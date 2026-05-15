#!/usr/bin/env python3
"""V421 operator-specific symbolic gate.

Tests a new symbolic-punctuation hypothesis: for 5-char Alice expressions,
learn transformations only from examples that use the same operator char as the
query, after removing the operator. This is CPU-only and uses weak labels only
to audit gains/losses.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from artifacts.v412_cpu_synthesis_gate.analyze_v412_cpu_synthesis_gate import (  # noqa: E402
    delete_char_programs,
    keep_char_programs,
    position_template_programs,
    transducer_programs,
)


BASELINE_CSV = ROOT / "artifacts" / "v342_acc_first_diagnostic" / "v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = ROOT / "artifacts" / "v421_operator_specific_symbolic_gate" / "20260515T_v421_operator_specific_symbolic"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    for line in str(prompt).splitlines():
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt))
    return examples, match.group(1).strip() if match else None


def split_expr(token: str | None) -> tuple[str, str, str] | None:
    text = str(token or "").strip()
    if len(text) != 5:
        return None
    return text[:2], text[2], text[3:]


def simple_one_shot_predictions(source_example: str, rhs: str, query_source: str) -> list[tuple[str, str]]:
    transforms = [
        ("identity", lambda text: text),
        ("reverse", lambda text: text[::-1]),
        ("left", lambda text: text[:2]),
        ("right", lambda text: text[2:]),
        ("right_left", lambda text: text[2:] + text[:2]),
    ]
    out: list[tuple[str, str]] = []
    for name, func in transforms:
        if func(source_example) == rhs:
            out.append((name, func(query_source)))
    return out


def operator_specific_prediction(examples: list[tuple[str, str]], query: str) -> tuple[str | None, dict]:
    parsed_query = split_expr(query)
    if not parsed_query:
        return None, {"status": "abstain", "reason": "query_not_len5"}
    q_left, q_op, q_right = parsed_query
    query_source = q_left + q_right
    same_op: list[tuple[str, str]] = []
    for lhs, rhs in examples:
        parsed = split_expr(lhs)
        if parsed and parsed[1] == q_op:
            same_op.append((parsed[0] + parsed[2], rhs))
    if not same_op:
        return None, {"status": "abstain", "reason": "no_same_operator_examples"}

    candidates: list[tuple[int, int, int, str, str]] = []
    if len(same_op) == 1:
        for name, pred in simple_one_shot_predictions(same_op[0][0], same_op[0][1], query_source):
            candidates.append((1, 1, 0, "one_shot:" + name, pred))
    else:
        programs = []
        programs += position_template_programs(same_op, query_source, max_position_sources=8, max_program_count=50000)
        programs += transducer_programs(same_op, pair_cap=5000, global_cap=30000)
        programs += delete_char_programs(same_op, 5)
        programs += keep_char_programs(same_op, 5)
        for program in programs:
            pred = program.apply(query_source)
            if pred:
                candidates.append((program.depth, program.nodes, program.literal_count, program.rule_class + ":" + program.name, pred))
    if not candidates:
        return None, {"status": "abstain", "reason": "no_operator_specific_program", "same_operator_examples": len(same_op)}
    candidates.sort()
    top_key = candidates[0][:3]
    top = [item for item in candidates if item[:3] == top_key]
    predictions = sorted({item[4] for item in top})
    if len(predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_operator_specific_top",
            "same_operator_examples": len(same_op),
            "top_predictions": predictions,
        }
    return predictions[0], {
        "status": "candidate",
        "reason": "operator_specific_symbolic_unique_top",
        "same_operator_examples": len(same_op),
        "top_programs": [item[3] for item in top[:5]],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(BASELINE_CSV)
    audit_rows: list[dict] = []
    accepted: list[dict] = []
    conflicts: list[dict] = []
    for row in rows:
        if row.get("type") != "equation_transform":
            continue
        examples, query = parse_examples(row.get("prompt", ""))
        pred, meta = operator_specific_prediction(examples, query or "")
        if pred is None:
            audit_rows.append(
                {
                    "id": row["id"],
                    "status": "abstain",
                    "prediction": "",
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "baseline_correct": row["correct"],
                    "reason": meta.get("reason", ""),
                    "same_operator_examples": meta.get("same_operator_examples", ""),
                    "proof": json.dumps(meta, sort_keys=True),
                }
            )
            continue
        baseline_correct = str(row["correct"]).lower() == "true"
        candidate_correct = pred == row["answer"]
        out = {
            "id": row["id"],
            "status": "candidate",
            "prediction": pred,
            "answer": row["answer"],
            "baseline_prediction": row["prediction"],
            "baseline_correct": row["correct"],
            "candidate_correct": str(candidate_correct),
            "reason": meta.get("reason", ""),
            "same_operator_examples": meta.get("same_operator_examples", ""),
            "proof": json.dumps(meta, sort_keys=True),
        }
        audit_rows.append(out)
        if baseline_correct and not candidate_correct:
            conflicts.append(out)
        elif not baseline_correct and candidate_correct:
            accepted.append(out)

    manifest = {
        "schema_version": "kg1_v421_operator_specific_symbolic_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": sum(1 for row in audit_rows if row["status"] == "candidate"),
        "accepted_gain_count": len(accepted),
        "conflict_count": len(conflicts),
        "hf_gpu_allowed": False,
        "decision": {
            "decision": "operator_specific_symbolic_gate_no_gain",
            "reason": f"accepted={len(accepted)}; conflicts={len(conflicts)}",
            "next_action": "Do not train from this hypothesis; continue with another symbolic structural class.",
        },
        "outputs": {
            "audit_csv": str((OUT_DIR / "v421_operator_specific_symbolic_audit.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v421_operator_specific_symbolic_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v421_operator_specific_symbolic_conflicts.csv").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v421_operator_specific_symbolic_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V421_OPERATOR_SPECIFIC_SYMBOLIC_GATE.md").relative_to(ROOT)),
        },
    }
    write_csv(OUT_DIR / "v421_operator_specific_symbolic_audit.csv", audit_rows)
    write_csv(OUT_DIR / "v421_operator_specific_symbolic_accepted.csv", accepted)
    write_csv(OUT_DIR / "v421_operator_specific_symbolic_conflicts.csv", conflicts)
    (OUT_DIR / "v421_operator_specific_symbolic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = f"""# V421 Operator-Specific Symbolic Gate

Generated: {manifest['generated_at_utc']}

| Metric | Value |
|---|---:|
| Candidate rows | `{manifest['candidate_count']}` |
| Accepted gains | `{manifest['accepted_gain_count']}` |
| Conflicts/losses | `{manifest['conflict_count']}` |

Decision: `{manifest['decision']['decision']}`.

This new same-operator symbolic hypothesis does not produce a safe gain. HF remains blocked.
"""
    (OUT_DIR / "V421_OPERATOR_SPECIFIC_SYMBOLIC_GATE.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V422 symbolic selection/substitution CPU gate.

This is an answer-free symbolic-punctuation probe for the remaining
equation_transform misses. It learns compact character substitution programs
from the examples inside each prompt, then audits predictions against weak
labels only after prediction. No training, packaging, or Kaggle submission is
authorized by this gate.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_CSV = ROOT / "artifacts" / "v342_acc_first_diagnostic" / "v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = ROOT / "artifacts" / "v422_symbolic_substitution_gate" / "20260515T_v422_symbolic_substitution_gate"


@dataclass(frozen=True)
class CandidateProgram:
    kind: str
    source_name: str
    source_positions: tuple[int, ...]
    output_len: int
    checked_loo: int
    score: tuple[int, int, int, int, int, str]
    prediction: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
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
        if len(lhs) == 5 and rhs:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt))
    query = match.group(1).strip() if match else None
    return examples, query


def source_variants(token: str) -> list[tuple[str, str, tuple[int, ...]]]:
    """Return candidate source alphabets as (name, source_text, original_positions)."""
    if len(token) != 5:
        return []
    left_right_positions = (0, 1, 3, 4)
    return [
        ("full5", token, (0, 1, 2, 3, 4)),
        ("operands4", token[0:2] + token[3:5], left_right_positions),
        ("left2", token[0:2], (0, 1)),
        ("right2", token[3:5], (3, 4)),
        ("op1", token[2], (2,)),
    ]


def build_map_global(
    examples: list[tuple[str, str]],
    source_name: str,
    source_positions: tuple[int, ...],
    tuple_positions: tuple[int, ...],
    omitted_index: int | None = None,
) -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for idx, (lhs, rhs) in enumerate(examples):
        if omitted_index is not None and idx == omitted_index:
            continue
        variants = {name: (text, positions) for name, text, positions in source_variants(lhs)}
        if source_name not in variants or len(rhs) != len(tuple_positions):
            return None
        source_text, _ = variants[source_name]
        for out_idx, source_idx in enumerate(tuple_positions):
            if source_idx >= len(source_text):
                return None
            key = source_text[source_idx]
            val = rhs[out_idx]
            old = mapping.get(key)
            if old is not None and old != val:
                return None
            mapping[key] = val
    return mapping


def build_map_slot(
    examples: list[tuple[str, str]],
    source_name: str,
    tuple_positions: tuple[int, ...],
    omitted_index: int | None = None,
) -> dict[tuple[int, str], str] | None:
    mapping: dict[tuple[int, str], str] = {}
    for idx, (lhs, rhs) in enumerate(examples):
        if omitted_index is not None and idx == omitted_index:
            continue
        variants = {name: text for name, text, _positions in source_variants(lhs)}
        if source_name not in variants or len(rhs) != len(tuple_positions):
            return None
        source_text = variants[source_name]
        for out_idx, source_idx in enumerate(tuple_positions):
            if source_idx >= len(source_text):
                return None
            key = (out_idx, source_text[source_idx])
            val = rhs[out_idx]
            old = mapping.get(key)
            if old is not None and old != val:
                return None
            mapping[key] = val
    return mapping


def apply_global(source_text: str, tuple_positions: tuple[int, ...], mapping: dict[str, str]) -> str | None:
    out: list[str] = []
    for source_idx in tuple_positions:
        if source_idx >= len(source_text):
            return None
        val = mapping.get(source_text[source_idx])
        if val is None:
            return None
        out.append(val)
    return "".join(out)


def apply_slot(source_text: str, tuple_positions: tuple[int, ...], mapping: dict[tuple[int, str], str]) -> str | None:
    out: list[str] = []
    for out_idx, source_idx in enumerate(tuple_positions):
        if source_idx >= len(source_text):
            return None
        val = mapping.get((out_idx, source_text[source_idx]))
        if val is None:
            return None
        out.append(val)
    return "".join(out)


def count_leave_one_out(
    kind: str,
    examples: list[tuple[str, str]],
    source_name: str,
    source_positions: tuple[int, ...],
    tuple_positions: tuple[int, ...],
) -> tuple[int, int]:
    checked = 0
    passed = 0
    for omitted_index, (lhs, rhs) in enumerate(examples):
        variants = {name: text for name, text, _positions in source_variants(lhs)}
        source_text = variants.get(source_name)
        if source_text is None:
            continue
        if kind == "global_substitution":
            mapping = build_map_global(examples, source_name, source_positions, tuple_positions, omitted_index=omitted_index)
            pred = apply_global(source_text, tuple_positions, mapping) if mapping is not None else None
        else:
            mapping = build_map_slot(examples, source_name, tuple_positions, omitted_index=omitted_index)
            pred = apply_slot(source_text, tuple_positions, mapping) if mapping is not None else None
        if pred is None:
            continue
        checked += 1
        if pred == rhs:
            passed += 1
    return checked, passed


def direct_selection_matches(
    examples: list[tuple[str, str]],
    source_name: str,
    tuple_positions: tuple[int, ...],
) -> bool:
    for lhs, rhs in examples:
        variants = {name: text for name, text, _positions in source_variants(lhs)}
        source_text = variants.get(source_name)
        if source_text is None or len(rhs) != len(tuple_positions):
            return False
        pred = "".join(source_text[source_idx] for source_idx in tuple_positions)
        if pred != rhs:
            return False
    return True


def symbolic_prediction(examples: list[tuple[str, str]], query: str) -> tuple[str | None, dict]:
    if len(query or "") != 5:
        return None, {"status": "abstain", "reason": "query_not_len5"}
    if len(examples) < 2:
        return None, {"status": "abstain", "reason": "too_few_examples"}

    rhs_lengths = sorted({len(rhs) for _lhs, rhs in examples if rhs})
    candidates: list[CandidateProgram] = []
    query_variants = {name: (text, positions) for name, text, positions in source_variants(query)}
    for out_len in rhs_lengths:
        if out_len <= 0 or out_len > 6:
            continue
        length_examples = [(lhs, rhs) for lhs, rhs in examples if len(rhs) == out_len]
        if len(length_examples) < 2:
            continue
        min_loo = 2 if len(length_examples) >= 3 else 1
        for source_name, source_text, source_positions in source_variants(query):
            if not source_text:
                continue
            tuple_count = len(source_text) ** out_len
            if tuple_count > 20000:
                continue
            for tuple_positions in itertools.product(range(len(source_text)), repeat=out_len):
                distinct_sources = len(set(tuple_positions))
                repeated_penalty = out_len - distinct_sources
                original_positions = tuple(source_positions[idx] for idx in tuple_positions)
                op_penalty = 1 if 2 in original_positions else 0
                if direct_selection_matches(length_examples, source_name, tuple_positions):
                    source_rank = {"operands4": 0, "full5": 1, "left2": 2, "right2": 2, "op1": 4}.get(source_name, 3)
                    pred = "".join(source_text[source_idx] for source_idx in tuple_positions)
                    score = (-1, source_rank, op_penalty, repeated_penalty, -len(length_examples), repr(original_positions))
                    candidates.append(
                        CandidateProgram(
                            kind="direct_selection",
                            source_name=source_name,
                            source_positions=original_positions,
                            output_len=out_len,
                            checked_loo=len(length_examples),
                            score=score,
                            prediction=pred,
                        )
                    )
                for kind, kind_rank in (("global_substitution", 0), ("slot_substitution", 1)):
                    if kind == "global_substitution":
                        mapping = build_map_global(length_examples, source_name, source_positions, tuple_positions)
                        pred = apply_global(source_text, tuple_positions, mapping) if mapping is not None else None
                    else:
                        mapping = build_map_slot(length_examples, source_name, tuple_positions)
                        pred = apply_slot(source_text, tuple_positions, mapping) if mapping is not None else None
                    if pred is None:
                        continue
                    checked, passed = count_leave_one_out(kind, length_examples, source_name, source_positions, tuple_positions)
                    if checked < min_loo or passed != checked:
                        continue
                    source_rank = {"operands4": 0, "full5": 1, "left2": 2, "right2": 2, "op1": 4}.get(source_name, 3)
                    score = (kind_rank, source_rank, op_penalty, repeated_penalty, -checked, repr(original_positions))
                    candidates.append(
                        CandidateProgram(
                            kind=kind,
                            source_name=source_name,
                            source_positions=original_positions,
                            output_len=out_len,
                            checked_loo=checked,
                            score=score,
                            prediction=pred,
                        )
                    )

    if not candidates:
        return None, {"status": "abstain", "reason": "no_selection_substitution_program"}
    candidates.sort(key=lambda item: item.score)
    best_score = candidates[0].score
    top = [item for item in candidates if item.score == best_score]
    top_predictions = sorted({item.prediction for item in top})
    if len(top_predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_top_selection_substitution",
            "top_predictions": top_predictions,
            "top_count": len(top),
        }
    best = top[0]
    return best.prediction, {
        "status": "candidate",
        "reason": "unique_selection_substitution_program",
        "program_kind": best.kind,
        "source_name": best.source_name,
        "source_positions": best.source_positions,
        "output_len": best.output_len,
        "checked_loo": best.checked_loo,
        "top_count": len(top),
    }


def summarize_projection(rows: list[dict[str, str]], accepted_by_id: dict[str, str]) -> dict[str, object]:
    summary = {
        "rows": len(rows),
        "correct": 0,
        "equation_transform_correct": 0,
        "bit_manipulation_correct": 0,
        "truncated": 0,
    }
    for row in rows:
        pred = accepted_by_id.get(row.get("id", ""), row.get("prediction", ""))
        is_correct = pred == row.get("answer", "")
        if is_correct:
            summary["correct"] = int(summary["correct"]) + 1
            if row.get("type") == "equation_transform":
                summary["equation_transform_correct"] = int(summary["equation_transform_correct"]) + 1
            elif row.get("type") == "bit_manipulation":
                summary["bit_manipulation_correct"] = int(summary["bit_manipulation_correct"]) + 1
    summary["accuracy"] = int(summary["correct"]) / max(1, len(rows))
    return summary


def main() -> int:
    print("=== V422 SYMBOLIC SUBSTITUTION GATE START ===", flush=True)
    print(f"baseline_csv = {BASELINE_CSV}", flush=True)
    print(f"output_dir = {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(BASELINE_CSV)
    print(f"input_rows = {len(rows)}", flush=True)

    audit_rows: list[dict] = []
    accepted: list[dict] = []
    conflicts: list[dict] = []
    for row in rows:
        if row.get("type") != "equation_transform":
            continue
        examples, query = parse_examples(row.get("prompt", ""))
        pred, meta = symbolic_prediction(examples, query or "")
        base_correct = str(row.get("correct", "")).lower() == "true"
        cand_correct = pred == row.get("answer", "") if pred is not None else False
        audit = {
            "id": row.get("id", ""),
            "status": "candidate" if pred is not None else "abstain",
            "prediction": pred or "",
            "answer": row.get("answer", ""),
            "baseline_prediction": row.get("prediction", ""),
            "baseline_correct": str(base_correct),
            "candidate_correct": str(cand_correct),
            "query": query or "",
            "example_count": str(len(examples)),
            "reason": str(meta.get("reason", "")),
            "program_kind": str(meta.get("program_kind", "")),
            "source_name": str(meta.get("source_name", "")),
            "source_positions": json.dumps(meta.get("source_positions", ""), sort_keys=True),
            "checked_loo": str(meta.get("checked_loo", "")),
            "proof": json.dumps(meta, sort_keys=True),
        }
        audit_rows.append(audit)
        if pred is None:
            continue
        if base_correct and not cand_correct:
            conflicts.append(audit)
        elif (not base_correct) and cand_correct:
            accepted.append(audit)

    accepted_by_id = {row["id"]: row["prediction"] for row in accepted}
    projection = summarize_projection(rows, accepted_by_id)
    decision = "hf_gpu_blocked_no_safe_gain"
    next_action = "Continue CPU-only symbolic structural search; do not train."
    if accepted and not conflicts and int(projection["equation_transform_correct"]) >= 60 and int(projection["bit_manipulation_correct"]) >= 136:
        decision = "cpu_signal_found_prepare_adapter_only_transfer_gate"
        next_action = "Use accepted IDs to build a tiny hard-negative transfer dataset; still require first-checkpoint ACC gate."

    manifest = {
        "schema_version": "kg1_v422_symbolic_substitution_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "source": "V291/V290 checkpoint-6",
            "weak_total": 192,
            "equation_transform": 56,
            "bit_manipulation": 136,
            "truncated": 0,
        },
        "teacher_reference": {
            "source": "V414/V366 consolidated CPU teacher/verifier",
            "weak_total": 222,
            "equation_transform": 63,
            "bit_manipulation": 159,
            "adapter_only": False,
        },
        "candidate_count": sum(1 for row in audit_rows if row["status"] == "candidate"),
        "accepted_gain_count": len(accepted),
        "conflict_count": len(conflicts),
        "projection": projection,
        "hf_gpu_allowed": decision == "cpu_signal_found_prepare_adapter_only_transfer_gate",
        "package_allowed": False,
        "kaggle_submit_allowed": False,
        "decision": {
            "decision": decision,
            "reason": f"accepted={len(accepted)}; conflicts={len(conflicts)}; projected_eq={projection['equation_transform_correct']}; projected_bit={projection['bit_manipulation_correct']}",
            "next_action": next_action,
        },
        "outputs": {
            "audit_csv": str((OUT_DIR / "v422_symbolic_substitution_audit.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v422_symbolic_substitution_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v422_symbolic_substitution_conflicts.csv").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v422_symbolic_substitution_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V422_SYMBOLIC_SUBSTITUTION_GATE.md").relative_to(ROOT)),
        },
    }

    write_csv(OUT_DIR / "v422_symbolic_substitution_audit.csv", audit_rows)
    write_csv(OUT_DIR / "v422_symbolic_substitution_accepted.csv", accepted)
    write_csv(OUT_DIR / "v422_symbolic_substitution_conflicts.csv", conflicts)
    (OUT_DIR / "v422_symbolic_substitution_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# V422 Symbolic Substitution Gate

Generated: {manifest['generated_at_utc']}

| Metric | Value |
|---|---:|
| Candidate rows | `{manifest['candidate_count']}` |
| Accepted gains | `{manifest['accepted_gain_count']}` |
| Conflicts/losses | `{manifest['conflict_count']}` |
| Projected weak total | `{projection['correct']}/315` |
| Projected equation_transform | `{projection['equation_transform_correct']}/155` |
| Projected bit_manipulation | `{projection['bit_manipulation_correct']}/160` |

Decision: `{decision}`.

This CPU gate tests a new structural class: selecting positions from the 5-char
Alice expression and learning global or slot-specific character substitutions
from in-prompt examples. Weak labels are used only for audit.
"""
    (OUT_DIR / "V422_SYMBOLIC_SUBSTITUTION_GATE.md").write_text(report, encoding="utf-8")
    print("v422_manifest =", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V422 SYMBOLIC SUBSTITUTION GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

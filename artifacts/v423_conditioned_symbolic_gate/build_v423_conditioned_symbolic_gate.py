#!/usr/bin/env python3
"""V423 conditioned symbolic CPU gate.

Tests a new class after V421/V422 failed: symbolic selection/substitution
programs conditioned on input invariants such as duplicated characters,
left/right equality, cross-position equality, and operator membership.
Weak labels are used only after prediction for audit.
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
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "artifacts" / "v422_symbolic_substitution_gate"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v422_symbolic_substitution_gate import (  # noqa: E402
    apply_global,
    apply_slot,
    build_map_global,
    build_map_slot,
    direct_selection_matches,
    source_variants,
)


BASELINE_CSV = ROOT / "artifacts" / "v342_acc_first_diagnostic" / "v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = ROOT / "artifacts" / "v423_conditioned_symbolic_gate" / "20260515T_v423_conditioned_symbolic_gate"


@dataclass(frozen=True)
class Condition:
    name: str
    fn: Callable[[str], bool]


@dataclass(frozen=True)
class Candidate:
    condition: str
    kind: str
    source_name: str
    source_positions: tuple[int, ...]
    output_len: int
    checked_loo: int
    prediction: str
    score: tuple[int, int, int, int, int, str]


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


def conditions() -> list[Condition]:
    return [
        Condition("left_chars_equal", lambda t: len(t) == 5 and t[0] == t[1]),
        Condition("right_chars_equal", lambda t: len(t) == 5 and t[3] == t[4]),
        Condition("left_equals_right", lambda t: len(t) == 5 and t[:2] == t[3:5]),
        Condition("left0_eq_right0", lambda t: len(t) == 5 and t[0] == t[3]),
        Condition("left0_eq_right1", lambda t: len(t) == 5 and t[0] == t[4]),
        Condition("left1_eq_right0", lambda t: len(t) == 5 and t[1] == t[3]),
        Condition("left1_eq_right1", lambda t: len(t) == 5 and t[1] == t[4]),
        Condition("op_in_left", lambda t: len(t) == 5 and t[2] in t[:2]),
        Condition("op_in_right", lambda t: len(t) == 5 and t[2] in t[3:5]),
        Condition("any_duplicate", lambda t: len(t) == 5 and len(set(t)) < 5),
        Condition("operand_duplicate", lambda t: len(t) == 5 and len(set(t[0:2] + t[3:5])) < 4),
        Condition("left_right_share_any", lambda t: len(t) == 5 and bool(set(t[:2]) & set(t[3:5]))),
        Condition("left_right_disjoint", lambda t: len(t) == 5 and not (set(t[:2]) & set(t[3:5]))),
    ]


def group_examples(examples: list[tuple[str, str]], condition: Condition) -> dict[bool, list[tuple[str, str]]]:
    groups: dict[bool, list[tuple[str, str]]] = {True: [], False: []}
    for lhs, rhs in examples:
        groups[bool(condition.fn(lhs))].append((lhs, rhs))
    return groups


def group_direct_prediction(
    examples: list[tuple[str, str]],
    source_name: str,
    tuple_positions: tuple[int, ...],
    query_source_text: str,
) -> str | None:
    if not examples:
        return None
    if not direct_selection_matches(examples, source_name, tuple_positions):
        return None
    if max(tuple_positions, default=-1) >= len(query_source_text):
        return None
    return "".join(query_source_text[idx] for idx in tuple_positions)


def group_map_prediction(
    kind: str,
    examples: list[tuple[str, str]],
    source_name: str,
    source_positions: tuple[int, ...],
    tuple_positions: tuple[int, ...],
    query_source_text: str,
) -> str | None:
    if not examples:
        return None
    if kind == "global_substitution":
        mapping = build_map_global(examples, source_name, source_positions, tuple_positions)
        return apply_global(query_source_text, tuple_positions, mapping) if mapping is not None else None
    mapping = build_map_slot(examples, source_name, tuple_positions)
    return apply_slot(query_source_text, tuple_positions, mapping) if mapping is not None else None


def conditioned_prediction_for_kind(
    condition: Condition,
    kind: str,
    examples: list[tuple[str, str]],
    source_name: str,
    source_positions: tuple[int, ...],
    tuple_positions: tuple[int, ...],
    query: str,
) -> tuple[str | None, int, int]:
    query_variants = {name: text for name, text, _pos in source_variants(query)}
    query_source_text = query_variants.get(source_name, "")
    if not query_source_text:
        return None, 0, 0
    groups = group_examples(examples, condition)
    query_group_key = bool(condition.fn(query))
    group_predictions: dict[bool, str] = {}
    for key, group in groups.items():
        if not group:
            continue
        if kind == "direct_selection":
            pred = group_direct_prediction(group, source_name, tuple_positions, query_source_text if key == query_group_key else source_variants(group[0][0])[0][1])
        else:
            pred = group_map_prediction(kind, group, source_name, source_positions, tuple_positions, query_source_text)
        if pred is None and key == query_group_key:
            return None, 0, 0
        if pred is not None:
            group_predictions[key] = pred
    if query_group_key not in group_predictions:
        return None, 0, 0

    checked = 0
    passed = 0
    for omitted_index, (lhs, rhs) in enumerate(examples):
        train = [item for idx, item in enumerate(examples) if idx != omitted_index]
        omitted_condition = bool(condition.fn(lhs))
        omitted_group = [item for item in train if bool(condition.fn(item[0])) == omitted_condition]
        if len(omitted_group) < 1:
            continue
        omitted_source_text = {name: text for name, text, _pos in source_variants(lhs)}.get(source_name, "")
        if not omitted_source_text:
            continue
        if kind == "direct_selection":
            if not direct_selection_matches(omitted_group, source_name, tuple_positions):
                continue
            if max(tuple_positions, default=-1) >= len(omitted_source_text):
                continue
            pred = "".join(omitted_source_text[idx] for idx in tuple_positions)
        else:
            pred = group_map_prediction(kind, omitted_group, source_name, source_positions, tuple_positions, omitted_source_text)
        if pred is None:
            continue
        checked += 1
        if pred == rhs:
            passed += 1
    return group_predictions[query_group_key], checked, passed


def conditioned_symbolic_prediction(examples: list[tuple[str, str]], query: str) -> tuple[str | None, dict]:
    if len(query or "") != 5:
        return None, {"status": "abstain", "reason": "query_not_len5"}
    if len(examples) < 3:
        return None, {"status": "abstain", "reason": "too_few_examples"}

    candidates: list[Candidate] = []
    rhs_lengths = sorted({len(rhs) for _lhs, rhs in examples if 0 < len(rhs) <= 6})
    for out_len in rhs_lengths:
        length_examples = [(lhs, rhs) for lhs, rhs in examples if len(rhs) == out_len]
        if len(length_examples) < 3:
            continue
        for condition in conditions():
            groups = group_examples(length_examples, condition)
            if len(groups[bool(condition.fn(query))]) < 2:
                continue
            if not groups[True] or not groups[False]:
                continue
            for source_name, source_text, source_positions in source_variants(query):
                if not source_text:
                    continue
                if len(source_text) ** out_len > 20000:
                    continue
                source_rank = {"operands4": 0, "full5": 1, "left2": 2, "right2": 2, "op1": 4}.get(source_name, 3)
                for tuple_positions in itertools.product(range(len(source_text)), repeat=out_len):
                    original_positions = tuple(source_positions[idx] for idx in tuple_positions)
                    op_penalty = 1 if 2 in original_positions else 0
                    repeated_penalty = out_len - len(set(tuple_positions))
                    for kind, kind_rank in (("direct_selection", -1), ("global_substitution", 0), ("slot_substitution", 1)):
                        pred, checked, passed = conditioned_prediction_for_kind(
                            condition,
                            kind,
                            length_examples,
                            source_name,
                            source_positions,
                            tuple_positions,
                            query,
                        )
                        if pred is None or checked < 2 or passed != checked:
                            continue
                        score = (kind_rank, source_rank, op_penalty, repeated_penalty, -checked, condition.name + repr(original_positions))
                        candidates.append(
                            Candidate(
                                condition=condition.name,
                                kind=kind,
                                source_name=source_name,
                                source_positions=original_positions,
                                output_len=out_len,
                                checked_loo=checked,
                                prediction=pred,
                                score=score,
                            )
                        )

    if not candidates:
        return None, {"status": "abstain", "reason": "no_conditioned_symbolic_program"}
    candidates.sort(key=lambda item: item.score)
    best_score = candidates[0].score
    top = [item for item in candidates if item.score == best_score]
    predictions = sorted({item.prediction for item in top})
    if len(predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_top_conditioned_symbolic",
            "top_predictions": predictions,
            "top_count": len(top),
        }
    best = top[0]
    return best.prediction, {
        "status": "candidate",
        "reason": "unique_conditioned_symbolic_program",
        "condition": best.condition,
        "program_kind": best.kind,
        "source_name": best.source_name,
        "source_positions": best.source_positions,
        "checked_loo": best.checked_loo,
        "output_len": best.output_len,
        "top_count": len(top),
    }


def summarize_projection(rows: list[dict[str, str]], accepted_by_id: dict[str, str]) -> dict[str, object]:
    summary = {"rows": len(rows), "correct": 0, "equation_transform_correct": 0, "bit_manipulation_correct": 0, "truncated": 0}
    for row in rows:
        pred = accepted_by_id.get(row.get("id", ""), row.get("prediction", ""))
        if pred == row.get("answer", ""):
            summary["correct"] = int(summary["correct"]) + 1
            if row.get("type") == "equation_transform":
                summary["equation_transform_correct"] = int(summary["equation_transform_correct"]) + 1
            elif row.get("type") == "bit_manipulation":
                summary["bit_manipulation_correct"] = int(summary["bit_manipulation_correct"]) + 1
    summary["accuracy"] = int(summary["correct"]) / max(1, len(rows))
    return summary


def main() -> int:
    print("=== V423 CONDITIONED SYMBOLIC GATE START ===", flush=True)
    print(f"baseline_csv = {BASELINE_CSV}", flush=True)
    print(f"output_dir = {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(BASELINE_CSV)
    audit_rows: list[dict] = []
    accepted: list[dict] = []
    conflicts: list[dict] = []
    for row in rows:
        if row.get("type") != "equation_transform":
            continue
        examples, query = parse_examples(row.get("prompt", ""))
        pred, meta = conditioned_symbolic_prediction(examples, query or "")
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
            "condition": str(meta.get("condition", "")),
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
    next_action = "Do not train; continue CPU-only search with a different symbolic class."
    if accepted and not conflicts and int(projection["equation_transform_correct"]) >= 60 and int(projection["bit_manipulation_correct"]) >= 136:
        decision = "cpu_signal_found_prepare_adapter_transfer_gate"
        next_action = "Build tiny hard-negative transfer set and require first-checkpoint ACC gate."

    manifest = {
        "schema_version": "kg1_v423_conditioned_symbolic_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {"weak_total": 192, "equation_transform": 56, "bit_manipulation": 136, "truncated": 0},
        "candidate_count": sum(1 for row in audit_rows if row["status"] == "candidate"),
        "accepted_gain_count": len(accepted),
        "conflict_count": len(conflicts),
        "projection": projection,
        "hf_gpu_allowed": decision != "hf_gpu_blocked_no_safe_gain",
        "package_allowed": False,
        "kaggle_submit_allowed": False,
        "decision": {
            "decision": decision,
            "reason": f"accepted={len(accepted)}; conflicts={len(conflicts)}; projected_eq={projection['equation_transform_correct']}; projected_bit={projection['bit_manipulation_correct']}",
            "next_action": next_action,
        },
        "outputs": {
            "audit_csv": str((OUT_DIR / "v423_conditioned_symbolic_audit.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v423_conditioned_symbolic_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v423_conditioned_symbolic_conflicts.csv").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v423_conditioned_symbolic_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V423_CONDITIONED_SYMBOLIC_GATE.md").relative_to(ROOT)),
        },
    }
    write_csv(OUT_DIR / "v423_conditioned_symbolic_audit.csv", audit_rows)
    write_csv(OUT_DIR / "v423_conditioned_symbolic_accepted.csv", accepted)
    write_csv(OUT_DIR / "v423_conditioned_symbolic_conflicts.csv", conflicts)
    (OUT_DIR / "v423_conditioned_symbolic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# V423 Conditioned Symbolic Gate

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

This CPU gate tests feature-conditioned symbolic rules. It remains answer-free
during prediction and uses weak labels only for audit.
"""
    (OUT_DIR / "V423_CONDITIONED_SYMBOLIC_GATE.md").write_text(report, encoding="utf-8")
    print("v423_manifest =", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V423 CONDITIONED SYMBOLIC GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

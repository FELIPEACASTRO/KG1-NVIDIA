#!/usr/bin/env python3
"""V433 string/multiset operator CPU gate for Alice equation rows.

This gate tests a symbolic class not covered by V431/V432: learn simple
operator-local string and multiset transforms from examples that share the
query operator. It is answer-free during prediction and uses weak labels only
for audit after candidates are produced.

No training, packaging, submission, or HF GPU is authorized by this script.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = ROOT / "artifacts/v433_string_multiset_operator_gate/20260515T_v433_string_multiset_operator"

EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}

AUDIT_COLUMNS = [
    "id",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "status",
    "reason",
    "same_operator_examples",
    "candidate_predictions",
    "candidate_count",
    "correct_candidate_present",
    "label_free_prediction",
    "label_free_correct",
    "accepted",
    "conflict",
    "proof",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_prompt(prompt: str) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str] | None]:
    examples: list[tuple[str, str, str, str]] = []
    for line in str(prompt or "").splitlines():
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs[:2], lhs[2], lhs[3:], rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query = match.group(1).strip() if match else ""
    if len(query) != 5:
        return examples, None
    return examples, (query[:2], query[2], query[3:])


def unique_order(text: str) -> str:
    out: list[str] = []
    for char in text:
        if char not in out:
            out.append(char)
    return "".join(out)


def ordered_intersection(left: str, right: str) -> str:
    return "".join(char for char in left if char in right)


def ordered_difference(left: str, right: str) -> str:
    return "".join(char for char in left if char not in right)


def counted_intersection(left: str, right: str) -> str:
    remaining = Counter(right)
    out: list[str] = []
    for char in left:
        if remaining[char] > 0:
            out.append(char)
            remaining[char] -= 1
    return "".join(out)


def prefix(text: str, size: int) -> str:
    if not text:
        return ""
    return (text * size)[:size]


def transform_values(left: str, right: str) -> dict[str, str]:
    base: dict[str, str] = {
        "left": left,
        "right": right,
        "concat_lr": left + right,
        "concat_rl": right + left,
        "reverse_concat_lr": (left + right)[::-1],
        "reverse_concat_rl": (right + left)[::-1],
        "reverse_left": left[::-1],
        "reverse_right": right[::-1],
        "intersection_lr": ordered_intersection(left, right),
        "intersection_rl": ordered_intersection(right, left),
        "difference_lr": ordered_difference(left, right),
        "difference_rl": ordered_difference(right, left),
        "symmetric_difference_lr": ordered_difference(left, right) + ordered_difference(right, left),
        "symmetric_difference_rl": ordered_difference(right, left) + ordered_difference(left, right),
        "union_lr": unique_order(left + right),
        "union_rl": unique_order(right + left),
        "counted_intersection_lr": counted_intersection(left, right),
        "counted_intersection_rl": counted_intersection(right, left),
        "left_first": left[:1],
        "left_second": left[1:],
        "right_first": right[:1],
        "right_second": right[1:],
        "outer": left[:1] + right[1:],
        "inner": left[1:] + right[:1],
        "swap_outer": right[1:] + left[:1],
        "swap_inner": right[:1] + left[1:],
    }
    values = dict(base)
    for name, value in list(base.items()):
        if value:
            values[name + "_x2"] = value * 2
            values[name + "_prefix2"] = prefix(value, 2)
            values[name + "_prefix3"] = prefix(value, 3)
            values[name + "_prefix4"] = prefix(value, 4)
    return values


def predict_operator_local(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    q_left, q_op, q_right = query
    same_operator = [(left, right, rhs) for left, op, right, rhs in examples if op == q_op]
    if not same_operator:
        return [], {"status": "abstain", "reason": "no_same_operator_examples", "same_operator_examples": 0}

    candidates: list[tuple[str, str]] = []
    query_values = transform_values(q_left, q_right)
    for name, query_prediction in sorted(query_values.items()):
        ok = True
        for left, right, rhs in same_operator:
            if transform_values(left, right).get(name) != rhs:
                ok = False
                break
        if ok:
            candidates.append((name, query_prediction))
    if not candidates:
        return [], {
            "status": "abstain",
            "reason": "no_matching_string_multiset_transform",
            "same_operator_examples": len(same_operator),
        }
    return candidates, {
        "status": "candidate",
        "reason": "operator_local_string_multiset_transform",
        "same_operator_examples": len(same_operator),
    }


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    truncated = 0
    families: dict[str, Counter[str]] = {"equation_transform": Counter(), "bit_manipulation": Counter()}
    for row in rows:
        family = str(row.get("family") or row.get("type") or classify_puzzle(str(row.get("prompt", ""))))
        if family not in families:
            continue
        correct = verify_answer(str(row.get("answer", "")), str(row.get(prediction_key, "")))
        total += int(correct)
        truncated += int(truthy(row.get("truncated", False)))
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
    return {
        "correct": total,
        "truncated": truncated,
        "families": {family: dict(counter) for family, counter in sorted(families.items())},
    }


def assert_expected_baseline(rows: list[dict[str, str]]) -> None:
    score = score_rows(rows, "prediction")
    observed = {
        "correct": int(score["correct"]),
        "equation_transform_correct": int(score["families"]["equation_transform"]["correct"]),
        "bit_manipulation_correct": int(score["families"]["bit_manipulation"]["correct"]),
        "truncated": int(score["truncated"]),
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError("unexpected baseline score: " + json.dumps(observed, sort_keys=True))


def render_report(path: Path, manifest: dict[str, Any], accepted: list[dict[str, Any]], ambiguous_correct: list[dict[str, Any]]) -> None:
    lines = [
        "# V433 String Multiset Operator Gate",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "CPU-only gate for operator-local string/multiset transforms in Alice equation rows.",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline V291/V290 | V433 projection | Delta |",
        "|---|---:|---:|---:|",
        f"| Total weak correct | `192/315` | `{manifest['projection']['total_correct']}/315` | `{manifest['projection']['total_delta']}` |",
        f"| equation_transform | `56/155` | `{manifest['projection']['equation_transform_correct']}/155` | `{manifest['projection']['equation_delta']}` |",
        f"| bit_manipulation | `136/160` | `{manifest['projection']['bit_manipulation_correct']}/160` | `{manifest['projection']['bit_delta']}` |",
        f"| Truncated | `0` | `{manifest['projection']['truncated']}` | `{manifest['projection']['truncated_delta']}` |",
        "",
        "## Gate Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate rows | `{manifest['candidate_rows']}` |",
        f"| Ambiguous rows | `{manifest['ambiguous_rows']}` |",
        f"| Ambiguous rows containing answer | `{manifest['ambiguous_correct_candidate_rows']}` |",
        f"| Accepted new gains | `{manifest['accepted_new_gains']}` |",
        f"| Conflicts | `{manifest['conflict_rows']}` |",
        "",
        "## Accepted Rows",
        "",
        "| id | prediction | answer |",
        "|---|---|---|",
    ]
    if accepted:
        for row in accepted:
            lines.append(f"| `{row['id']}` | `{row['label_free_prediction']}` | `{row['answer']}` |")
    else:
        lines.append("| none | none | none |")
    lines.extend(["", "## Ambiguous Correct Candidates", "", "| id | answer | candidate_predictions |", "|---|---|---|"])
    if ambiguous_correct:
        for row in ambiguous_correct:
            lines.append(f"| `{row['id']}` | `{row['answer']}` | `{row['candidate_predictions']}` |")
    else:
        lines.append("| none | none | none |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No GPU. The class produced no unique label-free gain; rows with the answer in the candidate set remain ambiguous and cannot be promoted without oracle selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=== V433 STRING MULTISET OPERATOR GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_csv =", BASELINE_CSV, flush=True)
    print("output_dir =", OUT_DIR, flush=True)

    rows = read_csv(BASELINE_CSV)
    assert_expected_baseline(rows)
    projected_rows = [dict(row) for row in rows]
    projected_by_id = {row["id"]: row for row in projected_rows}
    audit_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    ambiguous_correct: list[dict[str, Any]] = []

    equation_rows = [row for row in rows if str(row.get("type") or row.get("family")) == "equation_transform"]
    for index, row in enumerate(equation_rows, start=1):
        if index == 1 or index % 25 == 0 or index == len(equation_rows):
            print(f"v433_progress = {index}/{len(equation_rows)}", flush=True)
        examples, query = parse_prompt(row.get("prompt", ""))
        if query is None:
            continue
        candidates, meta = predict_operator_local(examples, query)
        predictions = sorted({prediction for _name, prediction in candidates})
        baseline_correct = truthy(row.get("correct", False))
        correct_candidate_present = any(verify_answer(row.get("answer", ""), prediction) for prediction in predictions)
        status = str(meta.get("status", "abstain"))
        reason = str(meta.get("reason", ""))
        label_free_prediction = ""
        label_free_correct = False
        accepted = False
        conflict = False
        if len(predictions) == 1:
            label_free_prediction = predictions[0]
            label_free_correct = verify_answer(row.get("answer", ""), label_free_prediction)
            if baseline_correct and not label_free_correct:
                conflict = True
                conflicts.append({"id": row["id"], "prediction": label_free_prediction, "answer": row["answer"]})
            elif (not baseline_correct) and label_free_correct:
                accepted = True
                projected_by_id[row["id"]]["v433_prediction"] = label_free_prediction
        elif len(predictions) > 1:
            status = "ambiguous"
            reason = "multiple_operator_local_predictions"
            if correct_candidate_present:
                ambiguous_correct.append(
                    {
                        "id": row["id"],
                        "answer": row["answer"],
                        "candidate_predictions": "|".join(predictions[:12]),
                    }
                )

        audit_row = {
            "id": row.get("id", ""),
            "answer": row.get("answer", ""),
            "baseline_prediction": row.get("prediction", ""),
            "baseline_correct": baseline_correct,
            "status": status,
            "reason": reason,
            "same_operator_examples": meta.get("same_operator_examples", ""),
            "candidate_predictions": "|".join(predictions[:12]),
            "candidate_count": len(predictions),
            "correct_candidate_present": correct_candidate_present,
            "label_free_prediction": label_free_prediction,
            "label_free_correct": label_free_correct,
            "accepted": accepted,
            "conflict": conflict,
            "proof": "|".join(name for name, prediction in candidates if prediction == label_free_prediction)[:500],
        }
        audit_rows.append(audit_row)
        if accepted:
            accepted_rows.append(audit_row)

    for row in projected_rows:
        if "v433_prediction" not in row:
            row["v433_prediction"] = row.get("prediction", "")
    projected_score = score_rows(projected_rows, "v433_prediction")
    projection = {
        "total_correct": int(projected_score["correct"]),
        "total_delta": int(projected_score["correct"]) - EXPECTED_BASELINE["correct"],
        "equation_transform_correct": int(projected_score["families"]["equation_transform"]["correct"]),
        "equation_delta": int(projected_score["families"]["equation_transform"]["correct"])
        - EXPECTED_BASELINE["equation_transform_correct"],
        "bit_manipulation_correct": int(projected_score["families"]["bit_manipulation"]["correct"]),
        "bit_delta": int(projected_score["families"]["bit_manipulation"]["correct"])
        - EXPECTED_BASELINE["bit_manipulation_correct"],
        "truncated": int(projected_score["truncated"]),
        "truncated_delta": int(projected_score["truncated"]) - EXPECTED_BASELINE["truncated"],
    }
    status_counts = Counter(str(row.get("status", "")) for row in audit_rows)
    manifest = {
        "schema_version": "kg1_v433_string_multiset_operator_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline_csv": str(BASELINE_CSV),
        "projection": projection,
        "audit_rows": len(audit_rows),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "candidate_rows": int(status_counts.get("candidate", 0)),
        "ambiguous_rows": int(status_counts.get("ambiguous", 0)),
        "ambiguous_correct_candidate_rows": len(ambiguous_correct),
        "accepted_new_gains": len(accepted_rows),
        "conflict_rows": len(conflicts),
        "hf_gpu_allowed": bool(
            len(accepted_rows) > 0
            and not conflicts
            and projection["total_correct"] > EXPECTED_BASELINE["correct"]
            and projection["equation_transform_correct"] > EXPECTED_BASELINE["equation_transform_correct"]
            and projection["bit_manipulation_correct"] >= EXPECTED_BASELINE["bit_manipulation_correct"]
            and projection["truncated"] == 0
        ),
        "decision": {
            "decision": "string_multiset_operator_no_unique_gain",
            "reason": "No unique label-free gain; correct candidates only appeared inside ambiguous candidate sets.",
            "next_action": "Do not launch GPU from V433. Continue only with another materially different CPU-gated class.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "v433_string_multiset_operator_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(OUT_DIR / "v433_string_multiset_operator_accepted.csv", accepted_rows, AUDIT_COLUMNS)
    write_csv(OUT_DIR / "v433_string_multiset_operator_conflicts.csv", conflicts, ["id", "prediction", "answer"])
    write_csv(OUT_DIR / "v433_string_multiset_operator_ambiguous_correct.csv", ambiguous_correct, ["id", "answer", "candidate_predictions"])
    write_json(OUT_DIR / "v433_string_multiset_operator_manifest.json", manifest)
    render_report(OUT_DIR / "V433_STRING_MULTISET_OPERATOR_GATE.md", manifest, accepted_rows, ambiguous_correct)

    print("candidate_status_counts =", json.dumps(manifest["candidate_status_counts"], sort_keys=True), flush=True)
    print("projection =", json.dumps(projection, sort_keys=True), flush=True)
    print("accepted_new_gains =", manifest["accepted_new_gains"], flush=True)
    print("ambiguous_correct_candidate_rows =", manifest["ambiguous_correct_candidate_rows"], flush=True)
    print("conflict_rows =", manifest["conflict_rows"], flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", OUT_DIR / "v433_string_multiset_operator_manifest.json", flush=True)
    print("=== V433 STRING MULTISET OPERATOR GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

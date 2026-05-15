#!/usr/bin/env python3
"""V432 label-free tiebreak audit for V431 symbolic cryptarithm candidates.

V431 found one real correction and several ambiguous/false candidates. This
gate asks a stricter submit-safety question: can any V431 candidate be promoted
without using the weak answer label, row id, or postprocessor-only oracle?

The gate is CPU-only and diagnostic. It does not train, package, submit, or
launch HF GPU.
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
V431_AUDIT_CSV = (
    ROOT
    / "artifacts/v431_signed_cryptarithm_gate/20260515T_v431_signed_cryptarithm/"
    / "v431_signed_cryptarithm_audit.csv"
)
V431_MANIFEST_JSON = (
    ROOT
    / "artifacts/v431_signed_cryptarithm_gate/20260515T_v431_signed_cryptarithm/"
    / "v431_signed_cryptarithm_manifest.json"
)
OUT_DIR = ROOT / "artifacts/v432_v431_label_free_tiebreak_gate/20260515T_v432_v431_label_free_tiebreak"


EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}


AUDIT_COLUMNS = [
    "id",
    "status_v431",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "candidate_predictions",
    "candidate_count",
    "correct_candidate_present",
    "query_expr",
    "query_operator",
    "query_operator_seen_in_examples",
    "example_operator_counts",
    "label_free_policy",
    "label_free_prediction",
    "label_free_correct",
    "label_free_promotable",
    "block_reason",
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_prompt(prompt: str) -> tuple[list[str], str]:
    examples: list[str] = []
    for raw in str(prompt or "").splitlines():
        if " = " not in raw:
            continue
        lhs, _rhs = raw.split(" = ", 1)
        lhs = lhs.strip()
        if len(lhs) == 5:
            examples.append(lhs)
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query = match.group(1).strip() if match else ""
    return examples, query


def operator_counts(expressions: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for expression in expressions:
        if len(expression) == 5:
            counts[expression[2]] += 1
    return counts


def split_candidates(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    return [part for part in value.split("|") if part]


def pick_label_free(candidates: list[str]) -> tuple[str, str, str]:
    """Return policy, prediction, block reason.

    Deliberately conservative. A unique candidate can be considered; multiple
    candidates are blocked because any row-level choice among them would require
    a verifier signal V431 does not provide.
    """

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return "unique_candidate_only", unique[0], ""
    if len(unique) > 1:
        return "blocked_multiple_candidates", "", "multiple_predictions_no_label_free_tiebreak"
    return "blocked_no_candidate", "", "no_candidate"


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


def render_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V432 V431 Label-Free Tiebreak Gate",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "CPU-only audit for whether V431 ambiguous/candidate rows can be promoted without using labels.",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline V291/V290 | V431 | V432 label-free |",
        "|---|---:|---:|---:|",
        f"| Total weak correct | `192/315` | `{manifest['v431_projection']['total_correct']}/315` | `{manifest['v432_projection']['total_correct']}/315` |",
        f"| equation_transform | `56/155` | `{manifest['v431_projection']['equation_transform_correct']}/155` | `{manifest['v432_projection']['equation_transform_correct']}/155` |",
        f"| bit_manipulation | `136/160` | `{manifest['v431_projection']['bit_manipulation_correct']}/160` | `{manifest['v432_projection']['bit_manipulation_correct']}/160` |",
        f"| Truncated | `0` | `{manifest['v431_projection']['truncated']}` | `{manifest['v432_projection']['truncated']}` |",
        "",
        "## Tiebreak Result",
        "",
        f"- Audited V431 non-abstain rows: `{manifest['audited_rows']}`.",
        f"- Label-free promotable new gains: `{manifest['label_free_new_gains']}`.",
        f"- Rows blocked by multiple candidates: `{manifest['blocked_multiple_candidates']}`.",
        f"- False unique candidates blocked by verification: `{manifest['blocked_false_unique_candidates']}`.",
        f"- `hf_gpu_allowed = {str(manifest['hf_gpu_allowed']).lower()}`.",
        "",
        "## Rows",
        "",
        "| id | V431 status | candidates | answer | policy | decision |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | `{row['status_v431']}` | `{row['candidate_predictions']}` | "
            f"`{row['answer']}` | `{row['label_free_policy']}` | `{row['block_reason'] or 'promotable'}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "V431 does not create a new submit-safe signal. The only label-free unique correct row is already known by V414; ambiguous rows with correct candidates require a row-specific choice that is not available at submission time.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=== V432 V431 LABEL FREE TIEBREAK GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_csv =", BASELINE_CSV, flush=True)
    print("v431_audit_csv =", V431_AUDIT_CSV, flush=True)
    print("v431_manifest_json =", V431_MANIFEST_JSON, flush=True)
    print("output_dir =", OUT_DIR, flush=True)

    baseline_rows = read_csv(BASELINE_CSV)
    assert_expected_baseline(baseline_rows)
    rows_by_id = {row["id"]: row for row in baseline_rows}
    v431_rows = read_csv(V431_AUDIT_CSV)
    v431_manifest = read_json(V431_MANIFEST_JSON)
    target_rows = [row for row in v431_rows if row.get("status") in {"accepted", "ambiguous", "candidate"}]
    print("target_rows =", len(target_rows), flush=True)

    projected_rows = [dict(row) for row in baseline_rows]
    projected_by_id = {row["id"]: row for row in projected_rows}
    audit_rows: list[dict[str, Any]] = []

    for index, row in enumerate(target_rows, start=1):
        print(f"v432_tiebreak_progress = {index}/{len(target_rows)} id={row.get('id','')}", flush=True)
        source = rows_by_id[row["id"]]
        examples, query = parse_prompt(source.get("prompt", ""))
        counts = operator_counts(examples)
        query_operator = query[2] if len(query) == 5 else ""
        candidates = split_candidates(row.get("candidate_predictions", ""))
        policy, prediction, block_reason = pick_label_free(candidates)
        baseline_correct = truthy(row.get("baseline_correct", ""))
        correct_candidate_present = any(verify_answer(row.get("answer", ""), candidate) for candidate in candidates)
        label_free_correct = verify_answer(row.get("answer", ""), prediction) if prediction else False
        promotable = bool(prediction and (not baseline_correct) and label_free_correct)
        if prediction and baseline_correct and not label_free_correct:
            block_reason = "unique_candidate_would_regress_baseline_correct"
            promotable = False
        elif prediction and not label_free_correct:
            block_reason = "unique_candidate_false_positive"
            promotable = False
        elif promotable and row.get("already_known_v414", "").lower() == "true":
            block_reason = "promotable_but_already_known_v414_not_new_signal"
            projected_by_id[row["id"]]["v432_prediction"] = prediction
        elif promotable:
            block_reason = ""
            projected_by_id[row["id"]]["v432_prediction"] = prediction

        audit_rows.append(
            {
                "id": row.get("id", ""),
                "status_v431": row.get("status", ""),
                "answer": row.get("answer", ""),
                "baseline_prediction": row.get("baseline_prediction", ""),
                "baseline_correct": baseline_correct,
                "candidate_predictions": "|".join(candidates),
                "candidate_count": len(candidates),
                "correct_candidate_present": correct_candidate_present,
                "query_expr": query,
                "query_operator": query_operator,
                "query_operator_seen_in_examples": bool(query_operator and counts.get(query_operator, 0) > 0),
                "example_operator_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                "label_free_policy": policy,
                "label_free_prediction": prediction,
                "label_free_correct": label_free_correct,
                "label_free_promotable": promotable,
                "block_reason": block_reason,
            }
        )

    for row in projected_rows:
        if "v432_prediction" not in row:
            row["v432_prediction"] = row.get("prediction", "")
    projection_score = score_rows(projected_rows, "v432_prediction")
    v432_projection = {
        "total_correct": int(projection_score["correct"]),
        "equation_transform_correct": int(projection_score["families"]["equation_transform"]["correct"]),
        "bit_manipulation_correct": int(projection_score["families"]["bit_manipulation"]["correct"]),
        "truncated": int(projection_score["truncated"]),
    }
    label_free_new_gains = sum(
        1
        for row in audit_rows
        if truthy(row["label_free_promotable"]) and row["block_reason"] == ""
    )
    blocked_false_unique = sum(1 for row in audit_rows if row["block_reason"] == "unique_candidate_false_positive")
    blocked_multiple = sum(1 for row in audit_rows if row["block_reason"] == "multiple_predictions_no_label_free_tiebreak")
    status_counts = Counter(str(row["block_reason"] or "promotable") for row in audit_rows)

    manifest = {
        "schema_version": "kg1_v432_v431_label_free_tiebreak_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline_csv": str(BASELINE_CSV),
        "v431_audit_csv": str(V431_AUDIT_CSV),
        "v431_manifest_json": str(V431_MANIFEST_JSON),
        "v431_projection": v431_manifest.get("projection", {}),
        "v432_projection": v432_projection,
        "audited_rows": len(audit_rows),
        "label_free_new_gains": int(label_free_new_gains),
        "blocked_false_unique_candidates": int(blocked_false_unique),
        "blocked_multiple_candidates": int(blocked_multiple),
        "block_reason_counts": dict(sorted(status_counts.items())),
        "hf_gpu_allowed": bool(
            label_free_new_gains > 0
            and v432_projection["total_correct"] > EXPECTED_BASELINE["correct"]
            and v432_projection["equation_transform_correct"] > EXPECTED_BASELINE["equation_transform_correct"]
            and v432_projection["bit_manipulation_correct"] >= EXPECTED_BASELINE["bit_manipulation_correct"]
            and v432_projection["truncated"] == 0
        ),
        "decision": {
            "decision": "v431_tiebreak_no_new_submit_safe_signal",
            "reason": (
                "V431 ambiguous rows either have multiple predictions without a label-free tie-break "
                "or a unique false positive; the only correct promoted row is already known by V414."
            ),
            "next_action": "Do not launch GPU from V431. Continue only with a materially new CPU-gated solver class or adapter-direct signal.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "v432_v431_label_free_tiebreak_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_json(OUT_DIR / "v432_v431_label_free_tiebreak_manifest.json", manifest)
    render_report(OUT_DIR / "V432_V431_LABEL_FREE_TIEBREAK_GATE.md", manifest, audit_rows)

    print("v432_projection =", json.dumps(v432_projection, sort_keys=True), flush=True)
    print("block_reason_counts =", json.dumps(manifest["block_reason_counts"], sort_keys=True), flush=True)
    print("label_free_new_gains =", label_free_new_gains, flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", OUT_DIR / "v432_v431_label_free_tiebreak_manifest.json", flush=True)
    print("=== V432 V431 LABEL FREE TIEBREAK GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

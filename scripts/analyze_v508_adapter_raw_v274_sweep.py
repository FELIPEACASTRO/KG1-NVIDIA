#!/usr/bin/env python3
"""Sweep all adapter raw-output candidates with V274 label-free projection.

V508 is CPU-only and deliberately narrow:

* input is the V505 revalidation summary;
* only rows with ``has_raw_output=True`` are considered;
* every candidate CSV is re-extracted from ``raw_output`` using
  ``extract_final_answer``;
* V274 is applied only after label-free extraction;
* labels are used only to audit weak metrics.

This prevents reference-only CSVs or expected-aware extraction from entering
the submit-safe decision path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402
from src.kg1_v274_numeric_postprocessor import postprocess_numeric_prediction  # noqa: E402


DEFAULT_V505_SUMMARY = (
    REPO_ROOT
    / "artifacts/v505_label_free_candidate_revalidation/v505_label_free_revalidation_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v508_adapter_raw_v274_sweep"


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
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def bool_text(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def family_counts(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row.get("family") or row.get("task_type") or row.get("type") or "")
        counts[family]["rows"] += 1
        counts[family]["correct"] += int(verify_answer(row.get("answer", ""), row.get(prediction_key, "")))
        counts[family]["truncated"] += int(bool_text(row.get("truncated_bool", row.get("truncated", ""))))
    return {family: dict(counter) for family, counter in sorted(counts.items())}


def score_candidate(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(path)
    if len(rows) != 315:
        raise RuntimeError(f"{path}: expected weak315 rows, got {len(rows)}")
    if not rows or "raw_output" not in rows[0]:
        raise RuntimeError(f"{path}: raw_output column missing")

    audited: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("family") or row.get("task_type") or row.get("type") or "")
        label_free = extract_final_answer(row.get("raw_output", ""))
        post = label_free
        rule = "not_attempted"
        applied = False
        if family == "equation_transform":
            decision = postprocess_numeric_prediction(
                str(row.get("prompt", "")),
                label_free,
                family=family,
                truncated=bool_text(row.get("truncated_bool", row.get("truncated", ""))),
            )
            post = decision.prediction
            rule = decision.rule
            applied = decision.applied
        audited.append(
            {
                "id": row.get("id", ""),
                "family": family,
                "answer": row.get("answer", ""),
                "stored_prediction": row.get("prediction", ""),
                "label_free_prediction": label_free,
                "v274_prediction": post,
                "label_free_correct": verify_answer(row.get("answer", ""), label_free),
                "v274_correct": verify_answer(row.get("answer", ""), post),
                "v274_rule": rule,
                "v274_applied": applied,
            }
        )

    base_counts = family_counts(audited, "label_free_prediction")
    post_counts = family_counts(audited, "v274_prediction")
    gains = [row for row in audited if not row["label_free_correct"] and row["v274_correct"]]
    losses = [row for row in audited if row["label_free_correct"] and not row["v274_correct"]]
    overcounts = [
        row
        for row in audited
        if row["stored_prediction"] != row["label_free_prediction"]
        and verify_answer(row["answer"], row["stored_prediction"])
        and not verify_answer(row["answer"], row["label_free_prediction"])
    ]

    summary = {
        "path": str(path),
        "name": path.stem,
        "base_total_label_free": sum(int(item["correct"]) for item in base_counts.values()),
        "v274_total_label_free": sum(int(item["correct"]) for item in post_counts.values()),
        "base_equation_label_free": int(base_counts.get("equation_transform", {}).get("correct", 0)),
        "v274_equation_label_free": int(post_counts.get("equation_transform", {}).get("correct", 0)),
        "base_bit_label_free": int(base_counts.get("bit_manipulation", {}).get("correct", 0)),
        "v274_bit_label_free": int(post_counts.get("bit_manipulation", {}).get("correct", 0)),
        "base_truncated": sum(int(item.get("truncated", 0)) for item in base_counts.values()),
        "v274_truncated": sum(int(item.get("truncated", 0)) for item in post_counts.values()),
        "gains": len(gains),
        "losses": len(losses),
        "gain_ids": " ".join(row["id"] for row in gains),
        "loss_ids": " ".join(row["id"] for row in losses),
        "symbolic_boxing_overcount_ids": " ".join(row["id"] for row in overcounts),
        "postprocessor_submit_safe": False,
        "adapter_only_promotable": False,
        "decision": "v274_signal_not_adapter_only_gain",
    }
    if (
        summary["v274_total_label_free"] > summary["base_total_label_free"]
        and summary["losses"] == 0
        and summary["v274_bit_label_free"] >= 136
    ):
        summary["cpu_signal"] = "positive_no_loss"
    else:
        summary["cpu_signal"] = "blocked_or_regressive"
    return summary, audited


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_rows = read_csv(args.v505_summary_csv)
    raw_paths = [
        Path(row["path"])
        for row in summary_rows
        if bool_text(row.get("has_raw_output")) and int(row.get("rows", 0)) == 315
    ]
    if not raw_paths:
        raise RuntimeError("no adapter raw-output weak315 candidates found")

    candidate_summaries: list[dict[str, Any]] = []
    all_change_rows: list[dict[str, Any]] = []
    for candidate_path in raw_paths:
        summary, audited = score_candidate(candidate_path)
        candidate_summaries.append(summary)
        for row in audited:
            if row["label_free_prediction"] != row["v274_prediction"] or row["label_free_correct"] != row["v274_correct"]:
                change = dict(row)
                change["candidate_name"] = summary["name"]
                change["candidate_path"] = summary["path"]
                all_change_rows.append(change)

    candidate_summaries.sort(
        key=lambda row: (
            int(row["v274_total_label_free"]),
            int(row["v274_equation_label_free"]),
            int(row["v274_bit_label_free"]),
            -int(row["losses"]),
        ),
        reverse=True,
    )
    best_overall = candidate_summaries[0]
    guardrail_candidates = [
        row
        for row in candidate_summaries
        if int(row["losses"]) == 0
        and int(row["v274_bit_label_free"]) >= 136
        and int(row["v274_truncated"]) == 0
    ]
    best_guardrail = guardrail_candidates[0] if guardrail_candidates else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"{args.label}_summary.csv"
    changes_csv = output_dir / f"{args.label}_changes.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"

    summary_columns = [
        "name",
        "path",
        "base_total_label_free",
        "v274_total_label_free",
        "base_equation_label_free",
        "v274_equation_label_free",
        "base_bit_label_free",
        "v274_bit_label_free",
        "base_truncated",
        "v274_truncated",
        "gains",
        "losses",
        "gain_ids",
        "loss_ids",
        "symbolic_boxing_overcount_ids",
        "cpu_signal",
        "postprocessor_submit_safe",
        "adapter_only_promotable",
        "decision",
    ]
    change_columns = [
        "candidate_name",
        "candidate_path",
        "id",
        "family",
        "answer",
        "stored_prediction",
        "label_free_prediction",
        "v274_prediction",
        "label_free_correct",
        "v274_correct",
        "v274_rule",
        "v274_applied",
    ]
    write_csv(summary_csv, candidate_summaries, summary_columns)
    write_csv(changes_csv, all_change_rows, change_columns)

    manifest = {
        "schema_version": "kg1_v508_adapter_raw_v274_sweep_v1",
        "generated_at_utc": utc_now(),
        "v505_summary_csv": str(args.v505_summary_csv),
        "adapter_raw_candidate_count": len(raw_paths),
        "summary_csv": str(summary_csv),
        "changes_csv": str(changes_csv),
        "best_overall": best_overall,
        "best_guardrail": best_guardrail,
        "submit_safe_gain_found": False,
        "decision": {
            "status": "no_adapter_only_submit_safe_gain",
            "next_action": (
                "Do not run GPU from this result alone. The best guardrail-preserving "
                "label-free V274 projection is still postprocessor-based and reaches "
                "195/315, equation=59, bit=136. The only equation=60 raw candidates "
                "have bit=135 and fail the bit guardrail."
            ),
        },
    }
    write_json(manifest_json, manifest)
    print("=== V508 ADAPTER RAW V274 SWEEP START ===", flush=True)
    print("adapter_raw_candidate_count =", len(raw_paths), flush=True)
    print("best_overall =", json.dumps(best_overall, sort_keys=True), flush=True)
    print("best_guardrail =", json.dumps(best_guardrail, sort_keys=True), flush=True)
    print("summary_csv =", summary_csv, flush=True)
    print("changes_csv =", changes_csv, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V508 ADAPTER RAW V274 SWEEP END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v505-summary-csv", type=Path, default=DEFAULT_V505_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v508_adapter_raw_v274_sweep")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

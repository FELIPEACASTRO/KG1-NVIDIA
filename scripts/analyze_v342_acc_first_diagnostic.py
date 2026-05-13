#!/usr/bin/env python3
"""ACC-first diagnostic for V341 vs the best adapter-only baseline and V336A.

This script intentionally treats family exact-match accuracy as the promotion
metric. Training loss and preference accuracy are not used as promotion signals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = ("bit_manipulation", "equation_transform")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_of(row: dict[str, str]) -> str:
    return (row.get("family") or row.get("type") or "").strip()


def normalize_prediction(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name]).strip()
    return ""


def keyed(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"{label}: row without id")
        if row_id in out:
            raise ValueError(f"{label}: duplicate id {row_id}")
        out[row_id] = row
    return out


def summarize(rows: list[dict[str, Any]], correct_col: str, truncated_col: str | None = None) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    fam: dict[str, dict[str, int]] = {
        family: {"rows": 0, "correct": 0, "truncated": 0} for family in EXPECTED_FAMILIES
    }
    for row in rows:
        family = str(row["family"])
        if family not in fam:
            continue
        total += 1
        fam[family]["rows"] += 1
        if as_bool(row[correct_col]):
            correct += 1
            fam[family]["correct"] += 1
        if truncated_col and as_bool(row.get(truncated_col, False)):
            truncated += 1
            fam[family]["truncated"] += 1
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": fam,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = Path(args.baseline_predictions_csv)
    v341_path = Path(args.v341_predictions_csv)
    v336_path = Path(args.v336_predictions_csv)
    trace_path = Path(args.v336_candidate_trace_csv) if args.v336_candidate_trace_csv else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows_raw = read_csv_rows(baseline_path)
    v341_rows_raw = read_csv_rows(v341_path)
    v336_rows_raw = read_csv_rows(v336_path)
    trace_rows = read_csv_rows(trace_path) if trace_path and trace_path.exists() else []

    baseline = keyed(baseline_rows_raw, "baseline")
    v341 = keyed(v341_rows_raw, "v341")
    v336 = keyed(v336_rows_raw, "v336")

    ids = list(baseline.keys())
    if set(ids) != set(v341.keys()) or set(ids) != set(v336.keys()):
        raise ValueError(
            "row contract id mismatch: "
            f"baseline={len(baseline)} v341={len(v341)} v336={len(v336)}"
        )
    if len(ids) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(ids)}")

    trace_by_id = {row["id"]: row for row in trace_rows if row.get("id")}

    comparison_rows: list[dict[str, Any]] = []
    counts = Counter()
    counts_by_family: dict[str, Counter] = defaultdict(Counter)
    for row_id in ids:
        b = baseline[row_id]
        c = v341[row_id]
        s = v336[row_id]
        family = family_of(b)
        if family != family_of(c) or family != family_of(s):
            raise ValueError(f"family mismatch for {row_id}: {family}, {family_of(c)}, {family_of(s)}")
        if str(b.get("answer", "")).strip() != str(c.get("answer", "")).strip():
            raise ValueError(f"answer mismatch baseline/v341 for {row_id}")
        if str(b.get("answer", "")).strip() != str(s.get("answer", "")).strip():
            raise ValueError(f"answer mismatch baseline/v336 for {row_id}")

        baseline_correct = as_bool(b.get("correct", ""))
        v341_correct = as_bool(c.get("correct", ""))
        v336_correct = as_bool(s.get("integrated_correct", ""))
        row_trace = trace_by_id.get(row_id, {})

        has_accepted_solver_trace = (
            bool(row_trace)
            and as_bool(row_trace.get("accepted", ""))
            and str(row_trace.get("rule_class", "")).strip() != ""
        )

        if v341_correct and not baseline_correct:
            bucket = "v341_gain_vs_baseline"
        elif baseline_correct and not v341_correct:
            bucket = "v341_loss_vs_baseline"
        elif v336_correct and not baseline_correct and not v341_correct and has_accepted_solver_trace:
            bucket = "v336_solver_salvage_not_learned_by_v341"
        elif v336_correct and not baseline_correct and not v341_correct:
            bucket = "v336_reference_correct_without_solver_trace"
        elif v336_correct and not baseline_correct and has_accepted_solver_trace:
            bucket = "v336_solver_gain_vs_baseline"
        elif v336_correct and not baseline_correct:
            bucket = "v336_reference_gain_without_solver_trace"
        else:
            bucket = "neutral"
        counts[bucket] += 1
        counts_by_family[family][bucket] += 1

        comparison_rows.append(
            {
                "id": row_id,
                "family": family,
                "answer": str(b.get("answer", "")).strip(),
                "baseline_prediction": normalize_prediction(b, "prediction"),
                "v341_prediction": normalize_prediction(c, "prediction"),
                "v336_prediction": normalize_prediction(s, "prediction"),
                "baseline_correct": baseline_correct,
                "v341_correct": v341_correct,
                "v336_integrated_correct": v336_correct,
                "baseline_truncated": as_bool(b.get("truncated", "")),
                "v341_truncated": as_bool(c.get("truncated", "")),
                "v341_delta_vs_baseline": int(v341_correct) - int(baseline_correct),
                "v336_delta_vs_baseline": int(v336_correct) - int(baseline_correct),
                "bucket": bucket,
                "v336_rule_class": row_trace.get("rule_class", ""),
                "v336_candidate_count": row_trace.get("candidate_count", ""),
                "v336_conflict_count": row_trace.get("conflict_count", ""),
                "v336_reason": row_trace.get("reason", ""),
            }
        )

    baseline_norm_rows = [
        {"family": family_of(row), "correct": as_bool(row.get("correct", "")), "truncated": as_bool(row.get("truncated", ""))}
        for row in baseline_rows_raw
    ]
    v341_norm_rows = [
        {"family": family_of(row), "correct": as_bool(row.get("correct", "")), "truncated": as_bool(row.get("truncated", ""))}
        for row in v341_rows_raw
    ]
    v336_norm_rows = [
        {"family": family_of(row), "correct": as_bool(row.get("integrated_correct", "")), "truncated": as_bool(row.get("truncated", ""))}
        for row in v336_rows_raw
    ]

    salvage_rows = [
        row for row in comparison_rows if row["bucket"] == "v336_solver_salvage_not_learned_by_v341"
    ]
    reference_correct_without_trace_rows = [
        row for row in comparison_rows if row["bucket"] == "v336_reference_correct_without_solver_trace"
    ]
    v341_loss_rows = [row for row in comparison_rows if row["bucket"] == "v341_loss_vs_baseline"]
    v341_gain_rows = [row for row in comparison_rows if row["bucket"] == "v341_gain_vs_baseline"]

    outputs = {
        "comparison_csv": str(output_dir / "v342_acc_first_row_comparison.csv"),
        "v336_solver_salvage_not_learned_csv": str(output_dir / "v342_v336_solver_salvage_not_learned_by_v341.csv"),
        "v336_reference_correct_without_solver_trace_csv": str(
            output_dir / "v342_v336_reference_correct_without_solver_trace.csv"
        ),
        "v341_losses_csv": str(output_dir / "v342_v341_losses_vs_baseline.csv"),
        "v341_gains_csv": str(output_dir / "v342_v341_gains_vs_baseline.csv"),
        "manifest_json": str(output_dir / "v342_acc_first_diagnostic_manifest.json"),
    }

    write_csv(Path(outputs["comparison_csv"]), comparison_rows)
    write_csv(Path(outputs["v336_solver_salvage_not_learned_csv"]), salvage_rows)
    write_csv(Path(outputs["v336_reference_correct_without_solver_trace_csv"]), reference_correct_without_trace_rows)
    write_csv(Path(outputs["v341_losses_csv"]), v341_loss_rows)
    write_csv(Path(outputs["v341_gains_csv"]), v341_gain_rows)

    decision = {
        "decision": "stop_gpu_preference_path_and_return_to_cpu_solver_gate",
        "reason": (
            "V341 checkpoint-2 regressed weak ACC versus baseline; "
            "the verified V336A solver rule gains were not learned by the adapter."
        ),
        "next_action": (
            "Use V336A accepted rows/rule classes as CPU no-loss solver/package inputs; "
            "only launch GPU if a new CPU gate creates a non-saturated training set with "
            "no-loss weak gain over 192/315."
        ),
    }

    manifest: dict[str, Any] = {
        "schema_version": "kg1_v342_acc_first_diagnostic_v1",
        "inputs": {
            "baseline_predictions_csv": str(baseline_path),
            "baseline_predictions_sha256": sha256_path(baseline_path),
            "v341_predictions_csv": str(v341_path),
            "v341_predictions_sha256": sha256_path(v341_path),
            "v336_predictions_csv": str(v336_path),
            "v336_predictions_sha256": sha256_path(v336_path),
            "v336_candidate_trace_csv": str(trace_path) if trace_path else "",
            "v336_candidate_trace_sha256": sha256_path(trace_path) if trace_path and trace_path.exists() else "",
        },
        "summaries": {
            "baseline_adapter_only": summarize(baseline_norm_rows, "correct", "truncated"),
            "v341_clean_preference_checkpoint2": summarize(v341_norm_rows, "correct", "truncated"),
            "v336a_integrated_no_loss_solver": summarize(v336_norm_rows, "correct", "truncated"),
        },
        "delta_counts": dict(counts),
        "delta_counts_by_family": {family: dict(counter) for family, counter in counts_by_family.items()},
        "v341_gain_ids": [row["id"] for row in v341_gain_rows],
        "v341_loss_ids": [row["id"] for row in v341_loss_rows],
        "v336_solver_salvage_not_learned_ids": [row["id"] for row in salvage_rows],
        "v336_solver_salvage_rule_classes": dict(Counter(row["v336_rule_class"] for row in salvage_rows)),
        "v336_reference_correct_without_solver_trace_ids": [
            row["id"] for row in reference_correct_without_trace_rows
        ],
        "decision": decision,
        "outputs": outputs,
    }
    Path(outputs["manifest_json"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("=== V342 ACC-FIRST DIAGNOSTIC SUMMARY ===")
    print(json.dumps(manifest["summaries"], indent=2, sort_keys=True))
    print("delta_counts =", json.dumps(manifest["delta_counts"], sort_keys=True))
    print("delta_counts_by_family =", json.dumps(manifest["delta_counts_by_family"], sort_keys=True))
    print("decision =", json.dumps(decision, indent=2, sort_keys=True))
    print("manifest_json =", outputs["manifest_json"])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions-csv", required=True)
    parser.add_argument("--v341-predictions-csv", required=True)
    parser.add_argument("--v336-predictions-csv", required=True)
    parser.add_argument("--v336-candidate-trace-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=315)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

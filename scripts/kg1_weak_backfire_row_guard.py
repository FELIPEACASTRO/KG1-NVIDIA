#!/usr/bin/env python3
"""Guard weak eval candidates against known row-level backfires.

This script is intentionally CPU-only.  It is meant to run after a weak eval
CSV is produced and before any full eval/package/submit decision.  Family
totals are necessary, but not sufficient: V518 showed that a small equation
gain can be canceled by a single bit regression while loss still improves.
Loss movement alone is not actionable for promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_CSV = Path("artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv")
DEFAULT_PROTECTED = ["8740ed31=01101000"]


def truthy(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y"}


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise RuntimeError(f"missing id in {path}")
        if row_id in out:
            raise RuntimeError(f"duplicate id {row_id} in {path}")
        out[row_id] = row
    return out


def parse_expected(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected ID=ANSWER, got {value!r}")
        row_id, answer = value.split("=", 1)
        row_id = row_id.strip()
        if not row_id:
            raise argparse.ArgumentTypeError(f"missing id in {value!r}")
        out[row_id] = answer.strip()
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    baseline = read_csv_by_id(args.baseline_csv)
    candidate = read_csv_by_id(args.candidate_csv)
    protected = parse_expected(args.protected_id_answer)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []

    for row_id, expected_answer in protected.items():
        base = baseline.get(row_id)
        cand = candidate.get(row_id)
        if base is None:
            blockers.append(f"protected_id_missing_in_baseline:{row_id}")
            continue
        if cand is None:
            blockers.append(f"protected_id_missing_in_candidate:{row_id}")
            continue
        base_prediction = str(base.get(args.baseline_prediction_column, "")).strip()
        cand_prediction = str(cand.get(args.candidate_prediction_column, "")).strip()
        base_correct = truthy(base.get(args.baseline_correct_column))
        cand_correct = truthy(cand.get(args.candidate_correct_column))
        answer_matches = cand_prediction == expected_answer
        protected_ok = cand_correct and answer_matches
        if not protected_ok:
            blockers.append(f"protected_id_backfire:{row_id}")
        rows.append(
            {
                "id": row_id,
                "expected_answer": expected_answer,
                "baseline_prediction": base_prediction,
                "baseline_correct": base_correct,
                "candidate_prediction": cand_prediction,
                "candidate_correct": cand_correct,
                "answer_matches_expected": answer_matches,
                "protected_ok": protected_ok,
            }
        )

    candidate_correct = Counter()
    for row in candidate.values():
        family = str(row.get("family") or row.get("type") or row.get("task_type") or "").strip()
        if family:
            candidate_correct[family] += int(truthy(row.get(args.candidate_correct_column)))

    report = {
        "schema_version": "kg1_weak_backfire_row_guard_v1",
        "baseline_csv": str(args.baseline_csv),
        "candidate_csv": str(args.candidate_csv),
        "candidate_correct_by_family": dict(sorted(candidate_correct.items())),
        "protected_id_answer": protected,
        "rows": rows,
        "blockers": blockers,
        "passed": not blockers,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if blockers and not args.allow_blocked:
        raise SystemExit(1)
    return report


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = root / "baseline.csv"
        candidate_ok = root / "candidate_ok.csv"
        candidate_bad = root / "candidate_bad.csv"
        header = "id,family,answer,prediction,correct\n"
        baseline.write_text(header + "8740ed31,bit_manipulation,01101000,01101000,True\n", encoding="utf-8")
        candidate_ok.write_text(header + "8740ed31,bit_manipulation,01101000,01101000,True\n", encoding="utf-8")
        candidate_bad.write_text(header + "8740ed31,bit_manipulation,01101000,01111000,False\n", encoding="utf-8")
        ok_args = parse_args(
            [
                "--baseline-csv",
                str(baseline),
                "--candidate-csv",
                str(candidate_ok),
                "--protected-id-answer",
                "8740ed31=01101000",
            ]
        )
        bad_args = parse_args(
            [
                "--baseline-csv",
                str(baseline),
                "--candidate-csv",
                str(candidate_bad),
                "--protected-id-answer",
                "8740ed31=01101000",
                "--allow-blocked",
            ]
        )
        if not audit(ok_args)["passed"]:
            raise AssertionError("expected ok candidate to pass")
        if audit(bad_args)["passed"]:
            raise AssertionError("expected bad candidate to block")
    print("kg1_weak_backfire_row_guard_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--protected-id-answer", action="append", default=list(DEFAULT_PROTECTED))
    parser.add_argument("--baseline-prediction-column", default="prediction")
    parser.add_argument("--candidate-prediction-column", default="prediction")
    parser.add_argument("--baseline-correct-column", default="correct")
    parser.add_argument("--candidate-correct-column", default="correct")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and args.candidate_csv is None:
        parser.error("--candidate-csv is required unless --self-test is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

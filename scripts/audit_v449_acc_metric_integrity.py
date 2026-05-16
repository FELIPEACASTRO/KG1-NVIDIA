#!/usr/bin/env python3
"""Audit that KG1 ACC scoring uses the strict Kaggle-style verifier.

This is a CPU-only validation script. It does not run inference, train,
package, or submit. Its purpose is to catch metric drift before a weak/full
gate is trusted for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    answers_equivalent,
    canonical_family,
    extract_final_answer,
    extract_final_answer_for_expected,
    verify_answer,
)


DEFAULT_WEAK_CSV = (
    ROOT
    / "artifacts"
    / "v290_rank19_micro_patch_reference"
    / "runtime_artifacts"
    / "v245_weak_eval_bridge"
    / "v245-weak-bridge-hfonly-20260510T1950Z"
    / "v221_weak_315.csv"
)

PREDICTION_COLUMN_CANDIDATES = [
    "prediction",
    "integrated_prediction",
    "v414_projection",
    "reexecuted_solver_patch_prediction",
    "strict_independent_patch_prediction",
    "baseline_prediction",
    "current_prediction",
    "v366_prediction",
    "v365_prediction",
    "v364_prediction",
    "v363_prediction",
    "v357_prediction",
    "v350_prediction",
]

RAW_OUTPUT_COLUMN_CANDIDATES = [
    "raw_output",
    "generated",
    "completion",
    "assistant",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_prediction_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted({"id", "answer"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    prediction_columns = [column for column in PREDICTION_COLUMN_CANDIDATES if column in frame.columns]
    if not prediction_columns:
        raise ValueError(
            f"{path} has no known prediction column; checked {PREDICTION_COLUMN_CANDIDATES}"
        )
    frame.attrs["prediction_columns"] = prediction_columns
    family_col = "type" if "type" in frame.columns else "task_type" if "task_type" in frame.columns else "family" if "family" in frame.columns else ""
    if family_col:
        frame["family_norm"] = frame[family_col].map(canonical_family)
    else:
        frame["family_norm"] = ""
    return frame


def builtin_metric_tests() -> list[dict[str, Any]]:
    cases = [
        {
            "name": "bit_exact_match",
            "answer": "10101010",
            "prediction": "10101010",
            "expected": True,
            "reason": "8-bit binary answers must match exactly.",
        },
        {
            "name": "bit_near_numeric_must_not_pass",
            "answer": "11100011",
            "prediction": "11100010",
            "expected": False,
            "reason": "Numeric tolerance would overcount binary strings; strict verifier must reject this.",
        },
        {
            "name": "numeric_tolerance_non_binary",
            "answer": "42",
            "prediction": "42.1",
            "expected": True,
            "reason": "Non-binary numeric answers use 1% relative tolerance.",
        },
        {
            "name": "binary_like_answer_exact",
            "answer": "101",
            "prediction": "101.0",
            "expected": False,
            "reason": "The project verifier mirrors the historical Kaggle verify_kaggle helper: all [01]+ answers are exact.",
        },
        {
            "name": "case_insensitive_symbolic",
            "answer": "Alice",
            "prediction": "alice",
            "expected": True,
            "reason": "Non-numeric symbolic answers compare case-insensitively.",
        },
    ]
    out: list[dict[str, Any]] = []
    for case in cases:
        observed = verify_answer(case["answer"], case["prediction"])
        out.append({**case, "observed": observed, "passed": observed is case["expected"]})
    return out


def weak_answer_audit(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    family_col = "family" if "family" in frame.columns else "type" if "type" in frame.columns else ""
    if family_col:
        frame["family_norm"] = frame[family_col].map(canonical_family)
    else:
        frame["family_norm"] = ""
    frame["answer_s"] = frame["answer"].fillna("").astype(str).str.strip()
    frame["binary_like_answer"] = frame["answer_s"].map(lambda value: bool(re.fullmatch(r"[01]+", value)))
    binary_like = frame[frame["binary_like_answer"]].copy()
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "binary_like_answer_by_family": {
            str(k): int(v) for k, v in binary_like.groupby("family_norm").size().to_dict().items()
        },
        "binary_like_equation_rows": binary_like[binary_like["family_norm"].eq("equation_transform")][
            ["id", "answer_s"]
        ].to_dict(orient="records"),
    }


def prediction_metric_audit(path: Path) -> dict[str, Any]:
    frame = read_prediction_csv(path)
    column_audits: list[dict[str, Any]] = []
    for prediction_column in frame.attrs["prediction_columns"]:
        working = frame.copy()
        working["strict_correct"] = working.apply(
            lambda row: verify_answer(row["answer"], row[prediction_column]), axis=1
        )
        working["permissive_correct"] = working.apply(
            lambda row: answers_equivalent(row["answer"], row[prediction_column]), axis=1
        )
        disagreement = working[working["strict_correct"] != working["permissive_correct"]].copy()
        column_audits.append(
            {
                "prediction_column": prediction_column,
                "strict_correct": int(working["strict_correct"].sum()),
                "strict_accuracy": float(working["strict_correct"].mean()) if len(working) else 0.0,
                "permissive_correct": int(working["permissive_correct"].sum()),
                "permissive_accuracy": float(working["permissive_correct"].mean()) if len(working) else 0.0,
                "strict_vs_permissive_disagreement_rows": int(len(disagreement)),
                "disagreement_by_family": {
                    str(k): int(v) for k, v in disagreement.groupby("family_norm").size().to_dict().items()
                },
                "disagreement_examples": disagreement[
                    ["id", "family_norm", "answer", prediction_column, "strict_correct", "permissive_correct"]
                ]
                .head(20)
                .rename(columns={prediction_column: "prediction"})
                .to_dict(orient="records"),
            }
        )
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "prediction_columns": list(frame.attrs["prediction_columns"]),
        "column_audits": column_audits,
    }


def raw_extraction_audit(path: Path, answer_csv: Path | None) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "answer" not in frame.columns and answer_csv is not None:
        answer_frame = pd.read_csv(answer_csv, dtype=str, keep_default_na=False)
        answer_cols = ["id", "answer"]
        family_cols = [column for column in ["type", "task_type", "family"] if column in answer_frame.columns]
        answer_frame = answer_frame[answer_cols + family_cols].copy()
        rename_family = {}
        for family_col in family_cols:
            if family_col in frame.columns:
                rename_family[family_col] = f"answer_{family_col}"
        if rename_family:
            answer_frame = answer_frame.rename(columns=rename_family)
        frame = frame.merge(answer_frame, on="id", how="left", validate="one_to_one")
    missing = sorted({"id", "answer"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns for raw extraction audit: {missing}")
    missing_answer_rows = int(frame["answer"].isna().sum()) if frame["answer"].isna().any() else 0
    if missing_answer_rows:
        raise ValueError(f"{path} has {missing_answer_rows} raw rows without joined answers")
    raw_columns = [column for column in RAW_OUTPUT_COLUMN_CANDIDATES if column in frame.columns]
    if not raw_columns:
        raise ValueError(f"{path} has no known raw output column; checked {RAW_OUTPUT_COLUMN_CANDIDATES}")
    raw_column = raw_columns[0]
    family_col = (
        "type"
        if "type" in frame.columns
        else "task_type"
        if "task_type" in frame.columns
        else "family"
        if "family" in frame.columns
        else "answer_type"
        if "answer_type" in frame.columns
        else "answer_task_type"
        if "answer_task_type" in frame.columns
        else "answer_family"
        if "answer_family" in frame.columns
        else ""
    )
    if family_col:
        frame["family_norm"] = frame[family_col].map(canonical_family)
    else:
        frame["family_norm"] = ""
    frame["simple_extracted"] = frame[raw_column].map(extract_final_answer)
    frame["expected_aware_extracted"] = frame.apply(
        lambda row: extract_final_answer_for_expected(row[raw_column], row["answer"]), axis=1
    )
    frame["simple_correct"] = frame.apply(
        lambda row: verify_answer(row["answer"], row["simple_extracted"]), axis=1
    )
    frame["expected_aware_correct"] = frame.apply(
        lambda row: verify_answer(row["answer"], row["expected_aware_extracted"]), axis=1
    )
    disagreement = frame[frame["simple_extracted"] != frame["expected_aware_extracted"]].copy()
    correctness_delta = frame[frame["simple_correct"] != frame["expected_aware_correct"]].copy()
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "raw_output_column": raw_column,
        "answer_csv": str(answer_csv) if answer_csv else "",
        "simple_correct": int(frame["simple_correct"].sum()),
        "expected_aware_correct": int(frame["expected_aware_correct"].sum()),
        "expected_aware_minus_simple_correct": int(frame["expected_aware_correct"].sum())
        - int(frame["simple_correct"].sum()),
        "extraction_disagreement_rows": int(len(disagreement)),
        "correctness_delta_rows": int(len(correctness_delta)),
        "correctness_delta_by_family": {
            str(k): int(v) for k, v in correctness_delta.groupby("family_norm").size().to_dict().items()
        },
        "correctness_delta_examples": correctness_delta[
            [
                "id",
                "family_norm",
                "answer",
                "simple_extracted",
                "expected_aware_extracted",
                "simple_correct",
                "expected_aware_correct",
            ]
        ]
        .head(20)
        .to_dict(orient="records"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tests = builtin_metric_tests()
    weak = weak_answer_audit(args.weak_csv) if args.weak_csv else {}
    prediction_audits = [prediction_metric_audit(path) for path in args.prediction_csv]
    raw_extraction_audits = [
        raw_extraction_audit(path, args.weak_csv) for path in args.raw_prediction_csv
    ]
    report = {
        "schema_version": "kg1_v449_acc_metric_integrity_v1",
        "generated_at_utc": utc_now(),
        "decision": "metric_path_ok" if all(item["passed"] for item in tests) else "metric_path_failed",
        "builtin_metric_tests": tests,
        "weak_answer_audit": weak,
        "prediction_metric_audits": prediction_audits,
        "raw_extraction_audits": raw_extraction_audits,
        "rule": (
            "Promotion ACC must be computed with src.competition_utils.verify_answer. "
            "answers_equivalent is diagnostic-only because it numerically overcounts binary bit strings. "
            "Expected-aware extraction may only disambiguate the last boxed payload; it must not select "
            "an earlier boxed answer because that would leak the validation answer into extraction."
        ),
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{args.label}_manifest.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("metric_integrity_manifest =", out_path, flush=True)
    print("metric_integrity_report =", json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not all(item["passed"] for item in tests):
        raise RuntimeError("builtin metric integrity tests failed")
    return report


def run_self_test() -> None:
    print("=== V449 ACC METRIC INTEGRITY SELF TEST START ===", flush=True)
    tests = builtin_metric_tests()
    payload = {
        "schema_version": "kg1_v449_acc_metric_integrity_self_test_v1",
        "builtin_metric_tests": tests,
        "passed": bool(all(item["passed"] for item in tests)),
    }
    print("v449_acc_metric_integrity_self_test =", json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["passed"]:
        raise RuntimeError("V449 ACC metric integrity self-test failed")
    symbolic = r"prefix \boxed{wrong} final \boxed{]\}\\!}"
    assert extract_final_answer_for_expected(symbolic, r"]}\!") == r"]}\!"
    earlier_correct_later_wrong = r"prefix \boxed{1010} final \boxed{1011}"
    assert extract_final_answer_for_expected(earlier_correct_later_wrong, "1010") == "1011"
    assert not verify_answer("1010", extract_final_answer_for_expected(earlier_correct_later_wrong, "1010"))
    print("=== V449 ACC METRIC INTEGRITY SELF TEST END ===", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run built-in strict metric checks and exit.")
    parser.add_argument("--weak-csv", type=Path, default=DEFAULT_WEAK_CSV)
    parser.add_argument("--prediction-csv", type=Path, action="append", default=[])
    parser.add_argument("--raw-prediction-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v449_acc_metric_integrity_audit/20260515T_cpu_gate"))
    parser.add_argument("--label", default="v449_acc_metric_integrity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-score weak/full prediction CSVs with the current label-free extractor.

This script exists to prevent stale remote evaluators from creating false
promotion or false rejection.  It uses only ``raw_output`` plus the current
``extract_final_answer``/``verify_answer`` path, then compares the result to a
baseline CSV and to any stored remote ``prediction``/``correct`` columns.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


FAMILY_COLS = ("family", "type", "task_type")


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def family_column(frame: pd.DataFrame) -> str:
    for name in FAMILY_COLS:
        if name in frame.columns:
            return name
    raise ValueError(f"CSV is missing family/type column. columns={list(frame.columns)}")


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}; columns={list(frame.columns)}")


def read_predictions(path: Path, *, require_raw_output: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    required = ["id", "answer"]
    if require_raw_output:
        required.append("raw_output")
    require_columns(frame, path, required)
    fam = family_column(frame)
    if fam != "family":
        frame = frame.rename(columns={fam: "family"})
    frame["id"] = frame["id"].astype(str)
    frame["family"] = frame["family"].astype(str)
    return frame


def current_rescore(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    require_columns(out, Path("<candidate>"), ["raw_output", "answer"])
    out["local_current_prediction"] = out["raw_output"].map(extract_final_answer)
    out["local_current_correct"] = out.apply(
        lambda row: bool(verify_answer(row["answer"], row["local_current_prediction"])),
        axis=1,
    )
    if "correct" in out.columns:
        out["stored_correct_bool"] = as_bool_series(out["correct"])
        out["stored_vs_local_correct_changed"] = out["stored_correct_bool"] != out["local_current_correct"]
    else:
        out["stored_correct_bool"] = False
        out["stored_vs_local_correct_changed"] = False
    if "prediction" in out.columns:
        out["stored_prediction_str"] = out["prediction"].astype(str)
        out["stored_vs_local_prediction_changed"] = (
            out["stored_prediction_str"] != out["local_current_prediction"].astype(str)
        )
    else:
        out["stored_prediction_str"] = ""
        out["stored_vs_local_prediction_changed"] = False
    return out


def baseline_correct(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "raw_output" in out.columns:
        out["baseline_prediction_current"] = out["raw_output"].map(extract_final_answer)
        out["baseline_correct_current"] = out.apply(
            lambda row: bool(verify_answer(row["answer"], row["baseline_prediction_current"])),
            axis=1,
        )
    elif "correct" in out.columns:
        out["baseline_correct_current"] = as_bool_series(out["correct"])
        out["baseline_prediction_current"] = out["prediction"].astype(str) if "prediction" in out.columns else ""
    elif "prediction" in out.columns:
        out["baseline_prediction_current"] = out["prediction"].astype(str)
        out["baseline_correct_current"] = out.apply(
            lambda row: bool(verify_answer(row["answer"], row["baseline_prediction_current"])),
            axis=1,
        )
    else:
        raise ValueError("baseline CSV needs raw_output, correct, or prediction")
    return out


def summarize(
    merged: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label: str,
    thresholds: dict[str, int],
    paths: dict[str, str],
) -> dict[str, Any]:
    total = int(candidate["local_current_correct"].sum())
    truncated = 0
    if "truncated" in candidate.columns:
        truncated = int(as_bool_series(candidate["truncated"]).sum())
    stored_correct = int(candidate["stored_correct_bool"].sum()) if "stored_correct_bool" in candidate.columns else None
    label_aware_correct = None
    if "label_aware_debug_correct" in candidate.columns:
        label_aware_correct = int(as_bool_series(candidate["label_aware_debug_correct"]).sum())
    summary: dict[str, Any] = {
        "label": label,
        "paths": paths,
        "rows": int(len(candidate)),
        "stored_remote_correct": stored_correct,
        "local_current_correct": total,
        "delta_local_minus_stored": None if stored_correct is None else int(total - stored_correct),
        "stored_vs_local_prediction_changed_rows": int(candidate["stored_vs_local_prediction_changed"].sum()),
        "stored_vs_local_correct_changed_rows": int(candidate["stored_vs_local_correct_changed"].sum()),
        "label_aware_debug_correct": label_aware_correct,
        "label_aware_minus_label_free_correct": (
            None if label_aware_correct is None else int(label_aware_correct - total)
        ),
        "truncated": truncated,
        "baseline_correct": int(merged["baseline_correct_current"].sum()),
        "delta_vs_baseline": int(total - merged["baseline_correct_current"].sum()),
        "by_family": {},
        "flips": {str(k): int(v) for k, v in merged["flip"].value_counts().sort_index().items()},
        "thresholds": thresholds,
    }
    for family, group in merged.groupby("family", dropna=False):
        summary["by_family"][str(family)] = {
            "rows": int(len(group)),
            "baseline_correct": int(group["baseline_correct_current"].sum()),
            "local_current_correct": int(group["local_current_correct"].sum()),
            "delta": int(group["local_current_correct"].sum() - group["baseline_correct_current"].sum()),
            "flips": {str(k): int(v) for k, v in group["flip"].value_counts().sort_index().items()},
            "prediction_changed_vs_baseline": int(group["prediction_changed_vs_baseline"].sum()),
        }

    equation = int(summary["by_family"].get("equation_transform", {}).get("local_current_correct", 0))
    bit = int(summary["by_family"].get("bit_manipulation", {}).get("local_current_correct", 0))
    blockers = []
    if total < thresholds["total_min"]:
        blockers.append(f"correct_lt_{thresholds['total_min']}")
    if equation < thresholds["equation_min"]:
        blockers.append(f"equation_lt_{thresholds['equation_min']}")
    if bit < thresholds["bit_min"]:
        blockers.append(f"bit_lt_{thresholds['bit_min']}")
    if truncated > thresholds["trunc_max"]:
        blockers.append(f"truncated_gt_{thresholds['trunc_max']}")
    if summary["label_aware_minus_label_free_correct"] not in (None, 0):
        blockers.append("label_aware_delta_nonzero")
    summary["decision"] = {
        "status": "promotion_candidate" if not blockers else "blocked",
        "blocking_reasons": blockers,
        "note": "Decision uses current label-free extraction only.",
    }
    return summary


def build_audit(
    candidate_csv: Path,
    baseline_csv: Path,
    output_dir: Path,
    *,
    label: str,
    thresholds: dict[str, int],
) -> dict[str, Any]:
    candidate = current_rescore(read_predictions(candidate_csv, require_raw_output=True))
    baseline = baseline_correct(read_predictions(baseline_csv, require_raw_output=False))
    if len(candidate) != len(baseline):
        raise ValueError(f"row count mismatch: candidate={len(candidate)} baseline={len(baseline)}")
    merged = baseline[
        ["id", "family", "answer", "baseline_prediction_current", "baseline_correct_current"]
    ].merge(
        candidate[
            [
                "id",
                "local_current_prediction",
                "local_current_correct",
                "raw_output",
                "stored_prediction_str",
                "stored_correct_bool",
                "stored_vs_local_prediction_changed",
                "stored_vs_local_correct_changed",
            ]
            + [column for column in ["completion_tokens", "finish_reason", "pred_type", "truncated"] if column in candidate.columns]
        ],
        on="id",
        how="inner",
    )
    if len(merged) != len(candidate):
        raise ValueError("candidate and baseline IDs do not match exactly")
    merged["flip"] = merged.apply(
        lambda row: "T->F"
        if row["baseline_correct_current"] and not row["local_current_correct"]
        else (
            "F->T"
            if (not row["baseline_correct_current"]) and row["local_current_correct"]
            else ("T->T" if row["baseline_correct_current"] else "F->F")
        ),
        axis=1,
    )
    merged["prediction_changed_vs_baseline"] = (
        merged["baseline_prediction_current"].astype(str) != merged["local_current_prediction"].astype(str)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = label.replace("/", "_").replace("\\", "_")
    summary = summarize(
        merged,
        candidate,
        label=label,
        thresholds=thresholds,
        paths={"candidate_csv": str(candidate_csv), "baseline_csv": str(baseline_csv), "output_dir": str(output_dir)},
    )
    merged.sort_values(["family", "id"]).to_csv(output_dir / f"{prefix}_full_audit.csv", index=False)
    merged[merged["flip"].isin(["T->F", "F->T"])].sort_values(["family", "flip", "id"]).to_csv(
        output_dir / f"{prefix}_flips.csv",
        index=False,
    )
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_rescore_") as tmp_raw:
        tmp = Path(tmp_raw)
        candidate = tmp / "candidate.csv"
        baseline = tmp / "baseline.csv"
        output = tmp / "out"
        pd.DataFrame(
            [
                {
                    "id": "symbolic_brace",
                    "answer": r"]}\!",
                    "family": "equation_transform",
                    "raw_output": r"Final answer: \boxed{]}\!}",
                    "prediction": "]",
                    "correct": False,
                    "truncated": False,
                },
                {
                    "id": "bit_ok",
                    "answer": "01101000",
                    "family": "bit_manipulation",
                    "raw_output": r"Final answer: \boxed{01101000}",
                    "prediction": "01101000",
                    "correct": True,
                    "truncated": False,
                },
            ]
        ).to_csv(candidate, index=False)
        pd.DataFrame(
            [
                {
                    "id": "symbolic_brace",
                    "answer": r"]}\!",
                    "family": "equation_transform",
                    "prediction": r"]}\!",
                    "correct": True,
                    "truncated": False,
                },
                {
                    "id": "bit_ok",
                    "answer": "01101000",
                    "family": "bit_manipulation",
                    "prediction": "01101000",
                    "correct": True,
                    "truncated": False,
                },
            ]
        ).to_csv(baseline, index=False)
        summary = build_audit(
            candidate,
            baseline,
            output,
            label="self_test",
            thresholds={"total_min": 2, "equation_min": 1, "bit_min": 1, "trunc_max": 0},
        )
        assert summary["stored_remote_correct"] == 1
        assert summary["local_current_correct"] == 2
        assert summary["delta_local_minus_stored"] == 1
        assert summary["by_family"]["equation_transform"]["local_current_correct"] == 1
        assert summary["decision"]["status"] == "promotion_candidate"
    print("kg1_rescore_predictions_label_free_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=False)
    parser.add_argument("--baseline-csv", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--total-min", type=int, default=196)
    parser.add_argument("--equation-min", type=int, default=60)
    parser.add_argument("--bit-min", type=int, default=136)
    parser.add_argument("--trunc-max", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.candidate_csv or not args.baseline_csv or not args.output_dir:
        raise SystemExit("--candidate-csv, --baseline-csv, and --output-dir are required unless --self-test is used")
    summary = build_audit(
        args.candidate_csv,
        args.baseline_csv,
        args.output_dir,
        label=args.label,
        thresholds={
            "total_min": args.total_min,
            "equation_min": args.equation_min,
            "bit_min": args.bit_min,
            "trunc_max": args.trunc_max,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

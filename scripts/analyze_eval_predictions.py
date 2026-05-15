#!/usr/bin/env python3
"""Analyze KG1 eval prediction CSVs without loading a model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_boxed_answers, extract_final_answer, verify_answer


def first_boxed_answer(text: object) -> str:
    matches = [item.strip() for item in extract_boxed_answers(str(text or "")) if item.strip()]
    return matches[0] if matches else "NOT_FOUND"


def early_window_answer(text: object, chars: int) -> str:
    return extract_final_answer(str(text or "")[:chars])


def count_boxed(text: object) -> int:
    return len(extract_boxed_answers(str(text or "")))


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "raw_output", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame


def add_alternatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["raw_chars"] = out["raw_output"].fillna("").astype(str).str.len()
    out["boxed_count"] = out["raw_output"].map(count_boxed)
    out["first_boxed_prediction"] = out["raw_output"].map(first_boxed_answer)
    out["early_512_prediction"] = out["raw_output"].map(lambda value: early_window_answer(value, 512))
    out["early_1024_prediction"] = out["raw_output"].map(lambda value: early_window_answer(value, 1024))
    out["early_2048_prediction"] = out["raw_output"].map(lambda value: early_window_answer(value, 2048))
    if "answer" in out.columns:
        out["official_correct"] = out.apply(lambda row: verify_answer(row["answer"], row["prediction"]), axis=1)
        out["first_boxed_correct"] = out.apply(lambda row: verify_answer(row["answer"], row["first_boxed_prediction"]), axis=1)
        for chars in (512, 1024, 2048):
            col = f"early_{chars}_prediction"
            out[f"early_{chars}_correct"] = out.apply(lambda row, name=col: verify_answer(row["answer"], row[name]), axis=1)
    return out


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "raw_chars_total": int(frame["raw_chars"].sum()),
        "raw_chars_mean": float(frame["raw_chars"].mean()) if len(frame) else 0.0,
        "boxed_count_mean": float(frame["boxed_count"].mean()) if len(frame) else 0.0,
    }
    if "completion_tokens" in frame.columns:
        summary["completion_tokens_total"] = int(frame["completion_tokens"].fillna(0).sum())
        summary["completion_tokens_mean"] = float(frame["completion_tokens"].fillna(0).mean()) if len(frame) else 0.0
    if "finish_reason" in frame.columns:
        truncated = frame["finish_reason"].fillna("").astype(str).eq("length")
        summary["truncated"] = int(truncated.sum())
        summary["truncation_rate"] = float(truncated.mean()) if len(frame) else 0.0
    for col in [
        "official_correct",
        "first_boxed_correct",
        "early_512_correct",
        "early_1024_correct",
        "early_2048_correct",
    ]:
        if col in frame.columns:
            summary[col] = int(frame[col].sum())
            summary[f"{col}_rate"] = float(frame[col].mean()) if len(frame) else 0.0
    family_col = "type" if "type" in frame.columns else "task_type" if "task_type" in frame.columns else ""
    if family_col and "official_correct" in frame.columns:
        per_family = []
        for family, group in frame.groupby(family_col, dropna=False):
            row: dict[str, Any] = {"family": str(family), "rows": int(len(group))}
            for col in [
                "official_correct",
                "first_boxed_correct",
                "early_512_correct",
                "early_1024_correct",
                "early_2048_correct",
            ]:
                if col in group.columns:
                    row[col] = int(group[col].sum())
            if "finish_reason" in group.columns:
                row["truncated"] = int(group["finish_reason"].fillna("").astype(str).eq("length").sum())
            per_family.append(row)
        summary["per_family"] = per_family
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="prediction_analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_alternatives(load_frame(args.predictions_csv))
    summary = summarize(frame)
    detail_path = args.output_dir / f"{args.label}_details.csv"
    summary_path = args.output_dir / f"{args.label}_summary.json"
    frame.to_csv(detail_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("prediction_analysis_summary =", json.dumps(summary, indent=2, sort_keys=True))
    print("prediction_analysis_details_csv =", detail_path)
    print("prediction_analysis_summary_json =", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

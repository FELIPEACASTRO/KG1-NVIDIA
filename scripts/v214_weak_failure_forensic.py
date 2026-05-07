#!/usr/bin/env python3
"""Forensic report for the rejected V214 weak evaluation.

This script does not train, submit, or call an LLM. It only reads evaluation
CSV/JSON artifacts and produces row-level and aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.competition_utils import classify_puzzle, verify_answer
except Exception:  # pragma: no cover - Colab fallback if src is absent.
    classify_puzzle = None

    def verify_answer(expected: object, observed: object) -> bool:
        return str(expected).strip().lower() == str(observed).strip().lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def boxed_count(text: object) -> int:
    return len(re.findall(r"\\boxed\s*\{", safe_str(text)))


def final_marker_count(text: object) -> int:
    return len(re.findall(r"\b(final answer|therefore|answer\s*:)\b", safe_str(text), re.I))


def line_stats(text: object) -> tuple[int, int, float]:
    lines = [line.strip() for line in safe_str(text).splitlines() if line.strip()]
    if not lines:
        return 0, 0, 0.0
    counts = Counter(lines)
    max_repeat = max(counts.values())
    return len(lines), max_repeat, max_repeat / len(lines)


def classify_failure(row: pd.Series) -> str:
    if parse_bool(row.get("correct", False)):
        return "CORRECT"
    if parse_bool(row.get("truncated", False)):
        return "TRUNCATED"
    if parse_bool(row.get("loop_suspected", False)):
        return "LOOP_OR_VERBOSITY"
    if int(row.get("boxed_count", 0) or 0) == 0:
        return "FORMAT_NO_BOXED"
    if safe_str(row.get("prediction")) in {"", "NOT_FOUND", "nan"}:
        return "FORMAT_NO_EXTRACTED_ANSWER"
    return "WRONG_NON_TRUNCATED"


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "id"})
    df["id"] = df["id"].astype(str)

    if "prompt" not in df.columns:
        for candidate in ("generated_prompt", "prompt_x", "prompt_y"):
            if candidate in df.columns:
                df["prompt"] = df[candidate].fillna("").astype(str)
                break
        else:
            df["prompt"] = ""

    if "type" not in df.columns:
        for candidate in ("task_type", "family", "pred_type"):
            if candidate in df.columns:
                df["type"] = df[candidate].fillna("").astype(str)
                break
        else:
            if classify_puzzle is not None:
                df["type"] = df["prompt"].map(classify_puzzle)
            else:
                df["type"] = "unknown"
    df["type"] = df["type"].fillna("unknown").astype(str).replace({"": "unknown"})

    for column in ("raw_output", "prediction", "answer", "finish_reason"):
        if column not in df.columns:
            df[column] = ""

    if "completion_tokens" not in df.columns:
        df["completion_tokens"] = 0
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).astype(int)

    if "correct" not in df.columns:
        df["correct"] = df.apply(lambda r: verify_answer(r.get("answer", ""), r.get("prediction", "")), axis=1)
    else:
        df["correct"] = df["correct"].map(parse_bool)

    if "truncated" not in df.columns:
        df["truncated"] = df["finish_reason"].fillna("").astype(str).eq("length")
    else:
        df["truncated"] = df["truncated"].map(parse_bool)

    df["raw_char_len"] = df["raw_output"].map(lambda value: len(safe_str(value)))
    df["boxed_count"] = df["raw_output"].map(boxed_count)
    df["final_marker_count"] = df["raw_output"].map(final_marker_count)
    stats = df["raw_output"].map(line_stats)
    df["line_count"] = stats.map(lambda item: item[0])
    df["max_repeated_line_count"] = stats.map(lambda item: item[1])
    df["max_repeated_line_ratio"] = stats.map(lambda item: item[2])
    df["loop_suspected"] = (
        df["truncated"]
        | df["completion_tokens"].ge(7000)
        | df["max_repeated_line_count"].ge(5)
        | (df["raw_char_len"].ge(4000) & df["max_repeated_line_ratio"].ge(0.30))
    )
    df["failure_bucket"] = df.apply(classify_failure, axis=1)
    return df


def quantile(series: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.quantile(q))


def per_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_type, group in df.groupby("type", dropna=False):
        total = int(len(group))
        correct = int(group["correct"].sum())
        truncated = int(group["truncated"].sum())
        rows.append(
            {
                "task_type": str(task_type),
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "truncated": truncated,
                "truncation_rate": truncated / total if total else 0.0,
                "no_boxed": int(group["boxed_count"].eq(0).sum()),
                "no_boxed_rate": float(group["boxed_count"].eq(0).mean()) if total else 0.0,
                "loop_suspected": int(group["loop_suspected"].sum()),
                "avg_completion_tokens": float(group["completion_tokens"].mean()) if total else 0.0,
                "p50_completion_tokens": quantile(group["completion_tokens"], 0.50),
                "p90_completion_tokens": quantile(group["completion_tokens"], 0.90),
                "p99_completion_tokens": quantile(group["completion_tokens"], 0.99),
                "wrong_non_truncated": int(group["failure_bucket"].eq("WRONG_NON_TRUNCATED").sum()),
                "format_no_boxed": int(group["failure_bucket"].eq("FORMAT_NO_BOXED").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["truncated", "total"], ascending=[False, False])


def failure_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(df)
    for bucket, group in df.groupby("failure_bucket", dropna=False):
        rows.append(
            {
                "failure_bucket": str(bucket),
                "rows": int(len(group)),
                "share": len(group) / total if total else 0.0,
                "avg_completion_tokens": float(group["completion_tokens"].mean()) if len(group) else 0.0,
                "truncated": int(group["truncated"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows"], ascending=False)


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_empty_"
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in view.itertuples(index=False):
        values = [safe_str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compare_baseline(v214: pd.DataFrame, baseline_path: Path | None) -> pd.DataFrame:
    if baseline_path is None or not baseline_path.exists():
        return pd.DataFrame()
    baseline = normalize_predictions(pd.read_csv(baseline_path))
    merged = v214.merge(
        baseline[["id", "correct", "prediction", "completion_tokens", "truncated"]].rename(
            columns={
                "correct": "baseline_correct",
                "prediction": "baseline_prediction",
                "completion_tokens": "baseline_completion_tokens",
                "truncated": "baseline_truncated",
            }
        ),
        on="id",
        how="left",
        validate="one_to_one",
    )
    merged["baseline_correct"] = merged["baseline_correct"].fillna(False).map(parse_bool)
    merged["delta_bucket"] = "same_wrong"
    merged.loc[merged["baseline_correct"] & merged["correct"], "delta_bucket"] = "same_correct"
    merged.loc[merged["baseline_correct"] & ~merged["correct"], "delta_bucket"] = "REGRESSED"
    merged.loc[~merged["baseline_correct"] & merged["correct"], "delta_bucket"] = "RECOVERED"
    return merged


def write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
    per_type: pd.DataFrame,
    buckets: pd.DataFrame,
    top_truncated: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# V214 Weak Failure Forensic",
        "",
        f"Generated UTC: {summary['generated_at_utc']}",
        "",
        "## Verdict",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- V214 weak: `{summary['v214_correct']}/{summary['rows']}` = `{summary['v214_accuracy']:.6f}`",
        f"- Protected V194 weak baseline: `{summary['baseline_weak_correct']}/{summary['baseline_weak_total']}`",
        f"- Weak delta vs baseline: `{summary['weak_delta_vs_baseline']}`",
        f"- Truncation: `{summary['truncated']}/{summary['rows']}` = `{summary['truncation_rate']:.6f}`",
        f"- Max possible full score if strong stays 632/632: `{summary['max_full_if_strong_default']}/947`",
        "",
        "## Gate Results",
        "",
        f"- Weak full-eval gate: `{summary['weak_gate_pass']}`",
        f"- Truncation gate: `{summary['truncation_gate_pass']}`",
        f"- Full eval should run: `{summary['full_eval_allowed']}`",
        "",
        "## Failure Buckets",
        "",
        dataframe_to_markdown(buckets),
        "",
        "## Per-Type Summary",
        "",
        dataframe_to_markdown(per_type),
        "",
        "## Top Truncated/Long Rows",
        "",
        dataframe_to_markdown(
            top_truncated[
                ["id", "type", "correct", "truncated", "completion_tokens", "boxed_count", "failure_bucket"]
            ],
            max_rows=20,
        ),
        "",
    ]
    if not comparison.empty:
        delta_counts = comparison["delta_bucket"].value_counts().rename_axis("delta_bucket").reset_index(name="rows")
        lines.extend(
            [
                "## Optional Baseline Comparison",
                "",
                dataframe_to_markdown(delta_counts),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--raw-predictions-csv", type=Path, default=None)
    parser.add_argument("--per-task-csv", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--baseline-predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v214_micro_weak")
    parser.add_argument("--baseline-weak-correct", type=int, default=190)
    parser.add_argument("--baseline-weak-total", type=int, default=315)
    parser.add_argument("--strong-default-correct", type=int, default=632)
    parser.add_argument("--full-gate", type=int, default=828)
    parser.add_argument("--full-preferred", type=int, default=830)
    parser.add_argument("--weak-full-gate", type=int, default=191)
    parser.add_argument("--trunc-gate", type=int, default=3)
    args = parser.parse_args()

    print("=== V214 WEAK FAILURE FORENSIC START ===", flush=True)
    print("predictions_csv =", args.predictions_csv, flush=True)
    print("raw_predictions_csv =", args.raw_predictions_csv, flush=True)
    print("per_task_csv =", args.per_task_csv, flush=True)
    print("report_json =", args.report_json, flush=True)
    print("baseline_predictions_csv =", args.baseline_predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    if not args.predictions_csv.exists():
        raise FileNotFoundError(f"missing predictions CSV: {args.predictions_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = read_json(args.report_json)
    df = normalize_predictions(pd.read_csv(args.predictions_csv))
    per_type = per_type_summary(df)
    buckets = failure_bucket_summary(df)
    comparison = compare_baseline(df, args.baseline_predictions_csv)

    rows = int(len(df))
    correct = int(df["correct"].sum())
    truncated = int(df["truncated"].sum())
    weak_delta = correct - args.baseline_weak_correct
    max_full = correct + args.strong_default_correct
    weak_gate_pass = correct >= args.weak_full_gate
    trunc_gate_pass = truncated <= args.trunc_gate
    full_eval_allowed = bool(weak_gate_pass and trunc_gate_pass and max_full >= args.full_gate)
    decision = "REJECT_V214_NO_FULL_EVAL"
    if full_eval_allowed:
        decision = "WEAK_GATE_PASS_FULL_EVAL_ALLOWED"

    summary = {
        "generated_at_utc": utc_now(),
        "label": args.label,
        "decision": decision,
        "rows": rows,
        "v214_correct": correct,
        "v214_accuracy": correct / rows if rows else 0.0,
        "baseline_weak_correct": args.baseline_weak_correct,
        "baseline_weak_total": args.baseline_weak_total,
        "weak_delta_vs_baseline": weak_delta,
        "truncated": truncated,
        "truncation_rate": truncated / rows if rows else 0.0,
        "max_full_if_strong_default": max_full,
        "full_gate": args.full_gate,
        "full_preferred": args.full_preferred,
        "weak_full_gate": args.weak_full_gate,
        "trunc_gate": args.trunc_gate,
        "weak_gate_pass": weak_gate_pass,
        "truncation_gate_pass": trunc_gate_pass,
        "full_eval_allowed": full_eval_allowed,
        "completion_tokens_total": int(df["completion_tokens"].sum()),
        "completion_tokens_p50": quantile(df["completion_tokens"], 0.50),
        "completion_tokens_p90": quantile(df["completion_tokens"], 0.90),
        "completion_tokens_p99": quantile(df["completion_tokens"], 0.99),
        "source_report": report_json,
        "inputs": {
            "predictions_csv": str(args.predictions_csv),
            "raw_predictions_csv": str(args.raw_predictions_csv) if args.raw_predictions_csv else "",
            "per_task_csv": str(args.per_task_csv) if args.per_task_csv else "",
            "report_json": str(args.report_json) if args.report_json else "",
            "baseline_predictions_csv": str(args.baseline_predictions_csv) if args.baseline_predictions_csv else "",
        },
    }

    label = args.label.replace("/", "_").replace("\\", "_")
    rows_path = args.output_dir / f"{label}_forensic_rows.csv"
    per_type_path = args.output_dir / f"{label}_forensic_per_type.csv"
    buckets_path = args.output_dir / f"{label}_forensic_failure_buckets.csv"
    top_truncated_path = args.output_dir / f"{label}_forensic_top_truncated.csv"
    comparison_path = args.output_dir / f"{label}_forensic_baseline_comparison.csv"
    summary_path = args.output_dir / f"{label}_forensic_summary.json"
    report_path = args.output_dir / f"{label}_forensic_report.md"

    top_truncated = df.sort_values(["truncated", "completion_tokens"], ascending=[False, False]).head(80)
    df.to_csv(rows_path, index=False)
    per_type.to_csv(per_type_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    top_truncated.to_csv(top_truncated_path, index=False)
    if not comparison.empty:
        comparison.to_csv(comparison_path, index=False)
    summary["outputs"] = {
        "rows_csv": str(rows_path),
        "per_type_csv": str(per_type_path),
        "failure_buckets_csv": str(buckets_path),
        "top_truncated_csv": str(top_truncated_path),
        "baseline_comparison_csv": str(comparison_path) if not comparison.empty else "",
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(
        report_path,
        summary=summary,
        per_type=per_type,
        buckets=buckets,
        top_truncated=top_truncated,
        comparison=comparison,
    )

    print("summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("rows_csv =", rows_path, flush=True)
    print("per_type_csv =", per_type_path, flush=True)
    print("failure_buckets_csv =", buckets_path, flush=True)
    print("top_truncated_csv =", top_truncated_path, flush=True)
    print("summary_json =", summary_path, flush=True)
    print("report_md =", report_path, flush=True)
    print("=== V214 WEAK FAILURE FORENSIC END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Solve-rate promotion gate for Nemotron LoRA candidates.

This is the score-facing gate: compare candidate vs baseline by decoded
answers, official answer verification, and per-family regressions. It supports
two modes:

1. CSV mode: compare existing prediction CSVs.
2. Adapter mode: run vLLM evaluation for baseline and candidate adapters.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_lora_adapter import evaluate_adapter, parse_seeds, resolve_base_model_path
from src.competition_utils import (
    OFFICIAL_INFERENCE_CONFIG,
    classify_puzzle,
    extract_boxed_answers,
    extract_closed_boxed_answers,
    extract_final_answer,
    has_unclosed_boxed_answer,
    verify_answer,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("id", "row_id"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns.to_list()[0])


def family_for_row(row: pd.Series) -> str:
    for key in ("type", "family", "task_family"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return classify_puzzle(str(row.get("prompt", "")))


def normalize_solution(solution_csv: Path, limit: int = 0) -> pd.DataFrame:
    solution = pd.read_csv(solution_csv)
    required = {"prompt", "answer"}
    missing = sorted(required - set(solution.columns))
    if missing:
        raise ValueError(f"solution CSV missing required columns: {missing}")
    id_col = row_id_column(solution)
    if id_col != "id":
        solution = solution.rename(columns={id_col: "id"})
    solution["id"] = solution["id"].astype(str)
    solution["family_gate"] = solution.apply(family_for_row, axis=1)
    if limit > 0:
        solution = solution.head(limit).copy()
    return solution


def predictions_from_csv(predictions_csv: Path, label: str, *, allow_prediction_only: bool = False) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_csv)
    id_col = row_id_column(predictions)
    if id_col != "id":
        predictions = predictions.rename(columns={id_col: "id"})
    predictions["id"] = predictions["id"].astype(str)

    if "raw_output" not in predictions.columns and "prediction" not in predictions.columns:
        raise ValueError(f"{label} predictions need a 'raw_output' column")
    if "raw_output" not in predictions.columns:
        if not allow_prediction_only:
            raise ValueError(
                f"{label} predictions need raw_output. "
                "Use --allow-prediction-only only for diagnostics, not promotion."
            )
        predictions["raw_output"] = predictions["prediction"].astype(str)
    if "prediction" not in predictions.columns:
        predictions["prediction"] = predictions["raw_output"].map(extract_final_answer)
    else:
        missing_prediction = predictions["prediction"].isna() | (predictions["prediction"].astype(str) == "")
        predictions.loc[missing_prediction, "prediction"] = predictions.loc[
            missing_prediction, "raw_output"
        ].map(extract_final_answer)
    optional = [
        column
        for column in ("finish_reason", "completion_tokens", "truncated", "truncated_bool", "was_truncated", "is_truncated")
        if column in predictions.columns
    ]
    return predictions[["id", "prediction", "raw_output", *optional]].copy()


TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
    "truncated",
}


def parse_boolish(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_intish(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def is_truncated_row(row: pd.Series, max_tokens: int) -> bool:
    finish_reason = str(row.get("finish_reason", "") or "").strip().lower()
    if finish_reason in TRUNCATED_FINISH_REASONS:
        return True
    if any(parse_boolish(row.get(key)) for key in ("truncated", "truncated_bool", "was_truncated", "is_truncated")):
        return True
    completion_tokens = parse_intish(row.get("completion_tokens"))
    return bool(completion_tokens is not None and completion_tokens >= int(max_tokens))


def score_predictions(
    solution: pd.DataFrame,
    predictions: pd.DataFrame,
    label: str,
    seed: int | None = None,
    *,
    max_tokens: int | None = None,
) -> pd.DataFrame:
    max_tokens = int(max_tokens or OFFICIAL_INFERENCE_CONFIG["max_tokens"])
    merged = solution.merge(predictions, on="id", how="left", validate="one_to_one")
    merged["label"] = label
    merged["seed"] = seed if seed is not None else 0
    if "raw_output" not in merged.columns:
        merged["raw_output"] = ""
    if "prediction" not in merged.columns:
        merged["prediction"] = merged["raw_output"].map(extract_final_answer)
    merged["prediction"] = merged["prediction"].fillna("NOT_FOUND").astype(str)
    merged["raw_output"] = merged["raw_output"].fillna("").astype(str)
    for column in ("finish_reason", "completion_tokens", "truncated", "truncated_bool", "was_truncated", "is_truncated"):
        if column not in merged.columns:
            merged[column] = ""
    merged["final_answer"] = merged.apply(
        lambda row: extract_final_answer(row["raw_output"]) if row["raw_output"] else row["prediction"],
        axis=1,
    )
    merged["boxed_marker_count"] = merged["raw_output"].map(lambda value: str(value).count(r"\boxed{"))
    merged["closed_boxed_count"] = merged["raw_output"].map(lambda value: len(extract_closed_boxed_answers(value)))
    merged["boxed_present"] = merged["raw_output"].map(lambda value: len(extract_boxed_answers(value)) > 0)
    merged["unclosed_boxed"] = merged["raw_output"].map(has_unclosed_boxed_answer)
    merged["exact_one_closed_boxed"] = (
        merged["boxed_marker_count"].eq(1)
        & merged["closed_boxed_count"].eq(1)
        & ~merged["unclosed_boxed"].astype(bool)
    )
    merged["boxed_valid"] = merged["exact_one_closed_boxed"]
    merged["truncated_detected"] = merged.apply(lambda row: is_truncated_row(row, max_tokens), axis=1)
    merged["correct"] = merged.apply(
        lambda row: verify_answer(str(row["answer"]), str(row["final_answer"])),
        axis=1,
    )
    return merged


def prediction_frames_from_adapters(
    solution: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    baseline_adapter: Path,
    candidate_adapter: Path,
    base_model_path: str,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    prediction_columns = [
        "id",
        "prediction",
        "raw_output",
        "finish_reason",
        "completion_tokens",
        "truncated",
    ]
    for seed in seeds:
        _, baseline_merged = evaluate_adapter(
            solution,
            questions,
            lora_path=str(baseline_adapter),
            base_model_path=base_model_path,
            config=OFFICIAL_INFERENCE_CONFIG,
            seed=seed,
        )
        _, candidate_merged = evaluate_adapter(
            solution,
            questions,
            lora_path=str(candidate_adapter),
            base_model_path=base_model_path,
            config=OFFICIAL_INFERENCE_CONFIG,
            seed=seed,
        )
        baseline_merged = baseline_merged.rename(columns={"type": "family_gate"})
        candidate_merged = candidate_merged.rename(columns={"type": "family_gate"})
        baseline_cols = [column for column in prediction_columns if column in baseline_merged.columns]
        candidate_cols = [column for column in prediction_columns if column in candidate_merged.columns]
        baseline_frames.append(
            score_predictions(
                solution,
                baseline_merged[baseline_cols],
                "baseline",
                seed,
                max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]),
            )
        )
        candidate_frames.append(
            score_predictions(
                solution,
                candidate_merged[candidate_cols],
                "candidate",
                seed,
                max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]),
            )
        )
    return pd.concat(baseline_frames, ignore_index=True), pd.concat(candidate_frames, ignore_index=True)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family, group in frame.groupby("family_gate"):
        by_family[str(family)] = {
            "rows": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()) if len(group) else 0.0,
            "boxed_format_rate": float(group["boxed_valid"].mean()) if len(group) else 0.0,
            "boxed_presence_rate": float(group["boxed_present"].mean()) if len(group) else 0.0,
            "truncated": int(group["truncated_detected"].sum()) if "truncated_detected" in group else 0,
        }

    seed_summaries = []
    for seed, group in frame.groupby("seed"):
        seed_summaries.append(
            {
                "seed": int(seed),
                "rows": int(len(group)),
                "correct": int(group["correct"].sum()),
                "accuracy": float(group["correct"].mean()) if len(group) else 0.0,
                "boxed_format_rate": float(group["boxed_valid"].mean()) if len(group) else 0.0,
                "boxed_presence_rate": float(group["boxed_present"].mean()) if len(group) else 0.0,
                "truncated": int(group["truncated_detected"].sum()) if "truncated_detected" in group else 0,
            }
        )
    accuracies = [item["accuracy"] for item in seed_summaries]
    return {
        "rows": int(len(frame)),
        "correct": int(frame["correct"].sum()),
        "accuracy": float(frame["correct"].mean()) if len(frame) else 0.0,
        "boxed_format_rate": float(frame["boxed_valid"].mean()) if len(frame) else 0.0,
        "accuracy_min_seed": float(min(accuracies)) if accuracies else 0.0,
        "accuracy_mean_seed": float(statistics.mean(accuracies)) if accuracies else 0.0,
        "accuracy_max_seed": float(max(accuracies)) if accuracies else 0.0,
        "truncated": int(frame["truncated_detected"].sum()) if "truncated_detected" in frame else 0,
        "boxed_presence_rate": float(frame["boxed_present"].mean()) if "boxed_present" in frame and len(frame) else 0.0,
        "by_seed": seed_summaries,
        "by_family": by_family,
    }


def compare(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    family_regression_tolerance: float,
    min_net_gain: float,
    min_boxed_rate: float,
    max_candidate_truncated: int,
) -> tuple[bool, list[str], dict[str, Any]]:
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    reasons: list[str] = []

    baseline_accuracy = float(baseline_summary["accuracy"])
    candidate_accuracy = float(candidate_summary["accuracy"])
    net_gain = candidate_accuracy - baseline_accuracy
    if net_gain <= min_net_gain:
        reasons.append(f"net_gain_not_above_threshold:{net_gain:.6f}<={min_net_gain:.6f}")

    if float(candidate_summary["boxed_format_rate"]) < min_boxed_rate:
        reasons.append(
            "candidate_exact_boxed_rate_below_threshold:"
            f"{candidate_summary['boxed_format_rate']:.6f}<{min_boxed_rate:.6f}"
        )
    if int(candidate_summary.get("truncated", 0)) > max_candidate_truncated:
        reasons.append(f"candidate_truncated_above_threshold:{candidate_summary.get('truncated', 0)}>{max_candidate_truncated}")

    baseline_families = baseline_summary["by_family"]
    candidate_families = candidate_summary["by_family"]
    family_deltas: dict[str, dict[str, float]] = {}
    for family in sorted(set(baseline_families) | set(candidate_families)):
        b_acc = float(baseline_families.get(family, {}).get("accuracy", 0.0))
        c_acc = float(candidate_families.get(family, {}).get("accuracy", 0.0))
        delta = c_acc - b_acc
        family_deltas[family] = {
            "baseline_accuracy": b_acc,
            "candidate_accuracy": c_acc,
            "delta": delta,
        }
        if delta < -family_regression_tolerance:
            reasons.append(
                f"family_regression:{family}:{c_acc:.6f}<{b_acc:.6f}-"
                f"{family_regression_tolerance:.6f}"
            )

    comparison = {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "net_gain": net_gain,
        "family_deltas": family_deltas,
    }
    return len(reasons) == 0, reasons, comparison


def write_failures(path: Path, baseline: pd.DataFrame, candidate: pd.DataFrame) -> None:
    cols = ["id", "family_gate", "answer", "final_answer", "correct"]
    b = baseline[cols].rename(columns={"final_answer": "baseline_final_answer", "correct": "baseline_correct"})
    c = candidate[cols].rename(columns={"final_answer": "candidate_final_answer", "correct": "candidate_correct"})
    merged = b.merge(c, on=["id", "family_gate", "answer"], how="outer")
    merged["regressed"] = merged["baseline_correct"].fillna(False) & ~merged["candidate_correct"].fillna(False)
    merged["improved"] = ~merged["baseline_correct"].fillna(False) & merged["candidate_correct"].fillna(False)
    merged.to_csv(path, index=False)


def run_self_test(solution_csv: Path, output_dir: Path, limit: int) -> int:
    if solution_csv.exists():
        solution = normalize_solution(solution_csv, limit=limit or 20)
    else:
        solution = pd.DataFrame(
            [
                {
                    "id": "self_bit",
                    "prompt": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
                    "answer": "10101010",
                },
                {
                    "id": "self_gravity",
                    "prompt": "In Alice's Wonderland, the gravitational constant has been secretly changed.",
                    "answer": "24.64",
                },
                {
                    "id": "self_numeral",
                    "prompt": "In Alice's Wonderland, numbers are secretly converted into a different numeral system.",
                    "answer": "XLVII",
                },
                {
                    "id": "self_cipher",
                    "prompt": "In Alice's Wonderland, secret encryption rules are used on text.",
                    "answer": "the queen smiles",
                },
                {
                    "id": "self_unit",
                    "prompt": "In Alice's Wonderland, a secret unit conversion is applied to measurements.",
                    "answer": "13.37",
                },
                {
                    "id": "self_equation",
                    "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
                    "answer": "}52",
                },
            ]
        )
        solution["family_gate"] = solution.apply(family_for_row, axis=1)
        if limit > 0:
            solution = solution.head(limit).copy()
    baseline_predictions = pd.DataFrame(
        {
            "id": solution["id"],
            "prediction": solution["answer"].astype(str),
            "raw_output": "\\boxed{" + solution["answer"].astype(str) + "}",
        }
    )
    candidate_predictions = baseline_predictions.copy()
    if len(candidate_predictions):
        candidate_predictions.loc[candidate_predictions.index[-1], "prediction"] = "INTENTIONAL_WRONG"
        candidate_predictions.loc[candidate_predictions.index[-1], "raw_output"] = "\\boxed{INTENTIONAL_WRONG}"
    baseline = score_predictions(solution, baseline_predictions, "baseline", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    candidate = score_predictions(solution, candidate_predictions, "candidate", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_only_csv = output_dir / "self_test_prediction_only.csv"
    baseline_predictions[["id", "prediction"]].to_csv(prediction_only_csv, index=False)
    prediction_only_blocked = False
    try:
        predictions_from_csv(prediction_only_csv, "self_test_prediction_only")
    except ValueError as exc:
        prediction_only_blocked = "raw_output" in str(exc)
    approved, reasons, comparison = compare(
        baseline,
        candidate,
        family_regression_tolerance=0.0,
        min_net_gain=0.0,
        min_boxed_rate=1.0,
        max_candidate_truncated=0,
    )
    candidate_multi_box = baseline_predictions.copy()
    if len(candidate_multi_box):
        candidate_multi_box.loc[candidate_multi_box.index[-1], "raw_output"] = (
            "\\boxed{INTENTIONAL_WRONG}\\n\\boxed{" + solution["answer"].astype(str).iloc[-1] + "}"
        )
    multi_baseline = score_predictions(solution, baseline_predictions, "baseline", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    multi_candidate = score_predictions(solution, candidate_multi_box, "candidate", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    multi_approved, multi_reasons, _ = compare(
        multi_baseline,
        multi_candidate,
        family_regression_tolerance=0.0,
        min_net_gain=0.0,
        min_boxed_rate=1.0,
        max_candidate_truncated=0,
    )
    candidate_truncated = baseline_predictions.copy()
    if len(candidate_truncated):
        candidate_truncated["finish_reason"] = "stop"
        candidate_truncated["completion_tokens"] = "8"
        candidate_truncated.loc[candidate_truncated.index[-1], "finish_reason"] = "max_tokens"
        candidate_truncated.loc[candidate_truncated.index[-1], "completion_tokens"] = str(
            int(OFFICIAL_INFERENCE_CONFIG["max_tokens"])
        )
    trunc_baseline = score_predictions(solution, baseline_predictions, "baseline", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    trunc_candidate = score_predictions(solution, candidate_truncated, "candidate", max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    trunc_approved, trunc_reasons, _ = compare(
        trunc_baseline,
        trunc_candidate,
        family_regression_tolerance=0.0,
        min_net_gain=0.0,
        min_boxed_rate=1.0,
        max_candidate_truncated=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "self_test": True,
        "approved": approved,
        "reasons": reasons,
        "prediction_only_blocked": prediction_only_blocked,
        "multi_box_blocked": bool(not multi_approved and any("boxed_rate" in reason for reason in multi_reasons)),
        "truncation_blocked": bool(not trunc_approved and any("truncated" in reason for reason in trunc_reasons)),
        "multi_box_reasons": multi_reasons,
        "truncation_reasons": trunc_reasons,
        "comparison": comparison,
    }
    (output_dir / "self_test_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_failures(output_dir / "self_test_row_deltas.csv", baseline, candidate)
    print(json.dumps(payload, indent=2))
    return 0 if not approved and reasons and prediction_only_blocked and payload["multi_box_blocked"] and payload["truncation_blocked"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-csv", type=Path, default=ROOT / "data" / "splits" / "val_public_proxy.csv")
    parser.add_argument("--questions-csv", type=Path, default=None)
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--candidate-predictions", type=Path, default=None)
    parser.add_argument("--baseline-adapter", type=Path, default=None)
    parser.add_argument("--candidate-adapter", type=Path, default=None)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--family-regression-tolerance", type=float, default=0.0)
    parser.add_argument("--min-net-gain", type=float, default=0.0)
    parser.add_argument("--min-boxed-rate", type=float, default=0.98)
    parser.add_argument("--max-candidate-truncated", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "solve_rate_gate")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--allow-prediction-only",
        action="store_true",
        help="Diagnostic only. Promotion requires raw_output from real generation.",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.solution_csv, args.output_dir, args.limit)

    solution = normalize_solution(args.solution_csv, args.limit)
    questions_csv = args.questions_csv or args.solution_csv
    questions = pd.read_csv(questions_csv)
    id_col = row_id_column(questions)
    if id_col != "id":
        questions = questions.rename(columns={id_col: "id"})
    questions["id"] = questions["id"].astype(str)
    if args.limit > 0:
        questions = questions[questions["id"].isin(set(solution["id"]))].copy()

    csv_mode = args.baseline_predictions is not None and args.candidate_predictions is not None
    adapter_mode = args.baseline_adapter is not None and args.candidate_adapter is not None
    if csv_mode == adapter_mode:
        raise SystemExit("Choose exactly one mode: prediction CSVs or adapter paths.")

    if csv_mode:
        baseline = score_predictions(
            solution,
            predictions_from_csv(
                args.baseline_predictions,  # type: ignore[arg-type]
                "baseline",
                allow_prediction_only=bool(args.allow_prediction_only),
            ),
            "baseline",
            max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]),
        )
        candidate = score_predictions(
            solution,
            predictions_from_csv(
                args.candidate_predictions,  # type: ignore[arg-type]
                "candidate",
                allow_prediction_only=bool(args.allow_prediction_only),
            ),
            "candidate",
            max_tokens=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]),
        )
    else:
        baseline, candidate = prediction_frames_from_adapters(
            solution,
            questions,
            baseline_adapter=args.baseline_adapter,  # type: ignore[arg-type]
            candidate_adapter=args.candidate_adapter,  # type: ignore[arg-type]
            base_model_path=resolve_base_model_path(args.base_model_path),
            seeds=parse_seeds(args.seeds),
        )

    approved, reasons, comparison = compare(
        baseline,
        candidate,
        family_regression_tolerance=args.family_regression_tolerance,
        min_net_gain=args.min_net_gain,
        min_boxed_rate=args.min_boxed_rate,
        max_candidate_truncated=args.max_candidate_truncated,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.json_output or args.output_dir / "solve_rate_gate_report.json"
    report = {
        "generated_at_utc": utc_now(),
        "status": "approve" if approved else "reject",
        "approved": approved,
        "reasons": reasons,
        "thresholds": {
            "family_regression_tolerance": args.family_regression_tolerance,
            "min_net_gain": args.min_net_gain,
            "min_boxed_rate": args.min_boxed_rate,
            "max_candidate_truncated": args.max_candidate_truncated,
        },
        "inputs": {
            "solution_csv": str(args.solution_csv),
            "questions_csv": str(questions_csv),
            "baseline_predictions": str(args.baseline_predictions) if args.baseline_predictions else "",
            "candidate_predictions": str(args.candidate_predictions) if args.candidate_predictions else "",
            "baseline_adapter": str(args.baseline_adapter) if args.baseline_adapter else "",
            "candidate_adapter": str(args.candidate_adapter) if args.candidate_adapter else "",
            "seeds": parse_seeds(args.seeds),
            "limit": args.limit,
        },
        "comparison": comparison,
    }
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_failures(args.output_dir / "solve_rate_row_deltas.csv", baseline, candidate)
    print(json.dumps(report, indent=2))
    return 0 if approved else 2


if __name__ == "__main__":
    raise SystemExit(main())

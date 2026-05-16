#!/usr/bin/env python3
"""Analyze V225 equation-only decode sweep outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def load_predictions(path: Path, candidate: str, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "prompt", "answer", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    if "type" in out.columns:
        out["family"] = out["type"].astype(str)
    elif "task_type" in out.columns:
        out["family"] = out["task_type"].astype(str)
    elif "family" in out.columns:
        out["family"] = out["family"].astype(str)
    else:
        out["family"] = out["prompt"].map(classify_puzzle)
    if "correct" in out.columns:
        out["correct_bool"] = out["correct"].map(parse_bool)
    else:
        out["correct_bool"] = out.apply(lambda item: verify_answer(item["answer"], item["prediction"]), axis=1)
    if "truncated" in out.columns:
        out["truncated_bool"] = out["truncated"].map(parse_bool)
    elif "finish_reason" in out.columns:
        out["truncated_bool"] = out["finish_reason"].fillna("").astype(str).eq("length")
    else:
        out["truncated_bool"] = False
    out["candidate"] = candidate
    out["variant"] = variant
    out["prediction_len"] = out["prediction"].fillna("").astype(str).str.len()
    return out


def load_v221_baseline(batch_summary_json: Path, baseline_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    batch = read_json(batch_summary_json)
    rows = batch.get("rows", [])
    for row in rows:
        if str(row.get("name")) != baseline_name:
            continue
        report_path = Path(str(row.get("report_json") or ""))
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = read_json(report_path)
        predictions_csv = Path(report.get("outputs", {}).get("predictions_csv", ""))
        if not predictions_csv.exists():
            raise FileNotFoundError(predictions_csv)
        frame = load_predictions(predictions_csv, baseline_name, "v221_default_baseline")
        return frame, {"report_json": str(report_path), "predictions_csv": str(predictions_csv), "summary": row}
    raise RuntimeError(f"baseline not found in V221 batch summary: {baseline_name}")


def discover_sweep_reports(sweep_root: Path) -> list[tuple[str, Path]]:
    reports: list[tuple[str, Path]] = []
    for variant_dir in sorted(sweep_root.glob("variant_*")):
        summary_json = variant_dir / "batch_candidate_summary.json"
        if not summary_json.exists():
            continue
        variant = variant_dir.name.replace("variant_", "", 1)
        reports.append((variant, summary_json))
    return reports


def load_sweep_predictions(sweep_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    meta: list[dict[str, Any]] = []
    for variant, summary_json in discover_sweep_reports(sweep_root):
        summary = read_json(summary_json)
        for row in summary.get("rows", []):
            if row.get("status") != "ok":
                meta.append({"variant": variant, "candidate": row.get("name"), "status": row.get("status"), "error": row.get("error", "")})
                continue
            report_path = Path(str(row.get("report_json") or ""))
            if not report_path.exists():
                raise FileNotFoundError(report_path)
            report = read_json(report_path)
            predictions_csv = Path(report.get("outputs", {}).get("predictions_csv", ""))
            if not predictions_csv.exists():
                raise FileNotFoundError(predictions_csv)
            candidate = str(row.get("name"))
            frames.append(load_predictions(predictions_csv, candidate, variant))
            meta.append(
                {
                    "variant": variant,
                    "candidate": candidate,
                    "status": "ok",
                    "report_json": str(report_path),
                    "predictions_csv": str(predictions_csv),
                    "rows": int(row.get("correct", 0) + (report.get("rows", 0) - row.get("correct", 0))),
                    "correct": int(row.get("correct", 0)),
                    "truncated": int(row.get("truncated", 0)),
                }
            )
    if not frames:
        raise RuntimeError(f"no sweep prediction frames found under {sweep_root}")
    return pd.concat(frames, ignore_index=True), meta


def summarize_equation_variants(
    baseline: pd.DataFrame,
    sweep: pd.DataFrame,
    thresholds: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    baseline_eq = baseline[baseline["family"] == "equation_transform"].copy()
    baseline_bit = baseline[baseline["family"] == "bit_manipulation"].copy()
    baseline_eq_correct = int(baseline_eq["correct_bool"].sum())
    baseline_bit_correct = int(baseline_bit["correct_bool"].sum())
    baseline_bit_truncated = int(baseline_bit["truncated_bool"].sum())
    baseline_lookup = baseline_eq[["id", "prediction", "correct_bool", "truncated_bool", "prediction_len"]].rename(
        columns={
            "prediction": "baseline_prediction",
            "correct_bool": "baseline_correct",
            "truncated_bool": "baseline_truncated",
            "prediction_len": "baseline_prediction_len",
        }
    )
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for (variant, candidate), group in sweep.groupby(["variant", "candidate"], sort=False):
        eq = group[group["family"] == "equation_transform"].copy()
        merged = baseline_lookup.merge(
            eq[["id", "prediction", "correct_bool", "truncated_bool", "prediction_len"]].rename(
                columns={
                    "prediction": "candidate_prediction",
                    "correct_bool": "candidate_correct",
                    "truncated_bool": "candidate_truncated",
                    "prediction_len": "candidate_prediction_len",
                }
            ),
            on="id",
            how="inner",
            validate="one_to_one",
        )
        merged.insert(0, "variant", variant)
        merged.insert(1, "candidate", candidate)
        merged["gained_vs_baseline_eq"] = ~merged["baseline_correct"] & merged["candidate_correct"]
        merged["lost_vs_baseline_eq"] = merged["baseline_correct"] & ~merged["candidate_correct"]
        merged["extra_truncated_vs_baseline_eq"] = ~merged["baseline_truncated"] & merged["candidate_truncated"]
        merged["prediction_len_delta"] = merged["candidate_prediction_len"] - merged["baseline_prediction_len"]
        detail_frames.append(merged)
        eq_correct = int(eq["correct_bool"].sum())
        eq_truncated = int(eq["truncated_bool"].sum())
        simulated_total = baseline_bit_correct + eq_correct
        simulated_truncated = baseline_bit_truncated + eq_truncated
        weak_gate = (
            simulated_total >= thresholds["total"]
            and eq_correct >= thresholds["equation_transform"]
            and baseline_bit_correct >= thresholds["bit_manipulation"]
            and simulated_truncated <= thresholds["truncated"]
        )
        rows.append(
            {
                "variant": variant,
                "candidate": candidate,
                "eq_rows": int(len(eq)),
                "eq_correct": eq_correct,
                "eq_truncated": eq_truncated,
                "eq_gain_vs_baseline": eq_correct - baseline_eq_correct,
                "eq_gained_rows": int(merged["gained_vs_baseline_eq"].sum()),
                "eq_lost_rows": int(merged["lost_vs_baseline_eq"].sum()),
                "eq_net_row_gain": int(merged["gained_vs_baseline_eq"].sum() - merged["lost_vs_baseline_eq"].sum()),
                "eq_extra_truncated": int(merged["extra_truncated_vs_baseline_eq"].sum()),
                "prediction_len_delta_mean": float(merged["prediction_len_delta"].mean()) if len(merged) else 0.0,
                "simulated_weak_total_with_baseline_bit": simulated_total,
                "simulated_weak_eq": eq_correct,
                "simulated_weak_bit": baseline_bit_correct,
                "simulated_weak_truncated": simulated_truncated,
                "weak_gate_pass_with_baseline_bit": weak_gate,
                "gate_total_gap": max(0, thresholds["total"] - simulated_total),
                "gate_eq_gap": max(0, thresholds["equation_transform"] - eq_correct),
                "gate_bit_gap": max(0, thresholds["bit_manipulation"] - baseline_bit_correct),
                "gate_trunc_gap": max(0, simulated_truncated - thresholds["truncated"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["weak_gate_pass_with_baseline_bit", "eq_correct", "eq_truncated", "eq_net_row_gain"],
        ascending=[False, False, True, False],
    )
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, detail


def choose_decision(summary: pd.DataFrame) -> dict[str, Any]:
    passing = summary[summary["weak_gate_pass_with_baseline_bit"]]
    if len(passing):
        best = passing.iloc[0].to_dict()
        return {
            "decision": "equation_decode_candidate_found_confirm_full_weak",
            "best_variant": str(best["variant"]),
            "best_candidate": str(best["candidate"]),
            "reason": (
                f"eq_correct={int(best['eq_correct'])}; "
                f"simulated_total={int(best['simulated_weak_total_with_baseline_bit'])}; "
                f"simulated_truncated={int(best['simulated_weak_truncated'])}"
            ),
            "next_action": "Run full weak confirmation for this adapter/config before full eval.",
        }
    best = summary.iloc[0].to_dict()
    return {
        "decision": "no_equation_decode_candidate_passed_weak_gate",
        "best_variant": str(best["variant"]),
        "best_candidate": str(best["candidate"]),
        "reason": (
            f"best_eq_correct={int(best['eq_correct'])}; "
            f"eq_gap={int(best['gate_eq_gap'])}; "
            f"total_gap={int(best['gate_total_gap'])}; "
            f"trunc_gap={int(best['gate_trunc_gap'])}"
        ),
        "next_action": "Build V226 targeted equation data/training or inspect gained/lost rows from V225.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v221-batch-summary-json", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="v217_final_existing")
    parser.add_argument("--label", default="v225_equation_decode_sweep")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=136)
    parser.add_argument("--weak-trunc-max", type=int, default=0)
    args = parser.parse_args()

    thresholds = {
        "total": int(args.weak_total_min),
        "equation_transform": int(args.weak_eq_min),
        "bit_manipulation": int(args.weak_bit_min),
        "truncated": int(args.weak_trunc_max),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(args.label)

    print("=== V225 EQUATION DECODE SWEEP ANALYSIS START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v221_batch_summary_json =", args.v221_batch_summary_json, flush=True)
    print("sweep_root =", args.sweep_root, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("baseline =", args.baseline, flush=True)
    print("thresholds =", json.dumps(thresholds, indent=2, sort_keys=True), flush=True)

    baseline_frame, baseline_meta = load_v221_baseline(args.v221_batch_summary_json, args.baseline)
    sweep_frame, sweep_meta = load_sweep_predictions(args.sweep_root)
    shared_ids = set(baseline_frame["id"].astype(str)) & set(sweep_frame["id"].astype(str))
    print("baseline_meta =", json.dumps(baseline_meta, indent=2, sort_keys=True), flush=True)
    print("sweep_meta_count =", len(sweep_meta), flush=True)
    print("shared_id_count =", len(shared_ids), flush=True)
    if len(shared_ids) < 150:
        raise RuntimeError(f"shared equation id count too low: {len(shared_ids)}")
    # Keep the full weak baseline so the simulated weak score preserves the
    # original 160 bit-manipulation rows. Only the sweep frame is equation-only.
    sweep_frame = sweep_frame[sweep_frame["id"].isin(shared_ids)].copy()

    summary, detail = summarize_equation_variants(baseline_frame, sweep_frame, thresholds)
    decision = choose_decision(summary)

    paths = {
        "variant_summary_csv": args.output_dir / f"{prefix}_variant_summary.csv",
        "row_detail_csv": args.output_dir / f"{prefix}_row_detail.csv",
        "sweep_predictions_long_csv": args.output_dir / f"{prefix}_sweep_predictions_long.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary.to_csv(paths["variant_summary_csv"], index=False)
    detail.to_csv(paths["row_detail_csv"], index=False)
    sweep_frame.to_csv(paths["sweep_predictions_long_csv"], index=False)
    manifest = {
        "generated_at_utc": utc_now(),
        "thresholds": thresholds,
        "baseline": args.baseline,
        "baseline_meta": baseline_meta,
        "sweep_meta": sweep_meta,
        "variant_summary": summary.to_dict(orient="records"),
        "decision": decision,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("variant_summary =", summary.to_string(index=False), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V225 EQUATION DECODE SWEEP ANALYSIS END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

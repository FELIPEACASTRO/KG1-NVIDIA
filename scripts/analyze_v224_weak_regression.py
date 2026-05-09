#!/usr/bin/env python3
"""V224 weak-regression forensic analysis.

This script is CPU-only. It reads prediction/report artifacts already produced
by V221/V223 weak evaluations and explains whether V223 should be kept,
rolled back, or used only as a source of examples for a future run.
"""

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

from src.competition_utils import classify_puzzle, extract_final_answer, verify_answer  # noqa: E402


WEAK_FAMILIES = ("bit_manipulation", "equation_transform")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def resolve_report_predictions(report_path: Path) -> Path:
    report = read_json(report_path)
    output_path = report.get("outputs", {}).get("predictions_csv", "")
    if output_path and Path(output_path).exists():
        return Path(output_path)
    matches = sorted(report_path.parent.glob("*_predictions.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"could not resolve predictions csv from report: {report_path}")


def resolve_predictions_csv_from_v221_row(row: dict[str, Any]) -> Path:
    report_path = Path(str(row.get("report_json") or ""))
    if report_path.exists():
        return resolve_report_predictions(report_path)
    report_parent = report_path.parent
    if report_parent.exists():
        matches = sorted(report_parent.glob("*_predictions.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not resolve predictions_csv for candidate {row.get('name')!r}")


def candidate_specs_from_v221(batch_summary_json: Path) -> list[dict[str, str]]:
    if not batch_summary_json.exists():
        return []
    batch = read_json(batch_summary_json)
    specs: list[dict[str, str]] = []
    for row in batch.get("rows", []):
        if row.get("status") != "ok":
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        path = resolve_predictions_csv_from_v221_row(row)
        specs.append(
            {
                "name": name,
                "kind": "v221_candidate",
                "predictions_csv": str(path),
                "report_json": str(row.get("report_json") or ""),
            }
        )
    return specs


def candidate_spec_from_report(name: str, report_path: Path, kind: str) -> dict[str, str] | None:
    if not report_path.exists():
        return None
    predictions_csv = resolve_report_predictions(report_path)
    return {
        "name": name,
        "kind": kind,
        "predictions_csv": str(predictions_csv),
        "report_json": str(report_path),
    }


def parse_extra_candidate(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise ValueError("--extra-candidate must use NAME=PATH")
    name, value = raw.split("=", 1)
    name = name.strip()
    path = Path(value.strip())
    if not name:
        raise ValueError("--extra-candidate name is empty")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        predictions_csv = resolve_report_predictions(path)
        return {"name": name, "kind": "extra_report", "predictions_csv": str(predictions_csv), "report_json": str(path)}
    return {"name": name, "kind": "extra_predictions_csv", "predictions_csv": str(path), "report_json": ""}


def load_predictions(spec: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(spec["predictions_csv"])
    frame = pd.read_csv(path)
    required = {"id", "prompt", "answer", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    family_col = first_existing_column(out, ["type", "task_type", "family"])
    if family_col:
        out["family"] = out[family_col].astype(str)
    else:
        out["family"] = out["prompt"].map(classify_puzzle)
    if "truncated" in out.columns:
        out["truncated_bool"] = out["truncated"].map(parse_bool)
    elif "finish_reason" in out.columns:
        out["truncated_bool"] = out["finish_reason"].fillna("").astype(str).eq("length")
    else:
        out["truncated_bool"] = False
    out["correct_bool"] = out.apply(lambda item: verify_answer(item["answer"], item["prediction"]), axis=1)
    out["extracted_final_answer"] = out["prediction"].map(extract_final_answer)
    out["prediction_len"] = out["prediction"].fillna("").astype(str).str.len()
    out["boxed_count"] = out["prediction"].fillna("").astype(str).str.count(r"\\boxed")
    out["candidate"] = spec["name"]
    out["candidate_kind"] = spec["kind"]
    mismatch = 0
    if "correct" in out.columns:
        mismatch = int((out["correct"].map(parse_bool) != out["correct_bool"]).sum())
    meta = {
        "candidate": spec["name"],
        "kind": spec["kind"],
        "predictions_csv": str(path),
        "rows": int(len(out)),
        "correct_mismatch_vs_csv": mismatch,
    }
    return out, meta


def candidate_summary(long_df: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_df.groupby("candidate", sort=False):
        by_family = group.groupby("family")["correct_bool"].sum().to_dict()
        correct = int(group["correct_bool"].sum())
        truncated = int(group["truncated_bool"].sum())
        pred_len = group["prediction_len"]
        rows.append(
            {
                "candidate": candidate,
                "kind": str(group["candidate_kind"].iloc[0]),
                "rows": int(len(group)),
                "correct": correct,
                "accuracy": float(correct / len(group)) if len(group) else 0.0,
                "equation_transform_correct": int(by_family.get("equation_transform", 0)),
                "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
                "truncated": truncated,
                "truncation_rate": float(truncated / len(group)) if len(group) else 0.0,
                "prediction_len_mean": float(pred_len.mean()) if len(group) else 0.0,
                "prediction_len_p50": float(pred_len.quantile(0.50)) if len(group) else 0.0,
                "prediction_len_p95": float(pred_len.quantile(0.95)) if len(group) else 0.0,
                "prediction_len_max": int(pred_len.max()) if len(group) else 0,
                "boxed_rate": float((group["boxed_count"] > 0).mean()) if len(group) else 0.0,
                "gate_total_gap": max(0, thresholds["total"] - correct),
                "gate_eq_gap": max(0, thresholds["equation_transform"] - int(by_family.get("equation_transform", 0))),
                "gate_bit_gap": max(0, thresholds["bit_manipulation"] - int(by_family.get("bit_manipulation", 0))),
                "gate_trunc_gap": max(0, truncated - thresholds["truncated"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[False, False, False, True],
    )


def per_family_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, family), group in long_df.groupby(["candidate", "family"], sort=False):
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "rows": int(len(group)),
                "correct": int(group["correct_bool"].sum()),
                "truncated": int(group["truncated_bool"].sum()),
                "prediction_len_p95": float(group["prediction_len"].quantile(0.95)) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def pairwise_vs_baseline(long_df: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline not in set(long_df["candidate"].astype(str)):
        raise ValueError(f"baseline candidate not found: {baseline}")
    baseline_df = long_df[long_df["candidate"] == baseline][
        ["id", "family", "answer", "prompt", "prediction", "correct_bool", "truncated_bool", "prediction_len"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "correct_bool": "baseline_correct",
            "truncated_bool": "baseline_truncated",
            "prediction_len": "baseline_prediction_len",
        }
    )
    rows: list[pd.DataFrame] = []
    for candidate, candidate_df in long_df.groupby("candidate", sort=False):
        if candidate == baseline:
            continue
        current = candidate_df[
            ["id", "prediction", "correct_bool", "truncated_bool", "prediction_len", "extracted_final_answer"]
        ].rename(
            columns={
                "prediction": "candidate_prediction",
                "correct_bool": "candidate_correct",
                "truncated_bool": "candidate_truncated",
                "prediction_len": "candidate_prediction_len",
                "extracted_final_answer": "candidate_extracted_final_answer",
            }
        )
        merged = baseline_df.merge(current, on="id", how="inner", validate="one_to_one")
        merged.insert(1, "candidate", candidate)
        merged["lost_vs_baseline"] = merged["baseline_correct"] & ~merged["candidate_correct"]
        merged["gained_vs_baseline"] = ~merged["baseline_correct"] & merged["candidate_correct"]
        merged["candidate_extra_truncated"] = ~merged["baseline_truncated"] & merged["candidate_truncated"]
        merged["prediction_len_delta"] = merged["candidate_prediction_len"] - merged["baseline_prediction_len"]
        rows.append(merged)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    if not detail.empty:
        for (candidate, family), group in detail.groupby(["candidate", "family"], sort=False):
            summary_rows.append(
                {
                    "candidate": candidate,
                    "family": family,
                    "rows": int(len(group)),
                    "lost_vs_baseline": int(group["lost_vs_baseline"].sum()),
                    "gained_vs_baseline": int(group["gained_vs_baseline"].sum()),
                    "net_gain_vs_baseline": int(group["gained_vs_baseline"].sum() - group["lost_vs_baseline"].sum()),
                    "candidate_extra_truncated": int(group["candidate_extra_truncated"].sum()),
                    "prediction_len_delta_mean": float(group["prediction_len_delta"].mean()),
                    "prediction_len_delta_p95": float(group["prediction_len_delta"].quantile(0.95)),
                }
            )
    return pd.DataFrame(summary_rows), detail


def oracle_summary(long_df: pd.DataFrame, thresholds: dict[str, int], excluded: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded or set()
    rows: list[dict[str, Any]] = []
    for row_id, group in long_df[~long_df["candidate"].isin(excluded)].groupby("id", sort=False):
        non_trunc_correct = group[group["correct_bool"] & ~group["truncated_bool"]]
        any_correct = group[group["correct_bool"]]
        selected = non_trunc_correct.iloc[0] if len(non_trunc_correct) else (any_correct.iloc[0] if len(any_correct) else group.iloc[0])
        rows.append(
            {
                "id": row_id,
                "family": selected["family"],
                "chosen_candidate": selected["candidate"],
                "oracle_correct": bool(selected["correct_bool"]),
                "oracle_truncated": bool(selected["truncated_bool"]),
            }
        )
    frame = pd.DataFrame(rows)
    correct = int(frame["oracle_correct"].sum())
    truncated = int(frame["oracle_truncated"].sum())
    by_family = frame.groupby("family")["oracle_correct"].sum().to_dict()
    result = {
        "excluded": sorted(excluded),
        "rows": int(len(frame)),
        "correct": correct,
        "equation_transform_correct": int(by_family.get("equation_transform", 0)),
        "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
        "truncated": truncated,
    }
    result["gate_pass"] = (
        result["correct"] >= thresholds["total"]
        and result["equation_transform_correct"] >= thresholds["equation_transform"]
        and result["bit_manipulation_correct"] >= thresholds["bit_manipulation"]
        and result["truncated"] <= thresholds["truncated"]
    )
    return result


def choose_decision(summary_df: pd.DataFrame, pairwise_summary: pd.DataFrame, baseline: str, v223_name: str) -> dict[str, Any]:
    by_name = {str(row.candidate): row for row in summary_df.itertuples(index=False)}
    if v223_name not in by_name:
        return {"decision": "v223_candidate_missing", "reason": f"{v223_name} was not loaded"}
    v223 = by_name[v223_name]
    baseline_row = by_name.get(baseline)
    if baseline_row is None:
        return {"decision": "baseline_missing", "reason": f"{baseline} was not loaded"}
    reason_parts = []
    if int(v223.correct) < int(baseline_row.correct):
        reason_parts.append(f"v223_correct_drop={int(baseline_row.correct) - int(v223.correct)}")
    if int(v223.truncated) > 3:
        reason_parts.append(f"v223_truncated={int(v223.truncated)}")
    if int(v223.equation_transform_correct) < int(baseline_row.equation_transform_correct):
        reason_parts.append(
            f"v223_equation_drop={int(baseline_row.equation_transform_correct) - int(v223.equation_transform_correct)}"
        )
    v223_pair = pairwise_summary[pairwise_summary["candidate"] == v223_name] if not pairwise_summary.empty else pd.DataFrame()
    total_lost = int(v223_pair["lost_vs_baseline"].sum()) if not v223_pair.empty else 0
    total_gained = int(v223_pair["gained_vs_baseline"].sum()) if not v223_pair.empty else 0
    if total_lost > total_gained:
        reason_parts.append(f"row_level_net_loss={total_lost - total_gained}")
    if reason_parts:
        return {
            "decision": "reject_v223_adapter_and_rollback",
            "reason": "; ".join(reason_parts),
            "next_action": "Use V224 outputs to build V225 rollback/router or decode-only rescue; do not full-eval V223.",
        }
    return {
        "decision": "v223_not_worse_than_baseline_review_manually",
        "reason": "summary did not show a baseline regression",
        "next_action": "Manual inspect before full eval.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v221-batch-summary-json", type=Path, default=Path(""))
    parser.add_argument("--v223-report-json", type=Path, default=Path(""))
    parser.add_argument("--extra-candidate", action="append", default=[], help="Additional candidate as NAME=predictions.csv or NAME=report.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v224_weak_regression")
    parser.add_argument("--baseline", default="v217_final_existing")
    parser.add_argument("--v223-name", default="v223_eqrescue_from_v217_lr1e8_s12")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=133)
    parser.add_argument("--weak-trunc-max", type=int, default=3)
    args = parser.parse_args()

    thresholds = {
        "total": int(args.weak_total_min),
        "equation_transform": int(args.weak_eq_min),
        "bit_manipulation": int(args.weak_bit_min),
        "truncated": int(args.weak_trunc_max),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(args.label)

    print("=== V224 WEAK REGRESSION SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v221_batch_summary_json =", args.v221_batch_summary_json, flush=True)
    print("v223_report_json =", args.v223_report_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("baseline =", args.baseline, flush=True)
    print("v223_name =", args.v223_name, flush=True)
    print("thresholds =", json.dumps(thresholds, indent=2, sort_keys=True), flush=True)

    specs = []
    if str(args.v221_batch_summary_json):
        specs.extend(candidate_specs_from_v221(args.v221_batch_summary_json))
    if str(args.v223_report_json):
        v223_spec = candidate_spec_from_report(args.v223_name, args.v223_report_json, "v223_report")
        if v223_spec:
            specs.append(v223_spec)
    for raw in args.extra_candidate:
        specs.append(parse_extra_candidate(raw))
    if not specs:
        raise RuntimeError("no candidate artifacts were loaded")

    dedup: dict[str, dict[str, str]] = {}
    for spec in specs:
        dedup[spec["name"]] = spec
    specs = list(dedup.values())
    print("candidate_specs =", json.dumps(specs, indent=2, sort_keys=True), flush=True)

    frames: list[pd.DataFrame] = []
    load_meta: list[dict[str, Any]] = []
    for spec in specs:
        frame, meta = load_predictions(spec)
        print("loaded_candidate =", json.dumps(meta, sort_keys=True), flush=True)
        frames.append(frame)
        load_meta.append(meta)
    long_df = pd.concat(frames, ignore_index=True)
    id_sets = {meta["candidate"]: set(frame["id"].astype(str)) for meta, frame in zip(load_meta, frames)}
    shared_ids = set.intersection(*id_sets.values())
    print("candidate_count =", len(frames), flush=True)
    print("shared_row_count =", len(shared_ids), flush=True)
    if len(shared_ids) < 300:
        raise RuntimeError(f"shared weak row count too low: {len(shared_ids)}")
    long_df = long_df[long_df["id"].isin(shared_ids)].copy()

    summary_df = candidate_summary(long_df, thresholds)
    family_df = per_family_summary(long_df)
    pair_summary_df, pair_detail_df = pairwise_vs_baseline(long_df, args.baseline)
    oracle_all = oracle_summary(long_df, thresholds)
    oracle_without_v223 = oracle_summary(long_df, thresholds, excluded={args.v223_name})
    decision = choose_decision(summary_df, pair_summary_df, args.baseline, args.v223_name)

    v223_regression_df = pd.DataFrame()
    if not pair_detail_df.empty and args.v223_name in set(pair_detail_df["candidate"].astype(str)):
        v223_regression_df = pair_detail_df[
            (pair_detail_df["candidate"] == args.v223_name)
            & (
                pair_detail_df["lost_vs_baseline"]
                | pair_detail_df["gained_vs_baseline"]
                | pair_detail_df["candidate_extra_truncated"]
            )
        ].sort_values(["family", "lost_vs_baseline", "candidate_extra_truncated", "gained_vs_baseline"], ascending=[True, False, False, False])

    paths = {
        "candidate_summary_csv": args.output_dir / f"{prefix}_candidate_summary.csv",
        "per_family_summary_csv": args.output_dir / f"{prefix}_per_family_summary.csv",
        "pairwise_vs_baseline_csv": args.output_dir / f"{prefix}_pairwise_vs_{safe_name(args.baseline)}.csv",
        "pairwise_vs_baseline_summary_csv": args.output_dir / f"{prefix}_pairwise_vs_{safe_name(args.baseline)}_summary.csv",
        "v223_regression_cases_csv": args.output_dir / f"{prefix}_v223_regression_cases.csv",
        "all_predictions_long_csv": args.output_dir / f"{prefix}_all_predictions_long.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary_df.to_csv(paths["candidate_summary_csv"], index=False)
    family_df.to_csv(paths["per_family_summary_csv"], index=False)
    pair_detail_df.to_csv(paths["pairwise_vs_baseline_csv"], index=False)
    pair_summary_df.to_csv(paths["pairwise_vs_baseline_summary_csv"], index=False)
    v223_regression_df.to_csv(paths["v223_regression_cases_csv"], index=False)
    long_df.to_csv(paths["all_predictions_long_csv"], index=False)

    manifest = {
        "generated_at_utc": utc_now(),
        "thresholds": thresholds,
        "baseline": args.baseline,
        "v223_name": args.v223_name,
        "load_meta": load_meta,
        "candidate_summary": summary_df.to_dict(orient="records"),
        "per_family_summary": family_df.to_dict(orient="records"),
        "pairwise_vs_baseline_summary": pair_summary_df.to_dict(orient="records"),
        "oracle_all_candidates": oracle_all,
        "oracle_without_v223": oracle_without_v223,
        "v223_regression_case_rows": int(len(v223_regression_df)),
        "decision": decision,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("candidate_summary =", summary_df.to_string(index=False), flush=True)
    print("per_family_summary =", family_df.to_string(index=False), flush=True)
    print("pairwise_vs_baseline_summary =", pair_summary_df.to_string(index=False), flush=True)
    print("oracle_all_candidates =", json.dumps(oracle_all, indent=2, sort_keys=True), flush=True)
    print("oracle_without_v223 =", json.dumps(oracle_without_v223, indent=2, sort_keys=True), flush=True)
    print("v223_regression_case_rows =", len(v223_regression_df), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V224 WEAK REGRESSION SCRIPT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

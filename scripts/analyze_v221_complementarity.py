#!/usr/bin/env python3
"""Analyze V221 weak candidate prediction complementarity.

This is CPU-only and reads the prediction CSVs already produced by V221. It
does not load a model, does not run full evaluation, and does not submit.
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

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


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


def resolve_predictions_csv(row: dict[str, Any]) -> Path:
    report_path = Path(str(row.get("report_json") or ""))
    if report_path.exists():
        report = read_json(report_path)
        output_path = report.get("outputs", {}).get("predictions_csv", "")
        if output_path:
            path = Path(output_path)
            if path.exists():
                return path
        matches = sorted(report_path.parent.glob("*_predictions.csv"))
        if matches:
            return matches[0]
    adapter_dir = Path(str(row.get("report_json") or "")).parent
    if adapter_dir.exists():
        matches = sorted(adapter_dir.glob("*_predictions.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not resolve predictions_csv for candidate {row.get('name')!r}")


def load_candidate_predictions(candidate: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_predictions_csv(candidate)
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
    if "correct" in out.columns:
        original_correct = out["correct"].map(parse_bool)
        mismatch = int((original_correct != out["correct_bool"]).sum())
    else:
        mismatch = 0
    out["candidate"] = str(candidate["name"])
    meta = {
        "candidate": str(candidate["name"]),
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
        rows.append(
            {
                "candidate": candidate,
                "rows": int(len(group)),
                "correct": correct,
                "accuracy": float(correct / len(group)) if len(group) else 0.0,
                "equation_transform_correct": int(by_family.get("equation_transform", 0)),
                "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
                "truncated": truncated,
                "truncation_rate": float(truncated / len(group)) if len(group) else 0.0,
                "gate_total_gap": max(0, thresholds["total"] - correct),
                "gate_eq_gap": max(0, thresholds["equation_transform"] - int(by_family.get("equation_transform", 0))),
                "gate_bit_gap": max(0, thresholds["bit_manipulation"] - int(by_family.get("bit_manipulation", 0))),
                "gate_trunc_gap": max(0, truncated - thresholds["truncated"]),
            }
        )
    return pd.DataFrame(rows)


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
            }
        )
    return pd.DataFrame(rows)


def build_correctness_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        long_df[["id", "family", "answer", "prompt"]]
        .drop_duplicates(subset=["id"])
        .sort_values(["family", "id"])
        .reset_index(drop=True)
    )
    correct = long_df.pivot_table(index="id", columns="candidate", values="correct_bool", aggfunc="first")
    trunc = long_df.pivot_table(index="id", columns="candidate", values="truncated_bool", aggfunc="first")
    correct.columns = [f"correct__{col}" for col in correct.columns]
    trunc.columns = [f"truncated__{col}" for col in trunc.columns]
    matrix = base.merge(correct.reset_index(), on="id", how="left").merge(trunc.reset_index(), on="id", how="left")
    candidate_correct_cols = [col for col in matrix.columns if col.startswith("correct__")]
    matrix["oracle_correct_count"] = matrix[candidate_correct_cols].fillna(False).sum(axis=1).astype(int)
    matrix["oracle_any_correct"] = matrix["oracle_correct_count"] > 0
    return matrix


def gate_detail(row: dict[str, Any], thresholds: dict[str, int]) -> dict[str, Any]:
    missing = {
        "weak_total": max(0, thresholds["total"] - as_int(row.get("correct"))),
        "equation_transform": max(0, thresholds["equation_transform"] - as_int(row.get("equation_transform_correct"))),
        "bit_manipulation": max(0, thresholds["bit_manipulation"] - as_int(row.get("bit_manipulation_correct"))),
        "truncated_excess": max(0, as_int(row.get("truncated"), 999999) - thresholds["truncated"]),
    }
    return {**row, "missing_for_gate": missing, "gate_pass": all(value == 0 for value in missing.values())}


def summarize_assignments(assignments: pd.DataFrame, thresholds: dict[str, int], strategy: str, deployable: bool) -> dict[str, Any]:
    correct = int(assignments["chosen_correct"].sum())
    truncated = int(assignments["chosen_truncated"].sum())
    by_family = assignments.groupby("family")["chosen_correct"].sum().to_dict()
    row = {
        "strategy": strategy,
        "deployable_without_row_labels": deployable,
        "rows": int(len(assignments)),
        "correct": correct,
        "accuracy": float(correct / len(assignments)) if len(assignments) else 0.0,
        "equation_transform_correct": int(by_family.get("equation_transform", 0)),
        "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
        "truncated": truncated,
    }
    return gate_detail(row, thresholds)


def assignment_frame(
    long_df: pd.DataFrame,
    base_ids: pd.DataFrame,
    chosen_by_id: dict[str, str],
    strategy: str,
) -> pd.DataFrame:
    lookup = long_df.set_index(["id", "candidate"])
    rows: list[dict[str, Any]] = []
    for item in base_ids.itertuples(index=False):
        row_id = str(item.id)
        chosen = chosen_by_id[row_id]
        selected = lookup.loc[(row_id, chosen)]
        rows.append(
            {
                "strategy": strategy,
                "id": row_id,
                "family": str(item.family),
                "chosen_candidate": chosen,
                "chosen_prediction": selected["prediction"],
                "answer": selected["answer"],
                "chosen_correct": bool(selected["correct_bool"]),
                "chosen_truncated": bool(selected["truncated_bool"]),
            }
        )
    return pd.DataFrame(rows)


def candidate_priority(summary_df: pd.DataFrame, preferred_default: str) -> list[str]:
    ordered = summary_df.sort_values(
        ["correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[False, False, False, True],
    )["candidate"].tolist()
    if preferred_default in ordered:
        ordered.remove(preferred_default)
        ordered.insert(0, preferred_default)
    return ordered


def choose_first_correct(row: pd.DataFrame, priority: list[str], allowed: set[str]) -> str | None:
    by_candidate = {str(item.candidate): item for item in row.itertuples(index=False)}
    for candidate in priority:
        if candidate in allowed and candidate in by_candidate:
            item = by_candidate[candidate]
            if bool(item.correct_bool) and not bool(item.truncated_bool):
                return candidate
    return None


def simulate_strategies(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_df: pd.DataFrame,
    thresholds: dict[str, int],
    preferred_default: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_ids = (
        long_df[["id", "family", "answer", "prompt"]]
        .drop_duplicates(subset=["id"])
        .sort_values(["family", "id"])
        .reset_index(drop=True)
    )
    candidates = summary_df["candidate"].tolist()
    safe_candidates = set(summary_df[summary_df["truncated"] <= thresholds["truncated"]]["candidate"].tolist())
    if preferred_default not in candidates:
        preferred_default = str(summary_df.sort_values("correct", ascending=False).iloc[0]["candidate"])
    priority = candidate_priority(summary_df, preferred_default)
    strategy_frames: list[pd.DataFrame] = []
    strategy_rows: list[dict[str, Any]] = []

    def add_strategy(name: str, chosen_by_id: dict[str, str], deployable: bool) -> None:
        frame = assignment_frame(long_df, base_ids, chosen_by_id, name)
        strategy_frames.append(frame)
        strategy_rows.append(summarize_assignments(frame, thresholds, name, deployable))

    for candidate in candidates:
        add_strategy(f"single::{candidate}", {row_id: candidate for row_id in base_ids["id"].astype(str)}, True)

    family_choice: dict[str, str] = {}
    for family, group in family_df[family_df["candidate"].isin(safe_candidates)].groupby("family", sort=False):
        merged = group.merge(summary_df[["candidate", "correct", "truncated"]], on="candidate", how="left")
        best = merged.sort_values(["correct_x", "truncated_x", "correct_y"], ascending=[False, True, False]).iloc[0]
        family_choice[str(family)] = str(best["candidate"])
    add_strategy(
        "family_best_safe",
        {str(item.id): family_choice.get(str(item.family), preferred_default) for item in base_ids.itertuples(index=False)},
        True,
    )

    add_strategy(
        f"default::{preferred_default}",
        {str(row_id): preferred_default for row_id in base_ids["id"].astype(str)},
        True,
    )

    grouped = {row_id: group for row_id, group in long_df.groupby("id", sort=False)}
    all_allowed = set(candidates)
    oracle_all: dict[str, str] = {}
    oracle_safe: dict[str, str] = {}
    default_plus_safe: dict[str, str] = {}
    equation_patch_safe: dict[str, str] = {}
    for item in base_ids.itertuples(index=False):
        row_id = str(item.id)
        group = grouped[row_id]
        oracle_all[row_id] = choose_first_correct(group, priority, all_allowed) or preferred_default
        oracle_safe[row_id] = choose_first_correct(group, priority, safe_candidates) or preferred_default
        default_row = group[group["candidate"] == preferred_default].iloc[0]
        if bool(default_row["correct_bool"]):
            default_plus_safe[row_id] = preferred_default
            equation_patch_safe[row_id] = preferred_default
        else:
            safe_hit = choose_first_correct(group, priority, safe_candidates)
            default_plus_safe[row_id] = safe_hit or preferred_default
            if str(item.family) == "equation_transform":
                equation_patch_safe[row_id] = safe_hit or preferred_default
            else:
                equation_patch_safe[row_id] = preferred_default

    add_strategy("oracle_any_candidate_by_row", oracle_all, False)
    add_strategy("oracle_safe_candidate_by_row", oracle_safe, False)
    add_strategy(f"default_plus_safe_oracle_misses::{preferred_default}", default_plus_safe, False)
    add_strategy(f"equation_only_safe_oracle_patch::{preferred_default}", equation_patch_safe, False)

    assignments = pd.concat(strategy_frames, ignore_index=True)
    return pd.DataFrame(strategy_rows), assignments


def unique_wins(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_id, group in long_df.groupby("id", sort=False):
        correct = group[group["correct_bool"] & ~group["truncated_bool"]]
        if len(correct) == 1:
            item = correct.iloc[0]
            rows.append(
                {
                    "id": row_id,
                    "family": item["family"],
                    "unique_correct_candidate": item["candidate"],
                    "answer": item["answer"],
                    "prediction": item["prediction"],
                }
            )
    return pd.DataFrame(rows)


def default_miss_hits(long_df: pd.DataFrame, preferred_default: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_id, group in long_df.groupby("id", sort=False):
        default_group = group[group["candidate"] == preferred_default]
        if default_group.empty:
            continue
        default_row = default_group.iloc[0]
        if bool(default_row["correct_bool"]):
            continue
        hits = group[group["correct_bool"] & ~group["truncated_bool"]]["candidate"].astype(str).tolist()
        rows.append(
            {
                "id": row_id,
                "family": default_row["family"],
                "answer": default_row["answer"],
                "default_prediction": default_row["prediction"],
                "correct_alternative_candidates": ";".join(hits),
                "correct_alternative_count": len(hits),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v222_v221_weak")
    parser.add_argument("--preferred-default", default="v217_final_existing")
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

    print("=== V222 COMPLEMENTARITY SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("batch_summary_json =", args.batch_summary_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("preferred_default =", args.preferred_default, flush=True)
    print("thresholds =", json.dumps(thresholds, indent=2, sort_keys=True), flush=True)

    batch_summary = read_json(args.batch_summary_json)
    candidate_rows = [row for row in batch_summary.get("rows", []) if row.get("status") == "ok"]
    if not candidate_rows:
        raise RuntimeError("batch summary has no ok candidate rows")
    print("candidate_count =", len(candidate_rows), flush=True)
    print("candidate_names =", [row.get("name") for row in candidate_rows], flush=True)

    frames: list[pd.DataFrame] = []
    load_meta: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        frame, meta = load_candidate_predictions(candidate)
        print("loaded_candidate =", json.dumps(meta, sort_keys=True), flush=True)
        frames.append(frame)
        load_meta.append(meta)
    long_df = pd.concat(frames, ignore_index=True)
    id_counts = long_df.groupby("candidate")["id"].nunique().to_dict()
    if len(set(id_counts.values())) != 1:
        raise RuntimeError(f"candidate row-count mismatch: {id_counts}")
    expected_ids = set(frames[0]["id"].astype(str))
    for meta, frame in zip(load_meta, frames):
        ids = set(frame["id"].astype(str))
        if ids != expected_ids:
            raise RuntimeError(f"id set mismatch for {meta['candidate']}")
    print("shared_row_count =", len(expected_ids), flush=True)

    summary_df = candidate_summary(long_df, thresholds)
    summary_df["gate_detail_json"] = summary_df.apply(lambda row: json.dumps(gate_detail(row.to_dict(), thresholds), sort_keys=True), axis=1)
    family_df = per_family_summary(long_df)
    matrix_df = build_correctness_matrix(long_df)
    unique_df = unique_wins(long_df)
    default_hits_df = default_miss_hits(long_df, args.preferred_default)
    strategy_df, assignments_df = simulate_strategies(long_df, summary_df, family_df, thresholds, args.preferred_default)

    prefix = safe_name(args.label)
    paths = {
        "candidate_summary_csv": args.output_dir / f"{prefix}_candidate_summary.csv",
        "per_family_summary_csv": args.output_dir / f"{prefix}_per_family_summary.csv",
        "correctness_matrix_csv": args.output_dir / f"{prefix}_correctness_matrix.csv",
        "unique_wins_csv": args.output_dir / f"{prefix}_unique_wins.csv",
        "default_miss_hits_csv": args.output_dir / f"{prefix}_default_miss_hits.csv",
        "router_simulation_csv": args.output_dir / f"{prefix}_router_simulation.csv",
        "router_assignments_csv": args.output_dir / f"{prefix}_router_assignments.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary_df.to_csv(paths["candidate_summary_csv"], index=False)
    family_df.to_csv(paths["per_family_summary_csv"], index=False)
    matrix_df.to_csv(paths["correctness_matrix_csv"], index=False)
    unique_df.to_csv(paths["unique_wins_csv"], index=False)
    default_hits_df.to_csv(paths["default_miss_hits_csv"], index=False)
    strategy_df.to_csv(paths["router_simulation_csv"], index=False)
    assignments_df.to_csv(paths["router_assignments_csv"], index=False)

    deployable_pass = strategy_df[strategy_df["deployable_without_row_labels"] & strategy_df["gate_pass"]]
    oracle_safe_pass = strategy_df[(strategy_df["strategy"] == "oracle_safe_candidate_by_row") & strategy_df["gate_pass"]]
    oracle_any_pass = strategy_df[(strategy_df["strategy"] == "oracle_any_candidate_by_row") & strategy_df["gate_pass"]]
    if len(deployable_pass):
        decision = "deployable_router_candidate_found_review_before_full_eval"
    elif len(oracle_safe_pass):
        decision = "row_level_complementarity_exists_build_rules_or_train_router_before_full_eval"
    elif len(oracle_any_pass):
        decision = "complementarity_exists_only_with_unsafe_candidates_fix_truncation_or_train"
    else:
        decision = "no_weak_gate_oracle_pass_train_equation_rescue"

    manifest = {
        "generated_at_utc": utc_now(),
        "batch_summary_json": str(args.batch_summary_json),
        "thresholds": thresholds,
        "preferred_default": args.preferred_default,
        "load_meta": load_meta,
        "candidate_summary": summary_df.drop(columns=["gate_detail_json"]).to_dict(orient="records"),
        "router_simulation": strategy_df.to_dict(orient="records"),
        "unique_wins": int(len(unique_df)),
        "default_miss_hits": int(len(default_hits_df)),
        "decision": decision,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("candidate_summary =", summary_df.drop(columns=["gate_detail_json"]).to_string(index=False), flush=True)
    print("per_family_summary =", family_df.to_string(index=False), flush=True)
    print("router_simulation =", strategy_df.to_string(index=False), flush=True)
    print("unique_wins_rows =", len(unique_df), flush=True)
    print("default_miss_hits_rows =", len(default_hits_df), flush=True)
    print("decision =", decision, flush=True)
    print("outputs =", json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V222 COMPLEMENTARITY SCRIPT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

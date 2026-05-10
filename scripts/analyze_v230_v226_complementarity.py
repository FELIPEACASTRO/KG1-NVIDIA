#!/usr/bin/env python3
"""Analyze V221/V226 weak prediction complementarity around the V226 best.

This script is CPU-only. It reads existing batch-summary/report artifacts,
loads their prediction CSVs, and checks whether a deployable family router or
row-level signal can close the weak gate before any new training or full eval.
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


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


def specs_from_batch_summary(path: Path, source: str) -> list[dict[str, str]]:
    if not path.exists():
        print(f"batch_summary_missing = {path}", flush=True)
        return []
    batch = read_json(path)
    specs: list[dict[str, str]] = []
    for row in batch.get("rows", []):
        if row.get("status") != "ok":
            continue
        original_name = str(row.get("name") or "").strip()
        report_json = Path(str(row.get("report_json") or ""))
        if not original_name or not report_json.exists():
            print(
                "candidate_artifact_skip =",
                json.dumps(
                    {
                        "source": source,
                        "name": original_name,
                        "report_json": str(report_json),
                        "report_exists": report_json.exists(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        predictions_csv = resolve_report_predictions(report_json)
        specs.append(
            {
                "name": f"{safe_name(source)}__{safe_name(original_name)}",
                "original_name": original_name,
                "source": source,
                "adapter": str(row.get("adapter") or ""),
                "report_json": str(report_json),
                "predictions_csv": str(predictions_csv),
            }
        )
    return specs


def parse_extra_candidate(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise ValueError("--extra-candidate must use NAME=PATH")
    name, value = raw.split("=", 1)
    name = safe_name(name.strip())
    path = Path(value.strip())
    if not name:
        raise ValueError("--extra-candidate name is empty")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        predictions_csv = resolve_report_predictions(path)
        report_json = str(path)
    else:
        predictions_csv = path
        report_json = ""
    return {
        "name": name,
        "original_name": name,
        "source": "extra",
        "adapter": "",
        "report_json": report_json,
        "predictions_csv": str(predictions_csv),
    }


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
    out = out[out["family"].isin(WEAK_FAMILIES)].copy()
    if "truncated" in out.columns:
        out["truncated_bool"] = out["truncated"].map(parse_bool)
    elif "finish_reason" in out.columns:
        out["truncated_bool"] = out["finish_reason"].fillna("").astype(str).eq("length")
    else:
        out["truncated_bool"] = False
    out["correct_bool"] = out.apply(lambda item: verify_answer(item["answer"], item["prediction"]), axis=1)
    if "correct" in out.columns:
        mismatch = int((out["correct"].map(parse_bool) != out["correct_bool"]).sum())
    else:
        mismatch = 0
    out["candidate"] = spec["name"]
    out["original_name"] = spec["original_name"]
    out["source"] = spec["source"]
    out["adapter"] = spec.get("adapter", "")
    meta = {
        "candidate": spec["name"],
        "original_name": spec["original_name"],
        "source": spec["source"],
        "predictions_csv": str(path),
        "rows": int(len(out)),
        "correct_mismatch_vs_csv": mismatch,
    }
    return out, meta


def candidate_summary(long_df: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_df.groupby("candidate", sort=False):
        by_family = group.groupby("family")["correct_bool"].sum().to_dict()
        trunc_by_family = group.groupby("family")["truncated_bool"].sum().to_dict()
        correct = int(group["correct_bool"].sum())
        truncated = int(group["truncated_bool"].sum())
        row = {
            "candidate": candidate,
            "original_name": str(group["original_name"].iloc[0]),
            "source": str(group["source"].iloc[0]),
            "adapter": str(group["adapter"].iloc[0]),
            "rows": int(len(group)),
            "correct": correct,
            "accuracy": float(correct / len(group)) if len(group) else 0.0,
            "equation_transform_correct": int(by_family.get("equation_transform", 0)),
            "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
            "truncated": truncated,
            "equation_transform_truncated": int(trunc_by_family.get("equation_transform", 0)),
            "bit_manipulation_truncated": int(trunc_by_family.get("bit_manipulation", 0)),
        }
        row["weak_gate_pass_for_full"] = (
            row["correct"] >= thresholds["total"]
            and row["equation_transform_correct"] >= thresholds["equation_transform"]
            and row["bit_manipulation_correct"] >= thresholds["bit_manipulation"]
            and row["truncated"] <= thresholds["truncated"]
        )
        row["gate_total_gap"] = max(0, thresholds["total"] - row["correct"])
        row["gate_eq_gap"] = max(0, thresholds["equation_transform"] - row["equation_transform_correct"])
        row["gate_bit_gap"] = max(0, thresholds["bit_manipulation"] - row["bit_manipulation_correct"])
        row["gate_trunc_gap"] = max(0, row["truncated"] - thresholds["truncated"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "weak_gate_pass_for_full",
            "correct",
            "equation_transform_correct",
            "bit_manipulation_correct",
            "truncated",
        ],
        ascending=[False, False, False, False, True],
    )


def per_family_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, family), group in long_df.groupby(["candidate", "family"], sort=False):
        rows.append(
            {
                "candidate": candidate,
                "original_name": str(group["original_name"].iloc[0]),
                "source": str(group["source"].iloc[0]),
                "family": family,
                "rows": int(len(group)),
                "correct": int(group["correct_bool"].sum()),
                "truncated": int(group["truncated_bool"].sum()),
            }
        )
    return pd.DataFrame(rows)


def pick_baseline(summary_df: pd.DataFrame, preferred: str) -> str:
    candidates = summary_df["candidate"].astype(str).tolist()
    if preferred in candidates:
        return preferred
    contains = [
        item
        for item in candidates
        if "v226" in item.lower()
        and ("checkpoint1" in item.lower() or "checkpoint_1" in item.lower() or "checkpoint-1" in item.lower())
    ]
    if contains:
        ranked = summary_df[summary_df["candidate"].isin(contains)].sort_values("correct", ascending=False)
        return str(ranked.iloc[0]["candidate"])
    v226_any = [item for item in candidates if "v226" in item.lower()]
    if v226_any:
        ranked = summary_df[summary_df["candidate"].isin(v226_any)].sort_values("correct", ascending=False)
        return str(ranked.iloc[0]["candidate"])
    return str(summary_df.sort_values("correct", ascending=False).iloc[0]["candidate"])


def pairwise_vs_baseline(long_df: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = long_df[long_df["candidate"] == baseline][
        ["id", "family", "answer", "prompt", "prediction", "correct_bool", "truncated_bool"]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "correct_bool": "baseline_correct",
            "truncated_bool": "baseline_truncated",
        }
    )
    if baseline_df.empty:
        raise ValueError(f"baseline candidate not found: {baseline}")
    rows: list[pd.DataFrame] = []
    for candidate, candidate_df in long_df.groupby("candidate", sort=False):
        if candidate == baseline:
            continue
        current = candidate_df[["id", "prediction", "correct_bool", "truncated_bool"]].rename(
            columns={
                "prediction": "candidate_prediction",
                "correct_bool": "candidate_correct",
                "truncated_bool": "candidate_truncated",
            }
        )
        merged = baseline_df.merge(current, on="id", how="inner", validate="one_to_one")
        merged.insert(1, "candidate", candidate)
        merged["lost_vs_baseline"] = merged["baseline_correct"] & ~merged["candidate_correct"]
        merged["gained_vs_baseline"] = ~merged["baseline_correct"] & merged["candidate_correct"]
        merged["candidate_extra_truncated"] = ~merged["baseline_truncated"] & merged["candidate_truncated"]
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
                }
            )
    return pd.DataFrame(summary_rows), detail


def assignment_summary(frame: pd.DataFrame, strategy: str, deployable: bool, thresholds: dict[str, int]) -> dict[str, Any]:
    correct = int(frame["chosen_correct"].sum())
    truncated = int(frame["chosen_truncated"].sum())
    by_family = frame.groupby("family")["chosen_correct"].sum().to_dict()
    row = {
        "strategy": strategy,
        "deployable_without_row_labels": deployable,
        "rows": int(len(frame)),
        "correct": correct,
        "accuracy": float(correct / len(frame)) if len(frame) else 0.0,
        "equation_transform_correct": int(by_family.get("equation_transform", 0)),
        "bit_manipulation_correct": int(by_family.get("bit_manipulation", 0)),
        "truncated": truncated,
    }
    row["weak_gate_pass_for_full"] = (
        row["correct"] >= thresholds["total"]
        and row["equation_transform_correct"] >= thresholds["equation_transform"]
        and row["bit_manipulation_correct"] >= thresholds["bit_manipulation"]
        and row["truncated"] <= thresholds["truncated"]
    )
    row["gate_total_gap"] = max(0, thresholds["total"] - row["correct"])
    row["gate_eq_gap"] = max(0, thresholds["equation_transform"] - row["equation_transform_correct"])
    row["gate_bit_gap"] = max(0, thresholds["bit_manipulation"] - row["bit_manipulation_correct"])
    row["gate_trunc_gap"] = max(0, row["truncated"] - thresholds["truncated"])
    return row


def build_assignments(long_df: pd.DataFrame, chosen_by_id: dict[str, str], strategy: str) -> pd.DataFrame:
    lookup = long_df.set_index(["id", "candidate"], drop=False)
    base = long_df[["id", "family", "answer", "prompt"]].drop_duplicates("id").sort_values(["family", "id"])
    rows: list[dict[str, Any]] = []
    for item in base.itertuples(index=False):
        row_id = str(item.id)
        chosen = chosen_by_id[row_id]
        selected = lookup.loc[(row_id, chosen)]
        rows.append(
            {
                "strategy": strategy,
                "id": row_id,
                "family": str(item.family),
                "chosen_candidate": chosen,
                "answer": selected["answer"],
                "chosen_prediction": selected["prediction"],
                "chosen_correct": bool(selected["correct_bool"]),
                "chosen_truncated": bool(selected["truncated_bool"]),
            }
        )
    return pd.DataFrame(rows)


def simulate_routers(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_df: pd.DataFrame,
    baseline: str,
    thresholds: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = long_df[["id", "family"]].drop_duplicates("id").sort_values(["family", "id"])
    candidates = summary_df["candidate"].astype(str).tolist()
    baseline_family = family_df[family_df["candidate"] == baseline].set_index("family")
    family_best_any: dict[str, str] = {}
    family_no_loss: dict[str, str] = {}
    family_choice_detail: dict[str, Any] = {}
    for family in WEAK_FAMILIES:
        fam_rows = family_df[family_df["family"] == family].copy()
        fam_rows = fam_rows.merge(summary_df[["candidate", "correct", "truncated"]], on="candidate", suffixes=("_family", "_total"))
        fam_rows = fam_rows.sort_values(
            ["correct_family", "truncated_family", "correct_total"],
            ascending=[False, True, False],
        )
        family_best_any[family] = str(fam_rows.iloc[0]["candidate"])
        baseline_correct = as_int(baseline_family.loc[family, "correct"]) if family in baseline_family.index else 0
        baseline_trunc = as_int(baseline_family.loc[family, "truncated"]) if family in baseline_family.index else 999999
        no_loss = fam_rows[
            (fam_rows["correct_family"].astype(int) >= baseline_correct)
            & (fam_rows["truncated_family"].astype(int) <= baseline_trunc)
        ].copy()
        if no_loss.empty:
            family_no_loss[family] = baseline
            chosen_row = fam_rows[fam_rows["candidate"] == baseline].iloc[0].to_dict()
        else:
            family_no_loss[family] = str(no_loss.iloc[0]["candidate"])
            chosen_row = no_loss.iloc[0].to_dict()
        family_choice_detail[family] = {
            "baseline_correct": baseline_correct,
            "baseline_truncated": baseline_trunc,
            "best_any": family_best_any[family],
            "no_loss": family_no_loss[family],
            "no_loss_row": {k: (int(v) if hasattr(v, "item") else v) for k, v in chosen_row.items()},
        }

    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []

    def add_strategy(strategy: str, chosen_by_id: dict[str, str], deployable: bool) -> None:
        frame = build_assignments(long_df, chosen_by_id, strategy)
        frames.append(frame)
        rows.append(assignment_summary(frame, strategy, deployable, thresholds))

    for candidate in candidates:
        add_strategy(f"single::{candidate}", {str(row_id): candidate for row_id in base["id"].astype(str)}, True)

    add_strategy(
        "family_best_any",
        {str(item.id): family_best_any.get(str(item.family), baseline) for item in base.itertuples(index=False)},
        True,
    )
    add_strategy(
        "family_no_loss_vs_baseline",
        {str(item.id): family_no_loss.get(str(item.family), baseline) for item in base.itertuples(index=False)},
        True,
    )

    grouped = {row_id: group for row_id, group in long_df.groupby("id", sort=False)}
    priority = summary_df.sort_values(["correct", "equation_transform_correct", "bit_manipulation_correct"], ascending=False)[
        "candidate"
    ].astype(str).tolist()
    if baseline in priority:
        priority.remove(baseline)
        priority.insert(0, baseline)
    oracle_any: dict[str, str] = {}
    default_plus_oracle_miss: dict[str, str] = {}
    for item in base.itertuples(index=False):
        row_id = str(item.id)
        group = grouped[row_id]
        correct = group[group["correct_bool"] & ~group["truncated_bool"]]
        chosen = baseline
        for candidate in priority:
            hit = correct[correct["candidate"] == candidate]
            if not hit.empty:
                chosen = candidate
                break
        oracle_any[row_id] = chosen
        baseline_row = group[group["candidate"] == baseline].iloc[0]
        default_plus_oracle_miss[row_id] = baseline if bool(baseline_row["correct_bool"]) else chosen
    add_strategy("oracle_any_candidate_by_row", oracle_any, False)
    add_strategy(f"baseline_plus_oracle_misses::{baseline}", default_plus_oracle_miss, False)

    router_df = pd.DataFrame(rows).sort_values(
        ["weak_gate_pass_for_full", "deployable_without_row_labels", "correct", "truncated"],
        ascending=[False, False, False, True],
    )
    assignments_df = pd.concat(frames, ignore_index=True)
    return router_df, assignments_df, family_choice_detail


def baseline_miss_hits(long_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {row_id: group for row_id, group in long_df.groupby("id", sort=False)}
    for row_id, group in grouped.items():
        baseline_row = group[group["candidate"] == baseline].iloc[0]
        if bool(baseline_row["correct_bool"]):
            continue
        hits = group[group["correct_bool"] & ~group["truncated_bool"]]["candidate"].astype(str).tolist()
        rows.append(
            {
                "id": row_id,
                "family": baseline_row["family"],
                "answer": baseline_row["answer"],
                "baseline_prediction": baseline_row["prediction"],
                "correct_alternative_count": len(hits),
                "correct_alternative_candidates": ";".join(hits),
                "prompt": baseline_row["prompt"],
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "correct_alternative_count", "id"], ascending=[True, False, True])


def choose_decision(
    summary_df: pd.DataFrame,
    router_df: pd.DataFrame,
    baseline_summary: dict[str, Any],
    thresholds: dict[str, int],
) -> dict[str, Any]:
    deployable_pass = router_df[router_df["deployable_without_row_labels"] & router_df["weak_gate_pass_for_full"]]
    single_pass = summary_df[summary_df["weak_gate_pass_for_full"]]
    row_oracle_pass = router_df[(~router_df["deployable_without_row_labels"]) & router_df["weak_gate_pass_for_full"]]
    best_router = router_df.iloc[0].to_dict()
    baseline_correct = as_int(baseline_summary.get("correct"))
    if len(deployable_pass):
        best = deployable_pass.iloc[0].to_dict()
        return {
            "decision": "deployable_router_passed_weak_gate_confirm_separately",
            "best_candidate": best.get("strategy"),
            "reason": f"correct={as_int(best.get('correct'))}; eq={as_int(best.get('equation_transform_correct'))}; bit={as_int(best.get('bit_manipulation_correct'))}; truncated={as_int(best.get('truncated'))}",
            "next_action": "Run a separate confirmation/full-eval notebook for the deployable router. Do not package or submit automatically.",
        }
    if len(single_pass):
        best = single_pass.iloc[0].to_dict()
        return {
            "decision": "single_candidate_passed_weak_gate_confirm_separately",
            "best_candidate": best.get("candidate"),
            "reason": f"correct={as_int(best.get('correct'))}; eq={as_int(best.get('equation_transform_correct'))}; bit={as_int(best.get('bit_manipulation_correct'))}; truncated={as_int(best.get('truncated'))}",
            "next_action": "Run a separate confirmation/full-eval notebook for this adapter. Do not package or submit automatically.",
        }
    if len(row_oracle_pass):
        best = row_oracle_pass.iloc[0].to_dict()
        return {
            "decision": "row_level_complementarity_passes_but_needs_rules",
            "best_candidate": best.get("strategy"),
            "reason": f"oracle_correct={as_int(best.get('correct'))}; deployable=false; baseline_correct={baseline_correct}",
            "next_action": "Build solver/confidence rules from the baseline-miss hit pack before any new training.",
        }
    if as_int(best_router.get("correct")) > baseline_correct:
        return {
            "decision": "router_improves_baseline_but_misses_weak_gate",
            "best_candidate": best_router.get("strategy"),
            "reason": f"router_correct={as_int(best_router.get('correct'))}; baseline_correct={baseline_correct}; total_gap={as_int(best_router.get('gate_total_gap'))}; eq_gap={as_int(best_router.get('gate_eq_gap'))}; bit_gap={as_int(best_router.get('gate_bit_gap'))}",
            "next_action": "Use the gained/lost rows to build a small verified solver or router, not another blind continuation.",
        }
    total_gap = max(0, thresholds["total"] - baseline_correct)
    return {
        "decision": "no_deployable_oracle_gain_over_v226",
        "best_candidate": best_router.get("strategy"),
        "reason": f"baseline_correct={baseline_correct}; total_gap={total_gap}; best_router_correct={as_int(best_router.get('correct'))}",
        "next_action": "Create a verified equation/bit solver dataset from baseline misses; do not full-eval or continue V227.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v221-batch-summary-json", type=Path, default=Path(""))
    parser.add_argument("--v226-batch-summary-json", type=Path, required=True)
    parser.add_argument("--v229-analysis-manifest-json", type=Path, default=Path(""))
    parser.add_argument("--extra-candidate", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v230_v226_complementarity")
    parser.add_argument("--preferred-baseline", default="v226__v226_best_checkpoint1_observed_191")
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

    print("=== V230 COMPLEMENTARITY SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v221_batch_summary_json =", args.v221_batch_summary_json, flush=True)
    print("v226_batch_summary_json =", args.v226_batch_summary_json, flush=True)
    print("v229_analysis_manifest_json =", args.v229_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("preferred_baseline =", args.preferred_baseline, flush=True)
    print("thresholds =", json.dumps(thresholds, indent=2, sort_keys=True), flush=True)

    specs: list[dict[str, str]] = []
    if str(args.v221_batch_summary_json):
        specs.extend(specs_from_batch_summary(args.v221_batch_summary_json, "v221"))
    specs.extend(specs_from_batch_summary(args.v226_batch_summary_json, "v226"))
    for raw in args.extra_candidate:
        specs.append(parse_extra_candidate(raw))
    dedup: dict[tuple[str, str], dict[str, str]] = {}
    for spec in specs:
        dedup[(spec["name"], spec["predictions_csv"])] = spec
    specs = list(dedup.values())
    if not specs:
        raise RuntimeError("no prediction specs were loaded")
    print("candidate_spec_count =", len(specs), flush=True)
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
    print("shared_row_count =", len(shared_ids), flush=True)
    if len(shared_ids) != 315:
        raise RuntimeError(f"expected exactly 315 shared weak rows, got {len(shared_ids)}")
    long_df = long_df[long_df["id"].isin(shared_ids)].copy()
    family_counts = long_df.drop_duplicates("id")["family"].value_counts().to_dict()
    print("family_counts =", json.dumps(family_counts, sort_keys=True), flush=True)
    if int(family_counts.get("equation_transform", 0)) != 155:
        raise RuntimeError("unexpected equation_transform row count")
    if int(family_counts.get("bit_manipulation", 0)) != 160:
        raise RuntimeError("unexpected bit_manipulation row count")

    summary_df = candidate_summary(long_df, thresholds)
    family_df = per_family_summary(long_df)
    baseline = pick_baseline(summary_df, args.preferred_baseline)
    print("resolved_baseline =", baseline, flush=True)
    baseline_summary = summary_df[summary_df["candidate"] == baseline].iloc[0].to_dict()
    pair_summary_df, pair_detail_df = pairwise_vs_baseline(long_df, baseline)
    router_df, assignments_df, family_choice_detail = simulate_routers(long_df, summary_df, family_df, baseline, thresholds)
    misses_df = baseline_miss_hits(long_df, baseline)
    equation_misses_df = misses_df[misses_df["family"].eq("equation_transform")].copy()
    bit_misses_df = misses_df[misses_df["family"].eq("bit_manipulation")].copy()
    decision = choose_decision(summary_df, router_df, baseline_summary, thresholds)

    v229_manifest: dict[str, Any] = {}
    if str(args.v229_analysis_manifest_json) and args.v229_analysis_manifest_json.exists():
        v229_manifest = read_json(args.v229_analysis_manifest_json)
        print("loaded_v229_manifest_decision =", json.dumps(v229_manifest.get("decision", {}), sort_keys=True), flush=True)

    paths = {
        "candidate_summary_csv": args.output_dir / f"{prefix}_candidate_summary.csv",
        "per_family_summary_csv": args.output_dir / f"{prefix}_per_family_summary.csv",
        "pairwise_summary_csv": args.output_dir / f"{prefix}_pairwise_summary.csv",
        "pairwise_detail_csv": args.output_dir / f"{prefix}_pairwise_detail.csv",
        "router_simulation_csv": args.output_dir / f"{prefix}_router_simulation.csv",
        "router_assignments_csv": args.output_dir / f"{prefix}_router_assignments.csv",
        "baseline_miss_hits_csv": args.output_dir / f"{prefix}_baseline_miss_hits.csv",
        "equation_miss_pack_csv": args.output_dir / f"{prefix}_equation_miss_pack.csv",
        "bit_miss_pack_csv": args.output_dir / f"{prefix}_bit_miss_pack.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary_df.to_csv(paths["candidate_summary_csv"], index=False)
    family_df.to_csv(paths["per_family_summary_csv"], index=False)
    pair_summary_df.to_csv(paths["pairwise_summary_csv"], index=False)
    pair_detail_df.to_csv(paths["pairwise_detail_csv"], index=False)
    router_df.to_csv(paths["router_simulation_csv"], index=False)
    assignments_df.to_csv(paths["router_assignments_csv"], index=False)
    misses_df.to_csv(paths["baseline_miss_hits_csv"], index=False)
    equation_misses_df.to_csv(paths["equation_miss_pack_csv"], index=False)
    bit_misses_df.to_csv(paths["bit_miss_pack_csv"], index=False)

    manifest = {
        "generated_at_utc": utc_now(),
        "label": args.label,
        "thresholds": thresholds,
        "candidate_count": len(summary_df),
        "resolved_baseline": baseline,
        "baseline_summary": baseline_summary,
        "family_choice_detail": family_choice_detail,
        "decision": decision,
        "v229_decision": v229_manifest.get("decision", {}),
        "candidate_summary": summary_df.to_dict(orient="records"),
        "router_simulation": router_df.to_dict(orient="records"),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_json(paths["manifest_json"], manifest)

    print("candidate_summary =", summary_df.to_string(index=False), flush=True)
    print("per_family_summary =", family_df.to_string(index=False), flush=True)
    print("router_simulation =", router_df.to_string(index=False), flush=True)
    print("family_choice_detail =", json.dumps(family_choice_detail, indent=2, sort_keys=True), flush=True)
    print("baseline_miss_hit_counts =", json.dumps(misses_df["family"].value_counts().to_dict(), sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V230 COMPLEMENTARITY SCRIPT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze V221/V226 weak prediction complementarity around the V226 best.

This script is CPU-only. It reads existing batch-summary/report artifacts,
loads their prediction CSVs, and checks whether a deployable family router or
row-level signal can close the weak gate before any new training or full eval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import box_answer, canonical_family, classify_puzzle, extract_final_answer, verify_answer  # noqa: E402


WEAK_FAMILIES = ("bit_manipulation", "equation_transform")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value in (0, 0.0):
            return False
        if value in (1, 1.0):
            return True
        raise ValueError(f"unsupported boolean numeric value: {value!r}")
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "0.0", "false", "no", "n", "f", "", "nan"}:
        return False
    raise ValueError(f"unsupported boolean text value: {value!r}")


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
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = report_path.parent / path
        if path.exists():
            return path
        raise FileNotFoundError(f"report predictions_csv does not exist: {path}")
    matches = sorted(report_path.parent.glob("*_predictions.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous prediction CSV fallback for {report_path}: {[str(path) for path in matches]}")
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
        if not original_name:
            raise ValueError(f"{path} has an ok row with an empty candidate name")
        if not report_json.is_absolute():
            report_json = path.parent / report_json
        if not report_json.exists():
            raise FileNotFoundError(f"{path} has an ok row with missing report_json: {report_json}")
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


def load_predictions(spec: dict[str, str], allow_rescore_mismatch: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(spec["predictions_csv"])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "prompt", "answer", "prediction", "raw_output"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    empty_required: dict[str, int] = {}
    for column in ["id", "prompt", "answer", "raw_output"]:
        empty_count = int(out[column].astype(str).str.strip().eq("").sum())
        if empty_count:
            empty_required[column] = empty_count
    if empty_required:
        raise RuntimeError(f"{path} has empty required fields: " + json.dumps(empty_required, sort_keys=True))
    duplicate_ids = int(out["id"].duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(f"{path} has duplicate ids after string normalization: {duplicate_ids}")
    family_col = first_existing_column(out, ["type", "task_type", "family"])
    derived_family = out["prompt"].map(classify_puzzle).map(canonical_family)
    if family_col:
        out["family"] = out[family_col].map(canonical_family)
        family_mismatch = int((out["family"] != derived_family).sum())
        if family_mismatch:
            sample = out.loc[out["family"] != derived_family, ["id", family_col, "prompt"]].head(5).to_dict(orient="records")
            raise RuntimeError(
                f"{path} has {family_mismatch} rows where declared family disagrees with prompt classifier: "
                + json.dumps(sample, sort_keys=True)
            )
    else:
        out["family"] = derived_family
    non_weak_rows = int((~out["family"].isin(WEAK_FAMILIES)).sum())
    if non_weak_rows:
        raise RuntimeError(f"{path} contains {non_weak_rows} non-weak rows; V230 requires exactly the weak 315-row slice")
    out = out[out["family"].isin(WEAK_FAMILIES)].copy()
    out["answer"] = out["answer"].astype(str)
    out["prompt"] = out["prompt"].astype(str)
    out["prediction"] = out["prediction"].fillna("").astype(str)
    out["raw_output"] = out["raw_output"].fillna("").astype(str)
    out["prompt_sha256"] = out["prompt"].map(sha256_text)
    extractor_mismatch = int((out["raw_output"].map(extract_final_answer) != out["prediction"]).sum())
    if extractor_mismatch:
        raise RuntimeError(f"{path} has {extractor_mismatch} rows where raw_output extraction differs from prediction")
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
    if mismatch and not allow_rescore_mismatch:
        raise RuntimeError(f"{path} has {mismatch} rows where CSV correct disagrees with current verifier")
    out["candidate"] = spec["name"]
    out["original_name"] = spec["original_name"]
    out["source"] = spec["source"]
    out["adapter"] = spec.get("adapter", "")
    meta = {
        "candidate": spec["name"],
        "original_name": spec["original_name"],
        "source": spec["source"],
        "predictions_csv": str(path),
        "predictions_csv_sha256": sha256_file(path),
        "input_rows": int(len(frame)),
        "rows": int(len(out)),
        "family_counts": {str(k): int(v) for k, v in out["family"].value_counts().to_dict().items()},
        "correct_mismatch_vs_csv": mismatch,
        "extractor_mismatch_vs_prediction": extractor_mismatch,
    }
    return out, meta


def validate_shared_row_contract(frames: list[pd.DataFrame], load_meta: list[dict[str, Any]], shared_ids: set[str]) -> None:
    contract_cols = ["id", "family", "answer", "prompt_sha256"]
    reference: dict[str, tuple[str, str, str]] | None = None
    reference_candidate = ""
    for frame, meta in zip(frames, load_meta):
        candidate = str(meta["candidate"])
        contract = frame[frame["id"].isin(shared_ids)][contract_cols].copy().sort_values("id")
        if int(contract["id"].duplicated().sum()):
            raise RuntimeError(f"{candidate} has duplicate shared ids")
        records = {
            str(row.id): (str(row.family), str(row.answer), str(row.prompt_sha256))
            for row in contract.itertuples(index=False)
        }
        digest_payload = "\n".join(
            f"{row_id}\t{family}\t{answer}\t{prompt_hash}"
            for row_id, (family, answer, prompt_hash) in sorted(records.items())
        )
        meta["shared_row_contract_sha256"] = sha256_text(digest_payload)
        if reference is None:
            reference = records
            reference_candidate = candidate
            continue
        mismatches = []
        assert reference is not None
        for row_id, value in records.items():
            if reference.get(row_id) != value:
                mismatches.append(
                    {
                        "id": row_id,
                        "reference_candidate": reference_candidate,
                        "reference": reference.get(row_id),
                        "candidate": candidate,
                        "observed": value,
                    }
                )
            if len(mismatches) >= 10:
                break
        if mismatches:
            raise RuntimeError(
                "prediction CSV row contract mismatch by id: "
                + json.dumps(mismatches, sort_keys=True, ensure_ascii=True)
            )


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
        add_gate_metrics(row, thresholds)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "weak_gate_pass_for_full",
            "gate_normalized_gap",
            "correct",
            "equation_transform_correct",
            "bit_manipulation_correct",
            "truncated",
        ],
        ascending=[False, True, False, False, False, True],
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
                "accuracy": float(group["correct_bool"].sum() / len(group)) if len(group) else 0.0,
                "truncated": int(group["truncated_bool"].sum()),
            }
        )
    return pd.DataFrame(rows)


def add_gate_metrics(row: dict[str, Any], thresholds: dict[str, int]) -> dict[str, Any]:
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
    row["gate_total_margin"] = row["correct"] - thresholds["total"]
    row["gate_eq_margin"] = row["equation_transform_correct"] - thresholds["equation_transform"]
    row["gate_bit_margin"] = row["bit_manipulation_correct"] - thresholds["bit_manipulation"]
    row["gate_trunc_margin"] = thresholds["truncated"] - row["truncated"]
    normalized_gaps = {
        "total": row["gate_total_gap"] / max(1, thresholds["total"]),
        "equation_transform": row["gate_eq_gap"] / max(1, thresholds["equation_transform"]),
        "bit_manipulation": row["gate_bit_gap"] / max(1, thresholds["bit_manipulation"]),
        "truncated": row["gate_trunc_gap"] / max(1, thresholds["truncated"]),
    }
    row["gate_normalized_gap"] = round(float(sum(normalized_gaps.values())), 9)
    row["gate_bottleneck"] = "none" if not any(normalized_gaps.values()) else max(normalized_gaps, key=normalized_gaps.get)
    return row


def family_calibration_summary(
    family_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    thresholds: dict[str, int],
    baseline: str,
) -> pd.DataFrame:
    merged = family_df.merge(
        summary_df[["candidate", "correct", "truncated"]].rename(
            columns={"correct": "total_correct", "truncated": "total_truncated"}
        ),
        on="candidate",
        how="left",
        validate="many_to_one",
    )
    baseline_rows = family_df[family_df["candidate"] == baseline].set_index("family")
    rows: list[dict[str, Any]] = []
    for item in merged.itertuples(index=False):
        family = str(item.family)
        threshold = int(thresholds[family])
        baseline_correct = as_int(baseline_rows.loc[family, "correct"]) if family in baseline_rows.index else 0
        baseline_truncated = as_int(baseline_rows.loc[family, "truncated"]) if family in baseline_rows.index else 0
        correct = as_int(item.correct)
        truncated = as_int(item.truncated)
        gap = max(0, threshold - correct)
        rows.append(
            {
                "candidate": str(item.candidate),
                "original_name": str(item.original_name),
                "source": str(item.source),
                "family": family,
                "rows": as_int(item.rows),
                "family_threshold": threshold,
                "correct": correct,
                "accuracy": float(getattr(item, "accuracy", 0.0)),
                "truncated": truncated,
                "family_gap": gap,
                "family_margin": correct - threshold,
                "family_threshold_attainment": round(float(correct / max(1, threshold)), 9),
                "baseline_correct": baseline_correct,
                "baseline_truncated": baseline_truncated,
                "delta_correct_vs_baseline_family": correct - baseline_correct,
                "delta_truncated_vs_baseline_family": truncated - baseline_truncated,
                "total_correct": as_int(getattr(item, "total_correct", 0)),
                "total_truncated": as_int(getattr(item, "total_truncated", 0)),
                "calibrated_family_gap_score": round(float(gap / max(1, threshold)), 9),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "family",
            "calibrated_family_gap_score",
            "correct",
            "delta_correct_vs_baseline_family",
            "truncated",
            "total_correct",
        ],
        ascending=[True, True, False, False, True, False],
    )


def pick_baseline(summary_df: pd.DataFrame, preferred: str, allow_fallback: bool = False) -> str:
    candidates = summary_df["candidate"].astype(str).tolist()
    preferred_names = [item.strip() for item in re.split(r"[,|]", str(preferred)) if item.strip()]
    for preferred_name in preferred_names:
        if preferred_name in candidates:
            return preferred_name
    if not allow_fallback:
        raise RuntimeError(f"required preferred baseline was not found: {preferred}")
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
    return add_gate_metrics(row, thresholds)


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
            "no_loss_row": {k: json_scalar(v) for k, v in chosen_row.items()},
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
        "weak_tuned_family_best_any",
        {str(item.id): family_best_any.get(str(item.family), baseline) for item in base.itertuples(index=False)},
        True,
    )
    add_strategy(
        "family_aggregate_non_regression_vs_baseline",
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
        baseline_effective_ok = bool(baseline_row["correct_bool"]) and not bool(baseline_row["truncated_bool"])
        default_plus_oracle_miss[row_id] = baseline if baseline_effective_ok else chosen
    add_strategy("oracle_any_candidate_by_row", oracle_any, False)
    add_strategy(f"baseline_plus_oracle_misses::{baseline}", default_plus_oracle_miss, False)

    router_df = pd.DataFrame(rows).sort_values(
        [
            "weak_gate_pass_for_full",
            "deployable_without_row_labels",
            "gate_normalized_gap",
            "correct",
            "truncated",
        ],
        ascending=[False, False, True, False, True],
    )
    assignments_df = pd.concat(frames, ignore_index=True)
    return router_df, assignments_df, family_choice_detail


def baseline_miss_hits(long_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {row_id: group for row_id, group in long_df.groupby("id", sort=False)}
    for row_id, group in grouped.items():
        baseline_row = group[group["candidate"] == baseline].iloc[0]
        baseline_effective_ok = bool(baseline_row["correct_bool"]) and not bool(baseline_row["truncated_bool"])
        if baseline_effective_ok:
            continue
        hits = group[group["correct_bool"] & ~group["truncated_bool"]]["candidate"].astype(str).tolist()
        rows.append(
            {
                "id": row_id,
                "family": baseline_row["family"],
                "answer": baseline_row["answer"],
                "baseline_prediction": baseline_row["prediction"],
                "baseline_correct": bool(baseline_row["correct_bool"]),
                "baseline_truncated": bool(baseline_row["truncated_bool"]),
                "correct_alternative_count": len(hits),
                "correct_alternative_candidates": ";".join(hits),
                "prompt": baseline_row["prompt"],
            }
        )
    columns = [
        "id",
        "family",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "baseline_truncated",
        "correct_alternative_count",
        "correct_alternative_candidates",
        "prompt",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["family", "correct_alternative_count", "id"], ascending=[True, False, True]
    )


def choose_decision(
    summary_df: pd.DataFrame,
    router_df: pd.DataFrame,
    baseline_summary: dict[str, Any],
    thresholds: dict[str, int],
) -> dict[str, Any]:
    router_strategy = router_df["strategy"].astype(str)
    is_single_strategy = router_strategy.str.startswith("single::")
    deployable_pass = router_df[
        router_df["deployable_without_row_labels"] & router_df["weak_gate_pass_for_full"] & ~is_single_strategy
    ]
    single_pass = summary_df[summary_df["weak_gate_pass_for_full"]]
    row_oracle_pass = router_df[(~router_df["deployable_without_row_labels"]) & router_df["weak_gate_pass_for_full"]]
    deployable_routers = router_df[router_df["deployable_without_row_labels"] & ~is_single_strategy].sort_values(
        ["gate_normalized_gap", "correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[True, False, False, False, True],
    )
    row_oracle_routers = router_df[~router_df["deployable_without_row_labels"]].sort_values(
        ["gate_normalized_gap", "correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[True, False, False, False, True],
    )
    best_deployable = deployable_routers.iloc[0].to_dict() if len(deployable_routers) else {}
    best_row_oracle = row_oracle_routers.iloc[0].to_dict() if len(row_oracle_routers) else {}
    baseline_correct = as_int(baseline_summary.get("correct"))
    best_single = summary_df.sort_values(
        ["gate_normalized_gap", "correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[True, False, False, False, True],
    ).iloc[0].to_dict()
    if len(deployable_pass):
        best = deployable_pass.iloc[0].to_dict()
        return {
            "decision": "weak_tuned_family_router_passed_weak_gate_confirm_separately",
            "best_candidate": best.get("strategy"),
            "reason": f"correct={as_int(best.get('correct'))}; eq={as_int(best.get('equation_transform_correct'))}; bit={as_int(best.get('bit_manipulation_correct'))}; truncated={as_int(best.get('truncated'))}",
            "next_action": "Run a separate confirmation/full-eval notebook for the weak-tuned family router. Do not package or submit automatically.",
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
    if best_deployable and as_int(best_deployable.get("correct")) > baseline_correct:
        return {
            "decision": "weak_tuned_family_router_improves_baseline_but_misses_weak_gate",
            "best_candidate": best_deployable.get("strategy"),
            "reason": f"router_correct={as_int(best_deployable.get('correct'))}; baseline_correct={baseline_correct}; total_gap={as_int(best_deployable.get('gate_total_gap'))}; eq_gap={as_int(best_deployable.get('gate_eq_gap'))}; bit_gap={as_int(best_deployable.get('gate_bit_gap'))}",
            "next_action": "Use the gained/lost rows to build a small verified solver or router, not another blind continuation.",
        }
    if best_single and as_int(best_single.get("correct")) > baseline_correct:
        return {
            "decision": "single_candidate_improves_baseline_but_misses_weak_gate",
            "best_candidate": best_single.get("candidate"),
            "reason": f"candidate_correct={as_int(best_single.get('correct'))}; baseline_correct={baseline_correct}; total_gap={as_int(best_single.get('gate_total_gap'))}; eq_gap={as_int(best_single.get('gate_eq_gap'))}; bit_gap={as_int(best_single.get('gate_bit_gap'))}",
            "next_action": "Use pairwise gained/lost rows to decide whether this adapter deserves a separate confirmation eval.",
        }
    if best_row_oracle and as_int(best_row_oracle.get("correct")) > baseline_correct:
        return {
            "decision": "row_level_oracle_improves_but_misses_weak_gate",
            "best_candidate": best_row_oracle.get("strategy"),
            "reason": f"oracle_correct={as_int(best_row_oracle.get('correct'))}; baseline_correct={baseline_correct}; total_gap={as_int(best_row_oracle.get('gate_total_gap'))}; eq_gap={as_int(best_row_oracle.get('gate_eq_gap'))}; bit_gap={as_int(best_row_oracle.get('gate_bit_gap'))}",
            "next_action": "Mine the baseline-miss hit pack for deterministic solver/confidence rules before another training run.",
        }
    total_gap = max(0, thresholds["total"] - baseline_correct)
    return {
        "decision": "no_deployable_oracle_gain_over_v226",
        "best_candidate": best_deployable.get("strategy", ""),
        "reason": f"baseline_correct={baseline_correct}; total_gap={total_gap}; best_deployable_correct={as_int(best_deployable.get('correct'))}; best_oracle_correct={as_int(best_row_oracle.get('correct'))}",
        "next_action": "Create a verified equation/bit solver dataset from baseline misses; do not full-eval or continue V227.",
    }


def synthetic_prediction_rows(candidate: str, mutation: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(315):
        family = "bit_manipulation" if index < 160 else "equation_transform"
        if family == "bit_manipulation":
            prompt = f"Solve this bit manipulation puzzle with an 8-bit binary value. Row {index}."
            answer = "00101010" if index == 0 else "1"
        else:
            prompt = f"Apply the transformation rule to this equation puzzle. Row {index}."
            answer = "1"
        prediction = answer if (candidate == "baseline" and index < 191) or (candidate == "v221" and index < 12) else "0"
        correct = prediction == answer
        raw_output = box_answer(prediction)
        declared_family = family
        if mutation == "bad_correct" and candidate == "baseline" and index == 0:
            correct = False
        if mutation == "bad_raw" and candidate == "baseline" and index == 0:
            raw_output = box_answer("9")
        if mutation == "prompt_mismatch" and candidate == "baseline" and index == 0:
            prompt = "Solve this bit manipulation puzzle with an 8-bit binary value. Changed row zero."
        if mutation == "family_mismatch" and candidate == "baseline" and index == 0:
            declared_family = "equation_transform"
        if mutation == "empty_required_field" and candidate == "baseline" and index == 0:
            answer = ""
            prediction = ""
            raw_output = box_answer(prediction)
            correct = True
        rows.append(
            {
                "id": f"weak_{index:03d}",
                "prompt": prompt,
                "answer": answer,
                "prediction": prediction,
                "raw_output": raw_output,
                "correct": str(correct),
                "type": declared_family,
                "truncated": "False",
            }
        )
    if mutation == "extra_weak_row" and candidate == "baseline":
        rows.append(
            {
                "id": "weak_extra_999",
                "prompt": "Solve this bit manipulation puzzle with an 8-bit binary value. Extra row.",
                "answer": "00101010",
                "prediction": "00101010",
                "raw_output": box_answer("00101010"),
                "correct": "True",
                "type": "bit_manipulation",
                "truncated": "False",
            }
        )
    return rows


def write_synthetic_case(root: Path, name: str, mutation: str | None = None) -> Path:
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    for source, candidate in [("v221", "v221"), ("v226", "baseline")]:
        prediction_path = case_dir / f"{source}_predictions.csv"
        rows = synthetic_prediction_rows(candidate, mutation=mutation)
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report_path = case_dir / f"{source}_report.json"
        write_json(report_path, {"outputs": {"predictions_csv": prediction_path.name}})
        row_name = "v226_best_checkpoint1_observed_191" if source == "v226" else "candidate_a"
        if mutation == "missing_baseline" and source == "v226":
            row_name = "checkpoint_without_required_name"
        write_json(
            case_dir / f"{source}_summary.json",
            {"rows": [{"status": "ok", "name": row_name, "adapter": "", "report_json": str(report_path)}]},
        )
    return case_dir


def run_synthetic_cli_case(
    root: Path,
    name: str,
    *,
    mutation: str | None = None,
    extra_args: list[str] | None = None,
    expect_ok: bool = True,
    expected_message: str = "",
) -> None:
    case_dir = write_synthetic_case(root, name, mutation=mutation)
    output_dir = case_dir / "out"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--v221-batch-summary-json",
        str(case_dir / "v221_summary.json"),
        "--v226-batch-summary-json",
        str(case_dir / "v226_summary.json"),
        "--output-dir",
        str(output_dir),
        "--label",
        name,
    ]
    if extra_args:
        command.extend(extra_args)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    if (completed.returncode == 0) != expect_ok:
        raise AssertionError(
            f"{name}: expected ok={expect_ok}, returncode={completed.returncode}, "
            f"tail={completed.stdout[-4000:]}"
        )
    if expected_message and expected_message not in completed.stdout:
        raise AssertionError(f"{name}: missing expected message {expected_message!r}; tail={completed.stdout[-4000:]}")
    if expect_ok:
        manifest = read_json(output_dir / f"{safe_name(name)}_manifest.json")
        if "family_calibration_summary" not in manifest:
            raise AssertionError(f"{name}: missing family_calibration_summary")
        if "family_calibration_csv" not in manifest.get("outputs", {}):
            raise AssertionError(f"{name}: missing family_calibration_csv output")
        if int(manifest["load_meta"][0]["rows"]) != 315:
            raise AssertionError(f"{name}: synthetic row count drift")
        calibration = manifest["family_calibration_summary"]
        if not any(row.get("family") == "bit_manipulation" for row in calibration):
            raise AssertionError(f"{name}: missing bit_manipulation calibration")


def run_self_test() -> int:
    print("=== V230 ANALYZER SELF TEST START ===", flush=True)
    boxed_binary = box_answer("00101010")
    if boxed_binary != r"\boxed{00101010}":
        raise AssertionError(f"boxed binary escape mismatch: {boxed_binary!r}")
    if extract_final_answer(boxed_binary) != "00101010":
        raise AssertionError("boxed binary extraction did not preserve leading zeros")
    if extract_final_answer(r"Answer \boxed{42}. \text{done}") != "42":
        raise AssertionError("boxed extraction consumed text after the balanced closing brace")
    if extract_final_answer(r"Answer \boxed{\frac{1}{2}} trailing") != r"\frac{1}{2}":
        raise AssertionError("boxed extraction failed nested LaTeX payload")
    if not verify_answer("00101010", "00101010"):
        raise AssertionError("binary exact verify failed")
    if verify_answer("00101010", "101010"):
        raise AssertionError("binary leading-zero verify incorrectly passed")

    root = Path(tempfile.mkdtemp(prefix="v230_analyzer_self_test_"))
    try:
        run_synthetic_cli_case(root, "valid_leading_zero_binary")
        run_synthetic_cli_case(
            root,
            "missing_baseline",
            mutation="missing_baseline",
            expect_ok=False,
            expected_message="required preferred baseline was not found",
        )
        run_synthetic_cli_case(
            root,
            "bad_correct",
            mutation="bad_correct",
            expect_ok=False,
            expected_message="CSV correct disagrees",
        )
        run_synthetic_cli_case(
            root,
            "bad_raw",
            mutation="bad_raw",
            expect_ok=False,
            expected_message="raw_output extraction differs",
        )
        run_synthetic_cli_case(
            root,
            "prompt_mismatch",
            mutation="prompt_mismatch",
            expect_ok=False,
            expected_message="row contract mismatch",
        )
        run_synthetic_cli_case(
            root,
            "family_mismatch",
            mutation="family_mismatch",
            expect_ok=False,
            expected_message="declared family disagrees",
        )
        run_synthetic_cli_case(
            root,
            "empty_required_field",
            mutation="empty_required_field",
            expect_ok=False,
            expected_message="empty required fields",
        )
        run_synthetic_cli_case(
            root,
            "extra_weak_row",
            mutation="extra_weak_row",
            expect_ok=False,
            expected_message="weak row set differs from shared 315-row contract",
        )
        run_synthetic_cli_case(
            root,
            "wrong_contract_hash",
            extra_args=["--expected-shared-row-contract-sha256", "definitely_wrong"],
            expect_ok=False,
            expected_message="shared row contract hash mismatch",
        )
        run_synthetic_cli_case(
            root,
            "missing_required_contract_hash",
            extra_args=["--require-shared-row-contract-sha256"],
            expect_ok=False,
            expected_message="--expected-shared-row-contract-sha256 is empty",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("v230_analyzer_self_test=ok", flush=True)
    print("=== V230 ANALYZER SELF TEST END ===", flush=True)
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return run_self_test()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--v221-batch-summary-json", type=Path, default=Path(""))
    parser.add_argument("--v226-batch-summary-json", type=Path, required=True)
    parser.add_argument("--v229-analysis-manifest-json", type=Path, default=Path(""))
    parser.add_argument("--extra-candidate", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v230_v226_complementarity")
    parser.add_argument("--preferred-baseline", default="v226__v226_best_checkpoint1_observed_191")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=136)
    parser.add_argument("--weak-trunc-max", type=int, default=0)
    parser.add_argument("--expected-baseline-correct", type=int, default=191)
    parser.add_argument("--expected-baseline-rows", type=int, default=315)
    parser.add_argument("--expected-baseline-adapter", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default="")
    parser.add_argument("--require-shared-row-contract-sha256", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument("--allow-rescore-mismatch", action="store_true")
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
    print("expected_baseline_correct =", args.expected_baseline_correct, flush=True)
    print("expected_baseline_rows =", args.expected_baseline_rows, flush=True)
    print("expected_baseline_adapter =", args.expected_baseline_adapter, flush=True)
    print("expected_shared_row_contract_sha256 =", args.expected_shared_row_contract_sha256, flush=True)
    print("required_shared_row_contract_sha256 =", args.require_shared_row_contract_sha256, flush=True)
    print("allow_baseline_fallback =", args.allow_baseline_fallback, flush=True)
    print("allow_rescore_mismatch =", args.allow_rescore_mismatch, flush=True)
    print("thresholds =", json.dumps(thresholds, indent=2, sort_keys=True), flush=True)

    if args.require_shared_row_contract_sha256 and not str(args.expected_shared_row_contract_sha256).strip():
        raise RuntimeError(
            "--require-shared-row-contract-sha256 was set but --expected-shared-row-contract-sha256 is empty"
        )

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
    duplicate_names = sorted({spec["name"] for spec in specs if [item["name"] for item in specs].count(spec["name"]) > 1})
    if duplicate_names:
        raise RuntimeError("duplicate candidate names after normalization: " + json.dumps(duplicate_names))
    if not specs:
        raise RuntimeError("no prediction specs were loaded")
    source_counts: dict[str, int] = {}
    for spec in specs:
        source_counts[spec["source"]] = source_counts.get(spec["source"], 0) + 1
    if source_counts.get("v226", 0) < 1:
        raise RuntimeError("at least one V226 prediction artifact is required")
    if source_counts.get("v221", 0) < 1:
        raise RuntimeError("at least one V221 prediction artifact is required")
    print("candidate_spec_count =", len(specs), flush=True)
    print("candidate_source_counts =", json.dumps(source_counts, sort_keys=True), flush=True)
    print("candidate_specs =", json.dumps(specs, indent=2, sort_keys=True), flush=True)

    frames: list[pd.DataFrame] = []
    load_meta: list[dict[str, Any]] = []
    for spec in specs:
        frame, meta = load_predictions(spec, allow_rescore_mismatch=bool(args.allow_rescore_mismatch))
        print("loaded_candidate =", json.dumps(meta, sort_keys=True), flush=True)
        frames.append(frame)
        load_meta.append(meta)
    long_df = pd.concat(frames, ignore_index=True)
    id_sets = {meta["candidate"]: set(frame["id"].astype(str)) for meta, frame in zip(load_meta, frames)}
    shared_ids = set.intersection(*id_sets.values())
    print("shared_row_count =", len(shared_ids), flush=True)
    if len(shared_ids) != 315:
        raise RuntimeError(f"expected exactly 315 shared weak rows, got {len(shared_ids)}")
    for meta in load_meta:
        candidate = str(meta["candidate"])
        candidate_ids = id_sets[candidate]
        extra_ids = sorted(candidate_ids - shared_ids)
        missing_ids = sorted(shared_ids - candidate_ids)
        if extra_ids or missing_ids:
            raise RuntimeError(
                f"{candidate} weak row set differs from shared 315-row contract: "
                + json.dumps(
                    {
                        "extra_count": len(extra_ids),
                        "extra_sample": extra_ids[:5],
                        "missing_count": len(missing_ids),
                        "missing_sample": missing_ids[:5],
                    },
                    sort_keys=True,
                )
            )
        if int(meta.get("rows", -1)) != 315 or int(meta.get("input_rows", -1)) != 315:
            raise RuntimeError(
                f"{candidate} must contain exactly 315 weak rows; "
                f"input_rows={meta.get('input_rows')}, weak_rows={meta.get('rows')}"
            )
    validate_shared_row_contract(frames, load_meta, shared_ids)
    observed_contract_hashes = {str(item["shared_row_contract_sha256"]) for item in load_meta}
    if len(observed_contract_hashes) != 1:
        raise RuntimeError("shared row contract hashes diverged after validation")
    observed_shared_row_contract_sha256 = sorted(observed_contract_hashes)[0]
    if args.expected_shared_row_contract_sha256:
        expected = str(args.expected_shared_row_contract_sha256)
        if observed_shared_row_contract_sha256 != expected:
            raise RuntimeError(
                f"shared row contract hash mismatch: expected {expected}, got {observed_shared_row_contract_sha256}"
            )
    print(
        "validated_shared_row_contracts =",
        json.dumps(
            {str(item["candidate"]): item.get("shared_row_contract_sha256", "") for item in load_meta},
            sort_keys=True,
        ),
        flush=True,
    )
    long_df = long_df[long_df["id"].isin(shared_ids)].copy()
    family_counts = long_df.drop_duplicates("id")["family"].value_counts().to_dict()
    print("family_counts =", json.dumps(family_counts, sort_keys=True), flush=True)
    if int(family_counts.get("equation_transform", 0)) != 155:
        raise RuntimeError("unexpected equation_transform row count")
    if int(family_counts.get("bit_manipulation", 0)) != 160:
        raise RuntimeError("unexpected bit_manipulation row count")

    summary_df = candidate_summary(long_df, thresholds)
    family_df = per_family_summary(long_df)
    baseline = pick_baseline(summary_df, args.preferred_baseline, allow_fallback=bool(args.allow_baseline_fallback))
    print("resolved_baseline =", baseline, flush=True)
    baseline_summary = summary_df[summary_df["candidate"] == baseline].iloc[0].to_dict()
    baseline_specs = [spec for spec in specs if spec["name"] == baseline]
    if not baseline_specs:
        raise RuntimeError("resolved baseline spec missing: " + baseline)
    baseline_spec = baseline_specs[0]
    if str(args.expected_baseline_adapter).strip():
        expected_adapter = Path(str(args.expected_baseline_adapter)).as_posix()
        observed_adapter = Path(str(baseline_spec.get("adapter") or "")).as_posix()
        if observed_adapter != expected_adapter:
            raise RuntimeError(
                "baseline adapter mismatch: expected "
                + expected_adapter
                + ", got "
                + observed_adapter
            )
    if args.expected_baseline_correct >= 0 and as_int(baseline_summary.get("correct")) != args.expected_baseline_correct:
        raise RuntimeError(
            f"baseline correct mismatch: expected {args.expected_baseline_correct}, got {as_int(baseline_summary.get('correct'))}"
        )
    if args.expected_baseline_rows >= 0 and as_int(baseline_summary.get("rows")) != args.expected_baseline_rows:
        raise RuntimeError(
            f"baseline row count mismatch: expected {args.expected_baseline_rows}, got {as_int(baseline_summary.get('rows'))}"
        )
    if as_int(baseline_summary.get("truncated")) > thresholds["truncated"]:
        raise RuntimeError("baseline truncation exceeds weak truncation ceiling")
    calibration_df = family_calibration_summary(family_df, summary_df, thresholds, baseline)
    pair_summary_df, pair_detail_df = pairwise_vs_baseline(long_df, baseline)
    router_df, assignments_df, family_choice_detail = simulate_routers(long_df, summary_df, family_df, baseline, thresholds)
    misses_df = baseline_miss_hits(long_df, baseline)
    equation_misses_df = misses_df[misses_df["family"].eq("equation_transform")].copy()
    bit_misses_df = misses_df[misses_df["family"].eq("bit_manipulation")].copy()
    decision = choose_decision(summary_df, router_df, baseline_summary, thresholds)

    v229_manifest: dict[str, Any] = {}
    v229_path_text = str(args.v229_analysis_manifest_json)
    if v229_path_text not in {"", "."} and args.v229_analysis_manifest_json.is_file():
        v229_manifest = read_json(args.v229_analysis_manifest_json)
        print("loaded_v229_manifest_decision =", json.dumps(v229_manifest.get("decision", {}), sort_keys=True), flush=True)
    elif v229_path_text not in {"", "."}:
        print("v229_manifest_skip_missing_or_not_file =", args.v229_analysis_manifest_json, flush=True)

    paths = {
        "candidate_summary_csv": args.output_dir / f"{prefix}_candidate_summary.csv",
        "per_family_summary_csv": args.output_dir / f"{prefix}_per_family_summary.csv",
        "family_calibration_csv": args.output_dir / f"{prefix}_family_calibration.csv",
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
    calibration_df.to_csv(paths["family_calibration_csv"], index=False)
    pair_summary_df.to_csv(paths["pairwise_summary_csv"], index=False)
    pair_detail_df.to_csv(paths["pairwise_detail_csv"], index=False)
    router_df.to_csv(paths["router_simulation_csv"], index=False)
    assignments_df.to_csv(paths["router_assignments_csv"], index=False)
    misses_df.to_csv(paths["baseline_miss_hits_csv"], index=False)
    equation_misses_df.to_csv(paths["equation_miss_pack_csv"], index=False)
    bit_misses_df.to_csv(paths["bit_miss_pack_csv"], index=False)
    output_artifact_hashes = {name: file_meta(path) for name, path in paths.items() if name != "manifest_json"}
    input_artifact_hashes: dict[str, Any] = {
        "v221_batch_summary_json": file_meta(args.v221_batch_summary_json) if str(args.v221_batch_summary_json) else {},
        "v226_batch_summary_json": file_meta(args.v226_batch_summary_json),
        "v229_analysis_manifest_json": file_meta(args.v229_analysis_manifest_json)
        if v229_path_text not in {"", "."}
        else {},
        "candidate_reports": {
            spec["name"]: file_meta(Path(spec["report_json"])) if spec.get("report_json") else {}
            for spec in specs
        },
    }

    manifest = {
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v221_batch_summary_json": str(args.v221_batch_summary_json),
            "v226_batch_summary_json": str(args.v226_batch_summary_json),
            "v229_analysis_manifest_json": str(args.v229_analysis_manifest_json),
            "preferred_baseline": args.preferred_baseline,
            "expected_baseline_correct": int(args.expected_baseline_correct),
            "expected_baseline_rows": int(args.expected_baseline_rows),
            "expected_baseline_adapter": str(args.expected_baseline_adapter),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "required_shared_row_contract_sha256": bool(args.require_shared_row_contract_sha256),
            "allow_baseline_fallback": bool(args.allow_baseline_fallback),
            "allow_rescore_mismatch": bool(args.allow_rescore_mismatch),
        },
        "thresholds": thresholds,
        "candidate_count": len(summary_df),
        "candidate_source_counts": source_counts,
        "candidate_specs": specs,
        "input_artifact_hashes": input_artifact_hashes,
        "load_meta": load_meta,
        "resolved_baseline": baseline,
        "observed_shared_row_contract_sha256": observed_shared_row_contract_sha256,
        "baseline_summary": baseline_summary,
        "family_choice_detail": family_choice_detail,
        "decision": decision,
        "v229_decision": v229_manifest.get("decision", {}),
        "candidate_summary": summary_df.to_dict(orient="records"),
        "per_family_summary": family_df.to_dict(orient="records"),
        "family_calibration_summary": calibration_df.to_dict(orient="records"),
        "router_simulation": router_df.to_dict(orient="records"),
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_artifact_hashes": output_artifact_hashes,
    }
    write_json(paths["manifest_json"], manifest)

    print("candidate_summary =", summary_df.to_string(index=False), flush=True)
    print("per_family_summary =", family_df.to_string(index=False), flush=True)
    print("family_calibration_summary =", calibration_df.to_string(index=False), flush=True)
    print("router_simulation =", router_df.to_string(index=False), flush=True)
    print("family_choice_detail =", json.dumps(family_choice_detail, indent=2, sort_keys=True), flush=True)
    print("baseline_miss_hit_counts =", json.dumps(misses_df["family"].value_counts().to_dict(), sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V230 COMPLEMENTARITY SCRIPT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

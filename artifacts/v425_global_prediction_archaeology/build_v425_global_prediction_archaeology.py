#!/usr/bin/env python3
"""V425 global prediction archaeology.

This CPU-only gate scans local CSV artifacts for row-level predictions that may
have been missed by narrower audits. It is intentionally read-only against model
weights and Kaggle/HF. Promotion is allowed only for adapter-like prediction
artifacts that beat the current submit-safe baseline on the weak contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for item in (REPO_ROOT, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, extract_final_answer, verify_answer  # noqa: E402


BASELINE_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V414_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate/"
    / "v414_accepted_union.csv"
)
OUT_DIR = REPO_ROOT / "artifacts/v425_global_prediction_archaeology/20260515T_v425_global_prediction_archaeology"

EXPECTED_BASELINE = {
    "rows": 315,
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}

ID_COLUMNS = ("id", "row_id", "question_id", "uuid")
PREDICTION_COLUMNS = (
    "prediction",
    "pred",
    "predictions",
    "final_answer",
    "model_answer",
    "extracted_answer",
    "extracted_final_answer",
    "candidate_prediction",
    "new_prediction",
    "integrated_prediction",
    "combined_prediction",
    "solver_answer",
    "solver_prediction",
    "stride_prediction",
    "tong_prediction",
    "postprocessed_prediction",
    "override_prediction",
    "current_prediction",
    "v336_prediction",
    "v343_prediction",
    "v350_prediction",
    "v355_prediction",
    "v357_prediction",
    "v363_prediction",
    "v364_prediction",
    "v365_prediction",
    "v366_prediction",
    "v368_prediction",
    "v414_prediction",
    "v258_prediction",
    "v260_prediction",
    "v269_checkpoint2_prediction",
    "v269_final_prediction",
    "raw_output",
    "candidate_raw_output",
    "baseline_raw_output",
    "response",
    "output",
    "text",
)
TRUNCATION_COLUMNS = ("truncated", "truncated_bool", "candidate_truncated", "integrated_truncated")

ADAPTER_HINTS = (
    "adapter",
    "checkpoint",
    "weak_eval",
    "official_like",
    "h200",
    "a100",
    "hf_",
    "_hf",
    "eval_",
    "/eval/",
    "/evals/",
    "soup",
    "prompt_sweep",
)
NON_ADAPTER_HINTS = (
    "postprocessor",
    "postprocessed",
    "numeric_override",
    "solver_projection",
    "integrated_solver",
    "cpu_gate",
    "reasoner_gate",
    "teacher_meta",
    "formal_solver",
    "pbe_gate",
    "cryptarithm_gate",
    "residual_gate",
    "synthesis_gate",
    "taxonomy",
    "public_train_pattern",
    "v274",
    "v275",
    "v278",
    "v300",
    "v301",
    "v302",
    "v306",
    "v324",
    "v329",
    "v336",
    "v343",
    "v357",
    "v366",
    "v403",
    "v405",
    "v409",
    "v412",
    "v414",
    "v418",
    "v420",
    "v421",
    "v422",
    "v423",
    "v424",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def lower_field_map(fieldnames: list[str] | None) -> dict[str, str]:
    return {str(name).strip().lower(): str(name) for name in (fieldnames or []) if str(name).strip()}


def first_column(fieldnames: list[str] | None, candidates: tuple[str, ...]) -> str | None:
    fmap = lower_field_map(fieldnames)
    for candidate in candidates:
        if candidate in fmap:
            return fmap[candidate]
    return None


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or row.get("task_type") or classify_puzzle(str(row.get("prompt", ""))))


def normalize_prediction(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "\\boxed{" in text or "\n" in text or len(text) > 96:
        return extract_final_answer(text)
    return text


def is_truncated(row: dict[str, str]) -> bool:
    for column in TRUNCATION_COLUMNS:
        for key in (column, column.upper(), column.capitalize()):
            if key in row:
                value = str(row.get(key, "")).strip().lower()
                if value in {"1", "true", "yes", "y"}:
                    return True
    finish = str(row.get("finish_reason", "") or row.get("stop_reason", "")).strip().lower()
    return finish == "length"


def classify_source(path: Path, pred_col: str) -> tuple[str, bool]:
    text = str(path.relative_to(REPO_ROOT)).replace("\\", "/").lower()
    pred = pred_col.lower()
    non_adapter = any(token in text for token in NON_ADAPTER_HINTS) or any(
        token in pred for token in ("solver", "postprocessed", "integrated", "override", "v414")
    )
    adapter_like = any(token in text for token in ADAPTER_HINTS) and not non_adapter
    if adapter_like:
        return "adapter_like", True
    if non_adapter:
        return "teacher_solver_or_postprocessor", False
    return "unknown_or_diagnostic", False


def iter_csvs() -> list[Path]:
    out: list[Path] = []
    for path in sorted((REPO_ROOT / "artifacts").rglob("*.csv")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > 100 * 1024 * 1024:
            continue
        if OUT_DIR in path.parents:
            continue
        out.append(path)
    return out


def baseline_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    families: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    for row in rows:
        family = family_for(row)
        correct = verify_answer(row["answer"], row["prediction"])
        truncated = is_truncated(row)
        total["rows"] += 1
        total["correct"] += int(correct)
        total["truncated"] += int(truncated)
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
        families[family]["truncated"] += int(truncated)
    return {"total": dict(total), "families": {key: dict(value) for key, value in sorted(families.items())}}


def assert_baseline(summary: dict[str, Any]) -> None:
    observed = {
        "rows": int(summary["total"].get("rows", -1)),
        "correct": int(summary["total"].get("correct", -1)),
        "equation_transform_correct": int(summary["families"].get("equation_transform", {}).get("correct", -1)),
        "bit_manipulation_correct": int(summary["families"].get("bit_manipulation", {}).get("correct", -1)),
        "truncated": int(summary["total"].get("truncated", -1)),
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError(f"baseline drift: expected {EXPECTED_BASELINE}, got {observed}")


def score_candidate(
    path: Path,
    rows: list[dict[str, str]],
    id_col: str,
    pred_col: str,
    baseline_by_id: dict[str, dict[str, str]],
    baseline_correct: dict[str, bool],
    v414_ids: set[str],
) -> dict[str, Any] | None:
    overlap = [row for row in rows if str(row.get(id_col, "")).strip() in baseline_by_id and str(row.get(pred_col, "")).strip()]
    if len(overlap) < 100:
        return None

    families: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    gains: list[str] = []
    losses: list[str] = []
    v414_hits: list[str] = []
    v414_misses: list[str] = []
    prediction_values: Counter[str] = Counter()

    for row in overlap:
        row_id = str(row.get(id_col, "")).strip()
        ref = baseline_by_id[row_id]
        prediction = normalize_prediction(row.get(pred_col, ""))
        prediction_values[prediction] += 1
        family = family_for(ref)
        correct = verify_answer(ref["answer"], prediction)
        base_correct = baseline_correct[row_id]
        truncated = is_truncated(row)

        total["rows"] += 1
        total["correct"] += int(correct)
        total["truncated"] += int(truncated)
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
        families[family]["truncated"] += int(truncated)

        if (not base_correct) and correct:
            gains.append(row_id)
        if base_correct and not correct:
            losses.append(row_id)
        if row_id in v414_ids:
            if correct:
                v414_hits.append(row_id)
            else:
                v414_misses.append(row_id)

    source_type, submit_like = classify_source(path, pred_col)
    baseline_overlap_correct = sum(int(baseline_correct[str(row.get(id_col, "")).strip()]) for row in overlap)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "source_type": source_type,
        "submit_like_adapter_eval": submit_like,
        "id_column": id_col,
        "prediction_column": pred_col,
        "overlap_rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "delta_vs_baseline_on_overlap": int(total["correct"]) - int(baseline_overlap_correct),
        "equation_transform_correct": int(families["equation_transform"]["correct"]),
        "bit_manipulation_correct": int(families["bit_manipulation"]["correct"]),
        "truncated": int(total["truncated"]),
        "gains_vs_baseline": len(gains),
        "losses_vs_baseline": len(losses),
        "v414_gain_hits": len(v414_hits),
        "v414_gain_misses": len(v414_misses),
        "v414_equation_hits": sum(1 for row_id in v414_hits if family_for(baseline_by_id[row_id]) == "equation_transform"),
        "v414_bit_hits": sum(1 for row_id in v414_hits if family_for(baseline_by_id[row_id]) == "bit_manipulation"),
        "unique_prediction_count": len(prediction_values),
        "top_prediction": prediction_values.most_common(1)[0][0] if prediction_values else "",
        "gain_ids": ";".join(sorted(gains)),
        "loss_ids": ";".join(sorted(losses)),
        "v414_hit_ids": ";".join(sorted(v414_hits)),
    }


def main() -> int:
    baseline_rows = read_csv(BASELINE_CSV)
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    baseline_correct = {row_id: verify_answer(row["answer"], row["prediction"]) for row_id, row in baseline_by_id.items()}
    base_summary = baseline_summary(baseline_rows)
    assert_baseline(base_summary)

    accepted_rows = read_csv(V414_ACCEPTED_CSV)
    v414_ids = {str(row["id"]) for row in accepted_rows if str(row.get("id", "")).strip()}

    candidate_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in iter_csvs():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                id_col = first_column(fieldnames, ID_COLUMNS)
                prediction_cols = [name for name in fieldnames if name.strip().lower() in PREDICTION_COLUMNS]
                if not id_col or not prediction_cols:
                    skipped.append(
                        {
                            "path": str(path.relative_to(REPO_ROOT)),
                            "reason": "missing_id_or_prediction",
                            "columns": "|".join(fieldnames[:40]),
                        }
                    )
                    continue
                rows = list(reader)
        except Exception as exc:
            skipped.append({"path": str(path.relative_to(REPO_ROOT)), "reason": repr(exc), "columns": ""})
            continue

        scored_any = False
        for pred_col in prediction_cols:
            scored = score_candidate(path, rows, id_col, pred_col, baseline_by_id, baseline_correct, v414_ids)
            if scored:
                candidate_rows.append(scored)
                scored_any = True
        if not scored_any:
            skipped.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "reason": "weak_overlap_too_small",
                    "columns": "|".join(fieldnames[:40]),
                }
            )

    candidate_rows.sort(
        key=lambda row: (
            row["source_type"] != "adapter_like",
            -int(row["correct"]),
            -int(row["equation_transform_correct"]),
            -int(row["bit_manipulation_correct"]),
            int(row["truncated"]),
            -int(row["v414_gain_hits"]),
            int(row["losses_vs_baseline"]),
            row["path"],
            row["prediction_column"],
        )
    )

    baseline_total = EXPECTED_BASELINE["correct"]
    baseline_eq = EXPECTED_BASELINE["equation_transform_correct"]
    baseline_bit = EXPECTED_BASELINE["bit_manipulation_correct"]
    promotable = [
        row
        for row in candidate_rows
        if bool(row["submit_like_adapter_eval"])
        and int(row["overlap_rows"]) == EXPECTED_BASELINE["rows"]
        and int(row["correct"]) > baseline_total
        and int(row["equation_transform_correct"]) > baseline_eq
        and int(row["bit_manipulation_correct"]) >= baseline_bit
        and int(row["truncated"]) == 0
    ]
    adapter_like = [row for row in candidate_rows if row["source_type"] == "adapter_like"]
    non_adapter_best = [row for row in candidate_rows if row["source_type"] != "adapter_like"][:30]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "path",
        "source_type",
        "submit_like_adapter_eval",
        "id_column",
        "prediction_column",
        "overlap_rows",
        "correct",
        "delta_vs_baseline_on_overlap",
        "equation_transform_correct",
        "bit_manipulation_correct",
        "truncated",
        "gains_vs_baseline",
        "losses_vs_baseline",
        "v414_gain_hits",
        "v414_gain_misses",
        "v414_equation_hits",
        "v414_bit_hits",
        "unique_prediction_count",
        "top_prediction",
        "gain_ids",
        "loss_ids",
        "v414_hit_ids",
    ]
    write_csv(OUT_DIR / "v425_all_scored_prediction_columns.csv", candidate_rows, columns)
    write_csv(OUT_DIR / "v425_adapter_like_candidates.csv", adapter_like, columns)
    write_csv(OUT_DIR / "v425_promotable_adapter_candidates.csv", promotable, columns)
    write_csv(OUT_DIR / "v425_best_non_adapter_teacher_candidates.csv", non_adapter_best, columns)
    write_csv(OUT_DIR / "v425_skipped_csvs.csv", skipped, ["path", "reason", "columns"])

    by_source = Counter(str(row["source_type"]) for row in candidate_rows)
    best_adapter = adapter_like[0] if adapter_like else {}
    best_overall = candidate_rows[0] if candidate_rows else {}
    manifest = {
        "schema_version": "kg1_v425_global_prediction_archaeology_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "baseline_csv": str(BASELINE_CSV.relative_to(REPO_ROOT)),
            "baseline_sha256": sha256_file(BASELINE_CSV),
            "v414_accepted_csv": str(V414_ACCEPTED_CSV.relative_to(REPO_ROOT)),
            "v414_accepted_sha256": sha256_file(V414_ACCEPTED_CSV),
        },
        "baseline": EXPECTED_BASELINE,
        "scan": {
            "csv_files_considered": len(iter_csvs()),
            "scored_prediction_columns": len(candidate_rows),
            "adapter_like_scored": len(adapter_like),
            "skipped_csvs": len(skipped),
            "source_type_counts": dict(sorted(by_source.items())),
        },
        "best_overall": best_overall,
        "best_adapter_like": best_adapter,
        "promotable_adapter_candidate_count": len(promotable),
        "decision": {
            "decision": "v425_promotable_adapter_candidate_found" if promotable else "v425_no_promotable_adapter_candidate_found",
            "hf_gpu_allowed": False,
            "reason": (
                "At least one existing adapter-like prediction artifact beats weak baseline; route through full/package gates."
                if promotable
                else "No existing adapter-like CSV beats total>192, equation>56, bit>=136, trunc=0. Do not spend GPU on archaeology result."
            ),
            "next_action": (
                "Promote the best adapter-like candidate through official-like full/package gates."
                if promotable
                else "Continue CPU-only rule/adapter-behavior probes; new GPU SFT remains blocked by V417/V425."
            ),
        },
        "outputs": {
            "manifest_json": str((OUT_DIR / "v425_global_prediction_archaeology_manifest.json").relative_to(REPO_ROOT)),
            "report_md": str((OUT_DIR / "V425_GLOBAL_PREDICTION_ARCHAEOLOGY.md").relative_to(REPO_ROOT)),
            "all_scored_csv": str((OUT_DIR / "v425_all_scored_prediction_columns.csv").relative_to(REPO_ROOT)),
            "adapter_like_csv": str((OUT_DIR / "v425_adapter_like_candidates.csv").relative_to(REPO_ROOT)),
            "promotable_csv": str((OUT_DIR / "v425_promotable_adapter_candidates.csv").relative_to(REPO_ROOT)),
            "non_adapter_best_csv": str((OUT_DIR / "v425_best_non_adapter_teacher_candidates.csv").relative_to(REPO_ROOT)),
            "skipped_csv": str((OUT_DIR / "v425_skipped_csvs.csv").relative_to(REPO_ROOT)),
        },
    }
    write_json(OUT_DIR / "v425_global_prediction_archaeology_manifest.json", manifest)

    report = [
        "# V425 Global Prediction Archaeology",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Baseline Contract",
        "",
        "| Candidate | Total | equation_transform | bit_manipulation | Truncated |",
        "|---|---:|---:|---:|---:|",
        f"| V291/V290 checkpoint-6 | `{baseline_total}/315` | `{baseline_eq}/155` | `{baseline_bit}/160` | `0` |",
        "",
        "## Scan Summary",
        "",
        f"- CSV files considered: `{manifest['scan']['csv_files_considered']}`.",
        f"- Scored prediction columns: `{len(candidate_rows)}`.",
        f"- Adapter-like scored columns: `{len(adapter_like)}`.",
        f"- Promotable adapter-like candidates: `{len(promotable)}`.",
        "",
        "## Best Adapter-Like Candidates",
        "",
        "| CSV | Col | Total | equation | bit | trunc | V414 hits | Losses | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in adapter_like[:15]:
        decision = "promotable" if row in promotable else "reject"
        report.append(
            f"| `{row['path']}` | `{row['prediction_column']}` | `{row['correct']}` | "
            f"`{row['equation_transform_correct']}` | `{row['bit_manipulation_correct']}` | "
            f"`{row['truncated']}` | `{row['v414_gain_hits']}` | `{row['losses_vs_baseline']}` | {decision} |"
        )
    report += [
        "",
        "## Best Non-Adapter/Teacher Signals",
        "",
        "| CSV | Col | Source | Total | equation | bit | trunc | V414 hits |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in non_adapter_best[:10]:
        report.append(
            f"| `{row['path']}` | `{row['prediction_column']}` | `{row['source_type']}` | "
            f"`{row['correct']}` | `{row['equation_transform_correct']}` | `{row['bit_manipulation_correct']}` | "
            f"`{row['truncated']}` | `{row['v414_gain_hits']}` |"
        )
    report += [
        "",
        "## Decision",
        "",
        f"`{manifest['decision']['decision']}`: {manifest['decision']['reason']}",
        "",
        "GPU spending remains blocked unless an adapter-like artifact or a new CPU gate proves a path that can beat the weak baseline without bit/truncation regression.",
    ]
    (OUT_DIR / "V425_GLOBAL_PREDICTION_ARCHAEOLOGY.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

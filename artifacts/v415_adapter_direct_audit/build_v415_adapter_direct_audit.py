#!/usr/bin/env python3
"""V415 adapter-direct audit.

Scan local row-level adapter evaluation CSVs and check whether any already
captures the V414 CPU-teacher gains without regressing the V291/V290 baseline.
This is a no-training, no-submit gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V414_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate/"
    / "v414_accepted_union.csv"
)
OUT_DIR = REPO_ROOT / "artifacts/v415_adapter_direct_audit/20260515T_v415_adapter_direct_audit"


PREDICTION_COLUMNS = [
    "prediction",
    "pred",
    "final_answer",
    "model_answer",
    "extracted_answer",
    "new_prediction",
]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("family") or classify_puzzle(row["prompt"]))


def prediction_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    lower_map = {name.lower(): name for name in fieldnames}
    for column in PREDICTION_COLUMNS:
        if column in lower_map:
            return lower_map[column]
    return None


def is_adapter_eval_path(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    include_tokens = [
        "hf_",
        "_hf",
        "weak_eval",
        "official_like",
        "eval_",
        "/eval/",
        "/evals/",
        "checkpoint",
        "soup",
        "prompt_sweep",
    ]
    exclude_tokens = [
        "postprocessor",
        "postprocessed",
        "numeric_override",
        "solver_projection",
        "cpu_gate",
        "reasoner_gate",
        "integrated_solver",
        "v414_cpu",
        "v415_adapter",
        "v366_bit",
        "v357_bit",
        "v336_integrated",
        "v343_equation",
        "v350_cpu",
        "v355_cpu",
        "v363_equation",
        "v364_symbolic",
        "v365_bit",
        "v374_cpu",
        "v380_solver",
        "v405_integrated",
        "v301_bit_postprocessor",
        "v302_combined_postprocessor",
    ]
    return any(token in text for token in include_tokens) and not any(token in text for token in exclude_tokens)


def iter_candidate_csvs() -> list[Path]:
    paths = []
    for path in sorted((REPO_ROOT / "artifacts").rglob("*.csv")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 80 * 1024 * 1024:
            continue
        if is_adapter_eval_path(path):
            paths.append(path)
    return paths


def score_baseline(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        correct = verify_answer(row["answer"], row["prediction"])
        total["rows"] += 1
        total["correct"] += int(correct)
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
    return {"total": dict(total), "families": {key: dict(value) for key, value in sorted(families.items())}}


def main() -> int:
    baseline_rows = read_csv(BASELINE_CSV)
    by_id = {row["id"]: row for row in baseline_rows}
    baseline_correct = {row["id"]: verify_answer(row["answer"], row["prediction"]) for row in baseline_rows}
    baseline_summary = score_baseline(baseline_rows)

    accepted = read_csv(V414_ACCEPTED_CSV)
    v414_gain_ids = {row["id"] for row in accepted}
    v414_by_id = {row["id"]: row for row in accepted}

    candidate_summaries: list[dict[str, Any]] = []
    gain_matrix: dict[str, dict[str, Any]] = {
        row_id: {
            "id": row_id,
            "family": family_for(by_id[row_id]),
            "baseline_prediction": by_id[row_id]["prediction"],
            "v414_prediction": v414_by_id[row_id]["new_prediction"],
            "answer": by_id[row_id]["answer"],
            "adapter_hit_count": 0,
            "adapter_hit_files": [],
        }
        for row_id in sorted(v414_gain_ids)
        if row_id in by_id
    }
    skipped: list[dict[str, Any]] = []

    for path in iter_candidate_csvs():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                pred_col = prediction_column(reader.fieldnames)
                if not reader.fieldnames or "id" not in reader.fieldnames or not pred_col:
                    skipped.append({"path": str(path.relative_to(REPO_ROOT)), "reason": "missing_id_or_prediction"})
                    continue
                rows = list(reader)
        except Exception as exc:  # pragma: no cover - diagnostic robustness
            skipped.append({"path": str(path.relative_to(REPO_ROOT)), "reason": repr(exc)})
            continue

        overlap_rows = [row for row in rows if row.get("id") in by_id and str(row.get(pred_col, "")).strip()]
        if len(overlap_rows) < 100:
            skipped.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "reason": f"weak_overlap_too_small:{len(overlap_rows)}",
                }
            )
            continue

        totals = Counter()
        family_counts: dict[str, Counter[str]] = defaultdict(Counter)
        gains: list[str] = []
        losses: list[str] = []
        v414_hits: list[str] = []
        v414_misses: list[str] = []
        truncated = 0
        for row in overlap_rows:
            row_id = str(row["id"])
            base = by_id[row_id]
            pred = str(row.get(pred_col, "")).strip()
            family = family_for(base)
            correct = verify_answer(base["answer"], pred)
            base_correct = baseline_correct[row_id]
            totals["rows"] += 1
            totals["correct"] += int(correct)
            family_counts[family]["rows"] += 1
            family_counts[family]["correct"] += int(correct)
            finish_reason = str(row.get("finish_reason", "")).strip().lower()
            row_truncated = str(row.get("truncated", "")).strip().lower() in {"true", "1", "yes"} or finish_reason == "length"
            truncated += int(row_truncated)
            if (not base_correct) and correct:
                gains.append(row_id)
            if base_correct and not correct:
                losses.append(row_id)
            if row_id in v414_gain_ids:
                if correct:
                    v414_hits.append(row_id)
                    if row_id in gain_matrix:
                        gain_matrix[row_id]["adapter_hit_count"] += 1
                        gain_matrix[row_id]["adapter_hit_files"].append(str(path.relative_to(REPO_ROOT)))
                else:
                    v414_misses.append(row_id)

        equation_correct = family_counts["equation_transform"]["correct"]
        bit_correct = family_counts["bit_manipulation"]["correct"]
        candidate_summaries.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "prediction_column": pred_col,
                "overlap_rows": totals["rows"],
                "correct": totals["correct"],
                "delta_vs_baseline_on_overlap": totals["correct"]
                - sum(int(baseline_correct[row["id"]]) for row in overlap_rows),
                "equation_transform_correct": equation_correct,
                "bit_manipulation_correct": bit_correct,
                "truncated": truncated,
                "gains_vs_baseline": len(gains),
                "losses_vs_baseline": len(losses),
                "v414_gain_hits": len(v414_hits),
                "v414_gain_misses": len(v414_misses),
                "v414_equation_hits": sum(1 for row_id in v414_hits if family_for(by_id[row_id]) == "equation_transform"),
                "v414_bit_hits": sum(1 for row_id in v414_hits if family_for(by_id[row_id]) == "bit_manipulation"),
                "gain_ids": ";".join(sorted(gains)),
                "loss_ids": ";".join(sorted(losses)),
                "v414_hit_ids": ";".join(sorted(v414_hits)),
            }
        )

    candidate_summaries.sort(
        key=lambda row: (
            -int(row["v414_gain_hits"]),
            int(row["losses_vs_baseline"]),
            -int(row["correct"]),
            str(row["path"]),
        )
    )
    matrix_rows = []
    for row in gain_matrix.values():
        row = dict(row)
        row["adapter_hit_files"] = ";".join(sorted(row["adapter_hit_files"]))
        matrix_rows.append(row)
    matrix_rows.sort(key=lambda row: (row["family"], row["id"]))

    best = candidate_summaries[0] if candidate_summaries else {}
    promotable = [
        row
        for row in candidate_summaries
        if int(row["overlap_rows"]) == 315
        and int(row["correct"]) > baseline_summary["total"]["correct"]
        and int(row["equation_transform_correct"]) > baseline_summary["families"]["equation_transform"]["correct"]
        and int(row["bit_manipulation_correct"]) >= baseline_summary["families"]["bit_manipulation"]["correct"]
        and int(row["truncated"]) == 0
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_columns = [
        "path",
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
        "gain_ids",
        "loss_ids",
        "v414_hit_ids",
    ]
    matrix_columns = [
        "id",
        "family",
        "baseline_prediction",
        "v414_prediction",
        "answer",
        "adapter_hit_count",
        "adapter_hit_files",
    ]
    write_csv(OUT_DIR / "v415_adapter_candidate_summary.csv", candidate_summaries, summary_columns)
    write_csv(OUT_DIR / "v415_v414_gain_hit_matrix.csv", matrix_rows, matrix_columns)
    write_csv(OUT_DIR / "v415_skipped_csvs.csv", skipped, ["path", "reason"])
    write_csv(OUT_DIR / "v415_promotable_candidates.csv", promotable, summary_columns)

    manifest = {
        "schema_version": "kg1_v415_adapter_direct_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_csv": str(BASELINE_CSV.relative_to(REPO_ROOT)),
            "baseline_sha256": sha256_file(BASELINE_CSV),
            "v414_accepted_csv": str(V414_ACCEPTED_CSV.relative_to(REPO_ROOT)),
            "v414_accepted_sha256": sha256_file(V414_ACCEPTED_CSV),
        },
        "baseline": {
            "total": baseline_summary["total"]["correct"],
            "equation_transform": baseline_summary["families"]["equation_transform"]["correct"],
            "bit_manipulation": baseline_summary["families"]["bit_manipulation"]["correct"],
        },
        "scanned_candidate_count": len(candidate_summaries),
        "skipped_csv_count": len(skipped),
        "v414_gain_count": len(v414_gain_ids),
        "best_v414_hit_candidate": best,
        "promotable_candidate_count": len(promotable),
        "decision": {
            "decision": "v415_no_existing_adapter_direct_promotion" if not promotable else "v415_promotable_candidate_found",
            "hf_gpu_allowed": False,
            "reason": (
                "No existing row-level adapter eval beats baseline with equation>56, bit>=136 and truncation=0."
                if not promotable
                else "At least one existing candidate passed the weak promotion screen."
            ),
            "next_action": (
                "If no promotable candidate exists, design a materially different transfer mechanism; do not repeat V368/V413."
            ),
        },
        "outputs": {
            "summary_md": str((OUT_DIR / "V415_ADAPTER_DIRECT_AUDIT.md").relative_to(REPO_ROOT)),
            "candidate_summary_csv": str((OUT_DIR / "v415_adapter_candidate_summary.csv").relative_to(REPO_ROOT)),
            "gain_hit_matrix_csv": str((OUT_DIR / "v415_v414_gain_hit_matrix.csv").relative_to(REPO_ROOT)),
            "promotable_candidates_csv": str((OUT_DIR / "v415_promotable_candidates.csv").relative_to(REPO_ROOT)),
            "manifest_json": str((OUT_DIR / "v415_adapter_direct_audit_manifest.json").relative_to(REPO_ROOT)),
        },
    }
    write_json(OUT_DIR / "v415_adapter_direct_audit_manifest.json", manifest)

    report = [
        "# V415 Adapter Direct Audit",
        "",
        "V415 scans existing row-level adapter eval CSVs and asks whether any candidate already captures V414 teacher gains without baseline regressions.",
        "",
        "## Baseline",
        "",
        f"- V291/V290 weak: `{baseline_summary['total']['correct']}/315`, equation `{baseline_summary['families']['equation_transform']['correct']}/155`, bit `{baseline_summary['families']['bit_manipulation']['correct']}/160`.",
        f"- V414 teacher gain rows audited: `{len(v414_gain_ids)}`.",
        "",
        "## Best Existing Adapter-Like Candidates By V414 Hits",
        "",
        "| Candidate CSV | Correct | equation | bit | trunc | V414 hits | Losses vs baseline | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidate_summaries[:12]:
        decision = "promotable" if row in promotable else "reject"
        report.append(
            f"| `{row['path']}` | `{row['correct']}` | `{row['equation_transform_correct']}` | "
            f"`{row['bit_manipulation_correct']}` | `{row['truncated']}` | `{row['v414_gain_hits']}` | "
            f"`{row['losses_vs_baseline']}` | {decision} |"
        )
    report += [
        "",
        "## Decision",
        "",
    ]
    if promotable:
        report.append("At least one existing candidate passed the weak screen; promote it through full/package gates before any new training.")
    else:
        report.append(
            "No existing adapter-like row-level eval passed the promotion screen. Some candidates may hit isolated V414 rows, but they also lose too many baseline rows or keep equation at `56`."
        )
        report.append(
            "Next step remains a new transfer mechanism, not another broad SFT of the same teacher rows."
        )
    (OUT_DIR / "V415_ADAPTER_DIRECT_AUDIT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

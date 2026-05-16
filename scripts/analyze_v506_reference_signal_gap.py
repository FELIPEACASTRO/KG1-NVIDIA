"""Compare best adapter raw-output predictions with best reference-only signal.

This CPU-only audit turns the V505 result into concrete transfer targets.  It
does not train, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_V505_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "v505_label_free_candidate_revalidation"
    / "v505_label_free_candidate_revalidation_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v506_reference_signal_gap"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row.get("id")}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def family_of(row: dict[str, str]) -> str:
    return str(row.get("family") or row.get("type") or row.get("task_type") or "unknown")


def adapter_prediction(row: dict[str, str]) -> str:
    raw_output = row.get("raw_output", "")
    if raw_output:
        return extract_final_answer(raw_output)
    return row.get("prediction", "")


def reference_rule_metadata(row: dict[str, str]) -> dict[str, str]:
    keys = [
        "v350_source_rule",
        "v357_source_rule",
        "v366_source_rule",
        "v343_prediction",
        "v350_prediction",
        "v357_prediction",
        "v366_prediction",
        "current_prediction",
        "baseline_prediction",
    ]
    return {key: row.get(key, "") for key in keys}


def run(args: argparse.Namespace) -> int:
    print("=== V506 REFERENCE SIGNAL GAP START ===", flush=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    print("generated_at_utc =", generated_at, flush=True)
    print("v505_manifest =", args.v505_manifest, flush=True)
    manifest = read_json(args.v505_manifest)
    adapter_info = manifest.get("best_adapter_raw_weak315") or {}
    reference_info = manifest.get("best_reference_only_weak315") or {}
    adapter_path = Path(str(adapter_info.get("path") or ""))
    reference_path = Path(str(reference_info.get("path") or ""))
    if not adapter_path.is_file():
        raise FileNotFoundError(f"best adapter raw CSV not found: {adapter_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"best reference-only CSV not found: {reference_path}")
    print("adapter_csv =", adapter_path, flush=True)
    print("reference_csv =", reference_path, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_rows = read_csv_map(adapter_path)
    reference_rows = read_csv_map(reference_path)
    common_ids = sorted(set(adapter_rows) & set(reference_rows))
    print("adapter_rows =", len(adapter_rows), flush=True)
    print("reference_rows =", len(reference_rows), flush=True)
    print("common_ids =", len(common_ids), flush=True)
    if len(common_ids) != 315:
        raise RuntimeError(f"expected 315 common weak ids, got {len(common_ids)}")

    rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    family_status_counts: Counter[tuple[str, str]] = Counter()
    for row_id in common_ids:
        adapter_row = adapter_rows[row_id]
        reference_row = reference_rows[row_id]
        answer = adapter_row.get("answer") or reference_row.get("answer") or ""
        family = family_of(adapter_row if family_of(adapter_row) != "unknown" else reference_row)
        raw_adapter_prediction = adapter_prediction(adapter_row)
        ref_prediction = reference_row.get("prediction", "")
        adapter_correct = verify_answer(answer, raw_adapter_prediction)
        reference_correct = verify_answer(answer, ref_prediction)
        if reference_correct and not adapter_correct:
            status = "reference_gain_target"
        elif adapter_correct and not reference_correct:
            status = "reference_loss_risk"
        elif adapter_correct and reference_correct:
            status = "both_correct"
        else:
            status = "both_wrong"
        status_counts[status] += 1
        family_status_counts[(family, status)] += 1
        rows.append(
            {
                "id": row_id,
                "family": family,
                "status": status,
                "answer": answer,
                "adapter_prediction_label_free": raw_adapter_prediction,
                "reference_prediction": ref_prediction,
                "adapter_correct": adapter_correct,
                "reference_correct": reference_correct,
                "adapter_raw_output": adapter_row.get("raw_output", ""),
                "prompt": adapter_row.get("prompt") or reference_row.get("prompt") or "",
                **reference_rule_metadata(reference_row),
            }
        )

    gap_csv = args.output_dir / "v506_reference_signal_gap_rows.csv"
    write_csv(
        gap_csv,
        rows,
        [
            "id",
            "family",
            "status",
            "answer",
            "adapter_prediction_label_free",
            "reference_prediction",
            "adapter_correct",
            "reference_correct",
            "adapter_raw_output",
            "prompt",
            "v350_source_rule",
            "v357_source_rule",
            "v366_source_rule",
            "v343_prediction",
            "v350_prediction",
            "v357_prediction",
            "v366_prediction",
            "current_prediction",
            "baseline_prediction",
        ],
    )
    target_csv = args.output_dir / "v506_reference_gain_targets.csv"
    write_csv(
        target_csv,
        [row for row in rows if row["status"] == "reference_gain_target"],
        [
            "id",
            "family",
            "answer",
            "adapter_prediction_label_free",
            "reference_prediction",
            "adapter_raw_output",
            "prompt",
            "v350_source_rule",
            "v357_source_rule",
            "v366_source_rule",
            "v343_prediction",
            "v350_prediction",
            "v357_prediction",
            "v366_prediction",
            "current_prediction",
            "baseline_prediction",
        ],
    )
    rule_counts = Counter(
        "|".join(
            [
                str(row.get("family", "")),
                str(row.get("v350_source_rule", "")),
                str(row.get("v357_source_rule", "")),
                str(row.get("v366_source_rule", "")),
            ]
        )
        for row in rows
        if row["status"] == "reference_gain_target"
    )
    family_summary = {
        family: {status: family_status_counts[(family, status)] for status in sorted(status_counts)}
        for family in sorted({key[0] for key in family_status_counts})
    }
    manifest_out = {
        "schema_version": "kg1_v506_reference_signal_gap_v1",
        "generated_at_utc": generated_at,
        "v505_manifest": str(args.v505_manifest),
        "adapter_csv": str(adapter_path),
        "reference_csv": str(reference_path),
        "common_ids": len(common_ids),
        "status_counts": dict(status_counts),
        "family_status_counts": family_summary,
        "reference_gain_rule_counts": dict(rule_counts),
        "gap_csv": str(gap_csv),
        "target_csv": str(target_csv),
        "decision": {
            "status": "reference_signal_not_adapter_gain",
            "next_action": (
                "Use v506_reference_gain_targets.csv only as a transfer-target inventory; "
                "do not promote without a new adapter raw-output weak eval that beats the gate."
            ),
        },
    }
    manifest_path = args.output_dir / "v506_reference_signal_gap_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8")
    print("status_counts =", json.dumps(dict(status_counts), sort_keys=True), flush=True)
    print("family_status_counts =", json.dumps(family_summary, sort_keys=True), flush=True)
    print("gap_csv =", gap_csv, flush=True)
    print("target_csv =", target_csv, flush=True)
    print("manifest_path =", manifest_path, flush=True)
    print("decision =", json.dumps(manifest_out["decision"], sort_keys=True), flush=True)
    print("=== V506 REFERENCE SIGNAL GAP END ===", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v505-manifest", type=Path, default=DEFAULT_V505_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

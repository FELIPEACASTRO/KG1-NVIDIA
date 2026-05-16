#!/usr/bin/env python3
"""V507 CPU-only audit for V274 postprocessor under label-free extraction.

This script answers one narrow question:

    If we apply the already-deployable V274 numeric postprocessor to the best
    adapter raw outputs, what is the *label-free* weak impact?

It does not train, launch HF, package, or submit. Labels are used only to audit
the weak gate after predictions have been produced from raw outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402
from src.kg1_v274_numeric_postprocessor import postprocess_numeric_prediction  # noqa: E402


DEFAULT_ADAPTER_CSV = (
    REPO_ROOT
    / "artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/"
    / "v333_tong_bit_reasoner_gate_tong_bit_replace_predictions.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v507_v274_label_free_projection"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def family_counts(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row.get("family") or row.get("task_type") or row.get("type") or "")
        counts[family]["rows"] += 1
        counts[family]["correct"] += int(verify_answer(row.get("answer", ""), row.get(prediction_key, "")))
        counts[family]["truncated"] += int(str(row.get("truncated_bool", row.get("truncated", ""))).lower() == "true")
    return {family: dict(counter) for family, counter in sorted(counts.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(args.adapter_csv)
    if len(rows) != 315:
        raise RuntimeError(f"expected weak315 CSV, got {len(rows)} rows")
    if not rows or "raw_output" not in rows[0]:
        raise RuntimeError("adapter CSV must include raw_output for label-free extraction")

    audited_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("family") or row.get("task_type") or row.get("type") or "")
        label_free_prediction = extract_final_answer(row.get("raw_output", ""))
        post_prediction = label_free_prediction
        post_rule = "not_attempted"
        post_applied = False
        post_proof = ""
        if family == "equation_transform":
            decision = postprocess_numeric_prediction(
                str(row.get("prompt", "")),
                label_free_prediction,
                family=family,
                truncated=str(row.get("truncated_bool", row.get("truncated", ""))).lower() == "true",
            )
            post_prediction = decision.prediction
            post_rule = decision.rule
            post_applied = decision.applied
            post_proof = decision.proof

        base_correct = verify_answer(row.get("answer", ""), label_free_prediction)
        post_correct = verify_answer(row.get("answer", ""), post_prediction)
        audited = {
            "id": row.get("id", ""),
            "family": family,
            "answer": row.get("answer", ""),
            "stored_prediction": row.get("prediction", ""),
            "label_free_prediction": label_free_prediction,
            "v274_prediction": post_prediction,
            "label_free_correct": base_correct,
            "v274_correct": post_correct,
            "v274_applied": post_applied,
            "v274_rule": post_rule,
            "v274_proof": post_proof,
            "prompt": row.get("prompt", ""),
            "raw_output": row.get("raw_output", ""),
        }
        audited_rows.append(audited)
        if label_free_prediction != post_prediction or base_correct != post_correct:
            changes.append(audited)

    base_counts = family_counts(audited_rows, "label_free_prediction")
    v274_counts = family_counts(audited_rows, "v274_prediction")
    gains = [row for row in changes if not row["label_free_correct"] and row["v274_correct"]]
    losses = [row for row in changes if row["label_free_correct"] and not row["v274_correct"]]
    symbolic_boxing_overcounts = [
        row
        for row in audited_rows
        if row["family"] == "equation_transform"
        and row["stored_prediction"] != row["label_free_prediction"]
        and verify_answer(row["answer"], row["stored_prediction"])
        and not verify_answer(row["answer"], row["label_free_prediction"])
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{args.label}_rows.csv"
    changes_csv = output_dir / f"{args.label}_changes.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"

    columns = [
        "id",
        "family",
        "answer",
        "stored_prediction",
        "label_free_prediction",
        "v274_prediction",
        "label_free_correct",
        "v274_correct",
        "v274_applied",
        "v274_rule",
        "v274_proof",
        "prompt",
        "raw_output",
    ]
    write_csv(rows_csv, audited_rows, columns)
    write_csv(changes_csv, changes, columns)

    base_total = sum(int(item["correct"]) for item in base_counts.values())
    post_total = sum(int(item["correct"]) for item in v274_counts.values())
    decision = {
        "status": "postprocessor_signal_not_adapter_only_gain",
        "next_action": (
            "Do not package/submit this as adapter-only. Use the four label-free V274 gains "
            "and the symbolic boxing failure as transfer/debug evidence only."
        ),
    }
    if post_total > base_total and not losses:
        decision["cpu_signal"] = "positive_no_loss"
    else:
        decision["cpu_signal"] = "blocked_or_regressive"

    manifest = {
        "schema_version": "kg1_v507_v274_label_free_projection_v1",
        "generated_at_utc": utc_now(),
        "adapter_csv": str(args.adapter_csv),
        "rows_csv": str(rows_csv),
        "changes_csv": str(changes_csv),
        "base_family_counts_label_free": base_counts,
        "v274_family_counts_label_free": v274_counts,
        "base_total_label_free": base_total,
        "v274_total_label_free": post_total,
        "gains": len(gains),
        "losses": len(losses),
        "gain_ids": [row["id"] for row in gains],
        "loss_ids": [row["id"] for row in losses],
        "symbolic_boxing_overcount_ids": [row["id"] for row in symbolic_boxing_overcounts],
        "submit_safe": False,
        "package_allowed": False,
        "decision": decision,
    }
    write_json(manifest_json, manifest)
    print("=== V507 V274 LABEL-FREE PROJECTION START ===", flush=True)
    print("adapter_csv =", args.adapter_csv, flush=True)
    print("base_total_label_free =", base_total, flush=True)
    print("v274_total_label_free =", post_total, flush=True)
    print("base_family_counts_label_free =", json.dumps(base_counts, sort_keys=True), flush=True)
    print("v274_family_counts_label_free =", json.dumps(v274_counts, sort_keys=True), flush=True)
    print("gains =", len(gains), "losses =", len(losses), flush=True)
    print("symbolic_boxing_overcount_ids =", json.dumps(manifest["symbolic_boxing_overcount_ids"]), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V507 V274 LABEL-FREE PROJECTION END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-csv", type=Path, default=DEFAULT_ADAPTER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v507_v274_label_free_projection")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

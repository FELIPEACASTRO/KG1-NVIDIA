#!/usr/bin/env python3
"""Gate combined V274 equation and V300 bit postprocessors on labeled predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import canonical_family, classify_puzzle, verify_answer
from kg1_v274_numeric_postprocessor import postprocess_numeric_prediction
from kg1_v300_bit_fullbyte_postprocessor import postprocess_bit_prediction


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": str(row.get("answer", "")).strip(),
        "prediction": str(row.get("prediction", "")).strip(),
        "family": canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt)),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        ok = verify_answer(row["answer"], row.get(prediction_key, ""))
        total += 1
        correct += int(ok)
        truncated += int(row.get("truncated_bool", False))
        families[family]["rows"] += 1
        families[family]["correct"] += int(ok)
        families[family]["truncated"] += int(row.get("truncated_bool", False))
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": {key: dict(value) for key, value in sorted(families.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [normalize_row(row) for row in read_csv(args.input_csv)]
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        original = str(row["prediction"])
        eq_decision = postprocess_numeric_prediction(
            row["prompt"],
            original,
            family=row["family"],
            truncated=bool(row["truncated_bool"]),
        )
        bit_decision = postprocess_bit_prediction(
            row["prompt"],
            eq_decision.prediction,
            family=row["family"],
            truncated=bool(row["truncated_bool"]),
        )
        final_prediction = bit_decision.prediction
        baseline_ok = verify_answer(row["answer"], original)
        final_ok = verify_answer(row["answer"], final_prediction)
        out = dict(row)
        out["baseline_prediction"] = original
        out["prediction"] = final_prediction
        out["v274_eq_applied"] = eq_decision.applied
        out["v274_eq_rule"] = eq_decision.rule
        out["v274_eq_proof"] = eq_decision.proof
        out["v300_bit_applied"] = bit_decision.applied
        out["v300_bit_rule"] = bit_decision.rule
        out["v300_bit_proof"] = bit_decision.proof
        output_rows.append(out)
        audit_rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "baseline_prediction": original,
                "final_prediction": final_prediction,
                "v274_eq_applied": eq_decision.applied,
                "v274_eq_rule": eq_decision.rule,
                "v300_bit_applied": bit_decision.applied,
                "v300_bit_rule": bit_decision.rule,
                "baseline_correct": baseline_ok,
                "final_correct": final_ok,
                "gain": (not baseline_ok) and final_ok,
                "loss": baseline_ok and (not final_ok),
                "truncated": row["truncated_bool"],
            }
        )

    baseline_summary = summarize(rows, "prediction")
    final_summary = summarize(output_rows, "prediction")
    gains = [row for row in audit_rows if row["gain"]]
    losses = [row for row in audit_rows if row["loss"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions_csv": args.output_dir / "v302_combined_postprocessed_predictions.csv",
        "audit_csv": args.output_dir / "v302_combined_postprocessor_audit.csv",
        "manifest_json": args.output_dir / "v302_combined_postprocessor_gate_manifest.json",
    }
    write_csv(outputs["predictions_csv"], output_rows, list(output_rows[0]) if output_rows else [])
    write_csv(
        outputs["audit_csv"],
        audit_rows,
        [
            "id",
            "family",
            "baseline_prediction",
            "final_prediction",
            "v274_eq_applied",
            "v274_eq_rule",
            "v300_bit_applied",
            "v300_bit_rule",
            "baseline_correct",
            "final_correct",
            "gain",
            "loss",
            "truncated",
        ],
    )
    manifest = {
        "schema_version": "kg1_v302_combined_postprocessor_gate_v1",
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "baseline_summary": baseline_summary,
        "postprocessed_summary": final_summary,
        "gains": len(gains),
        "losses": len(losses),
        "gain_ids": [row["id"] for row in gains],
        "loss_ids": [row["id"] for row in losses],
        "rule_applied": {
            "v274_eq": dict(Counter(row["v274_eq_rule"] for row in audit_rows if row["v274_eq_applied"])),
            "v300_bit": dict(Counter(row["v300_bit_rule"] for row in audit_rows if row["v300_bit_applied"])),
        },
        "decision": {
            "decision": "combined_postprocessor_positive_signal" if gains and not losses else "combined_postprocessor_blocked",
            "reason": f"baseline={baseline_summary['correct']}; postprocessed={final_summary['correct']}; gains={len(gains)}; losses={len(losses)}",
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

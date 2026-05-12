#!/usr/bin/env python3
"""Gate the V300 bit postprocessor on labeled weak predictions."""

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
from kg1_v300_bit_fullbyte_postprocessor import postprocess_bit_prediction

FORBIDDEN_SOURCE_TERMS = ("answer", "correct", "verify_answer", "solution")


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
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "family": family,
        "answer": str(row.get("answer", "")).strip(),
        "prediction": str(row.get("prediction", "")).strip(),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    total_correct = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        pred = str(row.get(prediction_key, ""))
        ok = verify_answer(row["answer"], pred)
        total += 1
        total_correct += int(ok)
        truncated += int(row.get("truncated_bool", False))
        families[family]["rows"] += 1
        families[family]["correct"] += int(ok)
        families[family]["truncated"] += int(row.get("truncated_bool", False))
    return {
        "rows": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0.0,
        "truncated": truncated,
        "family": {key: dict(value) for key, value in sorted(families.items())},
    }


def source_guard(module_path: Path) -> dict[str, Any]:
    text = module_path.read_text(encoding="utf-8")
    hits = [term for term in FORBIDDEN_SOURCE_TERMS if term in text]
    return {
        "module_path": str(module_path),
        "sha256": sha256_file(module_path),
        "forbidden_terms": list(FORBIDDEN_SOURCE_TERMS),
        "forbidden_hits": hits,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [normalize_row(row) for row in read_csv(args.input_csv)]
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        original = str(row["prediction"])
        decision = postprocess_bit_prediction(
            row["prompt"],
            original,
            family=row["family"],
            truncated=bool(row["truncated_bool"]),
        )
        out["baseline_prediction"] = original
        out["prediction"] = decision.prediction
        out["bit_postprocessor_applied"] = decision.applied
        out["bit_postprocessor_rule"] = decision.rule
        out["bit_postprocessor_proof"] = decision.proof
        baseline_ok = verify_answer(row["answer"], original)
        post_ok = verify_answer(row["answer"], decision.prediction)
        audit_rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "baseline_prediction": original,
                "postprocessed_prediction": decision.prediction,
                "applied": decision.applied,
                "rule": decision.rule,
                "proof": decision.proof,
                "baseline_correct": baseline_ok,
                "postprocessed_correct": post_ok,
                "gain": (not baseline_ok) and post_ok,
                "loss": baseline_ok and (not post_ok),
            }
        )
        output_rows.append(out)

    baseline_summary = summarize(rows, "prediction")
    post_summary = summarize(output_rows, "prediction")
    applied = [row for row in audit_rows if row["applied"]]
    gains = [row for row in audit_rows if row["gain"]]
    losses = [row for row in audit_rows if row["loss"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "postprocessed_predictions_csv": args.output_dir / "v301_bit_postprocessed_predictions.csv",
        "audit_csv": args.output_dir / "v301_bit_postprocessor_audit.csv",
        "manifest_json": args.output_dir / "v301_bit_postprocessor_gate_manifest.json",
    }
    columns = list(output_rows[0]) if output_rows else []
    write_csv(outputs["postprocessed_predictions_csv"], output_rows, columns)
    write_csv(
        outputs["audit_csv"],
        audit_rows,
        [
            "id",
            "family",
            "baseline_prediction",
            "postprocessed_prediction",
            "applied",
            "rule",
            "proof",
            "baseline_correct",
            "postprocessed_correct",
            "gain",
            "loss",
        ],
    )
    bit_after = post_summary["family"].get("bit_manipulation", {})
    eq_after = post_summary["family"].get("equation_transform", {})
    guard = source_guard(REPO_ROOT / "src" / "kg1_v300_bit_fullbyte_postprocessor.py")
    weak_gate_pass = (
        int(post_summary["correct"]) >= args.weak_total_min
        and int(bit_after.get("correct", 0)) >= args.weak_bit_min
        and int(eq_after.get("correct", 0)) >= args.weak_eq_min
        and not losses
        and not guard["forbidden_hits"]
    )
    manifest = {
        "schema_version": "kg1_v301_bit_postprocessor_gate_v1",
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "baseline_summary": baseline_summary,
        "postprocessed_summary": post_summary,
        "applied_rows": len(applied),
        "gains": len(gains),
        "losses": len(losses),
        "source_guard": guard,
        "weak_gate": {
            "pass": weak_gate_pass,
            "weak_total_min": args.weak_total_min,
            "weak_eq_min": args.weak_eq_min,
            "weak_bit_min": args.weak_bit_min,
        },
        "decision": {
            "decision": "v301_bit_postprocessor_passes_weak_gate" if weak_gate_pass else "v301_bit_postprocessor_blocked",
            "reason": f"baseline={baseline_summary['correct']}; postprocessed={post_summary['correct']}; "
            f"bit={bit_after.get('correct', 0)}; eq={eq_after.get('correct', 0)}; gains={len(gains)}; losses={len(losses)}",
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
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=137)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

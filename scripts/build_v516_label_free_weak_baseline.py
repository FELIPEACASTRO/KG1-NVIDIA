#!/usr/bin/env python3
"""Build a submit-safe label-free weak baseline CSV from a raw-output CSV.

This CPU-only utility exists to prevent a subtle evaluation bug: historical
weak CSVs may contain a `prediction` column created by an older extractor while
also containing `raw_output`.  Solver gates must use the answer extracted from
`raw_output` with the public label-free extractor, not the stored prediction.

The script does not train, launch jobs, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import (  # noqa: E402
    canonical_family,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)


DEFAULT_INPUT_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v516_label_free_weak_baseline"


OUTPUT_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "family",
    "type",
    "prediction",
    "raw_output",
    "stored_prediction",
    "stored_correct",
    "label_free_correct",
    "correct",
    "truncated",
    "metric_source",
    "prompt_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y"}


def infer_family(row: dict[str, str]) -> str:
    value = str(row.get("family") or row.get("type") or row.get("task_type") or "").strip()
    if value:
        return canonical_family(value)
    return canonical_family(classify_puzzle(str(row.get("prompt", ""))))


def convert_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    raw_output = str(row.get("raw_output", ""))
    stored_prediction = str(row.get("prediction", "")).strip()
    if raw_output.strip():
        label_free_prediction = extract_final_answer(raw_output)
        metric_source = "raw_output_label_free"
    else:
        label_free_prediction = stored_prediction
        metric_source = "stored_prediction_only"
    family = infer_family(row)
    stored_correct = verify_answer(answer, stored_prediction)
    label_free_correct = verify_answer(answer, label_free_prediction)
    return {
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "type": family,
        "prediction": label_free_prediction,
        "raw_output": raw_output,
        "stored_prediction": stored_prediction,
        "stored_correct": stored_correct,
        "label_free_correct": label_free_correct,
        "correct": label_free_correct,
        "truncated": truthy(row.get("truncated", row.get("truncated_bool", ""))),
        "metric_source": metric_source,
        "prompt_sha256": sha256_text(prompt),
    }


def family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        out[family]["rows"] += 1
        out[family]["label_free_correct"] += int(bool(row["label_free_correct"]))
        out[family]["stored_correct"] += int(bool(row["stored_correct"]))
        out[family]["truncated"] += int(bool(row["truncated"]))
    return {key: dict(counter) for key, counter in sorted(out.items())}


def row_contract(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    return sha256_text(payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V516 LABEL-FREE WEAK BASELINE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("input_csv =", args.input_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    source_rows = read_csv(args.input_csv)
    rows = [convert_row(row) for row in source_rows]
    if len(rows) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(rows)}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate ids in weak baseline")
    metric_sources = Counter(str(row["metric_source"]) for row in rows)
    family = family_summary(rows)
    output_csv = args.output_dir / "v516_label_free_v290_checkpoint6_baseline.csv"
    manifest_json = args.output_dir / "v516_label_free_weak_baseline_manifest.json"
    write_csv(output_csv, rows, OUTPUT_COLUMNS)
    manifest = {
        "schema_version": "kg1_v516_label_free_weak_baseline_v1",
        "generated_at_utc": utc_now(),
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "output_csv": str(output_csv),
        "output_sha256": sha256_file(output_csv),
        "rows": len(rows),
        "family_summary": family,
        "label_free_correct": sum(int(bool(row["label_free_correct"])) for row in rows),
        "stored_correct": sum(int(bool(row["stored_correct"])) for row in rows),
        "stored_vs_label_free_diff_rows": sum(
            int(bool(row["stored_correct"]) != bool(row["label_free_correct"])) for row in rows
        ),
        "prediction_changed_rows": sum(str(row["stored_prediction"]) != str(row["prediction"]) for row in rows),
        "metric_sources": dict(metric_sources),
        "row_contract_sha256": row_contract(rows),
        "outputs": {
            "csv": str(output_csv),
            "manifest_json": str(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    print("family_summary =", json.dumps(family, sort_keys=True), flush=True)
    print("label_free_correct =", manifest["label_free_correct"], flush=True)
    print("stored_correct =", manifest["stored_correct"], flush=True)
    print("stored_vs_label_free_diff_rows =", manifest["stored_vs_label_free_diff_rows"], flush=True)
    print("row_contract_sha256 =", manifest["row_contract_sha256"], flush=True)
    print("output_csv =", output_csv, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V516 LABEL-FREE WEAK BASELINE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    row = {
        "id": "x",
        "prompt": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        "answer": "00000001",
        "type": "bit_manipulation",
        "prediction": "00000000",
        "raw_output": "scratch \\boxed{00000001}",
        "truncated": "false",
    }
    converted = convert_row(row)
    if converted["prediction"] != "00000001" or not converted["label_free_correct"]:
        raise AssertionError(converted)
    print("v516_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

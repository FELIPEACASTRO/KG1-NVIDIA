#!/usr/bin/env python3
"""Analyze V435C adapter probe outputs against public train labels.

This is a CPU analysis gate. It consumes V435C raw outputs collected without
labels, then joins public train labels by id after collection to identify real
adapter misses in permitted public train rows. It does not train or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import canonical_answer, classify_puzzle, verify_answer  # noqa: E402


DEFAULT_TRAIN_CSV = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_REFERENCE_WEAK_CSV = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_REFERENCE_FULL_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435d_adapter_probe_output_analysis"

DETAIL_COLUMNS = [
    "id",
    "family",
    "correct",
    "truncated",
    "answer",
    "prediction",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "completion_tokens",
    "finish_reason",
    "adapter_repo",
    "adapter_subfolder",
    "decode_config_sha256",
    "prompt",
    "raw_output",
]

SUMMARY_COLUMNS = ["family", "rows", "correct", "accuracy", "misses", "truncated"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def reference_set(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            ids.add(rid)
        prompt = str(row.get("prompt") or row.get("generated_prompt") or "")
        if prompt:
            prompt_hashes.add(sha256_text(normalize_prompt(prompt)))
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows), "ids": ids, "prompt_hashes": prompt_hashes}


def summarize(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0, "truncated": 0})
    for row in detail_rows:
        family = str(row.get("family", "unknown"))
        counts[family]["rows"] += 1
        counts[family]["correct"] += int(truthy(row.get("correct")))
        counts[family]["truncated"] += int(truthy(row.get("truncated")))
    out: list[dict[str, Any]] = []
    for family in sorted(counts):
        item = counts[family]
        rows = item["rows"]
        correct = item["correct"]
        out.append(
            {
                "family": family,
                "rows": rows,
                "correct": correct,
                "accuracy": correct / rows if rows else 0.0,
                "misses": rows - correct,
                "truncated": item["truncated"],
            }
        )
    total_rows = sum(item["rows"] for item in counts.values())
    total_correct = sum(item["correct"] for item in counts.values())
    total_truncated = sum(item["truncated"] for item in counts.values())
    out.append(
        {
            "family": "OVERALL",
            "rows": total_rows,
            "correct": total_correct,
            "accuracy": total_correct / total_rows if total_rows else 0.0,
            "misses": total_rows - total_correct,
            "truncated": total_truncated,
        }
    )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435D ADAPTER PROBE OUTPUT ANALYSIS START ===", flush=True)
    print("raw_outputs_csv =", args.raw_outputs_csv, flush=True)
    print("competition_train_csv =", args.competition_train_csv, flush=True)
    print("reference_weak_csv =", args.reference_weak_csv, flush=True)
    print("reference_full_csv =", args.reference_full_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.raw_outputs_csv.is_file():
        raise FileNotFoundError(args.raw_outputs_csv)
    if not args.competition_train_csv.is_file():
        raise FileNotFoundError(args.competition_train_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weak_ref = reference_set(args.reference_weak_csv)
    full_ref = reference_set(args.reference_full_csv)
    reference_ids = set(weak_ref["ids"]) | set(full_ref["ids"])
    reference_hashes = set(weak_ref["prompt_hashes"]) | set(full_ref["prompt_hashes"])

    train_rows = read_csv(args.competition_train_csv)
    train_by_id = {str(row.get("id", "")).strip(): row for row in train_rows}
    raw_rows = read_csv(args.raw_outputs_csv)
    detail_rows: list[dict[str, Any]] = []
    blocked_reasons: Counter[str] = Counter()

    for row in raw_rows:
        rid = str(row.get("id", "")).strip()
        train = train_by_id.get(rid)
        if not train:
            blocked_reasons["missing_public_train_label"] += 1
            continue
        prompt = str(row.get("prompt") or train.get("prompt") or "")
        prompt_norm_hash = str(row.get("prompt_normalized_sha256") or sha256_text(normalize_prompt(prompt)))
        if rid in reference_ids:
            blocked_reasons["reference_id_overlap"] += 1
            continue
        if prompt_norm_hash in reference_hashes:
            blocked_reasons["reference_prompt_overlap"] += 1
            continue
        answer = canonical_answer(train.get("answer", ""))
        prediction = canonical_answer(row.get("prediction", ""))
        correct = verify_answer(answer, prediction)
        finish_reason = str(row.get("finish_reason", ""))
        family = str(row.get("family") or classify_puzzle(prompt))
        detail_rows.append(
            {
                **row,
                "family": family,
                "answer": answer,
                "prediction": prediction,
                "correct": correct,
                "truncated": finish_reason == "length",
                "prompt": prompt,
                "prompt_sha256": str(row.get("prompt_sha256") or sha256_text(prompt.replace("\r\n", "\n"))),
                "prompt_normalized_sha256": prompt_norm_hash,
            }
        )

    summary_rows = summarize(detail_rows)
    miss_rows = [row for row in detail_rows if not truthy(row.get("correct"))]
    detail_csv = args.output_dir / f"{args.label}_detail.csv"
    miss_csv = args.output_dir / f"{args.label}_misses.csv"
    summary_csv = args.output_dir / f"{args.label}_summary.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)
    write_csv(miss_csv, miss_rows, DETAIL_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    manifest = {
        "schema_version": "kg1_v435d_adapter_probe_output_analysis_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_public_train": True,
            "weak_full_used_for_training": False,
            "purpose": "Identify permitted public-train adapter misses for the next hard-negative builder.",
        },
        "inputs": {
            "raw_outputs_csv": str(args.raw_outputs_csv),
            "raw_outputs_csv_sha256": sha256_file(args.raw_outputs_csv),
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
            "reference_weak_csv": str(args.reference_weak_csv),
            "reference_weak_sha256": weak_ref["sha256"],
            "reference_full_csv": str(args.reference_full_csv),
            "reference_full_sha256": full_ref["sha256"],
        },
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "summary": summary_rows,
        "outputs": {
            "detail_csv": str(detail_csv),
            "detail_sha256": sha256_file(detail_csv),
            "misses_csv": str(miss_csv),
            "misses_sha256": sha256_file(miss_csv),
            "summary_csv": str(summary_csv),
            "summary_sha256": sha256_file(summary_csv),
            "manifest_json": str(manifest_json),
        },
        "next_action": "If equation misses exist, generate certified rule-class pairs from these locked raw outputs; do not train directly from uncategorized misses.",
    }
    write_json(manifest_json, manifest)
    print("summary =", json.dumps(summary_rows, sort_keys=True), flush=True)
    print("blocked_reasons =", json.dumps(dict(sorted(blocked_reasons.items())), sort_keys=True), flush=True)
    print("miss_rows =", len(miss_rows), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V435D ADAPTER PROBE OUTPUT ANALYSIS END ===", flush=True)
    return manifest


def self_test() -> None:
    assert verify_answer("1010", "1010")
    assert not verify_answer("1010", "0101")
    rows = summarize(
        [
            {"family": "equation_transform", "correct": True, "truncated": False},
            {"family": "equation_transform", "correct": False, "truncated": False},
            {"family": "bit_manipulation", "correct": True, "truncated": True},
        ]
    )
    overall = [row for row in rows if row["family"] == "OVERALL"][0]
    assert overall["rows"] == 3 and overall["correct"] == 2 and overall["truncated"] == 1
    print("v435d_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-outputs-csv", type=Path, required=True)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--reference-weak-csv", type=Path, default=DEFAULT_REFERENCE_WEAK_CSV)
    parser.add_argument("--reference-full-csv", type=Path, default=DEFAULT_REFERENCE_FULL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435d_adapter_probe_output_analysis")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

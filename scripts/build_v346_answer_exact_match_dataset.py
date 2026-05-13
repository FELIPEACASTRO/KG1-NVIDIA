#!/usr/bin/env python3
"""Build V346 answer-exact-match transfer dataset from V344.

V344 preserved the V343 rule classes but failed to move weak exact-match ACC.
This builder removes long equation traces and keeps only the final boxed answer,
while preserving bit replay rows as a guardrail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V344_DIR = REPO_ROOT / "artifacts/v344_v343_transfer_dataset/20260513T_minimal_transfer_v343"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v346_answer_exact_match_dataset/20260513T_cpu_gate"
SCHEMA_VERSION = "kg1_v346_answer_exact_match_dataset_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assistant_answer(answer: str) -> str:
    return "Final answer: " + r"\boxed{" + str(answer).strip() + "}"


def normalize_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise RuntimeError(f"bad messages for {row.get('id')}")
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise RuntimeError(f"bad message roles for {row.get('id')}")
    answer = str(row.get("answer", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    if not answer or not prompt:
        raise RuntimeError(f"empty answer/prompt for {row.get('id')}")
    metadata = dict(row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {})
    family = str(row.get("family", metadata.get("family", ""))).strip()
    source_dataset = str(metadata.get("source_dataset", row.get("source", ""))).strip()
    subcategory = str(row.get("subcategory", metadata.get("subcategory", ""))).strip()
    if family not in {"bit_manipulation", "equation_transform"}:
        raise RuntimeError(f"unexpected family {family!r} for {row.get('id')}")
    if metadata.get("weak_gate_rows_used_for_training") is not False:
        raise RuntimeError(f"weak gate row flag is not false for {row.get('id')}")
    if metadata.get("full_gate_rows_used_for_training") is not False:
        raise RuntimeError(f"full gate row flag is not false for {row.get('id')}")

    system = (
        "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
        "Infer the hidden rule from the examples and return exactly one boxed final answer."
    )
    metadata.update(
        {
            "source_dataset": source_dataset,
            "source_v344_id": row.get("id", ""),
            "source_v344_split": split,
            "v346_transform": "answer_exact_match_boxed_only",
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    return {
        "id": "v346_" + split + "_" + str(row.get("id", "")),
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "subcategory": subcategory,
        "source": source_dataset,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_answer(answer)},
        ],
        "metadata": metadata,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    subcategory_counts = Counter(str(row.get("subcategory", "")) for row in rows)
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [str(row.get("prompt", "")) for row in rows]
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "unique_ids": len(set(ids)),
        "unique_prompts": len(set(prompts)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V346 ANSWER EXACT MATCH DATASET START ===", flush=True)
    print("generated_at_utc =", datetime.now(timezone.utc).isoformat(), flush=True)
    print("v344_dir =", args.v344_dir, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.v344_dir / "v344_v343_minimal_transfer_manifest.json"
    train_path = args.v344_dir / "v344_v343_minimal_transfer_train.jsonl"
    val_path = args.v344_dir / "v344_v343_minimal_transfer_val.jsonl"
    v344_manifest = read_json(manifest_path)
    if v344_manifest.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("unexpected V344 manifest schema")
    expected_train_sha = str(v344_manifest.get("outputs", {}).get("train_sha256", ""))
    expected_val_sha = str(v344_manifest.get("outputs", {}).get("val_sha256", ""))
    if sha256_file(train_path) != expected_train_sha:
        raise RuntimeError("V344 train hash mismatch")
    if sha256_file(val_path) != expected_val_sha:
        raise RuntimeError("V344 val hash mismatch")

    train_rows = [normalize_row(row, "train") for row in read_jsonl(train_path)]
    val_rows = [normalize_row(row, "validation") for row in read_jsonl(val_path)]

    for split, rows in (("train", train_rows), ("validation", val_rows)):
        summary = summarize(rows)
        print(f"{split}_summary =", json.dumps(summary, sort_keys=True), flush=True)
        if summary["duplicate_ids"] or summary["duplicate_prompts"]:
            raise RuntimeError(f"{split} duplicate ids/prompts detected")
        if "bit_manipulation" not in summary["family_counts"] or "equation_transform" not in summary["family_counts"]:
            raise RuntimeError(f"{split} missing required family")
        if summary["family_counts"]["bit_manipulation"] < 160:
            raise RuntimeError(f"{split} bit guardrail too small")

    out_train = args.output_dir / "v346_answer_exact_match_train.jsonl"
    out_val = args.output_dir / "v346_answer_exact_match_val.jsonl"
    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v344_manifest_json": str(manifest_path),
            "v344_manifest_sha256": sha256_file(manifest_path),
            "v344_train_jsonl": str(train_path),
            "v344_train_sha256": sha256_file(train_path),
            "v344_val_jsonl": str(val_path),
            "v344_val_sha256": sha256_file(val_path),
        },
        "policy": {
            "purpose": "answer-exact-match LoRA transfer smoke after V344 preference objective failed",
            "assistant_format": r"Final answer: \boxed{answer}",
            "uses_weak_or_full_gate_rows": False,
            "equation_trace_removed": True,
            "bit_replay_preserved": True,
        },
        "validation": {
            "train": summarize(train_rows),
            "validation": summarize(val_rows),
        },
        "outputs": {
            "train_jsonl": str(out_train),
            "train_sha256": sha256_file(out_train),
            "val_jsonl": str(out_val),
            "val_sha256": sha256_file(out_val),
            "manifest_json": str(args.output_dir / "v346_answer_exact_match_manifest.json"),
        },
        "required_next_gate": [
            "scripts/run_v286_generic_tokenization_gate.py --assistant-final-answer-mode boxed_exact",
            "HF A100 smoke only after debug and push",
            "weak eval checkpoint-2 before any continuation",
        ],
    }
    write_json(Path(manifest["outputs"]["manifest_json"]), manifest)
    print("manifest_json =", manifest["outputs"]["manifest_json"], flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("=== V346 ANSWER EXACT MATCH DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v344-dir", type=Path, default=DEFAULT_V344_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

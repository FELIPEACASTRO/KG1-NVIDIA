#!/usr/bin/env python3
"""Build V439 final-answer-only pairs from V435E hard negatives.

V436B showed that full-text chosen/rejected preference targets moved the model
the wrong way. V438 found the chosen target mentioned the wrong adapter answer
and public-train label audit text. V439 keeps the same permitted hard-negative
rows but normalizes both sides to the same short final-answer template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("artifacts/v439_final_answer_only_pairs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def escape_boxed_answer(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def box_answer(value: object) -> str:
    return "\\boxed{" + escape_boxed_answer(value) + "}"


def final_answer(value: object) -> str:
    return "Final answer: " + box_answer(value)


def convert_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    answer = metadata.get("answer")
    adapter_prediction = metadata.get("adapter_prediction")
    if answer is None or adapter_prediction is None:
        raise ValueError(f"missing answer/adapter_prediction in row {row.get('id')}")
    if metadata.get("negative_type") != "hard_negative_adapter_exact_wrong":
        raise ValueError(f"non hard-negative row in V439 input: {row.get('id')}")

    chosen = final_answer(answer)
    rejected = final_answer(adapter_prediction)
    new_metadata = dict(metadata)
    new_metadata.update(
        {
            "schema_version": "kg1_v439_final_answer_only_pair_v1",
            "source_schema_version": metadata.get("schema_version", ""),
            "source_pair_id": row.get("id", ""),
            "target_style": "final_answer_only_equalized",
            "chosen_mentions_adapter_prediction": False,
            "chosen_mentions_public_train_label_audit": False,
            "split": split,
        }
    )
    messages = [dict(item) for item in row.get("messages", [])]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"unexpected messages format in row {row.get('id')}")
    messages[-1]["content"] = chosen
    return {
        **row,
        "id": "v439_" + str(row.get("id", "")),
        "chosen": chosen,
        "rejected": rejected,
        "messages": messages,
        "metadata": new_metadata,
        "source": "v439_final_answer_only_pairs",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter(str(row.get("family") or row.get("metadata", {}).get("family", "")) for row in rows)
    subcategories = Counter(
        str(row.get("subcategory") or row.get("metadata", {}).get("rule_class", "")) for row in rows
    )
    negative_types = Counter(str(row.get("metadata", {}).get("negative_type", "")) for row in rows)
    bad_chosen_text = sum(
        ("public-train label audit" in str(row.get("chosen", "")))
        or ("frozen adapter" in str(row.get("chosen", "")))
        or ("Rejected adapter" in str(row.get("chosen", "")))
        for row in rows
    )
    bad_rejected_text = sum(
        ("public-train label audit" in str(row.get("rejected", "")))
        or ("frozen adapter" in str(row.get("rejected", "")))
        or ("Rejected adapter" in str(row.get("rejected", "")))
        for row in rows
    )
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "negative_type_counts": dict(sorted(negative_types.items())),
        "bad_chosen_text_rows": bad_chosen_text,
        "bad_rejected_text_rows": bad_rejected_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "20260515T_v439_final_answer_only")
    parser.add_argument("--label", default="v439_final_answer_only_pairs")
    args = parser.parse_args()

    print("=== V439 FINAL ANSWER ONLY PAIRS START ===", flush=True)
    print(f"train_jsonl = {args.train_jsonl}", flush=True)
    print(f"val_jsonl = {args.val_jsonl}", flush=True)
    print(f"output_dir = {args.output_dir}", flush=True)
    train_rows = [convert_row(row, "train") for row in read_jsonl(args.train_jsonl)]
    val_rows = [convert_row(row, "validation") for row in read_jsonl(args.val_jsonl)]

    train_out = args.output_dir / f"{args.label}_train.jsonl"
    val_out = args.output_dir / f"{args.label}_val.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    manifest = {
        "schema_version": "kg1_v439_final_answer_only_pairs_manifest_v1",
        "label": args.label,
        "source_train_jsonl": str(args.train_jsonl),
        "source_val_jsonl": str(args.val_jsonl),
        "train_jsonl": str(train_out),
        "val_jsonl": str(val_out),
        "train_sha256": sha256_file(train_out),
        "val_sha256": sha256_file(val_out),
        "train_summary": summarize(train_rows),
        "val_summary": summarize(val_rows),
        "decision": {
            "builder_decision": "post_build_audit_required",
            "hf_gpu_allowed_by_builder_alone": False,
            "reason": "builder only creates pairs; V438 structural audit and pre-paid integration gate are required before GPU",
        },
    }
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("train_summary =", json.dumps(manifest["train_summary"], sort_keys=True), flush=True)
    print("val_summary =", json.dumps(manifest["val_summary"], sort_keys=True), flush=True)
    print("train_sha256 =", manifest["train_sha256"], flush=True)
    print("val_sha256 =", manifest["val_sha256"], flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V439 FINAL ANSWER ONLY PAIRS END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

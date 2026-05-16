#!/usr/bin/env python3
"""Build V515 by adding only verified full-byte residual bit traces to V514.

V514 dropped bit rows that could not be reproduced by the bit solver or the
V296 stride solver. V515 checks only those dropped rows with the conservative
full-byte solver and accepts a row only when the solver returns one unique
prediction that exactly matches the known training answer.

This is CPU-only. It never trains, launches HF, packages, or submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402
from src.kg1_v300_bit_fullbyte_postprocessor import solve_fullbyte  # noqa: E402


DEFAULT_V510_ROOT = (
    REPO_ROOT
    / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool"
)
DEFAULT_V514_ROOT = REPO_ROOT / "artifacts/v514_traceable_bit_v510_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v515_v514_fullbyte_residual_dataset"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def set_assistant(row: dict[str, Any], text: str) -> None:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = text
            return
    raise RuntimeError("row missing assistant message")


def original_id_from_v514(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("v514_original_id") or row.get("id") or "")


def accepted_fullbyte_trace(answer: str, proof: str) -> str:
    compact_proof = " ".join(str(proof).split())
    if len(compact_proof) > 420:
        compact_proof = compact_proof[:420].rstrip() + "..."
    return "\n".join(
        [
            "Rule: use the unique full-byte expression that matches every example.",
            "Gate: exactly one predicted query output survived the full-byte search.",
            "Proof summary: " + compact_proof,
            "Applying that same rule to the query gives the final 8-bit output.",
            f"Final answer: \\boxed{{{answer}}}",
        ]
    )


def make_v515_row(row: dict[str, Any], split: str, prediction: str, proof: str) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip()
    if prediction != answer:
        raise RuntimeError(f"full-byte prediction does not match answer for {row.get('id')}")
    out = json.loads(json.dumps(row))
    metadata = dict(out.get("metadata") if isinstance(out.get("metadata"), dict) else {})
    metadata.update(
        {
            "schema_version": "kg1_v515_v514_fullbyte_residual_dataset_v1",
            "source": "v515_fullbyte_residual_from_v510",
            "v515_original_id": str(row.get("id", "")),
            "v515_original_source": str(row.get("source") or metadata.get("source") or ""),
            "v515_trace_method": "fullbyte_unique_prediction",
            "v515_split": split,
            "v515_recovered_from_v514_dropped_bit": True,
            "weak_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    out["id"] = "v515_" + str(row.get("id", ""))
    out["source"] = "v515_fullbyte_residual_from_v510"
    out["source_dataset"] = "v515_v514_fullbyte_residual_dataset"
    out["metadata"] = metadata
    set_assistant(out, accepted_fullbyte_trace(answer, proof))
    if not verify_answer(answer, extract_final_answer(assistant_text(out))):
        raise RuntimeError(f"trace final answer mismatch for {row.get('id')}")
    return out


def recover_split(
    *,
    split: str,
    v510_rows: list[dict[str, Any]],
    v514_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted_original_ids = {
        original_id_from_v514(row)
        for row in v514_rows
        if str(row.get("family", "")).strip() == "bit_manipulation"
    }
    output_rows = list(v514_rows)
    counters: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    accepted_examples: list[str] = []
    ambiguous_examples: list[str] = []
    no_expression_examples: list[str] = []
    width_parse_examples: list[str] = []
    trace_words: list[int] = []

    for row in v510_rows:
        if str(row.get("family", "")).strip() != "bit_manipulation":
            continue
        original_id = str(row.get("id", ""))
        if original_id in converted_original_ids:
            counters["already_converted_by_v514"] += 1
            continue
        counters["v514_residual_bit_seen"] += 1
        prediction, rule, proof = solve_fullbyte(str(row.get("prompt", "")))
        rule_counts[rule] += 1
        if rule != "fullbyte_unique_prediction" or prediction is None:
            counters["residual_not_unique_fullbyte"] += 1
            if rule == "ambiguous_fullbyte_expression":
                if len(ambiguous_examples) < 20:
                    ambiguous_examples.append(original_id)
            elif rule == "no_fullbyte_expression":
                if len(no_expression_examples) < 20:
                    no_expression_examples.append(original_id)
            elif len(width_parse_examples) < 20:
                width_parse_examples.append(original_id)
            continue
        if not verify_answer(str(row.get("answer", "")).strip(), prediction):
            counters["unique_fullbyte_wrong_answer"] += 1
            continue
        recovered = make_v515_row(row, split, prediction, proof)
        output_rows.append(recovered)
        counters["residual_fullbyte_accepted"] += 1
        trace_words.append(len(assistant_text(recovered).split()))
        if len(accepted_examples) < 20:
            accepted_examples.append(original_id)

    family_counts = Counter(str(row.get("family", "")) for row in output_rows)
    summary = {
        "split": split,
        "v510_input_rows": len(v510_rows),
        "v514_input_rows": len(v514_rows),
        "output_rows": len(output_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "counts": dict(sorted(counters.items())),
        "fullbyte_rule_counts": dict(sorted(rule_counts.items())),
        "accepted_examples": accepted_examples,
        "ambiguous_examples": ambiguous_examples,
        "no_expression_examples": no_expression_examples,
        "parse_or_width_examples": width_parse_examples,
        "v515_trace_word_min": min(trace_words) if trace_words else 0,
        "v515_trace_word_p50": sorted(trace_words)[len(trace_words) // 2] if trace_words else 0,
        "v515_trace_word_max": max(trace_words) if trace_words else 0,
    }
    return output_rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V515 V514 FULLBYTE RESIDUAL DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v510_train_jsonl =", args.v510_train_jsonl, flush=True)
    print("v510_val_jsonl =", args.v510_val_jsonl, flush=True)
    print("v514_train_jsonl =", args.v514_train_jsonl, flush=True)
    print("v514_val_jsonl =", args.v514_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    v510_train = read_jsonl(args.v510_train_jsonl)
    v510_val = read_jsonl(args.v510_val_jsonl)
    v514_train = read_jsonl(args.v514_train_jsonl)
    v514_val = read_jsonl(args.v514_val_jsonl)

    train_out, train_summary = recover_split(split="train", v510_rows=v510_train, v514_rows=v514_train)
    val_out, val_summary = recover_split(split="validation", v510_rows=v510_val, v514_rows=v514_val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "v515_v514_fullbyte_residual_train.jsonl"
    val_path = args.output_dir / "v515_v514_fullbyte_residual_val.jsonl"
    manifest_path = args.output_dir / "v515_v514_fullbyte_residual_manifest.json"

    write_jsonl(train_path, train_out)
    write_jsonl(val_path, val_out)
    manifest = {
        "schema_version": "kg1_v515_v514_fullbyte_residual_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v510_train_jsonl": str(args.v510_train_jsonl),
            "v510_train_sha256": sha256_file(args.v510_train_jsonl),
            "v510_val_jsonl": str(args.v510_val_jsonl),
            "v510_val_sha256": sha256_file(args.v510_val_jsonl),
            "v514_train_jsonl": str(args.v514_train_jsonl),
            "v514_train_sha256": sha256_file(args.v514_train_jsonl),
            "v514_val_jsonl": str(args.v514_val_jsonl),
            "v514_val_sha256": sha256_file(args.v514_val_jsonl),
        },
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "decision": {
            "status": "dataset_ready_for_tokenization_and_v513_gate",
            "reason": "V515 adds only residual bit rows with unique verified full-byte predictions.",
            "gpu_allowed": False,
            "next_action": "Run V286 tokenization and V513 trace learnability gates before any paid GPU.",
        },
    }
    write_json(manifest_path, manifest)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V515 V514 FULLBYTE RESIDUAL DATASET END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v510-train-jsonl", type=Path, default=DEFAULT_V510_ROOT / "v510_canonical_active_training_pool_train.jsonl")
    parser.add_argument("--v510-val-jsonl", type=Path, default=DEFAULT_V510_ROOT / "v510_canonical_active_training_pool_val.jsonl")
    parser.add_argument("--v514-train-jsonl", type=Path, default=DEFAULT_V514_ROOT / "v514_traceable_bit_v510_train.jsonl")
    parser.add_argument("--v514-val-jsonl", type=Path, default=DEFAULT_V514_ROOT / "v514_traceable_bit_v510_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

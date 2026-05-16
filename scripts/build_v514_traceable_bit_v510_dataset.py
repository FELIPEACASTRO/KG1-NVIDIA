#!/usr/bin/env python3
"""Build V514 by replacing V510 bit answer-only rows with verified bit traces.

V513 showed the active V510 pool is blocked for GPU because every bit row is a
one-line answer target. V514 keeps the V510 equation rows unchanged and rewrites
only bit rows for which a deterministic local solver reproduces the row answer.
Unverified bit rows are dropped instead of kept as answer-only noise.

This is CPU-only. It never trains, launches HF, packages, or submits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402
from src.solvers.bit_manipulation_solver import BitManipulationSolver  # noqa: E402
from scripts.run_v296_bit_stride_solver_audit import solve_stride  # noqa: E402


DEFAULT_V510_ROOT = (
    REPO_ROOT
    / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool"
)
DEFAULT_TRAIN = DEFAULT_V510_ROOT / "v510_canonical_active_training_pool_train.jsonl"
DEFAULT_VAL = DEFAULT_V510_ROOT / "v510_canonical_active_training_pool_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v514_traceable_bit_v510_dataset"


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
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
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
    messages = row.get("messages", [])
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = text
            return
    raise RuntimeError("row missing assistant message")


def normalize_solver_cot(cot: str, answer: str) -> str:
    text = str(cot).strip()
    boxed_pattern = re.compile(r"\\boxed\{[^{}]*\}\s*$")
    text = boxed_pattern.sub("", text).rstrip()
    if text:
        text = text + "\n"
    return text + f"Final answer: \\boxed{{{answer}}}"


def stride_trace(prompt: str, answer: str, meta: dict[str, Any]) -> str:
    vector = meta.get("vector") if isinstance(meta.get("vector"), list) else []
    default_bits = int(meta.get("default_bits", 0) or 0)
    return "\n".join(
        [
            "Rule: match each output bit to verified input-bit relations, then concatenate b0..b7.",
            "Selected per-bit vector: " + " ".join(str(item) for item in vector),
            f"Default bit count: {default_bits}.",
            "The selected vector reproduces the examples and gives the query output.",
            f"Final answer: \\boxed{{{answer}}}",
        ]
    )


def traceable_bit_row(row: dict[str, Any], solver: BitManipulationSolver) -> tuple[dict[str, Any] | None, str]:
    answer = str(row.get("answer", "")).strip()
    prompt = str(row.get("prompt", ""))
    current_answer, current_cot, _solved = solver.solve(prompt)
    method = ""
    trace = ""
    if current_answer == answer:
        method = "bit_solver_v4"
        trace = normalize_solver_cot(str(current_cot), answer)
    else:
        stride_answer, stride_meta = solve_stride(prompt)
        if stride_answer == answer:
            method = "v296_stride_solver"
            trace = stride_trace(prompt, answer, stride_meta)
    if not method:
        return None, "no_verified_trace"

    out = json.loads(json.dumps(row))
    metadata = dict(out.get("metadata") if isinstance(out.get("metadata"), dict) else {})
    metadata.update(
        {
            "schema_version": "kg1_v514_traceable_bit_v510_dataset_v1",
            "source": "v514_traceable_bit_from_v510",
            "v514_trace_method": method,
            "v514_original_source": str(row.get("source") or metadata.get("source") or ""),
            "v514_original_id": str(row.get("id", "")),
            "v514_replaced_answer_only_bit_target": True,
            "weak_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    out["id"] = "v514_" + str(row.get("id", ""))
    out["source"] = "v514_traceable_bit_from_v510"
    out["source_dataset"] = "v514_traceable_bit_v510_dataset"
    out["metadata"] = metadata
    set_assistant(out, trace)
    if not verify_answer(answer, extract_final_answer(assistant_text(out))):
        raise RuntimeError(f"trace final answer mismatch for {row.get('id')}")
    return out, method


def convert_split(rows: list[dict[str, Any]], split: str, solver: BitManipulationSolver) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    dropped_examples: list[str] = []
    assistant_words: list[int] = []

    for row in rows:
        family = str(row.get("family", "")).strip()
        if family != "bit_manipulation":
            out_rows.append(row)
            counts["kept_equation_rows"] += 1
            continue
        counts["bit_rows_seen"] += 1
        converted, method = traceable_bit_row(row, solver)
        if converted is None:
            counts["bit_rows_dropped_unverified"] += 1
            if len(dropped_examples) < 20:
                dropped_examples.append(str(row.get("id", "")))
            continue
        methods[method] += 1
        counts["bit_rows_converted_to_trace"] += 1
        assistant_words.append(len(assistant_text(converted).split()))
        out_rows.append(converted)

    family_counts = Counter(str(row.get("family", "")) for row in out_rows)
    summary = {
        "split": split,
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "counts": dict(sorted(counts.items())),
        "methods": dict(sorted(methods.items())),
        "dropped_unverified_examples": dropped_examples,
        "bit_trace_word_min": min(assistant_words) if assistant_words else 0,
        "bit_trace_word_p50": sorted(assistant_words)[len(assistant_words) // 2] if assistant_words else 0,
        "bit_trace_word_max": max(assistant_words) if assistant_words else 0,
    }
    return out_rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V514 TRACEABLE BIT V510 DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("train_jsonl =", args.train_jsonl, flush=True)
    print("val_jsonl =", args.val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    solver = BitManipulationSolver()
    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    train_out, train_summary = convert_split(train_rows, "train", solver)
    val_out, val_summary = convert_split(val_rows, "validation", solver)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "v514_traceable_bit_v510_train.jsonl"
    val_path = args.output_dir / "v514_traceable_bit_v510_val.jsonl"
    manifest_path = args.output_dir / "v514_traceable_bit_v510_manifest.json"
    write_jsonl(train_path, train_out)
    write_jsonl(val_path, val_out)
    manifest = {
        "schema_version": "kg1_v514_traceable_bit_v510_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "train_jsonl": str(args.train_jsonl),
            "train_sha256": sha256_file(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "val_sha256": sha256_file(args.val_jsonl),
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
            "status": "dataset_ready_for_v513_and_tokenization_gate",
            "reason": "bit answer-only rows were replaced only where solver/stride reproduced the answer; unverified bit rows dropped",
            "gpu_allowed": False,
            "next_action": "Run V513 against V514 plus real tokenization gate before any HF GPU job.",
        },
    }
    write_json(manifest_path, manifest)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V514 TRACEABLE BIT V510 DATASET END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

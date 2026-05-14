"""Build V371 trace-style transfer dataset from V367.

V370 showed that boxed-only targets did not control V368 inference: all bit
rows still emitted the old long reasoning trace. V371 keeps the V367 synthetic
prompts and answers, but replaces the assistant target with a compact trace
that matches the model's observed bit-reasoning format and ends in a boxed
answer.

This script does not authorize HF. It only builds a CPU-gated dataset candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_bit_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples = re.findall(r"\b([01]{8})\s*->\s*([01]{8})\b", prompt)
    query_match = re.search(r"determine the output for:\s*([01]{8})", prompt, flags=re.IGNORECASE)
    if not examples or not query_match:
        raise ValueError("failed to parse bit prompt")
    return examples, query_match.group(1)


def bitsum(chars: str) -> str:
    ones = chars.count("1")
    if ones == 0 or ones == len(chars):
        return "a"
    return str(ones)


def trace_target(prompt: str, answer: str, expr: str) -> str:
    examples, query = parse_bit_prompt(prompt)
    lines: list[str] = ["We need to deduce the transformation by matching the example outputs.", ""]
    for idx, (_, output) in enumerate(examples):
        lines.append(f"Output {idx}: {output}")
        for bit_idx, bit in enumerate(output):
            lines.append(f"{bit_idx} {bit}")
        lines.append("")
    lines.append("Output bit columns (with bitsum as hash)")
    for bit_idx in range(8):
        col = "".join(output[bit_idx] for _, output in examples)
        lines.append(f"{bit_idx} {col} {bitsum(col)}")
    lines.extend(
        [
            "",
            f"Selected rule: {expr}",
            f"Applying the same rule to {query} gives {answer}.",
            f"Final answer: \\boxed{{{answer}}}",
        ]
    )
    return "\n".join(lines)


def convert_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row.get("metadata", {}))
        expr = str(meta.get("expr", "")).strip()
        answer = str(row.get("answer", "")).strip()
        prompt = str(row.get("prompt", ""))
        assistant = trace_target(prompt, answer, expr)
        messages = []
        for msg in row.get("messages", []):
            if msg.get("role") == "assistant":
                messages.append({"role": "assistant", "content": assistant})
            else:
                messages.append({"role": msg.get("role", ""), "content": msg.get("content", "")})
        meta.update(
            {
                "schema_version": "kg1_v371_v367_trace_style_transfer_dataset_v1",
                "completion_format": "trace_style_boxed_suffix",
                "source_v367_row_id": row.get("id", ""),
                "split": split,
            }
        )
        new_row = dict(row)
        new_row["id"] = str(row.get("id", "")).replace("v367_", "v371_", 1)
        new_row["messages"] = messages
        new_row["metadata"] = meta
        new_row["source"] = "v371_trace_style_from_v367"
        out.append(new_row)
    return out


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    families = Counter(str(row.get("family", "")) for row in rows)
    subcategories = Counter(str(row.get("subcategory", "")) for row in rows)
    source_ids = Counter(str(row.get("metadata", {}).get("source_id", "")) for row in rows)
    assistants = []
    prompt_hashes = set()
    ids = set()
    for row in rows:
        ids.add(str(row.get("id", "")))
        prompt_hashes.add(sha256_text(str(row.get("prompt", ""))))
        assistant = ""
        for msg in row.get("messages", []):
            if msg.get("role") == "assistant":
                assistant = str(msg.get("content", ""))
                break
        assistants.append(assistant)
    return {
        "split": split,
        "rows": len(rows),
        "unique_ids": len(ids),
        "prompt_hash_count": len(prompt_hashes),
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "source_id_count": len(source_ids),
        "assistant_trace_style_rows": sum(text.startswith("We need to deduce") for text in assistants),
        "assistant_contains_output_bit_columns_rows": sum("Output bit columns" in text for text in assistants),
        "assistant_boxed_suffix_rows": sum(re.search(r"\\boxed\{[01]{8}\}\s*$", text) is not None for text in assistants),
        "assistant_char_min": min(len(text) for text in assistants),
        "assistant_char_max": max(len(text) for text in assistants),
        "assistant_char_mean": sum(len(text) for text in assistants) / len(assistants),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v367-train-jsonl",
        type=Path,
        default=Path("artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_train.jsonl"),
    )
    parser.add_argument(
        "--v367-val-jsonl",
        type=Path,
        default=Path("artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/v367_v366_bit_ternary_transfer_val.jsonl"),
    )
    parser.add_argument(
        "--v370-manifest-json",
        type=Path,
        default=Path("artifacts/v370_v367_format_transfer_audit/20260514T_cpu_audit/v370_v367_format_transfer_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v371_v367_trace_style_transfer_dataset/20260514T_cpu_gate"),
    )
    args = parser.parse_args()

    print("=== V371 TRACE STYLE DATASET BUILD START ===", flush=True)
    print("v367_train_jsonl =", args.v367_train_jsonl, flush=True)
    print("v367_val_jsonl =", args.v367_val_jsonl, flush=True)
    print("v370_manifest_json =", args.v370_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    for path in [args.v367_train_jsonl, args.v367_val_jsonl, args.v370_manifest_json]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train = convert_rows(load_jsonl(args.v367_train_jsonl), "train")
    val = convert_rows(load_jsonl(args.v367_val_jsonl), "validation")
    train_hashes = {sha256_text(str(row.get("prompt", ""))) for row in train}
    val_hashes = {sha256_text(str(row.get("prompt", ""))) for row in val}
    prompt_overlap = train_hashes & val_hashes
    if prompt_overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(prompt_overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "v371_v367_trace_style_transfer_train.jsonl"
    val_out = args.output_dir / "v371_v367_trace_style_transfer_val.jsonl"
    manifest_out = args.output_dir / "v371_v367_trace_style_transfer_manifest.json"
    summary_out = args.output_dir.parent / "V371_RESULT_SUMMARY.md"
    write_jsonl(train_out, train)
    write_jsonl(val_out, val)

    train_summary = summarize(train, "train")
    val_summary = summarize(val, "validation")
    manifest = {
        "schema_version": "kg1_v371_v367_trace_style_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v367_train_jsonl": str(args.v367_train_jsonl),
            "v367_train_sha256": sha256_file(args.v367_train_jsonl),
            "v367_val_jsonl": str(args.v367_val_jsonl),
            "v367_val_sha256": sha256_file(args.v367_val_jsonl),
            "v370_manifest_json": str(args.v370_manifest_json),
            "v370_manifest_sha256": sha256_file(args.v370_manifest_json),
        },
        "outputs": {
            "train_jsonl": str(train_out),
            "train_sha256": sha256_file(train_out),
            "val_jsonl": str(val_out),
            "val_sha256": sha256_file(val_out),
            "manifest_json": str(manifest_out),
        },
        "validation": {
            "train": train_summary,
            "validation": val_summary,
            "train_val_prompt_overlap": len(prompt_overlap),
        },
        "policy": {
            "hf_gpu_allowed": False,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "assistant_final_answer_mode": "boxed_suffix",
            "next_gate": "run_v286_generic_tokenization_gate --assistant-final-answer-mode boxed_suffix",
            "finops": "HF remains blocked until tokenization and a new transfer rationale pass.",
        },
        "decision": {
            "decision": "v371_dataset_built_tokenization_required",
            "hf_gpu_allowed": False,
            "next_action": "Run V286 real tokenization gate. Do not launch HF from V371 without explicit weak-risk review.",
            "reason": (
                f"trace_train={train_summary['assistant_trace_style_rows']}/{train_summary['rows']}; "
                f"trace_val={val_summary['assistant_trace_style_rows']}/{val_summary['rows']}; "
                f"prompt_overlap={len(prompt_overlap)}"
            ),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_out.write_text(
        "\n".join(
            [
                "# V371 V367 trace-style transfer dataset",
                "",
                "Generated: 2026-05-14",
                "",
                "## Result",
                "",
                f"- Train rows: `{train_summary['rows']}`.",
                f"- Validation rows: `{val_summary['rows']}`.",
                f"- Train trace-style assistant rows: `{train_summary['assistant_trace_style_rows']}/{train_summary['rows']}`.",
                f"- Validation trace-style assistant rows: `{val_summary['assistant_trace_style_rows']}/{val_summary['rows']}`.",
                f"- Train boxed-suffix rows: `{train_summary['assistant_boxed_suffix_rows']}/{train_summary['rows']}`.",
                f"- Validation boxed-suffix rows: `{val_summary['assistant_boxed_suffix_rows']}/{val_summary['rows']}`.",
                f"- Train/validation prompt overlap: `{len(prompt_overlap)}`.",
                "",
                "## Decision",
                "",
                "Dataset built for CPU/tokenization review only. HF remains blocked until V286 real tokenization passes and the roadmap explicitly accepts a new smoke test.",
                "",
                "## Local artifacts",
                "",
                f"- Manifest: `{manifest_out}`",
                f"- Train: `{train_out}`",
                f"- Validation: `{val_out}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("val_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_out, flush=True)
    print("=== V371 TRACE STYLE DATASET BUILD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

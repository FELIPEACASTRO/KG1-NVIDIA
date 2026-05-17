#!/usr/bin/env python3
"""Build V523 targeted source-only trace pack.

V523 is the first dataset after the V521/V522 plateau audit. It intentionally
does not use weak/full rows as training examples. V522 uses weak/reference rows
only to choose rule families; V523 draws rows from permitted source-side trace
datasets and standardizes the assistant final answer to boxed suffix format.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v523_targeted_source_trace_pack"
DEFAULT_V304_TRAIN = ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_train.jsonl"
DEFAULT_V304_VAL = ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_val.jsonl"
DEFAULT_V390_TRAIN = ROOT / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/v390_v325_equation_no_loss_distill_sft_train.jsonl"
DEFAULT_V390_VAL = ROOT / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/v390_v325_equation_no_loss_distill_sft_val.jsonl"
DEFAULT_V516_WEAK = ROOT / "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv"
DEFAULT_FULL = ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_V522_MANIFEST = ROOT / "artifacts/v522_source_target_alignment_audit/v522_source_target_alignment_manifest.json"


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        for raw in handle:
            text = raw.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def row_family(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(row.get("family") or metadata.get("family") or "")


def row_subcategory(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("subtype") or "")


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row_metadata(row).get("answer") or "")


def row_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return ""


def assistant_index(row: dict[str, Any]) -> int | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            return index
    return None


def row_text(row: dict[str, Any]) -> str:
    parts = [row_prompt(row), row_answer(row), row_subcategory(row)]
    metadata = row_metadata(row)
    parts.extend(str(value) for value in metadata.values() if value is not None)
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def reference_prompt_hashes(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("prompt_sha256"):
                    out.add(str(row["prompt_sha256"]))
                elif row.get("prompt"):
                    out.add(sha256_text(str(row["prompt"])))
    return out


def normalize_boxed(row: dict[str, Any], *, bucket: str, split: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    answer = row_answer(out)
    index = assistant_index(out)
    if index is not None:
        messages = out["messages"]
        content = str(messages[index].get("content", ""))
        boxed = f"Final answer: \\\\boxed{{{answer}}}"
        if "\\boxed{" not in content:
            if "Final answer:" in content:
                content = re.sub(r"Final answer:\s*.*$", boxed, content, flags=re.DOTALL)
            else:
                content = content.rstrip() + "\n" + boxed
        messages[index]["content"] = content
    metadata = dict(row_metadata(out))
    metadata.update(
        {
            "schema_version": "kg1_v523_targeted_source_trace_pack_v1",
            "source": "v523_targeted_source_trace_pack",
            "source_dataset": "v523_targeted_source_trace_pack",
            "v523_bucket": bucket,
            "v523_split": split,
            "v523_original_id": str(row.get("id", "")),
            "v523_source_only": True,
            "gate_rows_used_for_training": False,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    out["metadata"] = metadata
    out["id"] = f"v523_{split}_{bucket}_{row.get('id', '')}"
    out["source"] = "v523_targeted_source_trace_pack"
    out["source_dataset"] = "v523_targeted_source_trace_pack"
    out["subcategory"] = bucket
    return out


def select_bucket(
    rows: list[dict[str, Any]],
    *,
    split: str,
    bucket: str,
    predicate: Callable[[dict[str, Any]], bool],
    limit: int,
    used_hashes: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if len(out) >= limit:
            break
        if not predicate(row):
            continue
        key = sha256_text(row_prompt(row) + "\n---ANSWER---\n" + row_answer(row))
        if key in used_hashes:
            continue
        used_hashes.add(key)
        out.append(normalize_boxed(row, bucket=bucket, split=split))
    return out


def family_is(value: str) -> Callable[[dict[str, Any]], bool]:
    return lambda row: row_family(row) == value


def text_has(pattern: str) -> Callable[[dict[str, Any]], bool]:
    regex = re.compile(pattern)
    return lambda row: bool(regex.search(row_text(row)))


def all_of(*predicates: Callable[[dict[str, Any]], bool]) -> Callable[[dict[str, Any]], bool]:
    return lambda row: all(predicate(row) for predicate in predicates)


def build_split(v304_rows: list[dict[str, Any]], v390_rows: list[dict[str, Any]], *, split: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    used_hashes: set[str] = set()
    quotas = {
        "train": {
            "bit_cho_trace": 260,
            "bit_maj3_trace": 260,
            "bit_par3_trace": 90,
            "bit_v300_gain_pattern_other": 180,
            "bit_fullbyte_ternary_other": 100,
            "equation_numeric_minus_signed": 80,
            "equation_numeric_add_direct": 80,
            "equation_numeric_colon_trailing_zero": 80,
            "equation_numeric_colon_absdiff": 80,
        },
        "validation": {
            "bit_cho_trace": 54,
            "bit_maj3_trace": 60,
            "bit_par3_trace": 20,
            "bit_v300_gain_pattern_other": 50,
            "bit_fullbyte_ternary_other": 30,
            "equation_numeric_minus_signed": 20,
            "equation_numeric_add_direct": 20,
            "equation_numeric_colon_trailing_zero": 20,
            "equation_numeric_colon_absdiff": 20,
        },
    }[split]

    selected: list[dict[str, Any]] = []
    bit = family_is("bit_manipulation")
    equation = family_is("equation_transform")
    bit_not_primary = lambda row: not any(marker in row_text(row) for marker in ("CHO(", "MAJ3(", "PAR3("))
    specs: list[tuple[str, list[dict[str, Any]], Callable[[dict[str, Any]], bool]]] = [
        ("bit_cho_trace", v304_rows, all_of(bit, text_has(r"\bCHO\("))),
        ("bit_maj3_trace", v304_rows, all_of(bit, text_has(r"\bMAJ3\("))),
        ("bit_par3_trace", v304_rows, all_of(bit, text_has(r"\bPAR3\("))),
        (
            "bit_v300_gain_pattern_other",
            v304_rows,
            all_of(bit, lambda row: "bit_fullbyte_v300_gain_pattern" in row_subcategory(row), bit_not_primary),
        ),
        (
            "bit_fullbyte_ternary_other",
            v304_rows,
            all_of(bit, lambda row: "bit_fullbyte_safe_ternary" in row_subcategory(row), bit_not_primary),
        ),
        (
            "equation_numeric_minus_signed",
            v390_rows,
            all_of(equation, lambda row: "equation_numeric_minus_signed" in row_subcategory(row)),
        ),
        (
            "equation_numeric_add_direct",
            v390_rows,
            all_of(equation, lambda row: "equation_numeric_add_direct" in row_subcategory(row)),
        ),
        (
            "equation_numeric_colon_trailing_zero",
            v390_rows,
            all_of(equation, lambda row: "equation_numeric_colon_trailing_zero" in row_subcategory(row)),
        ),
        (
            "equation_numeric_colon_absdiff",
            v390_rows,
            all_of(equation, lambda row: "equation_numeric_colon_absdiff" in row_subcategory(row)),
        ),
    ]
    bucket_counts: Counter[str] = Counter()
    for bucket, source_rows, predicate in specs:
        rows = select_bucket(
            source_rows,
            split=split,
            bucket=bucket,
            predicate=predicate,
            limit=quotas[bucket],
            used_hashes=used_hashes,
        )
        selected.extend(rows)
        bucket_counts[bucket] = len(rows)
    return selected, bucket_counts


def summarize(rows: list[dict[str, Any]], reference_hashes: set[str]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    prompt_answer_hashes: Counter[str] = Counter()
    prompt_hash_overlap = 0
    flags: Counter[str] = Counter()
    for row in rows:
        family_counts[row_family(row)] += 1
        bucket_counts[row_subcategory(row)] += 1
        prompt = row_prompt(row)
        answer = row_answer(row)
        prompt_answer_hashes[sha256_text(prompt + "\n---ANSWER---\n" + answer)] += 1
        if sha256_text(prompt) in reference_hashes:
            prompt_hash_overlap += 1
        metadata = row_metadata(row)
        for flag in ("weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if metadata.get(flag) not in (False, None):
                flags[flag] += 1
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "duplicate_prompt_answer": sum(1 for count in prompt_answer_hashes.values() if count > 1),
        "reference_prompt_overlap": prompt_hash_overlap,
        "training_flag_counts": dict(sorted(flags.items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir / utc_compact()
    output_dir.mkdir(parents=True, exist_ok=True)
    v304_train = read_jsonl(args.v304_train_jsonl)
    v304_val = read_jsonl(args.v304_val_jsonl)
    v390_train = read_jsonl(args.v390_train_jsonl)
    v390_val = read_jsonl(args.v390_val_jsonl)
    train_rows, train_bucket_counts = build_split(v304_train, v390_train, split="train")
    val_rows, val_bucket_counts = build_split(v304_val, v390_val, split="validation")
    reference_hashes = reference_prompt_hashes([args.weak_reference_csv, args.full_reference_csv])

    train_summary = summarize(train_rows, reference_hashes)
    val_summary = summarize(val_rows, reference_hashes)
    blockers: list[str] = []
    for label, summary in (("train", train_summary), ("validation", val_summary)):
        if summary["reference_prompt_overlap"]:
            blockers.append(f"{label}:reference_prompt_overlap")
        if summary["duplicate_prompt_answer"]:
            blockers.append(f"{label}:duplicate_prompt_answer")
        if summary["training_flag_counts"]:
            blockers.append(f"{label}:weak_full_training_flags")
    if train_summary["family_counts"].get("bit_manipulation", 0) < 700:
        blockers.append("train:bit_rows_lt_700")
    if train_summary["family_counts"].get("equation_transform", 0) < 300:
        blockers.append("train:equation_rows_lt_300")
    if val_summary["family_counts"].get("bit_manipulation", 0) < 120:
        blockers.append("validation:bit_rows_lt_120")
    if val_summary["family_counts"].get("equation_transform", 0) < 70:
        blockers.append("validation:equation_rows_lt_70")

    train_path = output_dir / "v523_targeted_source_trace_pack_train.jsonl"
    val_path = output_dir / "v523_targeted_source_trace_pack_val.jsonl"
    manifest_path = output_dir / "v523_targeted_source_trace_pack_manifest.json"
    comparison_path = output_dir / "V523_VS_V515.md"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    decision = {
        "gpu_allowed": False,
        "dataset_ready_for_gates": not blockers,
        "status": "dataset_ready_for_cpu_gates" if not blockers else "dataset_blocked",
        "reason": (
            "V523 is source-only and targeted to V522 no-loss rule families. "
            "It still requires V286 tokenization, V513 learnability, V521 transfer blocker, and pre-paid gates before GPU."
        ),
        "next_action": "Run V286 boxed_suffix tokenization gate and V513 trace learnability gate; do not run GPU from V523 until both pass.",
    }
    v522_manifest = read_json(args.v522_manifest_json) if args.v522_manifest_json.is_file() else {}
    manifest = {
        "version": "V523",
        "schema_version": "kg1_v523_targeted_source_trace_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "blockers": blockers,
        "inputs": {
            "v304_train_jsonl": str(args.v304_train_jsonl),
            "v304_train_sha256": sha256_file(args.v304_train_jsonl),
            "v304_val_jsonl": str(args.v304_val_jsonl),
            "v304_val_sha256": sha256_file(args.v304_val_jsonl),
            "v390_train_jsonl": str(args.v390_train_jsonl),
            "v390_train_sha256": sha256_file(args.v390_train_jsonl),
            "v390_val_jsonl": str(args.v390_val_jsonl),
            "v390_val_sha256": sha256_file(args.v390_val_jsonl),
            "v522_manifest_json": str(args.v522_manifest_json),
        },
        "v522_signal": v522_manifest.get("reference_signal_summary", {}),
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "train_bucket_counts": dict(sorted(train_bucket_counts.items())),
        "validation_bucket_counts": dict(sorted(val_bucket_counts.items())),
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
            "comparison_md": str(comparison_path),
        },
    }
    write_json(manifest_path, manifest)
    write_comparison(comparison_path, manifest)
    return manifest


def write_comparison(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V523 vs V515",
        "",
        "| Metric | V515 | V523 |",
        "|---|---:|---:|",
        "| train rows | 2491 | {rows} |".format(rows=manifest["train_summary"]["rows"]),
        "| val rows | 620 | {rows} |".format(rows=manifest["validation_summary"]["rows"]),
        "| train bit rows | 473 | {rows} |".format(rows=manifest["train_summary"]["family_counts"].get("bit_manipulation", 0)),
        "| train equation rows | 2018 | {rows} |".format(rows=manifest["train_summary"]["family_counts"].get("equation_transform", 0)),
        "| train CHO/MAJ3/PAR3 targeted | 7 approx in V515 | {rows} |".format(
            rows=sum(
                manifest["train_bucket_counts"].get(bucket, 0)
                for bucket in ("bit_cho_trace", "bit_maj3_trace", "bit_par3_trace")
            )
        ),
        "| GPU allowed now | no | no |",
        "",
        "V523 changes the training signal shape: it uses a smaller, targeted source-only pack "
        "with much more CHO/MAJ3/global ternary coverage than V515, but it still must pass CPU gates before any paid job.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    row = {
        "id": "r1",
        "family": "bit_manipulation",
        "prompt": "p",
        "answer": "01010101",
        "messages": [{"role": "assistant", "content": "Rule uses CHO(a,b,c).\nFinal answer: 01010101"}],
        "metadata": {"subcategory": "bit_fullbyte_safe_ternary"},
    }
    out = normalize_boxed(row, bucket="bit_cho_trace", split="train")
    content = out["messages"][0]["content"]
    if "\\boxed{01010101}" not in content or out["metadata"].get("weak_gate_rows_used_for_training") is not False:
        raise SystemExit("self-test failed")
    print("build_v523_targeted_source_trace_pack_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v304-train-jsonl", type=Path, default=DEFAULT_V304_TRAIN)
    parser.add_argument("--v304-val-jsonl", type=Path, default=DEFAULT_V304_VAL)
    parser.add_argument("--v390-train-jsonl", type=Path, default=DEFAULT_V390_TRAIN)
    parser.add_argument("--v390-val-jsonl", type=Path, default=DEFAULT_V390_VAL)
    parser.add_argument("--weak-reference-csv", type=Path, default=DEFAULT_V516_WEAK)
    parser.add_argument("--full-reference-csv", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--v522-manifest-json", type=Path, default=DEFAULT_V522_MANIFEST)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    manifest = build(args)
    print("v523_manifest =", manifest["outputs"]["manifest_json"], flush=True)
    print("v523_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("v523_train_summary =", json.dumps(manifest["train_summary"], sort_keys=True), flush=True)
    print("v523_validation_summary =", json.dumps(manifest["validation_summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

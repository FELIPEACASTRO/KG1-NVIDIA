#!/usr/bin/env python3
"""Build V361 boxed-only transfer data from the V358 verified bit dataset.

V360 showed that V359 trained long rule/check/final-answer completions and did
not use the preference hard negatives.  V361 keeps the same verified prompts and
answers, but makes the supervised target match the weak-eval instruction:
exactly one boxed answer and no reasoning text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kg1_v361_v357_boxed_only_transfer_dataset_v1"
DEFAULT_V358_MANIFEST = Path(
    "artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/"
    "v358_v357_bit_ternary_transfer_manifest.json"
)
DEFAULT_V360_MANIFEST = Path(
    "artifacts/v360_v359_transfer_failure_audit/20260514T_cpu_audit/"
    "v360_v359_transfer_failure_audit_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/v361_v357_boxed_only_transfer_dataset/20260514T_cpu_gate")

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden bit rule from the examples and return exactly one boxed answer."
)
BOXED_RE = re.compile(r"^\\boxed\{[01]{8}\}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
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


def prompt_key(row: dict[str, Any]) -> str:
    return sha256_text(re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip())


def boxed_answer(answer: str) -> str:
    return r"\boxed{" + answer + "}"


def flip_bit(answer: str, index: int) -> str:
    bits = list(answer)
    bits[index % len(bits)] = "0" if bits[index % len(bits)] == "1" else "1"
    return "".join(bits)


def convert_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    answer = str(row.get("answer", ""))
    if not re.fullmatch(r"[01]{8}", answer):
        raise RuntimeError(f"row {row.get('id')} has non-bit answer {answer!r}")
    prompt = str(row.get("prompt", ""))
    metadata = dict(row.get("metadata", {}))
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_dataset": "v358_v357_bit_ternary_transfer",
            "source_row_id": row.get("id"),
            "source_schema_version": row.get("metadata", {}).get("schema_version", ""),
            "completion_format": "boxed_only",
            "split": split,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    converted_id = "v361_boxed_only_" + str(row.get("id", "")).replace("v358_", "", 1)
    assistant = boxed_answer(answer)
    return {
        "id": converted_id,
        "prompt": prompt,
        "answer": answer,
        "family": row.get("family", "bit_manipulation"),
        "subcategory": row.get("subcategory", ""),
        "source": "v361_boxed_only_from_v358_verified_bit_rules",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def make_preferences(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    pref_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = str(row["answer"])
        wrong = flip_bit(answer, index)
        prompt = str(row["prompt"])
        base_metadata = dict(row.get("metadata", {}))
        for negative_type, rejected in (
            ("hard_negative_one_bit_flip_boxed_only", boxed_answer(wrong)),
            ("format_negative_raw_answer_no_box", answer),
        ):
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "negative_type": negative_type,
                    "preference_source_row_id": row["id"],
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                }
            )
            pref_rows.append(
                {
                    "id": row["id"] + "_" + negative_type,
                    "prompt": prompt,
                    "chosen": boxed_answer(answer),
                    "rejected": rejected,
                    "metadata": metadata,
                }
            )
    return pref_rows


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    subcategory_counts = Counter(str(row.get("subcategory", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    rule_counts = Counter(str(row.get("metadata", {}).get("rule_slug", "")) for row in rows)
    expr_counts = Counter(str(row.get("metadata", {}).get("expr", "")) for row in rows)
    ids = [str(row.get("id", "")) for row in rows]
    prompt_hashes = [prompt_key(row) for row in rows]
    assistant_texts = [str(row.get("messages", [{}, {}, {}])[2].get("content", "")) for row in rows]
    bad_rows = []
    for row, assistant in zip(rows, assistant_texts):
        if not BOXED_RE.fullmatch(assistant):
            bad_rows.append({"id": row.get("id"), "reason": "assistant_not_boxed_only", "assistant": assistant})
        metadata = row.get("metadata", {})
        if metadata.get("weak_gate_rows_used_for_training") is not False:
            bad_rows.append({"id": row.get("id"), "reason": "weak_gate_flag_not_false"})
        if metadata.get("full_gate_rows_used_for_training") is not False:
            bad_rows.append({"id": row.get("id"), "reason": "full_gate_flag_not_false"})
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompt_hashes) - len(set(prompt_hashes))
    if duplicate_ids:
        raise RuntimeError(f"{split} duplicate ids: {duplicate_ids}")
    if bad_rows:
        raise RuntimeError(f"{split} bad rows: {bad_rows[:20]}")
    return {
        "split": split,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "unique_rule_slugs": len(rule_counts),
        "unique_exprs": len(expr_counts),
        "unique_ids": len(set(ids)),
        "prompt_hash_count": len(set(prompt_hashes)),
        "duplicate_prompt_count_within_split": duplicate_prompts,
        "assistant_boxed_only_rows": sum(1 for text in assistant_texts if BOXED_RE.fullmatch(text)),
        "assistant_char_min": min(len(text) for text in assistant_texts) if assistant_texts else 0,
        "assistant_char_max": max(len(text) for text in assistant_texts) if assistant_texts else 0,
    }


def validate_against_v360(v360_manifest: dict[str, Any]) -> dict[str, Any]:
    decision = v360_manifest.get("decision", {})
    findings = v360_manifest.get("findings", [])
    if decision.get("decision") != "v360_blocks_more_hf_on_v358_v359":
        raise RuntimeError("V361 requires the V360 failure audit decision.")
    required_findings = [
        "The hard-negative preference files were not used",
        "Completion format differs",
    ]
    serialized = json.dumps(findings, sort_keys=True)
    missing = [item for item in required_findings if item not in serialized]
    if missing:
        raise RuntimeError("V360 manifest does not contain required findings: " + json.dumps(missing))
    return {
        "v360_decision": decision,
        "v360_manifest_findings_checked": required_findings,
    }


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    v358_manifest = read_json(args.v358_manifest_json)
    v360_manifest = read_json(args.v360_manifest_json)
    v360_check = validate_against_v360(v360_manifest)
    outputs = v358_manifest.get("outputs", {})
    train_source = Path(outputs["train_jsonl"])
    val_source = Path(outputs["val_jsonl"])
    train_rows = [convert_row(row, "train") for row in read_jsonl(train_source)]
    val_rows = [convert_row(row, "validation") for row in read_jsonl(val_source)]
    pref_train_rows = make_preferences(train_rows, "train")
    pref_val_rows = make_preferences(val_rows, "validation")

    train_prompt_overlap = {prompt_key(row) for row in train_rows} & {prompt_key(row) for row in val_rows}
    if train_prompt_overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(train_prompt_overlap)}")

    train_summary = summarize(train_rows, "train")
    val_summary = summarize(val_rows, "validation")

    train_path = output_dir / "v361_v357_boxed_only_transfer_train.jsonl"
    val_path = output_dir / "v361_v357_boxed_only_transfer_val.jsonl"
    pref_train_path = output_dir / "v361_v357_boxed_only_transfer_preferences_train.jsonl"
    pref_val_path = output_dir / "v361_v357_boxed_only_transfer_preferences_val.jsonl"
    manifest_path = output_dir / "v361_v357_boxed_only_transfer_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(pref_train_path, pref_train_rows)
    write_jsonl(pref_val_path, pref_val_rows)

    preference_summary = {
        "train_rows": len(pref_train_rows),
        "val_rows": len(pref_val_rows),
        "negative_types": {
            "hard_negative_one_bit_flip_boxed_only": len(train_rows) + len(val_rows),
            "format_negative_raw_answer_no_box": len(train_rows) + len(val_rows),
        },
    }
    decision = {
        "decision": "v361_boxed_only_dataset_built",
        "hf_gpu_allowed": False,
        "reason": (
            "V361 fixes V360 completion-format evidence and regenerates hard negatives, "
            "but still requires real tokenization gate before any HF smoke."
        ),
        "next_action": "Run V286 tokenization gate with assistant_final_answer_mode=boxed_only.",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "inputs": {
            "v358_manifest_json": str(args.v358_manifest_json),
            "v358_manifest_sha256": sha256_file(args.v358_manifest_json),
            "v360_manifest_json": str(args.v360_manifest_json),
            "v360_manifest_sha256": sha256_file(args.v360_manifest_json),
        },
        "source_v358_outputs": outputs,
        "v360_check": v360_check,
        "policy": {
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "completion_format": "boxed_only",
            "purpose": "CPU-gated format repair after V359 failed to transfer V357 teacher gains.",
        },
        "validation": {
            "train": train_summary,
            "validation": val_summary,
            "train_validation_prompt_overlap": len(train_prompt_overlap),
            "preference": preference_summary,
        },
        "outputs": {
            "manifest_json": str(manifest_path),
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "preferences_train_jsonl": str(pref_train_path),
            "preferences_train_sha256": sha256_file(pref_train_path),
            "preferences_val_jsonl": str(pref_val_path),
            "preferences_val_sha256": sha256_file(pref_val_path),
        },
        "decision": decision,
    }
    write_json(manifest_path, manifest)
    return manifest


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        source_manifest = tmp / "v358.json"
        train = tmp / "train.jsonl"
        val = tmp / "val.jsonl"
        sample = {
            "id": "v358_train_rule_000",
            "prompt": "00000000 -> 00000000\nNow, determine the output for: 11111111",
            "answer": "11111111",
            "family": "bit_manipulation",
            "subcategory": "bit_exact_global_ternary",
            "source": "toy",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "Final answer: \\boxed{11111111}"},
            ],
            "metadata": {
                "schema_version": "toy",
                "rule_slug": "toy_rule",
                "expr": "ID",
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
            },
        }
        val_sample = dict(sample)
        val_sample["id"] = "v358_validation_rule_000"
        val_sample["prompt"] = "11111111 -> 11111111\nNow, determine the output for: 00000000"
        val_sample["answer"] = "00000000"
        train.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        val.write_text(json.dumps(val_sample) + "\n", encoding="utf-8")
        source_manifest.write_text(
            json.dumps(
                {
                    "outputs": {
                        "train_jsonl": str(train),
                        "val_jsonl": str(val),
                        "preferences_train_jsonl": str(train),
                        "preferences_val_jsonl": str(val),
                    }
                }
            ),
            encoding="utf-8",
        )
        v360 = tmp / "v360.json"
        v360.write_text(
            json.dumps(
                {
                    "decision": {"decision": "v360_blocks_more_hf_on_v358_v359"},
                    "findings": [
                        {"finding": "The hard-negative preference files were not used"},
                        {"finding": "Completion format differs"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(v358_manifest_json=source_manifest, v360_manifest_json=v360, output_dir=tmp / "out")
        manifest = run_build(args)
        if manifest["validation"]["train"]["assistant_boxed_only_rows"] != 1:
            raise AssertionError("boxed-only train validation failed")
        if manifest["decision"]["decision"] != "v361_boxed_only_dataset_built":
            raise AssertionError("unexpected V361 decision")
    print("v361_boxed_only_transfer_dataset_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v358-manifest-json", type=Path, default=DEFAULT_V358_MANIFEST)
    parser.add_argument("--v360-manifest-json", type=Path, default=DEFAULT_V360_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()

    print("=== V361 BOXED ONLY TRANSFER DATASET START ===", flush=True)
    print("v358_manifest_json =", args.v358_manifest_json, flush=True)
    print("v360_manifest_json =", args.v360_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    manifest = run_build(args)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("validation =", json.dumps(manifest["validation"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V361 BOXED ONLY TRANSFER DATASET END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

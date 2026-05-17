#!/usr/bin/env python3
"""Generic CPU tokenization/leakage gate for KG1 chat JSONL datasets.

The gate validates JSONL structure, family/source counts, train/validation
separation, assistant loss masks, and prompt truncation before any HF GPU job.
It does not train, load base model weights, run generation, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    box_answer,
    extract_final_answer,
    extract_final_answer_for_expected,
    verify_answer,
)


DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"

BLOCKED_DATASET_MARKERS = {
    "v461_synthetic_numeric_probe_pack": "V461 contained a full-reference exact prompt/answer seed.",
    "v463_v462_synthetic_numeric_hard_negative_audit": "V463 depends on the quarantined V461/V462 route.",
    "v464_v463_numeric_multirule_dataset": "V464 contains contradictory rejected candidates.",
    "v468_v464_symbol_fix_dataset": "V468 still contains a full-reference exact prompt/answer seed.",
    "v447_v446_trace_dataset": "Current V447 contains hypothesis_formed contradictory traces.",
}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def blocked_dataset_matches(data_identity: str) -> list[dict[str, str]]:
    return [
        {"marker": marker, "reason": reason}
        for marker, reason in BLOCKED_DATASET_MARKERS.items()
        if marker in data_identity
    ]


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def normalize_answer(answer: Any) -> str:
    return re.sub(r"\s+", "", str(answer)).strip()


def prompt_key(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


def prompt_answer_key(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")) + "\0" + normalize_answer(row.get("answer", "")))


def load_reference_csv(path: Path) -> dict[str, set[str]]:
    ids: set[str] = set()
    prompts: set[str] = set()
    prompt_answers: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "prompt" not in fieldnames:
            raise RuntimeError(f"reference CSV missing prompt column: {path}")
        for row in reader:
            rid = str(row.get("id", "") or row.get("row_id", "")).strip()
            prompt = str(row.get("prompt", ""))
            answer = str(row.get("answer", ""))
            if rid:
                ids.add(rid)
            if prompt:
                prompts.add(sha256_text(normalize_prompt(prompt)))
            if prompt and answer:
                prompt_answers.add(sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer)))
    return {"ids": ids, "prompts": prompts, "prompt_answers": prompt_answers}


def reference_overlap_report(rows: list[dict[str, Any]], reference_csvs: list[Path]) -> dict[str, Any]:
    if not reference_csvs:
        return {
            "enabled": False,
            "reference_csvs": [],
            "id_overlap_count": 0,
            "prompt_overlap_count": 0,
            "prompt_answer_overlap_count": 0,
        }
    row_ids = {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}
    row_prompts = {prompt_key(row) for row in rows}
    row_prompt_answers = {prompt_answer_key(row) for row in rows}
    ref_ids: set[str] = set()
    ref_prompts: set[str] = set()
    ref_prompt_answers: set[str] = set()
    for path in reference_csvs:
        loaded = load_reference_csv(path)
        ref_ids.update(loaded["ids"])
        ref_prompts.update(loaded["prompts"])
        ref_prompt_answers.update(loaded["prompt_answers"])
    id_overlap = sorted(row_ids & ref_ids)
    prompt_overlap = sorted(row_prompts & ref_prompts)
    prompt_answer_overlap = sorted(row_prompt_answers & ref_prompt_answers)
    return {
        "enabled": True,
        "reference_csvs": [str(path) for path in reference_csvs],
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "prompt_answer_overlap_count": len(prompt_answer_overlap),
        "id_overlap_first10": id_overlap[:10],
        "prompt_overlap_first10": prompt_overlap[:10],
        "prompt_answer_overlap_first10": prompt_answer_overlap[:10],
    }


def validate_rows(
    rows: list[dict[str, Any]],
    split: str,
    min_rows: int,
    expected_sha256: str,
    assistant_final_answer_mode: str,
) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise RuntimeError(f"{split} row count below minimum: {len(rows)} < {min_rows}")
    ids: list[str] = []
    bad_rows: list[str] = []
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    answer_empty: list[str] = []
    for row in rows:
        rid = str(row.get("id", ""))
        ids.append(rid)
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        if not prompt or not answer:
            answer_empty.append(rid)
        family_counts[str(row.get("family", ""))] += 1
        subcategory_counts[str(row.get("subcategory", ""))] += 1
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        source_counts[str(metadata.get("source_dataset", row.get("source", "")))] += 1
        messages = row.get("messages")
        prompt_contract = str(metadata.get("prompt_contract", "legacy_system"))
        if not isinstance(messages, list):
            bad_rows.append(rid)
            continue
        roles = [message.get("role") for message in messages]
        if prompt_contract == "official_like":
            expected_suffix = str(metadata.get("prompt_suffix", PROMPT_SUFFIX))
            if len(messages) != 2 or roles != ["user", "assistant"]:
                bad_rows.append(rid)
                continue
            if messages[0].get("content") != prompt + expected_suffix:
                bad_rows.append(rid)
                continue
            assistant_content = str(messages[1].get("content", ""))
        else:
            if len(messages) != 3 or roles != ["system", "user", "assistant"]:
                bad_rows.append(rid)
                continue
            if messages[1].get("content") != prompt:
                bad_rows.append(rid)
                continue
            assistant_content = str(messages[2].get("content", ""))
        if assistant_final_answer_mode == "submit_safe_suffix":
            if "Final answer:" not in assistant_content:
                bad_rows.append(rid)
                continue
            extracted_answer = extract_final_answer(assistant_content)
            if not verify_answer(answer, extracted_answer):
                bad_rows.append(rid)
                continue
        else:
            final_answer_line = "Final answer: " + answer
            if assistant_final_answer_mode.startswith("boxed_"):
                final_answer_line = "Final answer: " + box_answer(answer)
            if assistant_final_answer_mode == "boxed_only":
                final_answer_line = box_answer(answer)
            if assistant_final_answer_mode == "exact" and assistant_content != final_answer_line:
                bad_rows.append(rid)
                continue
            if assistant_final_answer_mode == "suffix" and not assistant_content.rstrip().endswith(final_answer_line):
                bad_rows.append(rid)
                continue
            if assistant_final_answer_mode == "boxed_exact" and assistant_content != final_answer_line:
                bad_rows.append(rid)
                continue
            if assistant_final_answer_mode == "boxed_suffix" and not assistant_content.rstrip().endswith(final_answer_line):
                bad_rows.append(rid)
                continue
            if assistant_final_answer_mode == "boxed_only" and assistant_content != final_answer_line:
                bad_rows.append(rid)
                continue
            extracted_answer = extract_final_answer_for_expected(assistant_content, answer)
            if not verify_answer(answer, extracted_answer):
                bad_rows.append(rid)
                continue
        if assistant_final_answer_mode not in {
            "exact",
            "suffix",
            "boxed_exact",
            "boxed_suffix",
            "boxed_only",
            "submit_safe_suffix",
        }:
            raise RuntimeError(f"unknown assistant_final_answer_mode={assistant_final_answer_mode!r}")
        for flag in (
            "weak_gate_rows_used_for_training",
            "full_gate_rows_used_for_training",
            "gate_rows_used_for_training",
        ):
            if metadata.get(flag) is not False:
                bad_rows.append(rid)
                break
        rejected = str(metadata.get("rejected_candidate", "")).strip()
        if rejected and verify_answer(answer, rejected):
            bad_rows.append(rid)
            continue
        for candidate in re.findall(r"candidate '([^']*)' is rejected", assistant_content):
            if verify_answer(answer, candidate):
                bad_rows.append(rid)
                break
    if answer_empty:
        raise RuntimeError(f"{split} rows with empty prompt/answer: {answer_empty[:20]}")
    if bad_rows:
        raise RuntimeError(f"{split} bad rows: {bad_rows[:20]}")
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise RuntimeError(f"{split} duplicate ids: {duplicate_ids}")
    return {
        "rows": len(rows),
        "sha256_expected": expected_sha256,
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "unique_ids": len(set(ids)),
        "prompt_hash_count": len({prompt_key(row) for row in rows}),
        "prompt_answer_hash_count": len({prompt_answer_key(row) for row in rows}),
    }


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_completion_mask(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    require_offset_mask: bool,
) -> tuple[list[int], list[int], bool]:
    full_text = apply_chat_template(tokenizer, messages)
    assistant_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            assistant_text = str(message.get("content", ""))
            break
    if not assistant_text:
        raise RuntimeError("missing assistant text")
    assistant_start = full_text.rfind(assistant_text)
    if assistant_start < 0:
        raise RuntimeError("assistant text not found in rendered chat")
    try:
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(encoded["input_ids"])
        offsets = encoded.get("offset_mapping")
        if offsets and len(offsets) == len(input_ids):
            loss_mask = [1 if int(end) > assistant_start else 0 for _, end in offsets]
            return input_ids, loss_mask, True
    except (NotImplementedError, TypeError, ValueError):
        pass
    if require_offset_mask:
        raise RuntimeError("tokenizer did not provide offset mappings")
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_messages = [message for message in messages if message.get("role") != "assistant"]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    loss_mask = [0] * min(len(prompt_ids), len(input_ids)) + [1] * max(0, len(input_ids) - len(prompt_ids))
    return input_ids, loss_mask[: len(input_ids)], False


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return int(values[index])


def tokenize_rows(
    rows: list[dict[str, Any]],
    split: str,
    tokenizer: Any,
    max_length: int,
    max_prompt_truncation_rate: float,
    require_offset_mask: bool,
) -> dict[str, Any]:
    lengths: list[int] = []
    loss_lengths: list[int] = []
    by_family: dict[str, list[dict[str, int]]] = {}
    prompt_truncated = 0
    completion_tokens_dropped = 0
    offset_masks = 0
    fallback_masks = 0
    no_loss_rows: list[str] = []
    for row in rows:
        input_ids, loss_mask, used_offsets = build_completion_mask(tokenizer, row["messages"], require_offset_mask)
        if used_offsets:
            offset_masks += 1
        else:
            fallback_masks += 1
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            dropped_loss = sum(loss_mask[:overflow])
            completion_tokens_dropped += int(dropped_loss)
            if dropped_loss:
                raise RuntimeError(f"{split} completion tokens would be truncated for {row.get('id')}")
            prompt_truncated += 1
            input_ids = input_ids[overflow:]
            loss_mask = loss_mask[overflow:]
        loss_count = int(sum(loss_mask))
        if loss_count <= 0:
            no_loss_rows.append(str(row.get("id", "")))
        lengths.append(len(input_ids))
        loss_lengths.append(loss_count)
        family = str(row.get("family", "unknown"))
        by_family.setdefault(family, []).append({"tokens": len(input_ids), "loss_tokens": loss_count})
    if no_loss_rows:
        raise RuntimeError(f"{split} rows without assistant loss tokens: {no_loss_rows[:20]}")
    trunc_rate = prompt_truncated / max(1, len(rows))
    if trunc_rate > max_prompt_truncation_rate:
        raise RuntimeError(
            f"{split} prompt truncation rate {trunc_rate:.6f} exceeds {max_prompt_truncation_rate:.6f}"
        )
    family_summary: dict[str, Any] = {}
    for family, items in sorted(by_family.items()):
        token_values = [item["tokens"] for item in items]
        loss_values = [item["loss_tokens"] for item in items]
        family_summary[family] = {
            "rows": len(items),
            "token_p50": percentile(token_values, 0.50),
            "token_p90": percentile(token_values, 0.90),
            "token_p99": percentile(token_values, 0.99),
            "token_max": max(token_values) if token_values else 0,
            "loss_token_min": min(loss_values) if loss_values else 0,
            "loss_token_p50": percentile(loss_values, 0.50),
            "loss_token_max": max(loss_values) if loss_values else 0,
        }
    return {
        "rows": len(rows),
        "token_min": min(lengths) if lengths else 0,
        "token_p50": percentile(lengths, 0.50),
        "token_p90": percentile(lengths, 0.90),
        "token_p99": percentile(lengths, 0.99),
        "token_max": max(lengths) if lengths else 0,
        "token_mean": round(float(statistics.mean(lengths)), 3) if lengths else 0,
        "loss_token_min": min(loss_lengths) if loss_lengths else 0,
        "loss_token_p50": percentile(loss_lengths, 0.50),
        "loss_token_max": max(loss_lengths) if loss_lengths else 0,
        "offset_masks": offset_masks,
        "fallback_masks": fallback_masks,
        "prompt_truncated": prompt_truncated,
        "prompt_truncation_rate": trunc_rate,
        "completion_tokens_dropped": completion_tokens_dropped,
        "family_summary": family_summary,
    }


def load_tokenizer(args: argparse.Namespace) -> Any:
    if args.use_toy_tokenizer:
        class ToyTokenizer:
            is_fast = True
            pad_token = None
            eos_token = "<eos>"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
                text = "\n".join(str(message.get("content", "")) for message in messages)
                if add_generation_prompt:
                    text += "\nassistant:"
                return text

            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                ids = [ord(ch) % 127 for ch in text]
                result = {"input_ids": ids}
                if return_offsets_mapping:
                    result["offset_mapping"] = [(idx, idx + 1) for idx in range(len(ids))]
                return result

            def encode(self, text, add_special_tokens=False):
                return [ord(ch) % 127 for ch in text]

        return ToyTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the real tokenizer gate") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision or None,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V286 GENERIC TOKENIZATION GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("dataset_manifest_json =", args.dataset_manifest_json, flush=True)
    print("model_name =", args.model_name, flush=True)
    print("model_revision =", args.model_revision, flush=True)
    print("max_length =", args.max_length, flush=True)
    print("max_prompt_truncation_rate =", args.max_prompt_truncation_rate, flush=True)
    print("require_offset_mask =", args.require_offset_mask, flush=True)
    print("assistant_final_answer_mode =", args.assistant_final_answer_mode, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = read_json(args.dataset_manifest_json)
    data_identity = " ".join(
        [
            str(args.dataset_manifest_json),
            json.dumps(dataset_manifest.get("outputs", {}), sort_keys=True),
            json.dumps(dataset_manifest.get("inputs", {}), sort_keys=True),
            str(dataset_manifest.get("schema_version", "")),
            str(dataset_manifest.get("label", "")),
        ]
    )
    blocked = blocked_dataset_matches(data_identity)
    if blocked:
        raise RuntimeError("blocked quarantined dataset marker detected: " + json.dumps(blocked, sort_keys=True))
    train_path = Path(dataset_manifest["outputs"]["train_jsonl"])
    val_path = Path(dataset_manifest["outputs"]["val_jsonl"])
    expected_train_sha = str(dataset_manifest["outputs"].get("train_sha256", ""))
    expected_val_sha = str(dataset_manifest["outputs"].get("val_sha256", ""))
    observed_train_sha = sha256_file(train_path)
    observed_val_sha = sha256_file(val_path)
    print("train_jsonl =", train_path, flush=True)
    print("validation_jsonl =", val_path, flush=True)
    print("observed_train_sha256 =", observed_train_sha, flush=True)
    print("observed_val_sha256 =", observed_val_sha, flush=True)
    if expected_train_sha and observed_train_sha != expected_train_sha:
        raise RuntimeError(f"train SHA mismatch: expected {expected_train_sha}, got {observed_train_sha}")
    if expected_val_sha and observed_val_sha != expected_val_sha:
        raise RuntimeError(f"validation SHA mismatch: expected {expected_val_sha}, got {observed_val_sha}")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    train_validation = validate_rows(
        train_rows,
        "train",
        args.min_train_rows,
        expected_train_sha,
        args.assistant_final_answer_mode,
    )
    val_validation = validate_rows(
        val_rows,
        "validation",
        args.min_val_rows,
        expected_val_sha,
        args.assistant_final_answer_mode,
    )
    train_prompt_answer = {prompt_answer_key(row) for row in train_rows}
    val_prompt_answer = {prompt_answer_key(row) for row in val_rows}
    train_prompt = {prompt_key(row) for row in train_rows}
    val_prompt = {prompt_key(row) for row in val_rows}
    prompt_answer_overlap = len(train_prompt_answer & val_prompt_answer)
    prompt_overlap = len(train_prompt & val_prompt)
    print("train_val_prompt_answer_overlap =", prompt_answer_overlap, flush=True)
    print("train_val_prompt_overlap =", prompt_overlap, flush=True)
    if prompt_answer_overlap or prompt_overlap:
        raise RuntimeError(
            f"train/validation overlap detected: prompt_answer={prompt_answer_overlap} prompt={prompt_overlap}"
        )
    reference_csvs = [Path(item) for item in getattr(args, "reference_csvs", []) or []]
    manifest_reference_csvs = [
        Path(item)
        for item in dataset_manifest.get("forbidden_reference_csvs", dataset_manifest.get("reference_csvs", []))
        if str(item).strip()
    ]
    reference_csvs.extend(manifest_reference_csvs)
    reference_csvs = [path for path in reference_csvs if path.exists()]
    reference_overlap = reference_overlap_report(train_rows + val_rows, reference_csvs)
    print("reference_overlap =", json.dumps(reference_overlap, sort_keys=True), flush=True)
    if reference_overlap["id_overlap_count"] or reference_overlap["prompt_overlap_count"] or reference_overlap["prompt_answer_overlap_count"]:
        raise RuntimeError("reference overlap detected: " + json.dumps(reference_overlap, sort_keys=True))

    tokenizer = load_tokenizer(args)
    tokenizer_info = {
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "toy": bool(args.use_toy_tokenizer),
    }
    print("tokenizer_info =", json.dumps(tokenizer_info, sort_keys=True), flush=True)
    train_tokenization = tokenize_rows(
        train_rows,
        "train",
        tokenizer,
        args.max_length,
        args.max_prompt_truncation_rate,
        args.require_offset_mask,
    )
    val_tokenization = tokenize_rows(
        val_rows,
        "validation",
        tokenizer,
        args.max_length,
        args.max_prompt_truncation_rate,
        args.require_offset_mask,
    )

    manifest_path = args.output_dir / "v286_generic_tokenization_gate_manifest.json"
    manifest = {
        "schema_version": "kg1_v286_generic_tokenization_gate_v1",
        "generated_at_utc": utc_now(),
        "dataset_manifest_json": str(args.dataset_manifest_json),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest_json),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "config": {
            "max_length": args.max_length,
            "max_prompt_truncation_rate": args.max_prompt_truncation_rate,
            "require_offset_mask": args.require_offset_mask,
            "assistant_final_answer_mode": args.assistant_final_answer_mode,
            "min_train_rows": args.min_train_rows,
            "min_val_rows": args.min_val_rows,
        },
        "validation": {
            "train": train_validation,
            "validation": val_validation,
            "train_val_prompt_answer_overlap": prompt_answer_overlap,
            "train_val_prompt_overlap": prompt_overlap,
            "reference_overlap": reference_overlap,
        },
        "tokenizer_info": tokenizer_info,
        "tokenization": {
            "train": train_tokenization,
            "validation": val_tokenization,
        },
        "outputs": {
            "manifest_json": str(manifest_path),
        },
        "blocked_actions": [
            "full_eval",
            "package",
            "kaggle_submit",
        ],
        "decision": {
            "status": "tokenization_gate_passed",
            "next_action": "Only consider a tiny HF smoke train if roadmap risk/budget gates approve it.",
            "reason": (
                f"train_rows={len(train_rows)}; val_rows={len(val_rows)}; "
                f"train_token_max={train_tokenization['token_max']}; "
                f"val_token_max={val_tokenization['token_max']}; completion_truncation=0"
            ),
        },
    }
    write_json(manifest_path, manifest)
    print("v286_manifest_json =", manifest_path, flush=True)
    print("v286_tokenization_train =", json.dumps(train_tokenization, sort_keys=True), flush=True)
    print("v286_tokenization_validation =", json.dumps(val_tokenization, sort_keys=True), flush=True)
    print("v286_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("=== V286 GENERIC TOKENIZATION GATE END ===", flush=True)
    return manifest


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        train = tmp / "train.jsonl"
        val = tmp / "val.jsonl"
        manifest_path = tmp / "dataset_manifest.json"
        rows = []
        for idx, split in enumerate(["train", "train", "validation"], 1):
            prompt = f"Question {idx}"
            answer = str(idx)
            row = {
                "id": f"r{idx}",
                "prompt": prompt,
                "answer": answer,
                "family": "bit_manipulation" if idx == 1 else "equation_transform",
                "subcategory": "toy",
                "source": "toy",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Final answer: " + answer},
                ],
                "metadata": {
                    "source_dataset": "toy",
                    "weak_gate_rows_used_for_training": False,
                    "full_gate_rows_used_for_training": False,
                    "gate_rows_used_for_training": False,
                },
            }
            rows.append((split, row))
        with train.open("w", encoding="utf-8", newline="\n") as handle:
            for split, row in rows:
                if split == "train":
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        with val.open("w", encoding="utf-8", newline="\n") as handle:
            for split, row in rows:
                if split == "validation":
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        write_json(
            manifest_path,
            {
                "outputs": {
                    "train_jsonl": str(train),
                    "train_sha256": sha256_file(train),
                    "val_jsonl": str(val),
                    "val_sha256": sha256_file(val),
                }
            },
        )
        args = argparse.Namespace(
            dataset_manifest_json=manifest_path,
            output_dir=tmp / "out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=2,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="exact",
            reference_csvs=[],
        )
        manifest = run(args)
        assert manifest["decision"]["status"] == "tokenization_gate_passed"

        reference_csv = tmp / "reference.csv"
        reference_csv.write_text("id,prompt,answer\nr1,Question 1,1\n", encoding="utf-8")
        ref_args = argparse.Namespace(
            **{
                **vars(args),
                "output_dir": tmp / "ref_overlap_out",
                "reference_csvs": [reference_csv],
            }
        )
        try:
            run(ref_args)
        except RuntimeError as exc:
            assert "reference overlap detected" in str(exc)
        else:
            raise AssertionError("reference overlap must fail")

        blocked_manifest_path = tmp / "blocked_dataset_manifest.json"
        write_json(
            blocked_manifest_path,
            {
                "label": "v468_v464_symbol_fix_dataset",
                "outputs": {
                    "train_jsonl": str(train),
                    "train_sha256": sha256_file(train),
                    "val_jsonl": str(val),
                    "val_sha256": sha256_file(val),
                },
            },
        )
        blocked_args = argparse.Namespace(**{**vars(args), "dataset_manifest_json": blocked_manifest_path})
        try:
            run(blocked_args)
        except RuntimeError as exc:
            assert "blocked quarantined dataset marker" in str(exc)
        else:
            raise AssertionError("blocked dataset marker must fail")

        trace_row = {
            "id": "trace",
            "prompt": "Trace question",
            "answer": "42",
            "family": "equation_transform",
            "subcategory": "toy_trace",
            "source": "toy",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Trace question"},
                {"role": "assistant", "content": "Reason with a rule.\nFinal answer: 42"},
            ],
            "metadata": {
                "source_dataset": "toy",
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
            },
        }
        trace_train = tmp / "trace_train.jsonl"
        trace_val = tmp / "trace_val.jsonl"
        trace_train.write_text(json.dumps(trace_row, sort_keys=True) + "\n", encoding="utf-8")
        trace_val_row = {
            **trace_row,
            "id": "trace_val",
            "prompt": "Trace validation question",
            "answer": "43",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Trace validation question"},
                {"role": "assistant", "content": "Reason with a validation rule.\nFinal answer: 43"},
            ],
        }
        trace_val.write_text(json.dumps(trace_val_row, sort_keys=True) + "\n", encoding="utf-8")
        trace_manifest = tmp / "trace_dataset_manifest.json"
        write_json(
            trace_manifest,
            {
                "outputs": {
                    "train_jsonl": str(trace_train),
                    "train_sha256": sha256_file(trace_train),
                    "val_jsonl": str(trace_val),
                    "val_sha256": sha256_file(trace_val),
                }
            },
        )
        trace_args = argparse.Namespace(
            dataset_manifest_json=trace_manifest,
            output_dir=tmp / "trace_out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=1,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="suffix",
            reference_csvs=[],
        )
        trace_manifest_out = run(trace_args)
        assert trace_manifest_out["decision"]["status"] == "tokenization_gate_passed"
        boxed_row = {
            "id": "boxed_trace",
            "prompt": "Boxed trace question",
            "answer": "00000101",
            "family": "bit_manipulation",
            "subcategory": "toy_boxed_trace",
            "source": "toy",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Boxed trace question"},
                {"role": "assistant", "content": r"Reason with a boxed rule." + "\n" + r"Final answer: \boxed{00000101}"},
            ],
            "metadata": {
                "source_dataset": "toy",
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
            },
        }
        boxed_train = tmp / "boxed_train.jsonl"
        boxed_val = tmp / "boxed_val.jsonl"
        boxed_train.write_text(json.dumps(boxed_row, sort_keys=True) + "\n", encoding="utf-8")
        boxed_val_row = {
            **boxed_row,
            "id": "boxed_trace_val",
            "prompt": "Boxed trace validation question",
            "answer": "00000110",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Boxed trace validation question"},
                {"role": "assistant", "content": r"Reason with a boxed validation rule." + "\n" + r"Final answer: \boxed{00000110}"},
            ],
        }
        boxed_val.write_text(json.dumps(boxed_val_row, sort_keys=True) + "\n", encoding="utf-8")
        boxed_manifest = tmp / "boxed_dataset_manifest.json"
        write_json(
            boxed_manifest,
            {
                "outputs": {
                    "train_jsonl": str(boxed_train),
                    "train_sha256": sha256_file(boxed_train),
                    "val_jsonl": str(boxed_val),
                    "val_sha256": sha256_file(boxed_val),
                }
            },
        )
        boxed_args = argparse.Namespace(
            dataset_manifest_json=boxed_manifest,
            output_dir=tmp / "boxed_out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=1,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="boxed_suffix",
            reference_csvs=[],
        )
        boxed_manifest_out = run(boxed_args)
        assert boxed_manifest_out["decision"]["status"] == "tokenization_gate_passed"

        symbolic_boxed_row = {
            **boxed_row,
            "id": "symbolic_boxed_trace",
            "prompt": "Symbolic boxed trace question",
            "answer": "a{b}\\c",
            "family": "equation_transform",
            "subcategory": "toy_symbolic_boxed_trace",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Symbolic boxed trace question"},
                {"role": "assistant", "content": r"Reason with symbols." + "\n" + r"Final answer: \boxed{a\{b\}\\c}"},
            ],
        }
        symbolic_boxed_train = tmp / "symbolic_boxed_train.jsonl"
        symbolic_boxed_val = tmp / "symbolic_boxed_val.jsonl"
        symbolic_boxed_train.write_text(json.dumps(symbolic_boxed_row, sort_keys=True) + "\n", encoding="utf-8")
        symbolic_boxed_val.write_text(
            json.dumps(
                {
                    **symbolic_boxed_row,
                    "id": "symbolic_boxed_trace_val",
                    "prompt": "Symbolic boxed validation question",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "Symbolic boxed validation question"},
                        {"role": "assistant", "content": r"Reason with validation symbols." + "\n" + r"Final answer: \boxed{a\{b\}\\c}"},
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        symbolic_boxed_manifest = tmp / "symbolic_boxed_dataset_manifest.json"
        write_json(
            symbolic_boxed_manifest,
            {
                "outputs": {
                    "train_jsonl": str(symbolic_boxed_train),
                    "train_sha256": sha256_file(symbolic_boxed_train),
                    "val_jsonl": str(symbolic_boxed_val),
                    "val_sha256": sha256_file(symbolic_boxed_val),
                }
            },
        )
        symbolic_boxed_args = argparse.Namespace(
            dataset_manifest_json=symbolic_boxed_manifest,
            output_dir=tmp / "symbolic_boxed_out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=1,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="boxed_suffix",
            reference_csvs=[],
        )
        symbolic_boxed_manifest_out = run(symbolic_boxed_args)
        assert symbolic_boxed_manifest_out["decision"]["status"] == "tokenization_gate_passed"

        unescaped_symbolic_bad_row = {
            **symbolic_boxed_row,
            "id": "unescaped_symbolic_bad",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Symbolic boxed trace question"},
                {"role": "assistant", "content": r"Reason with unescaped symbols." + "\n" + r"Final answer: \boxed{a{b}\c}"},
            ],
        }
        unescaped_symbolic_bad_train = tmp / "unescaped_symbolic_bad_train.jsonl"
        unescaped_symbolic_bad_val = tmp / "unescaped_symbolic_bad_val.jsonl"
        unescaped_symbolic_bad_train.write_text(
            json.dumps(unescaped_symbolic_bad_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unescaped_symbolic_bad_val.write_text(
            json.dumps(
                {
                    **unescaped_symbolic_bad_row,
                    "id": "unescaped_symbolic_bad_val",
                    "prompt": "Unescaped symbolic boxed validation question",
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "Unescaped symbolic boxed validation question"},
                        {"role": "assistant", "content": r"Reason with unescaped validation symbols." + "\n" + r"Final answer: \boxed{a{b}\c}"},
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        unescaped_symbolic_bad_manifest = tmp / "unescaped_symbolic_bad_dataset_manifest.json"
        write_json(
            unescaped_symbolic_bad_manifest,
            {
                "outputs": {
                    "train_jsonl": str(unescaped_symbolic_bad_train),
                    "train_sha256": sha256_file(unescaped_symbolic_bad_train),
                    "val_jsonl": str(unescaped_symbolic_bad_val),
                    "val_sha256": sha256_file(unescaped_symbolic_bad_val),
                }
            },
        )
        unescaped_symbolic_bad_args = argparse.Namespace(
            dataset_manifest_json=unescaped_symbolic_bad_manifest,
            output_dir=tmp / "unescaped_symbolic_bad_out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=1,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="boxed_suffix",
            reference_csvs=[],
        )
        try:
            run(unescaped_symbolic_bad_args)
        except RuntimeError as exc:
            assert "bad rows" in str(exc)
        else:
            raise AssertionError("unescaped symbolic boxed answer must fail")

        bad_row = {
            **boxed_row,
            "id": "bad_rejected_candidate",
            "prompt": "Bad rejected candidate question",
            "answer": "30",
            "family": "equation_transform",
            "subcategory": "toy_bad_rejected",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Bad rejected candidate question"},
                {
                    "role": "assistant",
                    "content": "The shortened candidate '30' is rejected.\n" + r"Final answer: \boxed{30}",
                },
            ],
            "metadata": {
                "source_dataset": "toy",
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
                "rejected_candidate": "30",
            },
        }
        bad_train = tmp / "bad_train.jsonl"
        bad_val = tmp / "bad_val.jsonl"
        bad_train.write_text(json.dumps(bad_row, sort_keys=True) + "\n", encoding="utf-8")
        bad_val.write_text(json.dumps({**bad_row, "id": "bad_rejected_candidate_val"}, sort_keys=True) + "\n", encoding="utf-8")
        bad_manifest = tmp / "bad_dataset_manifest.json"
        write_json(
            bad_manifest,
            {
                "outputs": {
                    "train_jsonl": str(bad_train),
                    "train_sha256": sha256_file(bad_train),
                    "val_jsonl": str(bad_val),
                    "val_sha256": sha256_file(bad_val),
                }
            },
        )
        bad_args = argparse.Namespace(
            dataset_manifest_json=bad_manifest,
            output_dir=tmp / "bad_out",
            model_name="toy",
            model_revision="",
            max_length=2048,
            max_prompt_truncation_rate=0.0,
            require_offset_mask=True,
            min_train_rows=1,
            min_val_rows=1,
            use_toy_tokenizer=True,
            assistant_final_answer_mode="boxed_suffix",
            reference_csvs=[],
        )
        try:
            run(bad_args)
        except RuntimeError as exc:
            assert "bad rows" in str(exc)
        else:
            raise AssertionError("contradictory rejected candidate must fail the tokenization gate")
    print("v286_generic_tokenization_gate_self_test=ok", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v286_generic_tokenization_gate") / utc_compact())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-truncation-rate", type=float, default=0.0)
    parser.add_argument("--require-offset-mask", action="store_true", default=True)
    parser.add_argument("--allow-fallback-mask", dest="require_offset_mask", action="store_false")
    parser.add_argument("--min-train-rows", type=int, default=600)
    parser.add_argument("--min-val-rows", type=int, default=60)
    parser.add_argument(
        "--assistant-final-answer-mode",
        choices=("exact", "suffix", "boxed_exact", "boxed_suffix", "boxed_only", "submit_safe_suffix"),
        default="exact",
        help=(
            "Use exact for short-answer rows, suffix for solver traces, boxed_* for rows ending in "
            "\\boxed{answer}, submit_safe_suffix for mixed boxed/unboxed rows that must pass label-free extraction."
        ),
    )
    parser.add_argument("--use-toy-tokenizer", action="store_true")
    parser.add_argument(
        "--reference-csv",
        dest="reference_csvs",
        action="append",
        default=[],
        type=Path,
        help="Forbidden reference CSV. Fails on id, prompt, or prompt+answer overlap.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.dataset_manifest_json is None:
        parser.error("--dataset-manifest-json is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

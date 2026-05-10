#!/usr/bin/env python3
"""Tokenization and leakage gate for the V249 public non-weak target dataset.

This is a CPU-only validation job. It downloads the V249 dataset artifacts,
verifies hashes/counts/no weak overlap, runs the real Nemotron tokenizer with
chat template and offset-based assistant masks, checks truncation, and compares
prompt/answer overlap against the checked-in V217 corpus. It does not train,
load model weights, generate predictions, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_V249_PATH = "data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z"
DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
EXPECTED_TRAIN_SHA256 = "81c8624b7e0a330a720e22b5e4fc254b238a7c618e1c0cdcdea3cf1fd96d9f41"
EXPECTED_VAL_SHA256 = "43dd9f5fbb6864e85e60b1a6cc2ad7060a667e914a67dda2aa3a22771efb4783"
EXPECTED_BLOCKED_SHA256 = "5392c44fda7e0522910735c9a8b560d9c504a136d6141ed25091f4c858c3d4ce"
EXPECTED_TRAIN_ROWS = 2558
EXPECTED_VAL_ROWS = 284
EXPECTED_BLOCKED_ROWS = 315
EXPECTED_TRAIN_FAMILY_COUNTS = {"bit_manipulation": 1298, "equation_transform": 1260}
EXPECTED_VAL_FAMILY_COUNTS = {"bit_manipulation": 144, "equation_transform": 140}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def token_value(cli_token: str) -> str | None:
    if cli_token or os.environ.get("HF_TOKEN"):
        return cli_token or os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def download_file(repo_id: str, filename: str, repo_type: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V250") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
    )


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V250 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V250 tokenizer gate {path_in_repo.strip('/')}",
    )
    return str(info)


def upload_manifest(repo_id: str, manifest_path: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V250 manifest") from exc
    api = HfApi(token=token)
    info = api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(manifest_path),
        path_in_repo=(path_in_repo.strip("/") + "/" + manifest_path.name).strip("/"),
        commit_message=f"Refresh KG1 V250 tokenizer gate manifest {path_in_repo.strip('/')}",
    )
    return str(info)


def assert_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    print(f"{label}_sha256 = {observed}", flush=True)
    if expected and observed != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, got {observed}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_answer(value: Any) -> str:
    return "".join(str(value).strip().split())


def prompt_answer_key(row: dict[str, Any]) -> str:
    raw = normalize_text(row.get("prompt", "")) + "\0" + normalize_answer(row.get("answer", ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prompt_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(normalize_text(row.get("prompt", "")).encode("utf-8")).hexdigest()


def validate_rows(
    rows: list[dict[str, Any]],
    split: str,
    expected_rows: int,
    expected_family_counts: dict[str, int],
    weak_ids: set[str],
) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"{split} row count mismatch: expected {expected_rows}, got {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    original_ids = [str(row.get("original_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{split} duplicate id detected")
    weak_overlap = sorted(set(original_ids) & weak_ids)
    if weak_overlap:
        raise RuntimeError(f"{split} weak ID overlap detected: {weak_overlap[:10]}")
    family_counts = dict(Counter(str(row.get("family", "")) for row in rows))
    if family_counts != expected_family_counts:
        raise RuntimeError(
            f"{split} family count mismatch: expected {expected_family_counts}, got {family_counts}"
        )
    bad_messages: list[str] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            bad_messages.append(str(row.get("id", "")))
            continue
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            bad_messages.append(str(row.get("id", "")))
            continue
        if messages[1].get("content") != row.get("prompt"):
            bad_messages.append(str(row.get("id", "")))
            continue
        if messages[2].get("content") != "Final answer: " + str(row.get("answer", "")):
            bad_messages.append(str(row.get("id", "")))
    if bad_messages:
        raise RuntimeError(f"{split} bad message rows: {bad_messages[:20]}")
    return {"rows": len(rows), "unique_ids": len(set(ids)), "family_counts": family_counts}


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
        raise RuntimeError("assistant text not found in rendered chat template")
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
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    loss_mask = [0] * min(len(prompt_ids), len(input_ids)) + [1] * max(0, len(input_ids) - len(prompt_ids))
    return input_ids, loss_mask[: len(input_ids)], False


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return int(values[idx])


def tokenize_rows(
    rows: list[dict[str, Any]],
    split: str,
    tokenizer: Any,
    max_length: int,
    max_prompt_truncation_rate: float,
    require_offset_mask: bool,
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    lengths: list[int] = []
    loss_lengths: list[int] = []
    prompt_truncated = 0
    completion_tokens_dropped = 0
    offset_masks = 0
    fallback_masks = 0
    no_loss_rows: list[str] = []
    for row in rows:
        input_ids, loss_mask, used_offsets = build_completion_mask(
            tokenizer, row["messages"], require_offset_mask
        )
        if used_offsets:
            offset_masks += 1
        else:
            fallback_masks += 1
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            dropped_loss = sum(loss_mask[:overflow])
            completion_tokens_dropped += dropped_loss
            if dropped_loss:
                raise RuntimeError(f"{split} completion tokens would be truncated for {row.get('id')}")
            if overflow:
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
    prompt_truncation_rate = prompt_truncated / max(1, len(rows))
    if prompt_truncation_rate > max_prompt_truncation_rate:
        raise RuntimeError(
            f"{split} prompt truncation rate {prompt_truncation_rate:.6f} exceeds "
            f"{max_prompt_truncation_rate:.6f}"
        )
    family_summary: dict[str, Any] = {}
    for family, items in sorted(by_family.items()):
        token_values = [int(item["tokens"]) for item in items]
        loss_values = [int(item["loss_tokens"]) for item in items]
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
        "token_p50": percentile(lengths, 0.50),
        "token_p90": percentile(lengths, 0.90),
        "token_p99": percentile(lengths, 0.99),
        "token_max": max(lengths) if lengths else 0,
        "loss_token_min": min(loss_lengths) if loss_lengths else 0,
        "loss_token_p50": percentile(loss_lengths, 0.50),
        "loss_token_max": max(loss_lengths) if loss_lengths else 0,
        "offset_masks": offset_masks,
        "fallback_masks": fallback_masks,
        "prompt_truncated": prompt_truncated,
        "prompt_truncation_rate": prompt_truncation_rate,
        "completion_tokens_dropped": completion_tokens_dropped,
        "family_summary": family_summary,
    }


def load_repo_v217_rows(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_path = repo_root / "data" / "v217" / "v217_short_answer_train.jsonl"
    val_path = repo_root / "data" / "v217" / "v217_short_answer_val.jsonl"
    train_rows = read_jsonl(train_path) if train_path.exists() else []
    val_rows = read_jsonl(val_path) if val_path.exists() else []
    return train_rows, val_rows


def overlap_report(v249_rows: list[dict[str, Any]], v217_train: list[dict[str, Any]], v217_val: list[dict[str, Any]]) -> dict[str, Any]:
    def keyset(rows: list[dict[str, Any]], fn: Any) -> set[str]:
        return {fn(row) for row in rows}

    v249_pa = keyset(v249_rows, prompt_answer_key)
    v249_p = keyset(v249_rows, prompt_key)
    train_pa = keyset(v217_train, prompt_answer_key)
    val_pa = keyset(v217_val, prompt_answer_key)
    train_p = keyset(v217_train, prompt_key)
    val_p = keyset(v217_val, prompt_key)
    prompt_answer_overlap_train = len(v249_pa & train_pa)
    prompt_answer_overlap_val = len(v249_pa & val_pa)
    prompt_only_overlap_train = len(v249_p & train_p)
    prompt_only_overlap_val = len(v249_p & val_p)
    return {
        "v249_rows": len(v249_rows),
        "v217_train_rows": len(v217_train),
        "v217_val_rows": len(v217_val),
        "prompt_answer_overlap_v217_train": prompt_answer_overlap_train,
        "prompt_answer_overlap_v217_val": prompt_answer_overlap_val,
        "prompt_answer_overlap_v217_total": prompt_answer_overlap_train + prompt_answer_overlap_val,
        "prompt_answer_novel_vs_v217": len(v249_rows) - (prompt_answer_overlap_train + prompt_answer_overlap_val),
        "prompt_only_overlap_v217_train": prompt_only_overlap_train,
        "prompt_only_overlap_v217_val": prompt_only_overlap_val,
        "prompt_only_overlap_v217_total": prompt_only_overlap_train + prompt_only_overlap_val,
        "prompt_only_novel_vs_v217": len(v249_rows) - (prompt_only_overlap_train + prompt_only_overlap_val),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V250 V249 TOKENIZATION GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("data_repo =", args.data_repo, flush=True)
    print("v249_path =", args.v249_path, flush=True)
    print("model_name =", args.model_name, flush=True)
    print("model_revision =", args.model_revision, flush=True)
    print("max_length =", args.max_length, flush=True)
    print("max_prompt_truncation_rate =", args.max_prompt_truncation_rate, flush=True)
    print("require_offset_mask =", args.require_offset_mask, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN is required for V250")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "download"
        train_path = download_file(
            args.data_repo,
            args.v249_path.rstrip("/") + "/v249_public_nonweak_target_train.jsonl",
            "dataset",
            download_root,
            token,
        )
        val_path = download_file(
            args.data_repo,
            args.v249_path.rstrip("/") + "/v249_public_nonweak_target_val.jsonl",
            "dataset",
            download_root,
            token,
        )
        blocked_path = download_file(
            args.data_repo,
            args.v249_path.rstrip("/") + "/v249_blocked_weak_ids.csv",
            "dataset",
            download_root,
            token,
        )
        manifest_path = download_file(
            args.data_repo,
            args.v249_path.rstrip("/") + "/v249_public_nonweak_target_manifest.json",
            "dataset",
            download_root,
            token,
        )
        print("train_path =", train_path, flush=True)
        print("val_path =", val_path, flush=True)
        print("blocked_path =", blocked_path, flush=True)
        print("v249_manifest_path =", manifest_path, flush=True)
        assert_sha(train_path, args.expected_train_sha256, "train")
        assert_sha(val_path, args.expected_val_sha256, "validation")
        assert_sha(blocked_path, args.expected_blocked_sha256, "blocked_weak_ids")
        train_rows = read_jsonl(train_path)
        val_rows = read_jsonl(val_path)
        blocked_rows = read_csv_rows(blocked_path)
        v249_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_artifact_hashes = {
            "train_jsonl": file_meta(train_path),
            "val_jsonl": file_meta(val_path),
            "blocked_weak_ids_csv": file_meta(blocked_path),
            "v249_manifest_json": file_meta(manifest_path),
        }

    if len(blocked_rows) != args.expected_blocked_rows:
        raise RuntimeError(
            f"blocked weak row count mismatch: expected {args.expected_blocked_rows}, got {len(blocked_rows)}"
        )
    weak_ids = {str(row.get("id", "")) for row in blocked_rows}
    train_validation = validate_rows(
        train_rows,
        "train",
        args.expected_train_rows,
        args.expected_train_family_counts,
        weak_ids,
    )
    val_validation = validate_rows(
        val_rows,
        "validation",
        args.expected_val_rows,
        args.expected_val_family_counts,
        weak_ids,
    )
    if {row["original_id"] for row in train_rows} & {row["original_id"] for row in val_rows}:
        raise RuntimeError("train/validation original_id overlap detected")
    if {prompt_answer_key(row) for row in train_rows} & {prompt_answer_key(row) for row in val_rows}:
        raise RuntimeError("train/validation prompt+answer overlap detected")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for V250 tokenizer gate") from exc

    print("loading_tokenizer =", args.model_name, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision or None,
        trust_remote_code=True,
        token=token,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_info = {
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
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
    v217_train, v217_val = load_repo_v217_rows(args.repo_root)
    overlap = overlap_report(train_rows + val_rows, v217_train, v217_val)

    decision = {
        "decision": "v249_tokenization_gate_passed_training_still_requires_small_gpu_smoke",
        "reason": (
            f"train_tokenized={train_tokenization['rows']}; "
            f"val_tokenized={val_tokenization['rows']}; "
            f"train_prompt_trunc={train_tokenization['prompt_truncated']}; "
            f"val_prompt_trunc={val_tokenization['prompt_truncated']}; "
            f"fallback_masks={train_tokenization['fallback_masks'] + val_tokenization['fallback_masks']}; "
            f"novel_prompt_answer_vs_v217={overlap['prompt_answer_novel_vs_v217']}"
        ),
        "next_action": "Run a tiny HF GPU smoke train with strict weak-eval gate before any longer H200 run.",
    }
    manifest = {
        "schema_version": "kg1_v250_v249_tokenization_gate_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "data_repo": args.data_repo,
            "v249_path": args.v249_path,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "max_length": args.max_length,
            "max_prompt_truncation_rate": args.max_prompt_truncation_rate,
            "require_offset_mask": args.require_offset_mask,
        },
        "input_artifact_hashes": input_artifact_hashes,
        "v249_manifest_counts": v249_manifest.get("counts", {}),
        "train_validation": train_validation,
        "val_validation": val_validation,
        "tokenizer_info": tokenizer_info,
        "train_tokenization": train_tokenization,
        "val_tokenization": val_tokenization,
        "overlap_vs_v217": overlap,
        "decision": decision,
        "blocked_actions": ["long_train", "full_scoring", "package", "kaggle_submit"],
    }
    manifest_path_out = args.output_dir / "v250_v249_tokenization_gate_manifest.json"
    upload_info = "upload_disabled"
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.data_repo,
        "path_in_repo": str(args.output_path_in_repo or ""),
        "upload_info": upload_info,
        "manifest_uploaded_after_folder_upload": False,
    }
    write_json(manifest_path_out, manifest)
    manifest_refresh_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.data_repo, args.output_dir, args.output_path_in_repo, token)
        manifest["hf_upload"]["upload_info"] = upload_info
        manifest["hf_upload"]["manifest_uploaded_after_folder_upload"] = True
        write_json(manifest_path_out, manifest)
        manifest_refresh_info = upload_manifest(args.data_repo, manifest_path_out, args.output_path_in_repo, token)
    print("train_tokenization =", json.dumps(train_tokenization, sort_keys=True), flush=True)
    print("val_tokenization =", json.dumps(val_tokenization, sort_keys=True), flush=True)
    print("overlap_vs_v217 =", json.dumps(overlap, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("manifest_refresh_info =", manifest_refresh_info, flush=True)
    print("manifest_path =", manifest_path_out, flush=True)
    print("=== V250 V249 TOKENIZATION GATE END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo", default=DEFAULT_DATA_REPO)
    parser.add_argument("--v249-path", default=DEFAULT_V249_PATH)
    parser.add_argument("--expected-train-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument("--expected-val-sha256", default=EXPECTED_VAL_SHA256)
    parser.add_argument("--expected-blocked-sha256", default=EXPECTED_BLOCKED_SHA256)
    parser.add_argument("--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--expected-val-rows", type=int, default=EXPECTED_VAL_ROWS)
    parser.add_argument("--expected-blocked-rows", type=int, default=EXPECTED_BLOCKED_ROWS)
    parser.add_argument("--expected-train-family-counts", type=json.loads, default=EXPECTED_TRAIN_FAMILY_COUNTS)
    parser.add_argument("--expected-val-family-counts", type=json.loads, default=EXPECTED_VAL_FAMILY_COUNTS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-truncation-rate", type=float, default=0.0)
    parser.add_argument("--require-offset-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v250_v249_tokenization_gate")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    class ToyTokenizer:
        is_fast = True
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool, **_: Any) -> str:
            text = "\n".join(str(m["content"]) for m in messages)
            if add_generation_prompt:
                text += "\n"
            return text

        def __call__(self, text: str, add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, Any]:
            del add_special_tokens
            ids = list(range(len(text)))
            offsets = [(idx, idx + 1) for idx in range(len(text))]
            return {"input_ids": ids, "offset_mapping": offsets if return_offsets_mapping else None}

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return list(range(len(text)))

    row = {
        "id": "v249_train_abc",
        "original_id": "abc",
        "family": "equation_transform",
        "prompt": "Solve this equation transform.",
        "answer": "42",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Solve this equation transform."},
            {"role": "assistant", "content": "Final answer: 42"},
        ],
    }
    validate_rows([row], "self", 1, {"equation_transform": 1}, set())
    report = tokenize_rows([row], "self", ToyTokenizer(), 4096, 0.0, True)
    if report["fallback_masks"] != 0 or report["offset_masks"] != 1:
        raise AssertionError("offset mask self-test failed")
    print("v250_v249_tokenization_gate_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload the V371 trace-style bit-transfer dataset to the KG1 HF dataset repo."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = REPO_ROOT / "artifacts/v371_v367_trace_style_transfer_dataset/20260514T_cpu_gate"
MANIFEST = DATASET_DIR / "v371_v367_trace_style_transfer_manifest.json"
TOKENIZATION_GATE = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
PATH_IN_REPO = "data/v371_v367_trace_style_transfer/20260514T_cpu_gate"

EXPECTED_TRAIN_SHA256 = "96278ed22d81ba2412ce9a1ec5f8ac87df64a5f8014ebb5c3f88b18103bfbe0a"
EXPECTED_VAL_SHA256 = "fe66489655ad6907705fa5d18bc8e777055b6d77d9f1ceaba7fb58005169c7d0"
EXPECTED_TRAIN_ROWS = 1128
EXPECTED_VAL_ROWS = 282
EXPECTED_TOKEN_MAX_FOR_HF = 1024


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_local_contract() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not TOKENIZATION_GATE.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE)

    manifest = read_json(MANIFEST)
    gate = read_json(TOKENIZATION_GATE)
    if manifest.get("schema_version") != "kg1_v371_v367_trace_style_transfer_dataset_v1":
        raise RuntimeError("Unexpected V371 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V371 real tokenization gate did not pass.")
    if gate.get("dataset_manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("V371 tokenization gate is stale relative to dataset manifest.")
    if gate.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V371 tokenization gate must use boxed_suffix mode.")
    if gate.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V371 HF upload requires the real tokenizer gate, not toy.")

    outputs = manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V371 train SHA drifted.")
    if outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V371 val SHA drifted.")

    required_paths = [
        resolve_repo_path(outputs["train_jsonl"]),
        resolve_repo_path(outputs["val_jsonl"]),
        MANIFEST,
        TOKENIZATION_GATE,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = manifest.get("validation", {})
    train_validation = validation.get("train", {})
    val_validation = validation.get("validation", {})
    if int(train_validation.get("rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Unexpected V371 train row count.")
    if int(val_validation.get("rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected V371 validation row count.")
    for split_name, split_validation, expected_rows in (
        ("train", train_validation, EXPECTED_TRAIN_ROWS),
        ("validation", val_validation, EXPECTED_VAL_ROWS),
    ):
        if split_validation.get("family_counts") != {"bit_manipulation": expected_rows}:
            raise RuntimeError(f"Unexpected V371 {split_name} family counts.")
        if int(split_validation.get("assistant_trace_style_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} trace-style count drifted.")
        if int(split_validation.get("assistant_contains_output_bit_columns_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} Output bit columns count drifted.")
        if int(split_validation.get("assistant_boxed_suffix_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} boxed-suffix count drifted.")
    if int(validation.get("train_val_prompt_overlap", -1)) != 0:
        raise RuntimeError("V371 train/validation prompt overlap is not zero.")

    gate_validation = gate.get("validation", {})
    for split_name, split_key, expected_rows in (
        ("train", "train", EXPECTED_TRAIN_ROWS),
        ("validation", "validation", EXPECTED_VAL_ROWS),
    ):
        token_summary = gate.get("tokenization", {}).get(split_key, {})
        split_validation = gate_validation.get(split_key, {})
        if int(split_validation.get("rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} gate row count drifted.")
        if split_validation.get("source_counts") != {"v371_trace_style_from_v367": expected_rows}:
            raise RuntimeError(f"Unexpected V371 {split_name} source counts.")
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V371 {split_name} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V371 {split_name} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V371 {split_name} used fallback masks.")
        if int(token_summary.get("token_max", 999999)) > EXPECTED_TOKEN_MAX_FOR_HF:
            raise RuntimeError(f"V371 {split_name} token max exceeds HF max length gate.")

    return {
        "dataset_manifest_sha256": sha256_file(MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
        "train_rows": train_validation.get("rows"),
        "val_rows": val_validation.get("rows"),
        "train_subcategory_counts": train_validation.get("subcategory_counts"),
        "val_subcategory_counts": val_validation.get("subcategory_counts"),
        "token_max_train": gate.get("tokenization", {}).get("train", {}).get("token_max"),
        "token_max_val": gate.get("tokenization", {}).get("validation", {}).get("token_max"),
        "assistant_final_answer_mode": "boxed_suffix",
    }


def main() -> int:
    print("=== V371 HF DATASET UPLOAD START ===", flush=True)
    local_gate_summary = validate_local_contract()
    print("local_gate_summary =", json.dumps(local_gate_summary, sort_keys=True), flush=True)
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V371 dataset upload.")
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add KG1 V371 trace-style bit transfer dataset",
    )
    out = {
        "version": "v371_v367_trace_style_transfer_hf_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": DATASET_REPO,
        "dataset_dir": str(DATASET_DIR),
        "path_in_repo": PATH_IN_REPO,
        "dataset_upload": str(info),
        "upload": str(info),
        "manifest_json": str(MANIFEST),
        "tokenization_gate_manifest": str(TOKENIZATION_GATE),
        "local_gate_summary": local_gate_summary,
    }
    out_path = RUN_DIR / "v371_v367_trace_style_transfer_hf_upload_manifest.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest =", out_path, flush=True)
    print("upload_url =", info, flush=True)
    print("=== V371 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

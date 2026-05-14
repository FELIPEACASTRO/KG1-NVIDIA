#!/usr/bin/env python3
"""Upload the V367 V366 bit ternary transfer dataset to the KG1 HF dataset repo."""

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
DATASET_DIR = REPO_ROOT / "artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate"
MANIFEST = DATASET_DIR / "v367_v366_bit_ternary_transfer_manifest.json"
TOKENIZATION_GATE = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
PATH_IN_REPO = "data/v367_v366_bit_ternary_transfer/20260514T_cpu_gate"

EXPECTED_TRAIN_SHA256 = "5ea3cef4d9f589c9c77aabf22ac90b5261cc77cdbdcf5c120f306c6c0edf95fc"
EXPECTED_VAL_SHA256 = "04623efbcfd6c1db9d3988f9efca48ee6f387ae67bede8f55969517ebf06fb00"
EXPECTED_PREFERENCES_TRAIN_SHA256 = "f1dcedaa23d9e35b672e08243123589bc28ac353e26d3cf59c40fa23729c99b7"
EXPECTED_PREFERENCES_VAL_SHA256 = "73cd19267568267c2dfb8d0d3188db043a75a1ed2f0f7601d4da590694884b29"
EXPECTED_TRAIN_ROWS = 1128
EXPECTED_VAL_ROWS = 282


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
    if manifest.get("schema_version") != "kg1_v367_v366_bit_ternary_transfer_dataset_v1":
        raise RuntimeError("Unexpected V367 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V367 real tokenization gate did not pass.")
    if gate.get("dataset_manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("V367 tokenization gate is stale relative to dataset manifest.")
    if gate.get("config", {}).get("assistant_final_answer_mode") != "boxed_only":
        raise RuntimeError("V367 tokenization gate must use boxed_only mode.")
    if gate.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V367 HF upload requires the real tokenizer gate, not toy.")

    outputs = manifest.get("outputs", {})
    expected_shas = {
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "val_sha256": EXPECTED_VAL_SHA256,
        "preferences_train_sha256": EXPECTED_PREFERENCES_TRAIN_SHA256,
        "preferences_val_sha256": EXPECTED_PREFERENCES_VAL_SHA256,
    }
    for key, expected in expected_shas.items():
        observed = outputs.get(key)
        if observed != expected:
            raise RuntimeError(f"V367 {key} drifted: {observed} != {expected}")

    required_paths = [
        resolve_repo_path(outputs["train_jsonl"]),
        resolve_repo_path(outputs["val_jsonl"]),
        resolve_repo_path(outputs["preferences_train_jsonl"]),
        resolve_repo_path(outputs["preferences_val_jsonl"]),
        MANIFEST,
        TOKENIZATION_GATE,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_validation = manifest.get("validation", {}).get("train", {})
    val_validation = manifest.get("validation", {}).get("validation", {})
    preference = manifest.get("validation", {}).get("preference", {})
    if int(train_validation.get("rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Unexpected V367 train row count.")
    if int(val_validation.get("rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected V367 validation row count.")
    if int(train_validation.get("assistant_boxed_only_rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("V367 train boxed-only count drifted.")
    if int(val_validation.get("assistant_boxed_only_rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("V367 validation boxed-only count drifted.")
    if train_validation.get("family_counts") != {"bit_manipulation": EXPECTED_TRAIN_ROWS}:
        raise RuntimeError("Unexpected V367 train family counts.")
    if val_validation.get("family_counts") != {"bit_manipulation": EXPECTED_VAL_ROWS}:
        raise RuntimeError("Unexpected V367 validation family counts.")
    if int(manifest.get("validation", {}).get("train_val_prompt_overlap", -1)) != 0:
        raise RuntimeError("V367 train/validation prompt overlap is not zero.")
    if int(preference.get("train_rows", -1)) != EXPECTED_TRAIN_ROWS * 2:
        raise RuntimeError("Unexpected V367 preference train row count.")
    if int(preference.get("val_rows", -1)) != EXPECTED_VAL_ROWS * 2:
        raise RuntimeError("Unexpected V367 preference validation row count.")

    for split in ("train", "validation"):
        token_summary = gate.get("tokenization", {}).get(split, {})
        expected_rows = EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_VAL_ROWS
        if int(token_summary.get("rows", -1)) != expected_rows:
            raise RuntimeError(f"V367 {split} tokenization row count drifted.")
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V367 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V367 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V367 {split} used fallback masks.")
        if int(token_summary.get("token_max", 999999)) > 512:
            raise RuntimeError(f"V367 {split} token max unexpectedly high.")

    return {
        "dataset_manifest_sha256": sha256_file(MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
        "preferences_train_sha256": outputs.get("preferences_train_sha256"),
        "preferences_val_sha256": outputs.get("preferences_val_sha256"),
        "train_rows": train_validation.get("rows"),
        "val_rows": val_validation.get("rows"),
        "train_subcategory_counts": train_validation.get("subcategory_counts"),
        "val_subcategory_counts": val_validation.get("subcategory_counts"),
        "token_max_train": gate.get("tokenization", {}).get("train", {}).get("token_max"),
        "token_max_val": gate.get("tokenization", {}).get("validation", {}).get("token_max"),
    }


def main() -> int:
    print("=== V367 HF DATASET UPLOAD START ===", flush=True)
    local_gate_summary = validate_local_contract()
    print("local_gate_summary =", json.dumps(local_gate_summary, sort_keys=True), flush=True)
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V367 dataset upload.")
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add KG1 V367 V366 bit ternary transfer dataset",
    )
    out = {
        "version": "v367_v366_bit_ternary_transfer_hf_upload",
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
    out_path = RUN_DIR / "v367_v366_bit_ternary_transfer_hf_upload_manifest.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest =", out_path, flush=True)
    print("upload_url =", info, flush=True)
    print("=== V367 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload the V361 boxed-only transfer dataset to the KG1 HF dataset repo."""

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
DATASET_DIR = REPO_ROOT / "artifacts/v361_v357_boxed_only_transfer_dataset/20260514T_cpu_gate"
MANIFEST = DATASET_DIR / "v361_v357_boxed_only_transfer_manifest.json"
TOKENIZATION_GATE = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
PATH_IN_REPO = "data/v361_v357_boxed_only_transfer/20260514T_cpu_gate"

EXPECTED_TRAIN_SHA256 = "be742d7a82bf1c98f33d67bed8903006068c139ab74f798055fcc7d435ffa4db"
EXPECTED_VAL_SHA256 = "4c93766e7fae72da14f879177e15c3c6300b7991e4efc8fba4d7fe75d3df5332"
EXPECTED_PREFERENCES_TRAIN_SHA256 = "f7f2e11540adbf16bcc93cc8f88a8f3f9c432086df59add4b4132daa821b4b06"
EXPECTED_PREFERENCES_VAL_SHA256 = "565a079d523f74cafe76e6e6161fd53f86d64ac90a684215d133b4b435763df3"


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
    if manifest.get("schema_version") != "kg1_v361_v357_boxed_only_transfer_dataset_v1":
        raise RuntimeError("Unexpected V361 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V361 real tokenization gate did not pass.")
    if gate.get("dataset_manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("V361 tokenization gate is stale relative to dataset manifest.")
    if gate.get("config", {}).get("assistant_final_answer_mode") != "boxed_only":
        raise RuntimeError("V361 tokenization gate must use boxed_only mode.")
    if gate.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V361 HF upload requires the real tokenizer gate, not toy.")

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
            raise RuntimeError(f"V361 {key} drifted: {observed} != {expected}")

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
    if int(train_validation.get("rows", -1)) != 1152:
        raise RuntimeError("Unexpected V361 train row count.")
    if int(val_validation.get("rows", -1)) != 288:
        raise RuntimeError("Unexpected V361 validation row count.")
    if int(train_validation.get("assistant_boxed_only_rows", -1)) != 1152:
        raise RuntimeError("V361 train boxed-only count drifted.")
    if int(val_validation.get("assistant_boxed_only_rows", -1)) != 288:
        raise RuntimeError("V361 validation boxed-only count drifted.")
    if int(manifest.get("validation", {}).get("train_validation_prompt_overlap", -1)) != 0:
        raise RuntimeError("V361 train/validation prompt overlap is not zero.")
    if int(preference.get("train_rows", -1)) != 2304:
        raise RuntimeError("Unexpected V361 preference train row count.")
    if int(preference.get("val_rows", -1)) != 576:
        raise RuntimeError("Unexpected V361 preference validation row count.")

    for split in ("train", "validation"):
        token_summary = gate.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V361 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V361 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V361 {split} used fallback masks.")
        if int(token_summary.get("token_max", 999999)) > 512:
            raise RuntimeError(f"V361 {split} token max unexpectedly high.")

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
    print("=== V361 HF DATASET UPLOAD START ===", flush=True)
    local_gate_summary = validate_local_contract()
    print("local_gate_summary =", json.dumps(local_gate_summary, sort_keys=True), flush=True)
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V361 dataset upload.")
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add KG1 V361 boxed-only transfer dataset",
    )
    out = {
        "version": "v361_v357_boxed_only_transfer_hf_upload",
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
    out_path = RUN_DIR / "v361_v357_boxed_only_transfer_hf_upload_manifest.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest =", out_path, flush=True)
    print("upload_url =", info, flush=True)
    print("=== V361 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

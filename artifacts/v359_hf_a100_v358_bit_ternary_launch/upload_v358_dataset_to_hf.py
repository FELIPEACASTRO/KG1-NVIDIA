#!/usr/bin/env python3
"""Upload the V358 V357 bit-ternary transfer dataset to the KG1 HF dataset repo."""

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
DATASET_DIR = REPO_ROOT / "artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate"
MANIFEST = DATASET_DIR / "v358_v357_bit_ternary_transfer_manifest.json"
TOKENIZATION_GATE = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
PATH_IN_REPO = "data/v358_v357_bit_ternary_transfer/20260514T_cpu_gate"
EXPECTED_TRAIN_SHA256 = "6881308c7e46167ea8752513dd6e986d14b39f04f661dbac8d9ed18d189f1a05"
EXPECTED_VAL_SHA256 = "d92f4bdf2e622be958ae09353bf3965d2a23e1e6fcea95fbd77c8bcbdf0b6b47"
EXPECTED_PREFERENCES_TRAIN_SHA256 = "1159e04b15ad8b17c2a088e4c011d1b2bbc3c313d09db690ad8039addece422c"
EXPECTED_PREFERENCES_VAL_SHA256 = "12fd0bc97940b7f85f411efd6fcd061e28f048bc8ce58f4216c6c854d8d7cbc0"


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
    if manifest.get("schema_version") != "kg1_v358_v357_bit_ternary_transfer_dataset_v1":
        raise RuntimeError("Unexpected V358 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V358 real tokenization gate did not pass.")
    if gate.get("dataset_manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("V358 tokenization gate is stale relative to dataset manifest.")
    if gate.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V358 tokenization gate must use boxed_suffix mode.")
    if gate.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V358 HF upload requires the real tokenizer gate, not toy.")

    outputs = manifest.get("outputs", {})
    expected_shas = {
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "val_sha256": EXPECTED_VAL_SHA256,
        "preferences_train_sha256": EXPECTED_PREFERENCES_TRAIN_SHA256,
        "preferences_val_sha256": EXPECTED_PREFERENCES_VAL_SHA256,
    }
    for key, expected in expected_shas.items():
        if outputs.get(key) != expected:
            raise RuntimeError(f"V358 {key} drifted: {outputs.get(key)} != {expected}")

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
    if int(train_validation.get("rows", -1)) != 1152:
        raise RuntimeError("Unexpected V358 train row count.")
    if int(val_validation.get("rows", -1)) != 288:
        raise RuntimeError("Unexpected V358 validation row count.")
    if train_validation.get("family_counts") != {"bit_manipulation": 1152}:
        raise RuntimeError("V358 train family counts drifted.")
    if val_validation.get("family_counts") != {"bit_manipulation": 288}:
        raise RuntimeError("V358 validation family counts drifted.")
    if train_validation.get("subcategory_counts") != {
        "bit_exact_global_binary_replay": 320,
        "bit_exact_global_ternary": 832,
    }:
        raise RuntimeError("V358 train subcategory counts drifted.")
    if val_validation.get("subcategory_counts") != {
        "bit_exact_global_binary_replay": 80,
        "bit_exact_global_ternary": 208,
    }:
        raise RuntimeError("V358 validation subcategory counts drifted.")
    if int(train_validation.get("id_overlap_with_reference", -1)) != 0:
        raise RuntimeError("V358 train ID overlap is not zero.")
    if int(val_validation.get("id_overlap_with_reference", -1)) != 0:
        raise RuntimeError("V358 validation ID overlap is not zero.")
    if int(train_validation.get("prompt_sha256_overlap_with_reference", -1)) != 0:
        raise RuntimeError("V358 train prompt overlap is not zero.")
    if int(val_validation.get("prompt_sha256_overlap_with_reference", -1)) != 0:
        raise RuntimeError("V358 validation prompt overlap is not zero.")
    if int(manifest.get("validation", {}).get("train_val_prompt_overlap", -1)) != 0:
        raise RuntimeError("V358 train/val prompt overlap is not zero.")

    for split in ("train", "validation"):
        token_summary = gate.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V358 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V358 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V358 {split} used fallback masks.")

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
    }


def main() -> int:
    print("=== V358 HF DATASET UPLOAD START ===", flush=True)
    local_gate_summary = validate_local_contract()
    print("local_gate_summary =", json.dumps(local_gate_summary, sort_keys=True), flush=True)

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V358 dataset upload.")
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add KG1 V358 V357 bit ternary transfer dataset",
    )
    out = {
        "version": "v358_v357_bit_ternary_transfer_hf_upload",
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
    out_path = RUN_DIR / "v358_v357_bit_ternary_transfer_hf_upload_manifest.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest =", out_path, flush=True)
    print("upload_url =", info, flush=True)
    print("=== V358 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

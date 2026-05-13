#!/usr/bin/env python3
"""Upload the V337D minimal transfer dataset and tokenization gate to HF."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate")
GATE_DIR = DATASET_DIR / "tokenization_gate_real"
DATASET_MANIFEST = DATASET_DIR / "v337d_minimal_transfer_manifest.json"
GATE_MANIFEST = GATE_DIR / "v286_generic_tokenization_gate_manifest.json"
DATASET_PATH_IN_REPO = "data/v337d_minimal_transfer/20260513T_cpu_gate"
GATE_PATH_IN_REPO = "runtime_artifacts/v337d_minimal_transfer_tokenization_gate/20260513T_cpu_gate"
EXPECTED_TRAIN_SHA256 = "df67214d3fdbb74ada96a9fc24609db5a3f5f6dc1d26dea5d4449eb39eb4147c"
EXPECTED_VAL_SHA256 = "50d4ee05a377ed4e111d27f9de0e1109eb0c09bfe01a9bce0717b63d704dbf80"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_local_gates() -> dict:
    for path in [DATASET_DIR, GATE_DIR, DATASET_MANIFEST, GATE_MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(path)
    dataset_manifest = read_json(DATASET_MANIFEST)
    gate_manifest = read_json(GATE_MANIFEST)
    if dataset_manifest.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("Unexpected V337D dataset schema.")
    if gate_manifest.get("schema_version") != "kg1_v286_generic_tokenization_gate_v1":
        raise RuntimeError("Unexpected V286 tokenization gate schema.")
    if gate_manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V337D tokenization gate did not pass.")
    if gate_manifest.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V337D tokenization gate was not run in boxed_suffix mode.")
    if gate_manifest.get("dataset_manifest_sha256") != sha256_file(DATASET_MANIFEST):
        raise RuntimeError("V337D tokenization gate is stale relative to dataset manifest.")
    outputs = dataset_manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256 or outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V337D train/validation hashes drifted.")
    tokenization = gate_manifest.get("tokenization", {})
    for split in ("train", "validation"):
        split_summary = tokenization.get(split, {})
        if int(split_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V337D {split} prompt truncation is not zero.")
        if int(split_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V337D {split} completion token drop is not zero.")
        if int(split_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V337D {split} used fallback masks.")
    return {
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "gate_manifest_sha256": sha256_file(GATE_MANIFEST),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
    }


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V337D dataset.")
    gate_summary = validate_local_gates()

    api = HfApi(token=token)
    print("=== V337D HF DATASET UPLOAD START ===", flush=True)
    print("data_repo =", DATA_REPO, flush=True)
    print("dataset_dir =", DATASET_DIR, flush=True)
    print("dataset_path_in_repo =", DATASET_PATH_IN_REPO, flush=True)
    dataset_upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=DATASET_PATH_IN_REPO,
        commit_message="Add V337D minimal transfer dataset",
        ignore_patterns=["tokenization_gate_real/**", "tokenization_gate_toy/**"],
    )
    print("dataset_upload =", dataset_upload, flush=True)
    print("gate_dir =", GATE_DIR, flush=True)
    print("gate_path_in_repo =", GATE_PATH_IN_REPO, flush=True)
    print("local_gate_summary =", json.dumps(gate_summary, sort_keys=True), flush=True)
    gate_upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(GATE_DIR),
        path_in_repo=GATE_PATH_IN_REPO,
        commit_message="Add V337D tokenization gate artifact",
    )
    print("gate_upload =", gate_upload, flush=True)

    manifest = {
        "version": "v337d_hf_dataset_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_repo": DATA_REPO,
        "dataset_dir": str(DATASET_DIR),
        "dataset_path_in_repo": DATASET_PATH_IN_REPO,
        "dataset_upload": str(dataset_upload),
        "gate_dir": str(GATE_DIR),
        "gate_path_in_repo": GATE_PATH_IN_REPO,
        "gate_upload": str(gate_upload),
        "local_gate_summary": gate_summary,
        "next_action": "Commit/push V338 launcher and run a short A100 smoke train with first-checkpoint FinOps kill-switch.",
    }
    out_path = Path(__file__).resolve().parent / "v337d_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V337D HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

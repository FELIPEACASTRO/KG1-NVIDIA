#!/usr/bin/env python3
"""Upload V344 transfer SFT and preference assets to the KG1 HF dataset repo."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v344_v343_transfer_dataset/20260513T_minimal_transfer_v343")
MANIFEST = DATASET_DIR / "v344_v343_minimal_transfer_manifest.json"
PATH_IN_REPO = "data/v344_v343_minimal_transfer/20260513T_minimal_transfer_v343"

EXPECTED_TRAIN_SHA256 = "cab6b8370f2208c3e3fa954527967683be06639d7c556ae7697077d1d2bf8e03"
EXPECTED_VAL_SHA256 = "a2df22315cbd837d6b15c9ff646d76fb7b8d8e3930485ac0b02677b9ed9c87cc"
EXPECTED_PREF_TRAIN_SHA256 = "cd2c3021731ff141173cd05bed51cb3086320ce4996cd5a55b8e38b678cfee90"
EXPECTED_PREF_VAL_SHA256 = "c9fe097ccebfeb9de895abda139688bb0256bcc00f35fb1fd292f53fdbce23a2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash drift: expected {expected}, got {observed}")


def validate_local_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = read_json(MANIFEST)
    if manifest.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("Unexpected V344 transfer manifest schema.")
    outputs = manifest.get("outputs", {})
    train_path = Path(outputs.get("train_jsonl", ""))
    val_path = Path(outputs.get("val_jsonl", ""))
    pref_train_path = Path(outputs.get("preferences_train_jsonl", ""))
    pref_val_path = Path(outputs.get("preferences_val_jsonl", ""))
    for path in (train_path, val_path, pref_train_path, pref_val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    assert_hash(train_path, EXPECTED_TRAIN_SHA256, "V344 SFT train")
    assert_hash(val_path, EXPECTED_VAL_SHA256, "V344 SFT validation")
    assert_hash(pref_train_path, EXPECTED_PREF_TRAIN_SHA256, "V344 preference train")
    assert_hash(pref_val_path, EXPECTED_PREF_VAL_SHA256, "V344 preference validation")
    validation = manifest.get("validation", {})
    train = validation.get("train", {})
    val = validation.get("validation", {})
    expected_counts = {
        "train_rows": 1760,
        "val_rows": 420,
        "train_bit": 720,
        "train_equation": 1040,
        "val_bit": 160,
        "val_equation": 260,
    }
    observed_counts = {
        "train_rows": int(train.get("rows", -1)),
        "val_rows": int(val.get("rows", -1)),
        "train_bit": int(train.get("family_counts", {}).get("bit_manipulation", -1)),
        "train_equation": int(train.get("family_counts", {}).get("equation_transform", -1)),
        "val_bit": int(val.get("family_counts", {}).get("bit_manipulation", -1)),
        "val_equation": int(val.get("family_counts", {}).get("equation_transform", -1)),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(f"V344 count drift: expected {expected_counts}, got {observed_counts}")
    for split_name, split in (("train", train), ("validation", val)):
        if int(split.get("reference_id_overlap", -1)) != 0:
            raise RuntimeError(f"V344 {split_name} reference_id_overlap drift")
        if int(split.get("reference_prompt_overlap", -1)) != 0:
            raise RuntimeError(f"V344 {split_name} reference_prompt_overlap drift")
    return {
        "manifest_sha256": sha256_file(MANIFEST),
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "val_sha256": EXPECTED_VAL_SHA256,
        "preferences_train_sha256": EXPECTED_PREF_TRAIN_SHA256,
        "preferences_val_sha256": EXPECTED_PREF_VAL_SHA256,
        **observed_counts,
    }


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V344 transfer assets.")
    local_summary = validate_local_manifest()
    api = HfApi(token=token)
    print("=== V344 HF TRANSFER UPLOAD START ===", flush=True)
    print("data_repo =", DATA_REPO, flush=True)
    print("dataset_dir =", DATASET_DIR, flush=True)
    print("path_in_repo =", PATH_IN_REPO, flush=True)
    print("local_summary =", json.dumps(local_summary, sort_keys=True), flush=True)
    upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add V344 V343 transfer assets",
    )
    print("upload =", upload, flush=True)
    manifest = {
        "version": "v344_v343_transfer_hf_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_repo": DATA_REPO,
        "dataset_dir": str(DATASET_DIR),
        "path_in_repo": PATH_IN_REPO,
        "upload": str(upload),
        "local_summary": local_summary,
        "next_action": "Run V340 with V344 preference launcher, then debug launch before paid HF smoke.",
    }
    out_path = Path(__file__).resolve().parent / "v344_v343_transfer_hf_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V344 HF TRANSFER UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

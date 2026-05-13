#!/usr/bin/env python3
"""Upload V341 cleaned preference files to the KG1 HF dataset repo."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v341_clean_preference_transfer_dataset/20260513T_cpu_gate")
MANIFEST = DATASET_DIR / "v341_clean_preference_transfer_manifest.json"
PATH_IN_REPO = "data/v341_clean_preference_transfer/20260513T_cpu_gate"
EXPECTED_PREF_TRAIN_SHA256 = "217068058e723063178c00e5b9de697a1a669839e85abdb86154663543a71ae2"
EXPECTED_PREF_VAL_SHA256 = "6aa1bfcd795a6bb11fa50551067e0744159a697331f2fff3bdd9084ce912cd8a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_local_manifest() -> dict:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = read_json(MANIFEST)
    if manifest.get("schema_version") != "kg1_v341_clean_preference_transfer_dataset_v1":
        raise RuntimeError("Unexpected V341 manifest schema.")
    outputs = manifest.get("outputs", {})
    train_path = Path(outputs.get("preferences_train_jsonl", ""))
    val_path = Path(outputs.get("preferences_val_jsonl", ""))
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("V341 preference files are missing.")
    if sha256_file(train_path) != EXPECTED_PREF_TRAIN_SHA256:
        raise RuntimeError("V341 cleaned preference train hash drift.")
    if sha256_file(val_path) != EXPECTED_PREF_VAL_SHA256:
        raise RuntimeError("V341 cleaned preference validation hash drift.")
    summary = manifest.get("summary", {})
    if summary.get("train", {}).get("removed_rows") != 37:
        raise RuntimeError("Unexpected V341 train removed_rows count.")
    if summary.get("validation", {}).get("removed_rows") != 5:
        raise RuntimeError("Unexpected V341 validation removed_rows count.")
    return {
        "manifest_sha256": sha256_file(MANIFEST),
        "preferences_train_sha256": EXPECTED_PREF_TRAIN_SHA256,
        "preferences_val_sha256": EXPECTED_PREF_VAL_SHA256,
        "preferences_train_rows": summary.get("train", {}).get("kept_rows"),
        "preferences_val_rows": summary.get("validation", {}).get("kept_rows"),
    }


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V341 cleaned preferences.")
    local_summary = validate_local_manifest()
    api = HfApi(token=token)
    print("=== V341 HF CLEAN PREFERENCE UPLOAD START ===", flush=True)
    print("data_repo =", DATA_REPO, flush=True)
    print("dataset_dir =", DATASET_DIR, flush=True)
    print("path_in_repo =", PATH_IN_REPO, flush=True)
    print("local_summary =", json.dumps(local_summary, sort_keys=True), flush=True)
    upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=PATH_IN_REPO,
        commit_message="Add V341 cleaned preference transfer files",
    )
    print("upload =", upload, flush=True)
    manifest = {
        "version": "v341_clean_preference_hf_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_repo": DATA_REPO,
        "dataset_dir": str(DATASET_DIR),
        "path_in_repo": PATH_IN_REPO,
        "upload": str(upload),
        "local_summary": local_summary,
        "next_action": "Run V340 cleaned gate, then a tiny preference smoke if still allowed.",
    }
    out_path = Path(__file__).resolve().parent / "v341_clean_preference_hf_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V341 HF CLEAN PREFERENCE UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

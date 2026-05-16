#!/usr/bin/env python3
"""Upload V479 objective-aligned dataset and gate artifacts to HF."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter")
DATASET_PATH_IN_REPO = "data/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter"


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V479 dataset.")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)
    api = HfApi(token=token)
    print("=== V480 V479 HF DATASET UPLOAD START ===", flush=True)
    print("data_repo =", DATA_REPO, flush=True)
    print("dataset_dir =", DATASET_DIR, flush=True)
    print("dataset_path_in_repo =", DATASET_PATH_IN_REPO, flush=True)
    upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=DATASET_PATH_IN_REPO,
        commit_message="Add V479 objective-aligned equation bit dataset",
    )
    print("dataset_upload =", upload, flush=True)
    manifest = {
        "version": "v480_v479_dataset_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_repo": DATA_REPO,
        "dataset_dir": str(DATASET_DIR),
        "dataset_path_in_repo": DATASET_PATH_IN_REPO,
        "dataset_upload": str(upload),
        "train_file": f"{DATASET_PATH_IN_REPO}/v479_objective_aligned_filter_train.jsonl",
        "val_file": f"{DATASET_PATH_IN_REPO}/v479_objective_aligned_filter_val.jsonl",
        "next_action": "Launch V480 only after local debug verifies hashes and V478 objective alignment.",
    }
    out_path = Path(__file__).resolve().parent / "v480_v479_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V480 V479 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload the V337D minimal transfer dataset and tokenization gate to HF."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate")
GATE_DIR = DATASET_DIR / "tokenization_gate_real"
DATASET_PATH_IN_REPO = "data/v337d_minimal_transfer/20260513T_cpu_gate"
GATE_PATH_IN_REPO = "runtime_artifacts/v337d_minimal_transfer_tokenization_gate/20260513T_cpu_gate"


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V337D dataset.")
    for path in [DATASET_DIR, GATE_DIR]:
        if not path.exists():
            raise FileNotFoundError(path)

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
        "next_action": "Commit/push V338 launcher and run a short A100 smoke train with first-checkpoint FinOps kill-switch.",
    }
    out_path = Path(__file__).resolve().parent / "v337d_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V337D HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

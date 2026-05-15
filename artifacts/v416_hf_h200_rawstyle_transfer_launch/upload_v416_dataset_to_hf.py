#!/usr/bin/env python3
"""Upload the V416 rawstyle transfer dataset and tokenization gate to HF."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_DIR = Path("artifacts/v416_rawstyle_transfer_dataset/20260515T_v416_rawstyle_transfer")
GATE_DIR = DATASET_DIR / "tokenization_gate_real"
DATASET_PATH_IN_REPO = "data/v416_rawstyle_transfer/20260515T_v416_rawstyle_transfer"
GATE_PATH_IN_REPO = "runtime_artifacts/v416_rawstyle_transfer_tokenization_gate/20260515T_v416_rawstyle_transfer"


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to upload V416 dataset.")
    for path in [DATASET_DIR, GATE_DIR]:
        print("required_path =", path, "exists =", path.exists(), flush=True)
        if not path.exists():
            raise FileNotFoundError(path)

    api = HfApi(token=token)
    print("=== V416 HF DATASET UPLOAD START ===", flush=True)
    print("data_repo =", DATA_REPO, flush=True)
    print("dataset_dir =", DATASET_DIR, flush=True)
    print("dataset_path_in_repo =", DATASET_PATH_IN_REPO, flush=True)
    dataset_upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATASET_DIR),
        path_in_repo=DATASET_PATH_IN_REPO,
        commit_message="Add V416 rawstyle transfer dataset",
        ignore_patterns=["tokenization_gate_real/**"],
    )
    print("dataset_upload =", dataset_upload, flush=True)

    print("gate_dir =", GATE_DIR, flush=True)
    print("gate_path_in_repo =", GATE_PATH_IN_REPO, flush=True)
    gate_upload = api.upload_folder(
        repo_id=DATA_REPO,
        repo_type="dataset",
        folder_path=str(GATE_DIR),
        path_in_repo=GATE_PATH_IN_REPO,
        commit_message="Add V416 rawstyle tokenization gate artifact",
    )
    print("gate_upload =", gate_upload, flush=True)

    manifest = {
        "version": "v416_hf_dataset_upload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_repo": DATA_REPO,
        "dataset_dir": str(DATASET_DIR),
        "dataset_path_in_repo": DATASET_PATH_IN_REPO,
        "dataset_upload": str(dataset_upload),
        "gate_dir": str(GATE_DIR),
        "gate_path_in_repo": GATE_PATH_IN_REPO,
        "gate_upload": str(gate_upload),
        "previous_version": "V415 adapter-direct audit",
        "next_action": "Commit/push V416 launcher and run a short H200 smoke train if budget gate allows.",
    }
    out_path = Path(__file__).resolve().parent / "v416_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    print("=== V416 HF DATASET UPLOAD END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

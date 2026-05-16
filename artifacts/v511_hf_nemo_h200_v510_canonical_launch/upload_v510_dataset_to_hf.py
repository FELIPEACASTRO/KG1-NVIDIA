#!/usr/bin/env python3
"""Upload the V510 canonical active training pool to the KG1 HF dataset repo.

This helper verifies local hashes, row counts, integrity audit, tokenization
gate, and objective-alignment gate before uploading. It does not train,
package, or submit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_REPO = "felipesp1983/kg1-nemotron-training"
LOCAL_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool"
)
PATH_IN_REPO = "data/v510_canonical_training_dataset/v510_canonical_active_training_pool"
TRAIN_FILE = LOCAL_OUTPUT_DIR / "v510_canonical_active_training_pool_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v510_canonical_active_training_pool_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v510_canonical_active_training_pool_manifest.json"
TRIAGE_FILE = LOCAL_OUTPUT_DIR / "V510_CANONICAL_DATASET_TRIAGE.md"
INTEGRITY_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v509_reaudit/v510_canonical_reaudit_manifest.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "tokenization_gate_real_local/v286_generic_tokenization_gate_manifest.json"
OBJECTIVE_ALIGNMENT_FILE = LOCAL_OUTPUT_DIR / "v510_objective_alignment_gate.json"
TRAIN_SHA256 = "9033e794bad98679f26bb2fc7f1eb5d4d7f32d06ef6231ee6e0fffc66fc70d3b"
VAL_SHA256 = "062514b8a74ba3656df44ad99667ba63dda69f56d41a20ffb0500f17393ceea8"
TRAIN_ROWS = 2627
VAL_ROWS = 637


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows += 1
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_local() -> dict[str, Any]:
    print("=== V511/V510 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [
        TRAIN_FILE,
        VAL_FILE,
        MANIFEST_FILE,
        TRIAGE_FILE,
        INTEGRITY_MANIFEST_FILE,
        TOKENIZATION_MANIFEST_FILE,
        OBJECTIVE_ALIGNMENT_FILE,
    ]
    for path in required:
        print("required_file =", path, "exists =", path.exists(), flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    train_sha = sha256_file(TRAIN_FILE)
    val_sha = sha256_file(VAL_FILE)
    train_rows = count_jsonl(TRAIN_FILE)
    val_rows = count_jsonl(VAL_FILE)
    print("train_sha256 =", train_sha, flush=True)
    print("val_sha256 =", val_sha, flush=True)
    print("train_rows =", train_rows, flush=True)
    print("val_rows =", val_rows, flush=True)
    if train_sha != TRAIN_SHA256:
        raise RuntimeError(f"train sha mismatch: {train_sha} != {TRAIN_SHA256}")
    if val_sha != VAL_SHA256:
        raise RuntimeError(f"val sha mismatch: {val_sha} != {VAL_SHA256}")
    if train_rows != TRAIN_ROWS:
        raise RuntimeError(f"train rows mismatch: {train_rows} != {TRAIN_ROWS}")
    if val_rows != VAL_ROWS:
        raise RuntimeError(f"val rows mismatch: {val_rows} != {VAL_ROWS}")

    integrity = read_json(INTEGRITY_MANIFEST_FILE)
    if int(integrity.get("blocked_dataset_count", -1)) != 0:
        raise RuntimeError("V510 integrity reaudit has blocked datasets")
    print("integrity_reaudit_status =", integrity.get("decision", {}).get("status"), flush=True)

    token_manifest = read_json(TOKENIZATION_MANIFEST_FILE)
    tokenization_status = token_manifest.get("decision", {}).get("status")
    if tokenization_status != "tokenization_gate_passed":
        raise RuntimeError("V510 tokenization gate did not pass")
    print("tokenization_gate_decision =", tokenization_status, flush=True)

    objective = read_json(OBJECTIVE_ALIGNMENT_FILE)
    if objective.get("hf_gpu_allowed") is not True or objective.get("findings"):
        raise RuntimeError("V510 objective alignment gate did not pass")
    print("objective_alignment_hf_gpu_allowed =", objective.get("hf_gpu_allowed"), flush=True)
    print("=== V511/V510 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "triage_file": str(TRIAGE_FILE),
        "integrity_manifest_file": str(INTEGRITY_MANIFEST_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "objective_alignment_file": str(OBJECTIVE_ALIGNMENT_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "integrity_reaudit_status": integrity.get("decision", {}).get("status"),
        "tokenization_gate_decision": tokenization_status,
        "objective_alignment_hf_gpu_allowed": objective.get("hf_gpu_allowed"),
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v511_v510_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V510 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v511_v510_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V510 dataset.")
        api = HfApi(token=token)
        print("=== V511/V510 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V510 canonical active training pool",
        )
        payload.update(
            {
                "mode": "uploaded",
                "upload_info": str(info),
                "uploaded_at_utc": utc_now(),
                "hf_train_file": f"{PATH_IN_REPO}/{TRAIN_FILE.name}",
                "hf_val_file": f"{PATH_IN_REPO}/{VAL_FILE.name}",
            }
        )
        print("upload_info =", info, flush=True)
        print("=== V511/V510 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

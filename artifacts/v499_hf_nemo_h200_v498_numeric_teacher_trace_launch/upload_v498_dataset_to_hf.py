#!/usr/bin/env python3
"""Upload the V498 numeric teacher trace pack to the KG1 HF dataset repo.

This is a controlled data movement helper. It verifies local hashes and row
counts before uploading, and it writes a manifest without exposing credentials.
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
    / "artifacts/v498_numeric_teacher_trace_pack/20260516T_v498_numeric_teacher"
)
PATH_IN_REPO = "data/v498_numeric_teacher_trace_pack/20260516T_v498_numeric_teacher"
TRAIN_FILE = LOCAL_OUTPUT_DIR / "v498_numeric_teacher_trace_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v498_numeric_teacher_trace_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v498_numeric_teacher_trace_manifest.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
TRAIN_SHA256 = "920b3c30b9ada9ad2685091194dcc53e717f72a9c037cafeef6e494f21511e79"
VAL_SHA256 = "68cda4162214359aaf7cda304c2a06902775b1aadb53fcadfd0edf7ff481ed80"
TRAIN_ROWS = 1712
VAL_ROWS = 428


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


def verify_local() -> dict[str, Any]:
    print("=== V498 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [TRAIN_FILE, VAL_FILE, MANIFEST_FILE, TOKENIZATION_MANIFEST_FILE]
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

    token_manifest = json.loads(TOKENIZATION_MANIFEST_FILE.read_text(encoding="utf-8"))
    decision = token_manifest.get("decision")
    tokenization_status = decision.get("status") if isinstance(decision, dict) else decision
    if tokenization_status != "tokenization_gate_passed":
        raise RuntimeError("V498 tokenization gate did not pass")
    print("tokenization_gate_decision =", tokenization_status, flush=True)
    print("=== V498 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "tokenization_gate_decision": tokenization_status,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v498_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V498 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v498_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V498 dataset.")
        api = HfApi(token=token)
        print("=== V498 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V498 numeric teacher trace pack",
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
        print("=== V498 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

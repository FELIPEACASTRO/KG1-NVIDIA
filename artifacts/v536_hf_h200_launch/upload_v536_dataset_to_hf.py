#!/usr/bin/env python3
"""Upload the V536 dataset and CPU gate artifacts to Hugging Face.

This helper performs local integrity checks first. It uploads only the V536
dataset folder; it does not train, evaluate, package, or submit.
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
DATA_REPO = "felipesp1983/kg1-v536-v534-bit-v523-equation-artifacts"
LOCAL_OUTPUT_DIR = REPO_ROOT / "artifacts/v536_v534_bit_v523_equation_pack/20260517T024752Z"
PATH_IN_REPO = "v536-v534-bit-v523-equation-20260517T024752Z"
TRAIN_FILE = LOCAL_OUTPUT_DIR / "v536_v534_bit_v523_equation_pack_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v536_v534_bit_v523_equation_pack_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v536_v534_bit_v523_equation_pack_manifest.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
LEARNABILITY_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v513_recheck/v513_trace_learnability_gate_manifest.json"
QUOTA_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v524_objective_audit/v524_quota_token_objective_manifest.json"
EXAMPLE_MEAN_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v526_example_mean_dry_run/v526_example_mean_objective_dry_run_manifest.json"
TRAIN_SHA256 = "6c7a91891156cdc666ffbd6478ddfe02bc0c258473615907e40675c2aa716700"
VAL_SHA256 = "26bf3b0393c3f9757e22b13d91583f15542a5b6f2d1ae13fbf3e34322262f093"
TRAIN_ROWS = 1026
VAL_ROWS = 219


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_local() -> dict[str, Any]:
    print("=== V536 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [
        TRAIN_FILE,
        VAL_FILE,
        MANIFEST_FILE,
        TOKENIZATION_MANIFEST_FILE,
        LEARNABILITY_MANIFEST_FILE,
        QUOTA_MANIFEST_FILE,
        EXAMPLE_MEAN_MANIFEST_FILE,
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

    dataset_manifest = read_json(MANIFEST_FILE)
    if dataset_manifest.get("decision", {}).get("status") != "dataset_ready_for_cpu_gates":
        raise RuntimeError("V536 dataset manifest is not ready for CPU gates")
    tokenization = read_json(TOKENIZATION_MANIFEST_FILE)
    if tokenization.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V536 tokenization gate did not pass")
    learnability = read_json(LEARNABILITY_MANIFEST_FILE)
    if learnability.get("decision", {}).get("status") != "passed_cpu_structure_only":
        raise RuntimeError("V536 learnability gate did not pass")
    example_mean = read_json(EXAMPLE_MEAN_MANIFEST_FILE)
    if example_mean.get("decision", {}).get("status") != "example_mean_dry_run_passed":
        raise RuntimeError("V536 example_mean dry-run did not pass")
    print("dataset_manifest_status =", dataset_manifest.get("decision", {}).get("status"), flush=True)
    print("tokenization_status =", tokenization.get("decision", {}).get("status"), flush=True)
    print("learnability_status =", learnability.get("decision", {}).get("status"), flush=True)
    print("example_mean_status =", example_mean.get("decision", {}).get("status"), flush=True)
    print("=== V536 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "learnability_manifest_file": str(LEARNABILITY_MANIFEST_FILE),
        "quota_manifest_file": str(QUOTA_MANIFEST_FILE),
        "example_mean_manifest_file": str(EXAMPLE_MEAN_MANIFEST_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v536_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V536 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v536_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V536 dataset.")
        api = HfApi(token=token)
        api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        print("=== V536 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V536 V534-bit V523-equation pack",
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
        print("=== V536 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

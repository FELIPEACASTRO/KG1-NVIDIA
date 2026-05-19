#!/usr/bin/env python3
"""Upload the gated V673 guarded equation/bit transfer dataset to Hugging Face.

This script uploads only after local CPU gates have passed. It does not train,
evaluate weak ACC, package, or submit.
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
DATA_REPO = "felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts"
LOCAL_OUTPUT_DIR = REPO_ROOT / "artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T174833Z"
PATH_IN_REPO = "v673-guarded-equation-bit-transfer-20260519T174833Z"

TRAIN_FILE = LOCAL_OUTPUT_DIR / "v673_guarded_equation_bit_transfer_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v673_guarded_equation_bit_transfer_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v673_guarded_equation_bit_transfer_manifest.json"
INTEGRITY_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v509_integrity/v673_guarded_equation_bit_transfer_manifest.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
LEARNABILITY_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v513_learnability/v513_trace_learnability_gate_manifest.json"
OBJECTIVE_ALIGNMENT_FILE = LOCAL_OUTPUT_DIR / "v478_objective_alignment/v673_v478_objective_alignment.json"

EXPECTED_TRAIN_SHA256 = "cdf85573584c2bb965f8fb19bb8b698e7b03a7231013d39a74ff0410e0d76343"
EXPECTED_VAL_SHA256 = "858c02fcc046d130c4405aac942c102aaf0ded38c347479734c5339d6960e057"
EXPECTED_TRAIN_ROWS = 720
EXPECTED_VAL_ROWS = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for raw in handle if raw.strip())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_local() -> dict[str, Any]:
    print("=== V673 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [
        TRAIN_FILE,
        VAL_FILE,
        MANIFEST_FILE,
        INTEGRITY_MANIFEST_FILE,
        TOKENIZATION_MANIFEST_FILE,
        LEARNABILITY_MANIFEST_FILE,
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
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f"train SHA mismatch: {train_sha} != {EXPECTED_TRAIN_SHA256}")
    if val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(f"val SHA mismatch: {val_sha} != {EXPECTED_VAL_SHA256}")
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"train rows mismatch: {train_rows} != {EXPECTED_TRAIN_ROWS}")
    if val_rows != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"val rows mismatch: {val_rows} != {EXPECTED_VAL_ROWS}")

    integrity = read_json(INTEGRITY_MANIFEST_FILE)
    tokenization = read_json(TOKENIZATION_MANIFEST_FILE)
    learnability = read_json(LEARNABILITY_MANIFEST_FILE)
    objective = read_json(OBJECTIVE_ALIGNMENT_FILE)
    checks = {
        "integrity_status": integrity.get("decision", {}).get("status"),
        "tokenization_status": tokenization.get("decision", {}).get("status"),
        "learnability_status": learnability.get("decision", {}).get("status"),
        "learnability_warning_count": learnability.get("finding_counts", {}).get("warning"),
        "learnability_blocker_count": learnability.get("finding_counts", {}).get("blocker"),
        "objective_hf_gpu_allowed": objective.get("hf_gpu_allowed"),
    }
    print("gate_checks =", json.dumps(checks, indent=2, sort_keys=True), flush=True)
    if checks["integrity_status"] != "datasets_pass_integrity_audit":
        raise RuntimeError("V673 integrity gate did not pass")
    if checks["tokenization_status"] != "tokenization_gate_passed":
        raise RuntimeError("V673 tokenization gate did not pass")
    if checks["learnability_status"] != "passed_cpu_structure_only":
        raise RuntimeError("V673 learnability gate did not pass")
    if checks["learnability_warning_count"] != 0 or checks["learnability_blocker_count"] != 0:
        raise RuntimeError("V673 learnability gate has blockers/warnings")
    if checks["objective_hf_gpu_allowed"] is not True:
        raise RuntimeError("V673 objective alignment did not allow HF GPU")
    print("=== V673 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "integrity_manifest_file": str(INTEGRITY_MANIFEST_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "learnability_manifest_file": str(LEARNABILITY_MANIFEST_FILE),
        "objective_alignment_file": str(OBJECTIVE_ALIGNMENT_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "checks": checks,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v673_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V673 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v673_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V673 dataset.")
        api = HfApi(token=token)
        api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        print("=== V673 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V673 guarded equation bit transfer pack",
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
        print("=== V673 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload the V573 source-only reference-weighted dataset to Hugging Face.

This helper uploads only the already gated CPU artifacts. It does not train,
evaluate, package, or submit.
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
DATA_REPO = "felipesp1983/kg1-v573-v571-bitpair-v551-equation-reference-artifacts"
LOCAL_OUTPUT_DIR = REPO_ROOT / "artifacts/v573_v571_bitpair_v551_equation_reference_mix/20260517T_v573_cpu_gate"
PATH_IN_REPO = "v573-v571-bitpair-v551-equation-reference-20260517T-v573-cpu-gate"
TRAIN_FILE = LOCAL_OUTPUT_DIR / "v572_v571_bitpair_v551_equation_mix_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v572_v571_bitpair_v551_equation_mix_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v572_v571_bitpair_v551_equation_mix_manifest.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v286_tokenization_real/v286_generic_tokenization_gate_manifest.json"
LEARNABILITY_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v513_learnability_real/v513_trace_learnability_gate_manifest.json"
QUOTA_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v524_objective_real/v524_quota_token_objective_manifest.json"
EXAMPLE_MEAN_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v526_example_mean_row_weight_dry_run/v526_example_mean_objective_dry_run_manifest.json"
OBJECTIVE_ALIGNMENT_FILE = LOCAL_OUTPUT_DIR / "v478_objective_alignment.json"


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
    print("=== V573 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [
        TRAIN_FILE,
        VAL_FILE,
        MANIFEST_FILE,
        TOKENIZATION_MANIFEST_FILE,
        LEARNABILITY_MANIFEST_FILE,
        QUOTA_MANIFEST_FILE,
        EXAMPLE_MEAN_MANIFEST_FILE,
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
    if train_rows != 757:
        raise RuntimeError(f"train rows mismatch: {train_rows} != 757")
    if val_rows != 159:
        raise RuntimeError(f"val rows mismatch: {val_rows} != 159")

    dataset_manifest = read_json(MANIFEST_FILE)
    tokenization = read_json(TOKENIZATION_MANIFEST_FILE)
    learnability = read_json(LEARNABILITY_MANIFEST_FILE)
    example_mean = read_json(EXAMPLE_MEAN_MANIFEST_FILE)
    objective = read_json(OBJECTIVE_ALIGNMENT_FILE)
    checks = {
        "dataset_status": dataset_manifest.get("decision", {}).get("status"),
        "tokenization_status": tokenization.get("decision", {}).get("status"),
        "learnability_status": learnability.get("decision", {}).get("status"),
        "example_mean_status": example_mean.get("decision", {}).get("status"),
        "objective_hf_gpu_allowed": objective.get("hf_gpu_allowed"),
    }
    print("gate_checks =", json.dumps(checks, indent=2, sort_keys=True), flush=True)
    if checks["dataset_status"] != "dataset_ready_for_cpu_gates":
        raise RuntimeError("V573 dataset manifest is not ready for CPU gates")
    if checks["tokenization_status"] != "tokenization_gate_passed":
        raise RuntimeError("V573 tokenization gate did not pass")
    if checks["learnability_status"] != "passed_cpu_structure_only":
        raise RuntimeError("V573 learnability gate did not pass")
    if checks["example_mean_status"] != "example_mean_dry_run_passed":
        raise RuntimeError("V573 row-weighted example_mean dry-run did not pass")
    if checks["objective_hf_gpu_allowed"] is not True:
        raise RuntimeError("V573 objective alignment did not allow HF GPU")
    print("=== V573 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "learnability_manifest_file": str(LEARNABILITY_MANIFEST_FILE),
        "quota_manifest_file": str(QUOTA_MANIFEST_FILE),
        "example_mean_manifest_file": str(EXAMPLE_MEAN_MANIFEST_FILE),
        "objective_alignment_file": str(OBJECTIVE_ALIGNMENT_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v573_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V573 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v573_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V573 dataset.")
        api = HfApi(token=token)
        api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        print("=== V573 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V573 source-only reference-weighted pack",
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
        print("=== V573 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

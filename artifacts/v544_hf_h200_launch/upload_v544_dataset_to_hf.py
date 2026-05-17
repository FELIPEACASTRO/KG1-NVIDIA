#!/usr/bin/env python3
"""Upload the V544 minimal distillation dataset to Hugging Face.

This helper performs local integrity checks first. It uploads only the verified
V544 dataset/gate artifacts; it does not train, evaluate, package, or submit.
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
DATA_REPO = "felipesp1983/kg1-v544-minimal-distillation-artifacts"
LOCAL_OUTPUT_DIR = REPO_ROOT / "artifacts/v544_minimal_distillation_dataset/20260517T_v544_cpu_gate"
PATH_IN_REPO = "v544-minimal-distillation-20260517T063045Z"

TRAIN_FILE = LOCAL_OUTPUT_DIR / "v544_minimal_distillation_train.jsonl"
VAL_FILE = LOCAL_OUTPUT_DIR / "v544_minimal_distillation_val.jsonl"
MANIFEST_FILE = LOCAL_OUTPUT_DIR / "v544_minimal_distillation_manifest.json"
TEACHER_AUDIT_FILE = LOCAL_OUTPUT_DIR / "v544_teacher_gain_audit.csv"
DATASET_DOUBLECHECK_FILE = LOCAL_OUTPUT_DIR / "v544_dataset_doublecheck_audit.json"
TOKENIZATION_MANIFEST_FILE = LOCAL_OUTPUT_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
OBJECTIVE_ALIGNMENT_FILE = LOCAL_OUTPUT_DIR / "v478_objective_alignment_row_weighted.json"
V543_MANIFEST_FILE = (
    REPO_ROOT
    / "artifacts/v542_cpu_equation_solver_gate/v543_symbolic_queryop_on_v350_v516_strict/v543_symbolic_queryop_refinement_manifest.json"
)

TRAIN_SHA256 = "09f542297d9bafe85015b2955c09289817487ebf9fc53746de4ea68cb5f3e4f3"
VAL_SHA256 = "894a1df7590ccd0ded77f307438e646f870aa2cc6e6e006cb536e88f8aedb921"
TRAIN_ROWS = 236
VAL_ROWS = 115


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
    print("=== V544 HF DATASET UPLOAD PREFLIGHT START ===", flush=True)
    required = [
        TRAIN_FILE,
        VAL_FILE,
        MANIFEST_FILE,
        TEACHER_AUDIT_FILE,
        DATASET_DOUBLECHECK_FILE,
        TOKENIZATION_MANIFEST_FILE,
        OBJECTIVE_ALIGNMENT_FILE,
        V543_MANIFEST_FILE,
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
    if dataset_manifest.get("decision", {}).get("status") != "dataset_ready_for_v286_tokenization_gate":
        raise RuntimeError("V544 dataset manifest is not ready for tokenization gate")
    if dataset_manifest.get("decision", {}).get("gpu_allowed") is not False:
        raise RuntimeError("V544 dataset manifest should not directly allow GPU")
    doublecheck = read_json(DATASET_DOUBLECHECK_FILE)
    if doublecheck.get("decision") != "dataset_doublecheck_passed" or doublecheck.get("issues"):
        raise RuntimeError("V544 dataset doublecheck did not pass")
    doublecheck_manifest = doublecheck.get("manifest", {})
    if doublecheck_manifest.get("train_sha256_observed") != TRAIN_SHA256:
        raise RuntimeError("V544 dataset doublecheck train hash is stale")
    if doublecheck_manifest.get("val_sha256_observed") != VAL_SHA256:
        raise RuntimeError("V544 dataset doublecheck val hash is stale")
    tokenization = read_json(TOKENIZATION_MANIFEST_FILE)
    if tokenization.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V544 tokenization gate did not pass")
    objective = read_json(OBJECTIVE_ALIGNMENT_FILE)
    if objective.get("hf_gpu_allowed") is not True or objective.get("findings"):
        raise RuntimeError("V544 row-weighted objective alignment did not pass")
    v543 = read_json(V543_MANIFEST_FILE)
    if v543.get("decision", {}).get("decision") != "v543_symbolic_queryop_refinement_gate_passed":
        raise RuntimeError("V543 CPU teacher gate did not pass")
    if v543.get("v543_summary", {}).get("correct") != 200:
        raise RuntimeError("V543 CPU teacher expected 200/315")

    print("dataset_manifest_status =", dataset_manifest.get("decision", {}).get("status"), flush=True)
    print("doublecheck_status =", doublecheck.get("decision"), flush=True)
    print("tokenization_status =", tokenization.get("decision", {}).get("status"), flush=True)
    print("objective_hf_gpu_allowed =", objective.get("hf_gpu_allowed"), flush=True)
    print("v543_summary =", json.dumps(v543.get("v543_summary", {}), sort_keys=True), flush=True)
    print("=== V544 HF DATASET UPLOAD PREFLIGHT END ===", flush=True)
    return {
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "teacher_audit_file": str(TEACHER_AUDIT_FILE),
        "dataset_doublecheck_file": str(DATASET_DOUBLECHECK_FILE),
        "tokenization_manifest_file": str(TOKENIZATION_MANIFEST_FILE),
        "objective_alignment_file": str(OBJECTIVE_ALIGNMENT_FILE),
        "v543_manifest_file": str(V543_MANIFEST_FILE),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / "v544_hf_dataset_upload_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("upload_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload the verified V544 folder to HF.")
    args = parser.parse_args()

    local_report = verify_local()
    payload: dict[str, Any] = {
        "schema_version": "kg1_v544_hf_dataset_upload_v1",
        "generated_at_utc": utc_now(),
        "mode": "dry_run_no_upload",
        "data_repo": DATA_REPO,
        "path_in_repo": PATH_IN_REPO,
        "local_report": local_report,
    }
    if args.upload:
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required to upload V544 dataset.")
        api = HfApi(token=token)
        api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        print("=== V544 HF DATASET UPLOAD START ===", flush=True)
        print("data_repo =", DATA_REPO, flush=True)
        print("path_in_repo =", PATH_IN_REPO, flush=True)
        info = api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(LOCAL_OUTPUT_DIR),
            path_in_repo=PATH_IN_REPO,
            commit_message="Upload KG1 V544 minimal distillation pack",
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
        print("=== V544 HF DATASET UPLOAD END ===", flush=True)

    write_manifest(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

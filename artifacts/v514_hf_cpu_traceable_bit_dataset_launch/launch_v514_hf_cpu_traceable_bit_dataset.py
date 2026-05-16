#!/usr/bin/env python3
"""Launch V514 traceable bit dataset gates on Hugging Face CPU.

This reproduces the local V514 CPU path in a clean HF environment:

1. Build V514 from tracked V510 inputs.
2. Run the real tokenization/leakage gate.
3. Run V513 trace learnability against the V514 outputs.
4. Upload only CPU gate artifacts to a private HF dataset repo.

It never trains, evaluates a model, packages, or submits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v514_hf_cpu_traceable_bit_dataset_gate"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "python:3.11"
FLAVOR = "cpu-upgrade"
RUN_ID = "v514-hf-cpu-traceable-bit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DATASET_REPO = "felipesp1983/kg1-v514-traceable-bit-v510-artifacts"


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
python - <<'PY'
import json, platform
print(json.dumps({"python": platform.python_version(), "platform": platform.platform()}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
python -m pip install -q --no-cache-dir --upgrade pip
python -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' 'transformers>=4.51.0' 'tokenizers>=0.20.0' 'pandas>=2.0.0' 'jinja2>=3.1.0'
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
python -m py_compile \
  scripts/build_v514_traceable_bit_v510_dataset.py \
  scripts/run_v286_generic_tokenization_gate.py \
  scripts/audit_v513_trace_learnability_gate.py

echo "=== V514 HF CPU TRACEABLE BIT DATASET START ==="
echo "run_id=$KG1_RUN_ID"
echo "output_dataset_repo=$KG1_OUTPUT_DATASET_REPO"
OUT="artifacts/v514_traceable_bit_v510_dataset/${KG1_RUN_ID}"
python scripts/build_v514_traceable_bit_v510_dataset.py --output-dir "$OUT"
python scripts/run_v286_generic_tokenization_gate.py \
  --dataset-manifest-json "$OUT/v514_traceable_bit_v510_manifest.json" \
  --output-dir "$OUT/tokenization_gate_real" \
  --assistant-final-answer-mode boxed_suffix \
  --min-train-rows 2400 \
  --min-val-rows 600 \
  --reference-csv artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv
python scripts/audit_v513_trace_learnability_gate.py \
  --train-jsonl "$OUT/v514_traceable_bit_v510_train.jsonl" \
  --val-jsonl "$OUT/v514_traceable_bit_v510_val.jsonl" \
  --tokenization-manifest "$OUT/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json" \
  --output-dir "$OUT/v513_recheck"
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

out = Path("artifacts/v514_traceable_bit_v510_dataset") / os.environ["KG1_RUN_ID"]
v514 = json.loads((out / "v514_traceable_bit_v510_manifest.json").read_text(encoding="utf-8"))
tok = json.loads((out / "tokenization_gate_real" / "v286_generic_tokenization_gate_manifest.json").read_text(encoding="utf-8"))
v513 = json.loads((out / "v513_recheck" / "v513_trace_learnability_gate_manifest.json").read_text(encoding="utf-8"))
print("v514_decision =", json.dumps(v514.get("decision", {}), sort_keys=True), flush=True)
print("v514_train_counts =", json.dumps(v514.get("train_summary", {}).get("counts", {}), sort_keys=True), flush=True)
print("v514_validation_counts =", json.dumps(v514.get("validation_summary", {}).get("counts", {}), sort_keys=True), flush=True)
print("v286_decision =", json.dumps(tok.get("decision", {}), sort_keys=True), flush=True)
print("v513_decision =", json.dumps(v513.get("decision", {}), sort_keys=True), flush=True)
print("v513_finding_counts =", json.dumps(v513.get("finding_counts", {}), sort_keys=True), flush=True)
repo_id = os.environ["KG1_OUTPUT_DATASET_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out), path_in_repo=os.environ["KG1_RUN_ID"])
print("v514_uploaded_repo =", repo_id, flush=True)
print("v514_uploaded_path =", os.environ["KG1_RUN_ID"], flush=True)
PY
echo "=== V514 HF CPU TRACEABLE BIT DATASET END ==="
"""


def hardware_to_dict(item: object) -> dict[str, object]:
    accelerator = getattr(item, "accelerator", None)
    return {
        "name": str(getattr(item, "name", "")),
        "pretty_name": str(getattr(item, "pretty_name", "")),
        "cpu": str(getattr(item, "cpu", "")),
        "ram": str(getattr(item, "ram", "")),
        "accelerator_model": str(getattr(accelerator, "model", "")) if accelerator else "",
        "accelerator_quantity": str(getattr(accelerator, "quantity", "")) if accelerator else "",
        "accelerator_vram": str(getattr(accelerator, "vram", "")) if accelerator else "",
        "unit_cost_usd": float(getattr(item, "unit_cost_usd", 0.0) or 0.0),
        "unit_label": str(getattr(item, "unit_label", "")),
    }


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_RUN_ID": RUN_ID,
        "KG1_OUTPUT_DATASET_REPO": OUTPUT_DATASET_REPO,
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.001",
        "KG1_TRAIN_ALLOWED": "0",
        "KG1_GPU_ALLOWED": "0",
        "KG1_PACKAGE_ALLOWED": "0",
        "KG1_KAGGLE_SUBMIT_ALLOWED": "0",
        "KG1_FINOPS_CANCEL_IF_NO_GAIN": "1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V514 CPU job.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    hardware = hardware_by_name.get(FLAVOR)
    if not hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if float(hardware["unit_cost_usd"]) > 0.001:
        raise RuntimeError(f"CPU flavor unit cost above gate: {hardware}")

    job_env = build_job_env(hardware)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_dataset_repo": OUTPUT_DATASET_REPO,
        "gates": {
            "cpu_only": True,
            "gpu_allowed": False,
            "train_allowed": False,
            "package_allowed": False,
            "kaggle_submit_allowed": False,
            "finops_cancel_if_no_gain": True,
        },
    }
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)

    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=2400,
            namespace=NAMESPACE,
        )
        manifest.update(
            {
                "mode": "launched",
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

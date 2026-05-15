#!/usr/bin/env python3
"""Launch V435C adapter raw-output probe on HF Jobs.

The job is inference-only and capped by family. It exists to collect real
V291/V290 checkpoint-6 raw outputs on V435B prompt-only rows, then uploads the
outputs for the next CPU gate. It does not train, score, package, or submit.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v435c_adapter_probe_raw_outputs"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = os.environ.get("KG1_V435C_IMAGE", "vllm/vllm-openai:v0.20.1").strip()
FLAVOR = os.environ.get("KG1_V435C_FLAVOR", "a100-large").strip()
DEFAULT_GPU_REGEX = "A100" if FLAVOR.startswith("a100") else ("H200" if FLAVOR.startswith("h200") else "")
DEFAULT_MIN_GPU_GIB = "79" if FLAVOR.startswith("a100") else ("130" if FLAVOR.startswith("h200") else "0")
REQUIRED_GPU_NAME_REGEX = os.environ.get("KG1_V435C_REQUIRED_GPU_NAME_REGEX", DEFAULT_GPU_REGEX).strip()
MIN_GPU_TOTAL_GIB = os.environ.get("KG1_V435C_MIN_GPU_TOTAL_GIB", DEFAULT_MIN_GPU_GIB).strip()
MAX_UNIT_COST_USD_PER_MIN = float(os.environ.get("KG1_V435C_MAX_UNIT_COST_USD_PER_MIN", "0.09"))
RUN_ID = "v435c-adapter-probe-raw-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = os.environ.get(
    "KG1_V435C_ADAPTER_REPO",
    "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
).strip()
ADAPTER_SUBFOLDER = os.environ.get("KG1_V435C_ADAPTER_SUBFOLDER", "checkpoint-6").strip()
OUTPUT_REPO = os.environ.get("KG1_V435C_OUTPUT_REPO", "felipesp1983/kg1-v435c-adapter-probe-raw-outputs").strip()
OUTPUT_REPO_TYPE = os.environ.get("KG1_V435C_OUTPUT_REPO_TYPE", "dataset").strip()
OUTPUT_PATH_IN_REPO = f"runs/{RUN_ID}"

MAX_EQUATION = int(os.environ.get("KG1_V435C_MAX_EQUATION", "200"))
MAX_BIT = int(os.environ.get("KG1_V435C_MAX_BIT", "80"))
MAX_NUM_SEQS = int(os.environ.get("KG1_V435C_MAX_NUM_SEQS", "32"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("KG1_V435C_GPU_MEMORY_UTILIZATION", "0.90"))
TIMEOUT_SECONDS = int(os.environ.get("KG1_V435C_TIMEOUT_SECONDS", "7200"))


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
PYBIN=$(command -v python || command -v python3)
echo "=== V435C HF JOB START ==="
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json, torch
try:
    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    vllm_version = repr(exc)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(json.dumps({
    "torch": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
    "vllm": vllm_version,
}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer
rm -rf /tmp/kg1
git init /tmp/kg1
cd /tmp/kg1
git remote add origin https://github.com/FELIPEACASTRO/KG1-NVIDIA.git
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT"
git checkout --detach FETCH_HEAD
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/run_v435c_adapter_probe_raw_outputs.py scripts/evaluate_lora_adapter.py src/competition_utils.py
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
OUTPUT_DIR="/tmp/kg1_v435c/${KG1_RUN_ID}"
mkdir -p "$OUTPUT_DIR"
$PYBIN scripts/run_v435c_adapter_probe_raw_outputs.py \
  --prompt-pack-jsonl artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_prompts.jsonl \
  --prompt-pack-manifest-json artifacts/v435b_adapter_probe_prompt_pack/20260515T_v435b_prompt_pack/v435b_adapter_probe_prompt_pack_manifest.json \
  --output-dir "$OUTPUT_DIR" \
  --label v435c_adapter_probe_raw_outputs \
  --adapter-repo "$KG1_ADAPTER_REPO" \
  --adapter-subfolder "$KG1_ADAPTER_SUBFOLDER" \
  --max-equation "$KG1_MAX_EQUATION" \
  --max-bit "$KG1_MAX_BIT" \
  --max-num-seqs "$KG1_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KG1_GPU_MEMORY_UTILIZATION" \
  --warmup-rows 2
$PYBIN - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
output_dir = Path(os.environ["KG1_OUTPUT_DIR"])
repo_id = os.environ["KG1_OUTPUT_REPO"]
repo_type = os.environ["KG1_OUTPUT_REPO_TYPE"]
path_in_repo = os.environ["KG1_OUTPUT_PATH_IN_REPO"]
api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
info = api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=str(output_dir),
    path_in_repo=path_in_repo,
    commit_message=f"Upload V435C adapter probe raw outputs {os.environ['KG1_RUN_ID']}",
)
print("hf_upload_info =", info, flush=True)
print("hf_output_url = https://huggingface.co/datasets/" + repo_id + "/tree/main/" + path_in_repo if repo_type == "dataset" else "hf_output_repo=" + repo_id, flush=True)
PY
echo "=== V435C HF JOB END ==="
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


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to launch V435C.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    selected_hardware = hardware[FLAVOR]
    if float(selected_hardware["unit_cost_usd"]) > MAX_UNIT_COST_USD_PER_MIN:
        raise RuntimeError(f"unit cost above V435C gate: {selected_hardware}")

    api.create_repo(repo_id=OUTPUT_REPO, repo_type=OUTPUT_REPO_TYPE, exist_ok=True)
    job_env = {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SUBFOLDER": ADAPTER_SUBFOLDER,
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_FLAVOR": FLAVOR,
        "KG1_GPU_MEMORY_UTILIZATION": str(GPU_MEMORY_UTILIZATION),
        "KG1_MAX_BIT": str(MAX_BIT),
        "KG1_MAX_EQUATION": str(MAX_EQUATION),
        "KG1_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
        "KG1_MIN_GPU_TOTAL_GIB": MIN_GPU_TOTAL_GIB,
        "KG1_OUTPUT_DIR": f"/tmp/kg1_v435c/{RUN_ID}",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_OUTPUT_REPO_TYPE": OUTPUT_REPO_TYPE,
        "KG1_REQUIRED_GPU_NAME_REGEX": REQUIRED_GPU_NAME_REGEX,
        "KG1_RUN_ID": RUN_ID,
    }
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(selected_hardware, indent=2, sort_keys=True), flush=True)
    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=TIMEOUT_SECONDS,
        namespace=NAMESPACE,
    )

    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": selected_hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapter_subfolder": ADAPTER_SUBFOLDER,
        "output_repo": OUTPUT_REPO,
        "output_repo_type": OUTPUT_REPO_TYPE,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "source_policy": {
            "training": False,
            "scoring": False,
            "submission": False,
            "answers_input": False,
            "purpose": "Collect raw outputs for CPU hard-negative gate.",
        },
        "finops_gate": {
            "max_unit_cost_usd_per_min": MAX_UNIT_COST_USD_PER_MIN,
            "timeout_seconds": TIMEOUT_SECONDS,
            "family_caps": {
                "equation_transform": MAX_EQUATION,
                "bit_manipulation": MAX_BIT,
            },
            "cancel_if": "Job fails hardware/load checks, stalls, or raw-output collection cannot feed V435.",
        },
        "next_action": "Monitor logs every 40 seconds, then rerun V435 with uploaded raw outputs.",
    }
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    print("hf_output_url =", f"https://huggingface.co/datasets/{OUTPUT_REPO}/tree/main/{OUTPUT_PATH_IN_REPO}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

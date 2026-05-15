#!/usr/bin/env python3
"""Launch V458 raw-output probe for the V457 numeric prompt pack.

This is inference-only. It collects V291/V290 checkpoint-6 adapter outputs on
the V457 prompt-only pack. It does not train, score, package, or submit.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v458_hf_v457_numeric_raw_probe"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

IMAGE = os.environ.get("KG1_V458_IMAGE", "vllm/vllm-openai:v0.20.1").strip()
FLAVOR = os.environ.get("KG1_V458_FLAVOR", "h200").strip()
MAX_UNIT_COST_USD_PER_MIN = float(os.environ.get("KG1_V458_MAX_UNIT_COST_USD_PER_MIN", "0.09"))
TIMEOUT_SECONDS = int(os.environ.get("KG1_V458_TIMEOUT_SECONDS", "3600"))
RUN_ID = "v458-v457-numeric-raw-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = os.environ.get(
    "KG1_V458_ADAPTER_REPO",
    "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
).strip()
ADAPTER_SUBFOLDER = os.environ.get("KG1_V458_ADAPTER_SUBFOLDER", "checkpoint-6").strip()
OUTPUT_REPO = os.environ.get("KG1_V458_OUTPUT_REPO", "felipesp1983/kg1-v458-v457-numeric-raw-probe").strip()
OUTPUT_REPO_TYPE = "dataset"
OUTPUT_PATH_IN_REPO = f"runs/{RUN_ID}"

PROMPT_PACK_JSONL = (
    "artifacts/v457_public_train_numeric_probe_pack/20260515T_cpu_gate/"
    "v457_public_train_numeric_probe_pack_prompts.jsonl"
)
PROMPT_PACK_MANIFEST = (
    "artifacts/v457_public_train_numeric_probe_pack/20260515T_cpu_gate/"
    "v457_public_train_numeric_probe_pack_manifest.json"
)

MAX_EQUATION = int(os.environ.get("KG1_V458_MAX_EQUATION", "32"))
MAX_BIT = int(os.environ.get("KG1_V458_MAX_BIT", "0"))
MAX_NUM_SEQS = int(os.environ.get("KG1_V458_MAX_NUM_SEQS", "16"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("KG1_V458_GPU_MEMORY_UTILIZATION", "0.90"))
REQUIRED_GPU_NAME_REGEX = os.environ.get("KG1_V458_REQUIRED_GPU_NAME_REGEX", "H200").strip()
MIN_GPU_TOTAL_GIB = float(os.environ.get("KG1_V458_MIN_GPU_TOTAL_GIB", "130"))


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
PYBIN=$(command -v python || command -v python3)
echo "=== V458 HF V457 NUMERIC RAW PROBE START ==="
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json, os, re, torch
try:
    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    vllm_version = repr(exc)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
payload = {
    "torch": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
    "vllm": vllm_version,
}
print(json.dumps(payload, sort_keys=True), flush=True)
pattern = os.environ["KG1_REQUIRED_GPU_NAME_REGEX"]
if pattern and not re.search(pattern, payload["gpu_name"], re.I):
    raise SystemExit(f"GPU name gate failed: pattern={pattern!r} gpu={payload['gpu_name']!r}")
if float(payload["gpu_total_gib"]) < float(os.environ["KG1_MIN_GPU_TOTAL_GIB"]):
    raise SystemExit(f"GPU memory gate failed: {payload['gpu_total_gib']} < {os.environ['KG1_MIN_GPU_TOTAL_GIB']}")
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
OUTPUT_DIR="/tmp/kg1_v458/${KG1_RUN_ID}"
mkdir -p "$OUTPUT_DIR"
$PYBIN scripts/run_v435c_adapter_probe_raw_outputs.py \
  --prompt-pack-jsonl "$KG1_PROMPT_PACK_JSONL" \
  --prompt-pack-manifest-json "$KG1_PROMPT_PACK_MANIFEST_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --label v458_v457_numeric_raw_probe \
  --adapter-repo "$KG1_ADAPTER_REPO" \
  --adapter-subfolder "$KG1_ADAPTER_SUBFOLDER" \
  --max-equation "$KG1_MAX_EQUATION" \
  --max-bit "$KG1_MAX_BIT" \
  --max-num-seqs "$KG1_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KG1_GPU_MEMORY_UTILIZATION" \
  --warmup-rows 1
$PYBIN - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
output_dir = Path(os.environ["KG1_OUTPUT_DIR"])
repo_id = os.environ["KG1_OUTPUT_REPO"]
path_in_repo = os.environ["KG1_OUTPUT_PATH_IN_REPO"]
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
info = api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(output_dir),
    path_in_repo=path_in_repo,
    commit_message=f"Upload V458 V457 numeric raw probe {os.environ['KG1_RUN_ID']}",
)
print("hf_upload_info =", info, flush=True)
print("hf_output_url = https://huggingface.co/datasets/" + repo_id + "/tree/main/" + path_in_repo, flush=True)
PY
echo "=== V458 HF V457 NUMERIC RAW PROBE END ==="
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
        raise RuntimeError("HF token is required to launch V458.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    selected_hardware = hardware[FLAVOR]
    if float(selected_hardware["unit_cost_usd"]) > MAX_UNIT_COST_USD_PER_MIN:
        raise RuntimeError(f"unit cost above V458 gate: {selected_hardware}")

    api.create_repo(repo_id=OUTPUT_REPO, repo_type=OUTPUT_REPO_TYPE, exist_ok=True)
    job_env = {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SUBFOLDER": ADAPTER_SUBFOLDER,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_GPU_MEMORY_UTILIZATION": str(GPU_MEMORY_UTILIZATION),
        "KG1_MAX_BIT": str(MAX_BIT),
        "KG1_MAX_EQUATION": str(MAX_EQUATION),
        "KG1_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
        "KG1_MIN_GPU_TOTAL_GIB": str(MIN_GPU_TOTAL_GIB),
        "KG1_OUTPUT_DIR": f"/tmp/kg1_v458/{RUN_ID}",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROMPT_PACK_JSONL": PROMPT_PACK_JSONL,
        "KG1_PROMPT_PACK_MANIFEST_JSON": PROMPT_PACK_MANIFEST,
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
        "prompt_pack_jsonl": PROMPT_PACK_JSONL,
        "prompt_pack_manifest_json": PROMPT_PACK_MANIFEST,
        "output_repo": OUTPUT_REPO,
        "output_repo_type": OUTPUT_REPO_TYPE,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "source_policy": {
            "training": False,
            "scoring": False,
            "submission": False,
            "answers_input": False,
            "purpose": "Collect raw outputs for V459 CPU hard-negative analysis.",
        },
        "finops_gate": {
            "max_unit_cost_usd_per_min": MAX_UNIT_COST_USD_PER_MIN,
            "timeout_seconds": TIMEOUT_SECONDS,
            "family_caps": {"equation_transform": MAX_EQUATION, "bit_manipulation": MAX_BIT},
            "cancel_if": "hardware/load failure, no progress, or raw-output collection cannot feed V459.",
        },
        "next_action": "Monitor logs every 40 seconds, then run V459 raw-output analysis.",
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

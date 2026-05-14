#!/usr/bin/env python3
"""Launch V387 official-like full eval for V382 checkpoint-4 on HF Jobs."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v387_official_like_v382_checkpoint4_full_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = os.environ.get("KG1_LAUNCH_FLAVOR", "h200").strip()
DEFAULT_GPU_REGEX = "H200" if FLAVOR.startswith("h200") else "A100"
DEFAULT_MIN_GPU_GIB = "130" if FLAVOR.startswith("h200") else "79"
REQUIRED_GPU_NAME_REGEX = os.environ.get("KG1_LAUNCH_REQUIRED_GPU_NAME_REGEX", DEFAULT_GPU_REGEX).strip()
MIN_GPU_TOTAL_GIB = os.environ.get("KG1_LAUNCH_MIN_GPU_TOTAL_GIB", DEFAULT_MIN_GPU_GIB).strip()
RUN_ID = f"v387-{FLAVOR}-officiallike-v382ckpt4-full947-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v382-v381-teacher-smoke"
ADAPTER_SUBFOLDERS = ["checkpoint-4"]
CANDIDATE_NAMES = ["v387_v382_checkpoint_4_official_like_full947"]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"

COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
python3 - <<'PY'
import json, torch
try:
    import vllm
    vllm_version = getattr(vllm, '__version__', 'unknown')
except Exception as exc:
    vllm_version = repr(exc)
print(json.dumps({
    'torch_before': getattr(torch, '__version__', 'unknown'),
    'cuda': getattr(torch.version, 'cuda', ''),
    'cuda_available': torch.cuda.is_available(),
    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',
    'vllm': vllm_version,
}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
python3 -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging hf_transfer
rm -rf /tmp/kg1
git init /tmp/kg1
cd /tmp/kg1
git remote add origin https://github.com/FELIPEACASTRO/KG1-NVIDIA.git
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT"
git checkout --detach FETCH_HEAD
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
python3 scripts/hf_job_official_like_eval_gate_v284.py
"""


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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
        raise RuntimeError("HF token is required to launch V387 official-like eval.")
    expected_commit = current_commit()
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.09:
        raise RuntimeError(f"HF unit cost above gate: {hardware[FLAVOR]}")

    specs = [
        {"name": name, "repo": ADAPTER_REPO, "subfolder": subfolder}
        for name, subfolder in zip(CANDIDATE_NAMES, ADAPTER_SUBFOLDERS, strict=True)
    ]
    job_env = {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_ADAPTER_SUBFOLDERS": ",".join(ADAPTER_SUBFOLDERS),
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CANDIDATE_NAMES": ",".join(CANDIDATE_NAMES),
        "KG1_DISABLE_THINKING": "0",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_EVAL_TIMEOUT_S": "7200",
        "KG1_EXPECTED_COMMIT": expected_commit,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_FULL_MAX_TRUNC": "4",
        "KG1_FULL_MIN_CANDIDATE": "824",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_LABEL_PREFIX": "v387_hf_official_like",
        "KG1_MIN_GPU_TOTAL_GIB": MIN_GPU_TOTAL_GIB,
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_OFFICIAL_LIKE_STRICT": "1",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PREDICTION_POSTPROCESSOR": "none",
        "KG1_REQUIRED_GPU_NAME_REGEX": REQUIRED_GPU_NAME_REGEX,
        "KG1_REQUIRE_CUDA": "1",
        "KG1_RUN_ID": RUN_ID,
        "KG1_UPLOAD_TO_HF": "1",
    }
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware[FLAVOR], indent=2, sort_keys=True), flush=True)
    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=9000,
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
        "hardware": hardware[FLAVOR],
        "expected_commit": expected_commit,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapter_subfolders": ADAPTER_SUBFOLDERS,
        "candidate_names": CANDIDATE_NAMES,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "full_eval_contract": {
            "rows": 947,
            "csv_sha256": "84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935",
            "max_tokens": 7680,
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.85,
            "thinking_enabled": True,
            "prediction_postprocessor": "none",
        },
        "promotion_gate": {
            "current_submit_correct": 823,
            "minimum_correct_for_ranking_attempt": 824,
            "full_candidate_gate": "correct>=824 and truncated<=4",
            "finops_note": "Package/submit only if this beats V291, not if it merely ties 823.",
        },
    }
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

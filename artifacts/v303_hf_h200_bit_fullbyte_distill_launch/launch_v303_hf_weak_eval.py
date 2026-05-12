#!/usr/bin/env python3
"""Launch V303 weak eval for all complete V303 checkpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v303_bit_fullbyte_distill_v221_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
EXPECTED_COMMIT = "d1d0281bfaf99d534163a7b0a5b0399e0f3bc9e3"
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v303-h200-v221contract-bit-fullbyte-distill-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v303-bit-fullbyte-distill-v290ckpt6"
REQUESTED_ADAPTERS = [
    ("checkpoint-3", "v303_checkpoint_3_v221_contract"),
    ("checkpoint-6", "v303_checkpoint_6_v221_contract"),
    ("checkpoint-9", "v303_checkpoint_9_v221_contract"),
    ("checkpoint-12", "v303_checkpoint_12_v221_contract"),
    ("final", "v303_final_v221_contract"),
]
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
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
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
python3 scripts/hf_job_weak_eval_v245.py
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
        raise RuntimeError("HF token is required to launch V303 weak eval.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware[FLAVOR]}")

    repo_files = set(api.list_repo_files(ADAPTER_REPO, repo_type="model"))
    available_adapters = [
        (subfolder, name)
        for subfolder, name in REQUESTED_ADAPTERS
        if f"{subfolder}/adapter_config.json" in repo_files and f"{subfolder}/adapter_model.safetensors" in repo_files
    ]
    missing_adapters = [
        subfolder
        for subfolder, _name in REQUESTED_ADAPTERS
        if f"{subfolder}/adapter_config.json" not in repo_files or f"{subfolder}/adapter_model.safetensors" not in repo_files
    ]
    if not available_adapters:
        raise RuntimeError(f"No complete V303 adapters found in {ADAPTER_REPO}: {sorted(repo_files)}")
    adapter_subfolders = [subfolder for subfolder, _name in available_adapters]
    candidate_names = [name for _subfolder, name in available_adapters]
    print("requested_adapter_subfolders =", [subfolder for subfolder, _name in REQUESTED_ADAPTERS], flush=True)
    print("available_adapter_subfolders =", adapter_subfolders, flush=True)
    print("missing_adapter_subfolders =", missing_adapters, flush=True)

    job_env = {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SUBFOLDERS": ",".join(adapter_subfolders),
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CANDIDATE_NAMES": ",".join(candidate_names),
        "KG1_DISABLE_THINKING": "0",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_EVAL_TIMEOUT_S": "4200",
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_LABEL_PREFIX": "v303_hf_weak",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
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
        timeout=5400,
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
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapter_subfolders": adapter_subfolders,
        "candidate_names": candidate_names,
        "requested_adapter_subfolders": [subfolder for subfolder, _name in REQUESTED_ADAPTERS],
        "missing_adapter_subfolders": missing_adapters,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "eval_contract": {
            "name": "V221 reproduced weak contract",
            "disable_thinking": False,
            "no_prompt_suffix": False,
            "max_tokens": 7680,
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "weak_csv_sha256": "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6",
            "shared_row_contract_sha256": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
        },
        "decision_gate_after_eval": {
            "minimum_total_correct": 192,
            "minimum_equation_transform_correct": 56,
            "minimum_bit_manipulation_correct": 136,
            "preferred_submit_threshold": {
                "minimum_total_correct": 193,
                "minimum_equation_transform_correct": 60,
                "minimum_bit_manipulation_correct": 137,
            },
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

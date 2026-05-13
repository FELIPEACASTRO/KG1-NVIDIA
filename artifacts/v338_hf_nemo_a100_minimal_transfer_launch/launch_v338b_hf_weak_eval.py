#!/usr/bin/env python3
"""Launch V338B V221-contract weak eval on Hugging Face H200.

This evaluates all complete V338B checkpoints currently uploaded. It is meant
to run once after the short A100 smoke train finishes, not repeatedly per
checkpoint, to keep HF spend bounded.
"""

from __future__ import annotations

import json
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v338b_minimal_transfer_balanced_v221_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v338b-h200-v221contract-minimal-transfer-balanced-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v338b-nemo-a100-minimal-transfer-balanced-v290ckpt6"
ADAPTERS = [
    ("checkpoint-2", "v338b_checkpoint_2_v221_contract"),
    ("checkpoint-4", "v338b_checkpoint_4_v221_contract"),
    ("checkpoint-6", "v338b_checkpoint_6_v221_contract"),
    ("checkpoint-8", "v338b_checkpoint_8_v221_contract"),
    ("checkpoint-10", "v338b_checkpoint_10_v221_contract"),
    ("checkpoint-12", "v338b_checkpoint_12_v221_contract"),
    ("checkpoint-14", "v338b_checkpoint_14_v221_contract"),
    ("final", "v338b_final_v221_contract"),
]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json, torch
try:
    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    vllm_version = repr(exc)
print(json.dumps({
    "torch_before": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "vllm": vllm_version,
}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
$PYBIN scripts/hf_job_weak_eval_v245.py
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


def adapter_exists(api: HfApi, repo_id: str, subfolder: str) -> bool:
    try:
        files = set(api.list_repo_files(repo_id, repo_type="model"))
    except Exception:
        return False
    prefix = f"{subfolder}/"
    required = {prefix + "adapter_config.json", prefix + "adapter_model.safetensors"}
    return required.issubset(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF H200 weak-eval job after debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to launch V338B weak eval.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware[FLAVOR]}")

    existing_adapters = [(subfolder, name) for subfolder, name in ADAPTERS if adapter_exists(api, ADAPTER_REPO, subfolder)]
    missing_adapters = [subfolder for subfolder, _ in ADAPTERS if subfolder not in {item[0] for item in existing_adapters}]
    if len(existing_adapters) < 2:
        raise RuntimeError(f"Too few complete V338B adapters found in {ADAPTER_REPO}. Existing={existing_adapters}; missing={missing_adapters}")

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in existing_adapters]
    job_env = {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware[FLAVOR]["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_RUN_ID": RUN_ID,
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_LABEL_PREFIX": "v338b_hf_weak",
        "KG1_DISABLE_THINKING": "0",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_PROMPT_SUFFIX": "\nReturn only one line: `\\boxed{answer}`. No reasoning. No explanation.",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_EVAL_TIMEOUT_S": "7200",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_EXPECTED_LORA_ALPHA": "32",
    }
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware[FLAVOR], indent=2, sort_keys=True), flush=True)

    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware[FLAVOR],
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapters": specs,
        "missing_optional_adapters": missing_adapters,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "promotion_gate": {
            "baseline_total": 192,
            "baseline_equation_transform": 56,
            "baseline_bit_manipulation": 136,
            "reject_if_total_lte": 192,
            "reject_if_equation_lte": 56,
            "reject_if_bit_lt": 136,
            "promote_if_total_gte": 193,
            "promote_if_equation_gte": 60,
            "promote_if_bit_gte": 136,
            "requires_full_eval_before_package_or_submit": True,
            "blocked_actions": ["package", "kaggle_submit"],
        },
    }
    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=7200,
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

#!/usr/bin/env python3
"""Launch/debug V383 weak eval for V382 checkpoints on Hugging Face H200.

V382 is a weighted teacher-transfer smoke train. Its loss is diagnostic only;
promotion must be decided by the V221 weak-family ACC contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v383_v382_checkpoint_sweep_v221_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v383-h200-v221contract-v382-checkpoint-sweep-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v382-v381-teacher-smoke"
REQUESTED_ADAPTERS = [
    ("checkpoint-2", "v382_v381_teacher_checkpoint_2_v221_contract"),
    ("checkpoint-4", "v382_v381_teacher_checkpoint_4_v221_contract"),
    ("checkpoint-6", "v382_v381_teacher_checkpoint_6_v221_contract"),
    ("checkpoint-8", "v382_v381_teacher_checkpoint_8_v221_contract"),
    ("checkpoint-10", "v382_v381_teacher_checkpoint_10_v221_contract"),
    ("final", "v382_v381_teacher_final_v221_contract"),
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
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(json.dumps({
    "torch_before": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": torch.cuda.is_available(),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
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
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    required = {prefix + "adapter_config.json", prefix + "adapter_model.safetensors"}
    return required.issubset(files)


def build_job_env(hardware: dict[str, object], specs: list[dict[str, str]]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_RUN_ID": RUN_ID,
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_LABEL_PREFIX": "v383_v382_hf_weak",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Create the paid HF H200 weak-eval job after debug gates pass.")
    parser.add_argument("--min-adapters", type=int, default=1, help="Minimum complete checkpoints required before launch.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V383 weak eval.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware_by_name:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    hardware = hardware_by_name[FLAVOR]
    if float(hardware["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware}")

    existing_adapters = [
        (subfolder, name)
        for subfolder, name in REQUESTED_ADAPTERS
        if adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    existing_subfolders = {subfolder for subfolder, _name in existing_adapters}
    missing_adapters = [subfolder for subfolder, _name in REQUESTED_ADAPTERS if subfolder not in existing_subfolders]
    if len(existing_adapters) < args.min_adapters:
        raise RuntimeError(
            f"Need at least {args.min_adapters} complete V382 adapters in {ADAPTER_REPO}. "
            f"Existing={existing_adapters}; missing={missing_adapters}"
        )

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in existing_adapters]
    job_env = build_job_env(hardware, specs)
    print("available_adapters =", json.dumps(specs, indent=2, sort_keys=True), flush=True)
    print("missing_adapters =", json.dumps(missing_adapters, indent=2, sort_keys=True), flush=True)
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)

    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapters": specs,
        "missing_optional_adapters": missing_adapters,
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
        "promotion_gate": {
            "baseline_total": 192,
            "baseline_equation_transform": 56,
            "baseline_bit_manipulation": 136,
            "reject_if_total_lte": 192,
            "reject_if_equation_lte": 56,
            "reject_if_bit_lt": 136,
            "promote_if_total_gte": 193,
            "promote_if_equation_gt": 56,
            "promote_if_bit_gte": 136,
            "preferred_submit_gate": {
                "total": 193,
                "equation_transform": 60,
                "bit_manipulation": 136,
            },
            "requires_full_eval_before_package_or_submit": True,
            "blocked_actions": ["package", "kaggle_submit"],
        },
        "finops_note": "Evaluate checkpoints only; no training. Stop this line if no checkpoint beats adapter-only baseline by weak ACC.",
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

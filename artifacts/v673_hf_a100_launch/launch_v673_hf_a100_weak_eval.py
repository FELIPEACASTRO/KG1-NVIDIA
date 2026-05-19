#!/usr/bin/env python3
"""Debug/launch V673 V221-contract weak eval on HF A100.

This is the promotion gate for the guarded V673 transfer checkpoints.  It uses
A100-large only, rejects H200 fallback, and evaluates only complete adapter
subfolders that are already present in the V673 output repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v673_a100_guarded_eqbit_v221_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

IMAGE = os.environ.get("KG1_V673_WEAK_EVAL_IMAGE", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel").strip()
FLAVOR = "a100-large"
MAX_UNIT_COST_USD = 0.05
MAX_TORCH_CUDA_MAJOR = 12
WEAK_EVAL_TIMEOUT_S = 2400
WEAK_GENERATION_TIMEOUT_S = 900
WEAK_HF_JOB_TIMEOUT_S = 4200
WEAK_MAX_TOKENS = 2048
VLLM_VERSION = "0.20.1"
VLLM_CUDA_FLAVOR = "cu129"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu129"
VLLM_WHEEL_URL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.20.1/"
    "vllm-0.20.1%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl"
)
RUN_ID = "v673-a100-v221contract-guarded-eqbit-weak-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6"
REQUESTED_ADAPTERS = [
    ("checkpoint-10", "v673_guarded_eqbit_checkpoint_10_v221_contract"),
    ("checkpoint-20", "v673_guarded_eqbit_checkpoint_20_v221_contract"),
    ("final", "v673_guarded_eqbit_final_v221_contract"),
]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json, os, sys, torch
try:
    import vllm
    vllm_version = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    vllm_version = repr(exc)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
cuda_runtime = str(getattr(torch.version, "cuda", "") or "")
cuda_major = int(cuda_runtime.split(".", 1)[0]) if cuda_runtime[:1].isdigit() else 0
max_cuda_major = int(os.environ.get("KG1_MAX_TORCH_CUDA_MAJOR", "0") or "0")
flavor = os.environ.get("KG1_HF_FLAVOR", "")
allow_cuda13_a100 = os.environ.get("KG1_ALLOW_CUDA13_ON_A100", "0").lower() in {"1", "true", "yes", "on"}
payload = {
    "torch_before": getattr(torch, "__version__", "unknown"),
    "cuda": cuda_runtime,
    "cuda_available": torch.cuda.is_available(),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
    "vllm": vllm_version,
    "flavor": flavor,
    "max_cuda_major": max_cuda_major,
    "allow_cuda13_a100": allow_cuda13_a100,
}
print("weak_eval_cuda_gate_start = " + json.dumps(payload, sort_keys=True), flush=True)
if max_cuda_major and cuda_major > max_cuda_major:
    print("weak_eval_cuda_gate_error = torch CUDA runtime exceeds gate before dependency install", flush=True)
    sys.exit(13)
if "a100" in flavor.lower() and cuda_major >= 13 and not allow_cuda13_a100:
    print("weak_eval_cuda_gate_error = CUDA13 runtime on A100 is blocked", flush=True)
    sys.exit(14)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip uninstall -y vllm >/dev/null 2>&1 || true
$PYBIN -m pip install -q --no-cache-dir --extra-index-url "$KG1_PYTORCH_CUDA_INDEX_URL" "$KG1_VLLM_WHEEL_URL"
$PYBIN - <<'PY'
import json, os, sys
try:
    import torch
    import vllm
    payload = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda": getattr(torch.version, "cuda", ""),
        "cuda_available": torch.cuda.is_available(),
        "vllm": getattr(vllm, "__version__", "unknown"),
    }
    print("weak_eval_vllm_import_preflight = " + json.dumps(payload, sort_keys=True), flush=True)
    cuda_runtime = str(payload["cuda"] or "")
    cuda_major = int(cuda_runtime.split(".", 1)[0]) if cuda_runtime[:1].isdigit() else 0
    max_cuda_major = int(os.environ.get("KG1_MAX_TORCH_CUDA_MAJOR", "0") or "0")
    if max_cuda_major and cuda_major > max_cuda_major:
        print("weak_eval_vllm_import_error = torch CUDA runtime exceeds gate after vLLM install", flush=True)
        sys.exit(15)
except Exception as exc:
    text = repr(exc)
    print("weak_eval_vllm_import_error = " + text, flush=True)
    if "libcudart.so.13" in text:
        sys.exit(16)
    sys.exit(17)
PY
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_weak_eval_v245.py scripts/evaluate_lora_adapters_batch.py scripts/kg1_weak_backfire_row_guard.py scripts/validate_answer_extraction_v1.py src/competition_utils.py
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def active_paid_jobs(api: HfApi) -> list[dict[str, str]]:
    active_stages = {"PENDING", "QUEUED", "RUNNING", "SCHEDULED", "STARTING"}
    active: list[dict[str, str]] = []
    for job in api.list_jobs(namespace=NAMESPACE):
        stage = str(getattr(getattr(job, "status", None), "stage", "") or "").upper()
        if stage not in active_stages:
            continue
        env = getattr(job, "environment", {}) or {}
        run_id = str(env.get("KG1_RUN_ID", ""))
        output_repo = str(env.get("KG1_OUTPUT_REPO", ""))
        if run_id.startswith("v") or "kg1-nemotron-lora" in output_repo:
            active.append(
                {
                    "id": str(getattr(job, "id", "")),
                    "stage": stage,
                    "run_id": run_id,
                    "output_repo": output_repo,
                    "url": str(getattr(job, "url", "")),
                }
            )
    return active


def runtime_image_gate() -> dict[str, Any]:
    known_bad_a100_cuda13_images = {"vllm/vllm-openai:v0.20.1"}
    known_good_a100_cuda12_images = {"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"}
    allow_known_bad = os.environ.get("KG1_V673_ALLOW_A100_CUDA13_IMAGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    gate = {
        "name": "a100_cuda12_weak_eval_runtime_gate",
        "image": IMAGE,
        "flavor": FLAVOR,
        "max_torch_cuda_major": MAX_TORCH_CUDA_MAJOR,
        "allow_cuda13_on_a100": False,
        "known_bad_a100_cuda13_images": sorted(known_bad_a100_cuda13_images),
        "known_good_a100_cuda12_images": sorted(known_good_a100_cuda12_images),
        "vllm_version": VLLM_VERSION,
        "vllm_cuda_flavor": VLLM_CUDA_FLAVOR,
        "vllm_wheel_url": VLLM_WHEEL_URL,
        "pytorch_cuda_index_url": PYTORCH_CUDA_INDEX_URL,
        "passed": True,
        "reason": "CUDA12-compatible PyTorch runtime selected with official vLLM 0.20.1 cu129 wheel.",
    }
    if FLAVOR.startswith("a100") and IMAGE in known_bad_a100_cuda13_images and not allow_known_bad:
        gate.update(
            {
                "passed": False,
                "reason": (
                    "Blocked known-bad A100 weak-eval runtime. "
                    "vllm/vllm-openai:v0.20.1 exposed torch CUDA 13 on HF A100, "
                    "while the observed driver only supports CUDA 12.09."
                ),
            }
        )
    return gate


def build_job_env(hardware: dict[str, object], specs: list[dict[str, str]]) -> dict[str, str]:
    return {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_ALLOW_CUDA13_ON_A100": "0",
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CATASTROPHIC_EVAL_GUARD": "1",
        "KG1_DISABLE_THINKING": "1",
        "KG1_ENFORCE_WEAK_RUNTIME_POLICY": "1",
        "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
        "KG1_EVAL_TIMEOUT_S": str(WEAK_EVAL_TIMEOUT_S),
        "KG1_EXPECTED_ADAPTER_BASE_MODEL_NAME_OR_PATH": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": str(MAX_UNIT_COST_USD),
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_LABEL_PREFIX": "v673_hf_weak",
        "KG1_GENERATION_TIMEOUT_S": str(WEAK_GENERATION_TIMEOUT_S),
        "KG1_LLM_INIT_TIMEOUT_S": "1800",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "8",
        "KG1_MAX_TOKENS": str(WEAK_MAX_TOKENS),
        "KG1_MAX_TORCH_CUDA_MAJOR": str(MAX_TORCH_CUDA_MAJOR),
        "KG1_MIN_GPU_TOTAL_GIB": "70",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROTECTED_BASELINE_CSV": "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv",
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
        "KG1_PROTECTED_ROW_GUARD": "1",
        "KG1_PYTORCH_CUDA_INDEX_URL": PYTORCH_CUDA_INDEX_URL,
        "KG1_PROMPT_SUFFIX": "\nReturn only one line: `\\boxed{answer}`. No reasoning. No explanation.",
        "KG1_REQUIRE_DISABLE_THINKING": "1",
        "KG1_REQUIRE_CUDA": "1",
        "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
        "KG1_RUN_ID": RUN_ID,
        "KG1_STOP_AFTER_CONSECUTIVE_FAILED_CANDIDATES": "2",
        "KG1_UPLOAD_INCREMENTAL_EVAL_DIAGNOSTICS": "1",
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_VLLM_CUDA_FLAVOR": VLLM_CUDA_FLAVOR,
        "KG1_VLLM_GPU_MEMORY_UTILIZATION": "0.90",
        "KG1_VLLM_VERSION": VLLM_VERSION,
        "KG1_VLLM_WHEEL_URL": VLLM_WHEEL_URL,
        "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "512",
        "KG1_WEAK_PROMOTE_BIT_MIN": "136",
        "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": "1.0",
        "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
        "KG1_WEAK_PROMOTE_LABEL_AWARE_DELTA_MAX": "0",
        "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "2048",
        "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": "0",
        "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
        "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    }


def validate_weak_runtime_policy(job_env: dict[str, str]) -> dict[str, Any]:
    max_tokens = int(job_env["KG1_MAX_TOKENS"])
    generation_timeout_s = int(job_env["KG1_GENERATION_TIMEOUT_S"])
    eval_timeout_s = int(job_env["KG1_EVAL_TIMEOUT_S"])
    promote_max_completion = int(job_env["KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX"])
    disable_thinking = job_env["KG1_DISABLE_THINKING"] == "1"
    required = {
        "KG1_ENFORCE_WEAK_RUNTIME_POLICY": "1",
        "KG1_REQUIRE_DISABLE_THINKING": "1",
        "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
        "KG1_PROTECTED_ROW_GUARD": "1",
        "KG1_CATASTROPHIC_EVAL_GUARD": "1",
    }
    mismatched = {
        key: {"expected": value, "observed": job_env.get(key)}
        for key, value in required.items()
        if job_env.get(key) != value
    }
    gate = {
        "name": "v673_weak_runtime_policy_gate",
        "passed": True,
        "max_tokens": max_tokens,
        "generation_timeout_s": generation_timeout_s,
        "eval_timeout_s": eval_timeout_s,
        "promote_max_completion_tokens": promote_max_completion,
        "disable_thinking": disable_thinking,
        "required_env": required,
        "mismatched_env": mismatched,
        "reason": "Weak eval is bounded so runaway generations cannot consume A100 budget or create non-promotable results.",
    }
    blockers: list[str] = []
    if mismatched:
        blockers.append("required_env_mismatch")
    if max_tokens != WEAK_MAX_TOKENS or max_tokens > promote_max_completion:
        blockers.append("max_tokens_exceeds_promotion_completion_gate")
    if generation_timeout_s != WEAK_GENERATION_TIMEOUT_S or generation_timeout_s <= 0:
        blockers.append("generation_timeout_not_bounded")
    if eval_timeout_s != WEAK_EVAL_TIMEOUT_S or eval_timeout_s <= 0:
        blockers.append("eval_timeout_not_bounded")
    if not disable_thinking:
        blockers.append("disable_thinking_required_for_v673_weak_eval")
    if blockers:
        gate["passed"] = False
        gate["blockers"] = blockers
        raise RuntimeError("Weak runtime policy gate blocked launch: " + json.dumps(gate, sort_keys=True))
    gate["blockers"] = []
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Create the paid HF A100 weak-eval job after debug gates pass.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V673 weak eval.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware_by_name:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available. Available={sorted(hardware_by_name)}")
    hardware = hardware_by_name[FLAVOR]
    if float(hardware["unit_cost_usd"]) > MAX_UNIT_COST_USD:
        raise RuntimeError(f"A100-large unit cost above gate: {hardware}")

    existing_adapters = [
        (subfolder, name)
        for subfolder, name in REQUESTED_ADAPTERS
        if adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    missing_adapters = [
        subfolder
        for subfolder, _name in REQUESTED_ADAPTERS
        if subfolder not in {item[0] for item in existing_adapters}
    ]
    if not existing_adapters:
        raise RuntimeError(f"No complete V673 adapters found in {ADAPTER_REPO}. Missing={missing_adapters}")
    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in existing_adapters]
    job_env = build_job_env(hardware, specs)
    weak_runtime_policy_gate = validate_weak_runtime_policy(job_env)
    active = active_paid_jobs(api)
    if args.launch and active:
        raise RuntimeError("Active paid KG1 jobs block V673 weak eval launch: " + json.dumps(active, sort_keys=True))

    serialized = COMMAND_SCRIPT + "\n" + json.dumps(job_env, sort_keys=True)
    runtime_gate = runtime_image_gate()
    required_snippets = [
        "scripts/hf_job_weak_eval_v245.py",
        "scripts/evaluate_lora_adapters_batch.py",
        "KG1_REQUIRED_GPU_NAME_REGEX",
        "KG1_MAX_TORCH_CUDA_MAJOR",
        "KG1_ALLOW_CUDA13_ON_A100",
        "KG1_VLLM_WHEEL_URL",
        "A100",
        "KG1_WEAK_PROMOTE_EQUATION_MIN",
        "KG1_WEAK_PROMOTE_TOTAL_MIN",
        "KG1_ENFORCE_WEAK_RUNTIME_POLICY",
        "KG1_PROTECTED_ID_ANSWERS",
        "KG1_PROTECTED_ROW_GUARD",
        "KG1_GENERATION_TIMEOUT_S",
        "KG1_MAX_TOKENS",
        "KG1_DISABLE_THINKING",
        ADAPTER_REPO,
    ]
    missing_snippets = [item for item in required_snippets if item not in serialized]
    if missing_snippets:
        raise RuntimeError("Weak-eval launcher missing required snippets: " + json.dumps(missing_snippets))

    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
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
        "active_job_blockers": active,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "promotion_gate": {
            "baseline_total": 192,
            "baseline_equation_transform": 56,
            "baseline_bit_manipulation": 136,
            "promote_if_total_gte": 196,
            "promote_if_equation_gte": 60,
            "promote_if_bit_gte": 136,
            "reject_if_truncated_gt": 0,
            "reject_if_label_aware_delta_gt": 0,
            "reject_if_no_box_fallback_gt": 0,
            "reject_if_boxed_rate_lt": 1.0,
            "requires_full_eval_before_package_or_submit": True,
            "blocked_actions": ["package", "kaggle_submit", "h200_fallback"],
            "runaway_cost_guard": {
                "max_tokens": WEAK_MAX_TOKENS,
                "generation_timeout_s": WEAK_GENERATION_TIMEOUT_S,
                "disable_thinking": True,
                "reason": (
                    "Weak promotion rejects max_completion_tokens >2048 and any truncation; "
                    "therefore 7680-token runaway generations are non-promotable and should fail fast."
                ),
            },
        },
        "runtime_image_gate": runtime_gate,
        "weak_runtime_policy_gate": weak_runtime_policy_gate,
        "finops_policy": "A100-large only; do not launch while another paid KG1 job is active.",
        "hf_job_timeout_s": WEAK_HF_JOB_TIMEOUT_S,
    }
    if args.launch:
        if not runtime_gate["passed"]:
            raise RuntimeError("Runtime image gate blocked launch: " + json.dumps(runtime_gate, sort_keys=True))
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=WEAK_HF_JOB_TIMEOUT_S,
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
    out_path = out_dir / f"{RUN_ID}_weak_eval_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    remote_command_path = out_dir / f"{RUN_ID}_weak_eval_remote_command.sh"
    remote_command_path.write_text(COMMAND_SCRIPT + "\n", encoding="utf-8")
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)
    print("launch_manifest_path =", out_path, flush=True)
    print("remote_command_path =", remote_command_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

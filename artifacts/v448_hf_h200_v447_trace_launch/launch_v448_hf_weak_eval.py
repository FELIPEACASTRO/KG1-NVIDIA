#!/usr/bin/env python3
"""Launch/debug V448 V221-contract weak eval on HF H200.

Default mode is local debug only. Pass ``--launch`` only after this file is
committed and pushed. The eval is adapter-only, uses the V245/V221 weak bridge,
does not enable any prediction postprocessor, and uploads reports back to the
V448 model repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v448_v447_clean_trace_v221_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
HF_JOB_TIMEOUT_SECONDS = 3600
RUN_ID = "v448-h200-v221contract-cleantrace-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v448-nemo-h200-v447-clean-trace-v290ckpt6"
REQUESTED_ADAPTERS = [
    ("checkpoint-3", "v448_clean_trace_checkpoint_3_v221_contract"),
    ("checkpoint-6", "v448_clean_trace_checkpoint_6_v221_contract"),
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
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py scripts/evaluate_lora_adapters_batch.py src/competition_utils.py
$PYBIN scripts/hf_job_preflight_gate.py --phase eval-preinstall
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging safetensors hf_transfer
$PYBIN scripts/hf_job_preflight_gate.py --phase eval-postinstall
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
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    prefix = f"{subfolder}/"
    return {prefix + "adapter_config.json", prefix + "adapter_model.safetensors"}.issubset(files)


def build_env(selected: dict[str, object], specs: list[dict[str, str]]) -> dict[str, str]:
    return {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_DISABLE_THINKING": "0",
        "KG1_EVAL_TIMEOUT_S": str(HF_JOB_TIMEOUT_SECONDS),
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_HF_UNIT_COST_USD": str(selected["unit_cost_usd"]),
        "KG1_LABEL_PREFIX": "v448_hf_weak",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROMPT_SUFFIX": "\nReturn only one line: `\\boxed{answer}`. No reasoning. No explanation.",
        "KG1_REQUIRE_CUDA": "1",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_RUN_ID": RUN_ID,
        "KG1_UPLOAD_TO_HF": "1",
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_weak_eval_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF eval job after debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V448 weak eval.")
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    selected = hardware[FLAVOR]
    if float(selected["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {selected}")

    available_adapters = [
        (subfolder, name)
        for subfolder, name in REQUESTED_ADAPTERS
        if adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    missing_adapters = [
        subfolder
        for subfolder, _name in REQUESTED_ADAPTERS
        if subfolder not in {item[0] for item in available_adapters}
    ]
    if not available_adapters:
        raise RuntimeError(f"No complete V448 adapters found in {ADAPTER_REPO}. Missing={missing_adapters}")
    if missing_adapters:
        print("missing_optional_adapters =", json.dumps(missing_adapters, sort_keys=True), flush=True)

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in available_adapters]
    job_env = build_env(selected, specs)
    print("=== V448 WEAK EVAL DEBUG START ===", flush=True)
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)
    print("adapter_specs =", json.dumps(specs, indent=2, sort_keys=True), flush=True)
    print("=== V448 WEAK EVAL DEBUG END ===", flush=True)

    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hf_job_timeout_seconds": HF_JOB_TIMEOUT_SECONDS,
        "hardware": selected,
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
            "reject_if_truncated_gt": 0,
            "full_eval_only_if": "total>192 and equation>56 and bit>=136 and truncated=0",
            "submit_only_if_full_official_like_gte": 824,
        },
        "version_comparison_artifact": "artifacts/v448_hf_h200_v447_trace_launch/V448_VS_PREVIOUS.md",
        "previous_version": "V448 H200 clean trace SFT train",
        "job_env": job_env,
    }
    if not args.launch:
        write_manifest({**manifest, "mode": "debug_only_no_job_launched", "next_action": "Commit/push, then rerun with --launch."})
        return 0

    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=HF_JOB_TIMEOUT_SECONDS,
        namespace=NAMESPACE,
    )
    launched_manifest = {
        **manifest,
        "mode": "launched",
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
        "next_action": "Monitor every 40 seconds; cancel only if HF reports fatal error or timeout risk.",
    }
    write_manifest(launched_manifest)
    print("job_url =", launched_manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

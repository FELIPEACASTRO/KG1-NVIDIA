#!/usr/bin/env python3
"""Launch V653 official-like weak eval for checkpoint-2.

This is the first post-checkpoint gate for the V653 compact-trace output-policy
route.  It evaluates adapter-only predictions with the current label-free weak
contract and blocks promotion on any protected-row, truncation, boxed-format,
or no-box fallback regression.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token


VERSION = "v653_checkpoint2_strict_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
).strip()

IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v653-h200-compacttrace-checkpoint2-weak-" + datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v653-h200-compacttrace-outputpolicy-v290ckpt6"
ADAPTER_SUBFOLDERS = ["checkpoint-2"]
CANDIDATE_NAMES = ["v653_checkpoint_2_compacttrace_outputpolicy"]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"

BASELINE_TOTAL = 192
BASELINE_BIT = 136
BASELINE_EQUATION = 56
PROMOTION_TOTAL_MIN = 196
PROMOTION_EQUATION_MIN = 60


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export PYTHONUTF8=1
export PYTHONCOERCECLOCALE=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LC_CTYPE=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=C.UTF-8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export TERM=dumb
export NO_COLOR=1
export VLLM_LOGGING_LEVEL=WARNING
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
    'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
    'vllm': vllm_version,
}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
python3 -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
python3 -m py_compile scripts/hf_job_weak_eval_v245.py scripts/evaluate_lora_adapters_batch.py scripts/kg1_weak_backfire_row_guard.py scripts/validate_answer_extraction_v1.py src/competition_utils.py
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DEBUG=1
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


def assert_adapters_complete(api: HfApi) -> None:
    files = set(api.list_repo_files(ADAPTER_REPO, repo_type="model"))
    missing: list[str] = []
    for subfolder in ADAPTER_SUBFOLDERS:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = f"{subfolder}/{name}"
            if path not in files:
                missing.append(path)
    if missing:
        raise RuntimeError("V653 weak eval target is incomplete: " + json.dumps(missing, sort_keys=True))


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to launch V653 weak eval.")

    api = HfApi(token=token)
    assert_adapters_complete(api)

    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if float(hardware[FLAVOR]["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware[FLAVOR]}")

    job_env = {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SUBFOLDERS": ",".join(ADAPTER_SUBFOLDERS),
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CANDIDATE_NAMES": ",".join(CANDIDATE_NAMES),
        "KG1_CATASTROPHIC_EVAL_GUARD": "1",
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
        "KG1_DISABLE_THINKING": "0",
        "KG1_ENFORCE_WEAK_PROMOTION_GATE": "1",
        "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
        "KG1_EVAL_TIMEOUT_S": "4200",
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
        "KG1_EXPECTED_WEAK_CSV_SHA256": "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6",
        "KG1_LABEL_PREFIX": "v653_hf_weak",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
        "KG1_PROTECTED_ROW_GUARD": "1",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_REQUIRE_CUDA": "1",
        "KG1_RUN_ID": RUN_ID,
        "KG1_STOP_AFTER_CONSECUTIVE_FAILED_CANDIDATES": "1",
        "KG1_UPLOAD_INCREMENTAL_EVAL_DIAGNOSTICS": "1",
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_VLLM_ENFORCE_EAGER": "1",
        "KG1_WEAK_PROMOTE_BIT_MIN": str(BASELINE_BIT),
        "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": "1.0",
        "KG1_WEAK_PROMOTE_EQUATION_MIN": str(PROMOTION_EQUATION_MIN),
        "KG1_WEAK_PROMOTE_LABEL_AWARE_DELTA_MAX": "0",
        "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": "0",
        "KG1_WEAK_PROMOTE_TOTAL_MIN": str(PROMOTION_TOTAL_MIN),
        "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    }

    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware[FLAVOR], indent=2, sort_keys=True), flush=True)
    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=3600,
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
        "adapter_subfolders": ADAPTER_SUBFOLDERS,
        "candidate_names": CANDIDATE_NAMES,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "decision_gate_after_eval": {
            "baseline_submit_safe_total": BASELINE_TOTAL,
            "baseline_submit_safe_bit": BASELINE_BIT,
            "baseline_submit_safe_equation": BASELINE_EQUATION,
            "minimum_total_correct": PROMOTION_TOTAL_MIN,
            "minimum_bit_manipulation_correct": BASELINE_BIT,
            "minimum_equation_transform_correct": PROMOTION_EQUATION_MIN,
            "maximum_truncated": 0,
            "minimum_boxed_rate": 1.0,
            "maximum_no_box_fallback_rows": 0,
            "note": "Submit remains blocked unless label-free weak gate passes and protected rows hold.",
        },
    }
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_weak_eval_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

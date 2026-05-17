#!/usr/bin/env python3
"""Launch V574 official-like weak eval for V573 checkpoint-2.

This launcher is measurement-first. It evaluates the V573 checkpoint with the
official prompt suffix, thinking enabled, and 7680 max tokens. The job uploads
diagnostics even when the strict promotion gate fails, because after the long
plateau the row-level diff is more valuable than another silent fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download


VERSION = "v574_v573_checkpoint2_official_like_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v574-h200-officiallike-v573-checkpoint2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v573-v571bit-v551eq-refmix-v290ckpt6"
REQUESTED_ADAPTERS = [
    ("checkpoint-2", "v574_v573_checkpoint_2_official_like"),
]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"

SCRIPT_PATCH_REPO = "felipesp1983/kg1-v547-contract-aligned-distillation-artifacts"
SCRIPT_PATCH_ROOT = "v574-eval-script-patch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SCRIPT_PATCH_FILES = [
    "scripts/hf_job_weak_eval_v245.py",
    "scripts/evaluate_lora_adapters_batch.py",
    "scripts/hf_job_preflight_gate.py",
    "scripts/kg1_weak_backfire_row_guard.py",
    "scripts/kg1_workspace_clean_gate.py",
]

OFFICIAL_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export HF_HUB_DISABLE_PROGRESS_BARS=1
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
apt-get update -qq && apt-get install -y -qq git locales >/dev/null
locale-gen C.UTF-8 >/dev/null || true
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging safetensors hf_transfer
$PYBIN - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo = os.environ["KG1_SCRIPT_PATCH_REPO"]
root = os.environ["KG1_SCRIPT_PATCH_ROOT"].strip("/")
expected = json.loads(os.environ["KG1_SCRIPT_PATCH_SHA256_JSON"])
for rel, expected_sha in expected.items():
    remote_name = f"{root}/{rel}"
    src = Path(hf_hub_download(repo_id=repo, filename=remote_name, repo_type="dataset", token=os.environ.get("HF_TOKEN")))
    observed_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if observed_sha != expected_sha:
        raise SystemExit(f"script patch sha mismatch for {rel}: expected={expected_sha} observed={observed_sha}")
    dst = Path(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"script_patch_applied {rel} sha256={observed_sha}", flush=True)
PY
$PYBIN -m py_compile scripts/hf_job_weak_eval_v245.py scripts/hf_job_preflight_gate.py scripts/evaluate_lora_adapters_batch.py scripts/kg1_workspace_clean_gate.py src/competition_utils.py
$PYBIN scripts/kg1_workspace_clean_gate.py --delete-safe
$PYBIN scripts/hf_job_preflight_gate.py --phase eval-preinstall
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


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def upload_script_patch(api: HfApi) -> dict[str, str]:
    patch_dir = Path(__file__).resolve().parent / "_v574_script_patch"
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for rel in SCRIPT_PATCH_FILES:
        src = REPO_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = patch_dir / SCRIPT_PATCH_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[rel] = sha256_path(src)
    api.upload_folder(
        repo_id=SCRIPT_PATCH_REPO,
        repo_type="dataset",
        folder_path=str(patch_dir / SCRIPT_PATCH_ROOT),
        path_in_repo=SCRIPT_PATCH_ROOT,
        commit_message=f"Add {RUN_ID} eval script patch",
    )
    print("script_patch_repo =", SCRIPT_PATCH_REPO, flush=True)
    print("script_patch_root =", SCRIPT_PATCH_ROOT, flush=True)
    print("script_patch_hashes =", json.dumps(hashes, indent=2, sort_keys=True), flush=True)
    return hashes


def verify_uploaded_script_patch(hashes: dict[str, str], token: str) -> str:
    commit = ""
    for rel, expected_sha in hashes.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=SCRIPT_PATCH_REPO,
                filename=f"{SCRIPT_PATCH_ROOT}/{rel}",
                repo_type="dataset",
                token=token,
            )
        )
        observed_sha = sha256_path(downloaded)
        if observed_sha != expected_sha:
            raise RuntimeError(f"Uploaded script patch mismatch for {rel}: {observed_sha} != {expected_sha}")
        commit = downloaded.parents[2].name
    return commit


def build_env(hardware: dict[str, dict[str, object]], script_patch_hashes: dict[str, str]) -> dict[str, str]:
    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in REQUESTED_ADAPTERS]
    return {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CATASTROPHIC_EVAL_GUARD": "1",
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
        "KG1_DISABLE_THINKING": "0",
        "KG1_ENFORCE_WEAK_PROMOTION_GATE": "0",
        "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
        "KG1_EVAL_TIMEOUT_S": "3600",
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_HF_UNIT_COST_USD": str(hardware[FLAVOR]["unit_cost_usd"]),
        "KG1_LABEL_PREFIX": "v574_hf_weak",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROMPT_SUFFIX": OFFICIAL_PROMPT_SUFFIX,
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101",
        "KG1_PROTECTED_ROW_GUARD": "1",
        "KG1_REQUIRE_CUDA": "1",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_RUN_ID": RUN_ID,
        "KG1_SCRIPT_PATCH_REPO": SCRIPT_PATCH_REPO,
        "KG1_SCRIPT_PATCH_ROOT": SCRIPT_PATCH_ROOT,
        "KG1_SCRIPT_PATCH_SHA256_JSON": json.dumps(script_patch_hashes, sort_keys=True),
        "KG1_STOP_AFTER_CONSECUTIVE_FAILED_CANDIDATES": "0",
        "KG1_UPLOAD_INCREMENTAL_EVAL_DIAGNOSTICS": "1",
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_WEAK_EVAL_DIAGNOSTIC_ONLY": "1",
        "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "5000",
        "KG1_WEAK_PROMOTE_BIT_MIN": "136",
        "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
        "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "7680",
        "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
        "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token and not args.dry_run:
        raise RuntimeError("HF token is required to launch V574 weak eval.")
    api = HfApi(token=token or None)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if hardware[FLAVOR]["unit_cost_usd"] > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware[FLAVOR]}")

    missing_adapters = [
        subfolder for subfolder, _name in REQUESTED_ADAPTERS if not adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    if missing_adapters:
        raise RuntimeError(f"Missing required V573 adapters in {ADAPTER_REPO}: {missing_adapters}")

    script_patch_hashes = {rel: sha256_path(REPO_ROOT / rel) for rel in SCRIPT_PATCH_FILES}
    script_patch_commit = "dry-run"
    if not args.dry_run:
        script_patch_hashes = upload_script_patch(api)
        script_patch_commit = verify_uploaded_script_patch(script_patch_hashes, token or "")

    job_env = build_env(hardware, script_patch_hashes)
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware[FLAVOR], indent=2, sort_keys=True), flush=True)

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in REQUESTED_ADAPTERS]
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": None,
        "job_url": None,
        "job_status": "dry_run" if args.dry_run else "created",
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware[FLAVOR],
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapters": specs,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "script_patch_repo": SCRIPT_PATCH_REPO,
        "script_patch_root": SCRIPT_PATCH_ROOT,
        "script_patch_commit": script_patch_commit,
        "script_patch_hashes": script_patch_hashes,
        "eval_contract": {
            "label_free": True,
            "official_prompt_suffix": True,
            "disable_thinking": False,
            "max_tokens": 7680,
            "diagnostic_upload_even_if_gate_fails": True,
        },
        "promotion_gate": {
            "baseline_total": 192,
            "baseline_equation_transform": 56,
            "baseline_bit_manipulation": 136,
            "promote_if_total_gte": 196,
            "promote_if_equation_gte": 60,
            "promote_if_bit_gte": 136,
            "reject_if_truncated_gt": 0,
            "protected_rows_required": ["8740ed31=01101000", "59bee375=10010101"],
            "submit_only_after_full_eval": True,
        },
    }

    if not args.dry_run:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=3600,
            namespace=NAMESPACE,
        )
        manifest["job_id"] = job.id
        manifest["job_url"] = f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}"
        manifest["job_status"] = str(job.status.stage if getattr(job, "status", None) else "unknown")

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

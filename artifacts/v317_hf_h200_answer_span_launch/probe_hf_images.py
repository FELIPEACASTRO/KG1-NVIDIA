#!/usr/bin/env python3
"""Probe candidate HF job images for Nemotron 3 Nano dependencies.

The probe is intentionally cheap: it uses small hardware first and only checks
imports/install feasibility. It does not download the 30B model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, get_token


NAMESPACE = "felipesp1983"
FLAVOR = "a10g-small"
TIMEOUT_S = 900
OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_IMAGES = [
    {
        "name": "nemo_nemotron3_nano_official",
        "image": "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano",
        "reason": "NVIDIA official Nemotron 3 Nano training container.",
    },
    {
        "name": "vllm_openai_0201",
        "image": "vllm/vllm-openai:v0.20.1",
        "reason": "Known to run V221 weak eval; may contain model runtime kernels.",
    },
    {
        "name": "vllm_openai_latest",
        "image": "vllm/vllm-openai:latest",
        "reason": "Newest official vLLM image may have newer Nemotron/Mamba support.",
    },
    {
        "name": "pytorch_280_cuda128",
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        "reason": "Current baseline image; included for control.",
    },
]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

COMMAND_SCRIPT = r"""set -eux
PYBIN=$(command -v python || command -v python3)
echo "probe_start=$(date -u +%FT%TZ)"
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import importlib
import json
import subprocess
import sys
import time

def mod_status(name):
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "ok")}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

mods = {
    name: mod_status(name)
    for name in [
        "torch",
        "transformers",
        "peft",
        "causal_conv1d",
        "mamba_ssm",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.selective_scan_interface",
    ]
}
runtime = {}
try:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    runtime = {
        "torch": getattr(torch, "__version__", ""),
        "cuda": getattr(torch.version, "cuda", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": props.name if props else "",
        "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
    }
except Exception as exc:
    runtime = {"torch_error": repr(exc)}

print("initial_probe=" + json.dumps({"runtime": runtime, "mods": mods}, sort_keys=True), flush=True)

missing_core = [
    name for name in ["transformers", "peft", "causal_conv1d", "mamba_ssm"]
    if not mods.get(name, {}).get("ok")
]
if missing_core:
    start = time.time()
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-cache-dir",
        "--prefer-binary",
        "transformers>=4.56.0",
        "peft>=0.17.0",
        "accelerate>=1.10.0",
        "causal-conv1d==1.6.1",
        "mamba-ssm==2.3.1",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=480)
        install = {
            "returncode": proc.returncode,
            "elapsed_s": round(time.time() - start, 3),
            "tail": proc.stdout[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        install = {
            "returncode": 124,
            "elapsed_s": round(time.time() - start, 3),
            "tail": str(exc),
        }
else:
    install = {"returncode": 0, "elapsed_s": 0.0, "tail": "all core deps already import"}

mods_after = {
    name: mod_status(name)
    for name in [
        "torch",
        "transformers",
        "peft",
        "causal_conv1d",
        "mamba_ssm",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.selective_scan_interface",
    ]
}
print("install_probe=" + json.dumps(install, sort_keys=True), flush=True)
print("final_probe=" + json.dumps({"mods": mods_after}, sort_keys=True), flush=True)

core_ok = (
    mods_after["transformers"]["ok"]
    and mods_after["peft"]["ok"]
    and mods_after["causal_conv1d"]["ok"]
    and mods_after["mamba_ssm"]["ok"]
)
if not core_ok:
    raise SystemExit(23)
PY
echo "probe_end=$(date -u +%FT%TZ)"
"""


def main() -> int:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required.")
    api = HfApi(token=token)
    expected_commit = git_head()
    jobs = []
    for candidate in CANDIDATE_IMAGES:
        job = api.run_job(
            image=candidate["image"],
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            flavor=FLAVOR,
            timeout=TIMEOUT_S,
            namespace=NAMESPACE,
            secrets={"HF_TOKEN": token},
        )
        item = {
            **candidate,
            "job_id": job.id,
            "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
            "stage": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            "expected_commit": expected_commit,
        }
        jobs.append(item)
        print("launched_probe =", json.dumps(item, sort_keys=True), flush=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_commit": expected_commit,
        "flavor": FLAVOR,
        "timeout_s": TIMEOUT_S,
        "jobs": jobs,
        "selection_rule": (
            "Use the cheapest image that imports torch/transformers/peft/"
            "causal_conv1d/mamba_ssm or installs them within 8 minutes."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("hf_image_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("probe_manifest_path =", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

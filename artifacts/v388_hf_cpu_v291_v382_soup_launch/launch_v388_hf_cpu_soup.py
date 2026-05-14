#!/usr/bin/env python3
"""Launch V388 CPU-only HF adapter soup job.

This job builds a tiny set of submit-compatible LoRA adapter soups from two
already measured adapter-only candidates:

- V291/V290 checkpoint-6: best submitted full baseline, 823/947.
- V382 checkpoint-4: weak-only candidate with different bit movement, but full
  eval tied V291.

The job does not evaluate, package, or submit. It only uploads adapter-only
soups for a later weak gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v388_hf_cpu_v291_v382_soup"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "python:3.11"
FLAVOR = "cpu-upgrade"
RUN_ID = "v388-cpu-v291-v382-soup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v388-v291-v382-soups"
INPUTS = [
    {
        "name": "v291_ckpt6",
        "repo": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
        "subfolder": "checkpoint-6",
    },
    {
        "name": "v382_ckpt4",
        "repo": "felipesp1983/kg1-nemotron-lora-v382-v381-teacher-smoke",
        "subfolder": "checkpoint-4",
    },
]
RECIPES = [
    {
        "name": "soup_v291_095_v382_005",
        "primary": "v291_ckpt6",
        "weights": {"v291_ckpt6": 0.95, "v382_ckpt4": 0.05},
    },
    {
        "name": "soup_v291_090_v382_010",
        "primary": "v291_ckpt6",
        "weights": {"v291_ckpt6": 0.90, "v382_ckpt4": 0.10},
    },
    {
        "name": "soup_v291_105_v382_neg005",
        "primary": "v291_ckpt6",
        "weights": {"v291_ckpt6": 1.05, "v382_ckpt4": -0.05},
    },
]


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
python - <<'PY'
import json, platform
print(json.dumps({"python": platform.python_version(), "platform": platform.platform()}, sort_keys=True), flush=True)
PY
apt-get update -qq && apt-get install -y -qq git >/dev/null
python -m pip install -q --no-cache-dir --upgrade pip
python -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' safetensors hf_transfer torch numpy
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
python -m py_compile scripts/run_v262_adapter_soup_hf.py
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_v262_adapter_soup_hf.py
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


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_REQUIRE_REPO_COMMIT": "1",
        "KG1_RUN_ID": RUN_ID,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_OUTPUT_PRIVATE": "1",
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_SOUP_INPUTS_JSON": json.dumps(INPUTS, sort_keys=True),
        "KG1_SOUP_RECIPES_JSON": json.dumps(RECIPES, sort_keys=True),
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.001",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V388.")
    api = HfApi(token=token)
    hardware_by_name = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    hardware = hardware_by_name.get(FLAVOR)
    if not hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    if float(hardware["unit_cost_usd"]) > 0.001:
        raise RuntimeError(f"CPU flavor unit cost above gate: {hardware}")

    job_env = build_job_env(hardware)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_repo": OUTPUT_REPO,
        "inputs": INPUTS,
        "recipes": RECIPES,
        "gates": {
            "cpu_only": True,
            "train_allowed": False,
            "eval_allowed": False,
            "package_allowed": False,
            "kaggle_submit_allowed": False,
            "next_gpu_allowed_only_after_soup_upload": True,
        },
        "promotion_gate": {
            "weak_eval_required": True,
            "promote_to_full_only_if_total_gt": 192,
            "promote_to_full_only_if_equation_gt": 56,
            "promote_to_full_only_if_bit_gte": 136,
            "promote_to_full_only_if_truncated_eq": 0,
        },
        "finops_note": "CPU-upgrade soup job avoids local disk/CPU pressure; GPU is used only for later weak eval if upload succeeds.",
    }

    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)

    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=3600,
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

#!/usr/bin/env python3
"""Launch/debug V362 checkpoint-1 V221-contract weak eval on HF H200.

This is the FinOps gate for the V361 boxed-only bit-transfer smoke.  It
evaluates checkpoint-1 only.  Checkpoint-2/final evaluation is intentionally
blocked until checkpoint-1 shows adapter-only weak ACC gain without family
regression.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V344_LAUNCHER = REPO_ROOT / "artifacts/v344_hf_a100_preference_abstain_launch/launch_v344_hf_weak_eval_checkpoint2.py"
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

VERSION = "v362_v361_boxed_only_checkpoint1_v221_weak_eval"
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v362-h200-v221contract-v361-boxed-only-cp1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6"
ADAPTERS = [
    ("checkpoint-1", "v362_v361_boxed_only_checkpoint_1_v221_contract"),
]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"


def load_v344_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v344_weak_eval_launcher", V344_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load weak-eval template from {V344_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "KG1_LABEL_PREFIX": "v362_v361_boxed_only_hf_weak",
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
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V362 weak eval.")
    template = load_v344_launcher()
    api = HfApi(token=token)
    hardware_by_name = {item.name: template.hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware_by_name:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    hardware = hardware_by_name[FLAVOR]
    if float(hardware["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {hardware}")

    existing_adapters = [(subfolder, name) for subfolder, name in ADAPTERS if template.adapter_exists(api, ADAPTER_REPO, subfolder)]
    missing_adapters = [subfolder for subfolder, _ in ADAPTERS if subfolder not in {item[0] for item in existing_adapters}]
    if len(existing_adapters) != 1:
        raise RuntimeError(
            f"Expected checkpoint-1 in {ADAPTER_REPO} before V362 weak eval. "
            f"Existing={existing_adapters}; missing={missing_adapters}"
        )

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in existing_adapters]
    job_env = build_job_env(hardware, specs)
    serialized = template.COMMAND_SCRIPT + "\n" + json.dumps(job_env, sort_keys=True)
    required = [
        "scripts/hf_job_weak_eval_v245.py",
        "v362_v361_boxed_only_checkpoint_1_v221_contract",
        ADAPTER_REPO,
    ]
    missing_required = [item for item in required if item not in serialized]
    if missing_required:
        raise RuntimeError("V362 weak eval launcher missing required snippets: " + json.dumps(missing_required))

    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)

    manifest: dict[str, object] = {
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
        "missing_adapters": missing_adapters,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "promotion_gate": {
            "baseline_total": 192,
            "baseline_equation_transform": 56,
            "baseline_bit_manipulation": 136,
            "reject_if_total_lte": 192,
            "reject_if_equation_lt": 56,
            "reject_if_bit_lt": 136,
            "reject_if_truncated_gt": 0,
            "promote_if_total_gte": 193,
            "promote_if_equation_gte": 56,
            "promote_if_bit_gte": 136,
            "requires_full_eval_before_package_or_submit": True,
            "blocked_actions": ["checkpoint_2_eval_without_checkpoint_1_gain", "package", "kaggle_submit"],
        },
        "finops_decision": "Evaluate checkpoint-1 only. Do not launch checkpoint-2/final weak eval unless checkpoint-1 beats adapter-only gate.",
    }
    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", template.COMMAND_SCRIPT],
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

    out_path = RUN_DIR / f"{RUN_ID}_weak_eval_checkpoint1_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

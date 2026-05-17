#!/usr/bin/env python3
"""Launch/debug V537 V221-contract weak eval for V536 checkpoints.

V537 does not train. It evaluates the V536 checkpoint-2 and
checkpoint-abort-step4 adapters that already exist in the HF output repo. The
launcher is debug-only by default; pass --launch to create the paid HF job.
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


VERSION = "v537_v536_checkpoint2_step4_weak_eval"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "vllm/vllm-openai:v0.20.1"
FLAVOR = "h200"
RUN_ID = "v537-h200-v221contract-v536-cp2-step4-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v536-nemo-h200-v534bit-v523eq-v290ckpt6"
REQUESTED_ADAPTERS = [
    ("checkpoint-2", "v537_v536_checkpoint_2_v221_contract"),
    ("checkpoint-abort-step4", "v537_v536_checkpoint_abort_step4_v221_contract"),
]
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"


def load_v494_base() -> Any:
    base_path = REPO_ROOT / "artifacts" / "v494_hf_h200_v493_weak_eval_launch" / "launch_v494_hf_weak_eval_v493_checkpoint2.py"
    spec = importlib.util.spec_from_file_location("kg1_v494_weak_eval_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V494 weak eval base from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(__file__).resolve())
    return module


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in REQUESTED_ADAPTERS]
    return {
        "KG1_ADAPTER_REPO": ADAPTER_REPO,
        "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
        "KG1_DISABLE_THINKING": "0",
        "KG1_ENFORCE_WEAK_PROMOTION_GATE": "1",
        "KG1_EVAL_TIMEOUT_S": "7200",
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_LORA_ALPHA": "32",
        "KG1_EXPECTED_LORA_R": "32",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_LABEL_PREFIX": "v537_hf_weak",
        "KG1_MAX_MODEL_LEN": "8192",
        "KG1_MAX_NUM_SEQS": "64",
        "KG1_MAX_TOKENS": "7680",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_MODEL_NAME": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "KG1_NO_PROMPT_SUFFIX": "0",
        "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000",
        "KG1_PROTECTED_ROW_GUARD": "1",
        "KG1_PROMPT_SUFFIX": "\nReturn only one line: `\\boxed{answer}`. No reasoning. No explanation.",
        "KG1_REQUIRE_CUDA": "1",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_RUN_ID": RUN_ID,
        "KG1_UPLOAD_TO_HF": "1",
        "KG1_WEAK_PROMOTE_BIT_MIN": "136",
        "KG1_WEAK_PROMOTE_EQUATION_MIN": "57",
        "KG1_WEAK_PROMOTE_TOTAL_MIN": "193",
        "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    }


def build_manifest(
    *,
    mode: str,
    hardware: dict[str, object],
    job_env: dict[str, str],
    job: Any | None = None,
) -> dict[str, Any]:
    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in REQUESTED_ADAPTERS]
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapters": specs,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "job_env": job_env,
        "promotion_gate": {
            "current_label_free_baseline_total": 191,
            "current_label_free_baseline_equation_transform": 55,
            "current_label_free_baseline_bit_manipulation": 136,
            "legacy_stored_baseline_total": 192,
            "legacy_stored_baseline_equation_transform": 56,
            "legacy_stored_baseline_bit_manipulation": 136,
            "reject_if_total_lt": 193,
            "reject_if_equation_lt": 57,
            "reject_if_bit_lt": 136,
            "reject_if_truncated_gt": 0,
            "full_eval_only_if": "total>=193 and equation>=57 and bit>=136 and truncated=0",
            "submit_only_if_full_official_like_gt": 823,
        },
        "version_comparison": {
            "previous": "V536 H200 training smoke failed only on VRAM kill-switch after uploading checkpoints",
            "current": "V537 weak ACC measurement of existing V536 checkpoint-2 and checkpoint-abort-step4",
            "expected_decision": "no relaunch/no package/no submit unless weak promotion gate passes label-free",
        },
        "version_comparison_artifact": "artifacts/version_diffs/V537_VS_V536.md",
        "next_action": "Monitor every 40 seconds if launched; stop after weak gate decision.",
    }
    if job is not None:
        manifest.update(
            {
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )
    return manifest


def write_manifest(manifest: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    if "job_url" in manifest:
        print("job_url =", manifest["job_url"], flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Create the paid HF H200 weak-eval job.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V537 weak eval debug/launch.")
    base = load_v494_base()
    api = HfApi(token=token)
    hardware = {item.name: base.hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    selected = hardware[FLAVOR]
    if float(selected["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {selected}")

    missing_adapters = [
        subfolder for subfolder, _name in REQUESTED_ADAPTERS if not base.adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    if missing_adapters:
        raise RuntimeError(f"Missing required V536 adapters in {ADAPTER_REPO}: {missing_adapters}")

    command_script = base.COMMAND_SCRIPT.replace("V494", "V537").replace("v494", "v537")
    job_env = build_job_env(selected)
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)
    print("requested_adapters =", json.dumps(REQUESTED_ADAPTERS, indent=2), flush=True)
    print("command_script_static_debug = ok", flush=True)

    if not args.launch:
        write_manifest(build_manifest(mode="debug_only_no_job_launched", hardware=selected, job_env=job_env))
        print("launch_skipped = pass --launch to create the paid HF job", flush=True)
        return 0

    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", command_script],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=7200,
        namespace=NAMESPACE,
    )
    write_manifest(build_manifest(mode="launched", hardware=selected, job_env=job_env, job=job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

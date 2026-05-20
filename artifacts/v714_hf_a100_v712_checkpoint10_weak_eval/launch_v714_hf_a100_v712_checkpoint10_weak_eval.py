#!/usr/bin/env python3
"""Launch guarded V714 weak-eval-only check for V712 checkpoint-10.

This job does not train, package, run full eval, or submit to Kaggle. It only
measures whether the already-uploaded V712 checkpoint-10 produces real weak ACC
under the V245 label-free evaluation contract after the scripts import fix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = ROOT / "artifacts" / "v680_hf_a100_launch" / "launch_v680_hf_a100_weak_eval.py"
VERSION = "v714_a100_v712_checkpoint10_weak_eval_only"
NAMESPACE = "felipesp1983"
ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal"
REQUESTED_ADAPTERS = [
    ("checkpoint-10", "v714_v712_checkpoint_10_best_loss"),
]
RUN_ID = "v714-a100-v712-ckpt10-weak-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_REPO = ADAPTER_REPO
OUTPUT_PATH_IN_REPO = f"evals/{RUN_ID}"

STATIC_OFFICIAL_LIKE_CONTRACT = {
    "KG1_DISABLE_THINKING": "0",
    "KG1_REQUIRE_DISABLE_THINKING": "0",
    "KG1_NO_PROMPT_SUFFIX": "0",
    "KG1_MAX_TOKENS": "7680",
    "KG1_MAX_MODEL_LEN": "8192",
    "KG1_MAX_NUM_SEQS": "64",
    "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
    "KG1_WEAK_PROMOTE_BIT_MIN": "136",
    "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
    "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": "1.0",
    "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": "0",
    "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "512",
    "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "7680",
    "KG1_PROTECTED_ROW_GUARD": "1",
    "KG1_STOP_ON_PROTECTED_BACKFIRE": "1",
    "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
    "KG1_ENFORCE_WEAK_RUNTIME_POLICY": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_base_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v680_weak_eval_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base weak-eval launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.ADAPTER_REPO = ADAPTER_REPO
    module.REQUESTED_ADAPTERS = REQUESTED_ADAPTERS
    module.OUTPUT_REPO = OUTPUT_REPO
    module.OUTPUT_PATH_IN_REPO = OUTPUT_PATH_IN_REPO
    module.RUN_ID = RUN_ID
    return module


def build_retry_command(base_command: str) -> str:
    """Add bounded retries around network-sensitive pip installs."""

    vllm_install = (
        '$PYBIN -m pip install -q --no-cache-dir --extra-index-url '
        '"$KG1_PYTORCH_CUDA_INDEX_URL" "$KG1_VLLM_WHEEL_URL"'
    )
    vllm_retry = """for attempt in 1 2 3; do
  echo "vllm_wheel_install_attempt=$attempt"
  if $PYBIN -m pip install -q --no-cache-dir --extra-index-url "$KG1_PYTORCH_CUDA_INDEX_URL" "$KG1_VLLM_WHEEL_URL"; then
    break
  fi
  rc=$?
  if [ "$attempt" = "3" ]; then
    echo "vllm_wheel_install_failed_rc=$rc"
    exit "$rc"
  fi
  sleep $((attempt * 20))
done"""
    helper_install = "$PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer"
    helper_retry = """for attempt in 1 2 3; do
  echo "weak_eval_helper_install_attempt=$attempt"
  if $PYBIN -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' pandas packaging peft safetensors hf_transfer; then
    break
  fi
  rc=$?
  if [ "$attempt" = "3" ]; then
    echo "weak_eval_helper_install_failed_rc=$rc"
    exit "$rc"
  fi
  sleep $((attempt * 15))
done"""
    patched = base_command.replace(vllm_install, vllm_retry)
    patched = patched.replace(helper_install, helper_retry)
    if patched == base_command:
        raise RuntimeError("failed to inject pip retry guards into weak-eval command")
    return patched


def build_v714_job_env(base: Any, hardware: dict[str, object], specs: list[dict[str, str]]) -> dict[str, str]:
    env = base.build_job_env(hardware, specs)
    env.update(
        {
            "KG1_ADAPTER_REPO": ADAPTER_REPO,
            "KG1_ADAPTER_SPECS_JSON": json.dumps(specs, sort_keys=True),
            "KG1_LABEL_PREFIX": "v714_v712_checkpoint10_hf_weak",
            "KG1_OUTPUT_REPO": OUTPUT_REPO,
            "KG1_OUTPUT_PATH_IN_REPO": OUTPUT_PATH_IN_REPO,
            "KG1_RUN_ID": RUN_ID,
            "KG1_REQUIRE_DISABLE_THINKING": "0",
            "KG1_WEAK_EVAL_DIAGNOSTIC_ONLY": "0",
            "KG1_ENFORCE_WEAK_PROMOTION_GATE": "1",
            "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
            "KG1_WEAK_PROMOTE_BIT_MIN": "136",
            "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
            "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
            "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": "1.0",
            "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": "0",
            "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "512",
            "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "7680",
        }
    )
    return env


def validate_static_contract(job_env: dict[str, str]) -> dict[str, Any]:
    mismatched = {
        key: {"expected": expected, "observed": job_env.get(key)}
        for key, expected in STATIC_OFFICIAL_LIKE_CONTRACT.items()
        if job_env.get(key) != expected
    }
    payload = {
        "name": "v714_checkpoint10_static_official_like_contract",
        "expected": STATIC_OFFICIAL_LIKE_CONTRACT,
        "mismatched": mismatched,
        "passed": not mismatched,
    }
    if mismatched:
        raise RuntimeError("V714 weak-eval static contract mismatch: " + json.dumps(payload, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Create the paid HF A100 weak-eval-only job after gates pass.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V714 checkpoint-10 weak eval.")
    api = HfApi(token=token)
    base = load_base_launcher()

    hardware_by_name = {item.name: base.hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if base.FLAVOR != "a100-large":
        raise RuntimeError(f"V714 weak eval is A100-large only, got {base.FLAVOR!r}")
    if base.FLAVOR not in hardware_by_name:
        raise RuntimeError(f"HF flavor {base.FLAVOR!r} is not available. Available={sorted(hardware_by_name)}")
    hardware = hardware_by_name[base.FLAVOR]
    if float(hardware["unit_cost_usd"]) > base.MAX_UNIT_COST_USD:
        raise RuntimeError(f"A100-large unit cost above gate: {hardware}")

    existing_adapters = [
        (subfolder, name)
        for subfolder, name in REQUESTED_ADAPTERS
        if base.adapter_exists(api, ADAPTER_REPO, subfolder)
    ]
    missing_adapters = [
        subfolder
        for subfolder, _name in REQUESTED_ADAPTERS
        if subfolder not in {item[0] for item in existing_adapters}
    ]
    if not existing_adapters:
        raise RuntimeError(f"No complete V712 checkpoint-10 adapter found in {ADAPTER_REPO}. Missing={missing_adapters}")

    specs = [{"repo": ADAPTER_REPO, "subfolder": subfolder, "name": name} for subfolder, name in existing_adapters]
    job_env = build_v714_job_env(base, hardware, specs)
    static_official_like_contract = validate_static_contract(job_env)
    weak_runtime_policy_gate = base.validate_weak_runtime_policy(job_env)
    active = base.active_paid_jobs(api)
    if args.launch and active:
        raise RuntimeError("Active paid KG1 jobs block V714 checkpoint-10 weak eval launch: " + json.dumps(active, sort_keys=True))

    command_script = build_retry_command(base.COMMAND_SCRIPT)
    serialized = command_script + "\n" + json.dumps(job_env, sort_keys=True)
    required_snippets = [
        "scripts/hf_job_weak_eval_v245.py",
        "scripts/evaluate_lora_adapter.py",
        "scripts/evaluate_lora_adapters_batch.py",
        "scripts_package_gate",
        "weak_eval_import_gate_ok",
        "KG1_REQUIRED_GPU_NAME_REGEX",
        "KG1_MAX_TORCH_CUDA_MAJOR",
        "KG1_VLLM_WHEEL_URL",
        "A100",
        "KG1_WEAK_PROMOTE_EQUATION_MIN",
        "KG1_WEAK_PROMOTE_TOTAL_MIN",
        "KG1_WEAK_PROMOTE_BIT_MIN",
        "KG1_WEAK_PROMOTE_TRUNC_MAX",
        "KG1_WEAK_PROMOTE_BOXED_RATE_MIN",
        "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX",
        "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX",
        "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX",
        "KG1_ENFORCE_WEAK_RUNTIME_POLICY",
        "KG1_ENFORCE_WEAK_PROMOTION_GATE",
        "KG1_PROTECTED_ID_ANSWERS",
        "KG1_PROTECTED_ROW_GUARD",
        "KG1_CATASTROPHIC_BASELINE_TOTAL_CORRECT",
        "KG1_CATASTROPHIC_BASELINE_BIT_CORRECT",
        "KG1_CATASTROPHIC_BASELINE_EQUATION_CORRECT",
        "KG1_STOP_ON_PROTECTED_BACKFIRE",
        "KG1_GENERATION_TIMEOUT_S",
        "KG1_MAX_TOKENS",
        "KG1_DISABLE_THINKING",
        "KG1_REQUIRE_DISABLE_THINKING",
        "KG1_MAX_NUM_SEQS",
        ADAPTER_REPO,
        "checkpoint-10",
    ]
    missing_snippets = [item for item in required_snippets if item not in serialized]
    if missing_snippets:
        raise RuntimeError("Weak-eval launcher missing required snippets: " + json.dumps(missing_snippets))

    runtime_gate = base.runtime_image_gate()
    manifest: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "mode": "debug_only_no_job_launched",
        "job_id": "",
        "job_url": "",
        "job_status": "not_launched",
        "namespace": NAMESPACE,
        "image": base.IMAGE,
        "flavor": base.FLAVOR,
        "hardware": hardware,
        "expected_commit": base.EXPECTED_COMMIT,
        "branch": base.REPO_BRANCH,
        "run_id": RUN_ID,
        "adapter_repo": ADAPTER_REPO,
        "adapters": specs,
        "missing_required_adapters": missing_adapters,
        "active_job_blockers": active,
        "output_repo": OUTPUT_REPO,
        "output_path_in_repo": OUTPUT_PATH_IN_REPO,
        "promotion_gate": {
            "baseline_total": 196,
            "baseline_equation_transform": 60,
            "baseline_bit_manipulation": 136,
            "promote_if_total_gte": 196,
            "promote_if_equation_gte": 60,
            "promote_if_bit_gte": 136,
            "reject_if_truncated_gt": 0,
            "reject_if_label_aware_delta_gt": 0,
            "reject_if_no_box_fallback_gt": 0,
            "reject_if_boxed_rate_lt": 1.0,
            "requires_full_eval_before_package_or_submit": True,
            "blocked_actions": ["train", "package", "kaggle_submit", "h200_fallback"],
            "policy": (
                "V714 is inference-only. Any truncation, protected-row backfire, label-aware-only gain, "
                "missing boxed output, or family regression is non-promotable."
            ),
        },
        "runtime_image_gate": runtime_gate,
        "static_official_like_contract": static_official_like_contract,
        "weak_runtime_policy_gate": weak_runtime_policy_gate,
        "finops_policy": "A100-large only; evaluate V712 checkpoint-10 only; do not launch while another paid KG1 job is active.",
        "hf_job_timeout_s": base.WEAK_HF_JOB_TIMEOUT_S,
    }
    if args.launch:
        if not runtime_gate["passed"]:
            raise RuntimeError("Runtime image gate blocked launch: " + json.dumps(runtime_gate, sort_keys=True))
        job = api.run_job(
            image=base.IMAGE,
            command=["/bin/bash", "-lc", command_script],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=base.FLAVOR,
            timeout=base.WEAK_HF_JOB_TIMEOUT_S,
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
    remote_command_path = out_dir / f"{RUN_ID}_weak_eval_remote_command.sh"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    remote_command_path.write_text(command_script + "\n", encoding="utf-8")
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("hf_hardware_selected =", json.dumps(hardware, indent=2, sort_keys=True), flush=True)
    print("launch_manifest_path =", out_path, flush=True)
    print("remote_command_path =", remote_command_path, flush=True)
    print("job_url =", manifest["job_url"] or "not_launched_debug_only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

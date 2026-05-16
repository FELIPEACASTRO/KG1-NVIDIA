#!/usr/bin/env python3
"""Launch/debug V480 V479 objective-aligned smoke train on HF H200.

V480 is the correction after V478 found that V476 made bit only 0.9492% of the
effective weighted objective. This launcher requires the V478 objective
alignment gate before any paid job and uses equal weights for the V479 dataset.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = (
    REPO_ROOT
    / "artifacts/v391_hf_nemo_h200_equation_bit_replay_launch/"
    / "launch_v391_hf_nemo_h200_equation_bit_replay.py"
)
DATASET_MANIFEST = (
    REPO_ROOT
    / "artifacts/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter/"
    / "v479_objective_aligned_filter_manifest.json"
)
OBJECTIVE_ALIGNMENT_GATE = REPO_ROOT / "scripts/audit_v478_training_objective_alignment.py"
HF_JOB_TIMEOUT_SECONDS = 3600


def load_base_module():
    spec = importlib.util.spec_from_file_location("v391_base_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v480_v479_objective_aligned_from_v290_checkpoint6_nemo_h200"
base.RUN_ID = "v480-nemo-h200-v479-objective-aligned-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base.DATASET_UPLOAD_COMMIT = "https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/dbb97ac737969a69794d2e93895049a8ccbb8eb5"
base.TRAIN_FILE = "data/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter/v479_objective_aligned_filter_train.jsonl"
base.VAL_FILE = "data/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter/v479_objective_aligned_filter_val.jsonl"
base.TRAIN_SHA256 = "d7236e27f6dc437441217434f40216128e4412e52ff03e523eeb36775c927971"
base.VAL_SHA256 = "c1124008f967a94ce380694201648bb09c3a73d5873581995763be7c604b62b6"
base.TRAIN_ROWS = 992
base.VAL_ROWS = 248
base.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v480-v479-objective-aligned-v290ckpt6"
base.MAX_STEPS = 8
base.SAVE_EVERY_STEPS = 2
base.EVAL_EVERY_STEPS = 2
base.EVAL_MAX_EXAMPLES = 96
base.ANSWER_SPAN_LOSS_WEIGHT = "12.0"
base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1000"
base.SOURCE_WEIGHTS = "v475_v325_equation_no_loss_distill=1.00,v475_v217_bit_replay_guardrail=1.00"
base.SUBCATEGORY_WEIGHTS = (
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_signed=1.00,"
    "bit_guardrail_replay=1.00"
)
base.COMMAND_SCRIPT = (
    base.COMMAND_SCRIPT
    .replace("export OUTPUT_DIR='/tmp/kg1_v391_output'", "export OUTPUT_DIR='/tmp/kg1_v480_output'")
    .replace("export MAX_STEPS=12", "export MAX_STEPS=8")
    .replace("export LEARNING_RATE=4.0e-8", "export LEARNING_RATE=4.0e-8")
    .replace("export FINAL_LEARNING_RATE=1.0e-8", "export FINAL_LEARNING_RATE=1.0e-8")
)

_base_build_job_env = base.build_job_env
_base_local_debug = base.local_debug


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    env = _base_build_job_env(hardware)
    env.update(
        {
            "KG1_HF_MAX_UNIT_COST_USD": "0.09",
            "KG1_EXPECTED_MAX_STEPS": "8",
            "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
                "bit_guardrail_replay,"
                "equation_numeric_add_direct,equation_numeric_colon_trailing_zero,"
                "equation_numeric_minus_signed"
            ),
            "KG1_REQUIRED_VAL_SUBCATEGORIES": (
                "bit_guardrail_replay,"
                "equation_numeric_add_direct,equation_numeric_colon_trailing_zero,"
                "equation_numeric_minus_signed"
            ),
            "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        }
    )
    return env


def run_objective_alignment_gate() -> None:
    output_json = Path(__file__).resolve().parent / "v480_objective_alignment_debug.json"
    cmd = [
        "python",
        str(OBJECTIVE_ALIGNMENT_GATE),
        "--dataset-manifest-json",
        str(DATASET_MANIFEST),
        "--source-weights",
        base.SOURCE_WEIGHTS,
        "--subcategory-weights",
        base.SUBCATEGORY_WEIGHTS,
        "--output-json",
        str(output_json),
        "--enforce",
    ]
    print("objective_alignment_cmd =", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print("objective_alignment_gate_json =", output_json, flush=True)


def local_debug(api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str]]:
    run_objective_alignment_gate()
    selected, job_env = _base_local_debug(api, token)
    forbidden_snippets = [
        "data/v390_equation_bit_replay_mix",
        "data/v475_equation_bit_replay_mix",
        "v464_" + "v463_numeric_multirule_dataset",
        "v468_" + "v464_symbol_fix_dataset",
        "timeout=5400",
        "export MAX_STEPS=12",
        "v475_v325_equation_no_loss_distill=8.00",
    ]
    found_forbidden = [item for item in forbidden_snippets if item in base.COMMAND_SCRIPT or item in json.dumps(job_env)]
    if found_forbidden:
        raise RuntimeError("V480 command/env contains stale forbidden snippets: " + json.dumps(found_forbidden))
    required_command_snippets = [
        "export MAX_STEPS=8",
        "export OUTPUT_DIR='/tmp/kg1_v480_output'",
        "export SOURCE_WEIGHTS=\"$KG1_SOURCE_WEIGHTS\"",
        "export SUBCATEGORY_WEIGHTS=\"$KG1_SUBCATEGORY_WEIGHTS\"",
    ]
    missing_command_snippets = [item for item in required_command_snippets if item not in base.COMMAND_SCRIPT]
    if missing_command_snippets:
        raise RuntimeError(
            "V480 command missing required launch snippets: " + json.dumps(missing_command_snippets)
        )
    required_env = {
        "KG1_TRAIN_FILE": base.TRAIN_FILE,
        "KG1_VAL_FILE": base.VAL_FILE,
        "KG1_TRAIN_SHA": base.TRAIN_SHA256,
        "KG1_VAL_SHA": base.VAL_SHA256,
        "KG1_TRAIN_ROWS": str(base.TRAIN_ROWS),
        "KG1_VAL_ROWS": str(base.VAL_ROWS),
        "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        "KG1_SOURCE_WEIGHTS": base.SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": base.SUBCATEGORY_WEIGHTS,
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
            "bit_guardrail_replay,"
            "equation_numeric_add_direct,equation_numeric_colon_trailing_zero,"
            "equation_numeric_minus_signed"
        ),
    }
    mismatched = {
        key: {"observed": job_env.get(key), "expected": expected}
        for key, expected in required_env.items()
        if job_env.get(key) != expected
    }
    if mismatched:
        raise RuntimeError("V480 job env mismatch: " + json.dumps(mismatched, sort_keys=True))
    print("v480_extra_static_debug = ok", flush=True)
    return selected, job_env


base.build_job_env = build_job_env
base.local_debug = local_debug


def write_manifest(payload: dict[str, Any]) -> Path:
    payload["previous_version"] = "V476 failed by objective skew; V479 CPU objective-aligned candidate"
    payload["version_comparison_artifact"] = (
        "artifacts/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter/V479_VS_PREVIOUS.md"
    )
    payload["objective_alignment_gate"] = "scripts/audit_v478_training_objective_alignment.py"
    payload["hf_job_timeout_seconds"] = HF_JOB_TIMEOUT_SECONDS
    payload["next_action"] = (
        "Monitor every 40 seconds; after training, weak-eval checkpoints 2/4/6/8; "
        "stop unless total>192, equation>56, bit>=136, truncated=0."
    )
    if isinstance(payload.get("recipe"), dict):
        payload["recipe"].update(
            {
                "previous_version": "V476 failed by objective skew; V479 CPU objective-aligned candidate",
                "version_comparison_artifact": (
                    "artifacts/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter/V479_VS_PREVIOUS.md"
                ),
                "learning_rate": "4.0e-8",
                "final_learning_rate": "1.0e-8",
                "objective_alignment_gate": "scripts/audit_v478_training_objective_alignment.py",
                "promotion_gate": (
                    "promote only if weak total>192, equation>56, bit>=136, truncated=0; "
                    "otherwise cancel by FinOps"
                ),
            }
        )
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base.RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("launch_manifest_path =", out_path, flush=True)
    return out_path


base.write_manifest = write_manifest


def base_manifest(selected_hardware: dict[str, object], job_env: dict[str, str]) -> dict[str, Any]:
    return {
        "version": base.VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": base.NAMESPACE,
        "image": base.IMAGE,
        "flavor": base.FLAVOR,
        "hardware": selected_hardware,
        "expected_commit": base.EXPECTED_COMMIT,
        "branch": base.REPO_BRANCH,
        "run_id": base.RUN_ID,
        "output_repo": base.OUTPUT_REPO,
        "dataset": {
            "data_repo": base.DATA_REPO,
            "dataset_upload_commit": base.DATASET_UPLOAD_COMMIT,
            "train_file": base.TRAIN_FILE,
            "val_file": base.VAL_FILE,
            "train_sha256": base.TRAIN_SHA256,
            "val_sha256": base.VAL_SHA256,
            "train_rows": base.TRAIN_ROWS,
            "val_rows": base.VAL_ROWS,
        },
        "init_adapter": {"repo": base.INIT_ADAPTER_REPO, "subfolder": base.INIT_ADAPTER_SUBFOLDER},
        "job_env": job_env,
        "recipe": {
            "max_steps": base.MAX_STEPS,
            "save_every_steps": base.SAVE_EVERY_STEPS,
            "eval_every_steps": base.EVAL_EVERY_STEPS,
            "eval_max_examples": base.EVAL_MAX_EXAMPLES,
            "max_length": 1024,
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "learning_rate": "4.0e-8",
            "final_learning_rate": "1.0e-8",
            "answer_span_loss_weight": base.ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": base.SOURCE_WEIGHTS,
            "subcategory_weights": base.SUBCATEGORY_WEIGHTS,
            "sampling_mode": "weighted_replacement_with_equal_weights",
            "promotion_gate": "weak total>192, equation>56, bit>=136, truncated=0; otherwise cancel by FinOps",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V480.")
    api = HfApi(token=token)
    selected_hardware, job_env = base.local_debug(api, token)
    manifest = base_manifest(selected_hardware, job_env)
    if not args.launch:
        base.write_manifest({**manifest, "mode": "debug_only_no_job_launched"})
        return 0

    job = api.run_job(
        image=base.IMAGE,
        command=["/bin/bash", "-lc", base.COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=base.FLAVOR,
        timeout=HF_JOB_TIMEOUT_SECONDS,
        namespace=base.NAMESPACE,
    )
    launched = {
        **manifest,
        "mode": "launched",
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{base.NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
    }
    base.write_manifest(launched)
    print("job_url =", launched["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

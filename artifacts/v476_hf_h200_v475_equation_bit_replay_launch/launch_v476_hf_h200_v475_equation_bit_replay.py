#!/usr/bin/env python3
"""Launch/debug V476 V475 equation+bit replay smoke train on HF H200.

Default mode is local debug only. Pass ``--launch`` only after:

1. V475 combined tokenization gate passed with zero truncation.
2. Static safety gate passed.
3. This launcher commit was pushed to GitHub.
4. The V475 dataset path is uploaded to the HF dataset repo.

The run is intentionally capped at one hour and must be followed by weak eval
with kill-switch: total>192, equation>56, bit>=136, truncated=0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
HF_JOB_TIMEOUT_SECONDS = 3600


def load_base_module():
    spec = importlib.util.spec_from_file_location("v391_base_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v476_v475_equation_bit_replay_from_v290_checkpoint6_nemo_h200"
base.RUN_ID = "v476-nemo-h200-v475-equation-bit-replay-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base.DATASET_UPLOAD_COMMIT = "https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/023ccb160c8998508041322c5a47296bbc73de2d"
base.TRAIN_FILE = "data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_train.jsonl"
base.VAL_FILE = "data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_val.jsonl"
base.TRAIN_SHA256 = "22aa443a0c7d2f0dea790fe7afcd7249cc4d775e5627671e7dc6377105095aa6"
base.VAL_SHA256 = "2a3945ffd7cc795043fdc14ef12d37a044a5a3d304c5a5d1b9c0fa37b89cc0e7"
base.TRAIN_ROWS = 1312
base.VAL_ROWS = 328
base.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v476-v475-equation-bit-replay-v290ckpt6"
base.MAX_STEPS = 12
base.SAVE_EVERY_STEPS = 2
base.EVAL_EVERY_STEPS = 2
base.EVAL_MAX_EXAMPLES = 96
base.ANSWER_SPAN_LOSS_WEIGHT = "16.0"
base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1000"
base.SOURCE_WEIGHTS = "v475_v325_equation_no_loss_distill=8.00,v475_v217_bit_replay_guardrail=1.25"
base.SUBCATEGORY_WEIGHTS = (
    "equation_numeric_add_direct=12.00,"
    "equation_numeric_colon_absdiff=12.00,"
    "equation_numeric_colon_trailing_zero=12.00,"
    "equation_numeric_minus_direct_negative=12.00,"
    "equation_numeric_minus_signed=12.00,"
    "bit_guardrail_replay=1.15"
)
base.COMMAND_SCRIPT = (
    base.COMMAND_SCRIPT
    .replace("export OUTPUT_DIR='/tmp/kg1_v391_output'", "export OUTPUT_DIR='/tmp/kg1_v476_output'")
    .replace("export LEARNING_RATE=4.0e-8", "export LEARNING_RATE=6.0e-8")
    .replace("export FINAL_LEARNING_RATE=1.0e-8", "export FINAL_LEARNING_RATE=1.5e-8")
)

_base_build_job_env = base.build_job_env
_base_local_debug = base.local_debug


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    env = _base_build_job_env(hardware)
    env.update(
        {
            "KG1_HF_MAX_UNIT_COST_USD": "0.09",
            "KG1_EXPECTED_MAX_STEPS": "12",
            "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
                "bit_guardrail_replay,"
                "equation_numeric_add_direct,equation_numeric_colon_absdiff,"
                "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
                "equation_numeric_minus_signed"
            ),
            "KG1_REQUIRED_VAL_SUBCATEGORIES": (
                "bit_guardrail_replay,"
                "equation_numeric_add_direct,equation_numeric_colon_absdiff,"
                "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
                "equation_numeric_minus_signed"
            ),
            "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        }
    )
    return env


def local_debug(api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str]]:
    selected, job_env = _base_local_debug(api, token)
    forbidden_snippets = [
        "data/v390_equation_bit_replay_mix",
        "data/v464",
        "data/v468",
        "v464_" + "v463_numeric_multirule_dataset",
        "v468_" + "v464_symbol_fix_dataset",
        "timeout=5400",
    ]
    found_forbidden = [item for item in forbidden_snippets if item in base.COMMAND_SCRIPT or item in json.dumps(job_env)]
    if found_forbidden:
        raise RuntimeError("V476 command/env contains stale forbidden snippets: " + json.dumps(found_forbidden))
    required_env = {
        "KG1_TRAIN_FILE": base.TRAIN_FILE,
        "KG1_VAL_FILE": base.VAL_FILE,
        "KG1_TRAIN_SHA": base.TRAIN_SHA256,
        "KG1_VAL_SHA": base.VAL_SHA256,
        "KG1_TRAIN_ROWS": str(base.TRAIN_ROWS),
        "KG1_VAL_ROWS": str(base.VAL_ROWS),
        "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
            "bit_guardrail_replay,"
            "equation_numeric_add_direct,equation_numeric_colon_absdiff,"
            "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
            "equation_numeric_minus_signed"
        ),
    }
    mismatched = {
        key: {"observed": job_env.get(key), "expected": expected}
        for key, expected in required_env.items()
        if job_env.get(key) != expected
    }
    if mismatched:
        raise RuntimeError("V476 job env mismatch: " + json.dumps(mismatched, sort_keys=True))
    print("v476_extra_static_debug = ok", flush=True)
    return selected, job_env


base.build_job_env = build_job_env
base.local_debug = local_debug


def write_manifest(payload: dict[str, Any]) -> Path:
    payload["previous_version"] = "V475 CPU dataset/token gate; V391 historical equation+bit replay route"
    payload["version_comparison_artifact"] = (
        "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/V475_VS_PREVIOUS.md"
    )
    payload["hf_job_timeout_seconds"] = HF_JOB_TIMEOUT_SECONDS
    payload["next_action"] = (
        "Monitor every 40 seconds; after training, weak-eval checkpoints 2/4/6/8/10/12; "
        "stop unless total>192, equation>56, bit>=136, truncated=0."
    )
    if isinstance(payload.get("recipe"), dict):
        payload["recipe"].update(
            {
                "previous_version": "V475 CPU dataset/token gate; V391 historical equation+bit replay route",
                "version_comparison_artifact": (
                    "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/V475_VS_PREVIOUS.md"
                ),
                "learning_rate": "6.0e-8",
                "final_learning_rate": "1.5e-8",
                "promotion_gate": (
                    "promote only if weak total>192, equation>56, bit>=136, truncated=0; "
                    "otherwise cancel by FinOps"
                ),
            }
        )
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base.RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            "learning_rate": "6.0e-8",
            "final_learning_rate": "1.5e-8",
            "answer_span_loss_weight": base.ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": base.SOURCE_WEIGHTS,
            "subcategory_weights": base.SUBCATEGORY_WEIGHTS,
            "promotion_gate": "weak total>192, equation>56, bit>=136, truncated=0; otherwise cancel by FinOps",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V476.")
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

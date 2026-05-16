#!/usr/bin/env python3
"""Launch/debug V495 V475 CPU-gated smoke on HF H200.

This is a narrow follow-up to V493/V494. V493 proved the MoE target
parameters can be trainable, but it used the older V390/V326 train mix. V495
keeps the same mechanism and switches only to the V475 CPU-gated dataset that
projected equation 56->60 with bit preserved.
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


VERSION = "v495_v475_moe_trainable_no_lmhead_equation_bit_replay_from_v290_checkpoint6_nemo_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
RUN_ID = "v495-nemo-h200-v475-moe-trainable-no-lmhead-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATASET_UPLOAD_COMMIT = "dataset:v475_equation_bit_replay_mix_already_uploaded"
TRAIN_FILE = "data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_train.jsonl"
VAL_FILE = "data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_val.jsonl"
TRAIN_SHA256 = "22aa443a0c7d2f0dea790fe7afcd7249cc4d775e5627671e7dc6377105095aa6"
VAL_SHA256 = "2a3945ffd7cc795043fdc14ef12d37a044a5a3d304c5a5d1b9c0fa37b89cc0e7"
TRAIN_ROWS = 1312
VAL_ROWS = 328

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v495-nemo-h200-v475-moe-trainable-no-lmhead-v290ckpt6"

MAX_STEPS = 2
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 96
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1"

SOURCE_WEIGHTS = "v475_v325_equation_no_loss_distill=1.00,v475_v217_bit_replay_guardrail=1.00"
SUBCATEGORY_WEIGHTS = (
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_absdiff=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_direct_negative=1.00,"
    "equation_numeric_minus_signed=1.00,"
    "bit_guardrail_replay=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "bit_guardrail_replay,"
    "equation_numeric_add_direct,"
    "equation_numeric_colon_absdiff,"
    "equation_numeric_colon_trailing_zero,"
    "equation_numeric_minus_direct_negative,"
    "equation_numeric_minus_signed"
)


def load_base_module() -> Any:
    base_path = (
        REPO_ROOT
        / "artifacts"
        / "v493_hf_nemo_h200_moe_trainable_no_lmhead_launch"
        / "launch_v493_hf_nemo_h200_moe_trainable_no_lmhead.py"
    )
    spec = importlib.util.spec_from_file_location("kg1_v493_base_launcher", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V493 base launcher from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(__file__).resolve())
    return module


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "",
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_TRAIN_FILE": TRAIN_FILE,
        "KG1_VAL_FILE": VAL_FILE,
        "KG1_TRAIN_SHA": TRAIN_SHA256,
        "KG1_VAL_SHA": VAL_SHA256,
        "KG1_TRAIN_ROWS": str(TRAIN_ROWS),
        "KG1_VAL_ROWS": str(VAL_ROWS),
        "KG1_INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "KG1_INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
        "KG1_LORA_TARGET_PARAMETERS": INIT_ADAPTER_TARGET_PARAMETERS,
        "KG1_ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
        "KG1_SOURCE_WEIGHTS": SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
        "KG1_REQUIRE_MAMBA_IMPORTS": "1",
    }


def configure_base(base: Any) -> None:
    base.VERSION = VERSION
    base.NAMESPACE = NAMESPACE
    base.REPO_BRANCH = REPO_BRANCH
    base.EXPECTED_COMMIT = EXPECTED_COMMIT
    base.IMAGE = IMAGE
    base.FLAVOR = FLAVOR
    base.RUN_ID = RUN_ID
    base.DATA_REPO = DATA_REPO
    base.DATASET_UPLOAD_COMMIT = DATASET_UPLOAD_COMMIT
    base.TRAIN_FILE = TRAIN_FILE
    base.VAL_FILE = VAL_FILE
    base.TRAIN_SHA256 = TRAIN_SHA256
    base.VAL_SHA256 = VAL_SHA256
    base.TRAIN_ROWS = TRAIN_ROWS
    base.VAL_ROWS = VAL_ROWS
    base.INIT_ADAPTER_REPO = INIT_ADAPTER_REPO
    base.INIT_ADAPTER_SUBFOLDER = INIT_ADAPTER_SUBFOLDER
    base.INIT_ADAPTER_TARGET_PARAMETERS = INIT_ADAPTER_TARGET_PARAMETERS
    base.OUTPUT_REPO = OUTPUT_REPO
    base.MAX_STEPS = MAX_STEPS
    base.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS
    base.EVAL_EVERY_STEPS = EVAL_EVERY_STEPS
    base.EVAL_MAX_EXAMPLES = EVAL_MAX_EXAMPLES
    base.ANSWER_SPAN_LOSS_WEIGHT = ANSWER_SPAN_LOSS_WEIGHT
    base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = ANSWER_SPAN_MIN_WEIGHTED_TOKENS
    base.SOURCE_WEIGHTS = SOURCE_WEIGHTS
    base.SUBCATEGORY_WEIGHTS = SUBCATEGORY_WEIGHTS
    base.build_job_env = build_job_env
    base.COMMAND_SCRIPT = (
        base.COMMAND_SCRIPT
        .replace("v493", "v495")
        .replace("V493", "V495")
        .replace("kg1_v493_output", "kg1_v495_output")
        .replace("kg1_v493_objective_alignment_gate", "kg1_v495_objective_alignment_gate")
    )


def write_manifest(payload: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V495.")
    api = HfApi(token=token)
    base = load_base_module()
    configure_base(base)

    selected_hardware, job_env, objective_alignment_info = base.local_debug(api, token)
    if not args.launch:
        write_manifest(
            {
                "version": VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "debug_only_no_job_launched",
                "namespace": NAMESPACE,
                "image": IMAGE,
                "flavor": FLAVOR,
                "hardware": selected_hardware,
                "expected_commit": EXPECTED_COMMIT,
                "branch": REPO_BRANCH,
                "run_id": RUN_ID,
                "output_repo": OUTPUT_REPO,
                "dataset": {
                    "data_repo": DATA_REPO,
                    "dataset_upload_commit": DATASET_UPLOAD_COMMIT,
                    "train_file": TRAIN_FILE,
                    "val_file": VAL_FILE,
                    "train_sha256": TRAIN_SHA256,
                    "val_sha256": VAL_SHA256,
                    "train_rows": TRAIN_ROWS,
                    "val_rows": VAL_ROWS,
                },
                "init_adapter": {"repo": INIT_ADAPTER_REPO, "subfolder": INIT_ADAPTER_SUBFOLDER},
                "job_env": job_env,
                "objective_alignment": objective_alignment_info,
                "previous_version": "V493/V494 used V390/V326 and failed weak promotion",
                "next_action": "Commit/push, then run this launcher with --launch for the one V475 fail-fast smoke.",
            }
        )
        return 0

    job = api.run_job(
        image=IMAGE,
        command=["/bin/bash", "-lc", base.COMMAND_SCRIPT],
        env=job_env,
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=3600,
        namespace=NAMESPACE,
    )
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "launched",
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": selected_hardware,
        "objective_alignment": objective_alignment_info,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_repo": OUTPUT_REPO,
        "init_adapter": {"repo": INIT_ADAPTER_REPO, "subfolder": INIT_ADAPTER_SUBFOLDER},
        "dataset": {
            "data_repo": DATA_REPO,
            "dataset_upload_commit": DATASET_UPLOAD_COMMIT,
            "train_file": TRAIN_FILE,
            "val_file": VAL_FILE,
            "train_sha256": TRAIN_SHA256,
            "val_sha256": VAL_SHA256,
            "train_rows": TRAIN_ROWS,
            "val_rows": VAL_ROWS,
        },
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj",
            "target_parameters_trainability": "required_trainable",
            "learning_rate": "2.0e-8",
            "final_learning_rate": "5.0e-9",
            "answer_span_loss_weight": ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": SOURCE_WEIGHTS,
            "subcategory_weights": SUBCATEGORY_WEIGHTS,
            "promotion_gate": "reject if total<=192 or bit<136 or truncated>0; inspect only if equation>56; promote only if total>192 and equation>56 and bit>=136 and truncated=0",
            "version_comparison_artifact": "artifacts/version_diffs/V495_V493.md",
            "previous_version": "V493 H200 MoE trainable no-lmhead smoke on V390/V326",
        },
        "next_action": "Monitor every 40 seconds; weak-eval checkpoint-2 after training completes.",
    }
    write_manifest(manifest)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Launch/debug V444 high-confidence reconstructed SFT smoke train on HF H200.

V444 is deliberately narrow: it removes reconstructed rows with
``rule_unknown`` status, keeps only ``rule_found`` and ``hypothesis_formed``,
and runs a four-step continuation from the locked V290 checkpoint-6 adapter.
Default mode is local debug only. Pass ``--launch`` only after this launcher is
committed and pushed.
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
    / "artifacts/v398_hf_nemo_h200_sft_reconstructed_launch/"
    / "launch_v398_hf_nemo_h200_sft_reconstructed.py"
)
HF_JOB_TIMEOUT_SECONDS = 3600


def load_base_module():
    spec = importlib.util.spec_from_file_location("v398_base_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v444_high_conf_sft_reconstructed_from_v290_checkpoint6_nemo_h200"
base.RUN_ID = "v444-nemo-h200-highconf-sft-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base.DATASET_UPLOAD_NOTE = "dataset:v444_high_conf_reconstructed_sft;gate:v286_tokenization_passed;rule_unknown_removed"
base.TRAIN_FILE = "data/v444_sft_reconstructed_high_conf/20260515T_cpu_gate/v397_sft_reconstructed_transfer_train.jsonl"
base.VAL_FILE = "data/v444_sft_reconstructed_high_conf/20260515T_cpu_gate/v397_sft_reconstructed_transfer_val.jsonl"
base.TRAIN_SHA256 = "4b064ed04401c6632798c470f76225688e0af3b0771dc65225d32cc283f439cc"
base.VAL_SHA256 = "7a6ba5a60575f34f04f721b3c2312147a33fbbea6d3e27fbf9063ab8f4ef361e"
base.TRAIN_ROWS = 1848
base.VAL_ROWS = 172
base.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v444-nemo-h200-highconf-sft-v290ckpt6"
base.MAX_STEPS = 4
base.SAVE_EVERY_STEPS = 2
base.EVAL_EVERY_STEPS = 2
base.EVAL_MAX_EXAMPLES = 64
base.ANSWER_SPAN_LOSS_WEIGHT = "4.0"
base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "800"
base.SOURCE_WEIGHTS = "local_sft_reconstructed_jsonl=1.00,v444_sft_reconstructed_high_conf=1.25"
base.SUBCATEGORY_WEIGHTS = (
    "bit_manipulation=1.15,"
    "cryptarithm_deduce=2.00,"
    "cryptarithm_guess=2.20,"
    "equation_numeric_deduce=1.45,"
    "equation_numeric_guess=1.70"
)

base.COMMAND_SCRIPT = (
    base.COMMAND_SCRIPT
    .replace("export OUTPUT_DIR='/tmp/kg1_v398_output'", "export OUTPUT_DIR='/tmp/kg1_v444_output'")
    .replace("export LEARNING_RATE=1.5e-8", "export LEARNING_RATE=1.2e-8")
    .replace("export FINAL_LEARNING_RATE=5.0e-9", "export FINAL_LEARNING_RATE=4.0e-9")
)

_base_build_job_env = base.build_job_env
_base_local_debug = base.local_debug


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    env = _base_build_job_env(hardware)
    env.update(
        {
            "KG1_HF_MAX_UNIT_COST_USD": "0.09",
            "KG1_EXPECTED_MAX_STEPS": "4",
            "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
                "bit_manipulation,cryptarithm_deduce,cryptarithm_guess,"
                "equation_numeric_deduce,equation_numeric_guess"
            ),
            "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        }
    )
    return env


def local_debug(api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str]]:
    selected, job_env = _base_local_debug(api, token)
    forbidden_snippets = [
        "data/v397_sft_reconstructed_transfer/20260514T_cpu_gate",
        "data/v390_equation_bit_replay_mix",
        "MAX_STEPS=12",
        "timeout=5400",
    ]
    found_forbidden = [item for item in forbidden_snippets if item in base.COMMAND_SCRIPT]
    if found_forbidden:
        raise RuntimeError("V444 command contains stale snippets: " + json.dumps(found_forbidden))
    required_env = {
        "KG1_TRAIN_FILE": base.TRAIN_FILE,
        "KG1_VAL_FILE": base.VAL_FILE,
        "KG1_TRAIN_SHA": base.TRAIN_SHA256,
        "KG1_VAL_SHA": base.VAL_SHA256,
        "KG1_TRAIN_ROWS": str(base.TRAIN_ROWS),
        "KG1_VAL_ROWS": str(base.VAL_ROWS),
        "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
    }
    mismatched = {
        key: {"observed": job_env.get(key), "expected": expected}
        for key, expected in required_env.items()
        if job_env.get(key) != expected
    }
    if mismatched:
        raise RuntimeError("V444 job env mismatch: " + json.dumps(mismatched, sort_keys=True))
    print("v444_extra_static_debug = ok", flush=True)
    return selected, job_env


base.build_job_env = build_job_env
base.local_debug = local_debug


def write_manifest(payload: dict[str, Any]) -> Path:
    payload["previous_version"] = "V398 broad reconstructed SFT; V443 certified string rule builder"
    payload["version_comparison_artifact"] = (
        "artifacts/v444_hf_h200_high_conf_sft_launch/V444_VS_PREVIOUS.md"
    )
    payload["hf_job_timeout_seconds"] = HF_JOB_TIMEOUT_SECONDS
    payload["next_action"] = (
        "Monitor every 40 seconds; weak-eval checkpoint-2/4; stop unless weak total>192, "
        "equation>56, bit>=136, truncated=0."
    )
    if isinstance(payload.get("recipe"), dict):
        payload["recipe"].update(
            {
                "previous_version": "V398 broad reconstructed SFT; V443 certified string rule builder",
                "version_comparison_artifact": (
                    "artifacts/v444_hf_h200_high_conf_sft_launch/V444_VS_PREVIOUS.md"
                ),
                "learning_rate": "1.2e-8",
                "final_learning_rate": "4.0e-9",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V444.")
    api = HfApi(token=token)
    selected_hardware, job_env = base.local_debug(api, token)
    base_manifest = {
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
            "dataset_upload_note": base.DATASET_UPLOAD_NOTE,
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
            "max_length": 8192,
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "learning_rate": "1.2e-8",
            "final_learning_rate": "4.0e-9",
            "answer_span_loss_weight": base.ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": base.SOURCE_WEIGHTS,
            "subcategory_weights": base.SUBCATEGORY_WEIGHTS,
            "promotion_gate": "weak total>192, equation>56, bit>=136, truncated=0; otherwise cancel by FinOps",
            "previous_version": "V398 broad reconstructed SFT; V443 certified string rule builder",
            "version_comparison_artifact": "artifacts/v444_hf_h200_high_conf_sft_launch/V444_VS_PREVIOUS.md",
        },
    }
    if not args.launch:
        base.write_manifest(
            {
                **base_manifest,
                "mode": "debug_only_no_job_launched",
                "next_action": "Commit/push this launcher, then run with --launch.",
            }
        )
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
    manifest = {
        **base_manifest,
        "mode": "launched",
        "job_id": job.id,
        "job_url": f"https://huggingface.co/jobs/{base.NAMESPACE}/{job.id}",
        "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
        "next_action": "Monitor every 40 seconds; weak-eval checkpoint-2/4 after training completes.",
    }
    base.write_manifest(manifest)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

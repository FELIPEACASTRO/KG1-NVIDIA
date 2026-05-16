#!/usr/bin/env python3
"""Launch/debug V469 fixed-symbol numeric multi-rule SFT smoke train on HF H200."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = (
    REPO_ROOT
    / "artifacts/v398_hf_nemo_h200_sft_reconstructed_launch/"
    / "launch_v398_hf_nemo_h200_sft_reconstructed.py"
)
HF_JOB_TIMEOUT_SECONDS = 3600
MAX_UNIT_COST_USD_PER_MIN = 0.09


def load_base_module():
    spec = importlib.util.spec_from_file_location("v398_base_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v469_v468_symbol_fix_from_v290_checkpoint6_nemo_h200"
base.RUN_ID = "v469-nemo-h200-v468-symbol-fix-v290ckpt6-" + datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)
base.DATASET_UPLOAD_NOTE = "dataset:v468_v464_symbol_fix;gate:v286_tokenization_passed;hardneg=22_train_3_rules;contradictory_rejected_candidates=0"
base.TRAIN_FILE = "data/v468_v464_symbol_fix_dataset/20260515T_cpu_gate/v468_v464_symbol_fix_dataset_train.jsonl"
base.VAL_FILE = "data/v468_v464_symbol_fix_dataset/20260515T_cpu_gate/v468_v464_symbol_fix_dataset_val.jsonl"
base.TRAIN_SHA256 = "6f04f07e0406cfab4eb599b868055d2abc0c043a8dde2eba4fb7f76cdc7f3cb8"
base.VAL_SHA256 = "b99c6d57f31636221a045dcfe108fbf2c908bffe8db26cb207b9f3de871da162"
base.TRAIN_ROWS = 558
base.VAL_ROWS = 138
base.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v469-v468-symbol-fix-v290ckpt6"
base.MAX_STEPS = 16
base.SAVE_EVERY_STEPS = 4
base.EVAL_EVERY_STEPS = 4
base.EVAL_MAX_EXAMPLES = 96
base.ANSWER_SPAN_LOSS_WEIGHT = "5.0"
base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1000"
base.SOURCE_WEIGHTS = "v468_v464_symbol_fix_dataset=2.50,v217_bit_replay_guardrail=0.75"
base.SUBCATEGORY_WEIGHTS = (
    "bit_guardrail_replay=0.65,"
    "v274_guarded_numeric_add_direct_over_model_add_variant=4.00,"
    "v274_guarded_numeric_colon_absdiff_restore_trailing_zero=0.75,"
    "v274_guarded_numeric_minus_direct_negative_restore_sign=4.00,"
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded=3.50"
)

TOKENIZATION_GATE_FILE = (
    "runtime_artifacts/v468_v464_symbol_fix_dataset/20260515T_tokenization_gate/"
    "v286_generic_tokenization_gate_manifest.json"
)
TOKENIZATION_GATE_SHA256 = "5228144c4ddf0f31df92d48867cf286606bceefd545fa258c41ea616bad9c4af"
REQUIRED_SUBCATEGORIES = (
    "bit_guardrail_replay,"
    "v274_guarded_numeric_add_direct_over_model_add_variant,"
    "v274_guarded_numeric_colon_absdiff_restore_trailing_zero,"
    "v274_guarded_numeric_minus_direct_negative_restore_sign,"
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded"
)

base.COMMAND_SCRIPT = (
    base.COMMAND_SCRIPT.replace("export OUTPUT_DIR='/tmp/kg1_v398_output'", "export OUTPUT_DIR='/tmp/kg1_v469_output'")
    .replace("export LEARNING_RATE=1.5e-8", "export LEARNING_RATE=2.8e-8")
    .replace("export FINAL_LEARNING_RATE=5.0e-9", "export FINAL_LEARNING_RATE=8.0e-9")
    .replace("export MAX_STEPS=4", "export MAX_STEPS=16")
    .replace("export SAVE_EVERY_STEPS=2", "export SAVE_EVERY_STEPS=4")
    .replace("export EVAL_EVERY_STEPS=2", "export EVAL_EVERY_STEPS=4")
    .replace("export EVAL_MAX_EXAMPLES=64", "export EVAL_MAX_EXAMPLES=96")
)


def hf_download_and_hash(repo_id: str, filename: str, expected_sha: str, token: str) -> dict[str, object]:
    local_dir = Path(__file__).resolve().parent / "_hf_debug_download"
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
        )
    )
    observed = base.sha256_file(path)
    if observed.lower() != expected_sha.lower():
        raise RuntimeError(f"HF hash mismatch for {filename}: {observed} != {expected_sha}")
    return {"filename": filename, "local_path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    env = base.build_job_env(hardware)
    env.update(
        {
            "KG1_HF_MAX_UNIT_COST_USD": str(MAX_UNIT_COST_USD_PER_MIN),
            "KG1_EXPECTED_MAX_STEPS": "16",
            "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
            "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        }
    )
    return env


def local_debug(api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str]]:
    print("=== V469 LAUNCHER DEBUG START ===", flush=True)
    print("version =", base.VERSION, flush=True)
    print("expected_commit =", base.EXPECTED_COMMIT, flush=True)
    print("image =", base.IMAGE, flush=True)
    print("flavor =", base.FLAVOR, flush=True)
    print("run_id =", base.RUN_ID, flush=True)

    hardware = {item.name: base.hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if base.FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {base.FLAVOR!r} is not available. Available={sorted(hardware)}")
    selected = hardware[base.FLAVOR]
    if float(selected["unit_cost_usd"]) > MAX_UNIT_COST_USD_PER_MIN:
        raise RuntimeError(f"H200 unit cost above gate: {selected}")
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)

    train_info = base.download_and_hash(base.DATA_REPO, base.TRAIN_FILE, base.TRAIN_SHA256, token)
    val_info = base.download_and_hash(base.DATA_REPO, base.VAL_FILE, base.VAL_SHA256, token)
    gate_info = hf_download_and_hash(base.DATA_REPO, TOKENIZATION_GATE_FILE, TOKENIZATION_GATE_SHA256, token)
    print("hf_train_file_ok =", json.dumps(train_info, sort_keys=True), flush=True)
    print("hf_val_file_ok =", json.dumps(val_info, sort_keys=True), flush=True)
    print("hf_tokenization_gate_file_ok =", json.dumps(gate_info, sort_keys=True), flush=True)

    adapter_files = set(api.list_repo_files(base.INIT_ADAPTER_REPO, repo_type="model"))
    required_adapter_files = {
        f"{base.INIT_ADAPTER_SUBFOLDER}/adapter_config.json",
        f"{base.INIT_ADAPTER_SUBFOLDER}/adapter_model.safetensors",
    }
    missing = sorted(required_adapter_files - adapter_files)
    if missing:
        raise RuntimeError("missing init adapter files: " + json.dumps(missing))
    print("init_adapter_files_ok =", json.dumps(sorted(required_adapter_files)), flush=True)

    forbidden_snippets = [
        "data/v397_sft_reconstructed_transfer/20260514T_cpu_gate",
        "data/v464_v463_numeric_multirule_dataset/20260515T_cpu_gate",
        "runtime_artifacts/v464_v463_numeric_multirule_dataset/20260515T_tokenization_gate",
        "data/v447_v446_trace_dataset",
        "MAX_LENGTH=1024",
        "export MAX_STEPS=4",
        "export SAVE_EVERY_STEPS=2",
        "export EVAL_EVERY_STEPS=2",
        "timeout=5400",
        "kg1_v465_output",
        "v465_v464_numeric_multirule",
    ]
    found_forbidden = [item for item in forbidden_snippets if item in base.COMMAND_SCRIPT]
    if found_forbidden:
        raise RuntimeError("V469 command contains stale snippets: " + json.dumps(found_forbidden))
    required_snippets = [
        "export DATA_FILE=\"$KG1_TRAIN_FILE\"",
        "export VAL_FILE=\"$KG1_VAL_FILE\"",
        "export OUTPUT_DIR='/tmp/kg1_v469_output'",
        "export MAX_LENGTH=8192",
        "export LEARNING_RATE=2.8e-8",
        "export FINAL_LEARNING_RATE=8.0e-9",
        "export MAX_STEPS=16",
        "export SAVE_EVERY_STEPS=4",
        "export EVAL_EVERY_STEPS=4",
        "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'",
        "$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts",
        "$PYBIN scripts/hf_job_train_v90.py",
    ]
    missing_snippets = [item for item in required_snippets if item not in base.COMMAND_SCRIPT]
    if missing_snippets:
        raise RuntimeError("V469 command missing required snippets: " + json.dumps(missing_snippets))
    print("command_script_static_debug = ok", flush=True)

    job_env = build_job_env(selected)
    required_env = {
        "KG1_TRAIN_FILE": base.TRAIN_FILE,
        "KG1_VAL_FILE": base.VAL_FILE,
        "KG1_TRAIN_SHA": base.TRAIN_SHA256,
        "KG1_VAL_SHA": base.VAL_SHA256,
        "KG1_TRAIN_ROWS": str(base.TRAIN_ROWS),
        "KG1_VAL_ROWS": str(base.VAL_ROWS),
        "KG1_SOURCE_WEIGHTS": base.SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": base.SUBCATEGORY_WEIGHTS,
        "KG1_EXPECTED_MAX_STEPS": "16",
        "KG1_HF_JOB_TIMEOUT_SECONDS": str(HF_JOB_TIMEOUT_SECONDS),
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    }
    mismatched = {
        key: {"observed": job_env.get(key), "expected": expected}
        for key, expected in required_env.items()
        if job_env.get(key) != expected
    }
    if mismatched:
        raise RuntimeError("V469 job env mismatch: " + json.dumps(mismatched, sort_keys=True))
    forbidden_env = [item for item in forbidden_snippets if item in json.dumps(job_env, sort_keys=True)]
    if forbidden_env:
        raise RuntimeError("V469 job env contains stale snippets: " + json.dumps(forbidden_env))
    print("hf_job_env_debug =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("=== V469 LAUNCHER DEBUG END ===", flush=True)
    return selected, job_env


def write_manifest(payload: dict[str, Any]) -> Path:
    payload["previous_version"] = "V465/V466 contaminated V464 smoke; V290 checkpoint-6 baseline"
    payload["version_comparison_artifact"] = "artifacts/v469_hf_h200_v468_symbol_fix_launch/V469_VS_PREVIOUS.md"
    payload["hf_job_timeout_seconds"] = HF_JOB_TIMEOUT_SECONDS
    if isinstance(payload.get("recipe"), dict):
        payload["recipe"].update(
            {
                "previous_version": payload["previous_version"],
                "version_comparison_artifact": payload["version_comparison_artifact"],
                "learning_rate": "2.8e-8",
                "final_learning_rate": "8.0e-9",
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


def build_manifest(selected_hardware: dict[str, object], job_env: dict[str, str]) -> dict[str, Any]:
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
            "dataset_upload_note": base.DATASET_UPLOAD_NOTE,
            "train_file": base.TRAIN_FILE,
            "val_file": base.VAL_FILE,
            "train_sha256": base.TRAIN_SHA256,
            "val_sha256": base.VAL_SHA256,
            "train_rows": base.TRAIN_ROWS,
            "val_rows": base.VAL_ROWS,
            "tokenization_gate_file": TOKENIZATION_GATE_FILE,
            "tokenization_gate_sha256": TOKENIZATION_GATE_SHA256,
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
            "learning_rate": "2.8e-8",
            "final_learning_rate": "8.0e-9",
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
        raise RuntimeError("HF token is required to debug or launch V469.")
    api = HfApi(token=token)
    selected_hardware, job_env = local_debug(api, token)
    base_manifest = build_manifest(selected_hardware, job_env)
    if not args.launch:
        write_manifest(
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
        "next_action": "Monitor every 40 seconds; weak-eval checkpoint-4 first; cancel if weak total<=192, equation<=56, bit<136, or truncated>0.",
    }
    write_manifest(manifest)
    print("job_url =", manifest["job_url"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

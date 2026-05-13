#!/usr/bin/env python3
"""Launch/debug V341 cleaned-preference smoke train on HF A100.

Default mode writes a local launch manifest only. Pass --launch to create the
paid HF job. This uses the existing V315 single-policy preference trainer, but
switches to V337D SFT files plus V341 cleaned preference pairs.
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


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V315_LAUNCHER = REPO_ROOT / "artifacts/v315_hf_h200_preference_launch/launch_v315_hf_h200_v312_preference.py"
V340_CLEAN_GATE = (
    REPO_ROOT
    / "artifacts/v340_hard_negative_abstain_gate/20260513T_cpu_gate_v341_cleaned/"
    / "v340_hard_negative_abstain_gate_manifest.json"
)
V341_UPLOAD_MANIFEST = (
    REPO_ROOT
    / "artifacts/v341_clean_preference_transfer_dataset/"
    / "v341_clean_preference_hf_upload_manifest.json"
)

VERSION = "v341_clean_preference_contrastive_from_v290_checkpoint6_a100"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
FLAVOR = "a100-large"
MAX_UNIT_COST_USD = 0.05
RUN_ID = "v341-clean-pref-a100-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
SFT_TRAIN_FILE = "data/v337d_minimal_transfer/20260513T_cpu_gate/v337d_minimal_transfer_train.jsonl"
SFT_VAL_FILE = "data/v337d_minimal_transfer/20260513T_cpu_gate/v337d_minimal_transfer_val.jsonl"
SFT_TRAIN_SHA256 = "df67214d3fdbb74ada96a9fc24609db5a3f5f6dc1d26dea5d4449eb39eb4147c"
SFT_VAL_SHA256 = "50d4ee05a377ed4e111d27f9de0e1109eb0c09bfe01a9bce0717b63d704dbf80"
SFT_TRAIN_ROWS = 1440
SFT_VAL_ROWS = 340

PREF_TRAIN_FILE = "data/v341_clean_preference_transfer/20260513T_cpu_gate/v341_clean_preference_transfer_preferences_train.jsonl"
PREF_VAL_FILE = "data/v341_clean_preference_transfer/20260513T_cpu_gate/v341_clean_preference_transfer_preferences_val.jsonl"
PREF_TRAIN_SHA256 = "217068058e723063178c00e5b9de697a1a669839e85abdb86154663543a71ae2"
PREF_VAL_SHA256 = "6aa1bfcd795a6bb11fa50551067e0744159a697331f2fff3bdd9084ce912cd8a"
PREF_TRAIN_ROWS = 2843
PREF_VAL_ROWS = 715

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v341-clean-pref-a100-v290ckpt6"

MAX_STEPS = 8
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 96


def load_v315_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v315_launcher", V315_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V315 launcher from {V315_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cpu_gates() -> dict[str, Any]:
    if not V340_CLEAN_GATE.is_file():
        raise FileNotFoundError(V340_CLEAN_GATE)
    if not V341_UPLOAD_MANIFEST.is_file():
        raise FileNotFoundError(V341_UPLOAD_MANIFEST)
    gate = read_json(V340_CLEAN_GATE)
    upload = read_json(V341_UPLOAD_MANIFEST)
    if gate.get("schema_version") != "kg1_v340_hard_negative_abstain_gate_v1":
        raise RuntimeError("unexpected V340 cleaned gate schema")
    if gate.get("assets_valid") is not True:
        raise RuntimeError("V340 cleaned gate assets_valid is not true")
    decision = gate.get("decision", {})
    if decision.get("hf_gpu_allowed") is not True or decision.get("preference_training_allowed") is not True:
        raise RuntimeError("V340 cleaned gate does not allow a preference smoke")
    local_summary = upload.get("local_summary", {})
    if local_summary.get("preferences_train_sha256") != PREF_TRAIN_SHA256:
        raise RuntimeError("V341 uploaded train hash drift")
    if local_summary.get("preferences_val_sha256") != PREF_VAL_SHA256:
        raise RuntimeError("V341 uploaded validation hash drift")
    return {
        "v340_clean_gate_manifest": str(V340_CLEAN_GATE),
        "v340_clean_gate_decision": decision,
        "v341_upload_manifest": str(V341_UPLOAD_MANIFEST),
        "v341_upload_commit": str(upload.get("upload", "")),
    }


def patched_command_script() -> str:
    module = load_v315_launcher()
    script = module.COMMAND_SCRIPT
    replacements = {
        "export OUTPUT_DIR='/tmp/kg1_v315_output'": "export OUTPUT_DIR='/tmp/kg1_v341_output'",
        "export MAX_STEPS=16": f"export MAX_STEPS={MAX_STEPS}",
        "export SAVE_EVERY_STEPS=4": f"export SAVE_EVERY_STEPS={SAVE_EVERY_STEPS}",
        "export EVAL_EVERY_STEPS=4": f"export EVAL_EVERY_STEPS={EVAL_EVERY_STEPS}",
        "export EVAL_MAX_EXAMPLES=24": f"export EVAL_MAX_EXAMPLES={EVAL_MAX_EXAMPLES}",
        "export ABORT_MAX_RESERVED_GIB=118": "export ABORT_MAX_RESERVED_GIB=78",
    }
    for old, new in replacements.items():
        if old not in script:
            raise RuntimeError("V315 command script changed; missing replacement target: " + old)
        script = script.replace(old, new)
    return script


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "2.8.0+cu128",
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "70",
        "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": str(MAX_UNIT_COST_USD),
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
            "equation_numeric_add_direct,equation_numeric_colon_absdiff,equation_numeric_minus_signed,"
            "equation_symbolic_cryptarithm_single_operator_mul,bit_manipulation,unknown"
        ),
        "KG1_REQUIRED_VAL_SUBCATEGORIES": (
            "equation_numeric_add_direct,equation_numeric_colon_absdiff,equation_numeric_minus_signed,"
            "equation_symbolic_cryptarithm_single_operator_mul,bit_manipulation,unknown"
        ),
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_SFT_TRAIN_FILE": SFT_TRAIN_FILE,
        "KG1_SFT_VAL_FILE": SFT_VAL_FILE,
        "KG1_SFT_TRAIN_SHA": SFT_TRAIN_SHA256,
        "KG1_SFT_VAL_SHA": SFT_VAL_SHA256,
        "KG1_SFT_TRAIN_ROWS": str(SFT_TRAIN_ROWS),
        "KG1_SFT_VAL_ROWS": str(SFT_VAL_ROWS),
        "KG1_PREF_TRAIN_FILE": PREF_TRAIN_FILE,
        "KG1_PREF_VAL_FILE": PREF_VAL_FILE,
        "KG1_PREF_TRAIN_SHA": PREF_TRAIN_SHA256,
        "KG1_PREF_VAL_SHA": PREF_VAL_SHA256,
        "KG1_PREF_TRAIN_ROWS": str(PREF_TRAIN_ROWS),
        "KG1_PREF_VAL_ROWS": str(PREF_VAL_ROWS),
        "KG1_INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "KG1_INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
    }


def write_manifest(payload: dict[str, Any]) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest =", out_path, flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Create the paid HF job.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for V341 launch/debug.")
    cpu_gate_summary = validate_cpu_gates()
    api = HfApi(token=token)
    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available.")
    selected = hardware[FLAVOR]
    if float(selected["unit_cost_usd"]) > MAX_UNIT_COST_USD:
        raise RuntimeError(f"HF flavor cost above gate: {selected}")
    command_script = patched_command_script()
    job_env = build_job_env(selected)
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch": bool(args.launch),
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": selected,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_repo": OUTPUT_REPO,
        "init_adapter": {"repo": INIT_ADAPTER_REPO, "subfolder": INIT_ADAPTER_SUBFOLDER},
        "dataset": {
            "data_repo": DATA_REPO,
            "sft_train_file": SFT_TRAIN_FILE,
            "sft_val_file": SFT_VAL_FILE,
            "sft_train_sha256": SFT_TRAIN_SHA256,
            "sft_val_sha256": SFT_VAL_SHA256,
            "preference_train_file": PREF_TRAIN_FILE,
            "preference_val_file": PREF_VAL_FILE,
            "preference_train_sha256": PREF_TRAIN_SHA256,
            "preference_val_sha256": PREF_VAL_SHA256,
            "preference_train_rows": PREF_TRAIN_ROWS,
            "preference_val_rows": PREF_VAL_ROWS,
        },
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "preference_loss": "single-policy contrastive chosen/rejected plus chosen CE",
            "trainable_lora_modules": "q_proj,k_proj,v_proj,o_proj,lm_head",
            "promotion_gate": "weak eval only; promote if total>192, equation>56, bit>=136",
        },
        "cpu_gate_summary": cpu_gate_summary,
        "job_env": job_env,
        "finops_kill_switch": {
            "first_checkpoint_total_min_exclusive": 192,
            "first_checkpoint_equation_min_exclusive": 56,
            "first_checkpoint_bit_min": 136,
            "cancel_if_failed": True,
        },
    }
    if args.launch:
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", command_script],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=5400,
            namespace=NAMESPACE,
        )
        manifest.update(
            {
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )
    else:
        manifest["job_url"] = ""
        manifest["job_status"] = "debug_only_not_launched"
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)
    print("hf_job_env =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    write_manifest(manifest)
    if args.launch:
        print("job_url =", manifest["job_url"], flush=True)
    else:
        print("debug_only = true; pass --launch after commit/push to create the paid job.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

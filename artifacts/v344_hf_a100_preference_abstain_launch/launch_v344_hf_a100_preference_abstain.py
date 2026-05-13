#!/usr/bin/env python3
"""Launch/debug V344 ACC-first preference/abstain smoke train on HF A100.

Default mode writes a local launch manifest only. Pass --launch to create the
paid HF job. This launcher is intentionally tiny: it consumes V344
chosen/rejected rows, saves checkpoint-2, and relies on weak family ACC for
promotion. Lower preference loss is not a promotion criterion.
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
V341_LAUNCHER = REPO_ROOT / "artifacts/v341_hf_a100_clean_preference_launch/launch_v341_hf_a100_clean_preference.py"
V340_GATE = (
    REPO_ROOT
    / "artifacts/v344_v343_transfer_dataset/20260513T_hard_negative_gate_with_launcher/"
    / "v344_v343_hard_negative_abstain_gate_manifest.json"
)
V344_UPLOAD_MANIFEST = RUN_DIR / "v344_v343_transfer_hf_upload_manifest.json"

VERSION = "v344_v343_preference_abstain_from_v290_checkpoint6_a100"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "a100-large"
MAX_UNIT_COST_USD = 0.05
RUN_ID = "v344-pref-abstain-a100-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-nemotron-training"
DATA_PATH = "data/v344_v343_minimal_transfer/20260513T_minimal_transfer_v343"
SFT_TRAIN_FILE = f"{DATA_PATH}/v344_v343_minimal_transfer_train.jsonl"
SFT_VAL_FILE = f"{DATA_PATH}/v344_v343_minimal_transfer_val.jsonl"
SFT_TRAIN_SHA256 = "cab6b8370f2208c3e3fa954527967683be06639d7c556ae7697077d1d2bf8e03"
SFT_VAL_SHA256 = "a2df22315cbd837d6b15c9ff646d76fb7b8d8e3930485ac0b02677b9ed9c87cc"
SFT_TRAIN_ROWS = 1760
SFT_VAL_ROWS = 420

PREF_TRAIN_FILE = f"{DATA_PATH}/v344_v343_minimal_transfer_preferences_train.jsonl"
PREF_VAL_FILE = f"{DATA_PATH}/v344_v343_minimal_transfer_preferences_val.jsonl"
PREF_TRAIN_SHA256 = "cd2c3021731ff141173cd05bed51cb3086320ce4996cd5a55b8e38b678cfee90"
PREF_VAL_SHA256 = "c9fe097ccebfeb9de895abda139688bb0256bcc00f35fb1fd292f53fdbce23a2"
PREF_TRAIN_ROWS = 4160
PREF_VAL_ROWS = 1040

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v344-pref-abstain-a100-v290ckpt6"

MAX_STEPS = 2
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 8


def load_v341_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v341_launcher", V341_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V341 launcher from {V341_LAUNCHER}")
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
    if not V340_GATE.is_file():
        raise FileNotFoundError(V340_GATE)
    if not V344_UPLOAD_MANIFEST.is_file():
        raise FileNotFoundError(V344_UPLOAD_MANIFEST)
    gate = read_json(V340_GATE)
    upload = read_json(V344_UPLOAD_MANIFEST)
    if gate.get("schema_version") != "kg1_v340_hard_negative_abstain_gate_v1":
        raise RuntimeError("unexpected V340/V344 gate schema")
    if gate.get("assets_valid") is not True:
        raise RuntimeError("V340/V344 gate assets_valid is not true")
    decision = gate.get("decision", {})
    if decision.get("hf_gpu_allowed") is not True or decision.get("preference_training_allowed") is not True:
        raise RuntimeError("V340/V344 gate does not allow a preference/abstain smoke")
    signal = gate.get("v336a_signal", {})
    expected_signal = {
        "correct": 199,
        "equation_transform_correct": 63,
        "bit_manipulation_correct": 136,
        "loss_count": 0,
        "accepted_candidate_count": 7,
    }
    observed_signal = {key: int(signal.get(key, -1)) for key in expected_signal}
    if observed_signal != expected_signal:
        raise RuntimeError(f"V344 CPU signal drift: expected {expected_signal}, got {observed_signal}")
    local_summary = upload.get("local_summary", {})
    expected_upload = {
        "train_sha256": SFT_TRAIN_SHA256,
        "val_sha256": SFT_VAL_SHA256,
        "preferences_train_sha256": PREF_TRAIN_SHA256,
        "preferences_val_sha256": PREF_VAL_SHA256,
    }
    for key, expected in expected_upload.items():
        if str(local_summary.get(key, "")) != expected:
            raise RuntimeError(f"V344 upload {key} drift")
    return {
        "v340_gate_manifest": str(V340_GATE),
        "v340_gate_decision": decision,
        "v344_upload_manifest": str(V344_UPLOAD_MANIFEST),
        "v344_upload_commit": str(upload.get("upload", "")),
    }


def patched_command_script() -> str:
    module = load_v341_launcher()
    script = module.patched_command_script()
    replacements = {
        "export OUTPUT_DIR='/tmp/kg1_v341_output'": "export OUTPUT_DIR='/tmp/kg1_v344_output'",
        "export MAX_STEPS=8": f"export MAX_STEPS={MAX_STEPS}",
        "export SAVE_EVERY_STEPS=2": f"export SAVE_EVERY_STEPS={SAVE_EVERY_STEPS}",
        "export EVAL_EVERY_STEPS=2": f"export EVAL_EVERY_STEPS={EVAL_EVERY_STEPS}",
        "export EVAL_MAX_EXAMPLES=96": (
            f"export EVAL_MAX_EXAMPLES={EVAL_MAX_EXAMPLES}\n"
            "export PREFERENCE_EVAL_PROGRESS_EVERY=2"
        ),
    }
    for old, new in replacements.items():
        if old not in script:
            raise RuntimeError("V341 command script changed; missing replacement target: " + old)
        script = script.replace(old, new)
    return script


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    required_subcategories = (
        "bit_manipulation,equation_numeric_add_direct,equation_numeric_colon_absdiff,"
        "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
        "equation_numeric_minus_signed,equation_symbolic_cryptarithm_single_operator_mul,unknown"
    )
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "2.9.0a0+50eac811a6.nv25.09",
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
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": required_subcategories,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": required_subcategories,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_REQUIRE_MAMBA_IMPORTS": "1",
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
        raise RuntimeError("HF token is required for V344 launch/debug.")
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
            "sft_train_rows": SFT_TRAIN_ROWS,
            "sft_val_rows": SFT_VAL_ROWS,
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
            "hard_block": "eval_loss and internal preference accuracy do not authorize promotion.",
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
            timeout=3600,
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

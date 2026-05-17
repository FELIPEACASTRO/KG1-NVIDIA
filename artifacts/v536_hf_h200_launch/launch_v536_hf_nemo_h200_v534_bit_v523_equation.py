#!/usr/bin/env python3
"""Launch/debug V536 V534-bit + V523-equation smoke on HF H200.

V536 is a controlled H200 smoke:

* continue from the V290 checkpoint-6 adapter;
* keep lm_head frozen;
* require MoE target parameters to be trainable;
* use the V536 dataset that passed V286, V513, V524, and V526;
* force LOSS_NORMALIZATION_MODE=example_mean;
* run only four steps with checkpoints at step 2 and 4.

Default mode is local debug only. Pass --launch to create the paid HF job.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


VERSION = "v536_v534_bit_v523_equation_example_mean_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
RUN_ID = "v536-nemo-h200-v534bit-v523eq-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v536-v534-bit-v523-equation-artifacts"
DATASET_UPLOAD_COMMIT = "d2f11d82b40e3e9aa0f5add58c3698a7428bf550"
DATA_ROOT = "v536-v534-bit-v523-equation-20260517T024752Z"
TRAIN_FILE = DATA_ROOT + "/v536_v534_bit_v523_equation_pack_train.jsonl"
VAL_FILE = DATA_ROOT + "/v536_v534_bit_v523_equation_pack_val.jsonl"
PREF_TRAIN_SHA256 = "6c7a91891156cdc666ffbd6478ddfe02bc0c258473615907e40675c2aa716700"
PREF_VAL_SHA256 = "26bf3b0393c3f9757e22b13d91583f15542a5b6f2d1ae13fbf3e34322262f093"
TRAIN_SHA256 = PREF_TRAIN_SHA256
VAL_SHA256 = PREF_VAL_SHA256
PREF_TRAIN_ROWS = 1026
PREF_VAL_ROWS = 219
TRAIN_ROWS = PREF_TRAIN_ROWS
VAL_ROWS = PREF_VAL_ROWS

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v536-nemo-h200-v534bit-v523eq-v290ckpt6"

MAX_STEPS = 4
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 219
MAX_LENGTH = 2048
ABORT_MAX_RESERVED_GIB = 84
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
# Weight 1.0 means this route intentionally uses ordinary completion CE.
# Keep the min-token floor at 0 so manifests cannot imply answer-span weighting
# was active when the weighting multiplier is disabled.
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "0"
LOSS_NORMALIZATION_MODE = "example_mean"

SOURCE_WEIGHTS = "v536_v534_bit_v523_equation_pack=1.00"
SUBCATEGORY_WEIGHTS = (
    "bit_huikang_synthetic_cho=1.00,"
    "bit_huikang_synthetic_maj=1.00,"
    "bit_konbu_high_confidence_trace=1.00,"
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_absdiff=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_signed=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "bit_huikang_synthetic_cho,"
    "bit_huikang_synthetic_maj,"
    "bit_konbu_high_confidence_trace,"
    "equation_numeric_add_direct,"
    "equation_numeric_colon_absdiff,"
    "equation_numeric_colon_trailing_zero,"
    "equation_numeric_minus_signed"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"

# Static/pre-paid gate documentation:
# KG1_CRISIS_MODE_BACKFIRE_GUARD=1
# export LOSS_NORMALIZATION_MODE='example_mean'
# export TRAINABLE_LORA_MODULES="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# DATA_ROOT = "v536-v534-bit-v523-equation-20260517T024752Z"
# PREF_TRAIN_SHA256 = "6c7a91891156cdc666ffbd6478ddfe02bc0c258473615907e40675c2aa716700"
# PREF_VAL_SHA256 = "26bf3b0393c3f9757e22b13d91583f15542a5b6f2d1ae13fbf3e34322262f093"
# PREF_TRAIN_ROWS = 1026
# PREF_VAL_ROWS = 219
# OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v536-nemo-h200-v534bit-v523eq-v290ckpt6"
# INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
# INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
# SAVE_EVERY_STEPS = 2
# EVAL_EVERY_STEPS = 2


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
        "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
        "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_DATASET_SCHEMA": "sft",
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
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
        "KG1_TRAINABLE_LORA_MODULES": TRAINABLE_LORA_MODULES,
        "KG1_ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
        "KG1_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_SOURCE_WEIGHTS": SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
        "KG1_REQUIRE_MAMBA_IMPORTS": "1",
    }


def run_local_objective_alignment(train_path: str, val_path: str) -> dict[str, object]:
    out_path = Path(__file__).resolve().parent / f"{RUN_ID}_objective_alignment_gate.json"
    cmd = [
        sys.executable,
        "scripts/audit_v478_training_objective_alignment.py",
        "--train-jsonl",
        train_path,
        "--val-jsonl",
        val_path,
        "--source-weights",
        SOURCE_WEIGHTS,
        "--subcategory-weights",
        SUBCATEGORY_WEIGHTS,
        "--min-bit-effective-share",
        "0.60",
        "--max-equation-effective-share",
        "0.40",
        "--max-any-family-effective-share",
        "0.75",
        "--output-json",
        str(out_path),
        "--enforce",
    ]
    print("local_objective_alignment_cmd =", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    print("local_objective_alignment_report =", json.dumps(report, indent=2, sort_keys=True), flush=True)
    return {"path": str(out_path), "report": report}


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
    base.run_local_objective_alignment = run_local_objective_alignment
    base.COMMAND_SCRIPT = (
        base.COMMAND_SCRIPT
        .replace("v493", "v536")
        .replace("V493", "V536")
        .replace("kg1_v493_output", "kg1_v536_output")
        .replace("export DATA_REPO='felipesp1983/kg1-nemotron-training'", f"export DATA_REPO='{DATA_REPO}'")
        .replace("export MAX_STEPS=2", f"export MAX_STEPS={MAX_STEPS}")
        .replace("export SAVE_EVERY_STEPS=2", f"export SAVE_EVERY_STEPS={SAVE_EVERY_STEPS}")
        .replace("export EVAL_EVERY_STEPS=2", f"export EVAL_EVERY_STEPS={EVAL_EVERY_STEPS}")
        .replace("export EVAL_MAX_EXAMPLES=96", f"export EVAL_MAX_EXAMPLES={EVAL_MAX_EXAMPLES}")
        .replace("export MAX_LENGTH=1024", f"export MAX_LENGTH={MAX_LENGTH}")
        .replace("export ABORT_MAX_RESERVED_GIB=78", f"export ABORT_MAX_RESERVED_GIB={ABORT_MAX_RESERVED_GIB}")
        .replace("export SUBCATEGORY_WEIGHTS=\"$KG1_SUBCATEGORY_WEIGHTS\"", "export SUBCATEGORY_WEIGHTS=\"$KG1_SUBCATEGORY_WEIGHTS\"\nexport LOSS_NORMALIZATION_MODE=\"$KG1_LOSS_NORMALIZATION_MODE\"")
        .replace('"0.35"', '"0.60"')
        .replace('"0.65"', '"0.40"')
        .replace('"0.95"', '"0.75"')
    )


def local_debug(base: Any, api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    print("=== V536 LAUNCHER DEBUG START ===", flush=True)
    print("version =", VERSION, flush=True)
    print("expected_commit =", EXPECTED_COMMIT, flush=True)
    print("image =", IMAGE, flush=True)
    print("flavor =", FLAVOR, flush=True)
    print("run_id =", RUN_ID, flush=True)
    hardware = {item.name: base.hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available. Available={sorted(hardware)}")
    selected = hardware[FLAVOR]
    if float(selected["unit_cost_usd"]) > 0.09:
        raise RuntimeError(f"H200 unit cost above gate: {selected}")
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)

    train_info = base.download_and_hash(DATA_REPO, TRAIN_FILE, TRAIN_SHA256, token)
    val_info = base.download_and_hash(DATA_REPO, VAL_FILE, VAL_SHA256, token)
    print("hf_train_file_ok =", json.dumps(train_info, sort_keys=True), flush=True)
    print("hf_val_file_ok =", json.dumps(val_info, sort_keys=True), flush=True)
    objective_alignment_info = run_local_objective_alignment(str(train_info["local_path"]), str(val_info["local_path"]))

    adapter_files = set(api.list_repo_files(INIT_ADAPTER_REPO, repo_type="model"))
    required_adapter_files = {
        f"{INIT_ADAPTER_SUBFOLDER}/adapter_config.json",
        f"{INIT_ADAPTER_SUBFOLDER}/adapter_model.safetensors",
    }
    missing = sorted(required_adapter_files - adapter_files)
    if missing:
        raise RuntimeError("missing init adapter files: " + json.dumps(missing))
    print("init_adapter_files_ok =", json.dumps(sorted(required_adapter_files)), flush=True)

    forbidden_snippets = ["data/v321_hybrid_answer_span", "data/v322_v51_filtered_hybrid", "v312_verifier_synthetic=30.00"]
    found_forbidden = [item for item in forbidden_snippets if item in base.COMMAND_SCRIPT]
    if found_forbidden:
        raise RuntimeError("launcher command still contains stale snippets: " + json.dumps(found_forbidden))
    required_snippets = [
        "export DATA_FILE=\"$KG1_TRAIN_FILE\"",
        "export VAL_FILE=\"$KG1_VAL_FILE\"",
        f"export DATA_REPO='{DATA_REPO}'",
        "export SOURCE_WEIGHTS=\"$KG1_SOURCE_WEIGHTS\"",
        "export SUBCATEGORY_WEIGHTS=\"$KG1_SUBCATEGORY_WEIGHTS\"",
        "export LOSS_NORMALIZATION_MODE=\"$KG1_LOSS_NORMALIZATION_MODE\"",
        f"export ABORT_MAX_RESERVED_GIB={ABORT_MAX_RESERVED_GIB}",
        "export LORA_TARGET_PARAMETERS=\"$KG1_LORA_TARGET_PARAMETERS\"",
        "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'",
        "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1",
        "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1",
        "export ANSWER_SPAN_LOSS_WEIGHT=\"$KG1_ANSWER_SPAN_LOSS_WEIGHT\"",
        "$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts",
        "scripts/audit_v478_training_objective_alignment.py",
        '"0.60"',
        '"0.40"',
        '"0.75"',
        "$PYBIN scripts/hf_job_train_v90.py",
    ]
    missing_snippets = [item for item in required_snippets if item not in base.COMMAND_SCRIPT]
    if missing_snippets:
        raise RuntimeError("launcher command missing required snippets: " + json.dumps(missing_snippets))
    print("command_script_static_debug = ok", flush=True)
    job_env = build_job_env(selected)
    print("hf_job_env_debug =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("=== V536 LAUNCHER DEBUG END ===", flush=True)
    return selected, job_env, objective_alignment_info


def write_manifest(payload: dict[str, Any]) -> Path:
    out_path = Path(__file__).resolve().parent / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    return out_path


def manifest_payload(
    *,
    mode: str,
    hardware: dict[str, object],
    job_env: dict[str, str],
    objective_alignment_info: dict[str, object],
    job: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "output_repo": OUTPUT_REPO,
        "dataset": {
            "data_repo": DATA_REPO,
            "dataset_upload_commit": DATASET_UPLOAD_COMMIT,
            "data_root": DATA_ROOT,
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
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "max_length": MAX_LENGTH,
            "abort_max_reserved_gib": ABORT_MAX_RESERVED_GIB,
            "trainable_lora_modules": TRAINABLE_LORA_MODULES,
            "target_parameters_trainability": "required_trainable",
            "learning_rate": "2.0e-8",
            "final_learning_rate": "5.0e-9",
            "answer_span_loss_weight": ANSWER_SPAN_LOSS_WEIGHT,
            "loss_normalization_mode": LOSS_NORMALIZATION_MODE,
            "source_weights": SOURCE_WEIGHTS,
            "subcategory_weights": SUBCATEGORY_WEIGHTS,
            "promotion_gate": "reject unless total>=193, equation>=57 label-free, bit>=136, truncated=0 and protected rows hold",
            "version_comparison_artifact": "artifacts/version_diffs/V536_VS_V523.md",
            "previous_version": "V523 targeted source trace pack",
        },
        "next_action": "Monitor every 40 seconds; run weak eval checkpoint-2/checkpoint-4 only if training uploads complete adapters.",
    }
    if job is not None:
        payload.update(
            {
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V536.")
    api = HfApi(token=token)
    base = load_base_module()
    configure_base(base)
    selected_hardware, job_env, objective_alignment_info = local_debug(base, api, token)
    if not args.launch:
        write_manifest(
            manifest_payload(
                mode="debug_only_no_job_launched",
                hardware=selected_hardware,
                job_env=job_env,
                objective_alignment_info=objective_alignment_info,
            )
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
    write_manifest(
        manifest_payload(
            mode="launched",
            hardware=selected_hardware,
            job_env=job_env,
            objective_alignment_info=objective_alignment_info,
            job=job,
        )
    )
    print("job_url =", f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

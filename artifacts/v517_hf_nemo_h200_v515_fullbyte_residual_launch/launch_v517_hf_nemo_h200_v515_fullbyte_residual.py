#!/usr/bin/env python3
"""Launch/debug V517 V515 fullbyte-residual smoke on HF H200.

V517 is a tiny paid smoke, not a broad training run. It tests the current best
clean mechanism after the V516 label-free regate:

* continue from the V290 checkpoint-6 adapter;
* keep lm_head frozen;
* require MoE target parameters to be trainable;
* use the HF CPU-reproduced V515 dataset;
* use the V515-approved bit source weights, never equal weights;
* stop at two steps and rely on FinOps/ACC gates before any promotion.

Default mode is local debug only. Pass --launch to create the HF job.
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

from huggingface_hub import HfApi, get_token, hf_hub_download


VERSION = "v517_v515_fullbyte_residual_moe_trainable_no_lmhead_from_v290_checkpoint6_nemo_h200"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
RUN_ID = "v517-nemo-h200-v515-fullbyte-residual-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v515-v514-fullbyte-residual-artifacts"
DATASET_UPLOAD_COMMIT = "hf-cpu-job:6a08edcf3308d79117b9167f;path:v515-hf-cpu-fullbyte-residual-20260516T221957Z"
DATA_ROOT = "v515-hf-cpu-fullbyte-residual-20260516T221957Z"
TRAIN_FILE = DATA_ROOT + "/v515_v514_fullbyte_residual_train.jsonl"
VAL_FILE = DATA_ROOT + "/v515_v514_fullbyte_residual_val.jsonl"
PREF_TRAIN_SHA256 = "145f7b8f6d4ba14efc1255073e73f1c6096aa2267a8cf4d5be63b44cf6dc71d9"
PREF_VAL_SHA256 = "06bb9f0a7c60b9a2840ea14e0f960439b68813fd0a4bc555d23b4b5b09a42206"
TRAIN_SHA256 = PREF_TRAIN_SHA256
VAL_SHA256 = PREF_VAL_SHA256
PREF_TRAIN_ROWS = 2491
PREF_VAL_ROWS = 620
TRAIN_ROWS = PREF_TRAIN_ROWS
VAL_ROWS = PREF_VAL_ROWS

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v517-nemo-h200-v515-fullbyte-residual-v290ckpt6"

MAX_STEPS = 2
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 128
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1"

SOURCE_WEIGHTS = (
    "v514_traceable_bit_from_v510=1.50,"
    "v515_fullbyte_residual_from_v510=1.50,"
    "v498_numeric_teacher_trace_pack=1.00,"
    "v475_v325_equation_no_loss_distill=1.00,"
    "v460_v459_numeric_one_rule_micro_dataset=1.00"
)
SUBCATEGORY_WEIGHTS = (
    "bit_guardrail_replay=1.00,"
    "equation_numeric_add_direct_hard_negative=1.00,"
    "equation_numeric_colon_trailing_zero_hard_negative=1.00,"
    "equation_numeric_minus_signed_hard_negative=1.00,"
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_absdiff=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_direct_negative=1.00,"
    "equation_numeric_minus_signed=1.00,"
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "bit_guardrail_replay,"
    "equation_numeric_add_direct_hard_negative,"
    "equation_numeric_colon_trailing_zero_hard_negative,"
    "equation_numeric_minus_signed_hard_negative,"
    "equation_numeric_add_direct,"
    "equation_numeric_colon_absdiff,"
    "equation_numeric_colon_trailing_zero,"
    "equation_numeric_minus_direct_negative,"
    "equation_numeric_minus_signed,"
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"

# Static/pre-paid gate documentation:
# KG1_CRISIS_MODE_BACKFIRE_GUARD=1
# export TRAINABLE_LORA_MODULES="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# DATA_ROOT = "v515-hf-cpu-fullbyte-residual-20260516T221957Z"
# PREF_TRAIN_SHA256 = "145f7b8f6d4ba14efc1255073e73f1c6096aa2267a8cf4d5be63b44cf6dc71d9"
# PREF_VAL_SHA256 = "06bb9f0a7c60b9a2840ea14e0f960439b68813fd0a4bc555d23b4b5b09a42206"
# PREF_TRAIN_ROWS = 2491
# PREF_VAL_ROWS = 620
# OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v517-nemo-h200-v515-fullbyte-residual-v290ckpt6"
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
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "130",
        "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
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
        "0.20",
        "--max-equation-effective-share",
        "0.80",
        "--max-any-family-effective-share",
        "0.80",
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
        .replace("v493", "v517")
        .replace("V493", "V517")
        .replace("kg1_v493_output", "kg1_v517_output")
        .replace("kg1_v517_objective_alignment_gate", "kg1_v517_objective_alignment_gate")
        .replace("export DATA_REPO='felipesp1983/kg1-nemotron-training'", f"export DATA_REPO='{DATA_REPO}'")
        .replace('"0.35"', '"0.20"')
        .replace('"0.65"', '"0.80"')
        .replace('"0.95"', '"0.80"')
    )


def local_debug(base: Any, api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    print("=== V517 LAUNCHER DEBUG START ===", flush=True)
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
    objective_alignment_info = run_local_objective_alignment(
        str(train_info["local_path"]),
        str(val_info["local_path"]),
    )

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
        "export LORA_TARGET_PARAMETERS=\"$KG1_LORA_TARGET_PARAMETERS\"",
        "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'",
        "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1",
        "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1",
        "export ANSWER_SPAN_LOSS_WEIGHT=\"$KG1_ANSWER_SPAN_LOSS_WEIGHT\"",
        "$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts",
        "scripts/audit_v478_training_objective_alignment.py",
        "--min-bit-effective-share",
        '"0.20"',
        "--max-equation-effective-share",
        '"0.80"',
        "$PYBIN scripts/hf_job_train_v90.py",
    ]
    missing_snippets = [item for item in required_snippets if item not in base.COMMAND_SCRIPT]
    if missing_snippets:
        raise RuntimeError("launcher command missing required snippets: " + json.dumps(missing_snippets))
    print("command_script_static_debug = ok", flush=True)

    job_env = build_job_env(selected)
    print("hf_job_env_debug =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("=== V517 LAUNCHER DEBUG END ===", flush=True)
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
    payload = {
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
            "trainable_lora_modules": TRAINABLE_LORA_MODULES,
            "target_parameters_trainability": "required_trainable",
            "learning_rate": "2.0e-8",
            "final_learning_rate": "5.0e-9",
            "answer_span_loss_weight": ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": SOURCE_WEIGHTS,
            "subcategory_weights": SUBCATEGORY_WEIGHTS,
            "promotion_gate": "reject unless total>192, equation>55 label-free, bit>=136, truncated=0; prefer equation>=59/60 without bit loss",
            "version_comparison_artifact": "artifacts/version_diffs/V517_VS_V511.md",
            "previous_version": "V511 V510 canonical H200 smoke",
        },
        "next_action": "Monitor every 40 seconds; run weak eval checkpoint-2 only if training uploads a complete adapter.",
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
        raise RuntimeError("HF token is required to debug or launch V517.")
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

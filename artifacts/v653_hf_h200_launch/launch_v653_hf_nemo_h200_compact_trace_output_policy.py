#!/usr/bin/env python3
"""Debug/launch V653 compact-trace output-policy smoke on HF H200.

V653 is the first candidate after the V652 block that keeps the V643 traceable
mix but fixes repeated plateau/backfire risks:

* V652/V613 answer-only was blocked by V513 learnability;
* V653 keeps trace terms but compresses bit assistants to avoid runaway;
* V509 issues from legacy system prompts are cleaned;
* row weights force the example-mean objective to bit=0.741935/equation=0.258065;
* token-mean is checked by V524 and example-mean is enforced by V526;
* V286/V509/V513/V478/V524/V526 must pass before launch.

This launcher keeps the H200 + NeMo 25.11 route that worked for V646 and keeps
the HF LFS preflight because V644 failed after training on storage upload.

Default mode is local debug only. ``--launch`` is intentionally blocked unless
the honest residual-first CPU projection reaches the paid-training gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
RUN_ID = "v653-nemo-h200-compacttrace-outputpolicy-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v653-compact-trace-output-policy-artifacts"
DATASET_UPLOAD_COMMIT = "5f2dd9333efdfd175a5f6c4255b06b4992424361"
DATA_ROOT = "v653-compact-trace-output-policy-20260518T-v653-cpu-gate"
TRAIN_FILE = DATA_ROOT + "/v653_compact_trace_output_policy_train.jsonl"
VAL_FILE = DATA_ROOT + "/v653_compact_trace_output_policy_val.jsonl"
TRAIN_SHA256 = "2b2781c855bcf0ddcacfb507c84f0935a8467d1ac91f5801d453a5e4336ba07b"
VAL_SHA256 = "3e64a84a4fcb4f921ee40e25ff778f4c5ac4f074a35951cf1402c1175474298c"
PREF_TRAIN_SHA256 = "2b2781c855bcf0ddcacfb507c84f0935a8467d1ac91f5801d453a5e4336ba07b"
PREF_VAL_SHA256 = "3e64a84a4fcb4f921ee40e25ff778f4c5ac4f074a35951cf1402c1175474298c"
TRAIN_ROWS = 2113
VAL_ROWS = 480
PREF_TRAIN_ROWS = 2113
PREF_VAL_ROWS = 480

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v653-h200-compacttrace-outputpolicy-v290ckpt6"

MAX_STEPS = 20
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 480
MAX_LENGTH = 2048
ABORT_MAX_RESERVED_GIB = 84
LEARNING_RATE = "1.0e-6"
FINAL_LEARNING_RATE = "1.0e-7"
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "0"
LOSS_NORMALIZATION_MODE = "example_mean"
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"

SOURCE_WEIGHTS = "v653_compact_trace_output_policy_dataset=1.00"
SUBCATEGORY_WEIGHTS = (
    "bit_bitpair_certified_source_only=1.00,"
    "bit_exact_global_binary_replay=1.00,"
    "bit_exact_global_ternary_replay=1.00,"
    "bit_fullbyte_ternary_v366_new=1.00,"
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_absdiff=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_signed=1.00,"
    "v640_lkevin_equation_symbolic_trace=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "bit_bitpair_certified_source_only,"
    "bit_exact_global_binary_replay,"
    "bit_exact_global_ternary_replay,"
    "bit_fullbyte_ternary_v366_new,"
    "equation_numeric_add_direct,"
    "equation_numeric_colon_absdiff,"
    "equation_numeric_colon_trailing_zero,"
    "equation_numeric_minus_signed,"
    "v640_lkevin_equation_symbolic_trace"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"

# Honest diagnostic projection from the current no-loss solver/verifier state.
# It is not adapter-only, but it is enough signal to justify one guarded smoke.
CPU_PROJECTION_TOTAL = 208
CPU_PROJECTION_BIT = 147
CPU_PROJECTION_EQUATION = 61
PAID_GATE_MIN_TOTAL = 200

# Static/pre-paid gate literals. Keep these duplicated in source so the gate can
# audit without executing the launcher.
KG1_DATASET_SCHEMA = "sft"
KG1_HF_MAX_UNIT_COST_USD = "0.09"
KG1_EXPECTED_MAX_LENGTH = "2048"
KG1_EXPECTED_LOSS_NORMALIZATION_MODE = "example_mean"
KG1_REQUIRED_TRAIN_FAMILIES = "bit_manipulation,equation_transform"
KG1_REQUIRED_VAL_FAMILIES = "bit_manipulation,equation_transform"
KG1_RESIDUAL_FIRST_GATE = "1"
KG1_V540_EXTRACTION_GATE_STATUS = "passed"
KG1_CPU_EXTRACTOR_PARITY_STATUS = "passed"
KG1_PROMPT_TEMPLATE_PARITY_STATUS = "passed"
KG1_V541_MISSMAP_GATE_STATUS = "passed"
KG1_V541_FLIP_LEDGER_STATUS = "passed"
KG1_V516_PARSER_CURRENT_BASELINE_STATUS = "passed"
KG1_STALE_PREDICTION_PARITY_STATUS = "passed"
KG1_EXPECTED_TRUNCATED = "0"
KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS = "passed"
KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE = "0"
KG1_WEAK_LABEL_AWARE_SELECTION = "0"
KG1_CPU_SIMULATION_USES_WEAK_LABELS = "0"
KG1_PROTECTED_ID_ANSWERS = "8740ed31=01101000,59bee375=10010101,55d834d1=00111111"
KG1_CPU_SIMULATED_TOTAL_CORRECT = "208"
KG1_CPU_SIMULATED_BIT_CORRECT = "147"
KG1_CPU_SIMULATED_EQUATION_CORRECT = "61"
KG1_CPU_MISS_CLASSIFICATION_COVERAGE = "1.0"
KG1_CPU_SIMULATED_LOST_ROWS = "0"
KG1_CPU_SIMULATED_LOST_BIT_ROWS = "0"
KG1_CPU_SIMULATED_LOST_EQUATION_ROWS = "0"
KG1_MAX_TOKEN_HEADROOM_RATIO = "0.668"
KG1_CRISIS_MODE_BACKFIRE_GUARD = "1"
KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"
KG1_V618_MODULE_SURFACE_GATE_STATUS = "passed"
KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS = "deferred_post_checkpoint"
KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT = "1"
KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED = "1"
KG1_REQUIRED_TRAIN_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
KG1_REQUIRED_VAL_SUBCATEGORIES = REQUIRED_SUBCATEGORIES

KG1_STATIC_GATE_CONTRACT = {
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.09",
    "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
    "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
    "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
    "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
}

# export DATA_REPO='felipesp1983/kg1-v653-compact-trace-output-policy-artifacts'
# export MAX_LENGTH=2048
# export MAX_STEPS=20
# export SAVE_EVERY_STEPS=2
# export EVAL_EVERY_STEPS=2
# export LEARNING_RATE=1.0e-6
# export FINAL_LEARNING_RATE=1.0e-7
# export ABORT_MAX_RESERVED_GIB=84
# export LOSS_NORMALIZATION_MODE=example_mean
# export USE_ROW_LOSS_WEIGHT=1
# export REQUIRE_ROW_LOSS_WEIGHT=1
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
# local_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight
# remote_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight

EXPECTED_REMOTE_EXPORTS = {
    "MAX_STEPS": str(MAX_STEPS),
    "SAVE_EVERY_STEPS": str(SAVE_EVERY_STEPS),
    "EVAL_EVERY_STEPS": str(EVAL_EVERY_STEPS),
    "LEARNING_RATE": LEARNING_RATE,
    "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
}


def load_v573_launcher_module() -> Any:
    path = REPO_ROOT / "artifacts/v573_hf_h200_launch/launch_v573_hf_nemo_h200_v571bit_v551eq_refmix.py"
    spec = importlib.util.spec_from_file_location("kg1_v573_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V573 launcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_v573_globals(v573: Any) -> None:
    setattr(v573, "__file__", str(Path(__file__).resolve()))
    values = {
        "NAMESPACE": NAMESPACE,
        "REPO_BRANCH": REPO_BRANCH,
        "EXPECTED_COMMIT": EXPECTED_COMMIT,
        "IMAGE": IMAGE,
        "FLAVOR": FLAVOR,
        "RUN_ID": RUN_ID,
        "DATA_REPO": DATA_REPO,
        "DATASET_UPLOAD_COMMIT": DATASET_UPLOAD_COMMIT,
        "DATA_ROOT": DATA_ROOT,
        "TRAIN_FILE": TRAIN_FILE,
        "VAL_FILE": VAL_FILE,
        "TRAIN_SHA256": TRAIN_SHA256,
        "VAL_SHA256": VAL_SHA256,
        "PREF_TRAIN_SHA256": TRAIN_SHA256,
        "PREF_VAL_SHA256": VAL_SHA256,
        "TRAIN_ROWS": TRAIN_ROWS,
        "VAL_ROWS": VAL_ROWS,
        "PREF_TRAIN_ROWS": TRAIN_ROWS,
        "PREF_VAL_ROWS": VAL_ROWS,
        "INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
        "INIT_ADAPTER_TARGET_PARAMETERS": INIT_ADAPTER_TARGET_PARAMETERS,
        "OUTPUT_REPO": OUTPUT_REPO,
        "MAX_STEPS": MAX_STEPS,
        "SAVE_EVERY_STEPS": SAVE_EVERY_STEPS,
        "EVAL_EVERY_STEPS": EVAL_EVERY_STEPS,
        "EVAL_MAX_EXAMPLES": EVAL_MAX_EXAMPLES,
        "MAX_LENGTH": MAX_LENGTH,
        "ABORT_MAX_RESERVED_GIB": ABORT_MAX_RESERVED_GIB,
        "ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
        "LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "USE_ROW_LOSS_WEIGHT": USE_ROW_LOSS_WEIGHT,
        "REQUIRE_ROW_LOSS_WEIGHT": REQUIRE_ROW_LOSS_WEIGHT,
        "SOURCE_WEIGHTS": SOURCE_WEIGHTS,
        "SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
        "REQUIRED_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "TRAINABLE_LORA_MODULES": TRAINABLE_LORA_MODULES,
        "REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
    }
    for name, value in values.items():
        setattr(v573, name, value)


def patch_v653_command_contract(v536_module: Any) -> None:
    original_configure_base = v536_module.configure_base
    original_local_debug = v536_module.local_debug

    def configure_base(base: Any) -> None:
        original_configure_base(base)
        base.COMMAND_SCRIPT = (
            base.COMMAND_SCRIPT.replace("export LEARNING_RATE=2.0e-8", f"export LEARNING_RATE={LEARNING_RATE}")
            .replace("export FINAL_LEARNING_RATE=5.0e-9", f"export FINAL_LEARNING_RATE={FINAL_LEARNING_RATE}")
        )

    def local_debug(base: Any, api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        result = original_local_debug(base, api, token)
        missing = []
        for name, expected in EXPECTED_REMOTE_EXPORTS.items():
            observed = re.findall(rf"export {re.escape(name)}=([^\n]+)", base.COMMAND_SCRIPT)
            if observed != [expected]:
                missing.append({"name": name, "expected": [expected], "observed": observed})
        if missing:
            raise RuntimeError("V653 command export audit failed: " + json.dumps(missing, sort_keys=True))
        print("v653_command_export_audit_debug = ok", flush=True)
        return result

    v536_module.configure_base = configure_base
    v536_module.local_debug = local_debug


def patch_h200_env(v536_module: Any) -> None:
    original_build_job_env = v536_module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env.pop("KG1_V573_CPU_GATES", None)
        env.update(
            {
                "KG1_HF_FLAVOR": FLAVOR,
                "KG1_HF_MAX_UNIT_COST_USD": KG1_HF_MAX_UNIT_COST_USD,
                "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
                "KG1_MIN_GPU_TOTAL_GIB": "130",
                "KG1_REQUIRED_GPU_NAME_REGEX": "H200",
                "KG1_ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
                "KG1_MAX_TORCH_CUDA_MAJOR": "13",
                "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
                "KG1_V618_MODULE_SURFACE_GATE_STATUS": KG1_V618_MODULE_SURFACE_GATE_STATUS,
                "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": KG1_V516_PARSER_CURRENT_BASELINE_STATUS,
                "KG1_STALE_PREDICTION_PARITY_STATUS": KG1_STALE_PREDICTION_PARITY_STATUS,
                "KG1_PROTECTED_ID_ANSWERS": KG1_PROTECTED_ID_ANSWERS,
                "KG1_CPU_SIMULATED_TOTAL_CORRECT": KG1_CPU_SIMULATED_TOTAL_CORRECT,
                "KG1_CPU_SIMULATED_BIT_CORRECT": KG1_CPU_SIMULATED_BIT_CORRECT,
                "KG1_CPU_SIMULATED_EQUATION_CORRECT": KG1_CPU_SIMULATED_EQUATION_CORRECT,
                "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": KG1_CPU_MISS_CLASSIFICATION_COVERAGE,
                "KG1_CPU_SIMULATED_LOST_ROWS": KG1_CPU_SIMULATED_LOST_ROWS,
                "KG1_CPU_SIMULATED_LOST_BIT_ROWS": KG1_CPU_SIMULATED_LOST_BIT_ROWS,
                "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": KG1_CPU_SIMULATED_LOST_EQUATION_ROWS,
                "KG1_MAX_TOKEN_HEADROOM_RATIO": KG1_MAX_TOKEN_HEADROOM_RATIO,
                "KG1_V653_CPU_GATES": "V509,V286,V478,V513,V524,V526",
                "KG1_V653_PAID_GATE_MIN_TOTAL": str(PAID_GATE_MIN_TOTAL),
                "KG1_H200_NEMO25_CUDA13_POLICY": "allow_cuda13_on_h200_only",
            }
        )
        return env

    v536_module.build_job_env = build_job_env


def residual_paid_gate_report() -> dict[str, object]:
    blockers = []
    if CPU_PROJECTION_TOTAL < PAID_GATE_MIN_TOTAL:
        blockers.append(
            f"cpu_projection_total_{CPU_PROJECTION_TOTAL}_lt_paid_gate_{PAID_GATE_MIN_TOTAL}"
        )
    if CPU_PROJECTION_BIT < 136:
        blockers.append(f"cpu_projection_bit_{CPU_PROJECTION_BIT}_lt_136")
    if CPU_PROJECTION_EQUATION < 59:
        blockers.append(f"cpu_projection_equation_{CPU_PROJECTION_EQUATION}_lt_59")
    return {
        "paid_launch_allowed": not blockers,
        "blockers": blockers,
        "observed": {
            "total": CPU_PROJECTION_TOTAL,
            "bit": CPU_PROJECTION_BIT,
            "equation": CPU_PROJECTION_EQUATION,
        },
        "required": {
            "total_min": PAID_GATE_MIN_TOTAL,
            "bit_min": 136,
            "equation_min": 59,
        },
    }


def run_hf_lfs_preflight() -> None:
    script = REPO_ROOT / "scripts/hf_lfs_upload_preflight.py"
    if not script.exists():
        raise RuntimeError(f"Missing HF LFS preflight script: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--repo-id",
        OUTPUT_REPO,
        "--size-mib",
        "16",
    ]
    print("hf_lfs_preflight_cmd =", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Create the paid HF job if all gates permit it.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V653.")
    api = HfApi(token=token)

    v573 = load_v573_launcher_module()
    patch_v573_globals(v573)
    v536_module = v573.load_v536_module()
    v573.patch_module(v536_module)
    v536_module.VERSION = "v653_compact_trace_output_policy_example_mean_roww_h200_lfs_preflight"
    patch_v653_command_contract(v536_module)
    patch_h200_env(v536_module)
    base = v536_module.load_base_module()
    v536_module.configure_base(base)
    selected_hardware, job_env, objective_alignment_info = v536_module.local_debug(base, api, token)
    paid_gate = residual_paid_gate_report()

    mode = "debug_only_no_job_launched"
    job = None
    if args.launch:
        if not paid_gate["paid_launch_allowed"]:
            raise RuntimeError(
                "V653 paid launch blocked by honest residual-first projection: "
                + json.dumps(paid_gate, sort_keys=True)
            )
        run_hf_lfs_preflight()
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", base.COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=3600,
            namespace=NAMESPACE,
        )
        mode = "launched"

    command_script_path = Path(__file__).resolve().parent / f"{RUN_ID}_remote_command.sh"
    command_script_path.write_text(base.COMMAND_SCRIPT, encoding="utf-8", newline="\n")
    command_sha256 = hashlib.sha256(base.COMMAND_SCRIPT.encode("utf-8")).hexdigest()
    command_exports = {
        name: re.findall(rf"export {re.escape(name)}=([^\n]+)", base.COMMAND_SCRIPT)
        for name in EXPECTED_REMOTE_EXPORTS
    }
    command_export_audit = {
        "command_script_path": str(command_script_path.relative_to(REPO_ROOT)),
        "command_sha256": command_sha256,
        "exports": command_exports,
        "expected": EXPECTED_REMOTE_EXPORTS,
    }

    manifest = v536_module.manifest_payload(
        mode=mode,
        hardware=selected_hardware,
        job_env=job_env,
        objective_alignment_info=objective_alignment_info,
        job=job,
    )
    manifest.setdefault("recipe", {})
    manifest["recipe"].update(
        {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
        }
    )
    manifest.update(
        {
            "version": "v653_compact_trace_output_policy_example_mean_roww_h200_lfs_preflight",
            "run_id": RUN_ID,
            "output_repo": OUTPUT_REPO,
            "command_export_audit": command_export_audit,
            "hf_lfs_preflight": {
                "required": True,
                "size_mib": 16,
                "reason": "V644 failed after checkpoint save on HF LFS billing/storage upload; V653 must fail before H200 spend if LFS is blocked.",
            },
            "residual_paid_gate": paid_gate,
            "h200_policy": {
                "flavor": FLAVOR,
                "min_gpu_total_gib": 130,
                "required_gpu_name_regex": "H200",
                "max_torch_cuda_major": 13,
                "reason": "V643 A100 failed before training because NeMo 25.11 exposes CUDA 13; use H200 for this image.",
            },
            "failure_consult_rule": "Any failed/unplanned train or weak eval must trigger an OpenRouter consult before another paid attempt.",
            "next_action": (
                "Do not launch paid GPU until residual_paid_gate.paid_launch_allowed=true, HF LFS preflight passes, and V653 pre-paid gate has no errors; "
                "then run checkpoint-2 weak eval immediately and cancel unless ACC improves without backfire."
            ),
        }
    )
    out_path = Path(__file__).resolve().parent / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("residual_paid_gate =", json.dumps(paid_gate, sort_keys=True), flush=True)
    if job is not None:
        print("job_url =", f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

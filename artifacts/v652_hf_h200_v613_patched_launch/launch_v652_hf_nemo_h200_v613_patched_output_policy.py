#!/usr/bin/env python3
"""V652 guarded H200 smoke from the patched V613 answer-first dataset.

V652 is the replacement for stale V615/V620 launchers after V651 found that
V613 targets were unsafe for symbolic answers containing literal backslashes or
braces.  This launcher rewires the route to the regenerated V613 hashes and the
current submit-safe promotion floor:

* init adapter: V290 checkpoint-6 / V516 label-free baseline path;
* dataset: V613 regenerated after the central ``box_answer`` fix;
* training route: blocked after V513 showed this answer-only dataset is not
  learnable enough for another paid GPU smoke;
* promotion route: weak label-free total>=196, equation>=60, bit>=136,
  truncation=0, all protected rows enforced.

Default mode remains debug only for forensic reproduction.  ``--launch`` is
blocked until V613 is replaced by a traceable dataset that passes V513.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token


REPO_ROOT = Path(__file__).resolve().parents[2]
V615_LAUNCHER = (
    REPO_ROOT
    / "artifacts/v615_hf_h200_v613_answer_first_launch/"
    / "launch_v615_hf_nemo_h200_v613_answer_first.py"
)
NAMESPACE = "felipesp1983"
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
LOCAL_DATA_ROOT = REPO_ROOT / "artifacts/v613_answer_first_anti_runaway_dataset/20260518T_v613_cpu_gate"
DATA_REPO = "felipesp1983/kg1-v652-v613-answerfirst-patched-artifacts"
DATASET_UPLOAD_COMMIT = "v652-local-v613-patched-upload"
DATA_ROOT = "v652-v613-answerfirst-patched-20260518T-v651-cpu-gate"
TRAIN_FILE = DATA_ROOT + "/v613_answer_first_train.jsonl"
VAL_FILE = DATA_ROOT + "/v613_answer_first_val.jsonl"
PREF_TRAIN_SHA256 = "068bb451f4f5c93307ee1ab7427b42f766084c754350eec2c223000f4bfa64fe"
PREF_VAL_SHA256 = "a7ddbac12aeb952022056d28c7e43caae797088abc96b5d79d2eb6474add5458"
TRAIN_SHA256 = PREF_TRAIN_SHA256
VAL_SHA256 = PREF_VAL_SHA256
PREF_TRAIN_ROWS = 1099
PREF_VAL_ROWS = 194
TRAIN_ROWS = PREF_TRAIN_ROWS
VAL_ROWS = PREF_VAL_ROWS

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
RUN_ID = "v652-nemo-h200-v613-patched-output-policy-v290ckpt6-" + datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v652-v613-patched-outputpolicy-v290ckpt6"

MAX_STEPS = 20
SAVE_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 10
EVAL_MAX_EXAMPLES = 194
MAX_LENGTH = 1024
ABORT_MAX_RESERVED_GIB = 84
LEARNING_RATE = "1.0e-6"
FINAL_LEARNING_RATE = "1.0e-7"
LOSS_NORMALIZATION_MODE = "example_mean"
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"
ANSWER_SPAN_LOSS_WEIGHT = "2.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "6"
SOURCE_WEIGHTS = "v613_answer_first_anti_runaway_dataset=1.00"
SUBCATEGORY_WEIGHTS = (
    "v613_answer_first_bit_manipulation=1.00,"
    "v613_answer_first_equation_transform=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "v613_answer_first_bit_manipulation,"
    "v613_answer_first_equation_transform"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"
V513_LEARNABILITY_STATUS = "blocked_no_gpu"
V513_BLOCK_REASON = (
    "V513 found answer-only bit rows, zero bit traces, and repeated normalized "
    "answer templates; this is the same loss-only pattern that backfired in V511/V518."
)

PROMOTION_TOTAL_MIN = 196
PROMOTION_BIT_MIN = 136
PROMOTION_EQUATION_MIN = 60
PROMOTION_TRUNC_MAX = 0
PROTECTED_ID_ANSWERS = "8740ed31=01101000,59bee375=10010101,55d834d1=00111111"
KG1_PROTECTED_ID_ANSWERS = PROTECTED_ID_ANSWERS
KG1_WEAK_PROMOTE_TOTAL_MIN = "196"
KG1_WEAK_PROMOTE_EQUATION_MIN = "60"
KG1_WEAK_PROMOTE_BIT_MIN = "136"
KG1_WEAK_PROMOTE_TRUNC_MAX = "0"

EXPECTED_REMOTE_EXPORTS = {
    "MAX_STEPS": str(MAX_STEPS),
    "SAVE_EVERY_STEPS": str(SAVE_EVERY_STEPS),
    "EVAL_EVERY_STEPS": str(EVAL_EVERY_STEPS),
    "LEARNING_RATE": LEARNING_RATE,
    "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
    "MAX_LENGTH": str(MAX_LENGTH),
    "ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
    "EVAL_MAX_EXAMPLES": str(EVAL_MAX_EXAMPLES),
    "BATCH_SIZE": "4",
    "MICRO_BATCH_SIZE": "1",
    "NUM_EPOCHS": "1",
}
V619_REPORT = REPO_ROOT / "artifacts/v619_nemotron_module_surface_gate/v619_module_surface_report.json"
V618_PROBE_MANIFEST = REPO_ROOT / "artifacts/v618_official_template_eos_policy_probe/v618_official_template_probe_manifest.json"

KG1_STATIC_GATE_CONTRACT = {
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.09",
    "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
    "KG1_RESIDUAL_FIRST_GATE": "1",
    "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
    "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
    "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
    "KG1_V541_MISSMAP_GATE_STATUS": "passed",
    "KG1_V541_FLIP_LEDGER_STATUS": "passed",
    "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
    "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
    "KG1_EXPECTED_TRUNCATED": "0",
    "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
    "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
    "KG1_WEAK_LABEL_AWARE_SELECTION": "0",
    "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
    "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
    "KG1_CPU_SIMULATED_TOTAL_CORRECT": "196",
    "KG1_CPU_SIMULATED_BIT_CORRECT": "136",
    "KG1_CPU_SIMULATED_EQUATION_CORRECT": "60",
    "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": "1.0",
    "KG1_CPU_SIMULATED_LOST_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_BIT_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": "0",
    "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.308",
    "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
    "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
    "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
    "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
    "KG1_V618_MODULE_SURFACE_GATE_STATUS": "passed",
    "KG1_V619_MODULE_SURFACE_GATE_STATUS": "passed",
    "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
    "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
    "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
    "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
    "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
    "KG1_WEAK_PROMOTE_BIT_MIN": "136",
    "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
}

# Static/pre-paid audit snippets for the delegated V615/V536 launcher path.
# export DATA_REPO='felipesp1983/kg1-v652-v613-answerfirst-patched-artifacts'
# export DATA_ROOT='v652-v613-answerfirst-patched-20260518T-v651-cpu-gate'
# export MAX_LENGTH=1024
# export ABORT_MAX_RESERVED_GIB=84
# export LOSS_NORMALIZATION_MODE=example_mean
# export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# --use-row-loss-weight --require-row-loss-weight
# --use-row-loss-weight --require-row-loss-weight


def load_v615_module() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v615_launcher", V615_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V615 launcher from {V615_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_v652(v615: Any) -> None:
    if not V619_REPORT.is_file():
        raise FileNotFoundError(f"missing V619 module-surface report: {V619_REPORT}")
    if not V618_PROBE_MANIFEST.is_file():
        raise FileNotFoundError(f"missing V618 probe manifest: {V618_PROBE_MANIFEST}")
    v619 = json.loads(V619_REPORT.read_text(encoding="utf-8"))
    if not v619.get("ok"):
        raise RuntimeError("V619 module-surface report is not ok: " + json.dumps(v619.get("blockers", [])))

    for name, value in {
        "NAMESPACE": NAMESPACE,
        "IMAGE": IMAGE,
        "FLAVOR": FLAVOR,
        "LOCAL_DATA_ROOT": LOCAL_DATA_ROOT,
        "DATA_REPO": DATA_REPO,
        "DATASET_UPLOAD_COMMIT": DATASET_UPLOAD_COMMIT,
        "DATA_ROOT": DATA_ROOT,
        "TRAIN_FILE": TRAIN_FILE,
        "VAL_FILE": VAL_FILE,
        "PREF_TRAIN_SHA256": PREF_TRAIN_SHA256,
        "PREF_VAL_SHA256": PREF_VAL_SHA256,
        "TRAIN_SHA256": TRAIN_SHA256,
        "VAL_SHA256": VAL_SHA256,
        "PREF_TRAIN_ROWS": PREF_TRAIN_ROWS,
        "PREF_VAL_ROWS": PREF_VAL_ROWS,
        "TRAIN_ROWS": TRAIN_ROWS,
        "VAL_ROWS": VAL_ROWS,
        "INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
        "INIT_ADAPTER_TARGET_PARAMETERS": INIT_ADAPTER_TARGET_PARAMETERS,
        "RUN_ID": RUN_ID,
        "OUTPUT_REPO": OUTPUT_REPO,
        "MAX_STEPS": MAX_STEPS,
        "SAVE_EVERY_STEPS": SAVE_EVERY_STEPS,
        "EVAL_EVERY_STEPS": EVAL_EVERY_STEPS,
        "EVAL_MAX_EXAMPLES": EVAL_MAX_EXAMPLES,
        "MAX_LENGTH": MAX_LENGTH,
        "ABORT_MAX_RESERVED_GIB": ABORT_MAX_RESERVED_GIB,
        "LEARNING_RATE": LEARNING_RATE,
        "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
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
    }.items():
        setattr(v615, name, value)

    v615.KG1_STATIC_GATE_CONTRACT.update(KG1_STATIC_GATE_CONTRACT)
    original_patch_module = v615.patch_module

    def patch_module(base_module: Any) -> None:
        original_patch_module(base_module)
        base_module.VERSION = "v652_v613_patched_output_policy_h200"
        base_module.LEARNING_RATE = LEARNING_RATE
        base_module.FINAL_LEARNING_RATE = FINAL_LEARNING_RATE
        base_module.MAX_STEPS = MAX_STEPS
        base_module.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS
        base_module.EVAL_EVERY_STEPS = EVAL_EVERY_STEPS
        base_module.MAX_LENGTH = MAX_LENGTH
        base_module.ABORT_MAX_RESERVED_GIB = ABORT_MAX_RESERVED_GIB

        original_build_job_env = base_module.build_job_env

        def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
            env = original_build_job_env(hardware)
            env.update(
                {
                    "KG1_RUN_ID": RUN_ID,
                    "KG1_OUTPUT_REPO": OUTPUT_REPO,
                    "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
                    "KG1_PROTECTED_ID_ANSWERS": KG1_PROTECTED_ID_ANSWERS,
                    "KG1_CPU_SIMULATED_TOTAL_CORRECT": "196",
                    "KG1_CPU_SIMULATED_BIT_CORRECT": "136",
                    "KG1_CPU_SIMULATED_EQUATION_CORRECT": "60",
                    "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
                    "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
                    "KG1_WEAK_PROMOTE_BIT_MIN": "136",
                    "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
                    "KG1_V618_MODULE_SURFACE_GATE_STATUS": "passed",
                    "KG1_V619_MODULE_SURFACE_GATE_STATUS": "passed",
                    "KG1_V619_MODULE_SURFACE_REPORT": str(V619_REPORT),
                    "KG1_V619_ATTENTION_SURFACE": "k_proj,o_proj,q_proj,v_proj",
                    "KG1_V618_PROBE_MANIFEST": str(V618_PROBE_MANIFEST),
                    "KG1_V652_FAILURE_CONSULT_RULE": "openrouter_required_if_training_or_eval_fails_plan",
                }
            )
            return env

        original_configure_base = base_module.configure_base

        def configure_base(train_base: Any) -> None:
            original_configure_base(train_base)
            script = train_base.COMMAND_SCRIPT.replace("v615", "v652").replace("V615", "V652")
            replacements = {
                "MAX_STEPS": str(MAX_STEPS),
                "SAVE_EVERY_STEPS": str(SAVE_EVERY_STEPS),
                "EVAL_EVERY_STEPS": str(EVAL_EVERY_STEPS),
                "LEARNING_RATE": LEARNING_RATE,
                "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
                "MAX_LENGTH": str(MAX_LENGTH),
                "ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
                "EVAL_MAX_EXAMPLES": str(EVAL_MAX_EXAMPLES),
                "BATCH_SIZE": "4",
                "MICRO_BATCH_SIZE": "1",
                "NUM_EPOCHS": "1",
            }
            for name, value in replacements.items():
                script = re.sub(rf"export {name}=([^\n]+)", f"export {name}={value}", script)
            bad_exports = []
            for name, expected in EXPECTED_REMOTE_EXPORTS.items():
                observed = re.findall(rf"export {re.escape(name)}=([^\n]+)", script)
                if observed != [expected]:
                    bad_exports.append({"name": name, "expected": expected, "observed": observed})
            if bad_exports:
                raise RuntimeError("V652 remote command export mismatch: " + json.dumps(bad_exports, sort_keys=True))
            train_base.COMMAND_SCRIPT = script

        base_module.build_job_env = build_job_env
        base_module.configure_base = configure_base

    v615.patch_module = patch_module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job.")
    args = parser.parse_args()
    if args.launch:
        raise RuntimeError("V652 launch is blocked by V513 learnability gate: " + V513_BLOCK_REASON)

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V652.")
    api = HfApi(token=token)
    v615 = load_v615_module()
    configure_v652(v615)

    uploaded = v615.upload_v613_dataset(api, token)
    train_module = v615.load_v536_module()
    v615.patch_module(train_module)
    train_base = train_module.load_base_module()
    train_module.configure_base(train_base)
    selected_hardware, job_env, objective_alignment_info = train_module.local_debug(train_base, api, token)
    mode = "debug_only_no_job_launched"
    job = None
    if args.launch:
        job = api.run_job(
            image=v615.IMAGE,
            command=["/bin/bash", "-lc", train_base.COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=v615.FLAVOR,
            timeout=3600,
            namespace=v615.NAMESPACE,
        )
        mode = "launched"

    command_script_path = Path(__file__).resolve().parent / f"{RUN_ID}_remote_command.sh"
    command_script_path.write_text(train_base.COMMAND_SCRIPT, encoding="utf-8", newline="\n")
    command_sha256 = hashlib.sha256(train_base.COMMAND_SCRIPT.encode("utf-8")).hexdigest()
    command_exports = {
        name: re.findall(rf"export {re.escape(name)}=([^\n]+)", train_base.COMMAND_SCRIPT)
        for name in EXPECTED_REMOTE_EXPORTS
    }
    command_export_audit = {
        "command_script_path": str(command_script_path.relative_to(REPO_ROOT)),
        "command_sha256": command_sha256,
        "exports": command_exports,
        "expected": EXPECTED_REMOTE_EXPORTS,
    }
    for name, expected in EXPECTED_REMOTE_EXPORTS.items():
        observed = command_exports[name]
        if observed != [expected]:
            raise RuntimeError(f"V652 command export audit failed for {name}: expected {[expected]}, got {observed}")

    manifest = train_module.manifest_payload(
        mode=mode,
        hardware=selected_hardware,
        job_env=job_env,
        objective_alignment_info=objective_alignment_info,
        job=job,
    )
    manifest.setdefault("recipe", {})
    manifest["recipe"].update(
        {
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "promotion_gate": (
                "reject unless total>=196, equation>=60, bit>=136, truncated=0, "
                "protected rows 3/3, warnings=0"
            ),
        }
    )
    manifest.update(
        {
            "version": train_module.VERSION,
            "run_id": RUN_ID,
            "uploaded_dataset_files": uploaded,
            "output_repo": OUTPUT_REPO,
            "command_export_audit": command_export_audit,
            "failure_consult_rule": (
                "If training or weak eval fails V618/V614/promotion expectations, build and run an OpenRouter "
                "consultation prompt with logs, metrics, dataset manifest, launcher parameters, blockers, and row diffs."
            ),
            "required_prelaunch_gates": {
                "v618_preflight": "must pass before --launch",
                "v619_module_surface": str(V619_REPORT),
                "pre_paid_integration_gate": "must pass before --launch",
            },
            "required_post_eval_gate": {
                "script": "scripts/run_v614_anti_runaway_promotion_gate.py",
                "min_total": PROMOTION_TOTAL_MIN,
                "min_bit": PROMOTION_BIT_MIN,
                "min_equation": PROMOTION_EQUATION_MIN,
                "max_truncated": PROMOTION_TRUNC_MAX,
                "finding_counts_warning_must_be_zero": True,
            },
            "next_action": "Do not launch V652; replace V613 answer-only rows with a V513-passing traceable dataset first.",
            "v513_learnability_status": V513_LEARNABILITY_STATUS,
            "v513_block_reason": V513_BLOCK_REASON,
        }
    )
    out_path = Path(__file__).resolve().parent / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    if job is not None:
        print("job_url =", f"https://huggingface.co/jobs/{v615.NAMESPACE}/{job.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Launch/debug V416 rawstyle transfer smoke train on HF H200."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = (
    REPO_ROOT
    / "artifacts/v391_hf_nemo_h200_equation_bit_replay_launch/"
    / "launch_v391_hf_nemo_h200_equation_bit_replay.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("v391_base_launcher", BASE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base launcher: {BASE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

base.VERSION = "v416_rawstyle_transfer_from_v290_checkpoint6_nemo_h200"
base.FLAVOR = "h200"
base.RUN_ID = "v416-nemo-h200-rawstyle-transfer-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base.DATASET_UPLOAD_COMMIT = "pending-upload-manifest"
base.TRAIN_FILE = "data/v416_rawstyle_transfer/20260515T_v416_rawstyle_transfer/v416_rawstyle_transfer_train.jsonl"
base.VAL_FILE = "data/v416_rawstyle_transfer/20260515T_v416_rawstyle_transfer/v416_rawstyle_transfer_val.jsonl"
base.TRAIN_SHA256 = "cc1ac4bc74af73b3e8b00b07519b458f90f8dc05146cc399e8d74dedf03ef9da"
base.VAL_SHA256 = "d02d38ce0cc179650170f359f825ceffd8437003ebf454bc2472dc2057c402e4"
base.TRAIN_ROWS = 2320
base.VAL_ROWS = 580
base.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v416-h200-rawstyle-transfer-v290ckpt6"
base.MAX_STEPS = 4
base.SAVE_EVERY_STEPS = 2
base.EVAL_EVERY_STEPS = 2
base.EVAL_MAX_EXAMPLES = 128
base.ANSWER_SPAN_LOSS_WEIGHT = "10.0"
base.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1200"
base.SOURCE_WEIGHTS = (
    "v410_v390_v325_equation_numeric_rawstyle_v416=8.00,"
    "v410_v330_symbolic_cryptarithm_rawstyle_v416=6.00,"
    "v410_v408_perbit_asym_rule_rawstyle_v416=3.50,"
    "v410_v403_exact_global_bit_rule_rawstyle_v416=2.50,"
    "v406_v217_bit_replay_guardrail_rawstyle_v416=1.25"
)
base.SUBCATEGORY_WEIGHTS = (
    "equation_numeric_add_direct=12.00,"
    "equation_numeric_colon_absdiff=12.00,"
    "equation_numeric_colon_trailing_zero=12.00,"
    "equation_numeric_minus_direct_negative=12.00,"
    "equation_numeric_minus_signed=12.00,"
    "equation_symbolic_cryptarithm_single_operator_mul=8.00,"
    "bit_solver_first_v410=3.00,"
    "bit_manipulation=1.25,"
    "unknown=1.00"
)

base.COMMAND_SCRIPT = (
    base.COMMAND_SCRIPT
    .replace("export OUTPUT_DIR='/tmp/kg1_v391_output'", "export OUTPUT_DIR='/tmp/kg1_v416_output'")
    .replace("export MAX_STEPS=12", "export MAX_STEPS=4")
    .replace("export EVAL_MAX_EXAMPLES=96", "export EVAL_MAX_EXAMPLES=128")
)

_base_build_job_env = base.build_job_env


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    env = _base_build_job_env(hardware)
    env.update(
        {
            "KG1_REQUIRED_TRAIN_SUBCATEGORIES": (
                "bit_solver_first_v410,equation_numeric_add_direct,equation_numeric_colon_absdiff,"
                "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
                "equation_numeric_minus_signed,equation_symbolic_cryptarithm_single_operator_mul"
            ),
            "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
            "KG1_HF_MAX_UNIT_COST_USD": "0.09",
        }
    )
    return env


base.build_job_env = build_job_env


def write_manifest(payload: dict) -> Path:
    payload["previous_version"] = "V415 adapter-direct audit"
    payload["version_comparison_artifact"] = "artifacts/v416_rawstyle_transfer_dataset/20260515T_v416_rawstyle_transfer/V416_VS_PREVIOUS.md"
    payload["next_action"] = "Monitor every 40 seconds; weak-eval checkpoint-2/4; cancel unless weak total>192, equation>56, bit>=136, truncated=0."
    if isinstance(payload.get("recipe"), dict):
        payload["recipe"]["previous_version"] = "V415 adapter-direct audit"
        payload["recipe"]["version_comparison_artifact"] = "artifacts/v416_rawstyle_transfer_dataset/20260515T_v416_rawstyle_transfer/V416_VS_PREVIOUS.md"
        payload["recipe"]["promotion_gate"] = "promote only if weak total>192, equation>56, bit>=136, truncated=0; otherwise cancel by FinOps"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base.RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    return out_path


base.write_manifest = write_manifest


if __name__ == "__main__":
    raise SystemExit(base.main())

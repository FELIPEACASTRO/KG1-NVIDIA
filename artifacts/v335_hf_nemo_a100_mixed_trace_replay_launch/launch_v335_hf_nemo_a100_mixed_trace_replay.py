#!/usr/bin/env python3
"""Launch/debug V335 mixed trace replay smoke train on HF A100.

This wrapper reuses the validated V331 HF launcher implementation and patches
only the V335 data contract, run identity, sample weights, and smoke length.
Default mode is local debug. Pass --launch to create the paid HF job.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = REPO_ROOT / "artifacts" / "v331_hf_nemo_a100_equation_bit_symbolic_launch" / "launch_v331_hf_nemo_a100_equation_bit_symbolic.py"


def load_v331_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v331_launcher", V331_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V331 launcher from {V331_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_dataset_upload_commit() -> str:
    manifest_path = RUN_DIR / "v335_hf_dataset_upload_manifest.json"
    if not manifest_path.exists():
        return "not_uploaded_yet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upload_url = str(manifest.get("dataset_upload", ""))
    return upload_url.rstrip("/").split("/")[-1] if upload_url else "unknown"


def patch_launcher(module: Any) -> None:
    run_id = "v335-nemo-a100-mixed-trace-replay-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    module.VERSION = "v335_mixed_trace_replay_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = read_dataset_upload_commit()
    module.TRAIN_FILE = "data/v335_mixed_trace_replay/20260513T_cpu_gate/v335_mixed_trace_replay_train.jsonl"
    module.VAL_FILE = "data/v335_mixed_trace_replay/20260513T_cpu_gate/v335_mixed_trace_replay_val.jsonl"
    module.TRAIN_SHA256 = "fed84002b6f9104869c743cce816a81e279400c8031ac3545846871fecc50654"
    module.VAL_SHA256 = "1af6a221d3539294163cd684ded1a0de49d3631d2357d8a8aa0f560de1f1866d"
    module.TRAIN_ROWS = 13542
    module.VAL_ROWS = 1149
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v335-nemo-a100-mixed-trace-replay-v290ckpt6"
    module.MAX_STEPS = 16
    module.SAVE_EVERY_STEPS = 2
    module.EVAL_EVERY_STEPS = 2
    module.EVAL_MAX_EXAMPLES = 96
    module.ANSWER_SPAN_LOSS_WEIGHT = "14.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "1400"
    module.SOURCE_WEIGHTS = (
        "v330_symbolic_cryptarithm_distill=8.00,"
        "v325_equation_no_loss_distill=6.00,"
        "v304_solver_trace_bit_fullbyte_distill_exact=2.00,"
        "v304_solver_trace_bit_fullbyte_distill_random=1.50,"
        "v216_base_clean_safe_strict_bit=1.20,"
        "v216_synthetic_kg1_bit_rules=1.20,"
        "v215_replay_anchor=1.00"
    )
    module.SUBCATEGORY_WEIGHTS = (
        "equation_symbolic_cryptarithm_single_operator_mul=14.00,"
        "equation_numeric_add_direct=10.00,"
        "equation_numeric_colon_absdiff=10.00,"
        "equation_numeric_minus_signed=10.00,"
        "bit_fullbyte_v300_gain_pattern=2.50,"
        "bit_fullbyte_safe_ternary=2.00,"
        "bit_fullbyte_binary=1.50,"
        "bit_manipulation=1.10,"
        "unknown=1.00"
    )
    command_script = module.COMMAND_SCRIPT
    command_script = command_script.replace("export OUTPUT_DIR='/tmp/kg1_v331_output'", "export OUTPUT_DIR='/tmp/kg1_v335_output'")
    command_script = command_script.replace("export MAX_STEPS=10", "export MAX_STEPS=16")
    module.COMMAND_SCRIPT = command_script

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["finops_kill_switch"] = {
            "first_checkpoint_total_min_exclusive": 192,
            "first_checkpoint_equation_min_exclusive": 56,
            "first_checkpoint_bit_min": 136,
            "action": "cancel HF job if first weak checkpoint fails these gates",
        }
        out_path = RUN_DIR / f"{module.RUN_ID}_launch_manifest.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("launch_manifest_path =", out_path, flush=True)
        return out_path

    module.write_manifest = write_manifest


def main() -> int:
    module = load_v331_launcher()
    patch_launcher(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

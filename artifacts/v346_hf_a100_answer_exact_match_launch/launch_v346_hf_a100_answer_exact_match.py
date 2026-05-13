#!/usr/bin/env python3
"""Launch/debug V346 answer-exact-match transfer smoke on HF A100.

Default mode is local debug only. Pass --launch after commit/push to create the
paid HF job. Promotion remains weak family ACC only.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = (
    REPO_ROOT
    / "artifacts/v331_hf_nemo_a100_equation_bit_symbolic_launch/launch_v331_hf_nemo_a100_equation_bit_symbolic.py"
)
DATASET_DIR = REPO_ROOT / "artifacts/v346_answer_exact_match_dataset/20260513T_cpu_gate"
DATASET_MANIFEST = DATASET_DIR / "v346_answer_exact_match_manifest.json"
TOKENIZATION_GATE_MANIFEST = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
UPLOAD_MANIFEST = RUN_DIR / "v346_answer_exact_match_hf_upload_manifest.json"

TRAIN_SHA256 = "cb2e244c04b88e4aa81e726a8a89740aa6ab554c07eb8778f6f2d2aa57cb1d34"
VAL_SHA256 = "d9f8f7b7c2f3106f7e2f6bf88a531f0fe895bd7a8b16ea84501c3d2c21897087"


def load_v331_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v331_launcher", V331_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V331 launcher from {V331_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_local_gates(module: Any) -> dict[str, Any]:
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(DATASET_MANIFEST)
    if not TOKENIZATION_GATE_MANIFEST.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE_MANIFEST)
    if not UPLOAD_MANIFEST.is_file():
        raise FileNotFoundError(UPLOAD_MANIFEST)
    dataset = read_json(DATASET_MANIFEST)
    gate = read_json(TOKENIZATION_GATE_MANIFEST)
    upload = read_json(UPLOAD_MANIFEST)
    if dataset.get("schema_version") != "kg1_v346_answer_exact_match_dataset_v1":
        raise RuntimeError("Unexpected V346 dataset schema.")
    if gate.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V346 tokenization gate did not pass.")
    outputs = dataset.get("outputs", {})
    if outputs.get("train_sha256") != TRAIN_SHA256 or outputs.get("val_sha256") != VAL_SHA256:
        raise RuntimeError("V346 dataset hashes drifted.")
    if gate.get("config", {}).get("assistant_final_answer_mode") != "boxed_exact":
        raise RuntimeError("V346 tokenization gate must use boxed_exact.")
    for split in ("train", "validation"):
        token_summary = gate.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V346 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V346 {split} completion tokens would be dropped.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V346 {split} used fallback masks.")
    upload_url = str(upload.get("upload", ""))
    if not upload_url:
        raise RuntimeError("V346 upload manifest has no upload URL.")
    return {
        "dataset_manifest_sha256": module.sha256_file(DATASET_MANIFEST),
        "tokenization_gate_manifest_sha256": module.sha256_file(TOKENIZATION_GATE_MANIFEST),
        "upload_url": upload_url,
        "train_sha256": TRAIN_SHA256,
        "val_sha256": VAL_SHA256,
    }


def patch_launcher(module: Any) -> None:
    local_gate_summary = verify_local_gates(module)
    run_id = "v346-nemo-a100-answer-exact-match-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    module.VERSION = "v346_answer_exact_match_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = local_gate_summary["upload_url"].rstrip("/").split("/")[-1]
    module.TRAIN_FILE = "data/v346_answer_exact_match/20260513T_cpu_gate/v346_answer_exact_match_train.jsonl"
    module.VAL_FILE = "data/v346_answer_exact_match/20260513T_cpu_gate/v346_answer_exact_match_val.jsonl"
    module.TRAIN_SHA256 = TRAIN_SHA256
    module.VAL_SHA256 = VAL_SHA256
    module.TRAIN_ROWS = 1760
    module.VAL_ROWS = 420
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v346-answer-exact-match-a100-v290ckpt6"
    module.MAX_STEPS = 6
    module.SAVE_EVERY_STEPS = 2
    module.EVAL_EVERY_STEPS = 2
    module.EVAL_MAX_EXAMPLES = 96
    module.ANSWER_SPAN_LOSS_WEIGHT = "24.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "700"
    module.SOURCE_WEIGHTS = (
        "v325_equation_no_loss_distill=6.00,"
        "v330_symbolic_cryptarithm_distill=6.00,"
        "v337d_v217_bit_replay=8.00"
    )
    module.SUBCATEGORY_WEIGHTS = (
        "equation_numeric_add_direct=12.00,"
        "equation_numeric_colon_absdiff=12.00,"
        "equation_numeric_colon_trailing_zero=12.00,"
        "equation_numeric_minus_direct_negative=12.00,"
        "equation_numeric_minus_signed=12.00,"
        "equation_symbolic_cryptarithm_single_operator_mul=12.00,"
        "bit_manipulation=4.00,"
        "unknown=4.00"
    )

    command_script = module.COMMAND_SCRIPT
    replacements = {
        "export OUTPUT_DIR='/tmp/kg1_v331_output'": "export OUTPUT_DIR='/tmp/kg1_v346_output'",
        "export LEARNING_RATE=4.0e-8": "export LEARNING_RATE=8.0e-8",
        "export FINAL_LEARNING_RATE=1.0e-8": "export FINAL_LEARNING_RATE=2.0e-8",
        "export MAX_STEPS=10": "export MAX_STEPS=6",
        "export SAVE_EVERY_STEPS=2": "export SAVE_EVERY_STEPS=2",
        "export EVAL_EVERY_STEPS=2": "export EVAL_EVERY_STEPS=2",
        "export EVAL_MAX_EXAMPLES=96": "export EVAL_MAX_EXAMPLES=96",
    }
    for old, new in replacements.items():
        if old not in command_script:
            raise RuntimeError("V331 command script changed; missing replacement target: " + old)
        command_script = command_script.replace(old, new)
    module.COMMAND_SCRIPT = command_script

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        required_subcategories = (
            "bit_manipulation,equation_numeric_add_direct,equation_numeric_colon_absdiff,"
            "equation_numeric_colon_trailing_zero,equation_numeric_minus_direct_negative,"
            "equation_numeric_minus_signed,equation_symbolic_cryptarithm_single_operator_mul,unknown"
        )
        env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"] = required_subcategories
        env["KG1_REQUIRED_VAL_SUBCATEGORIES"] = required_subcategories
        env["KG1_EXPECTED_MAX_STEPS"] = str(module.MAX_STEPS)
        return env

    module.build_job_env = build_job_env

    original_local_debug = module.local_debug

    def local_debug(api: Any, token: str) -> tuple[dict[str, object], dict[str, str]]:
        selected, env = original_local_debug(api, token)
        forbidden = [
            "data/v331_equation_bit_symbolic_mix",
            "data/v337d_minimal_transfer",
            "v344_v343_minimal_transfer_preferences",
            "PREF_TRAIN_FILE",
        ]
        found = [item for item in forbidden if item in module.COMMAND_SCRIPT or item in json.dumps(env, sort_keys=True)]
        if found:
            raise RuntimeError("V346 launcher contains stale snippets: " + json.dumps(found))
        required = [
            "data/v346_answer_exact_match/20260513T_cpu_gate/v346_answer_exact_match_train.jsonl",
            "export LEARNING_RATE=8.0e-8",
            "export FINAL_LEARNING_RATE=2.0e-8",
            "export ANSWER_SPAN_LOSS_WEIGHT=\"$KG1_ANSWER_SPAN_LOSS_WEIGHT\"",
        ]
        missing = [item for item in required if item not in module.COMMAND_SCRIPT and item not in json.dumps(env)]
        if missing:
            raise RuntimeError("V346 launcher missing required snippets: " + json.dumps(missing))
        return selected, env

    module.local_debug = local_debug

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["v346_local_gate_summary"] = local_gate_summary
        payload["recipe"] = dict(payload.get("recipe", {}))
        payload["recipe"].update(
            {
                "learning_rate": "8.0e-8",
                "final_learning_rate": "2.0e-8",
                "answer_span_loss_weight": module.ANSWER_SPAN_LOSS_WEIGHT,
                "promotion_gate": "weak eval only; continue only if total>192, equation>56, bit>=136, truncated=0",
                "finops_kill_switch": "cancel after first weak checkpoint if gate is not beaten",
            }
        )
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

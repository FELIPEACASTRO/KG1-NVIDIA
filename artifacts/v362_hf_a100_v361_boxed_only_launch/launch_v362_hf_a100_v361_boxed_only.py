#!/usr/bin/env python3
"""Launch/debug V362 V361 boxed-only smoke train on HF A100.

V362 is intentionally small: max 2 steps, save/eval every step, then immediate
weak eval before any continuation.  It tests the V360/V361 hypothesis that the
V359 transfer failed partly because the supervised completion format was too
long and misaligned with inference.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = REPO_ROOT / "artifacts/v331_hf_nemo_a100_equation_bit_symbolic_launch/launch_v331_hf_nemo_a100_equation_bit_symbolic.py"
DATASET_DIR = REPO_ROOT / "artifacts/v361_v357_boxed_only_transfer_dataset/20260514T_cpu_gate"
DATASET_MANIFEST = DATASET_DIR / "v361_v357_boxed_only_transfer_manifest.json"
TOKENIZATION_GATE_MANIFEST = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
UPLOAD_MANIFEST = RUN_DIR / "v361_v357_boxed_only_transfer_hf_upload_manifest.json"

EXPECTED_TRAIN_SHA256 = "be742d7a82bf1c98f33d67bed8903006068c139ab74f798055fcc7d435ffa4db"
EXPECTED_VAL_SHA256 = "4c93766e7fae72da14f879177e15c3c6300b7991e4efc8fba4d7fe75d3df5332"
EXPECTED_TRAIN_ROWS = 1152
EXPECTED_VAL_ROWS = 288


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v331_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v331_launcher", V331_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V331 launcher from {V331_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_dataset_upload_commit() -> str:
    if not UPLOAD_MANIFEST.is_file():
        raise FileNotFoundError("V361 HF upload manifest is required before V362 launch: " + str(UPLOAD_MANIFEST))
    manifest = read_json(UPLOAD_MANIFEST)
    upload_url = str(manifest.get("dataset_upload", ""))
    if not upload_url:
        raise RuntimeError("V361 HF upload manifest is missing dataset_upload URL.")
    gate_summary = manifest.get("local_gate_summary", {})
    if gate_summary.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V361 HF upload manifest train SHA drifted.")
    if gate_summary.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V361 HF upload manifest val SHA drifted.")
    return upload_url.rstrip("/").split("/")[-1]


def verify_v361_tokenization_gate() -> dict[str, Any]:
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(DATASET_MANIFEST)
    if not TOKENIZATION_GATE_MANIFEST.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE_MANIFEST)
    dataset_manifest = read_json(DATASET_MANIFEST)
    gate_manifest = read_json(TOKENIZATION_GATE_MANIFEST)
    if dataset_manifest.get("schema_version") != "kg1_v361_v357_boxed_only_transfer_dataset_v1":
        raise RuntimeError("Unexpected V361 dataset schema.")
    if gate_manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V361 real tokenization gate did not pass.")
    if gate_manifest.get("dataset_manifest_sha256") != sha256_file(DATASET_MANIFEST):
        raise RuntimeError("V361 tokenization gate is stale relative to dataset manifest.")
    if gate_manifest.get("config", {}).get("assistant_final_answer_mode") != "boxed_only":
        raise RuntimeError("V361 tokenization gate must use boxed_only mode.")
    if gate_manifest.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V361 launcher requires the real tokenizer gate, not toy.")

    outputs = dataset_manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V361 dataset train SHA drifted.")
    if outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V361 dataset val SHA drifted.")
    train_validation = dataset_manifest.get("validation", {}).get("train", {})
    val_validation = dataset_manifest.get("validation", {}).get("validation", {})
    if int(train_validation.get("rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Unexpected V361 train row count.")
    if int(val_validation.get("rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected V361 validation row count.")
    if int(train_validation.get("assistant_boxed_only_rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("V361 train boxed-only count drifted.")
    if int(val_validation.get("assistant_boxed_only_rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("V361 validation boxed-only count drifted.")
    if train_validation.get("family_counts") != {"bit_manipulation": EXPECTED_TRAIN_ROWS}:
        raise RuntimeError("Unexpected V361 train family counts.")
    if val_validation.get("family_counts") != {"bit_manipulation": EXPECTED_VAL_ROWS}:
        raise RuntimeError("Unexpected V361 val family counts.")
    if int(dataset_manifest.get("validation", {}).get("train_validation_prompt_overlap", -1)) != 0:
        raise RuntimeError("V361 train/validation prompt overlap is not zero.")

    for split in ("train", "validation"):
        token_summary = gate_manifest.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V361 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V361 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V361 {split} used fallback masks.")
        if int(token_summary.get("token_max", 999999)) > 512:
            raise RuntimeError(f"V361 {split} token max unexpectedly high.")

    return {
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE_MANIFEST),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
        "train_rows": EXPECTED_TRAIN_ROWS,
        "val_rows": EXPECTED_VAL_ROWS,
        "token_max_train": gate_manifest.get("tokenization", {}).get("train", {}).get("token_max"),
        "token_max_val": gate_manifest.get("tokenization", {}).get("validation", {}).get("token_max"),
        "completion_format": "boxed_only",
    }


def patch_launcher(module: Any) -> None:
    local_gate_summary = verify_v361_tokenization_gate()
    run_id = "v362-nemo-a100-v361-boxed-only-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    module.VERSION = "v362_v361_boxed_only_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = read_dataset_upload_commit()
    module.TRAIN_FILE = "data/v361_v357_boxed_only_transfer/20260514T_cpu_gate/v361_v357_boxed_only_transfer_train.jsonl"
    module.VAL_FILE = "data/v361_v357_boxed_only_transfer/20260514T_cpu_gate/v361_v357_boxed_only_transfer_val.jsonl"
    module.TRAIN_SHA256 = EXPECTED_TRAIN_SHA256
    module.VAL_SHA256 = EXPECTED_VAL_SHA256
    module.TRAIN_ROWS = EXPECTED_TRAIN_ROWS
    module.VAL_ROWS = EXPECTED_VAL_ROWS
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6"
    module.MAX_STEPS = 2
    module.SAVE_EVERY_STEPS = 1
    module.EVAL_EVERY_STEPS = 1
    module.EVAL_MAX_EXAMPLES = 128
    module.ANSWER_SPAN_LOSS_WEIGHT = "8.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "500"
    module.SOURCE_WEIGHTS = "v361_boxed_only_from_v358_verified_bit_rules=1.00"
    module.SUBCATEGORY_WEIGHTS = "bit_exact_global_ternary=1.40,bit_exact_global_binary_replay=1.00,bit_manipulation=1.00,unknown=1.00"

    command_script = module.COMMAND_SCRIPT
    command_script = command_script.replace("export OUTPUT_DIR='/tmp/kg1_v331_output'", "export OUTPUT_DIR='/tmp/kg1_v362_output'")
    command_script = command_script.replace("export MAX_LENGTH=1024", "export MAX_LENGTH=512")
    command_script = command_script.replace("export MAX_STEPS=10", "export MAX_STEPS=2")
    command_script = command_script.replace("export SAVE_EVERY_STEPS=2", "export SAVE_EVERY_STEPS=1")
    command_script = command_script.replace("export EVAL_EVERY_STEPS=2", "export EVAL_EVERY_STEPS=1")
    command_script = command_script.replace("export EVAL_MAX_EXAMPLES=96", "export EVAL_MAX_EXAMPLES=128")
    command_script = command_script.replace("export LEARNING_RATE=4.0e-8", "export LEARNING_RATE=4.0e-8")
    command_script = command_script.replace("export FINAL_LEARNING_RATE=1.0e-8", "export FINAL_LEARNING_RATE=1.0e-8")
    module.COMMAND_SCRIPT = command_script

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env["KG1_REQUIRED_TRAIN_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_VAL_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"] = "bit_exact_global_ternary,bit_exact_global_binary_replay"
        env["KG1_REQUIRED_VAL_SUBCATEGORIES"] = "bit_exact_global_ternary,bit_exact_global_binary_replay"
        env["KG1_EXPECTED_MAX_STEPS"] = str(module.MAX_STEPS)
        return env

    module.build_job_env = build_job_env
    original_local_debug = module.local_debug

    def local_debug(api: Any, token: str) -> tuple[dict[str, object], dict[str, str]]:
        selected, env = original_local_debug(api, token)
        serialized = module.COMMAND_SCRIPT + "\n" + json.dumps(env, sort_keys=True)
        required = [
            "data/v361_v357_boxed_only_transfer/20260514T_cpu_gate/v361_v357_boxed_only_transfer_train.jsonl",
            "v361_boxed_only_from_v358_verified_bit_rules",
            "bit_exact_global_ternary",
            "bit_exact_global_binary_replay",
            "MAX_STEPS=2",
            "SAVE_EVERY_STEPS=1",
            "MAX_LENGTH=512",
        ]
        missing = [item for item in required if item not in serialized]
        if missing:
            raise RuntimeError("V362 launcher missing required snippets: " + json.dumps(missing))
        stale = [
            "data/v331_equation_bit_symbolic_mix",
            "data/v358_v357_bit_ternary_transfer",
            "v325_equation_no_loss_distill",
            "v330_symbolic_cryptarithm_distill",
            "MAX_STEPS=4",
            "MAX_STEPS=10",
        ]
        found = [item for item in stale if item in serialized]
        if found:
            raise RuntimeError("V362 launcher contains stale snippets: " + json.dumps(found))
        return selected, env

    module.local_debug = local_debug

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["v361_local_gate_summary"] = local_gate_summary
        payload["finops_kill_switch"] = {
            "first_weak_eval_total_min_exclusive": 192,
            "first_weak_eval_equation_min_exclusive": 56,
            "first_weak_eval_bit_min": 136,
            "first_weak_eval_truncated_max": 0,
            "action": "weak-eval checkpoint-1 first; cancel checkpoint-2/final continuation if no adapter-only gain appears",
        }
        payload["recipe_override"] = {
            "max_steps": module.MAX_STEPS,
            "save_every_steps": module.SAVE_EVERY_STEPS,
            "eval_every_steps": module.EVAL_EVERY_STEPS,
            "learning_rate": "4.0e-8",
            "final_learning_rate": "1.0e-8",
            "max_length": 512,
            "answer_span_loss_weight": module.ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": module.SOURCE_WEIGHTS,
            "subcategory_weights": module.SUBCATEGORY_WEIGHTS,
            "preference_training_used": False,
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

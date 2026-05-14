#!/usr/bin/env python3
"""Launch/debug V372 V371 trace-style bit-transfer smoke train on HF A100."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = REPO_ROOT / "artifacts/v331_hf_nemo_a100_equation_bit_symbolic_launch/launch_v331_hf_nemo_a100_equation_bit_symbolic.py"
DATASET_DIR = REPO_ROOT / "artifacts/v371_v367_trace_style_transfer_dataset/20260514T_cpu_gate"
DATASET_MANIFEST = DATASET_DIR / "v371_v367_trace_style_transfer_manifest.json"
TOKENIZATION_GATE_MANIFEST = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
UPLOAD_MANIFEST = RUN_DIR / "v371_v367_trace_style_transfer_hf_upload_manifest.json"

EXPECTED_TRAIN_SHA256 = "96278ed22d81ba2412ce9a1ec5f8ac87df64a5f8014ebb5c3f88b18103bfbe0a"
EXPECTED_VAL_SHA256 = "fe66489655ad6907705fa5d18bc8e777055b6d77d9f1ceaba7fb58005169c7d0"
EXPECTED_TRAIN_ROWS = 1128
EXPECTED_VAL_ROWS = 282
EXPECTED_MAX_LENGTH = 1024


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
        raise FileNotFoundError("V371 HF upload manifest is required before V372 launch: " + str(UPLOAD_MANIFEST))
    manifest = read_json(UPLOAD_MANIFEST)
    upload_url = str(manifest.get("dataset_upload", ""))
    if not upload_url:
        raise RuntimeError("V371 HF upload manifest is missing dataset_upload URL.")
    gate_summary = manifest.get("local_gate_summary", {})
    if gate_summary.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V371 HF upload manifest train SHA drifted.")
    if gate_summary.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V371 HF upload manifest val SHA drifted.")
    if int(gate_summary.get("token_max_train", 999999)) > EXPECTED_MAX_LENGTH:
        raise RuntimeError("V371 HF upload manifest train token max exceeds V372 max length.")
    if int(gate_summary.get("token_max_val", 999999)) > EXPECTED_MAX_LENGTH:
        raise RuntimeError("V371 HF upload manifest val token max exceeds V372 max length.")
    return upload_url.rstrip("/").split("/")[-1]


def verify_v371_tokenization_gate() -> dict[str, Any]:
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(DATASET_MANIFEST)
    if not TOKENIZATION_GATE_MANIFEST.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE_MANIFEST)
    dataset_manifest = read_json(DATASET_MANIFEST)
    gate_manifest = read_json(TOKENIZATION_GATE_MANIFEST)
    if dataset_manifest.get("schema_version") != "kg1_v371_v367_trace_style_transfer_dataset_v1":
        raise RuntimeError("Unexpected V371 dataset schema.")
    if gate_manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V371 real tokenization gate did not pass.")
    if gate_manifest.get("dataset_manifest_sha256") != sha256_file(DATASET_MANIFEST):
        raise RuntimeError("V371 tokenization gate is stale relative to dataset manifest.")
    if gate_manifest.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V371 tokenization gate must use boxed_suffix mode.")
    if gate_manifest.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V371 launcher requires the real tokenizer gate, not toy.")

    outputs = dataset_manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V371 dataset train SHA drifted.")
    if outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V371 dataset val SHA drifted.")
    validation = dataset_manifest.get("validation", {})
    train_validation = validation.get("train", {})
    val_validation = validation.get("validation", {})
    if int(train_validation.get("rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Unexpected V371 train row count.")
    if int(val_validation.get("rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected V371 validation row count.")
    for split_name, split_validation, expected_rows in (
        ("train", train_validation, EXPECTED_TRAIN_ROWS),
        ("validation", val_validation, EXPECTED_VAL_ROWS),
    ):
        if split_validation.get("family_counts") != {"bit_manipulation": expected_rows}:
            raise RuntimeError(f"Unexpected V371 {split_name} family counts.")
        if int(split_validation.get("assistant_trace_style_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} trace-style count drifted.")
        if int(split_validation.get("assistant_contains_output_bit_columns_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} Output bit columns count drifted.")
        if int(split_validation.get("assistant_boxed_suffix_rows", -1)) != expected_rows:
            raise RuntimeError(f"V371 {split_name} boxed-suffix count drifted.")
    if int(validation.get("train_val_prompt_overlap", -1)) != 0:
        raise RuntimeError("V371 train/validation prompt overlap is not zero.")

    for split in ("train", "validation"):
        token_summary = gate_manifest.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V371 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V371 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V371 {split} used fallback masks.")
        if int(token_summary.get("token_max", 999999)) > EXPECTED_MAX_LENGTH:
            raise RuntimeError(f"V371 {split} token max exceeds V372 max length.")

    return {
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE_MANIFEST),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
        "train_rows": EXPECTED_TRAIN_ROWS,
        "val_rows": EXPECTED_VAL_ROWS,
        "train_subcategory_counts": train_validation.get("subcategory_counts"),
        "val_subcategory_counts": val_validation.get("subcategory_counts"),
        "token_max_train": gate_manifest.get("tokenization", {}).get("train", {}).get("token_max"),
        "token_max_val": gate_manifest.get("tokenization", {}).get("validation", {}).get("token_max"),
        "completion_format": "trace_style_boxed_suffix",
        "source_format_audit": "V370",
        "source_dataset": "V371",
    }


def patch_launcher(module: Any) -> None:
    local_gate_summary = verify_v371_tokenization_gate()
    run_id = "v372-nemo-a100-v371-trace-style-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    module.VERSION = "v372_v371_trace_style_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = read_dataset_upload_commit()
    module.TRAIN_FILE = "data/v371_v367_trace_style_transfer/20260514T_cpu_gate/v371_v367_trace_style_transfer_train.jsonl"
    module.VAL_FILE = "data/v371_v367_trace_style_transfer/20260514T_cpu_gate/v371_v367_trace_style_transfer_val.jsonl"
    module.TRAIN_SHA256 = EXPECTED_TRAIN_SHA256
    module.VAL_SHA256 = EXPECTED_VAL_SHA256
    module.TRAIN_ROWS = EXPECTED_TRAIN_ROWS
    module.VAL_ROWS = EXPECTED_VAL_ROWS
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v372-nemo-a100-v371-trace-style-v290ckpt6"
    module.MAX_STEPS = 2
    module.SAVE_EVERY_STEPS = 1
    module.EVAL_EVERY_STEPS = 1
    module.EVAL_MAX_EXAMPLES = 128
    module.ANSWER_SPAN_LOSS_WEIGHT = "4.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "500"
    module.SOURCE_WEIGHTS = "v371_trace_style_from_v367=1.00"
    module.SUBCATEGORY_WEIGHTS = (
        "bit_fullbyte_ternary_v366_new=2.50,"
        "bit_exact_global_ternary_replay=1.20,"
        "bit_exact_global_binary_replay=1.00,"
        "bit_manipulation=1.00,"
        "unknown=1.00"
    )

    command_script = module.COMMAND_SCRIPT
    command_script = command_script.replace("export OUTPUT_DIR='/tmp/kg1_v331_output'", "export OUTPUT_DIR='/tmp/kg1_v372_output'")
    command_script = command_script.replace("export MAX_STEPS=10", "export MAX_STEPS=2")
    command_script = command_script.replace("export SAVE_EVERY_STEPS=2", "export SAVE_EVERY_STEPS=1")
    command_script = command_script.replace("export EVAL_EVERY_STEPS=2", "export EVAL_EVERY_STEPS=1")
    command_script = command_script.replace("export EVAL_MAX_EXAMPLES=96", "export EVAL_MAX_EXAMPLES=128")
    if "export MAX_LENGTH=1024" not in command_script:
        raise RuntimeError("V372 launcher template must preserve MAX_LENGTH=1024.")
    if "export MAX_LENGTH=512" in command_script:
        raise RuntimeError("V372 launcher must not use MAX_LENGTH=512.")
    module.COMMAND_SCRIPT = command_script

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env["KG1_REQUIRED_TRAIN_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_VAL_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"] = (
            "bit_fullbyte_ternary_v366_new,bit_exact_global_ternary_replay,bit_exact_global_binary_replay"
        )
        env["KG1_REQUIRED_VAL_SUBCATEGORIES"] = (
            "bit_fullbyte_ternary_v366_new,bit_exact_global_ternary_replay,bit_exact_global_binary_replay"
        )
        env["KG1_EXPECTED_MAX_STEPS"] = str(module.MAX_STEPS)
        env["KG1_EXPECTED_ASSISTANT_FINAL_ANSWER_MODE"] = "boxed_suffix"
        return env

    module.build_job_env = build_job_env
    original_local_debug = module.local_debug

    def local_debug(api: Any, token: str) -> tuple[dict[str, object], dict[str, str]]:
        selected, env = original_local_debug(api, token)
        serialized = module.COMMAND_SCRIPT + "\n" + json.dumps(env, sort_keys=True)
        required = [
            "data/v371_v367_trace_style_transfer/20260514T_cpu_gate/v371_v367_trace_style_transfer_train.jsonl",
            "v371_trace_style_from_v367",
            "bit_fullbyte_ternary_v366_new",
            "bit_exact_global_ternary_replay",
            "bit_exact_global_binary_replay",
            "MAX_STEPS=2",
            "SAVE_EVERY_STEPS=1",
            "MAX_LENGTH=1024",
            "KG1_EXPECTED_ASSISTANT_FINAL_ANSWER_MODE",
        ]
        missing = [item for item in required if item not in serialized]
        if missing:
            raise RuntimeError("V372 launcher missing required snippets: " + json.dumps(missing))
        stale = [
            "data/v367_v366_bit_ternary_transfer",
            "v367_synthetic_from_v366_teacher_with_replay",
            "data/v361_v357_boxed_only_transfer",
            "data/v358_v357_bit_ternary_transfer",
            "data/v331_equation_bit_symbolic_mix",
            "MAX_LENGTH=512",
            "MAX_STEPS=4",
            "MAX_STEPS=10",
        ]
        found = [item for item in stale if item in serialized]
        if found:
            raise RuntimeError("V372 launcher contains stale snippets: " + json.dumps(found))
        return selected, env

    module.local_debug = local_debug

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["v371_local_gate_summary"] = local_gate_summary
        payload["finops_kill_switch"] = {
            "first_weak_eval_total_min_exclusive": 192,
            "first_weak_eval_equation_min_exclusive": 56,
            "first_weak_eval_bit_min": 136,
            "first_weak_eval_truncated_max": 0,
            "action": "weak-eval checkpoint-1 first; cancel continuation if no adapter-only family gain appears",
        }
        payload["recipe_override"] = {
            "max_steps": module.MAX_STEPS,
            "save_every_steps": module.SAVE_EVERY_STEPS,
            "eval_every_steps": module.EVAL_EVERY_STEPS,
            "learning_rate": "4.0e-8",
            "final_learning_rate": "1.0e-8",
            "max_length": EXPECTED_MAX_LENGTH,
            "answer_span_loss_weight": module.ANSWER_SPAN_LOSS_WEIGHT,
            "answer_span_min_weighted_tokens": module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
            "source_weights": module.SOURCE_WEIGHTS,
            "subcategory_weights": module.SUBCATEGORY_WEIGHTS,
            "preference_training_used": False,
            "format_hypothesis": "train the observed V368 bit trace style instead of boxed-only targets",
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

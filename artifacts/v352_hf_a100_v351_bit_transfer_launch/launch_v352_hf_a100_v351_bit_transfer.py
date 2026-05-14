#!/usr/bin/env python3
"""Launch/debug V352 V351 bit-transfer smoke train on HF A100.

Default mode is local debug only. Pass --launch to create the paid HF job.
The wrapper reuses the validated V331 A100 NeMo launcher and patches only the
V351 dataset contract, run identity, and short smoke hyperparameters.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = REPO_ROOT / "artifacts/v331_hf_nemo_a100_equation_bit_symbolic_launch/launch_v331_hf_nemo_a100_equation_bit_symbolic.py"
DATASET_DIR = REPO_ROOT / "artifacts/v351_v350_bit_transfer_dataset/20260514T_cpu_gate"
DATASET_MANIFEST = DATASET_DIR / "v351_v350_bit_transfer_manifest.json"
TOKENIZATION_GATE_MANIFEST = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
UPLOAD_MANIFEST = RUN_DIR / "v351_v350_bit_transfer_hf_upload_manifest.json"
EXPECTED_TRAIN_SHA256 = "be8192036a570711d0858620aaeae1b0736e86588e4494cbdb2c85e8f8dcd5ed"
EXPECTED_VAL_SHA256 = "8e928e38a691f41c42ea4080c1227e053031f243cbf30dd4a1a07a98e5907f93"
EXPECTED_TRAIN_ROWS = 640
EXPECTED_VAL_ROWS = 160


def sha256_file(path: Path) -> str:
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
        raise FileNotFoundError("V351 HF upload manifest is required before V352 launch: " + str(UPLOAD_MANIFEST))
    manifest = read_json(UPLOAD_MANIFEST)
    upload_url = str(manifest.get("dataset_upload", ""))
    if not upload_url:
        raise RuntimeError("V351 HF upload manifest is missing dataset_upload URL.")
    gate_summary = manifest.get("local_gate_summary", {})
    if gate_summary.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V351 HF upload manifest local gate summary points at unexpected train SHA.")
    if gate_summary.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V351 HF upload manifest local gate summary points at unexpected val SHA.")
    return upload_url.rstrip("/").split("/")[-1]


def verify_v351_tokenization_gate() -> dict[str, Any]:
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(DATASET_MANIFEST)
    if not TOKENIZATION_GATE_MANIFEST.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE_MANIFEST)

    dataset_manifest = read_json(DATASET_MANIFEST)
    gate_manifest = read_json(TOKENIZATION_GATE_MANIFEST)
    if dataset_manifest.get("schema_version") != "kg1_v351_v350_bit_transfer_dataset_v1":
        raise RuntimeError("Unexpected V351 dataset schema.")
    if gate_manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V351 real tokenization gate did not pass.")
    if gate_manifest.get("dataset_manifest_sha256") != sha256_file(DATASET_MANIFEST):
        raise RuntimeError("V351 tokenization gate is stale relative to dataset manifest.")
    if gate_manifest.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V351 tokenization gate must use boxed_suffix mode.")
    if gate_manifest.get("tokenizer_info", {}).get("toy") is not False:
        raise RuntimeError("V351 launcher requires the real tokenizer gate, not toy.")

    outputs = dataset_manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V351 dataset train SHA drifted.")
    if outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V351 dataset val SHA drifted.")

    train_validation = dataset_manifest.get("validation", {}).get("train", {})
    val_validation = dataset_manifest.get("validation", {}).get("validation", {})
    if int(train_validation.get("rows", -1)) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Unexpected V351 train row count.")
    if int(val_validation.get("rows", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError("Unexpected V351 val row count.")
    if train_validation.get("family_counts") != {"bit_manipulation": EXPECTED_TRAIN_ROWS}:
        raise RuntimeError("Unexpected V351 train family counts.")
    if val_validation.get("family_counts") != {"bit_manipulation": EXPECTED_VAL_ROWS}:
        raise RuntimeError("Unexpected V351 val family counts.")

    for split in ("train", "validation"):
        token_summary = gate_manifest.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V351 {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V351 {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V351 {split} used fallback masks.")

    return {
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE_MANIFEST),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
        "train_rows": EXPECTED_TRAIN_ROWS,
        "val_rows": EXPECTED_VAL_ROWS,
        "token_max_train": gate_manifest.get("tokenization", {}).get("train", {}).get("token_max"),
        "token_max_val": gate_manifest.get("tokenization", {}).get("validation", {}).get("token_max"),
    }


def patch_launcher(module: Any) -> None:
    local_gate_summary = verify_v351_tokenization_gate()
    run_id = "v352-nemo-a100-v351-bit-transfer-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    module.VERSION = "v352_v351_bit_transfer_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = read_dataset_upload_commit()
    module.TRAIN_FILE = "data/v351_v350_bit_transfer/20260514T_cpu_gate/v351_v350_bit_transfer_train.jsonl"
    module.VAL_FILE = "data/v351_v350_bit_transfer/20260514T_cpu_gate/v351_v350_bit_transfer_val.jsonl"
    module.TRAIN_SHA256 = EXPECTED_TRAIN_SHA256
    module.VAL_SHA256 = EXPECTED_VAL_SHA256
    module.TRAIN_ROWS = EXPECTED_TRAIN_ROWS
    module.VAL_ROWS = EXPECTED_VAL_ROWS
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v352-nemo-a100-v351-bit-transfer-v290ckpt6"
    module.MAX_STEPS = 8
    module.SAVE_EVERY_STEPS = 2
    module.EVAL_EVERY_STEPS = 2
    module.EVAL_MAX_EXAMPLES = 80
    module.ANSWER_SPAN_LOSS_WEIGHT = "12.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "600"
    module.SOURCE_WEIGHTS = "v351_synthetic_from_v350_bit_exact_global=1.00"
    module.SUBCATEGORY_WEIGHTS = "bit_exact_global_byte=1.00,bit_manipulation=1.00,unknown=1.00"

    command_script = module.COMMAND_SCRIPT
    command_script = command_script.replace("export OUTPUT_DIR='/tmp/kg1_v331_output'", "export OUTPUT_DIR='/tmp/kg1_v352_output'")
    command_script = command_script.replace("export MAX_STEPS=10", "export MAX_STEPS=8")
    command_script = command_script.replace("export EVAL_MAX_EXAMPLES=96", "export EVAL_MAX_EXAMPLES=80")
    command_script = command_script.replace("export LEARNING_RATE=4.0e-8", "export LEARNING_RATE=3.0e-8")
    command_script = command_script.replace("export FINAL_LEARNING_RATE=1.0e-8", "export FINAL_LEARNING_RATE=8.0e-9")
    module.COMMAND_SCRIPT = command_script

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env["KG1_REQUIRED_TRAIN_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_VAL_FAMILIES"] = "bit_manipulation"
        env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"] = "bit_exact_global_byte"
        env["KG1_REQUIRED_VAL_SUBCATEGORIES"] = "bit_exact_global_byte"
        env["KG1_EXPECTED_MAX_STEPS"] = str(module.MAX_STEPS)
        return env

    module.build_job_env = build_job_env

    original_local_debug = module.local_debug

    def local_debug(api: Any, token: str) -> tuple[dict[str, object], dict[str, str]]:
        selected, env = original_local_debug(api, token)
        serialized = module.COMMAND_SCRIPT + "\n" + json.dumps(env, sort_keys=True)
        required = [
            "data/v351_v350_bit_transfer/20260514T_cpu_gate/v351_v350_bit_transfer_train.jsonl",
            "v351_synthetic_from_v350_bit_exact_global",
            "bit_exact_global_byte",
            "LEARNING_RATE=3.0e-8",
            "MAX_STEPS=8",
        ]
        missing = [item for item in required if item not in serialized]
        if missing:
            raise RuntimeError("V352 launcher missing required snippets: " + json.dumps(missing))
        stale = [
            "data/v331_equation_bit_symbolic_mix",
            "data/v337d_minimal_transfer",
            "data/v346_answer_exact_match",
            "v325_equation_no_loss_distill",
            "v330_symbolic_cryptarithm_distill",
        ]
        found = [item for item in stale if item in serialized]
        if found:
            raise RuntimeError("V352 launcher contains stale snippets: " + json.dumps(found))
        return selected, env

    module.local_debug = local_debug

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["v351_local_gate_summary"] = local_gate_summary
        payload["finops_kill_switch"] = {
            "first_checkpoint_total_min_exclusive": 192,
            "first_checkpoint_equation_min": 56,
            "first_checkpoint_bit_min_exclusive": 136,
            "action": "cancel HF job if first weak checkpoint does not show adapter-only gain over the 192/315 baseline",
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

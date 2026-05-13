#!/usr/bin/env python3
"""Launch/debug V338 minimal-transfer smoke train on HF A100.

Default mode is local debug only. Pass --launch to create the paid HF job.
This wrapper reuses the validated V331 A100 NeMo launcher and patches only the
V337D data contract, run identity, sample weights, and short smoke length.
"""

from __future__ import annotations

import importlib.util
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
V331_LAUNCHER = REPO_ROOT / "artifacts" / "v331_hf_nemo_a100_equation_bit_symbolic_launch" / "launch_v331_hf_nemo_a100_equation_bit_symbolic.py"
DATASET_DIR = REPO_ROOT / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate"
DATASET_MANIFEST = DATASET_DIR / "v337d_minimal_transfer_manifest.json"
TOKENIZATION_GATE_MANIFEST = DATASET_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
EXPECTED_TRAIN_SHA256 = "df67214d3fdbb74ada96a9fc24609db5a3f5f6dc1d26dea5d4449eb39eb4147c"
EXPECTED_VAL_SHA256 = "50d4ee05a377ed4e111d27f9de0e1109eb0c09bfe01a9bce0717b63d704dbf80"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v331_launcher() -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v331_launcher", V331_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V331 launcher from {V331_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_dataset_upload_commit() -> str:
    manifest_path = RUN_DIR / "v337d_hf_dataset_upload_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("V337D HF upload manifest is required before V338 launch: " + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upload_url = str(manifest.get("dataset_upload", ""))
    if not upload_url:
        raise RuntimeError("V337D HF upload manifest is missing dataset_upload URL.")
    gate_summary = manifest.get("local_gate_summary")
    if isinstance(gate_summary, dict) and gate_summary.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V337D HF upload manifest local gate summary points at an unexpected dataset.")
    return upload_url.rstrip("/").split("/")[-1]


def verify_v337d_tokenization_gate() -> dict[str, Any]:
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(DATASET_MANIFEST)
    if not TOKENIZATION_GATE_MANIFEST.is_file():
        raise FileNotFoundError(TOKENIZATION_GATE_MANIFEST)
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    gate_manifest = json.loads(TOKENIZATION_GATE_MANIFEST.read_text(encoding="utf-8"))
    if dataset_manifest.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("Unexpected V337D dataset schema.")
    if gate_manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        raise RuntimeError("V337D tokenization gate did not pass.")
    if gate_manifest.get("dataset_manifest_sha256") != sha256_file(DATASET_MANIFEST):
        raise RuntimeError("V337D tokenization gate is stale relative to dataset manifest.")
    outputs = dataset_manifest.get("outputs", {})
    if outputs.get("train_sha256") != EXPECTED_TRAIN_SHA256 or outputs.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("V337D dataset hashes drifted.")
    if gate_manifest.get("config", {}).get("assistant_final_answer_mode") != "boxed_suffix":
        raise RuntimeError("V337D tokenization gate must use boxed_suffix mode.")
    for split in ("train", "validation"):
        token_summary = gate_manifest.get("tokenization", {}).get(split, {})
        if int(token_summary.get("prompt_truncated", -1)) != 0:
            raise RuntimeError(f"V337D {split} prompt truncation is not zero.")
        if int(token_summary.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"V337D {split} completion token drop is not zero.")
        if int(token_summary.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"V337D {split} used fallback masks.")
    return {
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "tokenization_gate_manifest_sha256": sha256_file(TOKENIZATION_GATE_MANIFEST),
        "train_sha256": outputs.get("train_sha256"),
        "val_sha256": outputs.get("val_sha256"),
    }


def patch_launcher(module: Any) -> None:
    local_gate_summary = verify_v337d_tokenization_gate()
    run_id = "v338b-nemo-a100-minimal-transfer-balanced-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    module.VERSION = "v338b_minimal_transfer_balanced_from_v290_checkpoint6_nemo_a100"
    module.RUN_ID = run_id
    module.DATASET_UPLOAD_COMMIT = read_dataset_upload_commit()
    module.TRAIN_FILE = "data/v337d_minimal_transfer/20260513T_cpu_gate/v337d_minimal_transfer_train.jsonl"
    module.VAL_FILE = "data/v337d_minimal_transfer/20260513T_cpu_gate/v337d_minimal_transfer_val.jsonl"
    module.TRAIN_SHA256 = "df67214d3fdbb74ada96a9fc24609db5a3f5f6dc1d26dea5d4449eb39eb4147c"
    module.VAL_SHA256 = "50d4ee05a377ed4e111d27f9de0e1109eb0c09bfe01a9bce0717b63d704dbf80"
    module.TRAIN_ROWS = 1440
    module.VAL_ROWS = 340
    module.OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v338b-nemo-a100-minimal-transfer-balanced-v290ckpt6"
    module.MAX_STEPS = 14
    module.SAVE_EVERY_STEPS = 2
    module.EVAL_EVERY_STEPS = 2
    module.EVAL_MAX_EXAMPLES = 96
    module.ANSWER_SPAN_LOSS_WEIGHT = "16.0"
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "900"
    module.SOURCE_WEIGHTS = (
        "v330_symbolic_cryptarithm_distill=5.00,"
        "v325_equation_no_loss_distill=4.00,"
        "v337d_v217_bit_replay=8.00"
    )
    module.SUBCATEGORY_WEIGHTS = (
        "equation_symbolic_cryptarithm_single_operator_mul=10.00,"
        "equation_numeric_add_direct=8.00,"
        "equation_numeric_colon_absdiff=8.00,"
        "equation_numeric_minus_signed=8.00,"
        "bit_manipulation=3.00,"
        "unknown=3.00"
    )

    command_script = module.COMMAND_SCRIPT
    command_script = command_script.replace("export OUTPUT_DIR='/tmp/kg1_v331_output'", "export OUTPUT_DIR='/tmp/kg1_v338_output'")
    command_script = command_script.replace("export MAX_STEPS=10", "export MAX_STEPS=14")
    module.COMMAND_SCRIPT = command_script

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"] = (
            "equation_numeric_add_direct,equation_numeric_colon_absdiff,equation_numeric_minus_signed,"
            "equation_symbolic_cryptarithm_single_operator_mul,bit_manipulation,unknown"
        )
        env["KG1_REQUIRED_VAL_SUBCATEGORIES"] = env["KG1_REQUIRED_TRAIN_SUBCATEGORIES"]
        env["KG1_EXPECTED_MAX_STEPS"] = str(module.MAX_STEPS)
        return env

    module.build_job_env = build_job_env

    original_local_debug = module.local_debug

    def local_debug(api: Any, token: str) -> tuple[dict[str, object], dict[str, str]]:
        selected, env = original_local_debug(api, token)
        stale = [
            "data/v331_equation_bit_symbolic_mix",
            "v304_solver_trace_bit_fullbyte_distill",
            "bit_fullbyte_v300_gain_pattern",
            "v335_mixed_trace_replay",
        ]
        found = [item for item in stale if item in module.COMMAND_SCRIPT or item in json.dumps(env, sort_keys=True)]
        if found:
            raise RuntimeError("V338 launcher contains stale V331/V335 snippets: " + json.dumps(found))
        return selected, env

    module.local_debug = local_debug

    def write_manifest(payload: dict[str, Any]) -> Path:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = module.VERSION
        payload["run_id"] = module.RUN_ID
        payload["finops_kill_switch"] = {
            "first_checkpoint_total_min_exclusive": 192,
            "first_checkpoint_equation_min_exclusive": 56,
            "first_checkpoint_bit_min": 136,
            "action": "cancel HF job if first weak checkpoint cannot beat baseline gates",
        }
        payload["v337d_local_gate_summary"] = local_gate_summary
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

#!/usr/bin/env python3
"""Launch/debug V608 compact V446-source smoke on HF H200.

V608 is intentionally a tiny first-checkpoint experiment:

* continue from the V290 checkpoint-6 adapter;
* train only 2 steps and save/evaluate immediately;
* use the V607 compact source-only V446/Tong-aligned trace dataset;
* force example_mean and required row loss weights;
* defer V568 drift only until the first checkpoint exists, then weak eval must
  decide whether to continue or cancel.

Default mode is local debug only. Pass --launch to create the paid HF job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download


NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "h200"
RUN_ID = "v608-nemo-h200-v607-compact-v446-source-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v607-compact-v446-source-artifacts"
DATASET_UPLOAD_COMMIT = "eedc4f29b3a2180b3a6c56b4b86b8d26d1362b9e"
DATA_ROOT = "v607-compact-v446-source-20260518T-v607-cpu-gate"
TRAIN_FILE = DATA_ROOT + "/v607_compact_v446_source_train.jsonl"
VAL_FILE = DATA_ROOT + "/v607_compact_v446_source_val.jsonl"
TRAIN_SHA256 = "ad7e3797f01168b8e5c5342e46248e8fdac67d69c7315cd2c6e4a2e65ce284d3"
VAL_SHA256 = "2ddb2ef12e4b015bed78138a67738ce1bc0c80607f273beac62352fe6ffc0bc8"
TRAIN_ROWS = 1099
VAL_ROWS = 194
PREF_TRAIN_SHA256 = "ad7e3797f01168b8e5c5342e46248e8fdac67d69c7315cd2c6e4a2e65ce284d3"
PREF_VAL_SHA256 = "2ddb2ef12e4b015bed78138a67738ce1bc0c80607f273beac62352fe6ffc0bc8"
PREF_TRAIN_ROWS = 1099
PREF_VAL_ROWS = 194

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v608-v607compact-v290ckpt6"

MAX_STEPS = 2
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
EVAL_MAX_EXAMPLES = 194
MAX_LENGTH = 2048
ABORT_MAX_RESERVED_GIB = 84
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "0"
LOSS_NORMALIZATION_MODE = "example_mean"
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"

# Static/pre-paid gate contract. These literals are intentionally duplicated so
# scripts/kg1_pre_paid_job_integration_gate.py can validate a dynamic launcher
# without executing it.
KG1_STATIC_GATE_CONTRACT = {
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.09",
    "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
    "KG1_REQUIRED_TRAIN_FAMILIES": "bit_manipulation,equation_transform",
    "KG1_REQUIRED_VAL_FAMILIES": "bit_manipulation,equation_transform",
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
    "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101",
    "KG1_CPU_SIMULATED_TOTAL_CORRECT": "201",
    "KG1_CPU_SIMULATED_BIT_CORRECT": "138",
    "KG1_CPU_SIMULATED_EQUATION_CORRECT": "63",
    "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": "1.0",
    "KG1_CPU_SIMULATED_LOST_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_BIT_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": "0",
    "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.336",
    "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
    "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": "1",
    "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
    "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
    "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
}

# export DATA_REPO='felipesp1983/kg1-v607-compact-v446-source-artifacts'
# export MAX_LENGTH=2048
# export MAX_STEPS=2
# export SAVE_EVERY_STEPS=2
# export EVAL_EVERY_STEPS=2
# export ABORT_MAX_RESERVED_GIB=84
# export LOSS_NORMALIZATION_MODE=example_mean
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
# MAX_STEPS = "2"
# SAVE_EVERY_STEPS = "2"
# EVAL_EVERY_STEPS = "2"

SOURCE_WEIGHTS = "v607_compact_v446_source_dataset=1.00"
SUBCATEGORY_WEIGHTS = (
    "v607_v446_bit_compact_source=1.00,"
    "v607_v446_equation_compact_source=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "v607_v446_bit_compact_source,"
    "v607_v446_equation_compact_source"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"


def load_v536_module() -> Any:
    path = REPO_ROOT / "artifacts/v536_hf_h200_launch/launch_v536_hf_nemo_h200_v534_bit_v523_equation.py"
    spec = importlib.util.spec_from_file_location("kg1_v536_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load V536 launcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_module(module: Any) -> None:
    module.VERSION = "v608_v607_compact_v446_source_example_mean_roww_h200"
    module.NAMESPACE = NAMESPACE
    module.REPO_BRANCH = REPO_BRANCH
    module.EXPECTED_COMMIT = EXPECTED_COMMIT
    module.IMAGE = IMAGE
    module.FLAVOR = FLAVOR
    module.RUN_ID = RUN_ID
    module.DATA_REPO = DATA_REPO
    module.DATASET_UPLOAD_COMMIT = DATASET_UPLOAD_COMMIT
    module.DATA_ROOT = DATA_ROOT
    module.TRAIN_FILE = TRAIN_FILE
    module.VAL_FILE = VAL_FILE
    module.PREF_TRAIN_SHA256 = TRAIN_SHA256
    module.PREF_VAL_SHA256 = VAL_SHA256
    module.TRAIN_SHA256 = TRAIN_SHA256
    module.VAL_SHA256 = VAL_SHA256
    module.PREF_TRAIN_ROWS = TRAIN_ROWS
    module.PREF_VAL_ROWS = VAL_ROWS
    module.TRAIN_ROWS = TRAIN_ROWS
    module.VAL_ROWS = VAL_ROWS
    module.INIT_ADAPTER_REPO = INIT_ADAPTER_REPO
    module.INIT_ADAPTER_SUBFOLDER = INIT_ADAPTER_SUBFOLDER
    module.INIT_ADAPTER_TARGET_PARAMETERS = INIT_ADAPTER_TARGET_PARAMETERS
    module.OUTPUT_REPO = OUTPUT_REPO
    module.MAX_STEPS = MAX_STEPS
    module.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS
    module.EVAL_EVERY_STEPS = EVAL_EVERY_STEPS
    module.EVAL_MAX_EXAMPLES = EVAL_MAX_EXAMPLES
    module.MAX_LENGTH = MAX_LENGTH
    module.ABORT_MAX_RESERVED_GIB = ABORT_MAX_RESERVED_GIB
    module.ANSWER_SPAN_LOSS_WEIGHT = ANSWER_SPAN_LOSS_WEIGHT
    module.ANSWER_SPAN_MIN_WEIGHTED_TOKENS = ANSWER_SPAN_MIN_WEIGHTED_TOKENS
    module.LOSS_NORMALIZATION_MODE = LOSS_NORMALIZATION_MODE
    module.SOURCE_WEIGHTS = SOURCE_WEIGHTS
    module.SUBCATEGORY_WEIGHTS = SUBCATEGORY_WEIGHTS
    module.REQUIRED_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
    module.TRAINABLE_LORA_MODULES = TRAINABLE_LORA_MODULES
    module.REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE

    original_build_job_env = module.build_job_env

    def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
        env = original_build_job_env(hardware)
        env.update(
            {
                "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
                "KG1_TRAIN_FILE": TRAIN_FILE,
                "KG1_VAL_FILE": VAL_FILE,
                "KG1_TRAIN_SHA": TRAIN_SHA256,
                "KG1_VAL_SHA": VAL_SHA256,
                "KG1_TRAIN_ROWS": str(TRAIN_ROWS),
                "KG1_VAL_ROWS": str(VAL_ROWS),
                "KG1_OUTPUT_REPO": OUTPUT_REPO,
                "KG1_SOURCE_WEIGHTS": SOURCE_WEIGHTS,
                "KG1_SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
                "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
                "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
                "KG1_USE_ROW_LOSS_WEIGHT": USE_ROW_LOSS_WEIGHT,
                "KG1_REQUIRE_ROW_LOSS_WEIGHT": REQUIRE_ROW_LOSS_WEIGHT,
                "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
                "KG1_CPU_SIMULATED_TOTAL_CORRECT": "201",
                "KG1_CPU_SIMULATED_BIT_CORRECT": "138",
                "KG1_CPU_SIMULATED_EQUATION_CORRECT": "63",
                "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.336",
                "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
                "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
                "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
                "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
                "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
                "KG1_V608_CPU_GATES": "V509,V286,V478,V513,V524,V526",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONIOENCODING": "utf-8",
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
            }
        )
        return env

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
            "--use-row-loss-weight",
            "--require-row-loss-weight",
            "--min-bit-effective-share",
            "0.60",
            "--max-equation-effective-share",
            "0.40",
            "--max-any-family-effective-share",
            "0.75",
            "--output-json",
            str(out_path),
            "--enforce",
        ]
        print("local_objective_alignment_cmd =", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        print("local_objective_alignment_report =", json.dumps(report, indent=2, sort_keys=True), flush=True)
        return {"path": str(out_path), "report": report}

    original_configure_base = module.configure_base
    original_local_debug = module.local_debug

    def configure_base(base: Any) -> None:
        original_configure_base(base)
        base.MAX_STEPS = MAX_STEPS
        base.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS
        base.EVAL_EVERY_STEPS = EVAL_EVERY_STEPS
        base.EVAL_MAX_EXAMPLES = EVAL_MAX_EXAMPLES
        base.MAX_LENGTH = MAX_LENGTH
        base.ABORT_MAX_RESERVED_GIB = ABORT_MAX_RESERVED_GIB
        base.COMMAND_SCRIPT = (
            base.COMMAND_SCRIPT
            .replace("v536", "v608")
            .replace("V536", "V608")
            .replace(
                "export HF_HUB_ENABLE_HF_TRANSFER=1",
                "export HF_HUB_ENABLE_HF_TRANSFER=1\n"
                "export HF_HUB_DISABLE_PROGRESS_BARS=1\n"
                "export PYTHONIOENCODING=utf-8\n"
                "export LC_ALL=C.UTF-8\n"
                "export LANG=C.UTF-8",
            )
            .replace(
                'export LOSS_NORMALIZATION_MODE="$KG1_LOSS_NORMALIZATION_MODE"',
                'export LOSS_NORMALIZATION_MODE="$KG1_LOSS_NORMALIZATION_MODE"\n'
                'export USE_ROW_LOSS_WEIGHT="$KG1_USE_ROW_LOSS_WEIGHT"\n'
                'export REQUIRE_ROW_LOSS_WEIGHT="$KG1_REQUIRE_ROW_LOSS_WEIGHT"',
            )
            .replace(
                '    os.environ["SUBCATEGORY_WEIGHTS"],\n'
                '    "--min-bit-effective-share",',
                '    os.environ["SUBCATEGORY_WEIGHTS"],\n'
                '    "--use-row-loss-weight",\n'
                '    "--require-row-loss-weight",\n'
                '    "--min-bit-effective-share",',
            )
            .replace(f"export MAX_STEPS={module.MAX_STEPS}", f"export MAX_STEPS={MAX_STEPS}")
            .replace(f"export EVAL_MAX_EXAMPLES={module.EVAL_MAX_EXAMPLES}", f"export EVAL_MAX_EXAMPLES={EVAL_MAX_EXAMPLES}")
        )
        base.download_and_hash = download_and_hash_no_symlink_cache

    def local_debug(base: Any, api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        result = original_local_debug(base, api, token)
        row_weight_required = [
            'export USE_ROW_LOSS_WEIGHT="$KG1_USE_ROW_LOSS_WEIGHT"',
            'export REQUIRE_ROW_LOSS_WEIGHT="$KG1_REQUIRE_ROW_LOSS_WEIGHT"',
            '"--use-row-loss-weight"',
            '"--require-row-loss-weight"',
        ]
        missing = [item for item in row_weight_required if item not in base.COMMAND_SCRIPT]
        if missing:
            raise RuntimeError("remote command missing row-loss-weight contract snippets: " + json.dumps(missing))
        print("remote_row_loss_weight_contract_debug = ok", flush=True)
        return result

    module.build_job_env = build_job_env
    module.run_local_objective_alignment = run_local_objective_alignment
    module.configure_base = configure_base
    module.local_debug = local_debug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_hash_no_symlink_cache(repo_id: str, filename: str, expected_sha256: str, token: str) -> dict[str, Any]:
    download_root = Path(__file__).resolve().parent / "downloaded_debug" / RUN_ID
    download_root.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(download_root),
        )
    )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"HF downloaded file hash mismatch for {filename}: {observed} != {expected_sha256}")
    return {
        "repo_id": repo_id,
        "filename": filename,
        "local_path": str(path),
        "sha256": observed,
        "size": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V608.")
    api = HfApi(token=token)
    module = load_v536_module()
    patch_module(module)
    base = module.load_base_module()
    module.configure_base(base)
    selected_hardware, job_env, objective_alignment_info = module.local_debug(base, api, token)
    mode = "debug_only_no_job_launched"
    job = None
    if args.launch:
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

    manifest = module.manifest_payload(
        mode=mode,
        hardware=selected_hardware,
        job_env=job_env,
        objective_alignment_info=objective_alignment_info,
        job=job,
    )
    manifest["version"] = module.VERSION
    manifest["run_id"] = RUN_ID
    manifest["output_repo"] = OUTPUT_REPO
    manifest["next_action"] = "Run weak eval on checkpoint-2 immediately; cancel/stop unless ACC gate improves."
    out_path = Path(__file__).resolve().parent / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    if job is not None:
        print("job_url =", f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

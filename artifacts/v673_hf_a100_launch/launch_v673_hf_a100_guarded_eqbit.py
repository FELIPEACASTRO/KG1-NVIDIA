#!/usr/bin/env python3
"""Debug/launch V673 guarded equation+bit transfer on HF A100.

V673 is the current today-gain route:

* train only from synthetic/teacher rows, never weak labels;
* preserve the V290 checkpoint-6 LoRA contract;
* target the 14 direct residual opportunities found by the V672 ledger;
* use A100-large by default, with H200 blocked unless a later route proves A100
  cannot run.

Default mode is a local debug only. Pass ``--launch`` to create the paid HF job
after dataset hashes, adapter files, hardware cost, and objective-alignment
gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download


VERSION = "v673_a100_guarded_equation_bit_transfer_v290ckpt6"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "a100-large"
RUN_ID = "v673-a100-guarded-eqbit-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts"
DATASET_UPLOAD_COMMIT = "bb5459dc70434087ffc731da79333bf26b1a45ab"
DATA_ROOT = "v673-guarded-equation-bit-transfer-20260519T174833Z"
TRAIN_FILE = DATA_ROOT + "/v673_guarded_equation_bit_transfer_train.jsonl"
VAL_FILE = DATA_ROOT + "/v673_guarded_equation_bit_transfer_val.jsonl"
TRAIN_SHA256 = "cdf85573584c2bb965f8fb19bb8b698e7b03a7231013d39a74ff0410e0d76343"
VAL_SHA256 = "858c02fcc046d130c4405aac942c102aaf0ded38c347479734c5339d6960e057"
TRAIN_ROWS = 720
VAL_ROWS = 180
PREF_TRAIN_SHA256 = "cdf85573584c2bb965f8fb19bb8b698e7b03a7231013d39a74ff0410e0d76343"
PREF_VAL_SHA256 = "858c02fcc046d130c4405aac942c102aaf0ded38c347479734c5339d6960e057"
PREF_TRAIN_ROWS = 720
PREF_VAL_ROWS = 180

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6"

MAX_STEPS = 20
SAVE_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 10
EVAL_MAX_EXAMPLES = 180
MAX_LENGTH = 1024
ABORT_MAX_RESERVED_GIB = 72
LEARNING_RATE = "5.0e-7"
FINAL_LEARNING_RATE = "1.0e-7"
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "0"
LOSS_NORMALIZATION_MODE = "example_mean"
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"
LOSS_MASK_STOP_AFTER_EOS = "1"

SOURCE_WEIGHTS = "v673_guarded_equation_bit_transfer_dataset=1.00"
SUBCATEGORY_WEIGHTS = (
    "bit_exact_global_binary_replay=1.00,"
    "bit_exact_global_ternary_replay=1.00,"
    "bit_fullbyte_ternary_v366_new=1.00,"
    "equation_numeric_add_direct=1.00,"
    "equation_numeric_colon_trailing_zero=1.00,"
    "equation_numeric_minus_signed=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "bit_exact_global_binary_replay,"
    "bit_exact_global_ternary_replay,"
    "bit_fullbyte_ternary_v366_new,"
    "equation_numeric_add_direct,"
    "equation_numeric_colon_trailing_zero,"
    "equation_numeric_minus_signed"
)
TRAINABLE_LORA_MODULES = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
TRAINABLE_LORA_NAME_SUBSTRINGS = ""
REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS = (
    "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
)
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "1"

# Static/pre-paid gate literals. Keep these duplicated in source so the gate can
# audit without executing the launcher.
KG1_DATASET_SCHEMA = "sft"
KG1_HF_MAX_UNIT_COST_USD = "0.05"
KG1_EXPECTED_MAX_LENGTH = "1024"
KG1_EXPECTED_MAX_STEPS = "20"
KG1_EXPECTED_SAVE_EVERY_STEPS = "10"
KG1_EXPECTED_EVAL_EVERY_STEPS = "10"
KG1_EXPECTED_LOSS_NORMALIZATION_MODE = "example_mean"
KG1_REQUIRED_TRAIN_FAMILIES = "bit_manipulation,equation_transform"
KG1_REQUIRED_VAL_FAMILIES = "bit_manipulation,equation_transform"
KG1_REQUIRED_TRAIN_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
KG1_REQUIRED_VAL_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
KG1_RESIDUAL_FIRST_GATE = "1"
KG1_CRISIS_MODE_BACKFIRE_GUARD = "1"
KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS = "deferred_post_checkpoint"
KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT = "1"
KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED = "1"
KG1_V618_MODULE_SURFACE_GATE_STATUS = "passed"
KG1_V619_MODULE_SURFACE_GATE_STATUS = "passed"
KG1_ALLOW_CUDA13_ON_A100 = "1"
KG1_CUDA13_A100_DRIVER_GATE_STATUS = "inline_smoke_required"
KG1_V666_CPU_GATE_STACK_STATUS = "passed"
KG1_V666_CPU_GATE_STACK_REPORT = "artifacts/v673_hf_a100_launch/v673_v666_cpu_gate_stack.json"

KG1_STATIC_GATE_CONTRACT = {
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.05",
    "KG1_EXPECTED_MAX_LENGTH": KG1_EXPECTED_MAX_LENGTH,
    "KG1_EXPECTED_MAX_STEPS": KG1_EXPECTED_MAX_STEPS,
    "KG1_EXPECTED_SAVE_EVERY_STEPS": KG1_EXPECTED_SAVE_EVERY_STEPS,
    "KG1_EXPECTED_EVAL_EVERY_STEPS": KG1_EXPECTED_EVAL_EVERY_STEPS,
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": KG1_EXPECTED_LOSS_NORMALIZATION_MODE,
    "KG1_REQUIRED_TRAIN_FAMILIES": KG1_REQUIRED_TRAIN_FAMILIES,
    "KG1_REQUIRED_VAL_FAMILIES": KG1_REQUIRED_VAL_FAMILIES,
    "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_RESIDUAL_FIRST_GATE": KG1_RESIDUAL_FIRST_GATE,
    "KG1_CRISIS_MODE_BACKFIRE_GUARD": KG1_CRISIS_MODE_BACKFIRE_GUARD,
    "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS,
    "KG1_V618_MODULE_SURFACE_GATE_STATUS": KG1_V618_MODULE_SURFACE_GATE_STATUS,
    "KG1_ALLOW_CUDA13_ON_A100": KG1_ALLOW_CUDA13_ON_A100,
    "KG1_V666_CPU_GATE_STACK_STATUS": KG1_V666_CPU_GATE_STACK_STATUS,
    "KG1_V666_CPU_GATE_STACK_REPORT": KG1_V666_CPU_GATE_STACK_REPORT,
}

# Static searchable literals for integration gates.
# export DATA_REPO='felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts'
# export MAX_LENGTH=1024
# export MAX_STEPS=20
# export SAVE_EVERY_STEPS=10
# export EVAL_EVERY_STEPS=10
# export ABORT_MAX_RESERVED_GIB=72
# export LOSS_NORMALIZATION_MODE=example_mean
# export USE_ROW_LOSS_WEIGHT=1
# export REQUIRE_ROW_LOSS_WEIGHT=1
# export LOSS_MASK_STOP_AFTER_EOS=1
# export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
# export TRAINABLE_LORA_NAME_SUBSTRINGS=''
# export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
# export LORA_R=32
# export LORA_ALPHA=32
# local_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight --require-validation-row-loss-weight
# remote_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight --require-validation-row-loss-weight


COMMAND_SCRIPT = r"""set -eux
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
apt-get update -qq && apt-get install -y -qq git >/dev/null
$PYBIN - <<'PY'
import json
import torch

props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
payload = {
    "torch": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": bool(torch.cuda.is_available()),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
}
print("cuda13_a100_driver_gate_start = " + json.dumps(payload, sort_keys=True), flush=True)
if not payload["cuda_available"]:
    raise SystemExit("CUDA unavailable")
if "A100" not in payload["device"]:
    raise SystemExit(f"expected A100, observed {payload['device']!r}")
if payload["gpu_total_gib"] < 70:
    raise SystemExit(f"expected >=70GiB, observed {payload['gpu_total_gib']}")
x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x.T
torch.cuda.synchronize()
print("cuda13_a100_driver_gate_ok = " + json.dumps({
    "shape": list(y.shape),
    "mean": float(y.float().mean().detach().cpu()),
    "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
}, sort_keys=True), flush=True)
del x, y
torch.cuda.empty_cache()
PY
$PYBIN -m pip install -q --no-cache-dir --upgrade pip
$PYBIN -m pip install -q --no-cache-dir --upgrade 'huggingface_hub>=0.36.0' packaging wheel setuptools 'transformers==4.57.6' 'peft==0.19.1' 'accelerate>=1.10.0' safetensors sentencepiece hf_transfer ninja einops
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
git fetch --depth 1 origin "$KG1_EXPECTED_COMMIT" || true
git checkout --detach "$KG1_EXPECTED_COMMIT"
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch: expected=$KG1_EXPECTED_COMMIT observed=$observed" >&2; exit 12; fi
$PYBIN -m py_compile scripts/hf_job_train_v90.py scripts/hf_job_preflight_gate.py scripts/run_v485_peft_roundtrip_gate.py scripts/audit_v478_training_objective_alignment.py
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts'
export DATA_FILE="$KG1_TRAIN_FILE"
export VAL_FILE="$KG1_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_VAL_SHA"
export MIN_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_VAL_EXAMPLES="$KG1_VAL_ROWS"
export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_VAL_ROWS"
export OUTPUT_DIR='/tmp/kg1_v673_output'
export OUTPUT_REPO="$KG1_OUTPUT_REPO"
export RUN_ID="$KG1_RUN_ID"
export UPLOAD_TO_HF=1
export UPLOAD_CHECKPOINTS_DURING_TRAINING=1
export INIT_ADAPTER_REPO="$KG1_INIT_ADAPTER_REPO"
export INIT_ADAPTER_SUBFOLDER="$KG1_INIT_ADAPTER_SUBFOLDER"
export INIT_ADAPTER_LOAD_MODE='peft'
export PEFT_MANUAL_LOAD_METHOD='auto'
export FAIL_ON_MISSING_ADAPTER_KEYS=1
export LORA_R=32
export LORA_ALPHA=32
export LORA_DROPOUT=0.0
export LORA_TARGET_MODULES='down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj'
export LORA_TARGET_PARAMETERS="$KG1_LORA_TARGET_PARAMETERS"
export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
export TRAINABLE_LORA_NAME_SUBSTRINGS=''
export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'
export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1
export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
export MAX_TRAINABLE_PARAM_RATIO=0.060
export MAX_LENGTH=1024
export BATCH_SIZE=2
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=5.0e-7
export FINAL_LEARNING_RATE=1.0e-7
export NUM_EPOCHS=1
export MAX_STEPS=20
export SAVE_EVERY_STEPS=10
export EVAL_EVERY_STEPS=10
export EVAL_MAX_EXAMPLES=180
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export ANSWER_SPAN_LOSS_WEIGHT="$KG1_ANSWER_SPAN_LOSS_WEIGHT"
export ANSWER_SPAN_MIN_WEIGHTED_TOKENS="$KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS"
export LOSS_NORMALIZATION_MODE="$KG1_LOSS_NORMALIZATION_MODE"
export USE_ROW_LOSS_WEIGHT="$KG1_USE_ROW_LOSS_WEIGHT"
export REQUIRE_ROW_LOSS_WEIGHT="$KG1_REQUIRE_ROW_LOSS_WEIGHT"
export LOSS_MASK_STOP_AFTER_EOS="$KG1_LOSS_MASK_STOP_AFTER_EOS"
export BASELINE_EVAL_BEFORE_TRAIN=1
export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=0.16
export MAX_FINAL_EVAL_REGRESSION=0
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=72
export SAMPLING_MODE='weighted_replacement'
export SUBCATEGORY_WEIGHTS="$KG1_SUBCATEGORY_WEIGHTS"
export SOURCE_WEIGHTS="$KG1_SOURCE_WEIGHTS"
export MAX_PROMPT_TRUNCATION_RATE=0.0
export REQUIRE_OFFSET_MASK=1
export TOKENIZE_ONLY_DRY_RUN=0
export DRY_RUN_VALIDATE_ONLY=0
export USE_BITSANDBYTES=0
export MODEL_DEVICE_MAP='auto'
export ATTN_IMPLEMENTATION='eager'
export TORCH_ALLOW_TF32=1
export TORCH_FLOAT32_MATMUL_PRECISION='high'
export GRADIENT_CHECKPOINTING=1
$PYBIN scripts/hf_job_preflight_gate.py --phase preinstall
$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts
$PYBIN scripts/run_v485_peft_roundtrip_gate.py \
  --adapter-repo "$KG1_INIT_ADAPTER_REPO" \
  --adapter-subfolder "$KG1_INIT_ADAPTER_SUBFOLDER" \
  --expected-r 32 \
  --expected-alpha 32 \
  --expected-target-modules 'down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj' \
  --expected-target-parameters "$KG1_LORA_TARGET_PARAMETERS" \
  --output-json /tmp/kg1_v485_peft_roundtrip_gate_manifest.json
$PYBIN - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

train_path = hf_hub_download(
    repo_id=os.environ["DATA_REPO"],
    filename=os.environ["DATA_FILE"],
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN"),
)
val_path = hf_hub_download(
    repo_id=os.environ["DATA_REPO"],
    filename=os.environ["VAL_FILE"],
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN"),
)
output_json = "/tmp/kg1_v673_objective_alignment_gate.json"
cmd = [
    sys.executable,
    "scripts/audit_v478_training_objective_alignment.py",
    "--train-jsonl",
    train_path,
    "--val-jsonl",
    val_path,
    "--source-weights",
    os.environ["SOURCE_WEIGHTS"],
    "--subcategory-weights",
    os.environ["SUBCATEGORY_WEIGHTS"],
    "--use-row-loss-weight",
    "--require-row-loss-weight",
    "--require-validation-row-loss-weight",
    "--min-bit-effective-share",
    "0.10",
    "--max-equation-effective-share",
    "0.90",
    "--max-any-family-effective-share",
    "0.90",
    "--output-json",
    output_json,
    "--enforce",
]
print("objective_alignment_cmd = " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print("objective_alignment_gate_json = " + output_json, flush=True)
print(Path(output_json).read_text(encoding="utf-8"), flush=True)
PY
$PYBIN - <<'PY'
import inspect
import json
from peft import LoraConfig
import causal_conv1d
import mamba_ssm

payload = {
    "peft_lora_config_has_target_parameters": "target_parameters" in inspect.signature(LoraConfig.__init__).parameters,
    "causal_conv1d": getattr(causal_conv1d, "__version__", "ok"),
    "mamba_ssm": getattr(mamba_ssm, "__version__", "ok"),
}
print("nemo_dependency_probe = " + json.dumps(payload, sort_keys=True), flush=True)
if not payload["peft_lora_config_has_target_parameters"]:
    raise SystemExit("PEFT LoraConfig does not support target_parameters after upgrade")
PY
$PYBIN scripts/hf_job_preflight_gate.py --phase postinstall
$PYBIN scripts/hf_job_train_v90.py
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_to_dict(item: object) -> dict[str, object]:
    accelerator = getattr(item, "accelerator", None)
    return {
        "name": str(getattr(item, "name", "")),
        "pretty_name": str(getattr(item, "pretty_name", "")),
        "cpu": str(getattr(item, "cpu", "")),
        "ram": str(getattr(item, "ram", "")),
        "accelerator_model": str(getattr(accelerator, "model", "")) if accelerator else "",
        "accelerator_quantity": str(getattr(accelerator, "quantity", "")) if accelerator else "",
        "accelerator_vram": str(getattr(accelerator, "vram", "")) if accelerator else "",
        "unit_cost_usd": float(getattr(item, "unit_cost_usd", 0.0) or 0.0),
        "unit_label": str(getattr(item, "unit_label", "")),
    }


def download_and_hash(repo_id: str, filename: str, expected_sha: str, token: str) -> dict[str, object]:
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
    if observed.lower() != expected_sha.lower():
        raise RuntimeError(f"HF dataset hash mismatch for {filename}: {observed} != {expected_sha}")
    return {"filename": filename, "local_path": str(path), "sha256": observed, "bytes": path.stat().st_size}


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
        "--require-validation-row-loss-weight",
        "--min-bit-effective-share",
        "0.10",
        "--max-equation-effective-share",
        "0.90",
        "--max-any-family-effective-share",
        "0.90",
        "--output-json",
        str(out_path),
        "--enforce",
    ]
    print("local_objective_alignment_cmd =", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    print("local_objective_alignment_report =", json.dumps(report, indent=2, sort_keys=True), flush=True)
    return {"path": str(out_path), "report": report}


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "KG1_EXPECTED_TORCH_VERSION": "",
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
        "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "70",
        "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware["unit_cost_usd"]),
        "KG1_HF_MAX_UNIT_COST_USD": "0.05",
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_ALLOW_CUDA13_ON_A100": "1",
        "KG1_CUDA13_A100_DRIVER_GATE_STATUS": KG1_CUDA13_A100_DRIVER_GATE_STATUS,
        "KG1_DATASET_SCHEMA": "sft",
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
        "KG1_REQUIRED_TRAIN_FAMILIES": KG1_REQUIRED_TRAIN_FAMILIES,
        "KG1_REQUIRED_VAL_FAMILIES": KG1_REQUIRED_VAL_FAMILIES,
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_TRAIN_FILE": TRAIN_FILE,
        "KG1_VAL_FILE": VAL_FILE,
        "KG1_TRAIN_SHA": TRAIN_SHA256,
        "KG1_VAL_SHA": VAL_SHA256,
        "KG1_TRAIN_ROWS": str(TRAIN_ROWS),
        "KG1_VAL_ROWS": str(VAL_ROWS),
        "KG1_INIT_ADAPTER_REPO": INIT_ADAPTER_REPO,
        "KG1_INIT_ADAPTER_SUBFOLDER": INIT_ADAPTER_SUBFOLDER,
        "KG1_LORA_TARGET_PARAMETERS": INIT_ADAPTER_TARGET_PARAMETERS,
        "KG1_TRAINABLE_LORA_MODULES": TRAINABLE_LORA_MODULES,
        "KG1_TRAINABLE_LORA_NAME_SUBSTRINGS": TRAINABLE_LORA_NAME_SUBSTRINGS,
        "KG1_REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS": REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS,
        "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
        "KG1_ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
        "KG1_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_USE_ROW_LOSS_WEIGHT": USE_ROW_LOSS_WEIGHT,
        "KG1_REQUIRE_ROW_LOSS_WEIGHT": REQUIRE_ROW_LOSS_WEIGHT,
        "KG1_LOSS_MASK_STOP_AFTER_EOS": LOSS_MASK_STOP_AFTER_EOS,
        "KG1_SOURCE_WEIGHTS": SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
        "KG1_REQUIRE_MAMBA_IMPORTS": "1",
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
        "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.327",
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS,
        "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
        "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
        "KG1_V618_MODULE_SURFACE_GATE_STATUS": "passed",
        "KG1_V619_MODULE_SURFACE_GATE_STATUS": "passed",
        "KG1_V666_CPU_GATE_STACK_STATUS": KG1_V666_CPU_GATE_STACK_STATUS,
        "KG1_V666_CPU_GATE_STACK_REPORT": KG1_V666_CPU_GATE_STACK_REPORT,
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def check_no_active_paid_train_jobs(api: HfApi) -> list[dict[str, str]]:
    active_stages = {"PENDING", "QUEUED", "RUNNING", "SCHEDULED", "STARTING"}
    active: list[dict[str, str]] = []
    for job in api.list_jobs(namespace=NAMESPACE):
        stage = str(getattr(getattr(job, "status", None), "stage", "") or "").upper()
        if stage not in active_stages:
            continue
        env = getattr(job, "environment", {}) or {}
        run_id = str(env.get("KG1_RUN_ID", ""))
        output_repo = str(env.get("KG1_OUTPUT_REPO", ""))
        weak_eval = str(env.get("KG1_WEAK_EVAL_DIAGNOSTIC_ONLY", "0")) == "1"
        if weak_eval:
            continue
        if run_id.startswith("v") or "kg1-nemotron-lora" in output_repo:
            active.append(
                {
                    "id": str(getattr(job, "id", "")),
                    "stage": stage,
                    "run_id": run_id,
                    "output_repo": output_repo,
                    "url": str(getattr(job, "url", "")),
                }
            )
    return active


def local_debug(api: HfApi, token: str) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    print("=== V673 LAUNCHER DEBUG START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("version =", VERSION, flush=True)
    print("expected_commit =", EXPECTED_COMMIT, flush=True)
    print("image =", IMAGE, flush=True)
    print("flavor =", FLAVOR, flush=True)
    print("run_id =", RUN_ID, flush=True)

    hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
    if FLAVOR not in hardware:
        raise RuntimeError(f"HF flavor {FLAVOR!r} is not available. Available={sorted(hardware)}")
    selected = hardware[FLAVOR]
    if float(selected["unit_cost_usd"]) > float(KG1_HF_MAX_UNIT_COST_USD):
        raise RuntimeError(f"A100 unit cost above gate: {selected}")
    print("hf_hardware_selected =", json.dumps(selected, indent=2, sort_keys=True), flush=True)

    train_info = download_and_hash(DATA_REPO, TRAIN_FILE, TRAIN_SHA256, token)
    val_info = download_and_hash(DATA_REPO, VAL_FILE, VAL_SHA256, token)
    print("hf_train_file_ok =", json.dumps(train_info, sort_keys=True), flush=True)
    print("hf_val_file_ok =", json.dumps(val_info, sort_keys=True), flush=True)
    objective_alignment_info = run_local_objective_alignment(
        str(train_info["local_path"]),
        str(val_info["local_path"]),
    )

    adapter_files = set(api.list_repo_files(INIT_ADAPTER_REPO, repo_type="model"))
    required_adapter_files = {
        f"{INIT_ADAPTER_SUBFOLDER}/adapter_config.json",
        f"{INIT_ADAPTER_SUBFOLDER}/adapter_model.safetensors",
    }
    missing = sorted(required_adapter_files - adapter_files)
    if missing:
        raise RuntimeError("missing init adapter files: " + json.dumps(missing))
    print("init_adapter_files_ok =", json.dumps(sorted(required_adapter_files)), flush=True)

    required_snippets = [
        f"export DATA_REPO='{DATA_REPO}'",
        "export DATA_FILE=\"$KG1_TRAIN_FILE\"",
        "export VAL_FILE=\"$KG1_VAL_FILE\"",
        "cuda13_a100_driver_gate_ok",
        "export LORA_TARGET_PARAMETERS=\"$KG1_LORA_TARGET_PARAMETERS\"",
        "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'",
        "export TRAINABLE_LORA_NAME_SUBSTRINGS=''",
        "export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'",
        "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1",
        "export MAX_LENGTH=1024",
        "export MAX_STEPS=20",
        "export SAVE_EVERY_STEPS=10",
        "export EVAL_EVERY_STEPS=10",
        "export LEARNING_RATE=5.0e-7",
        "export FINAL_LEARNING_RATE=1.0e-7",
        "export LOSS_NORMALIZATION_MODE=\"$KG1_LOSS_NORMALIZATION_MODE\"",
        "export USE_ROW_LOSS_WEIGHT=\"$KG1_USE_ROW_LOSS_WEIGHT\"",
        "export REQUIRE_ROW_LOSS_WEIGHT=\"$KG1_REQUIRE_ROW_LOSS_WEIGHT\"",
        "export LOSS_MASK_STOP_AFTER_EOS=\"$KG1_LOSS_MASK_STOP_AFTER_EOS\"",
        "scripts/audit_v478_training_objective_alignment.py",
        "\"--use-row-loss-weight\"",
        "\"--require-row-loss-weight\"",
        "\"--require-validation-row-loss-weight\"",
        "$PYBIN scripts/hf_job_preflight_gate.py --phase artifacts",
        "$PYBIN scripts/hf_job_preflight_gate.py --phase postinstall",
        "$PYBIN scripts/hf_job_train_v90.py",
    ]
    missing_snippets = [item for item in required_snippets if item not in COMMAND_SCRIPT]
    if missing_snippets:
        raise RuntimeError("launcher command missing required snippets: " + json.dumps(missing_snippets))
    forbidden_snippets = [
        "h200",
        "v643-v641-plus-v367-bit-signal",
        "v664-close-think-boxed",
        "v312_verifier_synthetic=30.00",
        "WEAK_LABEL_AWARE_SELECTION=1",
    ]
    found_forbidden = [
        item
        for item in forbidden_snippets
        if item in COMMAND_SCRIPT or item in json.dumps(build_job_env(selected), sort_keys=True)
    ]
    if found_forbidden:
        raise RuntimeError("launcher contains forbidden stale snippets: " + json.dumps(found_forbidden))
    print("command_script_static_debug = ok", flush=True)

    job_env = build_job_env(selected)
    print("hf_job_env_debug =", json.dumps(job_env, indent=2, sort_keys=True), flush=True)
    print("=== V673 LAUNCHER DEBUG END ===", flush=True)
    return selected, job_env, objective_alignment_info


def manifest_payload(
    *,
    mode: str,
    hardware: dict[str, object],
    job_env: dict[str, str],
    objective_alignment_info: dict[str, object],
    active_job_blockers: list[dict[str, str]],
    job: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "namespace": NAMESPACE,
        "image": IMAGE,
        "flavor": FLAVOR,
        "hardware": hardware,
        "expected_commit": EXPECTED_COMMIT,
        "branch": REPO_BRANCH,
        "run_id": RUN_ID,
        "output_repo": OUTPUT_REPO,
        "dataset": {
            "data_repo": DATA_REPO,
            "dataset_upload_commit": DATASET_UPLOAD_COMMIT,
            "data_root": DATA_ROOT,
            "train_file": TRAIN_FILE,
            "val_file": VAL_FILE,
            "train_sha256": TRAIN_SHA256,
            "val_sha256": VAL_SHA256,
            "train_rows": TRAIN_ROWS,
            "val_rows": VAL_ROWS,
        },
        "init_adapter": {
            "repo": INIT_ADAPTER_REPO,
            "subfolder": INIT_ADAPTER_SUBFOLDER,
            "contract": "V290 r=32 alpha=32 target_modules plus target_parameters preserved",
        },
        "job_env": job_env,
        "objective_alignment": objective_alignment_info,
        "active_job_blockers": active_job_blockers,
        "recipe": {
            "max_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "eval_max_examples": EVAL_MAX_EXAMPLES,
            "max_length": MAX_LENGTH,
            "batch_size": 2,
            "micro_batch_size": 1,
            "abort_max_reserved_gib": ABORT_MAX_RESERVED_GIB,
            "lora_r": 32,
            "lora_alpha": 32,
            "target_modules": "down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj",
            "target_parameters": INIT_ADAPTER_TARGET_PARAMETERS,
            "trainable_lora_modules": TRAINABLE_LORA_MODULES,
            "trainable_lora_name_substrings": TRAINABLE_LORA_NAME_SUBSTRINGS,
            "required_trainable_lora_name_substrings": REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS,
            "target_parameters_trainability": "required_trainable",
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "loss_normalization_mode": LOSS_NORMALIZATION_MODE,
            "use_row_loss_weight": USE_ROW_LOSS_WEIGHT,
            "loss_mask_stop_after_eos": LOSS_MASK_STOP_AFTER_EOS,
            "source_weights": SOURCE_WEIGHTS,
            "subcategory_weights": SUBCATEGORY_WEIGHTS,
            "cuda13_a100_policy": "allowed only after inline torch matmul smoke gate in command",
            "promotion_gate": "first checkpoint weak eval required; promote only if bit>=136 equation>=60 total>=196, truncated=0, protected rows do not backfire",
        },
        "blocked_actions": ["kaggle_submit", "h200_fallback_without_new_a100_failure_analysis"],
        "next_action": "Launch with --launch only if no active paid train jobs; monitor logs, then weak-eval checkpoint-10 before any longer run.",
    }
    if job is not None:
        payload.update(
            {
                "job_id": job.id,
                "job_url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
                "job_status": str(job.status.stage if getattr(job, "status", None) else "unknown"),
            }
        )
    return payload


def write_manifest(payload: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    remote_path = out_dir / f"{RUN_ID}_remote_command.sh"
    remote_path.write_text(COMMAND_SCRIPT + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("remote_command_path =", remote_path, flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually create the paid HF job after local debug passes.")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("HF token is required to debug or launch V673.")
    api = HfApi(token=token)
    selected_hardware, job_env, objective_alignment_info = local_debug(api, token)
    active_job_blockers = check_no_active_paid_train_jobs(api)

    mode = "debug_only_no_job_launched"
    job = None
    if args.launch:
        if active_job_blockers:
            raise RuntimeError("Active paid train jobs block V673 launch: " + json.dumps(active_job_blockers, sort_keys=True))
        job = api.run_job(
            image=IMAGE,
            command=["/bin/bash", "-lc", COMMAND_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": token},
            flavor=FLAVOR,
            timeout=3600,
            namespace=NAMESPACE,
        )
        mode = "launched"

    manifest = manifest_payload(
        mode=mode,
        hardware=selected_hardware,
        job_env=job_env,
        objective_alignment_info=objective_alignment_info,
        active_job_blockers=active_job_blockers,
        job=job,
    )
    write_manifest(manifest)
    if job is not None:
        print("job_url =", f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

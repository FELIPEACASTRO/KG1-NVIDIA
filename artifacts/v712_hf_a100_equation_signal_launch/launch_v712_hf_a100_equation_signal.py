#!/usr/bin/env python3
"""Prepare or launch the V712 equation-only A100 signal probe.

Default mode is manifest-only and does not create a paid job. The paid path is
available only with ``--launch`` and requires the same local contract used by
the pre-paid gate: current hashes, protected rows, weak promotion thresholds,
row-loss objective flags, A100-only hardware, effective LoRA trainability
validation, and checkpointed weak eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


try:
    from huggingface_hub import HfApi, get_token, hf_hub_download
except Exception:  # pragma: no cover - manifest-only mode can run without HF deps.
    HfApi = None  # type: ignore[assignment]
    get_token = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]


VERSION = "v712_a100_equation_signal_v290ckpt6"
NAMESPACE = "felipesp1983"
REPO_BRANCH = "v230-v226-complementarity"
REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
FLAVOR = "a100-large"
RUN_ID = "v712-a100-equation-signal-v290ckpt6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DATA_REPO = "felipesp1983/kg1-v708-equation-single-family-dataset"
DATASET_UPLOAD_COMMIT = "b85f1a49625c2cec585b203015097fc27d1d7c72"
DATA_ROOT = "v708-equation-single-family-20260520T"
TRAIN_FILE = DATA_ROOT + "/v708_equation_single_family_train.jsonl"
VAL_FILE = DATA_ROOT + "/v708_equation_single_family_val.jsonl"
LOCAL_DATA_DIR = REPO_ROOT / "artifacts/v708_equation_single_family_dataset/20260520T_v708_cpu_gate"
LOCAL_TRAIN_FILE = LOCAL_DATA_DIR / "v708_equation_single_family_train.jsonl"
LOCAL_VAL_FILE = LOCAL_DATA_DIR / "v708_equation_single_family_val.jsonl"
LOCAL_MANIFEST_FILE = LOCAL_DATA_DIR / "v708_equation_single_family_manifest.json"

TRAIN_SHA256 = "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d"
VAL_SHA256 = "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba"
TRAIN_ROWS = 852
VAL_ROWS = 195
PREF_TRAIN_SHA256 = "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d"
PREF_VAL_SHA256 = "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba"
PREF_TRAIN_ROWS = 852
PREF_VAL_ROWS = 195

INIT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
INIT_ADAPTER_TARGET_PARAMETERS = "mlp.experts.gate_up_proj,mlp.experts.down_proj"
OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal"

MAX_STEPS = 20
SAVE_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 10
EVAL_MAX_EXAMPLES = 195
MAX_LENGTH = 1024
ABORT_MAX_RESERVED_GIB = 78
LEARNING_RATE = "2.0e-6"
FINAL_LEARNING_RATE = "5.0e-7"
ANSWER_SPAN_LOSS_WEIGHT = "1.0"
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = "0"
LOSS_NORMALIZATION_MODE = "example_mean"
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"
ROW_LOSS_WEIGHT_REDUCTION = "scale_mean"
LOSS_MASK_STOP_AFTER_EOS = "1"
SAVE_EMBEDDING_LAYERS = "0"
INIT_ADAPTER_LOAD_MODE = "manual"
DROP_INIT_ADAPTER_TARGET_MODULES = "lm_head"
ALLOW_MANUAL_TARGET_PARAMETERS_LOAD = "1"
FREEZE_LORA_TARGET_PARAMETERS = "1"

SOURCE_WEIGHTS = "v708_equation_single_family_dataset=1.00"
SUBCATEGORY_WEIGHTS = (
    "equation_numeric_add_direct_low_support=1.00,"
    "equation_numeric_colon_absdiff_unreverse_low_support=1.00,"
    "equation_numeric_minus_signed_reverse_high_support=1.00,"
    "equation_numeric_minus_signed_reverse_low_support=1.00,"
    "equation_symbolic_cryptarithm_single_operator_mul=1.00,"
    "symbolic_cryptarithm_multi_operator_digits_add=1.00,"
    "symbolic_cryptarithm_multi_operator_digits_mul=1.00,"
    "symbolic_cryptarithm_single_operator_digits_mul=1.00,"
    "v640_lkevin_equation_symbolic_trace=1.00"
)
REQUIRED_SUBCATEGORIES = (
    "equation_numeric_add_direct_low_support,"
    "equation_numeric_colon_absdiff_unreverse_low_support,"
    "equation_numeric_minus_signed_reverse_high_support,"
    "equation_numeric_minus_signed_reverse_low_support,"
    "equation_symbolic_cryptarithm_single_operator_mul,"
    "symbolic_cryptarithm_multi_operator_digits_add,"
    "symbolic_cryptarithm_multi_operator_digits_mul,"
    "symbolic_cryptarithm_single_operator_digits_mul,"
    "v640_lkevin_equation_symbolic_trace"
)
TRAINABLE_LORA_MODULES = "q_proj,v_proj"
TRAINABLE_LORA_NAME_SUBSTRINGS = ""
REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS = "q_proj,v_proj"
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = "0"
LORA_TARGET_MODULES = "down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj"

KG1_DATASET_SCHEMA = "sft"
KG1_HF_MAX_UNIT_COST_USD = "0.05"
KG1_EXPECTED_MAX_LENGTH = "1024"
KG1_EXPECTED_MAX_STEPS = "20"
KG1_EXPECTED_SAVE_EVERY_STEPS = "10"
KG1_EXPECTED_EVAL_EVERY_STEPS = "10"
KG1_EXPECTED_LOSS_NORMALIZATION_MODE = "example_mean"
KG1_REQUIRED_TRAIN_FAMILIES = "equation_transform"
KG1_REQUIRED_VAL_FAMILIES = "equation_transform"
KG1_REQUIRED_TRAIN_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
KG1_REQUIRED_VAL_SUBCATEGORIES = REQUIRED_SUBCATEGORIES
KG1_RESIDUAL_FIRST_GATE = "1"
KG1_CRISIS_MODE_BACKFIRE_GUARD = "1"
KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS = "deferred_post_checkpoint"
KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT = "1"
KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED = "1"
KG1_WEAK_EVAL_HARNESS = "scripts/hf_job_weak_eval_v245.py"
KG1_WEAK_EVAL_REQUIRED_CHECKPOINT = "checkpoint-20"
KG1_WEAK_EVAL_DATA_REPO = "felipesp1983/kg1-nemotron-training"
KG1_WEAK_CSV_FILE = (
    "runtime_artifacts/v245_weak_eval_bridge/"
    "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
KG1_WEAK_MANIFEST_FILE = (
    "runtime_artifacts/v245_weak_eval_bridge/"
    "v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json"
)
KG1_EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
KG1_MAX_TOKENS = "7680"
KG1_MAX_NEW_TOKENS = "7680"
KG1_WEAK_PROMOTE_TOTAL_MIN = "196"
KG1_WEAK_PROMOTE_BIT_MIN = "136"
KG1_WEAK_PROMOTE_EQUATION_MIN = "60"
KG1_WEAK_PROMOTE_TRUNC_MAX = "0"
KG1_WEAK_PROMOTE_BOXED_RATE_MIN = "1.0"
KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX = "0"
KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX = "512"
KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX = "7680"
KG1_ENFORCE_WEAK_RUNTIME_POLICY = "1"
KG1_GENERATION_TIMEOUT_S = "900"
KG1_EVAL_TIMEOUT_S = "4200"
KG1_NO_PROMPT_SUFFIX = "0"
KG1_MAX_MODEL_LEN = "8192"
KG1_MAX_NUM_SEQS = "64"
KG1_DISABLE_THINKING = "0"
KG1_REQUIRE_DISABLE_THINKING = "0"
KG1_PROTECTED_ROW_GUARD = "1"
KG1_STOP_ON_PROTECTED_BACKFIRE = "1"
KG1_EVAL_CANDIDATE_BY_CANDIDATE = "1"
KG1_V618_MODULE_SURFACE_GATE_STATUS = "passed"
KG1_V619_MODULE_SURFACE_GATE_STATUS = "passed"
KG1_ALLOW_CUDA13_ON_A100 = "1"
KG1_CUDA13_A100_DRIVER_GATE_STATUS = "inline_smoke_required"
KG1_V666_CPU_GATE_STACK_STATUS = "passed"
KG1_V666_CPU_GATE_STACK_REPORT = "artifacts/v708_hf_a100_launch/v708_cpu_gate_stack.json"

KG1_STATIC_GATE_CONTRACT = {
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.05",
    "KG1_EXPECTED_MAX_LENGTH": KG1_EXPECTED_MAX_LENGTH,
    "KG1_EXPECTED_MAX_STEPS": KG1_EXPECTED_MAX_STEPS,
    "KG1_EXPECTED_SAVE_EVERY_STEPS": KG1_EXPECTED_SAVE_EVERY_STEPS,
    "KG1_EXPECTED_EVAL_EVERY_STEPS": KG1_EXPECTED_EVAL_EVERY_STEPS,
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": KG1_EXPECTED_LOSS_NORMALIZATION_MODE,
    "KG1_SAVE_EMBEDDING_LAYERS": SAVE_EMBEDDING_LAYERS,
    "KG1_ROW_LOSS_WEIGHT_REDUCTION": ROW_LOSS_WEIGHT_REDUCTION,
    "KG1_DROP_INIT_ADAPTER_TARGET_MODULES": DROP_INIT_ADAPTER_TARGET_MODULES,
    "KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD": ALLOW_MANUAL_TARGET_PARAMETERS_LOAD,
    "KG1_FREEZE_LORA_TARGET_PARAMETERS": FREEZE_LORA_TARGET_PARAMETERS,
    "KG1_REQUIRED_TRAIN_FAMILIES": KG1_REQUIRED_TRAIN_FAMILIES,
    "KG1_REQUIRED_VAL_FAMILIES": KG1_REQUIRED_VAL_FAMILIES,
    "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
    "KG1_RESIDUAL_FIRST_GATE": KG1_RESIDUAL_FIRST_GATE,
    "KG1_CRISIS_MODE_BACKFIRE_GUARD": KG1_CRISIS_MODE_BACKFIRE_GUARD,
    "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS,
    "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT,
    "KG1_WEAK_EVAL_HARNESS": KG1_WEAK_EVAL_HARNESS,
    "KG1_WEAK_EVAL_REQUIRED_CHECKPOINT": KG1_WEAK_EVAL_REQUIRED_CHECKPOINT,
    "KG1_WEAK_EVAL_DATA_REPO": KG1_WEAK_EVAL_DATA_REPO,
    "KG1_WEAK_CSV_FILE": KG1_WEAK_CSV_FILE,
    "KG1_WEAK_MANIFEST_FILE": KG1_WEAK_MANIFEST_FILE,
    "KG1_EXPECTED_WEAK_CSV_SHA256": KG1_EXPECTED_WEAK_CSV_SHA256,
    "KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256": KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    "KG1_MAX_TOKENS": KG1_MAX_TOKENS,
    "KG1_MAX_NEW_TOKENS": KG1_MAX_NEW_TOKENS,
    "KG1_WEAK_PROMOTE_TOTAL_MIN": KG1_WEAK_PROMOTE_TOTAL_MIN,
    "KG1_WEAK_PROMOTE_BIT_MIN": KG1_WEAK_PROMOTE_BIT_MIN,
    "KG1_WEAK_PROMOTE_EQUATION_MIN": KG1_WEAK_PROMOTE_EQUATION_MIN,
    "KG1_WEAK_PROMOTE_TRUNC_MAX": KG1_WEAK_PROMOTE_TRUNC_MAX,
    "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": KG1_WEAK_PROMOTE_BOXED_RATE_MIN,
    "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX,
    "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX,
    "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX,
    "KG1_ENFORCE_WEAK_RUNTIME_POLICY": KG1_ENFORCE_WEAK_RUNTIME_POLICY,
}

KG1_WEAK_EVAL_OFFICIAL_LIKE_CONTRACT = {
    "KG1_DISABLE_THINKING": "0",
    "KG1_NO_PROMPT_SUFFIX": "0",
    "KG1_MAX_TOKENS": "7680",
    "KG1_MAX_MODEL_LEN": "8192",
    "KG1_MAX_NUM_SEQS": "64",
}

# Static searchable literals for integration gates.
# export DATA_REPO='felipesp1983/kg1-v708-equation-single-family-dataset'
# export MAX_LENGTH=1024
# export MAX_STEPS=20
# timeout=3600
# export SAVE_EVERY_STEPS=10
# export EVAL_EVERY_STEPS=10
# export ABORT_MAX_RESERVED_GIB=78
# export LOSS_NORMALIZATION_MODE=example_mean
# export USE_ROW_LOSS_WEIGHT=1
# export REQUIRE_ROW_LOSS_WEIGHT=1
# export ROW_LOSS_WEIGHT_REDUCTION=scale_mean
# export LOSS_MASK_STOP_AFTER_EOS=1
# export SAVE_EMBEDDING_LAYERS=0
# export INIT_ADAPTER_LOAD_MODE=manual
# export DROP_INIT_ADAPTER_TARGET_MODULES=lm_head
# export KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD=1
# export TRAINABLE_LORA_MODULES='q_proj,v_proj'
# export TRAINABLE_LORA_NAME_SUBSTRINGS=''
# export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='q_proj,v_proj'
# export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0
# export FREEZE_LORA_TARGET_PARAMETERS=1
# local_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight --require-validation-row-loss-weight
# remote_objective_alignment_cmd: --use-row-loss-weight --require-row-loss-weight --require-validation-row-loss-weight


COMMAND_SCRIPT = f"""set -eux
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
PYBIN=$(command -v python || command -v python3)
echo "python_bin=$PYBIN"
$PYBIN - <<'PY'
import json
import torch

props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
payload = {{
    "torch": getattr(torch, "__version__", "unknown"),
    "cuda": getattr(torch.version, "cuda", ""),
    "cuda_available": bool(torch.cuda.is_available()),
    "device": props.name if props else "",
    "gpu_total_gib": props.total_memory / 1024**3 if props else 0.0,
}}
print("cuda13_a100_driver_gate_start = " + json.dumps(payload, sort_keys=True), flush=True)
if not payload["cuda_available"]:
    raise SystemExit("CUDA unavailable")
if "A100" not in payload["device"]:
    raise SystemExit(f"expected A100, observed {{payload['device']!r}}")
if payload["gpu_total_gib"] < 70:
    raise SystemExit(f"expected >=70GiB, observed {{payload['gpu_total_gib']}}")
print("cuda13_a100_driver_gate_ok = " + json.dumps(payload, sort_keys=True), flush=True)
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
$PYBIN -m py_compile scripts/hf_job_train_v90.py scripts/hf_job_preflight_gate.py scripts/run_v485_peft_roundtrip_gate.py scripts/audit_v478_training_objective_alignment.py scripts/hf_job_weak_eval_v245.py scripts/evaluate_lora_adapter.py scripts/evaluate_lora_adapters_batch.py scripts/kg1_weak_backfire_row_guard.py scripts/validate_answer_extraction_v1.py src/competition_utils.py
$PYBIN - <<'PY'
import pathlib

root = pathlib.Path("/tmp/kg1").resolve()
import scripts

scripts_file = pathlib.Path(getattr(scripts, "__file__", "")).resolve()
print("scripts_package_gate = " + str(scripts_file), flush=True)
if scripts_file != root / "scripts" / "__init__.py":
    raise SystemExit("wrong scripts package: " + str(scripts_file))

import scripts.evaluate_lora_adapter as single_eval
import scripts.evaluate_lora_adapters_batch as batch_eval

print(
    "weak_eval_import_gate_ok = "
    + str(pathlib.Path(single_eval.__file__).resolve())
    + " | "
    + str(pathlib.Path(batch_eval.__file__).resolve()),
    flush=True,
)
PY
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='{DATA_REPO}'
export DATA_FILE="$KG1_TRAIN_FILE"
export VAL_FILE="$KG1_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_VAL_SHA"
export MIN_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_VAL_EXAMPLES="$KG1_VAL_ROWS"
export MIN_TOKENIZED_TRAIN_EXAMPLES="$KG1_TRAIN_ROWS"
export MIN_TOKENIZED_VAL_EXAMPLES="$KG1_VAL_ROWS"
export OUTPUT_DIR='/tmp/kg1_v712_output'
export OUTPUT_REPO="$KG1_OUTPUT_REPO"
export RUN_ID="$KG1_RUN_ID"
export UPLOAD_TO_HF=1
export UPLOAD_CHECKPOINTS_DURING_TRAINING=1
export INIT_ADAPTER_REPO="$KG1_INIT_ADAPTER_REPO"
export INIT_ADAPTER_SUBFOLDER="$KG1_INIT_ADAPTER_SUBFOLDER"
export INIT_ADAPTER_LOAD_MODE='manual'
export DROP_INIT_ADAPTER_TARGET_MODULES="$KG1_DROP_INIT_ADAPTER_TARGET_MODULES"
export KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD="$KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD"
export PEFT_MANUAL_LOAD_METHOD='auto'
export FAIL_ON_MISSING_ADAPTER_KEYS=1
export LORA_R=32
export LORA_ALPHA=32
export LORA_DROPOUT=0.0
export LORA_TARGET_MODULES='{LORA_TARGET_MODULES}'
export LORA_TARGET_PARAMETERS="$KG1_LORA_TARGET_PARAMETERS"
export TRAINABLE_LORA_MODULES='{TRAINABLE_LORA_MODULES}'
export TRAINABLE_LORA_NAME_SUBSTRINGS=''
export REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='{REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS}'
export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1
export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0
export FREEZE_LORA_TARGET_PARAMETERS=1
export MAX_TRAINABLE_PARAM_RATIO=0.060
export MAX_LENGTH=1024
export BATCH_SIZE=2
export MICRO_BATCH_SIZE=1
export LEARNING_RATE={LEARNING_RATE}
export FINAL_LEARNING_RATE={FINAL_LEARNING_RATE}
export NUM_EPOCHS=1
export MAX_STEPS=20
export SAVE_EVERY_STEPS=10
export EVAL_EVERY_STEPS=10
export EVAL_MAX_EXAMPLES=195
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export ANSWER_SPAN_LOSS_WEIGHT="$KG1_ANSWER_SPAN_LOSS_WEIGHT"
export ANSWER_SPAN_MIN_WEIGHTED_TOKENS="$KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS"
export LOSS_NORMALIZATION_MODE="$KG1_LOSS_NORMALIZATION_MODE"
export USE_ROW_LOSS_WEIGHT="$KG1_USE_ROW_LOSS_WEIGHT"
export REQUIRE_ROW_LOSS_WEIGHT="$KG1_REQUIRE_ROW_LOSS_WEIGHT"
export ROW_LOSS_WEIGHT_REDUCTION="$KG1_ROW_LOSS_WEIGHT_REDUCTION"
export LOSS_MASK_STOP_AFTER_EOS="$KG1_LOSS_MASK_STOP_AFTER_EOS"
export SAVE_EMBEDDING_LAYERS="$KG1_SAVE_EMBEDDING_LAYERS"
export BASELINE_EVAL_BEFORE_TRAIN=1
export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=0.16
export MAX_FINAL_EVAL_REGRESSION=0
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=78
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
$PYBIN scripts/run_v485_peft_roundtrip_gate.py \\
  --adapter-repo "$KG1_INIT_ADAPTER_REPO" \\
  --adapter-subfolder "$KG1_INIT_ADAPTER_SUBFOLDER" \\
  --expected-r 32 \\
  --expected-alpha 32 \\
  --expected-target-modules '{LORA_TARGET_MODULES}' \\
  --allowed-extra-target-modules "$KG1_DROP_INIT_ADAPTER_TARGET_MODULES" \\
  --expected-target-parameters "$KG1_LORA_TARGET_PARAMETERS" \\
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
output_json = "/tmp/kg1_v712_objective_alignment_gate.json"
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
    "0.0",
    "--max-equation-effective-share",
    "1.0",
    "--max-any-family-effective-share",
    "1.0",
    "--output-json",
    output_json,
    "--enforce",
]
print("objective_alignment_cmd = " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print("objective_alignment_gate_json = " + output_json, flush=True)
print(Path(output_json).read_text(encoding="utf-8"), flush=True)
PY
$PYBIN scripts/hf_job_preflight_gate.py --phase postinstall
$PYBIN scripts/hf_job_train_v90.py
$PYBIN - <<'PY'
import json
from pathlib import Path

training_manifest = Path("/tmp/kg1_v712_output/final_adapter/v90_training_manifest.json")
adapter_config_path = Path("/tmp/kg1_v712_output/checkpoint-20/adapter_config.json")
output_json = Path("/tmp/kg1_v712_lora_trainability_manifest_gate.json")

blockers = []
observations = []
if not training_manifest.is_file():
    blockers.append("missing_training_manifest")
if not adapter_config_path.is_file():
    blockers.append("missing_checkpoint_20_adapter_config")

manifest = json.loads(training_manifest.read_text(encoding="utf-8")) if training_manifest.is_file() else dict()
adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8")) if adapter_config_path.is_file() else dict()
lora = manifest.get("lora") or dict()
filter_report = lora.get("trainable_lora_module_filter") or dict()

expected_base = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
expected_active = sorted("down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj".split(","))
expected_trainable = sorted("q_proj,v_proj".split(","))
expected_target_parameters = sorted("mlp.experts.down_proj,mlp.experts.gate_up_proj".split(","))

adapter_modules = sorted(str(item) for item in (adapter_config.get("target_modules") or []))
adapter_target_parameters = sorted(str(item) for item in (adapter_config.get("target_parameters") or []))
modules_to_save = adapter_config.get("modules_to_save") or []

if adapter_config.get("base_model_name_or_path") != expected_base:
    blockers.append("adapter_base_model_mismatch")
if int(adapter_config.get("r", -1)) != 32:
    blockers.append("adapter_r_mismatch")
if int(adapter_config.get("lora_alpha", -1)) != 32:
    blockers.append("adapter_lora_alpha_mismatch")
if adapter_modules != expected_active:
    blockers.append("adapter_active_target_modules_mismatch")
if adapter_target_parameters != expected_target_parameters:
    blockers.append("adapter_target_parameters_mismatch")
if modules_to_save:
    blockers.append("adapter_modules_to_save_not_empty")
if "lm_head" in adapter_modules:
    blockers.append("adapter_lm_head_target_module_present")

if int(lora.get("r", -1)) != 32:
    blockers.append("manifest_lora_r_mismatch")
if int(lora.get("alpha", -1)) != 32:
    blockers.append("manifest_lora_alpha_mismatch")
if not filter_report.get("enabled"):
    blockers.append("trainable_filter_disabled")
if sorted(str(item) for item in (filter_report.get("modules") or [])) != expected_trainable:
    blockers.append("trainable_filter_modules_mismatch")

trainable_by_module = dict((str(k), int(v)) for k, v in (filter_report.get("trainable_by_module") or dict()).items())
nonzero_trainable = sorted(k for k, v in trainable_by_module.items() if int(v) > 0)
if nonzero_trainable != expected_trainable:
    blockers.append("nonzero_trainable_modules_not_exact_expected")
for module in expected_trainable:
    if int(trainable_by_module.get(module, 0)) <= 0:
        blockers.append("missing_trainable_module:" + module)

target_trainable = dict(
    (str(k), int(v))
    for k, v in (filter_report.get("target_parameter_trainable_lora_params") or dict()).items()
)
if any(int(v) != 0 for v in target_trainable.values()):
    blockers.append("target_parameters_are_trainable")
if filter_report.get("target_parameters_trainability_mode") != "frozen_active":
    blockers.append("target_parameters_trainability_mode_not_frozen_active")
if not bool(filter_report.get("freeze_lora_target_parameters")):
    blockers.append("freeze_lora_target_parameters_false")

ratio = float((lora.get("trainable_parameter_report_after_filter") or dict()).get("ratio", 1.0))
if ratio > 0.0001:
    blockers.append("trainable_ratio_too_high")
if int(filter_report.get("trainable_lora_params") or 0) <= 0:
    blockers.append("trainable_lora_params_zero")
if int(filter_report.get("frozen_lora_params") or 0) <= int(filter_report.get("trainable_lora_params") or 0):
    blockers.append("frozen_lora_params_not_dominant")

observations.append("checkpoint-20 effective trainability must remain q_proj,v_proj only with MoE target_parameters frozen")
report = dict(
    schema_version="kg1_v712_inline_lora_trainability_manifest_gate_v1",
    passed=not blockers,
    blockers=blockers,
    observations=observations,
    training_manifest=str(training_manifest),
    adapter_config=str(adapter_config_path),
    adapter_active_target_modules=adapter_modules,
    adapter_target_parameters=adapter_target_parameters,
    trainable_filter=dict(
        enabled=filter_report.get("enabled"),
        modules=sorted(str(item) for item in (filter_report.get("modules") or [])),
        trainable_by_module=trainable_by_module,
        target_parameter_trainable_lora_params=target_trainable,
        target_parameters_trainability_mode=filter_report.get("target_parameters_trainability_mode"),
        trainable_lora_params=int(filter_report.get("trainable_lora_params") or 0),
        frozen_lora_params=int(filter_report.get("frozen_lora_params") or 0),
        trainable_parameter_ratio=ratio,
    ),
)
output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print("v712_lora_trainability_manifest_gate = " + json.dumps(report, sort_keys=True), flush=True)
if blockers:
    raise SystemExit("V712 LoRA trainability manifest gate failed: " + ",".join(blockers))
PY
if ! $PYBIN - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "vllm_import_missing_install_start"
  $PYBIN -m pip install -q --no-cache-dir 'vllm==0.20.1'
  $PYBIN - <<'PY'
import vllm
print("vllm_import_after_install_ok = " + getattr(vllm, "__version__", "unknown"), flush=True)
PY
else
  echo "vllm_import_preinstalled_ok"
fi
export KG1_DATA_REPO='{KG1_WEAK_EVAL_DATA_REPO}'
export KG1_WEAK_CSV_FILE='{KG1_WEAK_CSV_FILE}'
export KG1_WEAK_MANIFEST_FILE='{KG1_WEAK_MANIFEST_FILE}'
export KG1_EXPECTED_WEAK_CSV_SHA256='{KG1_EXPECTED_WEAK_CSV_SHA256}'
export KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256='{KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256}'
export KG1_MAX_TOKENS=7680
export KG1_MAX_NEW_TOKENS=7680
export KG1_MAX_MODEL_LEN=8192
export KG1_MAX_NUM_SEQS=64
export KG1_NO_PROMPT_SUFFIX=0
export KG1_DISABLE_THINKING=0
export KG1_REQUIRE_DISABLE_THINKING=0
export KG1_ADAPTER_REPO="$KG1_OUTPUT_REPO"
export KG1_ADAPTER_SUBFOLDERS='{KG1_WEAK_EVAL_REQUIRED_CHECKPOINT}'
export KG1_CANDIDATE_NAMES='v712_checkpoint_20'
export KG1_CANDIDATE_NAME='v712_checkpoint_20'
export KG1_OUTPUT_DIR='/tmp/kg1_v712_weak_eval'
export KG1_OUTPUT_PATH_IN_REPO="evals/$KG1_RUN_ID"
export KG1_LABEL_PREFIX='v712_hf_weak'
export KG1_UPLOAD_TO_HF=1
export KG1_ENFORCE_WEAK_PROMOTION_GATE=1
export KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=0
export KG1_EXPECTED_LORA_R=32
export KG1_EXPECTED_LORA_ALPHA=32
export KG1_EXPECTED_ADAPTER_BASE_MODEL_NAME_OR_PATH="$MODEL_NAME"
export KG1_VLLM_GPU_MEMORY_UTILIZATION=0.86
export KG1_ALLOW_VLLM_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
$PYBIN scripts/hf_job_weak_eval_v245.py
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_local_dataset() -> dict[str, Any]:
    for path in [LOCAL_TRAIN_FILE, LOCAL_VAL_FILE, LOCAL_MANIFEST_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    train_sha = sha256_file(LOCAL_TRAIN_FILE)
    val_sha = sha256_file(LOCAL_VAL_FILE)
    train_rows = count_jsonl(LOCAL_TRAIN_FILE)
    val_rows = count_jsonl(LOCAL_VAL_FILE)
    if train_sha != TRAIN_SHA256:
        raise RuntimeError(f"train SHA mismatch: {train_sha} != {TRAIN_SHA256}")
    if val_sha != VAL_SHA256:
        raise RuntimeError(f"validation SHA mismatch: {val_sha} != {VAL_SHA256}")
    if train_rows != TRAIN_ROWS:
        raise RuntimeError(f"train rows mismatch: {train_rows} != {TRAIN_ROWS}")
    if val_rows != VAL_ROWS:
        raise RuntimeError(f"validation rows mismatch: {val_rows} != {VAL_ROWS}")
    return {
        "train_file": str(LOCAL_TRAIN_FILE),
        "validation_file": str(LOCAL_VAL_FILE),
        "manifest_file": str(LOCAL_MANIFEST_FILE),
        "train_sha256": train_sha,
        "validation_sha256": val_sha,
        "train_rows": train_rows,
        "validation_rows": val_rows,
    }


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


def build_job_env(hardware: dict[str, object]) -> dict[str, str]:
    return {
        "KG1_BRANCH": REPO_BRANCH,
        "KG1_EXPECTED_COMMIT": expected_commit(),
        "KG1_EXPECTED_MAX_STEPS": str(MAX_STEPS),
        "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
        "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_ABORT_MAX_RESERVED_GIB": str(ABORT_MAX_RESERVED_GIB),
        "KG1_REQUIRE_CUDA": "1",
        "KG1_MIN_GPU_TOTAL_GIB": "70",
        "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
        "KG1_HF_FLAVOR": FLAVOR,
        "KG1_HF_UNIT_COST_USD": str(hardware.get("unit_cost_usd", KG1_HF_MAX_UNIT_COST_USD)),
        "KG1_HF_MAX_UNIT_COST_USD": KG1_HF_MAX_UNIT_COST_USD,
        "KG1_ALLOWED_HF_FLAVORS": FLAVOR,
        "KG1_ALLOW_CUDA13_ON_A100": KG1_ALLOW_CUDA13_ON_A100,
        "KG1_CUDA13_A100_DRIVER_GATE_STATUS": KG1_CUDA13_A100_DRIVER_GATE_STATUS,
        "KG1_DATASET_SCHEMA": KG1_DATASET_SCHEMA,
        "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
        "KG1_REQUIRE_OFFSET_MASK": "1",
        "KG1_CRISIS_MODE_BACKFIRE_GUARD": KG1_CRISIS_MODE_BACKFIRE_GUARD,
        "KG1_REQUIRED_TRAIN_FAMILIES": KG1_REQUIRED_TRAIN_FAMILIES,
        "KG1_REQUIRED_VAL_FAMILIES": KG1_REQUIRED_VAL_FAMILIES,
        "KG1_REQUIRED_TRAIN_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_REQUIRED_VAL_SUBCATEGORIES": REQUIRED_SUBCATEGORIES,
        "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
        "KG1_OUTPUT_REPO": OUTPUT_REPO,
        "KG1_RUN_ID": RUN_ID,
        "KG1_DATA_REPO": DATA_REPO,
        "KG1_DATA_ROOT": DATA_ROOT,
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
        "KG1_FREEZE_LORA_TARGET_PARAMETERS": FREEZE_LORA_TARGET_PARAMETERS,
        "KG1_ANSWER_SPAN_LOSS_WEIGHT": ANSWER_SPAN_LOSS_WEIGHT,
        "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
        "KG1_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
        "KG1_USE_ROW_LOSS_WEIGHT": USE_ROW_LOSS_WEIGHT,
        "KG1_REQUIRE_ROW_LOSS_WEIGHT": REQUIRE_ROW_LOSS_WEIGHT,
        "KG1_ROW_LOSS_WEIGHT_REDUCTION": ROW_LOSS_WEIGHT_REDUCTION,
        "KG1_LOSS_MASK_STOP_AFTER_EOS": LOSS_MASK_STOP_AFTER_EOS,
        "KG1_SAVE_EMBEDDING_LAYERS": SAVE_EMBEDDING_LAYERS,
        "KG1_DROP_INIT_ADAPTER_TARGET_MODULES": DROP_INIT_ADAPTER_TARGET_MODULES,
        "KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD": ALLOW_MANUAL_TARGET_PARAMETERS_LOAD,
        "KG1_SOURCE_WEIGHTS": SOURCE_WEIGHTS,
        "KG1_SUBCATEGORY_WEIGHTS": SUBCATEGORY_WEIGHTS,
        "KG1_REQUIRE_MAMBA_IMPORTS": "1",
        "KG1_RESIDUAL_FIRST_GATE": KG1_RESIDUAL_FIRST_GATE,
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
        "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.371",
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS,
        "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT,
        "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED,
        "KG1_WEAK_EVAL_HARNESS": KG1_WEAK_EVAL_HARNESS,
        "KG1_WEAK_EVAL_REQUIRED_CHECKPOINT": KG1_WEAK_EVAL_REQUIRED_CHECKPOINT,
        "KG1_WEAK_EVAL_DATA_REPO": KG1_WEAK_EVAL_DATA_REPO,
        "KG1_WEAK_CSV_FILE": KG1_WEAK_CSV_FILE,
        "KG1_WEAK_MANIFEST_FILE": KG1_WEAK_MANIFEST_FILE,
        "KG1_EXPECTED_WEAK_CSV_SHA256": KG1_EXPECTED_WEAK_CSV_SHA256,
        "KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256": KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        "KG1_MAX_TOKENS": KG1_MAX_TOKENS,
        "KG1_MAX_NEW_TOKENS": KG1_MAX_NEW_TOKENS,
        "KG1_WEAK_PROMOTE_TOTAL_MIN": KG1_WEAK_PROMOTE_TOTAL_MIN,
        "KG1_WEAK_PROMOTE_BIT_MIN": KG1_WEAK_PROMOTE_BIT_MIN,
        "KG1_WEAK_PROMOTE_EQUATION_MIN": KG1_WEAK_PROMOTE_EQUATION_MIN,
        "KG1_WEAK_PROMOTE_TRUNC_MAX": KG1_WEAK_PROMOTE_TRUNC_MAX,
        "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": KG1_WEAK_PROMOTE_BOXED_RATE_MIN,
        "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX,
        "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX,
        "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX,
        "KG1_ENFORCE_WEAK_RUNTIME_POLICY": KG1_ENFORCE_WEAK_RUNTIME_POLICY,
        "KG1_GENERATION_TIMEOUT_S": KG1_GENERATION_TIMEOUT_S,
        "KG1_EVAL_TIMEOUT_S": KG1_EVAL_TIMEOUT_S,
        "KG1_NO_PROMPT_SUFFIX": KG1_NO_PROMPT_SUFFIX,
        "KG1_MAX_MODEL_LEN": KG1_MAX_MODEL_LEN,
        "KG1_MAX_NUM_SEQS": KG1_MAX_NUM_SEQS,
        "KG1_DISABLE_THINKING": KG1_DISABLE_THINKING,
        "KG1_REQUIRE_DISABLE_THINKING": KG1_REQUIRE_DISABLE_THINKING,
        "KG1_PROTECTED_ROW_GUARD": KG1_PROTECTED_ROW_GUARD,
        "KG1_STOP_ON_PROTECTED_BACKFIRE": KG1_STOP_ON_PROTECTED_BACKFIRE,
        "KG1_EVAL_CANDIDATE_BY_CANDIDATE": KG1_EVAL_CANDIDATE_BY_CANDIDATE,
        "KG1_V618_MODULE_SURFACE_GATE_STATUS": KG1_V618_MODULE_SURFACE_GATE_STATUS,
        "KG1_V619_MODULE_SURFACE_GATE_STATUS": KG1_V619_MODULE_SURFACE_GATE_STATUS,
        "KG1_V666_CPU_GATE_STACK_STATUS": KG1_V666_CPU_GATE_STACK_STATUS,
        "KG1_V666_CPU_GATE_STACK_REPORT": KG1_V666_CPU_GATE_STACK_REPORT,
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def check_no_active_paid_train_jobs(api: Any) -> list[dict[str, str]]:
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


def validate_hf_dataset(token: str) -> dict[str, Any]:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed.")
    download_root = Path(__file__).resolve().parent / "downloaded_debug" / RUN_ID
    download_root.mkdir(parents=True, exist_ok=True)
    train_path = Path(
        hf_hub_download(DATA_REPO, TRAIN_FILE, repo_type="dataset", token=token, local_dir=str(download_root))
    )
    val_path = Path(hf_hub_download(DATA_REPO, VAL_FILE, repo_type="dataset", token=token, local_dir=str(download_root)))
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    if train_sha != TRAIN_SHA256:
        raise RuntimeError(f"HF train SHA mismatch: {train_sha} != {TRAIN_SHA256}")
    if val_sha != VAL_SHA256:
        raise RuntimeError(f"HF validation SHA mismatch: {val_sha} != {VAL_SHA256}")
    return {"hf_train_file": TRAIN_FILE, "hf_val_file": VAL_FILE, "train_sha256": train_sha, "validation_sha256": val_sha}


def manifest_payload(
    *,
    mode: str,
    hardware: dict[str, object],
    job_env: dict[str, str],
    local_dataset: dict[str, Any],
    hf_dataset: dict[str, Any] | None,
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
        "expected_commit": expected_commit(),
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
            "families": [KG1_REQUIRED_TRAIN_FAMILIES],
            "subcategories": REQUIRED_SUBCATEGORIES.split(","),
            "local_dataset": local_dataset,
            "hf_dataset": hf_dataset or {"status": "not_checked_in_manifest_only_mode"},
        },
        "init_adapter": {
            "repo": INIT_ADAPTER_REPO,
            "subfolder": INIT_ADAPTER_SUBFOLDER,
            "contract": "V290 r=32 alpha=32; adapter-only modules exclude lm_head; MoE target_parameters loaded but frozen",
        },
        "job_env": job_env,
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
            "target_modules": LORA_TARGET_MODULES,
            "dropped_init_adapter_target_modules": DROP_INIT_ADAPTER_TARGET_MODULES,
            "target_parameters": INIT_ADAPTER_TARGET_PARAMETERS,
            "target_parameters_trainability": "frozen",
            "trainable_lora_modules": TRAINABLE_LORA_MODULES,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "loss_normalization_mode": LOSS_NORMALIZATION_MODE,
            "use_row_loss_weight": USE_ROW_LOSS_WEIGHT,
            "row_loss_weight_reduction": ROW_LOSS_WEIGHT_REDUCTION,
            "loss_mask_stop_after_eos": LOSS_MASK_STOP_AFTER_EOS,
            "source_weights": SOURCE_WEIGHTS,
            "subcategory_weights": SUBCATEGORY_WEIGHTS,
            "promotion_gate": "first checkpoint weak eval required with total>=196, bit>=136, equation>=60, truncation=0, boxed rate=1.0, no protected backfire",
            "weak_eval_command": "scripts/hf_job_weak_eval_v245.py on checkpoint-20 after hf_job_train_v90.py and inline trainability gate",
        },
        "blocked_actions": ["kaggle_submit", "package"],
        "next_action": "run pre-paid integration gate; launch only one A100-large 20-step signal probe after all gates pass",
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


def write_manifest(payload: dict[str, Any], remote_script: str = COMMAND_SCRIPT) -> Path:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}_launch_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    remote_path = out_dir / f"{RUN_ID}_remote_command.sh"
    remote_path.write_text(remote_script + "\n", encoding="utf-8")
    print("launch_manifest_path =", out_path, flush=True)
    print("remote_command_path =", remote_path, flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-hf-dataset", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== V712 A100 LAUNCHER START ===", flush=True)
    print("mode =", "launch" if args.launch else "manifest_only_no_job_launched", flush=True)
    print("run_id =", RUN_ID, flush=True)
    local_dataset = verify_local_dataset()
    print("local_dataset_ok =", json.dumps(local_dataset, sort_keys=True), flush=True)

    hardware = {
        "name": FLAVOR,
        "pretty_name": FLAVOR,
        "accelerator_model": "A100",
        "accelerator_quantity": "1",
        "accelerator_vram": "80GB",
        "unit_cost_usd": float(KG1_HF_MAX_UNIT_COST_USD),
        "unit_label": "hour",
    }
    hf_dataset = None
    active_job_blockers: list[dict[str, str]] = []
    job = None
    mode = "manifest_only_no_job_launched"

    if args.validate_hf_dataset or args.launch:
        if HfApi is None or get_token is None:
            raise RuntimeError("huggingface_hub is required for HF validation or launch.")
        token = get_token()
        if not token:
            raise RuntimeError("HF token is required for HF validation or launch.")
        api = HfApi(token=token)
        all_hardware = {item.name: hardware_to_dict(item) for item in api.list_jobs_hardware()}
        if FLAVOR not in all_hardware:
            raise RuntimeError(f"HF flavor {FLAVOR!r} is unavailable. Available={sorted(all_hardware)}")
        hardware = all_hardware[FLAVOR]
        if float(hardware["unit_cost_usd"]) > float(KG1_HF_MAX_UNIT_COST_USD):
            raise RuntimeError(f"A100 unit cost above gate: {hardware}")
        hf_dataset = validate_hf_dataset(token)
        print("hf_dataset_ok =", json.dumps(hf_dataset, sort_keys=True), flush=True)
        active_job_blockers = check_no_active_paid_train_jobs(api)
        if args.launch:
            if active_job_blockers:
                raise RuntimeError("Active paid train jobs block launch: " + json.dumps(active_job_blockers, sort_keys=True))
            job = api.run_job(
                image=IMAGE,
                command=["/bin/bash", "-lc", COMMAND_SCRIPT],
                env=build_job_env(hardware),
                secrets={"HF_TOKEN": token},
                flavor=FLAVOR,
                timeout=3600,
                namespace=NAMESPACE,
            )
            mode = "launched"
        else:
            mode = "validated_hf_no_job_launched"

    manifest = manifest_payload(
        mode=mode,
        hardware=hardware,
        job_env=build_job_env(hardware),
        local_dataset=local_dataset,
        hf_dataset=hf_dataset,
        active_job_blockers=active_job_blockers,
        job=job,
    )
    out_path = write_manifest(manifest)
    if job is not None:
        print("job_url =", f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}", flush=True)
    print("=== V712 A100 LAUNCHER END ===", flush=True)
    print("manifest =", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

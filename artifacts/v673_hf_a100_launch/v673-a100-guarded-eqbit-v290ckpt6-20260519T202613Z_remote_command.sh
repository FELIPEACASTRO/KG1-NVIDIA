set -eux
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
export LORA_TARGET_MODULES='down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj'
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
$PYBIN scripts/run_v485_peft_roundtrip_gate.py \
  --adapter-repo "$KG1_INIT_ADAPTER_REPO" \
  --adapter-subfolder "$KG1_INIT_ADAPTER_SUBFOLDER" \
  --expected-r 32 \
  --expected-alpha 32 \
  --expected-target-modules 'down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj' \
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

#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo "V80 HF Jobs - dgxchen v7 EXACT replica wrapper"
echo "=================================================================="

export DATA_REPO="${DATA_REPO:-${HF_DATA_REPO:-felipesp1983/kg1-nemotron-training}}"
export HF_DATA_REPO="${HF_DATA_REPO:-$DATA_REPO}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

# Report GPU status at start
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Upgrade pip + install base deps (cxx11 ABI may need tuning for Unsloth)
python -m pip install -q --upgrade pip
python -m pip install -q --upgrade setuptools wheel ninja packaging

# Core deps (pinned for dgxchen recipe compatibility)
python -m pip install -q \
  "transformers>=4.48,<4.58" \
  "peft>=0.14,<0.18" \
  "trl>=0.14,<0.26" \
  "accelerate>=1.0,<2.0" \
  "datasets>=3.2,<5" \
  "bitsandbytes" \
  huggingface_hub safetensors einops sentencepiece pandas

# Unsloth + unsloth_zoo (dgxchen recipe framework)
# May fail on some Docker images due to CUDA/torch version mismatch.
# Training script has fallback to pure transformers+peft.
python -m pip install -q "unsloth" "unsloth_zoo" || \
  echo "Unsloth install failed - training script will use pure transformers+peft fallback"

# xformers (for attention, if Unsloth installed)
python -m pip install -q xformers || \
  echo "xformers install failed - non-fatal"

# mamba-ssm + causal-conv1d (required by NemotronH architecture)
# Try pre-built wheels for PyTorch 2.5.1 / CUDA 12.4 / Python 3.11
python -m pip install -q \
  "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  || echo "mamba-ssm wheel failed; trying generic pip install"

python -m pip install -q \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  || echo "causal-conv1d wheel failed - non-fatal, transformers may fallback to slow path"

# Fallback: generic pip install if wheels failed
python -c "import mamba_ssm" 2>/dev/null || \
  python -m pip install -q mamba-ssm --no-build-isolation || \
  echo "WARN: mamba-ssm unavailable - NemotronH may error on attn 'eager' path if Mamba kernel required"

# Verify critical imports before downloading training script
python -c "
import torch, transformers, peft, trl, accelerate
print(f'torch={torch.__version__} cuda_avail={torch.cuda.is_available()}')
print(f'transformers={transformers.__version__}')
print(f'peft={peft.__version__} trl={trl.__version__} accelerate={accelerate.__version__}')
try:
    import unsloth
    print(f'unsloth={unsloth.__version__}')
except ImportError:
    print('unsloth=NOT_AVAILABLE (will use pure transformers+peft)')
try:
    import mamba_ssm
    print(f'mamba_ssm=AVAILABLE')
except ImportError:
    print('mamba_ssm=NOT_AVAILABLE')
"

# Download + execute V80 training script from HF dataset repo
python - <<'PY'
import os
from huggingface_hub import get_token, hf_hub_download

repo_id = os.environ.get("HF_DATA_REPO", "felipesp1983/kg1-nemotron-training")
token = os.environ.get("HF_TOKEN") or get_token()
path = hf_hub_download(
    repo_id=repo_id,
    filename="scripts/hf_job_train_v80.py",
    repo_type="dataset",
    token=token,
)
print(f"Executing {path}")
with open(path, encoding="utf-8") as f:
    code = compile(f.read(), path, "exec")
exec(code, {"__name__": "__main__", "__file__": path})
PY

echo "=================================================================="
echo "V80 HF Jobs wrapper DONE"
echo "=================================================================="

#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo "V80 HF Jobs v2 - dgxchen v7 EXACT + torch pin fix"
echo "=================================================================="

export DATA_REPO="${DATA_REPO:-${HF_DATA_REPO:-felipesp1983/kg1-nemotron-training}}"
export HF_DATA_REPO="${HF_DATA_REPO:-$DATA_REPO}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

# Report GPU status
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ============================================================
# CRITICAL: torch 2.5.1 pin BEFORE anything else
# Unsloth's pip install pulls torch>=2.5 but gets 2.10 (latest),
# breaking ABI with mamba-ssm wheels compiled for torch 2.5.
# Fix: pin torch==2.5.1 as FIRST install, then everything respects it.
# ============================================================
echo ""
echo "[1/6] PIN torch==2.5.1 + torchvision + torchaudio (match mamba-ssm wheels)..."
python -m pip install -q --upgrade pip
python -m pip install -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall || {
    echo "Torch pin failed - trying without --index-url"
    python -m pip install -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --force-reinstall
}

python -c "import torch; print(f'After pin: torch={torch.__version__}')"

# ============================================================
# [2/6] Install mamba-ssm + causal-conv1d wheels for torch 2.5 + cu12 + cp311
# ============================================================
echo ""
echo "[2/6] Install mamba-ssm + causal-conv1d wheels (cu12torch2.5 cp311)..."
python -m pip install -q \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  || echo "mamba-ssm 2.2.4 wheel failed - will retry with v2.3.0"

python -c "import mamba_ssm; print(f'mamba_ssm={mamba_ssm.__version__}')" 2>/dev/null || {
    python -m pip install -q \
      "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
      || echo "mamba-ssm 2.3.0 wheel failed"
}

python -m pip install -q \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  || echo "causal-conv1d wheel failed"

# Verify critical imports BEFORE proceeding (fail fast if ABI broken)
echo ""
echo "[3/6] Verify mamba-ssm import (fail fast if broken)..."
python -c "
import torch
print(f'torch={torch.__version__} cuda_avail={torch.cuda.is_available()}')
import mamba_ssm
print(f'mamba_ssm={mamba_ssm.__version__}')
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
print('OK rmsnorm_fn')
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('OK selective_scan_fn')
try:
    import causal_conv1d
    print(f'causal_conv1d OK')
except ImportError:
    print('causal_conv1d NOT_AVAILABLE (may work without)')
" || {
    echo "CRITICAL: mamba-ssm import broken after install. Aborting."
    exit 1
}

# ============================================================
# [4/6] Install ML stack (transformers, peft, trl, accelerate) PINNED to torch 2.5 compat
# No torch upgrade because torch is already installed.
# ============================================================
echo ""
echo "[4/6] Install transformers/peft/trl/accelerate (torch 2.5 compat range)..."
python -m pip install -q --upgrade setuptools wheel ninja packaging
python -m pip install -q \
  "transformers>=4.48,<4.58" \
  "peft>=0.14,<0.18" \
  "trl>=0.14,<0.26" \
  "accelerate>=1.0,<2.0" \
  "datasets>=3.2,<5" \
  "bitsandbytes" \
  huggingface_hub safetensors einops sentencepiece pandas \
  --no-deps
# --no-deps prevents torch from being upgraded by transitive deps

# Install remaining missing deps (without upgrading torch)
python -m pip install -q \
  "transformers>=4.48,<4.58" \
  "peft>=0.14,<0.18" \
  "trl>=0.14,<0.26" \
  "accelerate>=1.0,<2.0" \
  "datasets>=3.2,<5" \
  "bitsandbytes" \
  huggingface_hub safetensors einops sentencepiece pandas

# ============================================================
# [5/6] Install Unsloth (may try to upgrade torch - we check + rollback)
# ============================================================
echo ""
echo "[5/6] Install unsloth + unsloth_zoo (with --no-deps to prevent torch upgrade)..."
python -m pip install -q "unsloth" "unsloth_zoo" --no-deps \
  || echo "Unsloth install failed - training script has fallback to pure transformers+peft"

# After Unsloth, verify torch is STILL 2.5.1
python -c "
import torch
v = torch.__version__
if not v.startswith('2.5'):
    print(f'WARN: torch upgraded to {v} - forcing back to 2.5.1')
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
                    'torch==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cu124'])
else:
    print(f'OK torch={v} (pinned)')
"

python -m pip install -q xformers --no-deps || echo "xformers skipped (non-fatal)"

# ============================================================
# [6/6] Final verification + launch training
# ============================================================
echo ""
echo "[6/6] Final verification..."
python -c "
import torch
print(f'torch={torch.__version__} cuda_avail={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
import peft; print(f'peft={peft.__version__}')
import trl; print(f'trl={trl.__version__}')
import accelerate; print(f'accelerate={accelerate.__version__}')
import mamba_ssm; print(f'mamba_ssm={mamba_ssm.__version__}')
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('OK mamba_ssm cuda extension working')
except ImportError as e:
    print(f'FAIL mamba_ssm cuda: {e}')
    raise SystemExit(1)
try:
    import unsloth
    print(f'unsloth={unsloth.__version__}')
except ImportError:
    print('unsloth=NOT_AVAILABLE (training script will use pure transformers+peft fallback)')
"

# Download + execute V80 training script from HF dataset repo
echo ""
echo "=================================================================="
echo "Launching V80 training script..."
echo "=================================================================="
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

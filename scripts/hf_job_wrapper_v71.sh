#!/bin/bash
# HF Job wrapper for V71 training (huikang Tinker recipe)
# Usage: bash hf_job_wrapper_v71.sh
# Hardware: A100 80GB recommended

set -e

echo "=== Installing dependencies ==="
pip install -q transformers>=4.48 peft>=0.14 datasets huggingface_hub safetensors einops sentencepiece bitsandbytes

echo "=== Installing mamba-ssm ==="
pip install -q mamba-ssm --no-build-isolation

echo "=== Installing causal-conv1d (optional) ==="
pip install -q causal-conv1d --no-build-isolation || echo "causal-conv1d failed (optional - stub will be used)"

echo "=== Downloading V71 training script ==="
python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='felipesp1983/kg1-nemotron-training',
    filename='scripts/hf_job_train_v71.py',
    repo_type='dataset',
    token=os.environ['HF_TOKEN']
)
print(f'Script downloaded: {path}')

# Execute the script
exec(open(path).read())
"

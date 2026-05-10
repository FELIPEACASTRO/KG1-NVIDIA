#!/usr/bin/env python3
"""Build the V244 HF H200 smoke-train launcher notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "KG1_V244_HF_H200_SMOKE_TRAIN_COLAB.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    colab_url = (
        "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
        "v230-v226-complementarity/notebooks/KG1_V244_HF_H200_SMOKE_TRAIN_COLAB.ipynb"
    )
    github_url = (
        "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
        "v230-v226-complementarity/notebooks/KG1_V244_HF_H200_SMOKE_TRAIN_COLAB.ipynb"
    )
    cells = [
        markdown_cell(
            "# KG1 V244 HF H200 Smoke Train\n\n"
            f"Colab URL: {colab_url}\n\n"
            f"GitHub URL: {github_url}\n\n"
            "This notebook is a guarded Hugging Face Jobs launcher. It does not "
            "submit to Kaggle and it does not run training unless `RUN_TRAIN=1`."
        ),
        code_cell(
            """# CELL: global configuration, hard locks, and HF smoke-train gates.
print('=== V244 CONFIG START ===', flush=True)

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

VERSION = 'V244_HF_H200_SMOKE_TRAIN_20260510'
REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
REPO_BRANCH = 'v230-v226-complementarity'
ROOT = pathlib.Path('/content/kg1')
OUT_ROOT = pathlib.Path('/content/kg1_v244_hf_h200_smoke_train')
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = os.environ.get('KG1_V244_RUN_ID', 'v244-h200-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))

EXPECTED_REPO_COMMIT = os.environ.get('KG1_V244_EXPECTED_REPO_COMMIT', '').strip()
EXPECTED_TRAIN_SHA256 = 'c290555bffade5f4fa4e5c14f6f66c36745bd31a22c4b004709afd5a5f33f6d1'
EXPECTED_VAL_SHA256 = '54eda74b1ea01e6e3b165af23c99eac5dc6e21f29cbc49888503ea7a3d707764'
KG1_TRAIN_SHA = EXPECTED_TRAIN_SHA256
KG1_VAL_SHA = EXPECTED_VAL_SHA256
MIN_TRAIN_EXAMPLES = 12006
MIN_VAL_EXAMPLES = 921
MIN_TOKENIZED_TRAIN_EXAMPLES = 12006
MIN_TOKENIZED_VAL_EXAMPLES = 921

DATA_REPO = 'felipesp1983/kg1-nemotron-training'
DATA_FILE = 'runtime_artifacts/v243_training_mix/local_upload_20260510T180200Z/v243_training_mix_train.jsonl'
VAL_FILE = 'runtime_artifacts/v243_training_mix/local_upload_20260510T180200Z/v243_training_mix_validation.jsonl'
OUTPUT_REPO = 'felipesp1983/kg1-nemotron-lora-v243-safe-equation-fixtures'

MODEL_NAME = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
MODEL_REVISION = 'cbd3fa9f933d55ef16a84236559f4ee2a0526848'
INIT_ADAPTER_REPO = 'felipesp1983/kg1-nemotron-lora-v188-equation-lmhead'
INIT_ADAPTER_SUBFOLDER = 'checkpoint-40'
INIT_ADAPTER_DIR = ''
V194_ADAPTER = '/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter'

HF_FLAVOR = os.environ.get('KG1_V244_HF_FLAVOR', 'h200')
RUN_TRAIN = os.environ.get('KG1_V244_RUN_TRAIN', '0') == '1'
RUN_HF_JOB = RUN_TRAIN
ALLOW_KAGGLE_SUBMIT = False
ALLOW_HF_JOB_CANCEL = os.environ.get('KG1_V244_ALLOW_CANCEL', '0') == '1'
PREVIOUS_HF_JOB_ID = os.environ.get('KG1_V244_PREVIOUS_HF_JOB_ID', '')

TOKENIZE_ONLY_DRY_RUN = False
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
SAMPLING_MODE = 'weighted_replacement'
MAX_STEPS = 4
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 4
EVAL_MAX_EXAMPLES = 128
WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
weak_gate_pass_for_full = False

if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V244.')
if SAMPLING_MODE not in {'shuffle', 'weighted_replacement'}:
    raise RuntimeError('Invalid SAMPLING_MODE: ' + SAMPLING_MODE)
if MAX_STEPS != 4:
    raise RuntimeError('V244 smoke train must keep MAX_STEPS=4 unless a new notebook version is created.')

for key, value in {
    'VERSION': VERSION,
    'REPO_BRANCH': REPO_BRANCH,
    'EXPECTED_REPO_COMMIT': EXPECTED_REPO_COMMIT,
    'RUN_ID': RUN_ID,
    'RUN_TRAIN': RUN_TRAIN,
    'HF_FLAVOR': HF_FLAVOR,
    'DATA_REPO': DATA_REPO,
    'DATA_FILE': DATA_FILE,
    'VAL_FILE': VAL_FILE,
    'KG1_TRAIN_SHA': KG1_TRAIN_SHA,
    'KG1_VAL_SHA': KG1_VAL_SHA,
    'OUTPUT_REPO': OUTPUT_REPO,
    'MAX_STEPS': MAX_STEPS,
    'SAMPLING_MODE': SAMPLING_MODE,
    'ALLOW_KAGGLE_SUBMIT': ALLOW_KAGGLE_SUBMIT,
    'V194_ADAPTER': V194_ADAPTER,
    'INIT_ADAPTER_DIR': INIT_ADAPTER_DIR,
}.items():
    print(f'{key} = {value}', flush=True)

print('adapter_config.json and adapter_model.safetensors are checked by hf_job_train_v90.py during adapter load.', flush=True)
print('target_modules and target_parameters are logged and guarded by hf_job_train_v90.py.', flush=True)
print('If EXPECTED_REPO_COMMIT is empty, V244 pins the HF Job to the cloned branch HEAD observed in repo preflight.', flush=True)
print('=== V244 CONFIG END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: helper functions with command logging and HF hardware probes.
print('=== V244 HELPERS START ===', flush=True)

import hashlib
import subprocess


def run_cmd(cmd, *, cwd='.', log_path=None, check=True, timeout_s=None):
    printable = ' '.join(str(x) for x in cmd)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', printable, flush=True)
    print('timeout_s =', timeout_s, flush=True)
    if log_path is not None:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print('log_path =', log_path, flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    if log_path is not None:
        log_path.write_text(proc.stdout, encoding='utf-8')
    if proc.stdout:
        print(proc.stdout[-12000:], flush=True)
    print('returncode =', proc.returncode, flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError('command failed rc=' + str(proc.returncode) + ': ' + printable)
    return proc.returncode


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hf_token():
    token = os.environ.get('HF_TOKEN', '')
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token() or ''
        except Exception:
            token = ''
    if RUN_HF_JOB and not token:
        raise RuntimeError('HF_TOKEN is required when RUN_TRAIN=1.')
    return token


def hardware_to_dict(item):
    accelerator = getattr(item, 'accelerator', None)
    return {
        'name': str(getattr(item, 'name', '')),
        'pretty_name': str(getattr(item, 'pretty_name', '')),
        'cpu': str(getattr(item, 'cpu', '')),
        'ram': str(getattr(item, 'ram', '')),
        'accelerator_model': str(getattr(accelerator, 'model', '')) if accelerator else '',
        'accelerator_quantity': str(getattr(accelerator, 'quantity', '')) if accelerator else '',
        'accelerator_vram': str(getattr(accelerator, 'vram', '')) if accelerator else '',
        'unit_cost': float(getattr(item, 'unit_cost_usd', 0.0) or 0.0),
        'unit_label': str(getattr(item, 'unit_label', '')),
    }


print('COMMAND START and COMMAND END markers are emitted by run_cmd.', flush=True)
print('returncode and log_path are emitted by run_cmd.', flush=True)
print('=== V244 HELPERS END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: clone repo and compile training scripts.
print('=== V244 REPO PREFLIGHT START ===', flush=True)

if ROOT.exists():
    run_cmd([sys.executable, '-c', 'import shutil, pathlib; shutil.rmtree(pathlib.Path("/content/kg1"))'], cwd='/content', log_path=OUT_ROOT / 'repo_cleanup.log', check=True, timeout_s=120)

run_cmd(
    ['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)],
    cwd='/content',
    log_path=OUT_ROOT / 'repo_clone.log',
    check=True,
    timeout_s=300,
)

repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
print('repo_commit =', repo_commit, flush=True)
if EXPECTED_REPO_COMMIT and repo_commit != EXPECTED_REPO_COMMIT:
    raise RuntimeError('repo commit mismatch: expected ' + EXPECTED_REPO_COMMIT + ', got ' + repo_commit)
if not EXPECTED_REPO_COMMIT:
    EXPECTED_REPO_COMMIT = repo_commit
    print('EXPECTED_REPO_COMMIT inferred from cloned branch HEAD =', EXPECTED_REPO_COMMIT, flush=True)

compile_targets = [
    ROOT / 'scripts' / 'hf_job_train_v90.py',
    ROOT / 'scripts' / 'hf_job_preflight_gate.py',
    ROOT / 'scripts' / 'build_v243_training_mix.py',
    ROOT / 'scripts' / 'audit_jsonl_overlap.py',
    ROOT / 'scripts' / 'notebook_release_gate.py',
]
for target in compile_targets:
    print('compile_target =', target, 'exists =', target.exists(), flush=True)
    if not target.exists():
        raise FileNotFoundError(target)
run_cmd([sys.executable, '-m', 'py_compile'] + [str(x) for x in compile_targets], cwd=ROOT, log_path=OUT_ROOT / 'py_compile.log', check=True, timeout_s=180)
run_cmd([sys.executable, str(ROOT / 'scripts' / 'notebook_release_gate.py'), str(ROOT / 'notebooks' / 'KG1_V244_HF_H200_SMOKE_TRAIN_COLAB.ipynb')], cwd=ROOT, log_path=OUT_ROOT / 'notebook_release_gate.log', check=True, timeout_s=180)

print('=== V244 REPO PREFLIGHT END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: HF hardware, token, runtime, and data contract preflight.
print('=== V244 HF HARDWARE PREFLIGHT START ===', flush=True)

try:
    import torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = props.name if props else ''
    gpu_total_gib = props.total_memory / 1024**3 if props else 0.0
except Exception as exc:
    cuda_available = False
    gpu_name = ''
    gpu_total_gib = 0.0
    print('torch_cuda_probe_error =', repr(exc), flush=True)
content_free_gib = 0.0
try:
    import shutil
    content_free_gib = shutil.disk_usage('/content').free / 1024**3
except Exception as exc:
    print('content_disk_probe_error =', repr(exc), flush=True)
print('cuda_available =', cuda_available, flush=True)
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)

run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.36.0'], cwd='/content', log_path=OUT_ROOT / 'pip_install_huggingface_hub.log', check=True, timeout_s=180)
from huggingface_hub import HfApi

HF_TOKEN = require_hf_token()
api = HfApi(token=HF_TOKEN or None)
hardware = [hardware_to_dict(item) for item in api.list_jobs_hardware()]
print('hf_hardware =', json.dumps(hardware, indent=2, sort_keys=True), flush=True)
by_name = {item['name']: item for item in hardware}
for required_flavor in ['a100-large', 'h200']:
    if required_flavor not in by_name:
        raise RuntimeError('Required HF hardware flavor missing: ' + required_flavor)
    print('hf_required_flavor =', json.dumps(by_name[required_flavor], sort_keys=True), flush=True)
if HF_FLAVOR not in by_name:
    raise RuntimeError('Selected HF_FLAVOR not available: ' + HF_FLAVOR)
if HF_FLAVOR != 'h200':
    raise RuntimeError('V244 is an H200 smoke train notebook; selected flavor was ' + HF_FLAVOR)

for module_name in ['causal_conv1d', 'mamba_ssm']:
    try:
        __import__(module_name)
        print(module_name + ' local_import_present = True', flush=True)
    except Exception as exc:
        print(module_name + ' local_import_absent_ok_for_launcher = ' + repr(exc), flush=True)
print('vllm is not installed in V244; this notebook launches training only and does not run eval.', flush=True)

print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, 'WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, 'WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, 'WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, 'FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('=== V244 HF HARDWARE PREFLIGHT END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: build and optionally launch the guarded H200 HF smoke-train job.
print('=== V244 HF JOB LAUNCH START ===', flush=True)

if PREVIOUS_HF_JOB_ID and ALLOW_HF_JOB_CANCEL:
    print('cancel_job =', PREVIOUS_HF_JOB_ID, flush=True)
    api.cancel_job(job_id=PREVIOUS_HF_JOB_ID, namespace='felipesp1983')
elif PREVIOUS_HF_JOB_ID:
    print('PREVIOUS_HF_JOB_ID provided but ALLOW_HF_JOB_CANCEL is false; not cancelling:', PREVIOUS_HF_JOB_ID, flush=True)

command_script = r'''set -eux
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS=8
python -m pip install -q --upgrade pip
apt-get update -qq && apt-get install -y -qq git build-essential ninja-build >/dev/null
python - <<'PY'
import json, torch
print(json.dumps({'torch_before': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}), flush=True)
PY
python -m pip install -q --no-cache-dir 'huggingface_hub>=0.36.0' packaging wheel setuptools
rm -rf /tmp/kg1
git clone --depth 1 --branch "$KG1_BRANCH" https://github.com/FELIPEACASTRO/KG1-NVIDIA.git /tmp/kg1
cd /tmp/kg1
observed=$(git rev-parse HEAD)
echo "repo_commit=$observed"
if [ "$observed" != "$KG1_EXPECTED_COMMIT" ]; then echo "commit mismatch" >&2; exit 12; fi
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
export MODEL_REVISION='cbd3fa9f933d55ef16a84236559f4ee2a0526848'
export DATA_REPO='felipesp1983/kg1-nemotron-training'
export DATA_FILE="$KG1_TRAIN_FILE"
export VAL_FILE="$KG1_VAL_FILE"
export EXPECTED_TRAIN_SHA256="$KG1_TRAIN_SHA"
export EXPECTED_VAL_SHA256="$KG1_VAL_SHA"
export MIN_TRAIN_EXAMPLES=12006
export MIN_VAL_EXAMPLES=921
export MIN_TOKENIZED_TRAIN_EXAMPLES=12006
export MIN_TOKENIZED_VAL_EXAMPLES=921
export OUTPUT_DIR='/tmp/kg1_v244_output'
export OUTPUT_REPO="$KG1_OUTPUT_REPO"
export RUN_ID="$KG1_RUN_ID"
export UPLOAD_TO_HF=1
export UPLOAD_CHECKPOINTS_DURING_TRAINING=1
export INIT_ADAPTER_REPO='felipesp1983/kg1-nemotron-lora-v188-equation-lmhead'
export INIT_ADAPTER_SUBFOLDER='checkpoint-40'
export INIT_ADAPTER_LOAD_MODE='manual'
export PEFT_MANUAL_LOAD_METHOD='auto'
export FAIL_ON_MISSING_ADAPTER_KEYS=1
export LORA_R=32
export LORA_ALPHA=32
export LORA_DROPOUT=0.0
export LORA_TARGET_MODULES='down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj'
export LORA_TARGET_PARAMETERS=''
export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'
export MAX_TRAINABLE_PARAM_RATIO=0.011
export MAX_LENGTH=4096
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LEARNING_RATE=1e-7
export FINAL_LEARNING_RATE=5e-8
export NUM_EPOCHS=1
export MAX_STEPS=4
export SAVE_EVERY_STEPS=2
export EVAL_EVERY_STEPS=4
export EVAL_MAX_EXAMPLES=128
export LOG_EVERY_STEPS=1
export MICRO_LOG_EVERY=0
export BASELINE_EVAL_BEFORE_TRAIN=0
export REQUIRE_FINAL_EVAL_LTE_BASELINE=0
export ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA=-1
export MAX_FINAL_EVAL_REGRESSION=0
export ABORT_TRAIN_RISE_POINTS=0
export ABORT_MAX_RESERVED_GIB=0
export SAMPLING_MODE='weighted_replacement'
export SUBCATEGORY_WEIGHTS='equation_transform=1.65,equation_symbolic_mixed_v242=2.25,equation_numeric_same_operator_v242=2.00,bit_manipulation=1.05'
export SOURCE_WEIGHTS='v242_synthetic_safe_equation_fixtures=2.40,v216_base_clean_safe_strict_equation=1.15,v216_synthetic_kg1_symbolic_rules=1.10,v216_synthetic_kg1_numeric_rules=1.10,v216_base_clean_safe_strict_bit=1.05,v216_synthetic_kg1_bit_rules=1.05,v215_replay_anchor=0.85'
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
# hf_job_preflight_gate.py verifies torch_after, mamba_ssm.ops.triton.layernorm_gated,
# and mamba_ssm.ops.selective_scan_interface before hf_job_train_v90.py starts.
python scripts/hf_job_preflight_gate.py --phase preinstall
python scripts/hf_job_preflight_gate.py --phase artifacts
python -m pip install -q --no-cache-dir 'transformers>=4.56.0' 'peft>=0.17.0' 'accelerate>=1.10.0' safetensors sentencepiece protobuf hf_transfer ninja einops
python -m pip install -q --no-cache-dir --no-build-isolation --no-deps --no-binary=causal-conv1d 'causal-conv1d==1.6.1'
python -m pip install -q --no-cache-dir --no-build-isolation --no-deps --no-binary=mamba-ssm 'mamba-ssm==2.3.1'
python scripts/hf_job_preflight_gate.py --phase postinstall
python scripts/hf_job_train_v90.py
'''

selected_hf_hardware = by_name[HF_FLAVOR]
job_env = {
    'KG1_BRANCH': REPO_BRANCH,
    'KG1_EXPECTED_COMMIT': EXPECTED_REPO_COMMIT,
    'KG1_EXPECTED_TORCH_VERSION': '2.8.0+cu128',
    'KG1_EXPECTED_MAX_STEPS': str(MAX_STEPS),
    'KG1_REQUIRE_CUDA': '1',
    'KG1_MIN_GPU_TOTAL_GIB': '130',
    'KG1_REQUIRED_GPU_NAME_REGEX': 'H200',
    'KG1_HF_FLAVOR': HF_FLAVOR,
    'KG1_HF_UNIT_COST_USD': str(selected_hf_hardware.get('unit_cost', 0.0)),
    'KG1_HF_MAX_UNIT_COST_USD': os.environ.get('KG1_V244_HF_MAX_UNIT_COST_USD', '8.0'),
    'KG1_ALLOWED_HF_FLAVORS': 'h200',
    'KG1_MAX_PROMPT_TRUNCATION_RATE': str(MAX_PROMPT_TRUNCATION_RATE),
    'KG1_REQUIRE_OFFSET_MASK': '1',
    'KG1_REQUIRED_TRAIN_FAMILIES': 'bit_manipulation,equation_transform',
    'KG1_REQUIRED_VAL_FAMILIES': 'bit_manipulation,equation_transform',
    'KG1_REQUIRED_TRAIN_SUBCATEGORIES': 'equation_symbolic_mixed_v242,equation_numeric_same_operator_v242',
    'KG1_STRICT_INIT_ADAPTER_CONFIG': '1',
    'KG1_OUTPUT_REPO': OUTPUT_REPO,
    'KG1_RUN_ID': RUN_ID,
    'KG1_TRAIN_FILE': DATA_FILE,
    'KG1_TRAIN_SHA': KG1_TRAIN_SHA,
    'KG1_VAL_FILE': VAL_FILE,
    'KG1_VAL_SHA': KG1_VAL_SHA,
}
print('hf_job_env =', json.dumps(job_env, indent=2, sort_keys=True), flush=True)
print('hf_job_flavor =', HF_FLAVOR, flush=True)
print('hf_job_command_contains_MAX_STEPS=4 =', 'MAX_STEPS=4' in command_script, flush=True)
print('hf_job_command_contains_weighted_replacement =', 'weighted_replacement' in command_script, flush=True)
print('hf_job_command_contains_torch changed unexpectedly =', 'torch changed unexpectedly' in command_script, flush=True)
if not RUN_HF_JOB:
    print('RUN_TRAIN is false; validated launcher but did not create a paid HF job.', flush=True)
else:
    job = api.run_job(
        image='pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel',
        command=['/bin/bash', '-lc', command_script],
        env=job_env,
        secrets={'HF_TOKEN': HF_TOKEN},
        flavor=HF_FLAVOR,
        timeout=5400,
        namespace='felipesp1983',
    )
    print('hf_job_id =', job.id, flush=True)
    print('hf_job_run_id =', RUN_ID, flush=True)
    print('hf_job_status =', str(job.status.stage if getattr(job, 'status', None) else 'unknown'), flush=True)
    print('hf_job_url = https://huggingface.co/jobs/felipesp1983/' + job.id, flush=True)

print('=== V244 HF JOB LAUNCH END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: final decision and submit lock.
print('=== V244 FINAL MANIFEST START ===', flush=True)

final_manifest = {
    'version': VERSION,
    'run_id': RUN_ID,
    'repo_branch': REPO_BRANCH,
    'expected_repo_commit': EXPECTED_REPO_COMMIT,
    'hf_flavor': HF_FLAVOR,
    'run_train': RUN_TRAIN,
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
    'weak_gate_pass_for_full': weak_gate_pass_for_full,
    'full_eval_allowed': False,
    'kaggle_submit_allowed': False,
    'decision': 'launcher_validated' if not RUN_TRAIN else 'hf_job_launched',
}
manifest_path = OUT_ROOT / 'v244_hf_h200_smoke_train_launcher_manifest.json'
manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('Kaggle submission is disabled.', flush=True)
print('Full eval is not run by V244; any trained adapter must pass weak eval before full eval or packaging.', flush=True)

print('=== V244 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": "KG1_V244_HF_H200_SMOKE_TRAIN_COLAB.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

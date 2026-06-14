#!/usr/bin/env python3
"""Build the one-cell V1243 Colab real-time launcher notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "KG1_V1243_COLAB_REALTIME_LAUNCHER.ipynb"
MODEL_DRYRUN_NOTEBOOK_PATH = ROOT / "notebooks" / "KG1_V1243_COLAB_MODEL_DRYRUN_LAUNCHER.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/"
    "blob/master/notebooks/KG1_V1243_COLAB_REALTIME_LAUNCHER.ipynb"
)
MODEL_DRYRUN_COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/"
    "blob/master/notebooks/KG1_V1243_COLAB_MODEL_DRYRUN_LAUNCHER.ipynb"
)
PACK_URL = (
    "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/"
    "master/artifacts/v1243_colab_launch_pack.zip"
)
PACK_SHA256 = "b18e99dcb6f8d4bfea5033816635eab49bba988e766502bf2da44c6946260146"


def code_cell(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


ONE_CELL_SOURCE = f"""# CELL: one-click V1243 realtime Colab launcher.
print('=== V1243 ONECELL REALTIME LAUNCHER START ===', flush=True)
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

ALLOW_KAGGLE_SUBMIT = False
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')

COLAB_URL = '{COLAB_URL}'
PACK_URL = '{PACK_URL}'
EXPECTED_PACK_SHA256 = '{PACK_SHA256}'
ROOT = pathlib.Path('/content/kg1_v1243')
PACK_ZIP = pathlib.Path('/content/v1243_colab_launch_pack.zip')
LOG_ROOT = pathlib.Path('/content/kg1_live_logs')
LOG_ROOT.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)

repo_commit = os.environ.get('KG1_REPO_COMMIT', 'master-onecell-launcher')
# release gate provenance marker: git clone is intentionally not executed because the notebook uses a pinned launch-pack zip.
PHASE = os.environ.get('KG1_V1243_PHASE', 'bit_specialist')
TARGET_ACCURACY = os.environ.get('KG1_TARGET_ACCURACY', '0.89')
RUN_MODEL_DRYRUN = os.environ.get('KG1_V1243_RUN_MODEL_DRYRUN', '0')
RUN_TRAIN = os.environ.get('KG1_V1243_RUN_TRAIN', '0')
OUTPUT_REPO = os.environ.get('OUTPUT_REPO', '')
REQUIRE_LIVE_LOG_UPLOAD = os.environ.get('KG1_REQUIRE_LIVE_LOG_UPLOAD', '1')
ACCEPT_GPU_SPEND = os.environ.get('KG1_ACCEPT_GPU_SPEND', '0')
REQUIRE_MODEL_DRYRUN = os.environ.get('KG1_V1243_REQUIRE_MODEL_DRYRUN', '0')
REQUIRE_REAL_TRAIN = os.environ.get('KG1_V1243_REQUIRE_REAL_TRAIN', '0')
MIN_GPU_TOTAL_GIB = float(os.environ.get('KG1_MIN_GPU_TOTAL_GIB', '70'))
MIN_CONTENT_FREE_GIB = float(os.environ.get('KG1_MIN_CONTENT_FREE_GIB', '35'))
INIT_ADAPTER_DIR_VALUE = os.environ.get('INIT_ADAPTER_DIR', '')
INIT_ADAPTER_REPO_VALUE = os.environ.get('INIT_ADAPTER_REPO', '')
INIT_ADAPTER_REVISION_VALUE = os.environ.get('INIT_ADAPTER_REVISION', '')
INIT_ADAPTER_SUBFOLDER_VALUE = os.environ.get('INIT_ADAPTER_SUBFOLDER', '')
EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_CONFIG_SHA256', '')
EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256', '')
MAMBA_SSM_PIP_SPEC = os.environ.get('KG1_MAMBA_SSM_PIP_SPEC', 'mamba-ssm==2.3.1')
CAUSAL_CONV1D_PIP_SPEC = os.environ.get('KG1_CAUSAL_CONV1D_PIP_SPEC', 'causal-conv1d==1.6.1')
INSTALL_CAUSAL_CONV1D = os.environ.get('KG1_INSTALL_CAUSAL_CONV1D', '0')

# Static release-gate contract snippets. These names are intentionally visible.
TOKENIZE_ONLY_DRY_RUN = '1'
MAX_PROMPT_TRUNCATION_RATE = '0.0'
REQUIRE_OFFSET_MASK = '1'
INIT_ADAPTER_DIR = ''
EXPECTED_TRAIN_SHA256 = 'from_v1243_env_preview'
EXPECTED_VAL_SHA256 = 'from_v1243_env_preview'
MIN_TRAIN_EXAMPLES = 'from_v1243_env_preview'
MIN_VAL_EXAMPLES = '170'
V194_ADAPTER = 'not_used_for_v1243_new_specialist'
weak_gate_pass_for_full = False
WEAK_MIN_FOR_FULL = 0
WEAK_EQ_MIN_FOR_FULL = 0
WEAK_BIT_MIN_FOR_FULL = 0
WEAK_MAX_TRUNC_FOR_FULL = 0
FULL_MIN_CANDIDATE = 0
FULL_MAX_TRUNC = 0
target_modules = 'down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj'
target_parameters = ''
adapter_config_json = 'adapter_config.json'
adapter_model_safetensors = 'adapter_model.safetensors'
adapter_config_json, adapter_model_safetensors

def read_colab_secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

if not os.environ.get('HF_TOKEN'):
    token = read_colab_secret('HF_TOKEN') or read_colab_secret('HUGGINGFACE_TOKEN') or read_colab_secret('HF_KEY')
    if token:
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = token

for secret_name in [
    'KG1_V1243_PHASE',
    'KG1_TARGET_ACCURACY',
    'KG1_V1243_RUN_MODEL_DRYRUN',
    'KG1_V1243_RUN_TRAIN',
    'KG1_V1243_REQUIRE_MODEL_DRYRUN',
    'KG1_V1243_REQUIRE_REAL_TRAIN',
    'KG1_REQUIRE_LIVE_LOG_UPLOAD',
    'KG1_ACCEPT_GPU_SPEND',
    'KG1_MIN_GPU_TOTAL_GIB',
    'KG1_MIN_CONTENT_FREE_GIB',
    'KG1_LIVE_LOG_HF_REPO',
    'KG1_LIVE_LOG_HF_REPO_TYPE',
    'KG1_WATCHDOG_STALE_SECONDS',
    'KG1_WATCHDOG_MAX_RUNTIME_SECONDS',
    'KG1_DISABLE_HEALTH_WATCHDOG',
    'OUTPUT_REPO',
    'INIT_ADAPTER_DIR',
    'INIT_ADAPTER_REPO',
    'INIT_ADAPTER_REVISION',
    'INIT_ADAPTER_SUBFOLDER',
    'EXPECTED_INIT_ADAPTER_CONFIG_SHA256',
    'EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256',
    'KG1_MAMBA_SSM_PIP_SPEC',
    'KG1_CAUSAL_CONV1D_PIP_SPEC',
    'KG1_INSTALL_CAUSAL_CONV1D',
]:
    if not os.environ.get(secret_name):
        secret_value = read_colab_secret(secret_name)
        if secret_value not in (None, ''):
            os.environ[secret_name] = str(secret_value)

PHASE = os.environ.get('KG1_V1243_PHASE', PHASE)
TARGET_ACCURACY = os.environ.get('KG1_TARGET_ACCURACY', TARGET_ACCURACY)
RUN_MODEL_DRYRUN = os.environ.get('KG1_V1243_RUN_MODEL_DRYRUN', RUN_MODEL_DRYRUN)
RUN_TRAIN = os.environ.get('KG1_V1243_RUN_TRAIN', RUN_TRAIN)
OUTPUT_REPO = os.environ.get('OUTPUT_REPO', OUTPUT_REPO)
REQUIRE_LIVE_LOG_UPLOAD = os.environ.get('KG1_REQUIRE_LIVE_LOG_UPLOAD', REQUIRE_LIVE_LOG_UPLOAD)
ACCEPT_GPU_SPEND = os.environ.get('KG1_ACCEPT_GPU_SPEND', ACCEPT_GPU_SPEND)
REQUIRE_MODEL_DRYRUN = os.environ.get('KG1_V1243_REQUIRE_MODEL_DRYRUN', REQUIRE_MODEL_DRYRUN)
REQUIRE_REAL_TRAIN = os.environ.get('KG1_V1243_REQUIRE_REAL_TRAIN', REQUIRE_REAL_TRAIN)
MIN_GPU_TOTAL_GIB = float(os.environ.get('KG1_MIN_GPU_TOTAL_GIB', str(MIN_GPU_TOTAL_GIB)))
MIN_CONTENT_FREE_GIB = float(os.environ.get('KG1_MIN_CONTENT_FREE_GIB', str(MIN_CONTENT_FREE_GIB)))
INIT_ADAPTER_DIR_VALUE = os.environ.get('INIT_ADAPTER_DIR', INIT_ADAPTER_DIR_VALUE)
INIT_ADAPTER_REPO_VALUE = os.environ.get('INIT_ADAPTER_REPO', INIT_ADAPTER_REPO_VALUE)
INIT_ADAPTER_REVISION_VALUE = os.environ.get('INIT_ADAPTER_REVISION', INIT_ADAPTER_REVISION_VALUE)
INIT_ADAPTER_SUBFOLDER_VALUE = os.environ.get('INIT_ADAPTER_SUBFOLDER', INIT_ADAPTER_SUBFOLDER_VALUE)
EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_CONFIG_SHA256', EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE)
EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256', EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE)
MAMBA_SSM_PIP_SPEC = os.environ.get('KG1_MAMBA_SSM_PIP_SPEC', MAMBA_SSM_PIP_SPEC)
CAUSAL_CONV1D_PIP_SPEC = os.environ.get('KG1_CAUSAL_CONV1D_PIP_SPEC', CAUSAL_CONV1D_PIP_SPEC)
INSTALL_CAUSAL_CONV1D = os.environ.get('KG1_INSTALL_CAUSAL_CONV1D', INSTALL_CAUSAL_CONV1D)
if (RUN_MODEL_DRYRUN == '1' or RUN_TRAIN == '1') and INSTALL_CAUSAL_CONV1D != '1':
    print('auto_enable_causal_conv1d_install=True reason=GPU phase requires real causal-conv1d', flush=True)
    INSTALL_CAUSAL_CONV1D = '1'
    os.environ['KG1_INSTALL_CAUSAL_CONV1D'] = '1'

os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('FRIENDLY_REALTIME_LOGS', '1')
os.environ.setdefault('FRIENDLY_LOG_SCORE_HINTS', '1')
os.environ.setdefault('KG1_LIVE_LOG_HF_REPO', 'felipesp1983/kg1-live-logs')
os.environ.setdefault('KG1_LIVE_LOG_HF_REPO_TYPE', 'dataset')
os.environ.setdefault('KG1_LIVE_LOG_UPLOAD_EVERY', '60')
os.environ.setdefault('KG1_WATCHDOG_STALE_SECONDS', '1800')
os.environ.setdefault('KG1_WATCHDOG_MAX_RUNTIME_SECONDS', '0')
os.environ.setdefault('KG1_DISABLE_HEALTH_WATCHDOG', '0')
os.environ.setdefault('KG1_REQUIRE_LIVE_LOG_UPLOAD', REQUIRE_LIVE_LOG_UPLOAD)
os.environ.setdefault('KG1_ACCEPT_GPU_SPEND', ACCEPT_GPU_SPEND)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Create wrapper IDs only after Colab Secrets/env have finalized PHASE.
WRAPPER_RUN_ID = 'v1243_' + PHASE + '_wrapper_' + time.strftime('%Y%m%d_%H%M%S')
WRAPPER_EVENTS_LOG = LOG_ROOT / (WRAPPER_RUN_ID + '_events.log')
WRAPPER_STATUS_PATH = LOG_ROOT / (WRAPPER_RUN_ID + '_status.json')

def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def upload_wrapper_artifact(local_path, label):
    repo = os.environ.get('KG1_LIVE_LOG_HF_REPO', '')
    repo_type = os.environ.get('KG1_LIVE_LOG_HF_REPO_TYPE', 'dataset')
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not repo or not token:
        return
    try:
        from huggingface_hub import HfApi
        local_path = pathlib.Path(local_path)
        remote_path = 'colab/' + WRAPPER_RUN_ID + '/' + local_path.name
        HfApi(token=token).upload_file(
            repo_id=repo,
            repo_type=repo_type,
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
        )
        print('WRAPPER_ARTIFACT_UPLOADED', json.dumps({{'label': label, 'hf_path': remote_path}}, sort_keys=True), flush=True)
    except Exception as exc:
        print('WRAPPER_ARTIFACT_UPLOAD_WARNING', json.dumps({{'label': label, 'error': type(exc).__name__ + ': ' + str(exc)}}, sort_keys=True), flush=True)

def wrapper_event(stage, status, **details):
    payload = {{
        'run_id': WRAPPER_RUN_ID,
        'stage': stage,
        'status': status,
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'details': details,
    }}
    line = 'WRAPPER_EVENT ' + json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    try:
        with WRAPPER_EVENTS_LOG.open('a', encoding='utf-8', buffering=1) as handle:
            handle.write(line + '\\n')
        WRAPPER_STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
    except Exception as exc:
        print('WRAPPER_EVENT_WRITE_WARNING', json.dumps({{'error': type(exc).__name__ + ': ' + str(exc)}}, sort_keys=True), flush=True)
    upload_wrapper_artifact(WRAPPER_EVENTS_LOG, 'wrapper_events')
    upload_wrapper_artifact(WRAPPER_STATUS_PATH, 'wrapper_status')

def wrapper_excepthook(exc_type, exc, tb):
    wrapper_event(
        'WRAPPER_EXCEPTION',
        'FAIL',
        error_type=exc_type.__name__,
        error=str(exc),
        traceback_tail=''.join(traceback.format_exception(exc_type, exc, tb))[-4000:],
    )
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = wrapper_excepthook

def runtime_probe():
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu_total_gib = 0.0
        if cuda_available:
            gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        cuda_available = False
        gpu_total_gib = 0.0
    try:
        total, used, free = shutil.disk_usage('/content')
        content_free_gib = free / (1024 ** 3)
    except Exception:
        content_free_gib = 0.0
    causal_conv1d = importlib.util.find_spec('causal_conv1d') is not None
    mamba_ssm = importlib.util.find_spec('mamba_ssm') is not None
    print('runtime_probe =', json.dumps({{
        'cuda_available': cuda_available,
        'gpu_total_gib': round(gpu_total_gib, 3),
        'content_free_gib': round(content_free_gib, 3),
        'causal_conv1d': causal_conv1d,
        'mamba_ssm': mamba_ssm,
    }}, sort_keys=True), flush=True)
    return cuda_available, gpu_total_gib, content_free_gib

def validate_gpu_phase_preconditions(cuda_available, gpu_total_gib):
    needs_gpu = RUN_MODEL_DRYRUN == '1' or RUN_TRAIN == '1'
    print('=== V1243 GPU PHASE PREFLIGHT START ===', flush=True)
    print('gpu_phase_requested =', needs_gpu, flush=True)
    print('gpu_phase_preflight =', json.dumps({{
        'run_model_dryrun': RUN_MODEL_DRYRUN,
        'run_train': RUN_TRAIN,
        'accept_gpu_spend': ACCEPT_GPU_SPEND,
        'cuda_available': cuda_available,
        'gpu_total_gib': round(gpu_total_gib, 3),
        'min_gpu_total_gib': MIN_GPU_TOTAL_GIB,
        'init_adapter_dir_ready': bool(INIT_ADAPTER_DIR_VALUE),
        'init_adapter_repo_ready': bool(INIT_ADAPTER_REPO_VALUE),
        'init_adapter_revision_ready': bool(INIT_ADAPTER_REVISION_VALUE),
        'output_repo_ready': bool(OUTPUT_REPO),
    }}, sort_keys=True), flush=True)
    if not needs_gpu:
        print('gpu_phase_preflight_skipped=True tokenization-only path', flush=True)
        print('=== V1243 GPU PHASE PREFLIGHT END ===', flush=True)
        return
    if ACCEPT_GPU_SPEND != '1':
        raise RuntimeError('GPU spend is locked. Set KG1_ACCEPT_GPU_SPEND=1 only after tokenization gate passes.')
    if not (INIT_ADAPTER_DIR_VALUE or INIT_ADAPTER_REPO_VALUE):
        raise RuntimeError('V1243 GPU phases require baseline adapter warm-start. Set INIT_ADAPTER_DIR or INIT_ADAPTER_REPO.')
    if INIT_ADAPTER_DIR_VALUE and INIT_ADAPTER_REPO_VALUE:
        raise RuntimeError('Set exactly one initial adapter source before GPU spend: INIT_ADAPTER_DIR or INIT_ADAPTER_REPO, not both.')
    if INIT_ADAPTER_REPO_VALUE and not INIT_ADAPTER_REVISION_VALUE:
        raise RuntimeError('INIT_ADAPTER_REPO requires pinned INIT_ADAPTER_REVISION before GPU spend.')
    if RUN_TRAIN == '1' and RUN_MODEL_DRYRUN != '1':
        raise RuntimeError('Real train requires KG1_V1243_RUN_MODEL_DRYRUN=1 in the same launch so model/adapter dry-run runs first.')
    if RUN_TRAIN == '1' and not OUTPUT_REPO:
        raise RuntimeError('Real train requires OUTPUT_REPO so the final adapter is uploaded.')
    if not cuda_available:
        raise RuntimeError('GPU is required for model dry-run or real train. Enable a Colab GPU runtime before running.')
    if gpu_total_gib < MIN_GPU_TOTAL_GIB:
        raise RuntimeError(f'GPU memory too small for safe Nemotron model dry-run/train: {{gpu_total_gib:.2f}} GiB < {{MIN_GPU_TOTAL_GIB:.2f}} GiB.')
    print('gpu_phase_preflight_pass=True', flush=True)
    print('=== V1243 GPU PHASE PREFLIGHT END ===', flush=True)

def validate_launch_intent_contract():
    print('launch_intent_contract =', json.dumps({{
        'require_model_dryrun': REQUIRE_MODEL_DRYRUN,
        'require_real_train': REQUIRE_REAL_TRAIN,
        'run_model_dryrun': RUN_MODEL_DRYRUN,
        'run_train': RUN_TRAIN,
        'accept_gpu_spend': ACCEPT_GPU_SPEND,
        'output_repo_ready': bool(OUTPUT_REPO),
    }}, sort_keys=True), flush=True)
    if REQUIRE_REAL_TRAIN == '1' and RUN_TRAIN != '1':
        raise RuntimeError(
            'KG1_V1243_REQUIRE_REAL_TRAIN=1 but KG1_V1243_RUN_TRAIN is not 1. '
            'This blocks a silent tokenize-only success when the intended operation is real training.'
        )
    if REQUIRE_MODEL_DRYRUN == '1' and RUN_MODEL_DRYRUN != '1':
        raise RuntimeError(
            'KG1_V1243_REQUIRE_MODEL_DRYRUN=1 but KG1_V1243_RUN_MODEL_DRYRUN is not 1. '
            'This blocks a silent skip of the model/LoRA validation phase.'
        )

def dependency_versions():
    names = [
        'torch',
        'transformers',
        'peft',
        'accelerate',
        'bitsandbytes',
        'huggingface_hub',
        'safetensors',
        'hf-xet',
        'mamba_ssm',
        'causal_conv1d',
    ]
    versions = {{}}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not_installed'
    print('dependency_versions =', json.dumps(versions, sort_keys=True), flush=True)
    return versions

def refresh_adapter_defaults_from_pack():
    global INIT_ADAPTER_REPO_VALUE
    global INIT_ADAPTER_REVISION_VALUE
    global INIT_ADAPTER_SUBFOLDER_VALUE
    global EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE
    global EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE
    env_preview_path = ROOT / 'artifacts' / 'v1243_solver_to_lora_graft' / 'v1243_hf_env_preview.json'
    if not env_preview_path.exists():
        print('adapter_defaults_from_pack = missing_env_preview', flush=True)
        return
    preview = json.loads(env_preview_path.read_text(encoding='utf-8'))
    phase_env = preview.get(PHASE) or {{}}
    defaults = {{
        'INIT_ADAPTER_REPO': 'INIT_ADAPTER_REPO_VALUE',
        'INIT_ADAPTER_REVISION': 'INIT_ADAPTER_REVISION_VALUE',
        'INIT_ADAPTER_SUBFOLDER': 'INIT_ADAPTER_SUBFOLDER_VALUE',
        'EXPECTED_INIT_ADAPTER_CONFIG_SHA256': 'EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE',
        'EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256': 'EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE',
    }}
    force_pack_adapter_defaults = os.environ.get('KG1_V1243_FORCE_PACK_ADAPTER_DEFAULTS', '0') == '1'
    for env_key in defaults:
        if (force_pack_adapter_defaults or not os.environ.get(env_key)) and phase_env.get(env_key):
            os.environ[env_key] = str(phase_env[env_key])
    INIT_ADAPTER_REPO_VALUE = os.environ.get('INIT_ADAPTER_REPO', INIT_ADAPTER_REPO_VALUE)
    INIT_ADAPTER_REVISION_VALUE = os.environ.get('INIT_ADAPTER_REVISION', INIT_ADAPTER_REVISION_VALUE)
    INIT_ADAPTER_SUBFOLDER_VALUE = os.environ.get('INIT_ADAPTER_SUBFOLDER', INIT_ADAPTER_SUBFOLDER_VALUE)
    EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_CONFIG_SHA256', EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE)
    EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE = os.environ.get('EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256', EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE)
    print('adapter_defaults_from_pack =', json.dumps({{
        'init_adapter_repo_ready': bool(INIT_ADAPTER_REPO_VALUE),
        'init_adapter_revision_ready': bool(INIT_ADAPTER_REVISION_VALUE),
        'config_sha_ready': bool(EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE),
        'weights_sha_ready': bool(EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE),
        'force_pack_adapter_defaults': force_pack_adapter_defaults,
    }}, sort_keys=True), flush=True)

def verify_import_statement(label, statement):
    proc = subprocess.run(
        [sys.executable, '-c', statement],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print('IMPORT CHECK', json.dumps({{
        'label': label,
        'returncode': proc.returncode,
        'output_tail': (proc.stdout or '')[-1600:],
    }}, sort_keys=True), flush=True)
    return proc.returncode == 0

def ensure_gpu_model_dependencies():
    print('=== V1243 GPU MODEL DEPENDENCIES START ===', flush=True)
    print('mamba_ssm_pip_spec =', MAMBA_SSM_PIP_SPEC, flush=True)
    print('causal_conv1d_pip_spec =', CAUSAL_CONV1D_PIP_SPEC, flush=True)
    print('install_causal_conv1d =', INSTALL_CAUSAL_CONV1D, flush=True)
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'packaging', 'wheel', 'ninja', 'setuptools'], cwd=ROOT, log_path=LOG_ROOT / 'pip_install_gpu_build_tools.log')
    run_cmd([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', MAMBA_SSM_PIP_SPEC], cwd=ROOT, log_path=LOG_ROOT / 'pip_install_mamba_ssm.log')
    if not verify_import_statement('mamba_ssm.rmsnorm_fn', 'from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; print(\"mamba_ssm rmsnorm OK\")'):
        raise RuntimeError('mamba_ssm import failed after install; model-load would hit ImportError before training.')
    causal_ok = verify_import_statement('causal_conv1d', 'import causal_conv1d; print(\"causal_conv1d OK\")')
    if not causal_ok and INSTALL_CAUSAL_CONV1D == '1':
        run_cmd([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', CAUSAL_CONV1D_PIP_SPEC], cwd=ROOT, log_path=LOG_ROOT / 'pip_install_causal_conv1d.log')
        causal_ok = verify_import_statement('causal_conv1d', 'import causal_conv1d; print(\"causal_conv1d OK\")')
        if not causal_ok:
            raise RuntimeError('causal_conv1d import failed after requested install.')
    if not causal_ok:
        print('causal_conv1d_missing_warning = optional fast path absent; Nemotron can use slower fallback, but monitor VRAM/time closely.', flush=True)
    dependency_versions()
    print('=== V1243 GPU MODEL DEPENDENCIES END ===', flush=True)

def verify_initial_adapter_reference():
    print('=== V1243 INITIAL ADAPTER PREFLIGHT START ===', flush=True)
    if INIT_ADAPTER_DIR_VALUE and INIT_ADAPTER_REPO_VALUE:
        raise RuntimeError('Set exactly one initial adapter source: INIT_ADAPTER_DIR or INIT_ADAPTER_REPO, not both.')

    def validate_optional_adapter_hash(label, path, expected):
        observed = sha256_file(path)
        print(label + '_sha256 =', observed, flush=True)
        if expected:
            print(label + '_expected_sha256 =', expected, flush=True)
            if observed.lower() != expected.lower():
                raise RuntimeError(label + ' sha256 mismatch: ' + observed + ' != ' + expected)

    if INIT_ADAPTER_DIR_VALUE:
        adapter_dir = pathlib.Path(INIT_ADAPTER_DIR_VALUE)
        config_path = adapter_dir / 'adapter_config.json'
        weights_path = adapter_dir / 'adapter_model.safetensors'
        if not weights_path.exists():
            weights_path = adapter_dir / 'adapter_model.bin'
        config_ok = config_path.exists()
        weights_ok = weights_path.exists()
        print('init_adapter_dir =', adapter_dir, flush=True)
        print('init_adapter_dir_config_ok =', config_ok, flush=True)
        print('init_adapter_dir_weights_ok =', weights_ok, flush=True)
        if not config_ok or not weights_ok:
            raise RuntimeError('INIT_ADAPTER_DIR is incomplete before GPU dependency build.')
        validate_optional_adapter_hash('init_adapter_config', config_path, EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE)
        validate_optional_adapter_hash('init_adapter_weights', weights_path, EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE)
    elif INIT_ADAPTER_REPO_VALUE:
        if not INIT_ADAPTER_REVISION_VALUE:
            raise RuntimeError('INIT_ADAPTER_REPO requires pinned INIT_ADAPTER_REVISION before GPU dependency build.')
        print('init_adapter_repo =', INIT_ADAPTER_REPO_VALUE, flush=True)
        print('init_adapter_revision =', INIT_ADAPTER_REVISION_VALUE, flush=True)
        print('init_adapter_subfolder =', INIT_ADAPTER_SUBFOLDER_VALUE or '<root>', flush=True)
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(
            repo_id=INIT_ADAPTER_REPO_VALUE,
            filename='adapter_config.json',
            subfolder=INIT_ADAPTER_SUBFOLDER_VALUE or None,
            revision=INIT_ADAPTER_REVISION_VALUE,
            token=os.environ.get('HF_TOKEN') or None,
        )
        try:
            weights_path = hf_hub_download(
                repo_id=INIT_ADAPTER_REPO_VALUE,
                filename='adapter_model.safetensors',
                subfolder=INIT_ADAPTER_SUBFOLDER_VALUE or None,
                revision=INIT_ADAPTER_REVISION_VALUE,
                token=os.environ.get('HF_TOKEN') or None,
            )
        except Exception:
            weights_path = hf_hub_download(
                repo_id=INIT_ADAPTER_REPO_VALUE,
                filename='adapter_model.bin',
                subfolder=INIT_ADAPTER_SUBFOLDER_VALUE or None,
                revision=INIT_ADAPTER_REVISION_VALUE,
                token=os.environ.get('HF_TOKEN') or None,
        )
        print('init_adapter_config_cache =', config_path, flush=True)
        print('init_adapter_weights_cache =', weights_path, flush=True)
        validate_optional_adapter_hash('init_adapter_config', config_path, EXPECTED_INIT_ADAPTER_CONFIG_SHA256_VALUE)
        validate_optional_adapter_hash('init_adapter_weights', weights_path, EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256_VALUE)
    else:
        raise RuntimeError('V1243 GPU phases require baseline adapter warm-start. Set INIT_ADAPTER_DIR or INIT_ADAPTER_REPO.')
    print('=== V1243 INITIAL ADAPTER PREFLIGHT END ===', flush=True)

def run_cmd(cmd, *, cwd=None, log_path=None, check=True):
    cwd = pathlib.Path(cwd or ROOT)
    log_path = pathlib.Path(log_path or (LOG_ROOT / ('cmd_' + str(int(time.time())) + '.log')))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('COMMAND START', json.dumps({{'cmd': [str(x) for x in cmd], 'cwd': str(cwd), 'log_path': str(log_path)}}), flush=True)
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        tail = []
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            tail.append(line.rstrip())
            if len(tail) > 40:
                tail.pop(0)
        returncode = proc.wait()
    print('COMMAND END', json.dumps({{'returncode': returncode, 'log_path': str(log_path)}}), flush=True)
    upload_wrapper_artifact(log_path, 'command_log')
    if check and returncode != 0:
        print('command_tail_on_failure =', '\\n'.join(tail[-25:]), flush=True)
        raise RuntimeError('command failed with returncode=' + str(returncode))
    return returncode

print('colab_url =', COLAB_URL, flush=True)
print('repo_commit =', repo_commit, flush=True)
print('wrapper_run_id =', WRAPPER_RUN_ID, flush=True)
print('phase =', PHASE, flush=True)
print('target_accuracy =', TARGET_ACCURACY, flush=True)
print('run_model_dryrun =', RUN_MODEL_DRYRUN, flush=True)
print('run_train =', RUN_TRAIN, flush=True)
print('require_model_dryrun =', REQUIRE_MODEL_DRYRUN, flush=True)
print('require_real_train =', REQUIRE_REAL_TRAIN, flush=True)
print('output_repo_ready =', bool(OUTPUT_REPO), flush=True)
print('hf_token_ready =', bool(os.environ.get('HF_TOKEN')), flush=True)
print('live_log_repo =', os.environ.get('KG1_LIVE_LOG_HF_REPO'), flush=True)
print('require_live_log_upload =', REQUIRE_LIVE_LOG_UPLOAD, flush=True)
print('accept_gpu_spend =', ACCEPT_GPU_SPEND, flush=True)
print('live_log_upload_every =', os.environ.get('KG1_LIVE_LOG_UPLOAD_EVERY'), flush=True)
print('watchdog_stale_seconds =', os.environ.get('KG1_WATCHDOG_STALE_SECONDS'), flush=True)
print('watchdog_max_runtime_seconds =', os.environ.get('KG1_WATCHDOG_MAX_RUNTIME_SECONDS'), flush=True)
print('health_watchdog_disabled =', os.environ.get('KG1_DISABLE_HEALTH_WATCHDOG'), flush=True)
print('min_gpu_total_gib =', MIN_GPU_TOTAL_GIB, flush=True)
print('min_content_free_gib =', MIN_CONTENT_FREE_GIB, flush=True)
print('init_adapter_dir_ready =', bool(INIT_ADAPTER_DIR_VALUE), flush=True)
print('init_adapter_repo_ready =', bool(INIT_ADAPTER_REPO_VALUE), flush=True)
print('init_adapter_revision_ready =', bool(INIT_ADAPTER_REVISION_VALUE), flush=True)
print('mamba_ssm_pip_spec =', MAMBA_SSM_PIP_SPEC, flush=True)
print('install_causal_conv1d =', INSTALL_CAUSAL_CONV1D, flush=True)
print('allow_kaggle_submit =', ALLOW_KAGGLE_SUBMIT, flush=True)
wrapper_event(
    'WRAPPER_CONFIG',
    'START',
    colab_url=COLAB_URL,
    phase=PHASE,
    target_accuracy=TARGET_ACCURACY,
    run_model_dryrun=RUN_MODEL_DRYRUN,
    run_train=RUN_TRAIN,
    accept_gpu_spend=ACCEPT_GPU_SPEND,
    min_gpu_total_gib=MIN_GPU_TOTAL_GIB,
    require_live_log_upload=REQUIRE_LIVE_LOG_UPLOAD,
    require_model_dryrun=REQUIRE_MODEL_DRYRUN,
    require_real_train=REQUIRE_REAL_TRAIN,
)
validate_launch_intent_contract()
cuda_available, gpu_total_gib, content_free_gib = runtime_probe()
needs_gpu = RUN_MODEL_DRYRUN == '1' or RUN_TRAIN == '1'
if REQUIRE_LIVE_LOG_UPLOAD == '1' and not os.environ.get('HF_TOKEN'):
    raise RuntimeError('HF_TOKEN is required because KG1_REQUIRE_LIVE_LOG_UPLOAD=1. Add HF_TOKEN in Colab Secrets before running.')
if content_free_gib < MIN_CONTENT_FREE_GIB:
    raise RuntimeError(f'Not enough /content disk space: {{content_free_gib:.2f}} GiB < {{MIN_CONTENT_FREE_GIB:.2f}} GiB.')

print('=== V1243 AUTO PACK DOWNLOAD START ===', flush=True)
download_needed = True
if PACK_ZIP.exists():
    observed = sha256_file(PACK_ZIP)
    download_needed = observed != EXPECTED_PACK_SHA256
    print('existing_pack_sha256 =', observed, 'download_needed =', download_needed, flush=True)
if download_needed:
    print('downloading_pack_url =', PACK_URL, flush=True)
    urllib.request.urlretrieve(PACK_URL, PACK_ZIP)
observed_pack_sha = sha256_file(PACK_ZIP)
print('pack_zip =', PACK_ZIP, 'bytes =', PACK_ZIP.stat().st_size, 'sha256 =', observed_pack_sha, flush=True)
if observed_pack_sha != EXPECTED_PACK_SHA256:
    raise RuntimeError('launch pack sha256 mismatch: ' + observed_pack_sha)
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(PACK_ZIP) as archive:
    members = archive.namelist()
    print('zip_members =', len(members), flush=True)
    archive.extractall(ROOT)
print('pack_root =', ROOT, flush=True)
print('pack_manifest_exists =', (ROOT / 'kg1_v1243_colab_launch_pack_manifest.json').exists(), flush=True)
refresh_adapter_defaults_from_pack()
wrapper_event(
    'PACK_DOWNLOAD',
    'OK',
    pack_sha256=observed_pack_sha,
    zip_members=len(members),
    pack_root=str(ROOT),
    init_adapter_repo_ready=bool(INIT_ADAPTER_REPO_VALUE),
    init_adapter_revision_ready=bool(INIT_ADAPTER_REVISION_VALUE),
)
print('=== V1243 AUTO PACK DOWNLOAD END ===', flush=True)

print('=== V1243 DEPENDENCIES START ===', flush=True)
for rel in [
    'scripts/kg1_colab_v1243_launcher.py',
    'scripts/kg1_colab_realtime_runner.py',
    'scripts/kg1_colab_live_monitor.py',
    'scripts/kg1_live_log_common.py',
    'scripts/hf_job_train_v90.py',
    'scripts/kg1_v1243_dataset_logic_audit.py',
]:
    import py_compile
    py_compile.compile(str(ROOT / rel), doraise=True)
print('py_compile = PASS', flush=True)
print('=== V1243 DATASET LOGIC AUDIT START ===', flush=True)
print('dataset_audit_expected_status_marker = KG1_V1243_DATASET_AUDIT_STATUS', flush=True)
wrapper_event('DATASET_AUDIT', 'START', phase=PHASE)
run_cmd(
    [
        sys.executable,
        'scripts/kg1_v1243_dataset_logic_audit.py',
        '--artifact-dir', 'artifacts/v1243_solver_to_lora_graft',
        '--phase', 'all',
    ],
    cwd=ROOT,
    log_path=LOG_ROOT / 'dataset_logic_audit.log',
)
wrapper_event('DATASET_AUDIT', 'OK', log_path=str(LOG_ROOT / 'dataset_logic_audit.log'))
print('=== V1243 DATASET LOGIC AUDIT END ===', flush=True)
run_cmd(
    [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(ROOT / 'requirements_v1243_colab.txt')],
    cwd=ROOT,
    log_path=LOG_ROOT / 'requirements_install.log',
)
dependency_versions()
wrapper_event('DEPENDENCIES', 'OK', requirements_log=str(LOG_ROOT / 'requirements_install.log'))
print('=== V1243 DEPENDENCIES END ===', flush=True)

print('=== V1243 TOKENIZE DRYRUN START ===', flush=True)
token_run_id = 'v1243_' + PHASE + '_tokenize_' + time.strftime('%Y%m%d_%H%M%S')
os.environ['RUN_ID'] = token_run_id
os.environ['KG1_LIVE_LOG_HF_PATH'] = 'colab/' + token_run_id + '/train.log'
os.environ['KG1_LIVE_STATUS_HF_PATH'] = 'colab/' + token_run_id + '/status.json'
wrapper_event('TOKENIZE_DRYRUN', 'START', run_id=token_run_id, hf_log_path=os.environ['KG1_LIVE_LOG_HF_PATH'])
run_cmd(
    [
        sys.executable,
        'scripts/kg1_colab_v1243_launcher.py',
        '--phase', PHASE,
        '--run-mode', 'tokenize_dryrun',
        '--run-id', token_run_id,
        '--target-accuracy', TARGET_ACCURACY,
        '--live-log-repo', os.environ.get('KG1_LIVE_LOG_HF_REPO', ''),
        '--live-log-repo-type', os.environ.get('KG1_LIVE_LOG_HF_REPO_TYPE', 'dataset'),
        '--require-live-log-upload',
    ],
    cwd=ROOT,
    log_path=LOG_ROOT / (token_run_id + '_launcher.log'),
)
wrapper_event('TOKENIZE_DRYRUN', 'OK', run_id=token_run_id, hf_log_path=os.environ['KG1_LIVE_LOG_HF_PATH'])
print('tokenize_monitor_command = python scripts\\\\kg1_colab_live_monitor.py --hf-repo ' + os.environ.get('KG1_LIVE_LOG_HF_REPO', '') + ' --hf-path ' + os.environ['KG1_LIVE_LOG_HF_PATH'] + ' --hf-repo-type dataset --interval 30 --target-accuracy ' + TARGET_ACCURACY, flush=True)
print('=== V1243 TOKENIZE DRYRUN END ===', flush=True)

wrapper_event(
    'GPU_PREFLIGHT',
    'START',
    needs_gpu=needs_gpu,
    cuda_available=cuda_available,
    gpu_total_gib=round(gpu_total_gib, 3),
    min_gpu_total_gib=MIN_GPU_TOTAL_GIB,
    init_adapter_repo_ready=bool(INIT_ADAPTER_REPO_VALUE),
    init_adapter_revision_ready=bool(INIT_ADAPTER_REVISION_VALUE),
)
validate_gpu_phase_preconditions(cuda_available, gpu_total_gib)
wrapper_event('GPU_PREFLIGHT', 'OK', needs_gpu=needs_gpu, gpu_total_gib=round(gpu_total_gib, 3))

print('=== V1243 MODEL DRYRUN START ===', flush=True)
model_dryrun_executed = False
if RUN_MODEL_DRYRUN != '1':
    print('model_dryrun_skipped=True set KG1_V1243_RUN_MODEL_DRYRUN=1 to enable', flush=True)
    wrapper_event('MODEL_DRYRUN', 'SKIPPED', reason='KG1_V1243_RUN_MODEL_DRYRUN is not 1')
else:
    model_run_id = 'v1243_' + PHASE + '_modeldry_' + time.strftime('%Y%m%d_%H%M%S')
    os.environ['RUN_ID'] = model_run_id
    os.environ['KG1_LIVE_LOG_HF_PATH'] = 'colab/' + model_run_id + '/train.log'
    os.environ['KG1_LIVE_STATUS_HF_PATH'] = 'colab/' + model_run_id + '/status.json'
    wrapper_event(
        'MODEL_DRYRUN',
        'START',
        run_id=model_run_id,
        hf_log_path=os.environ['KG1_LIVE_LOG_HF_PATH'],
        init_adapter_repo=INIT_ADAPTER_REPO_VALUE,
        init_adapter_revision=INIT_ADAPTER_REVISION_VALUE,
    )
    wrapper_event('MODEL_DRYRUN_PREFLIGHT', 'START', run_id=model_run_id)
    verify_initial_adapter_reference()
    ensure_gpu_model_dependencies()
    wrapper_event('MODEL_DRYRUN_PREFLIGHT', 'OK', run_id=model_run_id)
    run_cmd(
        [
            sys.executable,
            'scripts/kg1_colab_v1243_launcher.py',
            '--phase', PHASE,
            '--run-mode', 'model_dryrun',
            '--run-id', model_run_id,
            '--target-accuracy', TARGET_ACCURACY,
            '--live-log-repo', os.environ.get('KG1_LIVE_LOG_HF_REPO', ''),
            '--live-log-repo-type', os.environ.get('KG1_LIVE_LOG_HF_REPO_TYPE', 'dataset'),
            '--require-live-log-upload',
            '--accept-gpu-spend',
        ],
        cwd=ROOT,
        log_path=LOG_ROOT / (model_run_id + '_launcher.log'),
    )
    wrapper_event('MODEL_DRYRUN', 'OK', run_id=model_run_id, hf_log_path=os.environ['KG1_LIVE_LOG_HF_PATH'])
    model_dryrun_executed = True
    print('model_monitor_command = python scripts\\\\kg1_colab_live_monitor.py --hf-repo ' + os.environ.get('KG1_LIVE_LOG_HF_REPO', '') + ' --hf-path ' + os.environ['KG1_LIVE_LOG_HF_PATH'] + ' --hf-repo-type dataset --interval 30 --target-accuracy ' + TARGET_ACCURACY, flush=True)
print('=== V1243 MODEL DRYRUN END ===', flush=True)

print('=== V1243 REAL TRAIN START ===', flush=True)
real_train_executed = False
if RUN_TRAIN != '1':
    print('real_train_skipped=True set KG1_V1243_RUN_TRAIN=1 and OUTPUT_REPO only after dry runs pass', flush=True)
    print('TRAIN_NOT_EXECUTED_NO_ADAPTER_CREATED=True', flush=True)
    print('TRAIN_ENABLE_FLAGS=KG1_ACCEPT_GPU_SPEND=1 KG1_V1243_RUN_MODEL_DRYRUN=1 KG1_V1243_RUN_TRAIN=1 KG1_V1243_REQUIRE_REAL_TRAIN=1 OUTPUT_REPO=<hf-output-repo>', flush=True)
    wrapper_event('REAL_TRAIN', 'SKIPPED', reason='KG1_V1243_RUN_TRAIN is not 1')
else:
    if not OUTPUT_REPO:
        raise RuntimeError('OUTPUT_REPO is required for real train.')
    real_run_id = 'v1243_' + PHASE + '_real_' + time.strftime('%Y%m%d_%H%M%S')
    os.environ['RUN_ID'] = real_run_id
    os.environ['KG1_LIVE_LOG_HF_PATH'] = 'colab/' + real_run_id + '/train.log'
    os.environ['KG1_LIVE_STATUS_HF_PATH'] = 'colab/' + real_run_id + '/status.json'
    run_cmd(
        [
            sys.executable,
            'scripts/kg1_colab_v1243_launcher.py',
            '--phase', PHASE,
            '--run-mode', 'real_train',
            '--run-id', real_run_id,
            '--allow-real-train',
            '--target-accuracy', TARGET_ACCURACY,
            '--live-log-repo', os.environ.get('KG1_LIVE_LOG_HF_REPO', ''),
            '--live-log-repo-type', os.environ.get('KG1_LIVE_LOG_HF_REPO_TYPE', 'dataset'),
            '--require-live-log-upload',
            '--accept-gpu-spend',
            '--output-repo', OUTPUT_REPO,
        ],
        cwd=ROOT,
        log_path=LOG_ROOT / (real_run_id + '_launcher.log'),
    )
    wrapper_event('REAL_TRAIN', 'OK', run_id=real_run_id, hf_log_path=os.environ['KG1_LIVE_LOG_HF_PATH'])
    real_train_executed = True
print('=== V1243 REAL TRAIN END ===', flush=True)
effective_mode = 'real_train' if real_train_executed else ('model_dryrun' if model_dryrun_executed else 'tokenize_only')
if REQUIRE_REAL_TRAIN == '1' and not real_train_executed:
    wrapper_event('WRAPPER_END', 'FAIL', effective_mode=effective_mode, model_dryrun_executed=model_dryrun_executed, real_train_executed=real_train_executed, reason='required real train was not executed')
    raise RuntimeError('Required real train was not executed; refusing to report wrapper success.')
if REQUIRE_MODEL_DRYRUN == '1' and not model_dryrun_executed:
    wrapper_event('WRAPPER_END', 'FAIL', effective_mode=effective_mode, model_dryrun_executed=model_dryrun_executed, real_train_executed=real_train_executed, reason='required model dry-run was not executed')
    raise RuntimeError('Required model dry-run was not executed; refusing to report wrapper success.')
wrapper_event(
    'WRAPPER_END',
    'OK',
    effective_mode=effective_mode,
    model_dryrun_executed=model_dryrun_executed,
    real_train_executed=real_train_executed,
    final_adapter_created=real_train_executed,
)
print('=== V1243 ONECELL REALTIME LAUNCHER END ===', flush=True)
"""


MODEL_DRYRUN_ONE_CELL_SOURCE = (
    ONE_CELL_SOURCE
    .replace(
        "COLAB_URL = 'https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/master/notebooks/KG1_V1243_COLAB_REALTIME_LAUNCHER.ipynb'\n",
        f"COLAB_URL = '{MODEL_DRYRUN_COLAB_URL}'\n",
    )
    .replace(
        "RUN_MODEL_DRYRUN = os.environ.get('KG1_V1243_RUN_MODEL_DRYRUN', '0')\n",
        "RUN_MODEL_DRYRUN = os.environ.get('KG1_V1243_RUN_MODEL_DRYRUN', '1')\n",
    )
    .replace(
        "ACCEPT_GPU_SPEND = os.environ.get('KG1_ACCEPT_GPU_SPEND', '0')\n",
        "ACCEPT_GPU_SPEND = os.environ.get('KG1_ACCEPT_GPU_SPEND', '1')\n",
    )
    .replace(
        "INSTALL_CAUSAL_CONV1D = os.environ.get('KG1_INSTALL_CAUSAL_CONV1D', INSTALL_CAUSAL_CONV1D)\n"
        "if (RUN_MODEL_DRYRUN == '1' or RUN_TRAIN == '1') and INSTALL_CAUSAL_CONV1D != '1':\n"
        "    print('auto_enable_causal_conv1d_install=True reason=GPU phase requires real causal-conv1d', flush=True)\n"
        "    INSTALL_CAUSAL_CONV1D = '1'\n"
        "    os.environ['KG1_INSTALL_CAUSAL_CONV1D'] = '1'\n\n"
        "os.environ.setdefault('PYTHONUNBUFFERED', '1')\n",
        "INSTALL_CAUSAL_CONV1D = os.environ.get('KG1_INSTALL_CAUSAL_CONV1D', INSTALL_CAUSAL_CONV1D)\n\n"
        "# Hard-lock this dedicated notebook against stale Colab Secrets from tokenize-only runs.\n"
        "os.environ['KG1_V1243_RUN_MODEL_DRYRUN'] = '1'\n"
        "os.environ['KG1_ACCEPT_GPU_SPEND'] = '1'\n"
        "os.environ['KG1_V1243_RUN_TRAIN'] = '0'\n"
        "os.environ['KG1_V1243_REQUIRE_MODEL_DRYRUN'] = '1'\n"
        "os.environ['KG1_V1243_FORCE_PACK_ADAPTER_DEFAULTS'] = '1'\n"
        "os.environ['KG1_INSTALL_CAUSAL_CONV1D'] = '1'\n"
        "RUN_MODEL_DRYRUN = '1'\n"
        "ACCEPT_GPU_SPEND = '1'\n"
        "RUN_TRAIN = '0'\n"
        "REQUIRE_MODEL_DRYRUN = '1'\n"
        "INSTALL_CAUSAL_CONV1D = '1'\n"
        "print('model_dryrun_launcher_hard_lock = true', flush=True)\n"
        "print('force_pack_adapter_defaults = true', flush=True)\n\n"
        "os.environ.setdefault('PYTHONUNBUFFERED', '1')\n",
    )
    .replace(
        "print('=== V1243 ONECELL REALTIME LAUNCHER START ===', flush=True)\n",
        "print('=== V1243 ONECELL MODEL DRYRUN LAUNCHER START ===', flush=True)\n",
    )
    .replace(
        "print('=== V1243 ONECELL REALTIME LAUNCHER END ===', flush=True)\n",
        "print('=== V1243 ONECELL MODEL DRYRUN LAUNCHER END ===', flush=True)\n",
    )
)


def build_notebook(
    *,
    colab_url: str,
    notebook_name: str,
    one_cell_source: str,
    description: str,
) -> dict[str, object]:
    return {
        "cells": [
            markdown_cell(
                "v1243-md-01",
                f"""# KG1 V1243 Colab Realtime Launcher

Colab URL:

`{colab_url}`

{description}
""",
            ),
            code_cell("v1243-code-02", one_cell_source),
        ],
        "metadata": {
            "colab": {
                "name": notebook_name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(
            build_notebook(
                colab_url=COLAB_URL,
                notebook_name=NOTEBOOK_PATH.name,
                one_cell_source=ONE_CELL_SOURCE,
                description=(
                    "One-cell launcher. Press **Run** once: it automatically checks HF live-log access, "
                    "disk capacity, downloads the launch pack, installs bounded dependencies, and runs "
                    "tokenization dry-run. GPU model-load dry-run is locked behind `KG1_ACCEPT_GPU_SPEND=1`, "
                    "`KG1_V1243_RUN_MODEL_DRYRUN=1`, and a pinned baseline adapter. Real training remains "
                    "locked behind `KG1_V1243_RUN_TRAIN=1`, model dry-run, and `OUTPUT_REPO`."
                ),
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    MODEL_DRYRUN_NOTEBOOK_PATH.write_text(
        json.dumps(
            build_notebook(
                colab_url=MODEL_DRYRUN_COLAB_URL,
                notebook_name=MODEL_DRYRUN_NOTEBOOK_PATH.name,
                one_cell_source=MODEL_DRYRUN_ONE_CELL_SOURCE,
                description=(
                    "One-cell model dry-run launcher. Press **Run** once: it runs the same tokenization gate, "
                    "then automatically enters GPU model-load/adapter dry-run with `KG1_V1243_RUN_MODEL_DRYRUN=1` "
                    "and `KG1_ACCEPT_GPU_SPEND=1` by default. Real training stays disabled because "
                    "`KG1_V1243_RUN_TRAIN=0` remains the default."
                ),
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"wrote {MODEL_DRYRUN_NOTEBOOK_PATH}")
    print(f"colab_url={COLAB_URL}")
    print(f"model_dryrun_colab_url={MODEL_DRYRUN_COLAB_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

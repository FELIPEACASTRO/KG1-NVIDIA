#!/usr/bin/env python3
"""Build the V216 equation score-push gated Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V216_EQUATION_SCORE_PUSH_COLAB.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v216-equation-score-push/notebooks/KG1_V216_EQUATION_SCORE_PUSH_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v216-equation-score-push/notebooks/KG1_V216_EQUATION_SCORE_PUSH_COLAB.ipynb"
)
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
V194_ADAPTER_DRIVE = "/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter"
V194_VAL_CSV_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/"
    "validation/official_train_seed42_stratified10_val.csv"
)
TRAIN_SHA = "8cfd065c102187b12c131aae7475c35e28073721175b4e6108004b0afc4d5d03"
VAL_SHA = "80efe71260c8589b998699543c85aff3ff140bc90e431dfa0ec33bce3e0921c0"
TRAIN_ROWS = 10210
VAL_ROWS = 681

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v216-{prefix}-{_CELL_COUNTER:02d}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md"),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            f"""# KG1 V216 Equation Score-Push Colab

Purpose: run a gated same-day continuation focused on `equation_transform` while preserving `bit_manipulation`.

This notebook:

- uses the V216 score-push dataset, not the raw V216 focus file;
- rejects the raw V216 empty-answer rows by hash-gated manifest;
- starts from the protected V194 adapter;
- trains a small attention/SSM projection delta;
- runs weak per-family gates before full validation;
- packages only if full proxy gates pass;
- never submits to Kaggle.

Colab URL:

`{COLAB_URL}`
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V216 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V216 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration and hard submit lock.
print('=== V216 CONFIG START ===', flush=True)
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('KG1_ALLOW_VLLM_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_MOE_USE_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM_E8M0', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES', '0')
os.environ.setdefault('VLLM_DEEP_GEMM_WARMUP', 'skip')
os.environ.setdefault('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS', '0')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', os.environ.get('KG1_TORCH_CUDA_ARCH_LIST', '9.0'))
os.environ.setdefault('MAX_JOBS', os.environ.get('KG1_BUILD_MAX_JOBS', '4'))

VERSION = 'V216_EQUATION_SCORE_PUSH_20260507'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'v216-equation-score-push')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V216')
OUT_ROOT = DRIVE_ROOT / 'output_v216_equation_score_push'
BASELINE_OUT = OUT_ROOT / 'baseline_v194_weak'
DRY_OUT = OUT_ROOT / 'dry_run_v216_eqpush_lr3e8_s24'
TRAIN_OUT = OUT_ROOT / 'train_v216_eqpush_lr3e8_s24'
EVAL_OUT = OUT_ROOT / 'eval_v216_eqpush_lr3e8_s24'
PACKAGE_OUT = OUT_ROOT / 'package_v216_eqpush_lr3e8_s24'
V194_ADAPTER = pathlib.Path('{V194_ADAPTER_DRIVE}')
V194_VAL_CSV = pathlib.Path('{V194_VAL_CSV_DRIVE}')
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

RUN_BASELINE_WEAK = os.environ.get('KG1_V216_RUN_BASELINE_WEAK', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
RUN_DRY_RUN = os.environ.get('KG1_V216_RUN_DRY_RUN', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
RUN_TRAIN = os.environ.get('KG1_V216_RUN_TRAIN', '1').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_EVAL = os.environ.get('KG1_V216_RUN_EVAL', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
FORCE_RETRAIN = os.environ.get('KG1_V216_FORCE_RETRAIN', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
FORCE_REEVAL = os.environ.get('KG1_V216_FORCE_REEVAL', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_WEAK_SMOKE = os.environ.get('KG1_V216_RUN_WEAK_SMOKE', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}

TRAIN_SHA = '{TRAIN_SHA}'
VAL_SHA = '{VAL_SHA}'
TRAIN_ROWS_EXPECTED = {TRAIN_ROWS}
VAL_ROWS_EXPECTED = {VAL_ROWS}
V216_LR = os.environ.get('KG1_V216_LR', '3e-8')
V216_MAX_STEPS = os.environ.get('KG1_V216_MAX_STEPS', '24')
V216_TRAINABLE_MODULES = os.environ.get('KG1_V216_TRAINABLE_MODULES', 'q_proj,k_proj,v_proj,o_proj,out_proj,in_proj')
V216_VLLM_PIP_SPEC = os.environ.get('KG1_V216_VLLM_PIP_SPEC', 'vllm==0.20.1')
V216_CAUSAL_CONV1D_PIP_SPEC = os.environ.get('KG1_V216_CAUSAL_CONV1D_PIP_SPEC', 'causal-conv1d==1.6.1')
V216_MAMBA_SSM_PIP_SPEC = os.environ.get('KG1_V216_MAMBA_SSM_PIP_SPEC', 'mamba-ssm==2.3.1')
V216_MAX_LENGTH = int(os.environ.get('KG1_V216_MAX_LENGTH', '4096'))
V216_BATCH_SIZE = int(os.environ.get('KG1_V216_BATCH_SIZE', '4'))
V216_MICRO_BATCH_SIZE = int(os.environ.get('KG1_V216_MICRO_BATCH_SIZE', '1'))

BASELINE_WEAK_EXPECTED = 190
BASELINE_EQ_EXPECTED = 55
BASELINE_BIT_EXPECTED = 135
BASELINE_WEAK_MAX_TRUNC = 3
WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, BASELINE_OUT, DRY_OUT, TRAIN_OUT, EVAL_OUT, PACKAGE_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V194_VAL_CSV =', V194_VAL_CSV, flush=True)
print('RUN_BASELINE_WEAK =', RUN_BASELINE_WEAK, flush=True)
print('RUN_DRY_RUN =', RUN_DRY_RUN, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('FORCE_RETRAIN =', FORCE_RETRAIN, flush=True)
print('FORCE_REEVAL =', FORCE_REEVAL, flush=True)
print('V216_LR =', V216_LR, flush=True)
print('V216_MAX_STEPS =', V216_MAX_STEPS, flush=True)
print('V216_TRAINABLE_MODULES =', V216_TRAINABLE_MODULES, flush=True)
print('V216_VLLM_PIP_SPEC =', V216_VLLM_PIP_SPEC, flush=True)
print('V216_CAUSAL_CONV1D_PIP_SPEC =', V216_CAUSAL_CONV1D_PIP_SPEC, flush=True)
print('V216_MAMBA_SSM_PIP_SPEC =', V216_MAMBA_SSM_PIP_SPEC, flush=True)
print('TORCH_CUDA_ARCH_LIST =', os.environ.get('TORCH_CUDA_ARCH_LIST', ''), flush=True)
print('MAX_JOBS =', os.environ.get('MAX_JOBS', ''), flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')
print('=== V216 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging and heartbeat.
print('=== V216 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resource_snapshot_line():
    parts = []
    try:
        meminfo = {}
        with open('/proc/meminfo', encoding='utf-8') as handle:
            for line in handle:
                key, raw = line.split(':', 1)
                meminfo[key] = int(raw.strip().split()[0]) / 1024 / 1024
        parts.append('ram_total={:.1f}GiB ram_available={:.1f}GiB'.format(meminfo.get('MemTotal', 0.0), meminfo.get('MemAvailable', 0.0)))
    except Exception as exc:
        parts.append(f'ram=unavailable:{type(exc).__name__}')
    try:
        usage = shutil.disk_usage('/content')
        parts.append('disk_content_free={:.1f}GiB disk_content_total={:.1f}GiB'.format(usage.free / 1024**3, usage.total / 1024**3))
    except Exception as exc:
        parts.append(f'disk=unavailable:{type(exc).__name__}')
    try:
        gpu = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
            timeout=10,
        ).strip().replace('\\n', ' | ')
        parts.append(f'gpu=[{gpu}]')
    except Exception as exc:
        parts.append(f'gpu=unavailable:{type(exc).__name__}')
    return ' '.join(parts)

def run_cmd(cmd, cwd=None, env=None, log_path=None, check=True, heartbeat_s=60):
    cmd = [str(x) for x in cmd]
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or pathlib.Path.cwd(), flush=True)
    print('+', ' '.join(cmd), flush=True)
    handle = None
    if log_path:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open('w', encoding='utf-8')
        print('log_path =', log_path, flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    q = queue.Queue()
    def reader():
        try:
            for output_line in proc.stdout:
                q.put(output_line)
        finally:
            q.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    live_lines = 0
    suppressed_lines = 0
    tail = []
    last_heartbeat = time.time()
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            item = 'HEARTBEAT'
        now = time.time()
        if item == 'HEARTBEAT':
            if heartbeat_s and now - last_heartbeat >= heartbeat_s and proc.poll() is None:
                print('[V216 heartbeat] elapsed_s={:.1f} {}'.format(now - started, resource_snapshot_line()), flush=True)
                last_heartbeat = now
            continue
        if item is None:
            break
        if handle:
            handle.write(item)
            handle.flush()
        tail.append(item.rstrip('\\n'))
        tail = tail[-40:]
        if live_lines < 80 or any(marker in item for marker in ['summary =', 'returncode', 'ERROR', 'Traceback', 'gate', 'report_json', 'weak_', 'full_']):
            print(item, end='', flush=True)
            live_lines += 1
        else:
            suppressed_lines += 1
    rc = proc.wait()
    elapsed = time.time() - started
    if handle:
        handle.close()
    print('returncode =', rc, flush=True)
    print('elapsed_s = {:.1f}'.format(elapsed), flush=True)
    if suppressed_lines:
        print('command_output_suppressed_lines =', suppressed_lines, flush=True)
    if rc != 0:
        print('command_tail_on_failure =', flush=True)
        for line in tail[-30:]:
            print(line, flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and rc != 0:
        raise RuntimeError(f'Command failed rc={rc}: {cmd}')
    return rc

def ensure_import(import_name, pip_spec=None):
    try:
        module = importlib.import_module(import_name)
        print(import_name, 'version=', getattr(module, '__version__', 'unknown'), flush=True)
        return module
    except Exception as exc:
        print(import_name, 'missing:', repr(exc), flush=True)
        if not pip_spec:
            raise
        safe_name = import_name.replace('.', '_').replace('-', '_')
        run_cmd(
            [sys.executable, '-m', 'pip', 'install', '-q', pip_spec],
            log_path=OUT_ROOT / f'pip_install_{safe_name}.log',
        )
        module = importlib.import_module(import_name)
        print(import_name, 'version=', getattr(module, '__version__', 'unknown'), flush=True)
        return module

def verify_import_subprocess(import_name, label=None, check=False):
    label = label or import_name
    code = (
        "import importlib; "
        f"m=importlib.import_module({import_name!r}); "
        f"print({label!r} + ' subprocess_version=' + str(getattr(m, '__version__', 'unknown')))"
    )
    return run_cmd(
        [sys.executable, '-c', code],
        log_path=OUT_ROOT / f'verify_import_{label.replace(".", "_").replace("-", "_")}.log',
        check=check,
        heartbeat_s=0,
    )

def install_pip_spec(spec, label, force=False):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q']
    if force:
        cmd.extend(['--force-reinstall'])
    cmd.append(spec)
    return run_cmd(
        cmd,
        log_path=OUT_ROOT / f'pip_install_{label.replace(".", "_").replace("-", "_")}.log',
        check=True,
    )

def uninstall_pip_specs(specs, label):
    specs = [str(spec) for spec in specs if str(spec).strip()]
    if not specs:
        return 0
    return run_cmd(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', *specs],
        log_path=OUT_ROOT / f'pip_uninstall_{label.replace(".", "_").replace("-", "_")}.log',
        check=False,
    )

def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.is_dir() and (path / 'adapter_config.json').exists() and (
        (path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists()
    )

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

print('python =', sys.version, flush=True)
print('=== V216 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone branch, compile scripts, and verify V216 score-push data hashes.
print('=== V216 REPO SETUP START ===', flush=True)
if ROOT.exists():
    print('removing existing ROOT before fresh clone:', ROOT, flush=True)
    shutil.rmtree(ROOT)
run_cmd(
    ['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, ROOT],
    log_path=OUT_ROOT / 'repo_clone.log',
)
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
print('repo_commit =', commit, flush=True)

for rel in [
    'src/competition_utils.py',
    'scripts/evaluate_lora_adapter.py',
    'scripts/hf_job_train_v90.py',
    'scripts/build_v216_score_push_dataset.py',
]:
    py_path = ROOT / rel
    print('compile_target =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)
    print('compiled', rel, flush=True)

train_path = ROOT / 'data/v216/v216_score_push_train.jsonl'
val_path = ROOT / 'data/v216/v216_score_push_val.jsonl'
manifest_path = ROOT / 'data/v216/v216_score_push_manifest.json'
for required in [train_path, val_path, manifest_path]:
    print('required_path =', required, 'exists =', required.exists(), flush=True)
    if not required.exists():
        raise FileNotFoundError(required)
observed_train_sha = sha256_file(train_path)
observed_val_sha = sha256_file(val_path)
manifest = read_json(manifest_path)
print('observed_train_sha256 =', observed_train_sha, flush=True)
print('observed_val_sha256 =', observed_val_sha, flush=True)
print('manifest_status =', manifest.get('status'), flush=True)
print('prefilter_rejected =', json.dumps(manifest.get('prefilter_rejected', {}), sort_keys=True), flush=True)
print('train_summary =', json.dumps(manifest.get('train', {}), indent=2, sort_keys=True)[:4000], flush=True)
print('validation_summary =', json.dumps(manifest.get('validation', {}), indent=2, sort_keys=True)[:4000], flush=True)
if observed_train_sha != TRAIN_SHA:
    raise RuntimeError('V216 score-push train SHA mismatch')
if observed_val_sha != VAL_SHA:
    raise RuntimeError('V216 score-push val SHA mismatch')
if manifest.get('status') != 'PASS':
    raise RuntimeError('V216 score-push manifest is not PASS')
if int(manifest.get('train', {}).get('rows', -1)) != TRAIN_ROWS_EXPECTED:
    raise RuntimeError('V216 score-push train row count mismatch')
if int(manifest.get('validation', {}).get('rows', -1)) != VAL_ROWS_EXPECTED:
    raise RuntimeError('V216 score-push validation row count mismatch')
print('=== V216 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: dependency, GPU, disk, and adapter audit.
print('=== V216 RUNTIME AUDIT START ===', flush=True)
ensure_import('pandas', 'pandas')
ensure_import('safetensors', 'safetensors')
ensure_import('huggingface_hub', 'huggingface_hub')
if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1':
    ensure_import('hf_transfer', 'hf_transfer')

# Avoid importing torch/vLLM in the notebook kernel before dependency repair.
# Pip may replace torch/vLLM wheels; subprocess checks keep the notebook process
# from retaining stale partial modules after an install.
for import_name, spec in [
    ('transformers', 'transformers'),
    ('peft', 'peft>=0.18.1'),
]:
    if verify_import_subprocess(import_name, check=False) != 0:
        install_pip_spec(spec, import_name, force=False)
        verify_import_subprocess(import_name, check=True)

if verify_import_subprocess('mamba_ssm', check=False) != 0:
    print('mamba_ssm subprocess import failed; installing causal-conv1d first, then mamba-ssm.', flush=True)
    if verify_import_subprocess('vllm', check=False) == 0:
        print(
            'vLLM is installed before mamba_ssm; uninstalling vLLM before the causal-conv1d/mamba build.',
            flush=True,
        )
        uninstall_pip_specs(['vllm'], 'vllm_pre_mamba_repair')
        if verify_import_subprocess('vllm', check=False) == 0:
            raise RuntimeError('Could not remove vLLM before mamba_ssm repair; see pip_uninstall_vllm_pre_mamba_repair.log')
    ensure_import('ninja', 'ninja')
    if verify_import_subprocess('causal_conv1d', check=False) != 0:
        causal_rc = run_cmd(
            [sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', V216_CAUSAL_CONV1D_PIP_SPEC],
            log_path=OUT_ROOT / 'pip_install_causal_conv1d.log',
            check=False,
        )
        if causal_rc != 0:
            print('causal-conv1d build failed once; refreshing build tooling and retrying without pip cache.', flush=True)
            run_cmd(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '-q',
                    '--upgrade',
                    'pip',
                    'setuptools',
                    'wheel',
                    'packaging',
                    'ninja',
                ],
                log_path=OUT_ROOT / 'pip_install_build_tooling_retry.log',
            )
            causal_rc = run_cmd(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--progress-bar',
                    'off',
                    '--no-cache-dir',
                    '--no-build-isolation',
                    V216_CAUSAL_CONV1D_PIP_SPEC,
                ],
                log_path=OUT_ROOT / 'pip_install_causal_conv1d_retry.log',
                check=False,
            )
        if causal_rc != 0:
            raise RuntimeError('causal-conv1d build failed after retry; see pip_install_causal_conv1d*.log')
        verify_import_subprocess('causal_conv1d', check=True)
    else:
        print('causal_conv1d subprocess import already OK; skipping install.', flush=True)
    run_cmd(
        [sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', V216_MAMBA_SSM_PIP_SPEC],
        log_path=OUT_ROOT / 'pip_install_mamba_ssm.log',
    )
    verify_import_subprocess('mamba_ssm', check=True)

vllm_rc = verify_import_subprocess('vllm', check=False)
if vllm_rc != 0:
    print('vLLM subprocess import failed; installing pinned V216_VLLM_PIP_SPEC =', V216_VLLM_PIP_SPEC, flush=True)
    install_pip_spec(V216_VLLM_PIP_SPEC, 'vllm', force=False)
    verify_import_subprocess('vllm', check=True)
else:
    print('vLLM subprocess import already OK; skipping install.', flush=True)

torch_check_code = (
    "import json, torch; "
    "props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
    "print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), "
    "'cuda_available': torch.cuda.is_available(), "
    "'gpu_name': props.name if props else '', "
    "'gpu_total_gib': props.total_memory/1024**3 if props else 0.0}))"
)
torch_audit_path = OUT_ROOT / 'verify_torch_cuda.jsonl'
torch_rc = run_cmd(
    [sys.executable, '-c', torch_check_code],
    log_path=torch_audit_path,
    check=False,
    heartbeat_s=0,
)
if torch_rc != 0:
    raise RuntimeError('Torch CUDA subprocess audit failed; restart the runtime and rerun from the top.')
torch_audit_lines = [line.strip() for line in torch_audit_path.read_text(encoding='utf-8').splitlines() if line.strip()]
torch_audit = json.loads(torch_audit_lines[-1])
print('torch_audit =', json.dumps(torch_audit, sort_keys=True), flush=True)
if not torch_audit.get('cuda_available'):
    raise RuntimeError('CUDA GPU is required for V216 train/eval.')
gpu_name = torch_audit.get('gpu_name', '')
gpu_total_gib = float(torch_audit.get('gpu_total_gib', 0.0))
content_usage = shutil.disk_usage('/content')
content_free_gib = content_usage.free / 1024**3
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if gpu_total_gib < 70:
    raise RuntimeError(f'Need H100/A100 80GB-class GPU; found {gpu_name} {gpu_total_gib:.1f}GiB')
if content_free_gib < 70:
    raise RuntimeError(f'/content free disk too small: {content_free_gib:.1f}GiB < 70GiB')

if not is_complete_adapter_dir(V194_ADAPTER):
    raise RuntimeError(f'V194 adapter incomplete or missing: {V194_ADAPTER}')
adapter_config = read_json(V194_ADAPTER / 'adapter_config.json')
print('v194_adapter_r =', adapter_config.get('r'), flush=True)
print('v194_target_modules =', adapter_config.get('target_modules'), flush=True)
print('v194_target_parameters =', adapter_config.get('target_parameters'), flush=True)
if int(adapter_config.get('r', 999)) > 32:
    raise RuntimeError('V194 adapter rank exceeds rank 32 gate')
weights_path = V194_ADAPTER / 'adapter_model.safetensors'
if weights_path.exists():
    from safetensors import safe_open
    with safe_open(str(weights_path), framework='pt', device='cpu') as handle:
        key_count = len(list(handle.keys()))
    print('v194_adapter_tensor_count =', key_count, flush=True)
    print('v194_adapter_weight_bytes =', weights_path.stat().st_size, flush=True)
    if key_count < 1000:
        raise RuntimeError('V194 adapter tensor count unexpectedly low')
print('=== V216 RUNTIME AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: build validation CSVs and optionally verify protected V194 weak baseline.
print('=== V216 VALIDATION AND BASELINE START ===', flush=True)
import pandas as pd
sys.path.insert(0, str(ROOT))
from src.competition_utils import classify_puzzle

if not V194_VAL_CSV.exists():
    raise FileNotFoundError(f'Missing V194 validation CSV on Drive: {V194_VAL_CSV}')
full_df = pd.read_csv(V194_VAL_CSV)
if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
    raise RuntimeError('V194 validation CSV must contain prompt and answer')
full_df['type'] = full_df['prompt'].map(classify_puzzle)
weak_df = full_df[full_df['type'].isin({'bit_manipulation', 'equation_transform'})].copy()
strong_df = full_df[full_df['type'].isin({'gravity_constant', 'numeral_system', 'text_encryption', 'unit_conversion'})].copy()
full_eval_csv = EVAL_OUT / 'v216_full_947.csv'
weak_eval_csv = EVAL_OUT / 'v216_weak_315.csv'
strong_eval_csv = EVAL_OUT / 'v216_strong_632.csv'
full_df.to_csv(full_eval_csv, index=False)
weak_df.to_csv(weak_eval_csv, index=False)
strong_df.to_csv(strong_eval_csv, index=False)
print('full_rows =', len(full_df), 'path =', full_eval_csv, flush=True)
print('weak_rows =', len(weak_df), 'path =', weak_eval_csv, flush=True)
print('strong_rows =', len(strong_df), 'path =', strong_eval_csv, flush=True)
print('per_family_counts =', full_df['type'].value_counts().sort_index().to_dict(), flush=True)
if len(full_df) != 947 or len(weak_df) != 315 or len(strong_df) != 632:
    raise RuntimeError('Validation row counts are not 947/315/632')

baseline_report_path = BASELINE_OUT / 'v194_baseline_weak_eval_report.json'
baseline_per_task_path = BASELINE_OUT / 'v194_baseline_weak_per_task.csv'
if RUN_BASELINE_WEAK and (FORCE_REEVAL or not baseline_report_path.exists()):
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(weak_eval_csv),
            '--questions-csv', str(weak_eval_csv),
            '--adapter', str(V194_ADAPTER),
            '--base-model-path', MODEL_NAME,
            '--label', 'v194_baseline_weak',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(BASELINE_OUT),
        ],
        cwd=ROOT,
        log_path=BASELINE_OUT / 'v194_baseline_weak.log',
        check=True,
    )
    print('baseline weak eval returncode =', rc, flush=True)
elif RUN_BASELINE_WEAK:
    print('reusing existing baseline_report_path =', baseline_report_path, flush=True)
else:
    print('RUN_BASELINE_WEAK is false; baseline gate skipped only for diagnostics.', flush=True)

baseline_report = None
baseline_per_task = None
if baseline_report_path.exists():
    baseline_report = read_json(baseline_report_path)
    print('baseline_weak_report =', json.dumps(baseline_report, indent=2, sort_keys=True), flush=True)
    if int(baseline_report.get('correct', -1)) < BASELINE_WEAK_EXPECTED:
        raise RuntimeError('V194 weak baseline below expected; stop before training')
    if int(baseline_report.get('truncated', 999)) > BASELINE_WEAK_MAX_TRUNC:
        raise RuntimeError('V194 weak truncation too high; stop before training')
    if baseline_per_task_path.exists():
        baseline_per_task = pd.read_csv(baseline_per_task_path)
        print('baseline_weak_per_task =', baseline_per_task.to_string(index=False), flush=True)
        by_task = {row['task_type']: int(row['correct']) for _, row in baseline_per_task.iterrows()}
        if by_task.get('equation_transform', -1) < BASELINE_EQ_EXPECTED:
            raise RuntimeError('V194 equation_transform baseline below expected')
        if by_task.get('bit_manipulation', -1) < BASELINE_BIT_EXPECTED:
            raise RuntimeError('V194 bit_manipulation baseline below expected')
print('=== V216 VALIDATION AND BASELINE END ===', flush=True)
"""
        ),
        code(
            """# CELL: dry-run and train environment builder.
print('=== V216 TRAINING ENV SETUP START ===', flush=True)
source_weights = manifest['recommended_training_env']['SOURCE_WEIGHTS']
subcategory_weights = manifest['recommended_training_env']['SUBCATEGORY_WEIGHTS']
print('SOURCE_WEIGHTS =', source_weights, flush=True)
print('SUBCATEGORY_WEIGHTS =', subcategory_weights, flush=True)

def training_env(output_dir, dry_run):
    env = os.environ.copy()
    target_modules = ','.join(adapter_config.get('target_modules') or [])
    target_parameters = ','.join(adapter_config.get('target_parameters') or [])
    env.update({
        'MODEL_NAME': MODEL_NAME,
        'MODEL_REVISION': MODEL_REVISION,
        'MODEL_DEVICE_MAP': 'cuda',
        'ATTN_IMPLEMENTATION': 'eager',
        'TORCH_ALLOW_TF32': '1',
        'TORCH_FLOAT32_MATMUL_PRECISION': 'high',
        'TORCH_DISABLE_CUDNN_SDP': '1',
        'TORCH_FORCE_MATH_SDP': os.environ.get('KG1_V216_FORCE_MATH_SDP', '0'),
        'GRADIENT_CHECKPOINTING': '1',
        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM', 'false'),
        'HF_HUB_ENABLE_HF_TRANSFER': os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '1'),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'),
        'DATA_REPO': 'local',
        'DATA_FILE': str(ROOT / 'data/v216/v216_score_push_train.jsonl'),
        'VAL_FILE': str(ROOT / 'data/v216/v216_score_push_val.jsonl'),
        'EXPECTED_TRAIN_SHA256': TRAIN_SHA,
        'EXPECTED_VAL_SHA256': VAL_SHA,
        'MIN_TRAIN_EXAMPLES': str(TRAIN_ROWS_EXPECTED),
        'MIN_VAL_EXAMPLES': str(VAL_ROWS_EXPECTED),
        'MIN_TOKENIZED_TRAIN_EXAMPLES': str(TRAIN_ROWS_EXPECTED),
        'MIN_TOKENIZED_VAL_EXAMPLES': str(VAL_ROWS_EXPECTED),
        'OUTPUT_DIR': str(output_dir),
        'OUTPUT_REPO': '',
        'RUN_ID': 'v216_eqpush_lr3e8_s24',
        'INIT_ADAPTER_DIR': str(V194_ADAPTER),
        'INIT_ADAPTER_LOAD_MODE': 'manual',
        'PEFT_MANUAL_LOAD_METHOD': 'direct',
        'FAIL_ON_MISSING_ADAPTER_KEYS': '1',
        'UPLOAD_TO_HF': '0',
        'UPLOAD_CHECKPOINTS_DURING_TRAINING': '0',
        'USE_BITSANDBYTES': '0',
        'DRY_RUN_VALIDATE_ONLY': '1' if dry_run else '0',
        'LORA_R': str(adapter_config.get('r', 32)),
        'LORA_ALPHA': str(adapter_config.get('lora_alpha', 32)),
        'LORA_DROPOUT': str(adapter_config.get('lora_dropout', 0.0)),
        'LORA_TARGET_MODULES': target_modules,
        'LORA_TARGET_PARAMETERS': target_parameters,
        'MAX_LENGTH': str(V216_MAX_LENGTH),
        'MAX_PROMPT_TRUNCATION_RATE': '0.0',
        'BATCH_SIZE': str(V216_BATCH_SIZE),
        'MICRO_BATCH_SIZE': str(V216_MICRO_BATCH_SIZE),
        'LEARNING_RATE': V216_LR,
        'FINAL_LEARNING_RATE': V216_LR,
        'ADAM_BETA1': '0.9',
        'ADAM_BETA2': '0.95',
        'ADAM_EPS': '1e-8',
        'WEIGHT_DECAY': '0.0',
        'GRAD_CLIP_NORM': os.environ.get('KG1_V216_GRAD_CLIP_NORM', '0.25'),
        'NUM_EPOCHS': '1',
        'MAX_STEPS': V216_MAX_STEPS,
        'SAVE_EVERY_STEPS': '0',
        'EVAL_EVERY_STEPS': '0',
        'EVAL_MAX_EXAMPLES': os.environ.get('KG1_V216_EVAL_MAX_EXAMPLES', '32'),
        'LOG_EVERY_STEPS': '1',
        'MICRO_LOG_EVERY': '0',
        'SEED': '216',
        'MAX_TRAINABLE_PARAM_RATIO': '0.025',
        'TRAINABLE_LORA_MODULES': V216_TRAINABLE_MODULES,
        'SAMPLING_MODE': 'weighted_replacement',
        'SOURCE_WEIGHTS': source_weights,
        'SUBCATEGORY_WEIGHTS': subcategory_weights,
        'BASELINE_EVAL_BEFORE_TRAIN': '0',
        'REQUIRE_FINAL_EVAL_LTE_BASELINE': '0',
        'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': '-1',
        'ABORT_MAX_RESERVED_GIB': os.environ.get('KG1_V216_ABORT_MAX_RESERVED_GIB', '78'),
        'COMPUTE_PROVIDER': 'colab_h100',
    })
    return env

print('training script =', ROOT / 'scripts/hf_job_train_v90.py', flush=True)
print('dry-run output dir =', DRY_OUT, flush=True)
print('train output dir =', TRAIN_OUT, flush=True)
print('=== V216 TRAINING ENV SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: dry-run model/adapter trainability check.
print('=== V216 DRY RUN START ===', flush=True)
if RUN_DRY_RUN:
    rc = run_cmd(
        [sys.executable, str(ROOT / 'scripts/hf_job_train_v90.py')],
        cwd=ROOT,
        env=training_env(DRY_OUT, dry_run=True),
        log_path=DRY_OUT / 'dry_run.log',
        check=True,
    )
    dry_report = DRY_OUT / 'dry_run_model_recipe_report.json'
    print('dry_run_report =', dry_report, 'exists =', dry_report.exists(), flush=True)
    if not dry_report.exists():
        raise RuntimeError('dry_run_model_recipe_report.json was not written')
    report = read_json(dry_report)
    print('dry_run_decision =', json.dumps(report.get('decision', {}), indent=2, sort_keys=True), flush=True)
    print('trainable_parameters =', json.dumps(report.get('trainable_parameters', {}), indent=2, sort_keys=True), flush=True)
else:
    print('RUN_DRY_RUN is false; skipping dry-run. This is not recommended.', flush=True)
print('=== V216 DRY RUN END ===', flush=True)
"""
        ),
        code(
            """# CELL: V216 small delta training.
print('=== V216 TRAIN START ===', flush=True)
final_adapter = TRAIN_OUT / 'final_adapter'
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('final_adapter =', final_adapter, flush=True)
print('final_adapter_exists =', final_adapter.exists(), flush=True)
print('final_adapter_complete =', is_complete_adapter_dir(final_adapter), flush=True)

if FORCE_RETRAIN and final_adapter.exists():
    backup = final_adapter.with_name(final_adapter.name + '_backup_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    print('FORCE_RETRAIN moving existing final_adapter aside:', final_adapter, '->', backup, flush=True)
    shutil.move(str(final_adapter), str(backup))

if not RUN_TRAIN:
    print('RUN_TRAIN is false. To train, set os.environ[\"KG1_V216_RUN_TRAIN\"]=\"1\" and rerun this cell.', flush=True)
elif is_complete_adapter_dir(final_adapter):
    print('complete final_adapter already exists; skipping retrain:', final_adapter, flush=True)
else:
    rc = run_cmd(
        [sys.executable, str(ROOT / 'scripts/hf_job_train_v90.py')],
        cwd=ROOT,
        env=training_env(TRAIN_OUT, dry_run=False),
        log_path=TRAIN_OUT / 'train.log',
        check=True,
    )
    print('training returncode =', rc, flush=True)

print('final_adapter_exists_after =', final_adapter.exists(), flush=True)
print('final_adapter_complete_after =', is_complete_adapter_dir(final_adapter), flush=True)
if RUN_TRAIN and not is_complete_adapter_dir(final_adapter):
    raise RuntimeError('Training finished but final_adapter is incomplete; refusing eval.')
print('=== V216 TRAIN END ===', flush=True)
"""
        ),
        code(
            """# CELL: adapter integrity audit after train.
print('=== V216 FINAL ADAPTER INTEGRITY START ===', flush=True)
adapter_integrity = {'exists': final_adapter.exists(), 'complete': is_complete_adapter_dir(final_adapter)}
if final_adapter.exists():
    config_path = final_adapter / 'adapter_config.json'
    weights_path = final_adapter / 'adapter_model.safetensors'
    adapter_integrity['config_exists'] = config_path.exists()
    adapter_integrity['weights_exists'] = weights_path.exists()
    if config_path.exists():
        final_config = read_json(config_path)
        adapter_integrity['r'] = final_config.get('r')
        adapter_integrity['target_modules'] = final_config.get('target_modules')
        adapter_integrity['target_parameters'] = final_config.get('target_parameters')
    if weights_path.exists():
        from safetensors import safe_open
        with safe_open(str(weights_path), framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
        adapter_integrity['tensor_count'] = len(keys)
        adapter_integrity['weight_bytes'] = weights_path.stat().st_size
        adapter_integrity['sample_keys'] = keys[:10]
        if len(keys) < 1000:
            raise RuntimeError(f'Final adapter tensor count too low: {len(keys)}')
        if weights_path.stat().st_size < 100 * 1024 * 1024:
            raise RuntimeError('Final adapter weights unexpectedly small')
print('adapter_integrity =', json.dumps(adapter_integrity, indent=2, sort_keys=True), flush=True)
(TRAIN_OUT / 'final_adapter_integrity.json').write_text(json.dumps(adapter_integrity, indent=2, sort_keys=True), encoding='utf-8')
print('=== V216 FINAL ADAPTER INTEGRITY END ===', flush=True)
"""
        ),
        code(
            """# CELL: weak eval gate with per-family requirements.
print('=== V216 WEAK EVAL START ===', flush=True)
weak_report = None
weak_per_task = None
weak_eval_dir = EVAL_OUT / 'weak_eval'
weak_report_path = weak_eval_dir / 'v216_eqpush_weak_eval_report.json'
weak_per_task_path = weak_eval_dir / 'v216_eqpush_weak_per_task.csv'
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping weak eval.', flush=True)
elif not is_complete_adapter_dir(final_adapter):
    print('No complete final_adapter exists; skipping weak eval.', flush=True)
else:
    weak_eval_dir.mkdir(parents=True, exist_ok=True)
    if RUN_WEAK_SMOKE:
        weak_smoke_dir = EVAL_OUT / 'weak_eval_smoke'
        weak_smoke_dir.mkdir(parents=True, exist_ok=True)
        print('weak smoke eval enabled: limit=8', flush=True)
        rc = run_cmd(
            [
                sys.executable,
                str(ROOT / 'scripts/evaluate_lora_adapter.py'),
                '--solution-csv', str(weak_eval_csv),
                '--questions-csv', str(weak_eval_csv),
                '--adapter', str(final_adapter),
                '--base-model-path', MODEL_NAME,
                '--label', 'v216_eqpush_weak_smoke',
                '--seed', '42',
                '--limit', '8',
                '--output-dir', str(weak_smoke_dir),
            ],
            cwd=ROOT,
            log_path=weak_smoke_dir / 'weak_eval_smoke.log',
            check=True,
        )
        print('weak smoke returncode =', rc, flush=True)
    if FORCE_REEVAL or not weak_report_path.exists():
        rc = run_cmd(
            [
                sys.executable,
                str(ROOT / 'scripts/evaluate_lora_adapter.py'),
                '--solution-csv', str(weak_eval_csv),
                '--questions-csv', str(weak_eval_csv),
                '--adapter', str(final_adapter),
                '--base-model-path', MODEL_NAME,
                '--label', 'v216_eqpush_weak',
                '--seed', '42',
                '--limit', '0',
                '--output-dir', str(weak_eval_dir),
            ],
            cwd=ROOT,
            log_path=weak_eval_dir / 'weak_eval.log',
            check=True,
        )
        print('weak eval returncode =', rc, flush=True)
    else:
        print('reusing existing weak_report_path =', weak_report_path, flush=True)
    weak_report = read_json(weak_report_path)
    weak_per_task = pd.read_csv(weak_per_task_path)
    print('weak_report =', json.dumps(weak_report, indent=2, sort_keys=True), flush=True)
    print('weak_per_task =', weak_per_task.to_string(index=False), flush=True)
    weak_by_task = {row['task_type']: int(row['correct']) for _, row in weak_per_task.iterrows()}
    weak_correct = int(weak_report.get('correct', 0))
    weak_truncated = int(weak_report.get('truncated', 999))
    weak_eq_correct = weak_by_task.get('equation_transform', 0)
    weak_bit_correct = weak_by_task.get('bit_manipulation', 0)
    weak_gate_pass_for_full = (
        weak_correct >= WEAK_MIN_FOR_FULL
        and weak_eq_correct >= WEAK_EQ_MIN_FOR_FULL
        and weak_bit_correct >= WEAK_BIT_MIN_FOR_FULL
        and weak_truncated <= BASELINE_WEAK_MAX_TRUNC
    )
    print('weak_correct =', weak_correct, flush=True)
    print('weak_eq_correct =', weak_eq_correct, flush=True)
    print('weak_bit_correct =', weak_bit_correct, flush=True)
    print('weak_truncated =', weak_truncated, flush=True)
    print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V216 WEAK EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval only if weak gate passes.
print('=== V216 FULL EVAL START ===', flush=True)
full_report = None
full_per_task = None
full_eval_dir = EVAL_OUT / 'full_eval'
full_report_path = full_eval_dir / 'v216_eqpush_full_eval_report.json'
full_per_task_path = full_eval_dir / 'v216_eqpush_full_per_task.csv'
weak_gate_pass_for_full = bool(
    weak_report
    and weak_per_task is not None
    and int(weak_report.get('correct', 0)) >= WEAK_MIN_FOR_FULL
    and int(weak_report.get('truncated', 999)) <= BASELINE_WEAK_MAX_TRUNC
)
if weak_per_task is not None:
    by_task_tmp = {row['task_type']: int(row['correct']) for _, row in weak_per_task.iterrows()}
    weak_gate_pass_for_full = bool(
        weak_gate_pass_for_full
        and by_task_tmp.get('equation_transform', 0) >= WEAK_EQ_MIN_FOR_FULL
        and by_task_tmp.get('bit_manipulation', 0) >= WEAK_BIT_MIN_FOR_FULL
    )
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping full eval.', flush=True)
elif not is_complete_adapter_dir(final_adapter):
    print('No complete final_adapter exists; skipping full eval.', flush=True)
elif not weak_gate_pass_for_full:
    print('Weak gate failed; full eval blocked.', flush=True)
    print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', BASELINE_WEAK_MAX_TRUNC, flush=True)
elif FORCE_REEVAL or not full_report_path.exists():
    full_eval_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(full_eval_csv),
            '--questions-csv', str(full_eval_csv),
            '--adapter', str(final_adapter),
            '--base-model-path', MODEL_NAME,
            '--label', 'v216_eqpush_full',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(full_eval_dir),
        ],
        cwd=ROOT,
        log_path=full_eval_dir / 'full_eval.log',
        check=True,
    )
    print('full eval returncode =', rc, flush=True)
else:
    print('reusing existing full_report_path =', full_report_path, flush=True)

if full_report_path.exists():
    full_report = read_json(full_report_path)
    full_per_task = pd.read_csv(full_per_task_path)
    print('full_report =', json.dumps(full_report, indent=2, sort_keys=True), flush=True)
    print('full_per_task =', full_per_task.to_string(index=False), flush=True)
    full_correct = int(full_report.get('correct', 0))
    full_truncated = int(full_report.get('truncated', 999))
    full_candidate_gate = full_correct >= FULL_MIN_CANDIDATE and full_truncated <= FULL_MAX_TRUNC
    print('full_correct =', full_correct, flush=True)
    print('full_truncated =', full_truncated, flush=True)
    print('full_candidate_gate =', full_candidate_gate, flush=True)
print('=== V216 FULL EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: package candidate zip only if local full gate passes. No submit.
print('=== V216 PACKAGE START ===', flush=True)
package_report = None
full_candidate_gate = bool(full_report and int(full_report.get('correct', 0)) >= FULL_MIN_CANDIDATE and int(full_report.get('truncated', 999)) <= FULL_MAX_TRUNC)
if not full_candidate_gate:
    print('Full candidate gate did not pass; package step skipped.', flush=True)
else:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import zipfile

    source_weights_path = final_adapter / 'adapter_model.safetensors'
    out_adapter_dir = PACKAGE_OUT / 'adapter'
    out_zip_dir = PACKAGE_OUT / 'zip'
    out_adapter_dir.mkdir(parents=True, exist_ok=True)
    out_zip_dir.mkdir(parents=True, exist_ok=True)
    out_weights = out_adapter_dir / 'adapter_model.safetensors'
    out_config = out_adapter_dir / 'adapter_config.json'
    zip_path = out_zip_dir / 'v216_eqpush_adapter_only.zip'

    training_prefix = 'base_model.model.backbone.'
    training_lm_head_prefix = 'base_model.model.backbone.lm_head.'
    kaggle_prefix = 'base_model.model.model.'
    kaggle_lm_head_prefix = 'base_model.model.lm_head.'
    converted = {}
    renamed = 0
    already = 0
    unchanged = []
    with safe_open(str(source_weights_path), framework='pt', device='cpu') as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.startswith(training_lm_head_prefix):
                new_key = kaggle_lm_head_prefix + key[len(training_lm_head_prefix):]
                renamed += 1
            elif key.startswith(training_prefix):
                new_key = kaggle_prefix + key[len(training_prefix):]
                renamed += 1
            elif key.startswith(kaggle_prefix) or key.startswith(kaggle_lm_head_prefix):
                new_key = key
                already += 1
            else:
                new_key = key
                unchanged.append(key)
            converted[new_key] = tensor
    if unchanged:
        raise RuntimeError('Unexpected adapter key prefixes during package conversion: ' + repr(unchanged[:20]))
    save_file(converted, str(out_weights))
    cfg = dict(read_json(final_adapter / 'adapter_config.json'))
    cfg['inference_mode'] = True
    cfg['base_model_name_or_path'] = MODEL_NAME
    out_config.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_config, 'adapter_config.json')
        archive.write(out_weights, 'adapter_model.safetensors')

    package_report = {
        'zip_path': str(zip_path),
        'zip_bytes': zip_path.stat().st_size,
        'zip_sha256': sha256_file(zip_path),
        'adapter_model_sha256': sha256_file(out_weights),
        'tensor_count': len(converted),
        'renamed_count': renamed,
        'already_kaggle_count': already,
        'submit_authorized': False,
        'note': 'No Kaggle submit. Human approval required after reviewing local gates.',
    }
    (PACKAGE_OUT / 'package_report.json').write_text(json.dumps(package_report, indent=2, sort_keys=True), encoding='utf-8')
    print('package_report =', json.dumps(package_report, indent=2, sort_keys=True), flush=True)
print('=== V216 PACKAGE END ===', flush=True)
"""
        ),
        code(
            """# CELL: write final run manifest and decision.
print('=== V216 FINAL MANIFEST START ===', flush=True)
weak_task_counts = {}
if weak_per_task is not None:
    weak_task_counts = {row['task_type']: int(row['correct']) for _, row in weak_per_task.iterrows()}
full_task_counts = {}
if full_per_task is not None:
    full_task_counts = {row['task_type']: int(row['correct']) for _, row in full_per_task.iterrows()}

decision = {
    'run_train': bool(RUN_TRAIN),
    'baseline_weak_correct': int(baseline_report.get('correct', -1)) if baseline_report else None,
    'weak_correct': int(weak_report.get('correct', -1)) if weak_report else None,
    'weak_truncated': int(weak_report.get('truncated', -1)) if weak_report else None,
    'weak_equation_transform_correct': weak_task_counts.get('equation_transform'),
    'weak_bit_manipulation_correct': weak_task_counts.get('bit_manipulation'),
    'full_correct': int(full_report.get('correct', -1)) if full_report else None,
    'full_truncated': int(full_report.get('truncated', -1)) if full_report else None,
    'full_task_counts': full_task_counts,
    'weak_gate_pass_for_full': bool(
        weak_report
        and int(weak_report.get('correct', 0)) >= WEAK_MIN_FOR_FULL
        and int(weak_report.get('truncated', 999)) <= BASELINE_WEAK_MAX_TRUNC
        and weak_task_counts.get('equation_transform', 0) >= WEAK_EQ_MIN_FOR_FULL
        and weak_task_counts.get('bit_manipulation', 0) >= WEAK_BIT_MIN_FOR_FULL
    ),
    'full_candidate_gate': bool(full_report and int(full_report.get('correct', 0)) >= FULL_MIN_CANDIDATE and int(full_report.get('truncated', 999)) <= FULL_MAX_TRUNC),
    'package_created': bool(package_report),
    'submit_authorized': False,
}
if decision['full_candidate_gate']:
    decision['roadmap_next'] = 'Candidate package can be manually reviewed for Kaggle submission. No auto-submit.'
elif decision['weak_correct'] is not None and not decision['weak_gate_pass_for_full']:
    decision['roadmap_next'] = 'Reject V216 train result; do not full-eval/submit unless manually overriding after inspecting predictions.'
else:
    decision['roadmap_next'] = 'Review diagnostics before any additional GPU spend.'

run_manifest = {
    'version': VERSION,
    'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'repo_branch': REPO_BRANCH,
    'repo_commit': globals().get('commit', ''),
    'dataset_manifest': manifest,
    'paths': {
        'out_root': str(OUT_ROOT),
        'baseline_out': str(BASELINE_OUT),
        'dry_out': str(DRY_OUT),
        'train_out': str(TRAIN_OUT),
        'eval_out': str(EVAL_OUT),
        'package_out': str(PACKAGE_OUT),
        'final_adapter': str(final_adapter) if 'final_adapter' in globals() else '',
    },
    'settings': {
        'lr': V216_LR,
        'max_steps': V216_MAX_STEPS,
        'trainable_modules': V216_TRAINABLE_MODULES,
        'batch_size': V216_BATCH_SIZE,
        'micro_batch_size': V216_MICRO_BATCH_SIZE,
        'max_length': V216_MAX_LENGTH,
        'source_weights': source_weights,
        'subcategory_weights': subcategory_weights,
    },
    'decision': decision,
}
manifest_path = OUT_ROOT / 'v216_equation_score_push_manifest.json'
manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('=== V216 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "H100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")
    print(f"colab_url={COLAB_URL}")
    print(f"github_url={GITHUB_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

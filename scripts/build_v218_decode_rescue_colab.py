#!/usr/bin/env python3
"""Build the V218 decode-rescue diagnostic Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v218-decode-rescue/notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v218-decode-rescue/notebooks/KG1_V218_DECODE_RESCUE_COLAB.ipynb"
)
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
V194_ADAPTER_DRIVE = "/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter"
V194_VAL_CSV_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/"
    "validation/official_train_seed42_stratified10_val.csv"
)
V217_FINAL_ADAPTER_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/"
    "train_v217_shortans_lr1e8_s16/final_adapter"
)
V217_WEAK_REPORT_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/"
    "eval_v217_shortans_lr1e8_s16/weak_eval/v217_shortans_weak_eval_report.json"
)
V217_WEAK_PREDICTIONS_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/"
    "eval_v217_shortans_lr1e8_s16/weak_eval/v217_shortans_weak_predictions.csv"
)
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"
TRAIN_ROWS = 10206
VAL_ROWS = 681

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v218-{prefix}-{_CELL_COUNTER:02d}"


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
            f"""# KG1 V218 Decode Rescue Colab

Purpose: diagnose why V217 produced long/truncated weak-eval outputs, then test a cheap decode-only rescue before spending GPU on another training run.

This notebook:

- reuses the V217 final adapter already produced on Drive;
- audits V217 weak predictions if the CSV exists;
- runs a diagnostic weak eval with shorter generation and thinking disabled;
- blocks full eval/package unless the weak gate passes;
- never submits to Kaggle.

Colab URL:

`{COLAB_URL}`
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V218 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V218 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration and hard submit lock.
print('=== V218 CONFIG START ===', flush=True)
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

try:
    from google.colab import userdata
    for secret_name in ['HF_TOKEN', 'HF_KEY']:
        if not os.environ.get('HF_TOKEN'):
            secret_value = userdata.get(secret_name)
            if secret_value:
                os.environ['HF_TOKEN'] = secret_value
                os.environ['HUGGING_FACE_HUB_TOKEN'] = secret_value
                print('loaded Hugging Face token from Colab secret:', secret_name, flush=True)
except Exception as exc:
    print('Colab secret probe skipped:', type(exc).__name__, flush=True)

VERSION = 'V218_DECODE_RESCUE_20260508'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'v218-decode-rescue')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V218')
OUT_ROOT = DRIVE_ROOT / 'output_v218_decode_rescue'
EVAL_OUT = OUT_ROOT / 'eval_v218_decode_rescue'
ANALYSIS_OUT = OUT_ROOT / 'analysis_v217_weak'
PACKAGE_OUT = OUT_ROOT / 'package_v218_decode_rescue'
V194_ADAPTER = pathlib.Path('{V194_ADAPTER_DRIVE}')
V194_VAL_CSV = pathlib.Path('{V194_VAL_CSV_DRIVE}')
V217_FINAL_ADAPTER = pathlib.Path('{V217_FINAL_ADAPTER_DRIVE}')
V217_WEAK_REPORT = pathlib.Path('{V217_WEAK_REPORT_DRIVE}')
V217_WEAK_PREDICTIONS = pathlib.Path('{V217_WEAK_PREDICTIONS_DRIVE}')
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

EXPECTED_TRAIN_SHA256 = '{TRAIN_SHA}'
EXPECTED_VAL_SHA256 = '{VAL_SHA}'
MIN_TRAIN_EXAMPLES = {TRAIN_ROWS}
MIN_VAL_EXAMPLES = {VAL_ROWS}
EXPECTED_V194_ADAPTER_BYTES = 4259069440
EXPECTED_V194_ADAPTER_TENSOR_COUNT = 12011
EXPECTED_V217_ADAPTER_BYTES = 4259063856
EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT = 12000
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']

RUN_TRAIN = os.environ.get('KG1_V218_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
TOKENIZE_ONLY_DRY_RUN = 'not_applicable_decode_eval_only'
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
INIT_ADAPTER_DIR = V217_FINAL_ADAPTER
RUN_EVAL = os.environ.get('KG1_V218_RUN_EVAL', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
FORCE_REEVAL = os.environ.get('KG1_V218_FORCE_REEVAL', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
V218_VLLM_PIP_SPEC = os.environ.get('KG1_V218_VLLM_PIP_SPEC', 'vllm==0.20.1')
V218_MAX_TOKENS = int(os.environ.get('KG1_V218_MAX_TOKENS', '1024'))
V218_MAX_NUM_SEQS = int(os.environ.get('KG1_V218_MAX_NUM_SEQS', '64'))
V218_PROMPT_SUFFIX = os.environ.get('KG1_V218_PROMPT_SUFFIX', '\\nReturn only the final answer inside `\\\\boxed{{}}`. Do not explain.')
V218_DISABLE_THINKING = os.environ.get('KG1_V218_DISABLE_THINKING', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
runtime_dependency_names = ['causal_conv1d', 'mamba_ssm', 'vllm']

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
WEAK_MAX_TRUNC_FOR_FULL = 0
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, EVAL_OUT, ANALYSIS_OUT, PACKAGE_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V194_VAL_CSV =', V194_VAL_CSV, flush=True)
print('V217_FINAL_ADAPTER =', V217_FINAL_ADAPTER, flush=True)
print('V217_WEAK_REPORT =', V217_WEAK_REPORT, flush=True)
print('V217_WEAK_PREDICTIONS =', V217_WEAK_PREDICTIONS, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('EXPECTED_V194_ADAPTER_BYTES =', EXPECTED_V194_ADAPTER_BYTES, flush=True)
print('EXPECTED_V194_ADAPTER_TENSOR_COUNT =', EXPECTED_V194_ADAPTER_TENSOR_COUNT, flush=True)
print('EXPECTED_V217_ADAPTER_BYTES =', EXPECTED_V217_ADAPTER_BYTES, flush=True)
print('EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT =', EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('FORCE_REEVAL =', FORCE_REEVAL, flush=True)
print('V218_MAX_TOKENS =', V218_MAX_TOKENS, flush=True)
print('V218_DISABLE_THINKING =', V218_DISABLE_THINKING, flush=True)
print('runtime_dependency_names =', runtime_dependency_names, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')
print('=== V218 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging and heartbeat.
print('=== V218 HELPERS START ===', flush=True)

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
                meminfo[key] = int(raw.strip().split()[0]) * 1024
        parts.append(f"ram_total={meminfo.get('MemTotal', 0)/1024**3:.1f}GiB")
        parts.append(f"ram_available={meminfo.get('MemAvailable', 0)/1024**3:.1f}GiB")
    except Exception as exc:
        parts.append(f'ram_probe_error={type(exc).__name__}')
    try:
        usage = shutil.disk_usage('/content')
        parts.append(f'disk_content_free={usage.free/1024**3:.1f}GiB')
        parts.append(f'disk_content_total={usage.total/1024**3:.1f}GiB')
    except Exception as exc:
        parts.append(f'disk_probe_error={type(exc).__name__}')
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            parts.append('gpu=[' + proc.stdout.strip().replace('\\n', '; ') + ']')
        else:
            parts.append('gpu_probe_rc=' + str(proc.returncode))
    except Exception as exc:
        parts.append(f'gpu_probe_error={type(exc).__name__}')
    return ' '.join(parts)

def run_cmd(cmd, cwd=None, env=None, log_path=None, check=True, heartbeat_s=60):
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or os.getcwd(), flush=True)
    print('+', ' '.join(map(str, cmd)), flush=True)
    if log_path:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print('log_path =', log_path, flush=True)
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    last_heartbeat = time.time()
    start = time.time()
    with (open(log_path, 'w', encoding='utf-8') if log_path else open(os.devnull, 'w', encoding='utf-8')) as log_handle:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip('\\n'))
            log_handle.write(line)
            log_handle.flush()
            if len(lines) <= 200:
                print(line, end='', flush=True)
            now = time.time()
            if now - last_heartbeat >= heartbeat_s:
                print(f'[V218 heartbeat] elapsed_s={now-start:.1f} {resource_snapshot_line()}', flush=True)
                last_heartbeat = now
    rc = proc.wait()
    elapsed = time.time() - start
    print('returncode =', rc, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if len(lines) > 200:
        print('command_output_suppressed_lines =', len(lines) - 200, flush=True)
    if rc != 0:
        print('command_tail_on_failure =', flush=True)
        for line in lines[-40:]:
            print(line, flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and rc != 0:
        raise RuntimeError(f'Command failed rc={rc}: {cmd}')
    return rc

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def verify_import_subprocess(import_name, check=True):
    return run_cmd(
        [sys.executable, '-c', f"import importlib; m=importlib.import_module({import_name!r}); print({import_name!r} + ' subprocess_version=' + str(getattr(m, '__version__', 'unknown')))"],
        log_path=OUT_ROOT / f'verify_import_{import_name}.log',
        check=check,
    )

def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.exists() and (path / 'adapter_config.json').exists() and (path / 'adapter_model.safetensors').exists()

def ensure_vllm_for_eval():
    print('=== V218 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    if verify_import_subprocess('vllm', check=False) != 0:
        print('vLLM subprocess import failed; installing pinned V218_VLLM_PIP_SPEC =', V218_VLLM_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', V218_VLLM_PIP_SPEC], log_path=OUT_ROOT / 'pip_install_vllm.log', check=True)
        verify_import_subprocess('vllm', check=True)
    print('=== V218 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)

print('python =', sys.version, flush=True)
print('=== V218 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone branch, compile scripts, and verify V217 data hashes.
print('=== V218 REPO SETUP START ===', flush=True)
if ROOT.exists():
    print('removing existing ROOT before fresh clone:', ROOT, flush=True)
    shutil.rmtree(ROOT)
run_cmd(
    ['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, ROOT],
    log_path=OUT_ROOT / 'repo_clone.log',
    check=True,
)
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
print('repo_commit =', commit, flush=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('repo_root_on_sys_path =', str(ROOT) in sys.path, flush=True)
for rel in [
    'scripts/evaluate_lora_adapter.py',
    'scripts/analyze_eval_predictions.py',
    'scripts/notebook_release_gate.py',
    'src/competition_utils.py',
]:
    py_path = ROOT / rel
    print('compile_check =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)

train_path = ROOT / 'data/v217/v217_short_answer_train.jsonl'
val_path = ROOT / 'data/v217/v217_short_answer_val.jsonl'
print('train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('val_path =', val_path, 'exists =', val_path.exists(), flush=True)
observed_train_sha256 = sha256_file(train_path)
observed_val_sha256 = sha256_file(val_path)
print('observed_train_sha256 =', observed_train_sha256, flush=True)
print('observed_val_sha256 =', observed_val_sha256, flush=True)
if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
    raise RuntimeError('V217 train SHA mismatch')
if observed_val_sha256 != EXPECTED_VAL_SHA256:
    raise RuntimeError('V217 val SHA mismatch')
train_rows_observed = sum(1 for _ in train_path.open(encoding='utf-8'))
val_rows_observed = sum(1 for _ in val_path.open(encoding='utf-8'))
print('train_rows_observed =', train_rows_observed, flush=True)
print('val_rows_observed =', val_rows_observed, flush=True)
if train_rows_observed < MIN_TRAIN_EXAMPLES:
    raise RuntimeError('train row count below MIN_TRAIN_EXAMPLES')
if val_rows_observed < MIN_VAL_EXAMPLES:
    raise RuntimeError('val row count below MIN_VAL_EXAMPLES')
print('=== V218 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, adapter, and validation CSV audit.
print('=== V218 RUNTIME AND DATA AUDIT START ===', flush=True)
import pandas as pd

run_cmd(
    [
        sys.executable,
        '-c',
        "import json, torch; props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), 'cuda_available': torch.cuda.is_available(), 'gpu_name': props.name if props else '', 'gpu_total_gib': props.total_memory/1024**3 if props else 0.0}))",
    ],
    log_path=OUT_ROOT / 'verify_torch_cuda.jsonl',
    check=True,
)
usage = shutil.disk_usage('/content')
content_free_gib = usage.free / 1024**3
print('content_free_gib =', round(content_free_gib, 2), flush=True)
print('causal_conv1d audit note: eval-only notebook does not compile causal_conv1d.', flush=True)
print('mamba_ssm audit note: vLLM handles Nemotron-H runtime path for eval.', flush=True)

print('V194_ADAPTER complete =', is_complete_adapter_dir(V194_ADAPTER), flush=True)
print('V217_FINAL_ADAPTER complete =', is_complete_adapter_dir(V217_FINAL_ADAPTER), flush=True)
if not V194_VAL_CSV.exists():
    raise FileNotFoundError(V194_VAL_CSV)
if not is_complete_adapter_dir(V217_FINAL_ADAPTER):
    raise RuntimeError('V217 final adapter is missing or incomplete; run V217 train first.')
try:
    from safetensors import safe_open
except ModuleNotFoundError:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], log_path=OUT_ROOT / 'pip_install_safetensors.log', check=True)
    from safetensors import safe_open
for adapter_name, adapter_path in [('v194', V194_ADAPTER), ('v217_final', V217_FINAL_ADAPTER)]:
    config_path = adapter_path / 'adapter_config.json'
    weights_path = adapter_path / 'adapter_model.safetensors'
    config = read_json(config_path)
    target_modules = config.get('target_modules')
    target_parameters = config.get('target_parameters')
    weight_bytes = weights_path.stat().st_size
    with safe_open(weights_path, framework='pt', device='cpu') as handle:
        tensor_count = len(handle.keys())
    print(adapter_name + '_adapter_config.json =', config_path, flush=True)
    print(adapter_name + '_adapter_model.safetensors =', weights_path, 'bytes =', weight_bytes, 'tensor_count =', tensor_count, flush=True)
    print(adapter_name + '_target_modules =', target_modules, flush=True)
    print(adapter_name + '_target_parameters =', target_parameters, flush=True)
    if set(target_modules or []) != set(EXPECTED_V194_TARGET_MODULES):
        raise RuntimeError(adapter_name + ' target_modules mismatch')
    if list(target_parameters or []) != EXPECTED_V194_TARGET_PARAMETERS:
        raise RuntimeError(adapter_name + ' target_parameters mismatch')
    if adapter_name == 'v194':
        if weight_bytes != EXPECTED_V194_ADAPTER_BYTES:
            raise RuntimeError('V194 adapter size mismatch')
        if tensor_count != EXPECTED_V194_ADAPTER_TENSOR_COUNT:
            raise RuntimeError('V194 adapter tensor count mismatch')
    if adapter_name == 'v217_final':
        if weight_bytes != EXPECTED_V217_ADAPTER_BYTES:
            raise RuntimeError('V217 final adapter size mismatch')
        if tensor_count < EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT:
            raise RuntimeError('V217 final adapter tensor count below expected floor')

from src.competition_utils import classify_puzzle
full_df = pd.read_csv(V194_VAL_CSV)
if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
    raise RuntimeError('V194 validation CSV must contain prompt and answer')
if 'type' not in full_df.columns:
    full_df['type'] = full_df['prompt'].map(classify_puzzle)
full_eval_csv = EVAL_OUT / 'v218_full_947.csv'
weak_eval_csv = EVAL_OUT / 'v218_weak_315.csv'
strong_eval_csv = EVAL_OUT / 'v218_strong_632.csv'
weak_df = full_df[full_df['type'].isin(['equation_transform', 'bit_manipulation'])].copy()
strong_df = full_df[~full_df['type'].isin(['equation_transform', 'bit_manipulation'])].copy()
full_df.to_csv(full_eval_csv, index=False)
weak_df.to_csv(weak_eval_csv, index=False)
strong_df.to_csv(strong_eval_csv, index=False)
print('full_rows =', len(full_df), 'path =', full_eval_csv, flush=True)
print('weak_rows =', len(weak_df), 'path =', weak_eval_csv, flush=True)
print('strong_rows =', len(strong_df), 'path =', strong_eval_csv, flush=True)
print('per_family_counts =', full_df['type'].value_counts().sort_index().to_dict(), flush=True)
if len(weak_df) != 315 or len(full_df) != 947:
    raise RuntimeError('validation split row count mismatch')
print('=== V218 RUNTIME AND DATA AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: analyze V217 weak prediction failure.
print('=== V218 V217 WEAK FORENSIC START ===', flush=True)
v217_report = None
if V217_WEAK_REPORT.exists():
    v217_report = read_json(V217_WEAK_REPORT)
    print('v217_weak_report =', json.dumps(v217_report, indent=2, sort_keys=True), flush=True)
else:
    print('V217 weak report missing:', V217_WEAK_REPORT, flush=True)
if V217_WEAK_PREDICTIONS.exists():
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/analyze_eval_predictions.py'),
            '--predictions-csv',
            str(V217_WEAK_PREDICTIONS),
            '--output-dir',
            str(ANALYSIS_OUT),
            '--label',
            'v217_weak_prediction_forensic',
        ],
        cwd=ROOT,
        log_path=ANALYSIS_OUT / 'v217_weak_prediction_forensic.log',
        check=True,
    )
    print('v217 weak forensic returncode =', rc, flush=True)
else:
    print('V217 weak predictions missing:', V217_WEAK_PREDICTIONS, flush=True)
print('=== V218 V217 WEAK FORENSIC END ===', flush=True)
"""
        ),
        code(
            """# CELL: V218 decode-rescue weak eval.
print('=== V218 DECODE WEAK EVAL START ===', flush=True)
decode_report = None
decode_per_task = None
weak_gate_pass_for_full = False
decode_eval_dir = EVAL_OUT / f'weak_eval_decode_mtok{V218_MAX_TOKENS}_think{int(not V218_DISABLE_THINKING)}'
decode_label = f'v218_decode_mtok{V218_MAX_TOKENS}_weak'
decode_report_path = decode_eval_dir / f'{decode_label}_eval_report.json'
decode_per_task_path = decode_eval_dir / f'{decode_label}_per_task.csv'
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping V218 decode eval.', flush=True)
else:
    ensure_vllm_for_eval()
    decode_eval_dir.mkdir(parents=True, exist_ok=True)
    run_new_decode_eval = FORCE_REEVAL or not decode_report_path.exists()
    print('run_new_decode_eval =', run_new_decode_eval, flush=True)
    if run_new_decode_eval:
        cmd = [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(weak_eval_csv),
            '--questions-csv', str(weak_eval_csv),
            '--adapter', str(V217_FINAL_ADAPTER),
            '--base-model-path', MODEL_NAME,
            '--label', decode_label,
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(decode_eval_dir),
            '--max-tokens', str(V218_MAX_TOKENS),
            '--max-num-seqs', str(V218_MAX_NUM_SEQS),
            '--prompt-suffix', V218_PROMPT_SUFFIX,
        ]
        if V218_DISABLE_THINKING:
            cmd.append('--disable-thinking')
        rc = run_cmd(cmd, cwd=ROOT, log_path=decode_eval_dir / 'weak_decode_eval.log', check=True)
        print('decode weak eval returncode =', rc, flush=True)
    else:
        print('reusing existing decode_report_path =', decode_report_path, flush=True)
    decode_report = read_json(decode_report_path)
    decode_per_task = pd.read_csv(decode_per_task_path)
    print('decode_report =', json.dumps(decode_report, indent=2, sort_keys=True), flush=True)
    print('decode_per_task =', decode_per_task.to_string(index=False), flush=True)
    decode_by_task = {row['task_type']: int(row['correct']) for _, row in decode_per_task.iterrows()}
    decode_correct = int(decode_report.get('correct', 0))
    decode_eq_correct = decode_by_task.get('equation_transform', 0)
    decode_bit_correct = decode_by_task.get('bit_manipulation', 0)
    decode_truncated = int(decode_report.get('truncated', 999))
    weak_gate_pass_for_full = (
        decode_correct >= WEAK_MIN_FOR_FULL
        and decode_eq_correct >= WEAK_EQ_MIN_FOR_FULL
        and decode_bit_correct >= WEAK_BIT_MIN_FOR_FULL
        and decode_truncated <= WEAK_MAX_TRUNC_FOR_FULL
    )
    v217_correct = int(v217_report.get('correct', 0)) if v217_report else 0
    v217_truncated = int(v217_report.get('truncated', 999)) if v217_report else 999
    decode_improves_v217 = decode_correct > v217_correct and decode_truncated < v217_truncated
    print('decode_correct =', decode_correct, flush=True)
    print('decode_eq_correct =', decode_eq_correct, flush=True)
    print('decode_bit_correct =', decode_bit_correct, flush=True)
    print('decode_truncated =', decode_truncated, flush=True)
    print('decode_improves_v217 =', decode_improves_v217, flush=True)
    print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V218 DECODE WEAK EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval is blocked unless decode weak gate passes.
print('=== V218 FULL EVAL GATE START ===', flush=True)
full_report = None
full_candidate_gate = False
if not weak_gate_pass_for_full:
    print('Decode weak gate failed or did not run; full eval blocked.', flush=True)
    print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
else:
    print('Decode weak gate passed. Full eval is intentionally not automatic in V218 diagnostic notebook.', flush=True)
print('=== V218 FULL EVAL GATE END ===', flush=True)
"""
        ),
        code(
            """# CELL: write V218 final manifest. No submit.
print('=== V218 FINAL MANIFEST START ===', flush=True)
decision = {
    'version': VERSION,
    'repo_commit': globals().get('commit', ''),
    'run_train': RUN_TRAIN,
    'run_eval': RUN_EVAL,
    'v217_report': v217_report,
    'decode_report': decode_report,
    'weak_gate_pass_for_full': bool(weak_gate_pass_for_full),
    'full_candidate_gate': bool(full_candidate_gate),
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
}
if not weak_gate_pass_for_full:
    decision['roadmap_next'] = 'If decode does not materially improve V217, build a new training dataset/prompt rather than full-eval or submit.'
manifest_path = OUT_ROOT / 'v218_decode_rescue_manifest.json'
manifest_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('=== V218 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "H100",
                "machine_shape": "hm",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")
    print(f"colab_url={COLAB_URL}")
    print(f"github_url={GITHUB_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

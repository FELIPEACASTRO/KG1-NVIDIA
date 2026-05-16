#!/usr/bin/env python3
"""Build the V226 equation checkpoint sweep Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V226_EQUATION_CHECKPOINT_SWEEP_COLAB.ipynb")
BRANCH = "v226-equation-checkpoint-sweep"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V226_EQUATION_CHECKPOINT_SWEEP_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V226_EQUATION_CHECKPOINT_SWEEP_COLAB.ipynb"
)
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v226-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return (
        source.replace("__BRANCH__", BRANCH)
        .replace("__COLAB_URL__", COLAB_URL)
        .replace("__GITHUB_URL__", GITHUB_URL)
        .replace("__TRAIN_SHA__", TRAIN_SHA)
        .replace("__VAL_SHA__", VAL_SHA)
    )


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md"),
        "metadata": {},
        "source": _subst(source).splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _subst(source).splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """# KG1 V226 Equation Checkpoint Sweep Colab

Purpose: move past the V225 prompt ceiling by evaluating existing V223 checkpoints and, when enabled, creating an ultra-conservative V226 micro-continuation from the protected V194 adapter.

The objective is explicit: find a weak candidate with at least `193/315` total, `60/155` equation_transform, `133/160` bit_manipulation, and no more than `3` truncations. Full eval, packaging, and Kaggle submit remain blocked in this notebook.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V226 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V226 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration, gates, and hard submit lock.
print('=== V226 CONFIG START ===', flush=True)
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
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

VERSION = 'V226_EQUATION_CHECKPOINT_SWEEP_20260509'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V226')
OUT_ROOT = DRIVE_ROOT / 'output_v226_equation_checkpoint_sweep'
TRAIN_OUT = OUT_ROOT / 'train_v226_v194_micro_lr2e9_s6'
EVAL_OUT = OUT_ROOT / 'eval_v226_checkpoint_sweep'
ANALYSIS_OUT = OUT_ROOT / 'analysis_v226_checkpoint_sweep'

MODEL_NAME = os.environ.get('KG1_V226_MODEL_NAME', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
V226_VLLM_PIP_SPEC = os.environ.get('KG1_V226_VLLM_PIP_SPEC', 'vllm==0.20.1')
V226_CAUSAL_CONV1D_PIP_SPEC = os.environ.get('KG1_V226_CAUSAL_CONV1D_PIP_SPEC', 'causal-conv1d==1.6.1')
V226_MAMBA_SSM_PIP_SPEC = os.environ.get('KG1_V226_MAMBA_SSM_PIP_SPEC', 'mamba-ssm==2.3.1')
V226_MAX_MODEL_LEN = int(os.environ.get('KG1_V226_MAX_MODEL_LEN', '8192'))
V226_MAX_NUM_SEQS = int(os.environ.get('KG1_V226_MAX_NUM_SEQS', '64'))
V226_MAX_TOKENS = int(os.environ.get('KG1_V226_MAX_TOKENS', '7680'))
V226_WARMUP_ROWS = int(os.environ.get('KG1_V226_WARMUP_ROWS', '0'))
V226_PROMPT_SUFFIX = os.environ.get('KG1_V226_PROMPT_SUFFIX', '\\nReturn exactly one line in this format: `\\\\boxed{answer}`.')

V221_EVAL_OUT = pathlib.Path(os.environ.get('KG1_V226_V221_EVAL_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab'))
V221_WEAK_CSV = pathlib.Path(os.environ.get('KG1_V226_V221_WEAK_CSV', str(V221_EVAL_OUT / 'v221_weak_315.csv')))
V221_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get('KG1_V226_V221_BATCH_SUMMARY_JSON', str(V221_EVAL_OUT / 'batch_candidate_summary.json')))
V225_FINAL_MANIFEST = pathlib.Path(os.environ.get('KG1_V226_V225_FINAL_MANIFEST', '/content/drive/MyDrive/KG1_NVIDIA_V225/output_v225_equation_decode_sweep/v225_equation_decode_sweep_final_manifest.json'))
V223_TRAIN_OUT = pathlib.Path(os.environ.get('KG1_V226_V223_TRAIN_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V223/output_v223_equation_rescue/train_v223_eqrescue_from_v217_lr1e8_s12'))

V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
V217_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter')
INIT_ADAPTER_DIR = pathlib.Path(os.environ.get('KG1_V226_INIT_ADAPTER', str(V194_ADAPTER)))
EXPECTED_TRAIN_SHA256 = '__TRAIN_SHA__'
EXPECTED_VAL_SHA256 = '__VAL_SHA__'
MIN_TRAIN_EXAMPLES = 10206
MIN_VAL_EXAMPLES = 681
TOKENIZE_ONLY_DRY_RUN = True
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True

EXPECTED_V194_ADAPTER_BYTES = 4259069440
EXPECTED_V194_ADAPTER_TENSOR_COUNT = 12011
MIN_V217_ADAPTER_BYTES = 4250000000
MIN_V217_ADAPTER_TENSOR_COUNT = 12000
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']

RUN_TRAIN = os.environ.get('KG1_V226_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_EVAL = os.environ.get('KG1_V226_RUN_EVAL', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_ANALYSIS = os.environ.get('KG1_V226_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_FULL_IF_GATE = False
FORCE_RETRAIN = os.environ.get('KG1_V226_FORCE_RETRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
FORCE_REEVAL = os.environ.get('KG1_V226_FORCE_REEVAL', '0').strip().lower() in {'1', 'true', 'yes', 'on'}

V226_LR = os.environ.get('KG1_V226_LR', '2e-9')
V226_FINAL_LR = os.environ.get('KG1_V226_FINAL_LR', '5e-10')
V226_MAX_STEPS = os.environ.get('KG1_V226_MAX_STEPS', '6')
V226_TRAINABLE_MODULES = os.environ.get('KG1_V226_TRAINABLE_MODULES', 'lm_head,o_proj,q_proj,k_proj')
V226_MAX_LENGTH = int(os.environ.get('KG1_V226_MAX_LENGTH', '4096'))
V226_BATCH_SIZE = int(os.environ.get('KG1_V226_BATCH_SIZE', '4'))
V226_MICRO_BATCH_SIZE = int(os.environ.get('KG1_V226_MICRO_BATCH_SIZE', '1'))
V226_MAX_CANDIDATES = int(os.environ.get('KG1_V226_MAX_CANDIDATES', '32'))

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
WEAK_MAX_TRUNC_FOR_FULL = 0
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, TRAIN_OUT, EVAL_OUT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('TRAIN_OUT =', TRAIN_OUT, flush=True)
print('EVAL_OUT =', EVAL_OUT, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('MODEL_NAME =', MODEL_NAME, flush=True)
print('V226_MAX_TOKENS =', V226_MAX_TOKENS, flush=True)
print('V226_PROMPT_SUFFIX =', repr(V226_PROMPT_SUFFIX), flush=True)
print('V221_WEAK_CSV =', V221_WEAK_CSV, flush=True)
print('V221_BATCH_SUMMARY_JSON =', V221_BATCH_SUMMARY_JSON, flush=True)
print('V225_FINAL_MANIFEST =', V225_FINAL_MANIFEST, flush=True)
print('V223_TRAIN_OUT =', V223_TRAIN_OUT, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V217_ADAPTER =', V217_ADAPTER, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('FORCE_RETRAIN =', FORCE_RETRAIN, flush=True)
print('FORCE_REEVAL =', FORCE_REEVAL, flush=True)
print('V226_LR =', V226_LR, flush=True)
print('V226_FINAL_LR =', V226_FINAL_LR, flush=True)
print('V226_MAX_STEPS =', V226_MAX_STEPS, flush=True)
print('V226_TRAINABLE_MODULES =', V226_TRAINABLE_MODULES, flush=True)
print('V226_MAX_CANDIDATES =', V226_MAX_CANDIDATES, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V226.')
print('=== V226 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging, heartbeat, hashes, and dependency installers.
print('=== V226 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def resource_snapshot_line():
    parts = []
    try:
        import psutil
        mem = psutil.virtual_memory()
        parts.append(f'ram_total={mem.total/1024**3:.1f}GiB')
        parts.append(f'ram_available={mem.available/1024**3:.1f}GiB')
    except Exception as exc:
        parts.append(f'ram_probe_error={type(exc).__name__}')
    try:
        usage = shutil.disk_usage('/content')
        parts.append(f'disk_content_free={usage.free/1024**3:.1f}GiB')
        parts.append(f'disk_content_total={usage.total/1024**3:.1f}GiB')
    except Exception as exc:
        parts.append(f'disk_probe_error={type(exc).__name__}')
    try:
        gpu_line = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip().splitlines()
        if gpu_line:
            parts.append('gpu=[' + gpu_line[0] + ']')
    except Exception as exc:
        parts.append(f'gpu_probe_error={type(exc).__name__}')
    return ' '.join(parts)

def run_cmd(cmd, cwd=None, log_path=None, check=True, heartbeat_s=0, suppress_after_lines=260):
    started = time.time()
    printable = ' '.join(str(x) for x in cmd)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or os.getcwd(), flush=True)
    print('+', printable, flush=True)
    log_handle = None
    if log_path is not None:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8', errors='replace')
        print('log_path =', log_path, flush=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    last_heartbeat = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip('\\n'))
        if log_handle:
            log_handle.write(line)
            log_handle.flush()
        if len(lines) <= suppress_after_lines:
            print(line, end='', flush=True)
        now = time.time()
        if heartbeat_s and now - last_heartbeat >= heartbeat_s:
            print('[V226 heartbeat] elapsed_s={:.1f} {}'.format(now - started, resource_snapshot_line()), flush=True)
            last_heartbeat = now
    returncode = proc.wait()
    if log_handle:
        log_handle.close()
    if len(lines) > suppress_after_lines:
        print('command_output_suppressed_lines =', len(lines) - suppress_after_lines, flush=True)
    elapsed = time.time() - started
    print('returncode =', returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if returncode != 0:
        print('command_tail_on_failure =', '\\n'.join(lines[-60:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and returncode != 0:
        raise RuntimeError(f'command failed rc={returncode}: {printable}')
    return returncode

def verify_import(module_name, log_name):
    cmd = [
        sys.executable,
        '-c',
        "import importlib; m=importlib.import_module(%r); print(%r + ' subprocess_version=' + str(getattr(m, '__version__', 'unknown')))" % (module_name, module_name),
    ]
    return run_cmd(cmd, cwd='/content', log_path=OUT_ROOT / log_name, check=False)

def ensure_train_dependencies():
    print('=== V226 TRAIN DEPENDENCY CHECK START ===', flush=True)
    if verify_import('causal_conv1d', 'verify_import_causal_conv1d.log') != 0:
        print('installing causal_conv1d =', V226_CAUSAL_CONV1D_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', V226_CAUSAL_CONV1D_PIP_SPEC], cwd='/content', log_path=OUT_ROOT / 'pip_install_causal_conv1d.log', check=True, heartbeat_s=60)
    if verify_import('mamba_ssm', 'verify_import_mamba_ssm.log') != 0:
        print('installing mamba_ssm =', V226_MAMBA_SSM_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '--progress-bar', 'off', '--no-build-isolation', V226_MAMBA_SSM_PIP_SPEC], cwd='/content', log_path=OUT_ROOT / 'pip_install_mamba_ssm.log', check=True, heartbeat_s=60)
    print('=== V226 TRAIN DEPENDENCY CHECK END ===', flush=True)

def ensure_vllm_for_eval():
    print('=== V226 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    if verify_import('vllm', 'verify_import_vllm.log') != 0:
        print('vLLM subprocess import failed; installing pinned V226_VLLM_PIP_SPEC =', V226_VLLM_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', V226_VLLM_PIP_SPEC], cwd='/content', log_path=OUT_ROOT / 'pip_install_vllm.log', check=True, heartbeat_s=60)
        if verify_import('vllm', 'verify_import_vllm.log') != 0:
            raise RuntimeError('vLLM import still failed after install.')
    print('=== V226 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)

def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.is_dir() and (path / 'adapter_config.json').exists() and (
        (path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists()
    )

print('=== V226 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and validate static data hashes.
print('=== V226 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
print('repo_commit =', repo_commit, flush=True)
compile_targets = [
    ROOT / 'src/competition_utils.py',
    ROOT / 'scripts/evaluate_lora_adapter.py',
    ROOT / 'scripts/evaluate_lora_adapters_batch.py',
    ROOT / 'scripts/hf_job_train_v90.py',
    ROOT / 'scripts/analyze_v226_checkpoint_sweep.py',
    ROOT / 'scripts/notebook_release_gate.py',
]
for py_path in compile_targets:
    print('compile_target =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)
    print('py_compile ok =', py_path.relative_to(ROOT), flush=True)
train_path = ROOT / 'data/v217/v217_short_answer_train.jsonl'
val_path = ROOT / 'data/v217/v217_short_answer_val.jsonl'
print('train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('val_path =', val_path, 'exists =', val_path.exists(), flush=True)
observed_train_sha256 = sha256_file(train_path)
observed_val_sha256 = sha256_file(val_path)
train_rows = sum(1 for _ in train_path.open('r', encoding='utf-8'))
val_rows = sum(1 for _ in val_path.open('r', encoding='utf-8'))
print('observed_train_sha256 =', observed_train_sha256, flush=True)
print('observed_val_sha256 =', observed_val_sha256, flush=True)
print('train_rows =', train_rows, flush=True)
print('val_rows =', val_rows, flush=True)
if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
    raise RuntimeError(f'train sha mismatch: {observed_train_sha256} != {EXPECTED_TRAIN_SHA256}')
if observed_val_sha256 != EXPECTED_VAL_SHA256:
    raise RuntimeError(f'val sha mismatch: {observed_val_sha256} != {EXPECTED_VAL_SHA256}')
if train_rows < MIN_TRAIN_EXAMPLES:
    raise RuntimeError(f'train row count too low: {train_rows} < {MIN_TRAIN_EXAMPLES}')
if val_rows < MIN_VAL_EXAMPLES:
    raise RuntimeError(f'val row count too low: {val_rows} < {MIN_VAL_EXAMPLES}')
print('=== V226 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, Drive artifact, adapter, and dependency audit.
print('=== V226 RUNTIME ARTIFACT AUDIT START ===', flush=True)
torch_probe_path = OUT_ROOT / 'verify_torch_cuda.jsonl'
torch_code = "import json, torch; props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), 'cuda_available': torch.cuda.is_available(), 'gpu_name': props.name if props else '', 'gpu_total_gib': props.total_memory/1024**3 if props else 0.0}))"
run_cmd([sys.executable, '-c', torch_code], cwd='/content', log_path=torch_probe_path, check=True)
torch_probe = json.loads([line for line in torch_probe_path.read_text(encoding='utf-8').splitlines() if line.strip()][-1])
cuda_available = bool(torch_probe.get('cuda_available'))
gpu_name = str(torch_probe.get('gpu_name', ''))
gpu_total_gib = float(torch_probe.get('gpu_total_gib', 0.0))
content_free_gib = shutil.disk_usage('/content').free / 1024**3
print('cuda_available =', cuda_available, flush=True)
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if not cuda_available:
    raise RuntimeError('CUDA is required for V226 checkpoint sweep.')
if gpu_total_gib < 70:
    raise RuntimeError(f'GPU memory too small for V226 sweep: {gpu_total_gib:.2f} GiB')
if content_free_gib < 40:
    raise RuntimeError(f'/content free disk too low: {content_free_gib:.2f} GiB')
for module_name in ['causal_conv1d', 'mamba_ssm']:
    try:
        module = importlib.import_module(module_name)
        print(module_name, 'version =', getattr(module, '__version__', 'unknown'), flush=True)
    except Exception as exc:
        print(module_name, 'import_warning =', repr(exc), flush=True)
print('V221_WEAK_CSV exists =', V221_WEAK_CSV.exists(), flush=True)
print('V221_BATCH_SUMMARY_JSON exists =', V221_BATCH_SUMMARY_JSON.exists(), flush=True)
print('V225_FINAL_MANIFEST exists =', V225_FINAL_MANIFEST.exists(), flush=True)
print('V223_TRAIN_OUT exists =', V223_TRAIN_OUT.exists(), flush=True)
for required_path in [V221_WEAK_CSV, V221_BATCH_SUMMARY_JSON]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)
try:
    from safetensors import safe_open
except Exception:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], cwd='/content', log_path=OUT_ROOT / 'pip_install_safetensors.log', check=True)
    from safetensors import safe_open
for label, adapter_path in [('V194', V194_ADAPTER), ('V217', V217_ADAPTER), ('INIT', INIT_ADAPTER_DIR)]:
    print(label, 'adapter path =', adapter_path, 'complete =', is_complete_adapter_dir(adapter_path), flush=True)
    if not is_complete_adapter_dir(adapter_path):
        raise RuntimeError(f'{label} adapter incomplete: {adapter_path}')
    cfg = read_json(adapter_path / 'adapter_config.json')
    print(label, 'target_modules =', cfg.get('target_modules'), flush=True)
    print(label, 'target_parameters =', cfg.get('target_parameters'), flush=True)
    if sorted(cfg.get('target_modules') or []) != sorted(EXPECTED_V194_TARGET_MODULES):
        raise RuntimeError(f'{label} target_modules mismatch')
    if sorted(cfg.get('target_parameters') or []) != sorted(EXPECTED_V194_TARGET_PARAMETERS):
        raise RuntimeError(f'{label} target_parameters mismatch')
    weights_path = adapter_path / 'adapter_model.safetensors'
    with safe_open(str(weights_path), framework='pt', device='cpu') as handle:
        tensor_count = len(handle.keys())
    print(label, 'adapter_tensor_count =', tensor_count, flush=True)
    print(label, 'adapter_weight_bytes =', weights_path.stat().st_size, flush=True)
    if label == 'V194':
        if tensor_count != EXPECTED_V194_ADAPTER_TENSOR_COUNT:
            raise RuntimeError('V194 adapter tensor count mismatch')
        if weights_path.stat().st_size != EXPECTED_V194_ADAPTER_BYTES:
            raise RuntimeError('V194 adapter weight size mismatch')
    if label == 'V217':
        if tensor_count < MIN_V217_ADAPTER_TENSOR_COUNT:
            raise RuntimeError('V217 final adapter tensor count below expected floor')
        if weights_path.stat().st_size < MIN_V217_ADAPTER_BYTES:
            raise RuntimeError('V217 final adapter size mismatch')
print('=== V226 RUNTIME ARTIFACT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: prepare weak eval CSV and baseline checkpoint metadata.
print('=== V226 WEAK DATA PREP START ===', flush=True)
import pandas as pd
weak_df = pd.read_csv(V221_WEAK_CSV)
if 'id' not in weak_df.columns:
    raise RuntimeError('V221 weak CSV must include id column.')
if 'prompt' not in weak_df.columns:
    raise RuntimeError('V221 weak CSV must include prompt column.')
family_col = 'type' if 'type' in weak_df.columns else 'family'
if family_col not in weak_df.columns:
    raise RuntimeError('V221 weak CSV must include type/family column.')
equation_df = weak_df[weak_df[family_col].astype(str).eq('equation_transform')].copy()
bit_df = weak_df[weak_df[family_col].astype(str).eq('bit_manipulation')].copy()
print('weak_rows =', len(weak_df), flush=True)
print('equation_rows =', len(equation_df), flush=True)
print('bit_rows =', len(bit_df), flush=True)
if len(weak_df) != 315:
    raise RuntimeError(f'unexpected weak row count: {len(weak_df)} != 315')
if len(equation_df) != 155:
    raise RuntimeError(f'unexpected equation row count: {len(equation_df)} != 155')
if len(bit_df) != 160:
    raise RuntimeError(f'unexpected bit row count: {len(bit_df)} != 160')
WEAK_EVAL_CSV = EVAL_OUT / 'v226_weak_315.csv'
weak_df.to_csv(WEAK_EVAL_CSV, index=False)
if V225_FINAL_MANIFEST.exists():
    v225_manifest = read_json(V225_FINAL_MANIFEST)
    print('v225_final_decision =', json.dumps(v225_manifest.get('decision', {}), sort_keys=True), flush=True)
else:
    print('v225_final_decision = missing manifest; continuing with V226 checkpoint sweep.', flush=True)
print('weak_eval_csv =', WEAK_EVAL_CSV, flush=True)
print('=== V226 WEAK DATA PREP END ===', flush=True)
"""
        ),
        code(
            """# CELL: enable V226 micro training.
print('=== V226 ENABLE TRAIN START ===', flush=True)
os.environ['KG1_V226_RUN_TRAIN'] = '1'
RUN_TRAIN = os.environ.get('KG1_V226_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
print('KG1_V226_RUN_TRAIN =', os.environ.get('KG1_V226_RUN_TRAIN'), flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
if not RUN_TRAIN:
    raise RuntimeError('RUN_TRAIN should be True after enabling KG1_V226_RUN_TRAIN=1.')
print('=== V226 ENABLE TRAIN END ===', flush=True)
"""
        ),
        code(
            """# CELL: optional V226 protected V194 micro-continuation training.
print('=== V226 TRAIN START ===', flush=True)
if not RUN_TRAIN:
    print('RUN_TRAIN is false; skipping V226 micro training and evaluating existing checkpoints only.', flush=True)
else:
    final_adapter = TRAIN_OUT / 'final_adapter'
    run_new_train = FORCE_RETRAIN or not is_complete_adapter_dir(final_adapter)
    print('train_out =', TRAIN_OUT, flush=True)
    print('final_adapter =', final_adapter, 'complete =', is_complete_adapter_dir(final_adapter), flush=True)
    print('run_new_train =', run_new_train, flush=True)
    if run_new_train:
        ensure_train_dependencies()
        train_overrides = {
            'MODEL_NAME': MODEL_NAME,
            'MODEL_REVISION': 'cbd3fa9f933d55ef16a84236559f4ee2a0526848',
            'DATA_FILE': 'data/v217/v217_short_answer_train.jsonl',
            'VAL_FILE': 'data/v217/v217_short_answer_val.jsonl',
            'EXPECTED_TRAIN_SHA256': EXPECTED_TRAIN_SHA256,
            'EXPECTED_VAL_SHA256': EXPECTED_VAL_SHA256,
            'MIN_TRAIN_EXAMPLES': str(MIN_TRAIN_EXAMPLES),
            'MIN_VAL_EXAMPLES': str(MIN_VAL_EXAMPLES),
            'MIN_TOKENIZED_TRAIN_EXAMPLES': '10000',
            'MIN_TOKENIZED_VAL_EXAMPLES': '681',
            'OUTPUT_DIR': str(TRAIN_OUT),
            'OUTPUT_REPO': '',
            'UPLOAD_TO_HF': '0',
            'UPLOAD_CHECKPOINTS_DURING_TRAINING': '0',
            'INIT_ADAPTER_DIR': str(INIT_ADAPTER_DIR),
            'INIT_ADAPTER_LOAD_MODE': 'manual',
            'FAIL_ON_MISSING_ADAPTER_KEYS': '1',
            'TRAINABLE_LORA_MODULES': V226_TRAINABLE_MODULES,
            'MAX_TRAINABLE_PARAM_RATIO': '0.030',
            'MAX_LENGTH': str(V226_MAX_LENGTH),
            'BATCH_SIZE': str(V226_BATCH_SIZE),
            'MICRO_BATCH_SIZE': str(V226_MICRO_BATCH_SIZE),
            'LEARNING_RATE': V226_LR,
            'FINAL_LEARNING_RATE': V226_FINAL_LR,
            'MAX_STEPS': V226_MAX_STEPS,
            'NUM_EPOCHS': '1',
            'SAVE_EVERY_STEPS': '1',
            'EVAL_EVERY_STEPS': '1',
            'EVAL_MAX_EXAMPLES': '256',
            'LOG_EVERY_STEPS': '1',
            'MICRO_LOG_EVERY': '0',
            'BASELINE_EVAL_BEFORE_TRAIN': '1',
            'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': '0.020',
            'MAX_FINAL_EVAL_REGRESSION': '0.020',
            'REQUIRE_FINAL_EVAL_LTE_BASELINE': '0',
            'ABORT_TRAIN_RISE_POINTS': '0',
            'SAMPLING_MODE': 'weighted',
            'SUBCATEGORY_WEIGHTS': 'equation_transform=1.65,equation_symbolic=1.65,equation_numeric=1.35,bit_manipulation=1.05',
            'SOURCE_WEIGHTS': 'v216=1.25,v217=1.10',
            'MAX_PROMPT_TRUNCATION_RATE': str(MAX_PROMPT_TRUNCATION_RATE),
            'REQUIRE_OFFSET_MASK': '1' if REQUIRE_OFFSET_MASK else '0',
            'TOKENIZE_ONLY_DRY_RUN': '0',
            'DRY_RUN_VALIDATE_ONLY': '0',
            'USE_BITSANDBYTES': '1',
            'RUN_ID': 'v226-v194-micro-lr2e9-s6',
        }
        for key, value in train_overrides.items():
            os.environ[key] = str(value)
        print('train_overrides =', json.dumps(train_overrides, indent=2, sort_keys=True), flush=True)
        rc = run_cmd([sys.executable, str(ROOT / 'scripts/hf_job_train_v90.py')], cwd=ROOT, log_path=TRAIN_OUT / 'v226_micro_train.log', check=True, heartbeat_s=60, suppress_after_lines=320)
        print('v226 train returncode =', rc, flush=True)
    else:
        print('reusing existing V226 final adapter:', final_adapter, flush=True)
print('=== V226 TRAIN END ===', flush=True)
"""
        ),
        code(
            """# CELL: collect V194, V217, V223, and V226 checkpoint candidates.
print('=== V226 CANDIDATE COLLECTION START ===', flush=True)
candidates = []
seen = set()

def add_adapter(name, adapter_path, source_kind):
    adapter_path = pathlib.Path(adapter_path)
    complete = is_complete_adapter_dir(adapter_path)
    print('candidate_probe =', json.dumps({'name': name, 'adapter': str(adapter_path), 'source_kind': source_kind, 'complete': complete}, sort_keys=True), flush=True)
    if not complete:
        return
    key = str(adapter_path.resolve())
    if key in seen:
        return
    seen.add(key)
    candidates.append({'name': name, 'adapter': str(adapter_path), 'source_kind': source_kind})

add_adapter('v194_protected_baseline', V194_ADAPTER, 'baseline')
add_adapter('v217_final_existing', V217_ADAPTER, 'baseline')
if V223_TRAIN_OUT.exists():
    add_adapter('v223_final_adapter', V223_TRAIN_OUT / 'final_adapter', 'v223_final')
    for path in sorted(V223_TRAIN_OUT.glob('checkpoint-*')):
        add_adapter('v223_' + path.name.replace('-', '_'), path, 'v223_checkpoint')
else:
    print('V223_TRAIN_OUT missing; no V223 checkpoints collected.', flush=True)
if TRAIN_OUT.exists():
    add_adapter('v226_final_adapter', TRAIN_OUT / 'final_adapter', 'v226_final')
    for path in sorted(TRAIN_OUT.glob('checkpoint-*')):
        add_adapter('v226_' + path.name.replace('-', '_'), path, 'v226_checkpoint')
if len(candidates) > V226_MAX_CANDIDATES:
    print('candidate_count_before_cap =', len(candidates), flush=True)
    baseline = [item for item in candidates if item['source_kind'] == 'baseline']
    rest = [item for item in candidates if item['source_kind'] != 'baseline']
    candidates = (baseline + rest)[:V226_MAX_CANDIDATES]
candidate_json = EVAL_OUT / 'v226_checkpoint_candidates.json'
write_json(candidate_json, candidates)
print('candidate_count =', len(candidates), flush=True)
print('candidate_json =', candidate_json, flush=True)
print('candidate_rows =', json.dumps(candidates, indent=2, sort_keys=True), flush=True)
if len(candidates) < 2:
    raise RuntimeError('Need at least V194 and V217 candidates for V226 checkpoint sweep.')
print('=== V226 CANDIDATE COLLECTION END ===', flush=True)
"""
        ),
        code(
            """# CELL: weak checkpoint batch eval with the best V225 prompt contract.
print('=== V226 WEAK CHECKPOINT EVAL START ===', flush=True)
weak_gate_pass_for_full = False
batch_summary_json = EVAL_OUT / 'batch_candidate_summary.json'
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping V226 weak checkpoint eval.', flush=True)
else:
    ensure_vllm_for_eval()
    run_new_eval = FORCE_REEVAL or not batch_summary_json.exists()
    print('run_new_eval =', run_new_eval, flush=True)
    if run_new_eval:
        cmd = [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapters_batch.py'),
            '--solution-csv', str(WEAK_EVAL_CSV),
            '--questions-csv', str(WEAK_EVAL_CSV),
            '--candidates-json', str(candidate_json),
            '--base-model-path', MODEL_NAME,
            '--label-prefix', 'v226_weak_ckpt',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(EVAL_OUT),
            '--max-tokens', str(V226_MAX_TOKENS),
            '--max-model-len', str(V226_MAX_MODEL_LEN),
            '--max-num-seqs', str(V226_MAX_NUM_SEQS),
            '--warmup-rows', str(V226_WARMUP_ROWS),
            '--prompt-suffix', V226_PROMPT_SUFFIX,
            '--continue-on-error',
        ]
        rc = run_cmd(cmd, cwd=ROOT, log_path=EVAL_OUT / 'weak_checkpoint_eval.log', check=True, heartbeat_s=60, suppress_after_lines=360)
        print('weak checkpoint eval returncode =', rc, flush=True)
    else:
        print('reusing batch summary:', batch_summary_json, flush=True)
print('batch_summary_json =', batch_summary_json, flush=True)
print('=== V226 WEAK CHECKPOINT EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: analyze V226 checkpoint sweep and decide gate action.
print('=== V226 CHECKPOINT ANALYSIS START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v226_checkpoint_sweep_manifest.json'
if not RUN_ANALYSIS:
    print('RUN_ANALYSIS is false; skipping analysis.', flush=True)
else:
    if not batch_summary_json.exists():
        raise FileNotFoundError(batch_summary_json)
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v226_checkpoint_sweep.py'),
        '--batch-summary-json', str(batch_summary_json),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v226_checkpoint_sweep',
        '--weak-total-min', str(WEAK_MIN_FOR_FULL),
        '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
        '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
        '--weak-trunc-max', str(WEAK_MAX_TRUNC_FOR_FULL),
    ]
    rc = run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v226_checkpoint_sweep.log', check=True)
    print('v226 analysis returncode =', rc, flush=True)
analysis_manifest = read_json(analysis_manifest_path)
decision = analysis_manifest.get('decision', {})
best = analysis_manifest.get('best', {})
weak_gate_pass_for_full = bool(best.get('weak_gate_pass_for_full', False))
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('best =', json.dumps(best, indent=2, sort_keys=True)[:6000], flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V226 CHECKPOINT ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval/package hard block and final manifest.
print('=== V226 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(ANALYSIS_OUT / 'v226_checkpoint_sweep_manifest.json')
decision = analysis_manifest.get('decision', {})
best = analysis_manifest.get('best', {})
weak_gate_pass_for_full = bool(best.get('weak_gate_pass_for_full', False))
full_candidate_gate = False
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('full_candidate_gate =', full_candidate_gate, flush=True)
print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('Full eval is blocked by default to avoid accidental GPU spend.', flush=True)
print('Full eval is intentionally not automatic in V226 checkpoint sweep notebook.', flush=True)
print('No package and no Kaggle submit can be created in V226.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('V226 hard block violated. Kaggle submission is disabled.')
final_manifest_path = OUT_ROOT / 'v226_equation_checkpoint_sweep_final_manifest.json'
final_manifest = {
    'version': VERSION,
    'repo_branch': REPO_BRANCH,
    'weak_gate_pass_for_full': weak_gate_pass_for_full,
    'full_candidate_gate': full_candidate_gate,
    'decision': decision,
    'best': best,
    'thresholds': {
        'weak_total': WEAK_MIN_FOR_FULL,
        'weak_equation_transform': WEAK_EQ_MIN_FOR_FULL,
        'weak_bit_manipulation': WEAK_BIT_MIN_FOR_FULL,
        'weak_truncated': WEAK_MAX_TRUNC_FOR_FULL,
        'full_min_candidate': FULL_MIN_CANDIDATE,
        'full_max_trunc': FULL_MAX_TRUNC,
    },
    'train_out': str(TRAIN_OUT),
    'eval_out': str(EVAL_OUT),
    'analysis_manifest': str(ANALYSIS_OUT / 'v226_checkpoint_sweep_manifest.json'),
    'roadmap_next': decision.get('next_action', 'Review V226 checkpoint sweep outputs.'),
}
write_json(final_manifest_path, final_manifest)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('roadmap_next =', final_manifest['roadmap_next'], flush=True)
print('=== V226 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")
    print(COLAB_URL)


if __name__ == "__main__":
    main()

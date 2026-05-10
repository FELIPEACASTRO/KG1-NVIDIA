#!/usr/bin/env python3
"""Build the V228 safe staged eval Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V228_SAFE_STAGED_EVAL_COLAB.ipynb")
BRANCH = "v228-safe-staged-eval"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V228_SAFE_STAGED_EVAL_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V228_SAFE_STAGED_EVAL_COLAB.ipynb"
)
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v228-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V228 Safe Staged Eval Colab

Purpose: replace the V227 monolithic weak eval that stalls near row 223 with a staged, timeout-aware evaluation flow. V228 does not train a new adapter by default. It audits the existing V226/V227 artifacts, evaluates V226 best vs V227 candidates in controlled windows, aggregates partial/full staged results, and blocks full official eval, packaging, and Kaggle submit.

Gate objective remains explicit: `193/315` total, `60/155` equation_transform, `133/160` bit_manipulation, and no more than `3` truncations.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V228 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V228 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration, gates, and hard submit lock.
print('=== V228 CONFIG START ===', flush=True)
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import shutil
import signal
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

VERSION = 'V228_SAFE_STAGED_EVAL_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V228')
OUT_ROOT = DRIVE_ROOT / 'output_v228_safe_staged_eval'
EVAL_OUT = OUT_ROOT / 'eval_v228_safe_staged'
ANALYSIS_OUT = OUT_ROOT / 'analysis_v228_safe_staged'

MODEL_NAME = os.environ.get('KG1_V228_MODEL_NAME', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
V228_VLLM_PIP_SPEC = os.environ.get('KG1_V228_VLLM_PIP_SPEC', 'vllm==0.20.1')
V228_MAX_MODEL_LEN = int(os.environ.get('KG1_V228_MAX_MODEL_LEN', '8192'))
V228_PROMPT_SUFFIX = os.environ.get('KG1_V228_PROMPT_SUFFIX', '\\nReturn exactly one line in this format: `\\\\boxed{answer}`.')
V228_STAGE_TIMEOUT_S = int(os.environ.get('KG1_V228_STAGE_TIMEOUT_S', '1800'))

V221_EVAL_OUT = pathlib.Path(os.environ.get('KG1_V228_V221_EVAL_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab'))
V221_WEAK_CSV = pathlib.Path(os.environ.get('KG1_V228_V221_WEAK_CSV', str(V221_EVAL_OUT / 'v221_weak_315.csv')))
V221_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get('KG1_V228_V221_BATCH_SUMMARY_JSON', str(V221_EVAL_OUT / 'batch_candidate_summary.json')))

V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
V217_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter')
V226_TRAIN_OUT = pathlib.Path(os.environ.get('KG1_V228_V226_TRAIN_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/train_v226_v194_micro_lr2e9_s6'))
V226_BEST_CHECKPOINT = pathlib.Path(os.environ.get('KG1_V228_V226_BEST_CHECKPOINT', str(V226_TRAIN_OUT / 'checkpoint-1')))
V227_TRAIN_OUT = pathlib.Path(os.environ.get('KG1_V228_V227_TRAIN_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V227/output_v227_targeted_equation_micro_sweep/train_v227_from_v226ckpt1_eq_nudge_lr5e10_s2'))
V227_FINAL_ADAPTER = pathlib.Path(os.environ.get('KG1_V228_V227_FINAL_ADAPTER', str(V227_TRAIN_OUT / 'final_adapter')))
V227_CHECKPOINT_1 = pathlib.Path(os.environ.get('KG1_V228_V227_CHECKPOINT_1', str(V227_TRAIN_OUT / 'checkpoint-1')))
INIT_ADAPTER_DIR = V227_FINAL_ADAPTER

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

RUN_TRAIN = os.environ.get('KG1_V228_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_EVAL = os.environ.get('KG1_V228_RUN_EVAL', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_ANALYSIS = os.environ.get('KG1_V228_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_CONFIRM_BEST = os.environ.get('KG1_V228_RUN_CONFIRM_BEST', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_FULL_IF_GATE = False
FORCE_REEVAL = os.environ.get('KG1_V228_FORCE_REEVAL', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
ALLOW_KAGGLE_SUBMIT = False

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4

for path in [DRIVE_ROOT, OUT_ROOT, EVAL_OUT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('EVAL_OUT =', EVAL_OUT, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('MODEL_NAME =', MODEL_NAME, flush=True)
print('V228_MAX_MODEL_LEN =', V228_MAX_MODEL_LEN, flush=True)
print('V228_PROMPT_SUFFIX =', repr(V228_PROMPT_SUFFIX), flush=True)
print('V228_STAGE_TIMEOUT_S =', V228_STAGE_TIMEOUT_S, flush=True)
print('V221_WEAK_CSV =', V221_WEAK_CSV, flush=True)
print('V221_BATCH_SUMMARY_JSON =', V221_BATCH_SUMMARY_JSON, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V217_ADAPTER =', V217_ADAPTER, flush=True)
print('V226_BEST_CHECKPOINT =', V226_BEST_CHECKPOINT, flush=True)
print('V227_FINAL_ADAPTER =', V227_FINAL_ADAPTER, flush=True)
print('V227_CHECKPOINT_1 =', V227_CHECKPOINT_1, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_CONFIRM_BEST =', RUN_CONFIRM_BEST, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('FORCE_REEVAL =', FORCE_REEVAL, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if RUN_TRAIN:
    raise RuntimeError('V228 is eval-only; RUN_TRAIN must stay false.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V228.')
print('=== V228 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging, heartbeat, timeouts, hashes, and dependency installers.
print('=== V228 HELPERS START ===', flush=True)

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

def run_cmd(cmd, cwd=None, log_path=None, check=True, heartbeat_s=0, suppress_after_lines=260, timeout_s=None):
    started = time.time()
    printable = ' '.join(str(x) for x in cmd)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or os.getcwd(), flush=True)
    print('+', printable, flush=True)
    if timeout_s:
        print('timeout_s =', timeout_s, flush=True)
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
        start_new_session=True,
    )
    lines = []
    timed_out = False
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
            print('[V228 heartbeat] elapsed_s={:.1f} {}'.format(now - started, resource_snapshot_line()), flush=True)
            last_heartbeat = now
        if timeout_s and now - started > timeout_s:
            timed_out = True
            print('timeout_reached =', timeout_s, flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(5)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception as exc:
                print('timeout_kill_warning =', repr(exc), flush=True)
                proc.kill()
            break
    returncode = proc.wait()
    if timed_out:
        returncode = -999
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

def ensure_vllm_for_eval():
    print('=== V228 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    if verify_import('vllm', 'verify_import_vllm.log') != 0:
        print('vLLM subprocess import failed; installing pinned V228_VLLM_PIP_SPEC =', V228_VLLM_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', V228_VLLM_PIP_SPEC], cwd='/content', log_path=OUT_ROOT / 'pip_install_vllm.log', check=True, heartbeat_s=60, timeout_s=1200)
        if verify_import('vllm', 'verify_import_vllm.log') != 0:
            raise RuntimeError('vLLM import still failed after install.')
    print('=== V228 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)

def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.is_dir() and (path / 'adapter_config.json').exists() and (
        (path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists()
    )

def safe_name(value):
    return str(value).replace('/', '_').replace('\\\\', '_').replace(' ', '_')

print('=== V228 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and validate static data hashes.
print('=== V228 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True, timeout_s=300)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
print('repo_commit =', repo_commit, flush=True)
compile_targets = [
    ROOT / 'src/competition_utils.py',
    ROOT / 'scripts/evaluate_lora_adapter.py',
    ROOT / 'scripts/evaluate_lora_adapters_batch.py',
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
    raise RuntimeError('train sha256 mismatch')
if observed_val_sha256 != EXPECTED_VAL_SHA256:
    raise RuntimeError('validation sha256 mismatch')
if train_rows < MIN_TRAIN_EXAMPLES:
    raise RuntimeError('train row count below minimum')
if val_rows < MIN_VAL_EXAMPLES:
    raise RuntimeError('validation row count below minimum')
print('=== V228 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, Drive artifact, adapter, and dependency audit.
print('=== V228 RUNTIME ARTIFACT AUDIT START ===', flush=True)
torch_probe_path = OUT_ROOT / 'verify_torch_cuda.jsonl'
run_cmd([
    sys.executable,
    '-c',
    "import json, torch; props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), 'cuda_available': torch.cuda.is_available(), 'gpu_name': props.name if props else '', 'gpu_total_gib': props.total_memory/1024**3 if props else 0.0}))",
], cwd='/content', log_path=torch_probe_path, check=True)
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
    raise RuntimeError('CUDA is required for V228 safe staged eval.')
if gpu_total_gib < 70:
    raise RuntimeError(f'GPU memory too small for V228 eval: {gpu_total_gib:.2f} GiB')
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
print('V226_BEST_CHECKPOINT complete =', is_complete_adapter_dir(V226_BEST_CHECKPOINT), flush=True)
print('V227_FINAL_ADAPTER complete =', is_complete_adapter_dir(V227_FINAL_ADAPTER), flush=True)
print('V227_CHECKPOINT_1 complete =', is_complete_adapter_dir(V227_CHECKPOINT_1), flush=True)
for required_path in [V221_WEAK_CSV, V221_BATCH_SUMMARY_JSON]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)
try:
    from safetensors import safe_open
except Exception:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], cwd='/content', log_path=OUT_ROOT / 'pip_install_safetensors.log', check=True, timeout_s=300)
    from safetensors import safe_open
for label, adapter_path in [('V194', V194_ADAPTER), ('V217', V217_ADAPTER), ('V226_BEST', V226_BEST_CHECKPOINT), ('V227_FINAL', V227_FINAL_ADAPTER), ('V227_CHECKPOINT_1', V227_CHECKPOINT_1), ('INIT', INIT_ADAPTER_DIR)]:
    print(label, 'adapter path =', adapter_path, 'complete =', is_complete_adapter_dir(adapter_path), flush=True)
    if not is_complete_adapter_dir(adapter_path):
        raise RuntimeError(f'{label} adapter incomplete: {adapter_path}')
    cfg = read_json(adapter_path / 'adapter_config.json')
    print(label, 'target_modules =', cfg.get('target_modules'), flush=True)
    print(label, 'target_parameters =', cfg.get('target_parameters'), flush=True)
    if sorted(cfg.get('target_modules') or []) != sorted(EXPECTED_V194_TARGET_MODULES):
        raise RuntimeError(f'{label} target_modules mismatch')
    target_parameters = cfg.get('target_parameters') or []
    if label in {'V194', 'V217'}:
        if sorted(target_parameters) != sorted(EXPECTED_V194_TARGET_PARAMETERS):
            raise RuntimeError(f'{label} target_parameters mismatch')
    elif sorted(target_parameters) != sorted(EXPECTED_V194_TARGET_PARAMETERS):
        print(label, 'target_parameters differ from V194/V217; accepting PEFT checkpoint format.', flush=True)
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
print('=== V228 RUNTIME ARTIFACT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: prepare weak CSV windows and focused candidates.
print('=== V228 WEAK WINDOW PREP START ===', flush=True)
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

candidates = [
    {'name': 'v226_best_checkpoint1_observed_191', 'adapter': str(V226_BEST_CHECKPOINT), 'source_kind': 'v226_best'},
    {'name': 'v227_final_adapter', 'adapter': str(V227_FINAL_ADAPTER), 'source_kind': 'v227_final'},
    {'name': 'v227_checkpoint_1', 'adapter': str(V227_CHECKPOINT_1), 'source_kind': 'v227_checkpoint'},
]
focused_candidates_json = EVAL_OUT / 'v228_focused_v226_v227_candidates.json'
write_json(focused_candidates_json, candidates)
print('focused_candidates_json =', focused_candidates_json, flush=True)
print('focused_candidates =', json.dumps(candidates, indent=2, sort_keys=True), flush=True)

stage_windows = [
    {'name': 'triage000_160', 'start': 0, 'end': 160, 'max_tokens': 768, 'max_num_seqs': 16, 'timeout_s': V228_STAGE_TIMEOUT_S},
    {'name': 'mid160_224', 'start': 160, 'end': 224, 'max_tokens': 512, 'max_num_seqs': 8, 'timeout_s': V228_STAGE_TIMEOUT_S},
    {'name': 'tail224_315', 'start': 224, 'end': 315, 'max_tokens': 256, 'max_num_seqs': 4, 'timeout_s': V228_STAGE_TIMEOUT_S},
]
for window in stage_windows:
    window_dir = EVAL_OUT / window['name']
    window_dir.mkdir(parents=True, exist_ok=True)
    window_csv = window_dir / 'weak_window.csv'
    weak_df.iloc[int(window['start']):int(window['end'])].to_csv(window_csv, index=False)
    window['csv'] = str(window_csv)
    window['output_dir'] = str(window_dir)
    print('window =', json.dumps(window, sort_keys=True), flush=True)
windows_json = EVAL_OUT / 'v228_eval_windows.json'
write_json(windows_json, stage_windows)
print('windows_json =', windows_json, flush=True)
print('=== V228 WEAK WINDOW PREP END ===', flush=True)
"""
        ),
        code(
            """# CELL: staged weak eval with bounded windows and timeout-aware continuation.
print('=== V228 STAGED WEAK EVAL START ===', flush=True)
staged_runs_path = EVAL_OUT / 'v228_staged_eval_runs.json'
staged_runs = []
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping V228 staged weak eval.', flush=True)
else:
    ensure_vllm_for_eval()
    for window in stage_windows:
        window_name = window['name']
        window_dir = pathlib.Path(window['output_dir'])
        summary_json = window_dir / 'batch_candidate_summary.json'
        run_new_window = FORCE_REEVAL or not summary_json.exists()
        print('stage_window_start =', json.dumps(window, sort_keys=True), flush=True)
        print('summary_json =', summary_json, flush=True)
        print('run_new_window =', run_new_window, flush=True)
        rc = 0
        if run_new_window:
            cmd = [
                sys.executable,
                str(ROOT / 'scripts/evaluate_lora_adapters_batch.py'),
                '--solution-csv', str(window['csv']),
                '--questions-csv', str(window['csv']),
                '--candidates-json', str(focused_candidates_json),
                '--base-model-path', MODEL_NAME,
                '--label-prefix', 'v228_' + window_name,
                '--seed', '42',
                '--limit', '0',
                '--output-dir', str(window_dir),
                '--max-tokens', str(window['max_tokens']),
                '--max-model-len', str(V228_MAX_MODEL_LEN),
                '--max-num-seqs', str(window['max_num_seqs']),
                '--warmup-rows', '0',
                '--prompt-suffix', V228_PROMPT_SUFFIX,
                '--continue-on-error',
            ]
            rc = run_cmd(
                cmd,
                cwd=ROOT,
                log_path=window_dir / (window_name + '_eval.log'),
                check=False,
                heartbeat_s=60,
                suppress_after_lines=260,
                timeout_s=int(window['timeout_s']),
            )
        else:
            print('reusing existing stage summary:', summary_json, flush=True)
        stage_record = {
            **window,
            'returncode': rc,
            'summary_json': str(summary_json),
            'summary_exists': summary_json.exists(),
            'timed_out': rc == -999,
        }
        staged_runs.append(stage_record)
        write_json(staged_runs_path, staged_runs)
        print('stage_window_result =', json.dumps(stage_record, indent=2, sort_keys=True), flush=True)
print('staged_runs_path =', staged_runs_path, flush=True)
print('staged_runs =', json.dumps(staged_runs, indent=2, sort_keys=True), flush=True)
print('=== V228 STAGED WEAK EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: aggregate V228 staged weak eval and decide next action.
print('=== V228 STAGED ANALYSIS START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v228_safe_staged_eval_manifest.json'
weak_gate_pass_for_full = False
if not RUN_ANALYSIS:
    print('RUN_ANALYSIS is false; skipping staged analysis.', flush=True)
else:
    candidate_totals = {}
    window_summaries = []
    for run in staged_runs:
        summary_json = pathlib.Path(run['summary_json'])
        window_name = run['name']
        window_rows = int(run['end']) - int(run['start'])
        if not summary_json.exists():
            window_summaries.append({**run, 'loaded': False, 'rows': []})
            print('missing_window_summary =', summary_json, flush=True)
            continue
        summary = read_json(summary_json)
        rows = summary.get('rows', [])
        window_summaries.append({**run, 'loaded': True, 'rows': rows})
        print('loaded_window_summary =', summary_json, 'rows =', len(rows), flush=True)
        for row in rows:
            name = row.get('name', '')
            total = candidate_totals.setdefault(name, {
                'name': name,
                'adapter': row.get('adapter', ''),
                'status': 'ok',
                'windows_completed': 0,
                'windows_expected': len(stage_windows),
                'rows_evaluated': 0,
                'correct': 0,
                'equation_transform_correct': 0,
                'bit_manipulation_correct': 0,
                'truncated': 0,
                'completion_tokens': 0,
                'window_details': [],
            })
            detail = {
                'window': window_name,
                'status': row.get('status'),
                'rows': window_rows,
                'correct': int(row.get('correct', 0)),
                'equation_transform_correct': int(row.get('equation_transform_correct', 0)),
                'bit_manipulation_correct': int(row.get('bit_manipulation_correct', 0)),
                'truncated': int(row.get('truncated', 999999)),
                'report_json': row.get('report_json', ''),
            }
            total['window_details'].append(detail)
            if row.get('status') == 'ok':
                total['windows_completed'] += 1
                total['rows_evaluated'] += window_rows
                total['correct'] += detail['correct']
                total['equation_transform_correct'] += detail['equation_transform_correct']
                total['bit_manipulation_correct'] += detail['bit_manipulation_correct']
                total['truncated'] += detail['truncated']
                total['completion_tokens'] += int(row.get('completion_tokens', 0))
            else:
                total['status'] = 'partial'
    totals = list(candidate_totals.values())
    for row in totals:
        all_windows = int(row['windows_completed']) == int(row['windows_expected']) and int(row['rows_evaluated']) == 315
        row['all_windows_completed'] = bool(all_windows)
        row['accuracy'] = float(row['correct'] / row['rows_evaluated']) if row['rows_evaluated'] else 0.0
        row['weak_gate_pass_for_full'] = bool(
            all_windows
            and int(row['correct']) >= WEAK_MIN_FOR_FULL
            and int(row['equation_transform_correct']) >= WEAK_EQ_MIN_FOR_FULL
            and int(row['bit_manipulation_correct']) >= WEAK_BIT_MIN_FOR_FULL
            and int(row['truncated']) <= WEAK_MAX_TRUNC_FOR_FULL
        )
        row['gate_total_gap'] = max(0, WEAK_MIN_FOR_FULL - int(row['correct']))
        row['gate_eq_gap'] = max(0, WEAK_EQ_MIN_FOR_FULL - int(row['equation_transform_correct']))
        row['gate_bit_gap'] = max(0, WEAK_BIT_MIN_FOR_FULL - int(row['bit_manipulation_correct']))
        row['gate_trunc_gap'] = max(0, int(row['truncated']) - WEAK_MAX_TRUNC_FOR_FULL)
    totals = sorted(
        totals,
        key=lambda item: (
            bool(item.get('weak_gate_pass_for_full')),
            int(item.get('rows_evaluated', 0)),
            int(item.get('correct', 0)),
            int(item.get('equation_transform_correct', 0)),
            int(item.get('bit_manipulation_correct', 0)),
            -int(item.get('truncated', 999999)),
        ),
        reverse=True,
    )
    best = totals[0] if totals else {}
    weak_gate_pass_for_full = bool(best.get('weak_gate_pass_for_full', False))
    if weak_gate_pass_for_full:
        decision = {
            'decision': 'staged_candidate_passed_weak_gate_confirm_full_eval_separately',
            'best_candidate': best.get('name'),
            'reason': f"correct={best.get('correct')}; eq={best.get('equation_transform_correct')}; bit={best.get('bit_manipulation_correct')}; truncated={best.get('truncated')}",
            'next_action': 'Run a separate full-eval confirmation notebook. Do not submit automatically.',
        }
    else:
        decision = {
            'decision': 'no_staged_candidate_passed_weak_gate',
            'best_candidate': best.get('name'),
            'reason': f"best_correct={best.get('correct')}; rows_evaluated={best.get('rows_evaluated')}; eq={best.get('equation_transform_correct')}; bit={best.get('bit_manipulation_correct')}; truncated={best.get('truncated')}; total_gap={best.get('gate_total_gap')}; eq_gap={best.get('gate_eq_gap')}; bit_gap={best.get('gate_bit_gap')}; trunc_gap={best.get('gate_trunc_gap')}",
            'next_action': 'Use the V228 staged totals to decide whether V227 is a regression and prepare V229 only if V227 beats V226 in the triage window.',
        }
    summary_csv = ANALYSIS_OUT / 'v228_safe_staged_eval_summary.csv'
    import pandas as pd
    flat_rows = []
    for row in totals:
        flat = {k: v for k, v in row.items() if k != 'window_details'}
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(summary_csv, index=False)
    manifest = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'version': VERSION,
        'staged_runs_path': str(staged_runs_path),
        'thresholds': {
            'weak_total': WEAK_MIN_FOR_FULL,
            'weak_equation_transform': WEAK_EQ_MIN_FOR_FULL,
            'weak_bit_manipulation': WEAK_BIT_MIN_FOR_FULL,
            'weak_truncated': WEAK_MAX_TRUNC_FOR_FULL,
            'full_min_candidate': FULL_MIN_CANDIDATE,
            'full_max_trunc': FULL_MAX_TRUNC,
        },
        'window_summaries': window_summaries,
        'totals': totals,
        'best': best,
        'decision': decision,
        'outputs': {
            'summary_csv': str(summary_csv),
            'manifest_json': str(analysis_manifest_path),
        },
    }
    write_json(analysis_manifest_path, manifest)
    print('staged_summary =', pd.DataFrame(flat_rows).to_string(index=False), flush=True)
    print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print('best =', json.dumps(best, indent=2, sort_keys=True)[:6000], flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V228 STAGED ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: optional best-candidate weak confirmation with safer cap.
print('=== V228 OPTIONAL BEST CONFIRM START ===', flush=True)
confirm_summary_json = None
if not RUN_CONFIRM_BEST:
    print('RUN_CONFIRM_BEST is false; skipping optional best-candidate weak confirmation.', flush=True)
else:
    analysis_manifest = read_json(analysis_manifest_path)
    best = analysis_manifest.get('best', {})
    best_name = best.get('name')
    if not best_name:
        raise RuntimeError('No best candidate available for confirmation.')
    best_candidates = [item for item in candidates if item['name'] == best_name]
    if not best_candidates:
        raise RuntimeError(f'Best candidate missing from focused candidate list: {best_name}')
    confirm_out = EVAL_OUT / 'confirm_best_mtok512'
    confirm_out.mkdir(parents=True, exist_ok=True)
    confirm_candidates_json = confirm_out / 'confirm_best_candidate.json'
    write_json(confirm_candidates_json, best_candidates)
    confirm_weak_csv = confirm_out / 'weak_315.csv'
    weak_df.to_csv(confirm_weak_csv, index=False)
    confirm_summary_json = confirm_out / 'batch_candidate_summary.json'
    print('confirm_best_name =', best_name, flush=True)
    print('confirm_candidates_json =', confirm_candidates_json, flush=True)
    print('confirm_summary_json =', confirm_summary_json, flush=True)
    ensure_vllm_for_eval()
    rc = run_cmd([
        sys.executable,
        str(ROOT / 'scripts/evaluate_lora_adapters_batch.py'),
        '--solution-csv', str(confirm_weak_csv),
        '--questions-csv', str(confirm_weak_csv),
        '--candidates-json', str(confirm_candidates_json),
        '--base-model-path', MODEL_NAME,
        '--label-prefix', 'v228_confirm_best',
        '--seed', '42',
        '--limit', '0',
        '--output-dir', str(confirm_out),
        '--max-tokens', '512',
        '--max-model-len', str(V228_MAX_MODEL_LEN),
        '--max-num-seqs', '4',
        '--warmup-rows', '0',
        '--prompt-suffix', V228_PROMPT_SUFFIX,
        '--continue-on-error',
    ], cwd=ROOT, log_path=confirm_out / 'confirm_best_eval.log', check=False, heartbeat_s=60, suppress_after_lines=260, timeout_s=3600)
    print('confirm_best_returncode =', rc, flush=True)
    print('confirm_summary_exists =', confirm_summary_json.exists(), flush=True)
print('confirm_summary_json =', confirm_summary_json, flush=True)
print('=== V228 OPTIONAL BEST CONFIRM END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval/package hard block and final manifest.
print('=== V228 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
decision = analysis_manifest.get('decision', {})
best = analysis_manifest.get('best', {})
weak_gate_pass_for_full = bool(best.get('weak_gate_pass_for_full', False))
full_candidate_gate = False
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('full_candidate_gate =', full_candidate_gate, flush=True)
print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('Full eval is blocked by default to avoid accidental GPU spend.', flush=True)
print('Full eval is intentionally not automatic in V228 safe staged eval notebook.', flush=True)
print('No package and no Kaggle submit can be created in V228.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('V228 hard block violated. Kaggle submission is disabled.')
final_manifest_path = OUT_ROOT / 'v228_safe_staged_eval_final_manifest.json'
final_manifest = {
    'version': VERSION,
    'repo_branch': REPO_BRANCH,
    'weak_gate_pass_for_full': weak_gate_pass_for_full,
    'full_candidate_gate': full_candidate_gate,
    'decision': decision,
    'best': best,
    'thresholds': analysis_manifest.get('thresholds', {}),
    'eval_out': str(EVAL_OUT),
    'analysis_manifest': str(analysis_manifest_path),
    'confirm_summary_json': str(confirm_summary_json) if confirm_summary_json else '',
    'roadmap_next': decision.get('next_action', 'Review V228 staged eval outputs.'),
}
write_json(final_manifest_path, final_manifest)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('roadmap_next =', final_manifest['roadmap_next'], flush=True)
print('=== V228 FINAL MANIFEST END ===', flush=True)
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

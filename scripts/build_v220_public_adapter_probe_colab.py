#!/usr/bin/env python3
"""Build the V220 public adapter probe Colab notebook.

V218 proved that disabling thinking and cutting output to 1024 tokens collapses
quality. This notebook runs a cheaper weak-only A/B that keeps thinking enabled
and evaluates a public Hugging Face PEFT adapter before any training or full evaluation spend.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v220-public-adapter-probe/notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v220-public-adapter-probe/notebooks/KG1_V220_PUBLIC_ADAPTER_PROBE_COLAB.ipynb"
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
V218_DECODE_REPORT_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V218/output_v218_decode_rescue/"
    "eval_v218_decode_rescue/weak_eval_decode_mtok1024_think0/"
    "v218_decode_mtok1024_weak_eval_report.json"
)
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"
TRAIN_ROWS = 10206
VAL_ROWS = 681

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v220-{prefix}-{_CELL_COUNTER:02d}"


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
            f"""# KG1 V220 Public Adapter Probe Colab

Purpose: run a weak-only diagnostic before any new training spend.

This notebook:

- keeps thinking enabled by default;
- evaluates the public Hugging Face adapter `Naribow/nemotron-sft-lora`;
- uses the same 315-row weak gate split from the V194 validation CSV;
- blocks full eval by default unless a candidate passes the weak gate and the user explicitly enables full eval;
- never submits to Kaggle.

Colab URL:

`{COLAB_URL}`
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V220 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V220 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration and hard submit lock.
print('=== V220 CONFIG START ===', flush=True)
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

VERSION = 'V220_PUBLIC_ADAPTER_PROBE_20260508'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'v220-public-adapter-probe')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V220')
OUT_ROOT = DRIVE_ROOT / 'output_v220_public_adapter_probe'
EVAL_OUT = OUT_ROOT / 'eval_v220_public_adapter_probe'
PACKAGE_OUT = OUT_ROOT / 'package_v220_public_adapter_probe'
V194_ADAPTER = pathlib.Path('{V194_ADAPTER_DRIVE}')
V194_VAL_CSV = pathlib.Path('{V194_VAL_CSV_DRIVE}')
V217_FINAL_ADAPTER = pathlib.Path('{V217_FINAL_ADAPTER_DRIVE}')
V217_WEAK_REPORT = pathlib.Path('{V217_WEAK_REPORT_DRIVE}')
V218_DECODE_REPORT = pathlib.Path('{V218_DECODE_REPORT_DRIVE}')
NARIBOW_ADAPTER_REPO = os.environ.get('KG1_V220_PUBLIC_ADAPTER_REPO', 'Naribow/nemotron-sft-lora')
NARIBOW_ADAPTER_REVISION = os.environ.get('KG1_V220_PUBLIC_ADAPTER_REVISION', 'main')
NARIBOW_ADAPTER = OUT_ROOT / 'hf_adapters' / NARIBOW_ADAPTER_REPO.replace('/', '__')
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

EXPECTED_TRAIN_SHA256 = '{TRAIN_SHA}'
EXPECTED_VAL_SHA256 = '{VAL_SHA}'
MIN_TRAIN_EXAMPLES = {TRAIN_ROWS}
MIN_VAL_EXAMPLES = {VAL_ROWS}
TOKENIZE_ONLY_DRY_RUN = 'public_adapter_probe_eval_only'
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
INIT_ADAPTER_DIR = V217_FINAL_ADAPTER
RUN_TRAIN = os.environ.get('KG1_V220_RUN_TRAIN', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_EVAL = os.environ.get('KG1_V220_RUN_EVAL', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
FORCE_REEVAL = os.environ.get('KG1_V220_FORCE_REEVAL', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_FULL_IF_GATE = os.environ.get('KG1_V220_RUN_FULL_IF_GATE', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
V220_VLLM_PIP_SPEC = os.environ.get('KG1_V220_VLLM_PIP_SPEC', 'vllm==0.20.1')
V220_MAX_TOKENS = int(os.environ.get('KG1_V220_MAX_TOKENS', '3584'))
V220_MAX_MODEL_LEN = int(os.environ.get('KG1_V220_MAX_MODEL_LEN', '8192'))
V220_MAX_NUM_SEQS = int(os.environ.get('KG1_V220_MAX_NUM_SEQS', '64'))
V220_WARMUP_ROWS = int(os.environ.get('KG1_V220_WARMUP_ROWS', '0'))
V220_PROMPT_SUFFIX = os.environ.get('KG1_V220_PROMPT_SUFFIX', '\\nPlease put your final answer inside `\\\\boxed{{}}`. For example: `\\\\boxed{{your answer}}`')
V220_DISABLE_THINKING_DEFAULT = False
runtime_dependency_names = ['causal_conv1d', 'mamba_ssm', 'vllm']

EXPECTED_V194_ADAPTER_BYTES = 4259069440
EXPECTED_V194_ADAPTER_TENSOR_COUNT = 12011
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']
EXPECTED_V217_ADAPTER_MIN_BYTES = 4250000000
EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT = 12000

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, EVAL_OUT, PACKAGE_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V217_FINAL_ADAPTER =', V217_FINAL_ADAPTER, flush=True)
print('NARIBOW_ADAPTER_REPO =', NARIBOW_ADAPTER_REPO, flush=True)
print('NARIBOW_ADAPTER_REVISION =', NARIBOW_ADAPTER_REVISION, flush=True)
print('NARIBOW_ADAPTER =', NARIBOW_ADAPTER, flush=True)
print('V194_VAL_CSV =', V194_VAL_CSV, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('V220_MAX_TOKENS =', V220_MAX_TOKENS, flush=True)
print('V220_MAX_MODEL_LEN =', V220_MAX_MODEL_LEN, flush=True)
print('V220_MAX_NUM_SEQS =', V220_MAX_NUM_SEQS, flush=True)
print('V220_WARMUP_ROWS =', V220_WARMUP_ROWS, flush=True)
print('V220_DISABLE_THINKING_DEFAULT =', V220_DISABLE_THINKING_DEFAULT, flush=True)
print('runtime_dependency_names =', runtime_dependency_names, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if RUN_TRAIN:
    raise RuntimeError('V220 is public adapter probe only; RUN_TRAIN must stay false.')
if V220_DISABLE_THINKING_DEFAULT:
    raise RuntimeError('V220 must keep thinking enabled by default.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')
print('=== V220 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging and heartbeat.
print('=== V220 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.exists() and (path / 'adapter_config.json').exists() and (path / 'adapter_model.safetensors').exists()


def command_heartbeat(stop_event, started_at, heartbeat_s=60):
    while not stop_event.wait(heartbeat_s):
        try:
            import psutil
            mem = psutil.virtual_memory()
            disk = shutil.disk_usage('/content')
            gpu = 'n/a'
            try:
                nvidia = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                gpu = nvidia.stdout.strip().replace('\\n', ' | ') or 'n/a'
            except Exception:
                gpu = 'n/a'
            print(
                '[V220 heartbeat] elapsed_s=%.1f ram_total=%.1fGiB ram_available=%.1fGiB disk_content_free=%.1fGiB disk_content_total=%.1fGiB gpu=[%s]'
                % (
                    time.time() - started_at,
                    mem.total / 1024**3,
                    mem.available / 1024**3,
                    disk.free / 1024**3,
                    disk.total / 1024**3,
                    gpu,
                ),
                flush=True,
            )
        except Exception as exc:
            print('[V220 heartbeat_error]', type(exc).__name__, str(exc), flush=True)


def run_cmd(cmd, cwd=None, env=None, log_path=None, check=True, heartbeat_s=60):
    cwd = pathlib.Path(cwd or '/content')
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    if log_path:
        print('log_path =', log_path, flush=True)
    started = time.time()
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=command_heartbeat, args=(stop_event, started, heartbeat_s), daemon=True)
    heartbeat_thread.start()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail = []
    suppressed = 0
    with (log_path.open('w', encoding='utf-8') if log_path else open(os.devnull, 'w', encoding='utf-8')) as log_handle:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
            tail.append(line.rstrip('\\n'))
            tail[:] = tail[-30:]
            if len(line) < 700:
                print(line, end='', flush=True)
            else:
                suppressed += 1
    rc = proc.wait()
    stop_event.set()
    heartbeat_thread.join(timeout=5)
    elapsed = time.time() - started
    print('returncode =', rc, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if suppressed:
        print('command_output_suppressed_lines =', suppressed, flush=True)
    if rc != 0:
        print('command_tail_on_failure =', '\\n'.join(tail[-25:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and rc != 0:
        raise RuntimeError(f'Command failed rc={rc}: {cmd}')
    return rc


def verify_import_subprocess(module_name, check=True):
    rc = run_cmd(
        [
            sys.executable,
            '-c',
            "import importlib; m=importlib.import_module(%r); print(%r + ' subprocess_version=' + str(getattr(m, '__version__', 'unknown')))" % (module_name, module_name),
        ],
        cwd='/content',
        log_path=OUT_ROOT / f'verify_import_{module_name}.log',
        check=False,
    )
    if check and rc != 0:
        raise RuntimeError(f'import failed for {module_name}')
    return rc


def ensure_vllm_for_eval():
    print('=== V220 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    if verify_import_subprocess('vllm', check=False) != 0:
        print('vLLM subprocess import failed; installing pinned V220_VLLM_PIP_SPEC =', V220_VLLM_PIP_SPEC, flush=True)
        run_cmd(
            [sys.executable, '-m', 'pip', 'install', '-q', V220_VLLM_PIP_SPEC],
            cwd='/content',
            log_path=OUT_ROOT / 'pip_install_vllm.log',
            check=True,
        )
        verify_import_subprocess('vllm', check=True)
    print('=== V220 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)


print('=== V220 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo and compile required scripts.
print('=== V220 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
repo_commit = commit
print('repo_commit =', repo_commit, flush=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('repo_root_on_sys_path =', str(ROOT) in sys.path, flush=True)

required_scripts = [
    ROOT / 'scripts/evaluate_lora_adapter.py',
    ROOT / 'scripts/analyze_eval_predictions.py',
    ROOT / 'scripts/notebook_release_gate.py',
    ROOT / 'src/competition_utils.py',
]
for py_path in required_scripts:
    print('compile_check =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)

train_path = ROOT / 'data/v217/v217_short_answer_train.jsonl'
val_path = ROOT / 'data/v217/v217_short_answer_val.jsonl'
print('train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('val_path =', val_path, 'exists =', val_path.exists(), flush=True)
if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
    raise RuntimeError('V217 train SHA mismatch')
if sha256_file(val_path) != EXPECTED_VAL_SHA256:
    raise RuntimeError('V217 val SHA mismatch')
train_rows_observed = sum(1 for _ in train_path.open(encoding='utf-8'))
val_rows_observed = sum(1 for _ in val_path.open(encoding='utf-8'))
print('train_rows_observed =', train_rows_observed, flush=True)
print('val_rows_observed =', val_rows_observed, flush=True)
if train_rows_observed < MIN_TRAIN_EXAMPLES:
    raise RuntimeError('train row count below MIN_TRAIN_EXAMPLES')
if val_rows_observed < MIN_VAL_EXAMPLES:
    raise RuntimeError('val row count below MIN_VAL_EXAMPLES')
print('=== V220 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, adapter, and weak split audit.
print('=== V220 RUNTIME AND DATA AUDIT START ===', flush=True)
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
torch_audit = json.loads((OUT_ROOT / 'verify_torch_cuda.jsonl').read_text(encoding='utf-8').strip().splitlines()[-1])
cuda_available = bool(torch_audit.get('cuda_available'))
gpu_total_gib = float(torch_audit.get('gpu_total_gib') or 0.0)
print('cuda_available =', cuda_available, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
if not cuda_available:
    raise RuntimeError('CUDA GPU is required for V220 vLLM eval')
if gpu_total_gib < 79.0:
    raise RuntimeError('V220 expects an 80GB GPU class runtime')
usage = shutil.disk_usage('/content')
content_free_gib = usage.free / 1024**3
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if content_free_gib < 80:
    raise RuntimeError('Not enough /content disk free space for model/eval cache')

for optional_module in ['causal_conv1d', 'mamba_ssm']:
    rc = verify_import_subprocess(optional_module, check=False)
    print(optional_module + '_optional_training_import_rc =', rc, flush=True)
print('causal_conv1d and mamba_ssm are audited but not installed here; this notebook is eval-only.', flush=True)

print('V194_ADAPTER complete =', is_complete_adapter_dir(V194_ADAPTER), flush=True)
print('V217_FINAL_ADAPTER complete_optional =', is_complete_adapter_dir(V217_FINAL_ADAPTER), flush=True)
if not V194_VAL_CSV.exists():
    raise FileNotFoundError(V194_VAL_CSV)
if not is_complete_adapter_dir(V194_ADAPTER):
    raise RuntimeError('V194 adapter is missing or incomplete')

try:
    from safetensors import safe_open
except ModuleNotFoundError:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], log_path=OUT_ROOT / 'pip_install_safetensors.log', check=True)
    from safetensors import safe_open
try:
    import huggingface_hub  # noqa: F401
except ModuleNotFoundError:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], log_path=OUT_ROOT / 'pip_install_huggingface_hub.log', check=True)
from huggingface_hub import snapshot_download

if not is_complete_adapter_dir(NARIBOW_ADAPTER) or FORCE_REEVAL:
    print('downloading public adapter from HF:', NARIBOW_ADAPTER_REPO, 'revision =', NARIBOW_ADAPTER_REVISION, flush=True)
    snapshot_download(
        repo_id=NARIBOW_ADAPTER_REPO,
        repo_type='model',
        revision=NARIBOW_ADAPTER_REVISION,
        local_dir=str(NARIBOW_ADAPTER),
        allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'README.md'],
        token=os.environ.get('HF_TOKEN') or None,
    )
print('NARIBOW_ADAPTER complete =', is_complete_adapter_dir(NARIBOW_ADAPTER), flush=True)
if not is_complete_adapter_dir(NARIBOW_ADAPTER):
    raise RuntimeError('Naribow public adapter is missing or incomplete after download')

for adapter_name, adapter_path in [('v194', V194_ADAPTER), ('naribow_public', NARIBOW_ADAPTER)]:
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
        if weight_bytes < EXPECTED_V217_ADAPTER_MIN_BYTES:
            raise RuntimeError('Naribow public adapter size below expected floor')
        if tensor_count < EXPECTED_V217_ADAPTER_MIN_TENSOR_COUNT:
            raise RuntimeError('Naribow public adapter tensor count below expected floor')

from src.competition_utils import classify_puzzle
full_df = pd.read_csv(V194_VAL_CSV)
if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
    raise RuntimeError('V194 validation CSV must contain prompt and answer')
if 'type' not in full_df.columns:
    full_df['type'] = full_df['prompt'].map(classify_puzzle)
full_eval_csv = EVAL_OUT / 'v220_full_947.csv'
weak_eval_csv = EVAL_OUT / 'v220_weak_315.csv'
strong_eval_csv = EVAL_OUT / 'v220_strong_632.csv'
weak_df = full_df[full_df['type'].isin(['equation_transform', 'bit_manipulation'])].copy()
strong_df = full_df[~full_df['type'].isin(['equation_transform', 'bit_manipulation'])].copy()
full_df.to_csv(full_eval_csv, index=False)
weak_df.to_csv(weak_eval_csv, index=False)
strong_df.to_csv(strong_eval_csv, index=False)
print('full_rows =', len(full_df), 'path =', full_eval_csv, flush=True)
print('weak_rows =', len(weak_df), 'path =', weak_eval_csv, flush=True)
print('strong_rows =', len(strong_df), 'path =', strong_eval_csv, flush=True)
print('per_family_counts =', full_df['type'].value_counts().sort_index().to_dict(), flush=True)
if len(weak_df) != 315 or len(full_df) != 947 or len(strong_df) != 632:
    raise RuntimeError('validation split row count mismatch')
if V217_WEAK_REPORT.exists():
    print('v217_weak_report_existing =', json.dumps(read_json(V217_WEAK_REPORT), indent=2, sort_keys=True), flush=True)
else:
    print('v217_weak_report_existing = missing', V217_WEAK_REPORT, flush=True)
if V218_DECODE_REPORT.exists():
    print('v218_decode_report_existing =', json.dumps(read_json(V218_DECODE_REPORT), indent=2, sort_keys=True), flush=True)
else:
    print('v218_decode_report_existing = missing', V218_DECODE_REPORT, flush=True)
print('=== V220 RUNTIME AND DATA AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: weak-only public adapter probe with thinking enabled.
print('=== V220 WEAK PUBLIC ADAPTER EVAL START ===', flush=True)
weak_gate_pass_for_full = False
best_candidate = None
candidate_reports = []
candidate_rows = []

if not RUN_EVAL:
    print('RUN_EVAL is false; skipping V220 public adapter probe.', flush=True)
else:
    ensure_vllm_for_eval()
    candidates = [
        {
            'name': 'naribow_public_think1_mtok3584',
            'adapter': NARIBOW_ADAPTER,
            'max_tokens': V220_MAX_TOKENS,
            'max_model_len': V220_MAX_MODEL_LEN,
            'max_num_seqs': V220_MAX_NUM_SEQS,
        },
    ]
    for candidate in candidates:
        label = 'v220_' + candidate['name'] + '_weak'
        eval_dir = EVAL_OUT / candidate['name']
        report_path = eval_dir / f'{label}_eval_report.json'
        per_task_path = eval_dir / f'{label}_per_task.csv'
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_new_eval = FORCE_REEVAL or not report_path.exists()
        print('candidate =', candidate, flush=True)
        print('run_new_eval =', run_new_eval, flush=True)
        if run_new_eval:
            cmd = [
                sys.executable,
                str(ROOT / 'scripts/evaluate_lora_adapter.py'),
                '--solution-csv', str(weak_eval_csv),
                '--questions-csv', str(weak_eval_csv),
                '--adapter', str(candidate['adapter']),
                '--base-model-path', MODEL_NAME,
                '--label', label,
                '--seed', '42',
                '--limit', '0',
                '--output-dir', str(eval_dir),
                '--max-tokens', str(candidate['max_tokens']),
                '--max-model-len', str(candidate['max_model_len']),
                '--max-num-seqs', str(candidate['max_num_seqs']),
                '--warmup-rows', str(V220_WARMUP_ROWS),
                '--prompt-suffix', V220_PROMPT_SUFFIX,
            ]
            rc = run_cmd(cmd, cwd=ROOT, log_path=eval_dir / 'weak_decode_eval.log', check=True)
            print('weak eval returncode for', candidate['name'], '=', rc, flush=True)
        else:
            print('reusing report_path =', report_path, flush=True)
        report = read_json(report_path)
        per_task = pd.read_csv(per_task_path)
        by_task = {row['task_type']: int(row['correct']) for _, row in per_task.iterrows()}
        row = {
            'name': candidate['name'],
            'adapter': str(candidate['adapter']),
            'correct': int(report.get('correct', 0)),
            'accuracy': float(report.get('accuracy', 0.0)),
            'truncated': int(report.get('truncated', 999)),
            'equation_transform_correct': by_task.get('equation_transform', 0),
            'bit_manipulation_correct': by_task.get('bit_manipulation', 0),
            'completion_tokens': int(report.get('completion_tokens', 0)),
            'tokens_per_second': float(report.get('tokens_per_second', 0.0)),
            'report_json': str(report_path),
        }
        candidate_reports.append(report)
        candidate_rows.append(row)
        print('candidate_summary =', json.dumps(row, indent=2, sort_keys=True), flush=True)
        print('candidate_per_task =', per_task.to_string(index=False), flush=True)
    candidates_csv = EVAL_OUT / 'v220_public_adapter_probe_candidates.csv'
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
    print('candidates_csv =', candidates_csv, flush=True)
    best_candidate = sorted(
        candidate_rows,
        key=lambda item: (
            int(item['correct']),
            int(item['equation_transform_correct']),
            int(item['bit_manipulation_correct']),
            -int(item['truncated']),
        ),
        reverse=True,
    )[0] if candidate_rows else None
    print('best_candidate =', json.dumps(best_candidate, indent=2, sort_keys=True), flush=True)
    if best_candidate:
        weak_gate_pass_for_full = (
            int(best_candidate['correct']) >= WEAK_MIN_FOR_FULL
            and int(best_candidate['equation_transform_correct']) >= WEAK_EQ_MIN_FOR_FULL
            and int(best_candidate['bit_manipulation_correct']) >= WEAK_BIT_MIN_FOR_FULL
            and int(best_candidate['truncated']) <= WEAK_MAX_TRUNC_FOR_FULL
        )
    print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V220 WEAK PUBLIC ADAPTER EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: optional full eval gate, default blocked.
print('=== V220 FULL EVAL GATE START ===', flush=True)
full_report = None
full_candidate_gate = False
if not weak_gate_pass_for_full:
    print('Weak gate failed or did not run; full eval blocked.', flush=True)
    print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
elif not RUN_FULL_IF_GATE:
    print('Weak gate passed, but RUN_FULL_IF_GATE is false. Full eval is blocked by default to avoid accidental GPU spend.', flush=True)
else:
    best_name = str(best_candidate['name'])
    adapter_path = NARIBOW_ADAPTER if best_name.startswith('naribow') else V194_ADAPTER
    full_label = 'v220_' + best_name + '_full'
    full_dir = EVAL_OUT / (best_name + '_full_eval')
    full_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(full_eval_csv),
            '--questions-csv', str(full_eval_csv),
            '--adapter', str(adapter_path),
            '--base-model-path', MODEL_NAME,
            '--label', full_label,
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(full_dir),
            '--max-tokens', str(V220_MAX_TOKENS),
            '--max-model-len', str(V220_MAX_MODEL_LEN),
            '--max-num-seqs', str(V220_MAX_NUM_SEQS),
            '--warmup-rows', str(V220_WARMUP_ROWS),
            '--prompt-suffix', V220_PROMPT_SUFFIX,
        ],
        cwd=ROOT,
        log_path=full_dir / 'full_eval.log',
        check=True,
    )
    print('full eval returncode =', rc, flush=True)
    full_report = read_json(full_dir / f'{full_label}_eval_report.json')
    full_candidate_gate = int(full_report.get('correct', 0)) >= FULL_MIN_CANDIDATE and int(full_report.get('truncated', 999)) <= FULL_MAX_TRUNC
    print('full_report =', json.dumps(full_report, indent=2, sort_keys=True), flush=True)
    print('full_candidate_gate =', full_candidate_gate, flush=True)
print('=== V220 FULL EVAL GATE END ===', flush=True)
"""
        ),
        code(
            """# CELL: write V220 final manifest. No submit.
print('=== V220 FINAL MANIFEST START ===', flush=True)
decision = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_train': RUN_TRAIN,
    'run_eval': RUN_EVAL,
    'run_full_if_gate': RUN_FULL_IF_GATE,
    'candidate_rows': candidate_rows,
    'best_candidate': best_candidate,
    'weak_gate_pass_for_full': bool(weak_gate_pass_for_full),
    'full_report': full_report,
    'full_candidate_gate': bool(full_candidate_gate),
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
}
if not weak_gate_pass_for_full:
    decision['roadmap_next'] = 'Do not full-eval or submit. If this adapter fails weak gate, build V221 solver-trace data audit for bit/equation families.'
elif not RUN_FULL_IF_GATE:
    decision['roadmap_next'] = 'Enable KG1_V220_RUN_FULL_IF_GATE=1 only if you intentionally want full eval spend.'
elif not full_candidate_gate:
    decision['roadmap_next'] = 'Full candidate failed. Do not package or submit.'
else:
    decision['roadmap_next'] = 'Manual review required before packaging; notebook still has hard submit lock.'
manifest_path = OUT_ROOT / 'v220_public_adapter_probe_manifest.json'
manifest_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('=== V220 FINAL MANIFEST END ===', flush=True)
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

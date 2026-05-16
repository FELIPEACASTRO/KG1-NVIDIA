#!/usr/bin/env python3
"""Build the V225 equation decode sweep Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V225_EQUATION_DECODE_SWEEP_COLAB.ipynb")
BRANCH = "v225-equation-decode-sweep"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V225_EQUATION_DECODE_SWEEP_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V225_EQUATION_DECODE_SWEEP_COLAB.ipynb"
)

TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v225-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V225 Equation Decode Sweep Colab

Purpose: run an equation-only decode/prompt sweep on the safe V221 candidate adapters, then simulate whether combining the best equation result with the V217/V221 bit baseline could clear the weak gate.

This notebook is evaluation-only. It does not train, does not package, does not run full eval automatically, and does not submit to Kaggle.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V225 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V225 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration, gates, decode variants, and hard submit lock.
print('=== V225 CONFIG START ===', flush=True)
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

VERSION = 'V225_EQUATION_DECODE_SWEEP_20260509'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V225')
OUT_ROOT = DRIVE_ROOT / 'output_v225_equation_decode_sweep'
SWEEP_ROOT = OUT_ROOT / 'sweeps'
ANALYSIS_OUT = OUT_ROOT / 'analysis_v225_equation_decode_sweep'

MODEL_NAME = os.environ.get('KG1_V225_MODEL_NAME', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
V225_MAX_MODEL_LEN = int(os.environ.get('KG1_V225_MAX_MODEL_LEN', '8192'))
V225_MAX_NUM_SEQS = int(os.environ.get('KG1_V225_MAX_NUM_SEQS', '64'))
V225_WARMUP_ROWS = int(os.environ.get('KG1_V225_WARMUP_ROWS', '0'))
V225_VLLM_PIP_SPEC = os.environ.get('KG1_V225_VLLM_PIP_SPEC', 'vllm==0.20.1')

V221_EVAL_OUT = pathlib.Path(os.environ.get('KG1_V225_V221_EVAL_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab'))
V221_WEAK_CSV = pathlib.Path(os.environ.get('KG1_V225_V221_WEAK_CSV', str(V221_EVAL_OUT / 'v221_weak_315.csv')))
V221_READY_CANDIDATES_JSON = pathlib.Path(os.environ.get('KG1_V225_V221_READY_CANDIDATES_JSON', str(V221_EVAL_OUT / 'v221_ready_candidates.json')))
V221_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get('KG1_V225_V221_BATCH_SUMMARY_JSON', str(V221_EVAL_OUT / 'batch_candidate_summary.json')))

V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
INIT_ADAPTER_DIR = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter')
EXPECTED_TRAIN_SHA256 = '__TRAIN_SHA__'
EXPECTED_VAL_SHA256 = '__VAL_SHA__'
MIN_TRAIN_EXAMPLES = 10206
MIN_VAL_EXAMPLES = 681
TOKENIZE_ONLY_DRY_RUN = True
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True

RUN_TRAIN = os.environ.get('KG1_V225_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_SWEEP = os.environ.get('KG1_V225_RUN_SWEEP', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_ANALYSIS = os.environ.get('KG1_V225_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_EVAL = RUN_SWEEP
RUN_FULL_IF_GATE = False
RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS = os.environ.get('KG1_V225_RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
FORCE_SWEEP = os.environ.get('KG1_V225_FORCE_SWEEP', '0').strip().lower() in {'1', 'true', 'yes', 'on'}

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
WEAK_MAX_TRUNC_FOR_FULL = 0
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

CANDIDATE_NAMES = [
    'v217_final_existing',
    'v194_protected_baseline',
    'dgxchen_trained_adapter',
    'konbu17_exp026_s012_lora',
    'kienngx_tinker_adapter',
]

DECODE_VARIANTS = [
    {
        'name': 'no_think_default_suffix',
        'disable_thinking': True,
        'no_prompt_suffix': False,
        'prompt_suffix': '',
        'max_tokens': 7680,
    },
    {
        'name': 'think_no_prompt_suffix',
        'disable_thinking': False,
        'no_prompt_suffix': True,
        'prompt_suffix': '',
        'max_tokens': 7680,
    },
    {
        'name': 'think_strict_boxed',
        'disable_thinking': False,
        'no_prompt_suffix': False,
        'prompt_suffix': '\\nReturn exactly one line in this format: `\\\\boxed{answer}`.',
        'max_tokens': 7680,
    },
    {
        'name': 'no_think_strict_boxed',
        'disable_thinking': True,
        'no_prompt_suffix': False,
        'prompt_suffix': '\\nReturn exactly one line in this format: `\\\\boxed{answer}`.',
        'max_tokens': 7680,
    },
]
variant_filter = os.environ.get('KG1_V225_VARIANT_FILTER', '').strip()
if variant_filter:
    wanted_variants = {item.strip() for item in variant_filter.split(',') if item.strip()}
    DECODE_VARIANTS = [item for item in DECODE_VARIANTS if item['name'] in wanted_variants]

for path in [DRIVE_ROOT, OUT_ROOT, SWEEP_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('SWEEP_ROOT =', SWEEP_ROOT, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('MODEL_NAME =', MODEL_NAME, flush=True)
print('V225_MAX_MODEL_LEN =', V225_MAX_MODEL_LEN, flush=True)
print('V225_MAX_NUM_SEQS =', V225_MAX_NUM_SEQS, flush=True)
print('V225_WARMUP_ROWS =', V225_WARMUP_ROWS, flush=True)
print('V225_VLLM_PIP_SPEC =', V225_VLLM_PIP_SPEC, flush=True)
print('V221_WEAK_CSV =', V221_WEAK_CSV, flush=True)
print('V221_READY_CANDIDATES_JSON =', V221_READY_CANDIDATES_JSON, flush=True)
print('V221_BATCH_SUMMARY_JSON =', V221_BATCH_SUMMARY_JSON, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_SWEEP =', RUN_SWEEP, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS =', RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS, flush=True)
print('FORCE_SWEEP =', FORCE_SWEEP, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('CANDIDATE_NAMES =', json.dumps(CANDIDATE_NAMES, indent=2), flush=True)
print('DECODE_VARIANTS =', json.dumps(DECODE_VARIANTS, indent=2, sort_keys=True), flush=True)
if RUN_TRAIN:
    raise RuntimeError('V225 is decode-sweep only; RUN_TRAIN must stay false.')
if RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS:
    raise RuntimeError('Full weak confirmation is intentionally blocked in V225; create a separate confirmation notebook after this sweep.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V225.')
if not DECODE_VARIANTS:
    raise RuntimeError('No decode variants selected.')
print('=== V225 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging, heartbeat, hashes, and vLLM dependency check.
print('=== V225 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return path


def heartbeat_line(started_at):
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total = mem.total / 1024**3
        ram_available = mem.available / 1024**3
    except Exception:
        ram_total = 0.0
        ram_available = 0.0
    disk = shutil.disk_usage('/content')
    gpu = 'unavailable'
    try:
        gpu = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        ).strip().replace('\\n', '; ')
    except Exception as exc:
        gpu = f'nvidia-smi-error={exc!r}'
    return (
        f"[V225 heartbeat] elapsed_s={time.time() - started_at:.1f} "
        f"ram_total={ram_total:.1f}GiB ram_available={ram_available:.1f}GiB "
        f"disk_content_free={disk.free / 1024**3:.1f}GiB disk_content_total={disk.total / 1024**3:.1f}GiB "
        f"gpu=[{gpu}]"
    )


def run_cmd(cmd, cwd=None, log_path=None, check=True, heartbeat_s=0, suppress_after_lines=240):
    cwd = pathlib.Path(cwd or '/content')
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = ' '.join(map(str, cmd))
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', printable, flush=True)
    if log_path:
        print('log_path =', log_path, flush=True)
    started = time.time()
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    captured = []
    last_heartbeat = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        print(line, end='', flush=True)
        if heartbeat_s and time.time() - last_heartbeat >= heartbeat_s:
            print(heartbeat_line(started), flush=True)
            last_heartbeat = time.time()
    returncode = proc.wait()
    elapsed = time.time() - started
    output = ''.join(captured)
    if log_path:
        log_path.write_text(output, encoding='utf-8')
    lines = output.splitlines()
    if suppress_after_lines and len(lines) > suppress_after_lines:
        print('command_output_suppressed_lines =', len(lines) - suppress_after_lines, flush=True)
    print('returncode =', returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if returncode and lines:
        print('command_tail_on_failure =', '\\n'.join(lines[-40:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and returncode:
        raise RuntimeError(f'command failed rc={returncode}: {printable}')
    return returncode


def verify_import(module_name, log_name):
    cmd = [
        sys.executable,
        '-c',
        "import importlib; m=importlib.import_module('" + module_name + "'); print('" + module_name + " subprocess_version=' + str(getattr(m, '__version__', 'unknown')))",
    ]
    return run_cmd(cmd, cwd='/content', log_path=OUT_ROOT / log_name, check=False)


def ensure_vllm_for_eval():
    print('=== V225 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    rc = verify_import('vllm', 'verify_import_vllm.log')
    if rc != 0:
        print('vLLM subprocess import failed; installing pinned V225_VLLM_PIP_SPEC =', V225_VLLM_PIP_SPEC, flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', V225_VLLM_PIP_SPEC], cwd='/content', log_path=OUT_ROOT / 'pip_install_vllm.log', check=True, heartbeat_s=60)
        verify_rc = verify_import('vllm', 'verify_import_vllm.log')
        if verify_rc != 0:
            raise RuntimeError('vLLM import still failed after install.')
    print('=== V225 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)


print('=== V225 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and validate static data hashes.
print('=== V225 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
print('repo_commit =', repo_commit, flush=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
compile_targets = [
    ROOT / 'src/competition_utils.py',
    ROOT / 'scripts/evaluate_lora_adapter.py',
    ROOT / 'scripts/evaluate_lora_adapters_batch.py',
    ROOT / 'scripts/analyze_v225_equation_decode_sweep.py',
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
if train_path.exists():
    observed_train_sha256 = sha256_file(train_path)
    train_rows = sum(1 for _ in train_path.open('r', encoding='utf-8'))
    print('observed_train_sha256 =', observed_train_sha256, flush=True)
    print('train_rows =', train_rows, flush=True)
    if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f'train sha mismatch: {observed_train_sha256} != {EXPECTED_TRAIN_SHA256}')
    if train_rows < MIN_TRAIN_EXAMPLES:
        raise RuntimeError(f'train row count too low: {train_rows} < {MIN_TRAIN_EXAMPLES}')
if val_path.exists():
    observed_val_sha256 = sha256_file(val_path)
    val_rows = sum(1 for _ in val_path.open('r', encoding='utf-8'))
    print('observed_val_sha256 =', observed_val_sha256, flush=True)
    print('val_rows =', val_rows, flush=True)
    if observed_val_sha256 != EXPECTED_VAL_SHA256:
        raise RuntimeError(f'val sha mismatch: {observed_val_sha256} != {EXPECTED_VAL_SHA256}')
    if val_rows < MIN_VAL_EXAMPLES:
        raise RuntimeError(f'val row count too low: {val_rows} < {MIN_VAL_EXAMPLES}')
print('=== V225 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, Drive artifact, adapter, and dependency audit.
print('=== V225 RUNTIME ARTIFACT AUDIT START ===', flush=True)
try:
    import torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    cuda_available = bool(torch.cuda.is_available())
    gpu_total_gib = float(props.total_memory / 1024**3) if props else 0.0
    gpu_name = props.name if props else ''
except Exception as exc:
    cuda_available = False
    gpu_total_gib = 0.0
    gpu_name = ''
    print('torch_cuda_probe_error =', repr(exc), flush=True)
content_free_gib = shutil.disk_usage('/content').free / 1024**3
print('cuda_available =', cuda_available, flush=True)
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if not cuda_available:
    raise RuntimeError('CUDA is required for V225 vLLM sweep.')
if gpu_total_gib < 70:
    raise RuntimeError(f'GPU memory too small for V225 sweep: {gpu_total_gib:.2f} GiB')
if content_free_gib < 40:
    raise RuntimeError(f'/content free disk too low: {content_free_gib:.2f} GiB')
for module_name in ['causal_conv1d', 'mamba_ssm']:
    try:
        module = importlib.import_module(module_name)
        print(module_name, 'version =', getattr(module, '__version__', 'unknown'), flush=True)
    except Exception as exc:
        print(module_name, 'import_warning =', repr(exc), flush=True)
print('V221_WEAK_CSV exists =', V221_WEAK_CSV.exists(), flush=True)
print('V221_READY_CANDIDATES_JSON exists =', V221_READY_CANDIDATES_JSON.exists(), flush=True)
print('V221_BATCH_SUMMARY_JSON exists =', V221_BATCH_SUMMARY_JSON.exists(), flush=True)
for required_path in [V221_WEAK_CSV, V221_READY_CANDIDATES_JSON, V221_BATCH_SUMMARY_JSON]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)
print('V194_ADAPTER exists =', V194_ADAPTER.exists(), flush=True)
print('V194 adapter_config.json exists =', (V194_ADAPTER / 'adapter_config.json').exists(), flush=True)
print('V194 adapter_model.safetensors exists =', (V194_ADAPTER / 'adapter_model.safetensors').exists(), flush=True)
if (V194_ADAPTER / 'adapter_config.json').exists():
    v194_config = read_json(V194_ADAPTER / 'adapter_config.json')
    print('V194 target_modules =', v194_config.get('target_modules'), flush=True)
    print('V194 target_parameters =', v194_config.get('target_parameters'), flush=True)
print('INIT_ADAPTER_DIR exists =', INIT_ADAPTER_DIR.exists(), flush=True)
print('INIT adapter_config.json exists =', (INIT_ADAPTER_DIR / 'adapter_config.json').exists(), flush=True)
print('INIT adapter_model.safetensors exists =', (INIT_ADAPTER_DIR / 'adapter_model.safetensors').exists(), flush=True)
if (INIT_ADAPTER_DIR / 'adapter_config.json').exists():
    init_config = read_json(INIT_ADAPTER_DIR / 'adapter_config.json')
    print('INIT target_modules =', init_config.get('target_modules'), flush=True)
    print('INIT target_parameters =', init_config.get('target_parameters'), flush=True)
print('=== V225 RUNTIME ARTIFACT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: prepare equation-only CSV and safe candidate subset.
print('=== V225 EQUATION SUBSET PREP START ===', flush=True)
import pandas as pd
from src.competition_utils import classify_puzzle

equation_csv = OUT_ROOT / 'v225_equation_155.csv'
candidate_subset_json = OUT_ROOT / 'v225_safe_candidate_subset.json'
weak_df = pd.read_csv(V221_WEAK_CSV)
if 'id' not in weak_df.columns:
    raise RuntimeError('V221 weak CSV must include id column.')
if 'prompt' not in weak_df.columns:
    raise RuntimeError('V221 weak CSV must include prompt column.')
if 'type' in weak_df.columns:
    weak_df['family'] = weak_df['type'].astype(str)
elif 'task_type' in weak_df.columns:
    weak_df['family'] = weak_df['task_type'].astype(str)
else:
    weak_df['family'] = weak_df['prompt'].map(classify_puzzle)
equation_df = weak_df[weak_df['family'].eq('equation_transform')].copy()
bit_df = weak_df[weak_df['family'].eq('bit_manipulation')].copy()
print('weak_rows =', len(weak_df), flush=True)
print('equation_rows =', len(equation_df), flush=True)
print('bit_rows =', len(bit_df), flush=True)
if len(weak_df) != 315:
    raise RuntimeError(f'unexpected weak row count: {len(weak_df)} != 315')
if len(equation_df) != 155:
    raise RuntimeError(f'unexpected equation row count: {len(equation_df)} != 155')
if len(bit_df) != 160:
    raise RuntimeError(f'unexpected bit row count: {len(bit_df)} != 160')
equation_df.to_csv(equation_csv, index=False)
ready_payload = read_json(V221_READY_CANDIDATES_JSON)
ready_candidates = ready_payload.get('candidates', ready_payload) if isinstance(ready_payload, dict) else ready_payload
if not isinstance(ready_candidates, list):
    raise RuntimeError('ready candidates JSON must be a list or candidates dict.')
ready_by_name = {str(item.get('name')): item for item in ready_candidates if isinstance(item, dict)}
missing = [name for name in CANDIDATE_NAMES if name not in ready_by_name]
print('available_candidate_names =', sorted(ready_by_name), flush=True)
print('requested_candidate_names =', CANDIDATE_NAMES, flush=True)
print('missing_candidate_names =', missing, flush=True)
if missing:
    raise RuntimeError(f'missing requested candidates: {missing}')
candidate_subset = []
for name in CANDIDATE_NAMES:
    item = dict(ready_by_name[name])
    adapter_path = pathlib.Path(str(item.get('adapter') or item.get('adapter_path') or ''))
    print('candidate_selected =', json.dumps({'name': name, 'adapter': str(adapter_path), 'adapter_exists': adapter_path.exists()}, sort_keys=True), flush=True)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    if not (adapter_path / 'adapter_config.json').exists():
        raise FileNotFoundError(adapter_path / 'adapter_config.json')
    if not ((adapter_path / 'adapter_model.safetensors').exists() or (adapter_path / 'adapter_model.bin').exists()):
        raise FileNotFoundError(f'adapter weights missing under {adapter_path}')
    item['name'] = name
    item['adapter'] = str(adapter_path)
    candidate_subset.append(item)
write_json(candidate_subset_json, {'candidates': candidate_subset})
print('equation_csv =', equation_csv, flush=True)
print('candidate_subset_json =', candidate_subset_json, flush=True)
print('candidate_subset_count =', len(candidate_subset), flush=True)
print('=== V225 EQUATION SUBSET PREP END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V225 equation-only decode sweep.
print('=== V225 EQUATION DECODE SWEEP START ===', flush=True)
if not RUN_SWEEP:
    print('RUN_SWEEP is false; reusing existing sweep outputs.', flush=True)
else:
    ensure_vllm_for_eval()
    for variant in DECODE_VARIANTS:
        variant_name = variant['name']
        variant_dir = SWEEP_ROOT / f'variant_{variant_name}'
        summary_json = variant_dir / 'batch_candidate_summary.json'
        run_new_variant = FORCE_SWEEP or not summary_json.exists()
        print('variant_start =', json.dumps(variant, sort_keys=True), flush=True)
        print('variant_dir =', variant_dir, flush=True)
        print('summary_json =', summary_json, 'exists =', summary_json.exists(), flush=True)
        print('run_new_variant =', run_new_variant, flush=True)
        if not run_new_variant:
            print('reusing variant summary:', summary_json, flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapters_batch.py'),
            '--solution-csv', str(equation_csv),
            '--questions-csv', str(equation_csv),
            '--candidates-json', str(candidate_subset_json),
            '--base-model-path', MODEL_NAME,
            '--label-prefix', f'v225_eq_{variant_name}',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(variant_dir),
            '--max-tokens', str(int(variant['max_tokens'])),
            '--max-model-len', str(V225_MAX_MODEL_LEN),
            '--max-num-seqs', str(V225_MAX_NUM_SEQS),
            '--warmup-rows', str(V225_WARMUP_ROWS),
            '--continue-on-error',
        ]
        if variant.get('disable_thinking'):
            cmd.append('--disable-thinking')
        if variant.get('no_prompt_suffix'):
            cmd.append('--no-prompt-suffix')
        elif variant.get('prompt_suffix'):
            cmd.extend(['--prompt-suffix', str(variant['prompt_suffix'])])
        rc = run_cmd(cmd, cwd=ROOT, log_path=variant_dir / 'equation_batch_eval.log', check=True, heartbeat_s=60)
        print('variant_returncode =', rc, flush=True)
print('sweep_root =', SWEEP_ROOT, flush=True)
print('=== V225 EQUATION DECODE SWEEP END ===', flush=True)
"""
        ),
        code(
            """# CELL: analyze V225 sweep and decide next gate action.
print('=== V225 SWEEP ANALYSIS START ===', flush=True)
manifest_path = ANALYSIS_OUT / 'v225_equation_decode_sweep_manifest.json'
if not RUN_ANALYSIS:
    print('RUN_ANALYSIS is false; skipping analysis.', flush=True)
else:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v225_equation_decode_sweep.py'),
        '--v221-batch-summary-json', str(V221_BATCH_SUMMARY_JSON),
        '--sweep-root', str(SWEEP_ROOT),
        '--output-dir', str(ANALYSIS_OUT),
        '--baseline', 'v217_final_existing',
        '--label', 'v225_equation_decode_sweep',
        '--weak-total-min', str(WEAK_MIN_FOR_FULL),
        '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
        '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
        '--weak-trunc-max', str(WEAK_MAX_TRUNC_FOR_FULL),
    ]
    rc = run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v225_equation_decode_sweep.log', check=True)
    print('v225 analysis returncode =', rc, flush=True)
analysis_manifest = read_json(manifest_path)
decision = analysis_manifest.get('decision', {})
variant_summary = analysis_manifest.get('variant_summary', [])
simulated_weak_gate_pass_for_full = decision.get('decision') == 'equation_decode_candidate_found_confirm_full_weak'
print('analysis_manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('variant_summary_top =', json.dumps(variant_summary[:8], indent=2, sort_keys=True), flush=True)
print('simulated_weak_gate_pass_for_full =', simulated_weak_gate_pass_for_full, flush=True)
print('=== V225 SWEEP ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval/package hard block and final manifest.
print('=== V225 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(ANALYSIS_OUT / 'v225_equation_decode_sweep_manifest.json')
decision = analysis_manifest.get('decision', {})
simulated_weak_gate_pass_for_full = decision.get('decision') == 'equation_decode_candidate_found_confirm_full_weak'
weak_gate_pass_for_full = False
full_candidate_gate = False
print('simulated_weak_gate_pass_for_full =', simulated_weak_gate_pass_for_full, flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('full_candidate_gate =', full_candidate_gate, flush=True)
print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('Full eval is blocked by default to avoid accidental GPU spend.', flush=True)
print('No package and no Kaggle submit can be created in V225.', flush=True)
if RUN_TRAIN or RUN_FULL_IF_GATE or RUN_FULL_WEAK_CONFIRM_IF_EQUATION_PASS or ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('V225 hard block violated. Kaggle submission is disabled.')
final_manifest_path = OUT_ROOT / 'v225_equation_decode_sweep_final_manifest.json'
final_manifest = {
    'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'version': VERSION,
    'repo_branch': REPO_BRANCH,
    'repo_commit': repo_commit,
    'model_name': MODEL_NAME,
    'v221_weak_csv': str(V221_WEAK_CSV),
    'v221_ready_candidates_json': str(V221_READY_CANDIDATES_JSON),
    'v221_batch_summary_json': str(V221_BATCH_SUMMARY_JSON),
    'sweep_root': str(SWEEP_ROOT),
    'analysis_out': str(ANALYSIS_OUT),
    'candidate_names': CANDIDATE_NAMES,
    'decode_variants': DECODE_VARIANTS,
    'thresholds': {
        'weak_total': WEAK_MIN_FOR_FULL,
        'weak_equation_transform': WEAK_EQ_MIN_FOR_FULL,
        'weak_bit_manipulation': WEAK_BIT_MIN_FOR_FULL,
        'weak_truncated': WEAK_MAX_TRUNC_FOR_FULL,
        'full_min_candidate': FULL_MIN_CANDIDATE,
        'full_max_trunc': FULL_MAX_TRUNC,
    },
    'simulated_weak_gate_pass_for_full': simulated_weak_gate_pass_for_full,
    'weak_gate_pass_for_full': weak_gate_pass_for_full,
    'full_candidate_gate': full_candidate_gate,
    'submit_authorized': False,
    'decision': decision,
    'analysis_outputs': analysis_manifest.get('outputs', {}),
    'roadmap_next': decision.get('next_action', 'Inspect V225 analysis outputs before any full eval.'),
}
write_json(final_manifest_path, final_manifest)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('roadmap_next =', final_manifest['roadmap_next'], flush=True)
print('=== V225 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")


if __name__ == "__main__":
    main()

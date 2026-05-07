#!/usr/bin/env python3
"""Build the V214 H100 micro-replay Colab notebook.

The notebook is intentionally submit-disabled. It bootstraps the V214 dataset,
audits the V194 adapter, runs a dry-run recipe check, and can run a one-step
continuation only when `KG1_V214_RUN_TRAIN=1` is set in the Colab environment.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb")
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
V194_ADAPTER_DRIVE = "/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter"
V194_VAL_CSV_DRIVE = (
    "/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/"
    "validation/official_train_seed42_stratified10_val.csv"
)

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v214-{prefix}-{_CELL_COUNTER:02d}"


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


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_first_existing_text(*paths: str) -> str:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"None of these files exist: {paths}")


def embedded_payload() -> str:
    files = {
        "src/__init__.py": read_text("src/__init__.py") if Path("src/__init__.py").exists() else "",
        "src/competition_utils.py": read_text("src/competition_utils.py"),
        "scripts/evaluate_lora_adapter.py": read_text("scripts/evaluate_lora_adapter.py"),
        "scripts/hf_job_train_v90.py": read_first_existing_text(
            "scripts/hf_job_train_v90.py",
            ".claude/worktrees/competent-shamir/scripts/hf_job_train_v90.py",
        ),
        "data/v214/v214_micro_train.jsonl": read_text("data/v214/v214_micro_train.jsonl"),
        "data/v214/v214_micro_val.jsonl": read_text("data/v214/v214_micro_val.jsonl"),
        "data/v214/v214_micro_replay_candidate_manifest.json": read_text(
            "data/v214/v214_micro_replay_candidate_manifest.json"
        ),
        "data/v214/v214_micro_split_manifest.json": read_text("data/v214/v214_micro_split_manifest.json"),
        "artifacts/V194_ADAPTER_AUDIT_2026-05-06.md": read_text("artifacts/V194_ADAPTER_AUDIT_2026-05-06.md"),
        "artifacts/ROADMAP_UPDATE_KG1_V214_PROBE_SOLVER_RESULTS_2026-05-06.md": read_text(
            "artifacts/ROADMAP_UPDATE_KG1_V214_PROBE_SOLVER_RESULTS_2026-05-06.md"
        ),
    }
    return json.dumps(files, ensure_ascii=False, indent=2)


def bootstrap_cell() -> str:
    payload = embedded_payload()
    return f"""# CELL: bootstrap V214 files into /content/kg1.
print('=== V214 BOOTSTRAP START ===', flush=True)
import json
import pathlib
import py_compile

ROOT = pathlib.Path('/content/kg1')
FILES = json.loads({json.dumps(payload)})
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline='\\n')
    print(f'wrote {{path}} bytes={{path.stat().st_size}}', flush=True)

for rel in [
    'src/competition_utils.py',
    'scripts/evaluate_lora_adapter.py',
    'scripts/hf_job_train_v90.py',
]:
    py_compile.compile(str(ROOT / rel), doraise=True)
    print(f'compiled {{rel}}', flush=True)

print('=== V214 BOOTSTRAP END ===', flush=True)
"""


def build_notebook() -> dict:
    cells = [
        md(
            """# KG1 V214 H100 Micro-Replay Colab

Purpose: execute the next V214 roadmap gate without submitting anything.

This notebook:

- bootstraps the local V214 train/val replay files into `/content/kg1`;
- audits the V194 adapter on Google Drive before touching training;
- validates the V214 dataset hashes, row counts, and family mix;
- runs a dry-run model/LoRA trainability check;
- runs the one-step V194 continuation only if `KG1_V214_RUN_TRAIN=1`;
- evaluates weak rows first and only runs full 947 eval if the weak gate passes;
- never packages and never submits to Kaggle.

Expected Colab runtime: H100 preferred. A100 may work for dry-run/eval, but this
notebook is designed around the existing H100 road map.
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V214 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V214 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration and hard submit lock.
print('=== V214 CONFIG START ===', flush=True)
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

VERSION = 'V214_H100_MICRO_REPLAY_20260506'
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V214')
OUT_ROOT = DRIVE_ROOT / 'output_v214_micro_replay'
TRAIN_OUT = OUT_ROOT / 'train_v214_v194_cont_lr3e7_s1'
DRY_OUT = OUT_ROOT / 'dry_run_v214_v194_cont_lr3e7_s1'
EVAL_OUT = OUT_ROOT / 'eval_v214_v194_cont_lr3e7_s1'
V194_ADAPTER = pathlib.Path('{V194_ADAPTER_DRIVE}')
V194_VAL_CSV = pathlib.Path('{V194_VAL_CSV_DRIVE}')
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'
RUN_DRY_RUN = os.environ.get('KG1_V214_RUN_DRY_RUN', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
RUN_TRAIN = os.environ.get('KG1_V214_RUN_TRAIN', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_EVAL = os.environ.get('KG1_V214_RUN_EVAL', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
V214_MODEL_DEVICE_MAP = os.environ.get('KG1_V214_MODEL_DEVICE_MAP', 'cuda')
V214_ATTN_IMPLEMENTATION = os.environ.get('KG1_V214_ATTN_IMPLEMENTATION', 'eager')
V214_BATCH_SIZE = int(os.environ.get('KG1_V214_BATCH_SIZE', '4'))
V214_MICRO_BATCH_SIZE = int(os.environ.get('KG1_V214_MICRO_BATCH_SIZE', '1'))
V214_MAX_LENGTH = int(os.environ.get('KG1_V214_MAX_LENGTH', '4096'))
V214_ABORT_MAX_RESERVED_GIB = float(os.environ.get('KG1_V214_ABORT_MAX_RESERVED_GIB', '78'))
ALLOW_KAGGLE_SUBMIT = False

WEAK_MIN_FOR_FULL = 191
WEAK_STRICT_TARGET = 198
FULL_STRICT_TARGET = 828
FULL_PREFERRED_TARGET = 830
STRONG_DEFAULT_TARGET = 632
WEAK_MAX_TRUNC_STRICT = 1
FULL_MAX_TRUNC_REVIEW = 4

for path in [DRIVE_ROOT, OUT_ROOT, TRAIN_OUT, DRY_OUT, EVAL_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION)
print('ROOT =', ROOT)
print('OUT_ROOT =', OUT_ROOT)
print('DRY_OUT =', DRY_OUT)
print('TRAIN_OUT =', TRAIN_OUT)
print('EVAL_OUT =', EVAL_OUT)
print('V194_ADAPTER =', V194_ADAPTER)
print('V194_VAL_CSV =', V194_VAL_CSV)
print('MODEL_NAME =', MODEL_NAME)
print('MODEL_REVISION =', MODEL_REVISION)
print('RUN_DRY_RUN =', RUN_DRY_RUN)
print('RUN_TRAIN =', RUN_TRAIN)
print('RUN_EVAL =', RUN_EVAL)
print('V214_MODEL_DEVICE_MAP =', V214_MODEL_DEVICE_MAP)
print('V214_ATTN_IMPLEMENTATION =', V214_ATTN_IMPLEMENTATION)
print('V214_BATCH_SIZE =', V214_BATCH_SIZE)
print('V214_MICRO_BATCH_SIZE =', V214_MICRO_BATCH_SIZE)
print('V214_MAX_LENGTH =', V214_MAX_LENGTH)
print('V214_ABORT_MAX_RESERVED_GIB =', V214_ABORT_MAX_RESERVED_GIB)
print('TOKENIZERS_PARALLELISM =', os.environ.get('TOKENIZERS_PARALLELISM'))
print('HF_HUB_ENABLE_HF_TRANSFER =', os.environ.get('HF_HUB_ENABLE_HF_TRANSFER'))
print('PYTORCH_CUDA_ALLOC_CONF =', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Submission is disabled in V214 by design.')
print('=== V214 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with explicit command logging.
print('=== V214 HELPERS START ===', flush=True)
import importlib
import json
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time

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
        parts.append(
            'ram_total={:.1f}GiB ram_available={:.1f}GiB'.format(
                meminfo.get('MemTotal', 0.0),
                meminfo.get('MemAvailable', 0.0),
            )
        )
    except Exception as exc:
        parts.append(f'ram=unavailable:{type(exc).__name__}')
    try:
        usage = shutil.disk_usage('/content')
        parts.append(
            'disk_content_free={:.1f}GiB disk_content_total={:.1f}GiB'.format(
                usage.free / 1024**3,
                usage.total / 1024**3,
            )
        )
    except Exception as exc:
        parts.append(f'disk=unavailable:{type(exc).__name__}')
    try:
        gpu = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
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
    if log_path:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open('w', encoding='utf-8')
        print('log_path =', log_path, flush=True)
    else:
        handle = None
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
    last_output = started
    last_heartbeat = started
    reader_done = False
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            item = '[[NO_LINE_READY]]'
        now = time.time()
        if item is None:
            reader_done = True
        elif item != '[[NO_LINE_READY]]':
            last_output = now
            print(item, end='', flush=True)
            if handle:
                handle.write(item)
                handle.flush()
        if heartbeat_s and now - last_heartbeat >= heartbeat_s and proc.poll() is None:
            last_heartbeat = now
            heartbeat = (
                f"[V214 heartbeat] elapsed_s={now - started:.1f} "
                f"no_output_s={now - last_output:.1f} {resource_snapshot_line()}"
            )
            print(heartbeat, flush=True)
            if handle:
                handle.write(heartbeat + '\\n')
                handle.flush()
        if reader_done and proc.poll() is not None:
            break
    rc = proc.wait()
    elapsed = time.time() - started
    if handle:
        handle.close()
    print(f'returncode = {rc}', flush=True)
    print(f'elapsed_s = {elapsed:.1f}', flush=True)
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
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', pip_spec])
        module = importlib.import_module(import_name)
        print(import_name, 'version=', getattr(module, '__version__', 'unknown'), flush=True)
        return module

print('python =', sys.version)
print('=== V214 HELPERS END ===', flush=True)
"""
        ),
        code(
            bootstrap_cell()
        ),
        code(
            """# CELL: dependency and GPU audit.
print('=== V214 DEPENDENCY AUDIT START ===', flush=True)
ensure_import('packaging', 'packaging')
from packaging.version import Version

def ensure_min_version(import_name, min_version, pip_spec):
    module = ensure_import(import_name, pip_spec)
    observed = getattr(module, '__version__', '0')
    print(f'{import_name}_observed_version = {observed}', flush=True)
    if Version(str(observed).split('+')[0]) < Version(min_version):
        print(f'{import_name} below required {min_version}; installing {pip_spec}', flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', pip_spec])
        module = importlib.import_module(import_name)
        observed = getattr(module, '__version__', '0')
        print(f'{import_name}_post_install_version = {observed}', flush=True)
        if Version(str(observed).split('+')[0]) < Version(min_version):
            raise RuntimeError(f'{import_name} version {observed} < required {min_version}')
    return module

def fresh_python_import_check(imports):
    code = (
        "import importlib, json; "
        "mods = {}; "
        f"names = {list(imports)!r}; "
        "ok = {}; "
        "\\nfor name in names:\\n"
        "    m = importlib.import_module(name)\\n"
        "    ok[name] = getattr(m, '__version__', 'unknown')\\n"
        "print(json.dumps(ok, sort_keys=True))"
    )
    run_cmd([sys.executable, '-c', code])

def fresh_python_code_check(label, code_text):
    print(f'=== FRESH PYTHON CHECK START: {label} ===', flush=True)
    run_cmd([sys.executable, '-c', code_text])
    print(f'=== FRESH PYTHON CHECK END: {label} ===', flush=True)

def ensure_mamba_stack():
    required_import = (
        "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn\\n"
        "import mamba_ssm, json\\n"
        "print(json.dumps({"
        "'mamba_ssm_version': getattr(mamba_ssm, '__version__', 'unknown'), "
        "'layernorm_gated_rmsnorm_fn': rmsnorm_fn is not None"
        "}, sort_keys=True))"
    )
    try:
        fresh_python_code_check('mamba_ssm_preinstall', required_import)
        return
    except Exception as exc:
        print('mamba_ssm required import unavailable before install:', repr(exc), flush=True)

    ensure_import('ninja', 'ninja')
    print('Installing mamba-ssm with causal-conv1d extra; this can take several minutes on a fresh Colab runtime.', flush=True)
    try:
        run_cmd([
            sys.executable,
            '-m',
            'pip',
            'install',
            '-q',
            '--no-build-isolation',
            'mamba-ssm[causal-conv1d]',
        ])
    except Exception as exc:
        print('Combined mamba-ssm extra install failed; retrying causal-conv1d and mamba-ssm separately:', repr(exc), flush=True)
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', 'causal-conv1d'])
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', 'mamba-ssm'])
    fresh_python_code_check('mamba_ssm_postinstall', required_import)

ensure_import('pandas', 'pandas')
ensure_import('huggingface_hub', 'huggingface_hub')
if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '').strip() == '1':
    ensure_import('hf_transfer', 'hf_transfer')
ensure_import('transformers', 'transformers')
ensure_min_version('peft', '0.18.1', 'peft>=0.18.1')
torch = ensure_import('torch')
print('torch_cuda_available =', torch.cuda.is_available(), flush=True)
print('torch_cuda_device_count =', torch.cuda.device_count() if torch.cuda.is_available() else 0, flush=True)
if torch.cuda.is_available():
    print('torch_cuda_device_name =', torch.cuda.get_device_name(0), flush=True)
    print('torch_cuda_version =', getattr(torch.version, 'cuda', 'unknown'), flush=True)
ensure_mamba_stack()
try:
    ensure_import('vllm')
except Exception as exc:
    print('vLLM unavailable before install:', repr(exc), flush=True)
    print('Installing vLLM; eval runs in fresh Python processes.', flush=True)
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'vllm'])
try:
    ensure_import('bitsandbytes', 'bitsandbytes')
except Exception as exc:
    print('bitsandbytes unavailable after install attempt; train script will fall back to torch Adam:', repr(exc), flush=True)
fresh_python_import_check(['torch', 'transformers', 'peft', 'vllm', 'mamba_ssm'])
print('=== V214 DEPENDENCY AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: H100/high-RAM sizing gate before any model load.
print('=== V214 H100 SIZE GATE START ===', flush=True)
MIN_GPU_TOTAL_GIB = float(os.environ.get('KG1_V214_MIN_GPU_GIB', '70'))
MIN_RAM_TOTAL_GIB = float(os.environ.get('KG1_V214_MIN_RAM_TOTAL_GIB', '45'))
MIN_RAM_AVAILABLE_GIB = float(os.environ.get('KG1_V214_MIN_RAM_AVAILABLE_GIB', '20'))
MIN_CONTENT_FREE_GIB = float(os.environ.get('KG1_V214_MIN_CONTENT_FREE_GIB', '55'))
WARN_CONTENT_FREE_GIB = float(os.environ.get('KG1_V214_WARN_CONTENT_FREE_GIB', '65'))
SAFE_DISK_CLEANUP = os.environ.get('KG1_V214_SAFE_DISK_CLEANUP', '1').strip().lower() not in {'0', 'false', 'no', 'off'}

def meminfo_gib():
    values = {}
    with open('/proc/meminfo', encoding='utf-8') as handle:
        for line in handle:
            key, raw = line.split(':', 1)
            values[key] = int(raw.strip().split()[0]) / 1024 / 1024
    return values

def disk_free_gib(path):
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3, usage.total / 1024**3

def path_size_gib(path):
    path = pathlib.Path(path)
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1024**3
    total = 0
    for child in path.rglob('*'):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            pass
    return total / 1024**3

def safe_remove_path(path):
    path = pathlib.Path(path)
    before = path_size_gib(path)
    if not path.exists():
        return {'path': str(path), 'existed': False, 'size_gib': 0.0, 'removed': False}
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return {'path': str(path), 'existed': True, 'size_gib': round(before, 3), 'removed': True}
    except Exception as exc:
        return {
            'path': str(path),
            'existed': True,
            'size_gib': round(before, 3),
            'removed': False,
            'error': f'{type(exc).__name__}: {exc}',
        }

def safe_disk_cleanup():
    cleanup_report = []
    if not SAFE_DISK_CLEANUP:
        print('SAFE_DISK_CLEANUP disabled by KG1_V214_SAFE_DISK_CLEANUP=0', flush=True)
        return cleanup_report
    targets = [
        '/content/sample_data',
        '/root/.cache/pip',
    ]
    for pattern in ['/tmp/pip-*']:
        targets.extend(str(item) for item in pathlib.Path('/tmp').glob(pathlib.Path(pattern).name))
    for target in targets:
        cleanup_report.append(safe_remove_path(target))
    print('safe_disk_cleanup_report =', json.dumps(cleanup_report, indent=2, sort_keys=True), flush=True)
    return cleanup_report

if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU is required for V214 dry-run/train/eval.')

content_free_before_cleanup, content_total_before_cleanup = disk_free_gib('/content')
cleanup_report = safe_disk_cleanup()
content_free_after_cleanup, content_total_after_cleanup = disk_free_gib('/content')

props = torch.cuda.get_device_properties(0)
gpu_name = props.name
gpu_total_gib = props.total_memory / 1024**3
mem = meminfo_gib()
ram_total_gib = mem.get('MemTotal', 0.0)
ram_available_gib = mem.get('MemAvailable', 0.0)
content_free_gib, content_total_gib = disk_free_gib('/content')
drive_free_gib, drive_total_gib = disk_free_gib('/content/drive') if pathlib.Path('/content/drive').exists() else (0.0, 0.0)
h100_detected = 'H100' in gpu_name.upper()

size_report = {
    'gpu_name': gpu_name,
    'h100_detected': h100_detected,
    'gpu_total_gib': round(gpu_total_gib, 2),
    'ram_total_gib': round(ram_total_gib, 2),
    'ram_available_gib': round(ram_available_gib, 2),
    'content_disk_free_before_cleanup_gib': round(content_free_before_cleanup, 2),
    'content_disk_free_after_cleanup_gib': round(content_free_after_cleanup, 2),
    'content_disk_free_gib': round(content_free_gib, 2),
    'content_disk_total_gib': round(content_total_gib, 2),
    'drive_disk_free_gib': round(drive_free_gib, 2),
    'drive_disk_total_gib': round(drive_total_gib, 2),
    'minimums': {
        'gpu_total_gib': MIN_GPU_TOTAL_GIB,
        'ram_total_gib': MIN_RAM_TOTAL_GIB,
        'ram_available_gib': MIN_RAM_AVAILABLE_GIB,
        'content_disk_free_gib': MIN_CONTENT_FREE_GIB,
        'content_disk_warning_gib': WARN_CONTENT_FREE_GIB,
    },
    'cleanup_report': cleanup_report,
}
print('size_report =', json.dumps(size_report, indent=2, sort_keys=True), flush=True)

if gpu_total_gib < MIN_GPU_TOTAL_GIB:
    raise RuntimeError(
        f'GPU memory too small for V214: {gpu_total_gib:.1f}GiB < {MIN_GPU_TOTAL_GIB:.1f}GiB. '
        'Use H100 80GB/high-RAM or another >=80GB-class GPU.'
    )
if ram_total_gib < MIN_RAM_TOTAL_GIB or ram_available_gib < MIN_RAM_AVAILABLE_GIB:
    raise RuntimeError(
        f'System RAM inadequate: total={ram_total_gib:.1f}GiB available={ram_available_gib:.1f}GiB. '
        'Use Colab high-RAM runtime.'
    )
if content_free_gib < MIN_CONTENT_FREE_GIB:
    raise RuntimeError(
        f'/content free disk too small: {content_free_gib:.1f}GiB < {MIN_CONTENT_FREE_GIB:.1f}GiB. '
        'Restart runtime, free disk, or lower KG1_V214_MIN_CONTENT_FREE_GIB only for dry-run diagnostics.'
    )
if content_free_gib < WARN_CONTENT_FREE_GIB:
    print(
        f'WARNING: /content free disk is tight: {content_free_gib:.1f}GiB < '
        f'warning threshold {WARN_CONTENT_FREE_GIB:.1f}GiB. This should be enough '
        'to try the V214 dry-run on this H100 runtime, but model download/cache may still fail. '
        'If download fails, restart runtime and avoid extra installs/files before this notebook.',
        flush=True,
    )
if not h100_detected:
    print('WARNING: H100 not detected. Memory gate passed, but the intended runtime is H100 high-RAM.', flush=True)
else:
    print('H100 detected and resource gate passed.', flush=True)
print('=== V214 H100 SIZE GATE END ===', flush=True)
"""
        ),
        code(
            """# CELL: dataset manifest, hash, and family-count gate.
print('=== V214 DATASET AUDIT START ===', flush=True)
import json
import pandas as pd
from collections import Counter

train_path = ROOT / 'data/v214/v214_micro_train.jsonl'
val_path = ROOT / 'data/v214/v214_micro_val.jsonl'
split_manifest_path = ROOT / 'data/v214/v214_micro_split_manifest.json'
candidate_manifest_path = ROOT / 'data/v214/v214_micro_replay_candidate_manifest.json'
split_manifest = json.loads(split_manifest_path.read_text(encoding='utf-8'))
candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding='utf-8'))
print('split_manifest =', json.dumps(split_manifest, indent=2, sort_keys=True), flush=True)
print('candidate_manifest rows =', candidate_manifest.get('rows'), flush=True)

observed_train_sha = sha256_file(train_path)
observed_val_sha = sha256_file(val_path)
print('observed_train_sha256 =', observed_train_sha, flush=True)
print('observed_val_sha256 =', observed_val_sha, flush=True)
if observed_train_sha != split_manifest['train_sha256']:
    raise RuntimeError('train sha256 mismatch')
if observed_val_sha != split_manifest['val_sha256']:
    raise RuntimeError('val sha256 mismatch')

def read_jsonl(path):
    rows = []
    with pathlib.Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

train_rows = read_jsonl(train_path)
val_rows = read_jsonl(val_path)
print('train_rows =', len(train_rows), flush=True)
print('val_rows =', len(val_rows), flush=True)
print('train_family_counts =', dict(sorted(Counter(r.get('family', 'unknown') for r in train_rows).items())), flush=True)
print('val_family_counts =', dict(sorted(Counter(r.get('family', 'unknown') for r in val_rows).items())), flush=True)
if len(train_rows) != split_manifest['train_rows'] or len(val_rows) != split_manifest['val_rows']:
    raise RuntimeError('row count mismatch vs split manifest')
if split_manifest.get('train_val_prompt_answer_overlap') != 0:
    raise RuntimeError('train/val overlap is nonzero')
print('=== V214 DATASET AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: V194 adapter audit on Google Drive.
print('=== V214 V194 ADAPTER AUDIT START ===', flush=True)
adapter_config_path = V194_ADAPTER / 'adapter_config.json'
adapter_model_safetensors = V194_ADAPTER / 'adapter_model.safetensors'
adapter_model_bin = V194_ADAPTER / 'adapter_model.bin'
print('adapter_config_path =', adapter_config_path, flush=True)
print('adapter_model_safetensors =', adapter_model_safetensors, adapter_model_safetensors.exists(), flush=True)
print('adapter_model_bin =', adapter_model_bin, adapter_model_bin.exists(), flush=True)
if not V194_ADAPTER.exists():
    raise FileNotFoundError(f'V194 adapter directory missing on Drive: {V194_ADAPTER}')
if not adapter_config_path.exists():
    raise FileNotFoundError(f'V194 adapter_config.json missing: {adapter_config_path}')
if not (adapter_model_safetensors.exists() or adapter_model_bin.exists()):
    raise FileNotFoundError(f'V194 adapter weights missing in: {V194_ADAPTER}')
adapter_config = json.loads(adapter_config_path.read_text(encoding='utf-8'))
print('adapter_config =', json.dumps(adapter_config, indent=2, sort_keys=True), flush=True)
if int(adapter_config.get('r', 999)) > 32:
    raise RuntimeError('V194 adapter rank exceeds LoRA rank gate')
if adapter_config.get('peft_type') != 'LORA':
    raise RuntimeError('V194 adapter is not a LoRA adapter')
print('lm_head_in_target_modules =', 'lm_head' in set(adapter_config.get('target_modules') or []), flush=True)
print('target_parameters =', adapter_config.get('target_parameters'), flush=True)
print('NOTE: V194 includes lm_head/target_parameters; this notebook preserves V194 and does not strip them.', flush=True)
print('=== V214 V194 ADAPTER AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: build weak/full/strong validation CSVs from the protected V194 gate file.
print('=== V214 VALIDATION CSV BUILD START ===', flush=True)
import pandas as pd
sys.path.insert(0, str(ROOT))
from src.competition_utils import classify_puzzle

if not V194_VAL_CSV.exists():
    raise FileNotFoundError(
        f'Missing V194 validation CSV: {V194_VAL_CSV}. Run/preserve the V207A ACC gate first.'
    )
full_df = pd.read_csv(V194_VAL_CSV)
if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
    raise RuntimeError(f'V194_VAL_CSV must include prompt and answer columns: {V194_VAL_CSV}')
full_df['type'] = full_df['prompt'].map(classify_puzzle)
weak_types = {'bit_manipulation', 'equation_transform'}
strong_types = {'gravity_constant', 'numeral_system', 'text_encryption', 'unit_conversion'}
weak_df = full_df[full_df['type'].isin(weak_types)].copy()
strong_df = full_df[full_df['type'].isin(strong_types)].copy()
full_eval_csv = EVAL_OUT / 'v214_full_947.csv'
weak_eval_csv = EVAL_OUT / 'v214_weak_315.csv'
strong_eval_csv = EVAL_OUT / 'v214_strong_632.csv'
full_df.to_csv(full_eval_csv, index=False)
weak_df.to_csv(weak_eval_csv, index=False)
strong_df.to_csv(strong_eval_csv, index=False)
print('full_rows =', len(full_df), 'path =', full_eval_csv, flush=True)
print('weak_rows =', len(weak_df), 'path =', weak_eval_csv, flush=True)
print('strong_rows =', len(strong_df), 'path =', strong_eval_csv, flush=True)
print('per_family_counts =', full_df['type'].value_counts().sort_index().to_dict(), flush=True)
if len(full_df) != 947:
    raise RuntimeError(f'Expected 947 full validation rows, got {len(full_df)}')
if len(weak_df) != 315:
    raise RuntimeError(f'Expected 315 weak validation rows, got {len(weak_df)}')
if len(strong_df) != 632:
    raise RuntimeError(f'Expected 632 strong validation rows, got {len(strong_df)}')
print('=== V214 VALIDATION CSV BUILD END ===', flush=True)
"""
        ),
        code(
            """# CELL: common training environment builder.
print('=== V214 TRAINING ENV SETUP START ===', flush=True)
TRAIN_SHA = split_manifest['train_sha256']
VAL_SHA = split_manifest['val_sha256']

def training_env(output_dir, dry_run):
    env = os.environ.copy()
    env.update({
        'MODEL_NAME': MODEL_NAME,
        'MODEL_REVISION': MODEL_REVISION,
        'MODEL_DEVICE_MAP': V214_MODEL_DEVICE_MAP,
        'ATTN_IMPLEMENTATION': V214_ATTN_IMPLEMENTATION,
        'TORCH_ALLOW_TF32': '1',
        'TORCH_FLOAT32_MATMUL_PRECISION': 'high',
        'GRADIENT_CHECKPOINTING': '1',
        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM', 'false'),
        'HF_HUB_ENABLE_HF_TRANSFER': os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '1'),
        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'),
        'DATA_REPO': 'local',
        'DATA_FILE': str(ROOT / 'data/v214/v214_micro_train.jsonl'),
        'VAL_FILE': str(ROOT / 'data/v214/v214_micro_val.jsonl'),
        'EXPECTED_TRAIN_SHA256': TRAIN_SHA,
        'EXPECTED_VAL_SHA256': VAL_SHA,
        'MIN_TRAIN_EXAMPLES': str(split_manifest['train_rows']),
        'MIN_VAL_EXAMPLES': str(split_manifest['val_rows']),
        'MIN_TOKENIZED_TRAIN_EXAMPLES': str(split_manifest['train_rows']),
        'MIN_TOKENIZED_VAL_EXAMPLES': str(split_manifest['val_rows']),
        'OUTPUT_DIR': str(output_dir),
        'OUTPUT_REPO': '',
        'RUN_ID': 'v214_v194_cont_lr3e7_s1',
        'INIT_ADAPTER_DIR': str(V194_ADAPTER),
        'INIT_ADAPTER_LOAD_MODE': 'peft',
        'FAIL_ON_MISSING_ADAPTER_KEYS': '1',
        'UPLOAD_TO_HF': '0',
        'UPLOAD_CHECKPOINTS_DURING_TRAINING': '0',
        'DRY_RUN_VALIDATE_ONLY': '1' if dry_run else '0',
        'LORA_R': '32',
        'LORA_ALPHA': '32',
        'LORA_DROPOUT': '0.0',
        'MAX_LENGTH': str(V214_MAX_LENGTH),
        'MAX_PROMPT_TRUNCATION_RATE': '0.0',
        'BATCH_SIZE': str(V214_BATCH_SIZE),
        'MICRO_BATCH_SIZE': str(V214_MICRO_BATCH_SIZE),
        'LEARNING_RATE': '3e-7',
        'FINAL_LEARNING_RATE': '3e-7',
        'NUM_EPOCHS': '1',
        'MAX_STEPS': '1',
        'SAVE_EVERY_STEPS': '1',
        'EVAL_EVERY_STEPS': '0',
        'EVAL_MAX_EXAMPLES': '32',
        'LOG_EVERY_STEPS': '1',
        'MICRO_LOG_EVERY': '1',
        'SEED': '214',
        'MAX_TRAINABLE_PARAM_RATIO': '0.04',
        'TRAINABLE_LORA_MODULES': 'q_proj,k_proj,v_proj,o_proj,in_proj,out_proj',
        'BASELINE_EVAL_BEFORE_TRAIN': '1',
        'REQUIRE_FINAL_EVAL_LTE_BASELINE': '0',
        'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': '-1',
        'ABORT_MAX_RESERVED_GIB': str(V214_ABORT_MAX_RESERVED_GIB),
        'COMPUTE_PROVIDER': 'colab_h100',
    })
    return env

print('training script =', ROOT / 'scripts/hf_job_train_v90.py', flush=True)
print('dry-run output dir =', DRY_OUT, flush=True)
print('train output dir =', TRAIN_OUT, flush=True)
print('=== V214 TRAINING ENV SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: dry-run model/adapter trainability check.
print('=== V214 DRY RUN START ===', flush=True)
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
    report = json.loads(dry_report.read_text(encoding='utf-8'))
    print('dry_run_decision =', json.dumps(report.get('decision', {}), indent=2, sort_keys=True), flush=True)
    print('trainable_parameters =', json.dumps(report.get('trainable_parameters', {}), indent=2, sort_keys=True), flush=True)
else:
    print('RUN_DRY_RUN is false; skipping dry-run. This is not recommended.', flush=True)
print('=== V214 DRY RUN END ===', flush=True)
"""
        ),
        code(
            """# CELL: optional one-step V194 continuation. Requires KG1_V214_RUN_TRAIN=1.
print('=== V214 TRAIN START ===', flush=True)
final_adapter = TRAIN_OUT / 'final_adapter'
if not RUN_TRAIN:
    print('RUN_TRAIN is false. Set environment KG1_V214_RUN_TRAIN=1 before running this cell to train.', flush=True)
    print('Skipping training; downstream eval will run only if final_adapter already exists:', final_adapter, flush=True)
elif final_adapter.exists():
    print('final_adapter already exists; skipping retrain:', final_adapter, flush=True)
else:
    rc = run_cmd(
        [sys.executable, str(ROOT / 'scripts/hf_job_train_v90.py')],
        cwd=ROOT,
        env=training_env(TRAIN_OUT, dry_run=False),
        log_path=TRAIN_OUT / 'train.log',
        check=True,
    )
    print('training returncode =', rc, flush=True)
print('final_adapter =', final_adapter, 'exists =', final_adapter.exists(), flush=True)
print('=== V214 TRAIN END ===', flush=True)
"""
        ),
        code(
            """# CELL: weak eval gate. Full eval is blocked until weak improves over V194.
print('=== V214 WEAK EVAL START ===', flush=True)
weak_report = None
weak_eval_dir = EVAL_OUT / 'weak_eval'
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping weak eval.', flush=True)
elif not final_adapter.exists():
    print('No final_adapter exists; skipping weak eval.', flush=True)
else:
    weak_eval_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(weak_eval_csv),
            '--questions-csv', str(weak_eval_csv),
            '--adapter', str(final_adapter),
            '--base-model-path', MODEL_NAME,
            '--label', 'v214_micro_weak',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(weak_eval_dir),
        ],
        cwd=ROOT,
        log_path=weak_eval_dir / 'weak_eval.log',
        check=True,
    )
    weak_report_path = weak_eval_dir / 'v214_micro_weak_eval_report.json'
    weak_report = json.loads(weak_report_path.read_text(encoding='utf-8'))
    print('weak_report =', json.dumps(weak_report, indent=2, sort_keys=True), flush=True)
    weak_correct = int(weak_report['correct'])
    weak_truncated = int(weak_report['truncated'])
    weak_gate_pass = weak_correct >= WEAK_MIN_FOR_FULL and weak_truncated <= 3
    print('weak_correct =', weak_correct, flush=True)
    print('weak_truncated =', weak_truncated, flush=True)
    print('weak_gate_pass_for_full =', weak_gate_pass, flush=True)
print('=== V214 WEAK EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval only if weak gate passes.
print('=== V214 FULL EVAL START ===', flush=True)
full_report = None
full_eval_dir = EVAL_OUT / 'full_eval'
weak_gate_pass = bool(weak_report and int(weak_report['correct']) >= WEAK_MIN_FOR_FULL and int(weak_report['truncated']) <= 3)
if not RUN_EVAL:
    print('RUN_EVAL is false; skipping full eval.', flush=True)
elif not final_adapter.exists():
    print('No final_adapter exists; skipping full eval.', flush=True)
elif not weak_gate_pass:
    print('Weak gate failed; skipping full eval.', flush=True)
else:
    full_eval_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapter.py'),
            '--solution-csv', str(full_eval_csv),
            '--questions-csv', str(full_eval_csv),
            '--adapter', str(final_adapter),
            '--base-model-path', MODEL_NAME,
            '--label', 'v214_micro_full',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(full_eval_dir),
        ],
        cwd=ROOT,
        log_path=full_eval_dir / 'full_eval.log',
        check=True,
    )
    full_report_path = full_eval_dir / 'v214_micro_full_eval_report.json'
    full_per_task_path = full_eval_dir / 'v214_micro_full_per_task.csv'
    full_report = json.loads(full_report_path.read_text(encoding='utf-8'))
    full_per_task = pd.read_csv(full_per_task_path)
    print('full_report =', json.dumps(full_report, indent=2, sort_keys=True), flush=True)
    print('full_per_task =')
    print(full_per_task.to_string(index=False))
    strong_rows = full_per_task[full_per_task['task_type'].isin(['gravity_constant', 'numeral_system', 'text_encryption', 'unit_conversion'])]
    strong_correct = int(strong_rows['correct'].sum())
    weak_rows = full_per_task[full_per_task['task_type'].isin(['bit_manipulation', 'equation_transform'])]
    weak_correct_full = int(weak_rows['correct'].sum())
    final_decision = {
        'full_correct': int(full_report['correct']),
        'weak_correct': weak_correct_full,
        'strong_correct': strong_correct,
        'truncated': int(full_report['truncated']),
        'strict_submit_candidate_after_human_review': (
            int(full_report['correct']) >= FULL_STRICT_TARGET
            and weak_correct_full >= WEAK_STRICT_TARGET
            and strong_correct == STRONG_DEFAULT_TARGET
            and int(full_report['truncated']) <= FULL_MAX_TRUNC_REVIEW
        ),
        'preferred_candidate': int(full_report['correct']) >= FULL_PREFERRED_TARGET,
        'note': 'This notebook never submits; human review is mandatory.',
    }
    print('final_decision =', json.dumps(final_decision, indent=2, sort_keys=True), flush=True)
print('=== V214 FULL EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: write final Colab run manifest.
print('=== V214 RUN MANIFEST START ===', flush=True)
manifest = {
    'version': VERSION,
    'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'run_flags': {
        'run_dry_run': RUN_DRY_RUN,
        'run_train': RUN_TRAIN,
        'run_eval': RUN_EVAL,
        'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
    },
    'paths': {
        'root': str(ROOT),
        'drive_root': str(DRIVE_ROOT),
        'dry_out': str(DRY_OUT),
        'train_out': str(TRAIN_OUT),
        'eval_out': str(EVAL_OUT),
        'final_adapter': str(final_adapter),
        'v194_adapter': str(V194_ADAPTER),
        'v194_val_csv': str(V194_VAL_CSV),
    },
    'gates': {
        'weak_min_for_full': WEAK_MIN_FOR_FULL,
        'weak_strict_target': WEAK_STRICT_TARGET,
        'full_strict_target': FULL_STRICT_TARGET,
        'full_preferred_target': FULL_PREFERRED_TARGET,
        'strong_default_target': STRONG_DEFAULT_TARGET,
        'weak_max_trunc_strict': WEAK_MAX_TRUNC_STRICT,
        'full_max_trunc_review': FULL_MAX_TRUNC_REVIEW,
    },
    'weak_report': weak_report,
    'full_report': full_report,
    'decision': 'no_submit_human_review_required',
}
manifest_path = OUT_ROOT / 'v214_colab_run_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
print('=== V214 RUN MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")
    print(
        "colab_url=https://colab.research.google.com/github/"
        "FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/"
        "KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb"
    )
    print("NOTE: the URL works after the notebook is pushed to the referenced branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

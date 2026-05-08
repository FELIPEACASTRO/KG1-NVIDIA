#!/usr/bin/env python3
"""Build the V221 candidate-registry weak A/B Colab notebook.

V221 is a no-training triage notebook. It downloads/validates candidate
adapters, evaluates the ready candidates on the weak split with a single vLLM
model load, and writes a family-level ranking. Full evaluation and Kaggle
submission are hard-blocked by default.
"""

from __future__ import annotations

import json
import pprint
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v221-candidate-registry/notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    "v221-candidate-registry/notebooks/KG1_V221_CANDIDATE_REGISTRY_WEAK_AB_COLAB.ipynb"
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
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"
TRAIN_ROWS = 10206
VAL_ROWS = 681

REGISTRY_SEED = {
    "schema_version": "kg1_v221_candidate_registry_v1",
    "candidates": [
        {
            "name": "v194_protected_baseline",
            "source_kind": "drive_adapter",
            "adapter_path": V194_ADAPTER_DRIVE,
            "priority": "P0",
            "target_families": ["all"],
            "required": True,
        },
        {
            "name": "v217_final_existing",
            "source_kind": "drive_adapter",
            "adapter_path": V217_FINAL_ADAPTER_DRIVE,
            "priority": "P0",
            "target_families": ["all"],
            "required": False,
        },
        {
            "name": "naribow_hf_nemotron_sft_lora",
            "source_kind": "hf_model_adapter",
            "ref": "Naribow/nemotron-sft-lora",
            "revision": "main",
            "priority": "P1",
            "target_families": ["all"],
            "required": False,
        },
        {
            "name": "dgxchen_trained_adapter",
            "source_kind": "kaggle_dataset_adapter",
            "ref": "dgxchen/trained-adapter",
            "priority": "P1",
            "target_families": ["all"],
            "required": False,
        },
        {
            "name": "konbu17_exp026_s012_lora",
            "source_kind": "kaggle_dataset_adapter",
            "ref": "konbu17/exp026-s012-lora",
            "priority": "P1",
            "target_families": ["all"],
            "required": False,
        },
        {
            "name": "konbu17_sft_lora_cot_selection",
            "source_kind": "kaggle_dataset_adapter",
            "ref": "konbu17/nemotron-sft-lora-cot-selection",
            "priority": "P1",
            "target_families": ["all"],
            "required": False,
        },
        {
            "name": "kienngx_tinker_adapter",
            "source_kind": "kaggle_model_adapter",
            "ref": "kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1",
            "priority": "P1",
            "target_families": ["all", "bit_manipulation", "equation_transform"],
            "required": False,
        },
    ],
}

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v221-{prefix}-{_CELL_COUNTER:02d}"


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
    registry_literal = pprint.pformat(REGISTRY_SEED, sort_dicts=True, width=120)
    cells = [
        md(
            f"""# KG1 V221 Candidate Registry Weak A/B Colab

Purpose: evaluate candidate adapters from a registry on the weak split before any new training or full-eval spend.

This notebook:

- does not train;
- keeps thinking enabled by default;
- validates Drive/HF/Kaggle adapter candidates;
- uses `scripts/evaluate_lora_adapters_batch.py` so the base model is loaded once for multiple LoRAs;
- writes candidate rankings by total, equation, bit, truncation, and throughput;
- blocks full eval and Kaggle submit by default.

Colab URL:

`{COLAB_URL}`
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V221 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V221 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration, embedded registry, and hard locks.
print('=== V221 CONFIG START ===', flush=True)
import csv
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
    for secret_name in ['KAGGLE_USERNAME', 'KAGGLE_KEY']:
        if not os.environ.get(secret_name):
            secret_value = userdata.get(secret_name)
            if secret_value:
                os.environ[secret_name] = secret_value
                print('loaded Kaggle secret from Colab userdata:', secret_name, flush=True)
except Exception as exc:
    print('Colab secret probe skipped:', type(exc).__name__, flush=True)

VERSION = 'V221_CANDIDATE_REGISTRY_WEAK_AB_20260508'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'v221-candidate-registry')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V221')
OUT_ROOT = DRIVE_ROOT / 'output_v221_candidate_registry_weak_ab'
DOWNLOAD_ROOT = OUT_ROOT / 'candidate_downloads'
EVAL_OUT = OUT_ROOT / 'eval_v221_candidate_registry_weak_ab'
PACKAGE_OUT = OUT_ROOT / 'package_v221_candidate_registry_weak_ab'
V194_ADAPTER = pathlib.Path('{V194_ADAPTER_DRIVE}')
V194_VAL_CSV = pathlib.Path('{V194_VAL_CSV_DRIVE}')
V217_FINAL_ADAPTER = pathlib.Path('{V217_FINAL_ADAPTER_DRIVE}')
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

EXPECTED_TRAIN_SHA256 = '{TRAIN_SHA}'
EXPECTED_VAL_SHA256 = '{VAL_SHA}'
MIN_TRAIN_EXAMPLES = {TRAIN_ROWS}
MIN_VAL_EXAMPLES = {VAL_ROWS}
TOKENIZE_ONLY_DRY_RUN = 'candidate_registry_weak_ab_eval_only'
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
INIT_ADAPTER_DIR = V194_ADAPTER
RUN_TRAIN = os.environ.get('KG1_V221_RUN_TRAIN', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_EVAL = os.environ.get('KG1_V221_RUN_EVAL', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
FORCE_DOWNLOAD = os.environ.get('KG1_V221_FORCE_DOWNLOAD', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
FORCE_REEVAL = os.environ.get('KG1_V221_FORCE_REEVAL', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_FULL_IF_GATE = os.environ.get('KG1_V221_RUN_FULL_IF_GATE', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
V221_VLLM_PIP_SPEC = os.environ.get('KG1_V221_VLLM_PIP_SPEC', 'vllm==0.20.1')
V221_MAX_TOKENS = int(os.environ.get('KG1_V221_MAX_TOKENS', '3584'))
V221_MAX_MODEL_LEN = int(os.environ.get('KG1_V221_MAX_MODEL_LEN', '8192'))
V221_MAX_NUM_SEQS = int(os.environ.get('KG1_V221_MAX_NUM_SEQS', '64'))
V221_WARMUP_ROWS = int(os.environ.get('KG1_V221_WARMUP_ROWS', '0'))
V221_MAX_CANDIDATES = int(os.environ.get('KG1_V221_MAX_CANDIDATES', '0'))
V221_PROMPT_SUFFIX = os.environ.get('KG1_V221_PROMPT_SUFFIX', '\\nPlease put your final answer inside `\\\\boxed{{}}`. For example: `\\\\boxed{{your answer}}`')
V221_DISABLE_THINKING_DEFAULT = False
runtime_dependency_names = ['causal_conv1d', 'mamba_ssm', 'vllm']

EXPECTED_V194_ADAPTER_BYTES = 4259069440
EXPECTED_V194_ADAPTER_TENSOR_COUNT = 12011
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

CANDIDATE_REGISTRY = {registry_literal}

for path in [DRIVE_ROOT, OUT_ROOT, DOWNLOAD_ROOT, EVAL_OUT, PACKAGE_OUT]:
    path.mkdir(parents=True, exist_ok=True)

registry_seed_path = OUT_ROOT / 'candidate_registry_seed_embedded.json'
registry_seed_path.write_text(json.dumps(CANDIDATE_REGISTRY, indent=2, sort_keys=True), encoding='utf-8')

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('DOWNLOAD_ROOT =', DOWNLOAD_ROOT, flush=True)
print('EVAL_OUT =', EVAL_OUT, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V217_FINAL_ADAPTER =', V217_FINAL_ADAPTER, flush=True)
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
print('V221_MAX_TOKENS =', V221_MAX_TOKENS, flush=True)
print('V221_MAX_MODEL_LEN =', V221_MAX_MODEL_LEN, flush=True)
print('V221_MAX_NUM_SEQS =', V221_MAX_NUM_SEQS, flush=True)
print('V221_WARMUP_ROWS =', V221_WARMUP_ROWS, flush=True)
print('V221_MAX_CANDIDATES =', V221_MAX_CANDIDATES, flush=True)
print('V221_DISABLE_THINKING_DEFAULT =', V221_DISABLE_THINKING_DEFAULT, flush=True)
print('runtime_dependency_names =', runtime_dependency_names, flush=True)
print('registry_seed_path =', registry_seed_path, flush=True)
print('registry_candidate_count =', len(CANDIDATE_REGISTRY['candidates']), flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if RUN_TRAIN:
    raise RuntimeError('V221 is candidate weak A/B only; RUN_TRAIN must stay false.')
if V221_DISABLE_THINKING_DEFAULT:
    raise RuntimeError('V221 must keep thinking enabled by default.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')
print('=== V221 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging, heartbeat, and candidate resolution.
print('=== V221 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def safe_name(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))


def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.exists() and (path / 'adapter_config.json').exists() and (
        (path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists()
    )


def find_adapter_dir(root):
    root = pathlib.Path(root)
    if is_complete_adapter_dir(root):
        return root
    matches = []
    if root.exists():
        for config_path in root.rglob('adapter_config.json'):
            candidate = config_path.parent
            if is_complete_adapter_dir(candidate):
                matches.append(candidate)
    if not matches:
        return None
    return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]


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
                '[V221 heartbeat] elapsed_s=%.1f ram_total=%.1fGiB ram_available=%.1fGiB disk_content_free=%.1fGiB disk_content_total=%.1fGiB gpu=[%s]'
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
            print('[V221 heartbeat_error]', type(exc).__name__, str(exc), flush=True)


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
    print('=== V221 VLLM EVAL DEPENDENCY CHECK START ===', flush=True)
    if verify_import_subprocess('vllm', check=False) != 0:
        print('vLLM subprocess import failed; installing pinned V221_VLLM_PIP_SPEC =', V221_VLLM_PIP_SPEC, flush=True)
        run_cmd(
            [sys.executable, '-m', 'pip', 'install', '-q', V221_VLLM_PIP_SPEC],
            cwd='/content',
            log_path=OUT_ROOT / 'pip_install_vllm.log',
            check=True,
        )
        verify_import_subprocess('vllm', check=True)
    print('=== V221 VLLM EVAL DEPENDENCY CHECK END ===', flush=True)


def ensure_python_package(import_name, pip_spec):
    if verify_import_subprocess(import_name, check=False) == 0:
        return
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', pip_spec], cwd='/content', log_path=OUT_ROOT / f'pip_install_{import_name}.log', check=True)
    verify_import_subprocess(import_name, check=True)


def has_kaggle_credentials():
    env_ok = bool(os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))
    file_ok = pathlib.Path('/root/.kaggle/kaggle.json').exists()
    return env_ok or file_ok


def download_or_resolve_candidate(candidate):
    status = {
        'name': candidate.get('name'),
        'source_kind': candidate.get('source_kind'),
        'ref': candidate.get('ref', ''),
        'adapter': '',
        'ready': False,
        'required': bool(candidate.get('required', False)),
        'error': '',
    }
    try:
        kind = candidate.get('source_kind')
        if kind == 'drive_adapter':
            adapter = pathlib.Path(candidate.get('adapter_path', ''))
            status['adapter'] = str(adapter)
            status['ready'] = is_complete_adapter_dir(adapter)
            if not status['ready']:
                status['error'] = 'drive adapter missing or incomplete'
            return status
        if kind == 'hf_model_adapter':
            ensure_python_package('huggingface_hub', 'huggingface_hub')
            from huggingface_hub import snapshot_download
            ref = candidate['ref']
            local_dir = DOWNLOAD_ROOT / safe_name(ref)
            if FORCE_DOWNLOAD and local_dir.exists():
                shutil.rmtree(local_dir)
            if not is_complete_adapter_dir(local_dir):
                print('downloading HF adapter:', ref, 'to', local_dir, flush=True)
                snapshot_download(
                    repo_id=ref,
                    repo_type='model',
                    revision=candidate.get('revision', 'main'),
                    local_dir=str(local_dir),
                    allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin', 'README.md'],
                    token=os.environ.get('HF_TOKEN') or None,
                )
            adapter = find_adapter_dir(local_dir)
            status['adapter'] = str(adapter or local_dir)
            status['ready'] = adapter is not None
            if not status['ready']:
                status['error'] = 'HF adapter download did not contain adapter files'
            return status
        if kind in {'kaggle_dataset_adapter', 'kaggle_model_adapter'}:
            if not has_kaggle_credentials():
                status['error'] = 'Kaggle credentials missing; set KAGGLE_USERNAME/KAGGLE_KEY or /root/.kaggle/kaggle.json'
                return status
            ref = candidate['ref']
            local_dir = DOWNLOAD_ROOT / safe_name(ref)
            if FORCE_DOWNLOAD and local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            if kind == 'kaggle_dataset_adapter':
                cmd = ['kaggle', 'datasets', 'download', ref, '--unzip', '-p', str(local_dir), '-q']
            else:
                cmd = ['kaggle', 'models', 'instances', 'versions', 'download', ref, '--untar', '-p', str(local_dir), '-q']
            if not find_adapter_dir(local_dir):
                rc = run_cmd(cmd, cwd='/content', log_path=OUT_ROOT / ('download_' + safe_name(ref) + '.log'), check=False, heartbeat_s=60)
                if rc != 0:
                    status['error'] = 'Kaggle download failed rc=' + str(rc)
                    return status
            adapter = find_adapter_dir(local_dir)
            status['adapter'] = str(adapter or local_dir)
            status['ready'] = adapter is not None
            if not status['ready']:
                status['error'] = 'Kaggle artifact did not contain adapter files'
            return status
        status['error'] = 'unknown source_kind=' + str(kind)
        return status
    except Exception as exc:
        status['error'] = type(exc).__name__ + ': ' + str(exc)
        return status


print('=== V221 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo and compile required scripts.
print('=== V221 REPO SETUP START ===', flush=True)
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
    ROOT / 'scripts/evaluate_lora_adapters_batch.py',
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
print('=== V221 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, weak split, V194 contract, and candidate registry resolution.
print('=== V221 PREFLIGHT START ===', flush=True)
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
    raise RuntimeError('CUDA GPU is required for V221 vLLM eval')
if gpu_total_gib < 79.0:
    raise RuntimeError('V221 expects an 80GB GPU class runtime')
usage = shutil.disk_usage('/content')
content_free_gib = usage.free / 1024**3
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if content_free_gib < 80:
    raise RuntimeError('Not enough /content disk free space for model/eval cache')

for optional_module in ['causal_conv1d', 'mamba_ssm']:
    rc = verify_import_subprocess(optional_module, check=False)
    print(optional_module + '_optional_training_import_rc =', rc, flush=True)
print('causal_conv1d and mamba_ssm are audited but not installed here; V221 is eval-only.', flush=True)

if not V194_VAL_CSV.exists():
    raise FileNotFoundError(V194_VAL_CSV)
if not is_complete_adapter_dir(V194_ADAPTER):
    raise RuntimeError('V194 adapter is missing or incomplete')

ensure_python_package('safetensors', 'safetensors')
from safetensors import safe_open

v194_config = read_json(V194_ADAPTER / 'adapter_config.json')
v194_weights = V194_ADAPTER / 'adapter_model.safetensors'
with safe_open(v194_weights, framework='pt', device='cpu') as handle:
    tensor_count = len(handle.keys())
print('V194 adapter_config.json =', V194_ADAPTER / 'adapter_config.json', flush=True)
print('V194 adapter_model.safetensors =', v194_weights, 'bytes =', v194_weights.stat().st_size, 'tensor_count =', tensor_count, flush=True)
print('target_modules =', v194_config.get('target_modules'), flush=True)
print('target_parameters =', v194_config.get('target_parameters'), flush=True)
if set(v194_config.get('target_modules') or []) != set(EXPECTED_V194_TARGET_MODULES):
    raise RuntimeError('V194 target_modules mismatch')
if list(v194_config.get('target_parameters') or []) != EXPECTED_V194_TARGET_PARAMETERS:
    raise RuntimeError('V194 target_parameters mismatch')
if v194_weights.stat().st_size != EXPECTED_V194_ADAPTER_BYTES:
    raise RuntimeError('V194 adapter size mismatch')
if tensor_count != EXPECTED_V194_ADAPTER_TENSOR_COUNT:
    raise RuntimeError('V194 adapter tensor count mismatch')

from src.competition_utils import classify_puzzle
full_df = pd.read_csv(V194_VAL_CSV)
if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
    raise RuntimeError('V194 validation CSV must contain prompt and answer')
if 'type' not in full_df.columns:
    full_df['type'] = full_df['prompt'].map(classify_puzzle)
full_eval_csv = EVAL_OUT / 'v221_full_947.csv'
weak_eval_csv = EVAL_OUT / 'v221_weak_315.csv'
strong_eval_csv = EVAL_OUT / 'v221_strong_632.csv'
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

resolved_candidates = []
candidate_resolution_rows = []
for candidate in CANDIDATE_REGISTRY['candidates']:
    status = download_or_resolve_candidate(candidate)
    candidate_resolution_rows.append({**candidate, **status})
    print('candidate_resolution =', json.dumps(status, indent=2, sort_keys=True), flush=True)
    if status['ready']:
        resolved_candidates.append({'name': status['name'], 'adapter': status['adapter']})
    elif status.get('required'):
        raise RuntimeError('required candidate is not ready: ' + json.dumps(status, sort_keys=True))

if V221_MAX_CANDIDATES > 0:
    resolved_candidates = resolved_candidates[:V221_MAX_CANDIDATES]
ready_candidates_path = EVAL_OUT / 'v221_ready_candidates.json'
resolution_csv = EVAL_OUT / 'v221_candidate_resolution.csv'
ready_candidates_path.write_text(json.dumps({'candidates': resolved_candidates}, indent=2, sort_keys=True), encoding='utf-8')
pd.DataFrame(candidate_resolution_rows).to_csv(resolution_csv, index=False)
print('ready_candidate_count =', len(resolved_candidates), flush=True)
print('ready_candidates_path =', ready_candidates_path, flush=True)
print('resolution_csv =', resolution_csv, flush=True)
if not resolved_candidates:
    raise RuntimeError('no candidates ready for weak eval')
print('=== V221 PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: weak-only batch adapter A/B with one vLLM model load.
print('=== V221 WEAK BATCH EVAL START ===', flush=True)
weak_gate_pass_for_full = False
best_candidate = None
candidate_rows = []
batch_summary_json = EVAL_OUT / 'batch_candidate_summary.json'
batch_summary_csv = EVAL_OUT / 'batch_candidate_summary.csv'

if not RUN_EVAL:
    print('RUN_EVAL is false; skipping V221 weak batch eval.', flush=True)
else:
    ensure_vllm_for_eval()
    run_new_eval = FORCE_REEVAL or not batch_summary_json.exists()
    print('run_new_eval =', run_new_eval, flush=True)
    if run_new_eval:
        cmd = [
            sys.executable,
            str(ROOT / 'scripts/evaluate_lora_adapters_batch.py'),
            '--solution-csv', str(weak_eval_csv),
            '--questions-csv', str(weak_eval_csv),
            '--candidates-json', str(ready_candidates_path),
            '--base-model-path', MODEL_NAME,
            '--label-prefix', 'v221_weak',
            '--seed', '42',
            '--limit', '0',
            '--output-dir', str(EVAL_OUT),
            '--max-tokens', str(V221_MAX_TOKENS),
            '--max-model-len', str(V221_MAX_MODEL_LEN),
            '--max-num-seqs', str(V221_MAX_NUM_SEQS),
            '--warmup-rows', str(V221_WARMUP_ROWS),
            '--prompt-suffix', V221_PROMPT_SUFFIX,
            '--continue-on-error',
        ]
        rc = run_cmd(cmd, cwd=ROOT, log_path=EVAL_OUT / 'weak_batch_eval.log', check=True, heartbeat_s=60)
        print('weak batch eval returncode =', rc, flush=True)
    else:
        print('reusing batch summary:', batch_summary_json, flush=True)
    batch_summary = read_json(batch_summary_json)
    candidate_rows = batch_summary.get('rows', [])
    print('candidate_rows =', json.dumps(candidate_rows, indent=2, sort_keys=True), flush=True)
    ok_rows = [row for row in candidate_rows if row.get('status') == 'ok']
    best_candidate = sorted(
        ok_rows,
        key=lambda item: (
            int(item.get('correct', 0)),
            int(item.get('equation_transform_correct', 0)),
            int(item.get('bit_manipulation_correct', 0)),
            -int(item.get('truncated', 999999)),
        ),
        reverse=True,
    )[0] if ok_rows else None
    print('best_candidate =', json.dumps(best_candidate, indent=2, sort_keys=True), flush=True)
    if best_candidate:
        weak_gate_pass_for_full = (
            int(best_candidate.get('correct', 0)) >= WEAK_MIN_FOR_FULL
            and int(best_candidate.get('equation_transform_correct', 0)) >= WEAK_EQ_MIN_FOR_FULL
            and int(best_candidate.get('bit_manipulation_correct', 0)) >= WEAK_BIT_MIN_FOR_FULL
            and int(best_candidate.get('truncated', 999999)) <= WEAK_MAX_TRUNC_FOR_FULL
        )
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('batch_summary_json =', batch_summary_json, flush=True)
print('batch_summary_csv =', batch_summary_csv, flush=True)
print('=== V221 WEAK BATCH EVAL END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval hard gate. Default blocked.
print('=== V221 FULL EVAL GATE START ===', flush=True)
full_report = None
full_candidate_gate = False
if not weak_gate_pass_for_full:
    print('Weak gate failed or did not run; full eval blocked.', flush=True)
    print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
elif not RUN_FULL_IF_GATE:
    print('Weak gate passed, but RUN_FULL_IF_GATE is false. Full eval is blocked by default to avoid accidental GPU spend.', flush=True)
else:
    print('Full eval is intentionally not automatic in V221 candidate registry notebook.', flush=True)
    print('Manual next step: run a dedicated full-eval notebook after reviewing weak family no-loss results.', flush=True)
print('=== V221 FULL EVAL GATE END ===', flush=True)
"""
        ),
        code(
            """# CELL: write V221 final manifest. No submit.
print('=== V221 FINAL MANIFEST START ===', flush=True)
decision = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_train': RUN_TRAIN,
    'run_eval': RUN_EVAL,
    'run_full_if_gate': RUN_FULL_IF_GATE,
    'registry_seed_path': str(registry_seed_path),
    'ready_candidates_path': str(globals().get('ready_candidates_path', '')),
    'candidate_resolution_csv': str(globals().get('resolution_csv', '')),
    'candidate_rows': candidate_rows,
    'best_candidate': best_candidate,
    'weak_gate_pass_for_full': bool(weak_gate_pass_for_full),
    'full_report': full_report,
    'full_candidate_gate': bool(full_candidate_gate),
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
}
if not weak_gate_pass_for_full:
    decision['roadmap_next'] = 'Do not full-eval or submit. Use per-family weak results to select adapters or build solver overrides.'
elif not RUN_FULL_IF_GATE:
    decision['roadmap_next'] = 'Review weak family results and enable a separate full-eval run only if no-loss criteria are satisfied.'
else:
    decision['roadmap_next'] = 'Manual review required; notebook still has hard submit lock.'
manifest_path = OUT_ROOT / 'v221_candidate_registry_weak_ab_manifest.json'
manifest_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in this notebook.')
print('=== V221 FINAL MANIFEST END ===', flush=True)
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

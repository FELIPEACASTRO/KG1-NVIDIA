#!/usr/bin/env python3
"""Build the V207B external/current adapter triage Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb")
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v207b-{prefix}-{_CELL_COUNTER:02d}"


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


def embed_sources_cell() -> str:
    files = {
        "src/__init__.py": read_text("src/__init__.py"),
        "src/competition_utils.py": read_text("src/competition_utils.py"),
        "scripts/evaluate_lora_adapter.py": read_text("scripts/evaluate_lora_adapter.py"),
        "scripts/solve_rate_gate.py": read_text("scripts/solve_rate_gate.py"),
    }
    payload = json.dumps(files, ensure_ascii=False, indent=2)
    return f"""# CELL: install/repair V207B metric scripts inside the cloned repo.
print('=== V207B SCRIPT BOOTSTRAP START ===')
import json, pathlib, py_compile
ROOT = pathlib.Path('/content/kg1')
FILES = json.loads({json.dumps(payload)})
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    print('wrote', path, 'bytes=', path.stat().st_size)
for rel in ['src/competition_utils.py', 'scripts/evaluate_lora_adapter.py', 'scripts/solve_rate_gate.py']:
    py_compile.compile(str(ROOT / rel), doraise=True)
    print('compiled', rel)
print('=== V207B SCRIPT BOOTSTRAP END ===')
"""


def build_notebook() -> dict:
    cells = [
        md(
            """# KG1 V207B External Adapter Triage Colab

Purpose: continue the V207 roadmap after V206B/V206C/V214 were rejected.

This notebook:

- reuses the V207A official-like validation artifacts already saved in Drive;
- downloads public Kaggle model adapters into Drive for gated testing;
- audits external/current adapter structures before spending H100/A100 time;
- screens only the weak families first: `equation_transform` and `bit_manipulation`;
- runs full 947-row official-like ACC only for weak-positive candidates;
- never trains and never submits to Kaggle.

Expected Colab URL after this notebook is pushed:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v207b-external-triage/notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb`
"""
        ),
        code(
            """# CELL: mount Drive.
print('=== V207B DRIVE MOUNT START ===')
from google.colab import drive
drive.mount('/content/drive')
print('=== V207B DRIVE MOUNT END ===')
"""
        ),
        code(
            f"""# CELL: runtime configuration.
print('=== V207B CONFIG START ===')
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V207B_EXTERNAL_ADAPTER_TRIAGE_20260507_PUBLIC_DOWNLOADS'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'v207a-colab')
ROOT = pathlib.Path('/content/kg1')
DRIVE_MY = pathlib.Path('/content/drive/MyDrive')
V207A_ROOT = DRIVE_MY / 'KG1_NVIDIA_V207A' / 'output_v207a_acc_gate'
OUT_ROOT = DRIVE_MY / 'KG1_NVIDIA_V207B' / 'output_v207b_external_adapter_triage'
REPORT_DIR = OUT_ROOT / 'reports'
PUBLIC_KAGGLE_ROOT = DRIVE_MY / 'KG1_PUBLIC_ADAPTERS'
FALLBACK_EXPORT_BASE = os.environ.get(
    'KG1_V207B_FALLBACK_EXPORT_BASE',
    'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/v207b-external-triage/artifacts/drive_exports',
)
MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

VAL_CSV = V207A_ROOT / 'validation' / 'official_train_seed42_stratified10_val.csv'
VAL_WEAK_CSV = OUT_ROOT / 'validation' / 'official_train_seed42_stratified10_val_weak_families.csv'
BASELINE_PREDICTIONS = V207A_ROOT / 'v194_baseline_eval' / 'v194_baseline_predictions.csv'
BASELINE_PER_TASK = V207A_ROOT / 'v194_baseline_eval' / 'v194_baseline_per_task.csv'
BASELINE_REPORT = V207A_ROOT / 'v194_baseline_eval' / 'v194_baseline_eval_report.json'

WEAK_FAMILIES = ['bit_manipulation', 'equation_transform']
FORCE_REEVAL = os.environ.get('KG1_V207B_FORCE_REEVAL', '0') == '1'
RUN_FULL_FOR_POSITIVE = os.environ.get('KG1_V207B_RUN_FULL_FOR_POSITIVE', '1') == '1'
HASH_WEIGHTS = os.environ.get('KG1_V207B_HASH_WEIGHTS', '0') == '1'
MAX_DISCOVERY_DIRS = int(os.environ.get('KG1_V207B_MAX_DISCOVERY_DIRS', '25000'))
INCLUDE_REJECTED_V206 = os.environ.get('KG1_V207B_INCLUDE_REJECTED_V206', '0') == '1'
RUN_KAGGLE_PUBLIC_DOWNLOAD = os.environ.get('KG1_V207B_RUN_KAGGLE_PUBLIC_DOWNLOAD', '1') == '1'
PUBLIC_DOWNLOAD_MAX_PRIORITY = int(os.environ.get('KG1_V207B_PUBLIC_DOWNLOAD_MAX_PRIORITY', '2'))
PUBLIC_DOWNLOAD_MAX_CANDIDATES = int(os.environ.get('KG1_V207B_PUBLIC_DOWNLOAD_MAX_CANDIDATES', '13'))
ALLOW_KAGGLE_SUBMIT = False

for path in [OUT_ROOT, REPORT_DIR, VAL_WEAK_CSV.parent, PUBLIC_KAGGLE_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION)
print('REPO_URL =', REPO_URL)
print('REPO_BRANCH =', REPO_BRANCH)
print('ROOT =', ROOT)
print('V207A_ROOT =', V207A_ROOT)
print('OUT_ROOT =', OUT_ROOT)
print('PUBLIC_KAGGLE_ROOT =', PUBLIC_KAGGLE_ROOT)
print('FALLBACK_EXPORT_BASE =', FALLBACK_EXPORT_BASE)
print('VAL_CSV =', VAL_CSV)
print('BASELINE_PREDICTIONS =', BASELINE_PREDICTIONS)
print('BASELINE_PER_TASK =', BASELINE_PER_TASK)
print('MODEL_NAME =', MODEL_NAME)
print('MODEL_REVISION =', MODEL_REVISION)
print('WEAK_FAMILIES =', WEAK_FAMILIES)
print('FORCE_REEVAL =', FORCE_REEVAL)
print('RUN_FULL_FOR_POSITIVE =', RUN_FULL_FOR_POSITIVE)
print('HASH_WEIGHTS =', HASH_WEIGHTS)
print('MAX_DISCOVERY_DIRS =', MAX_DISCOVERY_DIRS)
print('INCLUDE_REJECTED_V206 =', INCLUDE_REJECTED_V206)
print('RUN_KAGGLE_PUBLIC_DOWNLOAD =', RUN_KAGGLE_PUBLIC_DOWNLOAD)
print('PUBLIC_DOWNLOAD_MAX_PRIORITY =', PUBLIC_DOWNLOAD_MAX_PRIORITY)
print('PUBLIC_DOWNLOAD_MAX_CANDIDATES =', PUBLIC_DOWNLOAD_MAX_CANDIDATES)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook is submit-disabled by design.')
print('=== V207B CONFIG END ===')
"""
        ),
        code(
            """# CELL: helper functions and command logging.
print('=== V207B HELPERS START ===')
import importlib
import json
import pathlib
import subprocess
import sys
import time

def run_cmd(cmd, cwd=None, env=None, log_path=None, check=True):
    cmd = [str(x) for x in cmd]
    print('+', ' '.join(cmd))
    log_handle = None
    if log_path is not None:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8')
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
    for line in proc.stdout:
        print(line, end='')
        if log_handle:
            log_handle.write(line)
    rc = proc.wait()
    if log_handle:
        log_handle.close()
    print('returncode =', rc)
    if check and rc != 0:
        raise RuntimeError(f'Command failed with rc={rc}: {cmd}')
    return rc

def ensure_import(import_name, pip_spec=None):
    try:
        mod = importlib.import_module(import_name)
        print(import_name, 'version=', getattr(mod, '__version__', 'unknown'))
        return mod
    except Exception as exc:
        print(import_name, 'missing/import failed:', repr(exc))
        if not pip_spec:
            raise
        run_cmd([sys.executable, '-m', 'pip', 'install', '-q', pip_spec])
        mod = importlib.import_module(import_name)
        print(import_name, 'version=', getattr(mod, '__version__', 'unknown'))
        return mod

def fresh_python_import_check(import_name):
    code = (
        "import importlib, torch; "
        f"m=importlib.import_module('{import_name}'); "
        "print('fresh_python_import_ok', m.__name__, getattr(m, '__version__', 'unknown')); "
        "print('fresh_python_torch', torch.__version__, getattr(torch.version, 'cuda', 'unknown'))"
    )
    run_cmd([sys.executable, '-c', code])

def sha256_file(path, enabled=True):
    path = pathlib.Path(path)
    if not enabled:
        return ''
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_label(text):
    text = str(text).strip().replace('\\\\', '/')
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_.-')
    return text[-120:] or 'candidate'

print('python =', sys.version)
print('=== V207B HELPERS END ===')
"""
        ),
        code(
            """# CELL: install dependencies and validate GPU/vLLM import path.
print('=== V207B DEPENDENCY CHECK START ===')
ensure_import('pandas', 'pandas')
ensure_import('huggingface_hub', 'huggingface_hub')
ensure_import('transformers', 'transformers')
ensure_import('peft', 'peft')
ensure_import('safetensors', 'safetensors')
run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
run_cmd(['kaggle', '--version'])
torch = ensure_import('torch')
print('torch_cuda_available =', torch.cuda.is_available())
print('torch_cuda_device_count =', torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    print('torch_cuda_device_name =', torch.cuda.get_device_name(0))
    print('torch_cuda_version =', getattr(torch.version, 'cuda', 'unknown'))
try:
    ensure_import('vllm')
except Exception as exc:
    print('vLLM unavailable in current kernel before install:', repr(exc))
    print('Installing vLLM. Evaluation will run in fresh Python subprocesses.')
    run_cmd([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        '--extra-index-url',
        'https://download.pytorch.org/whl/cu128',
        'vllm',
    ])
    fresh_python_import_check('vllm')
else:
    fresh_python_import_check('vllm')
print('=== V207B DEPENDENCY CHECK END ===')
"""
        ),
        code(
            """# CELL: clone repo and install embedded metric scripts.
print('=== V207B REPO SETUP START ===')
if ROOT.exists():
    print('Repo exists; status follows:', ROOT)
    run_cmd(['git', 'status', '--short'], cwd=ROOT, check=False)
else:
    run_cmd(['git', 'clone', '--branch', REPO_BRANCH, '--depth', '1', REPO_URL, str(ROOT)])
run_cmd(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, check=False)
run_cmd(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=False)
print('=== V207B REPO SETUP END ===')
"""
        ),
        code(embed_sources_cell()),
        code(
            """# CELL: verify or bootstrap V207A baseline artifacts and build weak-family CSV.
print('=== V207B V207A ARTIFACT CHECK START ===')
import urllib.request
import pandas as pd
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.competition_utils import classify_puzzle

FALLBACK_EXPORTS = {
    BASELINE_PREDICTIONS: 'v194_baseline_predictions.csv',
    BASELINE_PER_TASK: 'v194_baseline_per_task.csv',
    BASELINE_REPORT: 'v194_baseline_eval_report.json',
}

def download_fallback_export(dst, filename):
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = FALLBACK_EXPORT_BASE.rstrip('/') + '/' + filename
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    print('fallback_download_url =', url)
    print('fallback_download_dst =', dst)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print('fallback_downloaded_bytes =', dst.stat().st_size)

def ensure_fallback_file(path, filename):
    path = pathlib.Path(path)
    print('checking', path, 'exists=', path.exists())
    if not path.exists():
        print('missing artifact; downloading validated fallback export:', filename)
        download_fallback_export(path, filename)
    print('artifact_ready =', path, 'bytes=', path.stat().st_size)

for path, filename in FALLBACK_EXPORTS.items():
    ensure_fallback_file(path, filename)

if not VAL_CSV.exists():
    print('VAL_CSV missing; reconstructing validation CSV from baseline predictions export.')
    pred = pd.read_csv(BASELINE_PREDICTIONS)
    prompt_col = next((c for c in ['prompt', 'prompt_x', 'prompt_y'] if c in pred.columns), None)
    family_source_col = next((c for c in ['family', 'type'] if c in pred.columns), None)
    required = {'id', 'answer'}
    missing = sorted(required - set(pred.columns))
    if missing or prompt_col is None:
        raise RuntimeError(
            f'Baseline predictions fallback cannot reconstruct validation CSV. '
            f'missing={missing}, prompt_col={prompt_col}, columns={list(pred.columns)}'
        )
    val_cols = ['id', prompt_col, 'answer']
    if family_source_col:
        val_cols.append(family_source_col)
    val = pred[val_cols].copy()
    val = val.rename(columns={prompt_col: 'prompt'})
    if family_source_col and family_source_col != 'family':
        val = val.rename(columns={family_source_col: 'family'})
    if 'family' not in val.columns:
        val['family'] = val['prompt'].map(classify_puzzle)
    val = val[['id', 'prompt', 'answer', 'family']]
    val.to_csv(VAL_CSV, index=False)
    print('VAL_CSV reconstructed rows =', len(val))
    print('validation_source = v194_baseline_predictions_drive_export_fallback')
else:
    print('VAL_CSV already exists:', VAL_CSV)

required_paths = [VAL_CSV, BASELINE_PREDICTIONS, BASELINE_PER_TASK, BASELINE_REPORT]
for path in required_paths:
    print('final_check', path, 'exists=', path.exists(), 'bytes=', path.stat().st_size if path.exists() else None)
    if not path.exists():
        raise FileNotFoundError('Missing required V207B artifact after fallback bootstrap: ' + str(path))

val = pd.read_csv(VAL_CSV)
family_col = 'family' if 'family' in val.columns else 'type'
if family_col not in val.columns:
    raise RuntimeError('Validation CSV needs family or type column.')
weak = val[val[family_col].isin(WEAK_FAMILIES)].copy()
weak.to_csv(VAL_WEAK_CSV, index=False)

base_per = pd.read_csv(BASELINE_PER_TASK)
base_weak = base_per[base_per['task_type'].isin(WEAK_FAMILIES)]
BASE_WEAK_CORRECT = int(base_weak['correct'].sum())
BASE_WEAK_TOTAL = int(base_weak['total'].sum())
BASE_WEAK_TRUNCATED = int(base_weak['truncated'].sum()) if 'truncated' in base_weak.columns else 0

print('VAL rows =', len(val))
print('VAL_WEAK_CSV =', VAL_WEAK_CSV, 'exists=', VAL_WEAK_CSV.exists())
print('weak rows =', len(weak))
print('weak family counts =')
print(weak[family_col].value_counts().sort_index().to_string())
print('baseline weak correct =', BASE_WEAK_CORRECT, '/', BASE_WEAK_TOTAL)
print('baseline weak truncated =', BASE_WEAK_TRUNCATED, '/', BASE_WEAK_TOTAL)
print('=== V207B V207A ARTIFACT CHECK END ===')
"""
        ),
        code(
            """# CELL: download public Kaggle adapter candidates into Drive.
print('=== V207B PUBLIC KAGGLE ADAPTER DOWNLOAD START ===')
from pathlib import Path

PUBLIC_KAGGLE_MODEL_CANDIDATES = [
    # Priority 1: Huikang public adapter versions used by public competition notebooks.
    {'label': 'huikang_default_v27', 'ref': 'huikang/nemotron-adapter/Transformers/default/27', 'priority': 1},
    {'label': 'huikang_default_v26', 'ref': 'huikang/nemotron-adapter/Transformers/default/26', 'priority': 1},
    {'label': 'huikang_default_v25', 'ref': 'huikang/nemotron-adapter/Transformers/default/25', 'priority': 1},
    {'label': 'huikang_default_v24', 'ref': 'huikang/nemotron-adapter/Transformers/default/24', 'priority': 1},
    {'label': 'huikang_default_v23', 'ref': 'huikang/nemotron-adapter/Transformers/default/23', 'priority': 1},
    {'label': 'huikang_default_v22', 'ref': 'huikang/nemotron-adapter/Transformers/default/22', 'priority': 1},
    {'label': 'huikang_default_v21', 'ref': 'huikang/nemotron-adapter/Transformers/default/21', 'priority': 1},
    {'label': 'huikang_default_v20', 'ref': 'huikang/nemotron-adapter/Transformers/default/20', 'priority': 1},

    # Priority 2: Kienngx variations referenced by public notebooks and model listings.
    {'label': 'kienngx_1200samples_cot_1e_5', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/1200samples-cot-1e-5/1', 'priority': 2},
    {'label': 'kienngx_1200samples_cot_5e_5', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/1200samples-cot-5e-5/1', 'priority': 2},
    {'label': 'kienngx_cot_labels_3000samples', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/cot-labels-3000samples/1', 'priority': 2},
    {'label': 'kienngx_600_samples_packing_false', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/600-samples-packing-false/1', 'priority': 2},
    {'label': 'kienngx_1800s_lora_rank32_false', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/1800s-lora-rank32-false/1', 'priority': 2},

    # Priority 3: extra variants. Keep default download priority at 2 to control Drive usage/time.
    {'label': 'kienngx_tinker_adapter', 'ref': 'kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1', 'priority': 3},
    {'label': 'kienngx_2400_1e_4_lr_all_linear_packingfalse', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/2400-1e-4_lr-all_linear-packingfalse/1', 'priority': 3},
    {'label': 'kienngx_9500s_batch1_lr1e_4', 'ref': 'kienngx/nemotron-nano-30b-trained/Transformers/9500s-batch1-lr1e-4/1', 'priority': 3},
]

PUBLIC_KAGGLE_EXPECTED_BYTES = {
    'huikang_default_v27': 1544348352,
    'huikang_default_v26': 1544348352,
    'huikang_default_v25': 1544348352,
    'huikang_default_v24': 772202848,
    'huikang_default_v23': 1544348352,
    'huikang_default_v22': 1544348352,
    'huikang_default_v21': 1544348352,
    'huikang_default_v20': 1544348352,
    'kienngx_1200samples_cot_1e_5': 3537299144,
    'kienngx_1200samples_cot_5e_5': 3537299144,
    'kienngx_cot_labels_3000samples': 3537299144,
    'kienngx_600_samples_packing_false': 1740420752,
    'kienngx_1800s_lora_rank32_false': 3479065680,
    'kienngx_tinker_adapter': 3554384888,
    'kienngx_2400_1e_4_lr_all_linear_packingfalse': 3537299144,
    'kienngx_9500s_batch1_lr1e_4': 58233016,
}
for _item in PUBLIC_KAGGLE_MODEL_CANDIDATES:
    _item['expected_bytes'] = PUBLIC_KAGGLE_EXPECTED_BYTES.get(_item['label'], 0)

def kaggle_adapter_ready(path):
    path = Path(path)
    return (
        path.is_dir()
        and (path / 'adapter_config.json').exists()
        and ((path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists())
    )

def configure_kaggle_credentials():
    kaggle_dir = Path('/root/.kaggle')
    token_path = kaggle_dir / 'kaggle.json'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    if token_path.exists():
        token_path.chmod(0o600)
        print('kaggle_token =', token_path, 'exists=True')
        return True

    candidate_paths = [
        DRIVE_MY / 'kaggle.json',
        DRIVE_MY / 'KG1_SECRETS' / 'kaggle.json',
        DRIVE_MY / '.kaggle' / 'kaggle.json',
    ]
    env_config_dir = os.environ.get('KAGGLE_CONFIG_DIR', '').strip()
    if env_config_dir:
        candidate_paths.append(Path(env_config_dir) / 'kaggle.json')

    for src in candidate_paths:
        print('checking_kaggle_token_source =', src, 'exists=', src.exists())
        if src.exists():
            shutil.copy2(src, token_path)
            token_path.chmod(0o600)
            print('copied_kaggle_token =', src, '->', token_path)
            return True

    print('Kaggle credentials were not found.')
    print('Place kaggle.json at /content/drive/MyDrive/kaggle.json or /content/drive/MyDrive/KG1_SECRETS/kaggle.json, then rerun this cell.')
    return False

download_status = []

if not RUN_KAGGLE_PUBLIC_DOWNLOAD:
    print('RUN_KAGGLE_PUBLIC_DOWNLOAD=False; skipping public model downloads.')
else:
    if not configure_kaggle_credentials():
        raise RuntimeError('Human action required: add Kaggle API token kaggle.json to Drive and rerun this cell.')
    run_cmd(['kaggle', '--version'])

    selected_candidates = [
        item for item in PUBLIC_KAGGLE_MODEL_CANDIDATES
        if int(item['priority']) <= PUBLIC_DOWNLOAD_MAX_PRIORITY
    ][:PUBLIC_DOWNLOAD_MAX_CANDIDATES]
    print('selected_public_download_count =', len(selected_candidates))
    remaining_expected_bytes = sum(
        int(item.get('expected_bytes') or 0)
        for item in selected_candidates
        if not kaggle_adapter_ready(PUBLIC_KAGGLE_ROOT / safe_label(item['label']) / 'adapter')
    )
    usage = shutil.disk_usage(PUBLIC_KAGGLE_ROOT)
    print('download_expected_remaining_gib =', round(remaining_expected_bytes / (1024 ** 3), 2))
    print('drive_total_gib =', round(usage.total / (1024 ** 3), 2))
    print('drive_free_gib =', round(usage.free / (1024 ** 3), 2))
    if remaining_expected_bytes and usage.free < remaining_expected_bytes + 5 * 1024 ** 3:
        raise RuntimeError(
            'Human action required: not enough free Drive space for public adapter downloads. '
            f'Need about {remaining_expected_bytes / (1024 ** 3):.2f} GiB plus 5 GiB buffer; '
            f'free={usage.free / (1024 ** 3):.2f} GiB.'
        )
    for item in selected_candidates:
        label = safe_label(item['label'])
        ref = item['ref']
        target = PUBLIC_KAGGLE_ROOT / label / 'adapter'
        log = REPORT_DIR / f'download_{label}.log'
        target.mkdir(parents=True, exist_ok=True)

        print('\\n' + '=' * 80)
        print('download_label =', label)
        print('download_ref =', ref)
        print('download_priority =', item['priority'])
        print('download_target =', target)
        print('already_ready =', kaggle_adapter_ready(target))

        if kaggle_adapter_ready(target):
            status = 'already_ready'
            rc = 0
        else:
            cmd = [
                'kaggle',
                'models',
                'instances',
                'versions',
                'download',
                ref,
                '-p',
                target,
                '--untar',
            ]
            rc = run_cmd(cmd, log_path=log, check=False)
            status = 'downloaded_ready' if rc == 0 and kaggle_adapter_ready(target) else f'download_or_structure_failed_{rc}'

        nested_ready_dirs = []
        if not kaggle_adapter_ready(target):
            for dirpath, dirnames, filenames in os.walk(target):
                p = Path(dirpath)
                if kaggle_adapter_ready(p):
                    nested_ready_dirs.append(str(p))

        top_files = sorted([p.name for p in target.glob('*')])[:25] if target.exists() else []
        row = {
            'label': label,
            'ref': ref,
            'priority': int(item['priority']),
            'target': str(target),
            'ready': kaggle_adapter_ready(target),
            'nested_ready_dirs': nested_ready_dirs,
            'status': status,
            'returncode': rc,
            'top_files': top_files,
            'log': str(log),
        }
        download_status.append(row)
        print('download_status =', json.dumps(row, indent=2, sort_keys=True))

download_manifest = OUT_ROOT / 'v207b_public_kaggle_download_manifest.json'
download_manifest.write_text(json.dumps(download_status, indent=2, sort_keys=True), encoding='utf-8')
print('download_manifest =', download_manifest)
print('=== V207B PUBLIC KAGGLE ADAPTER DOWNLOAD END ===')
"""
        ),
        code(
            """# CELL: discover and register candidate adapter directories.
print('=== V207B CANDIDATE DISCOVERY START ===')
from pathlib import Path

MANUAL_CANDIDATES = [
    # Baseline sanity / tied artifacts.
    ('v194_init_duplicate', DRIVE_MY / 'KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter'),
    ('v194_final_keep_duplicate', DRIVE_MY / 'KG1_NVIDIA_V202D/final_v194_keep_no_submit/adapter'),
    ('v199b_candidate', DRIVE_MY / 'KG1_NVIDIA_V199B/final_adapter'),
    ('v199b_candidate_adapter', DRIVE_MY / 'KG1_NVIDIA_V199B/adapter'),

    # Common external/current adapter landing zones. Missing paths are skipped.
    ('aaitdads_my_0p86', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/aaitdads_my_0p86_adapter'),
    ('huikang_default_v27', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v27/adapter'),
    ('huikang_default_v26', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v26/adapter'),
    ('huikang_default_v25', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v25/adapter'),
    ('huikang_default_v24', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v24/adapter'),
    ('huikang_default_v23', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v23/adapter'),
    ('huikang_default_v22', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v22/adapter'),
    ('huikang_default_v21', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v21/adapter'),
    ('huikang_default_v20', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_default_v20/adapter'),
    ('huikang_tinker_v27_legacy', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_tinker_v27/adapter'),
    ('huikang_tinker_v26_legacy', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_tinker_v26/adapter'),
    ('huikang_tinker_v20_legacy', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/huikang_tinker_v20/adapter'),
    ('kienngx_1200samples_cot_1e_5', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_1200samples_cot_1e_5/adapter'),
    ('kienngx_1200samples_cot_5e_5', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_1200samples_cot_5e_5/adapter'),
    ('kienngx_cot_labels_3000samples', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_cot_labels_3000samples/adapter'),
    ('kienngx_600_samples_packing_false', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_600_samples_packing_false/adapter'),
    ('kienngx_1800s_lora_rank32_false', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_1800s_lora_rank32_false/adapter'),
    ('kienngx_tinker_adapter', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_tinker_adapter/adapter'),
    ('kienngx_2400_1e_4_lr_all_linear_packingfalse', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_2400_1e_4_lr_all_linear_packingfalse/adapter'),
    ('kienngx_9500s_batch1_lr1e_4', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kienngx_9500s_batch1_lr1e_4/adapter'),
    ('kien_variant_legacy', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/kien_variant/adapter'),
    ('bugkeeper_v20', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/bugkeeper_v20/adapter'),
    ('dgxchen_trained', DRIVE_MY / 'KG1_PUBLIC_ADAPTERS/dgxchen_trained_adapter'),
]

DISCOVERY_ROOTS = [
    DRIVE_MY / 'KG1_PUBLIC_ADAPTERS',
    DRIVE_MY / 'KG1_NVIDIA_PUBLIC_ADAPTERS',
    DRIVE_MY / 'KG1_NVIDIA_EXTERNAL',
    DRIVE_MY / 'KG1_NVIDIA_V199B',
    DRIVE_MY / 'KG1_NVIDIA_V202D',
    DRIVE_MY / 'KG1_NVIDIA_V204',
    DRIVE_MY / 'KG1_NVIDIA_TINKER',
    DRIVE_MY / 'KG1_NVIDIA_KIEN',
]

REJECTED_SUBSTRINGS = [
    'KG1_NVIDIA_V206B',
    'KG1_NVIDIA_V206C',
    'adapter_s0p010',
    'adapter_s0p020',
    'adapter_s0p050',
    'adapter_s0p100',
]

def adapter_ready_dir(path):
    path = Path(path)
    return (
        path.is_dir()
        and (path / 'adapter_config.json').exists()
        and ((path / 'adapter_model.safetensors').exists() or (path / 'adapter_model.bin').exists())
    )

def is_rejected_path(path):
    if INCLUDE_REJECTED_V206:
        return False
    text = str(path)
    return any(chunk in text for chunk in REJECTED_SUBSTRINGS)

candidates = {}
for label, path in MANUAL_CANDIDATES:
    path = Path(path)
    print('manual_candidate', label, path, 'ready=', adapter_ready_dir(path))
    if adapter_ready_dir(path) and not is_rejected_path(path):
        candidates[str(path.resolve())] = {'label': safe_label(label), 'path': path}

for root in DISCOVERY_ROOTS:
    print('scan_root =', root, 'exists=', root.exists())
    if not root.exists():
        continue
    scanned_dirs = 0
    for dirpath, dirnames, filenames in os.walk(root):
        scanned_dirs += 1
        if scanned_dirs > MAX_DISCOVERY_DIRS:
            print('scan_root_limit_reached', root, MAX_DISCOVERY_DIRS)
            break
        p = Path(dirpath)
        if is_rejected_path(p):
            dirnames[:] = []
            continue
        names = set(filenames)
        if 'adapter_config.json' in names and (
            'adapter_model.safetensors' in names or 'adapter_model.bin' in names
        ):
            label = safe_label(str(p.relative_to(root)))
            candidates[str(p.resolve())] = {'label': label, 'path': p}
    print('scan_root_done =', root, 'scanned_dirs=', scanned_dirs)

CANDIDATES = list(candidates.values())
print('candidate_count =', len(CANDIDATES))
for item in CANDIDATES:
    print('candidate =', item['label'], item['path'])

manifest_path = OUT_ROOT / 'v207b_discovered_candidates.json'
manifest_path.write_text(
    json.dumps(
        [{'label': x['label'], 'path': str(x['path'])} for x in CANDIDATES],
        indent=2,
        sort_keys=True,
    ),
    encoding='utf-8',
)
print('candidate_manifest =', manifest_path)
print('=== V207B CANDIDATE DISCOVERY END ===')
"""
        ),
        code(
            """# CELL: structure audit candidate adapters.
print('=== V207B STRUCTURE AUDIT START ===')
from safetensors import safe_open

def audit_adapter(label, path):
    path = Path(path)
    cfg_path = path / 'adapter_config.json'
    model_path = path / 'adapter_model.safetensors'
    bin_path = path / 'adapter_model.bin'
    row = {
        'label': label,
        'path': str(path),
        'exists': path.exists(),
        'has_config': cfg_path.exists(),
        'has_safetensors': model_path.exists(),
        'has_bin': bin_path.exists(),
        'model_bytes': 0,
        'config_sha256': '',
        'model_sha256': '',
        'peft_type': '',
        'r': '',
        'lora_alpha': '',
        'target_modules': '',
        'base_model_name_or_path': '',
        'tensor_count': 0,
        'bad_lm_head_namespace_count': 0,
        'rank_ok': False,
        'ready_for_eval': False,
        'reason': '',
    }
    if not path.exists():
        row['reason'] = 'missing_path'
        return row
    if not cfg_path.exists():
        row['reason'] = 'missing_adapter_config'
        return row
    if not model_path.exists() and not bin_path.exists():
        row['reason'] = 'missing_adapter_weights'
        return row

    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    row['config_sha256'] = sha256_file(cfg_path, True)
    row['peft_type'] = str(cfg.get('peft_type', ''))
    row['r'] = str(cfg.get('r', ''))
    row['lora_alpha'] = str(cfg.get('lora_alpha', ''))
    row['target_modules'] = json.dumps(cfg.get('target_modules', ''), sort_keys=True)
    row['base_model_name_or_path'] = str(cfg.get('base_model_name_or_path', ''))
    try:
        rank = int(cfg.get('r', 0))
    except Exception:
        rank = 0
    row['rank_ok'] = 0 < rank <= 32

    weight_path = model_path if model_path.exists() else bin_path
    row['model_bytes'] = int(weight_path.stat().st_size)
    row['model_sha256'] = sha256_file(weight_path, HASH_WEIGHTS)

    if model_path.exists():
        try:
            with safe_open(str(model_path), framework='pt', device='cpu') as handle:
                keys = list(handle.keys())
            row['tensor_count'] = len(keys)
            row['bad_lm_head_namespace_count'] = sum(
                1 for key in keys if key.startswith('base_model.model.backbone.lm_head')
            )
        except Exception as exc:
            row['reason'] = 'safetensors_open_failed:' + repr(exc)
            return row

    if not row['rank_ok']:
        row['reason'] = 'rank_missing_or_gt32'
    elif row['bad_lm_head_namespace_count']:
        row['reason'] = 'bad_lm_head_namespace'
    else:
        row['ready_for_eval'] = True
        row['reason'] = 'ready'
    return row

audit_rows = []
for item in CANDIDATES:
    print('auditing', item['label'], item['path'])
    audit_rows.append(audit_adapter(item['label'], item['path']))

audit_df = pd.DataFrame(audit_rows)
audit_csv = OUT_ROOT / 'v207b_adapter_structure_audit.csv'
audit_json = OUT_ROOT / 'v207b_adapter_structure_audit.json'
audit_df.to_csv(audit_csv, index=False)
audit_json.write_text(json.dumps(audit_rows, indent=2, sort_keys=True), encoding='utf-8')

print('audit_rows =', len(audit_df))
if len(audit_df):
    print(audit_df[['label', 'ready_for_eval', 'r', 'tensor_count', 'model_bytes', 'reason', 'path']].to_string(index=False))
else:
    print('No candidates discovered. Add adapter paths to MANUAL_CANDIDATES or mount the Drive folder.')
print('audit_csv =', audit_csv)
print('audit_json =', audit_json)
READY_CANDIDATES = [row for row in audit_rows if row.get('ready_for_eval')]
print('ready_candidate_count =', len(READY_CANDIDATES))
print('=== V207B STRUCTURE AUDIT END ===')
"""
        ),
        code(
            """# CELL: weak-family screen ready candidates.
print('=== V207B WEAK FAMILY SCREEN START ===')
results = []

for row in READY_CANDIDATES:
    label = safe_label(row['label'] + '_weak')
    adapter = Path(row['path'])
    out = OUT_ROOT / f'{label}_eval'
    log = REPORT_DIR / f'{label}_eval.log'
    report_json = out / f'{label}_eval_report.json'
    per_task_csv = out / f'{label}_per_task.csv'

    print('\\n' + '=' * 80)
    print('label =', label)
    print('adapter =', adapter, 'exists=', adapter.exists())
    print('out =', out)
    print('report_json =', report_json, 'exists=', report_json.exists())

    if FORCE_REEVAL and out.exists():
        print('FORCE_REEVAL enabled; removing previous output:', out)
        shutil.rmtree(out)

    if not report_json.exists():
        cmd = [
            sys.executable,
            ROOT / 'scripts' / 'evaluate_lora_adapter.py',
            '--solution-csv', VAL_WEAK_CSV,
            '--questions-csv', VAL_WEAK_CSV,
            '--adapter', adapter,
            '--base-model-path', MODEL_NAME,
            '--label', label,
            '--seed', '42',
            '--limit', '0',
            '--output-dir', out,
        ]
        rc = run_cmd(cmd, cwd=ROOT, log_path=log, check=False)
        if rc != 0:
            results.append({
                'label': label,
                'path': str(adapter),
                'status': f'eval_failed_{rc}',
                'weak_correct': 0,
                'weak_total': BASE_WEAK_TOTAL,
                'weak_delta': -BASE_WEAK_CORRECT,
                'accuracy': 0.0,
                'truncated': None,
                'truncation_rate': None,
                'promote_to_full': False,
                'report_json': str(report_json),
            })
            continue
    else:
        print('existing report found; skipping eval')

    report = json.loads(report_json.read_text(encoding='utf-8'))
    per = pd.read_csv(per_task_csv)
    weak_per = per[per['task_type'].isin(WEAK_FAMILIES)]
    weak_correct = int(weak_per['correct'].sum())
    weak_total = int(weak_per['total'].sum())
    weak_truncated = int(weak_per['truncated'].sum()) if 'truncated' in weak_per.columns else int(report.get('truncated', 0))
    weak_delta = weak_correct - BASE_WEAK_CORRECT
    trunc_rate = weak_truncated / weak_total if weak_total else 0.0
    promote = weak_delta > 0 and weak_total == BASE_WEAK_TOTAL

    print(per.to_string(index=False))
    print('weak_correct =', weak_correct, '/', weak_total)
    print('weak_correct_delta_vs_v194 =', weak_delta)
    print('weak_truncated =', weak_truncated, '/', weak_total)
    print('promote_to_full =', promote)

    results.append({
        'label': label,
        'path': str(adapter),
        'status': 'done',
        'weak_correct': weak_correct,
        'weak_total': weak_total,
        'weak_delta': weak_delta,
        'accuracy': float(report['accuracy']),
        'truncated': weak_truncated,
        'truncation_rate': trunc_rate,
        'promote_to_full': bool(promote),
        'report_json': str(report_json),
        'predictions_csv': str(out / f'{label}_predictions.csv'),
        'per_task_csv': str(per_task_csv),
    })

weak_results = pd.DataFrame(results)
weak_csv = OUT_ROOT / 'v207b_weak_screen_results.csv'
weak_json = OUT_ROOT / 'v207b_weak_screen_results.json'
weak_results.to_csv(weak_csv, index=False)
weak_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding='utf-8')

print('\\nWEAK SCREEN SUMMARY')
if len(weak_results):
    print(weak_results[['label', 'status', 'weak_correct', 'weak_total', 'weak_delta', 'accuracy', 'truncated', 'promote_to_full']].to_string(index=False))
else:
    print('No ready candidates to evaluate.')
print('weak_csv =', weak_csv)
print('weak_json =', weak_json)
FULL_CANDIDATES = [item for item in results if item.get('promote_to_full')]
print('full_candidate_count =', len(FULL_CANDIDATES))
print('=== V207B WEAK FAMILY SCREEN END ===')
"""
        ),
        code(
            """# CELL: full 947-row gate only for weak-positive candidates.
print('=== V207B FULL GATE START ===')
full_results = []

if not RUN_FULL_FOR_POSITIVE:
    print('RUN_FULL_FOR_POSITIVE=False; skipping full gate by configuration.')
elif not FULL_CANDIDATES:
    print('No weak-positive candidates. Full 947-row evaluation skipped.')
else:
    for item in FULL_CANDIDATES:
        base_label = safe_label(item['label'].replace('_weak', ''))
        adapter = Path(item['path'])
        eval_label = safe_label(base_label + '_full')
        eval_out = OUT_ROOT / f'{eval_label}_eval'
        eval_log = REPORT_DIR / f'{eval_label}_eval.log'
        report_json = eval_out / f'{eval_label}_eval_report.json'
        candidate_predictions = eval_out / f'{eval_label}_predictions.csv'

        print('\\n' + '=' * 80)
        print('full_label =', eval_label)
        print('adapter =', adapter)
        print('eval_out =', eval_out)

        if FORCE_REEVAL and eval_out.exists():
            print('FORCE_REEVAL enabled; removing previous full output:', eval_out)
            shutil.rmtree(eval_out)

        if not report_json.exists():
            cmd = [
                sys.executable,
                ROOT / 'scripts' / 'evaluate_lora_adapter.py',
                '--solution-csv', VAL_CSV,
                '--questions-csv', VAL_CSV,
                '--adapter', adapter,
                '--base-model-path', MODEL_NAME,
                '--label', eval_label,
                '--seed', '42',
                '--limit', '0',
                '--output-dir', eval_out,
            ]
            rc = run_cmd(cmd, cwd=ROOT, log_path=eval_log, check=False)
            if rc != 0:
                full_results.append({'label': eval_label, 'status': f'eval_failed_{rc}', 'path': str(adapter)})
                continue
        else:
            print('existing full report found; skipping eval')

        gate_out = OUT_ROOT / f'{eval_label}_gate'
        gate_log = REPORT_DIR / f'{eval_label}_gate.log'
        cmd = [
            sys.executable,
            ROOT / 'scripts' / 'solve_rate_gate.py',
            '--solution-csv', VAL_CSV,
            '--baseline-predictions', BASELINE_PREDICTIONS,
            '--candidate-predictions', candidate_predictions,
            '--family-regression-tolerance', '0.0',
            '--min-net-gain', '0.0',
            '--min-boxed-rate', '0.98',
            '--output-dir', gate_out,
        ]
        rc = run_cmd(cmd, cwd=ROOT, log_path=gate_log, check=False)
        gate_report = gate_out / 'solve_rate_gate_report.json'
        gate_payload = json.loads(gate_report.read_text(encoding='utf-8')) if gate_report.exists() else {}
        full_results.append({
            'label': eval_label,
            'status': 'gate_approve' if rc == 0 else f'gate_reject_{rc}',
            'path': str(adapter),
            'gate_report': str(gate_report),
            'approved': bool(gate_payload.get('approved', False)),
            'net_gain': gate_payload.get('comparison', {}).get('net_gain'),
            'reasons': gate_payload.get('reasons', []),
        })

full_df = pd.DataFrame(full_results)
full_csv = OUT_ROOT / 'v207b_full_gate_results.csv'
full_json = OUT_ROOT / 'v207b_full_gate_results.json'
full_df.to_csv(full_csv, index=False)
full_json.write_text(json.dumps(full_results, indent=2, sort_keys=True), encoding='utf-8')

print('\\nFULL GATE SUMMARY')
if len(full_df):
    print(full_df.to_string(index=False))
else:
    print('No full-gate candidates were evaluated.')
print('full_csv =', full_csv)
print('full_json =', full_json)
print('=== V207B FULL GATE END ===')
"""
        ),
        code(
            """# CELL: final V207B summary. This cell does not submit.
print('=== V207B FINAL SUMMARY START ===')
summary = {
    'version': VERSION,
    'status': 'v207b_external_adapter_triage_completed',
    'submit_disabled': True,
    'v207a_root': str(V207A_ROOT),
    'out_root': str(OUT_ROOT),
    'candidate_manifest': str(OUT_ROOT / 'v207b_discovered_candidates.json'),
    'structure_audit_csv': str(OUT_ROOT / 'v207b_adapter_structure_audit.csv'),
    'weak_screen_csv': str(OUT_ROOT / 'v207b_weak_screen_results.csv'),
    'full_gate_csv': str(OUT_ROOT / 'v207b_full_gate_results.csv'),
    'full_candidate_count': len(FULL_CANDIDATES) if 'FULL_CANDIDATES' in globals() else 0,
    'next_human_action': (
        'Review any approved full gate candidate and explicitly approve Kaggle submission.'
        if 'full_results' in globals() and any(x.get('approved') for x in full_results)
        else 'No submission candidate approved. Add more external adapters or stop.'
    ),
}
summary_path = OUT_ROOT / 'V207B_FINAL_RUN_SUMMARY.json'
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(summary, indent=2, sort_keys=True))
print('summary_path =', summary_path)
print('=== V207B FINAL SUMMARY END ===')
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    required_markers = [
        "KG1 V207B External Adapter Triage Colab",
        "V207B DRIVE MOUNT START",
        "V207B PUBLIC KAGGLE ADAPTER DOWNLOAD START",
        "V207B CANDIDATE DISCOVERY START",
        "V207B WEAK FAMILY SCREEN START",
        "V207B FULL GATE START",
        "ALLOW_KAGGLE_SUBMIT = False",
        "submit_disabled",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(f"Generated notebook missing required markers: {missing}")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Cells: {len(notebook['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

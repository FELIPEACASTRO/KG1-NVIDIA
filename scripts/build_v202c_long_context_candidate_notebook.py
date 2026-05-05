#!/usr/bin/env python3
"""Build the V202C long-context candidate sweep Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_v202b_long_context_pretokenized_notebook import (
    BRANCH_SCRIPT_BASE,
    MODEL_NAME,
    MODEL_REVISION,
    TONG_CORPUS_JSONL_SHA256,
    TONG_DATASET_SLUG,
    TONG_DATASET_ZIP_NAME,
    TONG_DATASET_ZIP_SHA256,
    TRAIN_SCRIPT_URL,
    V194_RANK19_ADAPTER_CONFIG_SHA256,
    V194_RANK19_ADAPTER_MODEL_SHA256,
    V194_RANK19_PUBLIC_SCORE,
    V194_RANK19_RANK,
    V194_RANK19_ZIP_SHA256,
)


NOTEBOOK_PATH = Path("notebooks/KG1_V202C_H100_A100_LONG_CONTEXT_CANDIDATE_SWEEP_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v202_long_context_20260504/V202C_LONG_CONTEXT_CANDIDATE_SWEEP_NEXT_ACTIONS.md")

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v202c-{prefix}-{_CELL_COUNTER:02d}"


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


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source") or []) for cell in notebook["cells"])


def build_notebook() -> dict:
    cells = [
        md(
            """# KG1 V202C H100/A100 Long-Context Candidate Sweep

This notebook is the next step after V202B smoke passed:

`baseline_eval_loss=0.1668307316`, `final_eval_loss=0.1666177167`, `delta=-0.000213`, `passed_no_regression_gate=true`.

It runs three independent 8192-token candidates from the exact V194 rank-19 adapter. It does not submit to Kaggle and it does not promote any candidate unless the local no-regression gate passes.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            f"""import datetime, hashlib, json, os, pathlib, re, shutil, subprocess, sys, urllib.request, zipfile

VERSION = 'V202C_LONG_CONTEXT_CANDIDATE_SWEEP_20260505'
print('NOTEBOOK_VERSION =', VERSION)

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202C')
V202B_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B')
WORK_ROOT = pathlib.Path('/content/kg1_v202c')
DATA_DIR = WORK_ROOT / 'data'
SCRIPT_DIR = WORK_ROOT / 'scripts'
REPORT_DIR = DRIVE_ROOT / 'reports'
DRIVE_DATA_DIR = DRIVE_ROOT / 'data'
BASELINE_DIR = DRIVE_ROOT / 'baseline_v194_rank19'
RANK19_BUILD = DRIVE_ROOT / 'init_adapter_v194_rank19_build'
INIT_ADAPTER = RANK19_BUILD / 'adapter'
OUT_ROOT = DRIVE_ROOT / 'output_v202c_longctx_candidate_sweep'

for path in [DRIVE_ROOT, WORK_ROOT, DATA_DIR, SCRIPT_DIR, REPORT_DIR, DRIVE_DATA_DIR, BASELINE_DIR, OUT_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

RUN_CANDIDATE_SWEEP = True
ALLOW_KAGGLE_SUBMIT = False
REQUIRE_V202B_SMOKE_PASS = True

MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'
TRAIN_SCRIPT_URL = '{TRAIN_SCRIPT_URL}'

V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'
V194_RANK19_ADAPTER_MODEL_SHA256 = '{V194_RANK19_ADAPTER_MODEL_SHA256}'
V194_RANK19_ADAPTER_CONFIG_SHA256 = '{V194_RANK19_ADAPTER_CONFIG_SHA256}'
V194_RANK19_PUBLIC_SCORE = '{V194_RANK19_PUBLIC_SCORE}'
V194_RANK19_RANK = '{V194_RANK19_RANK}'

TONG_DATASET_SLUG = '{TONG_DATASET_SLUG}'
TONG_DATASET_ZIP_NAME = '{TONG_DATASET_ZIP_NAME}'
TONG_DATASET_ZIP_SHA256 = '{TONG_DATASET_ZIP_SHA256}'
TONG_CORPUS_JSONL_SHA256 = '{TONG_CORPUS_JSONL_SHA256}'

TRAINING_CONTRACT = {{
    'max_length': 8192,
    'max_model_len': 8192,
    'max_tokens': 7680,
    'max_lora_rank': 32,
    'temperature': 0.0,
    'top_p': 1.0,
    'base_model': MODEL_NAME,
    'model_revision': MODEL_REVISION,
    'no_submit_without_vllm_gate': True,
    'require_net_gain_vs_v194': True,
    'require_zero_anchor_regression': True,
}}

print('DRIVE_ROOT =', DRIVE_ROOT)
print('WORK_ROOT =', WORK_ROOT)
print('OUT_ROOT =', OUT_ROOT)
print('RUN_CANDIDATE_SWEEP =', RUN_CANDIDATE_SWEEP)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook is intentionally submit-disabled.')
"""
        ),
        code(
            """def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def assert_sha256(path: pathlib.Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    print(f'{label} sha256:', observed)
    if observed != expected:
        raise RuntimeError(f'{label} SHA mismatch for {path}: {observed} != {expected}')
    return observed

def stream_process(cmd, cwd=None, env=None, log_path=None):
    print('+', ' '.join(str(x) for x in cmd))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open('w', encoding='utf-8')
    else:
        log_file = None
    proc = subprocess.Popen(
        [str(x) for x in cmd],
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
        if log_file:
            log_file.write(line)
    rc = proc.wait()
    if log_file:
        log_file.close()
    print('returncode =', rc)
    return rc
"""
        ),
        code(
            """gpu_csv = subprocess.check_output(
    'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits',
    shell=True,
).decode().strip()
print('GPU:', gpu_csv)
parts = [part.strip() for part in gpu_csv.split(',')]
assert len(parts) >= 3, f'Unexpected nvidia-smi output: {gpu_csv}'
gpu_name = parts[0]
gpu_mem_mib = int(parts[1])
driver_version = parts[2]
assert ('H100' in gpu_name) or ('A100' in gpu_name and gpu_mem_mib >= 75000), (
    f'Use H100 or A100 80GB High-RAM for V202C; found {gpu_name} with {gpu_mem_mib} MiB.'
)
meminfo = pathlib.Path('/proc/meminfo').read_text(encoding='utf-8')
host_mem_kib = int(re.search(r'MemTotal:\\s+(\\d+)', meminfo).group(1))
host_mem_gib = host_mem_kib / 1024 / 1024
disk_free_gib = shutil.disk_usage('/content').free / 1024**3
print(f'Host RAM: {host_mem_gib:.1f} GiB')
print(f'/content free: {disk_free_gib:.1f} GiB')
print('Driver:', driver_version)
assert host_mem_gib >= 50, f'High-RAM runtime expected; host RAM is {host_mem_gib:.1f} GiB'
assert disk_free_gib >= 90, f'Need at least 90 GiB free on /content; found {disk_free_gib:.1f} GiB'
"""
        ),
        code(
            """import importlib.metadata as md
import importlib.util

os.environ.setdefault('MAX_JOBS', '4')
os.environ.setdefault('PIP_ROOT_USER_ACTION', 'ignore')

def pip_install(args):
    print('+ pip install', ' '.join(args))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

def pip_uninstall(package_name):
    print('+ pip uninstall -y', package_name)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name], check=False)

def install_exact(dist_name, expected_version, module_name, args):
    try:
        observed = md.version(dist_name)
    except md.PackageNotFoundError:
        observed = None
    if observed != expected_version:
        print(f'{dist_name} version {observed!r}; installing {expected_version}')
        pip_install(args)
    else:
        print(f'{dist_name} already at {expected_version}')
    assert importlib.util.find_spec(module_name) is not None, f'{module_name} import spec missing after install'
    assert md.version(dist_name) == expected_version, f'{dist_name} version mismatch after install: {md.version(dist_name)}'

pip_uninstall('torchao')
pip_install(['--upgrade', 'pip', 'setuptools', 'wheel', 'packaging', 'ninja==1.13.0'])
pip_install([
    'kaggle==2.0.2',
    'transformers==5.7.0',
    'accelerate==1.13.0',
    'peft==0.19.1',
    'datasets==4.8.5',
    'safetensors==0.7.0',
    'huggingface_hub==1.13.0',
    'sentencepiece==0.2.1',
    'protobuf==7.34.1',
])
install_exact('causal-conv1d', '1.6.1', 'causal_conv1d', ['causal-conv1d==1.6.1', '--no-build-isolation'])
install_exact('mamba-ssm', '2.3.1', 'mamba_ssm', ['mamba-ssm==2.3.1', '--no-build-isolation'])
assert importlib.util.find_spec('torchao') is None, 'torchao still installed; restart runtime and rerun cells from top'
import causal_conv1d, mamba_ssm
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
print('mamba_ssm OK:', getattr(mamba_ssm, '__version__', 'unknown'))
"""
        ),
        code(
            """def read_colab_secret(name):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value or ''
    except Exception:
        return ''

kaggle_dir = pathlib.Path('/root/.kaggle')
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_json = kaggle_dir / 'kaggle.json'

kaggle_username = os.environ.get('KAGGLE_USERNAME') or read_colab_secret('KAGGLE_USERNAME')
kaggle_key = os.environ.get('KAGGLE_KEY') or read_colab_secret('KAGGLE_KEY')
drive_kaggle_candidates = [
    pathlib.Path('/content/drive/MyDrive/.kaggle/kaggle.json'),
    pathlib.Path('/content/drive/MyDrive/kaggle.json'),
]
if kaggle_username and kaggle_key:
    kaggle_json.write_text(json.dumps({'username': kaggle_username, 'key': kaggle_key}), encoding='utf-8')
else:
    source = next((path for path in drive_kaggle_candidates if path.exists()), None)
    if source is None:
        raise RuntimeError('Missing Kaggle credentials. Add Colab secrets or Drive kaggle.json.')
    shutil.copy2(source, kaggle_json)
kaggle_json.chmod(0o600)
os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_dir)

hf_token = os.environ.get('HF_TOKEN') or read_colab_secret('HF_TOKEN') or read_colab_secret('HUGGINGFACE_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('HF_TOKEN staged from secrets/env.')
else:
    print('HF_TOKEN not found; model downloads will be unauthenticated.')
print('Kaggle credentials staged.')
"""
        ),
        code(
            """if REQUIRE_V202B_SMOKE_PASS:
    smoke_summaries = sorted(V202B_ROOT.glob('output_v202b_longctx_smoke_1s_*/v202b_smoke_summary.json'))
    passed = []
    for path in smoke_summaries:
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if item.get('passed_no_regression_gate') is True and item.get('returncode') == 0:
            passed.append((path, item))
    if not passed:
        raise RuntimeError('V202C requires a passed V202B smoke summary in Drive before candidate sweep.')
    smoke_path, smoke_item = passed[-1]
    print('Confirmed V202B smoke pass:', smoke_path)
    print(json.dumps({
        'baseline_eval_loss': smoke_item.get('baseline_eval_loss'),
        'final_eval_loss': smoke_item.get('final_eval_loss'),
        'delta_vs_baseline': smoke_item.get('delta_vs_baseline'),
    }, indent=2))
"""
        ),
        code(
            """def adapter_ready(path: pathlib.Path, min_model_bytes=4_000_000_000) -> bool:
    cfg = path / 'adapter_config.json'
    model = path / 'adapter_model.safetensors'
    if not cfg.exists() or not model.exists():
        return False
    if cfg.stat().st_size < 100 or model.stat().st_size < min_model_bytes:
        return False
    json.loads(cfg.read_text(encoding='utf-8'))
    return True

V194_ZIP_CANDIDATES = [
    pathlib.Path(os.environ['V194_RANK19_ZIP']) if os.environ.get('V194_RANK19_ZIP') else None,
    BASELINE_DIR / 'submission.zip',
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B/baseline_v194_rank19/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202/baseline_v194_rank19/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V201/baseline_v194_rank19/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/baseline_v194_rank19/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/Submit/submission.zip'),
    pathlib.Path('/content/drive/MyDrive/submission.zip'),
    pathlib.Path('/content/submission.zip'),
]
V194_ZIP_CANDIDATES = [path for path in V194_ZIP_CANDIDATES if path is not None]

def extract_v194_submission_zip(candidate: pathlib.Path) -> pathlib.Path:
    assert_sha256(candidate, V194_RANK19_ZIP_SHA256, 'V194 rank-19 submission.zip')
    shutil.rmtree(RANK19_BUILD, ignore_errors=True)
    INIT_ADAPTER.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidate) as zf:
        members = {name for name in zf.namelist() if not name.endswith('/')}
        expected = {'adapter_model.safetensors', 'adapter_config.json'}
        if members != expected:
            raise RuntimeError(f'Unexpected V194 zip members: {sorted(members)}')
        zf.extract('adapter_model.safetensors', INIT_ADAPTER)
        zf.extract('adapter_config.json', INIT_ADAPTER)
    cached_zip = RANK19_BUILD / 'submission.zip'
    cached_zip.parent.mkdir(parents=True, exist_ok=True)
    if candidate.resolve() != cached_zip.resolve():
        shutil.copy2(candidate, cached_zip)
    assert_sha256(INIT_ADAPTER / 'adapter_config.json', V194_RANK19_ADAPTER_CONFIG_SHA256, 'V194 adapter_config')
    assert_sha256(INIT_ADAPTER / 'adapter_model.safetensors', V194_RANK19_ADAPTER_MODEL_SHA256, 'V194 adapter_model')
    assert_sha256(cached_zip, V194_RANK19_ZIP_SHA256, 'cached V194 submission.zip')
    assert adapter_ready(INIT_ADAPTER)
    return INIT_ADAPTER

def ensure_v194_adapter() -> pathlib.Path:
    cached_zip = RANK19_BUILD / 'submission.zip'
    if adapter_ready(INIT_ADAPTER) and cached_zip.exists():
        try:
            assert_sha256(INIT_ADAPTER / 'adapter_config.json', V194_RANK19_ADAPTER_CONFIG_SHA256, 'cached V194 adapter_config')
            assert_sha256(INIT_ADAPTER / 'adapter_model.safetensors', V194_RANK19_ADAPTER_MODEL_SHA256, 'cached V194 adapter_model')
            assert_sha256(cached_zip, V194_RANK19_ZIP_SHA256, 'cached V194 submission.zip')
            return INIT_ADAPTER
        except RuntimeError:
            print('Cached V194 adapter mismatch; searching for exact zip.')
            shutil.rmtree(RANK19_BUILD, ignore_errors=True)
    print('Searching exact V194 rank-19 submission.zip candidates...')
    for candidate in V194_ZIP_CANDIDATES:
        print('  candidate:', candidate)
        if candidate.exists():
            return extract_v194_submission_zip(candidate)
    paths = '\\n'.join(f'  - {path}' for path in V194_ZIP_CANDIDATES)
    raise RuntimeError(
        'Exact V194 rank-19 submission.zip is required before V202C.\\n'
        f'Expected zip SHA256: {V194_RANK19_ZIP_SHA256}\\n'
        f'{paths}'
    )

INIT_ADAPTER = ensure_v194_adapter()
print('Confirmed immutable baseline:', {
    'label': 'V194 rank-19',
    'public_score': V194_RANK19_PUBLIC_SCORE,
    'rank': V194_RANK19_RANK,
    'adapter_model_sha256': sha256_path(INIT_ADAPTER / 'adapter_model.safetensors'),
})
"""
        ),
        code(
            """DRIVE_DATASET_ZIP = DRIVE_DATA_DIR / TONG_DATASET_ZIP_NAME
V202B_DATASET_ZIP = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B/data') / TONG_DATASET_ZIP_NAME
LOCAL_DATASET_ZIP = DATA_DIR / TONG_DATASET_ZIP_NAME

def download_tong_dataset_zip() -> pathlib.Path:
    for cached in [DRIVE_DATASET_ZIP, V202B_DATASET_ZIP]:
        if cached.exists():
            try:
                assert_sha256(cached, TONG_DATASET_ZIP_SHA256, f'cached Tong dataset zip {cached}')
                if cached != DRIVE_DATASET_ZIP:
                    shutil.copy2(cached, DRIVE_DATASET_ZIP)
                return DRIVE_DATASET_ZIP
            except RuntimeError:
                if cached == DRIVE_DATASET_ZIP:
                    cached.unlink()
                else:
                    print('Ignoring mismatched V202B dataset cache:', cached)
    tmp_dir = WORK_ROOT / 'tmp_kaggle_download'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'datasets', 'download', '-d', TONG_DATASET_SLUG, '-p', str(tmp_dir)], check=True)
    candidates = sorted(tmp_dir.glob('*.zip'))
    if not candidates:
        raise RuntimeError(f'Kaggle dataset download produced no zip in {tmp_dir}')
    downloaded = candidates[0]
    assert_sha256(downloaded, TONG_DATASET_ZIP_SHA256, 'downloaded Tong dataset zip')
    shutil.copy2(downloaded, DRIVE_DATASET_ZIP)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return DRIVE_DATASET_ZIP

source_zip = download_tong_dataset_zip()
if not LOCAL_DATASET_ZIP.exists() or sha256_path(LOCAL_DATASET_ZIP) != TONG_DATASET_ZIP_SHA256:
    shutil.copy2(source_zip, LOCAL_DATASET_ZIP)
assert_sha256(LOCAL_DATASET_ZIP, TONG_DATASET_ZIP_SHA256, 'local Tong dataset zip')

with zipfile.ZipFile(LOCAL_DATASET_ZIP) as zf:
    names = set(zf.namelist())
    required_members = {'corpus.jsonl', 'problems.jsonl', 'generation.jsonl'}
    missing = sorted(required_members - names)
    if missing:
        raise RuntimeError(f'Tong dataset zip missing required members: {missing}')
    corpus_sha = hashlib.sha256(zf.read('corpus.jsonl')).hexdigest()
    print('corpus.jsonl sha256:', corpus_sha)
    assert corpus_sha == TONG_CORPUS_JSONL_SHA256
    corpus_rows = [json.loads(line) for line in zf.read('corpus.jsonl').decode('utf-8').splitlines() if line.strip()]
    assert len(corpus_rows) == 15979, len(corpus_rows)
    max_tokens = max(int(row['token_count']) for row in corpus_rows)
    assert max_tokens <= 8192, max_tokens
print('Tong dataset ready:', LOCAL_DATASET_ZIP)
"""
        ),
        code(
            """TRAIN_SCRIPT = SCRIPT_DIR / 'hf_job_train_v90.py'
print('downloading training script:', TRAIN_SCRIPT_URL)
urllib.request.urlretrieve(TRAIN_SCRIPT_URL, TRAIN_SCRIPT)
script_text = TRAIN_SCRIPT.read_text(encoding='utf-8')
required_features = [
    'PRETOKENIZED_ARCHIVE_ZIP',
    'load_tong_pretokenized_archive',
    'DRY_RUN_VALIDATE_ONLY',
    'REQUIRE_FINAL_EVAL_LTE_BASELINE',
    'ABORT_MAX_RESERVED_GIB',
    'PEFT_MANUAL_LOAD_METHOD',
]
for feature in required_features:
    if feature not in script_text:
        raise RuntimeError(f'Training script missing required feature: {feature}')
print('Training script ready:', TRAIN_SCRIPT)
"""
        ),
        code(
            """COMMON_ENV = {
    'MODEL_NAME': MODEL_NAME,
    'MODEL_REVISION': MODEL_REVISION,
    'MODEL_DEVICE_MAP': 'auto',
    'PRETOKENIZED_ARCHIVE_ZIP': str(LOCAL_DATASET_ZIP),
    'EXPECTED_ARCHIVE_SHA256': TONG_DATASET_ZIP_SHA256,
    'PRETOKENIZED_VAL_COPY_ONLY': '0',
    'INIT_ADAPTER_DIR': str(INIT_ADAPTER),
    'INIT_ADAPTER_LOAD_MODE': 'manual',
    'PEFT_MANUAL_LOAD_METHOD': 'direct',
    'ADAPTER_LOAD_LOW_CPU_MEM_USAGE': '0',
    'UPLOAD_TO_HF': '0',
    'UPLOAD_CHECKPOINTS_DURING_TRAINING': '0',
    'FAIL_ON_MISSING_ADAPTER_KEYS': '1',
    'REQUIRE_OFFSET_MASK': '1',
    'LORA_R': '32',
    'LORA_ALPHA': '32',
    'LORA_DROPOUT': '0.0',
    'LORA_TARGET_MODULES': 'down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj',
    'TRAINABLE_LORA_MODULES': 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj',
    'MAX_TRAINABLE_PARAM_RATIO': '0.035',
    'MAX_LENGTH': '8192',
    'BATCH_SIZE': '1',
    'MICRO_BATCH_SIZE': '1',
    'NUM_EPOCHS': '1',
    'SAVE_EVERY_STEPS': '1',
    'EVAL_EVERY_STEPS': '999999',
    'LOG_EVERY_STEPS': '1',
    'MICRO_LOG_EVERY': '1',
    'SEED': '202',
    'SAMPLING_MODE': 'shuffle',
    'GRAD_CLIP_NORM': '1.0',
    'BASELINE_EVAL_BEFORE_TRAIN': '1',
    'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1',
    'MAX_FINAL_EVAL_REGRESSION': '0.0',
    'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': '0.001',
    'ABORT_MAX_RESERVED_GIB': '79',
    'COMPUTE_PROVIDER': 'colab_v202c_long_context',
    'DRY_RUN_VALIDATE_ONLY': '0',
}
if os.environ.get('HF_TOKEN'):
    COMMON_ENV['HF_TOKEN'] = os.environ['HF_TOKEN']

CANDIDATES = [
    {
        'label': 'A_all_shuffle_3s_lr2e8',
        'run_id': 'v202c-A-all-shuffle-3s-lr2e8',
        'max_steps': '3',
        'learning_rate': '2e-8',
        'final_learning_rate': '2e-8',
        'eval_max_examples': '128',
        'pretokenized_val_examples': '720',
        'exclude_categories': '',
        'min_train_examples': '15000',
        'min_val_examples': '720',
        'min_tokenized_train_examples': '15000',
        'min_tokenized_val_examples': '720',
    },
    {
        'label': 'B_official_only_3s_lr2e8',
        'run_id': 'v202c-B-official-only-3s-lr2e8',
        'max_steps': '3',
        'learning_rate': '2e-8',
        'final_learning_rate': '2e-8',
        'eval_max_examples': '128',
        'pretokenized_val_examples': '360',
        'exclude_categories': 'matching,concatenation,splitting,spelling,lstrip',
        'min_train_examples': '7000',
        'min_val_examples': '360',
        'min_tokenized_train_examples': '7000',
        'min_tokenized_val_examples': '360',
    },
    {
        'label': 'C_all_shuffle_5s_lr1e8',
        'run_id': 'v202c-C-all-shuffle-5s-lr1e8',
        'max_steps': '5',
        'learning_rate': '1e-8',
        'final_learning_rate': '1e-8',
        'eval_max_examples': '128',
        'pretokenized_val_examples': '720',
        'exclude_categories': '',
        'min_train_examples': '15000',
        'min_val_examples': '720',
        'min_tokenized_train_examples': '15000',
        'min_tokenized_val_examples': '720',
    },
]

contract_report = {
    'version': VERSION,
    'immutable_baseline': {
        'label': 'V194 rank-19',
        'public_score': V194_RANK19_PUBLIC_SCORE,
        'rank': V194_RANK19_RANK,
        'zip_sha256': V194_RANK19_ZIP_SHA256,
        'adapter_model_sha256': V194_RANK19_ADAPTER_MODEL_SHA256,
    },
    'dataset': {
        'slug': TONG_DATASET_SLUG,
        'zip_sha256': TONG_DATASET_ZIP_SHA256,
        'corpus_jsonl_sha256': TONG_CORPUS_JSONL_SHA256,
    },
    'common_env': COMMON_ENV,
    'candidates': CANDIDATES,
    'training_contract': TRAINING_CONTRACT,
    'policy': {
        'run_candidate_sweep': RUN_CANDIDATE_SWEEP,
        'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
        'do_not_use_raw_generation_or_traj_as_sft': True,
        'no_submit_without_vllm_gate': True,
    },
}
(REPORT_DIR / 'v202c_candidate_contract.json').write_text(json.dumps(contract_report, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('Candidate contract:', REPORT_DIR / 'v202c_candidate_contract.json')
"""
        ),
        code(
            """def parse_metrics(log_text: str) -> dict:
    metrics = {
        'baseline_eval_loss': None,
        'final_eval_loss': None,
        'best_eval_loss': None,
        'delta_vs_baseline': None,
    }
    baseline_match = re.search(r'baseline_eval_loss=([0-9.]+)', log_text)
    if baseline_match:
        metrics['baseline_eval_loss'] = float(baseline_match.group(1))
    final_match = re.search(r'Final eval loss:\\s*([0-9.]+);\\s*best eval loss:\\s*([0-9.]+)', log_text)
    if final_match:
        metrics['final_eval_loss'] = float(final_match.group(1))
        metrics['best_eval_loss'] = float(final_match.group(2))
    if metrics['baseline_eval_loss'] is not None and metrics['final_eval_loss'] is not None:
        metrics['delta_vs_baseline'] = round(metrics['final_eval_loss'] - metrics['baseline_eval_loss'], 6)
    return metrics

results = []
if RUN_CANDIDATE_SWEEP:
    for candidate in CANDIDATES:
        label = candidate['label']
        candidate_out = OUT_ROOT / label
        if candidate_out.exists():
            suffix = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
            candidate_out = OUT_ROOT / f'{label}_{suffix}'
        candidate_out.mkdir(parents=True, exist_ok=True)
        log_path = candidate_out / 'v202c_candidate_train.log'
        print('\\n' + '=' * 80)
        print('Starting V202C candidate:', label)
        print('Output:', candidate_out)
        env = os.environ.copy()
        env.update(COMMON_ENV)
        env.update({
            'OUTPUT_DIR': str(candidate_out),
            'RUN_ID': candidate['run_id'],
            'MAX_STEPS': candidate['max_steps'],
            'SAVE_EVERY_STEPS': candidate['max_steps'],
            'LEARNING_RATE': candidate['learning_rate'],
            'FINAL_LEARNING_RATE': candidate['final_learning_rate'],
            'EVAL_MAX_EXAMPLES': candidate['eval_max_examples'],
            'PRETOKENIZED_VAL_EXAMPLES': candidate['pretokenized_val_examples'],
            'PRETOKENIZED_EXCLUDE_CATEGORIES': candidate['exclude_categories'],
            'MIN_TRAIN_EXAMPLES': candidate['min_train_examples'],
            'MIN_VAL_EXAMPLES': candidate['min_val_examples'],
            'MIN_TOKENIZED_TRAIN_EXAMPLES': candidate['min_tokenized_train_examples'],
            'MIN_TOKENIZED_VAL_EXAMPLES': candidate['min_tokenized_val_examples'],
        })
        rc = stream_process([sys.executable, str(TRAIN_SCRIPT)], cwd=WORK_ROOT, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding='utf-8', errors='replace')
        metrics = parse_metrics(log_text)
        manifest_path = candidate_out / 'final_adapter/v90_training_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            gate = manifest.get('training', {}).get('baseline_gate', {})
            metrics['baseline_eval_loss'] = gate.get('baseline_eval_loss', metrics['baseline_eval_loss'])
            metrics['final_eval_loss'] = gate.get('final_eval_loss', metrics['final_eval_loss'])
            if metrics['baseline_eval_loss'] is not None and metrics['final_eval_loss'] is not None:
                metrics['delta_vs_baseline'] = round(metrics['final_eval_loss'] - metrics['baseline_eval_loss'], 6)
        passed = (
            rc == 0
            and manifest_path.exists()
            and metrics.get('baseline_eval_loss') is not None
            and metrics.get('final_eval_loss') is not None
            and metrics['final_eval_loss'] <= metrics['baseline_eval_loss']
        )
        result = {
            'label': label,
            'output_dir': str(candidate_out),
            'returncode': rc,
            'passed_no_regression_gate': bool(passed),
            'metrics': metrics,
            'manifest_path': str(manifest_path) if manifest_path.exists() else None,
            'log_path': str(log_path),
            'config': candidate,
        }
        (candidate_out / 'candidate_summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
        results.append(result)
        print('Candidate result:', json.dumps({
            'label': label,
            'returncode': rc,
            'passed_no_regression_gate': passed,
            'baseline_eval_loss': metrics.get('baseline_eval_loss'),
            'final_eval_loss': metrics.get('final_eval_loss'),
            'delta_vs_baseline': metrics.get('delta_vs_baseline'),
        }, indent=2))
else:
    print('RUN_CANDIDATE_SWEEP=False; no training was run.')

summary = {
    'version': VERSION,
    'root': str(OUT_ROOT),
    'baseline': {
        'label': 'V194 rank-19',
        'rank': V194_RANK19_RANK,
        'public_score': V194_RANK19_PUBLIC_SCORE,
        'zip_sha256': V194_RANK19_ZIP_SHA256,
        'adapter_model_sha256': V194_RANK19_ADAPTER_MODEL_SHA256,
    },
    'candidates': results,
    'passed_candidates': [item for item in results if item['passed_no_regression_gate']],
    'policy': 'No Kaggle submit was performed. Passed candidates still require vLLM/Kaggle-layout gates before submit.',
}
summary_path = OUT_ROOT / 'v202c_candidates_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('\\nV202C candidate summary:', summary_path)
print(json.dumps([
    {
        'label': item['label'],
        'passed': item['passed_no_regression_gate'],
        'baseline': item['metrics'].get('baseline_eval_loss'),
        'final': item['metrics'].get('final_eval_loss'),
        'delta': item['metrics'].get('delta_vs_baseline'),
    }
    for item in results
], indent=2))
print('No Kaggle submit was performed.')
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "colab": {"provenance": [], "name": NOTEBOOK_PATH.name},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_notebook(notebook: dict) -> None:
    source = all_source(notebook)
    required = [
        "KG1 V202C H100/A100 Long-Context Candidate Sweep",
        "REQUIRE_V202B_SMOKE_PASS = True",
        "RUN_CANDIDATE_SWEEP = True",
        "ALLOW_KAGGLE_SUBMIT = False",
        f"V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'",
        f"TONG_DATASET_ZIP_SHA256 = '{TONG_DATASET_ZIP_SHA256}'",
        "'MAX_LENGTH': '8192'",
        "'PRETOKENIZED_ARCHIVE_ZIP': str(LOCAL_DATASET_ZIP)",
        "A_all_shuffle_3s_lr2e8",
        "B_official_only_3s_lr2e8",
        "C_all_shuffle_5s_lr1e8",
        "'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1'",
        "'MAX_FINAL_EVAL_REGRESSION': '0.0'",
        "No Kaggle submit was performed.",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V202C notebook is missing {fragment!r}")
    forbidden = [
        "kaggle competitions submit",
        "kg1_v199b_safe_kaggle_submit.py",
        "MAX_LENGTH': '2048'",
        "MAX_LENGTH'] = '2048'",
        "data/v198",
        "ALLOW_KAGGLE_SUBMIT = True",
        "RUN_CANDIDATE_SWEEP = False",
        "bit_manipulation=2.5",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V202C notebook contains forbidden fragment {fragment!r}")


def main() -> int:
    notebook = build_notebook()
    validate_notebook(notebook)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V202C long-context candidate sweep\n\n"
        "- Notebook: `notebooks/KG1_V202C_H100_A100_LONG_CONTEXT_CANDIDATE_SWEEP_COLAB_PRO.ipynb`.\n"
        "- Requires a passed V202B smoke summary in Drive before running.\n"
        "- Runs three independent candidates from exact V194 rank-19, not chained from V202B smoke.\n"
        "- Candidate A: all Tong categories, 3 steps, LR `2e-8`.\n"
        "- Candidate B: official-category-only subset, 3 steps, LR `2e-8`.\n"
        "- Candidate C: all Tong categories, 5 steps, LR `1e-8`.\n"
        "- All candidates enforce `MAX_LENGTH=8192`, `BATCH_SIZE=1`, attention-only trainable filter, final eval <= baseline.\n"
        "- No Kaggle submit cell exists; passed candidates still require vLLM/Kaggle-layout gates.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

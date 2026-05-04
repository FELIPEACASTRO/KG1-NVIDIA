#!/usr/bin/env python3
"""Build the V202B long-context pretokenized dry-run/smoke Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V202B_H100_A100_LONG_CONTEXT_PRETOKENIZED_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v202_long_context_20260504/V202B_LONG_CONTEXT_PRETOKENIZED_NEXT_ACTIONS.md")

BRANCH_SCRIPT_BASE = "https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/claude/competent-shamir/scripts"
TRAIN_SCRIPT_URL = f"{BRANCH_SCRIPT_BASE}/hf_job_train_v90.py"

V194_RANK19_ZIP_SHA256 = "49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8"
V194_RANK19_ADAPTER_MODEL_SHA256 = "01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f"
V194_RANK19_ADAPTER_CONFIG_SHA256 = "e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743"
V194_RANK19_PUBLIC_SCORE = "0.86"
V194_RANK19_RANK = "19/2613"

TONG_DATASET_SLUG = "atahalam/tonghuikang-0-87-nemotron-dataset"
TONG_DATASET_ZIP_NAME = "tonghuikang-0-87-nemotron-dataset.zip"
TONG_DATASET_ZIP_SHA256 = "461776d6bc44d482988d23c4e584128b66a93d2500fe7c428f4e895ab42e9eb8"
TONG_CORPUS_JSONL_SHA256 = "309264659ba3f668b6b548ca16686d773868cd5bc63349a6721484308341e5c6"

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"


_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v202b-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V202B H100/A100 Long-Context Pretokenized Gate

This notebook is the first executable step after the V202 audit.

Policy:
- Start only from the exact V194 rank-19 adapter.
- Use the full Tong/Huikang pretokenized archive directly; do not retokenize the long traces.
- Enforce `MAX_LENGTH=8192`, rank <= 32, temperature/top-p inference contract in the manifest.
- Run a model-load dry run by default.
- Do not submit to Kaggle from this notebook.

Default behavior is safe: `RUN_DRY_RUN_VALIDATE=True` and `RUN_ONE_STEP_SMOKE_TRAIN=False`. Turn on the smoke train only after the dry-run report is reviewed.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            f"""import datetime, hashlib, json, os, pathlib, re, shutil, subprocess, sys, urllib.request, zipfile

VERSION = 'V202B_LONG_CONTEXT_PRETOKENIZED_20260504'
print('NOTEBOOK_VERSION =', VERSION)

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B')
WORK_ROOT = pathlib.Path('/content/kg1_v202b')
DATA_DIR = WORK_ROOT / 'data'
SCRIPT_DIR = WORK_ROOT / 'scripts'
REPORT_DIR = DRIVE_ROOT / 'reports'
DRIVE_DATA_DIR = DRIVE_ROOT / 'data'
BASELINE_DIR = DRIVE_ROOT / 'baseline_v194_rank19'
RANK19_BUILD = DRIVE_ROOT / 'init_adapter_v194_rank19_build'
INIT_ADAPTER = RANK19_BUILD / 'adapter'

for path in [DRIVE_ROOT, WORK_ROOT, DATA_DIR, SCRIPT_DIR, REPORT_DIR, DRIVE_DATA_DIR, BASELINE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RUN_DRY_RUN_VALIDATE = True
RUN_ONE_STEP_SMOKE_TRAIN = False
ALLOW_KAGGLE_SUBMIT = False

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
print('RUN_DRY_RUN_VALIDATE =', RUN_DRY_RUN_VALIDATE)
print('RUN_ONE_STEP_SMOKE_TRAIN =', RUN_ONE_STEP_SMOKE_TRAIN)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook is intentionally submit-disabled. Build a separate submit notebook after explicit authorization.')
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
    f'Use H100 or A100 80GB High-RAM for V202B; found {gpu_name} with {gpu_mem_mib} MiB.'
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
        md(
            """## Credentials

Kaggle is needed only to download the public dataset archive. Hugging Face token is optional but recommended to avoid rate limits.
"""
        ),
        code(
            """import atexit

def read_colab_secret(name):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value or ''
    except Exception:
        return ''

kaggle_dir = pathlib.Path('/root/.kaggle')
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_json = kaggle_dir / 'kaggle.json'

def cleanup_kaggle_credential():
    try:
        if kaggle_json.exists():
            kaggle_json.unlink()
            print('Removed transient Kaggle credential file:', kaggle_json)
    except Exception as exc:
        print('Kaggle credential cleanup warning:', repr(exc))

atexit.register(cleanup_kaggle_credential)

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

print('Kaggle credentials staged for dataset download.')
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
        'Exact V194 rank-19 submission.zip is required before any V202B run.\\n'
        f'Expected zip SHA256: {V194_RANK19_ZIP_SHA256}\\n'
        'Place it at one of these paths and rerun:\\n'
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
LOCAL_DATASET_ZIP = DATA_DIR / TONG_DATASET_ZIP_NAME

def download_tong_dataset_zip() -> pathlib.Path:
    if DRIVE_DATASET_ZIP.exists():
        try:
            assert_sha256(DRIVE_DATASET_ZIP, TONG_DATASET_ZIP_SHA256, 'cached Tong dataset zip')
            return DRIVE_DATASET_ZIP
        except RuntimeError:
            print('Cached Tong dataset zip SHA mismatch; deleting Drive cache copy.')
            DRIVE_DATASET_ZIP.unlink()

    tmp_dir = WORK_ROOT / 'tmp_kaggle_download'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'kaggle',
        'datasets',
        'download',
        '-d',
        TONG_DATASET_SLUG,
        '-p',
        str(tmp_dir),
    ], check=True)
    candidates = sorted(tmp_dir.glob('*.zip'))
    if not candidates:
        raise RuntimeError(f'Kaggle dataset download produced no zip in {tmp_dir}')
    downloaded = candidates[0]
    assert_sha256(downloaded, TONG_DATASET_ZIP_SHA256, 'downloaded Tong dataset zip')
    DRIVE_DATASET_ZIP.parent.mkdir(parents=True, exist_ok=True)
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
    print('corpus rows:', len(corpus_rows))
    assert len(corpus_rows) == 15979, len(corpus_rows)
    max_tokens = max(int(row['token_count']) for row in corpus_rows)
    p99_sample = sorted(int(row['token_count']) for row in corpus_rows)[int(len(corpus_rows) * 0.99)]
    print('token_count max:', max_tokens, 'p99:', p99_sample)
    assert max_tokens <= 8192, max_tokens
    sample_member = f\"corpus/{corpus_rows[0]['problem_id']}/{corpus_rows[0]['segment']}\"
    if sample_member not in names:
        raise RuntimeError(f'Missing sample segment member: {sample_member}')

dataset_manifest = {
    'dataset_slug': TONG_DATASET_SLUG,
    'zip_path': str(LOCAL_DATASET_ZIP),
    'zip_sha256': sha256_path(LOCAL_DATASET_ZIP),
    'corpus_jsonl_sha256': corpus_sha,
    'corpus_rows': len(corpus_rows),
    'max_token_count': max_tokens,
    'training_contract': TRAINING_CONTRACT,
}
(REPORT_DIR / 'v202b_dataset_manifest.json').write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('Dataset manifest:', REPORT_DIR / 'v202b_dataset_manifest.json')
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
            """COMMON_TRAIN_ENV = {
    'MODEL_NAME': MODEL_NAME,
    'MODEL_REVISION': MODEL_REVISION,
    'MODEL_DEVICE_MAP': 'auto',
    'PRETOKENIZED_ARCHIVE_ZIP': str(LOCAL_DATASET_ZIP),
    'EXPECTED_ARCHIVE_SHA256': TONG_DATASET_ZIP_SHA256,
    'PRETOKENIZED_VAL_EXAMPLES': '720',
    'PRETOKENIZED_VAL_COPY_ONLY': '0',
    'PRETOKENIZED_EXCLUDE_CATEGORIES': '',
    'MIN_TRAIN_EXAMPLES': '15000',
    'MIN_VAL_EXAMPLES': '720',
    'MIN_TOKENIZED_TRAIN_EXAMPLES': '15000',
    'MIN_TOKENIZED_VAL_EXAMPLES': '720',
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
    'MAX_STEPS': '1',
    'SAVE_EVERY_STEPS': '1',
    'EVAL_EVERY_STEPS': '1',
    'EVAL_MAX_EXAMPLES': '96',
    'LOG_EVERY_STEPS': '1',
    'MICRO_LOG_EVERY': '1',
    'SEED': '202',
    'SAMPLING_MODE': 'shuffle',
    'LEARNING_RATE': '2e-8',
    'FINAL_LEARNING_RATE': '2e-8',
    'GRAD_CLIP_NORM': '1.0',
    'BASELINE_EVAL_BEFORE_TRAIN': '1',
    'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1',
    'MAX_FINAL_EVAL_REGRESSION': '0.0',
    'ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA': '0.001',
    'ABORT_MAX_RESERVED_GIB': '79',
    'COMPUTE_PROVIDER': 'colab_v202b_long_context',
}
if os.environ.get('HF_TOKEN'):
    COMMON_TRAIN_ENV['HF_TOKEN'] = os.environ['HF_TOKEN']

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
    'env_contract': COMMON_TRAIN_ENV,
    'training_contract': TRAINING_CONTRACT,
    'policy': {
        'dry_run_default': RUN_DRY_RUN_VALIDATE,
        'smoke_train_default': RUN_ONE_STEP_SMOKE_TRAIN,
        'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
        'do_not_use_raw_generation_or_traj_as_sft': True,
        'no_submit_without_vllm_gate': True,
    },
}
(REPORT_DIR / 'v202b_training_contract.json').write_text(json.dumps(contract_report, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('Training contract:', REPORT_DIR / 'v202b_training_contract.json')
"""
        ),
        code(
            """DRY_RUN_OUT = DRIVE_ROOT / 'dry_run_longctx_8192'
if RUN_DRY_RUN_VALIDATE:
    env = os.environ.copy()
    env.update(COMMON_TRAIN_ENV)
    env.update({
        'OUTPUT_DIR': str(DRY_RUN_OUT),
        'RUN_ID': 'v202b-dryrun-v194-longctx-8192',
        'DRY_RUN_VALIDATE_ONLY': '1',
    })
    log_path = REPORT_DIR / 'v202b_dry_run.log'
    rc = stream_process([sys.executable, str(TRAIN_SCRIPT)], cwd=WORK_ROOT, env=env, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f'V202B dry-run failed; see {log_path}')
    dry_report = DRY_RUN_OUT / 'dry_run_model_recipe_report.json'
    if not dry_report.exists():
        raise RuntimeError(f'Dry-run report missing: {dry_report}')
    report = json.loads(dry_report.read_text(encoding='utf-8'))
    assert report['data']['pretokenized_archive_sha256'] == TONG_DATASET_ZIP_SHA256
    assert report['data']['tokenized_train_records'] >= 15000
    assert report['data']['tokenized_validation_records'] >= 720
    assert report['training']['max_length'] == 8192
    assert report['training']['batch_size'] == 1
    assert report['lora']['trainable_lora_module_filter']['enabled'] is True
    print('V202B dry-run passed:', dry_report)
else:
    print('RUN_DRY_RUN_VALIDATE=False; skipping dry-run.')
"""
        ),
        code(
            """if RUN_ONE_STEP_SMOKE_TRAIN:
    suffix = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
    TRAIN_OUT = DRIVE_ROOT / f'output_v202b_longctx_smoke_1s_{suffix}'
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(COMMON_TRAIN_ENV)
    env.update({
        'OUTPUT_DIR': str(TRAIN_OUT),
        'RUN_ID': 'v202b-h100-a100-v194-longctx-smoke-1s',
        'DRY_RUN_VALIDATE_ONLY': '0',
    })
    log_path = TRAIN_OUT / 'v202b_smoke_train.log'
    rc = stream_process([sys.executable, str(TRAIN_SCRIPT)], cwd=WORK_ROOT, env=env, log_path=log_path)
    manifest_path = TRAIN_OUT / 'final_adapter/v90_training_manifest.json'
    summary = {
        'output_dir': str(TRAIN_OUT),
        'returncode': rc,
        'manifest_path': str(manifest_path) if manifest_path.exists() else None,
        'log_path': str(log_path),
        'passed_no_regression_gate': False,
        'do_not_submit_without_vllm_gate': True,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        gate = manifest.get('training', {}).get('baseline_gate', {})
        baseline = gate.get('baseline_eval_loss')
        final = gate.get('final_eval_loss')
        summary['baseline_eval_loss'] = baseline
        summary['final_eval_loss'] = final
        summary['delta_vs_baseline'] = None if baseline is None or final is None else round(final - baseline, 6)
        summary['passed_no_regression_gate'] = bool(rc == 0 and baseline is not None and final is not None and final <= baseline)
    summary_path = TRAIN_OUT / 'v202b_smoke_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
    print('V202B smoke summary:', summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if rc != 0:
        raise RuntimeError(f'V202B smoke train failed or was blocked; see {summary_path}')
    if not summary['passed_no_regression_gate']:
        raise RuntimeError(f'V202B smoke train did not pass no-regression gate; see {summary_path}')
else:
    print('RUN_ONE_STEP_SMOKE_TRAIN=False; no training was run.')
    print('Next action after dry-run passes: set RUN_ONE_STEP_SMOKE_TRAIN=True and rerun from the training-contract cell downward.')
"""
        ),
        code(
            """final_report = {
    'version': VERSION,
    'status': 'V202B notebook completed configured stages',
    'dry_run_requested': RUN_DRY_RUN_VALIDATE,
    'smoke_train_requested': RUN_ONE_STEP_SMOKE_TRAIN,
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
    'dataset_zip': str(LOCAL_DATASET_ZIP),
    'dataset_zip_sha256': sha256_path(LOCAL_DATASET_ZIP),
    'init_adapter': str(INIT_ADAPTER),
    'init_adapter_sha256': sha256_path(INIT_ADAPTER / 'adapter_model.safetensors'),
    'reports': {
        'dataset_manifest': str(REPORT_DIR / 'v202b_dataset_manifest.json'),
        'training_contract': str(REPORT_DIR / 'v202b_training_contract.json'),
        'dry_run_log': str(REPORT_DIR / 'v202b_dry_run.log'),
    },
    'submit_policy': 'No Kaggle submit was performed. Build a separate submit notebook only after vLLM gate and explicit authorization.',
}
final_path = REPORT_DIR / 'v202b_final_notebook_report.json'
final_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print('=== V202B CONFIGURED STAGES COMPLETE ===')
print('final_report:', final_path)
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
        "KG1 V202B H100/A100 Long-Context Pretokenized Gate",
        "RUN_DRY_RUN_VALIDATE = True",
        "RUN_ONE_STEP_SMOKE_TRAIN = False",
        "ALLOW_KAGGLE_SUBMIT = False",
        f"V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'",
        f"V194_RANK19_ADAPTER_MODEL_SHA256 = '{V194_RANK19_ADAPTER_MODEL_SHA256}'",
        f"TONG_DATASET_ZIP_SHA256 = '{TONG_DATASET_ZIP_SHA256}'",
        "'MAX_LENGTH': '8192'",
        "'PRETOKENIZED_ARCHIVE_ZIP': str(LOCAL_DATASET_ZIP)",
        "'EXPECTED_ARCHIVE_SHA256': TONG_DATASET_ZIP_SHA256",
        "'TRAINABLE_LORA_MODULES': 'in_proj,out_proj,q_proj,k_proj,v_proj,o_proj'",
        "'REQUIRE_FINAL_EVAL_LTE_BASELINE': '1'",
        "'MAX_FINAL_EVAL_REGRESSION': '0.0'",
        "'ABORT_MAX_RESERVED_GIB': '79'",
        "load_tong_pretokenized_archive",
        "No Kaggle submit was performed.",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V202B notebook is missing {fragment!r}")

    forbidden = [
        "kaggle competitions submit",
        "kg1_v199b_safe_kaggle_submit.py",
        "DATA_FILE'] = '/content/kg1_v199/data/v198",
        "VAL_FILE'] = '/content/kg1_v199/data/v198",
        "MAX_LENGTH'] = '2048'",
        "MAX_LENGTH': '2048'",
        "bit_manipulation=2.5",
        "v198_v196_wrong_anti_regression=2.0",
        "PRETOKENIZED_VAL_COPY_ONLY': '1'",
        "ALLOW_KAGGLE_SUBMIT = True",
        "RUN_ONE_STEP_SMOKE_TRAIN = True",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V202B notebook contains forbidden fragment {fragment!r}")


def main() -> int:
    notebook = build_notebook()
    validate_notebook(notebook)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# V202B long-context pretokenized dry-run/smoke\n\n"
        "- Notebook: `notebooks/KG1_V202B_H100_A100_LONG_CONTEXT_PRETOKENIZED_COLAB_PRO.ipynb`.\n"
        f"- Starts only from exact V194 rank-19 adapter SHA `{V194_RANK19_ADAPTER_MODEL_SHA256}` and zip SHA `{V194_RANK19_ZIP_SHA256}`.\n"
        f"- Uses full Kaggle dataset `{TONG_DATASET_SLUG}` zip SHA `{TONG_DATASET_ZIP_SHA256}`.\n"
        "- Reads Tong/Huikang token/mask segments directly from the zip via `PRETOKENIZED_ARCHIVE_ZIP`; no chat-template retokenization.\n"
        "- Enforces `MAX_LENGTH=8192`, `LORA_R=32`, attention-only trainable filter, `BATCH_SIZE=1`, `MICRO_BATCH_SIZE=1`.\n"
        "- Default run performs model-load dry-run only and writes `dry_run_model_recipe_report.json`.\n"
        "- Optional one-step smoke train is disabled by default and remains no-submit with final eval no-regression gate.\n"
        "- Raw `generation.jsonl` and `nemotron_traj.csv` are deliberately not used as SFT because the V202 audit showed many false/partial outputs.\n"
        "- No Kaggle submit cell exists.\n",
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

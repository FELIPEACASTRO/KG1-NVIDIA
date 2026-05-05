#!/usr/bin/env python3
"""Build the V202D strict promotion gate Colab notebook."""

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


NOTEBOOK_PATH = Path("notebooks/KG1_V202D_H100_A100_STRICT_PROMOTION_GATE_COLAB_PRO.ipynb")
REPORT_PATH = Path("runs/v202_long_context_20260504/V202D_STRICT_PROMOTION_GATE_NEXT_ACTIONS.md")

EVAL_SCRIPT_URL = f"{BRANCH_SCRIPT_BASE}/kg1_v202_pretokenized_adapter_eval.py"
V202D_POSTTRAIN_GATE_URL = f"{BRANCH_SCRIPT_BASE}/kg1_v202d_posttrain_gate.py"
SCRIPT_URLS = {
    "hf_job_train_v90.py": TRAIN_SCRIPT_URL,
    "kg1_v202_pretokenized_adapter_eval.py": EVAL_SCRIPT_URL,
    "kg1_convert_local_training_adapter_to_kaggle_zip.py": (
        f"{BRANCH_SCRIPT_BASE}/kg1_convert_local_training_adapter_to_kaggle_zip.py"
    ),
    "kg1_v198_posttrain_gate.py": f"{BRANCH_SCRIPT_BASE}/kg1_v198_posttrain_gate.py",
    "kg1_v202d_posttrain_gate.py": V202D_POSTTRAIN_GATE_URL,
    "nemotron_submission_preflight.py": f"{BRANCH_SCRIPT_BASE}/nemotron_submission_preflight.py",
    "kg1_submission_gate.py": f"{BRANCH_SCRIPT_BASE}/kg1_submission_gate.py",
}

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v202d-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V202D H100/A100 Strict Promotion Gate

This notebook is the promotion gate after V202C:

- A passed: `delta=-0.000088`
- B passed: `delta=-0.000046`
- C passed but was effectively neutral: `delta=-0.000001`

V202D does not train and does not submit. It re-evaluates V194, A, and B on identical larger deterministic validation splits, compares overall and per-category losses, then packages only the selected candidate if it passes.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            f"""import datetime, hashlib, json, os, pathlib, re, shutil, subprocess, sys, urllib.request, zipfile

VERSION = 'V202D_STRICT_PROMOTION_GATE_20260505'
print('NOTEBOOK_VERSION =', VERSION)

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D')
V202C_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202C')
V202B_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B')
WORK_ROOT = pathlib.Path('/content/kg1_v202d')
DATA_DIR = WORK_ROOT / 'data'
SCRIPT_DIR = WORK_ROOT / 'scripts'
REPORT_DIR = DRIVE_ROOT / 'reports'
DRIVE_DATA_DIR = DRIVE_ROOT / 'data'
BASELINE_DIR = DRIVE_ROOT / 'baseline_v194_rank19'
RANK19_BUILD = DRIVE_ROOT / 'init_adapter_v194_rank19_build'
INIT_ADAPTER = RANK19_BUILD / 'adapter'
OUT_ROOT = DRIVE_ROOT / 'output_v202d_strict_promotion_gate'
EVAL_DIR = OUT_ROOT / 'eval_reports'

for path in [DRIVE_ROOT, WORK_ROOT, DATA_DIR, SCRIPT_DIR, REPORT_DIR, DRIVE_DATA_DIR, BASELINE_DIR, OUT_ROOT, EVAL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RUN_STRICT_EVAL = True
RUN_PACKAGE_SELECTED = True
FORCE_REEVAL = False
INCLUDE_C = False
ALLOW_KAGGLE_SUBMIT = False

MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'

V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'
V194_RANK19_ADAPTER_MODEL_SHA256 = '{V194_RANK19_ADAPTER_MODEL_SHA256}'
V194_RANK19_ADAPTER_CONFIG_SHA256 = '{V194_RANK19_ADAPTER_CONFIG_SHA256}'
V194_RANK19_PUBLIC_SCORE = '{V194_RANK19_PUBLIC_SCORE}'
V194_RANK19_RANK = '{V194_RANK19_RANK}'

TONG_DATASET_SLUG = '{TONG_DATASET_SLUG}'
TONG_DATASET_ZIP_NAME = '{TONG_DATASET_ZIP_NAME}'
TONG_DATASET_ZIP_SHA256 = '{TONG_DATASET_ZIP_SHA256}'
TONG_CORPUS_JSONL_SHA256 = '{TONG_CORPUS_JSONL_SHA256}'

SCRIPT_URLS = {json.dumps(SCRIPT_URLS, indent=4, sort_keys=True)}

PROMOTION_LABELS = [
    'A_all_shuffle_3s_lr2e8',
    'B_official_only_3s_lr2e8',
]
if INCLUDE_C:
    PROMOTION_LABELS.append('C_all_shuffle_5s_lr1e8')

SPLIT_SPECS = [
    {{
        'name': 'all720',
        'exclude_categories': '',
        'val_examples': 720,
        'eval_max_examples': 720,
    }},
    {{
        'name': 'official360',
        'exclude_categories': 'matching,concatenation,splitting,spelling,lstrip',
        'val_examples': 360,
        'eval_max_examples': 360,
    }},
]

CATEGORY_REGRESSION_TOLERANCE = 0.0005
OVERALL_REGRESSION_TOLERANCE = 0.0

print('DRIVE_ROOT =', DRIVE_ROOT)
print('V202C_ROOT =', V202C_ROOT)
print('OUT_ROOT =', OUT_ROOT)
print('RUN_STRICT_EVAL =', RUN_STRICT_EVAL)
print('RUN_PACKAGE_SELECTED =', RUN_PACKAGE_SELECTED)
print('FORCE_REEVAL =', FORCE_REEVAL)
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

def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')

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
    f'Use H100 or A100 80GB High-RAM for V202D; found {gpu_name} with {gpu_mem_mib} MiB.'
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
    pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202C/baseline_v194_rank19/submission.zip'),
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
        'Exact V194 rank-19 submission.zip is required before V202D.\\n'
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
V202C_DATASET_ZIP = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202C/data') / TONG_DATASET_ZIP_NAME
V202B_DATASET_ZIP = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202B/data') / TONG_DATASET_ZIP_NAME
LOCAL_DATASET_ZIP = DATA_DIR / TONG_DATASET_ZIP_NAME

def download_tong_dataset_zip() -> pathlib.Path:
    for cached in [DRIVE_DATASET_ZIP, V202C_DATASET_ZIP, V202B_DATASET_ZIP]:
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
                    print('Ignoring mismatched dataset cache:', cached)
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
print('Tong dataset ready:', LOCAL_DATASET_ZIP)
"""
        ),
        code(
            """for name, url in SCRIPT_URLS.items():
    target = SCRIPT_DIR / name
    print('downloading script:', name, url)
    urllib.request.urlretrieve(url, target)
    text = target.read_text(encoding='utf-8')
    if name == 'hf_job_train_v90.py':
        for feature in ['PRETOKENIZED_ARCHIVE_ZIP', 'load_tong_pretokenized_archive', 'evaluate_loss', 'PEFT_MANUAL_LOAD_METHOD']:
            if feature not in text:
                raise RuntimeError(f'Training script missing required feature: {feature}')
    if name == 'kg1_v202_pretokenized_adapter_eval.py':
        for feature in ['split-specs-json', 'load_trainable_adapter_or_create', 'overall_loss']:
            if feature not in text:
                raise RuntimeError(f'Eval script missing required feature: {feature}')
print('Scripts ready:', SCRIPT_DIR)
"""
        ),
        code(
            """summary_path = V202C_ROOT / 'output_v202c_longctx_candidate_sweep/v202c_candidates_summary.json'
if not summary_path.exists():
    candidates = sorted(V202C_ROOT.glob('output_v202c_longctx_candidate_sweep*/v202c_candidates_summary.json'))
    if not candidates:
        raise FileNotFoundError('V202C candidate summary not found in Drive.')
    summary_path = candidates[-1]
summary = json.loads(summary_path.read_text(encoding='utf-8'))
print('V202C summary:', summary_path)

candidate_by_label = {item['label']: item for item in summary.get('candidates', [])}
candidate_adapters = {}
for label in PROMOTION_LABELS:
    item = candidate_by_label.get(label)
    if item is None:
        raise RuntimeError(f'Missing V202C candidate in summary: {label}')
    if not item.get('passed_no_regression_gate'):
        raise RuntimeError(f'V202C candidate did not pass no-regression gate: {label}')
    out_dir = pathlib.Path(item['output_dir'])
    final_adapter = out_dir / 'final_adapter'
    if not adapter_ready(final_adapter):
        raise RuntimeError(f'Final adapter is not ready for {label}: {final_adapter}')
    candidate_adapters[label] = {
        'output_dir': str(out_dir),
        'adapter_dir': str(final_adapter),
        'v202c_metrics': item.get('metrics', {}),
    }
print('Promotion candidates:')
print(json.dumps(candidate_adapters, indent=2, sort_keys=True))
"""
        ),
        code(
            """eval_targets = [
    {
        'label': 'V194_rank19',
        'adapter_dir': str(INIT_ADAPTER),
        'output_dir': None,
        'kind': 'baseline',
    }
]
for label, payload in candidate_adapters.items():
    eval_targets.append({
        'label': label,
        'adapter_dir': payload['adapter_dir'],
        'output_dir': payload['output_dir'],
        'kind': 'candidate',
    })

split_specs_json = json.dumps(SPLIT_SPECS, sort_keys=True)
eval_reports = {}
if RUN_STRICT_EVAL:
    for target in eval_targets:
        safe_label = target['label'].replace('/', '_').replace(' ', '_')
        output_json = EVAL_DIR / f'{safe_label}_eval.json'
        log_path = EVAL_DIR / f'{safe_label}_eval.log'
        report = None
        if output_json.exists() and not FORCE_REEVAL:
            cached = json.loads(output_json.read_text(encoding='utf-8'))
            cached_split_names = [item.get('name') for item in cached.get('splits', [])]
            expected_split_names = [item['name'] for item in SPLIT_SPECS]
            if cached_split_names == expected_split_names:
                print('Reusing eval report:', output_json)
                report = cached
            else:
                print('Ignoring stale eval report with split names:', cached_split_names)
        if report is None:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / 'kg1_v202_pretokenized_adapter_eval.py'),
                '--training-script', str(SCRIPT_DIR / 'hf_job_train_v90.py'),
                '--adapter-dir', target['adapter_dir'],
                '--archive-zip', str(LOCAL_DATASET_ZIP),
                '--expected-archive-sha256', TONG_DATASET_ZIP_SHA256,
                '--output-json', str(output_json),
                '--label', target['label'],
                '--split-specs-json', split_specs_json,
                '--model-name', MODEL_NAME,
                '--model-revision', MODEL_REVISION,
                '--model-device-map', 'auto',
                '--max-length', '8192',
                '--seed', '202',
            ]
            rc = stream_process(cmd, cwd=WORK_ROOT, env=os.environ.copy(), log_path=log_path)
            if rc != 0:
                raise RuntimeError(f'Eval failed for {target["label"]}; see {log_path}')
            report = json.loads(output_json.read_text(encoding='utf-8'))
        report['target'] = target
        eval_reports[target['label']] = report
else:
    print('RUN_STRICT_EVAL=False; no evaluation was run.')

print('Eval reports loaded:', sorted(eval_reports))
"""
        ),
        code(
            """def split_map(report):
    return {item['name']: item for item in report['splits']}

def category_delta(candidate_split, baseline_split):
    rows = {}
    cats = sorted(set(candidate_split['per_category']) | set(baseline_split['per_category']))
    for cat in cats:
        cand = candidate_split['per_category'].get(cat)
        base = baseline_split['per_category'].get(cat)
        if cand is None or base is None:
            rows[cat] = {'delta': None, 'candidate_loss': None if cand is None else cand['loss'], 'baseline_loss': None if base is None else base['loss']}
        else:
            rows[cat] = {
                'delta': cand['loss'] - base['loss'],
                'candidate_loss': cand['loss'],
                'baseline_loss': base['loss'],
                'examples': cand['examples'],
            }
    return rows

baseline_report = eval_reports.get('V194_rank19')
if baseline_report is None:
    raise RuntimeError('Missing V194 eval report.')
baseline_splits = split_map(baseline_report)

promotion_results = []
for label in PROMOTION_LABELS:
    report = eval_reports[label]
    candidate_splits = split_map(report)
    split_results = []
    reasons = []
    total_delta = 0.0
    worst_category_delta = -999.0
    for split in SPLIT_SPECS:
        name = split['name']
        cand_split = candidate_splits[name]
        base_split = baseline_splits[name]
        overall_delta = cand_split['overall_loss'] - base_split['overall_loss']
        cat_rows = category_delta(cand_split, base_split)
        positive_cat_deltas = [
            value['delta'] for value in cat_rows.values()
            if value.get('delta') is not None
        ]
        max_cat_delta = max(positive_cat_deltas) if positive_cat_deltas else 0.0
        worst_category_delta = max(worst_category_delta, max_cat_delta)
        total_delta += overall_delta
        if overall_delta > OVERALL_REGRESSION_TOLERANCE:
            reasons.append(f'{name}:overall_regression:{overall_delta:.8f}')
        if max_cat_delta > CATEGORY_REGRESSION_TOLERANCE:
            reasons.append(f'{name}:category_regression:{max_cat_delta:.8f}')
        split_results.append({
            'name': name,
            'baseline_loss': base_split['overall_loss'],
            'candidate_loss': cand_split['overall_loss'],
            'overall_delta': overall_delta,
            'max_category_delta': max_cat_delta,
            'per_category_delta': cat_rows,
            'sample_examples': cand_split['sample_examples'],
        })
    ready = not reasons
    promotion_results.append({
        'label': label,
        'ready': ready,
        'reasons': reasons,
        'total_delta': total_delta,
        'worst_category_delta': worst_category_delta,
        'splits': split_results,
        'candidate': candidate_adapters[label],
    })

ready = [item for item in promotion_results if item['ready']]
selected = sorted(ready, key=lambda item: (item['total_delta'], item['worst_category_delta']))[0] if ready else None
promotion_report = {
    'schema_version': 1,
    'generated_at': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
    'version': VERSION,
    'v202c_summary_path': str(summary_path),
    'baseline': {
        'label': 'V194 rank-19',
        'rank': V194_RANK19_RANK,
        'public_score': V194_RANK19_PUBLIC_SCORE,
        'zip_sha256': V194_RANK19_ZIP_SHA256,
        'adapter_model_sha256': V194_RANK19_ADAPTER_MODEL_SHA256,
    },
    'split_specs': SPLIT_SPECS,
    'tolerances': {
        'overall_regression_tolerance': OVERALL_REGRESSION_TOLERANCE,
        'category_regression_tolerance': CATEGORY_REGRESSION_TOLERANCE,
    },
    'results': promotion_results,
    'selected_label': selected['label'] if selected else None,
    'decision': {
        'ready': selected is not None,
        'reasons': [] if selected else ['no_candidate_passed_strict_promotion_gate'],
        'do_not_submit_without_explicit_authorization': True,
    },
}
promotion_report_path = OUT_ROOT / 'v202d_strict_promotion_report.json'
write_json(promotion_report_path, promotion_report)
print('V202D strict promotion report:', promotion_report_path)
print(json.dumps([
    {
        'label': item['label'],
        'ready': item['ready'],
        'total_delta': item['total_delta'],
        'worst_category_delta': item['worst_category_delta'],
        'reasons': item['reasons'],
    }
    for item in promotion_results
], indent=2, sort_keys=True))
print('SELECTED_LABEL =', promotion_report['selected_label'])
"""
        ),
        code(
            """package_report = None
if RUN_PACKAGE_SELECTED and promotion_report['decision']['ready']:
    selected_label = promotion_report['selected_label']
    selected_output = pathlib.Path(candidate_adapters[selected_label]['output_dir'])
    posttrain_log = OUT_ROOT / f'{selected_label}_posttrain_gate.log'
    rc = stream_process([
        sys.executable,
        str(SCRIPT_DIR / 'kg1_v202d_posttrain_gate.py'),
        '--root', str(WORK_ROOT),
        '--output-root', str(selected_output),
        '--candidate-label', selected_label,
        '--fail-on-block',
    ], cwd=WORK_ROOT, env=os.environ.copy(), log_path=posttrain_log)
    if rc != 0:
        raise RuntimeError(f'V202D posttrain gate failed for {selected_label}; see {posttrain_log}')
    posttrain_report_path = selected_output / 'posttrain_kaggle_gate_v202d/v202d_posttrain_gate_report.json'
    posttrain_report = json.loads(posttrain_report_path.read_text(encoding='utf-8'))
    primary_zip = pathlib.Path(posttrain_report['decision']['primary_zip'])
    preflight_json = OUT_ROOT / f'{selected_label}_submission_preflight.json'
    preflight_log = OUT_ROOT / f'{selected_label}_submission_preflight.log'
    rc = stream_process([
        sys.executable,
        str(SCRIPT_DIR / 'nemotron_submission_preflight.py'),
        '--adapter-zip', str(primary_zip),
        '--output-json', str(preflight_json),
        '--fail-on-block',
    ], cwd=WORK_ROOT, env=os.environ.copy(), log_path=preflight_log)
    if rc != 0:
        raise RuntimeError(f'Nemotron submission preflight failed for {selected_label}; see {preflight_log}')
    preflight_report = json.loads(preflight_json.read_text(encoding='utf-8'))
    package_report = {
        'selected_label': selected_label,
        'selected_output_dir': str(selected_output),
        'posttrain_report_path': str(posttrain_report_path),
        'primary_zip': str(primary_zip),
        'primary_zip_sha256': sha256_path(primary_zip),
        'preflight_json': str(preflight_json),
        'preflight_production_ready': preflight_report['decision']['production_ready'],
        'do_not_submit_without_explicit_authorization': True,
    }
elif RUN_PACKAGE_SELECTED:
    print('No candidate passed strict promotion gate; packaging skipped.')
else:
    print('RUN_PACKAGE_SELECTED=False; packaging skipped.')

final_manifest = {
    'version': VERSION,
    'strict_promotion_report': str(promotion_report_path),
    'package_report': package_report,
    'policy': 'No Kaggle submit was performed. Submit only after explicit authorization and review of this manifest.',
}
final_manifest_path = OUT_ROOT / 'v202d_final_manifest.json'
write_json(final_manifest_path, final_manifest)
print('V202D final manifest:', final_manifest_path)
print(json.dumps(final_manifest, indent=2, sort_keys=True))
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
        "KG1 V202D H100/A100 Strict Promotion Gate",
        "ALLOW_KAGGLE_SUBMIT = False",
        "RUN_STRICT_EVAL = True",
        "RUN_PACKAGE_SELECTED = True",
        "FORCE_REEVAL = False",
        "A_all_shuffle_3s_lr2e8",
        "B_official_only_3s_lr2e8",
        "all720",
        "official360",
        "kg1_v202_pretokenized_adapter_eval.py",
        "kg1_v202d_posttrain_gate.py",
        "CATEGORY_REGRESSION_TOLERANCE = 0.0005",
        "No Kaggle submit was performed.",
        f"V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'",
        f"TONG_DATASET_ZIP_SHA256 = '{TONG_DATASET_ZIP_SHA256}'",
    ]
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(f"Built V202D notebook is missing {fragment!r}")
    forbidden = [
        "kaggle competitions submit",
        "ALLOW_KAGGLE_SUBMIT = True",
        "MAX_LENGTH': '2048'",
        "MAX_LENGTH'] = '2048'",
        "data/v198",
        "FORCE_REEVAL = True",
    ]
    for fragment in forbidden:
        if fragment in source:
            raise RuntimeError(f"Built V202D notebook contains forbidden fragment {fragment!r}")


def main() -> int:
    notebook = build_notebook()
    validate_notebook(notebook)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# V202D Strict Promotion Gate",
                "",
                "Notebook generated:",
                f"- `{NOTEBOOK_PATH}`",
                "",
                "Use after V202C passed A/B/C. V202D re-evaluates V194, A, and B on larger deterministic splits and packages only a selected candidate.",
                "",
                "Policy:",
                "- no training",
                "- no Kaggle submit",
                "- no promotion unless overall loss is <= V194 on all required splits",
                "- no category regression above 0.0005",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

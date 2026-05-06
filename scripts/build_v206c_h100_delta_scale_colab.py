#!/usr/bin/env python3
"""Build the V206C H100 delta-scaling Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    ".claude/worktrees/competent-shamir/notebooks/"
    "KG1_V206C_H100_DELTA_SCALE_COLAB.ipynb"
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
V194_RANK19_ZIP_SHA256 = "49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8"
V194_RANK19_ADAPTER_MODEL_SHA256 = "01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f"
V194_RANK19_ADAPTER_CONFIG_SHA256 = "e5499f128fde60d32d0595d427e4fe84d8abe6dbde1d80886c970e8184e4b743"
V198_VAL_SHA256 = "e59c907c6545e5e587097a64762e3e874508e8cd74d85d5c7c79354ebe56e73c"


_CELL_COUNTER = 0


def runtime_bootstrap_source() -> str:
    return f"""# V206C runtime bootstrap: keeps later cells robust after a runtime reset or out-of-order execution.
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

VERSION = globals().get('VERSION', 'V206C_H100_DELTA_SCALE_20260506')
REPO_URL = globals().get('REPO_URL', os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'))
REPO_BRANCH = globals().get('REPO_BRANCH', os.environ.get('KG1_REPO_BRANCH', 'claude/competent-shamir'))
ROOT = globals().get('ROOT', pathlib.Path('/content/kg1'))
SCRIPT_DIR = globals().get('SCRIPT_DIR', ROOT / 'scripts')
BUILD_SCALE_SCRIPT = globals().get('BUILD_SCALE_SCRIPT', SCRIPT_DIR / 'build_v206c_delta_scaled_adapters.py')
EVAL_SCRIPT = globals().get('EVAL_SCRIPT', SCRIPT_DIR / 'hf_eval_adapter_loss_v206c.py')
PREFLIGHT_SCRIPT = globals().get('PREFLIGHT_SCRIPT', SCRIPT_DIR / 'nemotron_submission_preflight.py')
DRIVE_ROOT = globals().get('DRIVE_ROOT', pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V206C'))
DRIVE_V202D = globals().get('DRIVE_V202D', pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D'))
DRIVE_V206B = globals().get('DRIVE_V206B', pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V206B'))
RANK19_BUILD = globals().get('RANK19_BUILD', DRIVE_V202D / 'init_adapter_v194_rank19_build')
INIT_ADAPTER = globals().get('INIT_ADAPTER', RANK19_BUILD / 'adapter')
FAILED_V206B_ADAPTER = globals().get('FAILED_V206B_ADAPTER', DRIVE_V206B / 'output_v206b_answer_only_h100_loss_gated/train_v206b_answer_only_1s_lr1e9/final_adapter')
OUT_ROOT = globals().get('OUT_ROOT', DRIVE_ROOT / 'output_v206c_delta_scale')
SCALE_OUT = globals().get('SCALE_OUT', OUT_ROOT / 'scaled_adapters')
REPORT_DIR = globals().get('REPORT_DIR', OUT_ROOT / 'reports')
MODEL_NAME = globals().get('MODEL_NAME', '{MODEL_NAME}')
MODEL_REVISION = globals().get('MODEL_REVISION', '{MODEL_REVISION}')
V198_VAL_SHA256 = globals().get('V198_VAL_SHA256', '{V198_VAL_SHA256}')
SCALES = globals().get('SCALES', '0.00,0.01,0.02,0.05,0.10')
ALLOW_KAGGLE_SUBMIT = globals().get('ALLOW_KAGGLE_SUBMIT', False)
for _path in [DRIVE_ROOT, OUT_ROOT, SCALE_OUT, REPORT_DIR]:
    _path.mkdir(parents=True, exist_ok=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook is intentionally submit-disabled.')

if 'stream_process' not in globals():
    def stream_process(cmd, cwd=None, env=None, log_path=None):
        cmd = [str(part) for part in cmd]
        print('+', ' '.join(cmd))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open('w', encoding='utf-8')
        else:
            log_handle = None
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
        return rc

if 'sha256_path' not in globals():
    def sha256_path(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

if 'write_json' not in globals():
    def write_json(path: pathlib.Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')

if not ROOT.exists():
    raise RuntimeError('Repo /content/kg1 is missing. Run the clone/setup cell before this cell.')
for _required in [BUILD_SCALE_SCRIPT, EVAL_SCRIPT, PREFLIGHT_SCRIPT]:
    if not _required.exists():
        raise FileNotFoundError(f'Missing required script. Run the clone/setup cell again: {{_required}}')
if not INIT_ADAPTER.exists():
    raise RuntimeError(f'V194 init adapter is missing. Run the V194 adapter setup cell: {{INIT_ADAPTER}}')
if not FAILED_V206B_ADAPTER.exists():
    raise RuntimeError(f'V206B forensic adapter is missing. Run V206B first: {{FAILED_V206B_ADAPTER}}')

"""


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v206c-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V206C H100 Delta-Scale Colab

Purpose: test a no-training rescue path after V206A and V206B both regressed validation loss.

This notebook:

- clones `FELIPEACASTRO/KG1-NVIDIA` branch `claude/competent-shamir`;
- requires the exact V194 rank-19 baseline adapter by SHA256;
- requires the failed V206B forensic adapter already saved in Drive;
- builds small delta-scaled adapters `V194 + scale * (V206B - V194)`;
- evaluates each scale on the full V198 strict validation loss proxy;
- preflights only the best non-baseline scale if it beats the V194 loss;
- never submits to Kaggle.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            f"""import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

VERSION = 'V206C_H100_DELTA_SCALE_20260506'
print('NOTEBOOK_VERSION =', VERSION)

REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', 'claude/competent-shamir')
ROOT = pathlib.Path('/content/kg1')
SCRIPT_DIR = ROOT / 'scripts'
BUILD_SCALE_SCRIPT = SCRIPT_DIR / 'build_v206c_delta_scaled_adapters.py'
EVAL_SCRIPT = SCRIPT_DIR / 'hf_eval_adapter_loss_v206c.py'
PREFLIGHT_SCRIPT = SCRIPT_DIR / 'nemotron_submission_preflight.py'

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V206C')
DRIVE_V202D = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D')
DRIVE_V206B = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V206B')
RANK19_BUILD = DRIVE_V202D / 'init_adapter_v194_rank19_build'
INIT_ADAPTER = RANK19_BUILD / 'adapter'
FAILED_V206B_ADAPTER = DRIVE_V206B / 'output_v206b_answer_only_h100_loss_gated/train_v206b_answer_only_1s_lr1e9/final_adapter'
OUT_ROOT = DRIVE_ROOT / 'output_v206c_delta_scale'
SCALE_OUT = OUT_ROOT / 'scaled_adapters'
REPORT_DIR = OUT_ROOT / 'reports'

MODEL_NAME = '{MODEL_NAME}'
MODEL_REVISION = '{MODEL_REVISION}'
V194_RANK19_ZIP_SHA256 = '{V194_RANK19_ZIP_SHA256}'
V194_RANK19_ADAPTER_MODEL_SHA256 = '{V194_RANK19_ADAPTER_MODEL_SHA256}'
V194_RANK19_ADAPTER_CONFIG_SHA256 = '{V194_RANK19_ADAPTER_CONFIG_SHA256}'
V198_VAL_SHA256 = '{V198_VAL_SHA256}'
SCALES = '0.00,0.01,0.02,0.05,0.10'
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, SCALE_OUT, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print('REPO_URL =', REPO_URL)
print('REPO_BRANCH =', REPO_BRANCH)
print('FAILED_V206B_ADAPTER =', FAILED_V206B_ADAPTER)
print('OUT_ROOT =', OUT_ROOT)
print('SCALES =', SCALES)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook is intentionally submit-disabled.')
"""
        ),
        code(
            """def run(cmd, cwd=None, env=None, check=True):
    cmd = [str(part) for part in cmd]
    print('+', ' '.join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def stream_process(cmd, cwd=None, env=None, log_path=None):
    cmd = [str(part) for part in cmd]
    print('+', ' '.join(cmd))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8')
    else:
        log_handle = None
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
    return rc


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def assert_sha256(path: pathlib.Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    print(f'{label} sha256:', observed)
    if observed.lower() != expected.lower():
        raise RuntimeError(f'{label} SHA mismatch for {path}: {observed} != {expected}')
    return observed


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')


def read_colab_secret(name):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value or ''
    except Exception:
        return ''
"""
        ),
        code(
            """gpu_csv = subprocess.check_output(
    'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits',
    shell=True,
).decode().strip()
print('GPU:', gpu_csv)
parts = [part.strip() for part in gpu_csv.split(',')]
gpu_name = parts[0]
gpu_mem_mib = int(parts[1])
driver_version = parts[2]
assert ('H100' in gpu_name) or ('A100' in gpu_name and gpu_mem_mib >= 75000), (
    f'Use H100 or A100 80GB High-RAM; found {gpu_name} with {gpu_mem_mib} MiB.'
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
            """if ROOT.exists():
    run(['git', '-C', ROOT, 'fetch', 'origin', REPO_BRANCH])
    run(['git', '-C', ROOT, 'checkout', REPO_BRANCH])
    run(['git', '-C', ROOT, 'pull', '--ff-only', 'origin', REPO_BRANCH])
else:
    run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, ROOT])

commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip()
print('Repo commit:', commit)
for required in [BUILD_SCALE_SCRIPT, EVAL_SCRIPT, PREFLIGHT_SCRIPT, ROOT / 'data/v198/v198_micro_val.strict.jsonl']:
    if not required.exists():
        raise FileNotFoundError(required)
assert_sha256(ROOT / 'data/v198/v198_micro_val.strict.jsonl', V198_VAL_SHA256, 'V198 strict validation')
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
            """hf_token = os.environ.get('HF_TOKEN') or read_colab_secret('HF_TOKEN') or read_colab_secret('HUGGINGFACE_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('HF_TOKEN staged from env/Colab secrets; value is not printed.')
else:
    print('HF_TOKEN not found. If the NVIDIA model is gated for this account, add HF_TOKEN as a Colab secret.')
"""
        ),
        code(
            """def adapter_ready(path: pathlib.Path) -> bool:
    return (path / 'adapter_config.json').exists() and (path / 'adapter_model.safetensors').exists()


def extract_v194_zip(candidate: pathlib.Path) -> pathlib.Path:
    safe_source = OUT_ROOT / 'source_v194_rank19_submission.zip'
    safe_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, safe_source)
    assert_sha256(safe_source, V194_RANK19_ZIP_SHA256, 'V194 rank-19 submission.zip')
    if RANK19_BUILD.exists():
        shutil.rmtree(RANK19_BUILD)
    INIT_ADAPTER.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(safe_source) as zf:
        members = {name for name in zf.namelist() if not name.endswith('/')}
        expected = {'adapter_model.safetensors', 'adapter_config.json'}
        if members != expected:
            raise RuntimeError(f'Unexpected V194 zip members: {sorted(members)}')
        zf.extract('adapter_model.safetensors', INIT_ADAPTER)
        zf.extract('adapter_config.json', INIT_ADAPTER)
    cached_zip = RANK19_BUILD / 'submission.zip'
    shutil.copy2(safe_source, cached_zip)
    assert_sha256(INIT_ADAPTER / 'adapter_model.safetensors', V194_RANK19_ADAPTER_MODEL_SHA256, 'V194 adapter_model')
    assert_sha256(INIT_ADAPTER / 'adapter_config.json', V194_RANK19_ADAPTER_CONFIG_SHA256, 'V194 adapter_config')
    assert_sha256(cached_zip, V194_RANK19_ZIP_SHA256, 'cached V194 submission.zip')
    return INIT_ADAPTER


def ensure_v194_adapter() -> pathlib.Path:
    cached_zip = RANK19_BUILD / 'submission.zip'
    if adapter_ready(INIT_ADAPTER) and cached_zip.exists():
        assert_sha256(INIT_ADAPTER / 'adapter_model.safetensors', V194_RANK19_ADAPTER_MODEL_SHA256, 'cached V194 adapter_model')
        assert_sha256(INIT_ADAPTER / 'adapter_config.json', V194_RANK19_ADAPTER_CONFIG_SHA256, 'cached V194 adapter_config')
        assert_sha256(cached_zip, V194_RANK19_ZIP_SHA256, 'cached V194 submission.zip')
        return INIT_ADAPTER

    candidates = []
    if os.environ.get('V194_RANK19_ZIP'):
        candidates.append(pathlib.Path(os.environ['V194_RANK19_ZIP']))
    candidates += [
        DRIVE_V202D / 'init_adapter_v194_rank19_build/submission.zip',
        DRIVE_V202D / 'baseline_v194_rank19/submission.zip',
        pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202/baseline_v194_rank19/submission.zip'),
        pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V199B/baseline_v194_rank19/submission.zip'),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return extract_v194_zip(candidate)
            except RuntimeError as exc:
                print('Skipping non-matching V194 candidate:', candidate, exc)
    raise FileNotFoundError('Exact V194 rank-19 baseline submission.zip not found.')


ensure_v194_adapter()
if not adapter_ready(FAILED_V206B_ADAPTER):
    raise FileNotFoundError(
        'V206B forensic adapter not found. Run KG1_V206B_H100_ANSWER_ONLY_LOSS_GATED_COLAB first; '
        f'expected {FAILED_V206B_ADAPTER}'
    )
print('V194 adapter ready:', INIT_ADAPTER)
print('V206B forensic adapter ready:', FAILED_V206B_ADAPTER)
"""
        ),
        code(
            runtime_bootstrap_source()
            + """build_log = OUT_ROOT / 'v206c_build_delta_scaled_adapters.log'
rc = stream_process(
    [
        sys.executable,
        BUILD_SCALE_SCRIPT,
        '--baseline-adapter-dir', INIT_ADAPTER,
        '--candidate-adapter-dir', FAILED_V206B_ADAPTER,
        '--output-dir', SCALE_OUT,
        '--scales', SCALES,
        '--run-id', 'v206c-v206b-delta-scale',
    ],
    cwd=ROOT,
    env=os.environ.copy(),
    log_path=build_log,
)
if rc != 0:
    raise RuntimeError(f'V206C delta-scale build failed; see {build_log}')
manifest_path = SCALE_OUT / 'v206c_delta_scale_manifest.json'
scale_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
print('V206C scaled adapters built:', manifest_path)
print('Alignment:', json.dumps(scale_manifest['alignment'], indent=2, sort_keys=True))
"""
        ),
        code(
            runtime_bootstrap_source()
            + """adapter_specs = []
manifest_path = SCALE_OUT / 'v206c_delta_scale_manifest.json'
if 'scale_manifest' not in globals():
    if not manifest_path.exists():
        raise FileNotFoundError(f'V206C scale manifest not found. Run the build cell first: {manifest_path}')
    scale_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
for output in scale_manifest['outputs']:
    label = output['label']
    adapter_dir = output['adapter_dir']
    adapter_specs.append(f'{label}={adapter_dir}')

eval_env = os.environ.copy()
eval_env.update({
    'MODEL_NAME': MODEL_NAME,
    'MODEL_REVISION': MODEL_REVISION,
    'MODEL_DEVICE_MAP': 'auto',
    'VAL_FILE': '/content/kg1/data/v198/v198_micro_val.strict.jsonl',
    'EXPECTED_VAL_SHA256': V198_VAL_SHA256,
    'MIN_VAL_EXAMPLES': '720',
    'MIN_TOKENIZED_VAL_EXAMPLES': '720',
    'REQUIRE_OFFSET_MASK': '1',
    'MAX_LENGTH': '8192',
    'LORA_R': '32',
    'LORA_ALPHA': '32',
    'LORA_DROPOUT': '0.0',
    'LORA_TARGET_MODULES': 'down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj',
    'PEFT_MANUAL_LOAD_METHOD': 'direct',
    'EVAL_MAX_EXAMPLES': '720',
    'ADAPTER_EVAL_DIRS': ';'.join(adapter_specs),
    'BASELINE_LABEL': 's0p000',
    'EVAL_OUTPUT_JSON': str(REPORT_DIR / 'v206c_delta_scale_eval_loss_report.json'),
    'UPLOAD_TO_HF': '0',
    'COMPUTE_PROVIDER': 'colab_v206c_h100_delta_scale_eval',
})
if os.environ.get('HF_TOKEN'):
    eval_env['HF_TOKEN'] = os.environ['HF_TOKEN']

eval_log = OUT_ROOT / 'v206c_delta_scale_eval_loss.log'
rc = stream_process([sys.executable, EVAL_SCRIPT], cwd=ROOT, env=eval_env, log_path=eval_log)
if rc != 0:
    raise RuntimeError(f'V206C loss evaluation failed; see {eval_log}')
eval_report = json.loads((REPORT_DIR / 'v206c_delta_scale_eval_loss_report.json').read_text(encoding='utf-8'))
print('V206C eval status:', eval_report['status'])
print('Baseline loss:', eval_report['baseline_eval_loss'])
print('Best non-baseline:', json.dumps(eval_report['best_non_baseline'], indent=2, sort_keys=True))
"""
        ),
        code(
            runtime_bootstrap_source()
            + """eval_report_path = REPORT_DIR / 'v206c_delta_scale_eval_loss_report.json'
if 'eval_report' not in globals():
    if not eval_report_path.exists():
        raise FileNotFoundError(f'V206C eval report not found. Run the evaluation cell first: {eval_report_path}')
    eval_report = json.loads(eval_report_path.read_text(encoding='utf-8'))
manifest_path = SCALE_OUT / 'v206c_delta_scale_manifest.json'
if 'scale_manifest' not in globals():
    scale_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

preflight_report = None
best = eval_report['best_non_baseline']
if eval_report['approved_loss_prefilter']:
    by_label = {item['label']: item for item in scale_manifest['outputs']}
    best_zip = pathlib.Path(by_label[best['label']]['zip_path'])
    preflight_json = REPORT_DIR / f\"v206c_preflight_{best['label']}.json\"
    preflight_log = OUT_ROOT / f\"v206c_preflight_{best['label']}.log\"
    rc = stream_process(
        [
            sys.executable,
            PREFLIGHT_SCRIPT,
            '--adapter-zip', best_zip,
            '--output-json', preflight_json,
            '--fail-on-block',
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        log_path=preflight_log,
    )
    if rc != 0:
        raise RuntimeError(f'V206C preflight failed for {best_zip}; see {preflight_log}')
    preflight_report = {
        'label': best['label'],
        'adapter_zip': str(best_zip),
        'adapter_zip_sha256': sha256_path(best_zip),
        'preflight_json': str(preflight_json),
        'human_approval_required_before_submit': True,
    }
    write_json(REPORT_DIR / 'v206c_candidate_package_summary.json', preflight_report)
    print('V206C loss-prefilter candidate passed preflight:')
    print(json.dumps(preflight_report, indent=2, sort_keys=True))
else:
    print('\\nV206C did not beat the V194 loss baseline. No package/preflight candidate will be promoted.')
    print('This is a clean reject, not a runtime error.')
"""
        ),
        code(
            runtime_bootstrap_source()
            + """eval_report_path = REPORT_DIR / 'v206c_delta_scale_eval_loss_report.json'
if 'eval_report' not in globals():
    if not eval_report_path.exists():
        raise FileNotFoundError(f'V206C eval report not found. Run the evaluation cell first: {eval_report_path}')
    eval_report = json.loads(eval_report_path.read_text(encoding='utf-8'))
if 'preflight_report' not in globals():
    preflight_report = None

final_summary = {
    'version': VERSION,
    'output_root': str(OUT_ROOT),
    'scale_manifest': str(SCALE_OUT / 'v206c_delta_scale_manifest.json'),
    'eval_report': str(REPORT_DIR / 'v206c_delta_scale_eval_loss_report.json'),
    'approved_loss_prefilter': eval_report['approved_loss_prefilter'],
    'best_non_baseline': eval_report['best_non_baseline'],
    'preflight_report': preflight_report,
    'no_training_performed': True,
    'no_kaggle_submit_performed': True,
}
write_json(OUT_ROOT / 'V206C_FINAL_RUN_SUMMARY.json', final_summary)
print(json.dumps(final_summary, indent=2, sort_keys=True))
print('FINAL_SUMMARY_PATH =', OUT_ROOT / 'V206C_FINAL_RUN_SUMMARY.json')
print('KAGGLE_SUBMIT = BLOCKED_IN_THIS_NOTEBOOK')
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "gpuClass": "premium",
            "colab": {
                "name": "KG1_V206C_H100_DELTA_SCALE_COLAB.ipynb",
                "provenance": [],
                "machine_shape": "hm",
            },
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


def main() -> None:
    notebook = build_notebook()
    text = "\n".join("".join(cell.get("source") or []) for cell in notebook["cells"])
    required = [
        "KG1 V206C H100 Delta-Scale Colab",
        "ALLOW_KAGGLE_SUBMIT = False",
        "build_v206c_delta_scaled_adapters.py",
        "hf_eval_adapter_loss_v206c.py",
        "V206B forensic adapter ready",
        "SCALES = '0.00,0.01,0.02,0.05,0.10'",
        "--adapter-zip",
        "KAGGLE_SUBMIT = BLOCKED_IN_THIS_NOTEBOOK",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Generated notebook missing markers: {missing}")
    forbidden = [
        "kaggle competitions submit",
        "ALLOW_KAGGLE_SUBMIT = True",
        "--submission-zip",
        "hf_job_train_v90.py",
        "'MAX_STEPS'",
    ]
    present = [item for item in forbidden if item in text]
    if present:
        raise RuntimeError(f"Generated notebook contains forbidden markers: {present}")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()

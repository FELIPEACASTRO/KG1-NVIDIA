import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V202_H100_A100_LONG_CONTEXT_EVAL_GATE_COLAB_PRO.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """# KG1 V202 H100/A100 Long-Context Eval Gate

This notebook is the next safe step after V201A/V201B/V201C failed the no-regression gate.

Purpose:
- Keep V194 rank-19 as the immutable submit floor.
- Stop direct 2048-token micro-continuation from V194.
- Audit and prepare the long-context Tong/Huikang-style data path.
- Build a manifest for a future 8192-token candidate and block blind submit.

Default behavior: data audit only. Training is disabled until `RUN_TRAINING = True` is changed manually after the audit passes.
"""
    ),
    code(
        """from google.colab import drive
drive.mount('/content/drive')
"""
    ),
    code(
        """import os, pathlib, re, shutil, subprocess, sys

VERSION = 'V202_LONG_CONTEXT_EVAL_GATE_20260504'
print('NOTEBOOK_VERSION =', VERSION)

V202_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202')
V202_ROOT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path('/content/kg1_v202')
WORK.mkdir(parents=True, exist_ok=True)
DATA_DIR = WORK / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = V202_ROOT / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

V194_ZIP_SHA256 = '49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8'
V194_ADAPTER_SHA256 = '01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f'
MODEL_NAME = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
MODEL_REVISION = 'cbd3fa9f933d55ef16a84236559f4ee2a0526848'

RUN_TRAINING = False
RUN_VLLM_GATE = False
ALLOW_KAGGLE_SUBMIT = False

REQUIRED_TRAINING_CONTRACT = {
    'max_length': 8192,
    'max_model_len': 8192,
    'max_tokens': 7680,
    'max_lora_rank': 32,
    'temperature': 0.0,
    'top_p': 1.0,
}

print('V202_ROOT =', V202_ROOT)
print('WORK =', WORK)
print('RUN_TRAINING =', RUN_TRAINING)
print('RUN_VLLM_GATE =', RUN_VLLM_GATE)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('This notebook must not submit directly. Submit only after explicit final authorization.')
"""
    ),
    code(
        """gpu_csv = subprocess.check_output(
    'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits',
    shell=True,
).decode().strip()
print('GPU:', gpu_csv)
parts = [p.strip() for p in gpu_csv.split(',')]
gpu_name = parts[0]
gpu_mem_mib = int(parts[1])
assert ('H100' in gpu_name) or ('A100' in gpu_name and gpu_mem_mib >= 75000), (
    f'Use H100 or A100 80GB High-RAM. Found {gpu_name} with {gpu_mem_mib} MiB.'
)

meminfo = pathlib.Path('/proc/meminfo').read_text(encoding='utf-8')
host_mem_kib = int(re.search(r'MemTotal:\\s+(\\d+)', meminfo).group(1))
host_mem_gib = host_mem_kib / 1024 / 1024
disk_free_gib = shutil.disk_usage('/content').free / 1024**3
print(f'Host RAM: {host_mem_gib:.1f} GiB')
print(f'/content free: {disk_free_gib:.1f} GiB')
assert host_mem_gib >= 50, f'High-RAM runtime expected; host RAM is {host_mem_gib:.1f} GiB'
assert disk_free_gib >= 90, f'Need at least 90 GiB free for data audit; found {disk_free_gib:.1f} GiB'
"""
    ),
    code(
        """import importlib.util, subprocess, sys

def pip_install(args):
    print('+ pip install', ' '.join(args))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

pip_install(['--upgrade', 'pip', 'setuptools', 'wheel'])
pip_install(['kaggle==2.0.2', 'pandas', 'numpy'])
print('deps installed')
"""
    ),
    md(
        """## Kaggle Credentials

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab secrets, or place `kaggle.json` in `/content/drive/MyDrive/.kaggle/kaggle.json`.
"""
    ),
    code(
        """import json, os, pathlib, shutil

kaggle_dir = pathlib.Path('/root/.kaggle')
kaggle_dir.mkdir(parents=True, exist_ok=True)
drive_kaggle = pathlib.Path('/content/drive/MyDrive/.kaggle/kaggle.json')
target_kaggle = kaggle_dir / 'kaggle.json'

if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
    target_kaggle.write_text(json.dumps({
        'username': os.environ['KAGGLE_USERNAME'],
        'key': os.environ['KAGGLE_KEY'],
    }), encoding='utf-8')
elif drive_kaggle.exists():
    shutil.copy2(drive_kaggle, target_kaggle)
else:
    raise RuntimeError('Missing Kaggle credentials. Add Colab secrets or /content/drive/MyDrive/.kaggle/kaggle.json')

target_kaggle.chmod(0o600)
print('Kaggle credentials staged at', target_kaggle)
"""
    ),
    code(
        """import hashlib, json, pathlib, subprocess, zipfile

def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_extract_zip(zf: zipfile.ZipFile, dest: pathlib.Path) -> None:
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        member_path = pathlib.PurePosixPath(member.filename.replace('\\\\', '/'))
        if member_path.is_absolute() or '..' in member_path.parts:
            raise RuntimeError(f'Unsafe zip member path: {member.filename}')
        target = (dest / pathlib.Path(*member_path.parts)).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise RuntimeError(f'Zip member escapes output dir: {member.filename}')
    zf.extractall(dest)

def kaggle_download_file(dataset: str, filename: str, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print('download:', dataset, filename)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', dataset,
        '-f', filename,
        '-p', str(out_dir),
    ], check=True)
    direct = out_dir / filename
    if direct.exists():
        return direct
    zips = sorted(out_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f'Downloaded file not found for {dataset}/{filename}')
    with zipfile.ZipFile(zips[0]) as zf:
        safe_extract_zip(zf, out_dir)
    if not direct.exists():
        matches = list(out_dir.rglob(pathlib.Path(filename).name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f'Extracted file not found: {filename}')

downloads = {}
downloads['tong_corpus'] = kaggle_download_file('atahalam/tonghuikang-0-87-nemotron-dataset', 'corpus.jsonl', DATA_DIR / 'tonghuikang_087')
downloads['tong_problems'] = kaggle_download_file('atahalam/tonghuikang-0-87-nemotron-dataset', 'problems.jsonl', DATA_DIR / 'tonghuikang_087')
downloads['tong_generation'] = kaggle_download_file('atahalam/tonghuikang-0-87-nemotron-dataset', 'generation.jsonl', DATA_DIR / 'tonghuikang_087')
downloads['kishan_traj_zip_or_csv'] = kaggle_download_file('kishanvavdara/nemotron-reasoning-traj', 'nemotron_traj.csv', DATA_DIR / 'kishan_traj')

optional = {}
for name in ['bit_manip_3input_synthesized_traces.jsonl', 'bit_manipulation_3input_traces.jsonl']:
    try:
        optional[name] = kaggle_download_file('samvalladares/huikang-nemotron-artifacts', name, DATA_DIR / 'samvalladares_huikang')
    except Exception as exc:
        print('optional download failed:', name, repr(exc))

manifest = {'required': {}, 'optional': {}}
for k, p in downloads.items():
    manifest['required'][k] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_path(p)}
for k, p in optional.items():
    manifest['optional'][k] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_path(p)}

expected_required_sha256 = {
    'tong_corpus': '309264659ba3f668b6b548ca16686d773868cd5bc63349a6721484308341e5c6',
    'tong_problems': '5b536b97b402fab985312003983bf4c59a928eb08dbb2705ca77d1030d4cf24e',
    'tong_generation': '42eb76d13bd81ea3ce6b55120a3e2a23782c18563e05dd4ac9eea59d631b9fbc',
    'kishan_traj_zip_or_csv': '01da9b309daedf18c9bcff9e0766b3deb7d736a1d350c73ded47775a8b66685e',
}
for key, expected in expected_required_sha256.items():
    actual = manifest['required'][key]['sha256']
    if actual != expected:
        raise RuntimeError(f'{key} sha256 mismatch: expected {expected}, got {actual}')

print(json.dumps(manifest, indent=2))
(REPORT_DIR / 'v202_download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
"""
    ),
    code(
        """import csv, json, pathlib
from collections import Counter

def summarize_jsonl(path: pathlib.Path, max_samples=3):
    counts = {}
    numeric = {}
    samples = []
    keys = Counter()
    rows = 0
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if len(samples) < max_samples:
                samples.append({k: str(v)[:300] for k, v in row.items()})
            for k, v in row.items():
                keys[k] += 1
            for k in ['category', 'status', 'included', 'segment']:
                if k in row:
                    counts.setdefault(k, Counter())[str(row[k])] += 1
            for k in ['masked_token_count', 'unmasked_token_count', 'token_count', 'latest_num_gen_tokens']:
                if k in row and isinstance(row[k], (int, float)):
                    numeric.setdefault(k, []).append(row[k])
    def pct(vals, q):
        vals = sorted(vals)
        return vals[min(len(vals)-1, int((len(vals)-1)*q))]
    return {
        'path': str(path),
        'rows': rows,
        'keys': keys.most_common(),
        'counts': {k: v.most_common(40) for k, v in counts.items()},
        'numeric': {
            k: {'count': len(v), 'min': min(v), 'p50': pct(v, .5), 'p90': pct(v, .9), 'p99': pct(v, .99), 'max': max(v)}
            for k, v in numeric.items()
        },
        'samples': samples,
    }

def summarize_csv(path: pathlib.Path, max_samples=3):
    rows = 0
    samples = []
    counts = {'problem type': Counter(), 'correctness': Counter()}
    gen_chars = []
    with path.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            rows += 1
            if len(samples) < max_samples:
                samples.append({k: str(v)[:300] for k, v in row.items()})
            for c in counts:
                if c in row:
                    counts[c][str(row[c])] += 1
            if 'generated' in row:
                gen_chars.append(len(row['generated']))
    def pct(vals, q):
        vals = sorted(vals)
        return vals[min(len(vals)-1, int((len(vals)-1)*q))]
    return {
        'path': str(path),
        'rows': rows,
        'columns': cols,
        'counts': {k: v.most_common(40) for k, v in counts.items()},
        'generated_chars': {
            'count': len(gen_chars),
            'min': min(gen_chars),
            'p50': pct(gen_chars, .5),
            'p90': pct(gen_chars, .9),
            'p99': pct(gen_chars, .99),
            'max': max(gen_chars),
        } if gen_chars else {},
        'samples': samples,
    }

audit = {
    'tong_corpus': summarize_jsonl(downloads['tong_corpus']),
    'tong_problems': summarize_jsonl(downloads['tong_problems']),
    'tong_generation': summarize_jsonl(downloads['tong_generation']),
    'kishan_traj': summarize_csv(downloads['kishan_traj_zip_or_csv']),
}
for k, p in optional.items():
    audit[f'samvalladares_{pathlib.Path(k).stem}'] = summarize_jsonl(p)

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ['samples']} for k, v in audit.items()}, indent=2))
(REPORT_DIR / 'v202_data_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

token_stats = audit['tong_corpus']['numeric'].get('token_count')
if not token_stats:
    found_keys = audit['tong_corpus'].get('keys', [])
    raise RuntimeError(f'tong_corpus missing numeric token_count. Found keys: {found_keys}')
if token_stats['p50'] <= 2048:
    raise RuntimeError(f'Expected long-context corpus p50 > 2048, got {token_stats["p50"]}')
if token_stats['p90'] <= 6000:
    raise RuntimeError(f'Expected long-context corpus p90 > 6000, got {token_stats["p90"]}')
print('Long-context audit passed: this route needs MAX_LENGTH=8192 or explicit distillation.')
"""
    ),
    code(
        """import json, pathlib

v202_plan = {
    'decision': 'do_not_submit_any_v201c_candidate',
    'why_v201c_blocked': [
        {'label': 'A_ultralow_shuffle_1s', 'baseline': 1.1149, 'final': 1.1173, 'delta': 0.0024},
        {'label': 'B_equation_crypt_ultralow_1s', 'baseline': 1.1149, 'final': 1.1182, 'delta': 0.0033},
        {'label': 'C_bit_cipher_ultralow_1s', 'baseline': 1.1160, 'final': 1.1166, 'delta': 0.0006},
    ],
    'immutable_fallback': {
        'label': 'V194 rank19',
        'public_score': 0.86,
        'zip_sha256': V194_ZIP_SHA256,
        'adapter_sha256': V194_ADAPTER_SHA256,
    },
    'next_candidate_family': 'Tong/Huikang long-context reproduction or distilled long-context data, not 2048-token micro-continuation',
    'required_training_contract': {
        'max_length': 8192,
        'base_model': MODEL_NAME,
        'model_revision': MODEL_REVISION,
        'rank_max': 32,
        'future_training_contract': REQUIRED_TRAINING_CONTRACT,
        'no_submit_without_vllm_gate': True,
        'vllm_gate': {
            'max_lora_rank': 32,
            'max_tokens': 7680,
            'max_model_len': 8192,
            'temperature': 0.0,
            'top_p': 1.0,
            'require_net_gain_vs_v194': True,
            'require_zero_anchor_regression': True,
        },
    },
    'hard_reject': [
        'any V201C final_adapter',
        'any candidate with final eval loss above its own baseline',
        'any public adapter not locally gated against V194',
        'any 2048-token training on Tong corpus without distillation',
        'any Kaggle submit from this notebook',
    ],
}
(REPORT_DIR / 'v202_plan_manifest.json').write_text(json.dumps(v202_plan, indent=2), encoding='utf-8')
print(json.dumps(v202_plan, indent=2))
"""
    ),
    code(
        """try:
    target_kaggle.unlink(missing_ok=True)
    print('Removed transient Kaggle credential file:', target_kaggle)
except NameError:
    pass

if RUN_TRAINING:
    raise RuntimeError(
        'RUN_TRAINING=True but this notebook intentionally contains no training cell. '
        'Create the next V202B training notebook only after this audit passes, and enforce REQUIRED_TRAINING_CONTRACT.'
    )

raise RuntimeError(
    'STOPPING BY DESIGN: V202 data audit and manifest are complete. '
    'Review /content/drive/MyDrive/KG1_NVIDIA_V202/reports/v202_data_audit.json and v202_plan_manifest.json. '
    'Next step is a separate V202B training notebook with REQUIRED_TRAINING_CONTRACT enforced.'
)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH)

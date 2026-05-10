#!/usr/bin/env python3
"""Build the V242 safe equation fixture generation Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V242_SAFE_EQUATION_FIXTURES_COLAB.ipynb"
)
DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_REFERENCE_PATH = "runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z/v232_equation_workitems.jsonl"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v242-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return (
        source.replace("__BRANCH__", BRANCH)
        .replace("__COLAB_URL__", COLAB_URL)
        .replace("__GITHUB_URL__", GITHUB_URL)
        .replace("__DEFAULT_HF_DATASET_REPO__", DEFAULT_HF_DATASET_REPO)
        .replace("__DEFAULT_REFERENCE_PATH__", DEFAULT_REFERENCE_PATH)
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
    global _CELL_COUNTER
    _CELL_COUNTER = 0
    cells = [
        md(
            """# KG1 V242 Safe Equation Fixtures Colab

Purpose: generate independent, leakage-guarded synthetic fixtures for the weak `equation_transform` bottleneck before any training is authorized.

This notebook is CPU-only and data-generation-only. It does not train, does not run model inference, does not score models, does not package artifacts, and does not submit to Kaggle.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V242 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V242 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V242 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V242_SAFE_EQUATION_FIXTURES_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V242')
OUT_ROOT = DRIVE_ROOT / 'output_v242_safe_equation_fixtures'
RUN_ID = os.environ.get('KG1_V242_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V242_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v242_safe_equation_fixtures' / RUN_ID

HF_DATASET_REPO = os.environ.get('KG1_V242_HF_DATASET_REPO', '__DEFAULT_HF_DATASET_REPO__').strip()
REFERENCE_PATH_IN_REPO = os.environ.get('KG1_V242_REFERENCE_PATH_IN_REPO', '__DEFAULT_REFERENCE_PATH__').strip()
HF_UPLOAD_PATH_PREFIX = os.environ.get('KG1_V242_HF_UPLOAD_PATH_PREFIX', 'runtime_artifacts/v242_safe_equation_fixtures').strip()
TRAIN_ROWS = int(os.environ.get('KG1_V242_TRAIN_ROWS', '1800'))
VALIDATION_ROWS = int(os.environ.get('KG1_V242_VALIDATION_ROWS', '240'))
SEED = int(os.environ.get('KG1_V242_SEED', '242'))
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V242_EXPECTED_REPO_COMMIT', '').strip()

RUN_GENERATION = os.environ.get('KG1_V242_RUN_GENERATION', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
UPLOAD_TO_HF = os.environ.get('KG1_V242_UPLOAD_TO_HF', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_TRAIN = False
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False
ALLOW_PACKAGE_OUTPUT = False
ALLOW_MODEL_INFERENCE = False
ALLOW_SOURCE_PAYLOAD_DOWNLOAD = False

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('RUN_ID =', RUN_ID, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('HF_DATASET_REPO =', HF_DATASET_REPO, flush=True)
print('REFERENCE_PATH_IN_REPO =', REFERENCE_PATH_IN_REPO, flush=True)
print('HF_UPLOAD_PATH_PREFIX =', HF_UPLOAD_PATH_PREFIX, flush=True)
print('TRAIN_ROWS =', TRAIN_ROWS, flush=True)
print('VALIDATION_ROWS =', VALIDATION_ROWS, flush=True)
print('SEED =', SEED, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('RUN_GENERATION =', RUN_GENERATION, flush=True)
print('UPLOAD_TO_HF =', UPLOAD_TO_HF, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('ALLOW_MODEL_INFERENCE =', ALLOW_MODEL_INFERENCE, flush=True)
print('ALLOW_SOURCE_PAYLOAD_DOWNLOAD =', ALLOW_SOURCE_PAYLOAD_DOWNLOAD, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V242: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V242: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V242 generates fixtures only; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V242 cannot run full scoring.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V242.')
if ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('Packaging is disabled in V242.')
if ALLOW_MODEL_INFERENCE:
    raise RuntimeError('Model inference is disabled in V242.')
if ALLOW_SOURCE_PAYLOAD_DOWNLOAD:
    raise RuntimeError('External source payload download is disabled in V242.')
if TRAIN_ROWS <= 0 or VALIDATION_ROWS <= 0:
    raise RuntimeError('TRAIN_ROWS and VALIDATION_ROWS must be positive.')
print('=== V242 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V242 HELPERS START ===', flush=True)

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_cmd(cmd, cwd=None, log_path=None, check=True, timeout_s=None, env=None):
    cwd = pathlib.Path(cwd or '/content')
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = ' '.join(map(str, cmd))
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', printable, flush=True)
    if timeout_s:
        print('timeout_s =', timeout_s, flush=True)
    if log_path:
        print('log_path =', log_path, flush=True)
    started = time.time()
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
        env=env,
    )
    elapsed = time.time() - started
    if log_path:
        log_path.write_text(proc.stdout or '', encoding='utf-8')
    if proc.stdout:
        print(proc.stdout, end='' if proc.stdout.endswith('\\n') else '\\n', flush=True)
    print('returncode =', proc.returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if proc.returncode and proc.stdout:
        print('command_tail_on_failure =', '\\n'.join(proc.stdout.splitlines()[-80:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError(f'command failed rc={proc.returncode}: {printable}')
    return proc.returncode


def resolve_hf_token():
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        print('hf_token_source = env', flush=True)
        return token
    try:
        from google.colab import userdata
        token = (userdata.get('HF_TOKEN') or '').strip()
        if token:
            print('hf_token_source = colab_secret', flush=True)
            os.environ['HF_TOKEN'] = token
            return token
    except Exception as exc:
        print('hf_token_colab_secret_probe =', type(exc).__name__, flush=True)
    print('hf_token_source = none', flush=True)
    return ''

print('=== V242 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo and validate scripts.
print('=== V242 REPO SETUP START ===', flush=True)
if ROOT.exists():
    print('removing_existing_root =', ROOT, flush=True)
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', timeout_s=300)
rev_proc = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
print('repo_rev_parse_returncode =', rev_proc.returncode, flush=True)
print('repo_rev_parse_output =', rev_proc.stdout.strip(), flush=True)
if rev_proc.returncode:
    raise RuntimeError('git rev-parse failed')
repo_commit = rev_proc.stdout.strip()
print('repo_commit =', repo_commit, flush=True)
if EXPECTED_REPO_COMMIT and repo_commit != EXPECTED_REPO_COMMIT:
    raise RuntimeError('repo commit mismatch: expected ' + EXPECTED_REPO_COMMIT + ', got ' + repo_commit)

compile_targets = [
    ROOT / 'scripts' / 'generate_v242_safe_equation_fixtures.py',
    ROOT / 'scripts' / 'audit_jsonl_overlap.py',
    ROOT / 'scripts' / 'notebook_release_gate.py',
]
for target in compile_targets:
    print('compile_target =', target, 'exists =', target.exists(), flush=True)
    if not target.exists():
        raise FileNotFoundError(target)
    run_cmd([sys.executable, '-m', 'py_compile', str(target)], cwd=ROOT, log_path=OUT_ROOT / ('py_compile_' + target.name + '.log'), timeout_s=120)
run_cmd([sys.executable, str(ROOT / 'scripts' / 'generate_v242_safe_equation_fixtures.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'v242_generator_self_test.log', timeout_s=180)
run_cmd([sys.executable, str(ROOT / 'scripts' / 'audit_jsonl_overlap.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'jsonl_overlap_audit_self_test.log', timeout_s=180)
print('=== V242 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: resolve weak reference artifact from HF dataset.
print('=== V242 REFERENCE RESOLUTION START ===', flush=True)
try:
    import huggingface_hub  # noqa: F401
    print('huggingface_hub_available = True', flush=True)
except ImportError:
    print('huggingface_hub_available = False; installing', flush=True)
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], cwd='/content', log_path=OUT_ROOT / 'pip_install_huggingface_hub.log', timeout_s=300)

from huggingface_hub import hf_hub_download

REFERENCE_DIR = OUT_ROOT / 'reference_artifacts' / RUN_ID
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
token = resolve_hf_token() or None
reference_jsonl = pathlib.Path(hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type='dataset',
    filename=REFERENCE_PATH_IN_REPO,
    local_dir=str(REFERENCE_DIR),
    token=token,
))
print('reference_jsonl =', reference_jsonl, flush=True)
print('reference_exists =', reference_jsonl.exists(), flush=True)
if not reference_jsonl.exists():
    raise FileNotFoundError(reference_jsonl)
print('reference_sha256 =', sha256_file(reference_jsonl), flush=True)
print('reference_rows =', sum(1 for line in reference_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()), flush=True)
print('=== V242 REFERENCE RESOLUTION END ===', flush=True)
"""
        ),
        code(
            """# CELL: generate V242 safe equation fixtures and overlap audit.
print('=== V242 FIXTURE GENERATION START ===', flush=True)
if RUN_GENERATION:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'generate_v242_safe_equation_fixtures.py'),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v242_safe_equation_fixtures',
        '--train-rows', str(TRAIN_ROWS),
        '--validation-rows', str(VALIDATION_ROWS),
        '--seed', str(SEED),
        '--reference-jsonl', str(reference_jsonl),
    ]
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v242_safe_equation_fixtures.log', check=True, timeout_s=300)
else:
    print('RUN_GENERATION is false; skipping fixture generation.', flush=True)

manifest_path = ANALYSIS_OUT / 'v242_safe_equation_fixtures_manifest.json'
print('manifest_path =', manifest_path, flush=True)
if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)
manifest = read_json(manifest_path)
print('manifest_sha256 =', sha256_file(manifest_path), flush=True)
print('counts =', json.dumps(manifest.get('counts', {}), indent=2, sort_keys=True), flush=True)
print('overlap_audit =', json.dumps(manifest.get('overlap_audit', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)

for key in ['train_jsonl', 'validation_jsonl', 'rule_summary_csv']:
    path = pathlib.Path(manifest['outputs'][key])
    print(key, '=', path, 'exists =', path.exists(), 'sha256 =', sha256_file(path), flush=True)

overlap_json = ANALYSIS_OUT / 'v242_overlap_audit_repeat.json'
run_cmd([
    sys.executable,
    str(ROOT / 'scripts' / 'audit_jsonl_overlap.py'),
    '--reference-jsonl', str(reference_jsonl),
    '--candidate-jsonl', manifest['outputs']['train_jsonl'],
    '--candidate-jsonl', manifest['outputs']['validation_jsonl'],
    '--output-json', str(overlap_json),
    '--fail-on-id-overlap',
    '--fail-on-prompt-overlap',
], cwd=ROOT, log_path=ANALYSIS_OUT / 'v242_overlap_audit_repeat.log', check=True, timeout_s=180)
print('overlap_json =', overlap_json, flush=True)
print('overlap_json_sha256 =', sha256_file(overlap_json), flush=True)
print('=== V242 FIXTURE GENERATION END ===', flush=True)
"""
        ),
        code(
            """# CELL: optional HF upload and final manifest.
print('=== V242 FINAL MANIFEST START ===', flush=True)
upload_info = 'skipped'
hf_path_in_repo = ''
if UPLOAD_TO_HF:
    token = resolve_hf_token()
    if not token:
        raise RuntimeError('UPLOAD_TO_HF requires HF_TOKEN in env or Colab Secrets.')
    from huggingface_hub import HfApi
    hf_path_in_repo = '/'.join(part.strip('/') for part in [HF_UPLOAD_PATH_PREFIX, RUN_ID] if part.strip('/'))
    print('hf_upload_path_in_repo =', hf_path_in_repo, flush=True)
    api = HfApi(token=token)
    upload_info = str(api.upload_folder(
        repo_id=HF_DATASET_REPO,
        repo_type='dataset',
        folder_path=str(ANALYSIS_OUT),
        path_in_repo=hf_path_in_repo,
        commit_message='Upload KG1 V242 safe equation fixtures ' + hf_path_in_repo,
    ))
    print('hf_upload_info =', upload_info, flush=True)
else:
    print('UPLOAD_TO_HF is false; keeping outputs in Drive only.', flush=True)

final_manifest = {
    'version': VERSION,
    'repo_commit': repo_commit,
    'run_id': RUN_ID,
    'analysis_manifest_json': str(manifest_path),
    'analysis_manifest_sha256': sha256_file(manifest_path),
    'counts': manifest.get('counts', {}),
    'overlap_audit': manifest.get('overlap_audit', {}),
    'decision': manifest.get('decision', {}),
    'hf_upload_info': upload_info,
    'hf_path_in_repo': hf_path_in_repo,
    'blocked_actions': ['train', 'model_inference', 'full_scoring', 'package', 'kaggle_submit'],
}
final_manifest_path = OUT_ROOT / 'v242_safe_equation_fixtures_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest_sha256 =', sha256_file(final_manifest_path), flush=True)
print('final_decision =', json.dumps(final_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('=== V242 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": NOTEBOOK_PATH.name},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)
    print(COLAB_URL)


if __name__ == "__main__":
    main()

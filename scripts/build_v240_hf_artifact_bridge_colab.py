#!/usr/bin/env python3
"""Build the V240 HF artifact bridge Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V240_HF_ARTIFACT_BRIDGE_COLAB.ipynb"
)
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v240-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return (
        source.replace("__BRANCH__", BRANCH)
        .replace("__COLAB_URL__", COLAB_URL)
        .replace("__GITHUB_URL__", GITHUB_URL)
        .replace("__EXPECTED_SHARED_ROW_CONTRACT_SHA256__", EXPECTED_SHARED_ROW_CONTRACT_SHA256)
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
            """# KG1 V240 HF Artifact Bridge Colab

Purpose: upload V232/V238 runtime diagnostic artifacts from Google Drive to a private Hugging Face dataset so future analysis jobs can run on HF without manual Colab work.

This notebook is CPU-only and bridge-only. It does not train, does not run model generation, does not run scoring, does not package artifacts, and does not submit to Kaggle.

Required one-time setup: Colab secret or environment variable named `HF_TOKEN` with write access to the target dataset.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V240 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V240 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V240 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V240_HF_ARTIFACT_BRIDGE_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V240')
OUT_ROOT = DRIVE_ROOT / 'output_v240_hf_artifact_bridge'
RUN_ID = os.environ.get('KG1_V240_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V240_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v240_hf_artifact_bridge' / RUN_ID

V232_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V240_V232_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V232/output_v232_verified_solver_workbench',
))
V238_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V240_V238_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V238/output_v238_alice_parser_probes',
))
V232_MANIFEST_TEXT = os.environ.get('KG1_V240_V232_MANIFEST_JSON', '').strip()
V238_MANIFEST_TEXT = os.environ.get('KG1_V240_V238_MANIFEST_JSON', '').strip()
if V232_MANIFEST_TEXT in {'.', './'}:
    print('V232_MANIFEST_TEXT ignored_directory_placeholder =', V232_MANIFEST_TEXT, flush=True)
    V232_MANIFEST_TEXT = ''
if V238_MANIFEST_TEXT in {'.', './'}:
    print('V238_MANIFEST_TEXT ignored_directory_placeholder =', V238_MANIFEST_TEXT, flush=True)
    V238_MANIFEST_TEXT = ''
V232_MANIFEST_JSON = pathlib.Path(V232_MANIFEST_TEXT) if V232_MANIFEST_TEXT else None
V238_MANIFEST_JSON = pathlib.Path(V238_MANIFEST_TEXT) if V238_MANIFEST_TEXT else None

HF_DATASET_REPO = os.environ.get('KG1_V240_HF_DATASET_REPO', 'felipesp1983/kg1-nemotron-training').strip()
HF_PATH_PREFIX = os.environ.get('KG1_V240_HF_PATH_PREFIX', 'runtime_artifacts/v240_hf_bridge').strip()
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = os.environ.get(
    'KG1_V240_EXPECTED_SHARED_ROW_CONTRACT_SHA256',
    '__EXPECTED_SHARED_ROW_CONTRACT_SHA256__',
).strip()
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V240_EXPECTED_REPO_COMMIT', '').strip()
DRY_RUN = os.environ.get('KG1_V240_DRY_RUN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}

RUN_ANALYSIS = os.environ.get('KG1_V240_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_TRAIN = False
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False
ALLOW_PACKAGE_OUTPUT = False
ALLOW_MODEL_GENERATION = False
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
print('V232_OUTPUT_ROOT =', V232_OUTPUT_ROOT, flush=True)
print('V238_OUTPUT_ROOT =', V238_OUTPUT_ROOT, flush=True)
print('V232_MANIFEST_JSON =', V232_MANIFEST_JSON or '', flush=True)
print('V238_MANIFEST_JSON =', V238_MANIFEST_JSON or '', flush=True)
print('HF_DATASET_REPO =', HF_DATASET_REPO, flush=True)
print('HF_PATH_PREFIX =', HF_PATH_PREFIX, flush=True)
print('EXPECTED_SHARED_ROW_CONTRACT_SHA256 =', EXPECTED_SHARED_ROW_CONTRACT_SHA256, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('DRY_RUN =', DRY_RUN, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('ALLOW_MODEL_GENERATION =', ALLOW_MODEL_GENERATION, flush=True)
print('ALLOW_SOURCE_PAYLOAD_DOWNLOAD =', ALLOW_SOURCE_PAYLOAD_DOWNLOAD, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V240: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V240: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V240 is artifact bridge only; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V240 cannot run full scoring.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V240.')
if ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('Packaging is disabled in V240.')
if ALLOW_MODEL_GENERATION:
    raise RuntimeError('Model generation is disabled in V240.')
if ALLOW_SOURCE_PAYLOAD_DOWNLOAD:
    raise RuntimeError('External payload download is disabled in V240.')
if not EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V240 requires KG1_V240_EXPECTED_SHARED_ROW_CONTRACT_SHA256.')
print('=== V240 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging and token resolution.
print('=== V240 HELPERS START ===', flush=True)

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
    printable = ' '.join('[REDACTED]' if 'hf_' in str(part).lower() and len(str(part)) > 20 else str(part) for part in cmd)
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


def resolve_latest_manifest(explicit_path, search_root, pattern, label):
    if explicit_path is not None:
        print(label + '_manifest_explicit =', explicit_path, flush=True)
        if not explicit_path.exists():
            raise FileNotFoundError(explicit_path)
        if not explicit_path.is_file():
            raise IsADirectoryError(label + ' manifest must be a JSON file: ' + str(explicit_path))
        return explicit_path
    print(label + '_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
    candidates = sorted(search_root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    print(label + '_manifest_candidate_count =', len(candidates), flush=True)
    for candidate in candidates[:10]:
        print(label + '_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
    if not candidates:
        raise FileNotFoundError('No ' + label + ' manifest found under: ' + str(search_root))
    return candidates[0]


def resolve_hf_token():
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        print('hf_token_source = environment', flush=True)
        return token
    try:
        from google.colab import userdata
        token = str(userdata.get('HF_TOKEN') or '').strip()
        if token:
            print('hf_token_source = colab_userdata', flush=True)
            return token
    except Exception as exc:
        print('hf_token_colab_userdata_warning =', repr(exc), flush=True)
    if DRY_RUN:
        print('hf_token_source = absent_but_dry_run', flush=True)
        return ''
    raise RuntimeError('HF_TOKEN is required. Add a Colab Secret named HF_TOKEN with write access to ' + HF_DATASET_REPO)


print('=== V240 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, install dependency, compile scripts, and run self-tests.
print('=== V240 REPO SETUP START ===', flush=True)
if ROOT.exists():
    print('removing_existing_root =', ROOT, flush=True)
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', timeout_s=300)
sys.path.insert(0, str(ROOT))
print('repo_root_on_sys_path =', sys.path[0], flush=True)

repo_commit_proc = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
repo_commit = repo_commit_proc.stdout.strip()
print('repo_rev_parse_returncode =', repo_commit_proc.returncode, flush=True)
print('repo_rev_parse_output =', repo_commit, flush=True)
if repo_commit_proc.returncode:
    raise RuntimeError('git rev-parse failed')
if EXPECTED_REPO_COMMIT and repo_commit != EXPECTED_REPO_COMMIT:
    raise RuntimeError('repo commit mismatch: expected ' + EXPECTED_REPO_COMMIT + ', got ' + repo_commit)
print('repo_commit =', repo_commit, flush=True)

run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=0.23.0'], cwd=ROOT, log_path=OUT_ROOT / 'pip_install_huggingface_hub.log', timeout_s=300)

compile_targets = [
    ROOT / 'scripts' / 'upload_runtime_artifacts_to_hf.py',
    ROOT / 'scripts' / 'run_v239_from_hf_bridge.py',
    ROOT / 'scripts' / 'analyze_v239_alice_abstain_mining.py',
    ROOT / 'scripts' / 'notebook_release_gate.py',
]
for target in compile_targets:
    print('compile_target =', target, 'exists =', target.exists(), flush=True)
    if not target.exists():
        raise FileNotFoundError(target)
    py_compile_result = subprocess.run([sys.executable, '-m', 'py_compile', str(target)], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if py_compile_result.stdout:
        print(py_compile_result.stdout, flush=True)
    print('py_compile_returncode =', py_compile_result.returncode, flush=True)
    if py_compile_result.returncode:
        raise RuntimeError('py_compile failed for ' + str(target))
    print('py_compile ok =', target.relative_to(ROOT), flush=True)

run_cmd([sys.executable, str(ROOT / 'scripts' / 'upload_runtime_artifacts_to_hf.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'v240_hf_artifact_bridge_self_test.log', timeout_s=180)
run_cmd([sys.executable, str(ROOT / 'scripts' / 'run_v239_from_hf_bridge.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'v239_hf_bridge_runner_self_test.log', timeout_s=180)
print('=== V240 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V232 and V238 artifacts.
print('=== V240 ARTIFACT PREFLIGHT START ===', flush=True)
resolved_v232_manifest = resolve_latest_manifest(
    V232_MANIFEST_JSON,
    V232_OUTPUT_ROOT / 'analysis_v232_verified_solver_workbench',
    '*/v232_verified_solver_workbench_manifest.json',
    'v232',
)
resolved_v238_manifest = resolve_latest_manifest(
    V238_MANIFEST_JSON,
    V238_OUTPUT_ROOT / 'analysis_v238_alice_parser_probes',
    '*/v238_alice_parser_probes_manifest.json',
    'v238',
)
print('resolved_v232_manifest =', resolved_v232_manifest, flush=True)
print('resolved_v238_manifest =', resolved_v238_manifest, flush=True)
for label, manifest_path in [('v232', resolved_v232_manifest), ('v238', resolved_v238_manifest)]:
    print(label + '_manifest_exists =', manifest_path.exists(), flush=True)
    print(label + '_manifest_is_file =', manifest_path.is_file(), flush=True)
    print(label + '_manifest_sha256 =', sha256_file(manifest_path), flush=True)
    manifest = read_json(manifest_path)
    print(label + '_manifest_keys =', sorted(manifest.keys()), flush=True)
    outputs = manifest.get('outputs', {})
    print(label + '_outputs =', json.dumps(outputs, indent=2, sort_keys=True), flush=True)
print('=== V240 ARTIFACT PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: upload runtime artifacts to HF dataset.
print('=== V240 HF ARTIFACT UPLOAD START ===', flush=True)
hf_token = resolve_hf_token()
env = os.environ.copy()
if hf_token:
    env['HF_TOKEN'] = hf_token
output_manifest_json = ANALYSIS_OUT / 'v240_hf_artifact_bridge_manifest.json'
cmd = [
    sys.executable,
    str(ROOT / 'scripts' / 'upload_runtime_artifacts_to_hf.py'),
    '--v232-manifest-json', str(resolved_v232_manifest),
    '--v238-manifest-json', str(resolved_v238_manifest),
    '--hf-dataset-repo', HF_DATASET_REPO,
    '--path-prefix', HF_PATH_PREFIX,
    '--run-id', RUN_ID,
    '--expected-shared-row-contract-sha256', EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    '--output-manifest-json', str(output_manifest_json),
]
if DRY_RUN:
    cmd.append('--dry-run')
run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v240_hf_artifact_bridge_upload.log', check=True, timeout_s=600, env=env)
bridge_manifest = read_json(output_manifest_json)
print('bridge_manifest_path =', output_manifest_json, flush=True)
print('bridge_manifest_sha256 =', sha256_file(output_manifest_json), flush=True)
print('bridge_repo_id =', bridge_manifest.get('repo_id'), flush=True)
print('bridge_path_in_repo =', bridge_manifest.get('path_in_repo'), flush=True)
print('bridge_upload_info =', bridge_manifest.get('upload_info'), flush=True)
print('bridge_files =', json.dumps(bridge_manifest.get('files', []), indent=2, sort_keys=True), flush=True)
print('=== V240 HF ARTIFACT UPLOAD END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V240 FINAL MANIFEST START ===', flush=True)
final_manifest = {
    'schema_version': 'kg1_v240_colab_final_manifest_v1',
    'version': VERSION,
    'repo_url': REPO_URL,
    'repo_branch': REPO_BRANCH,
    'repo_commit': repo_commit,
    'run_id': RUN_ID,
    'out_root': str(OUT_ROOT),
    'analysis_out': str(ANALYSIS_OUT),
    'bridge_manifest_path': str(output_manifest_json),
    'bridge_manifest_exists': output_manifest_json.exists(),
    'bridge_manifest_sha256': sha256_file(output_manifest_json) if output_manifest_json.exists() else '',
    'hf_dataset_repo': HF_DATASET_REPO,
    'hf_path_prefix': HF_PATH_PREFIX,
    'hf_bridge_path_in_repo': bridge_manifest.get('path_in_repo', '') if 'bridge_manifest' in globals() else '',
    'dry_run': DRY_RUN,
    'blocked_actions': ['train', 'model_generation', 'full_scoring', 'package', 'kaggle_submit', 'external_payload_download'],
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
    'allow_package_output': ALLOW_PACKAGE_OUTPUT,
    'allow_model_generation': ALLOW_MODEL_GENERATION,
    'allow_source_payload_download': ALLOW_SOURCE_PAYLOAD_DOWNLOAD,
}
final_manifest_path = OUT_ROOT / 'v240_hf_artifact_bridge_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('No package and no Kaggle submit can be created in V240.', flush=True)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest_sha256 =', sha256_file(final_manifest_path), flush=True)
print('hf_bridge_path_in_repo =', final_manifest.get('hf_bridge_path_in_repo'), flush=True)
print('=== V240 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")
    print(f"colab_url={COLAB_URL}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the V235 source access triage Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V235_SOURCE_ACCESS_TRIAGE_COLAB.ipynb"
)

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v235-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return source.replace("__BRANCH__", BRANCH).replace("__COLAB_URL__", COLAB_URL).replace("__GITHUB_URL__", GITHUB_URL)


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
            """# KG1 V235 Source Access Triage Colab

Purpose: consume the executed V234 external-intelligence manifest and build a controlled source-access, hash, and license plan before any source payload download, solver implementation, model generation, scoring, package, or Kaggle submit.

This notebook is CPU-only. It validates V234 outputs, audits credentials without printing secrets, optionally checks Hugging Face public metadata, and emits `source_access_inventory.csv`, `hf_metadata_audit.csv`, `kaggle_access_audit.csv`, `source_download_plan.csv`, and `license_gate_report.json`.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V235 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V235 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V235 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V235_SOURCE_ACCESS_TRIAGE_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V235')
OUT_ROOT = DRIVE_ROOT / 'output_v235_source_access_triage'
RUN_ID = os.environ.get('KG1_V235_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V235_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v235_source_access_triage' / RUN_ID

V234_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V235_V234_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V234/output_v234_external_intel_triage',
))
V234_ANALYSIS_MANIFEST_JSON_TEXT = os.environ.get('KG1_V235_V234_ANALYSIS_MANIFEST_JSON', '').strip()
if V234_ANALYSIS_MANIFEST_JSON_TEXT in {'.', './'}:
    print('V234_ANALYSIS_MANIFEST_JSON_TEXT ignored_directory_placeholder =', V234_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
    V234_ANALYSIS_MANIFEST_JSON_TEXT = ''
V234_ANALYSIS_MANIFEST_JSON = pathlib.Path(V234_ANALYSIS_MANIFEST_JSON_TEXT) if V234_ANALYSIS_MANIFEST_JSON_TEXT else None
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V235_EXPECTED_REPO_COMMIT', '').strip()

RUN_ANALYSIS = os.environ.get('KG1_V235_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
ENABLE_NETWORK_METADATA = os.environ.get('KG1_V235_ENABLE_NETWORK_METADATA', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
NETWORK_TIMEOUT_S = int(os.environ.get('KG1_V235_NETWORK_TIMEOUT_S', '20'))
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
print('V234_OUTPUT_ROOT =', V234_OUTPUT_ROOT, flush=True)
print('V234_ANALYSIS_MANIFEST_JSON_TEXT =', V234_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
print('V234_ANALYSIS_MANIFEST_JSON =', V234_ANALYSIS_MANIFEST_JSON or '', flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('ENABLE_NETWORK_METADATA =', ENABLE_NETWORK_METADATA, flush=True)
print('NETWORK_TIMEOUT_S =', NETWORK_TIMEOUT_S, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('ALLOW_MODEL_GENERATION =', ALLOW_MODEL_GENERATION, flush=True)
print('ALLOW_SOURCE_PAYLOAD_DOWNLOAD =', ALLOW_SOURCE_PAYLOAD_DOWNLOAD, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V235: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V235: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V235 is CPU-only source access triage; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V235 cannot run scoring.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V235.')
if ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('Packaging is disabled in V235.')
if ALLOW_MODEL_GENERATION:
    raise RuntimeError('Model generation is disabled in V235.')
if ALLOW_SOURCE_PAYLOAD_DOWNLOAD:
    raise RuntimeError('Source payload download is disabled in V235; this notebook only creates the guarded plan.')
print('=== V235 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V235 HELPERS START ===', flush=True)

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path):
    with pathlib.Path(path).open('r', encoding='utf-8', newline='') as handle:
        return max(0, sum(1 for _ in handle) - 1)


def run_cmd(cmd, cwd=None, log_path=None, check=True, timeout_s=None):
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


def resolve_latest_v234_manifest():
    if V234_ANALYSIS_MANIFEST_JSON is not None:
        print('v234_manifest_explicit =', V234_ANALYSIS_MANIFEST_JSON, flush=True)
        if not V234_ANALYSIS_MANIFEST_JSON.exists():
            raise FileNotFoundError(V234_ANALYSIS_MANIFEST_JSON)
        if not V234_ANALYSIS_MANIFEST_JSON.is_file():
            raise IsADirectoryError('KG1_V235_V234_ANALYSIS_MANIFEST_JSON must point to a JSON file, got: ' + str(V234_ANALYSIS_MANIFEST_JSON))
        return V234_ANALYSIS_MANIFEST_JSON
    search_root = V234_OUTPUT_ROOT / 'analysis_v234_external_intel_triage'
    print('v234_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
    candidates = sorted(search_root.glob('*/v234_external_intel_triage_manifest.json'), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    print('v234_manifest_candidate_count =', len(candidates), flush=True)
    for candidate in candidates[:10]:
        print('v234_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
    if not candidates:
        raise FileNotFoundError('No V234 manifest found under: ' + str(search_root))
    return candidates[0]


print('=== V235 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and run self-test.
print('=== V235 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True, timeout_s=300)
if EXPECTED_REPO_COMMIT:
    run_cmd(['git', 'fetch', '--depth', '1', 'origin', EXPECTED_REPO_COMMIT], cwd=ROOT, log_path=OUT_ROOT / 'repo_fetch_expected_commit.log', check=True, timeout_s=300)
    run_cmd(['git', 'checkout', '--detach', EXPECTED_REPO_COMMIT], cwd=ROOT, log_path=OUT_ROOT / 'repo_checkout_expected_commit.log', check=True, timeout_s=120)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
print('repo_commit =', repo_commit, flush=True)
if EXPECTED_REPO_COMMIT and repo_commit != EXPECTED_REPO_COMMIT:
    raise RuntimeError('repo commit mismatch')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('repo_root_on_sys_path =', str(ROOT) in sys.path, flush=True)

required_scripts = [
    ROOT / 'scripts/analyze_v235_source_access_triage.py',
    ROOT / 'scripts/notebook_release_gate.py',
]
for py_path in required_scripts:
    print('compile_target =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)
    print('py_compile ok =', py_path.relative_to(ROOT), flush=True)

run_cmd(
    [sys.executable, str(ROOT / 'scripts/analyze_v235_source_access_triage.py'), '--self-test', '--v234-analysis-manifest-json', 'dummy', '--output-dir', str(OUT_ROOT / 'self_test_dummy')],
    cwd=ROOT,
    log_path=OUT_ROOT / 'v235_source_access_triage_self_test.log',
    check=True,
    timeout_s=180,
)
print('=== V235 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V234 triage artifacts.
print('=== V235 V234 ARTIFACT PREFLIGHT START ===', flush=True)
resolved_v234_manifest = resolve_latest_v234_manifest()
print('resolved_v234_manifest =', resolved_v234_manifest, flush=True)
print('resolved_v234_manifest_exists =', resolved_v234_manifest.exists(), flush=True)
print('resolved_v234_manifest_is_file =', resolved_v234_manifest.is_file(), flush=True)
if not resolved_v234_manifest.exists():
    raise FileNotFoundError(resolved_v234_manifest)
if not resolved_v234_manifest.is_file():
    raise IsADirectoryError('V234 manifest must be a JSON file, got: ' + str(resolved_v234_manifest))
v234_manifest = read_json(resolved_v234_manifest)
coverage = v234_manifest.get('coverage', {})
metric_parity = v234_manifest.get('metric_parity', {})
print('v234_coverage =', json.dumps(coverage, indent=2, sort_keys=True), flush=True)
print('v234_metric_parity =', json.dumps(metric_parity, indent=2, sort_keys=True), flush=True)
if not coverage.get('passed'):
    raise RuntimeError('V234 coverage did not pass.')
if coverage.get('missing_refs') or coverage.get('refs_without_action_path'):
    raise RuntimeError('V234 coverage has unresolved refs.')
if not metric_parity.get('passed'):
    raise RuntimeError('V234 metric parity did not pass.')
required_outputs = [
    'external_metric_parity_report_json',
    'kaggle_kernel_triage_csv',
    'kaggle_dataset_triage_csv',
    'hf_dataset_triage_csv',
    'kaggle_model_triage_csv',
    'equation_numeric_operator_probe_results_csv',
    'bit_boolean_function_probe_results_csv',
    'external_adapter_registry_candidates_csv',
]
for name in required_outputs:
    path = pathlib.Path(str(v234_manifest.get('outputs', {}).get(name, '')))
    print('v234_output_artifact =', name, path, 'exists =', path.exists(), 'is_file =', path.is_file(), flush=True)
    if not path.exists():
        raise FileNotFoundError(name + ': ' + str(path))
    if not path.is_file():
        raise IsADirectoryError(name + ': ' + str(path))
    rows = csv_row_count(path) if path.suffix == '.csv' else None
    print('v234_output_artifact_meta =', json.dumps({'name': name, 'rows': rows, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V235 V234 ARTIFACT PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V235 source access triage.
print('=== V235 SOURCE ACCESS TRIAGE START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v235_source_access_triage_manifest.json'
if RUN_ANALYSIS:
    resolved_candidate_text = str(globals().get('resolved_v234_manifest', '')).strip()
    if resolved_candidate_text in {'', '.', './'} or not pathlib.Path(resolved_candidate_text).is_file():
        print('resolved_v234_manifest missing or invalid before source triage; resolving again.', flush=True)
        resolved_v234_manifest = resolve_latest_v234_manifest()
    else:
        resolved_v234_manifest = pathlib.Path(resolved_candidate_text)
    print('source_triage_v234_manifest =', resolved_v234_manifest, flush=True)
    print('source_triage_v234_manifest_exists =', resolved_v234_manifest.exists(), flush=True)
    print('source_triage_v234_manifest_is_file =', resolved_v234_manifest.is_file(), flush=True)
    if not resolved_v234_manifest.exists():
        raise FileNotFoundError(resolved_v234_manifest)
    if not resolved_v234_manifest.is_file():
        raise IsADirectoryError('V235 source triage requires V234 manifest JSON file, got: ' + str(resolved_v234_manifest))
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v235_source_access_triage.py'),
        '--v234-analysis-manifest-json', str(resolved_v234_manifest),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v235_source_access_triage',
        '--network-timeout-s', str(NETWORK_TIMEOUT_S),
    ]
    if ENABLE_NETWORK_METADATA:
        cmd.append('--enable-network-metadata')
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v235_source_access_triage.log', check=True, timeout_s=420)
else:
    print('RUN_ANALYSIS is false; skipping V235 source access command.', flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
print('summary =', json.dumps(analysis_manifest.get('summary', {}), indent=2, sort_keys=True), flush=True)
print('license_gate =', json.dumps(analysis_manifest.get('license_gate', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
required_outputs = [
    'source_access_inventory_csv',
    'hf_metadata_audit_csv',
    'kaggle_access_audit_csv',
    'source_download_plan_csv',
    'license_gate_report_json',
]
for name in required_outputs:
    path = pathlib.Path(str(analysis_manifest.get('outputs', {}).get(name, '')))
    print('v235_output_artifact =', name, path, 'exists =', path.exists(), 'is_file =', path.is_file(), flush=True)
    if not path.exists():
        raise FileNotFoundError(name + ': ' + str(path))
    if not path.is_file():
        raise IsADirectoryError(name + ': ' + str(path))
    rows = csv_row_count(path) if path.suffix == '.csv' else None
    print('v235_output_artifact_meta =', json.dumps({'name': name, 'rows': rows, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V235 SOURCE ACCESS TRIAGE END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V235 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
blocked_artifacts = []
for pattern in ['*.zip', '*submission*', '*kaggle*submit*', '*.safetensors', '*.bin', '*.pt']:
    blocked_artifacts.extend(str(path) for path in OUT_ROOT.rglob(pattern))
print('blocked_artifacts =', json.dumps(blocked_artifacts, indent=2, sort_keys=True), flush=True)
if blocked_artifacts:
    raise RuntimeError('V235 output contains package/model/submission-like artifacts: ' + json.dumps(blocked_artifacts, sort_keys=True))
print('Payload download, scoring, package, and Kaggle submit are intentionally not automatic in V235.', flush=True)
print('No package and no Kaggle submit can be created in V235.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT or ALLOW_PACKAGE_OUTPUT or ALLOW_MODEL_GENERATION or ALLOW_SOURCE_PAYLOAD_DOWNLOAD:
    raise RuntimeError('V235 hard block violated.')
final_manifest = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_id': RUN_ID,
    'v234_manifest': str(resolved_v234_manifest),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_manifest_sha256': sha256_file(analysis_manifest_path),
    'summary': analysis_manifest.get('summary', {}),
    'license_gate': analysis_manifest.get('license_gate', {}),
    'decision': analysis_manifest.get('decision', {}),
    'outputs': analysis_manifest.get('outputs', {}),
    'allowed_actions': ['review_source_access_plan', 'resolve_credentials_and_license_metadata', 'build_payload_downloader_after_gate'],
    'blocked_actions': ['payload_download_without_license_hash', 'train', 'model_generation', 'scoring', 'package', 'kaggle_submit'],
    'roadmap_next': analysis_manifest.get('decision', {}).get('next_action', 'Review V235 source access outputs.'),
}
final_manifest_path = OUT_ROOT / 'v235_source_access_triage_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('=== V235 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
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

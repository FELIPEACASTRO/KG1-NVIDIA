#!/usr/bin/env python3
"""Build the V237 prompt format audit Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V237_PROMPT_FORMAT_AUDIT_COLAB.ipynb"
)
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v237-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V237 Prompt Format Audit Colab

Purpose: audit the real prompt formats inside V232 solver workitems after V236 showed `0` deployable equation overrides.

This notebook is CPU-only and diagnostic-only. It does not train, does not run model generation, does not run scoring, does not package artifacts, does not download external payloads, and does not submit to Kaggle.

Primary outputs: `prompt_format_audit_csv`, `prompt_format_summary_csv`, `equation_prompt_sample_csv`, and `manifest_json`.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V237 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V237 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V237 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V237_PROMPT_FORMAT_AUDIT_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V237')
OUT_ROOT = DRIVE_ROOT / 'output_v237_prompt_format_audit'
RUN_ID = os.environ.get('KG1_V237_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V237_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v237_prompt_format_audit' / RUN_ID

V232_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V237_V232_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V232/output_v232_verified_solver_workbench',
))
V232_ANALYSIS_MANIFEST_JSON_TEXT = os.environ.get('KG1_V237_V232_ANALYSIS_MANIFEST_JSON', '').strip()
if V232_ANALYSIS_MANIFEST_JSON_TEXT in {'.', './'}:
    print('V232_ANALYSIS_MANIFEST_JSON_TEXT ignored_directory_placeholder =', V232_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
    V232_ANALYSIS_MANIFEST_JSON_TEXT = ''
V232_ANALYSIS_MANIFEST_JSON = pathlib.Path(V232_ANALYSIS_MANIFEST_JSON_TEXT) if V232_ANALYSIS_MANIFEST_JSON_TEXT else None
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = os.environ.get(
    'KG1_V237_EXPECTED_SHARED_ROW_CONTRACT_SHA256',
    '__EXPECTED_SHARED_ROW_CONTRACT_SHA256__',
).strip()
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V237_EXPECTED_REPO_COMMIT', '').strip()
SAMPLE_LIMIT = int(os.environ.get('KG1_V237_SAMPLE_LIMIT', '60'))
PREVIEW_LIMIT = int(os.environ.get('KG1_V237_PREVIEW_LIMIT', '20'))

RUN_ANALYSIS = os.environ.get('KG1_V237_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
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
print('V232_ANALYSIS_MANIFEST_JSON_TEXT =', V232_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
print('V232_ANALYSIS_MANIFEST_JSON =', V232_ANALYSIS_MANIFEST_JSON or '', flush=True)
print('EXPECTED_SHARED_ROW_CONTRACT_SHA256 =', EXPECTED_SHARED_ROW_CONTRACT_SHA256, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('SAMPLE_LIMIT =', SAMPLE_LIMIT, flush=True)
print('PREVIEW_LIMIT =', PREVIEW_LIMIT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('ALLOW_MODEL_GENERATION =', ALLOW_MODEL_GENERATION, flush=True)
print('ALLOW_SOURCE_PAYLOAD_DOWNLOAD =', ALLOW_SOURCE_PAYLOAD_DOWNLOAD, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V237: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V237: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V237 is CPU-only prompt audit; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V237 cannot run full scoring.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V237.')
if ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('Packaging is disabled in V237.')
if ALLOW_MODEL_GENERATION:
    raise RuntimeError('Model generation is disabled in V237.')
if ALLOW_SOURCE_PAYLOAD_DOWNLOAD:
    raise RuntimeError('External payload download is disabled in V237.')
if not EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V237 requires KG1_V237_EXPECTED_SHARED_ROW_CONTRACT_SHA256.')
print('=== V237 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V237 HELPERS START ===', flush=True)

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


def resolve_latest_v232_manifest():
    if V232_ANALYSIS_MANIFEST_JSON is not None:
        print('v232_manifest_explicit =', V232_ANALYSIS_MANIFEST_JSON, flush=True)
        if not V232_ANALYSIS_MANIFEST_JSON.exists():
            raise FileNotFoundError(V232_ANALYSIS_MANIFEST_JSON)
        if not V232_ANALYSIS_MANIFEST_JSON.is_file():
            raise IsADirectoryError('KG1_V237_V232_ANALYSIS_MANIFEST_JSON must point to a JSON file, got: ' + str(V232_ANALYSIS_MANIFEST_JSON))
        return V232_ANALYSIS_MANIFEST_JSON
    search_root = V232_OUTPUT_ROOT / 'analysis_v232_verified_solver_workbench'
    print('v232_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
    candidates = sorted(search_root.glob('*/v232_verified_solver_workbench_manifest.json'), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    print('v232_manifest_candidate_count =', len(candidates), flush=True)
    for candidate in candidates[:10]:
        print('v232_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
    if not candidates:
        raise FileNotFoundError('No V232 manifest found under: ' + str(search_root))
    return candidates[0]


print('=== V237 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and run self-test.
print('=== V237 REPO SETUP START ===', flush=True)
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

compile_targets = [
    ROOT / 'src' / 'competition_utils.py',
    ROOT / 'scripts' / 'analyze_v237_prompt_format_audit.py',
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

run_cmd([sys.executable, str(ROOT / 'scripts' / 'analyze_v237_prompt_format_audit.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'v237_prompt_format_audit_self_test.log', timeout_s=180)
print('=== V237 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V232 workbench artifacts.
print('=== V237 V232 ARTIFACT PREFLIGHT START ===', flush=True)
resolved_v232_manifest = resolve_latest_v232_manifest()
print('resolved_v232_manifest =', resolved_v232_manifest, flush=True)
print('resolved_v232_manifest_exists =', resolved_v232_manifest.exists(), flush=True)
print('resolved_v232_manifest_is_file =', resolved_v232_manifest.is_file(), flush=True)
if not resolved_v232_manifest.exists():
    raise FileNotFoundError(resolved_v232_manifest)
if not resolved_v232_manifest.is_file():
    raise IsADirectoryError('V232 manifest must be a JSON file: ' + str(resolved_v232_manifest))
v232_manifest = read_json(resolved_v232_manifest)
inputs = v232_manifest.get('inputs', {}) if isinstance(v232_manifest.get('inputs', {}), dict) else {}
observed_contract = str(inputs.get('observed_shared_row_contract_sha256') or v232_manifest.get('observed_shared_row_contract_sha256') or inputs.get('expected_shared_row_contract_sha256') or v232_manifest.get('expected_shared_row_contract_sha256') or '')
print('observed_shared_row_contract_sha256 =', observed_contract, flush=True)
if observed_contract != EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V232 shared row contract mismatch: expected ' + EXPECTED_SHARED_ROW_CONTRACT_SHA256 + ', got ' + observed_contract)
outputs = v232_manifest.get('outputs', {})
required_outputs = [
    'equation_solver_workitems_jsonl',
    'bit_guardrail_workitems_jsonl',
    'acceptance_matrix_csv',
    'solver_contracts_json',
]
for key in required_outputs:
    path = pathlib.Path(str(outputs.get(key, '')))
    print('v232_required_output =', key, 'path =', path, 'exists =', path.exists(), 'is_file =', path.is_file(), flush=True)
    if not path.exists():
        raise FileNotFoundError(str(key) + ': ' + str(path))
    if not path.is_file():
        raise IsADirectoryError(str(key) + ': ' + str(path))
    print('v232_output_artifact_meta =', json.dumps({'key': key, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V237 V232 ARTIFACT PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V237 prompt format audit.
print('=== V237 PROMPT FORMAT AUDIT START ===', flush=True)
required_runtime_names = [
    'ROOT',
    'ANALYSIS_OUT',
    'EXPECTED_SHARED_ROW_CONTRACT_SHA256',
    'SAMPLE_LIMIT',
    'RUN_ANALYSIS',
    'resolve_latest_v232_manifest',
    'run_cmd',
    'read_json',
    'sha256_file',
]
missing_runtime_names = [name for name in required_runtime_names if name not in globals()]
if missing_runtime_names:
    raise RuntimeError('Run V237 config/helper cells before this cell; missing: ' + ', '.join(missing_runtime_names))
if 'resolved_v232_manifest' not in globals() or not pathlib.Path(resolved_v232_manifest).exists():
    print('resolved_v232_manifest missing_or_stale; resolving latest V232 manifest now.', flush=True)
    resolved_v232_manifest = resolve_latest_v232_manifest()
print('resolved_v232_manifest =', resolved_v232_manifest, flush=True)
print('resolved_v232_manifest_exists =', pathlib.Path(resolved_v232_manifest).exists(), flush=True)
print('resolved_v232_manifest_is_file =', pathlib.Path(resolved_v232_manifest).is_file(), flush=True)
if not pathlib.Path(resolved_v232_manifest).exists():
    raise FileNotFoundError(resolved_v232_manifest)
if not pathlib.Path(resolved_v232_manifest).is_file():
    raise IsADirectoryError('V232 manifest must be a JSON file: ' + str(resolved_v232_manifest))
if RUN_ANALYSIS:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'analyze_v237_prompt_format_audit.py'),
        '--v232-analysis-manifest-json', str(resolved_v232_manifest),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v237_prompt_format_audit',
        '--expected-shared-row-contract-sha256', EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        '--sample-limit', str(SAMPLE_LIMIT),
        '--preview-limit', str(PREVIEW_LIMIT),
    ]
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v237_prompt_format_audit.log', check=True, timeout_s=300)
else:
    print('RUN_ANALYSIS is false; skipping V237 prompt audit command.', flush=True)

analysis_manifest_path = ANALYSIS_OUT / 'v237_prompt_format_audit_manifest.json'
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if RUN_ANALYSIS and not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path) if analysis_manifest_path.exists() else {}
if analysis_manifest:
    print('analysis_manifest_sha256 =', sha256_file(analysis_manifest_path), flush=True)
    print('counts =', json.dumps(analysis_manifest.get('counts', {}), indent=2, sort_keys=True), flush=True)
    print('equation_hint_summary =', json.dumps(analysis_manifest.get('equation_hint_summary', []), indent=2, sort_keys=True), flush=True)
    print('prompt_format_summary =', json.dumps(analysis_manifest.get('prompt_format_summary', []), indent=2, sort_keys=True), flush=True)
    print('equation_prompt_sample_preview =', json.dumps(analysis_manifest.get('equation_prompt_sample_preview', []), indent=2, sort_keys=True), flush=True)
    print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
    print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('=== V237 PROMPT FORMAT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V237 FINAL MANIFEST START ===', flush=True)
final_manifest = {
    'schema_version': 'kg1_v237_colab_final_manifest_v1',
    'version': VERSION,
    'repo_url': REPO_URL,
    'repo_branch': REPO_BRANCH,
    'repo_commit': repo_commit,
    'run_id': RUN_ID,
    'out_root': str(OUT_ROOT),
    'analysis_out': str(ANALYSIS_OUT),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_manifest_exists': analysis_manifest_path.exists(),
    'analysis_manifest_sha256': sha256_file(analysis_manifest_path) if analysis_manifest_path.exists() else '',
    'decision': analysis_manifest.get('decision', {}) if analysis_manifest else {},
    'blocked_actions': ['train', 'model_generation', 'full_scoring', 'package', 'kaggle_submit', 'external_payload_download'],
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
    'allow_package_output': ALLOW_PACKAGE_OUTPUT,
    'allow_model_generation': ALLOW_MODEL_GENERATION,
    'allow_source_payload_download': ALLOW_SOURCE_PAYLOAD_DOWNLOAD,
}
final_manifest_path = OUT_ROOT / 'v237_prompt_format_audit_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('No package and no Kaggle submit can be created in V237.', flush=True)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest_sha256 =', sha256_file(final_manifest_path), flush=True)
print('final_decision =', json.dumps(final_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('=== V237 FINAL MANIFEST END ===', flush=True)
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

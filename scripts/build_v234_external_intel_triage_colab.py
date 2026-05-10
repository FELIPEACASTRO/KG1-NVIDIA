#!/usr/bin/env python3
"""Build the V234 external intelligence triage Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V234_EXTERNAL_INTEL_TRIAGE_COLAB.ipynb"
)
ROADMAP_REL = "artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v234-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return (
        source.replace("__BRANCH__", BRANCH)
        .replace("__COLAB_URL__", COLAB_URL)
        .replace("__GITHUB_URL__", GITHUB_URL)
        .replace("__ROADMAP_REL__", ROADMAP_REL)
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
            """# KG1 V234 External Intel Triage Colab

Purpose: materialize the roadmap double-check into executable triage artifacts for external Kaggle, Hugging Face, OpenRouter, metric-parity, equation, bit, and external model findings.

This notebook is CPU-only. It does not train, does not run model generation, does not run scoring, does not package outputs, and does not submit to Kaggle.

Primary outputs: `external_metric_parity_report.json`, `kaggle_kernel_triage.csv`, `kaggle_dataset_triage.csv`, `hf_dataset_triage.csv`, `kaggle_model_triage.csv`, `equation_numeric_operator_probe_results.csv`, `bit_boolean_function_probe_results.csv`, and `external_adapter_registry_candidates.csv`.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V234 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V234 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V234 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V234_EXTERNAL_INTEL_TRIAGE_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V234')
OUT_ROOT = DRIVE_ROOT / 'output_v234_external_intel_triage'
RUN_ID = os.environ.get('KG1_V234_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V234_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v234_external_intel_triage' / RUN_ID

ROADMAP_MD_TEXT = os.environ.get('KG1_V234_ROADMAP_MD', '').strip()
if ROADMAP_MD_TEXT in {'.', './'}:
    print('ROADMAP_MD_TEXT ignored_directory_placeholder =', ROADMAP_MD_TEXT, flush=True)
    ROADMAP_MD_TEXT = ''
ROADMAP_MD = pathlib.Path(ROADMAP_MD_TEXT) if ROADMAP_MD_TEXT else None
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V234_EXPECTED_REPO_COMMIT', '').strip()

RUN_ANALYSIS = os.environ.get('KG1_V234_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_TRAIN = False
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False
ALLOW_PACKAGE_OUTPUT = False
ALLOW_MODEL_GENERATION = False

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('RUN_ID =', RUN_ID, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('ROADMAP_MD_TEXT =', ROADMAP_MD_TEXT, flush=True)
print('ROADMAP_MD =', ROADMAP_MD or '', flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('ALLOW_MODEL_GENERATION =', ALLOW_MODEL_GENERATION, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V234: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V234: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V234 is CPU-only external intel triage; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V234 cannot run scoring. Build a separate gated notebook after source triage passes.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V234.')
if ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('Packaging is disabled in V234.')
if ALLOW_MODEL_GENERATION:
    raise RuntimeError('Model generation is disabled in V234.')
print('=== V234 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V234 HELPERS START ===', flush=True)

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


def resolve_roadmap_md():
    if ROADMAP_MD is not None:
        print('roadmap_md_explicit =', ROADMAP_MD, flush=True)
        if not ROADMAP_MD.exists():
            raise FileNotFoundError(ROADMAP_MD)
        if not ROADMAP_MD.is_file():
            raise IsADirectoryError('KG1_V234_ROADMAP_MD must point to a markdown file, got: ' + str(ROADMAP_MD))
        return ROADMAP_MD
    candidate = ROOT / '__ROADMAP_REL__'
    print('roadmap_md_default =', candidate, 'exists =', candidate.exists(), flush=True)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if not candidate.is_file():
        raise IsADirectoryError('default roadmap path is not a file: ' + str(candidate))
    return candidate


print('=== V234 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and run self-test.
print('=== V234 REPO SETUP START ===', flush=True)
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
    ROOT / 'scripts/analyze_v234_external_intel_triage.py',
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
    [sys.executable, str(ROOT / 'scripts/analyze_v234_external_intel_triage.py'), '--self-test', '--output-dir', str(OUT_ROOT / 'self_test_dummy')],
    cwd=ROOT,
    log_path=OUT_ROOT / 'v234_external_intel_triage_self_test.log',
    check=True,
    timeout_s=180,
)
print('=== V234 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight roadmap artifact.
print('=== V234 ROADMAP PREFLIGHT START ===', flush=True)
resolved_roadmap_md = resolve_roadmap_md()
print('resolved_roadmap_md =', resolved_roadmap_md, flush=True)
print('resolved_roadmap_md_exists =', resolved_roadmap_md.exists(), flush=True)
print('resolved_roadmap_md_is_file =', resolved_roadmap_md.is_file(), flush=True)
if not resolved_roadmap_md.exists():
    raise FileNotFoundError(resolved_roadmap_md)
if not resolved_roadmap_md.is_file():
    raise IsADirectoryError('V234 roadmap must be a markdown file, got: ' + str(resolved_roadmap_md))
roadmap_text = resolved_roadmap_md.read_text(encoding='utf-8')
required_roadmap_markers = [
    'Double check OpenRouter e matriz de destino dos achados',
    'Matriz V234 obrigatoria',
    'external_metric_parity_report.json',
    'equation_numeric_operator_probe_results.csv',
    'bit_boolean_function_probe_results.csv',
    'external_adapter_registry_candidates.csv',
]
for marker in required_roadmap_markers:
    print('roadmap_marker =', marker, 'present =', marker in roadmap_text, flush=True)
    if marker not in roadmap_text:
        raise RuntimeError('roadmap marker missing: ' + marker)
print('roadmap_meta =', json.dumps({'bytes': resolved_roadmap_md.stat().st_size, 'sha256': sha256_file(resolved_roadmap_md)}, sort_keys=True), flush=True)
print('=== V234 ROADMAP PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V234 external intelligence triage.
print('=== V234 EXTERNAL INTEL TRIAGE START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v234_external_intel_triage_manifest.json'
if RUN_ANALYSIS:
    resolved_roadmap_md = pathlib.Path(globals().get('resolved_roadmap_md', '') or resolve_roadmap_md())
    print('triage_roadmap_md =', resolved_roadmap_md, flush=True)
    print('triage_roadmap_md_exists =', resolved_roadmap_md.exists(), flush=True)
    print('triage_roadmap_md_is_file =', resolved_roadmap_md.is_file(), flush=True)
    if not resolved_roadmap_md.exists():
        raise FileNotFoundError(resolved_roadmap_md)
    if not resolved_roadmap_md.is_file():
        raise IsADirectoryError('V234 triage requires roadmap markdown file, got: ' + str(resolved_roadmap_md))
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v234_external_intel_triage.py'),
        '--roadmap-md', str(resolved_roadmap_md),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v234_external_intel_triage',
    ]
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v234_external_intel_triage.log', check=True, timeout_s=300)
else:
    print('RUN_ANALYSIS is false; skipping V234 triage command.', flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
print('coverage =', json.dumps(analysis_manifest.get('coverage', {}), indent=2, sort_keys=True), flush=True)
print('metric_parity =', json.dumps(analysis_manifest.get('metric_parity', {}), indent=2, sort_keys=True), flush=True)
print('summary =', json.dumps(analysis_manifest.get('summary', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
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
    path = pathlib.Path(str(analysis_manifest.get('outputs', {}).get(name, '')))
    print('v234_output_artifact =', name, path, 'exists =', path.exists(), 'is_file =', path.is_file(), flush=True)
    if not path.exists():
        raise FileNotFoundError(name + ': ' + str(path))
    if not path.is_file():
        raise IsADirectoryError(name + ': ' + str(path))
    rows = csv_row_count(path) if path.suffix == '.csv' else None
    print('v234_output_artifact_meta =', json.dumps({'name': name, 'rows': rows, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V234 EXTERNAL INTEL TRIAGE END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V234 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
blocked_artifacts = []
for pattern in ['*.zip', '*submission*', '*kaggle*submit*']:
    blocked_artifacts.extend(str(path) for path in OUT_ROOT.rglob(pattern))
print('blocked_artifacts =', json.dumps(blocked_artifacts, indent=2, sort_keys=True), flush=True)
if blocked_artifacts:
    raise RuntimeError('V234 output contains package/submission-like artifacts: ' + json.dumps(blocked_artifacts, sort_keys=True))
print('Scoring is intentionally not automatic in V234 external intel triage.', flush=True)
print('No package and no Kaggle submit can be created in V234.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT or ALLOW_PACKAGE_OUTPUT or ALLOW_MODEL_GENERATION:
    raise RuntimeError('V234 hard block violated.')
final_manifest = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_id': RUN_ID,
    'roadmap_md': str(resolved_roadmap_md),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_manifest_sha256': sha256_file(analysis_manifest_path),
    'coverage': analysis_manifest.get('coverage', {}),
    'metric_parity': analysis_manifest.get('metric_parity', {}),
    'summary': analysis_manifest.get('summary', {}),
    'decision': analysis_manifest.get('decision', {}),
    'outputs': analysis_manifest.get('outputs', {}),
    'allowed_actions': ['review_triage_artifacts', 'download_sources_with_hash_license_guard', 'build_solver_probe_implementation'],
    'blocked_actions': ['train', 'model_generation', 'scoring', 'package', 'kaggle_submit'],
    'roadmap_next': analysis_manifest.get('decision', {}).get('next_action', 'Review V234 triage outputs.'),
}
final_manifest_path = OUT_ROOT / 'v234_external_intel_triage_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('=== V234 FINAL MANIFEST END ===', flush=True)
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

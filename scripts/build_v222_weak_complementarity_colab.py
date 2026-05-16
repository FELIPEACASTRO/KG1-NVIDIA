#!/usr/bin/env python3
"""Build the V222 weak-set complementarity Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V222_WEAK_COMPLEMENTARITY_AND_EQUATION_RESCUE_COLAB.ipynb")
BRANCH = "v222-weak-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V222_WEAK_COMPLEMENTARITY_AND_EQUATION_RESCUE_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V222_WEAK_COMPLEMENTARITY_AND_EQUATION_RESCUE_COLAB.ipynb"
)

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v222-{prefix}-{_CELL_COUNTER:02d}"


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
            f"""# KG1 V222 Weak Complementarity And Equation Rescue Colab

Purpose: read the V221 candidate prediction CSVs, measure per-question complementarity, and decide whether a router-style next step can close the weak-set gate before any new GPU spend.

This notebook is CPU-only. It does not train, does not run model generation, does not run full scoring, and does not submit to Kaggle.

Colab: {COLAB_URL}

GitHub: {GITHUB_URL}
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V222 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V222 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration and hard locks.
print('=== V222 CONFIG START ===', flush=True)
import csv
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

VERSION = 'V222_WEAK_COMPLEMENTARITY_AND_EQUATION_RESCUE_20260508'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '{BRANCH}')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V222')
OUT_ROOT = DRIVE_ROOT / 'output_v222_weak_complementarity'
V221_EVAL_OUT = pathlib.Path(os.environ.get('KG1_V222_V221_EVAL_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab'))
BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get('KG1_V222_BATCH_SUMMARY_JSON', str(V221_EVAL_OUT / 'batch_candidate_summary.json')))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v222_weak_complementarity'
WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
WEAK_MAX_TRUNC_FOR_FULL = 0
PREFERRED_DEFAULT = os.environ.get('KG1_V222_PREFERRED_DEFAULT', 'v217_final_existing')
RUN_ANALYSIS = os.environ.get('KG1_V222_RUN_ANALYSIS', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('V221_EVAL_OUT =', V221_EVAL_OUT, flush=True)
print('BATCH_SUMMARY_JSON =', BATCH_SUMMARY_JSON, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('PREFERRED_DEFAULT =', PREFERRED_DEFAULT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V222.')
print('=== V222 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V222 HELPERS START ===', flush=True)

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def run_cmd(cmd, cwd=None, log_path=None, check=True):
    cwd = pathlib.Path(cwd or '/content')
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = ' '.join(map(str, cmd))
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', printable, flush=True)
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
    )
    elapsed = time.time() - started
    if log_path:
        log_path.write_text(proc.stdout or '', encoding='utf-8')
    if proc.stdout:
        print(proc.stdout, end='' if proc.stdout.endswith('\\n') else '\\n', flush=True)
    print('returncode =', proc.returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if proc.returncode and proc.stdout:
        print('command_tail_on_failure =', '\\n'.join(proc.stdout.splitlines()[-40:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError(f'command failed rc={proc.returncode}: {printable}')
    return proc.returncode


print('=== V222 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo and compile analysis scripts.
print('=== V222 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
print('repo_commit =', repo_commit, flush=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('repo_root_on_sys_path =', str(ROOT) in sys.path, flush=True)

required_scripts = [
    ROOT / 'scripts/analyze_v221_complementarity.py',
    ROOT / 'scripts/notebook_release_gate.py',
    ROOT / 'src/competition_utils.py',
]
for py_path in required_scripts:
    print('compile_check =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)
print('=== V222 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V221 prediction artifacts.
print('=== V222 PREFLIGHT START ===', flush=True)
print('BATCH_SUMMARY_JSON exists =', BATCH_SUMMARY_JSON.exists(), flush=True)
if not BATCH_SUMMARY_JSON.exists():
    raise FileNotFoundError('V221 batch summary missing. Run V221 first or set KG1_V222_BATCH_SUMMARY_JSON.')
batch_summary = read_json(BATCH_SUMMARY_JSON)
candidate_rows = [row for row in batch_summary.get('rows', []) if row.get('status') == 'ok']
print('batch_summary_generated_at_utc =', batch_summary.get('generated_at_utc', ''), flush=True)
print('batch_summary_config =', json.dumps(batch_summary.get('config', {}), indent=2, sort_keys=True), flush=True)
print('candidate_count =', len(candidate_rows), flush=True)
missing_prediction_artifacts = []
for row in candidate_rows:
    report_path = pathlib.Path(str(row.get('report_json', '')))
    print('candidate_report =', row.get('name'), report_path, 'exists =', report_path.exists(), flush=True)
    if not report_path.exists():
        missing_prediction_artifacts.append(str(report_path))
        continue
    report = read_json(report_path)
    predictions_csv = pathlib.Path(str(report.get('outputs', {}).get('predictions_csv', '')))
    print('candidate_predictions =', row.get('name'), predictions_csv, 'exists =', predictions_csv.exists(), 'bytes =', predictions_csv.stat().st_size if predictions_csv.exists() else 0, flush=True)
    if not predictions_csv.exists():
        missing_prediction_artifacts.append(str(predictions_csv))
if missing_prediction_artifacts:
    print('missing_prediction_artifacts =', json.dumps(missing_prediction_artifacts, indent=2), flush=True)
    raise FileNotFoundError('Missing V221 prediction artifacts; cannot run V222 complementarity analysis.')
print('=== V222 PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V222 complementarity and router simulation.
print('=== V222 COMPLEMENTARITY ANALYSIS START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v222_v221_weak_manifest.json'
if not RUN_ANALYSIS:
    print('RUN_ANALYSIS is false; skipping analysis command.', flush=True)
else:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v221_complementarity.py'),
        '--batch-summary-json', str(BATCH_SUMMARY_JSON),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v222_v221_weak',
        '--preferred-default', PREFERRED_DEFAULT,
        '--weak-total-min', str(WEAK_MIN_FOR_FULL),
        '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
        '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
        '--weak-trunc-max', str(WEAK_MAX_TRUNC_FOR_FULL),
    ]
    print('analysis_cmd =', ' '.join(map(str, cmd)), flush=True)
    rc = run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v222_complementarity.log', check=True)
    print('analysis_returncode =', rc, flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
print('analysis_decision =', analysis_manifest.get('decision', ''), flush=True)
print('analysis_outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('router_simulation =', json.dumps(analysis_manifest.get('router_simulation', []), indent=2, sort_keys=True), flush=True)
print('=== V222 COMPLEMENTARITY ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: write V222 final manifest and next action.
print('=== V222 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
router_rows = analysis_manifest.get('router_simulation', [])
deployable_pass = [row for row in router_rows if row.get('deployable_without_row_labels') and row.get('gate_pass')]
oracle_safe_pass = [row for row in router_rows if row.get('strategy') == 'oracle_safe_candidate_by_row' and row.get('gate_pass')]
oracle_any_pass = [row for row in router_rows if row.get('strategy') == 'oracle_any_candidate_by_row' and row.get('gate_pass')]
if deployable_pass:
    next_action = 'Review deployable router candidate, then create a separate full-scoring notebook with hard gate.'
elif oracle_safe_pass:
    next_action = 'Complementarity exists, but row-label oracle is not deployable. Build rules/features or train a small router/rescue before full scoring.'
elif oracle_any_pass:
    next_action = 'Only unsafe candidate oracle passes. Fix truncation or train equation rescue; do not full score yet.'
else:
    next_action = 'No oracle pass on current candidates. Train equation rescue with bit replay protection.'
decision = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'batch_summary_json': str(BATCH_SUMMARY_JSON),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_decision': analysis_manifest.get('decision', ''),
    'deployable_pass': deployable_pass,
    'oracle_safe_pass': oracle_safe_pass,
    'oracle_any_pass': oracle_any_pass,
    'next_action': next_action,
    'allow_kaggle_submit': ALLOW_KAGGLE_SUBMIT,
}
manifest_path = OUT_ROOT / 'v222_weak_complementarity_manifest.json'
manifest_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V222.')
print('=== V222 FINAL MANIFEST END ===', flush=True)
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

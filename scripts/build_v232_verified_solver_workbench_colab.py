#!/usr/bin/env python3
"""Build the V232 verified solver workbench Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V232_VERIFIED_SOLVER_WORKBENCH_COLAB.ipynb"
)
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v232-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V232 Verified Solver Workbench Colab

Purpose: convert the executed V231 miss-pack mining outputs into a verified solver workbench with full-prompt work items, acceptance contracts, and no-submit hard locks.

This notebook is CPU-only. It does not train, does not run model generation, does not run full scoring, does not package outputs, and does not submit to Kaggle.

Primary outputs: `equation_solver_workitems_jsonl`, `bit_guardrail_workitems_jsonl`, `acceptance_matrix_csv`, and `solver_contracts_json`.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V232 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V232 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V232 CONFIG START ===', flush=True)
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V232_VERIFIED_SOLVER_WORKBENCH_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V232')
OUT_ROOT = DRIVE_ROOT / 'output_v232_verified_solver_workbench'
RUN_ID = os.environ.get('KG1_V232_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V232_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v232_verified_solver_workbench' / RUN_ID

V231_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V232_V231_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V231/output_v231_v230_miss_pack_mining',
))
V231_ANALYSIS_MANIFEST_JSON_TEXT = os.environ.get('KG1_V232_V231_ANALYSIS_MANIFEST_JSON', '').strip()
V231_ANALYSIS_MANIFEST_JSON = pathlib.Path(V231_ANALYSIS_MANIFEST_JSON_TEXT) if V231_ANALYSIS_MANIFEST_JSON_TEXT else None
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = os.environ.get(
    'KG1_V232_EXPECTED_SHARED_ROW_CONTRACT_SHA256',
    '__EXPECTED_SHARED_ROW_CONTRACT_SHA256__',
).strip()
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V232_EXPECTED_REPO_COMMIT', '').strip()

RUN_ANALYSIS = os.environ.get('KG1_V232_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_TRAIN = False
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False
ALLOW_PACKAGE_OUTPUT = False
EQUATION_TARGET_GAIN = 5
BIT_GUARDRAIL_MIN = 136

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('RUN_ID =', RUN_ID, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('V231_OUTPUT_ROOT =', V231_OUTPUT_ROOT, flush=True)
print('V231_ANALYSIS_MANIFEST_JSON_TEXT =', V231_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
print('V231_ANALYSIS_MANIFEST_JSON =', V231_ANALYSIS_MANIFEST_JSON or '', flush=True)
print('EXPECTED_SHARED_ROW_CONTRACT_SHA256 =', EXPECTED_SHARED_ROW_CONTRACT_SHA256, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('EQUATION_TARGET_GAIN =', EQUATION_TARGET_GAIN, flush=True)
print('BIT_GUARDRAIL_MIN =', BIT_GUARDRAIL_MIN, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V232: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V232: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V232 is CPU-only solver workbench; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V232 cannot run full scoring. Build a separate gated notebook after solver proof passes.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V232.')
if not EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V232 requires KG1_V232_EXPECTED_SHARED_ROW_CONTRACT_SHA256.')
print('=== V232 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V232 HELPERS START ===', flush=True)

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


def resolve_latest_v231_manifest():
    if V231_ANALYSIS_MANIFEST_JSON is not None:
        print('v231_manifest_explicit =', V231_ANALYSIS_MANIFEST_JSON, flush=True)
        if not V231_ANALYSIS_MANIFEST_JSON.exists():
            raise FileNotFoundError(V231_ANALYSIS_MANIFEST_JSON)
        if not V231_ANALYSIS_MANIFEST_JSON.is_file():
            raise IsADirectoryError('KG1_V232_V231_ANALYSIS_MANIFEST_JSON must point to a JSON file, got: ' + str(V231_ANALYSIS_MANIFEST_JSON))
        return V231_ANALYSIS_MANIFEST_JSON
    search_root = V231_OUTPUT_ROOT / 'analysis_v231_miss_pack_mining'
    print('v231_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
    candidates = sorted(search_root.glob('*/v231_v230_miss_pack_mining_manifest.json'), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    print('v231_manifest_candidate_count =', len(candidates), flush=True)
    for candidate in candidates[:10]:
        print('v231_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
    if not candidates:
        raise FileNotFoundError('No V231 manifest found under: ' + str(search_root))
    return candidates[0]


print('=== V232 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and run self-test.
print('=== V232 REPO SETUP START ===', flush=True)
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
    ROOT / 'scripts/analyze_v232_verified_solver_workbench.py',
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
    [sys.executable, str(ROOT / 'scripts/analyze_v232_verified_solver_workbench.py'), '--self-test', '--v231-analysis-manifest-json', 'dummy', '--output-dir', 'dummy'],
    cwd=ROOT,
    log_path=OUT_ROOT / 'v232_verified_solver_workbench_self_test.log',
    check=True,
    timeout_s=180,
)
print('=== V232 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V231 artifacts.
print('=== V232 V231 ARTIFACT PREFLIGHT START ===', flush=True)
resolved_v231_manifest = resolve_latest_v231_manifest()
print('resolved_v231_manifest =', resolved_v231_manifest, flush=True)
print('resolved_v231_manifest_exists =', resolved_v231_manifest.exists(), flush=True)
print('resolved_v231_manifest_is_file =', resolved_v231_manifest.is_file(), flush=True)
if not resolved_v231_manifest.exists():
    raise FileNotFoundError(resolved_v231_manifest)
if not resolved_v231_manifest.is_file():
    raise IsADirectoryError('V231 manifest must be a JSON file, got: ' + str(resolved_v231_manifest))
v231_manifest = read_json(resolved_v231_manifest)
v231_inputs = v231_manifest.get('inputs', {})
if not isinstance(v231_inputs, dict):
    v231_inputs = {}
observed_contract = str(
    v231_inputs.get('observed_shared_row_contract_sha256')
    or v231_manifest.get('observed_shared_row_contract_sha256')
    or v231_inputs.get('expected_shared_row_contract_sha256')
    or v231_manifest.get('expected_shared_row_contract_sha256')
    or ''
)
print('observed_shared_row_contract_sha256 =', observed_contract, flush=True)
if observed_contract != EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V231 shared row contract mismatch: expected ' + EXPECTED_SHARED_ROW_CONTRACT_SHA256 + ', got ' + observed_contract)
print('v231_decision =', json.dumps(v231_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('v231_miss_counts =', json.dumps(v231_manifest.get('miss_counts', {}), indent=2, sort_keys=True), flush=True)
required_outputs = [
    'equation_miss_taxonomy_csv',
    'bit_miss_taxonomy_csv',
    'equation_solver_candidate_rules_json',
    'bit_guardrail_candidates_json',
]
outputs = v231_manifest.get('outputs', {})
for name in required_outputs:
    path = pathlib.Path(str(outputs.get(name, '')))
    print('v231_output_artifact =', name, path, 'exists =', path.exists(), 'is_file =', path.is_file(), flush=True)
    if not path.exists():
        raise FileNotFoundError(name + ': ' + str(path))
    if not path.is_file():
        raise IsADirectoryError(name + ': ' + str(path))
    rows = csv_row_count(path) if path.suffix == '.csv' else None
    print('v231_output_artifact_meta =', json.dumps({'name': name, 'rows': rows, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V232 V231 ARTIFACT PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V232 verified solver workbench.
print('=== V232 VERIFIED SOLVER WORKBENCH START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v232_verified_solver_workbench_manifest.json'
if RUN_ANALYSIS:
    if 'resolved_v231_manifest' not in globals() or not pathlib.Path(resolved_v231_manifest).is_file():
        print('resolved_v231_manifest missing or invalid before workbench; resolving again.', flush=True)
        resolved_v231_manifest = resolve_latest_v231_manifest()
    resolved_v231_manifest = pathlib.Path(resolved_v231_manifest)
    print('workbench_v231_manifest =', resolved_v231_manifest, flush=True)
    print('workbench_v231_manifest_exists =', resolved_v231_manifest.exists(), flush=True)
    print('workbench_v231_manifest_is_file =', resolved_v231_manifest.is_file(), flush=True)
    if not resolved_v231_manifest.exists():
        raise FileNotFoundError(resolved_v231_manifest)
    if not resolved_v231_manifest.is_file():
        raise IsADirectoryError('V232 workbench requires V231 manifest JSON file, got: ' + str(resolved_v231_manifest))
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v232_verified_solver_workbench.py'),
        '--v231-analysis-manifest-json', str(resolved_v231_manifest),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v232_verified_solver_workbench',
        '--expected-shared-row-contract-sha256', EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        '--equation-target-gain', str(EQUATION_TARGET_GAIN),
        '--bit-guardrail-min', str(BIT_GUARDRAIL_MIN),
    ]
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v232_verified_solver_workbench.log', check=True, timeout_s=300)
else:
    print('RUN_ANALYSIS is false; skipping V232 workbench command.', flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
print('workitem_counts =', json.dumps(analysis_manifest.get('workitem_counts', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('=== V232 VERIFIED SOLVER WORKBENCH END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V232 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
blocked_artifacts = []
for pattern in ['*.zip', '*submission*', '*kaggle*submit*']:
    blocked_artifacts.extend(str(path) for path in OUT_ROOT.rglob(pattern))
print('blocked_artifacts =', json.dumps(blocked_artifacts, indent=2, sort_keys=True), flush=True)
if blocked_artifacts:
    raise RuntimeError('V232 output contains package/submission-like artifacts: ' + json.dumps(blocked_artifacts, sort_keys=True))
print('Full scoring is intentionally not automatic in V232 verified solver workbench.', flush=True)
print('No package and no Kaggle submit can be created in V232.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT or ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('V232 hard block violated.')
final_manifest = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_id': RUN_ID,
    'v231_manifest': str(resolved_v231_manifest),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_manifest_sha256': sha256_file(analysis_manifest_path),
    'expected_shared_row_contract_sha256': EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    'decision': analysis_manifest.get('decision', {}),
    'workitem_counts': analysis_manifest.get('workitem_counts', {}),
    'outputs': analysis_manifest.get('outputs', {}),
    'allowed_actions': ['review_workitems', 'implement_verified_solver_probes', 'prepare_v233_solver_probe_notebook'],
    'blocked_actions': ['train', 'full_scoring', 'package', 'kaggle_submit'],
    'roadmap_next': 'Implement V233 verified equation solver probes against V232 workitems before any training.',
}
final_manifest_path = OUT_ROOT / 'v232_verified_solver_workbench_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('=== V232 FINAL MANIFEST END ===', flush=True)
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

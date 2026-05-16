#!/usr/bin/env python3
"""Build the V231 miss-pack mining Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V231_MISS_PACK_MINING_COLAB.ipynb"
)
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v231-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 V231 Miss Pack Mining Colab

Purpose: mine the V230 miss-pack artifacts and build a taxonomy for verified equation and bit rescue rules before any GPU spend.

This notebook is CPU-only. It does not train, does not run model generation, does not run full scoring, does not package outputs, and does not submit to Kaggle.

Primary outputs: `equation_miss_taxonomy_csv`, `equation_solver_candidate_rules_json`, and `bit_guardrail_candidates_json`.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V231 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V231 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration and hard locks.
print('=== V231 CONFIG START ===', flush=True)
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

VERSION = 'V231_MISS_PACK_MINING_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
EXPECTED_REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
EXPECTED_REPO_BRANCH = '__BRANCH__'
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V231')
OUT_ROOT = DRIVE_ROOT / 'output_v231_v230_miss_pack_mining'
RUN_ID = os.environ.get('KG1_V231_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if not re.fullmatch(r'[A-Za-z0-9_.-]+', RUN_ID):
    raise RuntimeError('KG1_V231_RUN_ID contains unsafe characters: ' + repr(RUN_ID))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v231_miss_pack_mining' / RUN_ID

V230_OUTPUT_ROOT = pathlib.Path(os.environ.get(
    'KG1_V231_V230_OUTPUT_ROOT',
    '/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity',
))
V230_ANALYSIS_MANIFEST_JSON_TEXT = os.environ.get('KG1_V231_V230_ANALYSIS_MANIFEST_JSON', '').strip()
V230_ANALYSIS_MANIFEST_JSON = pathlib.Path(V230_ANALYSIS_MANIFEST_JSON_TEXT) if V230_ANALYSIS_MANIFEST_JSON_TEXT else None
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = os.environ.get(
    'KG1_V231_EXPECTED_SHARED_ROW_CONTRACT_SHA256',
    '__EXPECTED_SHARED_ROW_CONTRACT_SHA256__',
).strip()
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V231_EXPECTED_REPO_COMMIT', '').strip()

RUN_ANALYSIS = os.environ.get('KG1_V231_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
RUN_TRAIN = False
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False
ALLOW_PACKAGE_OUTPUT = False
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
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
print('V230_OUTPUT_ROOT =', V230_OUTPUT_ROOT, flush=True)
print('V230_ANALYSIS_MANIFEST_JSON_TEXT =', V230_ANALYSIS_MANIFEST_JSON_TEXT, flush=True)
print('V230_ANALYSIS_MANIFEST_JSON =', V230_ANALYSIS_MANIFEST_JSON or '', flush=True)
print('EXPECTED_SHARED_ROW_CONTRACT_SHA256 =', EXPECTED_SHARED_ROW_CONTRACT_SHA256, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('ALLOW_PACKAGE_OUTPUT =', ALLOW_PACKAGE_OUTPUT, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('EQUATION_TARGET_GAIN =', EQUATION_TARGET_GAIN, flush=True)
print('BIT_GUARDRAIL_MIN =', BIT_GUARDRAIL_MIN, flush=True)

if REPO_URL != EXPECTED_REPO_URL:
    raise RuntimeError('KG1_REPO_URL override is not allowed in V231: ' + REPO_URL)
if REPO_BRANCH != EXPECTED_REPO_BRANCH:
    raise RuntimeError('KG1_REPO_BRANCH override is not allowed in V231: ' + REPO_BRANCH)
if RUN_TRAIN:
    raise RuntimeError('V231 is CPU-only miss-pack mining; RUN_TRAIN must stay false.')
if RUN_FULL_IF_GATE:
    raise RuntimeError('V231 cannot run full scoring. Build a separate gated notebook after weak gates pass.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V231.')
if not EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V231 requires KG1_V231_EXPECTED_SHARED_ROW_CONTRACT_SHA256.')
print('=== V231 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging.
print('=== V231 HELPERS START ===', flush=True)

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
        print('command_tail_on_failure =', '\\n'.join(proc.stdout.splitlines()[-60:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError(f'command failed rc={proc.returncode}: {printable}')
    return proc.returncode


def resolve_latest_v230_manifest():
    if V230_ANALYSIS_MANIFEST_JSON is not None:
        print('v230_manifest_explicit =', V230_ANALYSIS_MANIFEST_JSON, flush=True)
        if not V230_ANALYSIS_MANIFEST_JSON.exists():
            raise FileNotFoundError(V230_ANALYSIS_MANIFEST_JSON)
        if not V230_ANALYSIS_MANIFEST_JSON.is_file():
            raise IsADirectoryError('KG1_V231_V230_ANALYSIS_MANIFEST_JSON must point to a JSON file, got: ' + str(V230_ANALYSIS_MANIFEST_JSON))
        return V230_ANALYSIS_MANIFEST_JSON
    search_root = V230_OUTPUT_ROOT / 'analysis_v230_v226_complementarity'
    print('v230_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
    candidates = sorted(search_root.glob('*/v230_v226_complementarity_manifest.json'), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    print('v230_manifest_candidate_count =', len(candidates), flush=True)
    for candidate in candidates[:10]:
        print('v230_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
    if not candidates:
        raise FileNotFoundError('No V230 manifest found under: ' + str(search_root))
    return candidates[0]


print('=== V231 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and run self-test.
print('=== V231 REPO SETUP START ===', flush=True)
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
    ROOT / 'scripts/analyze_v231_miss_packs.py',
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
    [sys.executable, str(ROOT / 'scripts/analyze_v231_miss_packs.py'), '--self-test', '--v230-analysis-manifest-json', 'dummy', '--output-dir', 'dummy'],
    cwd=ROOT,
    log_path=OUT_ROOT / 'v231_miss_pack_mining_self_test.log',
    check=True,
    timeout_s=180,
)
print('=== V231 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: preflight V230 miss-pack artifacts.
print('=== V231 V230 ARTIFACT PREFLIGHT START ===', flush=True)
resolved_v230_manifest = resolve_latest_v230_manifest()
print('resolved_v230_manifest =', resolved_v230_manifest, flush=True)
print('resolved_v230_manifest_exists =', resolved_v230_manifest.exists(), flush=True)
if not resolved_v230_manifest.exists():
    raise FileNotFoundError(resolved_v230_manifest)
v230_manifest = read_json(resolved_v230_manifest)
observed_contract = str(v230_manifest.get('observed_shared_row_contract_sha256', ''))
print('observed_shared_row_contract_sha256 =', observed_contract, flush=True)
if observed_contract != EXPECTED_SHARED_ROW_CONTRACT_SHA256:
    raise RuntimeError('V230 shared row contract mismatch: expected ' + EXPECTED_SHARED_ROW_CONTRACT_SHA256 + ', got ' + observed_contract)
print('v230_decision =', json.dumps(v230_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('v230_baseline_summary =', json.dumps(v230_manifest.get('baseline_summary', {}), indent=2, sort_keys=True), flush=True)
required_outputs = [
    'baseline_miss_hits_csv',
    'equation_miss_pack_csv',
    'bit_miss_pack_csv',
    'pairwise_detail_csv',
    'candidate_summary_csv',
    'router_simulation_csv',
]
outputs = v230_manifest.get('outputs', {})
for name in required_outputs:
    path = pathlib.Path(str(outputs.get(name, '')))
    print('v230_output_artifact =', name, path, 'exists =', path.exists(), flush=True)
    if not path.exists():
        raise FileNotFoundError(name + ': ' + str(path))
    print('v230_output_artifact_meta =', json.dumps({'name': name, 'rows': csv_row_count(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}, sort_keys=True), flush=True)
print('=== V231 V230 ARTIFACT PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V231 miss-pack mining.
print('=== V231 MISS PACK MINING START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v231_v230_miss_pack_mining_manifest.json'
if RUN_ANALYSIS:
    def _resolve_v230_manifest_for_mining():
        existing = pathlib.Path(str(globals().get('resolved_v230_manifest', '')))
        print('mining_existing_resolved_v230_manifest =', existing, 'is_file =', existing.is_file(), flush=True)
        if existing.is_file():
            return existing
        explicit_text = str(globals().get('V230_ANALYSIS_MANIFEST_JSON_TEXT', os.environ.get('KG1_V231_V230_ANALYSIS_MANIFEST_JSON', ''))).strip()
        print('mining_v230_manifest_explicit_text =', explicit_text, flush=True)
        if explicit_text and explicit_text not in {'.', './'}:
            explicit = pathlib.Path(explicit_text)
            print('mining_v230_manifest_explicit =', explicit, 'exists =', explicit.exists(), 'is_file =', explicit.is_file(), flush=True)
            if not explicit.exists():
                raise FileNotFoundError(explicit)
            if not explicit.is_file():
                raise IsADirectoryError('KG1_V231_V230_ANALYSIS_MANIFEST_JSON must point to a JSON file, got: ' + str(explicit))
            return explicit
        search_root = pathlib.Path(globals().get('V230_OUTPUT_ROOT', os.environ.get('KG1_V231_V230_OUTPUT_ROOT', '/content/drive/MyDrive/KG1_NVIDIA_V230/output_v230_v226_complementarity'))) / 'analysis_v230_v226_complementarity'
        print('mining_v230_manifest_search_root =', search_root, 'exists =', search_root.exists(), flush=True)
        candidates = sorted(search_root.glob('*/v230_v226_complementarity_manifest.json'), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
        print('mining_v230_manifest_candidate_count =', len(candidates), flush=True)
        for candidate in candidates[:10]:
            print('mining_v230_manifest_candidate =', candidate, 'mtime =', candidate.stat().st_mtime, flush=True)
        if not candidates:
            raise FileNotFoundError('No V230 manifest found under: ' + str(search_root))
        return candidates[0]

    resolved_v230_manifest = _resolve_v230_manifest_for_mining()
    print('mining_v230_manifest =', resolved_v230_manifest, flush=True)
    print('mining_v230_manifest_exists =', resolved_v230_manifest.exists(), flush=True)
    print('mining_v230_manifest_is_file =', resolved_v230_manifest.is_file(), flush=True)
    if not resolved_v230_manifest.exists():
        raise FileNotFoundError(resolved_v230_manifest)
    if not resolved_v230_manifest.is_file():
        raise IsADirectoryError('V231 mining requires V230 manifest JSON file, got: ' + str(resolved_v230_manifest))
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v231_miss_packs.py'),
        '--v230-analysis-manifest-json', str(resolved_v230_manifest),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v231_v230_miss_pack_mining',
        '--expected-shared-row-contract-sha256', EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
        '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
        '--equation-target-gain', str(EQUATION_TARGET_GAIN),
        '--bit-guardrail-min', str(BIT_GUARDRAIL_MIN),
    ]
    run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v231_miss_pack_mining.log', check=True, timeout_s=300)
else:
    print('RUN_ANALYSIS is false; skipping V231 mining command.', flush=True)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_exists =', analysis_manifest_path.exists(), flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
print('miss_counts =', json.dumps(analysis_manifest.get('miss_counts', {}), indent=2, sort_keys=True), flush=True)
print('route_summary =', json.dumps(analysis_manifest.get('route_summary', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('=== V231 MISS PACK MINING END ===', flush=True)
"""
        ),
        code(
            """# CELL: final manifest and hard block.
print('=== V231 FINAL MANIFEST START ===', flush=True)
analysis_manifest = read_json(analysis_manifest_path)
blocked_artifacts = []
for pattern in ['*.zip', '*submission*', '*kaggle*submit*']:
    blocked_artifacts.extend(str(path) for path in OUT_ROOT.rglob(pattern))
print('blocked_artifacts =', json.dumps(blocked_artifacts, indent=2, sort_keys=True), flush=True)
if blocked_artifacts:
    raise RuntimeError('V231 output contains package/submission-like artifacts: ' + json.dumps(blocked_artifacts, sort_keys=True))
print('Full scoring is intentionally not automatic in V231 miss-pack mining notebook.', flush=True)
print('No package and no Kaggle submit can be created in V231.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT or ALLOW_PACKAGE_OUTPUT:
    raise RuntimeError('V231 hard block violated.')
final_manifest = {
    'version': VERSION,
    'repo_commit': globals().get('repo_commit', ''),
    'run_id': RUN_ID,
    'v230_manifest': str(resolved_v230_manifest),
    'analysis_manifest_path': str(analysis_manifest_path),
    'analysis_manifest_sha256': sha256_file(analysis_manifest_path),
    'expected_shared_row_contract_sha256': EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    'decision': analysis_manifest.get('decision', {}),
    'miss_counts': analysis_manifest.get('miss_counts', {}),
    'outputs': analysis_manifest.get('outputs', {}),
    'allowed_actions': ['review_taxonomy', 'write_verified_solver_tests', 'prepare_separate_gated_rescue_notebook'],
    'blocked_actions': ['train', 'full_scoring', 'package', 'kaggle_submit'],
    'roadmap_next': 'Review equation taxonomy and implement verified equation solvers before any training.',
}
final_manifest_path = OUT_ROOT / 'v231_miss_pack_mining_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('=== V231 FINAL MANIFEST END ===', flush=True)
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

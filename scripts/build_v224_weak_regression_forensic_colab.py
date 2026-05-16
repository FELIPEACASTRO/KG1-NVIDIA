#!/usr/bin/env python3
"""Build the V224 weak-regression forensic Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V224_WEAK_REGRESSION_FORENSIC_COLAB.ipynb")
BRANCH = "v224-weak-regression-forensic"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V224_WEAK_REGRESSION_FORENSIC_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V224_WEAK_REGRESSION_FORENSIC_COLAB.ipynb"
)

TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v224-{prefix}-{_CELL_COUNTER:02d}"


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
            f"""# KG1 V224 Weak Regression Forensic Colab

Purpose: compare V223 weak predictions against the V221/V217/V194 weak baselines, quantify row-level losses/gains, and produce a rollback/router roadmap before any new training or full evaluation.

This notebook is CPU-only. It does not train, does not run model generation, does not run full eval, and does not submit to Kaggle.

Colab: {COLAB_URL}

GitHub: {GITHUB_URL}
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V224 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V224 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            f"""# CELL: global configuration, compatibility gates, and hard submit lock.
print('=== V224 CONFIG START ===', flush=True)
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

VERSION = 'V224_WEAK_REGRESSION_FORENSIC_20260509'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '{BRANCH}')
ROOT = pathlib.Path('/content/kg1')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V224')
OUT_ROOT = DRIVE_ROOT / 'output_v224_weak_regression_forensic'
ANALYSIS_OUT = OUT_ROOT / 'analysis_v224_weak_regression'

V221_EVAL_OUT = pathlib.Path(os.environ.get('KG1_V224_V221_EVAL_OUT', '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab'))
V221_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get('KG1_V224_V221_BATCH_SUMMARY_JSON', str(V221_EVAL_OUT / 'batch_candidate_summary.json')))
V223_WEAK_REPORT_JSON = pathlib.Path(os.environ.get('KG1_V224_V223_WEAK_REPORT_JSON', '/content/drive/MyDrive/KG1_NVIDIA_V223/output_v223_equation_rescue/eval_v223_eqrescue_from_v217_lr1e8_s12/weak_eval/v223_eqrescue_weak_eval_report.json'))
V223_CANDIDATE_NAME = os.environ.get('KG1_V224_V223_CANDIDATE_NAME', 'v223_eqrescue_from_v217_lr1e8_s12')
BASELINE_CANDIDATE = os.environ.get('KG1_V224_BASELINE_CANDIDATE', 'v217_final_existing')

V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
INIT_ADAPTER_DIR = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter')
EXPECTED_TRAIN_SHA256 = '{TRAIN_SHA}'
EXPECTED_VAL_SHA256 = '{VAL_SHA}'
MIN_TRAIN_EXAMPLES = 10206
MIN_VAL_EXAMPLES = 681
TOKENIZE_ONLY_DRY_RUN = True
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
RUN_TRAIN = os.environ.get('KG1_V224_RUN_TRAIN', '0').strip().lower() in {{'1', 'true', 'yes', 'on'}}
RUN_ANALYSIS = os.environ.get('KG1_V224_RUN_ANALYSIS', '1').strip().lower() not in {{'0', 'false', 'no', 'off'}}
RUN_EVAL = False
RUN_FULL_IF_GATE = False

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 136
WEAK_MAX_TRUNC_FOR_FULL = 0
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
ALLOW_KAGGLE_SUBMIT = False

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('V221_BATCH_SUMMARY_JSON =', V221_BATCH_SUMMARY_JSON, flush=True)
print('V223_WEAK_REPORT_JSON =', V223_WEAK_REPORT_JSON, flush=True)
print('V223_CANDIDATE_NAME =', V223_CANDIDATE_NAME, flush=True)
print('BASELINE_CANDIDATE =', BASELINE_CANDIDATE, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
if RUN_TRAIN:
    raise RuntimeError('V224 is forensic-only; RUN_TRAIN must stay false.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V224.')
print('=== V224 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging and hash checks.
print('=== V224 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def run_cmd(cmd, cwd=None, log_path=None, check=True, suppress_after_lines=220):
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
    output = proc.stdout or ''
    if log_path:
        log_path.write_text(output, encoding='utf-8')
    lines = output.splitlines()
    if suppress_after_lines and len(lines) > suppress_after_lines:
        shown = lines[:120] + ['... command_output_suppressed_lines = ' + str(len(lines) - 180) + ' ...'] + lines[-60:]
    else:
        shown = lines
    for line in shown:
        print(line, flush=True)
    print('returncode =', proc.returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if proc.returncode and lines:
        print('command_tail_on_failure =', '\\n'.join(lines[-40:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError(f'command failed rc={proc.returncode}: {printable}')
    return proc.returncode


print('=== V224 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and validate static artifacts.
print('=== V224 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
print('repo_commit =', repo_commit, flush=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
compile_targets = [
    ROOT / 'src/competition_utils.py',
    ROOT / 'scripts/analyze_v224_weak_regression.py',
    ROOT / 'scripts/evaluate_lora_adapter.py',
    ROOT / 'scripts/notebook_release_gate.py',
]
for py_path in compile_targets:
    print('compile_target =', py_path, 'exists =', py_path.exists(), flush=True)
    if not py_path.exists():
        raise FileNotFoundError(py_path)
    import py_compile
    py_compile.compile(str(py_path), doraise=True)
    print('py_compile ok =', py_path.relative_to(ROOT), flush=True)
train_path = ROOT / 'data/v217/v217_short_answer_train.jsonl'
val_path = ROOT / 'data/v217/v217_short_answer_val.jsonl'
print('train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('val_path =', val_path, 'exists =', val_path.exists(), flush=True)
if train_path.exists():
    observed_train_sha256 = sha256_file(train_path)
    print('observed_train_sha256 =', observed_train_sha256, flush=True)
    if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f'train sha mismatch: {observed_train_sha256} != {EXPECTED_TRAIN_SHA256}')
if val_path.exists():
    observed_val_sha256 = sha256_file(val_path)
    print('observed_val_sha256 =', observed_val_sha256, flush=True)
    if observed_val_sha256 != EXPECTED_VAL_SHA256:
        raise RuntimeError(f'val sha mismatch: {observed_val_sha256} != {EXPECTED_VAL_SHA256}')
print('=== V224 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, Drive artifact, and adapter audit.
print('=== V224 RUNTIME ARTIFACT AUDIT START ===', flush=True)
try:
    import torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    cuda_available = bool(torch.cuda.is_available())
    gpu_total_gib = float(props.total_memory / 1024**3) if props else 0.0
    gpu_name = props.name if props else ''
except Exception as exc:
    cuda_available = False
    gpu_total_gib = 0.0
    gpu_name = ''
    print('torch_cuda_probe_error =', repr(exc), flush=True)
content_free_gib = shutil.disk_usage('/content').free / 1024**3
print('cuda_available =', cuda_available, flush=True)
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
print('causal_conv1d = not required for V224 CPU forensic, audited as training-stack dependency name only', flush=True)
print('mamba_ssm = not required for V224 CPU forensic, audited as training-stack dependency name only', flush=True)
print('V221_BATCH_SUMMARY_JSON exists =', V221_BATCH_SUMMARY_JSON.exists(), flush=True)
print('V223_WEAK_REPORT_JSON exists =', V223_WEAK_REPORT_JSON.exists(), flush=True)
if not V221_BATCH_SUMMARY_JSON.exists():
    raise FileNotFoundError(f'Missing V221 batch summary: {V221_BATCH_SUMMARY_JSON}')
if not V223_WEAK_REPORT_JSON.exists():
    raise FileNotFoundError(f'Missing V223 weak report: {V223_WEAK_REPORT_JSON}')
batch_summary = read_json(V221_BATCH_SUMMARY_JSON)
v223_report = read_json(V223_WEAK_REPORT_JSON)
print('v221_batch_rows =', len(batch_summary.get('rows', [])), flush=True)
print('v223_report_correct =', v223_report.get('correct'), flush=True)
print('v223_report_truncated =', v223_report.get('truncated'), flush=True)
print('v223_report_predictions_csv =', v223_report.get('outputs', {}).get('predictions_csv'), flush=True)
print('V194_ADAPTER exists =', V194_ADAPTER.exists(), flush=True)
print('V194 adapter_config.json exists =', (V194_ADAPTER / 'adapter_config.json').exists(), flush=True)
print('V194 adapter_model.safetensors exists =', (V194_ADAPTER / 'adapter_model.safetensors').exists(), flush=True)
if (V194_ADAPTER / 'adapter_config.json').exists():
    v194_config = read_json(V194_ADAPTER / 'adapter_config.json')
    print('V194 target_modules =', v194_config.get('target_modules'), flush=True)
    print('V194 target_parameters =', v194_config.get('target_parameters'), flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('INIT adapter_config.json exists =', (INIT_ADAPTER_DIR / 'adapter_config.json').exists(), flush=True)
print('INIT adapter_model.safetensors exists =', (INIT_ADAPTER_DIR / 'adapter_model.safetensors').exists(), flush=True)
print('=== V224 RUNTIME ARTIFACT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V224 weak regression forensic analysis.
print('=== V224 FORENSIC ANALYSIS START ===', flush=True)
manifest_path = ANALYSIS_OUT / 'v224_weak_regression_manifest.json'
if not RUN_ANALYSIS:
    print('RUN_ANALYSIS is false; skipping forensic analysis.', flush=True)
else:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/analyze_v224_weak_regression.py'),
        '--v221-batch-summary-json', str(V221_BATCH_SUMMARY_JSON),
        '--v223-report-json', str(V223_WEAK_REPORT_JSON),
        '--output-dir', str(ANALYSIS_OUT),
        '--label', 'v224_weak_regression',
        '--baseline', BASELINE_CANDIDATE,
        '--v223-name', V223_CANDIDATE_NAME,
        '--weak-total-min', str(WEAK_MIN_FOR_FULL),
        '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
        '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
        '--weak-trunc-max', str(WEAK_MAX_TRUNC_FOR_FULL),
    ]
    rc = run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v224_weak_regression.log', check=True)
    print('v224 forensic returncode =', rc, flush=True)
analysis_manifest = read_json(manifest_path)
print('analysis_manifest_path =', manifest_path, flush=True)
print('analysis_decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('analysis_outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('=== V224 FORENSIC ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval, training, and package hard block.
print('=== V224 HARD BLOCK START ===', flush=True)
weak_gate_pass_for_full = False
full_candidate_gate = False
print('V224 is forensic-only; weak eval/model generation is not run here.', flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_EVAL =', RUN_EVAL, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('full_candidate_gate =', full_candidate_gate, flush=True)
print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('No package and no Kaggle submit can be created in V224.', flush=True)
if RUN_TRAIN or RUN_EVAL or RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('V224 hard block violated.')
print('=== V224 HARD BLOCK END ===', flush=True)
"""
        ),
        code(
            """# CELL: write final V224 manifest and roadmap.
print('=== V224 FINAL MANIFEST START ===', flush=True)
final_manifest_path = OUT_ROOT / 'v224_weak_regression_forensic_manifest.json'
analysis_manifest = read_json(ANALYSIS_OUT / 'v224_weak_regression_manifest.json')
decision = analysis_manifest.get('decision', {})
final_manifest = {
    'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'version': VERSION,
    'repo_branch': REPO_BRANCH,
    'repo_commit': repo_commit,
    'v221_batch_summary_json': str(V221_BATCH_SUMMARY_JSON),
    'v223_weak_report_json': str(V223_WEAK_REPORT_JSON),
    'baseline_candidate': BASELINE_CANDIDATE,
    'v223_candidate_name': V223_CANDIDATE_NAME,
    'weak_gate_pass_for_full': False,
    'full_candidate_gate': False,
    'submit_authorized': False,
    'decision': decision,
    'analysis_outputs': analysis_manifest.get('outputs', {}),
    'roadmap_next': decision.get('next_action', 'Inspect V224 outputs before creating V225.'),
}
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('roadmap_next =', final_manifest['roadmap_next'], flush=True)
print('=== V224 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")


if __name__ == "__main__":
    main()

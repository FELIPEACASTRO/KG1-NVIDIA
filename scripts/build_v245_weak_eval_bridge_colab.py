#!/usr/bin/env python3
"""Build the V245 weak-eval CSV bridge Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    colab_url = (
        "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
        "v230-v226-complementarity/notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb"
    )
    github_url = (
        "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
        "v230-v226-complementarity/notebooks/KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb"
    )
    cells = [
        markdown_cell(
            "# KG1 V245 Weak Eval Bridge\n\n"
            f"Colab URL: {colab_url}\n\n"
            f"GitHub URL: {github_url}\n\n"
            "This notebook publishes the exact 315-row weak eval CSV to the HF "
            "dataset so later V245 remote eval can evaluate V244 adapters without "
            "mounting Google Drive. It does not train, run vLLM, package, or submit."
        ),
        code_cell(
            """# CELL: mount Google Drive.
print('=== V245 DRIVE MOUNT START ===', flush=True)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('drive_mount_error =', repr(exc), flush=True)
    raise
print('=== V245 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: global configuration, hard locks, and bridge gates.
print('=== V245 CONFIG START ===', flush=True)

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone

VERSION = 'V245_WEAK_EVAL_BRIDGE_20260510'
REPO_URL = 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git'
REPO_BRANCH = 'v230-v226-complementarity'
ROOT = pathlib.Path('/content/kg1')
OUT_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V245/output_v245_weak_eval_bridge')
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = os.environ.get('KG1_V245_RUN_ID', 'v245-weak-eval-bridge-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))

HF_DATASET_REPO = os.environ.get('KG1_V245_HF_DATASET_REPO', 'felipesp1983/kg1-nemotron-training')
HF_UPLOAD_PREFIX = os.environ.get('KG1_V245_HF_UPLOAD_PREFIX', 'runtime_artifacts/v245_weak_eval_bridge')
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = 'bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff'
EXPECTED_TRAIN_SHA256 = 'a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984'
EXPECTED_VAL_SHA256 = '65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451'
MIN_TRAIN_EXAMPLES = 10206
MIN_VAL_EXAMPLES = 681

V221_WEAK_CSV = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab/v221_weak_315.csv')
V194_VAL_CSV = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv')
V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
BRIDGE_WEAK_CSV = OUT_ROOT / 'v221_weak_315.csv'
BRIDGE_MANIFEST_JSON = OUT_ROOT / 'v245_weak_eval_bridge_manifest.json'

RUN_UPLOAD_TO_HF = os.environ.get('KG1_V245_RUN_UPLOAD_TO_HF', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
ALLOW_KAGGLE_SUBMIT = False
TOKENIZE_ONLY_DRY_RUN = 'v245_cpu_only_bridge_no_training'
RUN_TRAIN = os.environ.get('KG1_V245_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_EVAL = False
RUN_FULL_IF_GATE = False
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True
INIT_ADAPTER_DIR = 'not_applicable_v245_cpu_only_bridge'
weak_gate_pass_for_full = False
WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('RUN_ID =', RUN_ID, flush=True)
print('HF_DATASET_REPO =', HF_DATASET_REPO, flush=True)
print('HF_UPLOAD_PREFIX =', HF_UPLOAD_PREFIX, flush=True)
print('EXPECTED_SHARED_ROW_CONTRACT_SHA256 =', EXPECTED_SHARED_ROW_CONTRACT_SHA256, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('V221_WEAK_CSV =', V221_WEAK_CSV, flush=True)
print('V194_VAL_CSV =', V194_VAL_CSV, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('BRIDGE_WEAK_CSV =', BRIDGE_WEAK_CSV, flush=True)
print('BRIDGE_MANIFEST_JSON =', BRIDGE_MANIFEST_JSON, flush=True)
print('RUN_UPLOAD_TO_HF =', RUN_UPLOAD_TO_HF, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V245.')
if RUN_TRAIN or RUN_EVAL or RUN_FULL_IF_GATE:
    raise RuntimeError('V245 bridge must not train, run model eval, or run full eval.')
print('=== V245 CONFIG END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: helper functions with command logging, hashes, and HF token loading.
print('=== V245 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd, cwd=None, log_path=None, check=True, timeout_s=300):
    printable = ' '.join(str(x) for x in cmd)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or os.getcwd(), flush=True)
    print('+', printable, flush=True)
    print('timeout_s =', timeout_s, flush=True)
    if log_path:
        print('log_path =', log_path, flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    if log_path:
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(log_path).write_text(proc.stdout, encoding='utf-8')
    print(proc.stdout[-6000:], flush=True)
    print('returncode =', proc.returncode, flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and proc.returncode:
        raise RuntimeError('command failed rc=' + str(proc.returncode) + ': ' + printable)
    return proc.returncode


def load_hf_token_if_available():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or ''
    try:
        from google.colab import userdata
        for name in ['HF_TOKEN', 'HF_KEY', 'HUGGING_FACE_HUB_TOKEN']:
            if not token:
                value = userdata.get(name)
                if value:
                    token = value
                    print('loaded HF token from Colab secret:', name, flush=True)
    except Exception as exc:
        print('Colab userdata probe skipped:', type(exc).__name__, flush=True)
    if token:
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
    return token


HF_TOKEN = load_hf_token_if_available()
print('hf_token_available =', bool(HF_TOKEN), flush=True)
print('=== V245 HELPERS END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: CPU runtime, data hash, and optional V194 adapter audit.
print('=== V245 RUNTIME DATA AUDIT START ===', flush=True)
runtime_probe = (
    "import json, shutil, torch; "
    "props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
    "print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), "
    "'cuda_available': torch.cuda.is_available(), "
    "'gpu_name': props.name if props else '', "
    "'gpu_total_gib': props.total_memory/1024**3 if props else 0.0, "
    "'content_free_gib': shutil.disk_usage('/content').free/1024**3}))"
)
runtime_log = OUT_ROOT / 'verify_runtime.jsonl'
run_cmd([sys.executable, '-c', runtime_probe], cwd='/content', log_path=runtime_log, check=True, timeout_s=120)
runtime_info = json.loads([line for line in runtime_log.read_text(encoding='utf-8').splitlines() if line.strip()][-1])
cuda_available = bool(runtime_info.get('cuda_available'))
gpu_total_gib = float(runtime_info.get('gpu_total_gib') or 0.0)
content_free_gib = float(runtime_info.get('content_free_gib') or 0.0)
print('cuda_available =', cuda_available, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
for module_name in ['causal_conv1d', 'mamba_ssm']:
    try:
        __import__(module_name)
        print(module_name, 'optional_import = ok', flush=True)
    except Exception as exc:
        print(module_name, 'optional_import_absent_cpu_only_bridge =', repr(exc), flush=True)

train_path = ROOT / 'data' / 'v217' / 'v217_short_answer_train.jsonl'
val_path = ROOT / 'data' / 'v217' / 'v217_short_answer_val.jsonl'
print('train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('val_path =', val_path, 'exists =', val_path.exists(), flush=True)
if train_path.exists() and val_path.exists():
    observed_train_sha256 = sha256_file(train_path)
    observed_val_sha256 = sha256_file(val_path)
    print('observed_train_sha256 =', observed_train_sha256, flush=True)
    print('observed_val_sha256 =', observed_val_sha256, flush=True)
    if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError('train sha256 mismatch')
    if observed_val_sha256 != EXPECTED_VAL_SHA256:
        raise RuntimeError('validation sha256 mismatch')
else:
    print('V217 hash files are not needed before repo clone; repo preflight will run next.', flush=True)

adapter_config_json = V194_ADAPTER / 'adapter_config.json'
adapter_model_safetensors = V194_ADAPTER / 'adapter_model.safetensors'
print('V194 adapter_config.json =', adapter_config_json, 'exists =', adapter_config_json.exists(), flush=True)
print('V194 adapter_model.safetensors =', adapter_model_safetensors, 'exists =', adapter_model_safetensors.exists(), flush=True)
if adapter_config_json.exists():
    adapter_cfg = json.loads(adapter_config_json.read_text(encoding='utf-8'))
    target_modules = adapter_cfg.get('target_modules')
    target_parameters = adapter_cfg.get('target_parameters')
    print('target_modules =', target_modules, flush=True)
    print('target_parameters =', target_parameters, flush=True)
else:
    print('V194 adapter is optional for V245 bridge; adapter_config.json not required for upload.', flush=True)
print('=== V245 RUNTIME DATA AUDIT END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: clone repo, compile scripts, self-test bridge, and run notebook gate.
print('=== V245 REPO PREFLIGHT START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True, timeout_s=300)
repo_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
print('repo_commit =', repo_commit, flush=True)

train_path = ROOT / 'data' / 'v217' / 'v217_short_answer_train.jsonl'
val_path = ROOT / 'data' / 'v217' / 'v217_short_answer_val.jsonl'
print('post_clone_train_path =', train_path, 'exists =', train_path.exists(), flush=True)
print('post_clone_val_path =', val_path, 'exists =', val_path.exists(), flush=True)
if not train_path.exists() or not val_path.exists():
    raise FileNotFoundError('V217 train/val files are required for release-gate data hash audit.')
observed_train_sha256 = sha256_file(train_path)
observed_val_sha256 = sha256_file(val_path)
print('post_clone_observed_train_sha256 =', observed_train_sha256, flush=True)
print('post_clone_observed_val_sha256 =', observed_val_sha256, flush=True)
if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
    raise RuntimeError('train sha256 mismatch')
if observed_val_sha256 != EXPECTED_VAL_SHA256:
    raise RuntimeError('validation sha256 mismatch')

compile_targets = [
    ROOT / 'src' / 'competition_utils.py',
    ROOT / 'scripts' / 'upload_v245_weak_csv_bridge_to_hf.py',
    ROOT / 'scripts' / 'notebook_release_gate.py',
]
for target in compile_targets:
    print('compile_target =', target, 'exists =', target.exists(), flush=True)
    if not target.exists():
        raise FileNotFoundError(target)
    run_cmd([sys.executable, '-m', 'py_compile', str(target)], cwd=ROOT, log_path=OUT_ROOT / ('py_compile_' + target.name + '.log'), check=True, timeout_s=180)

run_cmd([sys.executable, str(ROOT / 'scripts' / 'upload_v245_weak_csv_bridge_to_hf.py'), '--self-test'], cwd=ROOT, log_path=OUT_ROOT / 'v245_bridge_self_test.log', check=True, timeout_s=180)
run_cmd([sys.executable, str(ROOT / 'scripts' / 'notebook_release_gate.py'), str(ROOT / 'notebooks' / 'KG1_V245_WEAK_EVAL_BRIDGE_COLAB.ipynb')], cwd=ROOT, log_path=OUT_ROOT / 'notebook_release_gate.log', check=True, timeout_s=180)
print('=== V245 REPO PREFLIGHT END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: resolve exact weak CSV from Drive or reconstruct from exact V194 validation CSV.
print('=== V245 WEAK CSV RESOLUTION START ===', flush=True)
import pandas as pd

source_csv = None
if V221_WEAK_CSV.exists():
    source_csv = V221_WEAK_CSV
    print('weak_csv_source = existing_v221_weak_csv', flush=True)
elif V194_VAL_CSV.exists():
    print('weak_csv_source = reconstruct_from_v194_validation_csv', flush=True)
    from src.competition_utils import classify_puzzle
    full_df = pd.read_csv(V194_VAL_CSV)
    if 'prompt' not in full_df.columns or 'answer' not in full_df.columns:
        raise RuntimeError('V194 validation CSV must contain prompt and answer columns.')
    if 'type' not in full_df.columns:
        full_df['type'] = full_df['prompt'].map(classify_puzzle)
    weak_df = full_df[full_df['type'].isin(['equation_transform', 'bit_manipulation'])].copy()
    weak_df.to_csv(BRIDGE_WEAK_CSV, index=False)
    source_csv = BRIDGE_WEAK_CSV
else:
    raise FileNotFoundError('Neither V221 weak CSV nor V194 validation CSV exists: ' + str(V221_WEAK_CSV) + ' ; ' + str(V194_VAL_CSV))

weak_df = pd.read_csv(source_csv)
family_col = 'type' if 'type' in weak_df.columns else 'family'
family_counts = weak_df[family_col].astype(str).value_counts().sort_index().to_dict()
print('source_csv =', source_csv, flush=True)
print('source_csv_exists =', source_csv.exists(), flush=True)
print('source_csv_sha256 =', sha256_file(source_csv), flush=True)
print('weak_rows =', len(weak_df), flush=True)
print('family_counts =', json.dumps(family_counts, indent=2, sort_keys=True), flush=True)
if len(weak_df) != 315:
    raise RuntimeError('V245 weak CSV must have 315 rows.')
if family_counts != {'bit_manipulation': 160, 'equation_transform': 155}:
    raise RuntimeError('V245 weak CSV family counts mismatch: ' + json.dumps(family_counts, sort_keys=True))
print('=== V245 WEAK CSV RESOLUTION END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: validate weak row contract and upload bridge payload to Hugging Face dataset.
print('=== V245 HF UPLOAD START ===', flush=True)
if RUN_UPLOAD_TO_HF and not HF_TOKEN:
    raise RuntimeError('HF_TOKEN is required for RUN_UPLOAD_TO_HF=1. Add HF_TOKEN to Colab Secrets.')

cmd = [
    sys.executable,
    str(ROOT / 'scripts' / 'upload_v245_weak_csv_bridge_to_hf.py'),
    '--source-csv', str(source_csv),
    '--hf-dataset-repo', HF_DATASET_REPO,
    '--path-prefix', HF_UPLOAD_PREFIX,
    '--run-id', RUN_ID,
    '--expected-shared-row-contract-sha256', EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    '--output-manifest-json', str(BRIDGE_MANIFEST_JSON),
]
if not RUN_UPLOAD_TO_HF:
    cmd.append('--dry-run')
run_cmd(cmd, cwd=ROOT, log_path=OUT_ROOT / 'v245_hf_upload.log', check=True, timeout_s=300)
manifest = json.loads(BRIDGE_MANIFEST_JSON.read_text(encoding='utf-8'))
print('bridge_manifest =', json.dumps(manifest, indent=2, sort_keys=True), flush=True)
print('bridge_manifest_sha256 =', sha256_file(BRIDGE_MANIFEST_JSON), flush=True)
print('hf_weak_csv_path =', manifest.get('uploaded_files', {}).get('weak_csv', ''), flush=True)
print('=== V245 HF UPLOAD END ===', flush=True)
"""
        ),
        code_cell(
            """# CELL: final manifest and next-action lock.
print('=== V245 FINAL MANIFEST START ===', flush=True)
final_manifest = {
    'version': VERSION,
    'run_id': RUN_ID,
    'repo_commit': repo_commit,
    'source_csv': str(source_csv),
    'bridge_manifest_json': str(BRIDGE_MANIFEST_JSON),
    'bridge_manifest_sha256': sha256_file(BRIDGE_MANIFEST_JSON),
    'hf_dataset_repo': HF_DATASET_REPO,
    'hf_weak_csv_path': manifest.get('uploaded_files', {}).get('weak_csv', ''),
    'observed_shared_row_contract_sha256': manifest.get('canonical_weak_csv', {}).get('observed_shared_row_contract_sha256', ''),
    'blocked_actions': ['train', 'vllm_eval', 'full_eval', 'package', 'kaggle_submit'],
    'next_action': 'Launch V245 remote weak eval for V244 final/checkpoint-4/checkpoint-2 only after this bridge upload succeeds.',
}
final_manifest_path = OUT_ROOT / 'v245_weak_eval_bridge_final_manifest.json'
final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding='utf-8')
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_manifest =', json.dumps(final_manifest, indent=2, sort_keys=True), flush=True)
print('No training, no vLLM eval, no full eval, no package, and no Kaggle submit were run in V245 bridge.', flush=True)
print('=== V245 FINAL MANIFEST END ===', flush=True)
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "include_colab_link": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"Colab URL: {colab_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

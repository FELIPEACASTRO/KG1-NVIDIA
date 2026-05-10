#!/usr/bin/env python3
"""Build the V230 V226 complementarity analysis Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb")
BRANCH = "v230-v226-complementarity"
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb"
)
GITHUB_URL = (
    "https://github.com/FELIPEACASTRO/KG1-NVIDIA/blob/"
    f"{BRANCH}/notebooks/KG1_V230_V226_COMPLEMENTARITY_COLAB.ipynb"
)
TRAIN_SHA = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
VAL_SHA = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"

_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"v230-{prefix}-{_CELL_COUNTER:02d}"


def _subst(source: str) -> str:
    return (
        source.replace("__BRANCH__", BRANCH)
        .replace("__COLAB_URL__", COLAB_URL)
        .replace("__GITHUB_URL__", GITHUB_URL)
        .replace("__TRAIN_SHA__", TRAIN_SHA)
        .replace("__VAL_SHA__", VAL_SHA)
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
            """# KG1 V230 V226 Complementarity Colab

Purpose: run a CPU-only weak-result complementarity analysis around the known V226 best checkpoint before spending more GPU on training or full eval. V230 reads existing V221 and V226 prediction artifacts, simulates deployable family routers and row-level oracles, writes miss packs, and keeps full eval, packaging, and Kaggle submit blocked.

Gate objective remains explicit: `193/315` total, `60/155` equation_transform, `133/160` bit_manipulation, and no more than `3` truncations.

Colab: __COLAB_URL__

GitHub: __GITHUB_URL__
"""
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V230 DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V230 DRIVE MOUNT END ===', flush=True)
"""
        ),
        code(
            """# CELL: global configuration, gates, and hard submit lock.
print('=== V230 CONFIG START ===', flush=True)
import datetime
import csv
import hashlib
import importlib
import json
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('KG1_ALLOW_VLLM_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_MOE_USE_DEEP_GEMM', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM_E8M0', '0')
os.environ.setdefault('VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES', '0')
os.environ.setdefault('VLLM_DEEP_GEMM_WARMUP', 'skip')
os.environ.setdefault('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS', '0')

VERSION = 'V230_V226_COMPLEMENTARITY_20260510'
REPO_URL = os.environ.get('KG1_REPO_URL', 'https://github.com/FELIPEACASTRO/KG1-NVIDIA.git')
REPO_BRANCH = os.environ.get('KG1_REPO_BRANCH', '__BRANCH__')
ROOT = pathlib.Path('/content/kg1')

DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V230')
OUT_ROOT = DRIVE_ROOT / 'output_v230_v226_complementarity'
RUN_ID = os.environ.get('KG1_V230_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
ANALYSIS_OUT = OUT_ROOT / 'analysis_v230_v226_complementarity' / RUN_ID

V221_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get(
    'KG1_V230_V221_BATCH_SUMMARY_JSON',
    '/content/drive/MyDrive/KG1_NVIDIA_V221/output_v221_candidate_registry_weak_ab/eval_v221_candidate_registry_weak_ab/batch_candidate_summary.json',
))
V226_BATCH_SUMMARY_JSON = pathlib.Path(os.environ.get(
    'KG1_V230_V226_BATCH_SUMMARY_JSON',
    '/content/drive/MyDrive/KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/eval_v226_checkpoint_sweep/batch_candidate_summary.json',
))
V229_ANALYSIS_MANIFEST_JSON = pathlib.Path(os.environ.get(
    'KG1_V230_V229_ANALYSIS_MANIFEST_JSON',
    '/content/drive/MyDrive/KG1_NVIDIA_V229/output_v229_v227_only_fast_eval/analysis_v229_v227_only_fast/v229_v227_only_fast_eval_manifest.json',
))
EXPECTED_REPO_COMMIT = os.environ.get('KG1_V230_EXPECTED_REPO_COMMIT', '').strip()

V194_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter')
V217_ADAPTER = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V217/output_v217_short_answer_rescue/train_v217_shortans_lr1e8_s16/final_adapter')
V226_BEST_CHECKPOINT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V226/output_v226_equation_checkpoint_sweep/train_v226_v194_micro_lr2e9_s6/checkpoint-1')
INIT_ADAPTER_DIR = V226_BEST_CHECKPOINT

EXPECTED_TRAIN_SHA256 = '__TRAIN_SHA__'
EXPECTED_VAL_SHA256 = '__VAL_SHA__'
EXPECTED_TRAIN_ROWS = 10206
EXPECTED_VAL_ROWS = 681
MIN_TRAIN_EXAMPLES = EXPECTED_TRAIN_ROWS
MIN_VAL_EXAMPLES = EXPECTED_VAL_ROWS
EXPECTED_JSONL_KEYS = ['answer', 'family', 'id', 'messages', 'metadata', 'prompt', 'source', 'subcategory']
EXPECTED_TRAIN_FAMILY_COUNTS = {
    'bit_manipulation': 2695,
    'equation_transform': 6935,
    'gravity_constant': 144,
    'numeral_system': 144,
    'text_encryption': 144,
    'unit_conversion': 144,
}
EXPECTED_VAL_FAMILY_COUNTS = {
    'bit_manipulation': 164,
    'equation_transform': 453,
    'gravity_constant': 16,
    'numeral_system': 16,
    'text_encryption': 16,
    'unit_conversion': 16,
}
TOKENIZE_ONLY_DRY_RUN = True
MAX_PROMPT_TRUNCATION_RATE = 0.0
REQUIRE_OFFSET_MASK = True

EXPECTED_V194_ADAPTER_BYTES = 4259069440
EXPECTED_V194_ADAPTER_TENSOR_COUNT = 12011
MIN_V217_ADAPTER_BYTES = 4250000000
MIN_V217_ADAPTER_TENSOR_COUNT = 12000
MIN_V226_CHECKPOINT_BYTES = 4250000000
MIN_V226_CHECKPOINT_TENSOR_COUNT = 12000
EXPECTED_V194_TARGET_MODULES = ['k_proj', 'up_proj', 'down_proj', 'out_proj', 'v_proj', 'q_proj', 'lm_head', 'o_proj', 'in_proj']
EXPECTED_V194_TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']

RUN_TRAIN = os.environ.get('KG1_V230_RUN_TRAIN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_ANALYSIS = os.environ.get('KG1_V230_RUN_ANALYSIS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
ALLOW_V226_SUMMARY_SYNTHESIS = os.environ.get('KG1_V230_ALLOW_V226_SUMMARY_SYNTHESIS', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
RUN_FULL_IF_GATE = False
ALLOW_KAGGLE_SUBMIT = False

WEAK_MIN_FOR_FULL = 193
WEAK_EQ_MIN_FOR_FULL = 60
WEAK_BIT_MIN_FOR_FULL = 133
WEAK_MAX_TRUNC_FOR_FULL = 3
KNOWN_V226_WEAK_TOTAL = 191
FULL_MIN_CANDIDATE = 831
FULL_MAX_TRUNC = 4

for path in [DRIVE_ROOT, OUT_ROOT, ANALYSIS_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('VERSION =', VERSION, flush=True)
print('REPO_URL =', REPO_URL, flush=True)
print('REPO_BRANCH =', REPO_BRANCH, flush=True)
print('ROOT =', ROOT, flush=True)
print('OUT_ROOT =', OUT_ROOT, flush=True)
print('RUN_ID =', RUN_ID, flush=True)
print('ANALYSIS_OUT =', ANALYSIS_OUT, flush=True)
print('EXPECTED_REPO_COMMIT =', EXPECTED_REPO_COMMIT, flush=True)
print('V221_BATCH_SUMMARY_JSON =', V221_BATCH_SUMMARY_JSON, flush=True)
print('V226_BATCH_SUMMARY_JSON =', V226_BATCH_SUMMARY_JSON, flush=True)
print('V229_ANALYSIS_MANIFEST_JSON =', V229_ANALYSIS_MANIFEST_JSON, flush=True)
print('V194_ADAPTER =', V194_ADAPTER, flush=True)
print('V217_ADAPTER =', V217_ADAPTER, flush=True)
print('V226_BEST_CHECKPOINT =', V226_BEST_CHECKPOINT, flush=True)
print('INIT_ADAPTER_DIR =', INIT_ADAPTER_DIR, flush=True)
print('EXPECTED_TRAIN_SHA256 =', EXPECTED_TRAIN_SHA256, flush=True)
print('EXPECTED_VAL_SHA256 =', EXPECTED_VAL_SHA256, flush=True)
print('EXPECTED_TRAIN_ROWS =', EXPECTED_TRAIN_ROWS, flush=True)
print('EXPECTED_VAL_ROWS =', EXPECTED_VAL_ROWS, flush=True)
print('MIN_TRAIN_EXAMPLES =', MIN_TRAIN_EXAMPLES, flush=True)
print('MIN_VAL_EXAMPLES =', MIN_VAL_EXAMPLES, flush=True)
print('EXPECTED_TRAIN_FAMILY_COUNTS =', json.dumps(EXPECTED_TRAIN_FAMILY_COUNTS, sort_keys=True), flush=True)
print('EXPECTED_VAL_FAMILY_COUNTS =', json.dumps(EXPECTED_VAL_FAMILY_COUNTS, sort_keys=True), flush=True)
print('TOKENIZE_ONLY_DRY_RUN =', TOKENIZE_ONLY_DRY_RUN, flush=True)
print('MAX_PROMPT_TRUNCATION_RATE =', MAX_PROMPT_TRUNCATION_RATE, flush=True)
print('REQUIRE_OFFSET_MASK =', REQUIRE_OFFSET_MASK, flush=True)
print('tokenization_offset_mask_contract = not_applicable_for_v230_cpu_only_artifact_analysis', flush=True)
print('RUN_TRAIN =', RUN_TRAIN, flush=True)
print('RUN_ANALYSIS =', RUN_ANALYSIS, flush=True)
print('ALLOW_V226_SUMMARY_SYNTHESIS =', ALLOW_V226_SUMMARY_SYNTHESIS, flush=True)
print('RUN_FULL_IF_GATE =', RUN_FULL_IF_GATE, flush=True)
print('ALLOW_KAGGLE_SUBMIT =', ALLOW_KAGGLE_SUBMIT, flush=True)
print('WEAK_MIN_FOR_FULL =', WEAK_MIN_FOR_FULL, flush=True)
print('WEAK_EQ_MIN_FOR_FULL =', WEAK_EQ_MIN_FOR_FULL, flush=True)
print('WEAK_BIT_MIN_FOR_FULL =', WEAK_BIT_MIN_FOR_FULL, flush=True)
print('WEAK_MAX_TRUNC_FOR_FULL =', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('FULL_MIN_CANDIDATE =', FULL_MIN_CANDIDATE, flush=True)
print('FULL_MAX_TRUNC =', FULL_MAX_TRUNC, flush=True)
if RUN_TRAIN:
    raise RuntimeError('V230 is CPU-only analysis; RUN_TRAIN must stay false.')
if not RUN_ANALYSIS:
    raise RuntimeError('V230 complementarity analysis is mandatory; RUN_ANALYSIS must stay true.')
if ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('Kaggle submission is disabled in V230.')
print('=== V230 CONFIG END ===', flush=True)
"""
        ),
        code(
            """# CELL: helper functions with command logging, hashes, and adapter checks.
print('=== V230 HELPERS START ===', flush=True)

def sha256_file(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def resource_snapshot_line():
    parts = []
    try:
        usage = shutil.disk_usage('/content')
        parts.append(f'content_free_gib={usage.free/1024**3:.1f}')
        parts.append(f'content_total_gib={usage.total/1024**3:.1f}')
    except Exception as exc:
        parts.append(f'disk_probe_error={type(exc).__name__}')
    try:
        gpu_line = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        ).stdout.strip().splitlines()
        if gpu_line:
            parts.append('gpu=[' + gpu_line[0] + ']')
    except Exception as exc:
        parts.append(f'gpu_probe_error={type(exc).__name__}')
    return ' '.join(parts)

def run_cmd(cmd, cwd=None, log_path=None, check=True, heartbeat_s=0, suppress_after_lines=260, timeout_s=None):
    started = time.time()
    printable = ' '.join(str(x) for x in cmd)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd or os.getcwd(), flush=True)
    print('+', printable, flush=True)
    if timeout_s:
        print('timeout_s =', timeout_s, flush=True)
    log_handle = None
    if log_path is not None:
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8', errors='replace')
        print('log_path =', log_path, flush=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines = []
    timed_out = False
    last_heartbeat = time.time()
    assert proc.stdout is not None
    output_queue = queue.Queue()
    stdout_done = object()

    def _reader():
        try:
            for stdout_line in proc.stdout:
                output_queue.put(stdout_line)
        finally:
            output_queue.put(stdout_done)

    reader = threading.Thread(target=_reader, name='kg1-run-cmd-reader', daemon=True)
    reader.start()
    while True:
        try:
            item = output_queue.get(timeout=0.5)
        except queue.Empty:
            item = None
        if item is stdout_done:
            break
        if item is not None:
            line = item
            lines.append(line.rstrip('\\n'))
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
            if len(lines) <= suppress_after_lines:
                print(line, end='', flush=True)
        now = time.time()
        if heartbeat_s and now - last_heartbeat >= heartbeat_s:
            print('[V230 heartbeat] elapsed_s={:.1f} {}'.format(now - started, resource_snapshot_line()), flush=True)
            last_heartbeat = now
        if timeout_s and now - started > timeout_s:
            timed_out = True
            print('timeout_reached =', timeout_s, flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(5)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception as exc:
                print('timeout_kill_warning =', repr(exc), flush=True)
                proc.kill()
            break
        if item is None and proc.poll() is not None and not reader.is_alive():
            break
    returncode = proc.wait()
    if timed_out:
        returncode = -999
    if log_handle:
        log_handle.close()
    if len(lines) > suppress_after_lines:
        print('command_output_suppressed_lines =', len(lines) - suppress_after_lines, flush=True)
    elapsed = time.time() - started
    print('returncode =', returncode, flush=True)
    print('elapsed_s =', round(elapsed, 1), flush=True)
    if returncode != 0:
        print('command_tail_on_failure =', '\\n'.join(lines[-60:]), flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and returncode != 0:
        raise RuntimeError(f'command failed rc={returncode}: {printable}')
    return returncode

def is_complete_adapter_dir(path):
    path = pathlib.Path(path)
    return path.is_dir() and (path / 'adapter_config.json').exists() and (path / 'adapter_model.safetensors').exists()

def resolve_predictions_from_report(report_json):
    report_json = pathlib.Path(report_json)
    report = read_json(report_json)
    direct = report.get('outputs', {}).get('predictions_csv', '')
    if direct:
        direct_path = pathlib.Path(direct)
        if not direct_path.is_absolute():
            direct_path = report_json.parent / direct_path
        if direct_path.exists():
            return direct_path
        raise FileNotFoundError('report predictions_csv does not exist: ' + str(direct_path))
    matches = sorted(report_json.parent.glob('*_predictions.csv'))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError('ambiguous prediction CSV fallback for ' + str(report_json) + ': ' + json.dumps([str(p) for p in matches]))
    raise FileNotFoundError(report_json)

def csv_data_rows(path):
    path = pathlib.Path(path)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    with path.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)

def per_task_counts_from_report(report):
    per_task_csv = pathlib.Path(str(report.get('outputs', {}).get('per_task_csv', '')))
    counts = {'equation_transform_correct': 0, 'bit_manipulation_correct': 0}
    if not per_task_csv.exists():
        return counts
    with per_task_csv.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        for row in csv.DictReader(handle):
            task = str(row.get('task_type') or row.get('family') or '')
            correct = int(float(row.get('correct') or 0))
            if task == 'equation_transform':
                counts['equation_transform_correct'] = correct
            if task == 'bit_manipulation':
                counts['bit_manipulation_correct'] = correct
    return counts

def infer_v226_name(report_path, report):
    adapter = str(report.get('inputs', {}).get('adapter') or report.get('adapter_dir') or '')
    label = str(report.get('label') or report_path.stem.replace('_eval_report', ''))
    correct = int(report.get('correct', 0))

    def name_from_text(text):
        lowered = str(text).lower()
        if 'checkpoint-1' in lowered or 'checkpoint_1' in lowered or 'checkpoint1' in lowered:
            return 'v226_best_checkpoint1_observed_' + str(correct)
        if 'checkpoint-2' in lowered or 'checkpoint_2' in lowered or 'checkpoint2' in lowered:
            return 'v226_checkpoint_2_observed_' + str(correct)
        if 'checkpoint-3' in lowered or 'checkpoint_3' in lowered or 'checkpoint3' in lowered:
            return 'v226_checkpoint_3_observed_' + str(correct)
        return ''

    adapter_name = name_from_text(adapter)
    if adapter_name:
        return adapter_name
    fallback_name = name_from_text(label + ' ' + str(report_path))
    if fallback_name:
        return fallback_name
    return 'v226_report_' + label

def synthesize_batch_summary_from_reports(output_json, source_roots):
    report_paths = []
    for root in source_roots:
        root = pathlib.Path(root)
        print('synthesis_report_root =', root, 'exists =', root.exists(), flush=True)
        if root.exists():
            report_paths.extend(sorted(root.rglob('*_eval_report.json')))
    rows = []
    seen_reports = set()
    for report_path in report_paths:
        report_path = pathlib.Path(report_path)
        if str(report_path) in seen_reports:
            continue
        seen_reports.add(str(report_path))
        try:
            report = read_json(report_path)
            predictions_csv = resolve_predictions_from_report(report_path)
            prediction_rows = csv_data_rows(predictions_csv)
            if prediction_rows != 315:
                print('synthesis_skip_report_rows =', json.dumps({'report_json': str(report_path), 'prediction_rows': prediction_rows}, sort_keys=True), flush=True)
                continue
            adapter = str(report.get('inputs', {}).get('adapter') or report.get('adapter_dir') or '')
            counts = per_task_counts_from_report(report)
            row = {
                'name': infer_v226_name(report_path, report),
                'adapter': adapter,
                'status': 'ok',
                'correct': int(report.get('correct', 0)),
                'accuracy': float(report.get('accuracy', 0.0)),
                'truncated': int(report.get('truncated', 0)),
                'truncation_rate': float(report.get('truncation_rate', 0.0)),
                'equation_transform_correct': counts['equation_transform_correct'],
                'bit_manipulation_correct': counts['bit_manipulation_correct'],
                'completion_tokens': int(report.get('completion_tokens', 0)),
                'tokens_per_second': float(report.get('tokens_per_second', 0.0)),
                'report_json': str(report_path),
                'error': '',
            }
            rows.append(row)
            print('synthesis_candidate_row =', json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            print('synthesis_report_skip =', json.dumps({'report_json': str(report_path), 'error': repr(exc)}, sort_keys=True), flush=True)
    if not rows:
        raise FileNotFoundError('No usable V226 315-row eval reports found under: ' + ', '.join(str(p) for p in source_roots))
    rows = sorted(rows, key=lambda item: (int(item.get('correct', 0)), -int(item.get('truncated', 999999)), str(item.get('name', ''))), reverse=True)
    payload = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': 'v230_synthesized_from_individual_v226_eval_reports',
        'rows': rows,
    }
    write_json(output_json, payload)
    print('synthesized_batch_summary_json =', output_json, flush=True)
    print('synthesized_batch_summary_rows =', len(rows), flush=True)
    return pathlib.Path(output_json)

def resolve_existing_or_synthesize_v226_batch_summary(path):
    path = pathlib.Path(path)
    if path.exists():
        print('using_existing_v226_batch_summary_json =', path, flush=True)
        return path
    output_root = V226_BEST_CHECKPOINT.parents[1]
    candidates = [
        output_root / 'eval_v226_checkpoint_sweep' / 'batch_candidate_summary.json',
        output_root / 'eval_v226_checkpoint1_weak' / 'batch_candidate_summary.json',
        output_root / 'eval_v226_checkpoint2_weak' / 'batch_candidate_summary.json',
        output_root / 'eval_v226_checkpoint3_weak' / 'batch_candidate_summary.json',
    ]
    for candidate in candidates:
        print('v226_batch_summary_candidate =', candidate, 'exists =', candidate.exists(), flush=True)
        if candidate.exists():
            return candidate
    if not ALLOW_V226_SUMMARY_SYNTHESIS:
        raise FileNotFoundError('V226 batch summary missing and synthesis disabled: ' + str(path))
    return synthesize_batch_summary_from_reports(
        OUT_ROOT / 'v226_synthesized_batch_candidate_summary.json',
        [
            output_root / 'eval_v226_checkpoint_sweep',
            output_root / 'eval_v226_checkpoint1_weak',
            output_root / 'eval_v226_checkpoint2_weak',
            output_root / 'eval_v226_checkpoint3_weak',
            output_root,
        ],
    )

print('=== V230 HELPERS END ===', flush=True)
"""
        ),
        code(
            """# CELL: clone repo, compile scripts, and validate static data hashes.
print('=== V230 REPO SETUP START ===', flush=True)
if ROOT.exists():
    shutil.rmtree(ROOT)
run_cmd(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, str(ROOT)], cwd='/content', log_path=OUT_ROOT / 'repo_clone.log', check=True, timeout_s=300)
if EXPECTED_REPO_COMMIT:
    run_cmd(['git', 'fetch', '--depth', '1', 'origin', EXPECTED_REPO_COMMIT], cwd=ROOT, log_path=OUT_ROOT / 'repo_fetch_expected_commit.log', check=True, timeout_s=300)
    run_cmd(['git', 'checkout', '--detach', EXPECTED_REPO_COMMIT], cwd=ROOT, log_path=OUT_ROOT / 'repo_checkout_expected_commit.log', check=True, timeout_s=120)
repo_rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)
print('repo_rev_parse_returncode =', repo_rev.returncode, flush=True)
print('repo_rev_parse_output =', repo_rev.stdout.strip(), flush=True)
if repo_rev.returncode != 0:
    raise RuntimeError('git rev-parse HEAD failed')
repo_commit = repo_rev.stdout.strip()
print('repo_commit =', repo_commit, flush=True)
if EXPECTED_REPO_COMMIT and repo_commit != EXPECTED_REPO_COMMIT:
    raise RuntimeError('repo commit mismatch after checkout')
compile_targets = [
    ROOT / 'src/competition_utils.py',
    ROOT / 'scripts/analyze_v230_v226_complementarity.py',
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
observed_train_sha256 = sha256_file(train_path)
observed_val_sha256 = sha256_file(val_path)
print('observed_train_sha256 =', observed_train_sha256, flush=True)
print('observed_val_sha256 =', observed_val_sha256, flush=True)
if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
    raise RuntimeError('train sha256 mismatch')
if observed_val_sha256 != EXPECTED_VAL_SHA256:
    raise RuntimeError('validation sha256 mismatch')

def inspect_short_answer_jsonl(path, expected_rows, expected_family_counts, split_name):
    print(split_name, 'jsonl_audit_start =', path, flush=True)
    rows = 0
    ids = set()
    duplicate_ids = 0
    family_counts = {}
    source_counts = {}
    assistant_mismatch = 0
    bad_rows = []
    with pathlib.Path(path).open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            rows += 1
            try:
                item = json.loads(line)
            except Exception as exc:
                bad_rows.append({'line': line_no, 'error': 'json_parse', 'detail': repr(exc)})
                continue
            if sorted(item.keys()) != EXPECTED_JSONL_KEYS:
                bad_rows.append({'line': line_no, 'error': 'schema', 'keys': sorted(item.keys())})
            row_id = str(item.get('id', ''))
            if row_id in ids:
                duplicate_ids += 1
            ids.add(row_id)
            family = str(item.get('family', ''))
            source = str(item.get('source', ''))
            family_counts[family] = family_counts.get(family, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            answer = str(item.get('answer', '')).strip()
            prompt = str(item.get('prompt', '')).strip()
            messages = item.get('messages')
            if not row_id or not answer or not prompt:
                bad_rows.append({'line': line_no, 'error': 'empty_required_field'})
            if not isinstance(messages, list) or len(messages) < 3:
                bad_rows.append({'line': line_no, 'error': 'messages_shape'})
                continue
            final_message = messages[-1]
            if final_message.get('role') != 'assistant':
                bad_rows.append({'line': line_no, 'error': 'final_message_role', 'role': final_message.get('role')})
                continue
            final_text = str(final_message.get('content', '')).strip()
            expected_final = 'Final answer: ' + answer
            if final_text != expected_final:
                assistant_mismatch += 1
                if len(bad_rows) < 10:
                    bad_rows.append({'line': line_no, 'error': 'assistant_answer_mismatch', 'expected': expected_final, 'observed': final_text})
    summary = {
        'path': str(path),
        'rows': rows,
        'unique_ids': len(ids),
        'duplicate_ids': duplicate_ids,
        'family_counts': family_counts,
        'source_counts': source_counts,
        'assistant_mismatch': assistant_mismatch,
        'bad_rows_first10': bad_rows[:10],
    }
    print(split_name, 'jsonl_audit_summary =', json.dumps(summary, sort_keys=True), flush=True)
    if rows != expected_rows:
        raise RuntimeError(split_name + ' row count mismatch')
    if duplicate_ids:
        raise RuntimeError(split_name + ' duplicate ids found')
    if family_counts != expected_family_counts:
        raise RuntimeError(split_name + ' family counts mismatch')
    if assistant_mismatch:
        raise RuntimeError(split_name + ' assistant final answer mismatch')
    if bad_rows:
        raise RuntimeError(split_name + ' jsonl audit found bad rows')
    return summary

manifest_path = ROOT / 'data/v217/v217_short_answer_manifest.json'
print('manifest_path =', manifest_path, 'exists =', manifest_path.exists(), flush=True)
manifest = read_json(manifest_path)
print('manifest_version =', manifest.get('version'), flush=True)
if manifest.get('train', {}).get('sha256') != EXPECTED_TRAIN_SHA256:
    raise RuntimeError('manifest train sha256 mismatch')
if manifest.get('validation', {}).get('sha256') != EXPECTED_VAL_SHA256:
    raise RuntimeError('manifest validation sha256 mismatch')
train_audit = inspect_short_answer_jsonl(train_path, EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_FAMILY_COUNTS, 'train')
val_audit = inspect_short_answer_jsonl(val_path, EXPECTED_VAL_ROWS, EXPECTED_VAL_FAMILY_COUNTS, 'validation')
print('=== V230 REPO SETUP END ===', flush=True)
"""
        ),
        code(
            """# CELL: runtime, dependency, Drive artifact, and adapter audit.
print('=== V230 RUNTIME ARTIFACT AUDIT START ===', flush=True)
torch_probe_path = OUT_ROOT / 'verify_torch_cuda.jsonl'
run_cmd([
    sys.executable,
    '-c',
    "import json, torch; props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(json.dumps({'torch': getattr(torch, '__version__', 'unknown'), 'cuda_available': torch.cuda.is_available(), 'gpu_name': props.name if props else '', 'gpu_total_gib': props.total_memory/1024**3 if props else 0.0}))",
], cwd='/content', log_path=torch_probe_path, check=True, timeout_s=120)
torch_probe = json.loads([line for line in torch_probe_path.read_text(encoding='utf-8').splitlines() if line.strip()][-1])
cuda_available = bool(torch_probe.get('cuda_available'))
gpu_name = str(torch_probe.get('gpu_name', ''))
gpu_total_gib = float(torch_probe.get('gpu_total_gib', 0.0))
content_free_gib = shutil.disk_usage('/content').free / 1024**3
print('cuda_available =', cuda_available, flush=True)
print('gpu_name =', gpu_name, flush=True)
print('gpu_total_gib =', round(gpu_total_gib, 2), flush=True)
print('content_free_gib =', round(content_free_gib, 2), flush=True)
if not cuda_available:
    print('V230 is CPU-only analysis; CUDA absence is accepted.', flush=True)
elif gpu_total_gib < 70:
    print('GPU memory is below H100/A100 class; accepted because V230 does not run vLLM eval.', flush=True)
for module_name in ['causal_conv1d', 'mamba_ssm', 'vllm']:
    try:
        module = importlib.import_module(module_name)
        print(module_name, 'version =', getattr(module, '__version__', 'unknown'), flush=True)
    except Exception as exc:
        print(module_name, 'import_warning =', repr(exc), flush=True)
print('V230 CPU-only path does not install vLLM and runs after all training has been blocked.', flush=True)

V226_BATCH_SUMMARY_JSON = resolve_existing_or_synthesize_v226_batch_summary(V226_BATCH_SUMMARY_JSON)
print('V221_BATCH_SUMMARY_JSON exists =', V221_BATCH_SUMMARY_JSON.exists(), flush=True)
print('V226_BATCH_SUMMARY_JSON exists =', V226_BATCH_SUMMARY_JSON.exists(), flush=True)
print('V226_BATCH_SUMMARY_JSON resolved =', V226_BATCH_SUMMARY_JSON, flush=True)
print('V229_ANALYSIS_MANIFEST_JSON exists =', V229_ANALYSIS_MANIFEST_JSON.exists(), flush=True)
if not V221_BATCH_SUMMARY_JSON.exists():
    raise FileNotFoundError(V221_BATCH_SUMMARY_JSON)
if not V226_BATCH_SUMMARY_JSON.exists():
    raise FileNotFoundError(V226_BATCH_SUMMARY_JSON)
if not V229_ANALYSIS_MANIFEST_JSON.exists():
    raise FileNotFoundError(V229_ANALYSIS_MANIFEST_JSON)

try:
    from safetensors import safe_open
except Exception:
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], cwd='/content', log_path=OUT_ROOT / 'pip_install_safetensors.log', check=True, timeout_s=300)
    from safetensors import safe_open
for label, adapter_path in [('V194', V194_ADAPTER), ('V217', V217_ADAPTER), ('V226_BEST', V226_BEST_CHECKPOINT), ('INIT', INIT_ADAPTER_DIR)]:
    print(label, 'adapter path =', adapter_path, 'complete =', is_complete_adapter_dir(adapter_path), flush=True)
    if not is_complete_adapter_dir(adapter_path):
        raise RuntimeError(f'{label} adapter incomplete: {adapter_path}')
    cfg = read_json(adapter_path / 'adapter_config.json')
    print(label, 'target_modules =', cfg.get('target_modules'), flush=True)
    print(label, 'target_parameters =', cfg.get('target_parameters'), flush=True)
    if sorted(cfg.get('target_modules') or []) != sorted(EXPECTED_V194_TARGET_MODULES):
        raise RuntimeError(f'{label} target_modules mismatch')
    target_parameters = cfg.get('target_parameters') or []
    if label in {'V194', 'V217'}:
        if sorted(target_parameters) != sorted(EXPECTED_V194_TARGET_PARAMETERS):
            raise RuntimeError(f'{label} target_parameters mismatch')
    elif sorted(target_parameters) != sorted(EXPECTED_V194_TARGET_PARAMETERS):
        print(label, 'target_parameters differ from V194/V217; accepting PEFT checkpoint format.', flush=True)
    weights_path = adapter_path / 'adapter_model.safetensors'
    with safe_open(str(weights_path), framework='pt', device='cpu') as handle:
        tensor_count = len(handle.keys())
    print(label, 'adapter_tensor_count =', tensor_count, flush=True)
    print(label, 'adapter_weight_bytes =', weights_path.stat().st_size, flush=True)
    if label == 'V194':
        if tensor_count != EXPECTED_V194_ADAPTER_TENSOR_COUNT:
            raise RuntimeError('V194 adapter tensor count mismatch')
        if weights_path.stat().st_size != EXPECTED_V194_ADAPTER_BYTES:
            raise RuntimeError('V194 adapter weight size mismatch')
    if label == 'V217':
        if tensor_count < MIN_V217_ADAPTER_TENSOR_COUNT:
            raise RuntimeError('V217 final adapter tensor count below expected floor')
        if weights_path.stat().st_size < MIN_V217_ADAPTER_BYTES:
            raise RuntimeError('V217 final_adapter size mismatch')
    if label in {'V226_BEST', 'INIT'}:
        if tensor_count < MIN_V226_CHECKPOINT_TENSOR_COUNT:
            raise RuntimeError(f'{label} checkpoint tensor count below expected floor')
        if weights_path.stat().st_size < MIN_V226_CHECKPOINT_BYTES:
            raise RuntimeError(f'{label} checkpoint weight size below expected floor')
print('=== V230 RUNTIME ARTIFACT AUDIT END ===', flush=True)
"""
        ),
        code(
            """# CELL: prediction artifact preflight for V221 and V226 weak summaries.
print('=== V230 PREDICTION PREFLIGHT START ===', flush=True)
def summarize_batch_artifacts(summary_json, label):
    summary_json = pathlib.Path(summary_json)
    payload = read_json(summary_json)
    rows = [row for row in payload.get('rows', []) if row.get('status') == 'ok']
    print(label, 'summary_json =', summary_json, flush=True)
    print(label, 'ok_candidate_count =', len(rows), flush=True)
    inspected = []
    for row in rows:
        report_json = pathlib.Path(str(row.get('report_json', '')))
        report_exists = report_json.exists()
        predictions_csv = ''
        predictions_exists = False
        prediction_bytes = 0
        prediction_rows = 0
        prediction_sha256 = ''
        if report_exists:
            try:
                prediction_path = resolve_predictions_from_report(report_json)
                predictions_csv = str(prediction_path)
                predictions_exists = prediction_path.exists()
                prediction_bytes = prediction_path.stat().st_size if predictions_exists else 0
                prediction_rows = csv_data_rows(prediction_path) if predictions_exists else 0
                prediction_sha256 = sha256_file(prediction_path) if predictions_exists else ''
            except Exception as exc:
                predictions_csv = 'resolve_error:' + repr(exc)
        inspected.append({
            'name': row.get('name', ''),
            'status': row.get('status', ''),
            'correct': row.get('correct', ''),
            'equation_transform_correct': row.get('equation_transform_correct', ''),
            'bit_manipulation_correct': row.get('bit_manipulation_correct', ''),
            'truncated': row.get('truncated', ''),
            'report_json': str(report_json),
            'report_exists': report_exists,
            'predictions_csv': predictions_csv,
            'predictions_exists': predictions_exists,
            'prediction_bytes': prediction_bytes,
            'prediction_rows': prediction_rows,
            'prediction_sha256': prediction_sha256,
        })
    for item in inspected:
        print(label, 'candidate_artifact =', json.dumps(item, sort_keys=True), flush=True)
    missing = [item for item in inspected if not item['report_exists'] or not item['predictions_exists']]
    if missing:
        raise RuntimeError(label + ' missing prediction artifacts: ' + json.dumps(missing[:5], sort_keys=True))
    wrong_rows = [item for item in inspected if int(item.get('prediction_rows') or 0) != 315]
    if wrong_rows:
        raise RuntimeError(label + ' prediction CSV row count mismatch: ' + json.dumps(wrong_rows[:5], sort_keys=True))
    return inspected

v221_candidates = summarize_batch_artifacts(V221_BATCH_SUMMARY_JSON, 'V221')
v226_candidates = summarize_batch_artifacts(V226_BATCH_SUMMARY_JSON, 'V226')
if not v221_candidates:
    raise RuntimeError('V221 batch summary has no ok candidates.')
if not v226_candidates:
    raise RuntimeError('V226 batch summary has no ok candidates.')
preferred_v226 = [row for row in v226_candidates if str(row.get('name', '')) == 'v226_best_checkpoint1_observed_191']
if not preferred_v226:
    raise RuntimeError('Required V226 baseline row missing: v226_best_checkpoint1_observed_191')
if int(preferred_v226[0].get('correct') or 0) != KNOWN_V226_WEAK_TOTAL:
    raise RuntimeError('Required V226 baseline correct count mismatch')
known_v226_rows = [row for row in v226_candidates if 'checkpoint' in str(row.get('name', '')).lower()]
print('known_v226_checkpoint_rows =', json.dumps(known_v226_rows[:5], indent=2, sort_keys=True), flush=True)
print('=== V230 PREDICTION PREFLIGHT END ===', flush=True)
"""
        ),
        code(
            """# CELL: run V230 complementarity analyzer.
print('=== V230 COMPLEMENTARITY ANALYSIS START ===', flush=True)
analysis_manifest_path = ANALYSIS_OUT / 'v230_v226_complementarity_manifest.json'
if analysis_manifest_path.exists():
    analysis_manifest_path.unlink()
    print('removed_stale_analysis_manifest =', analysis_manifest_path, flush=True)
weak_gate_pass_for_full = False
cmd = [
    sys.executable,
    str(ROOT / 'scripts/analyze_v230_v226_complementarity.py'),
    '--v221-batch-summary-json', str(V221_BATCH_SUMMARY_JSON),
    '--v226-batch-summary-json', str(V226_BATCH_SUMMARY_JSON),
    '--v229-analysis-manifest-json', str(V229_ANALYSIS_MANIFEST_JSON),
    '--output-dir', str(ANALYSIS_OUT),
    '--label', 'v230_v226_complementarity',
    '--preferred-baseline', 'v226__v226_best_checkpoint1_observed_191',
    '--weak-total-min', str(WEAK_MIN_FOR_FULL),
    '--weak-eq-min', str(WEAK_EQ_MIN_FOR_FULL),
    '--weak-bit-min', str(WEAK_BIT_MIN_FOR_FULL),
    '--weak-trunc-max', str(WEAK_MAX_TRUNC_FOR_FULL),
]
run_cmd(cmd, cwd=ROOT, log_path=ANALYSIS_OUT / 'v230_v226_complementarity.log', check=True, heartbeat_s=30, timeout_s=600)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
deployable_pass = [
    row for row in analysis_manifest.get('router_simulation', [])
    if row.get('deployable_without_row_labels') and row.get('weak_gate_pass_for_full')
]
single_pass = [row for row in analysis_manifest.get('candidate_summary', []) if row.get('weak_gate_pass_for_full')]
row_level_oracle_pass = [
    row for row in analysis_manifest.get('router_simulation', [])
    if (not row.get('deployable_without_row_labels')) and row.get('weak_gate_pass_for_full')
]
weak_gate_pass_for_full = bool(deployable_pass or single_pass)
print('analysis_manifest_path =', analysis_manifest_path, flush=True)
print('analysis_manifest_sha256 =', sha256_file(analysis_manifest_path), flush=True)
print('resolved_baseline =', analysis_manifest.get('resolved_baseline'), flush=True)
print('candidate_source_counts =', json.dumps(analysis_manifest.get('candidate_source_counts', {}), sort_keys=True), flush=True)
print('deployable_weak_gate_pass_for_full =', bool(deployable_pass or single_pass), flush=True)
print('row_level_oracle_gate_pass =', bool(row_level_oracle_pass), flush=True)
print('baseline_summary =', json.dumps(analysis_manifest.get('baseline_summary', {}), indent=2, sort_keys=True), flush=True)
print('decision =', json.dumps(analysis_manifest.get('decision', {}), indent=2, sort_keys=True), flush=True)
print('router_top =', json.dumps(analysis_manifest.get('router_simulation', [])[:5], indent=2, sort_keys=True), flush=True)
print('outputs =', json.dumps(analysis_manifest.get('outputs', {}), indent=2, sort_keys=True), flush=True)
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('=== V230 COMPLEMENTARITY ANALYSIS END ===', flush=True)
"""
        ),
        code(
            """# CELL: full eval/package hard block and final manifest.
print('=== V230 FINAL MANIFEST START ===', flush=True)
if not analysis_manifest_path.exists():
    raise FileNotFoundError(analysis_manifest_path)
analysis_manifest = read_json(analysis_manifest_path)
analysis_manifest_sha256 = sha256_file(analysis_manifest_path)
decision = analysis_manifest.get('decision', {})
if not decision or decision.get('decision') == 'analysis_not_run':
    raise RuntimeError('V230 final manifest requires a completed analysis decision.')
weak_gate_pass_for_full = bool(weak_gate_pass_for_full)
row_level_oracle_gate_pass = any(
    (not row.get('deployable_without_row_labels')) and row.get('weak_gate_pass_for_full')
    for row in analysis_manifest.get('router_simulation', [])
)
full_candidate_gate = False
print('weak_gate_pass_for_full =', weak_gate_pass_for_full, flush=True)
print('row_level_oracle_gate_pass =', row_level_oracle_gate_pass, flush=True)
print('full_candidate_gate =', full_candidate_gate, flush=True)
print('Required weak_total >=', WEAK_MIN_FOR_FULL, 'eq >=', WEAK_EQ_MIN_FOR_FULL, 'bit >=', WEAK_BIT_MIN_FOR_FULL, 'trunc <=', WEAK_MAX_TRUNC_FOR_FULL, flush=True)
print('Full eval is intentionally not automatic in V230 complementarity notebook.', flush=True)
print('No package and no Kaggle submit can be created in V230.', flush=True)
if RUN_FULL_IF_GATE or ALLOW_KAGGLE_SUBMIT:
    raise RuntimeError('V230 hard block violated. Kaggle submission is disabled.')
blocked_artifacts = []
for pattern in ['*.zip', '*submission*.csv', '*kaggle*.json']:
    blocked_artifacts.extend(str(path) for path in OUT_ROOT.glob(pattern))
if blocked_artifacts:
    raise RuntimeError('V230 output contains package/submission-like artifacts: ' + json.dumps(blocked_artifacts, sort_keys=True))
router_rows = analysis_manifest.get('router_simulation', [])
candidate_rows = analysis_manifest.get('candidate_summary', [])
best_deployable_summary = next((row for row in router_rows if row.get('deployable_without_row_labels')), {})
best_oracle_summary = next((row for row in router_rows if not row.get('deployable_without_row_labels')), {})
final_manifest_path = OUT_ROOT / 'v230_v226_complementarity_final_manifest.json'
final_manifest = {
    'version': VERSION,
    'run_id': RUN_ID,
    'repo_branch': REPO_BRANCH,
    'repo_commit': repo_commit,
    'expected_repo_commit': EXPECTED_REPO_COMMIT,
    'analysis_complete': True,
    'weak_gate_pass_for_full': weak_gate_pass_for_full,
    'row_level_oracle_gate_pass': row_level_oracle_gate_pass,
    'full_candidate_gate': full_candidate_gate,
    'allowed_actions': {
        'train': False,
        'full_eval': False,
        'package': False,
        'kaggle_submit': False,
    },
    'decision': decision,
    'baseline_summary': analysis_manifest.get('baseline_summary', {}),
    'best_deployable_summary': best_deployable_summary,
    'best_oracle_summary': best_oracle_summary,
    'observed_shared_row_contract_sha256': analysis_manifest.get('observed_shared_row_contract_sha256', ''),
    'candidate_source_counts': analysis_manifest.get('candidate_source_counts', {}),
    'candidate_count': len(candidate_rows),
    'thresholds': {
        'weak_total': WEAK_MIN_FOR_FULL,
        'weak_equation_transform': WEAK_EQ_MIN_FOR_FULL,
        'weak_bit_manipulation': WEAK_BIT_MIN_FOR_FULL,
        'weak_truncated': WEAK_MAX_TRUNC_FOR_FULL,
        'full_min_candidate': FULL_MIN_CANDIDATE,
        'full_max_trunc': FULL_MAX_TRUNC,
    },
    'known_v226_weak_total': KNOWN_V226_WEAK_TOTAL,
    'analysis_manifest': str(analysis_manifest_path),
    'analysis_manifest_sha256': analysis_manifest_sha256,
    'analysis_out': str(ANALYSIS_OUT),
    'observed_input_artifacts': analysis_manifest.get('load_meta', []),
    'roadmap_next': decision.get('next_action', 'Review V230 complementarity outputs.'),
}
write_json(final_manifest_path, final_manifest)
print('final_manifest_path =', final_manifest_path, flush=True)
print('final_decision =', json.dumps(decision, indent=2, sort_keys=True), flush=True)
print('roadmap_next =', final_manifest['roadmap_next'], flush=True)
print('=== V230 FINAL MANIFEST END ===', flush=True)
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


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")
    print(COLAB_URL)


if __name__ == "__main__":
    main()

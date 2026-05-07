#!/usr/bin/env python3
"""Build the V214 weak-failure forensic Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


BRANCH = "v214-h100-micro-replay"
NOTEBOOK_PATH = Path("notebooks/KG1_V214_WEAK_FAILURE_FORENSIC_COLAB.ipynb")
COLAB_URL = (
    "https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/"
    f"blob/{BRANCH}/notebooks/KG1_V214_WEAK_FAILURE_FORENSIC_COLAB.ipynb"
)


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown(
            """# KG1 V214 Weak Failure Forensic

Reads the completed V214 weak-eval artifacts from Google Drive and writes an
audit report explaining why V214 is rejected. This notebook does not train,
submit, or call an LLM.
""",
            "v214-forensic-title",
        ),
        code(
            """# CELL: mount Google Drive.
print('=== V214 FORENSIC DRIVE MOUNT START ===', flush=True)
from google.colab import drive
drive.mount('/content/drive')
print('=== V214 FORENSIC DRIVE MOUNT END ===', flush=True)
""",
            "v214-forensic-code-01",
        ),
        code(
            f"""# CELL: configuration and path checks.
print('=== V214 FORENSIC CONFIG START ===', flush=True)
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

BRANCH = os.environ.get('KG1_FORENSIC_BRANCH', '{BRANCH}')
ROOT = pathlib.Path('/content/kg1_forensic')
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive/KG1_NVIDIA_V214')
EVAL_OUT = DRIVE_ROOT / 'output_v214_micro_replay' / 'eval_v214_v194_cont_lr3e7_s1'
WEAK_EVAL = EVAL_OUT / 'weak_eval'
PREDICTIONS_CSV = pathlib.Path(os.environ.get(
    'KG1_V214_WEAK_PREDICTIONS_CSV',
    str(WEAK_EVAL / 'v214_micro_weak_predictions.csv'),
))
RAW_PREDICTIONS_CSV = pathlib.Path(os.environ.get(
    'KG1_V214_WEAK_RAW_PREDICTIONS_CSV',
    str(WEAK_EVAL / 'v214_micro_weak_raw_predictions_pre_score.csv'),
))
PER_TASK_CSV = pathlib.Path(os.environ.get(
    'KG1_V214_WEAK_PER_TASK_CSV',
    str(WEAK_EVAL / 'v214_micro_weak_per_task.csv'),
))
REPORT_JSON = pathlib.Path(os.environ.get(
    'KG1_V214_WEAK_REPORT_JSON',
    str(WEAK_EVAL / 'v214_micro_weak_eval_report.json'),
))
BASELINE_PREDICTIONS_CSV = pathlib.Path(os.environ.get(
    'KG1_V194_BASELINE_PREDICTIONS_CSV',
    '',
))
FORENSIC_OUT = pathlib.Path(os.environ.get(
    'KG1_V214_FORENSIC_OUT',
    str(WEAK_EVAL / 'v214_rejected_forensic'),
))

for path in [ROOT, ROOT / 'scripts', ROOT / 'src', FORENSIC_OUT]:
    path.mkdir(parents=True, exist_ok=True)

print('BRANCH =', BRANCH)
print('ROOT =', ROOT)
print('DRIVE_ROOT =', DRIVE_ROOT)
print('EVAL_OUT =', EVAL_OUT)
print('WEAK_EVAL =', WEAK_EVAL)
print('PREDICTIONS_CSV =', PREDICTIONS_CSV, 'exists =', PREDICTIONS_CSV.exists())
print('RAW_PREDICTIONS_CSV =', RAW_PREDICTIONS_CSV, 'exists =', RAW_PREDICTIONS_CSV.exists())
print('PER_TASK_CSV =', PER_TASK_CSV, 'exists =', PER_TASK_CSV.exists())
print('REPORT_JSON =', REPORT_JSON, 'exists =', REPORT_JSON.exists())
print('BASELINE_PREDICTIONS_CSV =', BASELINE_PREDICTIONS_CSV, 'exists =', BASELINE_PREDICTIONS_CSV.exists() if str(BASELINE_PREDICTIONS_CSV) else False)
print('FORENSIC_OUT =', FORENSIC_OUT)

if not PREDICTIONS_CSV.exists():
    raise FileNotFoundError(f'Missing V214 weak predictions CSV: {{PREDICTIONS_CSV}}')
if not REPORT_JSON.exists():
    raise FileNotFoundError(f'Missing V214 weak report JSON: {{REPORT_JSON}}')
print('=== V214 FORENSIC CONFIG END ===', flush=True)
""",
            "v214-forensic-code-02",
        ),
        code(
            """# CELL: helper for explicit command logging.
print('=== V214 FORENSIC HELPERS START ===', flush=True)
import queue
import threading

def run_cmd(cmd, *, cwd=None, log_path=None, check=True, heartbeat_s=30):
    cwd = pathlib.Path(cwd or '.')
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    print('--- COMMAND START ---', flush=True)
    print('cwd =', cwd, flush=True)
    print('+', ' '.join(map(str, cmd)), flush=True)
    if log_path:
        print('log_path =', log_path, flush=True)
    start = time.time()
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    q = queue.Queue()
    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    last_output = time.time()
    with (log_path.open('w', encoding='utf-8') if log_path else open(os.devnull, 'w', encoding='utf-8')) as log:
        while proc.poll() is None or not q.empty():
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                if time.time() - last_output >= heartbeat_s:
                    elapsed = time.time() - start
                    print(f'[V214 forensic heartbeat] elapsed_s={elapsed:.1f}', flush=True)
                    last_output = time.time()
                continue
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            last_output = time.time()
    rc = proc.wait()
    elapsed = time.time() - start
    print('returncode =', rc, flush=True)
    print(f'elapsed_s = {elapsed:.1f}', flush=True)
    print('--- COMMAND END ---', flush=True)
    if check and rc != 0:
        raise RuntimeError(f'Command failed rc={rc}: {cmd}')
    return rc

print('=== V214 FORENSIC HELPERS END ===', flush=True)
""",
            "v214-forensic-code-03",
        ),
        code(
            """# CELL: download forensic script from GitHub and compile it.
print('=== V214 FORENSIC SCRIPT FETCH START ===', flush=True)
import py_compile

RAW_BASE = f'https://raw.githubusercontent.com/FELIPEACASTRO/KG1-NVIDIA/{BRANCH}'
FILES = {
    'scripts/v214_weak_failure_forensic.py': ROOT / 'scripts' / 'v214_weak_failure_forensic.py',
    'src/__init__.py': ROOT / 'src' / '__init__.py',
    'src/competition_utils.py': ROOT / 'src' / 'competition_utils.py',
}
for remote_path, local_path in FILES.items():
    url = f'{RAW_BASE}/{remote_path}'
    print('fetching =', url, flush=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(urllib.request.urlopen(url, timeout=60).read().decode('utf-8'), encoding='utf-8')
    py_compile.compile(str(local_path), doraise=True)
    print('compiled =', local_path, 'bytes =', local_path.stat().st_size, flush=True)

print('=== V214 FORENSIC SCRIPT FETCH END ===', flush=True)
""",
            "v214-forensic-code-04",
        ),
        code(
            """# CELL: run V214 weak failure forensic.
print('=== V214 FORENSIC RUN START ===', flush=True)
cmd = [
    sys.executable,
    str(ROOT / 'scripts' / 'v214_weak_failure_forensic.py'),
    '--predictions-csv', str(PREDICTIONS_CSV),
    '--raw-predictions-csv', str(RAW_PREDICTIONS_CSV),
    '--per-task-csv', str(PER_TASK_CSV),
    '--report-json', str(REPORT_JSON),
    '--output-dir', str(FORENSIC_OUT),
    '--label', 'v214_micro_weak_rejected',
    '--baseline-weak-correct', '190',
    '--baseline-weak-total', '315',
    '--strong-default-correct', '632',
    '--full-gate', '828',
    '--full-preferred', '830',
    '--weak-full-gate', '191',
    '--trunc-gate', '3',
]
if str(BASELINE_PREDICTIONS_CSV) and BASELINE_PREDICTIONS_CSV.exists():
    cmd.extend(['--baseline-predictions-csv', str(BASELINE_PREDICTIONS_CSV)])
rc = run_cmd(
    cmd,
    cwd=ROOT,
    log_path=FORENSIC_OUT / 'v214_weak_failure_forensic.log',
    check=True,
)
print('forensic returncode =', rc, flush=True)
print('=== V214 FORENSIC RUN END ===', flush=True)
""",
            "v214-forensic-code-05",
        ),
        code(
            """# CELL: display final forensic summary and required decision.
print('=== V214 FORENSIC SUMMARY START ===', flush=True)
summary_path = FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_summary.json'
report_path = FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_report.md'
rows_path = FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_rows.csv'
per_type_path = FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_per_type.csv'
buckets_path = FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_failure_buckets.csv'

for path in [summary_path, report_path, rows_path, per_type_path, buckets_path]:
    print('artifact =', path, 'exists =', path.exists(), flush=True)
    if not path.exists():
        raise FileNotFoundError(path)

summary = json.loads(summary_path.read_text(encoding='utf-8'))
print('decision =', summary['decision'], flush=True)
print('v214_correct =', summary['v214_correct'], '/', summary['rows'], flush=True)
print('weak_delta_vs_baseline =', summary['weak_delta_vs_baseline'], flush=True)
print('truncated =', summary['truncated'], '/', summary['rows'], flush=True)
print('max_full_if_strong_default =', summary['max_full_if_strong_default'], flush=True)
print('full_eval_allowed =', summary['full_eval_allowed'], flush=True)
if summary['decision'] != 'REJECT_V214_NO_FULL_EVAL':
    raise RuntimeError('Unexpected decision; review gates manually before any next step.')
print('report preview:')
print('\\n'.join(report_path.read_text(encoding='utf-8').splitlines()[:80]))
print('=== V214 FORENSIC SUMMARY END ===', flush=True)
""",
            "v214-forensic-code-06",
        ),
        code(
            """# CELL: write final forensic manifest.
print('=== V214 FORENSIC MANIFEST START ===', flush=True)
manifest = {
    'generated_at_utc': __import__('datetime').datetime.datetime.now(__import__('datetime').timezone.utc).isoformat(),
    'branch': BRANCH,
    'predictions_csv': str(PREDICTIONS_CSV),
    'report_json': str(REPORT_JSON),
    'forensic_out': str(FORENSIC_OUT),
    'summary_json': str(FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_summary.json'),
    'report_md': str(FORENSIC_OUT / 'v214_micro_weak_rejected_forensic_report.md'),
    'decision': summary['decision'],
    'v214_correct': summary['v214_correct'],
    'rows': summary['rows'],
    'truncated': summary['truncated'],
    'full_eval_allowed': summary['full_eval_allowed'],
}
manifest_path = FORENSIC_OUT / 'v214_weak_failure_forensic_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
print('manifest_path =', manifest_path, flush=True)
print('manifest =', json.dumps(manifest, indent=2, sort_keys=True), flush=True)
print('=== V214 FORENSIC MANIFEST END ===', flush=True)
""",
            "v214-forensic-code-07",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "none",
            "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} bytes={NOTEBOOK_PATH.stat().st_size}")
    print(f"colab_url={COLAB_URL}")
    print("NOTE: the URL works after the notebook is pushed to the referenced branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

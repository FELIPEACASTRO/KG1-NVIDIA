#!/usr/bin/env python3
"""Build a safe Google Drive cleanup Colab notebook for KG1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(".claude/worktrees/competent-shamir/notebooks/KG1_DRIVE_CLEANUP_SAFE_COLAB.ipynb")


_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"drive-cleanup-{prefix}-{_CELL_COUNTER:02d}"


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
            """# KG1 Drive Cleanup Safe Colab

This notebook frees Google Drive storage used by obsolete KG1/NVIDIA runs.

Default behavior is dry-run only. It never deletes anything unless both switches are set:

- `DRY_RUN = False`
- `CONFIRM_DELETE = "DELETE_KG1_UNUSED_ARTIFACTS"`

It keeps the current solution-critical artifacts:

- V194 rank-19 init adapter used as baseline;
- V206B failed final adapter used as forensic delta source for V206C;
- V206C current output folder.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            """import hashlib
import json
import os
import pathlib
import shutil
import time

VERSION = 'KG1_DRIVE_CLEANUP_SAFE_20260506'
DRIVE_ROOT = pathlib.Path('/content/drive/MyDrive')

DRY_RUN = True
CONFIRM_DELETE = ''

# Critical paths for the current roadmap.
KEEP_PATHS = [
    DRIVE_ROOT / 'KG1_NVIDIA_V202D/init_adapter_v194_rank19_build',
    DRIVE_ROOT / 'KG1_NVIDIA_V206B/output_v206b_answer_only_h100_loss_gated/train_v206b_answer_only_1s_lr1e9/final_adapter',
    DRIVE_ROOT / 'KG1_NVIDIA_V206C',
]

# Full folders known to be rejected/obsolete for the current solution.
DELETE_CANDIDATES = [
    DRIVE_ROOT / 'KG1_NVIDIA_V206',   # V206A rejected: final_eval_loss 1.1162 > 1.1149
    DRIVE_ROOT / 'KG1_NVIDIA_V202E',
    DRIVE_ROOT / 'KG1_NVIDIA_V202C',
    DRIVE_ROOT / 'KG1_NVIDIA_V202B',
    DRIVE_ROOT / 'KG1_NVIDIA_V202',
    DRIVE_ROOT / 'KG1_NVIDIA_V201',
    DRIVE_ROOT / 'KG1_NVIDIA_V199',
    DRIVE_ROOT / 'KG1_NVIDIA_V198',
    DRIVE_ROOT / 'KG1_NVIDIA_V195',
]

# Optional duplicate baseline backups. Keep them unless you need more space.
# These are only safe to delete after confirming that init_adapter_v194_rank19_build/submission.zip exists.
OPTIONAL_DUPLICATE_BASELINE_CANDIDATES = [
    DRIVE_ROOT / 'KG1_NVIDIA_V202D/baseline_v194_rank19',
    DRIVE_ROOT / 'KG1_NVIDIA_V202D/final_v194_keep_no_submit',
]

print('VERSION =', VERSION)
print('DRY_RUN =', DRY_RUN)
print('CONFIRM_DELETE =', repr(CONFIRM_DELETE))
"""
        ),
        code(
            """def is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def file_size(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def tree_size(path: pathlib.Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    if path.is_file():
        return file_size(path), 1, 0
    total = 0
    files = 0
    dirs = 0
    for root, dirnames, filenames in os.walk(path):
        dirs += len(dirnames)
        for name in filenames:
            files += 1
            total += file_size(pathlib.Path(root) / name)
    return total, files, dirs


def fmt_bytes(num: int) -> str:
    value = float(num)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def assert_safe_delete_target(path: pathlib.Path) -> None:
    resolved = path.resolve()
    if not is_relative_to(resolved, DRIVE_ROOT):
        raise RuntimeError(f'Refusing to delete outside MyDrive: {resolved}')
    for keep in KEEP_PATHS:
        if resolved == keep.resolve() or is_relative_to(keep, resolved):
            raise RuntimeError(f'Refusing to delete {resolved}; it contains KEEP path {keep}')
        if is_relative_to(resolved, keep):
            raise RuntimeError(f'Refusing to delete {resolved}; it is inside KEEP path {keep}')


def require_baseline_ready() -> None:
    adapter = DRIVE_ROOT / 'KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter'
    zip_path = DRIVE_ROOT / 'KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/submission.zip'
    required = [
        adapter / 'adapter_config.json',
        adapter / 'adapter_model.safetensors',
        zip_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Critical V194 baseline files missing: ' + json.dumps(missing, indent=2))
    print('V194 baseline ready:', adapter)
    print('V194 submission.zip sha256:', sha256_path(zip_path))


def require_v206b_forensic_ready() -> None:
    adapter = DRIVE_ROOT / 'KG1_NVIDIA_V206B/output_v206b_answer_only_h100_loss_gated/train_v206b_answer_only_1s_lr1e9/final_adapter'
    required = [adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Critical V206B forensic adapter files missing: ' + json.dumps(missing, indent=2))
    print('V206B forensic adapter ready:', adapter)


require_baseline_ready()
require_v206b_forensic_ready()
for keep in KEEP_PATHS:
    print('KEEP:', keep, 'exists=', keep.exists())
"""
        ),
        code(
            """def inventory(paths, group):
    rows = []
    for path in paths:
        size, files, dirs = tree_size(path)
        rows.append({
            'group': group,
            'path': str(path),
            'exists': path.exists(),
            'bytes': size,
            'size': fmt_bytes(size),
            'files': files,
            'dirs': dirs,
        })
    return rows


rows = []
rows.extend(inventory(KEEP_PATHS, 'KEEP'))
rows.extend(inventory(DELETE_CANDIDATES, 'DELETE_CANDIDATE'))
rows.extend(inventory(OPTIONAL_DUPLICATE_BASELINE_CANDIDATES, 'OPTIONAL_DUPLICATE_BASELINE'))

total_delete = sum(row['bytes'] for row in rows if row['group'] == 'DELETE_CANDIDATE' and row['exists'])
total_optional = sum(row['bytes'] for row in rows if row['group'] == 'OPTIONAL_DUPLICATE_BASELINE' and row['exists'])
print(json.dumps(rows, indent=2, sort_keys=True))
print('Candidate reclaim:', fmt_bytes(total_delete))
print('Optional duplicate baseline reclaim:', fmt_bytes(total_optional))

report_dir = DRIVE_ROOT / 'KG1_NVIDIA_V206C/output_v206c_delta_scale/reports'
report_dir.mkdir(parents=True, exist_ok=True)
inventory_path = report_dir / 'kg1_drive_cleanup_inventory.json'
inventory_path.write_text(json.dumps({
    'version': VERSION,
    'dry_run': DRY_RUN,
    'rows': rows,
    'candidate_reclaim_bytes': total_delete,
    'optional_duplicate_baseline_reclaim_bytes': total_optional,
}, indent=2, sort_keys=True), encoding='utf-8')
print('Inventory saved:', inventory_path)
"""
        ),
        code(
            """def delete_targets(paths, group):
    deleted = []
    skipped = []
    for path in paths:
        if not path.exists():
            skipped.append({'path': str(path), 'reason': 'not_exists'})
            continue
        assert_safe_delete_target(path)
        size, files, dirs = tree_size(path)
        if DRY_RUN:
            skipped.append({'path': str(path), 'reason': 'dry_run', 'size': fmt_bytes(size), 'files': files, 'dirs': dirs})
            continue
        if CONFIRM_DELETE != 'DELETE_KG1_UNUSED_ARTIFACTS':
            raise RuntimeError('Set CONFIRM_DELETE exactly to DELETE_KG1_UNUSED_ARTIFACTS before deleting.')
        print('Deleting:', path, fmt_bytes(size))
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append({'path': str(path), 'group': group, 'bytes': size, 'size': fmt_bytes(size), 'files': files, 'dirs': dirs})
    return deleted, skipped


deleted, skipped = delete_targets(DELETE_CANDIDATES, 'DELETE_CANDIDATE')
cleanup_report = {
    'version': VERSION,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'dry_run': DRY_RUN,
    'deleted': deleted,
    'skipped': skipped,
    'deleted_bytes': sum(item['bytes'] for item in deleted),
    'deleted_size': fmt_bytes(sum(item['bytes'] for item in deleted)),
    'optional_duplicate_baseline_not_deleted_by_default': [str(path) for path in OPTIONAL_DUPLICATE_BASELINE_CANDIDATES],
}
cleanup_path = report_dir / 'kg1_drive_cleanup_report.json'
cleanup_path.write_text(json.dumps(cleanup_report, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(cleanup_report, indent=2, sort_keys=True))
print('Cleanup report saved:', cleanup_path)
"""
        ),
        code(
            """# Optional second pass. Run only if the first pass is not enough and you accept removing duplicate V194 backups.
# Keep disabled by default because V194 is the current production baseline.
DELETE_OPTIONAL_DUPLICATE_BASELINES = False

if DELETE_OPTIONAL_DUPLICATE_BASELINES:
    require_baseline_ready()
    deleted_optional, skipped_optional = delete_targets(OPTIONAL_DUPLICATE_BASELINE_CANDIDATES, 'OPTIONAL_DUPLICATE_BASELINE')
    print(json.dumps({'deleted_optional': deleted_optional, 'skipped_optional': skipped_optional}, indent=2, sort_keys=True))
else:
    print('Optional duplicate baseline deletion is disabled.')
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "KG1_DRIVE_CLEANUP_SAFE_COLAB.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    text = "\n".join("".join(cell.get("source") or []) for cell in notebook["cells"])
    required = [
        "DRY_RUN = True",
        "CONFIRM_DELETE = ''",
        "DELETE_KG1_UNUSED_ARTIFACTS",
        "KG1_NVIDIA_V202D/init_adapter_v194_rank19_build",
        "KG1_NVIDIA_V206B/output_v206b_answer_only_h100_loss_gated/train_v206b_answer_only_1s_lr1e9/final_adapter",
        "KG1_NVIDIA_V206C",
        "KG1_NVIDIA_V206",
        "shutil.rmtree",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Generated cleanup notebook missing markers: {missing}")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()

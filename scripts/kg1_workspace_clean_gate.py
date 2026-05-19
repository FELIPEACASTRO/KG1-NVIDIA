#!/usr/bin/env python3
"""Audit and optionally remove KG1 workspace junk.

This gate is intentionally conservative. It only treats cache directories and
temporary editor/build files as auto-removable. Logs, manifests, datasets, and
analysis reports are reported but not deleted automatically, because they may be
needed to reproduce a decision or debug a plateau.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SAFE_JUNK_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".cache",
}

SAFE_JUNK_SUFFIXES = {
    ".tmp",
    ".bak",
    ".old",
    ".orig",
}

DOWNLOAD_ARCHIVE_SUFFIXES = {
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
    ".gz",
}

LARGE_BLOB_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".parquet",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    message: str
    bytes: int = 0


def is_relative_to(path: Path, parent: Path) -> bool:
    if os.name == "nt":
        child = normalized_windows_path_text(path)
        root = normalized_windows_path_text(parent)
        return child == root or child.startswith(root.rstrip("\\") + "\\")
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def normalized_windows_path_text(path: Path) -> str:
    text = str(path.resolve())
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.abspath(text))


def windows_long_path(path: Path) -> Path:
    """Return a Windows long-path version so audits do not silently miss files."""

    resolved = path.resolve()
    if os.name != "nt":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    return Path("\\\\?\\" + text)


def display_path(path: Path) -> str:
    text = str(path)
    if os.name == "nt" and text.startswith("\\\\?\\"):
        return text[4:]
    return text


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    for child in windows_long_path(path).rglob("*"):
        if child.is_file():
            total += file_size(child)
    return total


def audit_path(root: Path, large_blob_mb: float) -> list[Finding]:
    findings: list[Finding] = []
    root = root.resolve()
    walk_root = windows_long_path(root)

    for path in walk_root.rglob("*"):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not is_relative_to(resolved, root):
            findings.append(
                Finding(
                    "error",
                    "unsafe_path_outside_workspace",
                    display_path(path),
                    "Resolved path is outside the audited workspace.",
                )
            )
            continue

        name = path.name
        if path.is_dir() and name in SAFE_JUNK_DIR_NAMES:
            findings.append(
                Finding(
                    "error",
                    "safe_junk_dir_present",
                    display_path(path),
                    "Cache/checkpoint directory must not remain in the solution workspace.",
                    dir_size(path),
                )
            )
            continue

        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        size = file_size(path)
        if suffix in SAFE_JUNK_SUFFIXES:
            findings.append(
                Finding(
                    "error",
                    "safe_junk_file_present",
                    display_path(path),
                    "Temporary/editor backup file must not remain in the solution workspace.",
                    size,
                )
            )
        elif suffix in DOWNLOAD_ARCHIVE_SUFFIXES:
            findings.append(
                Finding(
                    "warning",
                    "download_archive_present",
                    display_path(path),
                    "Archive inside the repo must be justified by an active manifest or removed after extraction/audit.",
                    size,
                )
            )
        elif suffix in LARGE_BLOB_SUFFIXES and size >= int(large_blob_mb * 1024 * 1024):
            findings.append(
                Finding(
                    "warning",
                    "large_blob_present",
                    display_path(path),
                    "Large binary artifact should live in HF/Kaggle storage unless it is an active submit/package input.",
                    size,
                )
            )
    return findings


def delete_safe(root: Path, findings: list[Finding]) -> int:
    deleted = 0
    root = root.resolve()
    for item in findings:
        if item.code not in {"safe_junk_dir_present", "safe_junk_file_present"}:
            continue
        path = Path(item.path)
        resolved = path.resolve()
        if not is_relative_to(resolved, root):
            raise RuntimeError(f"refusing to delete outside workspace: {resolved}")
        target = windows_long_path(resolved)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=False)
            if not target.exists():
                deleted += 1
        elif target.exists():
            target.unlink()
            if not target.exists():
                deleted += 1
    return deleted


def report_payload(root: Path, findings: list[Finding], deleted: int = 0) -> dict[str, Any]:
    return {
        "schema_version": "kg1_workspace_clean_gate_v1",
        "root": str(root.resolve()),
        "passed": not any(item.severity == "error" for item in findings),
        "deleted_safe_items": deleted,
        "finding_counts": {
            "error": sum(1 for item in findings if item.severity == "error"),
            "warning": sum(1 for item in findings if item.severity == "warning"),
        },
        "findings": [item.__dict__ for item in findings],
        "policy": {
            "auto_deletes_only": sorted(SAFE_JUNK_DIR_NAMES) + sorted(SAFE_JUNK_SUFFIXES),
            "never_auto_delete": "logs, manifests, datasets, adapters, roadmaps, and analysis reports",
            "rule": "keep only active or future-reusable artifacts; remove caches, temp files, and extracted download leftovers",
        },
    }


def run_self_test() -> None:
    tmp = tempfile.mkdtemp()
    try:
        root = Path(tmp)
        (root / ".cache").mkdir()
        (root / ".cache" / "x").write_text("cache", encoding="utf-8")
        (root / "note.tmp").write_text("tmp", encoding="utf-8")
        (root / "evidence.log").write_text("keep", encoding="utf-8")
        deep = root / ("longpath_" + "x" * 80) / ("child_" + "y" * 80) / ("leaf_" + "z" * 80)
        deep_long = windows_long_path(deep)
        deep_long.mkdir(parents=True)
        (deep_long / "adapter_model.safetensors").write_bytes(b"0123456789")
        findings = audit_path(root, large_blob_mb=1.0)
        codes = {item.code for item in findings}
        if "safe_junk_dir_present" not in codes or "safe_junk_file_present" not in codes:
            raise AssertionError(f"self-test did not catch safe junk: {codes}")
        if not any(item.path.endswith("adapter_model.safetensors") for item in findings):
            # No finding is expected for this file; the check below verifies the
            # long path was at least traversable through a low threshold.
            long_findings = audit_path(root, large_blob_mb=0.000001)
            if not any("adapter_model.safetensors" in item.path for item in long_findings):
                raise AssertionError("self-test did not traverse long-path fixture")
        deleted = delete_safe(root, findings)
        if deleted != 2:
            raise AssertionError(f"self-test expected 2 deleted items, got {deleted}")
        findings_after = audit_path(root, large_blob_mb=1.0)
        if any(item.severity == "error" for item in findings_after):
            raise AssertionError(f"self-test cleanup left errors: {findings_after}")
        if not (root / "evidence.log").exists():
            raise AssertionError("self-test deleted evidence log")
    finally:
        shutil.rmtree(windows_long_path(Path(tmp)), ignore_errors=True)
    print("kg1_workspace_clean_gate_self_test=ok", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(ROOT))
    parser.add_argument("--delete-safe", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--large-blob-mb", type=float, default=25.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"audit path is not a directory: {root}")

    findings = audit_path(root, large_blob_mb=args.large_blob_mb)
    deleted = 0
    if args.delete_safe:
        deleted = delete_safe(root, findings)
        findings = audit_path(root, large_blob_mb=args.large_blob_mb)

    payload = report_payload(root, findings, deleted)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

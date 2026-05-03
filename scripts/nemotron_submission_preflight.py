#!/usr/bin/env python3
"""Strict preflight for Nemotron adapter submission zips.

This wraps the existing KG1 submission gate and adds the production contract we
want before any Kaggle upload: a root-level adapter_config.json plus a
root-level adapter_model.safetensors, with no training state, secrets, nested
archives, or diagnostic extras unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kg1_submission_gate import validate_adapter_zip  # noqa: E402


EXPECTED_ROOT_ENTRIES = ["adapter_config.json", "adapter_model.safetensors"]
NESTED_ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def inspect_layout(path: Path, allow_extra_files: bool) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
    except zipfile.BadZipFile:
        return {
            "valid": False,
            "reasons": ["bad_zip_file"],
            "warnings": [],
            "entries": [],
        }

    entries = [info.filename.replace("\\", "/") for info in infos]
    root_entries = [name for name in entries if "/" not in name.strip("/")]
    nested_entries = [name for name in entries if "/" in name.strip("/")]
    nested_archives = [
        name for name in entries
        if Path(name).suffix.lower() in NESTED_ARCHIVE_SUFFIXES
    ]
    compressed_bytes = sum(info.compress_size for info in infos)
    uncompressed_bytes = sum(info.file_size for info in infos)

    missing = [name for name in EXPECTED_ROOT_ENTRIES if name not in entries]
    if missing:
        reasons.append("missing_expected_root_entries")
    if nested_entries:
        reasons.append("nested_entries_present")
    if nested_archives:
        reasons.append("nested_archives_present")

    extras = sorted(set(entries) - set(EXPECTED_ROOT_ENTRIES))
    if extras and not allow_extra_files:
        reasons.append("extra_entries_present")
    elif extras:
        warnings.append("extra_entries_allowed_by_flag")

    if set(root_entries) != set(EXPECTED_ROOT_ENTRIES):
        warnings.append("root_entry_names_differ_from_expected")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "entries": entries,
        "root_entries": root_entries,
        "extra_entries": extras,
        "nested_entries_sample": nested_entries[:25],
        "nested_archives_sample": nested_archives[:25],
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": uncompressed_bytes,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    layout = inspect_layout(args.adapter_zip, args.allow_extra_files)
    gate = validate_adapter_zip(
        args.adapter_zip,
        reference_adapter_dir=args.reference_adapter_dir,
        strict_adapter_keys=args.strict_adapter_keys,
    )
    reasons: list[str] = []
    reasons.extend(f"layout:{reason}" for reason in layout["reasons"])
    reasons.extend(f"adapter_gate:{reason}" for reason in gate["reasons"])
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "adapter_zip": str(args.adapter_zip),
        "allow_extra_files": args.allow_extra_files,
        "strict_adapter_keys": args.strict_adapter_keys,
        "layout": layout,
        "adapter_gate": gate,
        "decision": {
            "production_ready": not reasons,
            "reasons": reasons,
            "policy": "Do not upload unless production_ready is true and score confidence gate is separately satisfied.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-zip", type=Path, required=True)
    parser.add_argument("--reference-adapter-dir", type=Path)
    parser.add_argument("--strict-adapter-keys", action="store_true")
    parser.add_argument("--allow-extra-files", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "score_analysis_20260426" / "submission_preflight.json",
    )
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.adapter_zip.exists():
        raise FileNotFoundError(args.adapter_zip)
    report = build_report(args)
    write_json(args.output_json, report)
    print(f"production_ready: {report['decision']['production_ready']}")
    print(f"report: {args.output_json}")
    if args.fail_on_block and report["decision"]["reasons"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

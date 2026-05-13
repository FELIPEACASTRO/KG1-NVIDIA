#!/usr/bin/env python3
"""Build V341 cleaned preference transfer files from V337D.

V340 found that some V337D hard-negative rows had the same boxed final answer
in chosen and rejected completions. This script preserves V337D SFT data, but
filters invalid preference pairs so a preference trainer does not learn a
contradictory signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V337D_MANIFEST = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_manifest.json"
)
DEFAULT_PREF_TRAIN = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_preferences_train.jsonl"
)
DEFAULT_PREF_VAL = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_preferences_val.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def boxed_values(text: str) -> list[str]:
    return re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))


def clean_preferences(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    bad: list[str] = []
    negative_counts: Counter[str] = Counter()
    kept_negative_counts: Counter[str] = Counter()
    removed_negative_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows, 1):
        rid = str(row.get("id", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        negative_type = str(metadata.get("negative_type", "unknown"))
        negative_counts[negative_type] += 1
        chosen_boxes = boxed_values(chosen)
        rejected_boxes = boxed_values(rejected)
        if not rid or not chosen or not rejected:
            bad.append(f"{split}:{idx}:missing_required_field")
            continue
        if len(chosen_boxes) != 1:
            bad.append(f"{split}:{rid}:chosen_box_count_{len(chosen_boxes)}")
            continue
        remove = False
        if negative_type == "hard_negative_equation_near_miss":
            if len(rejected_boxes) != 1:
                bad.append(f"{split}:{rid}:hard_negative_rejected_box_count_{len(rejected_boxes)}")
                continue
            if rejected_boxes[0] == chosen_boxes[0]:
                remove = True
        elif negative_type.startswith("format_negative_"):
            pass
        else:
            bad.append(f"{split}:{rid}:unexpected_negative_type_{negative_type}")
            continue
        if remove:
            removed.append(rid)
            removed_negative_counts[negative_type] += 1
            continue
        kept.append(row)
        kept_negative_counts[negative_type] += 1
        rule_counts[str(metadata.get("rule_class", ""))] += 1
    if bad:
        raise RuntimeError(f"{split} preference structural validation failed: " + json.dumps(bad[:30]))
    return kept, {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "kept_negative_type_counts": dict(sorted(kept_negative_counts.items())),
        "removed_negative_type_counts": dict(sorted(removed_negative_counts.items())),
        "rule_class_counts": dict(rule_counts.most_common(40)),
        "removed_id_examples": removed[:40],
    }


def assert_v337d_manifest(path: Path, pref_train: Path, pref_val: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("unexpected V337D schema")
    outputs = payload.get("outputs", {})
    expected_train = str(outputs.get("preferences_train_sha256", ""))
    expected_val = str(outputs.get("preferences_val_sha256", ""))
    if sha256_file(pref_train) != expected_train:
        raise RuntimeError("V337D preference train hash mismatch")
    if sha256_file(pref_val) != expected_val:
        raise RuntimeError("V337D preference validation hash mismatch")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V341 CLEAN PREFERENCE TRANSFER DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v337d_manifest_json =", args.v337d_manifest_json, flush=True)
    print("input_preferences_train_jsonl =", args.preferences_train_jsonl, flush=True)
    print("input_preferences_val_jsonl =", args.preferences_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v337d = assert_v337d_manifest(args.v337d_manifest_json, args.preferences_train_jsonl, args.preferences_val_jsonl)
    train_rows = read_jsonl(args.preferences_train_jsonl)
    val_rows = read_jsonl(args.preferences_val_jsonl)
    clean_train, train_summary = clean_preferences(train_rows, "train")
    clean_val, val_summary = clean_preferences(val_rows, "validation")

    train_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    val_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, clean_train)
    write_jsonl(val_path, clean_val)
    manifest = {
        "schema_version": "kg1_v341_clean_preference_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source": {
            "v337d_manifest_json": str(args.v337d_manifest_json),
            "v337d_manifest_sha256": sha256_file(args.v337d_manifest_json),
            "v337d_train_sha256": v337d.get("outputs", {}).get("train_sha256"),
            "v337d_val_sha256": v337d.get("outputs", {}).get("val_sha256"),
            "input_preferences_train_jsonl": str(args.preferences_train_jsonl),
            "input_preferences_train_sha256": sha256_file(args.preferences_train_jsonl),
            "input_preferences_val_jsonl": str(args.preferences_val_jsonl),
            "input_preferences_val_sha256": sha256_file(args.preferences_val_jsonl),
        },
        "cleaning_policy": {
            "removed": "hard_negative_equation_near_miss rows where rejected boxed answer equals chosen boxed answer",
            "kept": "format negatives and hard negatives with a different boxed final answer",
            "reason": "same-box hard negatives provide contradictory preference signal and were caught by V340",
        },
        "summary": {"train": train_summary, "validation": val_summary},
        "outputs": {
            "preferences_train_jsonl": str(train_path),
            "preferences_train_sha256": sha256_file(train_path),
            "preferences_val_jsonl": str(val_path),
            "preferences_val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "next_gate": "Run V340 with --allow-derived-preferences using these cleaned files.",
    }
    write_json(manifest_path, manifest)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("preferences_train_sha256 =", manifest["outputs"]["preferences_train_sha256"], flush=True)
    print("preferences_val_sha256 =", manifest["outputs"]["preferences_val_sha256"], flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V341 CLEAN PREFERENCE TRANSFER DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    rows = [
        {
            "id": "a",
            "chosen": r"Final answer: \boxed{11}",
            "rejected": r"Final answer: \boxed{11}",
            "metadata": {"negative_type": "hard_negative_equation_near_miss"},
        },
        {
            "id": "b",
            "chosen": r"Final answer: \boxed{11}",
            "rejected": r"Final answer: 11",
            "metadata": {"negative_type": "format_negative_no_box"},
        },
    ]
    kept, summary = clean_preferences(rows, "test")
    if len(kept) != 1 or summary["removed_rows"] != 1:
        raise AssertionError(summary)
    print("v341_clean_preference_transfer_dataset_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v337d-manifest-json", type=Path, default=DEFAULT_V337D_MANIFEST)
    parser.add_argument("--preferences-train-jsonl", type=Path, default=DEFAULT_PREF_TRAIN)
    parser.add_argument("--preferences-val-jsonl", type=Path, default=DEFAULT_PREF_VAL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/v341_clean_preference_transfer_dataset" / utc_compact(),
    )
    parser.add_argument("--label", default="v341_clean_preference_transfer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

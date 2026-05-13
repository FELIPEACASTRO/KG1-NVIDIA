#!/usr/bin/env python3
"""Build V321 hybrid answer-span dataset.

V321 keeps the broad V304 replay distribution and appends the focused V312
verifier-synthetic rows. Oversampling is intentionally left to the HF trainer
via SOURCE_WEIGHTS/SUBCATEGORY_WEIGHTS so the dataset remains auditable and does
not contain physical duplicate rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_V304_ROOT = Path("artifacts/v304_solver_trace_distill_dataset/20260512T1430Z")
DEFAULT_V312_ROOT = Path("artifacts/v312_verifier_synthetic_distill_dataset/20260512T1545Z")
DEFAULT_OUTPUT_ROOT = Path("artifacts/v321_hybrid_answer_span_dataset/20260513T0400Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids: list[str] = []
    families: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    weak_flags: Counter[str] = Counter()
    bad_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        row_id = str(row.get("id", ""))
        ids.append(row_id)
        family = str(row.get("family") or row.get("metadata", {}).get("family") or "")
        source = str(row.get("source") or row.get("metadata", {}).get("source") or "")
        subcategory = str(row.get("subcategory") or row.get("metadata", {}).get("subcategory") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        families[family] += 1
        sources[source] += 1
        subcategories[subcategory] += 1
        for flag in ("weak_gate_rows_used_for_training", "gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if bool(metadata.get(flag) or row.get(flag)):
                weak_flags[flag] += 1
        messages = row.get("messages")
        if not row_id or not row.get("prompt") or not row.get("answer") or not isinstance(messages, list) or len(messages) < 2:
            bad_rows.append({"index": index, "id": row_id, "reason": "missing required fields"})

    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        seen: set[str] = set()
        dupes = sorted({item for item in ids if item in seen or seen.add(item)})
        bad_rows.append({"reason": "duplicate ids", "sample": dupes[:10]})
    if weak_flags:
        bad_rows.append({"reason": "gate-row training flag present", "flags": dict(weak_flags)})

    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "family_counts": dict(sorted(families.items())),
        "source_counts": dict(sorted(sources.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "weak_training_flags": dict(sorted(weak_flags.items())),
        "bad_rows_first10": bad_rows[:10],
    }


def tag_rows(rows: list[dict[str, Any]], source_label: str) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metadata = dict(item.get("metadata") or {})
        metadata["v321_hybrid_source"] = source_label
        metadata["v321_physical_duplicate"] = False
        item["metadata"] = metadata
        tagged.append(item)
    return tagged


def build_dataset(v304_root: Path, v312_root: Path, output_root: Path) -> dict[str, Any]:
    print("=== V321 HYBRID DATASET BUILD START ===", flush=True)
    print(f"v304_root={v304_root}", flush=True)
    print(f"v312_root={v312_root}", flush=True)
    print(f"output_root={output_root}", flush=True)

    inputs = {
        "v304_train": v304_root / "v304_solver_trace_distill_train.jsonl",
        "v304_val": v304_root / "v304_solver_trace_distill_val.jsonl",
        "v312_train": v312_root / "v312_verifier_synthetic_distill_sft_train.jsonl",
        "v312_val": v312_root / "v312_verifier_synthetic_distill_sft_val.jsonl",
    }
    for label, path in inputs.items():
        print(f"input {label} exists={path.exists()} path={path}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    v304_train = tag_rows(read_jsonl(inputs["v304_train"]), "v304_solver_trace_distill")
    v304_val = tag_rows(read_jsonl(inputs["v304_val"]), "v304_solver_trace_distill")
    v312_train = tag_rows(read_jsonl(inputs["v312_train"]), "v312_verifier_synthetic_distill")
    v312_val = tag_rows(read_jsonl(inputs["v312_val"]), "v312_verifier_synthetic_distill")

    train_rows = v304_train + v312_train
    val_rows = v304_val + v312_val
    train_audit = audit_rows(train_rows, "train")
    val_audit = audit_rows(val_rows, "validation")
    print("train_audit=" + json.dumps(train_audit, sort_keys=True), flush=True)
    print("val_audit=" + json.dumps(val_audit, sort_keys=True), flush=True)

    if train_audit["bad_rows_first10"] or val_audit["bad_rows_first10"]:
        raise RuntimeError("V321 hybrid dataset audit failed")
    if "bit_manipulation" not in train_audit["family_counts"] or "equation_transform" not in train_audit["family_counts"]:
        raise RuntimeError("V321 train split missing target families")
    if "v312_verifier_synthetic" not in train_audit["source_counts"]:
        raise RuntimeError("V321 train split missing V312 focused rows")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "v321_hybrid_answer_span_train.jsonl"
    val_path = output_root / "v321_hybrid_answer_span_val.jsonl"
    manifest_path = output_root / "v321_hybrid_answer_span_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest: dict[str, Any] = {
        "version": "v321_hybrid_answer_span_dataset",
        "output_root": str(output_root),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "inputs": {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in inputs.items()},
        "train_audit": train_audit,
        "val_audit": val_audit,
        "policy": {
            "weak_gate_rows_used_for_training": False,
            "physical_duplicate_rows": False,
            "oversampling_location": "HF trainer weighted sampler only",
            "promotion_gate": "weak total>=193, equation>=60, bit>=136, truncation no worse",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("manifest_path=" + str(manifest_path), flush=True)
    print("manifest=" + json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V321 HYBRID DATASET BUILD END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v304-root", type=Path, default=DEFAULT_V304_ROOT)
    parser.add_argument("--v312-root", type=Path, default=DEFAULT_V312_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    build_dataset(args.v304_root, args.v312_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

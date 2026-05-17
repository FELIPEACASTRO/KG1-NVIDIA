#!/usr/bin/env python3
"""Audit the V544 minimal distillation dataset before any HF launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "artifacts/v544_minimal_distillation_dataset/20260517T_v544_cpu_gate"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if raw.strip():
                row = json.loads(raw)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def weight(value: Any) -> float:
    return float(value)


def audit_rows(rows: list[dict[str, Any]], split: str, issues: list[str], warnings: list[str]) -> dict[str, Any]:
    family = Counter()
    roles = Counter()
    formats = Counter()
    source_ids = Counter()
    weight_by_role: dict[str, set[float]] = defaultdict(set)
    prompt_hashes: set[str] = set()
    prompt_answer_hashes: set[str] = set()
    bad_label_free: list[str] = []
    bad_format_metadata: list[str] = []
    ids: set[str] = set()
    answer_chars: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in ids:
            issues.append(f"{split}:duplicate_or_missing_id:{row_id}")
        ids.add(row_id)
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", "")).strip()
        metadata = row.get("metadata") or {}
        role = str(metadata.get("role") or row.get("subcategory") or "")
        fmt = str(metadata.get("final_answer_format") or "")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3 or messages[-1].get("role") != "assistant":
            issues.append(f"{split}:{row_id}:bad_messages")
            continue
        assistant = str(messages[-1].get("content", ""))
        extracted = extract_final_answer(assistant)
        if not verify_answer(answer, extracted):
            bad_label_free.append(f"{row_id}:{answer!r}!={extracted!r}")
        if fmt == "boxed_raw_label_free" and "\\boxed{" not in assistant:
            bad_format_metadata.append(f"{row_id}:boxed_format_without_boxed")
        if fmt == "unboxed_label_free_fallback" and "\\boxed{" in assistant:
            bad_format_metadata.append(f"{row_id}:unboxed_format_contains_boxed")
        if metadata.get("weak_gate_rows_used_for_training") is not False:
            issues.append(f"{split}:{row_id}:weak_gate_rows_used_for_training")
        if metadata.get("full_gate_rows_used_for_training") is not False:
            issues.append(f"{split}:{row_id}:full_gate_rows_used_for_training")
        if metadata.get("gate_rows_used_for_training") is not False:
            issues.append(f"{split}:{row_id}:gate_rows_used_for_training")
        family[str(row.get("family", ""))] += 1
        roles[role] += 1
        formats[fmt] += 1
        source_id = str(metadata.get("source_row_id", ""))
        source_ids[source_id] += 1
        if "loss_weight" in metadata:
            weight_by_role[role].add(weight(metadata["loss_weight"]))
        prompt_hashes.add(sha256_text(prompt))
        prompt_answer_hashes.add(sha256_text(prompt + "\n" + answer))
        answer_chars.update(answer)
    if bad_label_free:
        issues.append(f"{split}:label_free_answer_mismatch:{bad_label_free[:20]}")
    if bad_format_metadata:
        issues.append(f"{split}:format_metadata_mismatch:{bad_format_metadata[:20]}")
    if not formats:
        warnings.append(f"{split}:missing_final_answer_format_counts")
    return {
        "rows": len(rows),
        "family": dict(sorted(family.items())),
        "roles": dict(sorted(roles.items())),
        "final_answer_formats": dict(sorted(formats.items())),
        "unique_source_ids": len(source_ids),
        "unique_prompt_hashes": len(prompt_hashes),
        "unique_prompt_answer_hashes": len(prompt_answer_hashes),
        "weights_by_role": {key: sorted(values) for key, values in sorted(weight_by_role.items())},
        "answer_character_set": "".join(sorted(answer_chars)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = dataset_dir / "v544_minimal_distillation_manifest.json"
    train_path = dataset_dir / "v544_minimal_distillation_train.jsonl"
    val_path = dataset_dir / "v544_minimal_distillation_val.jsonl"
    output_json = args.output_json or dataset_dir / "v544_dataset_doublecheck_audit.json"

    manifest = read_json(manifest_path)
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    issues: list[str] = []
    warnings: list[str] = []

    observed_train_sha = sha256_file(train_path)
    observed_val_sha = sha256_file(val_path)
    expected_train_sha = manifest.get("outputs", {}).get("train_sha256")
    expected_val_sha = manifest.get("outputs", {}).get("val_sha256")
    if observed_train_sha != expected_train_sha:
        issues.append(f"train_sha_mismatch:{observed_train_sha}!={expected_train_sha}")
    if observed_val_sha != expected_val_sha:
        issues.append(f"val_sha_mismatch:{observed_val_sha}!={expected_val_sha}")

    train_summary = audit_rows(train_rows, "train", issues, warnings)
    val_summary = audit_rows(val_rows, "val", issues, warnings)
    train_prompt_hashes = {sha256_text(str(row.get("prompt", ""))) for row in train_rows}
    val_prompt_hashes = {sha256_text(str(row.get("prompt", ""))) for row in val_rows}
    if train_prompt_hashes & val_prompt_hashes:
        issues.append(f"train_val_prompt_overlap:{len(train_prompt_hashes & val_prompt_hashes)}")
    teacher_sources = Counter(
        str(row.get("metadata", {}).get("source_row_id", ""))
        for row in train_rows
        if row.get("metadata", {}).get("role") == "teacher_gain"
    )
    if sorted(teacher_sources.values()) != [5] * 9:
        issues.append(f"teacher_gain_repeat_counts:{dict(sorted(teacher_sources.items()))}")
    protected = [
        row for row in train_rows if row.get("metadata", {}).get("source_row_id") == "8740ed31"
    ]
    if len(protected) != 1 or str(protected[0].get("answer", "")) != "01101000":
        issues.append("protected_row_8740ed31_not_preserved_once")

    payload = {
        "schema_version": "kg1_v544_dataset_doublecheck_audit_v2",
        "dataset_dir": str(dataset_dir),
        "manifest": {
            "train_sha256_expected": expected_train_sha,
            "train_sha256_observed": observed_train_sha,
            "val_sha256_expected": expected_val_sha,
            "val_sha256_observed": observed_val_sha,
        },
        "counts": {
            "train": train_summary,
            "validation": val_summary,
            "teacher_gain_source_counts": dict(sorted(teacher_sources.items())),
        },
        "issues": issues,
        "warnings": warnings,
        "decision": "dataset_doublecheck_passed" if not issues else "dataset_doublecheck_failed",
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

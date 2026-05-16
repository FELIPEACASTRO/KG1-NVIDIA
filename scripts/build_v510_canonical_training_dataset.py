#!/usr/bin/env python3
"""Build a single curated KG1 training dataset from audited JSONL sources.

This is not a blind concatenation. It consumes the V509 integrity audit,
includes only active clean sources, removes duplicates, preserves per-row source
metadata, and records every include/exclude decision.

The output is CPU-only. It never trains, launches HF, packages, or submits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_V509_SUMMARY = (
    REPO_ROOT
    / "artifacts/v509_training_dataset_integrity_audit/"
    / "v509_training_dataset_integrity_audit_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v510_canonical_training_dataset"

ACTIVE_INCLUDE_PATTERNS = {
    "v498_numeric_teacher_trace_pack": {
        "priority": 10,
        "reason": "latest clean numeric hard-negative trace pack plus bit replay guardrail",
    },
    "v475_equation_bit_replay_mix": {
        "priority": 20,
        "reason": "clean CPU-gated equation/bit replay mix with boxed final answers",
    },
    "v460_numeric_one_rule_micro_dataset": {
        "priority": 30,
        "reason": "small clean numeric sign-guard micro dataset",
    },
}

HISTORICAL_EXCLUDE_PATTERNS = {
    "v293_v274_distill_dataset": "old unboxed broad distill style; not compatible with current boxed short-answer recipe",
    "v390_v326_equation_bit_replay_mix_dataset": "historical broad mix already failed transfer in V391/V494",
    "v406_solver_first_transfer_dataset": "historical solver-first mix already failed to transfer",
    "v410_solver_first_transfer_dataset": "historical solver-first mix already failed to transfer",
    "v416_rawstyle_transfer_dataset": "historical rawstyle mix already failed to transfer",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_answer_hash(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    return sha256_text(prompt + "\n===ANSWER===\n" + answer)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_policy(path: Path, status: str, nonboxed_rows: int) -> tuple[bool, int, str]:
    text = str(path).replace("\\", "/")
    if status != "ok":
        return False, 999, "excluded: V509 integrity status is blocked"
    if nonboxed_rows:
        return False, 999, "excluded: nonboxed assistant answers under current boxed recipe"
    for pattern, reason in HISTORICAL_EXCLUDE_PATTERNS.items():
        if pattern in text:
            return False, 999, f"excluded: {reason}"
    for pattern, info in ACTIVE_INCLUDE_PATTERNS.items():
        if pattern in text:
            return True, int(info["priority"]), f"included: {info['reason']}"
    return False, 999, "excluded: not in active canonical source allowlist"


def split_for_path(path: Path) -> str:
    name = path.name.lower()
    if "_val" in name or "validation" in name:
        return "val"
    if "_train" in name:
        return "train"
    return "unknown"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def canonicalize_row(row: dict[str, Any], *, source_path: Path, source_dataset: str, label: str) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata") if isinstance(out.get("metadata"), dict) else {})
    metadata.update(
        {
            "v510_canonical_dataset": label,
            "v510_source_dataset": source_dataset,
            "v510_source_path": str(source_path),
            "v510_source_sha256": sha256_file(source_path),
            "v510_builder_kept_original_prompt": True,
            "weak_gate_rows_used_for_training": bool(metadata.get("weak_gate_rows_used_for_training", False)),
            "gate_rows_used_for_training": bool(metadata.get("gate_rows_used_for_training", False)),
            "full_gate_rows_used_for_training": bool(metadata.get("full_gate_rows_used_for_training", False)),
        }
    )
    out["metadata"] = metadata
    out["source_dataset"] = source_dataset
    return out


def row_is_valid(row: dict[str, Any]) -> tuple[bool, str]:
    answer = str(row.get("answer", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    family = str(row.get("family", "")).strip()
    row_id = str(row.get("id", "")).strip()
    messages = row.get("messages")
    if not row_id or not answer or not prompt or not family or not isinstance(messages, list):
        return False, "missing required id/prompt/answer/family/messages fields"
    assistant = ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant = str(message.get("content", ""))
            break
    if not assistant:
        return False, "missing assistant message"
    if "\\boxed{" not in assistant:
        return False, "assistant answer is not boxed"
    if not verify_answer(answer, extract_final_answer(assistant)):
        return False, "assistant final answer does not verify against row answer"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("weak_gate_rows_used_for_training") or metadata.get("gate_rows_used_for_training") or metadata.get("full_gate_rows_used_for_training"):
        return False, "anti-leakage metadata flag is true"
    return True, "ok"


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit_rows = read_csv_rows(args.v509_summary_csv)
    decisions: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    for audit in audit_rows:
        path = Path(audit["path"])
        nonboxed = int(audit.get("nonboxed_rows") or 0)
        include, priority, reason = source_policy(path, audit.get("status", ""), nonboxed)
        split = split_for_path(path)
        decision = {
            "path": str(path),
            "dataset": audit.get("dataset") or path.stem,
            "split": split,
            "status": audit.get("status", ""),
            "rows": int(audit.get("rows") or audit.get("row_count") or 0),
            "nonboxed_rows": nonboxed,
            "include": include,
            "priority": priority,
            "reason": reason,
        }
        decisions.append(decision)
        if include:
            source_entries.append(decision)

    source_entries.sort(key=lambda item: (int(item["priority"]), item["split"], item["path"]))
    kept_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen_by_split: dict[str, set[str]] = {"train": set(), "val": set()}
    train_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    duplicate_examples: list[dict[str, str]] = []

    for source in source_entries:
        split = str(source["split"])
        if split not in kept_by_split:
            skipped_counts["unknown_split"] += int(source["rows"])
            continue
        path = Path(source["path"])
        source_dataset = str(source["dataset"])
        for row in load_jsonl(path):
            valid, detail = row_is_valid(row)
            if not valid:
                skipped_counts[detail] += 1
                continue
            row_hash = prompt_answer_hash(row)
            if split == "val" and row_hash in train_hashes:
                skipped_counts["val_train_prompt_answer_overlap"] += 1
                continue
            if row_hash in seen_by_split[split]:
                skipped_counts[f"duplicate_prompt_answer_{split}"] += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append(
                        {
                            "split": split,
                            "source_dataset": source_dataset,
                            "id": str(row.get("id", "")),
                            "prompt_answer_hash": row_hash,
                        }
                    )
                continue
            out = canonicalize_row(row, source_path=path, source_dataset=source_dataset, label=args.label)
            kept_by_split[split].append(out)
            seen_by_split[split].add(row_hash)
            if split == "train":
                train_hashes.add(row_hash)
            source_counts[f"{split}:{source_dataset}"] += 1
            family_counts[f"{split}:{out.get('family', '')}"] += 1

    output_dir = args.output_dir / args.label
    train_jsonl = output_dir / f"{args.label}_train.jsonl"
    val_jsonl = output_dir / f"{args.label}_val.jsonl"
    decisions_csv = output_dir / f"{args.label}_source_decisions.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"

    write_jsonl(train_jsonl, kept_by_split["train"])
    write_jsonl(val_jsonl, kept_by_split["val"])
    write_csv(decisions_csv, decisions, ["path", "dataset", "split", "status", "rows", "nonboxed_rows", "include", "priority", "reason"])

    manifest = {
        "schema_version": "kg1_v510_canonical_training_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v509_summary_csv": str(args.v509_summary_csv),
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "val_sha256": sha256_file(val_jsonl),
            "source_decisions_csv": str(decisions_csv),
            "manifest_json": str(manifest_json),
        },
        "v509_summary_csv": str(args.v509_summary_csv),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "source_decisions_csv": str(decisions_csv),
        "train_rows": len(kept_by_split["train"]),
        "val_rows": len(kept_by_split["val"]),
        "included_source_count": sum(1 for row in decisions if row["include"]),
        "excluded_source_count": sum(1 for row in decisions if not row["include"]),
        "source_counts": dict(sorted(source_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "duplicate_examples": duplicate_examples,
        "decision": {
            "status": "canonical_dataset_ready_for_tokenization_gate"
            if kept_by_split["train"] and kept_by_split["val"]
            else "canonical_dataset_blocked",
            "next_action": "Run V509 integrity audit and tokenization/pre-paid gates on V510 before any HF job.",
        },
    }
    write_json(manifest_json, manifest)

    print("=== V510 CANONICAL TRAINING DATASET START ===", flush=True)
    print("included_source_count =", manifest["included_source_count"], flush=True)
    print("excluded_source_count =", manifest["excluded_source_count"], flush=True)
    print("train_rows =", manifest["train_rows"], flush=True)
    print("val_rows =", manifest["val_rows"], flush=True)
    print("skipped_counts =", json.dumps(manifest["skipped_counts"], sort_keys=True), flush=True)
    print("train_jsonl =", train_jsonl, flush=True)
    print("val_jsonl =", val_jsonl, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V510 CANONICAL TRAINING DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v509-summary-csv", type=Path, default=DEFAULT_V509_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v510_canonical_active_training_pool")
    return parser.parse_args()


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

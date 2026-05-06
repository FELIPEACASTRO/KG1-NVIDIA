#!/usr/bin/env python
"""Build a V214 non-validation micro-replay candidate dataset.

This prepares a review artifact, not a training launch. It excludes exact
prompt+answer overlap with the protected V194 947-row validation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition_utils import extract_boxed_answers, extract_final_answer, verify_answer  # noqa: E402

DEFAULT_QUOTAS = {
    "gravity_constant": 150,
    "numeral_system": 150,
    "text_encryption": 150,
    "unit_conversion": 150,
    "bit_manipulation": 160,
    "equation_transform": 120,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(prompt: object, answer: object) -> str:
    payload = json.dumps(
        {"prompt": str(prompt or "").strip(), "answer": str(answer or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_validation_signatures(path: Path) -> set[str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    sigs: set[str] = set()
    for _, row in df.iterrows():
        prompt = ""
        for col in ("prompt", "prompt_x", "prompt_y", "question"):
            if col in row and str(row[col]):
                prompt = str(row[col])
                break
        if prompt and str(row.get("answer", "")):
            sigs.add(signature(prompt, row["answer"]))
    return sigs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return str(row.get("completion") or row.get("assistant") or row.get("cot") or "")


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "").strip()
    family = str(row.get("family") or row.get("category") or "").strip()
    text = assistant_text(row)
    extracted = extract_final_answer(text)
    boxed = extract_boxed_answers(text)
    return {
        "verified": bool(answer and verify_answer(answer, extracted)),
        "boxed_count": len(boxed),
        "single_boxed": len(boxed) == 1,
        "assistant_word_count": len(text.split()),
        "bit_shape": family != "bit_manipulation" or bool(len(answer) == 8 and set(answer) <= {"0", "1"}),
    }


def source_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    source = str(row.get("source") or "")
    family = str(row.get("family") or "")
    text_len = audit_row(row)["assistant_word_count"]
    order = [
        "v206::v95_bit_rehearsal_train",
        "v206::v100_programmatic_repair_train",
        "v206::v90_gold_safe_train",
        "v206::v92_delta_train",
        "v206::v94_equation_crypt_train",
    ]
    source_rank = order.index(source) if source in order else len(order)
    # Prefer concise but non-empty CoT. Avoid ultra-long traces in the first candidate.
    length_penalty = 0 if 20 <= text_len <= 350 else 1
    if family == "text_encryption":
        length_penalty = 0 if 20 <= text_len <= 800 else 1
    return (source_rank, length_penalty, str(row.get("id") or ""))


def build(args: argparse.Namespace) -> int:
    source_path = Path(args.source_jsonl)
    validation_path = Path(args.validation_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "v214_micro_replay_candidate.jsonl"
    manifest_path = output_dir / "v214_micro_replay_candidate_manifest.json"

    print("[build_v214_micro_replay_candidate] START")
    print(f"[build_v214_micro_replay_candidate] source_jsonl={source_path}")
    print(f"[build_v214_micro_replay_candidate] validation_csv={validation_path}")
    print(f"[build_v214_micro_replay_candidate] output_dir={output_dir}")

    rng = random.Random(args.seed)
    validation_sigs = load_validation_signatures(validation_path)
    rows = load_jsonl(source_path)
    rejected = Counter()
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()

    for row in rows:
        family = str(row.get("family") or row.get("category") or "")
        prompt = str(row.get("prompt") or "")
        answer = str(row.get("answer") or "")
        if family not in DEFAULT_QUOTAS:
            rejected["family_not_targeted"] += 1
            continue
        if not prompt or not answer:
            rejected["missing_prompt_or_answer"] += 1
            continue
        sig = signature(prompt, answer)
        if sig in validation_sigs:
            rejected["validation_overlap"] += 1
            continue
        if sig in seen:
            rejected["duplicate_prompt_answer"] += 1
            continue
        audit = audit_row(row)
        if not audit["verified"]:
            rejected["not_verified"] += 1
            continue
        if not audit["single_boxed"]:
            rejected["not_single_boxed"] += 1
            continue
        if not audit["bit_shape"]:
            rejected["bad_bit_shape"] += 1
            continue
        seen.add(sig)
        copied = dict(row)
        metadata = dict(copied.get("metadata") or {})
        metadata.update(
            {
                "v214_role": "micro_replay_candidate",
                "train_allowed": True,
                "validation_overlap_excluded": True,
            }
        )
        copied["metadata"] = metadata
        by_family[family].append(copied)

    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    for family, quota in DEFAULT_QUOTAS.items():
        candidates = by_family[family]
        candidates.sort(key=source_priority)
        if args.shuffle_within_family:
            # Keep source/length priority buckets stable, shuffle inside equal keys.
            grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in candidates:
                key = source_priority(row)[:2]
                grouped[key].append(row)
            ordered: list[dict[str, Any]] = []
            for key in sorted(grouped):
                bucket = grouped[key]
                rng.shuffle(bucket)
                ordered.extend(bucket)
            candidates = ordered
        take = candidates[:quota]
        selected.extend(take)
        if len(take) < quota:
            shortfalls[family] = quota - len(take)

    rng.shuffle(selected)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    selected_counts = Counter(str(row.get("family") or "unknown") for row in selected)
    source_counts = Counter(str(row.get("source") or "unknown") for row in selected)
    audits = [audit_row(row) for row in selected]
    manifest = {
        "schema_version": "v214_micro_replay_candidate_v1",
        "generated_at_utc": utc_now(),
        "decision": "review_only_not_training_launch",
        "source_jsonl": str(source_path),
        "source_sha256": sha256_file(source_path),
        "validation_csv": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_signature_count": len(validation_sigs),
        "output_jsonl": str(out_jsonl),
        "output_sha256": sha256_file(out_jsonl),
        "seed": args.seed,
        "quotas": DEFAULT_QUOTAS,
        "rows": len(selected),
        "family_counts": dict(sorted(selected_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "shortfalls": shortfalls,
        "audit": {
            "verified": sum(1 for row in audits if row["verified"]),
            "single_boxed": sum(1 for row in audits if row["single_boxed"]),
            "bit_shape_ok": sum(1 for row in audits if row["bit_shape"]),
            "max_assistant_word_count": max((row["assistant_word_count"] for row in audits), default=0),
        },
        "hard_gate": {
            "train_allowed_after_review": not shortfalls and len(selected) == sum(DEFAULT_QUOTAS.values()),
            "requires_before_training": [
                "manual review of manifest",
                "adapter/tokenizer/template trainability audit",
                "no local validation rows in training data",
                "short dry-run only; no submit",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[build_v214_micro_replay_candidate] wrote {out_jsonl}")
    print(f"[build_v214_micro_replay_candidate] wrote {manifest_path}")
    print(
        "[build_v214_micro_replay_candidate] RESULT "
        f"rows={len(selected)} verified={manifest['audit']['verified']} "
        f"single_boxed={manifest['audit']['single_boxed']} shortfalls={shortfalls}"
    )
    print("[build_v214_micro_replay_candidate] END")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", default="data/v206/v206_curated_train.jsonl")
    parser.add_argument("--validation-csv", default="artifacts/drive_exports/v194_baseline_predictions.csv")
    parser.add_argument("--output-dir", default="data/v214")
    parser.add_argument("--seed", type=int, default=214)
    parser.add_argument("--shuffle-within-family", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))

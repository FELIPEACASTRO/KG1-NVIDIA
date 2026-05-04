#!/usr/bin/env python3
"""Validate KG1 training JSONL before launching expensive runs.

The gate focuses on failure modes that silently waste GPU time:

* malformed JSONL
* missing chat messages
* blank or unboxed assistant answers
* duplicate record IDs
* unknown puzzle families
* exact ID/prompt overlap with Kaggle test data
* likely secret material embedded in generated examples
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from competition_utils import FAMILIES, canonical_family  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use
    FAMILIES = {
        "bit_manipulation",
        "equation_transform",
        "text_encryption",
        "unit_conversion",
        "gravity_constant",
        "numeral_system",
    }

    def canonical_family(value: str) -> str:
        return value.strip().lower()


SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9_=-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|kaggle[_-]?key|hf[_-]?token|hugging_face_hub_token|password|authorization)\b"
        r".{0,80}['\"][^'\"]{8,}['\"]"
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_prompt(text: str) -> str:
    text = re.sub(r"Please put your final answer inside `?\\boxed\{\}`?.*", "", text, flags=re.I | re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_message(messages: list[dict[str, Any]], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""


def last_message(messages: list[dict[str, Any]], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""


def has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def extract_boxed_answer(text: str | None) -> str | None:
    """Return the last parseable boxed answer, handling literal escaped braces."""
    if not text:
        return None

    starts: list[int] = []
    search = 0
    marker = r"\boxed"
    while True:
        start = text.find(marker, search)
        if start < 0:
            break
        brace_start = text.find("{", start + len(marker))
        if brace_start >= 0 and not is_escaped(text, start):
            between = text[start + len(marker) : brace_start]
            if between.strip() == "":
                starts.append(brace_start)
        search = start + len(marker)

    for brace_start in reversed(starts):
        depth = 0
        payload: list[str] = []
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            escaped = ch in "{}" and is_escaped(text, idx)
            if idx == brace_start:
                depth = 1
                continue
            if escaped:
                payload.append(ch)
                continue
            if ch == "{":
                depth += 1
                payload.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    answer = "".join(payload).strip()
                    return answer or None
                payload.append(ch)
            else:
                payload.append(ch)
    return None


def validate_jsonl(jsonl_path: Path, train_csv: Path | None, test_csv: Path | None) -> dict[str, Any]:
    train_rows = read_csv_rows(train_csv)
    test_rows = read_csv_rows(test_csv)
    train_ids = {str(row.get("id", "")).strip() for row in train_rows if row.get("id")}
    test_ids = {str(row.get("id", "")).strip() for row in test_rows if row.get("id")}
    train_prompts = {normalize_prompt(str(row.get("prompt", ""))) for row in train_rows if row.get("prompt")}
    test_prompts = {normalize_prompt(str(row.get("prompt", ""))) for row in test_rows if row.get("prompt")}

    ids: list[str] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    assistant_lengths: list[int] = []
    parse_errors: list[dict[str, Any]] = []
    missing_messages: list[str] = []
    blank_assistant: list[str] = []
    unboxed_assistant: list[str] = []
    unknown_family: list[str] = []
    train_id_misses: list[str] = []
    test_id_overlap: list[str] = []
    test_prompt_overlap: list[str] = []
    secret_findings: list[str] = []
    duplicate_prompt_counter: Counter[str] = Counter()

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_no, "error": str(exc)})
                continue

            record_id = str(rec.get("id", f"line:{line_no}")).strip()
            ids.append(record_id)

            family_raw = rec.get("family", rec.get("category", rec.get("type", "")))
            family = canonical_family(str(family_raw))
            family_counts[family] += 1
            if family not in FAMILIES:
                unknown_family.append(record_id)

            source_counts[str(rec.get("source", "unknown"))] += 1
            messages = rec.get("messages")
            if not isinstance(messages, list) or not messages:
                missing_messages.append(record_id)
                continue

            user_text = first_message(messages, "user")
            assistant_text = last_message(messages, "assistant")
            assistant_lengths.append(len(assistant_text))
            prompt_norm = normalize_prompt(user_text)
            if prompt_norm:
                duplicate_prompt_counter[prompt_norm] += 1

            if not assistant_text.strip():
                blank_assistant.append(record_id)
            if extract_boxed_answer(assistant_text) is None:
                unboxed_assistant.append(record_id)
            if has_secret(raw):
                secret_findings.append(record_id)
            if train_ids and record_id not in train_ids:
                train_id_misses.append(record_id)
            if record_id in test_ids:
                test_id_overlap.append(record_id)
            if prompt_norm in test_prompts:
                test_prompt_overlap.append(record_id)

    id_counts = Counter(ids)
    duplicate_ids = sorted(record_id for record_id, count in id_counts.items() if count > 1)
    duplicate_prompts = sum(count - 1 for count in duplicate_prompt_counter.values() if count > 1)

    row_count = len(ids)
    avg_assistant_chars = sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0.0
    boxed_rate = 1.0 - (len(unboxed_assistant) / row_count if row_count else 1.0)

    reasons: list[str] = []
    if parse_errors:
        reasons.append("json_parse_errors")
    if missing_messages:
        reasons.append("missing_messages")
    if blank_assistant:
        reasons.append("blank_assistant")
    if unboxed_assistant:
        reasons.append("unboxed_assistant")
    if unknown_family:
        reasons.append("unknown_family")
    if duplicate_ids:
        reasons.append("duplicate_ids")
    if test_id_overlap:
        reasons.append("test_id_overlap")
    if test_prompt_overlap:
        reasons.append("test_prompt_overlap")
    if secret_findings:
        reasons.append("secret_findings")

    warnings: list[str] = []
    if train_id_misses:
        warnings.append("records_not_in_train_csv")
    if duplicate_prompts:
        warnings.append("duplicate_prompts")
    if boxed_rate < 0.99:
        warnings.append("boxed_rate_below_99pct")

    return {
        "path": str(jsonl_path),
        "sha256": sha256_file(jsonl_path),
        "generated_at": utc_now(),
        "rows": row_count,
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(source_counts.most_common()),
        "boxed_rate": boxed_rate,
        "avg_assistant_chars": avg_assistant_chars,
        "parse_errors_sample": parse_errors[:25],
        "missing_messages_sample": missing_messages[:25],
        "blank_assistant_sample": blank_assistant[:25],
        "unboxed_assistant_sample": unboxed_assistant[:25],
        "unknown_family_sample": unknown_family[:25],
        "duplicate_ids_sample": duplicate_ids[:25],
        "duplicate_prompt_excess_count": duplicate_prompts,
        "records_not_in_train_csv_sample": train_id_misses[:25],
        "test_id_overlap_sample": test_id_overlap[:25],
        "test_prompt_overlap_sample": test_prompt_overlap[:25],
        "secret_findings_sample": secret_findings[:25],
    }


def write_clean_jsonl(jsonl_path: Path, clean_path: Path, test_csv: Path) -> dict[str, Any]:
    test_rows = read_csv_rows(test_csv)
    test_ids = {str(row.get("id", "")).strip() for row in test_rows if row.get("id")}
    test_prompts = {normalize_prompt(str(row.get("prompt", ""))) for row in test_rows if row.get("prompt")}
    seen_ids: set[str] = set()
    stats = Counter()

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as src, clean_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line_no, raw in enumerate(src, start=1):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            stats["input_rows"] += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                stats["dropped_malformed_json"] += 1
                continue

            record_id = str(rec.get("id", f"line:{line_no}")).strip()
            if record_id in seen_ids:
                stats["dropped_duplicate_id"] += 1
                continue

            messages = rec.get("messages")
            if not isinstance(messages, list) or not messages:
                stats["dropped_missing_messages"] += 1
                continue
            assistant_text = last_message(messages, "assistant")
            if not assistant_text.strip() or extract_boxed_answer(assistant_text) is None:
                stats["dropped_bad_assistant"] += 1
                continue

            prompt_norm = normalize_prompt(first_message(messages, "user"))
            if record_id in test_ids:
                stats["dropped_test_id_overlap"] += 1
                continue
            if prompt_norm in test_prompts:
                stats["dropped_test_prompt_overlap"] += 1
                continue

            seen_ids.add(record_id)
            dst.write(raw + "\n")
            stats["kept_rows"] += 1

    return {
        "path": str(clean_path),
        "sha256": sha256_file(clean_path),
        **dict(stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "runs" / "kg1_training_data_gate.json")
    parser.add_argument("--clean-jsonl", type=Path)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jsonl.exists():
        print(f"[ERROR] JSONL not found: {args.jsonl}", file=sys.stderr)
        return 2
    try:
        report = validate_jsonl(args.jsonl, args.train_csv, args.test_csv)
        if args.clean_jsonl:
            if args.test_csv is None:
                print("[ERROR] --clean-jsonl requires --test-csv", file=sys.stderr)
                return 2
            report["clean_jsonl"] = write_clean_jsonl(args.jsonl, args.clean_jsonl, args.test_csv)
    except FileNotFoundError as exc:
        report = {
            "generated_at": utc_now(),
            "path": str(args.jsonl),
            "valid": False,
            "rows": 0,
            "boxed_rate": 0.0,
            "reasons": [str(exc)],
            "warnings": [],
        }
        write_json(args.output_json, report)
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(f"report: {args.output_json}")
        return 2
    write_json(args.output_json, report)
    print(f"valid: {report['valid']}")
    print(f"rows: {report['rows']}")
    print(f"boxed_rate: {report['boxed_rate']:.6f}")
    print(f"reasons: {report['reasons']}")
    print(f"warnings: {report['warnings']}")
    print(f"report: {args.output_json}")
    if args.fail_on_block and not report["valid"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

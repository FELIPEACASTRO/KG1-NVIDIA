#!/usr/bin/env python3
r"""V659 local output-policy and false-gain audit gate.

This CPU-only gate implements the V654/V657 crisis roadmap checks that must
run before another paid train/eval job:

* fail-closed dataset path and weak-reference validation;
* duplicate/contradictory prompt detection;
* weak overlap by id, prompt hash, and prompt+answer hash;
* exactly one boxed answer target, optional answer-first requirement;
* byte-equal boxed target against the row answer;
* label-free and expected-aware extraction parity;
* assistant length, first boxed position, non-ASCII, control-char checks;
* family/subcategory weights and bit-operation coverage summaries.

It does not train, launch HF, call Kaggle, or submit. Tokenization/mask checks
still require V286 because this script intentionally avoids loading the model
tokenizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    box_answer,
    canonical_answer,
    extract_boxed_answers,
    extract_final_answer,
    extract_final_answer_for_expected,
    verify_answer,
)


DEFAULT_WEAK_REFERENCE_CSV = (
    ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/"
    / "v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/v659_local_output_policy_gate"

ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)
BIT_OP_PATTERNS = {
    "AND": re.compile(r"\bAND\b|AND[-_ ]?NOT", re.IGNORECASE),
    "OR": re.compile(r"\bOR\b|OR[-_ ]?NOT", re.IGNORECASE),
    "XOR": re.compile(r"\bXOR\b", re.IGNORECASE),
    "NOT": re.compile(r"\bNOT\b|AND[-_ ]?NOT|OR[-_ ]?NOT", re.IGNORECASE),
    "SHL": re.compile(r"\bSHL\b|shift left", re.IGNORECASE),
    "SHR": re.compile(r"\bSHR\b|shift right", re.IGNORECASE),
    "ROT": re.compile(r"\bROT\b|rotate", re.IGNORECASE),
    "MAJ": re.compile(r"\bMAJ\b|majority", re.IGNORECASE),
    "CHO": re.compile(r"\bCHO\b|choice", re.IGNORECASE),
    "INPUT": re.compile(r"\bI\d+\b", re.IGNORECASE),
    "CONST": re.compile(r"\bC[01]\b", re.IGNORECASE),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value if value is not None else "").encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", canonical_answer(value)).strip()


def prompt_answer_hash(prompt: object, answer: object) -> str:
    return sha256_text(normalize_text(prompt) + "\n===ANSWER===\n" + normalize_text(answer))


def windows_long_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    return Path("\\\\?\\" + text)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def short_counter(counter: Counter[Any], limit: int = 50) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"weak reference CSV not found; fail-closed: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_reference(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    prompt_answer_hashes: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if row_id:
            ids.add(row_id)
        if prompt:
            prompt_hashes.add(sha256_text(normalize_text(prompt)))
            if answer:
                prompt_answer_hashes.add(prompt_answer_hash(prompt, answer))
    return {
        "ids": ids,
        "prompt_hashes": prompt_hashes,
        "prompt_answer_hashes": prompt_answer_hashes,
    }


def row_messages(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", "", "", []
    roles: list[str] = []
    contents: dict[str, str] = {}
    for item in messages:
        if not isinstance(item, dict):
            roles.append("<non-dict>")
            continue
        role = str(item.get("role", ""))
        roles.append(role)
        contents.setdefault(role, str(item.get("content", "")))
    return contents.get("system", ""), contents.get("user", ""), contents.get("assistant", ""), roles


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def row_family(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("task_type") or row.get("type") or metadata.get("family") or "").strip()


def row_subcategory(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("subtype") or "").strip()


def row_answer(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(row.get("answer") or metadata.get("answer") or "").strip()


def non_ascii_count(value: str) -> int:
    return sum(1 for char in value if ord(char) > 127)


def suspicious_control_count(value: str) -> int:
    allowed = {"\n", "\r", "\t"}
    return sum(1 for char in value if ord(char) < 32 and char not in allowed)


def word_count(value: str) -> int:
    return len(re.findall(r"\S+", value or ""))


def bit_ops_for_row(row: dict[str, Any], assistant: str, metadata: dict[str, Any]) -> set[str]:
    if row_family(row, metadata) != "bit_manipulation":
        return set()
    blob = assistant + "\n" + json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    return {name for name, pattern in BIT_OP_PATTERNS.items() if pattern.search(blob)}


def exact_box_text(answer: str) -> str:
    return box_answer(answer)


def audit_dataset_file(
    path: Path,
    split: str,
    weak_ref: dict[str, set[str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset JSONL not found; fail-closed: {path}")

    rows_for_csv: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except Exception as exc:
                blockers.append(
                    {"split": split, "path": str(path), "line": line_no, "code": "json_parse_error", "detail": repr(exc)}
                )
                continue
            if not isinstance(row, dict):
                blockers.append(
                    {
                        "split": split,
                        "path": str(path),
                        "line": line_no,
                        "code": "row_not_object",
                        "detail": type(row).__name__,
                    }
                )
                continue
            row["_line_no"] = line_no
            parsed_rows.append(row)

    id_counter: Counter[str] = Counter()
    prompt_hash_counter: Counter[str] = Counter()
    prompt_answer_counter: Counter[str] = Counter()
    prompt_to_answers: dict[str, set[str]] = defaultdict(set)
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    role_sequence_counts: Counter[str] = Counter()
    bit_op_counts: Counter[str] = Counter()
    family_weight_sum: Counter[str] = Counter()
    subcategory_weight_sum: Counter[str] = Counter()
    assistant_words: list[float] = []
    assistant_chars: list[float] = []
    first_box_word_idx_values: list[float] = []

    for row in parsed_rows:
        metadata = metadata_of(row)
        line_no = int(row.get("_line_no", 0))
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        answer = row_answer(row, metadata)
        family = row_family(row, metadata)
        subcategory = row_subcategory(row, metadata)
        system, user, assistant, roles = row_messages(row)
        issue_codes: list[str] = []

        if not row_id:
            issue_codes.append("missing_id")
        if not prompt:
            issue_codes.append("missing_prompt")
        if not answer:
            issue_codes.append("missing_answer")
        if not family:
            issue_codes.append("missing_family")
        if not isinstance(row.get("messages"), list):
            issue_codes.append("missing_messages")

        role_sequence = ",".join(roles)
        role_sequence_counts[role_sequence] += 1
        if role_sequence not in {"system,user,assistant", "user,assistant"}:
            issue_codes.append("unexpected_message_roles")
        if args.require_starts_boxed and re.search(r"\bverify\b.*\bbriefly\b|\bexplain\b|\bshow\b.*\bwork\b", system, re.IGNORECASE):
            issue_codes.append("system_prompt_conflicts_answer_only")

        allowed_user_prompts = {prompt.strip(), (prompt + PROMPT_SUFFIX).strip()}
        if user and prompt and user.strip() not in allowed_user_prompts:
            issue_codes.append("prompt_user_mismatch")

        for flag in ANTI_LEAK_FLAGS:
            if flag not in metadata:
                issue_codes.append(f"missing_anti_leak_flag:{flag}")
            elif metadata.get(flag) is not False:
                issue_codes.append(f"anti_leak_flag_true:{flag}")

        prompt_hash = sha256_text(normalize_text(prompt))
        pa_hash = prompt_answer_hash(prompt, answer)
        if row_id and row_id in weak_ref["ids"]:
            issue_codes.append("weak_overlap_id")
        if prompt and prompt_hash in weak_ref["prompt_hashes"]:
            issue_codes.append("weak_overlap_prompt_hash")
        if prompt and answer and pa_hash in weak_ref["prompt_answer_hashes"]:
            issue_codes.append("weak_overlap_prompt_answer_hash")

        boxed_answers = extract_boxed_answers(assistant)
        boxed_count = len(boxed_answers)
        exact_box = exact_box_text(answer)
        boxed_marker_idx = assistant.find(r"\boxed{")
        before_first_box = assistant[:boxed_marker_idx] if boxed_marker_idx >= 0 else assistant
        first_box_word_idx = word_count(before_first_box) if boxed_marker_idx >= 0 else None
        starts_boxed = assistant.lstrip().startswith(r"\boxed{")
        target_suffix_ok = bool(answer) and assistant.rstrip().endswith(exact_box)
        exact_box_present = bool(answer) and exact_box in assistant
        label_free_prediction = extract_final_answer(assistant)
        expected_aware_prediction = extract_final_answer_for_expected(assistant, answer)
        label_free_correct = bool(answer) and verify_answer(answer, label_free_prediction)
        expected_aware_correct = bool(answer) and verify_answer(answer, expected_aware_prediction)

        if boxed_count == 0:
            issue_codes.append("boxed_missing")
        if boxed_count > 1 and not args.allow_multiple_boxed:
            issue_codes.append("boxed_multiple")
        if args.require_starts_boxed and not starts_boxed:
            issue_codes.append("starts_boxed_required_failed")
        if first_box_word_idx is None:
            issue_codes.append("first_box_missing")
        elif first_box_word_idx > args.max_first_box_word_index:
            issue_codes.append("first_box_word_idx_gt_limit")
        if answer and not target_suffix_ok and not args.allow_trace_after_box:
            issue_codes.append("target_suffix_not_exact_box")
        if answer and not exact_box_present:
            issue_codes.append("exact_box_text_absent")
        if answer and not label_free_correct:
            issue_codes.append("label_free_extraction_mismatch")
        if answer and not expected_aware_correct:
            issue_codes.append("expected_aware_extraction_mismatch")
        if answer and label_free_prediction != expected_aware_prediction:
            issue_codes.append("label_free_expected_aware_diverge")

        assistant_non_ascii = non_ascii_count(assistant)
        prompt_non_ascii = non_ascii_count(prompt)
        assistant_controls = suspicious_control_count(assistant)
        prompt_controls = suspicious_control_count(prompt)
        if assistant_non_ascii and not args.allow_non_ascii_assistant:
            issue_codes.append("assistant_non_ascii")
        if assistant_controls:
            issue_codes.append("assistant_control_chars")
        if prompt_controls:
            issue_codes.append("prompt_control_chars")

        current_words = word_count(assistant)
        current_chars = len(assistant)
        assistant_words.append(float(current_words))
        assistant_chars.append(float(current_chars))
        if first_box_word_idx is not None:
            first_box_word_idx_values.append(float(first_box_word_idx))

        ops = bit_ops_for_row(row, assistant, metadata)
        for op in ops:
            bit_op_counts[op] += 1

        try:
            weight = float(metadata.get("loss_weight", row.get("loss_weight", 1.0)))
        except Exception:
            weight = 1.0
            issue_codes.append("loss_weight_not_numeric")
        family_weight_sum[family] += weight
        subcategory_weight_sum[subcategory] += weight

        id_counter[row_id] += 1
        prompt_hash_counter[prompt_hash] += 1
        prompt_answer_counter[pa_hash] += 1
        prompt_to_answers[prompt_hash].add(answer)
        family_counts[family] += 1
        subcategory_counts[subcategory] += 1
        schema_counts[str(metadata.get("schema_version") or row.get("schema_version") or "")] += 1
        source_counts[str(row.get("source") or metadata.get("source") or metadata.get("source_dataset") or "")] += 1

        for code in issue_codes:
            issue_counts[code] += 1

        row_record = {
            "split": split,
            "path": str(path),
            "line": line_no,
            "id": row_id,
            "family": family,
            "subcategory": subcategory,
            "answer": answer,
            "assistant_words": current_words,
            "assistant_chars": current_chars,
            "first_box_word_idx": "" if first_box_word_idx is None else first_box_word_idx,
            "boxed_count": boxed_count,
            "starts_boxed": starts_boxed,
            "target_suffix_ok": target_suffix_ok,
            "exact_box_present": exact_box_present,
            "label_free_prediction": label_free_prediction,
            "label_free_correct": label_free_correct,
            "expected_aware_prediction": expected_aware_prediction,
            "expected_aware_correct": expected_aware_correct,
            "assistant_non_ascii": assistant_non_ascii,
            "prompt_non_ascii": prompt_non_ascii,
            "assistant_controls": assistant_controls,
            "prompt_controls": prompt_controls,
            "loss_weight": weight,
            "issue_codes": "|".join(sorted(issue_codes)),
        }
        rows_for_csv.append(row_record)

    for row_id, count in id_counter.items():
        if row_id and count > 1:
            issue_counts["duplicate_id"] += count
            blockers.append({"split": split, "path": str(path), "code": "duplicate_id", "id": row_id, "count": count})
    for prompt_hash, answers in prompt_to_answers.items():
        if len(answers) > 1:
            issue_counts["prompt_answer_conflict"] += len(answers)
            blockers.append(
                {
                    "split": split,
                    "path": str(path),
                    "code": "prompt_answer_conflict",
                    "prompt_hash": prompt_hash,
                    "answers": sorted(answers),
                }
            )
    for prompt_hash, count in prompt_hash_counter.items():
        if count > 1:
            issue_counts["duplicate_prompt_hash"] += count
    for pa_hash, count in prompt_answer_counter.items():
        if count > 1:
            issue_counts["duplicate_prompt_answer_hash"] += count

    blocking_codes = {
        "json_parse_error",
        "row_not_object",
        "missing_id",
        "missing_prompt",
        "missing_answer",
        "missing_family",
        "missing_messages",
        "unexpected_message_roles",
        "prompt_user_mismatch",
        "system_prompt_conflicts_answer_only",
        "missing_anti_leak_flag:weak_gate_rows_used_for_training",
        "missing_anti_leak_flag:gate_rows_used_for_training",
        "missing_anti_leak_flag:full_gate_rows_used_for_training",
        "anti_leak_flag_true:weak_gate_rows_used_for_training",
        "anti_leak_flag_true:gate_rows_used_for_training",
        "anti_leak_flag_true:full_gate_rows_used_for_training",
        "weak_overlap_id",
        "weak_overlap_prompt_hash",
        "weak_overlap_prompt_answer_hash",
        "boxed_missing",
        "boxed_multiple",
        "starts_boxed_required_failed",
        "first_box_missing",
        "first_box_word_idx_gt_limit",
        "target_suffix_not_exact_box",
        "exact_box_text_absent",
        "label_free_extraction_mismatch",
        "expected_aware_extraction_mismatch",
        "label_free_expected_aware_diverge",
        "assistant_non_ascii",
        "assistant_control_chars",
        "prompt_control_chars",
        "loss_weight_not_numeric",
        "duplicate_id",
        "prompt_answer_conflict",
    }
    for code, count in issue_counts.items():
        payload = {"split": split, "path": str(path), "code": code, "count": int(count)}
        if code in blocking_codes:
            blockers.append(payload)
        else:
            warnings.append(payload)

    p95_words = percentile(assistant_words, 0.95)
    if p95_words is not None and p95_words > args.max_assistant_p95_words:
        blockers.append(
            {
                "split": split,
                "path": str(path),
                "code": "assistant_p95_words_gt_limit",
                "value": p95_words,
                "limit": args.max_assistant_p95_words,
            }
        )

    required_bit_ops = {"AND", "OR", "XOR", "NOT", "SHL", "SHR", "ROT"}
    if family_counts.get("bit_manipulation", 0):
        missing_ops = sorted(op for op in required_bit_ops if bit_op_counts.get(op, 0) == 0)
        if missing_ops:
            warnings.append(
                {
                    "split": split,
                    "path": str(path),
                    "code": "bit_required_op_absent_from_dataset",
                    "missing_ops": missing_ops,
                }
            )

    total_weight = sum(family_weight_sum.values())
    effective_family_weights = {
        family: (float(value) / float(total_weight) if total_weight else 0.0)
        for family, value in sorted(family_weight_sum.items())
    }
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "split": split,
        "rows": len(parsed_rows),
        "family_counts": short_counter(family_counts),
        "subcategory_counts": short_counter(subcategory_counts),
        "schema_counts": short_counter(schema_counts),
        "source_counts": short_counter(source_counts),
        "role_sequence_counts": short_counter(role_sequence_counts),
        "issue_counts": short_counter(issue_counts, limit=200),
        "assistant_word_stats": stats(assistant_words),
        "assistant_char_stats": stats(assistant_chars),
        "first_box_word_idx_stats": stats(first_box_word_idx_values),
        "family_weight_sum": {str(key): float(value) for key, value in sorted(family_weight_sum.items())},
        "subcategory_weight_sum_top": {str(key): float(value) for key, value in subcategory_weight_sum.most_common(50)},
        "effective_family_weights": effective_family_weights,
        "bit_op_counts": short_counter(bit_op_counts),
    }
    return rows_for_csv, blockers + warnings, summary


def cross_split_checks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    train_prompt_hashes: set[str] = set()
    train_pa_hashes: set[str] = set()
    val_prompt_hashes: set[str] = set()
    val_pa_hashes: set[str] = set()
    for row in rows:
        prompt_hash = sha256_text(normalize_text(row.get("prompt", "")))
        pa_hash = prompt_answer_hash(row.get("prompt", ""), row.get("answer", ""))
        if row.get("split") == "train":
            train_prompt_hashes.add(prompt_hash)
            train_pa_hashes.add(pa_hash)
        elif row.get("split") in {"val", "validation"}:
            val_prompt_hashes.add(prompt_hash)
            val_pa_hashes.add(pa_hash)
    prompt_overlap = len(train_prompt_hashes & val_prompt_hashes)
    prompt_answer_overlap = len(train_pa_hashes & val_pa_hashes)
    if prompt_overlap:
        blockers.append({"code": "train_val_prompt_overlap_nonzero", "count": prompt_overlap})
    if prompt_answer_overlap:
        blockers.append({"code": "train_val_prompt_answer_overlap_nonzero", "count": prompt_answer_overlap})
    return blockers, {
        "train_prompt_hashes": len(train_prompt_hashes),
        "val_prompt_hashes": len(val_prompt_hashes),
        "train_val_prompt_overlap": prompt_overlap,
        "train_val_prompt_answer_overlap": prompt_answer_overlap,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "path",
        "line",
        "id",
        "family",
        "subcategory",
        "answer",
        "assistant_words",
        "assistant_chars",
        "first_box_word_idx",
        "boxed_count",
        "starts_boxed",
        "target_suffix_ok",
        "exact_box_present",
        "label_free_prediction",
        "label_free_correct",
        "expected_aware_prediction",
        "expected_aware_correct",
        "assistant_non_ascii",
        "prompt_non_ascii",
        "assistant_controls",
        "prompt_controls",
        "loss_weight",
        "issue_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KG1 V659 Local Output Policy Gate",
        "",
        f"- generated_at_utc: `{manifest['generated_at_utc']}`",
        f"- label: `{manifest['label']}`",
        f"- status: `{manifest['decision']['status']}`",
        f"- submit_allowed: `{manifest['decision']['submit_allowed']}`",
        f"- train_or_eval_allowed: `{manifest['decision']['train_or_eval_allowed']}`",
        f"- blocker_count: `{manifest['decision']['blocker_count']}`",
        f"- warning_count: `{manifest['decision']['warning_count']}`",
        "",
        "## Inputs",
        "",
    ]
    for item in manifest["inputs"]["datasets"]:
        lines.append(f"- `{item['split']}`: `{item['path']}` sha256 `{item['sha256']}`")
    lines.extend(
        [
            f"- weak_reference_csv: `{manifest['inputs']['weak_reference_csv']['path']}`",
            f"- weak_reference_sha256: `{manifest['inputs']['weak_reference_csv']['sha256']}`",
            "",
            "## Dataset Summaries",
            "",
        ]
    )
    for item in manifest["dataset_summaries"]:
        lines.extend(
            [
                f"### {item['split']}",
                "",
                f"- rows: `{item['rows']}`",
                f"- family_counts: `{json.dumps(item['family_counts'], sort_keys=True)}`",
                f"- effective_family_weights: `{json.dumps(item['effective_family_weights'], sort_keys=True)}`",
                f"- assistant_word_stats: `{json.dumps(item['assistant_word_stats'], sort_keys=True)}`",
                f"- first_box_word_idx_stats: `{json.dumps(item['first_box_word_idx_stats'], sort_keys=True)}`",
                f"- bit_op_counts: `{json.dumps(item['bit_op_counts'], sort_keys=True)}`",
                f"- issue_counts: `{json.dumps(item['issue_counts'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross Split",
            "",
            f"`{json.dumps(manifest['cross_split'], sort_keys=True)}`",
            "",
            "## Top Blockers",
            "",
        ]
    )
    for item in manifest["blockers"][:50]:
        lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
    if not manifest["blockers"]:
        lines.append("- none")
    lines.extend(["", "## Top Warnings", ""])
    for item in manifest["warnings"][:50]:
        lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
    if not manifest["warnings"]:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None, help="Optional split names matching dataset order.")
    parser.add_argument("--weak-reference-csv", type=Path, default=DEFAULT_WEAK_REFERENCE_CSV)
    parser.add_argument("--label", default="v659_local_output_policy")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-first-box-word-index", type=int, default=50)
    parser.add_argument("--max-assistant-p95-words", type=int, default=128)
    parser.add_argument("--require-starts-boxed", action="store_true")
    parser.add_argument("--allow-trace-after-box", action="store_true")
    parser.add_argument("--allow-multiple-boxed", action="store_true")
    parser.add_argument("--allow-non-ascii-assistant", action="store_true")
    return parser.parse_args()


def infer_split(path: Path, index: int) -> str:
    lower = path.name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower or "validation" in lower:
        return "validation"
    return f"dataset_{index}"


def main() -> int:
    args = parse_args()
    dataset_paths = [path.resolve() for path in args.dataset_jsonl]
    missing = [str(path) for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("dataset JSONL path(s) not found; fail-closed: " + json.dumps(missing))

    split_names = args.split or [infer_split(path, index) for index, path in enumerate(dataset_paths)]
    if len(split_names) != len(dataset_paths):
        raise ValueError("--split count must match --dataset-jsonl count")

    weak_rows = read_csv_rows(args.weak_reference_csv)
    weak_ref = build_reference(weak_rows)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / (args.label + "_" + datetime.now().strftime("%Y%m%dT%H%M%SZ"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_row_audits: list[dict[str, Any]] = []
    all_blockers: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for split, path in zip(split_names, dataset_paths):
        row_audits, issues, summary = audit_dataset_file(path, split, weak_ref, args)
        all_row_audits.extend(row_audits)
        dataset_summaries.append(summary)
        for record in row_audits:
            cross_rows.append({"split": split, "prompt": "", "answer": record["answer"]})
        blocker_codes = {str(item.get("code", "")) for item in issues if isinstance(item, dict)}
        for item in issues:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", ""))
            if (
                code.startswith("duplicate_prompt")
                or code.startswith("duplicate_prompt_answer")
                or code == "bit_required_op_absent_from_dataset"
            ):
                all_warnings.append(item)
            elif code:
                all_blockers.append(item)
        _ = blocker_codes

    # Re-read minimal fields for cross-split overlap without storing prompts in
    # the public row CSV.
    cross_payload_rows: list[dict[str, Any]] = []
    for split, path in zip(split_names, dataset_paths):
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                metadata = metadata_of(row)
                cross_payload_rows.append({"split": split, "prompt": row.get("prompt", ""), "answer": row_answer(row, metadata)})
    cross_blockers, cross_summary = cross_split_checks(cross_payload_rows)
    all_blockers.extend(cross_blockers)

    blocker_count = len(all_blockers)
    warning_count = len(all_warnings)
    row_csv = output_dir / "v659_row_output_policy_audit.csv"
    manifest_json = output_dir / "v659_local_output_policy_gate_manifest.json"
    summary_md = output_dir / "KG1_V659_LOCAL_OUTPUT_POLICY_GATE.md"
    write_csv(row_csv, all_row_audits)
    manifest = {
        "label": args.label,
        "generated_at_utc": utc_now(),
        "schema_version": "kg1_v659_local_output_policy_gate_v1",
        "inputs": {
            "datasets": [
                {"split": split, "path": str(path), "sha256": sha256_file(path)}
                for split, path in zip(split_names, dataset_paths)
            ],
            "weak_reference_csv": {
                "path": str(args.weak_reference_csv),
                "sha256": sha256_file(args.weak_reference_csv),
                "rows": len(weak_rows),
            },
            "thresholds": {
                "max_first_box_word_index": args.max_first_box_word_index,
                "max_assistant_p95_words": args.max_assistant_p95_words,
                "require_starts_boxed": bool(args.require_starts_boxed),
                "allow_trace_after_box": bool(args.allow_trace_after_box),
                "allow_multiple_boxed": bool(args.allow_multiple_boxed),
                "allow_non_ascii_assistant": bool(args.allow_non_ascii_assistant),
            },
        },
        "outputs": {
            "manifest_json": str(manifest_json),
            "summary_md": str(summary_md),
            "row_csv": str(row_csv),
        },
        "dataset_summaries": dataset_summaries,
        "cross_split": cross_summary,
        "blockers": all_blockers,
        "warnings": all_warnings,
        "decision": {
            "status": "blocked" if blocker_count else "passed",
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "train_or_eval_allowed": blocker_count == 0,
            "submit_allowed": False,
            "next_action": (
                "Fix blockers before any train/eval job."
                if blocker_count
                else "Run V286 tokenization/mask gate and holdout non-weak eval before any paid job."
            ),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary_md, manifest)
    print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
    print(f"manifest={manifest_json}")
    print(f"summary={summary_md}")
    print(f"row_csv={row_csv}")
    return 1 if blocker_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

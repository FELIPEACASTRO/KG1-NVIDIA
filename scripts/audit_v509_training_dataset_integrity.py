#!/usr/bin/env python3
"""Audit KG1 training/validation JSONL datasets before any new paid job.

The goal is to catch cheap failures before HF/Kaggle compute:

* malformed JSONL rows;
* duplicate IDs, duplicate prompts, and prompt-answer conflicts;
* overlap with weak/full reference rows by id, prompt hash, or prompt+answer;
* missing anti-leakage flags;
* blocked/quarantined dataset markers;
* assistant final-answer mismatch with the row answer;
* non-boxed final answer formatting where the current training recipe expects
  boxed suffix style.

This script is CPU-only. It never trains, launches HF, packages, or submits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v509_training_dataset_integrity_audit"
DEFAULT_WEAK_REFERENCE_CSV = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_FULL_REFERENCE_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"

BLOCKED_MARKERS = {
    "v447_v446_trace_dataset": "V447 traces contain contradictory hypothesis_formed answers.",
    "v461_synthetic_numeric_probe_pack": "V461 contained full-reference exact prompt/answer seed.",
    "v463_v462_synthetic_numeric_hard_negative_audit": "V463 depends on quarantined V461/V462.",
    "v464_v463_numeric_multirule_dataset": "V464 rejected candidates can equal answers.",
    "v468_v464_symbol_fix_dataset": "V468 still contains full-reference exact prompt/answer seed.",
}
ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def prompt_answer_hash(prompt: str, answer: str) -> str:
    return sha256_text(str(prompt) + "\n===ANSWER===\n" + str(answer))


def compact_counter(counter: Counter[Any], limit: int = 20) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_reference(rows: list[dict[str, str]]) -> dict[str, set[str]]:
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
            prompt_hashes.add(sha256_text(prompt))
            if answer:
                prompt_answer_hashes.add(prompt_answer_hash(prompt, answer))
    return {
        "ids": ids,
        "prompt_hashes": prompt_hashes,
        "prompt_answer_hashes": prompt_answer_hashes,
    }


def discover_dataset_paths(explicit_paths: list[Path]) -> list[Path]:
    if explicit_paths:
        return sorted({path.resolve() for path in explicit_paths if path.exists()})
    candidates: set[Path] = set()
    for root in (REPO_ROOT / "artifacts", REPO_ROOT / "data"):
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            name = path.name.lower()
            if any(token in name for token in ("train", "val", "validation", "preferences")):
                candidates.add(path.resolve())
    return sorted(candidates)


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


def audit_file(path: Path, weak_ref: dict[str, set[str]], full_ref: dict[str, set[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                parse_errors += 1
                issues.append({"path": str(path), "line": line_no, "code": "json_parse_error", "detail": repr(exc)})
                continue
            if not isinstance(row, dict):
                issues.append({"path": str(path), "line": line_no, "code": "row_not_object", "detail": type(row).__name__})
                continue
            row["_line_no"] = line_no
            rows.append(row)

    id_counter: Counter[str] = Counter()
    prompt_counter: Counter[str] = Counter()
    prompt_answer_counter: Counter[str] = Counter()
    prompt_to_answers: dict[str, set[str]] = defaultdict(set)
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    assistant_style_counts: Counter[str] = Counter()
    missing_flags: Counter[str] = Counter()
    true_flags: Counter[str] = Counter()
    blocked_hits: Counter[str] = Counter()
    reference_overlap_counts: Counter[str] = Counter()
    assistant_mismatch_rows: list[str] = []
    nonboxed_rows: list[str] = []
    missing_required_rows: list[str] = []
    empty_answer_rows: list[str] = []
    empty_prompt_rows: list[str] = []
    raw_output_rows: list[str] = []

    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer") or metadata.get("answer") or "").strip()
        family = str(
            row.get("family") or row.get("task_type") or row.get("type") or metadata.get("family") or ""
        ).strip()
        system, user, assistant, roles = row_messages(row)

        for key in ("id", "prompt", "family", "messages"):
            if key not in row or row.get(key) in (None, ""):
                missing_required_rows.append(row_id or f"line:{row['_line_no']}:{key}")
        if not answer:
            missing_required_rows.append(row_id or f"line:{row['_line_no']}:answer")
        if not answer:
            empty_answer_rows.append(row_id or f"line:{row['_line_no']}")
        if not prompt:
            empty_prompt_rows.append(row_id or f"line:{row['_line_no']}")
        if "raw_output" in row:
            raw_output_rows.append(row_id or f"line:{row['_line_no']}")

        id_counter[row_id] += 1
        prompt_hash = sha256_text(prompt)
        pa_hash = prompt_answer_hash(prompt, answer)
        prompt_counter[prompt_hash] += 1
        prompt_answer_counter[pa_hash] += 1
        prompt_to_answers[prompt_hash].add(answer)
        family_counts[family] += 1
        source_counts[str(row.get("source") or metadata.get("source") or "")] += 1
        subcategory_counts[str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("subtype") or "")] += 1
        schema_counts[str(metadata.get("schema_version") or row.get("schema_version") or "")] += 1

        if "\\boxed{" in assistant:
            assistant_style_counts["boxed"] += 1
        elif "Final answer:" in assistant:
            assistant_style_counts["final_answer_unboxed"] += 1
            nonboxed_rows.append(row_id or f"line:{row['_line_no']}")
        else:
            assistant_style_counts["other_or_missing"] += 1
            nonboxed_rows.append(row_id or f"line:{row['_line_no']}")

        extracted = extract_final_answer(assistant)
        if assistant and answer and not verify_answer(answer, extracted):
            assistant_mismatch_rows.append(row_id or f"line:{row['_line_no']}")

        role_sequence = ",".join(roles)
        if role_sequence != "system,user,assistant":
            issues.append(
                {
                    "path": str(path),
                    "line": row["_line_no"],
                    "id": row_id,
                    "code": "unexpected_message_roles",
                    "detail": role_sequence,
                }
            )
        if user and prompt and user.strip() != prompt.strip():
            issues.append(
                {
                    "path": str(path),
                    "line": row["_line_no"],
                    "id": row_id,
                    "code": "prompt_user_mismatch",
                    "detail": "row prompt differs from user message",
                }
            )
        if system and "final answer" not in system.lower():
            issues.append(
                {
                    "path": str(path),
                    "line": row["_line_no"],
                    "id": row_id,
                    "code": "system_prompt_lacks_final_answer_instruction",
                    "detail": system[:160],
                }
            )

        for flag in ANTI_LEAK_FLAGS:
            if flag not in metadata:
                missing_flags[flag] += 1
            elif bool(metadata.get(flag)):
                true_flags[flag] += 1

        combined_text = "\n".join([str(path), json.dumps(row, sort_keys=True, ensure_ascii=False)])
        for marker in BLOCKED_MARKERS:
            if marker in combined_text:
                blocked_hits[marker] += 1

        if row_id in weak_ref["ids"]:
            reference_overlap_counts["weak_id"] += 1
        if prompt_hash in weak_ref["prompt_hashes"]:
            reference_overlap_counts["weak_prompt"] += 1
        if pa_hash in weak_ref["prompt_answer_hashes"]:
            reference_overlap_counts["weak_prompt_answer"] += 1
        if row_id in full_ref["ids"]:
            reference_overlap_counts["full_id"] += 1
        if prompt_hash in full_ref["prompt_hashes"]:
            reference_overlap_counts["full_prompt"] += 1
        if pa_hash in full_ref["prompt_answer_hashes"]:
            reference_overlap_counts["full_prompt_answer"] += 1

    duplicate_ids = sum(count - 1 for key, count in id_counter.items() if key and count > 1)
    duplicate_prompts = sum(count - 1 for count in prompt_counter.values() if count > 1)
    duplicate_prompt_answers = sum(count - 1 for count in prompt_answer_counter.values() if count > 1)
    prompt_answer_conflicts = sum(1 for answers in prompt_to_answers.values() if len(answers) > 1)

    hard_fail_codes: list[str] = []
    if not rows:
        hard_fail_codes.append("empty_dataset")
    if parse_errors:
        hard_fail_codes.append("json_parse_error")
    if duplicate_ids:
        hard_fail_codes.append("duplicate_ids")
    if prompt_answer_conflicts:
        hard_fail_codes.append("prompt_answer_conflicts")
    if assistant_mismatch_rows:
        hard_fail_codes.append("assistant_answer_mismatch")
    if true_flags:
        hard_fail_codes.append("anti_leak_flag_true")
    if blocked_hits:
        hard_fail_codes.append("blocked_marker_present")
    if reference_overlap_counts:
        hard_fail_codes.append("reference_overlap")
    if missing_required_rows:
        hard_fail_codes.append("missing_required_fields")

    issue_type_counts = Counter(issue["code"] for issue in issues if issue.get("path") == str(path))
    blocking_issue_counts = Counter()
    for code in hard_fail_codes:
        if code == "empty_dataset":
            blocking_issue_counts[code] = 1
        elif code == "json_parse_error":
            blocking_issue_counts[code] = parse_errors
        elif code == "duplicate_ids":
            blocking_issue_counts[code] = duplicate_ids
        elif code == "prompt_answer_conflicts":
            blocking_issue_counts[code] = prompt_answer_conflicts
        elif code == "assistant_answer_mismatch":
            blocking_issue_counts[code] = len(assistant_mismatch_rows)
        elif code == "anti_leak_flag_true":
            blocking_issue_counts[code] = sum(true_flags.values())
        elif code == "blocked_marker_present":
            blocking_issue_counts[code] = sum(blocked_hits.values())
        elif code == "reference_overlap":
            blocking_issue_counts[code] = sum(reference_overlap_counts.values())
        elif code == "missing_required_fields":
            blocking_issue_counts[code] = len(missing_required_rows)

    summary = {
        "dataset": path.stem,
        "path": str(path),
        "row_count": len(rows),
        "rows": len(rows),
        "blocking_issue_count": int(sum(blocking_issue_counts.values())),
        "blocking_issue_counts": json.dumps(compact_counter(blocking_issue_counts), sort_keys=True),
        "issue_type_counts": json.dumps(compact_counter(issue_type_counts), sort_keys=True),
        "parse_errors": parse_errors,
        "duplicate_ids": duplicate_ids,
        "duplicate_prompts": duplicate_prompts,
        "duplicate_prompt_answers": duplicate_prompt_answers,
        "prompt_answer_conflicts": prompt_answer_conflicts,
        "missing_required_rows": len(missing_required_rows),
        "empty_answer_rows": len(empty_answer_rows),
        "empty_prompt_rows": len(empty_prompt_rows),
        "raw_output_rows": len(raw_output_rows),
        "assistant_answer_mismatch_rows": len(assistant_mismatch_rows),
        "nonboxed_rows": len(nonboxed_rows),
        "missing_anti_leak_flags": json.dumps(compact_counter(missing_flags), sort_keys=True),
        "true_anti_leak_flags": json.dumps(compact_counter(true_flags), sort_keys=True),
        "blocked_markers": json.dumps(compact_counter(blocked_hits), sort_keys=True),
        "reference_overlaps": json.dumps(compact_counter(reference_overlap_counts), sort_keys=True),
        "family_counts": json.dumps(compact_counter(family_counts), sort_keys=True),
        "source_counts": json.dumps(compact_counter(source_counts), sort_keys=True),
        "subcategory_counts": json.dumps(compact_counter(subcategory_counts), sort_keys=True),
        "schema_counts": json.dumps(compact_counter(schema_counts), sort_keys=True),
        "assistant_style_counts": json.dumps(compact_counter(assistant_style_counts), sort_keys=True),
        "sample_assistant_mismatch_ids": " ".join(assistant_mismatch_rows[:20]),
        "sample_nonboxed_ids": " ".join(nonboxed_rows[:20]),
        "sample_missing_required": " ".join(missing_required_rows[:20]),
        "hard_fail_codes": " ".join(hard_fail_codes),
        "status": "blocked" if hard_fail_codes else "ok",
    }
    return summary, issues


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = discover_dataset_paths(args.dataset_jsonl)
    weak_ref = load_reference(read_csv_rows(args.weak_reference_csv))
    full_ref = load_reference(read_csv_rows(args.full_reference_csv))

    summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for path in paths:
        summary, file_issues = audit_file(path, weak_ref, full_ref)
        summaries.append(summary)
        issues.extend(file_issues)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"{args.label}_summary.csv"
    issues_csv = output_dir / f"{args.label}_issues.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"

    columns = [
        "dataset",
        "path",
        "row_count",
        "rows",
        "blocking_issue_count",
        "blocking_issue_counts",
        "issue_type_counts",
        "parse_errors",
        "duplicate_ids",
        "duplicate_prompts",
        "duplicate_prompt_answers",
        "prompt_answer_conflicts",
        "missing_required_rows",
        "empty_answer_rows",
        "empty_prompt_rows",
        "raw_output_rows",
        "assistant_answer_mismatch_rows",
        "nonboxed_rows",
        "missing_anti_leak_flags",
        "true_anti_leak_flags",
        "blocked_markers",
        "reference_overlaps",
        "family_counts",
        "source_counts",
        "subcategory_counts",
        "schema_counts",
        "assistant_style_counts",
        "sample_assistant_mismatch_ids",
        "sample_nonboxed_ids",
        "sample_missing_required",
        "hard_fail_codes",
        "status",
    ]
    write_csv(summary_csv, summaries, columns)
    write_csv(issues_csv, issues, ["path", "line", "id", "code", "detail"])

    blocked = [row for row in summaries if row["status"] != "ok"]
    manifest = {
        "schema_version": "kg1_v509_training_dataset_integrity_audit_v1",
        "generated_at_utc": utc_now(),
        "dataset_count": len(paths),
        "blocked_dataset_count": len(blocked),
        "summary_csv": str(summary_csv),
        "issues_csv": str(issues_csv),
        "weak_reference_csv": str(args.weak_reference_csv),
        "full_reference_csv": str(args.full_reference_csv),
        "decision": {
            "status": "datasets_need_triage" if blocked else "datasets_pass_integrity_audit",
            "next_action": (
                "Do not launch paid training with any dataset marked blocked. "
                "Inspect hard_fail_codes and either fix the dataset or remove it from active roadmap."
            )
            if blocked
            else "Dataset integrity checks passed for audited files; still run tokenization/pre-paid gates before HF.",
        },
        "blocked_paths": [row["path"] for row in blocked],
    }
    write_json(manifest_json, manifest)

    print("=== V509 TRAINING DATASET INTEGRITY AUDIT START ===", flush=True)
    print("dataset_count =", len(paths), flush=True)
    print("blocked_dataset_count =", len(blocked), flush=True)
    print("summary_csv =", summary_csv, flush=True)
    print("issues_csv =", issues_csv, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V509 TRAINING DATASET INTEGRITY AUDIT END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--weak-reference-csv", type=Path, default=DEFAULT_WEAK_REFERENCE_CSV)
    parser.add_argument("--full-reference-csv", type=Path, default=DEFAULT_FULL_REFERENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v509_training_dataset_integrity_audit")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build V447 trace dataset from V446 accepted rows.

The builder is CPU-only. It converts accepted V446 SFT traces into the KG1
chat JSONL contract required by the generic tokenization gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402

DEFAULT_V446_AUDIT_CSV = (
    REPO_ROOT
    / "artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/"
    / "v446_tong_source_target_alignment_gate_candidate_audit.csv"
)
DEFAULT_SFT_JSONL = Path(r"C:\Users\davis\Downloads\sft_reconstructed.jsonl")
DEFAULT_COMPETITION_TRAIN_CSV = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v447_v446_trace_dataset"

SYSTEM_PROMPT = (
    "You are solving NVIDIA Nemotron reasoning puzzles. Follow the examples, "
    "derive the hidden rule, and put the final answer in \\boxed{}."
)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def message_parts(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", ""))
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)
    return "\n".join(user_texts), "\n".join(assistant_texts)


def read_jsonl_by_row_no(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict):
                rows[row_no] = obj
    return rows


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for row in read_csv(path):
        rid = str(row.get("id", "")).strip()
        if rid:
            answers[rid] = str(row.get("answer", "")).strip()
    return answers


def normalize_assistant(content: str, answer: str, *, allow_different_boxed: bool = False) -> tuple[str, str, bool]:
    text = str(content or "").rstrip()
    final_line = r"Final answer: \boxed{" + str(answer) + "}"
    if text.endswith(final_line):
        if verify_answer(answer, extract_final_answer(text)):
            return text, "already_boxed_suffix", True
        return text, "dropped_metric_inextractable_already_boxed_suffix", False
    matches = BOXED_RE.findall(text)
    if matches and matches[-1].strip() == str(answer).strip():
        candidate = text + "\n" + final_line
        if verify_answer(answer, extract_final_answer(candidate)):
            return candidate, "appended_confirming_boxed_suffix", True
        return candidate, "dropped_metric_inextractable_confirming_boxed_suffix", False
    if matches:
        if allow_different_boxed:
            candidate = text + "\n" + final_line
            if verify_answer(answer, extract_final_answer(candidate)):
                return candidate, "appended_train_answer_after_different_boxed", True
            return candidate, "dropped_metric_inextractable_after_different_boxed", False
        return text, "dropped_different_boxed_mismatch", False
    candidate = text + "\n" + final_line
    if verify_answer(answer, extract_final_answer(candidate)):
        return candidate, "appended_train_answer_no_boxed", True
    return candidate, "dropped_metric_inextractable_no_boxed", False


def deterministic_split_key(row: dict[str, Any]) -> int:
    payload = f"{row.get('family','')}|{row.get('status','')}|{row.get('id','')}|{row.get('prompt_sha256','')}"
    return int(sha256_text(payload)[:8], 16)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows = read_csv(args.v446_audit_csv)
    accepted = [row for row in audit_rows if truthy(row.get("accepted"))]
    source_rows = read_jsonl_by_row_no(args.sft_jsonl)
    answers = load_answers(args.competition_train_csv)
    built: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "audit_rows": len(audit_rows),
        "accepted_audit_rows": len(accepted),
        "missing_source_row": 0,
        "missing_answer": 0,
        "dropped_non_rule_found_status": 0,
        "dropped_rows": 0,
        "normalization_status": Counter(),
        "family_counts": Counter(),
        "status_counts": Counter(),
    }
    for audit in accepted:
        row_no = int(str(audit.get("row_no", "0") or "0"))
        source = source_rows.get(row_no)
        if not source:
            stats["missing_source_row"] += 1
            continue
        rid = str(audit.get("id", "")).strip()
        answer = answers.get(rid, "")
        if not answer:
            stats["missing_answer"] += 1
            continue
        status = str(audit.get("status", "")).strip()
        if status != "rule_found":
            stats["dropped_non_rule_found_status"] += 1
            continue
        prompt, assistant = message_parts(source)
        if sha256_text(prompt) != str(audit.get("prompt_sha256", "")):
            raise RuntimeError(f"prompt hash mismatch for row_no={row_no} id={rid}")
        assistant_norm, norm_status, keep_row = normalize_assistant(
            assistant,
            answer,
            allow_different_boxed=args.allow_different_boxed,
        )
        stats["normalization_status"][norm_status] += 1
        if not keep_row:
            stats["dropped_rows"] += 1
            continue
        family = str(audit.get("family", "")).strip()
        row = {
            "id": f"v447_{rid}",
            "prompt": prompt,
            "answer": answer,
            "family": family,
            "subcategory": status,
            "source": "v447_v446_tong_source_target_alignment",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_norm},
            ],
            "metadata": {
                "source_dataset": "v447_v446_tong_source_target_alignment",
                "source_sft_jsonl": str(args.sft_jsonl),
                "source_v446_audit_csv": str(args.v446_audit_csv),
                "source_problem_id": rid,
                "source_row_no": row_no,
                "family": family,
                "status": status,
                "prompt_sha256": audit.get("prompt_sha256", ""),
                "prompt_normalized_sha256": audit.get("prompt_normalized_sha256", ""),
                "assistant_sha256": audit.get("assistant_sha256", ""),
                "v447_normalization_status": norm_status,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
            },
        }
        stats["family_counts"][family] += 1
        stats["status_counts"][status] += 1
        built.append(row)
    return built, stats


def split_rows(rows: list[dict[str, Any]], val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("family", "")), str(row.get("subcategory", "")))].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    for key, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=deterministic_split_key)
        val_count = max(1, round(len(ordered) * val_fraction)) if len(ordered) >= 10 else 1
        val.extend(ordered[:val_count])
        train.extend(ordered[val_count:])
        split_stats[f"{key[0]}::{key[1]}"] = {
            "rows": len(ordered),
            "train": len(ordered) - val_count,
            "val": val_count,
        }
    train = sorted(train, key=deterministic_split_key)
    val = sorted(val, key=deterministic_split_key)
    for row in train:
        row["metadata"]["split"] = "train"
    for row in val:
        row["metadata"]["split"] = "val"
    return train, val, split_stats


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    norm: Counter[str] = Counter()
    assistant_chars: list[int] = []
    for row in rows:
        families[str(row.get("family", ""))] += 1
        statuses[str(row.get("subcategory", ""))] += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        norm[str(metadata.get("v447_normalization_status", ""))] += 1
        assistant_chars.append(len(str(row["messages"][2]["content"])))
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(families.items())),
        "status_counts": dict(sorted(statuses.items())),
        "normalization_status": dict(sorted(norm.items())),
        "assistant_chars": {
            "min": min(assistant_chars) if assistant_chars else 0,
            "max": max(assistant_chars) if assistant_chars else 0,
            "avg": round(sum(assistant_chars) / len(assistant_chars), 2) if assistant_chars else 0,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / utc_compact())
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== V447 V446 TRACE DATASET BUILD START ===", flush=True)
    if not args.allow_quarantined_rebuild:
        raise RuntimeError(
            "V447/V446 trace builder is quarantined and fail-closed: crisis audit found "
            "hypothesis_formed contradictory traces and divergent internal boxed answers. "
            "Create a new rule_found-only builder/version with internal boxed-answer validation."
        )
    print("v446_audit_csv =", args.v446_audit_csv, "exists =", args.v446_audit_csv.exists(), flush=True)
    print("sft_jsonl =", args.sft_jsonl, "exists =", args.sft_jsonl.exists(), flush=True)
    print("competition_train_csv =", args.competition_train_csv, "exists =", args.competition_train_csv.exists(), flush=True)
    print("output_dir =", output_dir, flush=True)
    rows, build_stats = build_rows(args)
    if len(rows) < args.min_total_rows:
        raise RuntimeError(f"not enough rows after build: {len(rows)} < {args.min_total_rows}")
    train, val, split_stats = split_rows(rows, args.val_fraction)
    train_path = output_dir / "v447_v446_trace_train.jsonl"
    val_path = output_dir / "v447_v446_trace_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    manifest = {
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v446_audit_csv": str(args.v446_audit_csv),
            "v446_audit_sha256": sha256_file(args.v446_audit_csv),
            "sft_jsonl": str(args.sft_jsonl),
            "sft_jsonl_sha256": sha256_file(args.sft_jsonl),
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_csv_sha256": sha256_file(args.competition_train_csv),
        },
        "build_stats": {
            **build_stats,
            "normalization_status": dict(build_stats["normalization_status"]),
            "family_counts": dict(build_stats["family_counts"]),
            "status_counts": dict(build_stats["status_counts"]),
        },
        "split_stats": split_stats,
        "train_summary": summarize(train),
        "val_summary": summarize(val),
        "decision": {
            "status": "dataset_built",
            "next_action": "Run V286 generic tokenization gate with assistant_final_answer_mode=boxed_suffix.",
        },
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(output_dir / f"{args.label}_manifest.json"),
        },
    }
    write_json(output_dir / f"{args.label}_manifest.json", manifest)
    print("train_summary =", json.dumps(manifest["train_summary"], sort_keys=True), flush=True)
    print("val_summary =", json.dumps(manifest["val_summary"], sort_keys=True), flush=True)
    print("manifest_json =", output_dir / f"{args.label}_manifest.json", flush=True)
    print("=== V447 V446 TRACE DATASET BUILD END ===", flush=True)
    return manifest


def self_test() -> None:
    print("=== V447 SELF TEST START ===", flush=True)
    text, status, keep = normalize_assistant("abc " + r"\boxed{101}", "101")
    assert text.endswith(r"Final answer: \boxed{101}")
    assert status == "appended_confirming_boxed_suffix"
    assert keep is True
    text2, status2, keep2 = normalize_assistant("abc", "42")
    assert text2.endswith(r"Final answer: \boxed{42}")
    assert status2 == "appended_train_answer_no_boxed"
    assert keep2 is True
    text3, status3, keep3 = normalize_assistant("abc " + r"\boxed{999}", "42")
    assert text3.endswith(r"\boxed{999}")
    assert status3 == "dropped_different_boxed_mismatch"
    assert keep3 is False
    text4, status4, keep4 = normalize_assistant("abc " + r"\boxed{999}", "42", allow_different_boxed=True)
    assert text4.endswith(r"Final answer: \boxed{42}")
    assert status4 == "appended_train_answer_after_different_boxed"
    assert keep4 is True
    text5, status5, keep5 = normalize_assistant("abc", "a{b}\\c")
    assert text5.endswith(r"Final answer: \boxed{a{b}\c}")
    assert verify_answer("a{b}\\c", extract_final_answer(text5))
    assert status5 == "appended_train_answer_no_boxed"
    assert keep5 is True
    print("v447_self_test=ok", flush=True)
    print("=== V447 SELF TEST END ===", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--label", default="v447_v446_trace_dataset")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v446-audit-csv", type=Path, default=DEFAULT_V446_AUDIT_CSV)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_COMPETITION_TRAIN_CSV)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--min-total-rows", type=int, default=1000)
    parser.add_argument(
        "--allow-different-boxed",
        action="store_true",
        help="Keep traces whose last boxed answer differs from the official train answer. Default blocks them.",
    )
    parser.add_argument("--allow-quarantined-rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

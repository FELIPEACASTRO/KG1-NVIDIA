#!/usr/bin/env python3
"""V606 CPU audit for unused V446 public/source-only rows.

This audit does not train and does not authorize GPU by itself. It answers a
specific post-V605 question: after quarantining the failed V573/V579/V591/V596
routes, is there still clean, target-family source material that has not already
been consumed by those routes?

The gate validates each V446 accepted row against the source competition_train
answer using the same label-free extraction path used for weak promotion:

assistant raw text -> extract_final_answer -> verify_answer

Rows that only looked aligned by status/overlap but do not extract to the known
source answer are treated as dirty and cannot seed a new dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_boxed_answers, extract_final_answer, verify_answer  # noqa: E402


DEFAULT_V446_AUDIT_CSV = (
    ROOT
    / "artifacts/v446_tong_source_target_alignment_gate/20260515T_v446_cpu_gate/"
    / "v446_tong_source_target_alignment_gate_candidate_audit.csv"
)
DEFAULT_SFT_JSONL = Path(r"C:\Users\davis\Downloads\sft_reconstructed.jsonl")
DEFAULT_COMPETITION_TRAIN_CSV = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v606_unused_v446_source_pool_audit"

DEFAULT_USED_DATASETS = [
    ROOT / "artifacts/v523_targeted_source_trace_pack/20260516T235821Z/v523_targeted_source_trace_pack_train.jsonl",
    ROOT / "artifacts/v523_targeted_source_trace_pack/20260516T235821Z/v523_targeted_source_trace_pack_val.jsonl",
    ROOT / "artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v551_short_bit_trace_pack_train.jsonl",
    ROOT / "artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate/v551_short_bit_trace_pack_val.jsonl",
    ROOT / "artifacts/v571_bitpair_source_only_trace_pack/20260517T_v571_cpu_gate/v571_bitpair_source_only_trace_pack_train.jsonl",
    ROOT / "artifacts/v571_bitpair_source_only_trace_pack/20260517T_v571_cpu_gate/v571_bitpair_source_only_trace_pack_val.jsonl",
    ROOT
    / "artifacts/v573_v571_bitpair_v551_equation_reference_mix/20260517T_v573_cpu_gate/"
    / "v572_v571_bitpair_v551_equation_mix_train.jsonl",
    ROOT
    / "artifacts/v573_v571_bitpair_v551_equation_reference_mix/20260517T_v573_cpu_gate/"
    / "v572_v571_bitpair_v551_equation_mix_val.jsonl",
    ROOT / "artifacts/v579_v571_bitpair_v551_equation_strictedge_mix/20260517T_v579_cpu_gate/v572_v571_bitpair_v551_equation_mix_train.jsonl",
    ROOT / "artifacts/v579_v571_bitpair_v551_equation_strictedge_mix/20260517T_v579_cpu_gate/v572_v571_bitpair_v551_equation_mix_val.jsonl",
    ROOT / "artifacts/v591_v579_symbolic_queryop_source_mix/20260518T_v591_cpu_gate/v591_v579_symbolic_queryop_source_mix_train.jsonl",
    ROOT / "artifacts/v591_v579_symbolic_queryop_source_mix/20260518T_v591_cpu_gate/v591_v579_symbolic_queryop_source_mix_val.jsonl",
    ROOT / "artifacts/v596_queryop_answer_only_preference_dataset/20260518T_v596_cpu_gate/v596_queryop_answer_only_preference_train.jsonl",
    ROOT / "artifacts/v596_queryop_answer_only_preference_dataset/20260518T_v596_cpu_gate/v596_queryop_answer_only_preference_val.jsonl",
]

TARGET_FAMILIES = {"bit_manipulation", "equation_transform"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid jsonl: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def row_family(row: dict[str, Any]) -> str:
    meta = row.get("_metadata") if isinstance(row.get("_metadata"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("family") or row.get("task_type") or meta.get("category") or metadata.get("family") or "")


def source_id(row: dict[str, Any]) -> str:
    meta = row.get("_metadata") if isinstance(row.get("_metadata"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("id") or meta.get("problem_id") or metadata.get("problem_id") or metadata.get("original_id") or "")


def accepted_v446_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    out = []
    for row in rows:
        if str(row.get("accepted", "")).strip().lower() != "true":
            continue
        if str(row.get("family", "")).strip() not in TARGET_FAMILIES:
            continue
        out.append(row)
    return out


def competition_answer_index(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {str(row.get("id", "")): row for row in rows if str(row.get("id", ""))}


def used_prompt_assistant_hashes(paths: list[Path]) -> dict[str, Any]:
    prompt_hashes: set[str] = set()
    assistant_hashes: set[str] = set()
    prompt_answer_hashes: set[str] = set()
    existing_paths: list[str] = []
    missing_paths: list[str] = []
    rows_seen = 0
    for path in paths:
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        existing_paths.append(str(path))
        for row in read_jsonl(path):
            prompt = str(row.get("prompt") or message_content(row, "user"))
            assistant = message_content(row, "assistant")
            answer = str(row.get("answer") or (row.get("metadata") or {}).get("answer") or "")
            prompt_hashes.add(sha256_text(prompt))
            assistant_hashes.add(sha256_text(assistant))
            prompt_answer_hashes.add(sha256_text(prompt + "\n" + answer))
            rows_seen += 1
    return {
        "prompt_hashes": prompt_hashes,
        "assistant_hashes": assistant_hashes,
        "prompt_answer_hashes": prompt_answer_hashes,
        "existing_paths": existing_paths,
        "missing_paths": missing_paths,
        "rows_seen": rows_seen,
    }


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def audit(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[str] = []
    for label, path in (
        ("v446_audit_csv", args.v446_audit_csv),
        ("sft_jsonl", args.sft_jsonl),
        ("competition_train_csv", args.competition_train_csv),
    ):
        if not path.is_file():
            findings.append(f"missing_{label}:{path}")

    if findings:
        report = {
            "schema_version": "kg1_v606_unused_v446_source_pool_audit_v1",
            "generated_at_utc": utc_now(),
            "decision": "blocked",
            "findings": findings,
        }
        write_json(args.output_dir / "v606_unused_v446_source_pool_manifest.json", report)
        return report

    accepted = accepted_v446_rows(args.v446_audit_csv)
    sft_rows = read_jsonl(args.sft_jsonl)
    answer_index = competition_answer_index(args.competition_train_csv)
    used = used_prompt_assistant_hashes(args.used_dataset)

    clean_rows: list[dict[str, Any]] = []
    dirty_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    clean_family_counts: Counter[str] = Counter()
    dirty_reason_counts: Counter[str] = Counter()
    box_counts: Counter[int] = Counter()
    assistant_chars: list[int] = []
    clean_assistant_chars: list[int] = []

    for audit_row in accepted:
        row_no = int(str(audit_row["row_no"]))
        source = sft_rows[row_no - 1] if 0 < row_no <= len(sft_rows) else {}
        rid = str(audit_row.get("id") or source_id(source))
        family = str(audit_row.get("family") or row_family(source))
        family_counts[family] += 1
        source_answer = str(answer_index.get(rid, {}).get("answer", ""))
        prompt = message_content(source, "user")
        assistant = message_content(source, "assistant")
        extracted = extract_final_answer(assistant)
        boxed = extract_boxed_answers(assistant)
        prompt_sha = sha256_text(prompt)
        assistant_sha = sha256_text(assistant)
        prompt_answer_sha = sha256_text(prompt + "\n" + source_answer)
        verified = bool(source_answer and verify_answer(source_answer, extracted))
        chars = len(assistant)
        assistant_chars.append(chars)
        box_counts[len(boxed)] += 1
        exact_overlap = (
            prompt_sha in used["prompt_hashes"]
            or assistant_sha in used["assistant_hashes"]
            or prompt_answer_sha in used["prompt_answer_hashes"]
        )
        row_out = {
            "row_no": row_no,
            "id": rid,
            "family": family,
            "status": audit_row.get("status", ""),
            "answer": source_answer,
            "extracted": extracted,
            "verified": verified,
            "boxed_count": len(boxed),
            "assistant_chars": chars,
            "prompt_sha256": prompt_sha,
            "assistant_sha256": assistant_sha,
            "prompt_answer_sha256": prompt_answer_sha,
            "exact_overlap_with_quarantined_datasets": exact_overlap,
        }
        if not source:
            row_out["dirty_reason"] = "source_row_missing"
            dirty_reason_counts["source_row_missing"] += 1
            dirty_rows.append(row_out)
        elif not source_answer:
            row_out["dirty_reason"] = "competition_answer_missing"
            dirty_reason_counts["competition_answer_missing"] += 1
            dirty_rows.append(row_out)
        elif not verified:
            row_out["dirty_reason"] = "extracted_answer_mismatch"
            dirty_reason_counts["extracted_answer_mismatch"] += 1
            dirty_rows.append(row_out)
        elif exact_overlap:
            overlap_rows.append(row_out)
        else:
            clean_family_counts[family] += 1
            clean_assistant_chars.append(chars)
            clean_rows.append(row_out)

    # Directly training the raw accepted traces is blocked when the source pool is
    # mostly long multi-boxed CoT. The next usable action is to build compact,
    # answer-verified source-only traces from this pool and then re-run V286/V513.
    long_trace_rows = sum(1 for row in clean_rows if int(row["assistant_chars"]) > args.max_direct_assistant_chars)
    multi_box_rows = sum(1 for row in clean_rows if int(row["boxed_count"]) > args.max_direct_boxed_count)
    direct_training_allowed = not long_trace_rows and not multi_box_rows
    dataset_build_allowed = bool(clean_rows) and clean_family_counts.get("bit_manipulation", 0) >= args.min_clean_bit and clean_family_counts.get("equation_transform", 0) >= args.min_clean_equation
    gpu_allowed = False
    decision = "compact_source_pool_ready" if dataset_build_allowed else "blocked_no_clean_source_pool"
    if direct_training_allowed and dataset_build_allowed:
        decision = "source_pool_ready_but_cpu_gates_required"

    selected_preview = clean_rows[: args.preview_rows]
    write_csv(
        args.output_dir / "v606_clean_unused_v446_rows_preview.csv",
        selected_preview,
        [
            "row_no",
            "id",
            "family",
            "answer",
            "extracted",
            "boxed_count",
            "assistant_chars",
            "prompt_sha256",
            "assistant_sha256",
            "prompt_answer_sha256",
        ],
    )
    write_csv(
        args.output_dir / "v606_dirty_v446_rows.csv",
        dirty_rows[: max(args.preview_rows, 100)],
        [
            "row_no",
            "id",
            "family",
            "answer",
            "extracted",
            "dirty_reason",
            "boxed_count",
            "assistant_chars",
        ],
    )

    summary = {
        "schema_version": "kg1_v606_unused_v446_source_pool_audit_v1",
        "version": "V606",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v446_audit_csv": str(args.v446_audit_csv),
            "v446_audit_sha256": sha256_file(args.v446_audit_csv),
            "sft_jsonl": str(args.sft_jsonl),
            "sft_jsonl_sha256": sha256_file(args.sft_jsonl),
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
            "used_dataset_existing_paths": used["existing_paths"],
            "used_dataset_missing_paths": used["missing_paths"],
            "used_dataset_rows_seen": used["rows_seen"],
        },
        "accepted_v446_rows": len(accepted),
        "accepted_family_counts": dict(sorted(family_counts.items())),
        "clean_unused_rows": len(clean_rows),
        "clean_unused_family_counts": dict(sorted(clean_family_counts.items())),
        "dirty_rows": len(dirty_rows),
        "dirty_reason_counts": dict(sorted(dirty_reason_counts.items())),
        "overlap_with_quarantined_dataset_rows": len(overlap_rows),
        "assistant_chars": {
            "accepted_p50": percentile(assistant_chars, 0.50),
            "accepted_p95": percentile(assistant_chars, 0.95),
            "clean_p50": percentile(clean_assistant_chars, 0.50),
            "clean_p95": percentile(clean_assistant_chars, 0.95),
            "clean_max": max(clean_assistant_chars) if clean_assistant_chars else 0,
        },
        "boxed_count_distribution": dict(sorted((str(key), value) for key, value in box_counts.items())),
        "direct_full_trace_training_allowed": direct_training_allowed,
        "direct_full_trace_blockers": [
            *(["long_trace_rows_gt_limit:" + str(long_trace_rows)] if long_trace_rows else []),
            *(["multi_box_rows_gt_limit:" + str(multi_box_rows)] if multi_box_rows else []),
        ],
        "dataset_build_allowed": dataset_build_allowed,
        "gpu_allowed": gpu_allowed,
        "decision": decision,
        "next_action": (
            "Build a compact V607 source-only dataset from clean_unused_rows with one boxed final answer, "
            "top-level canonical family/subcategory, loss_weight calibrated by example_mean, and no raw "
            "long CoT copied verbatim; then run V509/V286/V513/V524/V575 before any GPU."
            if dataset_build_allowed
            else "Do not build/train from V446 accepted rows; clean source pool is insufficient."
        ),
        "outputs": {
            "manifest_json": str(args.output_dir / "v606_unused_v446_source_pool_manifest.json"),
            "clean_preview_csv": str(args.output_dir / "v606_clean_unused_v446_rows_preview.csv"),
            "dirty_preview_csv": str(args.output_dir / "v606_dirty_v446_rows.csv"),
            "report_md": str(args.output_dir / "KG1_V606_UNUSED_V446_SOURCE_POOL_AUDIT.md"),
        },
    }
    write_json(args.output_dir / "v606_unused_v446_source_pool_manifest.json", summary)
    write_markdown(args.output_dir / "KG1_V606_UNUSED_V446_SOURCE_POOL_AUDIT.md", summary)
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# KG1 V606 Unused V446 Source Pool Audit",
        "",
        f"Decision: `{summary['decision']}`",
        f"GPU allowed: `{summary['gpu_allowed']}`",
        f"Dataset build allowed: `{summary['dataset_build_allowed']}`",
        "",
        "## Counts",
        "",
        f"- accepted V446 target rows: `{summary['accepted_v446_rows']}`",
        f"- accepted families: `{json.dumps(summary['accepted_family_counts'], sort_keys=True)}`",
        f"- clean unused rows: `{summary['clean_unused_rows']}`",
        f"- clean unused families: `{json.dumps(summary['clean_unused_family_counts'], sort_keys=True)}`",
        f"- dirty rows: `{summary['dirty_rows']}`",
        f"- dirty reasons: `{json.dumps(summary['dirty_reason_counts'], sort_keys=True)}`",
        f"- overlap with quarantined datasets: `{summary['overlap_with_quarantined_dataset_rows']}`",
        "",
        "## Trace Shape",
        "",
        f"- clean assistant chars p50: `{summary['assistant_chars']['clean_p50']}`",
        f"- clean assistant chars p95: `{summary['assistant_chars']['clean_p95']}`",
        f"- clean assistant chars max: `{summary['assistant_chars']['clean_max']}`",
        f"- boxed-count distribution: `{json.dumps(summary['boxed_count_distribution'], sort_keys=True)}`",
        f"- direct full-trace training allowed: `{summary['direct_full_trace_training_allowed']}`",
        f"- direct full-trace blockers: `{json.dumps(summary['direct_full_trace_blockers'])}`",
        "",
        "## Next Action",
        "",
        summary["next_action"],
        "",
        "## Rule",
        "",
        (
            "Rows in this audit are still source-only candidates. They do not authorize H200. "
            "Any V607 dataset must be compacted, revalidated, and pass the existing anti-backfire "
            "and no-false-gain gates before a paid run."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v446-audit-csv", type=Path, default=DEFAULT_V446_AUDIT_CSV)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_COMPETITION_TRAIN_CSV)
    parser.add_argument("--used-dataset", type=Path, action="append", default=list(DEFAULT_USED_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-clean-bit", type=int, default=128)
    parser.add_argument("--min-clean-equation", type=int, default=128)
    parser.add_argument("--max-direct-assistant-chars", type=int, default=2500)
    parser.add_argument("--max-direct-boxed-count", type=int, default=1)
    parser.add_argument("--preview-rows", type=int, default=200)
    args = parser.parse_args()

    print("=== V606 UNUSED V446 SOURCE POOL AUDIT START ===", flush=True)
    print(f"v446_audit_csv = {args.v446_audit_csv}", flush=True)
    print(f"sft_jsonl = {args.sft_jsonl}", flush=True)
    print(f"competition_train_csv = {args.competition_train_csv}", flush=True)
    print(f"output_dir = {args.output_dir}", flush=True)
    summary = audit(args)
    print("decision =", summary.get("decision"), flush=True)
    print("dataset_build_allowed =", summary.get("dataset_build_allowed"), flush=True)
    print("gpu_allowed =", summary.get("gpu_allowed"), flush=True)
    print("clean_unused_family_counts =", json.dumps(summary.get("clean_unused_family_counts", {}), sort_keys=True), flush=True)
    print("=== V606 UNUSED V446 SOURCE POOL AUDIT END ===", flush=True)
    return 0 if summary.get("decision") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())

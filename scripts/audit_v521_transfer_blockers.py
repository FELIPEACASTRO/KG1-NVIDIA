#!/usr/bin/env python3
"""V521 CPU audit for KG1 teacher-to-adapter transfer blockers.

This audit exists because V517/V518 proved that lower eval_loss can still
backfire on submit-safe ACC: one equation row improved while a protected bit
row regressed. V521 does not launch training. It summarizes the active dataset
lineage, checks the known backfire rows are not being used as labels, and emits
a fail-closed decision before any paid GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v521_transfer_blocker_audit"
DEFAULT_V519_CHANGED_ROWS = ROOT / "artifacts/v519_v518_backfire_row_audit/v519_v518_backfire_changed_rows.csv"
DEFAULT_V520_MANIFEST = ROOT / "artifacts/v520_local_candidate_mining/v520_local_candidate_mining_manifest.json"

TEXT_TRACE_MARKERS = (
    "verify",
    "rule",
    "example",
    "output bit",
    "bitsum",
    "stride",
    "rot",
    "shl",
    "shr",
    "xor",
    "and",
    "or",
    "not",
    "majority",
    "choice",
    "fullbyte",
)

BIT_BINARY_RE = re.compile(r"^[01]{8}$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_jsonl: Path
    val_jsonl: Path
    manifest_json: Path | None
    role: str
    known_status: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def final_assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    for key in ("completion", "target", "response", "assistant"):
        if key in row:
            return str(row.get(key, ""))
    return ""


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def row_family(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(row.get("family") or metadata.get("family") or "")


def row_source(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(row.get("source") or row.get("source_dataset") or metadata.get("source") or metadata.get("source_dataset") or "")


def row_subcategory(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    return str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("subtype") or "")


def row_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return ""


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row_metadata(row).get("answer") or "")


def metadata_flag(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    metadata = row_metadata(row)
    return metadata.get(key)


def is_trace_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TEXT_TRACE_MARKERS)


def summarize_rows(rows: list[dict[str, Any]], *, split: str, backfire_prompt_hashes: set[str]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    duplicate_ids: Counter[str] = Counter()
    prompt_answer_hashes: Counter[str] = Counter()
    weak_or_full_flags: Counter[str] = Counter()
    bit_rows = 0
    bit_binary_answers = 0
    bit_trace_rows = 0
    bit_answer_only_rows = 0
    equation_rows = 0
    equation_trace_rows = 0
    boxed_rows = 0
    changed_prompt_overlaps = 0
    assistant_word_counts: list[int] = []

    for row in rows:
        family = row_family(row)
        family_counts[family] += 1
        source_counts[row_source(row)] += 1
        subcategory_counts[row_subcategory(row)] += 1
        duplicate_ids[str(row.get("id", ""))] += 1
        prompt = row_prompt(row)
        answer = row_answer(row)
        prompt_answer_hashes[sha256_text(prompt + "\n---ANSWER---\n" + answer)] += 1
        assistant = final_assistant_text(row)
        assistant_word_counts.append(len(assistant.split()))
        if "\\boxed{" in assistant:
            boxed_rows += 1
        if sha256_text(prompt) in backfire_prompt_hashes:
            changed_prompt_overlaps += 1
        for flag in (
            "weak_gate_rows_used_for_training",
            "full_gate_rows_used_for_training",
            "weak_or_full_gate_rows_used_for_training",
        ):
            value = metadata_flag(row, flag)
            if value not in (None, False):
                weak_or_full_flags[flag] += 1
        if family == "bit_manipulation":
            bit_rows += 1
            if BIT_BINARY_RE.match(answer):
                bit_binary_answers += 1
            if is_trace_text(assistant):
                bit_trace_rows += 1
            if len(assistant.split()) <= 6:
                bit_answer_only_rows += 1
        if family == "equation_transform":
            equation_rows += 1
            if is_trace_text(assistant):
                equation_trace_rows += 1

    duplicate_id_count = sum(1 for value in duplicate_ids.values() if value > 1)
    duplicate_prompt_answer_count = sum(1 for value in prompt_answer_hashes.values() if value > 1)
    word_counts_sorted = sorted(assistant_word_counts)
    p50 = word_counts_sorted[len(word_counts_sorted) // 2] if word_counts_sorted else 0
    return {
        "split": split,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(source_counts.most_common(12)),
        "subcategory_counts": dict(subcategory_counts.most_common(12)),
        "duplicate_id_count": duplicate_id_count,
        "duplicate_prompt_answer_count": duplicate_prompt_answer_count,
        "weak_or_full_training_flags": dict(sorted(weak_or_full_flags.items())),
        "changed_prompt_overlaps": changed_prompt_overlaps,
        "assistant_word_p50": p50,
        "assistant_word_min": min(assistant_word_counts) if assistant_word_counts else 0,
        "assistant_word_max": max(assistant_word_counts) if assistant_word_counts else 0,
        "boxed_rows": boxed_rows,
        "bit_rows": bit_rows,
        "bit_binary_answers": bit_binary_answers,
        "bit_trace_rows": bit_trace_rows,
        "bit_answer_only_rows": bit_answer_only_rows,
        "equation_rows": equation_rows,
        "equation_trace_rows": equation_trace_rows,
    }


def load_backfire_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    prompt_hashes: set[str] = set()
    if not path.is_file():
        return rows, prompt_hashes
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            prompt_sha256 = row.get("prompt_sha256") or ""
            if prompt_sha256:
                prompt_hashes.add(prompt_sha256)
            prompt = row.get("prompt") or ""
            if prompt:
                prompt_hashes.add(sha256_text(prompt))
    return rows, prompt_hashes


def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="v390_equation_no_loss_distill",
            train_jsonl=ROOT / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/v390_v325_equation_no_loss_distill_sft_train.jsonl",
            val_jsonl=ROOT / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/v390_v325_equation_no_loss_distill_sft_val.jsonl",
            manifest_json=ROOT / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/v390_v325_equation_no_loss_distill_manifest.json",
            role="equation_only_cpu_signal",
            known_status="blocked_direct_gpu_without_bit_replay",
        ),
        DatasetSpec(
            name="v475_equation_bit_replay_mix",
            train_jsonl=ROOT / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_train.jsonl",
            val_jsonl=ROOT / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_val.jsonl",
            manifest_json=ROOT / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/v475_equation_bit_replay_mix_manifest.json",
            role="equation_signal_plus_bit_replay",
            known_status="failed_transfer_in_v495_v496",
        ),
        DatasetSpec(
            name="v510_canonical_active_training_pool",
            train_jsonl=ROOT / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool/v510_canonical_active_training_pool_train.jsonl",
            val_jsonl=ROOT / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool/v510_canonical_active_training_pool_val.jsonl",
            manifest_json=ROOT / "artifacts/v510_canonical_training_dataset/v510_canonical_active_training_pool/v510_canonical_active_training_pool_manifest.json",
            role="canonical_merged_pool",
            known_status="failed_learnability_in_v513_as_bit_answer_only",
        ),
        DatasetSpec(
            name="v515_v514_fullbyte_residual",
            train_jsonl=ROOT / "artifacts/v515_v514_fullbyte_residual_dataset/v515_v514_fullbyte_residual_train.jsonl",
            val_jsonl=ROOT / "artifacts/v515_v514_fullbyte_residual_dataset/v515_v514_fullbyte_residual_val.jsonl",
            manifest_json=ROOT / "artifacts/v515_v514_fullbyte_residual_dataset/v515_v514_fullbyte_residual_manifest.json",
            role="traceable_bit_plus_equation_pool",
            known_status="failed_transfer_in_v517_v518",
        ),
        DatasetSpec(
            name="v304_solver_trace_distill",
            train_jsonl=ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_train.jsonl",
            val_jsonl=ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_val.jsonl",
            manifest_json=ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_manifest.json",
            role="historical_broad_solver_trace",
            known_status="historical_reference_only_not_current_active_pool",
        ),
    ]


def dataset_finding(spec: DatasetSpec, train_summary: dict[str, Any], val_summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if train_summary["changed_prompt_overlaps"] or val_summary["changed_prompt_overlaps"]:
        findings.append("blocks: exact prompt overlap with V519 changed weak rows")
    if train_summary["weak_or_full_training_flags"] or val_summary["weak_or_full_training_flags"]:
        findings.append("blocks: weak/full gate row flag present in training data")
    if spec.name == "v390_equation_no_loss_distill":
        findings.append("blocks direct GPU: equation-only dataset has no bit guardrail rows")
    if spec.name == "v475_equation_bit_replay_mix":
        findings.append("already tested: V495/V496 gained equation but lost bit and truncation")
    if spec.name == "v510_canonical_active_training_pool":
        bit_trace_ratio = train_summary["bit_trace_rows"] / max(train_summary["bit_rows"], 1)
        if bit_trace_ratio < 0.8:
            findings.append("blocks as-is: bit trace ratio below 80 percent")
        findings.append("already tested: V511/V513 showed no transferable bit trace signal as built")
    if spec.name == "v515_v514_fullbyte_residual":
        findings.append("already tested: V517/V518 lower loss still lost protected bit row")
        train_bit = train_summary["family_counts"].get("bit_manipulation", 0)
        train_rows = max(train_summary["rows"], 1)
        if train_bit / train_rows < 0.25:
            findings.append("risk: unweighted bit share below 25 percent")
    if train_summary["duplicate_id_count"] or val_summary["duplicate_id_count"]:
        findings.append("warning: duplicate ids present")
    if train_summary["duplicate_prompt_answer_count"] or val_summary["duplicate_prompt_answer_count"]:
        findings.append("warning: duplicate prompt-answer rows present")
    if not findings:
        findings.append("no structural blocker found by V521")
    return findings


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backfire_rows, backfire_prompt_hashes = load_backfire_rows(args.v519_changed_rows_csv)
    v520_manifest = read_json(args.v520_manifest_json) if args.v520_manifest_json.is_file() else {}

    dataset_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for spec in default_dataset_specs():
        missing = [str(path) for path in (spec.train_jsonl, spec.val_jsonl) if not path.is_file()]
        if missing:
            blockers.append(f"{spec.name}:missing_dataset_file")
            dataset_rows.append(
                {
                    "dataset": spec.name,
                    "split": "missing",
                    "rows": 0,
                    "family_counts": "{}",
                    "bit_rows": 0,
                    "bit_trace_rows": 0,
                    "equation_rows": 0,
                    "equation_trace_rows": 0,
                    "changed_prompt_overlaps": 0,
                    "weak_or_full_training_flags": "{}",
                    "finding": "missing files: " + "; ".join(missing),
                }
            )
            continue
        train_rows = read_jsonl(spec.train_jsonl)
        val_rows = read_jsonl(spec.val_jsonl)
        train_summary = summarize_rows(train_rows, split="train", backfire_prompt_hashes=backfire_prompt_hashes)
        val_summary = summarize_rows(val_rows, split="validation", backfire_prompt_hashes=backfire_prompt_hashes)
        findings = dataset_finding(spec, train_summary, val_summary)
        if any(item.startswith("blocks") or item.startswith("already tested") for item in findings):
            blockers.append(spec.name + ":" + "|".join(findings))
        for summary in (train_summary, val_summary):
            dataset_rows.append(
                {
                    "dataset": spec.name,
                    "split": summary["split"],
                    "role": spec.role,
                    "known_status": spec.known_status,
                    "rows": summary["rows"],
                    "sha256": sha256_file(spec.train_jsonl if summary["split"] == "train" else spec.val_jsonl),
                    "family_counts": json.dumps(summary["family_counts"], sort_keys=True),
                    "source_counts": json.dumps(summary["source_counts"], sort_keys=True),
                    "subcategory_counts": json.dumps(summary["subcategory_counts"], sort_keys=True),
                    "bit_rows": summary["bit_rows"],
                    "bit_binary_answers": summary["bit_binary_answers"],
                    "bit_trace_rows": summary["bit_trace_rows"],
                    "bit_answer_only_rows": summary["bit_answer_only_rows"],
                    "equation_rows": summary["equation_rows"],
                    "equation_trace_rows": summary["equation_trace_rows"],
                    "assistant_word_p50": summary["assistant_word_p50"],
                    "duplicate_id_count": summary["duplicate_id_count"],
                    "duplicate_prompt_answer_count": summary["duplicate_prompt_answer_count"],
                    "changed_prompt_overlaps": summary["changed_prompt_overlaps"],
                    "weak_or_full_training_flags": json.dumps(summary["weak_or_full_training_flags"], sort_keys=True),
                    "finding": "; ".join(findings),
                }
            )

    v520_submit_safe = int(v520_manifest.get("submit_safe_adapter_candidates_above_baseline", -1))
    if v520_submit_safe != 0:
        blockers.append("v520_manifest_unexpected_submit_safe_count")

    decision = {
        "gpu_allowed": False,
        "status": "blocked_until_new_cpu_transfer_signal",
        "reason": (
            "V518 showed loss/ACC divergence and V520 found zero submit-safe adapter candidates above baseline. "
            "The active datasets are either already failed as-is or have insufficient new bit transfer signal."
        ),
        "next_action": (
            "Build V522 CPU source-target alignment/learnability audit: mine only train/public solver traces, "
            "prove new coverage over the protected bit backfire class and at least one equation rule class, then "
            "permit GPU only if the no-GPU gate predicts a real label-free gain with bit>=136, trunc=0, and "
            "8740ed31 preserved."
        ),
    }

    summary_csv = output_dir / "v521_transfer_blocker_dataset_summary.csv"
    manifest_path = output_dir / "v521_transfer_blocker_audit_manifest.json"
    report_md = output_dir / "KG1_V521_TRANSFER_BLOCKER_AUDIT.md"
    fieldnames = [
        "dataset",
        "split",
        "role",
        "known_status",
        "rows",
        "sha256",
        "family_counts",
        "source_counts",
        "subcategory_counts",
        "bit_rows",
        "bit_binary_answers",
        "bit_trace_rows",
        "bit_answer_only_rows",
        "equation_rows",
        "equation_trace_rows",
        "assistant_word_p50",
        "duplicate_id_count",
        "duplicate_prompt_answer_count",
        "changed_prompt_overlaps",
        "weak_or_full_training_flags",
        "finding",
    ]
    write_csv(summary_csv, dataset_rows, fieldnames)
    manifest = {
        "version": "V521",
        "schema_version": "kg1_v521_transfer_blocker_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "blockers": blockers,
        "backfire_rows": [
            {
                "id": row.get("id", ""),
                "family": row.get("family", ""),
                "delta": row.get("delta", ""),
                "delta_correct": row.get("delta_correct", ""),
                "answer": row.get("answer", ""),
                "baseline": row.get("baseline_prediction", ""),
                "candidate": row.get("candidate_prediction", "") or row.get("v518_prediction", ""),
            }
            for row in backfire_rows
        ],
        "v520_summary": {
            "submit_safe_adapter_candidates_above_baseline": v520_submit_safe,
            "baseline_label_free": v520_manifest.get("baseline_label_free", {}),
            "decision": v520_manifest.get("decision", ""),
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "manifest_json": str(manifest_path),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_md, manifest, dataset_rows)
    return manifest


def write_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V521 Transfer Blocker Audit",
        "",
        "## Decision",
        "",
        f"- GPU allowed: `{manifest['decision']['gpu_allowed']}`",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Reason: {manifest['decision']['reason']}",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## Why this matters",
        "",
        "- V517 reduced loss, but V518 did not improve submit-safe ACC.",
        "- V518 gained one equation row and lost the protected bit row `8740ed31=01101000`.",
        "- V520 found zero local adapter-only CSVs above the label-free baseline without backfire.",
        "- Therefore, another paid job is blocked until a CPU-only transfer gate proves new signal.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Split | Rows | Family counts | Bit traces | Equation traces | Finding |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {split} | {rows} | `{family_counts}` | {bit_trace_rows}/{bit_rows} | "
            "{equation_trace_rows}/{equation_rows} | {finding} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Operational Rule",
            "",
            "Do not run H200/A100/HF GPU from these datasets as-is. A new job needs a V522-style CPU gate that proves:",
            "",
            "1. no exact prompt overlap with weak/full rows;",
            "2. no weak/full training flags;",
            "3. protected row `8740ed31` remains correct in weak eval;",
            "4. label-free total improves beyond baseline;",
            "5. `bit_manipulation>=136`, `equation_transform>55`, and `truncated=0`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    sample = {
        "id": "x",
        "family": "bit_manipulation",
        "prompt": "p",
        "answer": "01010101",
        "messages": [{"role": "assistant", "content": "Check bits with stride. Final answer: 01010101"}],
        "metadata": {"weak_gate_rows_used_for_training": False},
    }
    summary = summarize_rows([sample], split="train", backfire_prompt_hashes={sha256_text("other")})
    if summary["bit_rows"] != 1 or summary["bit_trace_rows"] != 1 or summary["weak_or_full_training_flags"]:
        raise SystemExit("self-test failed")
    print("audit_v521_transfer_blockers_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v519-changed-rows-csv", type=Path, default=DEFAULT_V519_CHANGED_ROWS)
    parser.add_argument("--v520-manifest-json", type=Path, default=DEFAULT_V520_MANIFEST)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    manifest = audit(args)
    print("v521_manifest =", manifest["outputs"]["manifest_json"], flush=True)
    print("v521_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

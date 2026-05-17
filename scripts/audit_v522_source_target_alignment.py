#!/usr/bin/env python3
"""V522 CPU source-target alignment audit.

V521 blocked paid GPU because active datasets had already failed as-is. V522
answers the next question: which solver/postprocessor rule classes actually
create no-loss weak gains, and whether permitted train/public sources contain
enough analogous traces to build a new source-only dataset without using weak
labels as training rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v522_source_target_alignment_audit"
DEFAULT_BASELINE_CSV = ROOT / "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv"
DEFAULT_TEACHER_CSV = ROOT / "artifacts/v380_solver_results_patch_gate/20260514T_cpu_gate/v380_reexecuted_teacher_predictions.csv"
DEFAULT_V357_RULES_CSV = ROOT / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_candidate_rules.csv"
DEFAULT_V366_RULES_CSV = ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_candidate_rules.csv"
DEFAULT_V516_EQUATION_ACCEPTED_CSV = ROOT / "artifacts/v516_equation_label_free_solver_gate/v324_equation_expanded_solver_accepted_candidates.csv"

from src.competition_utils import verify_answer  # noqa: E402


@dataclass(frozen=True)
class SourceSpec:
    name: str
    train_jsonl: Path
    val_jsonl: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if text:
                rows.append(json.loads(text))
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


def row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("prompt", "answer", "source", "source_dataset", "subcategory"):
        value = row.get(key)
        if value is not None:
            parts.append(str(value))
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(str(value) for value in metadata.values() if value is not None)
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def default_sources() -> list[SourceSpec]:
    return [
        SourceSpec(
            name="v304_solver_trace_distill",
            train_jsonl=ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_train.jsonl",
            val_jsonl=ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_val.jsonl",
        ),
        SourceSpec(
            name="v515_v514_fullbyte_residual",
            train_jsonl=ROOT / "artifacts/v515_v514_fullbyte_residual_dataset/v515_v514_fullbyte_residual_train.jsonl",
            val_jsonl=ROOT / "artifacts/v515_v514_fullbyte_residual_dataset/v515_v514_fullbyte_residual_val.jsonl",
        ),
    ]


def gain_rule(row: dict[str, str]) -> str:
    for key in ("v366_source_rule", "v357_source_rule", "v350_source_rule"):
        value = row.get(key, "").strip()
        if value:
            return value
    if row.get("family") == "equation_transform":
        return "equation_reference_gain_untyped"
    return "unknown_reference_gain"


def summarize_reference_signal(baseline_rows: list[dict[str, str]], teacher_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    gains: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()

    for teacher in teacher_rows:
        row_id = teacher.get("id", "")
        baseline = baseline_by_id.get(row_id)
        if not baseline:
            continue
        answer = teacher.get("answer", "")
        baseline_prediction = baseline.get("prediction", "")
        teacher_prediction = teacher.get("prediction", "")
        baseline_correct = verify_answer(answer, baseline_prediction)
        teacher_correct = verify_answer(answer, teacher_prediction)
        family = teacher.get("family", "")
        rule = gain_rule(teacher)
        if not baseline_correct and teacher_correct:
            gains.append(
                {
                    "id": row_id,
                    "family": family,
                    "answer": answer,
                    "baseline_prediction": baseline_prediction,
                    "teacher_prediction": teacher_prediction,
                    "rule": rule,
                    "prompt_sha256": teacher.get("prompt_sha256", ""),
                }
            )
            family_counts[family] += 1
            rule_counts[f"{family}:{rule}"] += 1
        elif baseline_correct and not teacher_correct:
            losses.append(
                {
                    "id": row_id,
                    "family": family,
                    "answer": answer,
                    "baseline_prediction": baseline_prediction,
                    "teacher_prediction": teacher_prediction,
                    "rule": rule,
                }
            )
    summary = {
        "gain_total": len(gains),
        "loss_total": len(losses),
        "gain_family_counts": dict(sorted(family_counts.items())),
        "gain_rule_counts": dict(rule_counts.most_common()),
    }
    return gains, summary


def accepted_rule_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows = read_csv(path)
    return [row for row in rows if str(row.get("accepted") or row.get("promotable_after_class_gate", "")).lower() in {"true", "1"}]


def source_marker_coverage(sources: list[SourceSpec]) -> list[dict[str, Any]]:
    markers = {
        "CHO": re.compile(r"\bCHO\("),
        "MAJ3": re.compile(r"\bMAJ3\("),
        "PAR3": re.compile(r"\bPAR3\("),
        "XOR": re.compile(r"\bXOR\b|\^"),
        "OR": re.compile(r"\bOR\b|\|"),
        "fullbyte_safe_ternary": re.compile(r"fullbyte_safe_ternary|full-byte expression|fullbyte"),
        "bit_v300_gain_pattern": re.compile(r"bit_fullbyte_v300_gain_pattern|v300_gain_pattern"),
    }
    out: list[dict[str, Any]] = []
    for source in sources:
        for split, path in (("train", source.train_jsonl), ("validation", source.val_jsonl)):
            rows = read_jsonl(path)
            counts: Counter[str] = Counter()
            family_counts: Counter[str] = Counter()
            for row in rows:
                text = row_text(row)
                family = str(row.get("family") or (row.get("metadata") or {}).get("family") or "")
                family_counts[family] += 1
                for marker, pattern in markers.items():
                    if pattern.search(text):
                        counts[marker] += 1
            out.append(
                {
                    "source": source.name,
                    "split": split,
                    "rows": len(rows),
                    "family_counts": dict(sorted(family_counts.items())),
                    **{f"marker_{marker}": counts.get(marker, 0) for marker in markers},
                }
            )
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_csv(args.baseline_csv)
    teacher_rows = read_csv(args.teacher_csv)
    gains, signal_summary = summarize_reference_signal(baseline_rows, teacher_rows)
    v357_accepted = accepted_rule_rows(args.v357_rules_csv)
    v366_accepted = accepted_rule_rows(args.v366_rules_csv)
    v516_equation = read_csv(args.v516_equation_accepted_csv) if args.v516_equation_accepted_csv.is_file() else []
    coverage = source_marker_coverage(default_sources())

    gain_csv = output_dir / "v522_reference_no_loss_gains.csv"
    coverage_csv = output_dir / "v522_source_marker_coverage.csv"
    manifest_path = output_dir / "v522_source_target_alignment_manifest.json"
    report_md = output_dir / "KG1_V522_SOURCE_TARGET_ALIGNMENT_AUDIT.md"

    write_csv(
        gain_csv,
        gains,
        ["id", "family", "answer", "baseline_prediction", "teacher_prediction", "rule", "prompt_sha256"],
    )
    write_csv(
        coverage_csv,
        coverage,
        [
            "source",
            "split",
            "rows",
            "family_counts",
            "marker_CHO",
            "marker_MAJ3",
            "marker_PAR3",
            "marker_XOR",
            "marker_OR",
            "marker_fullbyte_safe_ternary",
            "marker_bit_v300_gain_pattern",
        ],
    )

    current_equation_ids = [row.get("id", "") for row in v516_equation]
    decision = {
        "gpu_allowed": False,
        "dataset_build_allowed": bool(signal_summary["gain_total"] and not signal_summary["loss_total"]),
        "status": "source_signal_found_dataset_build_only",
        "reason": (
            "Reference teacher has no-loss gains, but those gains are not adapter behavior. "
            "Use them only to choose source-side trace families; do not train on weak labels."
        ),
        "next_action": (
            "Build V523 targeted source-only trace pack from permitted v304/v515-like sources: "
            "prioritize CHO/MAJ3/global ternary bit traces and current V516 label-free equation classes "
            f"{current_equation_ids}; then run V286/V513/V521 before any GPU."
        ),
    }
    manifest = {
        "version": "V522",
        "schema_version": "kg1_v522_source_target_alignment_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reference_signal_summary": signal_summary,
        "accepted_cpu_rule_sources": {
            "v357": v357_accepted,
            "v366": v366_accepted,
            "v516_equation_current_label_free": v516_equation,
        },
        "outputs": {
            "gain_csv": str(gain_csv),
            "coverage_csv": str(coverage_csv),
            "manifest_json": str(manifest_path),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_path, manifest)
    write_report(report_md, manifest, coverage)
    return manifest


def write_report(path: Path, manifest: dict[str, Any], coverage: list[dict[str, Any]]) -> None:
    lines = [
        "# V522 Source Target Alignment Audit",
        "",
        "## Decision",
        "",
        f"- GPU allowed: `{manifest['decision']['gpu_allowed']}`",
        f"- Dataset build allowed: `{manifest['decision']['dataset_build_allowed']}`",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Reason: {manifest['decision']['reason']}",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## Reference Signal",
        "",
        f"- No-loss teacher gains: `{manifest['reference_signal_summary']['gain_total']}`",
        f"- Teacher losses vs baseline: `{manifest['reference_signal_summary']['loss_total']}`",
        f"- Gain family counts: `{json.dumps(manifest['reference_signal_summary']['gain_family_counts'], sort_keys=True)}`",
        "",
        "Top gain rules:",
        "",
    ]
    for rule, count in list(manifest["reference_signal_summary"]["gain_rule_counts"].items())[:12]:
        lines.append(f"- `{rule}`: `{count}`")
    lines.extend(
        [
            "",
            "## Source Coverage",
            "",
            "| Source | Split | Rows | CHO | MAJ3 | PAR3 | XOR | OR | fullbyte | gain-pattern |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage:
        lines.append(
            "| {source} | {split} | {rows} | {marker_CHO} | {marker_MAJ3} | {marker_PAR3} | "
            "{marker_XOR} | {marker_OR} | {marker_fullbyte_safe_ternary} | {marker_bit_v300_gain_pattern} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "The gain rows in this audit are diagnostic targets only. They cannot be copied into training labels. "
            "V523 must draw training rows from source-side synthetic/public/train data with no weak/full prompt overlap.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    baseline = [{"id": "a", "answer": "1", "prediction": "0", "family": "equation_transform"}]
    teacher = [{"id": "a", "answer": "1", "prediction": "1", "family": "equation_transform"}]
    gains, summary = summarize_reference_signal(baseline, teacher)
    if len(gains) != 1 or summary["gain_total"] != 1 or summary["loss_total"] != 0:
        raise SystemExit("self-test failed")
    print("audit_v522_source_target_alignment_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--v357-rules-csv", type=Path, default=DEFAULT_V357_RULES_CSV)
    parser.add_argument("--v366-rules-csv", type=Path, default=DEFAULT_V366_RULES_CSV)
    parser.add_argument("--v516-equation-accepted-csv", type=Path, default=DEFAULT_V516_EQUATION_ACCEPTED_CSV)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    manifest = audit(args)
    print("v522_manifest =", manifest["outputs"]["manifest_json"], flush=True)
    print("v522_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("v522_reference_signal =", json.dumps(manifest["reference_signal_summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

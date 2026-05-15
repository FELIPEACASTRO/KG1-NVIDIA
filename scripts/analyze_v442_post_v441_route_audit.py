#!/usr/bin/env python3
"""V442 CPU audit after V441 boxed-payload preference failed to move metrics.

This script does not train and does not use weak/full answers. It audits the
current preference route and decides whether another paid GPU job is justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V439_MANIFEST = ROOT / "artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/v439_final_answer_only_pairs_manifest.json"
DEFAULT_V438_MANIFEST = ROOT / "artifacts/v438_preference_objective_audit/20260515T_v438_v439_final_answer_only/v438_v439_final_answer_only_audit_manifest.json"
DEFAULT_V441_MANIFEST = ROOT / "artifacts/v441_hf_h200_v439_boxed_payload_preference_launch/v441-v439-boxed-payload-pref-v290ckpt6-20260515T165533Z_manifest.json"
DEFAULT_V419_MANIFEST = ROOT / "artifacts/v419_residual_taxonomy/20260515T_v419_residual_taxonomy/v419_residual_taxonomy_manifest.json"
DEFAULT_V433_MANIFEST = ROOT / "artifacts/v433_string_multiset_operator_gate/20260515T_v433_string_multiset_operator/v433_string_multiset_operator_manifest.json"

EXISTING_CERT_FIELDS = [
    "adapter_exact_wrong_certificate",
    "raw_output_collected_without_labels",
    "locked_before_answer_audit",
    "labels_joined_after_collection_from_public_train",
    "weak_gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
]

REQUIRED_RULE_CERT_FIELDS = [
    "rule_unique_label_free",
    "rule_candidate_count",
    "program_or_rule",
    "mdl_score",
    "leave_one_out_pass",
    "renaming_stability_pass",
    "slot_alignment_stats",
    "rule_frozen_before_answer",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def to_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def audit_pair(split: str, row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata", {})
    missing_rule_cert = [field for field in REQUIRED_RULE_CERT_FIELDS if field not in meta]
    existing = {field: boolish(meta.get(field)) for field in EXISTING_CERT_FIELDS}
    weak_used = existing.get("weak_gate_rows_used_for_training", False)
    full_used = existing.get("full_gate_rows_used_for_training", False)
    existing_source_ok = (
        existing.get("adapter_exact_wrong_certificate", False)
        and existing.get("raw_output_collected_without_labels", False)
        and existing.get("locked_before_answer_audit", False)
        and not weak_used
        and not full_used
    )
    rule_certified = not missing_rule_cert and boolish(meta.get("rule_unique_label_free"))
    return {
        "split": split,
        "id": row.get("id", ""),
        "source_id": meta.get("source_id", ""),
        "family": row.get("family", meta.get("family", "")),
        "rule_class": meta.get("rule_class", row.get("subcategory", "")),
        "negative_type": meta.get("negative_type", ""),
        "existing_source_ok": existing_source_ok,
        "rule_certified": rule_certified,
        "missing_rule_cert_fields": "|".join(missing_rule_cert),
        "missing_rule_cert_count": len(missing_rule_cert),
        "adapter_exact_wrong_certificate": existing.get("adapter_exact_wrong_certificate", False),
        "raw_output_collected_without_labels": existing.get("raw_output_collected_without_labels", False),
        "locked_before_answer_audit": existing.get("locked_before_answer_audit", False),
        "labels_joined_after_collection_from_public_train": existing.get("labels_joined_after_collection_from_public_train", False),
        "weak_gate_rows_used_for_training": weak_used,
        "full_gate_rows_used_for_training": full_used,
        "target_style": meta.get("target_style", ""),
        "certification_grade": "rule_certified" if rule_certified else "format_clean_but_not_rule_certified",
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "family_counts": dict(Counter(str(row.get("family", "")) for row in rows)),
        "rule_class_counts": dict(Counter(str(row.get("rule_class", "")) for row in rows)),
        "split_counts": dict(Counter(str(row.get("split", "")) for row in rows)),
        "rule_certified_rows": sum(1 for row in rows if row["rule_certified"]),
        "existing_source_ok_rows": sum(1 for row in rows if row["existing_source_ok"]),
        "weak_full_training_rows": sum(
            1
            for row in rows
            if row["weak_gate_rows_used_for_training"] or row["full_gate_rows_used_for_training"]
        ),
        "missing_rule_cert_counts": dict(
            Counter(
                field
                for row in rows
                for field in str(row.get("missing_rule_cert_fields", "")).split("|")
                if field
            )
        ),
    }


def compare_v441(v441_manifest: dict[str, Any]) -> dict[str, Any]:
    result = v441_manifest.get("result", {})
    baseline = result.get("baseline_preference_eval", {})
    ckpt3 = result.get("checkpoint_3_preference_eval", {})
    return {
        "job_id": v441_manifest.get("job_id", ""),
        "job_url": v441_manifest.get("job_url", ""),
        "status": v441_manifest.get("job_status", ""),
        "decision": result.get("decision", ""),
        "baseline_total": baseline.get("preference_correct"),
        "checkpoint3_total": ckpt3.get("preference_correct"),
        "baseline_equation": baseline.get("equation_transform_correct"),
        "checkpoint3_equation": ckpt3.get("equation_transform_correct"),
        "baseline_bit": baseline.get("bit_manipulation_correct"),
        "checkpoint3_bit": ckpt3.get("bit_manipulation_correct"),
        "total_delta": (ckpt3.get("preference_correct") or 0) - (baseline.get("preference_correct") or 0),
        "equation_delta": (ckpt3.get("equation_transform_correct") or 0) - (baseline.get("equation_transform_correct") or 0),
        "bit_delta": (ckpt3.get("bit_manipulation_correct") or 0) - (baseline.get("bit_manipulation_correct") or 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "id",
        "source_id",
        "family",
        "rule_class",
        "negative_type",
        "existing_source_ok",
        "rule_certified",
        "missing_rule_cert_count",
        "missing_rule_cert_fields",
        "adapter_exact_wrong_certificate",
        "raw_output_collected_without_labels",
        "locked_before_answer_audit",
        "labels_joined_after_collection_from_public_train",
        "weak_gate_rows_used_for_training",
        "full_gate_rows_used_for_training",
        "target_style",
        "certification_grade",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(
    path: Path,
    manifest: dict[str, Any],
    pair_summary: dict[str, Any],
    v441_summary: dict[str, Any],
    v419_manifest: dict[str, Any],
    v433_manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy = v419_manifest.get("taxonomy", [])
    v433_projection = v433_manifest.get("projection", {})
    lines = [
        "# V442 Post-V441 Route Audit",
        "",
        "## Decision",
        "",
        f"- Decision: `{manifest['decision']['decision']}`",
        f"- Reason: {manifest['decision']['reason']}",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## What V441 Proved",
        "",
        "| Metric | Baseline | Checkpoint-3 | Delta |",
        "|---|---:|---:|---:|",
        f"| Preference total | {v441_summary['baseline_total']} | {v441_summary['checkpoint3_total']} | {v441_summary['total_delta']} |",
        f"| equation_transform | {v441_summary['baseline_equation']} | {v441_summary['checkpoint3_equation']} | {v441_summary['equation_delta']} |",
        f"| bit_manipulation | {v441_summary['baseline_bit']} | {v441_summary['checkpoint3_bit']} | {v441_summary['bit_delta']} |",
        "",
        "V441 was technically healthy but did not move the validation signal. This blocks another",
        "GPU relaunch on the same V439/V435E preference family.",
        "",
        "## Pair Certification Audit",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| audited rows | {pair_summary['rows']} |",
        f"| existing source-ok rows | {pair_summary['existing_source_ok_rows']} |",
        f"| rule-certified rows | {pair_summary['rule_certified_rows']} |",
        f"| weak/full training rows | {pair_summary['weak_full_training_rows']} |",
        "",
        "The source is clean enough for diagnostics, but not enough for another paid ranking job.",
        "The missing piece is a label-free rule certificate, not another loss variant.",
        "",
        "## Residual Evidence",
        "",
        "V419 residual taxonomy says the remaining hard equation work is mostly symbolic punctuation:",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for item in taxonomy:
        lines.append(f"| `{item.get('bucket', '')}` | {item.get('count', 0)} |")
    lines.extend(
        [
            "",
            "V433 found correct answers only inside ambiguous sets, not unique label-free gains:",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| accepted new gains | {v433_manifest.get('accepted_new_gains', 0)} |",
            f"| ambiguous correct candidate rows | {v433_manifest.get('ambiguous_correct_candidate_rows', 0)} |",
            f"| projected total | {v433_projection.get('total_correct', '')} |",
            f"| projected equation | {v433_projection.get('equation_transform_correct', '')} |",
            f"| projected bit | {v433_projection.get('bit_manipulation_correct', '')} |",
            "",
            "## Active Implementation Order",
            "",
            "1. Build a true CPU certified pair builder for `equation_symbolic_sequence` and `equation_symbolic_short`.",
            "2. Freeze each candidate rule before looking at the public-train answer.",
            "3. Require MDL, leave-one-out, renaming stability, and unique candidate count.",
            "4. Only after at least four independent equation modes pass, regenerate preference rows.",
            "5. Only then consider HF GPU; otherwise remain CPU-only.",
            "",
            "## Outputs",
            "",
            f"- Pair audit CSV: `{manifest['outputs']['pair_audit_csv']}`",
            f"- Manifest: `{manifest['outputs']['manifest_json']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v439_manifest = read_json(args.v439_manifest_json)
    v438_manifest = read_json(args.v438_manifest_json)
    v441_manifest = read_json(args.v441_manifest_json)
    v419_manifest = read_json(args.v419_manifest_json)
    v433_manifest = read_json(args.v433_manifest_json)

    train_rows = read_jsonl(to_path(v439_manifest["train_jsonl"]))
    val_rows = read_jsonl(to_path(v439_manifest["val_jsonl"]))
    audited_rows = [audit_pair("train", row) for row in train_rows]
    audited_rows.extend(audit_pair("validation", row) for row in val_rows)

    pair_summary = summarize_rows(audited_rows)
    v441_summary = compare_v441(v441_manifest)
    structural_ok = bool(v438_manifest.get("hf_gpu_allowed_for_same_objective"))

    same_route_gpu_blocked = (
        v441_summary["total_delta"] <= 0
        and v441_summary["equation_delta"] <= 0
        and pair_summary["rule_certified_rows"] == 0
    )
    decision = {
        "decision": "same_preference_route_blocked_return_to_cpu_certified_builder"
        if same_route_gpu_blocked
        else "manual_review_required",
        "reason": (
            "V441 tied baseline under boxed-payload scoring and V439 pairs have zero rule-certified rows."
            if same_route_gpu_blocked
            else "Unexpected audit state; inspect manifest before any HF job."
        ),
        "next_action": (
            "Implement CPU certified equation pair builder; do not launch another V435E/V439 preference GPU job."
            if same_route_gpu_blocked
            else "Manual review before spending GPU."
        ),
    }

    pair_audit_csv = output_dir / f"{args.label}_pair_certification_audit.csv"
    report_md = output_dir / f"{args.label}_report.md"
    manifest_json = output_dir / f"{args.label}_manifest.json"
    write_csv(pair_audit_csv, audited_rows)

    manifest: dict[str, Any] = {
        "schema_version": "kg1_v442_post_v441_route_audit_v1",
        "label": args.label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v439_manifest_json": rel(args.v439_manifest_json),
            "v438_manifest_json": rel(args.v438_manifest_json),
            "v441_manifest_json": rel(args.v441_manifest_json),
            "v419_manifest_json": rel(args.v419_manifest_json),
            "v433_manifest_json": rel(args.v433_manifest_json),
        },
        "v438_structural_audit_ok": structural_ok,
        "v441_summary": v441_summary,
        "pair_summary": pair_summary,
        "same_route_gpu_blocked": same_route_gpu_blocked,
        "hf_gpu_allowed": False,
        "decision": decision,
        "outputs": {
            "pair_audit_csv": rel(pair_audit_csv),
            "report_md": rel(report_md),
            "manifest_json": rel(manifest_json),
        },
    }

    write_markdown(report_md, manifest, pair_summary, v441_summary, v419_manifest, v433_manifest)
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v439-manifest-json", type=Path, default=DEFAULT_V439_MANIFEST)
    parser.add_argument("--v438-manifest-json", type=Path, default=DEFAULT_V438_MANIFEST)
    parser.add_argument("--v441-manifest-json", type=Path, default=DEFAULT_V441_MANIFEST)
    parser.add_argument("--v419-manifest-json", type=Path, default=DEFAULT_V419_MANIFEST)
    parser.add_argument("--v433-manifest-json", type=Path, default=DEFAULT_V433_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v442_post_v441_route_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(args)
    print("=== V442 POST V441 ROUTE AUDIT START ===", flush=True)
    print("label =", manifest["label"], flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("v441_summary =", json.dumps(manifest["v441_summary"], sort_keys=True), flush=True)
    print("pair_summary =", json.dumps(manifest["pair_summary"], sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", manifest["outputs"]["manifest_json"], flush=True)
    print("=== V442 POST V441 ROUTE AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

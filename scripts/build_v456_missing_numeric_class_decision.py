#!/usr/bin/env python3
"""V456 missing numeric class decision.

This CPU-only gate follows V455. It asks whether the three V324-verified
equation classes that V452 did not promote are ready for paid HF training.

The answer must be based on evidence already available in the repo:

* V455 class gaps: which verified weak-side rules are still missing.
* V452 public-train hard-negative audit: whether those rules have legal
  adapter-wrong pairs.
* Earlier synthetic datasets: whether "more synthetic SFT" for the same class
  has already been tried.

The script deliberately does not build train rows from weak/full examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v456_missing_numeric_class_decision" / "20260515T_cpu_gate"
DEFAULT_V455_CLASS_GAPS = (
    ROOT
    / "artifacts"
    / "v455_equation_target_audit"
    / "20260515T_cpu_gate"
    / "v455_equation_target_audit_class_gaps.csv"
)
DEFAULT_V452_AUDIT = (
    ROOT
    / "artifacts"
    / "v452_equation_dsl_v2_certified_builder"
    / "20260515T_cpu_gate"
    / "v452_equation_dsl_v2_certified_builder_audit.csv"
)

PRIOR_SYNTHETIC_MANIFESTS = [
    ROOT / "artifacts/v290_rank19_micro_patch_dataset/20260511T1925Z/v282_rank19_micro_patch_manifest.json",
    ROOT / "artifacts/v293_v274_distill_dataset/20260511T2338Z/v293_v274_distill_manifest.json",
    ROOT / "artifacts/v294_verified_equation_patch_dataset/20260512T012919Z/v294_verified_equation_patch_manifest.json",
]

RULE_SUFFIX = "v274_guarded_numeric_"

RULE_TO_SUBCATEGORY = {
    "add_direct_over_model_add_variant": "equation_numeric_add_direct",
    "colon_absdiff_restore_trailing_zero": "equation_numeric_colon_trailing_zero",
    "minus_signed_opposite_sign_guarded": "equation_numeric_minus_signed",
}

KNOWN_TRANSFER_FAILURES = {
    "add_direct_over_model_add_variant": (
        "V290/V293/V294 already generated large legal synthetic add_direct "
        "coverage, but later adapter-only weak evals stayed at equation=56."
    ),
    "minus_signed_opposite_sign_guarded": (
        "V290/V293/V294 already generated large legal synthetic minus_signed "
        "coverage, but later adapter-only weak evals stayed at equation=56."
    ),
    "colon_absdiff_restore_trailing_zero": (
        "The builder exists, but this exact trailing-zero variant was not part "
        "of the earlier generated rule mix; it still lacks a legal public-train "
        "adapter hard-negative pair."
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rule_name(rule_class: str) -> str:
    text = str(rule_class)
    return text[len(RULE_SUFFIX) :] if text.startswith(RULE_SUFFIX) else text


def count_key_recursive(payload: Any, key: str) -> int:
    total = 0
    if isinstance(payload, dict):
        for item_key, item_value in payload.items():
            if str(item_key) == key and isinstance(item_value, int):
                total += int(item_value)
            total += count_key_recursive(item_value, key)
    elif isinstance(payload, list):
        for item in payload:
            total += count_key_recursive(item, key)
    return total


def source_has_builder(source_text: str, rule: str, subcategory: str) -> bool:
    return rule in source_text or subcategory in source_text


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V456 MISSING NUMERIC CLASS DECISION START ===", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("v455_class_gaps_csv =", args.v455_class_gaps_csv, "exists =", args.v455_class_gaps_csv.exists(), flush=True)
    print("v452_audit_csv =", args.v452_audit_csv, "exists =", args.v452_audit_csv.exists(), flush=True)
    if not args.v455_class_gaps_csv.exists():
        raise FileNotFoundError(args.v455_class_gaps_csv)
    if not args.v452_audit_csv.exists():
        raise FileNotFoundError(args.v452_audit_csv)

    v455_rows = read_csv(args.v455_class_gaps_csv)
    v452_rows = read_csv(args.v452_audit_csv)
    builder_source_path = ROOT / "scripts" / "build_v282_rank19_micro_patch_dataset.py"
    builder_source = builder_source_path.read_text(encoding="utf-8") if builder_source_path.exists() else ""

    prior_manifests: list[dict[str, Any]] = []
    for manifest_path in PRIOR_SYNTHETIC_MANIFESTS:
        print("prior_manifest =", manifest_path, "exists =", manifest_path.exists(), flush=True)
        if manifest_path.exists():
            prior_manifests.append({"path": str(manifest_path), "payload": read_json(manifest_path)})

    promoted_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for row in v452_rows:
        rc = str(row.get("rule_class", ""))
        if not rc:
            continue
        if str(row.get("status", "")) == "candidate":
            candidate_counts[rc] = candidate_counts.get(rc, 0) + 1
        if boolish(row.get("row_promoted")):
            promoted_counts[rc] = promoted_counts.get(rc, 0) + 1

    decision_rows: list[dict[str, Any]] = []
    for row in v455_rows:
        if str(row.get("status", "")) == "covered":
            continue
        rc = str(row["rule_class"])
        rule = rule_name(rc)
        subcategory = RULE_TO_SUBCATEGORY.get(rule, "")
        prior_rule_count = 0
        prior_subcategory_count = 0
        prior_sources: list[str] = []
        for item in prior_manifests:
            payload = item["payload"]
            rule_count = count_key_recursive(payload, rule)
            subcat_count = count_key_recursive(payload, subcategory) if subcategory else 0
            if rule_count or subcat_count:
                prior_sources.append(str(item["path"]))
            prior_rule_count += rule_count
            prior_subcategory_count += subcat_count

        has_builder = source_has_builder(builder_source, rule, subcategory)
        public_promoted = int(promoted_counts.get(rc, 0))
        public_candidates = int(candidate_counts.get(rc, 0))

        if public_promoted > 0:
            action = "eligible_for_prepaid_gate"
            reason = "legal public-train hard-negative pair already promoted"
        elif prior_rule_count or prior_subcategory_count:
            action = "blocked_prior_synthetic_transfer_failed"
            reason = KNOWN_TRANSFER_FAILURES.get(rule, "synthetic coverage exists but no public hard-negative pair")
        elif has_builder:
            action = "needs_public_train_raw_probe_before_gpu"
            reason = KNOWN_TRANSFER_FAILURES.get(rule, "builder exists but no promoted public hard-negative pair")
        else:
            action = "needs_new_builder_before_gpu"
            reason = "no known builder and no promoted public hard-negative pair"

        decision_rows.append(
            {
                "rule_class": rc,
                "rule": rule,
                "v455_gap": row.get("gap_verified_not_promoted", ""),
                "v455_accepted_ids": row.get("accepted_ids", ""),
                "public_train_candidates": public_candidates,
                "public_train_promoted": public_promoted,
                "prior_synthetic_rule_count": prior_rule_count,
                "prior_synthetic_subcategory_count": prior_subcategory_count,
                "prior_synthetic_sources": ";".join(prior_sources),
                "builder_exists": has_builder,
                "action": action,
                "reason": reason,
            }
        )

    eligible = [row for row in decision_rows if row["action"] == "eligible_for_prepaid_gate"]
    needs_probe = [row for row in decision_rows if row["action"] == "needs_public_train_raw_probe_before_gpu"]
    synthetic_failed = [row for row in decision_rows if row["action"] == "blocked_prior_synthetic_transfer_failed"]
    needs_builder = [row for row in decision_rows if row["action"] == "needs_new_builder_before_gpu"]

    hf_gpu_allowed = bool(eligible) and not needs_probe and not synthetic_failed and not needs_builder
    if hf_gpu_allowed:
        decision_name = "v456_missing_numeric_classes_ready_for_prepaid_gate"
        next_action = "Run tokenization/leakage/pre-paid integration gates before a short HF job."
    else:
        decision_name = "v456_no_gpu_expand_public_train_raw_probe"
        next_action = (
            "Build V457 public-train numeric raw-output probe pack for the missing classes; "
            "only train if it yields legal adapter-wrong pairs with zero leakage."
        )

    manifest = {
        "schema_version": "kg1_v456_missing_numeric_class_decision_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v455_class_gaps_csv": str(args.v455_class_gaps_csv),
            "v452_audit_csv": str(args.v452_audit_csv),
            "prior_synthetic_manifests": [item["path"] for item in prior_manifests],
            "builder_source_path": str(builder_source_path),
        },
        "summary": {
            "missing_classes": len(decision_rows),
            "eligible_for_prepaid_gate": len(eligible),
            "blocked_prior_synthetic_transfer_failed": len(synthetic_failed),
            "needs_public_train_raw_probe_before_gpu": len(needs_probe),
            "needs_new_builder_before_gpu": len(needs_builder),
            "hf_gpu_allowed": hf_gpu_allowed,
        },
        "decision_rows": decision_rows,
        "decision": {
            "decision": decision_name,
            "hf_gpu_allowed": hf_gpu_allowed,
            "next_action": next_action,
            "reason": (
                f"eligible={len(eligible)}; synthetic_failed={len(synthetic_failed)}; "
                f"needs_probe={len(needs_probe)}; needs_builder={len(needs_builder)}"
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "v456_missing_numeric_class_decision_manifest.json"
    decision_csv = args.output_dir / "v456_missing_numeric_class_decision.csv"
    report_md = args.output_dir / "V456_MISSING_NUMERIC_CLASS_DECISION.md"
    write_json(manifest_path, manifest)
    write_csv(
        decision_csv,
        decision_rows,
        [
            "rule_class",
            "rule",
            "v455_gap",
            "v455_accepted_ids",
            "public_train_candidates",
            "public_train_promoted",
            "prior_synthetic_rule_count",
            "prior_synthetic_subcategory_count",
            "builder_exists",
            "action",
            "reason",
            "prior_synthetic_sources",
        ],
    )

    lines = [
        "# V456 Missing Numeric Class Decision",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Missing classes audited | `{len(decision_rows)}` |",
        f"| Eligible for prepaid gate | `{len(eligible)}` |",
        f"| Blocked by prior synthetic transfer failure | `{len(synthetic_failed)}` |",
        f"| Needs public-train raw probe | `{len(needs_probe)}` |",
        f"| Needs new builder | `{len(needs_builder)}` |",
        f"| `hf_gpu_allowed` | `{str(hf_gpu_allowed).lower()}` |",
        "",
        "## Class Decisions",
        "",
        "| Rule | Gap | Public promoted | Prior synthetic count | Builder | Action |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in decision_rows:
        lines.append(
            f"| `{row['rule']}` | `{row['v455_gap']}` | `{row['public_train_promoted']}` | "
            f"`{row['prior_synthetic_rule_count']}` | `{str(row['builder_exists']).lower()}` | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            manifest["decision"]["reason"],
            "",
            manifest["decision"]["next_action"],
            "",
            "Do not open HF GPU from V456. The missing classes still need legal public-train adapter-wrong evidence.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("manifest_json =", manifest_path, flush=True)
    print("report_md =", report_md, flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("=== V456 MISSING NUMERIC CLASS DECISION END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v455-class-gaps-csv", type=Path, default=DEFAULT_V455_CLASS_GAPS)
    parser.add_argument("--v452-audit-csv", type=Path, default=DEFAULT_V452_AUDIT)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

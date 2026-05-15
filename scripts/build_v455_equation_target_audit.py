#!/usr/bin/env python3
"""V455 equation target audit.

This CPU-only audit compares the strongest known equation CPU signal (V324) with
the certified pair builder output (V452). It identifies which verified equation
classes are still not converted into trainable adapter targets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v455_equation_target_audit" / "20260515T_cpu_gate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V455 EQUATION TARGET AUDIT START ===", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    paths = [
        args.v324_manifest_json,
        args.v324_accepted_csv,
        args.v324_rule_summary_csv,
        args.v452_manifest_json,
        args.v452_audit_csv,
    ]
    for path in paths:
        print("input =", path, "exists =", path.exists(), flush=True)
        if not path.exists():
            raise FileNotFoundError(path)

    v324_manifest = read_json(args.v324_manifest_json)
    v324_accepted = read_csv(args.v324_accepted_csv)
    v324_rule_summary = read_csv(args.v324_rule_summary_csv)
    v452_manifest = read_json(args.v452_manifest_json)
    v452_audit = read_csv(args.v452_audit_csv)

    v324_verified_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in v324_accepted:
        v324_verified_by_class[row["rule_class"]].append(row)

    v452_promoted_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    v452_candidate_by_class: Counter[str] = Counter()
    for row in v452_audit:
        rule = row.get("rule_class", "")
        if row.get("status") == "candidate" and rule:
            v452_candidate_by_class[rule] += 1
        if boolish(row.get("row_promoted")):
            v452_promoted_by_class[rule].append(row)

    class_rows: list[dict[str, Any]] = []
    missing_verified_rows: list[dict[str, Any]] = []
    for rule_class, rows in sorted(v324_verified_by_class.items()):
        promoted = v452_promoted_by_class.get(rule_class, [])
        row = {
            "rule_class": rule_class,
            "v324_verified": len(rows),
            "v452_candidates": int(v452_candidate_by_class.get(rule_class, 0)),
            "v452_promoted": len(promoted),
            "gap_verified_not_promoted": max(0, len(rows) - len(promoted)),
            "status": "covered" if len(promoted) >= len(rows) else "missing_builder_target",
            "accepted_ids": ",".join(item["id"] for item in rows),
            "promoted_ids": ",".join(item["source_id"] for item in promoted),
        }
        class_rows.append(row)
        if row["status"] != "covered":
            for item in rows:
                missing_verified_rows.append(
                    {
                        "id": item["id"],
                        "rule_class": rule_class,
                        "subtype": item.get("subtype", ""),
                        "query": item.get("query", ""),
                        "answer": item.get("answer", ""),
                        "baseline_prediction": item.get("baseline_prediction", ""),
                        "prediction": item.get("prediction", ""),
                        "candidate_source": item.get("candidate_source", ""),
                    }
                )

    subtype_counts = v324_manifest.get("subtype_counts", {})
    numeric_miss_rows = int(subtype_counts.get("equation_numeric_operator", 0) or 0)
    symbolic_miss_rows = int(subtype_counts.get("equation_symbolic_punct", 0) or 0)
    v324_accepted = int(v324_manifest.get("accepted_candidate_count", 0) or 0)
    v452_pairs = int(v452_manifest.get("summary", {}).get("certified_pair_rows", 0) or 0)
    missing_verified = len(missing_verified_rows)
    independent_missing_classes = len({row["rule_class"] for row in missing_verified_rows})
    symbolic_verified = sum(
        int(row.get("verified_candidates", 0) or 0)
        for row in v324_rule_summary
        if str(row.get("rule_class", "")).startswith("symbolic_")
    )

    hf_gpu_allowed = False
    if missing_verified == 0 and v452_pairs >= args.min_pairs and int(v452_manifest.get("summary", {}).get("independent_modes", 0)) >= args.min_modes:
        hf_gpu_allowed = True

    if missing_verified:
        decision_name = "v455_missing_verified_equation_targets_no_gpu"
        next_action = (
            "Implement V456 builder for the missing V324 verified numeric classes before any HF GPU."
        )
    elif symbolic_verified == 0 and symbolic_miss_rows:
        decision_name = "v455_symbolic_uncovered_no_gpu"
        next_action = "Build symbolic-specific probes; do not train from numeric-only signal."
    elif hf_gpu_allowed:
        decision_name = "v455_equation_targets_ready_for_prepaid_gate"
        next_action = "Run tokenization, leakage, pre-paid job integration gate, then one short HF smoke."
    else:
        decision_name = "v455_equation_targets_insufficient_no_gpu"
        next_action = "Expand CPU builder; do not launch paid GPU."

    decision = {
        "decision": decision_name,
        "hf_gpu_allowed": hf_gpu_allowed,
        "reason": (
            f"v324_accepted={v324_accepted}; v452_pairs={v452_pairs}; "
            f"missing_verified={missing_verified}; missing_classes={independent_missing_classes}; "
            f"symbolic_verified={symbolic_verified}"
        ),
        "next_action": next_action,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_csv = args.output_dir / "v455_equation_target_audit_class_gaps.csv"
    missing_csv = args.output_dir / "v455_equation_target_audit_missing_verified_rows.csv"
    manifest_json = args.output_dir / "v455_equation_target_audit_manifest.json"
    report_md = args.output_dir / "V455_EQUATION_TARGET_AUDIT.md"

    write_csv(
        class_csv,
        class_rows,
        [
            "rule_class",
            "v324_verified",
            "v452_candidates",
            "v452_promoted",
            "gap_verified_not_promoted",
            "status",
            "accepted_ids",
            "promoted_ids",
        ],
    )
    write_csv(
        missing_csv,
        missing_verified_rows,
        ["id", "rule_class", "subtype", "query", "answer", "baseline_prediction", "prediction", "candidate_source"],
    )

    manifest = {
        "schema_version": "kg1_v455_equation_target_audit_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v324_manifest_json": str(args.v324_manifest_json),
            "v324_accepted_csv": str(args.v324_accepted_csv),
            "v324_rule_summary_csv": str(args.v324_rule_summary_csv),
            "v452_manifest_json": str(args.v452_manifest_json),
            "v452_audit_csv": str(args.v452_audit_csv),
        },
        "summary": {
            "equation_miss_rows": int(v324_manifest.get("equation_miss_rows", 0) or 0),
            "numeric_miss_rows": numeric_miss_rows,
            "symbolic_miss_rows": symbolic_miss_rows,
            "v324_accepted_candidates": v324_accepted,
            "v452_certified_pairs": v452_pairs,
            "missing_verified_rows": missing_verified,
            "missing_verified_classes": independent_missing_classes,
            "symbolic_verified_candidates": symbolic_verified,
            "class_gaps": class_rows,
        },
        "decision": decision,
        "outputs": {
            "class_gaps_csv": str(class_csv),
            "missing_verified_rows_csv": str(missing_csv),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_json, manifest)

    lines = [
        "# V455 Equation Target Audit",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Equation misses audited by V324 | `{manifest['summary']['equation_miss_rows']}` |",
        f"| Numeric misses | `{numeric_miss_rows}` |",
        f"| Symbolic/punctuation misses | `{symbolic_miss_rows}` |",
        f"| V324 accepted no-loss candidates | `{v324_accepted}` |",
        f"| V452 certified trainable pairs | `{v452_pairs}` |",
        f"| Verified rows missing from builder | `{missing_verified}` |",
        f"| Missing verified classes | `{independent_missing_classes}` |",
        f"| Symbolic verified candidates | `{symbolic_verified}` |",
        f"| `hf_gpu_allowed` | `{str(hf_gpu_allowed).lower()}` |",
        "",
        "## Class Gap",
        "",
        "| Rule class | V324 verified | V452 promoted | Gap | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in class_rows:
        lines.append(
            f"| `{row['rule_class']}` | `{row['v324_verified']}` | `{row['v452_promoted']}` | "
            f"`{row['gap_verified_not_promoted']}` | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision["reason"],
            "",
            decision["next_action"],
            "",
            "No HF GPU is allowed from this audit until V456 closes the verified-class gap.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("manifest_json =", manifest_json, flush=True)
    print("report_md =", report_md, flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("=== V455 EQUATION TARGET AUDIT END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    base = REPO_ROOT / "artifacts" / "v394_equation_row_level_inventory" / "20260514T_cpu_gate" / "v324_on_v290_checkpoint6"
    v452 = REPO_ROOT / "artifacts" / "v452_equation_dsl_v2_certified_builder" / "20260515T_cpu_gate"
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v324-manifest-json", type=Path, default=base / "v324_equation_expanded_solver_manifest.json")
    parser.add_argument("--v324-accepted-csv", type=Path, default=base / "v324_equation_expanded_solver_accepted_candidates.csv")
    parser.add_argument("--v324-rule-summary-csv", type=Path, default=base / "v324_equation_expanded_solver_rule_summary.csv")
    parser.add_argument("--v452-manifest-json", type=Path, default=v452 / "v452_equation_dsl_v2_certified_builder_manifest.json")
    parser.add_argument("--v452-audit-csv", type=Path, default=v452 / "v452_equation_dsl_v2_certified_builder_audit.csv")
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--min-modes", type=int, default=4)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

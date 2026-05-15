#!/usr/bin/env python3
"""Build V419 residual taxonomy after V418.

The goal is not to relabel data or authorize training. It identifies which
unsolved weak rows remain after the best current CPU teacher so the next step is
a new rule class, not another wider run of the same DSL.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "v419_residual_taxonomy" / "20260515T_v419_residual_taxonomy"
BASELINE_CSV = ROOT / "artifacts" / "v342_acc_first_diagnostic" / "v290_checkpoint6_baseline_predictions.csv"
ACCEPTED_CSV = (
    ROOT
    / "artifacts"
    / "v418_cpu_synthesis_aggressive_gate"
    / "20260515T_v418_cpu_aggressive_gate"
    / "v412_integrated_accepted.csv"
)
AUDIT_CSV = (
    ROOT
    / "artifacts"
    / "v418_cpu_synthesis_aggressive_gate"
    / "20260515T_v418_cpu_aggressive_gate"
    / "v412_candidate_audit.csv"
)
FALSE_POSITIVE_CSV = (
    ROOT
    / "artifacts"
    / "v418_cpu_synthesis_aggressive_gate"
    / "20260515T_v418_cpu_aggressive_gate"
    / "v412_false_positive_candidates.csv"
)
CONFLICT_CSV = (
    ROOT
    / "artifacts"
    / "v418_cpu_synthesis_aggressive_gate"
    / "20260515T_v418_cpu_aggressive_gate"
    / "v412_conflicts.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def answer_class(answer: str) -> str:
    text = str(answer or "")
    if re.fullmatch(r"-?\d+", text):
        return "numeric_signed" if text.startswith("-") else "numeric_unsigned"
    if any(ch.isalnum() for ch in text) and any(not ch.isalnum() for ch in text):
        return "mixed_symbolic"
    if any(not ch.isalnum() for ch in text):
        return "punct_only"
    return "alpha_or_other"


def prompt_hint(prompt: str) -> str:
    text = str(prompt or "")
    if re.search(r"\d+\s*[-+*/:@#$%^&|?<>\\{}\[\]!]+\s*\d+", text):
        return "numeric_operator_prompt"
    if "equation" in text.lower() and any(ch in text for ch in "{}[]<>!?@#$%^&|\\"):
        return "symbolic_punctuation_prompt"
    return "other_equation_prompt"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [BASELINE_CSV, ACCEPTED_CSV, AUDIT_CSV, FALSE_POSITIVE_CSV, CONFLICT_CSV]:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline = read_csv(BASELINE_CSV)
    accepted = {row["id"] for row in read_csv(ACCEPTED_CSV)}
    audit_by_id = {row["id"]: row for row in read_csv(AUDIT_CSV)}
    false_positive = read_csv(FALSE_POSITIVE_CSV)
    conflicts = read_csv(CONFLICT_CSV)

    residual_rows: list[dict[str, str]] = []
    counters = Counter()
    for row in baseline:
        family = row.get("type") or row.get("family")
        if family != "equation_transform":
            continue
        baseline_correct = str(row.get("correct", "")).lower() == "true"
        if baseline_correct or row["id"] in accepted:
            continue
        audit = audit_by_id.get(row["id"], {})
        aclass = answer_class(row["answer"])
        phint = prompt_hint(row["prompt"])
        key = f"{aclass}::{phint}::{audit.get('reason', 'missing_audit')}"
        counters[key] += 1
        residual_rows.append(
            {
                "id": row["id"],
                "answer": row["answer"],
                "baseline_prediction": row.get("prediction", ""),
                "answer_class": aclass,
                "prompt_hint": phint,
                "audit_status": audit.get("status", ""),
                "audit_reason": audit.get("reason", ""),
                "candidate_prediction": audit.get("prediction", ""),
            }
        )

    taxonomy_rows = [
        {"bucket": key, "count": value}
        for key, value in sorted(counters.items(), key=lambda item: (-item[1], item[0]))
    ]

    next_targets = [
        {
            "priority": 1,
            "target": "symbolic_punctuation_structural_solver",
            "why": "Most remaining equation misses are punctuation/mixed symbolic rows where V412 reports no_consistent_vsa_program.",
            "gpu_allowed": False,
        },
        {
            "priority": 2,
            "target": "signed_numeric_guarded_solver",
            "why": "V418 conflicts show sign-handling candidates such as 7ac90433 would regress baseline if not explicitly guarded.",
            "gpu_allowed": False,
        },
        {
            "priority": 3,
            "target": "bit_fullbyte_formula_to_runtime_or_package_path",
            "why": "V414 proves many bit gains formally, but V415/V416 show they do not transfer to adapter-only SFT.",
            "gpu_allowed": False,
        },
    ]

    manifest = {
        "schema_version": "kg1_v419_residual_taxonomy_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_csv": str(BASELINE_CSV.relative_to(ROOT)),
        "accepted_csv": str(ACCEPTED_CSV.relative_to(ROOT)),
        "equation_residual_rows_after_v418": len(residual_rows),
        "taxonomy": taxonomy_rows,
        "false_positive_count": len(false_positive),
        "conflict_count": len(conflicts),
        "next_targets": next_targets,
        "hf_gpu_allowed": False,
        "decision": {
            "decision": "mine_new_symbolic_rule_class_before_gpu",
            "reason": "V418 found zero new safe gains by widening existing DSL parameters.",
            "next_action": "Implement a new symbolic punctuation structural solver gate; do not train.",
        },
        "outputs": {
            "residual_rows_csv": str((OUT_DIR / "v419_equation_residual_rows.csv").relative_to(ROOT)),
            "taxonomy_csv": str((OUT_DIR / "v419_residual_taxonomy.csv").relative_to(ROOT)),
            "next_targets_json": str((OUT_DIR / "v419_next_targets.json").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v419_residual_taxonomy_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V419_RESIDUAL_TAXONOMY.md").relative_to(ROOT)),
        },
    }

    write_csv(OUT_DIR / "v419_equation_residual_rows.csv", residual_rows)
    write_csv(OUT_DIR / "v419_residual_taxonomy.csv", taxonomy_rows)
    (OUT_DIR / "v419_next_targets.json").write_text(json.dumps(next_targets, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "v419_residual_taxonomy_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    top_rows = "\n".join(f"| {row['bucket']} | `{row['count']}` |" for row in taxonomy_rows[:12])
    report = f"""# V419 Residual Taxonomy

Generated: {manifest['generated_at_utc']}

V419 analyzes the equation rows still unsolved after the best current CPU projection from V409/V412/V418.

| Metric | Value |
|---|---:|
| Equation residual rows after V418 | `{len(residual_rows)}` |
| False positives blocked in V418 | `{len(false_positive)}` |
| Conflicts/losses blocked in V418 | `{len(conflicts)}` |

## Top Residual Buckets

| Bucket | Count |
|---|---:|
{top_rows}

## Decision

`hf_gpu_allowed = false`.

The next useful work is a new symbolic punctuation structural solver gate. Re-running V412 with wider caps already produced `0` new safe gains.
"""
    (OUT_DIR / "V419_RESIDUAL_TAXONOMY.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

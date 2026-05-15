#!/usr/bin/env python3
"""Build V417 transfer blocker gate artifacts.

This gate is intentionally CPU-only. It consolidates the failed transfer routes
and emits an explicit allow/block decision before any further HF GPU spend.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "v417_transfer_blocker_gate" / "20260515T_v417_transfer_blocker_gate"

BASELINE = {
    "name": "V291/V290 checkpoint-6 baseline",
    "total": 192,
    "equation_transform": 56,
    "bit_manipulation": 136,
    "truncated": 0,
}

V416_SUMMARY = (
    ROOT
    / "artifacts"
    / "v416_hf_h200_rawstyle_transfer_launch"
    / "eval_summary_v416_20260515T031825Z"
    / "batch_candidate_summary.json"
)
V414_STAGE_SUMMARY = (
    ROOT
    / "artifacts"
    / "v414_cpu_teacher_meta_gate"
    / "20260515T_v414_cpu_teacher_meta_gate"
    / "v414_stage_summary.csv"
)
V415_MATRIX = (
    ROOT
    / "artifacts"
    / "v415_adapter_direct_audit"
    / "20260515T_v415_adapter_direct_audit"
    / "v415_v414_gain_hit_matrix.csv"
)


def read_json(path: Path) -> dict:
    return json.loads(fs_path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with fs_path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fs_path(path: Path) -> Path:
    """Return a filesystem path that works with long Windows paths."""
    resolved = path.resolve()
    if os.name == "nt":
        text = str(resolved)
        if not text.startswith("\\\\?\\"):
            return Path("\\\\?\\" + text)
    return resolved


def candidate_decision(row: dict) -> dict:
    total = int(row["correct"])
    eq = int(row["equation_transform_correct"])
    bit = int(row["bit_manipulation_correct"])
    trunc = int(row["truncated"])
    passes = total > BASELINE["total"] and eq > BASELINE["equation_transform"] and bit >= BASELINE["bit_manipulation"] and trunc == 0
    return {
        "candidate": row["name"],
        "total": total,
        "equation_transform": eq,
        "bit_manipulation": bit,
        "truncated": trunc,
        "delta_total": total - BASELINE["total"],
        "delta_equation_transform": eq - BASELINE["equation_transform"],
        "delta_bit_manipulation": bit - BASELINE["bit_manipulation"],
        "delta_truncated": trunc - BASELINE["truncated"],
        "passes_promotion_gate": passes,
        "decision": "promote" if passes else "reject",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [V416_SUMMARY, V414_STAGE_SUMMARY, V415_MATRIX]
    missing = [str(p) for p in required if not fs_path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required V417 inputs: " + json.dumps(missing, indent=2))

    v416_summary = read_json(V416_SUMMARY)
    v416_rows = [candidate_decision(row) for row in v416_summary["rows"]]
    v414_rows = read_csv(V414_STAGE_SUMMARY)
    v415_rows = read_csv(V415_MATRIX)

    v414_best = next(row for row in v414_rows if row["state"] == "V414/V366 consolidated CPU teacher")
    adapter_hit_count = sum(int(row["adapter_hit_count"]) for row in v415_rows)
    v414_gain_count = len(v415_rows)

    blocked_recipes = [
        {
            "recipe": "solver_teacher_sft_v413",
            "reason": "H200 smoke failed: checkpoint-2 was 190/315, equation=56, bit=134, truncated=1.",
            "allowed_again": "no, unless CPU gate proves a new adapter/package signal",
        },
        {
            "recipe": "rawstyle_teacher_sft_v416",
            "reason": "H200 weak eval failed: best checkpoint was 191/315, equation=56, bit=135, truncated=1.",
            "allowed_again": "no",
        },
        {
            "recipe": "more_epochs_or_lr_only",
            "reason": "Loss improved in prior jobs without improving equation_transform; decision metric is family ACC.",
            "allowed_again": "no",
        },
        {
            "recipe": "adapter_soup_v291_v382",
            "reason": "V389 soups regressed to 190-191/315 and never moved equation above 56.",
            "allowed_again": "no",
        },
        {
            "recipe": "prompt_sweep_without_new_signal",
            "reason": "V393 tied or regressed the baseline; no_suffix broke bit_manipulation.",
            "allowed_again": "no",
        },
        {
            "recipe": "gpu_job_from_cpu_teacher_only",
            "reason": "V414 teacher is 222/315, but V368/V413/V416 show teacher strength alone does not transfer.",
            "allowed_again": "only if pre-GPU gate shows adapter/package behavior can change",
        },
    ]

    next_gate_contract = {
        "schema_version": "kg1_v417_next_gate_contract_v1",
        "hf_gpu_allowed": False,
        "baseline": BASELINE,
        "promotion_threshold": {
            "total_strictly_greater_than": BASELINE["total"],
            "equation_transform_strictly_greater_than": BASELINE["equation_transform"],
            "bit_manipulation_at_least": BASELINE["bit_manipulation"],
            "truncated_equal": 0,
        },
        "mandatory_pre_gpu_evidence": [
            "new adapter/package-level signal, not only solver/verifier teacher",
            "no direct weak/full training rows",
            "row-level comparison against V291/V290 baseline",
            "kill-switch plan at first checkpoint",
        ],
    }

    manifest = {
        "schema_version": "kg1_v417_transfer_blocker_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "v416_summary_json": str(V416_SUMMARY.relative_to(ROOT)),
        "v416_candidate_decisions": v416_rows,
        "v414_best_cpu_teacher": v414_best,
        "v415_adapter_hit_count_on_v414_gains": adapter_hit_count,
        "v415_v414_gain_rows": v414_gain_count,
        "blocked_recipes": blocked_recipes,
        "hf_gpu_allowed": False,
        "decision": {
            "decision": "block_new_gpu_sft_until_new_cpu_adapter_signal",
            "reason": "V413 and V416 both leave equation at 56 and regress bit/truncation despite stronger CPU teachers.",
            "next_action": "Mine a materially new CPU-gated adapter/package mechanism or formal solver path before any HF spend.",
        },
        "outputs": {
            "blocked_recipes_csv": str((OUT_DIR / "v417_blocked_recipes.csv").relative_to(ROOT)),
            "next_gate_contract_json": str((OUT_DIR / "v417_next_gate_contract.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V417_TRANSFER_BLOCKER_GATE.md").relative_to(ROOT)),
            "manifest_json": str((OUT_DIR / "v417_transfer_blocker_gate_manifest.json").relative_to(ROOT)),
        },
    }

    write_csv(OUT_DIR / "v417_v416_candidate_decisions.csv", v416_rows)
    write_csv(OUT_DIR / "v417_blocked_recipes.csv", blocked_recipes)
    (OUT_DIR / "v417_next_gate_contract.json").write_text(json.dumps(next_gate_contract, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "v417_transfer_blocker_gate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report = f"""# V417 Transfer Blocker Gate

Generated: {manifest['generated_at_utc']}

## Baseline

| Candidate | Total | equation_transform | bit_manipulation | Truncated |
|---|---:|---:|---:|---:|
| {BASELINE['name']} | `{BASELINE['total']}/315` | `{BASELINE['equation_transform']}/155` | `{BASELINE['bit_manipulation']}/160` | `{BASELINE['truncated']}` |

## V416 Result

| Candidate | Total | equation_transform | bit_manipulation | Truncated | Delta | Decision |
|---|---:|---:|---:|---:|---:|---|
"""
    for row in v416_rows:
        report += (
            f"| {row['candidate']} | `{row['total']}/315` | `{row['equation_transform']}/155` | "
            f"`{row['bit_manipulation']}/160` | `{row['truncated']}` | "
            f"`{row['delta_total']} total, {row['delta_equation_transform']} eq, {row['delta_bit_manipulation']} bit` | {row['decision']} |\n"
        )
    report += f"""
## Teacher Versus Adapter Gap

V414 CPU teacher projection reaches `{v414_best['total']}/315`, `equation={v414_best['equation_transform']}/155`, `bit={v414_best['bit_manipulation']}/160`, but V415 found only `{adapter_hit_count}` adapter hits across `{v414_gain_count}` V414 gain rows. The transfer gap is still the bottleneck.

## Decision

`hf_gpu_allowed = false`.

New GPU SFT is blocked until a CPU gate proves a materially new adapter/package-level signal. A stronger solver/verifier teacher alone is not enough, because V368, V413, and V416 all failed to transfer it into weak ACC.

## Allowed Next Work

1. CPU-only row-level mining.
2. Formal rule expansion with abstain/no-loss proofs.
3. Adapter/package behavior probes that do not train on weak/full rows.
4. HF GPU only after the pre-GPU gate can plausibly beat `total>192`, `equation>56`, `bit>=136`, `truncated=0`.
"""
    (OUT_DIR / "V417_TRANSFER_BLOCKER_GATE.md").write_text(report, encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

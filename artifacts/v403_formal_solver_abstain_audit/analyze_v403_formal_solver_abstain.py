#!/usr/bin/env python3
"""V403 CPU audit for formal solver-first KG1 strategy.

This script evaluates current local solvers only as abstaining candidate
generators against the locked V290 checkpoint-6 weak predictions. It does not
train, package, submit, or call GPU services.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402
from src.solvers.bit_manipulation_solver import BitManipulationSolver  # noqa: E402
from src.solvers.equation_solver_v2 import solve_equation_v2  # noqa: E402


INPUT_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = REPO_ROOT / "artifacts/v403_formal_solver_abstain_audit/20260514T_v403_solver_abstain"


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def bit_policy(cot: str) -> str:
    """Return the only submit-safe diagnostic policy for the current bit solver."""
    if cot.startswith("Global binary:") or cot.startswith("Global unary:") or cot.startswith("Ternary:"):
        return "accept_exact_global"
    if "CONSENSUS" in cot:
        return "reject_consensus"
    if "UNSOLVED" in cot:
        return "reject_unsolved"
    return "reject_non_global_per_bit"


def main() -> int:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    rows = read_csv(INPUT_CSV)
    bit_solver = BitManipulationSolver()
    audit_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    family_counts: Counter[tuple[str, str, str]] = Counter()

    for row in rows:
        family = classify_puzzle(row.get("prompt", ""))
        baseline_correct = verify_answer(row.get("answer", ""), row.get("prediction", ""))
        candidate = ""
        proof = ""
        policy = "not_target_family"
        candidate_correct = False
        accepted = False

        if family == "bit_manipulation":
            candidate, cot, solved_bits = bit_solver.solve(row["prompt"])
            candidate = candidate or ""
            policy = bit_policy(cot)
            proof = cot.splitlines()[0] if cot else ""
            candidate_correct = verify_answer(row["answer"], candidate)
            accepted = policy == "accept_exact_global"
            audit_rows.append(
                {
                    "id": row["id"],
                    "family": family,
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "baseline_correct": baseline_correct,
                    "candidate_prediction": candidate,
                    "candidate_correct": candidate_correct,
                    "policy": policy,
                    "accepted": accepted,
                    "delta": int(candidate_correct) - int(baseline_correct) if accepted else 0,
                    "solved_bits": solved_bits,
                    "proof": proof,
                }
            )
            if accepted:
                accepted_rows.append(audit_rows[-1])
        elif family == "equation_transform":
            candidate, cot, solved_independently = solve_equation_v2(row["prompt"], known_answer=None)
            candidate = candidate or ""
            policy = "accept_independent_equation_v2" if solved_independently else "reject_equation_v2_abstain"
            proof = cot.splitlines()[0] if cot else ""
            candidate_correct = verify_answer(row["answer"], candidate)
            accepted = solved_independently
            audit_rows.append(
                {
                    "id": row["id"],
                    "family": family,
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "baseline_correct": baseline_correct,
                    "candidate_prediction": candidate,
                    "candidate_correct": candidate_correct,
                    "policy": policy,
                    "accepted": accepted,
                    "delta": int(candidate_correct) - int(baseline_correct) if accepted else 0,
                    "solved_bits": "",
                    "proof": proof,
                }
            )
            if accepted:
                accepted_rows.append(audit_rows[-1])

        if family in {"bit_manipulation", "equation_transform"}:
            policy_counts[policy] += 1
            family_counts[(family, str(baseline_correct), str(candidate_correct))] += 1

    accepted_delta = sum(int(row["delta"]) for row in accepted_rows)
    accepted_losses = [row for row in accepted_rows if row["baseline_correct"] and not row["candidate_correct"]]
    accepted_gains = [row for row in accepted_rows if not row["baseline_correct"] and row["candidate_correct"]]

    summary = {
        "schema_version": "kg1_v403_formal_solver_abstain_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV),
        "rows": len(rows),
        "target_rows": len(audit_rows),
        "policy_counts": dict(sorted(policy_counts.items())),
        "family_correctness_counts": {
            "|".join(key): value for key, value in sorted(family_counts.items())
        },
        "accepted_rows": len(accepted_rows),
        "accepted_gains": len(accepted_gains),
        "accepted_losses": len(accepted_losses),
        "accepted_delta": accepted_delta,
        "decision": (
            "formal_solver_global_bit_signal_found_but_not_adapter_submit_safe"
            if accepted_gains and not accepted_losses
            else "no_safe_candidate_signal"
        ),
        "next_action": (
            "Use exact global bit rules only as verified trace fixtures; do not use consensus/per-bit fallback. "
            "No GPU job is authorized until a transfer gate proves adapter-only gains."
        ),
    }

    columns = [
        "id",
        "family",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "candidate_prediction",
        "candidate_correct",
        "policy",
        "accepted",
        "delta",
        "solved_bits",
        "proof",
    ]
    write_csv(OUT_DIR / "v403_formal_solver_abstain_audit.csv", audit_rows, columns)
    write_csv(OUT_DIR / "v403_accepted_formal_solver_candidates.csv", accepted_rows, columns)
    write_json(OUT_DIR / "v403_formal_solver_abstain_manifest.json", summary)

    report = [
        "# V403 Formal Solver Abstain Audit",
        "",
        "Baseline: V290 checkpoint-6 weak predictions.",
        "",
        "## Result",
        "",
        f"- Accepted candidates: `{len(accepted_rows)}`",
        f"- Accepted gains: `{len(accepted_gains)}`",
        f"- Accepted losses: `{len(accepted_losses)}`",
        f"- Accepted delta: `{accepted_delta}`",
        "",
        "## Policy",
        "",
        "- Accept only exact byte-global bit rules: global unary, global binary, or exact ternary.",
        "- Reject `CONSENSUS`, `UNSOLVED`, and non-global per-bit fallbacks.",
        "- Reject the old equation v2 parser as a source of weak gains; it abstains/fails on current equation rows.",
        "",
        "## Accepted Gain Rows",
        "",
    ]
    for row in accepted_gains:
        report.append(
            f"- `{row['id']}`: `{row['baseline_prediction']}` -> `{row['candidate_prediction']}` "
            f"via `{row['proof']}`"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "This is a CPU solver signal, not a Kaggle-submitable adapter gain.",
            "",
        ]
    )
    (OUT_DIR / "V403_FORMAL_SOLVER_ABSTAIN_AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

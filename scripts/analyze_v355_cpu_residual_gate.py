#!/usr/bin/env python3
"""V355 CPU residual gate after V352 transfer failure.

This gate deliberately avoids GPU work. It re-checks the next plausible
CPU-only candidates after V350:

* bit stride/bit-pair solver classes;
* current bit solver high-coverage classes;
* equation cryptarithm conflicts that V350 rejected.

Promotion is strict: a rule class must produce at least one gain and zero
losses. Row-specific weak-label cherry-picking is not allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import EXPECTED_ROW_CONTRACT_SHA256, normalize_row, row_contract, sha256_file  # noqa: E402
from run_v296_bit_stride_solver_audit import solve_stride  # noqa: E402
from solvers.bit_manipulation_solver import BitManipulationSolver  # noqa: E402


DEFAULT_V350_INTEGRATED = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_integrated_predictions.csv"
)
DEFAULT_V350_DECISIONS = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_candidate_decisions.csv"
)

DECISION_COLUMNS = [
    "id",
    "family",
    "candidate_source",
    "rule_class",
    "old_prediction",
    "new_prediction",
    "answer",
    "old_correct",
    "new_correct",
    "accepted",
    "rejection_reason",
    "candidate_count",
    "conflict_count",
    "proof",
]

RULE_COLUMNS = [
    "family",
    "candidate_source",
    "rule_class",
    "candidate_rows",
    "changed_rows",
    "gains",
    "losses",
    "new_correct",
    "accepted_rows",
    "accepted",
    "reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def current_prediction(row: dict[str, Any]) -> str:
    return str(row.get("v350_prediction") or row.get("prediction") or row.get("baseline_prediction") or "").strip()


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        fam = str(row["family"])
        ok = verify_answer(str(row["answer"]), str(row.get(prediction_key, "")))
        total["rows"] += 1
        total["correct"] += int(ok)
        family[fam]["rows"] += 1
        family[fam]["correct"] += int(ok)
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "accuracy": float(total["correct"] / total["rows"]) if total["rows"] else 0.0,
        "family": {key: dict(value) for key, value in sorted(family.items())},
    }


def bit_stride_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if str(row["family"]) != "bit_manipulation":
            continue
        old_prediction = current_prediction(row)
        answer, meta = solve_stride(str(row["prompt"]))
        if not answer or answer == old_prediction:
            continue
        default_bits = int(meta.get("default_bits", -1)) if str(meta.get("default_bits", "")).strip() else -1
        vector = meta.get("vector", [])
        vector_text = " ".join(str(item) for item in vector) if isinstance(vector, list) else ""
        decisions.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "candidate_source": "v355_bit_stride",
                "rule_class": f"bit_stride_default_bits_{default_bits}",
                "old_prediction": old_prediction,
                "new_prediction": answer,
                "answer": row["answer"],
                "old_correct": verify_answer(str(row["answer"]), old_prediction),
                "new_correct": verify_answer(str(row["answer"]), answer),
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": 1,
                "conflict_count": 0,
                "proof": vector_text,
            }
        )
    return decisions


def bit_current_solver_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    solver = BitManipulationSolver()
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if str(row["family"]) != "bit_manipulation":
            continue
        old_prediction = current_prediction(row)
        answer, trace, solved = solver.solve(str(row["prompt"]))
        if not answer or answer == old_prediction:
            continue
        mode = "global" if "Global" in str(trace) or "Ternary" in str(trace) else "perbit"
        decisions.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "candidate_source": "v355_current_bit_solver",
                "rule_class": f"current_bit_solver_{mode}_solved_{int(solved)}",
                "old_prediction": old_prediction,
                "new_prediction": str(answer),
                "answer": row["answer"],
                "old_correct": verify_answer(str(row["answer"]), old_prediction),
                "new_correct": verify_answer(str(row["answer"]), str(answer)),
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": 1,
                "conflict_count": 0,
                "proof": str(trace).splitlines()[0] if trace else "",
            }
        )
    return decisions


def equation_conflict_decisions(v350_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry forward V350 verified conflicts as explicitly rejected candidates."""
    out: list[dict[str, Any]] = []
    for row in v350_decisions:
        if str(row.get("family")) != "equation_transform":
            continue
        if not truthy(row.get("new_correct")):
            continue
        if str(row.get("rejection_reason")) != "reject_conflicting_predictions":
            continue
        item = {key: row.get(key, "") for key in DECISION_COLUMNS}
        item["candidate_source"] = "v355_equation_conflict_recheck"
        item["accepted"] = False
        item["rejection_reason"] = "reject_conflicting_predictions_still_ambiguous"
        out.append(item)
    return out


def apply_rule_gates(decisions: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)

    for _key, items in grouped.items():
        gains = [row for row in items if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
        losses = [row for row in items if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
        can_accept = bool(gains) and not losses and all(int(row.get("conflict_count") or 0) == 0 for row in items)
        for row in items:
            if can_accept and row in gains:
                row["accepted"] = True
                row["rejection_reason"] = ""
            elif losses:
                row["rejection_reason"] = "reject_rule_class_has_losses"
            elif int(row.get("conflict_count") or 0) > 0:
                row["rejection_reason"] = row.get("rejection_reason") or "reject_conflicting_predictions"
            elif not gains:
                row["rejection_reason"] = "reject_rule_class_no_gain"
            else:
                row["rejection_reason"] = row.get("rejection_reason") or "reject_not_promotable"


def rule_summary(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (family, source, rule_class), items in sorted(grouped.items()):
        changed = [row for row in items if row.get("new_prediction") != row.get("old_prediction")]
        gains = [row for row in changed if truthy(row.get("new_correct")) and not truthy(row.get("old_correct"))]
        losses = [row for row in changed if truthy(row.get("old_correct")) and not truthy(row.get("new_correct"))]
        accepted = [row for row in items if truthy(row.get("accepted"))]
        rows.append(
            {
                "family": family,
                "candidate_source": source,
                "rule_class": rule_class,
                "candidate_rows": len(items),
                "changed_rows": len(changed),
                "gains": len(gains),
                "losses": len(losses),
                "new_correct": sum(1 for row in items if truthy(row.get("new_correct"))),
                "accepted_rows": len(accepted),
                "accepted": bool(accepted),
                "reason": "accepted_no_loss" if accepted else "rejected_by_gate",
            }
        )
    rows.sort(key=lambda item: (int(item["accepted_rows"]), int(item["gains"]), -int(item["losses"])), reverse=True)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V355 CPU RESIDUAL GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v350_integrated_predictions_csv =", args.v350_integrated_predictions_csv, flush=True)
    print("v350_candidate_decisions_csv =", args.v350_candidate_decisions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(row) for row in read_csv(args.v350_integrated_predictions_csv)]
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    baseline = summarize(rows, "v350_prediction")
    print("v350_baseline_summary =", json.dumps(baseline, sort_keys=True), flush=True)

    v350_decisions = read_csv(args.v350_candidate_decisions_csv)
    decisions = bit_stride_decisions(rows) + bit_current_solver_decisions(rows) + equation_conflict_decisions(v350_decisions)
    apply_rule_gates(decisions)
    accepted = [row for row in decisions if truthy(row.get("accepted"))]
    rules = rule_summary(decisions)
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("accepted_candidate_count =", len(accepted), flush=True)
    print("candidate_rule_summary_top =", json.dumps(rules[:10], sort_keys=True), flush=True)

    # V355 currently audits next candidates only. Do not mutate predictions
    # unless accepted rows exist.
    integrated = []
    accepted_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_id[str(row["id"])].append(row)
    for row in rows:
        item = dict(row)
        item["v355_prediction"] = current_prediction(row)
        item["v355_source_rule"] = ""
        accepted_items = accepted_by_id.get(str(row["id"]), [])
        if len(accepted_items) == 1:
            item["v355_prediction"] = str(accepted_items[0]["new_prediction"])
            item["v355_source_rule"] = str(accepted_items[0]["rule_class"])
        elif len(accepted_items) > 1:
            raise RuntimeError(f"accepted conflict id={row['id']}")
        item["v355_correct"] = verify_answer(str(item["answer"]), str(item["v355_prediction"]))
        integrated.append(item)

    after = summarize(integrated, "v355_prediction")
    gains = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row["v355_prediction"]))
        and not verify_answer(str(row["answer"]), current_prediction(row))
    ]
    losses = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), current_prediction(row))
        and not verify_answer(str(row["answer"]), str(row["v355_prediction"]))
    ]
    eq_after = int(after["family"].get("equation_transform", {}).get("correct", 0))
    bit_after = int(after["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = not losses and (eq_after > args.expected_v350_equation_correct or bit_after > args.expected_v350_bit_correct)
    decision = {
        "decision": "v355_cpu_residual_gate_passed" if gate_pass else "v355_cpu_residual_gate_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            f"v355={after['correct']}/315; equation={eq_after}/155; bit={bit_after}/160; "
            f"gains={len(gains)}; losses={len(losses)}"
        ),
        "next_action": (
            "Build a new transfer dataset with stronger replay only after another CPU gain."
            if gate_pass
            else "Do not launch HF. Continue CPU search with a new equation ambiguity resolver or new bit rule family."
        ),
    }
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "candidate_decisions_csv": args.output_dir / "v355_candidate_decisions.csv",
        "candidate_rules_csv": args.output_dir / "v355_candidate_rules.csv",
        "integrated_predictions_csv": args.output_dir / "v355_integrated_predictions.csv",
        "manifest_json": args.output_dir / "v355_cpu_residual_gate_manifest.json",
    }
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["candidate_rules_csv"], rules, RULE_COLUMNS)
    write_csv(outputs["integrated_predictions_csv"], integrated, list(integrated[0]) if integrated else [])
    manifest = {
        "schema_version": "kg1_v355_cpu_residual_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v350_integrated_predictions_csv": str(args.v350_integrated_predictions_csv),
            "v350_integrated_predictions_sha256": sha256_file(args.v350_integrated_predictions_csv),
            "v350_candidate_decisions_csv": str(args.v350_candidate_decisions_csv),
            "v350_candidate_decisions_sha256": sha256_file(args.v350_candidate_decisions_csv),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "v350_summary": baseline,
        "v355_summary": after,
        "candidate_rule_summary_top": rules[:25],
        "accepted_candidate_count": len(accepted),
        "accepted_candidate_ids": [row["id"] for row in accepted],
        "gain_count": len(gains),
        "gain_ids": [row["id"] for row in gains],
        "loss_count": len(losses),
        "loss_ids": [row["id"] for row in losses],
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V355 CPU RESIDUAL GATE END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {
            "id": "a",
            "family": "bit_manipulation",
            "prompt": "bad",
            "answer": "1",
            "prediction": "0",
            "v350_prediction": "0",
        }
    ]
    assert summarize(rows, "v350_prediction")["correct"] == 0
    print("v355_cpu_residual_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v350-integrated-predictions-csv", type=Path, default=DEFAULT_V350_INTEGRATED)
    parser.add_argument("--v350-candidate-decisions-csv", type=Path, default=DEFAULT_V350_DECISIONS)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v355_cpu_residual_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--expected-v350-equation-correct", type=int, default=63)
    parser.add_argument("--expected-v350-bit-correct", type=int, default=138)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V394 CPU row-level equation_transform inventory.

This script consolidates the V324/V375 CPU evidence against the locked V290
checkpoint-6 weak predictions. It does not train, launch GPU jobs, package, or
submit. Its job is to separate verified CPU solver/verifier gains from actual
adapter-only gains and to compare the current signal against the prior V390
gate before any further spend.
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
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    EXPECTED_ROW_CONTRACT_SHA256,
    classify_subtype,
    normalize_row,
    parse_alice_prompt,
    row_contract,
    sha256_file,
)


DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
)
DEFAULT_V324_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v324_on_v290_checkpoint6/"
    / "v324_equation_expanded_solver_accepted_candidates.csv"
)
DEFAULT_V324_AUDIT_CSV = (
    REPO_ROOT
    / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v324_on_v290_checkpoint6/"
    / "v324_equation_expanded_solver_audit.csv"
)
DEFAULT_V375_RESIDUAL_CSV = (
    REPO_ROOT
    / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v375_residual_after_v324_on_v290/"
    / "v375_equation_residual_rows.csv"
)
DEFAULT_PREVIOUS_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v390_cpu_v324_equation_gate/20260514T193847Z/"
    / "v324_equation_expanded_solver_accepted_candidates.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/v394_inventory"


INVENTORY_COLUMNS = [
    "id",
    "subtype",
    "baseline_prediction",
    "answer",
    "baseline_correct",
    "accepted_cpu_prediction",
    "accepted_cpu_correct",
    "accepted_rule_class",
    "accepted_candidate_source",
    "accepted_proof",
    "cluster_key",
    "priority_reason",
    "status",
    "query",
    "examples_count",
]
SUMMARY_COLUMNS = ["family", "rows", "baseline_correct", "projected_correct", "delta"]
COMPARISON_COLUMNS = ["metric", "current", "previous", "delta", "decision"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_accepted(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    accepted: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        row_id = str(row.get("id", ""))
        if not row_id:
            continue
        if truthy(row.get("verified_by_weak_label")) and not truthy(row.get("incorrect_by_weak_label")):
            accepted[row_id] = row
    return accepted


def residual_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {str(row.get("id", "")): row for row in read_csv(path)}


def family_summary(rows: list[dict[str, Any]], accepted: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        base_prediction = str(row.get("prediction", ""))
        projected_prediction = base_prediction
        if family == "equation_transform" and str(row["id"]) in accepted:
            projected_prediction = str(accepted[str(row["id"])].get("prediction", base_prediction))
        counts[family]["rows"] += 1
        counts[family]["baseline_correct"] += int(verify_answer(str(row["answer"]), base_prediction))
        counts[family]["projected_correct"] += int(verify_answer(str(row["answer"]), projected_prediction))
    out = []
    for family, counter in sorted(counts.items()):
        out.append(
            {
                "family": family,
                "rows": int(counter["rows"]),
                "baseline_correct": int(counter["baseline_correct"]),
                "projected_correct": int(counter["projected_correct"]),
                "delta": int(counter["projected_correct"] - counter["baseline_correct"]),
            }
        )
    return out


def build_inventory(
    rows: list[dict[str, Any]],
    accepted: dict[str, dict[str, str]],
    residuals: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for row in rows:
        if row["family"] != "equation_transform":
            continue
        base_prediction = str(row.get("prediction", ""))
        baseline_correct = verify_answer(str(row["answer"]), base_prediction)
        if baseline_correct:
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        subtype = "parse_" + parse_status if parse_status != "ok" else classify_subtype(examples, query)
        row_id = str(row["id"])
        accepted_row = accepted.get(row_id, {})
        accepted_prediction = str(accepted_row.get("prediction", ""))
        accepted_correct = bool(accepted_prediction) and verify_answer(str(row["answer"]), accepted_prediction)
        residual = residuals.get(row_id, {})
        if accepted_correct:
            status = "cpu_solver_verified_gain_not_adapter_gain"
        elif accepted_prediction:
            status = "cpu_candidate_failed_verification"
        else:
            status = "unresolved_equation_miss"
        inventory.append(
            {
                "id": row_id,
                "subtype": subtype,
                "baseline_prediction": base_prediction,
                "answer": row["answer"],
                "baseline_correct": baseline_correct,
                "accepted_cpu_prediction": accepted_prediction,
                "accepted_cpu_correct": accepted_correct,
                "accepted_rule_class": accepted_row.get("rule_class", ""),
                "accepted_candidate_source": accepted_row.get("candidate_source", ""),
                "accepted_proof": accepted_row.get("proof", ""),
                "cluster_key": residual.get("cluster_key", ""),
                "priority_reason": residual.get("priority_reason", ""),
                "status": status,
                "query": query,
                "examples_count": len(examples),
            }
        )
    return inventory


def compare_with_previous(
    current: dict[str, dict[str, str]],
    previous: dict[str, dict[str, str]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_ids = set(current)
    previous_ids = set(previous)
    summary_by_family = {row["family"]: row for row in summary_rows}
    equation = summary_by_family.get("equation_transform", {})
    bit = summary_by_family.get("bit_manipulation", {})
    return [
        {
            "metric": "accepted_cpu_gain_ids",
            "current": ",".join(sorted(current_ids)),
            "previous": ",".join(sorted(previous_ids)),
            "delta": len(current_ids - previous_ids),
            "decision": "no_new_cpu_gain" if current_ids == previous_ids else "changed_cpu_gain_set",
        },
        {
            "metric": "projected_equation_correct",
            "current": equation.get("projected_correct", ""),
            "previous": 62,
            "delta": int(equation.get("projected_correct", 0)) - 62,
            "decision": "no_new_equation_projection"
            if int(equation.get("projected_correct", 0)) == 62
            else "projection_changed",
        },
        {
            "metric": "bit_guardrail",
            "current": bit.get("projected_correct", ""),
            "previous": 136,
            "delta": int(bit.get("projected_correct", 0)) - 136,
            "decision": "bit_guardrail_ok" if int(bit.get("projected_correct", 0)) >= 136 else "bit_regression",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V394 EQUATION ROW LEVEL INVENTORY START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_predictions_csv =", args.baseline_predictions_csv, flush=True)
    print("v324_accepted_candidates_csv =", args.v324_accepted_candidates_csv, flush=True)
    print("v375_residual_rows_csv =", args.v375_residual_rows_csv, flush=True)
    print("previous_accepted_candidates_csv =", args.previous_accepted_candidates_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [normalize_row(row) | row for row in read_csv(args.baseline_predictions_csv)]
    observed_contract = row_contract([normalize_row(row) for row in rows])
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    accepted = load_accepted(args.v324_accepted_candidates_csv)
    previous = load_accepted(args.previous_accepted_candidates_csv)
    residuals = residual_by_id(args.v375_residual_rows_csv)
    inventory = build_inventory(rows, accepted, residuals)
    summary_rows = family_summary(rows, accepted)
    comparison_rows = compare_with_previous(accepted, previous, summary_rows)

    status_counts = Counter(str(row["status"]) for row in inventory)
    subtype_counts = Counter(str(row["subtype"]) for row in inventory)
    accepted_ids = sorted(accepted)
    new_ids = sorted(set(accepted) - set(previous))

    projected_by_family = {row["family"]: row for row in summary_rows}
    equation_projected = int(projected_by_family["equation_transform"]["projected_correct"])
    bit_projected = int(projected_by_family["bit_manipulation"]["projected_correct"])
    if new_ids:
        decision = {
            "decision": "new_cpu_equation_signal_found",
            "reason": f"new_accepted_ids={new_ids}; projected_equation={equation_projected}; bit={bit_projected}",
            "next_action": "Build a tiny transfer probe only for new rows before any GPU spend.",
        }
    else:
        decision = {
            "decision": "reconfirmed_existing_cpu_signal_no_new_gpu_authorization",
            "reason": (
                f"accepted_ids_match_previous_v390; projected_equation={equation_projected}; "
                f"bit={bit_projected}; V391 already showed this signal did not transfer to LoRA"
            ),
            "next_action": "Do not launch HF training from the same six numeric rows. Expand symbolic DSL on unresolved rows.",
        }

    outputs = {
        "inventory_csv": args.output_dir / "v394_equation_row_level_inventory.csv",
        "family_summary_csv": args.output_dir / "v394_family_projection_summary.csv",
        "comparison_csv": args.output_dir / "v394_vs_v390_comparison.csv",
        "manifest_json": args.output_dir / "v394_equation_row_level_inventory_manifest.json",
    }
    write_csv(outputs["inventory_csv"], inventory, INVENTORY_COLUMNS)
    write_csv(outputs["family_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["comparison_csv"], comparison_rows, COMPARISON_COLUMNS)
    manifest = {
        "schema_version": "kg1_v394_equation_row_level_inventory_v1",
        "generated_at_utc": utc_now(),
        "baseline_predictions_csv": str(args.baseline_predictions_csv),
        "baseline_predictions_sha256": sha256_file(args.baseline_predictions_csv),
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "equation_miss_rows": len(inventory),
        "accepted_cpu_gain_count": len(accepted_ids),
        "accepted_cpu_gain_ids": accepted_ids,
        "new_cpu_gain_vs_v390_count": len(new_ids),
        "new_cpu_gain_vs_v390_ids": new_ids,
        "status_counts": dict(status_counts),
        "subtype_counts": dict(subtype_counts),
        "family_summary": summary_rows,
        "comparison": comparison_rows,
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("equation_miss_rows =", len(inventory), flush=True)
    print("accepted_cpu_gain_count =", len(accepted_ids), flush=True)
    print("accepted_cpu_gain_ids =", json.dumps(accepted_ids), flush=True)
    print("new_cpu_gain_vs_v390_ids =", json.dumps(new_ids), flush=True)
    print("status_counts =", json.dumps(dict(status_counts), sort_keys=True), flush=True)
    print("subtype_counts =", json.dumps(dict(subtype_counts), sort_keys=True), flush=True)
    print("family_summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("comparison =", json.dumps(comparison_rows, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V394 EQUATION ROW LEVEL INVENTORY END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--v324-accepted-candidates-csv", type=Path, default=DEFAULT_V324_ACCEPTED_CSV)
    parser.add_argument("--v324-audit-csv", type=Path, default=DEFAULT_V324_AUDIT_CSV)
    parser.add_argument("--v375-residual-rows-csv", type=Path, default=DEFAULT_V375_RESIDUAL_CSV)
    parser.add_argument("--previous-accepted-candidates-csv", type=Path, default=DEFAULT_PREVIOUS_ACCEPTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V497 CPU residual transfer audit.

This audit is intentionally CPU-only. It compares the locked V290 checkpoint-6
weak predictions, the V324/V475 CPU equation solver signal, and the V496 weak
eval diff. The goal is to decide whether another paid GPU job is justified.

Weak labels are used here only for auditing/gating. This script does not create
training data, run inference, package a submission, or submit to Kaggle.
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
)


DEFAULT_BASELINE_CSV = (
    REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
)
DEFAULT_V324_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v475_cpu_equation_solver_regate/20260516T_cpu_v324_current_baseline/"
    / "v324_equation_expanded_solver_accepted_candidates.csv"
)
DEFAULT_V324_MANIFEST_JSON = (
    REPO_ROOT
    / "artifacts/v475_cpu_equation_solver_regate/20260516T_cpu_v324_current_baseline/"
    / "v324_equation_expanded_solver_manifest.json"
)
DEFAULT_V394_RESIDUAL_ROWS_CSV = (
    REPO_ROOT
    / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/"
    / "v375_residual_after_v324_on_v290/v375_equation_residual_rows.csv"
)
DEFAULT_V496_CHANGED_CSV = (
    REPO_ROOT
    / "artifacts/v496_hf_h200_v495_weak_eval_launch/metric_audit/"
    / "v496_changed_vs_v290_checkpoint6.csv"
)
DEFAULT_V496_GAINS_CSV = (
    REPO_ROOT
    / "artifacts/v496_hf_h200_v495_weak_eval_launch/metric_audit/"
    / "v496_gains_vs_v290_checkpoint6.csv"
)
DEFAULT_V496_LOSSES_CSV = (
    REPO_ROOT
    / "artifacts/v496_hf_h200_v495_weak_eval_launch/metric_audit/"
    / "v496_losses_vs_v290_checkpoint6.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v497_cpu_residual_transfer_audit/20260516T_v497_cpu_audit"


EQUATION_COLUMNS = [
    "id",
    "subtype",
    "answer",
    "baseline_prediction",
    "query",
    "examples_count",
    "cluster_key",
    "priority_reason",
    "v324_prediction",
    "v324_rule_class",
    "v324_candidate_source",
    "v324_correct",
    "v496_prediction",
    "v496_correct",
    "v496_changed",
    "transfer_status",
    "next_action",
]
CLUSTER_COLUMNS = [
    "cluster_key",
    "subtype",
    "rows",
    "v324_verified_gains",
    "v496_verified_gains",
    "still_unresolved",
    "priority_reason_counts",
]
BIT_COLUMNS = [
    "id",
    "answer",
    "base_prediction",
    "v496_prediction",
    "base_correct",
    "v496_correct",
    "binary_shape_ok",
    "failure_type",
    "guardrail_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}


def load_v324_accepted(path: Path) -> dict[str, dict[str, str]]:
    accepted: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            continue
        if truthy(row.get("verified_by_weak_label")) and not truthy(row.get("incorrect_by_weak_label")):
            accepted[row_id] = row
    return accepted


def family_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        counts[family]["rows"] += 1
        counts[family]["correct"] += int(verify_answer(str(row["answer"]), str(row.get("prediction", ""))))
        counts[family]["truncated"] += int(bool(row.get("truncated_bool", False)))
    return {family: dict(counter) for family, counter in sorted(counts.items())}


def binary_shape_ok(answer: str, prediction: str) -> bool:
    return len(prediction) == len(answer) and set(answer) <= {"0", "1"} and set(prediction) <= {"0", "1"}


def classify_transfer_status(v324_correct: bool, v496_correct: bool, v496_changed: bool) -> tuple[str, str]:
    if v324_correct and v496_correct:
        return "cpu_signal_transferred", "freeze_as_evidence_no_new_gpu"
    if v324_correct and not v496_correct:
        return "cpu_signal_not_transferred", "needs_teacher_trace_or_different_objective_before_gpu"
    if (not v324_correct) and v496_correct:
        return "v496_unique_equation_gain", "mine_prompt_pattern_but_do_not_train_from_weak_label"
    if v496_changed:
        return "changed_but_still_wrong", "inspect_symbolic_format_regression"
    return "unresolved_equation_miss", "cluster_for_new_cpu_solver_only"


def build_equation_rows(
    baseline_rows: list[dict[str, Any]],
    residual_by_id: dict[str, dict[str, str]],
    v324_by_id: dict[str, dict[str, str]],
    v496_changed_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in baseline_rows:
        if row["family"] != "equation_transform":
            continue
        if verify_answer(str(row["answer"]), str(row.get("prediction", ""))):
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        subtype = "parse_" + parse_status if parse_status != "ok" else classify_subtype(examples, query)
        row_id = str(row["id"])
        residual = residual_by_id.get(row_id, {})
        v324 = v324_by_id.get(row_id, {})
        v496 = v496_changed_by_id.get(row_id, {})
        v324_prediction = str(v324.get("prediction", ""))
        v496_prediction = str(v496.get("v496_prediction", ""))
        v324_correct = bool(v324_prediction) and verify_answer(str(row["answer"]), v324_prediction)
        v496_correct = truthy(v496.get("v496_correct"))
        v496_changed = bool(v496)
        transfer_status, next_action = classify_transfer_status(v324_correct, v496_correct, v496_changed)
        out.append(
            {
                "id": row_id,
                "subtype": subtype,
                "answer": row["answer"],
                "baseline_prediction": row.get("prediction", ""),
                "query": query,
                "examples_count": len(examples),
                "cluster_key": residual.get("cluster_key", ""),
                "priority_reason": residual.get("priority_reason", ""),
                "v324_prediction": v324_prediction,
                "v324_rule_class": v324.get("rule_class", ""),
                "v324_candidate_source": v324.get("candidate_source", ""),
                "v324_correct": v324_correct,
                "v496_prediction": v496_prediction,
                "v496_correct": v496_correct,
                "v496_changed": v496_changed,
                "transfer_status": transfer_status,
                "next_action": next_action,
            }
        )
    return out


def build_cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("cluster_key") or f"{row.get('subtype')}|unclustered")
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        priority_counts = Counter(str(item.get("priority_reason", "")) for item in items)
        summary.append(
            {
                "cluster_key": key,
                "subtype": Counter(str(item.get("subtype", "")) for item in items).most_common(1)[0][0],
                "rows": len(items),
                "v324_verified_gains": sum(int(truthy(item.get("v324_correct"))) for item in items),
                "v496_verified_gains": sum(int(truthy(item.get("v496_correct"))) for item in items),
                "still_unresolved": sum(
                    int(item.get("transfer_status") in {"unresolved_equation_miss", "changed_but_still_wrong"})
                    for item in items
                ),
                "priority_reason_counts": json.dumps(dict(priority_counts), sort_keys=True),
            }
        )
    return summary


def build_bit_guardrail_rows(loss_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in loss_rows:
        if str(row.get("family_norm", "")) != "bit_manipulation":
            continue
        answer = str(row.get("answer", ""))
        prediction = str(row.get("v496_prediction", ""))
        shape_ok = binary_shape_ok(answer, prediction)
        failure_type = "binary_wrong_value" if shape_ok else "non_binary_or_wrong_length"
        out.append(
            {
                "id": row.get("id", ""),
                "answer": answer,
                "base_prediction": row.get("base_prediction", ""),
                "v496_prediction": prediction,
                "base_correct": row.get("base_correct", ""),
                "v496_correct": row.get("v496_correct", ""),
                "binary_shape_ok": shape_ok,
                "failure_type": failure_type,
                "guardrail_action": "block_candidate_family_until_bit_replay_preserves_this_row",
            }
        )
    return out


def run_analysis(args: argparse.Namespace) -> None:
    print("=== V497 CPU RESIDUAL TRANSFER AUDIT START ===", flush=True)
    print("baseline_csv =", args.baseline_csv, flush=True)
    print("v324_accepted_csv =", args.v324_accepted_csv, flush=True)
    print("v496_changed_csv =", args.v496_changed_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    output_dir = Path(args.output_dir)
    baseline_rows = [normalize_row(row) for row in read_csv(Path(args.baseline_csv))]
    observed_contract = row_contract(baseline_rows)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    v324_by_id = load_v324_accepted(Path(args.v324_accepted_csv))
    residual_by_id = by_id(read_csv(Path(args.v394_residual_rows_csv)))
    v496_changed_by_id = by_id(read_csv(Path(args.v496_changed_csv)))
    v496_gain_by_id = by_id(read_csv(Path(args.v496_gains_csv)))
    v496_loss_rows = read_csv(Path(args.v496_losses_csv))
    v324_manifest = read_json(Path(args.v324_manifest_json)) if Path(args.v324_manifest_json).exists() else {}

    equation_rows = build_equation_rows(baseline_rows, residual_by_id, v324_by_id, v496_changed_by_id)
    cluster_summary = build_cluster_summary(equation_rows)
    bit_guardrail_rows = build_bit_guardrail_rows(v496_loss_rows)
    counts = family_counts(baseline_rows)
    baseline_total = sum(item["correct"] for item in counts.values())
    v324_equation_gain = sum(int(truthy(row.get("v324_correct"))) for row in equation_rows)
    v496_equation_gain = sum(
        int(row.get("family_norm") == "equation_transform" and truthy(row.get("v496_correct")))
        for row in v496_gain_by_id.values()
    )
    v496_bit_losses = len(bit_guardrail_rows)
    v496_total_projection = baseline_total + len(v496_gain_by_id) - len(v496_loss_rows)
    v324_total_projection = baseline_total + v324_equation_gain

    decision: dict[str, Any]
    if v496_total_projection <= baseline_total or v496_bit_losses:
        decision = {
            "decision": "do_not_promote_v496_or_repeat_h200_sft",
            "reason": (
                f"v496_total_projection={v496_total_projection}; baseline_total={baseline_total}; "
                f"v496_bit_losses={v496_bit_losses}; v324_cpu_gain={v324_equation_gain}"
            ),
            "next_action": "Implement a CPU teacher/verifier that explains equation gains without bit regression before any paid GPU job.",
        }
    elif v324_total_projection > baseline_total and v496_bit_losses == 0:
        decision = {
            "decision": "cpu_signal_exists_but_needs_transfer_design",
            "reason": f"v324_total_projection={v324_total_projection}; v496 did not prove transfer",
            "next_action": "Build short deterministic traces and run a cheap adapter smoke only after bit guardrail passes.",
        }
    else:
        decision = {
            "decision": "no_actionable_signal",
            "reason": "No verified gain survived the audit.",
            "next_action": "Stop GPU spend and expand CPU solver only.",
        }

    outputs = {
        "equation_residual_transfer_audit_csv": str(output_dir / "v497_equation_residual_transfer_audit.csv"),
        "equation_cluster_summary_csv": str(output_dir / "v497_equation_cluster_summary.csv"),
        "bit_guardrail_failures_csv": str(output_dir / "v497_bit_guardrail_failures.csv"),
        "manifest_json": str(output_dir / "v497_cpu_residual_transfer_audit_manifest.json"),
        "summary_md": str(output_dir / "KG1_V497_CPU_RESIDUAL_TRANSFER_AUDIT_2026_05_16.md"),
    }
    manifest = {
        "schema_version": "v497_cpu_residual_transfer_audit_v1",
        "generated_at_utc": utc_now(),
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "baseline_counts": counts,
        "baseline_total_correct": baseline_total,
        "equation_miss_rows": len(equation_rows),
        "v324_verified_equation_gain": v324_equation_gain,
        "v324_projected_total_correct": v324_total_projection,
        "v324_manifest_decision": v324_manifest.get("decision", {}),
        "v496_changed_rows": len(v496_changed_by_id),
        "v496_verified_equation_gain": v496_equation_gain,
        "v496_loss_rows": len(v496_loss_rows),
        "v496_bit_loss_rows": v496_bit_losses,
        "v496_total_projection_from_diff": v496_total_projection,
        "decision": decision,
        "outputs": outputs,
    }

    write_csv(Path(outputs["equation_residual_transfer_audit_csv"]), equation_rows, EQUATION_COLUMNS)
    write_csv(Path(outputs["equation_cluster_summary_csv"]), cluster_summary, CLUSTER_COLUMNS)
    write_csv(Path(outputs["bit_guardrail_failures_csv"]), bit_guardrail_rows, BIT_COLUMNS)
    write_json(Path(outputs["manifest_json"]), manifest)
    write_summary_md(Path(outputs["summary_md"]), manifest, cluster_summary, bit_guardrail_rows)

    print("baseline_total_correct =", baseline_total, flush=True)
    print("equation_miss_rows =", len(equation_rows), flush=True)
    print("v324_verified_equation_gain =", v324_equation_gain, flush=True)
    print("v496_verified_equation_gain =", v496_equation_gain, flush=True)
    print("v496_bit_loss_rows =", v496_bit_losses, flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(outputs, indent=2, sort_keys=True), flush=True)
    print("=== V497 CPU RESIDUAL TRANSFER AUDIT END ===", flush=True)


def write_summary_md(
    path: Path,
    manifest: dict[str, Any],
    cluster_summary: list[dict[str, Any]],
    bit_guardrail_rows: list[dict[str, Any]],
) -> None:
    top_clusters = cluster_summary[:8]
    lines = [
        "# KG1 V497 CPU Residual Transfer Audit",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- Decision: `{manifest['decision']['decision']}`",
        f"- Reason: {manifest['decision']['reason']}",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline total correct | {manifest['baseline_total_correct']} |",
        f"| Equation miss rows | {manifest['equation_miss_rows']} |",
        f"| V324 verified equation gain | {manifest['v324_verified_equation_gain']} |",
        f"| V324 projected total | {manifest['v324_projected_total_correct']} |",
        f"| V496 changed rows | {manifest['v496_changed_rows']} |",
        f"| V496 verified equation gain | {manifest['v496_verified_equation_gain']} |",
        f"| V496 bit loss rows | {manifest['v496_bit_loss_rows']} |",
        f"| V496 total projection from diff | {manifest['v496_total_projection_from_diff']} |",
        "",
        "## Top Residual Equation Clusters",
        "",
        "| Cluster | Rows | V324 gains | V496 gains | Unresolved |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_clusters:
        lines.append(
            f"| `{row['cluster_key']}` | {row['rows']} | {row['v324_verified_gains']} | "
            f"{row['v496_verified_gains']} | {row['still_unresolved']} |"
        )
    lines.extend(["", "## Bit Guardrail Failures", ""])
    if bit_guardrail_rows:
        lines.extend(["| id | answer | V496 prediction | failure |", "|---|---|---|---|"])
        for row in bit_guardrail_rows:
            lines.append(
                f"| `{row['id']}` | `{row['answer']}` | `{row['v496_prediction']}` | `{row['failure_type']}` |"
            )
    else:
        lines.append("No bit guardrail failures in the compared diff.")
    lines.extend(
        [
            "",
            "## Implementation Consequence",
            "",
            "Do not launch another H200 SFT run from V475/V390/V326 directly. The next executable step is a CPU teacher/verifier redesign that explains at least four equation misses while preserving the exact bit guardrail before any paid GPU job.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    assert binary_shape_ok("10101010", "10101010")
    assert not binary_shape_ok("10101010", "1010101")
    assert not binary_shape_ok("10101010", "2")
    assert classify_transfer_status(True, False, False)[0] == "cpu_signal_not_transferred"
    assert classify_transfer_status(False, True, True)[0] == "v496_unique_equation_gain"
    print("v497_cpu_residual_transfer_audit_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--v324-accepted-csv", type=Path, default=DEFAULT_V324_ACCEPTED_CSV)
    parser.add_argument("--v324-manifest-json", type=Path, default=DEFAULT_V324_MANIFEST_JSON)
    parser.add_argument("--v394-residual-rows-csv", type=Path, default=DEFAULT_V394_RESIDUAL_ROWS_CSV)
    parser.add_argument("--v496-changed-csv", type=Path, default=DEFAULT_V496_CHANGED_CSV)
    parser.add_argument("--v496-gains-csv", type=Path, default=DEFAULT_V496_GAINS_CSV)
    parser.add_argument("--v496-losses-csv", type=Path, default=DEFAULT_V496_LOSSES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

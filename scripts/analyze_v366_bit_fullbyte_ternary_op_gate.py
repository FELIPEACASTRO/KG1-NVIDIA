#!/usr/bin/env python3
"""V366 CPU-only full-byte ternary bit operator gate.

V365 showed that free per-bit boolean grammar is unsafe. V366 tests a stricter
full-byte route: one ternary byte-level expression must match all examples, and
promotion is grouped by ternary operator family. A family is accepted only when
it has at least one gain and zero losses on the weak contract.

This is a teacher/verifier gate only. It does not train, launch HF, package, or
submit. Any accepted rows must still be transferred to LoRA and pass adapter-only
weak/full gates before Kaggle submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    normalize_row,
    row_contract,
    sha256_file,
)
from run_v300_bit_fullbyte_grammar_audit import solve_fullbyte  # noqa: E402


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_integrated_predictions.csv"
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
    for key in ("v366_prediction", "v357_prediction", "v350_prediction", "prediction", "baseline_prediction"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def normalize_prediction_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        base = normalize_row(row)
        base.update(row)
        base["current_prediction"] = current_prediction(base)
        base["current_correct"] = verify_answer(str(base.get("answer", "")), str(base["current_prediction"]))
        normalized.append(base)
    return normalized


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family_name = str(row["family"])
        ok = verify_answer(str(row["answer"]), str(row.get(prediction_key, "")))
        total["rows"] += 1
        total["correct"] += int(ok)
        family[family_name]["rows"] += 1
        family[family_name]["correct"] += int(ok)
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "accuracy": float(total["correct"] / total["rows"]) if total["rows"] else 0.0,
        "family": {key: dict(value) for key, value in sorted(family.items())},
    }


def ternary_op_from_expr(expr: str) -> str:
    match = re.match(r"^([A-Z0-9_]+)\(", str(expr))
    return match.group(1) if match else "unknown"


def build_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    bit_rows = [row for row in rows if str(row["family"]) == "bit_manipulation"]
    decisions: list[dict[str, Any]] = []
    for index, row in enumerate(bit_rows, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(bit_rows):
            print(f"v366_bit_fullbyte_progress = {index}/{len(bit_rows)} decisions={len(decisions)}", flush=True)
        candidate, meta = solve_fullbyte(str(row["prompt"]), max_level=3, ternary_allowlist=None)
        if meta.get("status") != "ok" or meta.get("level") != "ternary":
            continue
        old_prediction = current_prediction(row)
        if not candidate or candidate == old_prediction:
            continue
        expr = str(meta.get("expr", ""))
        op_name = ternary_op_from_expr(expr)
        old_correct = verify_answer(str(row["answer"]), old_prediction)
        new_correct = verify_answer(str(row["answer"]), str(candidate))
        decisions.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "candidate_source": "v366_bit_fullbyte_ternary",
                "rule_class": f"bit_fullbyte_ternary_op_{op_name}",
                "old_prediction": old_prediction,
                "new_prediction": candidate,
                "answer": row["answer"],
                "old_correct": old_correct,
                "new_correct": new_correct,
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": 1,
                "conflict_count": 0,
                "proof": expr,
            }
        )
    return decisions


def apply_rule_gates(decisions: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)

    for _key, items in grouped.items():
        gains = [row for row in items if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
        losses = [row for row in items if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
        can_accept = bool(gains) and not losses
        for row in items:
            if can_accept and row in gains:
                row["accepted"] = True
                row["rejection_reason"] = ""
            elif losses:
                row["rejection_reason"] = "reject_rule_class_has_losses"
            elif not gains:
                row["rejection_reason"] = "reject_rule_class_no_gain"
            else:
                row["rejection_reason"] = "reject_not_promotable"


def rule_summary(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (family, source, rule_class), items in sorted(grouped.items()):
        gains = [row for row in items if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
        losses = [row for row in items if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
        accepted = [row for row in items if truthy(row["accepted"])]
        rows.append(
            {
                "family": family,
                "candidate_source": source,
                "rule_class": rule_class,
                "candidate_rows": len(items),
                "changed_rows": len(items),
                "gains": len(gains),
                "losses": len(losses),
                "new_correct": sum(1 for row in items if truthy(row["new_correct"])),
                "accepted_rows": len(accepted),
                "accepted": bool(accepted),
                "reason": "accepted_no_loss" if accepted else "rejected_by_gate",
            }
        )
    rows.sort(key=lambda item: (int(item["accepted_rows"]), int(item["gains"]), -int(item["losses"])), reverse=True)
    return rows


def integrate(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id = {str(row["id"]): row for row in decisions if truthy(row["accepted"])}
    integrated = []
    for row in rows:
        item = dict(row)
        item["v366_prediction"] = current_prediction(row)
        item["v366_source_rule"] = ""
        accepted = accepted_by_id.get(str(row["id"]))
        if accepted:
            item["v366_prediction"] = str(accepted["new_prediction"])
            item["v366_source_rule"] = str(accepted["rule_class"])
        item["v366_correct"] = verify_answer(str(item["answer"]), str(item["v366_prediction"]))
        item["prediction"] = item["v366_prediction"]
        integrated.append(item)
    return integrated


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V366 BIT FULLBYTE TERNARY OP GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("input_predictions_csv =", args.input_predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_prediction_rows(read_csv(args.input_predictions_csv))
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    baseline = summarize(rows, "current_prediction")
    print("current_summary =", json.dumps(baseline, sort_keys=True), flush=True)

    decisions = build_decisions(rows, args)
    apply_rule_gates(decisions)
    rules = rule_summary(decisions)
    accepted = [row for row in decisions if truthy(row["accepted"])]
    candidate_gains = [row for row in decisions if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
    candidate_losses = [row for row in decisions if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
    integrated = integrate(rows, decisions)
    after = summarize(integrated, "v366_prediction")
    accepted_gains = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row["v366_prediction"]))
        and not verify_answer(str(row["answer"]), str(row.get("current_prediction", "")))
    ]
    accepted_losses = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row.get("current_prediction", "")))
        and not verify_answer(str(row["answer"]), str(row["v366_prediction"]))
    ]

    eq_before = int(baseline["family"].get("equation_transform", {}).get("correct", 0))
    bit_before = int(baseline["family"].get("bit_manipulation", {}).get("correct", 0))
    eq_after = int(after["family"].get("equation_transform", {}).get("correct", 0))
    bit_after = int(after["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = bool(accepted) and not accepted_losses and bit_after > bit_before
    decision = {
        "decision": "v366_cpu_gate_passed" if gate_pass else "v366_cpu_gate_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            f"v366={after['correct']}/315; equation={eq_after}/155; bit={bit_after}/160; "
            f"candidate_changes={len(decisions)}; candidate_gains={len(candidate_gains)}; "
            f"candidate_losses={len(candidate_losses)}; accepted={len(accepted)}; "
            f"accepted_gains={len(accepted_gains)}; accepted_losses={len(accepted_losses)}"
        ),
        "next_action": (
            "Build V367 transfer dataset from accepted MAJ3/CHO rows with strong replay and hard negatives."
            if gate_pass
            else "Do not launch HF. Continue CPU search."
        ),
    }
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("accepted_candidate_count =", len(accepted), flush=True)
    print("candidate_rule_summary =", json.dumps(rules, sort_keys=True), flush=True)
    print("v366_summary =", json.dumps(after, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "candidate_decisions_csv": args.output_dir / "v366_candidate_decisions.csv",
        "candidate_rules_csv": args.output_dir / "v366_candidate_rules.csv",
        "integrated_predictions_csv": args.output_dir / "v366_integrated_predictions.csv",
        "manifest_json": args.output_dir / "v366_bit_fullbyte_ternary_op_gate_manifest.json",
    }
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["candidate_rules_csv"], rules, RULE_COLUMNS)
    write_csv(outputs["integrated_predictions_csv"], integrated, list(integrated[0]) if integrated else [])
    manifest = {
        "schema_version": "kg1_v366_bit_fullbyte_ternary_op_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "input_predictions_csv": str(args.input_predictions_csv),
            "input_predictions_sha256": sha256_file(args.input_predictions_csv),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "current_summary": baseline,
        "v366_summary": after,
        "candidate_rule_summary": rules,
        "accepted_candidate_count": len(accepted),
        "accepted_candidate_ids": [row["id"] for row in accepted],
        "candidate_change_count": len(decisions),
        "candidate_gain_count": len(candidate_gains),
        "candidate_gain_ids": [row["id"] for row in candidate_gains],
        "candidate_loss_count": len(candidate_losses),
        "candidate_loss_ids": [row["id"] for row in candidate_losses],
        "accepted_gain_count": len(accepted_gains),
        "accepted_gain_ids": [row["id"] for row in accepted_gains],
        "accepted_loss_count": len(accepted_losses),
        "accepted_loss_ids": [row["id"] for row in accepted_losses],
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V366 BIT FULLBYTE TERNARY OP GATE END ===", flush=True)
    return manifest


def self_test() -> None:
    exprs = ["MAJ3(ROL5,SHL1,SHR4)", "CHO(SHL2,SHR3,ROL1)", "AND_OR(ROL3,SHL7,SHR3)"]
    ops = [ternary_op_from_expr(expr) for expr in exprs]
    if ops != ["MAJ3", "CHO", "AND_OR"]:
        raise AssertionError(ops)
    print("v366_bit_fullbyte_ternary_op_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--progress-every", type=int, default=20)
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

#!/usr/bin/env python3
"""V350 CPU residual no-loss gate.

This script is CPU-only. It does not train, package, submit, or run GPU
inference. It takes the current V343 integrated weak predictions, audits new
label-free candidate rules against the full weak row contract, and promotes a
candidate only when it improves ACC with zero losses.
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
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    EXPECTED_ROW_CONTRACT_SHA256,
    build_audit_row,
    classify_subtype,
    normalize_row,
    numeric_candidate,
    parse_alice_prompt,
    row_contract,
    sha256_file,
    symbolic_candidates,
)
from run_v296_bit_stride_solver_audit import solve_stride  # noqa: E402
from run_v324_equation_expanded_solver_gate import (  # noqa: E402
    symbolic_variable_operator_candidates,
    v274_guarded_candidate,
    v299_rows_as_v324,
)
from run_v329_symbolic_cryptarithm_gate import symbolic_cryptarithm_candidates  # noqa: E402
from src.solvers.bit_manipulation_solver import BITWISE_OPS, make_transforms, parse_bit_problem, to_bits, from_bits  # noqa: E402


DEFAULT_V343_PREDICTIONS = (
    REPO_ROOT
    / "artifacts/v343_equation_residual_solver_audit/20260513T_integrated_on_v290_v3/"
    / "v336a_integrated_no_loss_predictions.csv"
)
DEFAULT_V348_EQUATION_RESIDUALS = (
    REPO_ROOT
    / "artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_equation_residuals.csv"
)
DEFAULT_V348_BIT_RESIDUALS = (
    REPO_ROOT
    / "artifacts/v348_residual_no_loss_expansion/20260513T_cpu_gate/v348_bit_residuals.csv"
)
DEFAULT_V349_TRIAGE = (
    REPO_ROOT
    / "artifacts/v349_kaggle_discussion_double_check/20260514T003649Z/"
    / "v349_kaggle_discussion_triage.csv"
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
    "verified_candidates",
    "incorrect_candidates",
    "accepted_rows",
    "accepted",
    "reason",
]
FAMILY_COLUMNS = [
    "family",
    "rows",
    "v343_correct",
    "v350_correct",
    "delta_correct",
    "v343_truncated",
    "v350_truncated",
]


class SymbolicArgs:
    pair_mapping_cap = 1000
    global_mapping_cap = 20000
    max_char_subset_size = 4
    max_position_sources = 6


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
    return str(row.get("prediction") or row.get("integrated_prediction") or row.get("baseline_prediction") or "").strip()


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        fam = str(row["family"])
        family[fam]["rows"] += 1
        total["rows"] += 1
        ok = verify_answer(str(row["answer"]), str(row.get(prediction_key, "")))
        family[fam]["correct"] += int(ok)
        total["correct"] += int(ok)
        truncated = truthy(row.get("truncated", row.get("truncated_bool", "")))
        family[fam]["truncated"] += int(truncated)
        total["truncated"] += int(truncated)
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "accuracy": float(total["correct"] / total["rows"]) if total["rows"] else 0.0,
        "truncated": int(total["truncated"]),
        "family": {key: dict(value) for key, value in sorted(family.items())},
    }


def family_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before = summarize(rows, "v343_prediction")
    after = summarize(rows, "v350_prediction")
    out = []
    for family in sorted(before["family"]):
        before_item = before["family"][family]
        after_item = after["family"].get(family, {})
        out.append(
            {
                "family": family,
                "rows": before_item.get("rows", 0),
                "v343_correct": before_item.get("correct", 0),
                "v350_correct": after_item.get("correct", 0),
                "delta_correct": int(after_item.get("correct", 0)) - int(before_item.get("correct", 0)),
                "v343_truncated": before_item.get("truncated", 0),
                "v350_truncated": after_item.get("truncated", 0),
            }
        )
    return out


def solve_exact_global_byte(prompt: str) -> tuple[str, str, str]:
    examples, test_input = parse_bit_problem(prompt)
    if not examples or not test_input:
        return "", "bit_exact_global_byte_parse_failed", "parse_failed"
    transforms = make_transforms()
    inputs = [to_bits(left) for left, _right in examples]
    outputs = [to_bits(right) for _left, right in examples]
    test_bits = to_bits(test_input)
    all_inputs = inputs + [test_bits]
    pre = {name: [func(bits) for bits in all_inputs] for name, func in transforms}
    test_index = len(inputs)

    for name, _func in transforms:
        if all(pre[name][index] == outputs[index] for index in range(len(outputs))):
            return from_bits(pre[name][test_index]), "bit_exact_global_unary_" + name, f"output={name}(input)"

    for left_name, _left_func in transforms:
        for right_name, _right_func in transforms:
            for op_name, op_func in BITWISE_OPS:
                if all(
                    op_func(pre[left_name][index], pre[right_name][index]) == outputs[index]
                    for index in range(len(outputs))
                ):
                    answer = from_bits(op_func(pre[left_name][test_index], pre[right_name][test_index]))
                    rule_class = "bit_exact_global_binary_" + op_name
                    proof = f"output={op_name}({left_name}(input),{right_name}(input))"
                    return answer, rule_class, proof
    return "", "bit_exact_global_byte_abstain", "no exact global unary/binary rule"


def equation_audit_rows(rows: list[dict[str, Any]], *, progress_every: int) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    misses = [
        row
        for row in rows
        if str(row["family"]) == "equation_transform" and not verify_answer(str(row["answer"]), current_prediction(row))
    ]
    print("v350_equation_miss_rows =", len(misses), flush=True)
    for index, row in enumerate(misses, start=1):
        if index == 1 or index % progress_every == 0 or index == len(misses):
            print(f"v350_equation_audit_progress = {index}/{len(misses)}", flush=True)
        examples, query, parse_status = parse_alice_prompt(str(row["prompt"]))
        if parse_status != "ok":
            continue
        subtype = classify_subtype(examples, query)
        if subtype == "equation_numeric_operator":
            for result in (
                [numeric_candidate(examples, query, min_examples=2), v274_guarded_candidate(row)]
                + [dict(item) for item in []]
            ):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v350_equation_numeric_local"
                audit_rows.append(audit)
            for audit in v299_rows_as_v324(row, examples, query):
                audit["candidate_source"] = "v350_v299_numeric_dsl"
                audit_rows.append(audit)
        else:
            for result in symbolic_candidates(examples, query, SymbolicArgs()):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v350_v278_symbolic_dsl"
                audit_rows.append(audit)
            for result in symbolic_variable_operator_candidates(examples, query):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v350_v324_variable_operator"
                audit_rows.append(audit)
            for result in symbolic_cryptarithm_candidates(
                examples,
                query,
                max_operator_symbols=4,
                max_solutions_per_assignment=3,
                solver_time_limit_s=0.08,
            ):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v350_v329_symbolic_cryptarithm"
                audit_rows.append(audit)
    return audit_rows


def bit_candidate_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for row in rows:
        if str(row["family"]) != "bit_manipulation":
            continue
        old_prediction = current_prediction(row)
        answer, rule_class, proof = solve_exact_global_byte(str(row["prompt"]))
        if not answer or answer == old_prediction:
            continue
        old_correct = verify_answer(str(row["answer"]), old_prediction)
        new_correct = verify_answer(str(row["answer"]), answer)
        decisions.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "candidate_source": "v350_bit_exact_global_byte",
                "rule_class": rule_class,
                "old_prediction": old_prediction,
                "new_prediction": answer,
                "answer": row["answer"],
                "old_correct": old_correct,
                "new_correct": new_correct,
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": 1,
                "conflict_count": 0,
                "proof": proof,
            }
        )
    return decisions


def equation_candidate_decisions(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped_by_rule[str(row.get("rule_class", ""))].append(row)

    promotable_rules = set()
    for rule_class, items in grouped_by_rule.items():
        candidates = [row for row in items if row.get("status") == "candidate"]
        incorrect = [row for row in candidates if truthy(row.get("incorrect_by_weak_label", ""))]
        verified = [row for row in candidates if truthy(row.get("verified_by_weak_label", ""))]
        if candidates and verified and not incorrect:
            promotable_rules.add(rule_class)

    grouped_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        if row.get("status") == "candidate":
            grouped_by_id[str(row.get("id", ""))].append(row)

    decisions = []
    for row_id, items in sorted(grouped_by_id.items()):
        predictions = sorted({str(item.get("prediction", "")) for item in items if item.get("prediction")})
        conflict_count = len(predictions) if len(predictions) > 1 else 0
        for item in items:
            new_prediction = str(item.get("prediction", ""))
            old_prediction = str(item.get("baseline_prediction", ""))
            old_correct = verify_answer(str(item.get("answer", "")), old_prediction)
            new_correct = verify_answer(str(item.get("answer", "")), new_prediction)
            rule_class = str(item.get("rule_class", ""))
            accepted = (
                not conflict_count
                and rule_class in promotable_rules
                and (not old_correct)
                and new_correct
            )
            if accepted:
                reason = ""
            elif conflict_count:
                reason = "reject_conflicting_predictions"
            elif rule_class not in promotable_rules:
                reason = "reject_rule_class_has_incorrect_or_no_verified_candidates"
            elif old_correct:
                reason = "reject_old_already_correct"
            elif not new_correct:
                reason = "reject_candidate_not_correct"
            else:
                reason = "reject_unknown"
            decisions.append(
                {
                    "id": row_id,
                    "family": "equation_transform",
                    "candidate_source": item.get("candidate_source", ""),
                    "rule_class": rule_class,
                    "old_prediction": old_prediction,
                    "new_prediction": new_prediction,
                    "answer": item.get("answer", ""),
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "accepted": accepted,
                    "rejection_reason": reason,
                    "candidate_count": len(items),
                    "conflict_count": conflict_count,
                    "proof": item.get("proof", ""),
                }
            )
    return decisions


def rule_summary(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)
    rows = []
    for (family, source, rule_class), items in sorted(grouped.items()):
        changed = [row for row in items if row.get("new_prediction") != row.get("old_prediction")]
        gains = [row for row in changed if truthy(row.get("new_correct")) and not truthy(row.get("old_correct"))]
        losses = [row for row in changed if truthy(row.get("old_correct")) and not truthy(row.get("new_correct"))]
        verified = [row for row in items if truthy(row.get("new_correct"))]
        incorrect = [row for row in items if not truthy(row.get("new_correct"))]
        accepted_rows = [row for row in items if truthy(row.get("accepted"))]
        accepted = bool(accepted_rows) and not losses
        reason = (
            "accepted_no_loss"
            if accepted
            else "rejected_losses_or_no_verified_gain"
            if losses or not gains
            else "rejected_by_row_gate"
        )
        rows.append(
            {
                "family": family,
                "candidate_source": source,
                "rule_class": rule_class,
                "candidate_rows": len(items),
                "changed_rows": len(changed),
                "gains": len(gains),
                "losses": len(losses),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "accepted_rows": len(accepted_rows),
                "accepted": accepted,
                "reason": reason,
            }
        )
    rows.sort(key=lambda item: (int(item["gains"]), -int(item["losses"]), str(item["rule_class"])), reverse=True)
    return rows


def apply_accepted(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if truthy(decision.get("accepted")):
            accepted_by_id[str(decision["id"])].append(decision)
    out = []
    for row in rows:
        item = dict(row)
        item["v343_prediction"] = current_prediction(row)
        accepted = accepted_by_id.get(str(row["id"]), [])
        if accepted:
            unique_predictions = sorted({str(candidate["new_prediction"]) for candidate in accepted})
            if len(unique_predictions) != 1:
                raise RuntimeError(f"accepted prediction conflict for id={row['id']}: {unique_predictions}")
            item["v350_prediction"] = unique_predictions[0]
            item["v350_source_rule"] = ";".join(
                sorted({str(candidate["rule_class"]) for candidate in accepted if candidate.get("rule_class")})
            )
        else:
            item["v350_prediction"] = current_prediction(row)
            item["v350_source_rule"] = ""
        item["v350_correct"] = verify_answer(str(item["answer"]), str(item["v350_prediction"]))
        # Keep the prediction column as the integrated candidate for downstream tools.
        item["prediction"] = item["v350_prediction"]
        out.append(item)
    return out


def validate_expected_summary(summary: dict[str, Any], args: argparse.Namespace) -> None:
    if int(summary["correct"]) != args.expected_v343_correct:
        raise RuntimeError(f"expected V343 correct={args.expected_v343_correct}, got {summary['correct']}")
    eq = int(summary["family"].get("equation_transform", {}).get("correct", 0))
    bit = int(summary["family"].get("bit_manipulation", {}).get("correct", 0))
    if eq != args.expected_v343_equation_correct:
        raise RuntimeError(f"expected V343 equation={args.expected_v343_equation_correct}, got {eq}")
    if bit != args.expected_v343_bit_correct:
        raise RuntimeError(f"expected V343 bit={args.expected_v343_bit_correct}, got {bit}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V350 CPU RESIDUAL NO-LOSS GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v343_predictions_csv =", args.v343_predictions_csv, flush=True)
    print("v348_equation_residuals_csv =", args.v348_equation_residuals_csv, flush=True)
    print("v348_bit_residuals_csv =", args.v348_bit_residuals_csv, flush=True)
    print("v349_triage_csv =", args.v349_triage_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(row) for row in read_csv(args.v343_predictions_csv)]
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    v343_summary = summarize(rows, "prediction")
    validate_expected_summary(v343_summary, args)
    print("v343_summary =", json.dumps(v343_summary, sort_keys=True), flush=True)

    equation_residual_rows = read_csv(args.v348_equation_residuals_csv)
    bit_residual_rows = read_csv(args.v348_bit_residuals_csv)
    triage_rows = read_csv(args.v349_triage_csv) if args.v349_triage_csv and args.v349_triage_csv.exists() else []
    print("v348_equation_residual_count =", len(equation_residual_rows), flush=True)
    print("v348_bit_residual_count =", len(bit_residual_rows), flush=True)
    print("v349_triage_rows =", len(triage_rows), flush=True)

    equation_audit = equation_audit_rows(rows, progress_every=args.progress_every)
    equation_decisions = equation_candidate_decisions(equation_audit)
    bit_decisions = bit_candidate_decisions(rows)

    # Bit exact global byte is an all-or-nothing deployable rule family. Promote
    # only if the full changed set has at least one gain and zero losses.
    bit_gains = [row for row in bit_decisions if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
    bit_losses = [row for row in bit_decisions if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
    bit_promote = bool(bit_gains) and not bit_losses
    for row in bit_decisions:
        if bit_promote and truthy(row["new_correct"]) and not truthy(row["old_correct"]):
            row["accepted"] = True
        elif bit_losses:
            row["rejection_reason"] = "reject_bit_exact_global_byte_has_losses"
        elif not bit_gains:
            row["rejection_reason"] = "reject_bit_exact_global_byte_no_gain"
        else:
            row["rejection_reason"] = "reject_no_accuracy_gain"

    decisions = equation_decisions + bit_decisions
    accepted = [row for row in decisions if truthy(row.get("accepted"))]
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("accepted_candidate_count =", len(accepted), flush=True)
    print("accepted_candidate_ids =", [row["id"] for row in accepted], flush=True)

    integrated_rows = apply_accepted(rows, decisions)
    v350_summary = summarize(integrated_rows, "v350_prediction")
    family_rows = family_summary_rows(integrated_rows)
    print("v350_summary =", json.dumps(v350_summary, sort_keys=True), flush=True)
    print("v350_family_summary =", json.dumps(family_rows, sort_keys=True), flush=True)

    gains = [
        row
        for row in integrated_rows
        if verify_answer(str(row["answer"]), str(row["v350_prediction"]))
        and not verify_answer(str(row["answer"]), str(row["v343_prediction"]))
    ]
    losses = [
        row
        for row in integrated_rows
        if verify_answer(str(row["answer"]), str(row["v343_prediction"]))
        and not verify_answer(str(row["answer"]), str(row["v350_prediction"]))
    ]
    eq_correct = int(v350_summary["family"].get("equation_transform", {}).get("correct", 0))
    bit_correct = int(v350_summary["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = (
        not losses
        and (
            eq_correct > args.expected_v343_equation_correct
            or bit_correct > args.expected_v343_bit_correct
        )
        and bit_correct >= args.expected_v343_bit_correct
    )

    if gate_pass:
        decision = {
            "decision": "v350_cpu_residual_no_loss_gate_passed",
            "hf_gpu_allowed": False,
            "reason": (
                f"v350={v350_summary['correct']}/315; equation={eq_correct}/155; "
                f"bit={bit_correct}/160; gains={len(gains)}; losses={len(losses)}"
            ),
            "next_action": "Build V351 minimal transfer dataset from accepted V350 rows before any HF GPU job.",
        }
    else:
        decision = {
            "decision": "v350_cpu_residual_no_loss_gate_blocked",
            "hf_gpu_allowed": False,
            "reason": (
                f"v350={v350_summary['correct']}/315; equation={eq_correct}/155; "
                f"bit={bit_correct}/160; gains={len(gains)}; losses={len(losses)}"
            ),
            "next_action": "Do not launch HF. Expand CPU rules or inspect residuals.",
        }
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "candidate_rules_csv": args.output_dir / "v350_candidate_rules.csv",
        "candidate_decisions_csv": args.output_dir / "v350_candidate_decisions.csv",
        "equation_audit_csv": args.output_dir / "v350_equation_audit.csv",
        "integrated_predictions_csv": args.output_dir / "v350_integrated_predictions.csv",
        "family_summary_csv": args.output_dir / "v350_family_summary.csv",
        "manifest_json": args.output_dir / "v350_no_loss_gate_manifest.json",
    }
    candidate_rules = rule_summary(decisions)
    write_csv(outputs["candidate_rules_csv"], candidate_rules, RULE_COLUMNS)
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["equation_audit_csv"], equation_audit, list(equation_audit[0]) if equation_audit else [])
    write_csv(outputs["integrated_predictions_csv"], integrated_rows, list(integrated_rows[0]) if integrated_rows else [])
    write_csv(outputs["family_summary_csv"], family_rows, FAMILY_COLUMNS)

    manifest = {
        "schema_version": "kg1_v350_cpu_residual_no_loss_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v343_predictions_csv": str(args.v343_predictions_csv),
            "v343_predictions_sha256": sha256_file(args.v343_predictions_csv),
            "v348_equation_residuals_csv": str(args.v348_equation_residuals_csv),
            "v348_equation_residuals_sha256": sha256_file(args.v348_equation_residuals_csv),
            "v348_bit_residuals_csv": str(args.v348_bit_residuals_csv),
            "v348_bit_residuals_sha256": sha256_file(args.v348_bit_residuals_csv),
            "v349_triage_csv": str(args.v349_triage_csv),
            "v349_triage_sha256": sha256_file(args.v349_triage_csv) if args.v349_triage_csv.exists() else "",
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "v343_summary": v343_summary,
        "v350_summary": v350_summary,
        "family_summary": family_rows,
        "accepted_candidate_count": len(accepted),
        "accepted_candidate_ids": [row["id"] for row in accepted],
        "gain_count": len(gains),
        "gain_ids": [row["id"] for row in gains],
        "loss_count": len(losses),
        "loss_ids": [row["id"] for row in losses],
        "candidate_rule_summary_top": candidate_rules[:20],
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V350 CPU RESIDUAL NO-LOSS GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
10000000 -> 00000001
01000000 -> 10000000
00100000 -> 01000000

Now, determine the output for: 00010000"""
    answer, rule_class, proof = solve_exact_global_byte(prompt)
    if answer != "00100000":
        raise AssertionError((answer, rule_class, proof))
    print("v350_cpu_residual_no_loss_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v343-predictions-csv", type=Path, default=DEFAULT_V343_PREDICTIONS)
    parser.add_argument("--v348-equation-residuals-csv", type=Path, default=DEFAULT_V348_EQUATION_RESIDUALS)
    parser.add_argument("--v348-bit-residuals-csv", type=Path, default=DEFAULT_V348_BIT_RESIDUALS)
    parser.add_argument("--v349-triage-csv", type=Path, default=DEFAULT_V349_TRIAGE)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--expected-v343-correct", type=int, default=199)
    parser.add_argument("--expected-v343-equation-correct", type=int, default=63)
    parser.add_argument("--expected-v343-bit-correct", type=int, default=136)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V363 CPU-only equation residual operator-support gate.

This gate follows the post-V362 roadmap: no GPU, no package, no submit.
It audits the latest integrated weak predictions and checks whether the
remaining equation_transform misses contain a new label-free signal:

* symbolic rows with query operator seen in prompt examples;
* symbolic same-operator DSL candidates derived only from same-op examples;
* numeric query-operator priors learned from public train rows excluding the
  weak contract ids.

Weak labels are used only as the audit brake. HF is allowed only if a candidate
rule class gives a no-loss gain over the current integrated predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import parse_numeric_token  # noqa: E402
from analyze_v241_abstain_rule_candidate_audit import numeric_rule_functions  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    EXPECTED_ROW_CONTRACT_SHA256,
    classify_subtype,
    normalize_row,
    parse_alice_prompt,
    row_contract,
    sha256_file,
    symbolic_candidates,
)
from run_v324_equation_expanded_solver_gate import symbolic_variable_operator_candidates  # noqa: E402


DEFAULT_INPUT_CSV = (
    REPO_ROOT / "artifacts/v355_cpu_residual_gate/20260514T_cpu_gate/v355_integrated_predictions.csv"
)
DEFAULT_PUBLIC_TRAIN_CSV = (
    REPO_ROOT.parent.parent
    / "artifacts/api_kaggle_openrouter_audit_2026_05_06/competition_data/extracted/train.csv"
)
if not DEFAULT_PUBLIC_TRAIN_CSV.exists():
    DEFAULT_PUBLIC_TRAIN_CSV = REPO_ROOT / "data/train.csv"

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

RESIDUAL_COLUMNS = [
    "id",
    "family",
    "answer",
    "current_prediction",
    "current_correct",
    "subtype",
    "query",
    "query_shape",
    "query_operator",
    "query_operator_same_position_seen_count",
    "query_operator_anywhere_seen",
    "example_count",
    "route_hint",
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


def query_shape(query: str) -> str:
    out = []
    for ch in str(query):
        if ch.isdigit():
            out.append("D")
        elif ch.isalpha():
            out.append("A")
        else:
            out.append("P")
    return "".join(out)


def current_prediction(row: dict[str, Any]) -> str:
    for key in ("v355_prediction", "v350_prediction", "integrated_prediction", "prediction", "baseline_prediction"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


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


def normalize_prediction_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        base = normalize_row(row)
        pred = current_prediction(row)
        base.update(row)
        base["family"] = canonical_family(base.get("family") or classify_puzzle(str(base.get("prompt", ""))))
        base["current_prediction"] = pred
        base["current_correct"] = verify_answer(str(base.get("answer", "")), pred)
        normalized.append(base)
    return normalized


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(int(value)):02d}"[-2:]
    return int(text[0]), int(text[1])


def numeric_functions() -> dict[str, Callable[[int, int], str]]:
    funcs = dict(numeric_rule_functions())
    funcs.update(
        {
            "mul_rev": lambda a, b: str(a * b)[::-1],
            "add_rev": lambda a, b: str(a + b)[::-1],
            "abs_diff_rev": lambda a, b: str(abs(a - b))[::-1],
            "sub_ab_rev_keep_sign": lambda a, b: ("-" if a - b < 0 else "") + str(abs(a - b))[::-1],
            "sub_ba_rev_keep_sign": lambda a, b: ("-" if b - a < 0 else "") + str(abs(b - a))[::-1],
            "concat_ab_rev": lambda a, b: (f"{abs(a)}{abs(b)}")[::-1],
            "concat_ba_rev": lambda a, b: (f"{abs(b)}{abs(a)}")[::-1],
            "mul_plus1": lambda a, b: str(a * b + 1),
            "mul_minus1": lambda a, b: str(a * b - 1),
            "square_sum": lambda a, b: str(a * a + b * b),
            "square_diff_abs": lambda a, b: str(abs(a * a - b * b)),
            "first_digits_concat": lambda a, b: str(abs(a) // 10) + str(abs(b) // 10),
            "last_digits_concat": lambda a, b: str(abs(a) % 10) + str(abs(b) % 10),
            "cross_add_concat": lambda a, b: str((digits2(a)[0] + digits2(b)[1]) % 10)
            + str((digits2(a)[1] + digits2(b)[0]) % 10),
            "cross_absdiff_concat": lambda a, b: str(abs(digits2(a)[0] - digits2(b)[1]))
            + str(abs(digits2(a)[1] - digits2(b)[0])),
        }
    )
    return funcs


def build_numeric_operator_priors(
    train_csv: Path,
    excluded_ids: set[str],
    *,
    min_support: int,
    min_precision: float,
) -> dict[str, dict[str, Any]]:
    if not train_csv.exists():
        return {}
    funcs = numeric_functions()
    totals: Counter[str] = Counter()
    matches: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_csv(train_csv):
        if str(row.get("id", "")) in excluded_ids:
            continue
        if canonical_family(classify_puzzle(str(row.get("prompt", "")))) != "equation_transform":
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        if parse_status != "ok":
            continue
        parsed = parse_numeric_token(query)
        if not parsed:
            continue
        left, op, right = parsed
        totals[op] += 1
        for name, func in funcs.items():
            try:
                prediction = func(left, right)
            except Exception:
                continue
            if prediction == str(row.get("answer", "")):
                matches[op][name] += 1

    priors: dict[str, dict[str, Any]] = {}
    for op, total in totals.items():
        if total <= 0 or not matches.get(op):
            continue
        func_name, support = matches[op].most_common(1)[0]
        precision = float(support / total)
        if support >= min_support and precision >= min_precision:
            priors[op] = {
                "operator": op,
                "function": func_name,
                "support": int(support),
                "total": int(total),
                "precision": precision,
            }
    return priors


def decision_row(
    row: dict[str, Any],
    *,
    source: str,
    rule_class: str,
    new_prediction: str,
    proof: str,
    candidate_count: int = 1,
    conflict_count: int = 0,
) -> dict[str, Any]:
    old_prediction = current_prediction(row)
    return {
        "id": row["id"],
        "family": row["family"],
        "candidate_source": source,
        "rule_class": rule_class,
        "old_prediction": old_prediction,
        "new_prediction": new_prediction,
        "answer": row["answer"],
        "old_correct": verify_answer(str(row["answer"]), old_prediction),
        "new_correct": verify_answer(str(row["answer"]), new_prediction),
        "accepted": False,
        "rejection_reason": "",
        "candidate_count": candidate_count,
        "conflict_count": conflict_count,
        "proof": proof,
    }


def same_operator_symbolic_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    audit_args = argparse.Namespace(
        pair_mapping_cap=args.pair_mapping_cap,
        global_mapping_cap=args.global_mapping_cap,
        max_char_subset_size=args.max_char_subset_size,
        max_position_sources=args.max_position_sources,
    )
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("family")) != "equation_transform":
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        if parse_status != "ok" or classify_subtype(examples, query) != "equation_symbolic_punct":
            continue
        if len(query) < 3:
            continue
        query_op = query[2]
        same_examples = [(lhs, rhs) for lhs, rhs in examples if len(lhs) > 2 and lhs[2] == query_op]
        if len(same_examples) < args.min_same_operator_examples:
            continue
        candidates = []
        for result in symbolic_candidates(same_examples, query, audit_args):
            candidates.append(("v363_same_operator_symbolic", "same_op_" + str(result["rule_class"]), result))
        for result in symbolic_variable_operator_candidates(same_examples, query):
            candidates.append(("v363_same_operator_variable_symbolic", "same_op_" + str(result["rule_class"]), result))
        old_prediction = current_prediction(row)
        for source, rule_class, result in candidates:
            if str(result.get("status")) != "candidate":
                continue
            prediction = str(result.get("prediction", "")).strip()
            if not prediction or prediction == old_prediction:
                continue
            decisions.append(
                decision_row(
                    row,
                    source=source,
                    rule_class=rule_class,
                    new_prediction=prediction,
                    proof="same_operator_examples="
                    + str(len(same_examples))
                    + "; "
                    + str(result.get("proof", "")),
                    candidate_count=int(result.get("candidate_program_count") or 1),
                    conflict_count=max(0, int(result.get("unique_prediction_count") or 1) - 1),
                )
            )
    return decisions


def numeric_prior_decisions(
    rows: list[dict[str, Any]],
    priors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    funcs = numeric_functions()
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("family")) != "equation_transform":
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        if parse_status != "ok":
            continue
        parsed = parse_numeric_token(query)
        if not parsed:
            continue
        left, op, right = parsed
        prior = priors.get(op)
        if not prior:
            continue
        func = funcs[str(prior["function"])]
        try:
            prediction = func(left, right)
        except Exception:
            continue
        if not prediction or prediction == current_prediction(row):
            continue
        decisions.append(
            decision_row(
                row,
                source="v363_public_train_exweak_numeric_prior",
                rule_class="numeric_operator_prior_"
                + str(ord(op))
                + "_"
                + str(prior["function"]),
                new_prediction=prediction,
                proof=json.dumps(prior, sort_keys=True),
            )
        )
    return decisions


def apply_rule_gates(decisions: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)

    for _key, items in grouped.items():
        gains = [row for row in items if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
        losses = [row for row in items if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
        conflicts = [row for row in items if int(row.get("conflict_count") or 0) > 0]
        can_accept = bool(gains) and not losses and not conflicts
        for row in items:
            if can_accept and row in gains:
                row["accepted"] = True
                row["rejection_reason"] = ""
            elif losses:
                row["rejection_reason"] = "reject_rule_class_has_losses"
            elif conflicts:
                row["rejection_reason"] = "reject_conflicting_predictions"
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


def residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = current_prediction(row)
        if verify_answer(str(row.get("answer", "")), pred):
            continue
        if str(row.get("family")) != "equation_transform":
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        subtype = classify_subtype(examples, query) if parse_status == "ok" else parse_status
        query_op = query[2] if len(query) > 2 else ""
        same_seen = sum(1 for lhs, _rhs in examples if len(lhs) > 2 and lhs[2] == query_op)
        anywhere_seen = bool(query_op and query_op in "".join(lhs for lhs, _rhs in examples))
        if subtype == "equation_numeric_operator" and same_seen == 0:
            hint = "numeric_query_operator_unseen_in_examples_model_prior_required"
        elif subtype == "equation_symbolic_punct" and same_seen == 0:
            hint = "symbolic_query_operator_unseen_in_examples_no_label_free_tiebreaker"
        elif subtype == "equation_symbolic_punct":
            hint = "symbolic_query_operator_seen_but_same_op_dsl_no_unique_candidate"
        else:
            hint = "residual_unknown"
        out.append(
            {
                "id": row["id"],
                "family": row["family"],
                "answer": row.get("answer", ""),
                "current_prediction": pred,
                "current_correct": False,
                "subtype": subtype,
                "query": query,
                "query_shape": query_shape(query),
                "query_operator": query_op,
                "query_operator_same_position_seen_count": same_seen,
                "query_operator_anywhere_seen": anywhere_seen,
                "example_count": len(examples),
                "route_hint": hint,
            }
        )
    return out


def integrate(rows: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_id[str(row["id"])].append(row)
    integrated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["v363_prediction"] = current_prediction(row)
        item["v363_source_rule"] = ""
        accepted_items = accepted_by_id.get(str(row["id"]), [])
        if len(accepted_items) == 1:
            item["v363_prediction"] = str(accepted_items[0]["new_prediction"])
            item["v363_source_rule"] = str(accepted_items[0]["rule_class"])
        elif len(accepted_items) > 1:
            raise RuntimeError(f"accepted conflict id={row['id']}")
        item["v363_correct"] = verify_answer(str(item["answer"]), str(item["v363_prediction"]))
        integrated.append(item)
    return integrated


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V363 EQUATION RESIDUAL OPERATOR SUPPORT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("input_predictions_csv =", args.input_predictions_csv, flush=True)
    print("public_train_csv =", args.public_train_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("min_same_operator_examples =", args.min_same_operator_examples, flush=True)
    print("numeric_prior_min_support =", args.numeric_prior_min_support, flush=True)
    print("numeric_prior_min_precision =", args.numeric_prior_min_precision, flush=True)
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
    for row in rows:
        row["current_prediction_for_summary"] = current_prediction(row)
    baseline = summarize(rows, "current_prediction_for_summary")
    print("current_summary =", json.dumps(baseline, sort_keys=True), flush=True)

    residual = residual_rows(rows)
    residual_counts = Counter(str(row["route_hint"]) for row in residual)
    residual_subtypes = Counter(str(row["subtype"]) for row in residual)
    print("equation_residual_count =", len(residual), flush=True)
    print("equation_residual_subtypes =", json.dumps(dict(residual_subtypes), sort_keys=True), flush=True)
    print("equation_residual_route_hints =", json.dumps(dict(residual_counts), sort_keys=True), flush=True)

    excluded_ids = {str(row["id"]) for row in rows}
    priors = build_numeric_operator_priors(
        args.public_train_csv,
        excluded_ids,
        min_support=args.numeric_prior_min_support,
        min_precision=args.numeric_prior_min_precision,
    )
    print("numeric_operator_prior_count =", len(priors), flush=True)

    decisions = same_operator_symbolic_decisions(rows, args) + numeric_prior_decisions(rows, priors)
    apply_rule_gates(decisions)
    accepted = [row for row in decisions if truthy(row.get("accepted"))]
    rules = rule_summary(decisions)
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("accepted_candidate_count =", len(accepted), flush=True)
    print("candidate_rule_summary_top =", json.dumps(rules[:12], sort_keys=True), flush=True)

    integrated = integrate(rows, accepted)
    after = summarize(integrated, "v363_prediction")
    gains = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row["v363_prediction"]))
        and not verify_answer(str(row["answer"]), current_prediction(row))
    ]
    losses = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), current_prediction(row))
        and not verify_answer(str(row["answer"]), str(row["v363_prediction"]))
    ]
    eq_before = int(baseline["family"].get("equation_transform", {}).get("correct", 0))
    bit_before = int(baseline["family"].get("bit_manipulation", {}).get("correct", 0))
    eq_after = int(after["family"].get("equation_transform", {}).get("correct", 0))
    bit_after = int(after["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = bool(accepted) and not losses and (eq_after > eq_before or bit_after > bit_before)
    decision = {
        "decision": "v363_cpu_gate_passed" if gate_pass else "v363_cpu_gate_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            f"v363={after['correct']}/315; equation={eq_after}/155; bit={bit_after}/160; "
            f"gains={len(gains)}; losses={len(losses)}; accepted={len(accepted)}"
        ),
        "next_action": (
            "Build a transfer dataset only from accepted V363 rows after tokenization/no-regression gates."
            if gate_pass
            else "Do not launch HF. Remaining equation residuals are dominated by query-operator gaps or same-op DSL ambiguity."
        ),
    }
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "residuals_csv": args.output_dir / "v363_equation_residuals.csv",
        "candidate_decisions_csv": args.output_dir / "v363_candidate_decisions.csv",
        "candidate_rules_csv": args.output_dir / "v363_candidate_rules.csv",
        "integrated_predictions_csv": args.output_dir / "v363_integrated_predictions.csv",
        "manifest_json": args.output_dir / "v363_equation_residual_operator_support_manifest.json",
    }
    write_csv(outputs["residuals_csv"], residual, RESIDUAL_COLUMNS)
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["candidate_rules_csv"], rules, RULE_COLUMNS)
    write_csv(outputs["integrated_predictions_csv"], integrated, list(integrated[0]) if integrated else [])

    manifest = {
        "schema_version": "kg1_v363_equation_residual_operator_support_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "input_predictions_csv": str(args.input_predictions_csv),
            "input_predictions_sha256": sha256_file(args.input_predictions_csv),
            "public_train_csv": str(args.public_train_csv),
            "public_train_sha256": sha256_file(args.public_train_csv) if args.public_train_csv.exists() else "",
            "public_train_excluded_id_count": len(excluded_ids),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "current_summary": baseline,
        "v363_summary": after,
        "equation_residual_count": len(residual),
        "equation_residual_subtypes": dict(residual_subtypes),
        "equation_residual_route_hints": dict(residual_counts),
        "numeric_operator_prior_count": len(priors),
        "numeric_operator_priors": priors,
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
    print("=== V363 EQUATION RESIDUAL OPERATOR SUPPORT END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {
            "id": "x",
            "family": "equation_transform",
            "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\nab+cd = abcd\nxy+zz = xyzz\nNow, determine the result for: pq+rs",
            "answer": "pqrs",
            "prediction": "bad",
        }
    ]
    decisions = same_operator_symbolic_decisions(rows, argparse.Namespace(
        min_same_operator_examples=2,
        pair_mapping_cap=50,
        global_mapping_cap=100,
        max_char_subset_size=2,
        max_position_sources=8,
    ))
    assert any(row["new_prediction"] == "pqrs" for row in decisions)
    print("v363_equation_residual_operator_support_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--public-train-csv", type=Path, default=DEFAULT_PUBLIC_TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v363_equation_residual_operator_support" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--pair-mapping-cap", type=int, default=200)
    parser.add_argument("--global-mapping-cap", type=int, default=1000)
    parser.add_argument("--max-char-subset-size", type=int, default=3)
    parser.add_argument("--max-position-sources", type=int, default=8)
    parser.add_argument("--numeric-prior-min-support", type=int, default=3)
    parser.add_argument("--numeric-prior-min-precision", type=float, default=0.25)
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

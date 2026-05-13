#!/usr/bin/env python3
"""CPU gate for symbolic equation cryptarithm candidates.

V329 is intentionally CPU-only and diagnostic. It extends the V324 equation
gate with one narrow hypothesis for the remaining symbolic/punctuation misses:
some 5-character Alice expressions are two encoded two-digit numbers separated
by an arithmetic operator, and the RHS is the encoded decimal result.

Weak labels are used only as an audit brake. This script does not train, run
GPU inference, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover - explicit runtime gate
    raise RuntimeError("ortools is required for V329 symbolic cryptarithm gate") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    AUDIT_COLUMNS as V278_AUDIT_COLUMNS,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_BASELINE_REPO,
    EXPECTED_ROW_CONTRACT_SHA256,
    build_audit_row,
    classify_subtype,
    download_file,
    normalize_row,
    parse_alice_prompt,
    row_contract,
    sha256_file,
)
from run_v324_equation_expanded_solver_gate import (  # noqa: E402
    accepted_no_loss_candidates,
    family_counts,
    summarize_rule_classes,
)


DEFAULT_V324_MANIFEST = (
    REPO_ROOT
    / "artifacts/v324_equation_expanded_solver_gate/20260513T_cpu_gate/v324_equation_expanded_solver_manifest.json"
)

EXTRA_AUDIT_COLUMNS = V278_AUDIT_COLUMNS + ["candidate_source"]
SUMMARY_COLUMNS = [
    "rule_class",
    "rows",
    "candidate_rows",
    "verified_candidates",
    "incorrect_candidates",
    "abstain_rows",
    "promotable_after_class_gate",
]
ARITHMETIC_RULES = ("add", "sub_ab", "sub_ba", "abs_diff", "mul")


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_predictions(
    rule_class: str,
    predictions: list[str],
    proof: str,
    *,
    candidate_program_count: int,
) -> dict[str, Any]:
    unique = sorted({prediction for prediction in predictions if prediction != ""})
    if len(unique) == 1:
        return {
            "rule_class": rule_class,
            "status": "candidate",
            "prediction": unique[0],
            "proof": proof,
            "candidate_program_count": candidate_program_count,
            "unique_prediction_count": 1,
        }
    return {
        "rule_class": rule_class,
        "status": "abstain",
        "prediction": "",
        "proof": proof + f"; unique_prediction_count={len(unique)}",
        "candidate_program_count": candidate_program_count,
        "unique_prediction_count": len(unique),
    }


def split_encoded_binary_expr(token: str) -> tuple[str, str, str] | None:
    """Parse two encoded two-digit numbers around one operator char."""

    text = str(token or "").strip()
    if len(text) != 5:
        return None
    return text[:2], text[2], text[3:]


def apply_arithmetic_rule(rule: str, left: int, right: int) -> int:
    if rule == "add":
        return left + right
    if rule == "sub_ab":
        return left - right
    if rule == "sub_ba":
        return right - left
    if rule == "abs_diff":
        return abs(left - right)
    if rule == "mul":
        return left * right
    raise KeyError(rule)


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: dict[str, Any], limit: int) -> None:
        super().__init__()
        self.variables = variables
        self.limit = limit
        self.solutions: list[dict[str, int]] = []

    def OnSolutionCallback(self) -> None:
        self.solutions.append({key: int(self.Value(var)) for key, var in self.variables.items()})
        if len(self.solutions) >= self.limit:
            self.StopSearch()


def add_rule_constraint(model: cp_model.CpModel, out: Any, left: Any, right: Any, rule: str) -> None:
    if rule == "add":
        model.Add(out == left + right)
    elif rule == "sub_ab":
        model.Add(out == left - right)
    elif rule == "sub_ba":
        model.Add(out == right - left)
    elif rule == "abs_diff":
        diff = model.NewIntVar(-99, 99, "diff")
        abs_value = model.NewIntVar(0, 99, "abs_diff")
        model.Add(diff == left - right)
        model.AddAbsEquality(abs_value, diff)
        model.Add(out == abs_value)
    elif rule == "mul":
        product_value = model.NewIntVar(0, 9801, "product")
        model.AddMultiplicationEquality(product_value, [left, right])
        model.Add(out == product_value)
    else:
        raise KeyError(rule)


def encoded_number(token: str, digit_vars: dict[str, Any]) -> Any:
    return 10 * digit_vars[token[0]] + digit_vars[token[1]]


def encoded_output_value(token: str, digit_vars: dict[str, Any]) -> Any:
    total = 0
    width = len(token)
    for index, char in enumerate(token):
        total += (10 ** (width - index - 1)) * digit_vars[char]
    return total


def solve_cryptarithm_assignment(
    examples: list[tuple[str, str]],
    query: str,
    operator_rules: dict[str, str],
    *,
    max_solutions: int,
    solver_time_limit_s: float,
) -> list[dict[str, int]]:
    symbols: set[str] = set()
    parsed_examples: list[tuple[str, str, str, str]] = []
    for lhs, rhs in examples:
        parsed = split_encoded_binary_expr(lhs)
        if parsed is None:
            return []
        left, op, right = parsed
        if op not in operator_rules:
            return []
        symbols.update(left)
        symbols.update(right)
        symbols.update(str(rhs))
        parsed_examples.append((left, op, right, str(rhs)))
    parsed_query = split_encoded_binary_expr(query)
    if parsed_query is None:
        return []
    q_left, q_op, q_right = parsed_query
    if q_op not in operator_rules:
        return []
    symbols.update(q_left)
    symbols.update(q_right)
    if len(symbols) > 10:
        return []

    model = cp_model.CpModel()
    digit_vars = {char: model.NewIntVar(0, 9, "d_" + str(ord(char))) for char in sorted(symbols)}
    model.AddAllDifferent(list(digit_vars.values()))
    for left, op, right, rhs in parsed_examples:
        left_value = encoded_number(left, digit_vars)
        right_value = encoded_number(right, digit_vars)
        output_value = encoded_output_value(rhs, digit_vars)
        add_rule_constraint(model, output_value, left_value, right_value, operator_rules[op])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time_limit_s
    solver.parameters.num_search_workers = 1
    collector = SolutionCollector(digit_vars, max_solutions)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    return collector.solutions


def prediction_from_solution(solution: dict[str, int], query: str, rule: str) -> str:
    parsed = split_encoded_binary_expr(query)
    if parsed is None:
        return ""
    left, _, right = parsed
    inverse = {digit: char for char, digit in solution.items()}
    left_value = 10 * solution[left[0]] + solution[left[1]]
    right_value = 10 * solution[right[0]] + solution[right[1]]
    result_value = apply_arithmetic_rule(rule, left_value, right_value)
    if result_value < 0:
        return ""
    digits = str(result_value)
    chars: list[str] = []
    for digit in digits:
        char = inverse.get(int(digit))
        if char is None:
            return ""
        chars.append(char)
    return "".join(chars)


def symbolic_cryptarithm_candidates(
    examples: list[tuple[str, str]],
    query: str,
    *,
    max_operator_symbols: int,
    max_solutions_per_assignment: int,
    solver_time_limit_s: float,
) -> list[dict[str, Any]]:
    parsed_examples: list[tuple[str, str, str]] = []
    operator_symbols: set[str] = set()
    for lhs, _rhs in examples:
        parsed = split_encoded_binary_expr(lhs)
        if parsed is None:
            return [
                {
                    "rule_class": "symbolic_cryptarithm_gate",
                    "status": "abstain",
                    "prediction": "",
                    "proof": "one_or_more_examples_not_len5_binary_expr",
                    "candidate_program_count": 0,
                    "unique_prediction_count": 0,
                }
            ]
        left, op, right = parsed
        parsed_examples.append((left, op, right))
        operator_symbols.add(op)
    query_parsed = split_encoded_binary_expr(query)
    if query_parsed is None:
        return [
            {
                "rule_class": "symbolic_cryptarithm_gate",
                "status": "abstain",
                "prediction": "",
                "proof": "query_not_len5_binary_expr",
                "candidate_program_count": 0,
                "unique_prediction_count": 0,
            }
        ]
    _q_left, query_op, _q_right = query_parsed
    operator_symbols.add(query_op)
    if len(operator_symbols) > max_operator_symbols:
        return [
            {
                "rule_class": "symbolic_cryptarithm_gate",
                "status": "abstain",
                "prediction": "",
                "proof": f"operator_symbol_count_above_cap={len(operator_symbols)}",
                "candidate_program_count": 0,
                "unique_prediction_count": 0,
            }
        ]

    results_by_query_rule: dict[str, list[str]] = defaultdict(list)
    proof_bits_by_query_rule: dict[str, list[str]] = defaultdict(list)
    assignment_count_by_query_rule: Counter[str] = Counter()
    ordered_ops = sorted(operator_symbols)
    for rule_tuple in product(ARITHMETIC_RULES, repeat=len(ordered_ops)):
        operator_rules = dict(zip(ordered_ops, rule_tuple))
        query_rule = operator_rules[query_op]
        solutions = solve_cryptarithm_assignment(
            examples,
            query,
            operator_rules,
            max_solutions=max_solutions_per_assignment,
            solver_time_limit_s=solver_time_limit_s,
        )
        if not solutions:
            continue
        assignment_count_by_query_rule[query_rule] += 1
        for solution in solutions:
            prediction = prediction_from_solution(solution, query, query_rule)
            if prediction:
                results_by_query_rule[query_rule].append(prediction)
                if len(proof_bits_by_query_rule[query_rule]) < 5:
                    proof_bits_by_query_rule[query_rule].append(
                        "ops="
                        + json.dumps(operator_rules, sort_keys=True)
                        + "; map="
                        + json.dumps(dict(sorted(solution.items())), sort_keys=True)
                    )

    if not results_by_query_rule:
        return [
            {
                "rule_class": "symbolic_cryptarithm_operator_digits",
                "status": "abstain",
                "prediction": "",
                "proof": "no_operator_digit_solution",
                "candidate_program_count": 0,
                "unique_prediction_count": 0,
            }
        ]

    class_prefix = (
        "symbolic_cryptarithm_single_operator_digits_"
        if len(operator_symbols) == 1
        else "symbolic_cryptarithm_multi_operator_digits_"
    )
    outputs: list[dict[str, Any]] = []
    for query_rule in ARITHMETIC_RULES:
        predictions = results_by_query_rule.get(query_rule, [])
        if not predictions:
            continue
        proof = "; ".join(proof_bits_by_query_rule.get(query_rule, []))
        outputs.append(
            candidate_from_predictions(
                class_prefix + query_rule,
                predictions,
                proof,
                candidate_program_count=int(assignment_count_by_query_rule.get(query_rule, 0)),
            )
        )
    return outputs


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.input_csv:
        path = Path(args.input_csv)
        return [normalize_row(row) for row in read_csv(path)], {
            "source": "local_csv",
            "path": str(path),
            "sha256": sha256_file(path),
        }

    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v329_") as temp_name:
        path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(path)]
        meta = {
            "source": "hf_hub",
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": path.name,
            "sha256": sha256_file(path),
        }
    return rows, meta


def load_v324_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v324_equation_expanded_solver_gate_v1":
        raise RuntimeError("unexpected V324 schema: " + str(payload.get("schema_version")))
    if int(payload.get("conflict_count", -1)) != 0:
        raise RuntimeError("V324 has conflicts: " + str(payload.get("conflict_count")))
    if int(payload.get("accepted_candidate_count", -1)) < 4:
        raise RuntimeError("V324 accepted candidate count below expected floor")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V329 SYMBOLIC CRYPTARITHM GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("input_csv =", args.input_csv or "", flush=True)
    print("v324_manifest_json =", args.v324_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("target_new_symbolic_gain =", args.target_new_symbolic_gain, flush=True)
    print("max_operator_symbols =", args.max_operator_symbols, flush=True)
    print("solver_time_limit_s =", args.solver_time_limit_s, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, input_meta = load_rows(args)
    input_meta["rows"] = len(rows)
    v324_manifest = load_v324_manifest(args.v324_manifest_json)

    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    baseline_family = family_counts(rows, "prediction")
    baseline_equation_correct = int(baseline_family.get("equation_transform", {}).get("correct", 0))
    baseline_bit_correct = int(baseline_family.get("bit_manipulation", {}).get("correct", 0))
    print("baseline_family_counts =", json.dumps(baseline_family, sort_keys=True), flush=True)

    symbolic_misses = []
    for row in rows:
        if row["family"] != "equation_transform" or row["correct_bool"] or row["truncated_bool"]:
            continue
        examples, query, parse_status = parse_alice_prompt(str(row["prompt"]))
        if parse_status == "ok" and classify_subtype(examples, query) == "equation_symbolic_punct":
            symbolic_misses.append((row, examples, query))
    print("symbolic_equation_miss_rows =", len(symbolic_misses), flush=True)

    audit_rows: list[dict[str, Any]] = []
    for index, (row, examples, query) in enumerate(symbolic_misses, start=1):
        if index == 1 or index % 10 == 0 or index == len(symbolic_misses):
            print(f"symbolic_cryptarithm_audit_progress = {index}/{len(symbolic_misses)}", flush=True)
        for result in symbolic_cryptarithm_candidates(
            examples,
            query,
            max_operator_symbols=args.max_operator_symbols,
            max_solutions_per_assignment=args.max_solutions_per_assignment,
            solver_time_limit_s=args.solver_time_limit_s,
        ):
            audit = build_audit_row(row, result, examples, query)
            audit["candidate_source"] = "v329_symbolic_cryptarithm_cp_sat"
            audit_rows.append(audit)

    summary_rows = summarize_rule_classes(audit_rows)
    accepted, conflicts = accepted_no_loss_candidates(audit_rows)
    new_accepted_ids = [str(row["id"]) for row in accepted]
    v324_ids = [str(item) for item in v324_manifest.get("accepted_candidate_ids", [])]
    duplicate_with_v324 = sorted(set(new_accepted_ids) & set(v324_ids))
    if duplicate_with_v324:
        raise RuntimeError("V329 accepted ids overlap V324 ids: " + json.dumps(duplicate_with_v324))

    v324_gain = int(v324_manifest.get("accepted_candidate_count", 0))
    combined_gain = v324_gain + len(accepted)
    projected_equation_correct = baseline_equation_correct + combined_gain
    projected_weak_correct = sum(1 for row in rows if row["correct_bool"]) + combined_gain

    outputs = {
        "audit_csv": args.output_dir / "v329_symbolic_cryptarithm_audit.csv",
        "rule_summary_csv": args.output_dir / "v329_symbolic_cryptarithm_rule_summary.csv",
        "accepted_candidates_csv": args.output_dir / "v329_symbolic_cryptarithm_accepted_candidates.csv",
        "conflicts_csv": args.output_dir / "v329_symbolic_cryptarithm_conflicts.csv",
        "manifest_json": args.output_dir / "v329_symbolic_cryptarithm_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, EXTRA_AUDIT_COLUMNS)
    write_csv(outputs["rule_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["accepted_candidates_csv"], accepted, EXTRA_AUDIT_COLUMNS)
    write_csv(outputs["conflicts_csv"], conflicts, ["id", "predictions", "rule_classes"])

    if len(accepted) >= args.target_new_symbolic_gain and not conflicts and baseline_bit_correct >= args.bit_guardrail_min:
        decision = {
            "decision": "symbolic_cryptarithm_cpu_gate_found_new_signal",
            "reason": (
                f"new_symbolic_gain={len(accepted)}; v324_gain={v324_gain}; "
                f"combined_gain={combined_gain}; projected_equation_correct={projected_equation_correct}; "
                f"bit_correct_guardrail={baseline_bit_correct}>={args.bit_guardrail_min}; conflicts={len(conflicts)}"
            ),
            "next_action": "Build V330 distillation rows for this cryptarithm rule, then run tokenization/no-regression gates before any HF job.",
        }
    else:
        decision = {
            "decision": "symbolic_cryptarithm_signal_blocked",
            "reason": (
                f"new_symbolic_gain={len(accepted)}; target={args.target_new_symbolic_gain}; "
                f"conflicts={len(conflicts)}; bit_correct={baseline_bit_correct}; projected_equation_correct={projected_equation_correct}"
            ),
            "next_action": "Do not launch HF from V329; inspect symbolic misses or expand the constraint DSL.",
        }

    manifest = {
        "schema_version": "kg1_v329_symbolic_cryptarithm_gate_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "v324_manifest_json": str(args.v324_manifest_json),
        "v324_manifest_sha256": sha256_file(args.v324_manifest_json),
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "baseline_family_counts": baseline_family,
        "symbolic_equation_miss_rows": len(symbolic_misses),
        "rule_summary": summary_rows,
        "new_accepted_candidate_count": len(accepted),
        "new_accepted_candidate_ids": new_accepted_ids,
        "v324_accepted_candidate_count": v324_gain,
        "v324_accepted_candidate_ids": v324_ids,
        "combined_accepted_candidate_count": combined_gain,
        "combined_accepted_candidate_ids": v324_ids + new_accepted_ids,
        "projected_equation_correct": projected_equation_correct,
        "projected_weak_correct_if_equation_only": projected_weak_correct,
        "conflict_count": len(conflicts),
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("rule_summary_top =", json.dumps(summary_rows[:12], indent=2, sort_keys=True), flush=True)
    print("new_accepted_candidate_count =", len(accepted), flush=True)
    print("new_accepted_candidate_ids =", json.dumps(new_accepted_ids, sort_keys=True), flush=True)
    print("combined_accepted_candidate_count =", combined_gain, flush=True)
    print("projected_equation_correct =", projected_equation_correct, flush=True)
    print("projected_weak_correct_if_equation_only =", projected_weak_correct, flush=True)
    print("conflict_count =", len(conflicts), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V329 SYMBOLIC CRYPTARITHM GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    examples = [
        ("#[%[<", "#{^<"),
        ("[)%![", "!)?("),
        ("))%??", "[(`"),
        (")#%`?", "{([<"),
    ]
    results = symbolic_cryptarithm_candidates(
        examples,
        "(<%))",
        max_operator_symbols=2,
        max_solutions_per_assignment=3,
        solver_time_limit_s=0.5,
    )
    by_class = {row["rule_class"]: row for row in results}
    result = by_class.get("symbolic_cryptarithm_single_operator_digits_mul")
    if not result or result["status"] != "candidate" or result["prediction"] != "?()<":
        raise AssertionError(results)
    print("v329_symbolic_cryptarithm_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--v324-manifest-json", type=Path, default=DEFAULT_V324_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v329_symbolic_cryptarithm_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-new-symbolic-gain", type=int, default=1)
    parser.add_argument("--bit-guardrail-min", type=int, default=136)
    parser.add_argument("--max-operator-symbols", type=int, default=4)
    parser.add_argument("--max-solutions-per-assignment", type=int, default=3)
    parser.add_argument("--solver-time-limit-s", type=float, default=0.08)
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

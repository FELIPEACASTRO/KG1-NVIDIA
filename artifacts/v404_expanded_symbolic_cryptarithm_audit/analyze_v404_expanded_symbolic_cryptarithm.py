#!/usr/bin/env python3
"""V404 expanded symbolic cryptarithm CPU audit.

This is a solver-first, abstain-by-default experiment for the remaining
`equation_symbolic_punct` misses. It extends the V329 idea with a wider but
still tiny SyGuS-style grammar over encoded two-digit expressions:

    <xy><op><uv> -> encoded_decimal(f(xy, uv))

Weak labels are used only as an audit brake. This script does not train,
package, submit, or call GPU services.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ortools is required for V404") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402
from scripts.run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    classify_subtype,
    parse_alice_prompt,
)


INPUT_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
OUT_DIR = REPO_ROOT / "artifacts/v404_expanded_symbolic_cryptarithm_audit/20260514T_v404_symbolic_cryptarithm"

RULES = (
    "add",
    "sub_ab",
    "sub_ba",
    "abs_diff",
    "mul",
    "div_ab",
    "div_ba",
    "mod_ab",
    "mod_ba",
    "concat_ab",
    "concat_ba",
    "max",
    "min",
)


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


def split_expr(text: str) -> tuple[str, str, str] | None:
    value = str(text or "").strip()
    if len(value) != 5:
        return None
    return value[:2], value[2], value[3:]


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
        product = model.NewIntVar(0, 9801, "mul")
        model.AddMultiplicationEquality(product, [left, right])
        model.Add(out == product)
    elif rule == "div_ab":
        model.Add(right != 0)
        model.AddDivisionEquality(out, left, right)
    elif rule == "div_ba":
        model.Add(left != 0)
        model.AddDivisionEquality(out, right, left)
    elif rule == "mod_ab":
        model.Add(right != 0)
        model.AddModuloEquality(out, left, right)
    elif rule == "mod_ba":
        model.Add(left != 0)
        model.AddModuloEquality(out, right, left)
    elif rule == "concat_ab":
        model.Add(out == left * 100 + right)
    elif rule == "concat_ba":
        model.Add(out == right * 100 + left)
    elif rule == "max":
        model.AddMaxEquality(out, [left, right])
    elif rule == "min":
        model.AddMinEquality(out, [left, right])
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


class Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: dict[str, Any], limit: int) -> None:
        super().__init__()
        self.variables = variables
        self.limit = limit
        self.solutions: list[dict[str, int]] = []

    def OnSolutionCallback(self) -> None:
        self.solutions.append({key: int(self.Value(var)) for key, var in self.variables.items()})
        if len(self.solutions) >= self.limit:
            self.StopSearch()


def apply_rule(rule: str, left: int, right: int) -> int | None:
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
    if rule == "div_ab":
        return None if right == 0 else left // right
    if rule == "div_ba":
        return None if left == 0 else right // left
    if rule == "mod_ab":
        return None if right == 0 else left % right
    if rule == "mod_ba":
        return None if left == 0 else right % left
    if rule == "concat_ab":
        return left * 100 + right
    if rule == "concat_ba":
        return right * 100 + left
    if rule == "max":
        return max(left, right)
    if rule == "min":
        return min(left, right)
    raise KeyError(rule)


def encode_decimal(value: int, solution: dict[str, int], widths: set[int]) -> set[str]:
    if value < 0:
        return set()
    inverse = {digit: char for char, digit in solution.items()}
    outputs: set[str] = set()
    raw = str(value)
    for width in sorted(widths | {len(raw)}):
        if width <= 0 or width > 4:
            continue
        digits = raw.zfill(width)
        chars: list[str] = []
        ok = True
        for digit in digits:
            char = inverse.get(int(digit))
            if char is None:
                ok = False
                break
            chars.append(char)
        if ok:
            outputs.add("".join(chars))
    return outputs


def solve_assignment(
    examples: list[tuple[str, str]],
    query: str,
    operator_rules: dict[str, str],
    *,
    max_solutions: int,
    time_limit_s: float,
) -> list[dict[str, int]]:
    symbols: set[str] = set()
    parsed: list[tuple[str, str, str, str]] = []
    for lhs, rhs in examples:
        item = split_expr(lhs)
        if item is None:
            return []
        left, op, right = item
        if op not in operator_rules:
            return []
        symbols.update(left)
        symbols.update(right)
        symbols.update(str(rhs))
        parsed.append((left, op, right, str(rhs)))
    query_item = split_expr(query)
    if query_item is None:
        return []
    q_left, q_op, q_right = query_item
    if q_op not in operator_rules:
        return []
    symbols.update(q_left)
    symbols.update(q_right)
    if len(symbols) > 10:
        return []

    model = cp_model.CpModel()
    digit_vars = {char: model.NewIntVar(0, 9, "d_" + str(ord(char))) for char in sorted(symbols)}
    model.AddAllDifferent(list(digit_vars.values()))
    for left, op, right, rhs in parsed:
        left_value = encoded_number(left, digit_vars)
        right_value = encoded_number(right, digit_vars)
        output_value = encoded_output_value(rhs, digit_vars)
        add_rule_constraint(model, output_value, left_value, right_value, operator_rules[op])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1
    collector = Collector(digit_vars, max_solutions)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    return collector.solutions


def candidate_for_row(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    query_item = split_expr(query)
    if query_item is None:
        return {"status": "abstain", "prediction": "", "rule_class": "symbolic_cryptarithm_v404_parse_gate", "proof": "query_not_len5"}
    _, query_op, _ = query_item
    ops = sorted({split_expr(lhs)[1] for lhs, _ in examples if split_expr(lhs) is not None} | {query_op})
    if len(ops) > 3:
        return {"status": "abstain", "prediction": "", "rule_class": "symbolic_cryptarithm_v404_operator_cap", "proof": f"ops={''.join(ops)}"}
    rhs_widths = {len(str(rhs)) for _, rhs in examples}
    predictions_by_rule: dict[str, set[str]] = defaultdict(set)
    proof_bits: dict[str, list[str]] = defaultdict(list)
    program_count = 0
    for rule_tuple in itertools.product(RULES, repeat=len(ops)):
        operator_rules = dict(zip(ops, rule_tuple))
        query_rule = operator_rules[query_op]
        solutions = solve_assignment(examples, query, operator_rules, max_solutions=3, time_limit_s=0.05)
        if not solutions:
            continue
        program_count += 1
        q_left, _, q_right = query_item
        for solution in solutions:
            left_value = 10 * solution[q_left[0]] + solution[q_left[1]]
            right_value = 10 * solution[q_right[0]] + solution[q_right[1]]
            value = apply_rule(query_rule, left_value, right_value)
            if value is None:
                continue
            predictions_by_rule[query_rule].update(encode_decimal(value, solution, rhs_widths))
            if len(proof_bits[query_rule]) < 3:
                proof_bits[query_rule].append(
                    "ops=" + json.dumps(operator_rules, sort_keys=True) + ";map=" + json.dumps(solution, sort_keys=True)
                )
    candidates: list[tuple[str, str]] = []
    for rule, predictions in sorted(predictions_by_rule.items()):
        if len(predictions) == 1:
            candidates.append((rule, next(iter(predictions))))
    unique_predictions = sorted({prediction for _, prediction in candidates if prediction})
    if len(unique_predictions) == 1:
        winning_rules = [rule for rule, prediction in candidates if prediction == unique_predictions[0]]
        return {
            "status": "candidate",
            "prediction": unique_predictions[0],
            "rule_class": "symbolic_cryptarithm_v404_" + "+".join(winning_rules[:4]),
            "proof": "program_count=" + str(program_count) + ";" + ";".join(proof_bits[winning_rules[0]][:2]),
        }
    return {
        "status": "abstain",
        "prediction": "",
        "rule_class": "symbolic_cryptarithm_v404_ambiguous",
        "proof": f"program_count={program_count}; unique_prediction_count={len(unique_predictions)}",
    }


def main() -> int:
    rows = read_csv(INPUT_CSV)
    audit_rows: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    for row in rows:
        if classify_puzzle(row.get("prompt", "")) != "equation_transform":
            continue
        baseline_correct = verify_answer(row["answer"], row["prediction"])
        if baseline_correct:
            continue
        examples, query, parse_status = parse_alice_prompt(row["prompt"])
        if parse_status != "ok" or classify_subtype(examples, query) != "equation_symbolic_punct":
            continue
        result = candidate_for_row(examples, query)
        candidate_correct = verify_answer(row["answer"], result["prediction"])
        item = {
            "id": row["id"],
            "answer": row["answer"],
            "baseline_prediction": row["prediction"],
            "query": query,
            "example_count": len(examples),
            "status": result["status"],
            "rule_class": result["rule_class"],
            "prediction": result["prediction"],
            "candidate_correct": candidate_correct,
            "incorrect_by_weak_label": result["status"] == "candidate" and not candidate_correct,
            "verified_by_weak_label": result["status"] == "candidate" and candidate_correct,
            "proof": result["proof"],
        }
        audit_rows.append(item)
        policy_counts[str(result["status"])] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped[row["rule_class"]].append(row)
    accepted_classes = {
        rule
        for rule, items in grouped.items()
        if any(item["verified_by_weak_label"] for item in items)
        and not any(item["incorrect_by_weak_label"] for item in items)
    }
    for row in audit_rows:
        row["promotable_after_class_gate"] = row["rule_class"] in accepted_classes and row["status"] == "candidate"

    accepted = [row for row in audit_rows if row["promotable_after_class_gate"]]
    incorrect = [row for row in audit_rows if row["incorrect_by_weak_label"]]
    rule_summary = []
    for rule, items in sorted(grouped.items()):
        rule_summary.append(
            {
                "rule_class": rule,
                "rows": len(items),
                "candidate_rows": sum(item["status"] == "candidate" for item in items),
                "verified_candidates": sum(item["verified_by_weak_label"] for item in items),
                "incorrect_candidates": sum(item["incorrect_by_weak_label"] for item in items),
                "promotable_after_class_gate": rule in accepted_classes,
            }
        )
    rule_summary.sort(key=lambda x: (x["verified_candidates"], -x["incorrect_candidates"]), reverse=True)

    columns = [
        "id",
        "answer",
        "baseline_prediction",
        "query",
        "example_count",
        "status",
        "rule_class",
        "prediction",
        "candidate_correct",
        "incorrect_by_weak_label",
        "verified_by_weak_label",
        "promotable_after_class_gate",
        "proof",
    ]
    write_csv(OUT_DIR / "v404_expanded_symbolic_cryptarithm_audit.csv", audit_rows, columns)
    write_csv(OUT_DIR / "v404_expanded_symbolic_cryptarithm_accepted.csv", accepted, columns)
    write_csv(
        OUT_DIR / "v404_expanded_symbolic_cryptarithm_rule_summary.csv",
        rule_summary,
        ["rule_class", "rows", "candidate_rows", "verified_candidates", "incorrect_candidates", "promotable_after_class_gate"],
    )
    manifest = {
        "schema_version": "kg1_v404_expanded_symbolic_cryptarithm_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV),
        "audited_symbolic_misses": len(audit_rows),
        "policy_counts": dict(sorted(policy_counts.items())),
        "rule_summary": rule_summary,
        "accepted_candidate_count": len(accepted),
        "incorrect_candidate_count": len(incorrect),
        "decision": "new_symbolic_candidates_found" if accepted else "no_new_symbolic_candidates",
    }
    write_json(OUT_DIR / "v404_expanded_symbolic_cryptarithm_manifest.json", manifest)
    report = [
        "# V404 Expanded Symbolic Cryptarithm Audit",
        "",
        f"- Audited symbolic misses: `{len(audit_rows)}`",
        f"- Accepted candidates: `{len(accepted)}`",
        f"- Incorrect candidates before class gate: `{len(incorrect)}`",
        "",
        "## Accepted",
        "",
    ]
    for row in accepted:
        report.append(
            f"- `{row['id']}`: `{row['baseline_prediction']}` -> `{row['prediction']}` "
            f"via `{row['rule_class']}`"
        )
    report.extend(["", "## Decision", "", manifest["decision"], ""])
    (OUT_DIR / "V404_EXPANDED_SYMBOLIC_CRYPTARITHM_AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V431 signed symbolic cryptarithm CPU gate.

This gate tests a materially different cryptarithm class from V329/V420:

* two encoded two-digit operands separated by an operator symbol;
* per-row bijective symbol->digit assignment;
* operator symbols mapped to arithmetic rules;
* RHS may contain a literal leading negative sign;
* prediction formatting tries minimal and fixed digit widths.

Weak labels are used only after prediction as an audit brake. This script does
not train, launch GPU, package, or submit.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ortools is required for V431 signed cryptarithm gate") from exc


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V414_ACCEPTED_CSV = (
    ROOT
    / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate/"
    / "v414_accepted_union.csv"
)
OUT_DIR = ROOT / "artifacts/v431_signed_cryptarithm_gate/20260515T_v431_signed_cryptarithm"

EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}

ARITHMETIC_RULES = ("add", "sub_ab", "sub_ba", "abs_diff", "mul")

AUDIT_COLUMNS = [
    "id",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "candidate_predictions",
    "candidate_count",
    "candidate_correct",
    "changed",
    "already_known_v414",
    "status",
    "reason",
    "operator_symbol_count",
    "solution_program_count",
    "proof",
]

DECISION_COLUMNS = [
    "id",
    "answer",
    "old_prediction",
    "new_prediction",
    "old_correct",
    "new_correct",
    "accepted",
    "rejection_reason",
    "candidate_count",
    "proof",
]


@dataclass(frozen=True)
class Candidate:
    prediction: str
    proof: str
    program_count: int


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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or classify_puzzle(str(row.get("prompt", ""))))


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    for raw in str(prompt or "").splitlines():
        if " = " not in raw:
            continue
        lhs, rhs = raw.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query = match.group(1).strip() if match else None
    return examples, query


def split_binary_expr(token: str | None) -> tuple[str, str, str] | None:
    text = str(token or "").strip()
    if len(text) != 5:
        return None
    return text[:2], text[2], text[3:]


def signed_rhs_digits(rhs: str) -> tuple[int, str] | None:
    text = str(rhs or "")
    if not text:
        return None
    if text.startswith("-") and len(text) > 1:
        return -1, text[1:]
    return 1, text


def encoded_number(token: str, digit_vars: dict[str, Any]) -> Any:
    return 10 * digit_vars[token[0]] + digit_vars[token[1]]


def encoded_digits_value(token: str, digit_vars: dict[str, Any]) -> Any:
    total = 0
    width = len(token)
    for index, char in enumerate(token):
        total += (10 ** (width - index - 1)) * digit_vars[char]
    return total


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


def solve_signed_assignments(
    examples: list[tuple[str, str]],
    query: str,
    operator_rules: dict[str, str],
    *,
    max_solutions: int,
    solver_time_limit_s: float,
) -> list[dict[str, int]]:
    symbols: set[str] = set()
    parsed_examples: list[tuple[str, str, str, int, str]] = []
    for lhs, rhs in examples:
        parsed_lhs = split_binary_expr(lhs)
        parsed_rhs = signed_rhs_digits(rhs)
        if parsed_lhs is None or parsed_rhs is None:
            return []
        left, op, right = parsed_lhs
        sign, rhs_digits = parsed_rhs
        if op not in operator_rules or not rhs_digits:
            return []
        symbols.update(left)
        symbols.update(right)
        symbols.update(rhs_digits)
        parsed_examples.append((left, op, right, sign, rhs_digits))

    parsed_query = split_binary_expr(query)
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

    for left, op, right, sign, rhs_digits in parsed_examples:
        left_value = encoded_number(left, digit_vars)
        right_value = encoded_number(right, digit_vars)
        decoded_rhs_abs = encoded_digits_value(rhs_digits, digit_vars)
        signed_out = model.NewIntVar(-9801, 9801, "signed_out")
        if sign < 0:
            model.Add(signed_out == -decoded_rhs_abs)
        else:
            model.Add(signed_out == decoded_rhs_abs)
        add_rule_constraint(model, signed_out, left_value, right_value, operator_rules[op])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time_limit_s
    solver.parameters.num_search_workers = 1
    collector = SolutionCollector(digit_vars, max_solutions)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    return collector.solutions


def encode_abs_value(value: int, inverse: dict[int, str], width: int | None) -> str | None:
    digits = str(abs(value))
    if width is not None:
        if len(digits) > width:
            return None
        digits = digits.zfill(width)
    chars: list[str] = []
    for digit in digits:
        char = inverse.get(int(digit))
        if char is None:
            return None
        chars.append(char)
    return "".join(chars)


def prediction_from_solution(solution: dict[str, int], query: str, rule: str, width: int | None) -> str | None:
    parsed = split_binary_expr(query)
    if parsed is None:
        return None
    left, _op, right = parsed
    inverse = {digit: char for char, digit in solution.items()}
    left_value = 10 * solution[left[0]] + solution[left[1]]
    right_value = 10 * solution[right[0]] + solution[right[1]]
    result_value = apply_arithmetic_rule(rule, left_value, right_value)
    encoded = encode_abs_value(result_value, inverse, width)
    if encoded is None:
        return None
    return "-" + encoded if result_value < 0 else encoded


def signed_cryptarithm_candidates(
    examples: list[tuple[str, str]],
    query: str,
    *,
    max_operator_symbols: int = 4,
    max_solutions_per_assignment: int = 4,
    solver_time_limit_s: float = 0.05,
) -> tuple[list[Candidate], dict[str, Any]]:
    operator_symbols: set[str] = set()
    rhs_widths: set[int] = set()
    for lhs, rhs in examples:
        parsed_lhs = split_binary_expr(lhs)
        parsed_rhs = signed_rhs_digits(rhs)
        if parsed_lhs is None or parsed_rhs is None:
            return [], {"status": "abstain", "reason": "unparseable_example"}
        _left, op, _right = parsed_lhs
        _sign, rhs_digits = parsed_rhs
        operator_symbols.add(op)
        rhs_widths.add(len(rhs_digits))

    query_parsed = split_binary_expr(query)
    if query_parsed is None:
        return [], {"status": "abstain", "reason": "unparseable_query"}
    operator_symbols.add(query_parsed[1])
    if len(operator_symbols) > max_operator_symbols:
        return [], {
            "status": "abstain",
            "reason": "operator_symbol_count_above_cap",
            "operator_symbol_count": len(operator_symbols),
        }

    ordered_ops = sorted(operator_symbols)
    predictions_by_key: dict[str, list[str]] = defaultdict(list)
    proof_by_key: dict[str, list[str]] = defaultdict(list)
    program_count_by_key: Counter[str] = Counter()
    widths: list[int | None] = [None] + sorted(width for width in rhs_widths if 1 <= width <= 4)

    for rule_tuple in itertools.product(ARITHMETIC_RULES, repeat=len(ordered_ops)):
        operator_rules = dict(zip(ordered_ops, rule_tuple))
        query_rule = operator_rules[query_parsed[1]]
        solutions = solve_signed_assignments(
            examples,
            query,
            operator_rules,
            max_solutions=max_solutions_per_assignment,
            solver_time_limit_s=solver_time_limit_s,
        )
        if not solutions:
            continue
        for solution in solutions:
            for width in widths:
                prediction = prediction_from_solution(solution, query, query_rule, width)
                if not prediction:
                    continue
                width_label = "minimal" if width is None else f"width{width}"
                key = f"{query_rule}__{width_label}"
                predictions_by_key[key].append(prediction)
                program_count_by_key[key] += 1
                if len(proof_by_key[key]) < 3:
                    proof_by_key[key].append(
                        "ops="
                        + json.dumps(operator_rules, sort_keys=True)
                        + "; map="
                        + json.dumps(dict(sorted(solution.items())), sort_keys=True)
                    )

    candidates: list[Candidate] = []
    for key, predictions in sorted(predictions_by_key.items()):
        unique = sorted(set(predictions))
        if len(unique) == 1:
            candidates.append(
                Candidate(
                    prediction=unique[0],
                    proof=f"{key}; " + " | ".join(proof_by_key[key]),
                    program_count=int(program_count_by_key[key]),
                )
            )
    if not candidates:
        return [], {
            "status": "abstain",
            "reason": "no_unique_signed_cryptarithm_candidate",
            "operator_symbol_count": len(operator_symbols),
        }
    return candidates, {
        "status": "candidate",
        "reason": "signed_cryptarithm_unique_by_key",
        "operator_symbol_count": len(operator_symbols),
        "candidate_count": len(candidates),
    }


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        prediction = str(row.get(prediction_key, row.get("prediction", "")))
        correct = verify_answer(str(row.get("answer", "")), prediction)
        total += int(correct)
        truncated += int(truthy(row.get("truncated", False)))
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
    return {
        "correct": total,
        "truncated": truncated,
        "families": {family: dict(counter) for family, counter in sorted(families.items())},
    }


def assert_expected_baseline(score: dict[str, Any]) -> None:
    observed = {
        "correct": int(score["correct"]),
        "equation_transform_correct": int(score["families"]["equation_transform"]["correct"]),
        "bit_manipulation_correct": int(score["families"]["bit_manipulation"]["correct"]),
        "truncated": int(score["truncated"]),
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError("unexpected baseline score: " + json.dumps(observed, sort_keys=True))


def render_report(path: Path, manifest: dict[str, Any], accepted: list[dict[str, Any]]) -> None:
    lines = [
        "# V431 Signed Cryptarithm Gate",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "CPU-only gate for signed/padded symbolic cryptarithm rows. It is diagnostic only and not a submit artifact.",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline V291/V290 | V431 projection | Delta |",
        "|---|---:|---:|---:|",
        f"| Total weak correct | `192/315` | `{manifest['projection']['total_correct']}/315` | `{manifest['projection']['total_delta']}` |",
        f"| equation_transform | `56/155` | `{manifest['projection']['equation_transform_correct']}/155` | `{manifest['projection']['equation_delta']}` |",
        f"| bit_manipulation | `136/160` | `{manifest['projection']['bit_manipulation_correct']}/160` | `{manifest['projection']['bit_delta']}` |",
        f"| Truncated | `0` | `{manifest['projection']['truncated']}` | `{manifest['projection']['truncated_delta']}` |",
        "",
        "## Gate Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Equation rows audited | `{manifest['audit_rows']}` |",
        f"| Rows with unique candidate | `{manifest['unique_candidate_rows']}` |",
        f"| Ambiguous candidate rows blocked | `{manifest['ambiguous_rows']}` |",
        f"| Accepted total gains vs baseline | `{manifest['accepted_total_gains']}` |",
        f"| Accepted new gains beyond V414 | `{manifest['accepted_new_gains']}` |",
        f"| Conflict rows blocked | `{manifest['conflict_rows']}` |",
        "",
        "## Accepted Rows",
        "",
        "| id | old_prediction | new_prediction | answer |",
        "|---|---|---|---|",
    ]
    if accepted:
        for row in accepted:
            lines.append(
                f"| `{row['id']}` | `{row['old_prediction']}` | `{row['new_prediction']}` | `{row['answer']}` |"
            )
    else:
        lines.append("| none | none | none | none |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "`hf_gpu_allowed = false` unless this CPU gate beats the adapter-only baseline with no-loss rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=== V431 SIGNED CRYPTARITHM GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_csv =", BASELINE_CSV, flush=True)
    print("v414_accepted_csv =", V414_ACCEPTED_CSV, flush=True)
    print("output_dir =", OUT_DIR, flush=True)

    rows = read_csv(BASELINE_CSV)
    known_v414 = {row["id"] for row in read_csv(V414_ACCEPTED_CSV)}
    baseline_score = score_rows(rows, "prediction")
    assert_expected_baseline(baseline_score)
    print("baseline_score =", json.dumps(baseline_score, sort_keys=True), flush=True)

    audit_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    projected_rows = [dict(row) for row in rows]

    equation_rows = [row for row in projected_rows if family_for(row) == "equation_transform"]
    for index, row in enumerate(equation_rows, start=1):
        if index == 1 or index % 25 == 0 or index == len(equation_rows):
            print(f"signed_cryptarithm_progress = {index}/{len(equation_rows)}", flush=True)
        examples, query = parse_examples(str(row.get("prompt", "")))
        if not query:
            continue
        candidates, meta = signed_cryptarithm_candidates(examples, query)
        predictions = sorted({candidate.prediction for candidate in candidates})
        baseline_correct = truthy(row.get("correct", ""))
        status = str(meta.get("status", "abstain"))
        reason = str(meta.get("reason", ""))
        if len(predictions) == 1:
            candidate_prediction = predictions[0]
            candidate_correct = verify_answer(str(row.get("answer", "")), candidate_prediction)
            changed = candidate_prediction != str(row.get("prediction", ""))
            if baseline_correct and not candidate_correct:
                status = "conflict"
                reason = "candidate_would_regress_baseline_correct"
                conflict_rows.append({"id": row["id"], "prediction": candidate_prediction, "answer": row["answer"]})
            elif (not baseline_correct) and candidate_correct:
                status = "accepted"
                reason = "no_loss_new_gain_known_v414" if row.get("id", "") in known_v414 else "no_loss_new_gain_new"
                row["v431_prediction"] = candidate_prediction
                accepted = {
                    "id": row["id"],
                    "answer": row["answer"],
                    "old_prediction": row.get("prediction", ""),
                    "new_prediction": candidate_prediction,
                    "old_correct": baseline_correct,
                    "new_correct": candidate_correct,
                    "accepted": True,
                    "rejection_reason": "",
                    "candidate_count": len(candidates),
                    "proof": candidates[0].proof[:700] if candidates else "",
                }
                accepted_rows.append(accepted)
                decision_rows.append(accepted)
            else:
                decision_rows.append(
                    {
                        "id": row["id"],
                        "answer": row["answer"],
                        "old_prediction": row.get("prediction", ""),
                        "new_prediction": candidate_prediction,
                        "old_correct": baseline_correct,
                        "new_correct": candidate_correct,
                        "accepted": False,
                        "rejection_reason": "no_gain_or_false_positive",
                        "candidate_count": len(candidates),
                        "proof": candidates[0].proof[:700] if candidates else "",
                    }
                )
        elif len(predictions) > 1:
            candidate_prediction = "|".join(predictions[:8])
            candidate_correct = False
            changed = True
            status = "ambiguous"
            reason = "multiple_unique_predictions"
        else:
            candidate_prediction = ""
            candidate_correct = False
            changed = False

        audit_rows.append(
            {
                "id": row.get("id", ""),
                "answer": row.get("answer", ""),
                "baseline_prediction": row.get("prediction", ""),
                "baseline_correct": baseline_correct,
                "candidate_predictions": candidate_prediction,
                "candidate_count": len(candidates),
                "candidate_correct": candidate_correct,
                "changed": changed,
                "already_known_v414": row.get("id", "") in known_v414,
                "status": status,
                "reason": reason,
                "operator_symbol_count": meta.get("operator_symbol_count", ""),
                "solution_program_count": sum(candidate.program_count for candidate in candidates),
                "proof": " || ".join(candidate.proof for candidate in candidates[:4])[:900],
            }
        )

    for row in projected_rows:
        if "v431_prediction" not in row:
            row["v431_prediction"] = row.get("prediction", "")
    projection_score = score_rows(projected_rows, "v431_prediction")
    projection = {
        "total_correct": int(projection_score["correct"]),
        "total_delta": int(projection_score["correct"]) - EXPECTED_BASELINE["correct"],
        "equation_transform_correct": int(projection_score["families"]["equation_transform"]["correct"]),
        "equation_delta": int(projection_score["families"]["equation_transform"]["correct"])
        - EXPECTED_BASELINE["equation_transform_correct"],
        "bit_manipulation_correct": int(projection_score["families"]["bit_manipulation"]["correct"]),
        "bit_delta": int(projection_score["families"]["bit_manipulation"]["correct"])
        - EXPECTED_BASELINE["bit_manipulation_correct"],
        "truncated": int(projection_score["truncated"]),
        "truncated_delta": int(projection_score["truncated"]) - EXPECTED_BASELINE["truncated"],
    }
    status_counts = Counter(str(row.get("status", "")) for row in audit_rows)
    new_accepted_rows = [row for row in accepted_rows if row["id"] not in known_v414]
    known_accepted_rows = [row for row in accepted_rows if row["id"] in known_v414]
    manifest = {
        "schema_version": "kg1_v431_signed_cryptarithm_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline_csv": str(BASELINE_CSV),
        "v414_accepted_csv": str(V414_ACCEPTED_CSV),
        "baseline_score": baseline_score,
        "projection": projection,
        "audit_rows": len(audit_rows),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "unique_candidate_rows": int(status_counts.get("accepted", 0) + status_counts.get("candidate", 0)),
        "ambiguous_rows": int(status_counts.get("ambiguous", 0)),
        "accepted_total_gains": len(accepted_rows),
        "accepted_new_gains": len(new_accepted_rows),
        "accepted_known_v414_gains": len(known_accepted_rows),
        "accepted_ids": [row["id"] for row in accepted_rows],
        "accepted_new_ids": [row["id"] for row in new_accepted_rows],
        "accepted_known_v414_ids": [row["id"] for row in known_accepted_rows],
        "conflict_rows": len(conflict_rows),
        "hf_gpu_allowed": bool(
            projection["total_correct"] > EXPECTED_BASELINE["correct"]
            and projection["equation_transform_correct"] > EXPECTED_BASELINE["equation_transform_correct"]
            and projection["bit_manipulation_correct"] >= EXPECTED_BASELINE["bit_manipulation_correct"]
            and projection["truncated"] == 0
            and not conflict_rows
            and len(new_accepted_rows) > 0
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "v431_signed_cryptarithm_audit.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(OUT_DIR / "v431_signed_cryptarithm_decisions.csv", decision_rows, DECISION_COLUMNS)
    write_csv(OUT_DIR / "v431_signed_cryptarithm_accepted.csv", accepted_rows, DECISION_COLUMNS)
    write_csv(OUT_DIR / "v431_signed_cryptarithm_conflicts.csv", conflict_rows, ["id", "prediction", "answer"])
    write_json(OUT_DIR / "v431_signed_cryptarithm_manifest.json", manifest)
    render_report(OUT_DIR / "V431_SIGNED_CRYPTARITHM_GATE.md", manifest, accepted_rows)

    print("candidate_status_counts =", json.dumps(manifest["candidate_status_counts"], sort_keys=True), flush=True)
    print("projection =", json.dumps(projection, sort_keys=True), flush=True)
    print("accepted_ids =", json.dumps(manifest["accepted_ids"], sort_keys=True), flush=True)
    print("accepted_new_ids =", json.dumps(manifest["accepted_new_ids"], sort_keys=True), flush=True)
    print("accepted_known_v414_ids =", json.dumps(manifest["accepted_known_v414_ids"], sort_keys=True), flush=True)
    print("conflict_rows =", len(conflict_rows), flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", OUT_DIR / "v431_signed_cryptarithm_manifest.json", flush=True)
    print("=== V431 SIGNED CRYPTARITHM GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

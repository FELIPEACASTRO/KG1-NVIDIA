"""Label-free full-byte bit postprocessor for KG1 bit prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

BITS = 8
Vec = tuple[int, ...]


@dataclass(frozen=True)
class BitPostprocessDecision:
    prediction: str
    applied: bool
    rule: str
    proof: str


def parse_bit_problem(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    query = None
    for line in str(prompt or "").strip().splitlines():
        text = line.strip()
        pair = re.match(r"^([01]{8})\s*->\s*([01]{8})$", text)
        if pair:
            examples.append((pair.group(1), pair.group(2)))
            continue
        query_match = re.match(r".*(?:determine|find|output for)[:\s]+([01]{8})", text, re.I)
        if query_match:
            query = query_match.group(1)
    return examples, query


def to_vec(text: str) -> Vec:
    return tuple(1 if ch == "1" else 0 for ch in str(text).strip())


def from_vec(vec: Vec) -> str:
    return "".join(str(bit) for bit in vec)


def not_vec(a: Vec) -> Vec:
    return tuple(1 - x for x in a)


def and_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x & y for x, y in zip(a, b))


def or_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x | y for x, y in zip(a, b))


def xor_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x ^ y for x, y in zip(a, b))


def xnor_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(xor_vec(a, b))


def nand_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(and_vec(a, b))


def nor_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(or_vec(a, b))


def and_not_vec(a: Vec, b: Vec) -> Vec:
    return and_vec(a, not_vec(b))


def not_and_vec(a: Vec, b: Vec) -> Vec:
    return and_vec(not_vec(a), b)


def or_not_vec(a: Vec, b: Vec) -> Vec:
    return or_vec(a, not_vec(b))


def not_or_vec(a: Vec, b: Vec) -> Vec:
    return or_vec(not_vec(a), b)


BINARY_OPS: list[tuple[str, Callable[[Vec, Vec], Vec]]] = [
    ("AND", and_vec),
    ("OR", or_vec),
    ("XOR", xor_vec),
    ("XNOR", xnor_vec),
    ("NAND", nand_vec),
    ("NOR", nor_vec),
    ("AND_NOT", and_not_vec),
    ("NOT_AND", not_and_vec),
    ("OR_NOT", or_not_vec),
    ("NOT_OR", not_or_vec),
]

TERNARY_OPS: list[tuple[str, Callable[[Vec, Vec, Vec], Vec]]] = [
    ("PAR3", lambda a, b, c: xor_vec(xor_vec(a, b), c)),
    ("MAJ3", lambda a, b, c: tuple(1 if x + y + z >= 2 else 0 for x, y, z in zip(a, b, c))),
    ("CHO", lambda a, b, c: or_vec(and_vec(a, b), and_vec(not_vec(a), c))),
]


def transforms(vec: Vec) -> list[tuple[str, Vec]]:
    rows: list[tuple[str, Vec]] = [("ID", vec), ("NOT", not_vec(vec))]
    for k in range(1, BITS):
        rows.append((f"ROL{k}", vec[k:] + vec[:k]))
    for k in range(1, BITS):
        rows.append((f"ROR{k}", vec[-k:] + vec[:-k]))
    for k in range(1, BITS):
        rows.append((f"SHL{k}", vec[k:] + tuple(0 for _ in range(k))))
    for k in range(1, BITS):
        rows.append((f"SHR{k}", tuple(0 for _ in range(k)) + vec[:-k]))
    return rows


def solve_fullbyte(prompt: str) -> tuple[str | None, str, str]:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        return None, "parse_gate", "bit prompt was not parseable"
    inputs = [to_vec(inp) for inp, _ in examples]
    outputs = [to_vec(out) for _, out in examples]
    query_vec = to_vec(query)
    if any(len(vec) != BITS for vec in inputs + outputs + [query_vec]):
        return None, "width_gate", "bit width was not eight"

    all_inputs = inputs + [query_vec]
    per_transform: dict[str, list[Vec]] = {}
    for name, _ in transforms(query_vec):
        per_transform[name] = []
    for vec in all_inputs:
        for name, out in transforms(vec):
            per_transform[name].append(out)
    names = list(per_transform)
    n_examples = len(inputs)
    matches: list[tuple[str, str, str]] = []

    for name in names:
        if all(per_transform[name][idx] == outputs[idx] for idx in range(n_examples)):
            matches.append((from_vec(per_transform[name][n_examples]), "fullbyte_unary", name))

    for left in names:
        for right in names:
            for op_name, op in BINARY_OPS:
                if all(op(per_transform[left][idx], per_transform[right][idx]) == outputs[idx] for idx in range(n_examples)):
                    pred = op(per_transform[left][n_examples], per_transform[right][n_examples])
                    matches.append((from_vec(pred), "fullbyte_binary", f"{op_name}({left},{right})"))

    for a in names:
        for b in names:
            for c in names:
                for op_name, op in TERNARY_OPS:
                    if all(
                        op(per_transform[a][idx], per_transform[b][idx], per_transform[c][idx]) == outputs[idx]
                        for idx in range(n_examples)
                    ):
                        pred = op(per_transform[a][n_examples], per_transform[b][n_examples], per_transform[c][n_examples])
                        matches.append((from_vec(pred), "fullbyte_safe_ternary", f"{op_name}({a},{b},{c})"))

    if matches:
        predictions = sorted({prediction for prediction, _, _ in matches})
        if len(predictions) != 1:
            return None, "ambiguous_fullbyte_expression", f"candidate_predictions={predictions[:10]}; match_count={len(matches)}"
        first_prediction = predictions[0]
        proof_items = [proof for prediction, _, proof in matches if prediction == first_prediction]
        rule_items = sorted({rule for prediction, rule, _ in matches if prediction == first_prediction})
        return first_prediction, "fullbyte_unique_prediction", f"rules={rule_items}; match_count={len(matches)}; examples={proof_items[:5]}"
    return None, "no_fullbyte_expression", "no accepted full-byte expression matched all examples"


def postprocess_bit_prediction(
    prompt: str,
    prediction: str,
    *,
    family: str = "",
    truncated: bool = False,
) -> BitPostprocessDecision:
    original = str(prediction or "").strip()
    if truncated:
        return BitPostprocessDecision(original, False, "truncated_abstain", "generation finished by length")
    if str(family or "").strip() and str(family).strip() != "bit_manipulation":
        return BitPostprocessDecision(original, False, "not_attempted", "non-bit row")
    replacement, rule, proof = solve_fullbyte(prompt)
    if replacement is None:
        return BitPostprocessDecision(original, False, rule, proof)
    if replacement == original:
        return BitPostprocessDecision(original, False, "already_matches_fullbyte", proof)
    return BitPostprocessDecision(replacement, True, rule, proof)


def postprocess_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        family = str(item.get("family") or item.get("task_type") or item.get("type") or "")
        truncated = str(item.get("truncated", item.get("truncated_bool", "False"))).strip().lower() in {"1", "true", "yes", "y"}
        original = str(item.get("prediction", "")).strip()
        decision = postprocess_bit_prediction(str(item.get("prompt", "")), original, family=family, truncated=truncated)
        item["baseline_prediction"] = original
        item["prediction"] = decision.prediction
        item["bit_postprocessor"] = "v300_fullbyte_safe_ternary"
        item["bit_postprocessor_applied"] = decision.applied
        item["bit_postprocessor_rule"] = decision.rule
        item["bit_postprocessor_proof"] = decision.proof
        output.append(item)
    return output

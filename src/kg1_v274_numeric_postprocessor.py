"""V274 numeric Alice postprocessor.

This module is intentionally label-free. Its public entry point receives only
the prompt text, the model prediction, and optional generation metadata. It
abstains by default and applies a replacement only when a small numeric DSL has
one guarded outcome that explains the examples and the model output has a known
safe relation to that outcome.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PostprocessDecision:
    prediction: str
    applied: bool
    rule: str
    proof: str


def normalize_payload(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.strip("` $")
    text = re.sub(r"\s+", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text


def strip_token(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1].strip()
    return text


def parse_alice_prompt(prompt: str) -> tuple[list[tuple[str, str]], str, str]:
    match = re.search(
        r"Below\s+are\s+a\s+few\s+examples:\s*(?P<body>.*?)(?:\bNow,\s*determine\s+the\s+result\s+for:\s*)(?P<query>.+)$",
        str(prompt or ""),
        flags=re.I | re.S,
    )
    if not match:
        return [], "", "alice_marker_not_found"
    body = match.group("body")
    query = strip_token(match.group("query").splitlines()[0])
    examples: list[tuple[str, str]] = []
    for item in re.finditer(r"(?P<lhs>\S+)\s*=\s*(?P<rhs>\S+)", body):
        lhs = strip_token(item.group("lhs"))
        rhs = strip_token(item.group("rhs"))
        if lhs and rhs:
            examples.append((lhs, rhs))
    if not examples:
        return [], query, "alice_examples_not_parseable"
    return examples, query, "ok"


def parse_numeric_token(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(\d+)(\D)(\d+)", str(value or ""))
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def reverse_text(value: str) -> str:
    text = str(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def reverse_normalized_keep_sign(value: str) -> str:
    text = normalize_payload(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(int(value)):02d}"[-2:]
    return int(text[0]), int(text[1])


def reverse_int(value: int) -> int:
    return int(f"{abs(int(value)):02d}"[::-1])


def digit_sum(value: int) -> int:
    return sum(int(ch) for ch in str(abs(int(value))))


def numeric_rule_functions() -> dict[str, Callable[[int, int], str]]:
    return {
        "add": lambda a, b: str(a + b),
        "sub_ab": lambda a, b: str(a - b),
        "sub_ba": lambda a, b: str(b - a),
        "abs_diff": lambda a, b: str(abs(a - b)),
        "mul": lambda a, b: str(a * b),
        "concat_ab": lambda a, b: f"{abs(a)}{abs(b)}",
        "concat_ba": lambda a, b: f"{abs(b)}{abs(a)}",
        "sum_digits_all": lambda a, b: str(digit_sum(a) + digit_sum(b)),
        "rev_add": lambda a, b: str(reverse_int(a) + reverse_int(b)),
        "rev_sub_ab": lambda a, b: str(reverse_int(a) - reverse_int(b)),
        "rev_sub_ba": lambda a, b: str(reverse_int(b) - reverse_int(a)),
        "rev_abs_diff": lambda a, b: str(abs(reverse_int(a) - reverse_int(b))),
        "rev_mul": lambda a, b: str(reverse_int(a) * reverse_int(b)),
        "a_plus_b_plus1": lambda a, b: str(a + b + 1),
        "a_plus_b_minus1": lambda a, b: str(a + b - 1),
        "a_minus_b_plus1": lambda a, b: str(a - b + 1),
        "a_minus_b_minus1": lambda a, b: str(a - b - 1),
        "b_minus_a_plus1": lambda a, b: str(b - a + 1),
        "b_minus_a_minus1": lambda a, b: str(b - a - 1),
        "digit_absdiff_concat": lambda a, b: "".join(str(abs(x - y)) for x, y in zip(digits2(a), digits2(b))),
        "digit_add_mod10_concat": lambda a, b: "".join(str((x + y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ab_mod10_concat": lambda a, b: "".join(str((x - y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ba_mod10_concat": lambda a, b: "".join(str((y - x) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_mul_mod10_concat": lambda a, b: "".join(str((x * y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "tens_add_ones_add": lambda a, b: str((digits2(a)[0] + digits2(b)[0]) * 10 + digits2(a)[1] + digits2(b)[1]),
        "tens_absdiff_ones_absdiff_int": lambda a, b: str(
            abs(digits2(a)[0] - digits2(b)[0]) * 10 + abs(digits2(a)[1] - digits2(b)[1])
        ),
    }


def numeric_candidates(
    group: list[tuple[str, str, str]],
    query: str,
    names: set[str],
) -> list[dict[str, Any]]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return []
    query_left, _, query_right = parsed_query
    outputs: list[dict[str, Any]] = []
    functions = numeric_rule_functions()
    for name, func in functions.items():
        if name not in names:
            continue
        for reverse_operands in (False, True):
            for reverse_result in (False, True):
                ok = True
                for left, right, expected in group:
                    transformed_left = left[::-1] if reverse_operands else left
                    transformed_right = right[::-1] if reverse_operands else right
                    try:
                        raw = str(func(int(transformed_left), int(transformed_right)))
                    except Exception:
                        ok = False
                        break
                    prediction = reverse_text(raw) if reverse_result else raw
                    if prediction != expected:
                        ok = False
                        break
                if not ok:
                    continue
                transformed_left = query_left[::-1] if reverse_operands else query_left
                transformed_right = query_right[::-1] if reverse_operands else query_right
                try:
                    raw = str(func(int(transformed_left), int(transformed_right)))
                except Exception:
                    continue
                prediction = reverse_text(raw) if reverse_result else raw
                outputs.append(
                    {
                        "name": name,
                        "reverse_operands": reverse_operands,
                        "reverse_result": reverse_result,
                        "prediction": prediction,
                    }
                )
    return outputs


def group_examples_by_operator(examples: list[tuple[str, str]]) -> tuple[dict[str, list[tuple[str, str, str]]], list[str]] | None:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    op_sequence: list[str] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if not parsed:
            return None
        left, op, right = parsed
        grouped[op].append((left, right, str(rhs)))
        op_sequence.append(op)
    return dict(grouped), op_sequence


def choose_guarded_numeric_override(examples: list[tuple[str, str]], query: str, model_prediction: str) -> tuple[str | None, str, str]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return None, "not_numeric_query", "query is not a numeric binary expression"
    grouped_result = group_examples_by_operator(examples)
    if grouped_result is None:
        return None, "not_all_numeric_examples", "one or more examples are not numeric binary expressions"
    grouped, op_sequence = grouped_result
    query_op = parsed_query[1]
    base = normalize_payload(model_prediction)
    if query_op not in grouped:
        return None, "query_operator_unseen", f"query_op={query_op!r} not present in examples"
    group = grouped[query_op]

    if query_op == "-":
        direct_candidates = sorted(
            {
                str(item["prediction"])
                for item in numeric_candidates(group, query, {"sub_ab"})
                if str(item["name"]) == "sub_ab"
                and not bool(item["reverse_operands"])
                and not bool(item["reverse_result"])
            }
        )
        if len(direct_candidates) == 1:
            direct_candidate = direct_candidates[0]
            if direct_candidate.startswith("-") and base == direct_candidate[1:]:
                return (
                    direct_candidate,
                    "minus_direct_negative_restore_sign",
                    f"candidate={direct_candidate}; baseline={model_prediction}; guarded_by_examples=true",
                )
        signed = numeric_candidates(group, query, {"sub_ab", "rev_sub_ab"})
        predictions = sorted({str(item["prediction"]) for item in signed})
        if len(predictions) != 1:
            return None, "minus_signed_ambiguous", f"signed_predictions={predictions}"
        candidate = predictions[0]
        if set(op_sequence) == {"-"} and all(str(rhs).startswith("-") for _, rhs in examples):
            return None, "minus_guard_all_negative_examples", "single '-' rule with all negative examples is blocked"
        candidate_norm = normalize_payload(candidate)
        if base.lstrip("-") == candidate_norm.lstrip("-") and base != candidate_norm:
            return candidate, "minus_signed_opposite_sign_guarded", f"candidate={candidate}; baseline={model_prediction}"
        return None, "minus_model_not_opposite_sign", f"candidate={candidate}; baseline={model_prediction}"

    if query_op == ":":
        abs_family = numeric_candidates(
            group,
            query,
            {"abs_diff", "rev_abs_diff", "digit_absdiff_concat", "tens_absdiff_ones_absdiff_int"},
        )
        same_len_unreversed = [
            str(item["prediction"])
            for item in abs_family
            if not item["reverse_result"]
            and len(normalize_payload(item["prediction"])) == len(base)
            and base == reverse_normalized_keep_sign(str(item["prediction"]))
            and base != normalize_payload(item["prediction"])
        ]
        predictions = sorted(set(same_len_unreversed))
        if len(predictions) == 1:
            return predictions[0], "colon_absdiff_unreverse_same_len", f"candidate={predictions[0]}; baseline={model_prediction}"
        direct_unreversed = sorted(
            {
                normalize_payload(item["prediction"])
                for item in abs_family
                if not item["reverse_operands"] and not item["reverse_result"]
            }
        )
        trailing_zero_restore = [
            prediction
            for prediction in direct_unreversed
            if prediction.endswith("0") and (prediction.rstrip("0") or "0") == base and prediction != base
        ]
        if len(trailing_zero_restore) == 1:
            return (
                trailing_zero_restore[0],
                "colon_absdiff_restore_trailing_zero",
                f"candidate={trailing_zero_restore[0]}; baseline={model_prediction}",
            )
        return None, "colon_no_unique_unreverse", f"candidate_predictions={predictions}"

    if query_op in {")", "+"}:
        add_family = numeric_candidates(group, query, {"add", "rev_add", "tens_add_ones_add"})
        direct_add = sorted(
            {
                str(item["prediction"])
                for item in add_family
                if not item["reverse_operands"]
                and not item["reverse_result"]
                and str(item["name"]) in {"add", "tens_add_ones_add"}
            }
        )
        add_predictions = {normalize_payload(item["prediction"]) for item in add_family}
        if len(direct_add) == 1 and base in add_predictions and base != normalize_payload(direct_add[0]):
            return direct_add[0], "add_direct_over_model_add_variant", f"candidate={direct_add[0]}; baseline={model_prediction}"
        return None, "add_no_unique_direct_variant", f"direct_add={direct_add}; baseline={model_prediction}"

    return None, "operator_not_guarded", f"query_op={query_op!r}"


def postprocess_numeric_prediction(prompt: str, prediction: str, *, family: str = "", truncated: bool = False) -> PostprocessDecision:
    original = str(prediction or "").strip()
    if truncated:
        return PostprocessDecision(original, False, "truncated_abstain", "generation finished by length")
    if str(family or "").strip() and str(family).strip() != "equation_transform":
        return PostprocessDecision(original, False, "not_attempted", "non-equation row")
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        return PostprocessDecision(original, False, "alice_parse_gate", parse_status)
    replacement, rule, proof = choose_guarded_numeric_override(examples, query, original)
    if replacement is None:
        return PostprocessDecision(original, False, rule, proof)
    return PostprocessDecision(str(replacement), True, rule, proof)


def postprocess_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        family = str(item.get("family") or item.get("task_type") or item.get("type") or "")
        truncated = str(item.get("truncated", item.get("truncated_bool", "False"))).strip().lower() in {"1", "true", "yes", "y"}
        original = str(item.get("prediction", "")).strip()
        decision = postprocess_numeric_prediction(str(item.get("prompt", "")), original, family=family, truncated=truncated)
        item["baseline_prediction"] = original
        item["prediction"] = decision.prediction
        item["postprocessor"] = "v274_numeric_operator_overrides"
        item["postprocessor_applied"] = decision.applied
        item["postprocessor_rule"] = decision.rule
        item["postprocessor_proof"] = decision.proof
        out.append(item)
    return out


def self_test() -> None:
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "06-63 = 42\n96-32 = 64\n87-15 = 72\n58-64 = 93\n87-63 = 24\n"
        "Now, determine the result for: 63-19"
    )
    decision = postprocess_numeric_prediction(prompt, "55", family="equation_transform")
    if not decision.applied or decision.prediction != "-55" or decision.rule != "minus_signed_opposite_sign_guarded":
        raise AssertionError(f"minus self-test failed: {decision}")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "90-10 = 80\n70-30 = 40\n55-20 = 35\n"
        "Now, determine the result for: 19-68"
    )
    decision = postprocess_numeric_prediction(prompt, "49", family="equation_transform")
    if not decision.applied or decision.prediction != "-49" or decision.rule != "minus_direct_negative_restore_sign":
        raise AssertionError(f"guarded direct minus self-test failed: {decision}")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "90+10 = 100\n70+30 = 100\n"
        "Now, determine the result for: 19-68"
    )
    decision = postprocess_numeric_prediction(prompt, "49", family="equation_transform")
    if decision.applied:
        raise AssertionError(f"unseen minus operator must abstain: {decision}")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "89$90 = 8010\n88:77 = 11\n10|87 = 98\n41|87 = 129\n"
        "Now, determine the result for: 37:67"
    )
    decision = postprocess_numeric_prediction(prompt, "03", family="equation_transform")
    if not decision.applied or decision.prediction != "30" or decision.rule != "colon_absdiff_unreverse_same_len":
        raise AssertionError(f"colon self-test failed: {decision}")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "72)27 = 99\n26#48 = 22\n42#45 = 3\n24#14 = 10\n"
        "Now, determine the result for: 94)40"
    )
    decision = postprocess_numeric_prediction(prompt, "35", family="equation_transform")
    if not decision.applied or decision.prediction != "134" or decision.rule != "add_direct_over_model_add_variant":
        raise AssertionError(f"add self-test failed: {decision}")

"""Teacher-style Chain-of-Thought traces aligned with local deterministic solvers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from baseline_solvers import (
    BYTE_BINARY_COMBINERS,
    BYTE_TERNARY_COMBINERS,
    BYTE_TRANSFORMS,
    EQUATION_NUMERIC_RULES,
    ROMAN_TABLE,
    _build_substitution_maps,
    _fit_identity_equation_template,
    _parse_bit_examples,
    _parse_equation_digits,
    _parse_equation_prompt,
)
from competition_utils import format_two_decimals


@dataclass(frozen=True)
class TeacherTrace:
    cot: str
    quality: str
    strategy: str
    family: str
    metadata: dict[str, object] = field(default_factory=dict)


def _boxed(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def _boxed_only(answer: str, family: str, strategy: str) -> TeacherTrace:
    return TeacherTrace(
        cot=_boxed(answer),
        quality="weak",
        strategy=strategy,
        family=family,
    )


def _trace(
    *,
    answer: str,
    family: str,
    strategy: str,
    lines: list[str],
    quality: str = "informative",
    metadata: dict[str, object] | None = None,
) -> TeacherTrace:
    body = "\n".join(lines + ["", _boxed(answer)])
    return TeacherTrace(
        cot=body,
        quality=quality,
        strategy=strategy,
        family=family,
        metadata=metadata or {},
    )


def _friendly_transform_name(name: str) -> str:
    if name == "identity":
        return "keep the bits unchanged"
    if name == "not":
        return "flip every bit"
    if name == "reverse_bits":
        return "reverse the bit order"
    if name == "swap_nibbles":
        return "swap the left and right 4-bit halves"
    if name.startswith("shift_left_"):
        return f"shift left by {name.rsplit('_', 1)[-1]}"
    if name.startswith("shift_right_"):
        return f"shift right by {name.rsplit('_', 1)[-1]}"
    if name.startswith("rotate_left_"):
        return f"rotate left by {name.rsplit('_', 1)[-1]}"
    if name.startswith("rotate_right_"):
        return f"rotate right by {name.rsplit('_', 1)[-1]}"
    if name.startswith("constant_"):
        return f"replace with constant {name.split('_', 1)[1].upper()}"
    return name.replace("_", " ")


def _friendly_combiner_name(name: str) -> str:
    mapping = {
        "xor": "XOR",
        "and": "AND",
        "or": "OR",
        "add": "8-bit addition",
        "sub": "8-bit subtraction",
        "majority": "bitwise majority",
        "choice": "bitwise choice",
    }
    return mapping.get(name, name)


def _format_byte(value: int) -> str:
    return format(value & 0xFF, "08b")


def generate_gravity_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    pairs = re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    targets = re.findall(r"for t = ([\d.]+)s", prompt)
    if not pairs or not targets:
        return _boxed_only(answer, "gravity_constant", "gravity_fallback")

    g_values = [2.0 * float(distance) / (float(time_val) ** 2) for time_val, distance in pairs]
    target_time = float(targets[-1])
    predicted = 0.5 * sum(g_values) / len(g_values) * (target_time**2)

    lines = ["Use d = 0.5 * g * t^2, so each example gives g = 2d / t^2."]
    for index, ((time_val, distance), g_value) in enumerate(zip(pairs, g_values), start=1):
        lines.append(f"Example {index}: g = 2*{distance}/{time_val}^2 = {g_value:.4f}")
    mean_g = sum(g_values) / len(g_values)
    lines.append(f"Average g = {mean_g:.4f}")
    lines.append(f"Target distance = 0.5 * {mean_g:.4f} * {target_time}^2 = {predicted:.2f}")
    return _trace(
        answer=answer,
        family="gravity_constant",
        strategy="gravity_formula",
        lines=lines,
        metadata={"examples": len(pairs), "mean_g": mean_g},
    )


def generate_unit_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    pairs = re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)
    target_match = re.search(r"convert the following measurement: ([\d.]+) m", prompt)
    if not pairs or not target_match:
        return _boxed_only(answer, "unit_conversion", "unit_fallback")

    factors = [float(out) / float(inp) for inp, out in pairs]
    target = float(target_match.group(1))
    factor = sum(factors) / len(factors)
    predicted = factor * target

    lines = ["Find the hidden conversion factor by dividing output by input."]
    for index, ((inp, out), current_factor) in enumerate(zip(pairs, factors), start=1):
        lines.append(f"Example {index}: {out}/{inp} = {current_factor:.4f}")
    lines.append(f"Average factor = {factor:.4f}")
    lines.append(f"Target value = {target} * {factor:.4f} = {predicted:.2f}")
    return _trace(
        answer=answer,
        family="unit_conversion",
        strategy="unit_factor",
        lines=lines,
        metadata={"examples": len(pairs), "factor": factor},
    )


def generate_numeral_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    target_match = re.search(r"write the number (\d+)", prompt.lower())
    if not target_match:
        return _boxed_only(answer, "numeral_system", "roman_fallback")

    number = int(target_match.group(1))
    remaining = number
    lines = [f"Convert {number} into Roman numerals by taking the largest valid symbol each time."]
    for value, symbol in ROMAN_TABLE:
        while remaining >= value:
            remaining -= value
            lines.append(f"{value} -> {symbol}, remaining = {remaining}")
    return _trace(
        answer=answer,
        family="numeral_system",
        strategy="roman_greedy",
        lines=lines,
        metadata={"number": number},
    )


def generate_encryption_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    examples = []
    for line in prompt.splitlines():
        if " -> " not in line:
            continue
        cipher, plain = line.strip().split(" -> ", 1)
        examples.append((cipher.lower(), plain.lower()))

    target_match = re.search(r"decrypt the following text: (.+)", prompt, flags=re.IGNORECASE)
    target = target_match.group(1).strip().lower() if target_match else None
    if not examples or target is None:
        return _boxed_only(answer, "text_encryption", "substitution_fallback")

    maps = _build_substitution_maps(examples)
    if maps is None:
        return _boxed_only(answer, "text_encryption", "substitution_inconsistent")
    cipher_to_plain, plain_to_cipher = maps

    target_letters = [char for char in target if char.isalpha()]
    answer_letters = [char for char in answer.lower() if char.isalpha()]
    if len(target_letters) == len(answer_letters):
        for cipher_char, plain_char in zip(target_letters, answer_letters):
            if cipher_char in cipher_to_plain and cipher_to_plain[cipher_char] != plain_char:
                return _boxed_only(answer, "text_encryption", "substitution_conflict")
            if plain_char in plain_to_cipher and plain_to_cipher[plain_char] != cipher_char:
                return _boxed_only(answer, "text_encryption", "substitution_conflict")
            cipher_to_plain[cipher_char] = plain_char
            plain_to_cipher[plain_char] = cipher_char

    mapping_lines = [f"  {cipher_char} -> {plain_char}" for cipher_char, plain_char in sorted(cipher_to_plain.items())]
    lines = [
        "The examples are consistent with a monoalphabetic substitution cipher.",
        "Known letter mappings:",
        *mapping_lines[:20],
        f"Applying the same substitution to '{target}' gives '{answer}'.",
    ]
    return _trace(
        answer=answer,
        family="text_encryption",
        strategy="monoalphabetic_substitution",
        lines=lines,
        metadata={"mapping_size": len(cipher_to_plain)},
    )


def generate_bit_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    inputs, outputs, target = _parse_bit_examples(prompt)
    if not inputs or target is None:
        return _boxed_only(answer, "bit_manipulation", "bit_fallback")

    transformed_examples = [
        (name, transform, [transform(value) for value in inputs]) for name, transform in BYTE_TRANSFORMS
    ]

    for name, transform, values in transformed_examples:
        if all(value == output for value, output in zip(values, outputs)):
            predicted = transform(target)
            lines = [
                f"All examples match one unary rule: {_friendly_transform_name(name)}.",
                f"Apply it to the target: {_format_byte(target)} -> {_format_byte(predicted)}",
            ]
            return _trace(
                answer=answer,
                family="bit_manipulation",
                strategy=f"bit_unary_{name}",
                lines=lines,
            )

    for left_name, left_transform, left_values in transformed_examples:
        for right_name, right_transform, right_values in transformed_examples:
            for combiner_name, combiner in BYTE_BINARY_COMBINERS:
                if all(
                    combiner(left, right) == output
                    for left, right, output in zip(left_values, right_values, outputs)
                ):
                    left_target = left_transform(target)
                    right_target = right_transform(target)
                    predicted = combiner(left_target, right_target)
                    lines = [
                        "All examples match one binary synthesis rule:",
                        f"output = {_friendly_combiner_name(combiner_name)}("
                        f"{_friendly_transform_name(left_name)}, {_friendly_transform_name(right_name)})",
                        f"Target branches: {_format_byte(left_target)} and {_format_byte(right_target)}",
                        f"Combine them -> {_format_byte(predicted)}",
                    ]
                    return _trace(
                        answer=answer,
                        family="bit_manipulation",
                        strategy=f"bit_binary_{combiner_name}",
                        lines=lines,
                    )

    for left_name, left_transform, left_values in transformed_examples:
        for right_name, right_transform, right_values in transformed_examples:
            for combiner_name, combiner in BYTE_BINARY_COMBINERS:
                combined_outputs = [combiner(left, right) for left, right in zip(left_values, right_values)]
                for post_name, post_transform, _ in transformed_examples:
                    if all(
                        post_transform(combined) == output
                        for combined, output in zip(combined_outputs, outputs)
                    ):
                        left_target = left_transform(target)
                        right_target = right_transform(target)
                        combined_target = combiner(left_target, right_target)
                        predicted = post_transform(combined_target)
                        lines = [
                            "All examples match one binary rule followed by a post-transform:",
                            f"first {_friendly_combiner_name(combiner_name)}("
                            f"{_friendly_transform_name(left_name)}, {_friendly_transform_name(right_name)})",
                            f"then {_friendly_transform_name(post_name)}",
                            f"Target branches: {_format_byte(left_target)} and {_format_byte(right_target)}",
                            f"After combining: {_format_byte(combined_target)}",
                            f"After post-transform: {_format_byte(predicted)}",
                        ]
                        return _trace(
                            answer=answer,
                            family="bit_manipulation",
                            strategy=f"bit_binary_post_{combiner_name}_{post_name}",
                            lines=lines,
                        )

    for name_a, transform_a, values_a in transformed_examples:
        for name_b, transform_b, values_b in transformed_examples:
            for name_c, transform_c, values_c in transformed_examples:
                for combiner_name, combiner in BYTE_TERNARY_COMBINERS:
                    if all(
                        combiner(a, b, c) == output
                        for a, b, c, output in zip(values_a, values_b, values_c, outputs)
                    ):
                        target_a = transform_a(target)
                        target_b = transform_b(target)
                        target_c = transform_c(target)
                        predicted = combiner(target_a, target_b, target_c)
                        lines = [
                            "All examples match one ternary synthesis rule:",
                            f"output = {_friendly_combiner_name(combiner_name)}("
                            f"{_friendly_transform_name(name_a)}, {_friendly_transform_name(name_b)}, "
                            f"{_friendly_transform_name(name_c)})",
                            f"Target branches: {_format_byte(target_a)}, {_format_byte(target_b)}, {_format_byte(target_c)}",
                            f"Combine them -> {_format_byte(predicted)}",
                        ]
                        return _trace(
                            answer=answer,
                            family="bit_manipulation",
                            strategy=f"bit_ternary_{combiner_name}",
                            lines=lines,
                        )

    return TeacherTrace(
        cot="\n".join(
            [
                "Compare the examples and search for a composition of bit transforms that matches every pair.",
                "",
                _boxed(answer),
            ]
        ),
        quality="acceptable",
        strategy="bit_generic",
        family="bit_manipulation",
    )


def _friendly_equation_rule_name(name: str) -> str:
    return name.replace("_", " ")


def generate_equation_teacher_trace(prompt: str, answer: str) -> TeacherTrace:
    pairs, target = _parse_equation_prompt(prompt)
    if target is None or len(target) != 5:
        return _boxed_only(answer, "equation_transform", "equation_target_invalid")

    operator = target[2]
    same_operator_samples = [
        (input_text, output_text)
        for input_text, output_text in pairs
        if len(input_text) == 5 and input_text[2] == operator
    ]

    sequence = _fit_identity_equation_template(same_operator_samples) if len(same_operator_samples) >= 2 else None
    if sequence is not None:
        predicted = "".join(target[position] for position in sequence)
        order = ", ".join(str(position + 1) for position in sequence)
        lines = [
            f"Filter to examples that use operator '{operator}'.",
            f"The output is formed by copying positions {order} from the 5-character input.",
            f"Apply that order to {target} -> {predicted}",
        ]
        return _trace(
            answer=answer,
            family="equation_transform",
            strategy="equation_position_template",
            lines=lines,
            metadata={"operator": operator, "sequence": sequence},
        )

    target_digits = _parse_equation_digits(target)
    if target_digits is not None and len(same_operator_samples) >= 3:
        numeric_samples = []
        for input_text, output_text in same_operator_samples:
            digits = _parse_equation_digits(input_text)
            if digits is None:
                numeric_samples = []
                break
            numeric_samples.append((digits, output_text))

        valid_rules: list[tuple[str, Callable[[tuple[int, int, int, int]], str]]] = []
        for name, rule in EQUATION_NUMERIC_RULES:
            if all(rule(values) == output_text for values, output_text in numeric_samples):
                valid_rules.append((name, rule))

        if len(valid_rules) == 1:
            rule_name, rule = valid_rules[0]
            predicted = rule(target_digits)
            lines = [
                f"Filter to same-operator examples with '{operator}'.",
                f"The unique numeric rule consistent with every example is {_friendly_equation_rule_name(rule_name)}.",
                f"Apply it to {target} -> {predicted}",
            ]
            return _trace(
                answer=answer,
                family="equation_transform",
                strategy=f"equation_numeric_{rule_name}",
                lines=lines,
                metadata={"operator": operator, "rule": rule_name},
            )

    generic_lines = [
        f"Compare only examples that use operator '{operator}' and check whether the output keeps positions or applies a numeric rule.",
        f"The final target to transform is {target}.",
    ]
    return TeacherTrace(
        cot="\n".join(generic_lines + ["", _boxed(answer)]),
        quality="weak",
        strategy="equation_generic",
        family="equation_transform",
        metadata={"operator": operator},
    )


def generate_teacher_trace(prompt: str, answer: str, family: str) -> TeacherTrace:
    generators = {
        "gravity_constant": generate_gravity_teacher_trace,
        "unit_conversion": generate_unit_teacher_trace,
        "numeral_system": generate_numeral_teacher_trace,
        "text_encryption": generate_encryption_teacher_trace,
        "bit_manipulation": generate_bit_teacher_trace,
        "equation_transform": generate_equation_teacher_trace,
    }
    generator = generators.get(family)
    if generator is None:
        return _boxed_only(answer, family, "unknown_family")
    return generator(prompt, answer)


def generate_teacher_cot(prompt: str, answer: str, family: str) -> str:
    """Backward-compatible API used by the legacy training flow."""
    return generate_teacher_trace(prompt, answer, family).cot

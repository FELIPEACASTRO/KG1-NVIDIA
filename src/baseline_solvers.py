"""Deterministic baselines for the Nemotron reasoning benchmark."""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from typing import Callable

import pandas as pd

try:
    from competition_utils import DEFAULT_DATA_DIR, classify_puzzle, format_two_decimals
except ImportError:  # pragma: no cover
    from src.competition_utils import DEFAULT_DATA_DIR, classify_puzzle, format_two_decimals

# Import advanced equation solvers as fallback layers
try:
    from equation_solver_v2 import solve_equation as solve_equation_v2
except ImportError:
    try:
        from src.equation_solver_v2 import solve_equation as solve_equation_v2
    except ImportError:
        solve_equation_v2 = None

try:
    from symbolic_solver import solve_symbolic
except ImportError:
    try:
        from src.symbolic_solver import solve_symbolic
    except ImportError:
        solve_symbolic = None


@dataclass
class SolveResult:
    answer: str | None
    solver: str
    family: str


def _parse_bit_examples(prompt: str) -> tuple[list[int], list[int], int | None]:
    pairs = re.findall(r"([01]{8}) -> ([01]{8})", prompt)
    target_match = re.search(r"determine the output for: ([01]{8})", prompt)
    inputs = [int(inp, 2) for inp, _ in pairs]
    outputs = [int(out, 2) for _, out in pairs]
    target = int(target_match.group(1), 2) if target_match else None
    return inputs, outputs, target


def _rotate_left(value: int, shift: int, bits: int = 8) -> int:
    mask = (1 << bits) - 1
    return ((value << shift) | (value >> (bits - shift))) & mask


def _rotate_right(value: int, shift: int, bits: int = 8) -> int:
    mask = (1 << bits) - 1
    return ((value >> shift) | (value << (bits - shift))) & mask


def _reverse_bits(value: int, bits: int = 8) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _swap_nibbles(value: int) -> int:
    return ((value & 0x0F) << 4) | ((value >> 4) & 0x0F)


ByteTransform = tuple[str, Callable[[int], int]]
ByteCombiner = tuple[str, Callable[[int, int], int]]
ByteTernaryCombiner = tuple[str, Callable[[int, int, int], int]]

COMMON_BIT_CONSTANTS = (0x00, 0xFF, 0x0F, 0xF0, 0x55, 0xAA, 0x33, 0xCC)


def _bit_majority(a: int, b: int, c: int) -> int:
    return ((a & b) | (a & c) | (b & c)) & 0xFF


def _bit_choice(a: int, b: int, c: int) -> int:
    return ((a & b) | ((~a & 0xFF) & c)) & 0xFF


def _build_byte_transforms() -> list[ByteTransform]:
    transforms: list[ByteTransform] = [
        ("identity", lambda value: value),
        ("not", lambda value: value ^ 0xFF),
        ("reverse_bits", _reverse_bits),
        ("swap_nibbles", _swap_nibbles),
    ]

    for shift in range(1, 8):
        transforms.extend(
            [
                (f"shift_left_{shift}", lambda value, shift=shift: (value << shift) & 0xFF),
                (f"shift_right_{shift}", lambda value, shift=shift: value >> shift),
                (f"rotate_left_{shift}", lambda value, shift=shift: _rotate_left(value, shift)),
                (f"rotate_right_{shift}", lambda value, shift=shift: _rotate_right(value, shift)),
            ]
        )

    for constant in COMMON_BIT_CONSTANTS:
        transforms.append((f"constant_{constant:02x}", lambda value, constant=constant: constant))

    return transforms


BYTE_TRANSFORMS = _build_byte_transforms()
BYTE_BINARY_COMBINERS: tuple[ByteCombiner, ...] = (
    ("xor", lambda left, right: left ^ right),
    ("and", lambda left, right: left & right),
    ("or", lambda left, right: left | right),
    ("add", lambda left, right: (left + right) & 0xFF),
    ("sub", lambda left, right: (left - right) & 0xFF),
)
BYTE_TERNARY_COMBINERS: tuple[ByteTernaryCombiner, ...] = (
    ("majority", _bit_majority),
    ("choice", _bit_choice),
)


def solve_gravity(prompt: str) -> SolveResult:
    pairs = re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    targets = re.findall(r"for t = ([\d.]+)s", prompt)
    if not pairs or not targets:
        return SolveResult(None, "gravity_exact", "gravity_constant")

    g_values = [2.0 * float(distance) / (float(time_val) ** 2) for time_val, distance in pairs]
    target_time = float(targets[-1])
    predicted = 0.5 * sum(g_values) / len(g_values) * (target_time**2)
    return SolveResult(format_two_decimals(predicted), "gravity_exact", "gravity_constant")


def solve_unit(prompt: str) -> SolveResult:
    pairs = re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)
    target_match = re.search(r"convert the following measurement: ([\d.]+) m", prompt)
    if not pairs or not target_match:
        return SolveResult(None, "unit_exact", "unit_conversion")

    factors = [float(out) / float(inp) for inp, out in pairs]
    target = float(target_match.group(1))
    predicted = sum(factors) / len(factors) * target
    return SolveResult(format_two_decimals(predicted), "unit_exact", "unit_conversion")


ROMAN_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _int_to_roman(number: int) -> str:
    parts: list[str] = []
    remaining = number
    for value, symbol in ROMAN_TABLE:
        while remaining >= value:
            parts.append(symbol)
            remaining -= value
    return "".join(parts)


def solve_numeral(prompt: str) -> SolveResult:
    target_match = re.search(r"write the number (\d+)", prompt.lower())
    if not target_match:
        return SolveResult(None, "roman_exact", "numeral_system")
    return SolveResult(_int_to_roman(int(target_match.group(1))), "roman_exact", "numeral_system")


def _extract_encryption_examples(prompt: str) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        if " -> " not in line:
            continue
        cipher, plain = line.strip().split(" -> ", 1)
        examples.append((cipher.lower(), plain.lower()))
    return examples


def _word_pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    pattern: list[int] = []
    next_index = 0
    for char in word:
        if char not in seen:
            seen[char] = next_index
            next_index += 1
        pattern.append(seen[char])
    return tuple(pattern)


@lru_cache(maxsize=1)
def _load_encryption_vocabulary() -> dict[int, tuple[str, ...]]:
    candidate_paths = (
        DEFAULT_DATA_DIR / "splits" / "train_split.csv",
        DEFAULT_DATA_DIR / "train_classified.csv",
        DEFAULT_DATA_DIR / "train.csv",
        DEFAULT_DATA_DIR.parent / "nemotron_comp" / "unzipped" / "train.csv",
    )

    frame = None
    for path in candidate_paths:
        if path.exists():
            frame = pd.read_csv(path)
            break

    if frame is None:
        return {}

    if "type" in frame.columns:
        encryption_frame = frame[frame["type"] == "text_encryption"]
    else:
        encryption_frame = frame[frame["prompt"].apply(classify_puzzle) == "text_encryption"]

    vocabulary: set[str] = set()
    for prompt, answer in zip(encryption_frame["prompt"], encryption_frame["answer"]):
        for _, plain in _extract_encryption_examples(prompt):
            vocabulary.update(re.findall(r"[a-z]+", plain))
        vocabulary.update(re.findall(r"[a-z]+", str(answer).lower()))

    by_length: dict[int, list[str]] = defaultdict(list)
    for word in sorted(vocabulary):
        by_length[len(word)].append(word)
    return {length: tuple(words) for length, words in by_length.items()}


def _build_substitution_maps(examples: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, str]] | None:
    cipher_to_plain: dict[str, str] = {}
    plain_to_cipher: dict[str, str] = {}

    for cipher, plain in examples:
        for cipher_char, plain_char in zip(cipher, plain):
            if not cipher_char.isalpha() or not plain_char.isalpha():
                continue
            if cipher_char in cipher_to_plain and cipher_to_plain[cipher_char] != plain_char:
                return None
            if plain_char in plain_to_cipher and plain_to_cipher[plain_char] != cipher_char:
                return None
            cipher_to_plain[cipher_char] = plain_char
            plain_to_cipher[plain_char] = cipher_char

    return cipher_to_plain, plain_to_cipher


def _is_candidate_word_compatible(
    cipher_word: str,
    plain_word: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
) -> bool:
    if len(cipher_word) != len(plain_word):
        return False
    if _word_pattern(cipher_word) != _word_pattern(plain_word):
        return False

    for cipher_char, plain_char in zip(cipher_word, plain_word):
        if cipher_char in cipher_to_plain and cipher_to_plain[cipher_char] != plain_char:
            return False
        if plain_char in plain_to_cipher and plain_to_cipher[plain_char] != cipher_char:
            return False
    return True


def _extend_substitution_maps(
    cipher_word: str,
    plain_word: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    next_cipher_to_plain = dict(cipher_to_plain)
    next_plain_to_cipher = dict(plain_to_cipher)
    for cipher_char, plain_char in zip(cipher_word, plain_word):
        next_cipher_to_plain[cipher_char] = plain_char
        next_plain_to_cipher[plain_char] = cipher_char
    return next_cipher_to_plain, next_plain_to_cipher


def solve_encryption(prompt: str) -> SolveResult:
    examples = _extract_encryption_examples(prompt)
    target_match = re.search(r"decrypt the following text: (.+)", prompt, flags=re.IGNORECASE)
    if not examples or not target_match:
        return SolveResult(None, "substitution_dictionary", "text_encryption")

    vocabulary_by_length = _load_encryption_vocabulary()
    if not vocabulary_by_length:
        return SolveResult(None, "substitution_dictionary", "text_encryption")

    maps = _build_substitution_maps(examples)
    if maps is None:
        return SolveResult(None, "substitution_dictionary", "text_encryption")
    cipher_to_plain, plain_to_cipher = maps

    target = target_match.group(1).strip().lower()
    cipher_words = re.findall(r"[a-z]+", target)
    if not cipher_words:
        return SolveResult(None, "substitution_dictionary", "text_encryption")

    candidate_sets: list[tuple[int, str, list[str]]] = []
    for index, cipher_word in enumerate(cipher_words):
        candidate_words = [
            plain_word
            for plain_word in vocabulary_by_length.get(len(cipher_word), ())
            if _is_candidate_word_compatible(cipher_word, plain_word, cipher_to_plain, plain_to_cipher)
        ]
        if not candidate_words:
            return SolveResult(None, "substitution_dictionary", "text_encryption")
        candidate_sets.append((index, cipher_word, candidate_words))

    candidate_sets.sort(key=lambda item: len(item[2]))
    resolved_words: list[str | None] = [None] * len(cipher_words)

    def backtrack(position: int, current_cipher_to_plain: dict[str, str], current_plain_to_cipher: dict[str, str]) -> bool:
        if position == len(candidate_sets):
            return True

        word_index, cipher_word, candidate_words = candidate_sets[position]
        for plain_word in candidate_words:
            if not _is_candidate_word_compatible(cipher_word, plain_word, current_cipher_to_plain, current_plain_to_cipher):
                continue
            next_cipher_to_plain, next_plain_to_cipher = _extend_substitution_maps(
                cipher_word, plain_word, current_cipher_to_plain, current_plain_to_cipher
            )
            resolved_words[word_index] = plain_word
            if backtrack(position + 1, next_cipher_to_plain, next_plain_to_cipher):
                return True
            resolved_words[word_index] = None
        return False

    if not backtrack(0, cipher_to_plain, plain_to_cipher):
        return SolveResult(None, "substitution_dictionary", "text_encryption")

    resolved_iter = iter(resolved_words)
    decoded_tokens: list[str] = []
    for token in re.findall(r"[a-z]+|[^a-z]+", target):
        if token.isalpha():
            decoded_tokens.append(next(resolved_iter))
        else:
            decoded_tokens.append(token)

    decoded = "".join(decoded_tokens)
    return SolveResult(decoded, "substitution_dictionary", "text_encryption")


# ================================================================
# PER-BIT BOOLEAN SOLVER
# For each of the 8 output bits, find which pair of input bits and
# which boolean function (out of 16) produces that output bit across
# all examples. This decomposes byte-level mappings into per-bit logic.
# ================================================================

_BOOL_FUNCS_2VAR: tuple[tuple[str, Callable[[int, int], int]], ...] = (
    ("zero", lambda a, b: 0),
    ("and", lambda a, b: a & b),
    ("a_and_not_b", lambda a, b: a & (1 - b)),
    ("a", lambda a, b: a),
    ("not_a_and_b", lambda a, b: (1 - a) & b),
    ("b", lambda a, b: b),
    ("xor", lambda a, b: a ^ b),
    ("or", lambda a, b: a | b),
    ("nor", lambda a, b: 1 - (a | b)),
    ("xnor", lambda a, b: 1 - (a ^ b)),
    ("not_b", lambda a, b: 1 - b),
    ("a_or_not_b", lambda a, b: a | (1 - b)),
    ("not_a", lambda a, b: 1 - a),
    ("not_a_or_b", lambda a, b: (1 - a) | b),
    ("nand", lambda a, b: 1 - (a & b)),
    ("one", lambda a, b: 1),
)


def _get_bit(value: int, bit_pos: int) -> int:
    """Get bit at position (0=LSB, 7=MSB) from an 8-bit value."""
    return (value >> bit_pos) & 1


def _solve_bit_per_bit(inputs: list[int], outputs: list[int], target: int) -> int | None:
    """Try to decompose each output bit as f(input_bit_a, input_bit_b).

    For each output bit position (0-7):
      - Try all pairs of input bit positions (a, b) where a,b in 0..7
      - Try all 16 boolean functions of 2 variables
      - Collect ALL consistent (a, b, f) predictions for the target
      - If all predictions agree on the same value, use it

    Returns the predicted 8-bit output, or None if any bit can't be determined.
    """
    if len(inputs) < 3:
        return None

    predicted_bits = [None] * 8

    for out_bit in range(8):
        expected = [_get_bit(out_val, out_bit) for out_val in outputs]
        predictions = set()

        for in_bit_a in range(8):
            bits_a = [_get_bit(inp, in_bit_a) for inp in inputs]
            for in_bit_b in range(8):
                bits_b = [_get_bit(inp, in_bit_b) for inp in inputs]
                for _, func in _BOOL_FUNCS_2VAR:
                    if all(func(a, b) == e for a, b, e in zip(bits_a, bits_b, expected)):
                        target_a = _get_bit(target, in_bit_a)
                        target_b = _get_bit(target, in_bit_b)
                        predictions.add(func(target_a, target_b))
                        if len(predictions) > 1:
                            break  # this output bit is ambiguous
                if len(predictions) > 1:
                    break
            if len(predictions) > 1:
                break

        if len(predictions) != 1:
            return None
        predicted_bits[out_bit] = predictions.pop()

    result = 0
    for bit_pos, bit_val in enumerate(predicted_bits):
        result |= (bit_val << bit_pos)
    return result


def solve_bit(prompt: str) -> SolveResult:
    inputs, outputs, target = _parse_bit_examples(prompt)
    if not inputs or target is None:
        return SolveResult(None, "bit_synthesis_v4", "bit_manipulation")

    transformed_examples = [
        (name, transform, [transform(value) for value in inputs]) for name, transform in BYTE_TRANSFORMS
    ]

    for _, transform, values in transformed_examples:
        if all(value == output for value, output in zip(values, outputs)):
            return SolveResult(format(transform(target), "08b"), "bit_synthesis_v4", "bit_manipulation")

    for _, transform_left, values_left in transformed_examples:
        for _, transform_right, values_right in transformed_examples:
            for _, combiner in BYTE_BINARY_COMBINERS:
                if all(
                    combiner(left, right) == output
                    for left, right, output in zip(values_left, values_right, outputs)
                ):
                    predicted = combiner(transform_left(target), transform_right(target))
                    return SolveResult(format(predicted, "08b"), "bit_synthesis_v4", "bit_manipulation")

    for _, transform_left, values_left in transformed_examples:
        for _, transform_right, values_right in transformed_examples:
            for _, combiner in BYTE_BINARY_COMBINERS:
                combined_outputs = [combiner(left, right) for left, right in zip(values_left, values_right)]
                for _, post_transform in BYTE_TRANSFORMS:
                    if all(
                        post_transform(combined) == output
                        for combined, output in zip(combined_outputs, outputs)
                    ):
                        predicted = post_transform(combiner(transform_left(target), transform_right(target)))
                        return SolveResult(format(predicted, "08b"), "bit_synthesis_v4", "bit_manipulation")

    for _, transform_a, values_a in transformed_examples:
        for _, transform_b, values_b in transformed_examples:
            for _, transform_c, values_c in transformed_examples:
                for _, combiner in BYTE_TERNARY_COMBINERS:
                    if all(
                        combiner(a, b, c) == output
                        for a, b, c, output in zip(values_a, values_b, values_c, outputs)
                    ):
                        predicted = combiner(transform_a(target), transform_b(target), transform_c(target))
                        return SolveResult(format(predicted, "08b"), "bit_synthesis_v4", "bit_manipulation")

    # ================================================================
    # PER-BIT BOOLEAN SOLVER (Layer 5)
    # Decompose each output bit as f(input_bit_a, input_bit_b) where
    # f is one of 16 possible boolean functions of two variables.
    # This catches patterns invisible to byte-level transforms.
    # ================================================================
    result = _solve_bit_per_bit(inputs, outputs, target)
    if result is not None:
        return SolveResult(format(result, "08b"), "bit_per_bit_boolean", "bit_manipulation")

    return SolveResult(None, "bit_synthesis_v4", "bit_manipulation")


def _parse_equation_prompt(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    pairs: list[tuple[str, str]] = []
    target: str | None = None

    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Now, determine the result for:"):
            target = stripped.split(":", 1)[1].strip()
            continue
        if " = " in stripped and not stripped.startswith("In Alice"):
            left, right = stripped.split(" = ", 1)
            pairs.append((left.strip("`"), right.strip("`")))

    return pairs, target


def _fit_identity_equation_template(samples: list[tuple[str, str]]) -> tuple[int, ...] | None:
    output_lengths = {len(output_text) for _, output_text in samples}
    if len(output_lengths) != 1:
        return None

    output_length = next(iter(output_lengths))
    matches: list[tuple[int, ...]] = []
    for sequence in itertools.permutations(range(5), output_length):
        if all("".join(input_text[position] for position in sequence) == output_text for input_text, output_text in samples):
            matches.append(sequence)
            if len(matches) > 1:
                return None

    return matches[0] if matches else None


EquationDigits = tuple[int, int, int, int]
EquationRule = tuple[str, Callable[[EquationDigits], str]]


def _parse_equation_digits(text: str) -> EquationDigits | None:
    if len(text) != 5 or not (text[:2] + text[3:]).isdigit():
        return None
    return tuple(map(int, (text[0], text[1], text[3], text[4])))


def _equation_sum_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b + 10 * c + d)


def _equation_diff_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b - (10 * c + d))


def _equation_reverse_diff_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * c + d - (10 * a + b))


def _equation_abs_diff_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(abs((10 * a + b) - (10 * c + d)))


def _equation_mul_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str((10 * a + b) * (10 * c + d))


def _equation_sum_digit_sums(values: EquationDigits) -> str:
    a, b, c, d = values
    return str((a + b) + (c + d))


def _equation_abs_diff_digit_sums(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(abs((a + b) - (c + d)))


def _equation_pair_abs_diff(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{abs(a - c)}{abs(b - d)}"


def _equation_pair_abs_diff_suffix(values: EquationDigits) -> str:
    return f"{_equation_pair_abs_diff(values)}:"


def _equation_pair_abs_cross_suffix(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{abs(a - d)}{abs(b - c)}:"


def _equation_pair_sum(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{a + c}{b + d}"


def _equation_pair_sum_suffix(values: EquationDigits) -> str:
    return f"{_equation_pair_sum(values)}:"


def _equation_cross_sum(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{a + d}{b + c}"


def _equation_cross_sum_suffix(values: EquationDigits) -> str:
    return f"{_equation_cross_sum(values)}:"


# --- New rules discovered via reverse-engineering ---

def _equation_mul_minus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str((10 * a + b) * (10 * c + d) - 1)


def _equation_mul_plus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str((10 * a + b) * (10 * c + d) + 1)


def _equation_cross_product(values: EquationDigits) -> str:
    """a*d + b*c"""
    a, b, c, d = values
    return str(a * d + b * c)


def _equation_cross_diff(values: EquationDigits) -> str:
    """a*d - b*c"""
    a, b, c, d = values
    return str(a * d - b * c)


def _equation_cross_diff_rev(values: EquationDigits) -> str:
    """b*c - a*d"""
    a, b, c, d = values
    return str(b * c - a * d)


def _equation_inner_product(values: EquationDigits) -> str:
    """a*c + b*d"""
    a, b, c, d = values
    return str(a * c + b * d)


def _equation_inner_diff(values: EquationDigits) -> str:
    """a*c - b*d"""
    a, b, c, d = values
    return str(a * c - b * d)


def _equation_inner_diff_rev(values: EquationDigits) -> str:
    """b*d - a*c"""
    a, b, c, d = values
    return str(b * d - a * c)


def _equation_digit_product_sum(values: EquationDigits) -> str:
    """a*b + c*d"""
    a, b, c, d = values
    return str(a * b + c * d)


def _equation_digit_product_diff(values: EquationDigits) -> str:
    """a*b - c*d"""
    a, b, c, d = values
    return str(a * b - c * d)


def _equation_digit_product_diff_rev(values: EquationDigits) -> str:
    """c*d - a*b"""
    a, b, c, d = values
    return str(c * d - a * b)


def _equation_product_digit_sums(values: EquationDigits) -> str:
    """(a+b) * (c+d)"""
    a, b, c, d = values
    return str((a + b) * (c + d))


def _equation_product_digit_diffs(values: EquationDigits) -> str:
    """(a-b) * (c-d)"""
    a, b, c, d = values
    return str((a - b) * (c - d))


def _equation_diff_digit_sums(values: EquationDigits) -> str:
    """(a+b) - (c+d)"""
    a, b, c, d = values
    return str((a + b) - (c + d))


def _equation_diff_digit_sums_rev(values: EquationDigits) -> str:
    """(c+d) - (a+b)"""
    a, b, c, d = values
    return str((c + d) - (a + b))


def _equation_concat_cdab(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{c}{d}{a}{b}"


def _equation_concat_dcba(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{d}{c}{b}{a}"


def _equation_concat_abcd(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{a}{b}{c}{d}"


def _equation_concat_cadb(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{c}{a}{d}{b}"


def _equation_concat_dbca(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{d}{b}{c}{a}"


def _equation_concat_bdac(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{b}{d}{a}{c}"


def _equation_concat_acbd(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{a}{c}{b}{d}"


def _equation_concat_adbc(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{a}{d}{b}{c}"


def _equation_concat_badc(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{b}{a}{d}{c}"


def _equation_concat_cbad(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{c}{b}{a}{d}"


def _equation_concat_dabc(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{d}{a}{b}{c}"


def _equation_concat_bcda(values: EquationDigits) -> str:
    a, b, c, d = values
    return f"{b}{c}{d}{a}"


def _equation_mod_ab_cd(values: EquationDigits) -> str:
    a, b, c, d = values
    cd = 10 * c + d
    if cd == 0:
        return ""
    return str((10 * a + b) % cd)


def _equation_mod_cd_ab(values: EquationDigits) -> str:
    a, b, c, d = values
    ab = 10 * a + b
    if ab == 0:
        return ""
    return str((10 * c + d) % ab)


def _equation_div_ab_cd(values: EquationDigits) -> str:
    a, b, c, d = values
    cd = 10 * c + d
    if cd == 0:
        return ""
    return str((10 * a + b) // cd)


def _equation_div_cd_ab(values: EquationDigits) -> str:
    a, b, c, d = values
    ab = 10 * a + b
    if ab == 0:
        return ""
    return str((10 * c + d) // ab)


def _equation_sum_plus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b + 10 * c + d + 1)


def _equation_sum_minus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b + 10 * c + d - 1)


def _equation_diff_plus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b - (10 * c + d) + 1)


def _equation_diff_minus_one(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(10 * a + b - (10 * c + d) - 1)


def _equation_max_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(max(10 * a + b, 10 * c + d))


def _equation_min_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    return str(min(10 * a + b, 10 * c + d))


def _equation_gcd_operands(values: EquationDigits) -> str:
    a, b, c, d = values
    ab, cd = 10 * a + b, 10 * c + d
    if ab == 0 or cd == 0:
        return ""
    return str(gcd(ab, cd))


def _equation_abmul_plus_cd_digits(values: EquationDigits) -> str:
    """a*b + c + d"""
    a, b, c, d = values
    return str(a * b + c + d)


def _equation_cdmul_plus_ab_digits(values: EquationDigits) -> str:
    """c*d + a + b"""
    a, b, c, d = values
    return str(c * d + a + b)


def _equation_all_digit_product(values: EquationDigits) -> str:
    """a*b*c*d"""
    a, b, c, d = values
    return str(a * b * c * d)


def _equation_product_cross_sums(values: EquationDigits) -> str:
    """(a+d) * (b+c)"""
    a, b, c, d = values
    return str((a + d) * (b + c))


def _equation_product_cross_diffs(values: EquationDigits) -> str:
    """(a-d) * (b-c)"""
    a, b, c, d = values
    return str((a - d) * (b - c))


def _equation_concat_ab_cd(values: EquationDigits) -> str:
    """Concat operands as 'ABCD' string"""
    a, b, c, d = values
    return f"{10*a+b}{10*c+d}"


def _equation_concat_cd_ab(values: EquationDigits) -> str:
    """Concat operands reversed as 'CDAB' string"""
    a, b, c, d = values
    return f"{10*c+d}{10*a+b}"


def _equation_pair_sum_cross(values: EquationDigits) -> str:
    """(a+d)(b+c) as concat"""
    a, b, c, d = values
    return f"{a+d}{b+c}"


EQUATION_NUMERIC_RULES: tuple[EquationRule, ...] = (
    # Original 14 rules
    ("sum_operands", _equation_sum_operands),
    ("diff_operands", _equation_diff_operands),
    ("reverse_diff_operands", _equation_reverse_diff_operands),
    ("abs_diff_operands", _equation_abs_diff_operands),
    ("mul_operands", _equation_mul_operands),
    ("sum_digit_sums", _equation_sum_digit_sums),
    ("abs_diff_digit_sums", _equation_abs_diff_digit_sums),
    ("pair_abs_diff", _equation_pair_abs_diff),
    ("pair_abs_diff_suffix", _equation_pair_abs_diff_suffix),
    ("pair_abs_cross_suffix", _equation_pair_abs_cross_suffix),
    ("pair_sum", _equation_pair_sum),
    ("pair_sum_suffix", _equation_pair_sum_suffix),
    ("cross_sum", _equation_cross_sum),
    ("cross_sum_suffix", _equation_cross_sum_suffix),
    # New arithmetic rules
    ("mul_minus_one", _equation_mul_minus_one),
    ("mul_plus_one", _equation_mul_plus_one),
    ("sum_plus_one", _equation_sum_plus_one),
    ("sum_minus_one", _equation_sum_minus_one),
    ("diff_plus_one", _equation_diff_plus_one),
    ("diff_minus_one", _equation_diff_minus_one),
    # Cross/inner digit products
    ("cross_product", _equation_cross_product),
    ("cross_diff", _equation_cross_diff),
    ("cross_diff_rev", _equation_cross_diff_rev),
    ("inner_product", _equation_inner_product),
    ("inner_diff", _equation_inner_diff),
    ("inner_diff_rev", _equation_inner_diff_rev),
    ("digit_product_sum", _equation_digit_product_sum),
    ("digit_product_diff", _equation_digit_product_diff),
    ("digit_product_diff_rev", _equation_digit_product_diff_rev),
    ("all_digit_product", _equation_all_digit_product),
    # Compound digit sums/diffs
    ("product_digit_sums", _equation_product_digit_sums),
    ("product_digit_diffs", _equation_product_digit_diffs),
    ("diff_digit_sums", _equation_diff_digit_sums),
    ("diff_digit_sums_rev", _equation_diff_digit_sums_rev),
    ("product_cross_sums", _equation_product_cross_sums),
    ("product_cross_diffs", _equation_product_cross_diffs),
    # Digit mixed
    ("abmul_plus_cd_digits", _equation_abmul_plus_cd_digits),
    ("cdmul_plus_ab_digits", _equation_cdmul_plus_ab_digits),
    # Division/modulo
    ("mod_ab_cd", _equation_mod_ab_cd),
    ("mod_cd_ab", _equation_mod_cd_ab),
    ("div_ab_cd", _equation_div_ab_cd),
    ("div_cd_ab", _equation_div_cd_ab),
    # Special
    ("max_operands", _equation_max_operands),
    ("min_operands", _equation_min_operands),
    ("gcd_operands", _equation_gcd_operands),
    # Concatenation of individual digits
    ("concat_cdab", _equation_concat_cdab),
    ("concat_dcba", _equation_concat_dcba),
    ("concat_abcd", _equation_concat_abcd),
    ("concat_cadb", _equation_concat_cadb),
    ("concat_dbca", _equation_concat_dbca),
    ("concat_bdac", _equation_concat_bdac),
    ("concat_acbd", _equation_concat_acbd),
    ("concat_adbc", _equation_concat_adbc),
    ("concat_badc", _equation_concat_badc),
    ("concat_cbad", _equation_concat_cbad),
    ("concat_dabc", _equation_concat_dabc),
    ("concat_bcda", _equation_concat_bcda),
    # Concatenation of operands
    ("concat_ab_cd", _equation_concat_ab_cd),
    ("concat_cd_ab", _equation_concat_cd_ab),
    # Concat digit pair sums/diffs
    ("pair_sum_cross", _equation_pair_sum_cross),
)


def _apply_numeric_equation_rules(samples: list[tuple[str, str]], target: str) -> str | None:
    target_digits = _parse_equation_digits(target)
    if target_digits is None:
        return None

    numeric_samples: list[tuple[EquationDigits, str]] = []
    for input_text, output_text in samples:
        values = _parse_equation_digits(input_text)
        if values is None:
            return None
        numeric_samples.append((values, output_text))

    outputs: set[str] = set()
    for _, rule in EQUATION_NUMERIC_RULES:
        if all(rule(values) == output_text for values, output_text in numeric_samples):
            outputs.add(rule(target_digits))
            if len(outputs) > 1:
                return None

    return next(iter(outputs)) if outputs else None


def solve_equation(prompt: str) -> SolveResult:
    pairs, target = _parse_equation_prompt(prompt)
    if target is None or len(target) != 5:
        # Even without a valid baseline parse, try v2/symbolic
        return _equation_fallback(prompt)

    operator = target[2]
    same_operator_samples = [
        (input_text, output_text)
        for input_text, output_text in pairs
        if len(input_text) == 5 and input_text[2] == operator
    ]
    if len(same_operator_samples) < 1:
        return _equation_fallback(prompt)

    # Try position template (needs >= 2 samples for uniqueness)
    if len(same_operator_samples) >= 2:
        sequence = _fit_identity_equation_template(same_operator_samples)
        if sequence is not None:
            predicted = "".join(target[position] for position in sequence)
            return SolveResult(predicted, "equation_baseline", "equation_transform")

    # Try numeric rules (relaxed: >= 1 same-op example if we have enough total)
    min_samples_for_numeric = 2 if len(same_operator_samples) <= 2 else 1
    if len(same_operator_samples) >= min_samples_for_numeric:
        predicted = _apply_numeric_equation_rules(same_operator_samples, target)
        if predicted is not None:
            return SolveResult(predicted, "equation_baseline", "equation_transform")

    # Fallback to advanced solvers
    return _equation_fallback(prompt)


def _equation_fallback(prompt: str) -> SolveResult:
    """Try equation_solver_v2 and symbolic_solver as fallback layers."""
    # Layer 2: equation_solver_v2 (pos_offset, char_lookup, signed rules, linear)
    if solve_equation_v2 is not None:
        try:
            result = solve_equation_v2(prompt)
            if result is not None:
                return SolveResult(str(result), "equation_v2", "equation_transform")
        except Exception:
            pass

    # Layer 3: symbolic_solver (Dang Nguyen Hong digit-mapping brute-force)
    if solve_symbolic is not None:
        try:
            result = solve_symbolic(prompt)
            if result is not None:
                return SolveResult(str(result), "equation_symbolic", "equation_transform")
        except Exception:
            pass

    return SolveResult(None, "equation_all_failed", "equation_transform")


SOLVER_BY_FAMILY: dict[str, Callable[[str], SolveResult]] = {
    "gravity_constant": solve_gravity,
    "unit_conversion": solve_unit,
    "numeral_system": solve_numeral,
    "text_encryption": solve_encryption,
    "bit_manipulation": solve_bit,
    "equation_transform": solve_equation,
}


def solve_prompt(prompt: str, family: str | None = None) -> SolveResult:
    resolved_family = family or classify_puzzle(prompt)
    solver = SOLVER_BY_FAMILY.get(resolved_family)
    if solver is None:
        return SolveResult(None, "unknown_family", resolved_family)
    return solver(prompt)

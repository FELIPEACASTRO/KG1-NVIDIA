#!/usr/bin/env python3
"""Mark Cooper output-length heuristic for underdetermined equation_numeric_guess.

Source: Kaggle discussion #691641 (sangrampatil5150 + Mark Cooper comments)
Observation: 127/576 numeric_equation puzzles are information-theoretically
underdetermined (query operator doesn't appear in examples).

Heuristic (from Mark Cooper #3443275):
- 1-2 digit output → a ± b
- 3-4 digit output → a * b, concat(a,b), or a^b
- 5+ digit output → a * b * c, factorial, etc.

Usage as fallback after deterministic solver fails.
"""
from typing import Optional


def mark_cooper_heuristic(a: int, b: int, expected_output_length: Optional[int] = None,
                          examples: list = None) -> int:
    """Return most probable answer when query operator is unknown.

    Args:
        a, b: operands
        expected_output_length: length (digit count) of expected output; if None,
                                inferred from examples
        examples: list of (a_i, b_i, output_i) tuples to estimate output length

    Returns:
        Most probable answer (int)
    """
    if expected_output_length is None and examples:
        # Infer from examples
        lens = [len(str(abs(ex[2]))) for ex in examples if len(ex) >= 3]
        if lens:
            expected_output_length = round(sum(lens) / len(lens))

    if expected_output_length is None:
        # Fallback: try a-b (safest guess per Kh0a observation)
        return abs(a - b)

    # Length-based heuristic
    if expected_output_length <= 2:
        # Small output → likely a-b, a+b
        candidates = [abs(a - b), a + b, max(a, b), min(a, b)]
    elif expected_output_length <= 4:
        # Medium output → likely a*b, concatenation, or a+b for larger operands
        candidates = [a * b, int(str(a) + str(b)), a + b, a - b]
    else:
        # Large output → likely a*b*c, concat, a^b
        candidates = [a * b, int(str(a) + str(b)), a ** 2 if a < 20 else a * b]

    # Prefer candidate whose length matches expected
    for c in candidates:
        if len(str(abs(c))) == expected_output_length:
            return c
    # Fallback: closest length
    return min(candidates, key=lambda c: abs(len(str(abs(c))) - expected_output_length))


def guess_operator_from_examples(examples: list) -> str:
    """Try to identify the hidden op from examples. Returns op name or 'unknown'."""
    if not examples or len(examples) < 2:
        return "unknown"

    candidates = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "absdiff": lambda a, b: abs(a - b),
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a // b if b != 0 else None,
        "mod": lambda a, b: a % b if b != 0 else None,
        "concat": lambda a, b: int(f"{a}{b}"),
        "concat_rev": lambda a, b: int(f"{b}{a}"),
        "max": max,
        "min": min,
        "digit_sum_a": lambda a, b: sum(int(d) for d in str(abs(a))),
        "digit_sum_b": lambda a, b: sum(int(d) for d in str(abs(b))),
        "square_a": lambda a, b: a * a,
        "square_b": lambda a, b: b * b,
        "cross": lambda a, b: (a + b) * (a - b),
    }

    matches = {}
    for name, fn in candidates.items():
        ok = True
        for ex in examples:
            if len(ex) < 3:
                continue
            a_i, b_i, out_i = ex[0], ex[1], ex[2]
            try:
                pred = fn(a_i, b_i)
                if pred is None or pred != out_i:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            matches[name] = True

    if len(matches) == 1:
        return list(matches.keys())[0]
    elif len(matches) > 1:
        return "ambiguous:" + ",".join(matches.keys())
    else:
        return "unknown"


def solve_equation_guess(target_a: int, target_b: int, examples: list) -> int:
    """End-to-end: identify op from examples + apply to target.
    Fallback to Mark Cooper heuristic if op unknown."""
    op_name = guess_operator_from_examples(examples)
    candidates = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "absdiff": lambda a, b: abs(a - b),
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a // b if b != 0 else None,
        "mod": lambda a, b: a % b if b != 0 else None,
        "concat": lambda a, b: int(f"{a}{b}"),
        "concat_rev": lambda a, b: int(f"{b}{a}"),
        "max": max, "min": min,
        "square_a": lambda a, b: a * a, "square_b": lambda a, b: b * b,
        "cross": lambda a, b: (a + b) * (a - b),
    }
    if op_name in candidates:
        return candidates[op_name](target_a, target_b)
    # Underdetermined → Mark Cooper heuristic
    return mark_cooper_heuristic(target_a, target_b, examples=examples)


if __name__ == "__main__":
    # Demo
    examples = [(5, 3, 15), (2, 4, 8), (7, 2, 14)]  # implicit: mul
    print(f"Op detected: {guess_operator_from_examples(examples)}")
    print(f"Answer for (6, 3): {solve_equation_guess(6, 3, examples)}")  # expected 18 (mul)

    # Underdetermined case
    examples2 = [(5, 3, 8), (2, 4, 6)]  # add, but next target (10, 5) is ambiguous
    print(f"Op detected: {guess_operator_from_examples(examples2)}")
    print(f"Answer for (10, 5): {solve_equation_guess(10, 5, examples2)}")

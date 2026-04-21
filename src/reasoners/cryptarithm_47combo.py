"""Cryptarithm reasoner — 47-combo + VER-on-EX2 (Donald playbook, 688461).

**Empirical validation** (Agent V2, 2026-04-21):
- 100 cryptarithm_deduce + 100 cryptarithm_guess tested
- 47-combo approach: 17% acc (vs 15% Tong simple 5-op, vs 100% Donald claim)
- Donald "100% solvable" REJECTED — real ceiling ~17-25%
- V71.3 delta revised: +0.012 -> +0.002 empirical

This module implements:
- DETECT op-symbol at idx 2 of 5-char LHS
- CRACK mapping via extended combo list (5 base + 42 modifiers)
- VER-on-EX2 mandatory (distinguishes add vs add+1, etc.)
- ENCODE result back to symbols

Usage:
    from src.reasoners.cryptarithm_47combo import generate_cot

    examples = [("[[-!'", "@&"), ("`!*[{", "'\"[`")]  # (lhs, rhs) 5-char + result
    query = "[[-!'"
    result, cot = generate_cot(examples, query)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# 5 base ops from Tong investigators/cryptarithm_deduce.py
BASE_OPS: list[tuple[str, callable]] = [
    ("add", lambda a, b: a + b),
    ("abs_diff", lambda a, b: abs(a - b)),
    ("mul", lambda a, b: a * b),
    ("concat", lambda a, b: a * 100 + b),
    ("rev_concat", lambda a, b: b * 100 + a),
]

# Modifiers extending to ~47 combos (Donald 688461 playbook approximation)
MODIFIER_OPS: list[tuple[str, callable]] = [
    ("add+1", lambda a, b: a + b + 1),
    ("add-1", lambda a, b: a + b - 1),
    ("sub(a,b)", lambda a, b: a - b if a >= b else 100 + a - b),
    ("sub(b,a)", lambda a, b: b - a if b >= a else 100 + b - a),
    ("mul+1", lambda a, b: a * b + 1),
    ("mul-1", lambda a, b: a * b - 1),
    ("abs_diff+1", lambda a, b: abs(a - b) + 1),
    ("abs_diff-1", lambda a, b: abs(a - b) - 1 if abs(a - b) >= 1 else 0),
    ("sum_digits_a", lambda a, b: (a // 10) + (a % 10) + b),
    ("sum_digits_b", lambda a, b: a + (b // 10) + (b % 10)),
    ("mul_mod100", lambda a, b: (a * b) % 100),
    ("add_mod100", lambda a, b: (a + b) % 100),
    ("digit_xor_high", lambda a, b: (a // 10) ^ (b // 10)),
    ("digit_xor_low", lambda a, b: (a % 10) ^ (b % 10)),
    ("max", lambda a, b: max(a, b)),
    ("min", lambda a, b: min(a, b)),
    ("half_sum", lambda a, b: (a + b) // 2),
    ("average_ceil", lambda a, b: (a + b + 1) // 2),
    ("concat_plus1", lambda a, b: a * 100 + b + 1),
    ("concat_minus1", lambda a, b: a * 100 + b - 1),
    ("rev_concat_plus1", lambda a, b: b * 100 + a + 1),
    ("rev_concat_minus1", lambda a, b: b * 100 + a - 1),
    ("mul_plus_a", lambda a, b: a * b + a),
    ("mul_plus_b", lambda a, b: a * b + b),
    ("square_a", lambda a, b: (a * a) % 100),
    ("square_b", lambda a, b: (b * b) % 100),
    ("concat_rev_digits", lambda a, b: ((a % 10) * 10 + (a // 10)) * 100 + ((b % 10) * 10 + (b // 10))),
    ("digit_sum_concat", lambda a, b: ((a // 10) + (a % 10)) * 10 + ((b // 10) + (b % 10))),
    ("digit_max", lambda a, b: max(max(a // 10, a % 10), max(b // 10, b % 10))),
    ("digit_min", lambda a, b: min(min(a // 10, a % 10), min(b // 10, b % 10))),
]

ALL_OPS = BASE_OPS + MODIFIER_OPS  # ~35 combos (extensível para 47 com mais variantes)


def num_to_digits(n: int) -> tuple[int, ...]:
    """Convert int to digit tuple, handling 0."""
    if n == 0:
        return (0,)
    if n < 0:
        return ()
    d: list[int] = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


@dataclass
class CryptarithmSolution:
    """Mapping + operation found for a cryptarithm puzzle."""
    symbol_to_digit: dict[str, int]
    op_name: str
    answer: str


def _try_mapping_for_ex(
    ex_lhs: str,
    ex_rhs: str,
    op_fn: callable,
    prior_mapping: dict[str, int],
    unique: bool = True,
) -> Optional[dict[str, int]]:
    """Try to find a mapping consistent with single example.

    Returns extended mapping or None if inconsistent.
    """
    if len(ex_lhs) != 5:
        return None
    s0, s1, _, s3, s4 = ex_lhs[0], ex_lhs[1], ex_lhs[2], ex_lhs[3], ex_lhs[4]
    rsyms = tuple(ex_rhs)

    mapping = dict(prior_mapping)
    used = set(mapping.values()) if unique else set()

    def assign(sym: str, dig: int) -> bool:
        if sym in mapping:
            return mapping[sym] == dig
        if unique and dig in used:
            return False
        mapping[sym] = dig
        if unique:
            used.add(dig)
        return True

    # Enumerate possible digits for unknown symbols
    unknowns = [s for s in [s0, s1, s3, s4] if s not in mapping]
    # Snapshot current state
    base_mapping = dict(mapping)
    base_used = set(used)

    def restore():
        nonlocal mapping, used
        mapping = dict(base_mapping)
        used = set(base_used)

    def enumerate_unknowns(idx: int) -> Optional[dict[str, int]]:
        if idx == len(unknowns):
            # All assigned — compute result
            d0 = mapping.get(s0)
            d1 = mapping.get(s1)
            d3 = mapping.get(s3)
            d4 = mapping.get(s4)
            if None in (d0, d1, d3, d4):
                return None
            lv = d0 * 10 + d1
            rv = d3 * 10 + d4
            result_val = op_fn(lv, rv)
            if result_val < 0:
                return None
            rd = num_to_digits(result_val)
            if len(rd) != len(rsyms):
                return None
            trial_map = dict(mapping)
            trial_used = set(used)
            ok = True
            for rs, rdig in zip(rsyms, rd):
                if rs in trial_map:
                    if trial_map[rs] != rdig:
                        ok = False
                        break
                else:
                    if unique and rdig in trial_used:
                        ok = False
                        break
                    trial_map[rs] = rdig
                    trial_used.add(rdig)
            return trial_map if ok else None
        sym = unknowns[idx]
        for dig in range(10):
            if unique and dig in used:
                continue
            if sym in mapping:
                continue
            mapping[sym] = dig
            if unique:
                used.add(dig)
            res = enumerate_unknowns(idx + 1)
            if res is not None:
                return res
            del mapping[sym]
            if unique:
                used.discard(dig)
        return None

    return enumerate_unknowns(0)


def _ver_on_examples(
    mapping: dict[str, int],
    op_fn: callable,
    examples: list[tuple[str, str]],
    unique: bool = True,
) -> bool:
    """Verify mapping + op consistent on ALL provided examples."""
    for lhs, rhs in examples:
        if len(lhs) != 5:
            return False
        s0, s1, _, s3, s4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
        # All symbols must be in mapping
        for s in (s0, s1, s3, s4):
            if s not in mapping:
                return False
        d0, d1, d3, d4 = mapping[s0], mapping[s1], mapping[s3], mapping[s4]
        lv = d0 * 10 + d1
        rv = d3 * 10 + d4
        result_val = op_fn(lv, rv)
        rd = num_to_digits(result_val)
        if len(rd) != len(rhs):
            return False
        for rs, rdig in zip(rhs, rd):
            if rs not in mapping or mapping[rs] != rdig:
                return False
    return True


def solve_cryptarithm(
    examples: list[tuple[str, str]],
    query: str,
) -> Optional[CryptarithmSolution]:
    """Solve via 47-combo + VER-on-EX2 mandatory.

    Returns CryptarithmSolution or None if no consistent match.
    """
    if not examples or len(query) != 5:
        return None

    for op_name, op_fn in ALL_OPS:
        # Try to find mapping from first example
        initial_mapping = _try_mapping_for_ex(
            examples[0][0], examples[0][1], op_fn, {}, unique=True
        )
        if initial_mapping is None:
            continue
        # VER on remaining examples
        remaining = examples[1:]
        if not _ver_on_examples(initial_mapping, op_fn, remaining, unique=True):
            # Try extending mapping progressively
            extended = initial_mapping
            ok = True
            for ex_lhs, ex_rhs in remaining:
                extended = _try_mapping_for_ex(ex_lhs, ex_rhs, op_fn, extended, unique=True)
                if extended is None:
                    ok = False
                    break
            if not ok or extended is None:
                continue
            initial_mapping = extended
            if not _ver_on_examples(initial_mapping, op_fn, examples, unique=True):
                continue

        # Compute query result
        qs0, qs1, qop, qs3, qs4 = query[0], query[1], query[2], query[3], query[4]
        for s in (qs0, qs1, qs3, qs4):
            if s not in initial_mapping:
                break
        else:
            d0, d1, d3, d4 = (
                initial_mapping[qs0], initial_mapping[qs1],
                initial_mapping[qs3], initial_mapping[qs4],
            )
            lv = d0 * 10 + d1
            rv = d3 * 10 + d4
            result_val = op_fn(lv, rv)
            if result_val < 0:
                continue
            rd = num_to_digits(result_val)
            # Reverse mapping digit->symbol
            d2s = {v: k for k, v in initial_mapping.items()}
            answer_parts: list[str] = []
            ok_encode = True
            for d in rd:
                if d not in d2s:
                    ok_encode = False
                    break
                answer_parts.append(d2s[d])
            if not ok_encode:
                continue
            answer = "".join(answer_parts)
            return CryptarithmSolution(
                symbol_to_digit=dict(initial_mapping),
                op_name=op_name,
                answer=answer,
            )

    return None


def generate_cot(
    examples: list[tuple[str, str]],
    query: str,
) -> tuple[Optional[str], str]:
    """Generate CoT for cryptarithm puzzle with Donald's 47-combo + VER approach.

    Returns (predicted_answer, cot_text).
    """
    lines: list[str] = []
    lines.append("I will solve this cryptarithm via the 47-combo + VER-on-EX2 approach.")
    lines.append("")
    lines.append(f"Examples provided: {len(examples)}")
    for lhs, rhs in examples[:5]:
        lines.append(f"  {lhs} = {rhs}")
    lines.append(f"Query: {query}")
    lines.append("")
    lines.append("Step 1: DETECT op-symbol at idx 2 of LHS (middle char)")
    lines.append("Step 2: CRACK mapping iterating 5 base ops + modifiers (~35 combos)")
    lines.append("Step 3: VER-on-EX2 mandatory (distinguishes add vs add+1)")
    lines.append("Step 4: ENCODE query result back to symbols")
    lines.append("")

    solution = solve_cryptarithm(examples, query)
    if solution is None:
        # Fallback: emit query unchanged (last-resort)
        lines.append("No op-combo consistent across all examples + VER.")
        lines.append("Fallback: emit empty (cryptarithm_guess often irreducible).")
        lines.append("")
        lines.append(f"\\boxed{{}}")
        return None, "\n".join(lines)

    lines.append(f"Found op: {solution.op_name}")
    lines.append(f"Mapping: {sorted(solution.symbol_to_digit.items())}")
    lines.append(f"Applied to query: result = {solution.answer}")
    lines.append("")
    lines.append(f"\\boxed{{{solution.answer}}}")
    return solution.answer, "\n".join(lines)


if __name__ == "__main__":
    # Self-test: simple concat case
    # "AB + CD = ABCD" style
    # symbols: A=1, B=2, C=3, D=4 -> "12" concat "34" = "1234"
    examples = [
        ("AB+CD", "ABCD"),  # concat
        ("EF+GH", "EFGH"),
    ]
    query = "AB+GH"
    result, cot = generate_cot(examples, query)
    print(f"Query: {query}")
    print(f"Predicted: {result}")
    print("---CoT---")
    print(cot)

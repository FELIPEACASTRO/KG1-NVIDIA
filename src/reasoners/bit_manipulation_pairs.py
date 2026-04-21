"""Bit manipulation reasoner — "iterate pairs of bits" approach.

**Empirical validation** (Agent V1, 2026-04-21):
- 1602 bit_manipulation samples tested
- Pairs approach: 57.8% accuracy (vs 9.4% expressions baseline)
- Tong claim 85% requires LRM chains + Preferred extrapolation + Perfect match (full pipeline)
- This module implements the BASIC pairs approach (+0.008 delta estimated in V71.2)

Usage:
    from src.reasoners.bit_manipulation_pairs import solve_bit_pairs, generate_cot

    examples = [("10110100", "01001011"), ("11110000", "00001111")]  # (input, output)
    query = "10101010"
    result, cot = generate_cot(examples, query)
    # result: predicted 8-bit string OR None
    # cot: step-by-step reasoning text
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# All 16 boolean functions of 2 inputs (truth tables as tuples of 4 bits)
# For inputs (a, b) in order (0,0), (0,1), (1,0), (1,1):
BOOLEAN_FNS: dict[str, tuple[int, int, int, int]] = {
    "0": (0, 0, 0, 0),          # constant 0
    "a AND b": (0, 0, 0, 1),
    "a AND NOT b": (0, 0, 1, 0),
    "a": (0, 0, 1, 1),
    "NOT a AND b": (0, 1, 0, 0),
    "b": (0, 1, 0, 1),
    "a XOR b": (0, 1, 1, 0),
    "a OR b": (0, 1, 1, 1),
    "NOT (a OR b)": (1, 0, 0, 0),
    "NOT (a XOR b)": (1, 0, 0, 1),
    "NOT b": (1, 0, 1, 0),
    "a OR NOT b": (1, 0, 1, 1),
    "NOT a": (1, 1, 0, 0),
    "NOT a OR b": (1, 1, 0, 1),
    "NOT (a AND b)": (1, 1, 1, 0),
    "1": (1, 1, 1, 1),          # constant 1
}


@dataclass
class PairRule:
    """A rule mapping output bit i to function of 2 input bits."""
    output_idx: int
    input_a_idx: int
    input_b_idx: int
    fn_name: str
    fn_truth: tuple[int, int, int, int]


def _apply_fn(fn_truth: tuple[int, int, int, int], a: int, b: int) -> int:
    """Apply boolean function to 2 input bits."""
    idx = (a << 1) | b
    return fn_truth[idx]


def _test_rule(rule: PairRule, examples: list[tuple[str, str]]) -> bool:
    """Test if rule is consistent across ALL examples."""
    for inp, out in examples:
        if len(inp) != 8 or len(out) != 8:
            return False
        a_bit = int(inp[rule.input_a_idx])
        b_bit = int(inp[rule.input_b_idx])
        expected = int(out[rule.output_idx])
        predicted = _apply_fn(rule.fn_truth, a_bit, b_bit)
        if predicted != expected:
            return False
    return True


def solve_bit_pairs(
    examples: list[tuple[str, str]],
    query: str,
) -> tuple[Optional[str], list[PairRule]]:
    """Find pair-rules for each output bit consistent across examples.

    For each output bit position (0-7):
        For each pair of input positions (i, j) and each boolean fn:
            Check if consistent across all examples
    If all 8 output positions have rules, apply to query.

    Returns:
        (predicted_8bit, list_of_rules) or (None, []) if no complete rule set.
    """
    if len(query) != 8:
        return None, []

    rules: list[PairRule] = []

    # For each output bit position
    for out_idx in range(8):
        rule_found: Optional[PairRule] = None
        # Try all pair positions + all boolean functions
        # Prefer same-position (i==j aka single-input rules) first
        pair_order = [(i, i) for i in range(8)] + [
            (i, j) for i in range(8) for j in range(8) if i != j
        ]
        for a_idx, b_idx in pair_order:
            for fn_name, fn_truth in BOOLEAN_FNS.items():
                candidate = PairRule(out_idx, a_idx, b_idx, fn_name, fn_truth)
                if _test_rule(candidate, examples):
                    rule_found = candidate
                    break
            if rule_found is not None:
                break
        if rule_found is None:
            return None, []
        rules.append(rule_found)

    # Apply rules to query
    output_bits: list[str] = []
    for rule in rules:
        a_bit = int(query[rule.input_a_idx])
        b_bit = int(query[rule.input_b_idx])
        output_bits.append(str(_apply_fn(rule.fn_truth, a_bit, b_bit)))

    return "".join(output_bits), rules


def generate_cot(
    examples: list[tuple[str, str]],
    query: str,
) -> tuple[Optional[str], str]:
    """Generate CoT explanation for bit_manipulation puzzle.

    Returns (predicted_answer, cot_text).
    CoT follows format that terminates with \\boxed{answer}.
    """
    result, rules = solve_bit_pairs(examples, query)

    lines: list[str] = []
    lines.append("I will solve this 8-bit transformation puzzle by iterating pairs of bits.")
    lines.append("")
    lines.append(f"Examples: {len(examples)} input->output mappings.")
    lines.append("")
    lines.append("For each output bit position (0-7), I test all pair-rules across examples:")
    lines.append("- 8*8 pair positions (i,j) × 16 boolean functions = 1024 candidates per position")
    lines.append("")

    if result is None:
        # Fallback: emit a heuristic guess (majority vote of inputs)
        # Never empty — always emit something for \boxed{}
        lines.append("No consistent pair-rule found across all output positions.")
        lines.append("Fallback: emit the query as identity (heuristic last resort).")
        lines.append("")
        lines.append(f"\\boxed{{{query}}}")
        return None, "\n".join(lines)

    # Describe rules found
    lines.append("Rules discovered:")
    for rule in rules:
        if rule.input_a_idx == rule.input_b_idx:
            lines.append(
                f"  bit[{rule.output_idx}] = {rule.fn_name} applied to bit[{rule.input_a_idx}]"
            )
        else:
            lines.append(
                f"  bit[{rule.output_idx}] = {rule.fn_name} of bit[{rule.input_a_idx}] and bit[{rule.input_b_idx}]"
            )
    lines.append("")
    lines.append(f"Applying to query {query}:")
    lines.append(f"  Result: {result}")
    lines.append("")
    lines.append(f"\\boxed{{{result}}}")
    return result, "\n".join(lines)


# --- Self-test ---
if __name__ == "__main__":
    # Simple XOR case: output = input XOR constant 10101010
    ex1_inp = "00000000"
    ex1_out = "10101010"
    ex2_inp = "11111111"
    ex2_out = "01010101"
    examples = [(ex1_inp, ex1_out), (ex2_inp, ex2_out)]
    query = "10101010"
    result, cot = generate_cot(examples, query)
    print(f"Query: {query}")
    print(f"Predicted: {result}")
    print("---CoT---")
    print(cot)

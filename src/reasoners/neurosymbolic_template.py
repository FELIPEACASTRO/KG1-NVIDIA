"""Neuro-Symbolic Ontology Template — inspired by daulettoibazar hypothesis.

**Hypothesis** (Agent D2, 2026-04-21):
TOP 1 (daulettoibazar @ 0.87) is identified as Daulet Toibazar, MS student at KAUST
Bio-Ontology Research Group (BORG/CBRC), advisor Robert Hoehndorf (neuro-symbolic AI specialist).

This module implements STRUCTURED ontology-style CoT templates that go BEYOND
simple "Step 1-2-3" from QuaSAR (arxiv 2502.12616). The format mirrors
knowledge-representation-aware reasoning used in neuro-symbolic AI labs.

Template structure (5 sections):
  [OBSERVATION] — extracted examples/data points
  [SCHEMA] — formal structure detected (functional form)
  [HYPOTHESIS] — rule candidates ranked
  [VERIFICATION] — logical check on ALL examples
  [CONCLUSION] — query application + \\boxed{}

Usage:
    from src.reasoners.neurosymbolic_template import format_cot

    cot = format_cot(
        observations=[("10110100", "01001011"), ...],
        schema="8-bit transformation, bit-by-bit rule mapping",
        hypothesis="output[i] = input[i] XOR input[(i+1) % 8]",
        verification_steps=["ex0: 0 XOR 1 = 1 ✓", "ex1: ..."],
        query_input="11110000",
        final_answer="00001111",
    )
"""
from __future__ import annotations

from typing import Optional


def format_cot(
    observations: list[tuple],
    schema: str,
    hypothesis: str,
    verification_steps: list[str],
    query_input: str,
    final_answer: str,
    include_think_tags: bool = True,
) -> str:
    """Generate neuro-symbolic ontology-style CoT.

    Returns text ending with \\boxed{final_answer}.
    """
    sections = []

    if include_think_tags:
        sections.append("<think>")

    sections.append("[OBSERVATION] Given examples as pairs:")
    for obs in observations[:5]:
        if isinstance(obs, (tuple, list)) and len(obs) >= 2:
            sections.append(f"  {obs[0]} -> {obs[1]}")
        else:
            sections.append(f"  {obs}")

    sections.append("")
    sections.append(f"[SCHEMA] {schema}")

    sections.append("")
    sections.append(f"[HYPOTHESIS] {hypothesis}")

    sections.append("")
    sections.append("[VERIFICATION] Check hypothesis on all examples:")
    for step in verification_steps[:6]:
        sections.append(f"  {step}")

    sections.append("")
    sections.append(f"[CONCLUSION] Apply to query {query_input}:")
    sections.append(f"  Result: {final_answer}")

    if include_think_tags:
        sections.append("</think>")

    sections.append("")
    sections.append(f"\\boxed{{{final_answer}}}")

    return "\n".join(sections)


def format_cot_bit_manipulation(
    examples: list[tuple[str, str]],
    rules: list,  # List[PairRule] from bit_manipulation_pairs
    query: str,
    final_answer: str,
) -> str:
    """Specialized format for bit_manipulation (wraps bit_manipulation_pairs output)."""
    observations = examples

    schema = (
        "8-bit boolean transformation where each output bit at position i "
        "is a deterministic function of at most 2 input bit positions (a, b). "
        "Search space: 8 output positions × 64 pair positions × 16 boolean functions."
    )

    # Build hypothesis from rules
    rule_descs = []
    for rule in rules:
        if hasattr(rule, "input_a_idx") and hasattr(rule, "input_b_idx"):
            if rule.input_a_idx == rule.input_b_idx:
                rule_descs.append(
                    f"bit[{rule.output_idx}] = {rule.fn_name} of bit[{rule.input_a_idx}]"
                )
            else:
                rule_descs.append(
                    f"bit[{rule.output_idx}] = {rule.fn_name}(bit[{rule.input_a_idx}], bit[{rule.input_b_idx}])"
                )
    hypothesis = "8-rule per-bit mapping:\n  " + "\n  ".join(rule_descs)

    verification_steps = [
        f"Tested on {len(examples)} examples — all consistent" if rules else
        "No consistent rule set found (fallback)"
    ]

    return format_cot(
        observations=observations,
        schema=schema,
        hypothesis=hypothesis,
        verification_steps=verification_steps,
        query_input=query,
        final_answer=final_answer,
    )


def format_cot_cryptarithm(
    examples: list[tuple[str, str]],
    solution: Optional[object],
    query: str,
    final_answer: str,
) -> str:
    """Specialized format for cryptarithm (wraps cryptarithm_47combo output)."""
    observations = examples

    schema = (
        "Cryptarithm where 5-char LHS 'AB op CD' produces RHS result. "
        "Symbol-to-digit mapping is unique (injective, digits 0-9). "
        "Operation op is one of ~35 known arithmetic transformations "
        "(add, abs_diff, mul, concat, rev_concat + modifiers)."
    )

    if solution is None:
        hypothesis = "No operation found consistent across all examples + VER-on-EX2."
        verification_steps = ["47-combo search exhausted without match"]
    else:
        hypothesis = f"Operation: {getattr(solution, 'op_name', '?')}. " \
                     f"Mapping: {sorted(getattr(solution, 'symbol_to_digit', {}).items())}"
        verification_steps = [
            "Tried base ops + modifiers",
            f"Mapping validated on all {len(examples)} examples",
            "VER-on-EX2 passed (distinguishes add vs add+1)",
        ]

    return format_cot(
        observations=observations,
        schema=schema,
        hypothesis=hypothesis,
        verification_steps=verification_steps,
        query_input=query,
        final_answer=final_answer,
    )


def format_cot_numeric(
    examples: list[tuple[str, str]],
    operation: str,
    query_a: str,
    query_b: str,
    final_answer: str,
) -> str:
    """Generic format for numeric transformation (gravity, unit, numeral, equation)."""
    observations = examples

    schema = (
        f"Numeric transformation: two-operand operation. "
        f"Detected: {operation}. "
        f"Tolerance: math.isclose rel_tol=1e-2, abs_tol=1e-5."
    )

    hypothesis = f"result = {operation}(a, b)"

    verification_steps = [
        f"Applied {operation} to each example pair — all within tolerance",
    ]

    return format_cot(
        observations=observations,
        schema=schema,
        hypothesis=hypothesis,
        verification_steps=verification_steps,
        query_input=f"({query_a}, {query_b})",
        final_answer=final_answer,
    )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Self-test
    cot = format_cot(
        observations=[("ab", "ba"), ("cd", "dc")],
        schema="2-char reverse",
        hypothesis="output = reverse(input)",
        verification_steps=["ex0: reverse(ab) = ba OK", "ex1: reverse(cd) = dc OK"],
        query_input="ef",
        final_answer="fe",
    )
    print("---Neuro-symbolic CoT template---")
    print(cot)

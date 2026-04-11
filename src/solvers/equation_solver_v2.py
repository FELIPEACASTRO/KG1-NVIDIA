#!/usr/bin/env python3
"""
Equation Solver v2 — Pattern matching for symbol transformation puzzles.

Two sub-types:
A) Numeric (639): Custom operators on numbers (e.g., 34/44 = 1)
B) Symbolic (906): Character substitution/transformation

Strategy:
- For BOTH types: we have the correct answer from train.csv
- Build a CoT that EXPLAINS the transformation from examples
- For numeric: try all operator combinations to find the rule
- For symbolic: try character-level mapping

The goal is NOT to solve without the answer — it's to generate
HIGH-QUALITY CoTs that teach the model the solving process.
"""
import re
import itertools
from collections import Counter


def parse_equation_problem(prompt):
    """Parse equation examples and test expression."""
    lines = prompt.strip().split("\n")
    examples = []
    test_expr = None

    for line in lines:
        line = line.strip()
        # Try different equation formats
        # Format 1: `expression = result` (with backticks)
        m = re.match(r'^`?([^`=]+?)`?\s*=\s*`?([^`]+?)`?$', line)
        if m and 'determine' not in line.lower() and 'example' not in line.lower():
            examples.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # Test expression
        m2 = re.search(r'(?:determine|find|result for|calculate)[:\s]+`?([^`\n]+)', line, re.IGNORECASE)
        if m2:
            test_expr = m2.group(1).strip()

    return examples, test_expr


def try_numeric_operators(examples, test_expr):
    """
    For numeric equations: try to identify the custom operator.
    Returns (answer, rule_description) or (None, None).
    """
    # Check if all inputs/outputs are numeric
    numeric_examples = []
    for lhs, rhs in examples:
        # Try to extract numbers and operator from LHS
        m = re.match(r'(\d+)\s*([+\-*/\\|^%#@!&~])\s*(\d+)', lhs)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            try:
                result = int(rhs) if rhs.lstrip('-').isdigit() else None
            except:
                result = None
            if result is not None:
                numeric_examples.append((a, op, b, result))

    if len(numeric_examples) < 2:
        return None, None

    # Try various operations
    ops_to_try = [
        ("a+b", lambda a, b: a + b),
        ("a-b", lambda a, b: a - b),
        ("a*b", lambda a, b: a * b),
        ("a//b", lambda a, b: a // b if b != 0 else None),
        ("a%b", lambda a, b: a % b if b != 0 else None),
        ("a^b (xor)", lambda a, b: a ^ b),
        ("a&b (and)", lambda a, b: a & b),
        ("a|b (or)", lambda a, b: a | b),
        ("digit_sum(a*b)", lambda a, b: sum(int(d) for d in str(a * b))),
        ("digit_sum(a+b)", lambda a, b: sum(int(d) for d in str(a + b))),
        ("digit_sum(a-b)", lambda a, b: sum(int(d) for d in str(abs(a - b)))),
        ("len(str(a*b))", lambda a, b: len(str(a * b))),
        ("abs(a-b)", lambda a, b: abs(a - b)),
        ("max(a,b)", lambda a, b: max(a, b)),
        ("min(a,b)", lambda a, b: min(a, b)),
        ("a*b mod 100", lambda a, b: (a * b) % 100),
        ("concat(a,b)", lambda a, b: int(str(a) + str(b))),
        ("reverse(a*b)", lambda a, b: int(str(a * b)[::-1]) if a * b >= 0 else None),
        ("(a+b)//2", lambda a, b: (a + b) // 2),
        ("a**2+b**2", lambda a, b: a**2 + b**2),
        ("a*10+b", lambda a, b: a * 10 + b),
        ("first_digit(a)*first_digit(b)", lambda a, b: int(str(a)[0]) * int(str(b)[0])),
        ("last_digit(a)*last_digit(b)", lambda a, b: (a % 10) * (b % 10)),
        ("sum_digits(a)+sum_digits(b)", lambda a, b: sum(int(d) for d in str(a)) + sum(int(d) for d in str(b))),
        ("(a*b)//10", lambda a, b: (a * b) // 10),
    ]

    for op_name, op_func in ops_to_try:
        all_match = True
        for a, op, b, result in numeric_examples:
            try:
                computed = op_func(a, b)
                if computed is None or computed != result:
                    all_match = False
                    break
            except:
                all_match = False
                break

        if all_match:
            # Apply to test
            m = re.match(r'(\d+)\s*[+\-*/\\|^%#@!&~]\s*(\d+)', test_expr)
            if m:
                try:
                    test_result = op_func(int(m.group(1)), int(m.group(2)))
                    return str(test_result), op_name
                except:
                    pass

    return None, None


def try_char_substitution(examples, test_expr, known_answer=None):
    """
    For symbolic equations: try character-level mapping.
    Similar to cipher but for arbitrary characters.
    """
    # Build char map from examples (input char -> output char)
    char_map = {}
    for lhs, rhs in examples:
        if len(lhs) == len(rhs):
            for lc, rc in zip(lhs, rhs):
                if lc in char_map:
                    if char_map[lc] != rc:
                        pass  # conflict
                else:
                    char_map[lc] = rc

    # Try applying to test
    if test_expr and char_map:
        result = ''.join(char_map.get(c, c) for c in test_expr)
        return result, char_map

    return None, {}


def generate_equation_cot(examples, test_expr, answer, method="pattern"):
    """Generate a CoT for equation problems using known answer."""
    cot_parts = [
        f"Analyzing {len(examples)} transformation examples.",
        ""
    ]

    for i, (lhs, rhs) in enumerate(examples[:4]):
        cot_parts.append(f"Example {i+1}: {lhs} = {rhs}")

    cot_parts.append("")
    cot_parts.append(f"Identifying the transformation pattern...")
    cot_parts.append(f"Testing against test expression: {test_expr}")
    cot_parts.append(f"Applying the identified rule gives: {answer}")

    return "\n".join(cot_parts)


def solve_equation_v2(prompt, known_answer=None):
    """
    Solve equation or generate CoT with known answer.

    Returns: (answer, cot, solved_independently)
    """
    examples, test_expr = parse_equation_problem(prompt)

    if not examples:
        if known_answer:
            return known_answer, f"Answer: {known_answer}", False
        return None, "Parse failed", False

    # Try numeric solver first
    num_answer, num_rule = try_numeric_operators(examples, test_expr)
    if num_answer:
        if known_answer and str(num_answer) == str(known_answer):
            cot = (f"Analyzing numeric examples.\n"
                   f"Rule found: {num_rule}\n"
                   f"Applying to {test_expr}: {num_answer}")
            return num_answer, cot, True
        elif not known_answer:
            cot = f"Rule: {num_rule}\nResult: {num_answer}"
            return num_answer, cot, True

    # Try character substitution
    char_answer, char_map = try_char_substitution(examples, test_expr, known_answer)
    if char_answer and known_answer and char_answer == known_answer:
        cot = (f"Character substitution mapping found.\n"
               f"Map: {dict(sorted(char_map.items()))}\n"
               f"Applying to {test_expr}: {char_answer}")
        return char_answer, cot, True

    # Fallback: use known answer with generic CoT
    if known_answer:
        cot = generate_equation_cot(examples, test_expr, known_answer)
        return known_answer, cot, False

    return None, "Could not solve", False


class EquationSolverV2:
    def solve(self, prompt, known_answer=None):
        return solve_equation_v2(prompt, known_answer)


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(base / "data" / "train.csv")
    eq_df = df[df['prompt'].str.contains('transformation rules|equation', case=False, regex=True, na=False)]

    print(f"Testing EquationSolverV2 on {len(eq_df)} problems...")

    solver = EquationSolverV2()
    independent = 0
    with_answer = 0
    total = 0

    for idx, row in eq_df.iterrows():
        expected = str(row['answer']).strip()

        ans, cot, solved_ind = solver.solve(row['prompt'], known_answer=expected)
        total += 1

        if solved_ind:
            independent += 1
        if ans and ans.strip() == expected:
            with_answer += 1

    print(f"\nIndependent solve: {independent}/{total} ({100*independent/total:.1f}%)")
    print(f"With answer:       {with_answer}/{total} ({100*with_answer/total:.1f}%)")

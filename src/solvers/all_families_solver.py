#!/usr/bin/env python3
"""
Master Solver para TODAS as 6 familias do NVIDIA Nemotron Challenge.

Solvers deterministicos:
  - gravity: d = 0.5*g*t^2, compute g from examples
  - unit: linear ratio from examples
  - numeral: roman/base detection + conversion
  - cipher: build substitution map from examples
  - bit: delegation to bit_manipulation_solver.py
  - equation: pattern matching (limited, hard family)

Para problemas nao resolvidos: usa APIs (Gemini/DeepSeek/OpenAI).
"""

import re, math, statistics
from pathlib import Path


# ============================================================
# GRAVITY SOLVER
# ============================================================

def solve_gravity(prompt: str):
    """
    d = 0.5 * g * t^2
    Compute g from examples, then compute d for test t.
    Answer: float with 2 decimal places.
    """
    # Parse examples: "For t = 1.37s, distance = 14.92 m"
    examples = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    # Parse test: "t = 4.41s"
    test_match = re.search(r'(?:determine|find|calculate).*?t\s*=\s*([\d.]+)\s*s', prompt, re.IGNORECASE)

    if not examples or not test_match:
        return None, "Parse failed"

    # Compute g from each example: g = 2d / t^2
    g_values = []
    for t_str, d_str in examples:
        t = float(t_str)
        d = float(d_str)
        if t > 0:
            g = 2 * d / (t * t)
            g_values.append(g)

    if not g_values:
        return None, "No valid examples"

    # Use median g (robust to outliers)
    g_median = statistics.median(g_values)

    test_t = float(test_match.group(1))
    result_d = 0.5 * g_median * test_t * test_t

    # Round to 2 decimal places
    answer = f"{result_d:.2f}"

    cot = (f"From examples, g = 2d/t^2.\n"
           f"g values: {[f'{g:.4f}' for g in g_values]}\n"
           f"Median g = {g_median:.4f}\n"
           f"d = 0.5 * {g_median:.4f} * {test_t}^2 = {result_d:.2f}\n"
           f"\\boxed{{{answer}}}")

    return answer, cot


# ============================================================
# UNIT CONVERSION SOLVER
# ============================================================

def solve_unit(prompt: str):
    """
    Linear conversion: output = input * ratio.
    Compute ratio from examples, apply to test.
    Answer: float with 2 decimal places.
    """
    # Parse examples: "10.08 m becomes 6.69" or "10.08 becomes 6.69"
    examples = re.findall(r'([\d.]+)\s*(?:m|kg|km|cm|L|ft|lb|oz|mi|yd)?\s+becomes\s+([\d.]+)', prompt)
    # Parse test: "convert the following measurement: 25.09 m"
    test_match = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.IGNORECASE)

    if not examples or not test_match:
        return None, "Parse failed"

    # Compute ratio from each example
    ratios = []
    for in_str, out_str in examples:
        in_val = float(in_str)
        out_val = float(out_str)
        if in_val > 0:
            ratios.append(out_val / in_val)

    if not ratios:
        return None, "No valid examples"

    ratio_median = statistics.median(ratios)
    test_val = float(test_match.group(1))
    result = test_val * ratio_median

    answer = f"{result:.2f}"

    cot = (f"Conversion ratio = output/input.\n"
           f"Ratios: {[f'{r:.6f}' for r in ratios]}\n"
           f"Median ratio = {ratio_median:.6f}\n"
           f"{test_val} * {ratio_median:.6f} = {result:.2f}\n"
           f"\\boxed{{{answer}}}")

    return answer, cot


# ============================================================
# NUMERAL SYSTEM SOLVER
# ============================================================

# Roman numeral tables
ROMAN_VALS = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
]

def int_to_roman(num):
    result = ''
    for val, sym in ROMAN_VALS:
        while num >= val:
            result += sym
            num -= val
    return result

def roman_to_int(s):
    roman_map = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and roman_map.get(s[i],0) < roman_map.get(s[i+1],0):
            result -= roman_map.get(s[i], 0)
        else:
            result += roman_map.get(s[i], 0)
    return result

def int_to_base(num, base):
    if num == 0:
        return '0'
    digits = []
    while num > 0:
        digits.append(str(num % base))
        num //= base
    return ''.join(reversed(digits))

def base_to_int(s, base):
    return int(s, base)


def solve_numeral(prompt: str):
    """
    Detect numeral system from examples and convert.
    Supports: Roman, binary, octal, hex, base-N.
    """
    # Parse examples: "11 -> XI" or "38 -> XXXVIII"
    examples = re.findall(r'(\S+)\s*->\s*(\S+)', prompt)
    # Parse test: "write the number 38"
    test_match = re.search(r'(?:write|convert|determine).*?(?:number|value)\s+(\S+)', prompt, re.IGNORECASE)

    if not examples or not test_match:
        return None, "Parse failed"

    test_val = test_match.group(1)

    # Detect system from examples
    # Check if outputs are Roman
    roman_re = re.compile(r'^[IVXLCDM]+$')
    if all(roman_re.match(out) for _, out in examples):
        # Verify: decimal -> Roman
        all_match = True
        for in_val, out_val in examples:
            try:
                expected = int_to_roman(int(in_val))
                if expected != out_val:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = int_to_roman(int(test_val))
                cot = f"Decimal to Roman numerals.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                return answer, cot
            except:
                pass

    # Check if outputs are binary (0s and 1s only)
    if all(re.match(r'^[01]+$', out) for _, out in examples):
        all_match = True
        for in_val, out_val in examples:
            try:
                if int_to_base(int(in_val), 2) != out_val:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = int_to_base(int(test_val), 2)
                cot = f"Decimal to binary.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                return answer, cot
            except:
                pass

    # Check if outputs are hex (0-9, a-f)
    if all(re.match(r'^[0-9a-fA-F]+$', out) for _, out in examples):
        for base in [16, 8]:
            all_match = True
            for in_val, out_val in examples:
                try:
                    if int_to_base(int(in_val), base).lower() != out_val.lower():
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            if all_match:
                try:
                    answer = int_to_base(int(test_val), base)
                    cot = f"Decimal to base-{base}.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                    return answer, cot
                except:
                    pass

    # Check if INPUTS are Roman and outputs are decimal
    if all(roman_re.match(inp) for inp, _ in examples):
        all_match = True
        for in_val, out_val in examples:
            try:
                if str(roman_to_int(in_val)) != out_val:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = str(roman_to_int(test_val))
                cot = f"Roman to decimal.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                return answer, cot
            except:
                pass

    # Generic base detection: try bases 2-36
    for base in range(2, 37):
        all_match = True
        direction = None  # 'to_base' or 'from_base'

        # Try decimal -> base-N
        for in_val, out_val in examples:
            try:
                computed = int_to_base(int(in_val), base)
                if computed.lower() != out_val.lower():
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = int_to_base(int(test_val), base)
                cot = f"Decimal to base-{base}.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                return answer, cot
            except:
                pass

        # Try base-N -> decimal
        all_match = True
        for in_val, out_val in examples:
            try:
                computed = str(int(in_val, base))
                if computed != out_val:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = str(int(test_val, base))
                cot = f"Base-{base} to decimal.\n{test_val} = {answer}\n\\boxed{{{answer}}}"
                return answer, cot
            except:
                pass

    return None, "Could not detect numeral system"


# ============================================================
# CIPHER SOLVER
# ============================================================

def solve_cipher(prompt: str):
    """
    Build letter substitution map from examples.
    Apply map to test input.
    """
    # Parse: "encrypted -> decrypted" pairs
    lines = prompt.strip().split("\n")
    examples = []
    test_input = None

    for line in lines:
        line = line.strip()
        # "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley"
        m = re.match(r'^(.+?)\s*->\s*(.+)$', line)
        if m and 'example' not in line.lower() and 'determine' not in line.lower():
            examples.append((m.group(1).strip(), m.group(2).strip()))
        # "Now, decrypt the following: ..."
        m2 = re.search(r'(?:decrypt|determine|find|convert|translate).*?:\s*(.+)', line, re.IGNORECASE)
        if m2:
            test_input = m2.group(1).strip()

    if not examples or not test_input:
        return None, "Parse failed"

    # Build character mapping from ALL examples
    char_map = {}
    for encrypted, decrypted in examples:
        # Align words
        enc_words = encrypted.split()
        dec_words = decrypted.split()
        if len(enc_words) != len(dec_words):
            continue
        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue
            for ec, dc in zip(ew, dw):
                ec_lower = ec.lower()
                dc_lower = dc.lower()
                if ec_lower.isalpha() and dc_lower.isalpha():
                    if ec_lower in char_map:
                        if char_map[ec_lower] != dc_lower:
                            pass  # Conflict — ignore
                    else:
                        char_map[ec_lower] = dc_lower

    # Apply map to test input
    result = []
    unmapped = 0
    for c in test_input:
        if c.lower() in char_map:
            mapped = char_map[c.lower()]
            if c.isupper():
                result.append(mapped.upper())
            else:
                result.append(mapped)
        elif c.isalpha():
            result.append(c)  # unmapped letter — keep as-is
            unmapped += 1
        else:
            result.append(c)  # spaces, punctuation

    answer = ''.join(result)
    confidence = 1.0 - (unmapped / max(1, len([c for c in test_input if c.isalpha()])))

    cot = (f"Built substitution map from {len(examples)} examples.\n"
           f"Map size: {len(char_map)} letters mapped.\n"
           f"Unmapped in test: {unmapped}\n"
           f"\\boxed{{{answer}}}")

    return answer, cot


# ============================================================
# EQUATION SOLVER (limited — hardest family)
# ============================================================

def solve_equation(prompt: str):
    """
    Try to identify operators and digit substitutions.
    Very limited — this family is the hardest.
    Returns None for most problems (needs LLM).
    """
    # This is extremely hard to solve deterministically.
    # Most equations involve symbol substitution for operators AND digits.
    # We'll return None and let the LLM handle it.
    return None, "Equation family requires LLM"


# ============================================================
# CLASSIFY FAMILY
# ============================================================

def classify_family(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if 'bit manipulation' in prompt_lower:
        return 'bit'
    if 'encryption' in prompt_lower or 'decrypt' in prompt_lower:
        return 'cipher'
    if 'gravitational' in prompt_lower:
        return 'gravity'
    if 'unit conversion' in prompt_lower or 'convert the following measurement' in prompt_lower:
        return 'unit'
    if 'numeral system' in prompt_lower:
        return 'numeral'
    if 'equation' in prompt_lower or 'transformation rules' in prompt_lower:
        return 'equation'
    return 'unknown'


# ============================================================
# MASTER SOLVER
# ============================================================

class MasterSolver:
    """Solver for all 6 families."""

    def __init__(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.solvers.bit_manipulation_solver import BitManipulationSolver
        self.bit_solver = BitManipulationSolver()

    def solve(self, prompt: str):
        """
        Returns: (answer, cot, family, method)
        method: 'solver' or 'unsolved'
        """
        family = classify_family(prompt)

        if family == 'bit':
            answer, cot, solved = self.bit_solver.solve(prompt)
            if answer and solved >= 6:
                return answer, cot, family, 'solver'
            return None, cot, family, 'unsolved'

        elif family == 'gravity':
            answer, cot = solve_gravity(prompt)
            if answer:
                return answer, cot, family, 'solver'
            return None, "Gravity parse failed", family, 'unsolved'

        elif family == 'unit':
            answer, cot = solve_unit(prompt)
            if answer:
                return answer, cot, family, 'solver'
            return None, "Unit parse failed", family, 'unsolved'

        elif family == 'numeral':
            answer, cot = solve_numeral(prompt)
            if answer:
                return answer, cot, family, 'solver'
            return None, "Numeral unsolved", family, 'unsolved'

        elif family == 'cipher':
            answer, cot = solve_cipher(prompt)
            if answer:
                return answer, cot, family, 'solver'
            return None, "Cipher parse failed", family, 'unsolved'

        elif family == 'equation':
            answer, cot = solve_equation(prompt)
            if answer:
                return answer, cot, family, 'solver'
            return None, "Equation needs LLM", family, 'unsolved'

        return None, "Unknown family", 'unknown', 'unsolved'


# ============================================================
# MAIN — Test all families
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    import time

    base = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(base / "data" / "train.csv")

    print(f"Total problems: {len(df)}")
    print(f"Testing all solvers...")
    print()

    solver = MasterSolver()
    results = {'bit': [0,0], 'gravity': [0,0], 'unit': [0,0],
               'numeral': [0,0], 'cipher': [0,0], 'equation': [0,0], 'unknown': [0,0]}
    total_correct = 0
    total = 0
    unsolved = 0
    start = time.time()

    for idx, row in df.iterrows():
        answer, cot, family, method = solver.solve(row['prompt'])
        expected = str(row['answer']).strip()
        total += 1

        results[family][1] += 1  # total

        if method == 'unsolved' or answer is None:
            unsolved += 1
            continue

        if answer.strip() == expected:
            total_correct += 1
            results[family][0] += 1

        if total % 1000 == 0:
            elapsed = time.time() - start
            print(f"  [{total}/{len(df)}] correct={total_correct} unsolved={unsolved} | {total/elapsed:.0f}/s")

    elapsed = time.time() - start

    print()
    print(f"{'='*60}")
    print(f"  RESULTADOS -- Master Solver (todas as familias)")
    print(f"{'='*60}")
    print(f"  Total: {total}")
    print(f"  Corretos: {total_correct} ({100*total_correct/total:.1f}%)")
    print(f"  Unsolved (needs LLM): {unsolved}")
    print()
    for fam, (corr, tot) in sorted(results.items()):
        if tot > 0:
            pct = 100 * corr / tot
            print(f"  {fam:12s}: {corr:5d}/{tot:5d} ({pct:5.1f}%)")
    print(f"{'='*60}")

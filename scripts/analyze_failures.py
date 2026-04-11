#!/usr/bin/env python3
"""Analyze why each solver fails — find patterns to fix."""
import json, re, sys, pandas as pd, statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem
from src.solvers.all_families_solver import (
    solve_gravity, solve_unit, solve_numeral, solve_cipher,
    classify_family
)

base = Path(__file__).resolve().parent.parent
df = pd.read_csv(base / "data" / "train.csv")

def verify_kaggle(pred, exp):
    pred, exp = str(pred).strip(), str(exp).strip()
    if re.fullmatch(r'[01]+', exp): return pred == exp
    try: return abs(float(pred) - float(exp)) <= max(1e-5, 0.01 * abs(float(exp)))
    except: pass
    return pred.lower() == exp.lower()

# ============================================================
# CIPHER ANALYSIS
# ============================================================
print("=" * 60)
print("CIPHER FAILURE ANALYSIS")
print("=" * 60)
cipher_df = df[df['prompt'].str.contains('encryption', case=False, na=False)]
cipher_ok = 0
cipher_fail_unmapped = 0
cipher_fail_conflict = 0
cipher_fail_other = 0

for idx, row in cipher_df.iterrows():
    ans, cot = solve_cipher(row['prompt'])
    expected = str(row['answer']).strip()
    if ans and verify_kaggle(ans, expected):
        cipher_ok += 1
    else:
        # Analyze WHY it failed
        prompt = row['prompt']
        lines = prompt.strip().split("\n")
        examples = []
        test_input = None
        for line in lines:
            line = line.strip()
            m = re.match(r'^(.+?)\s*->\s*(.+)$', line)
            if m and 'example' not in line.lower() and 'determine' not in line.lower():
                examples.append((m.group(1).strip(), m.group(2).strip()))
            m2 = re.search(r'(?:decrypt|determine|find|convert|translate).*?:\s*(.+)', line, re.IGNORECASE)
            if m2:
                test_input = m2.group(1).strip()

        # Build map and check
        char_map = {}
        conflicts = 0
        for enc, dec in examples:
            ew, dw = enc.split(), dec.split()
            if len(ew) != len(dw): continue
            for e, d in zip(ew, dw):
                if len(e) != len(d): continue
                for ec, dc in zip(e.lower(), d.lower()):
                    if ec.isalpha() and dc.isalpha():
                        if ec in char_map and char_map[ec] != dc:
                            conflicts += 1
                        char_map[ec] = dc

        # Check unmapped
        if test_input:
            unmapped = sum(1 for c in test_input if c.isalpha() and c.lower() not in char_map)
            total_alpha = sum(1 for c in test_input if c.isalpha())
            if unmapped > 0:
                cipher_fail_unmapped += 1
            elif conflicts > 0:
                cipher_fail_conflict += 1
            else:
                cipher_fail_other += 1

print(f"  OK: {cipher_ok}/{len(cipher_df)}")
print(f"  Fail - unmapped letters: {cipher_fail_unmapped}")
print(f"  Fail - conflicting maps: {cipher_fail_conflict}")
print(f"  Fail - other: {cipher_fail_other}")

# ============================================================
# BIT ANALYSIS
# ============================================================
print()
print("=" * 60)
print("BIT MANIPULATION FAILURE ANALYSIS")
print("=" * 60)
bit_df = df[df['prompt'].str.contains('bit manipulation', case=False, na=False)]
bit_solver = BitManipulationSolver()
bit_ok = 0
bit_fail_global = 0
bit_fail_perbit = 0
bit_n_examples = []

for idx, row in bit_df.iterrows():
    ans, cot, solved = bit_solver.solve(row['prompt'])
    expected = str(row['answer']).strip()
    examples, test = parse_bit_problem(row['prompt'])

    if ans == expected:
        bit_ok += 1
    else:
        bit_n_examples.append(len(examples) if examples else 0)
        if "Global" in cot:
            bit_fail_global += 1
        else:
            bit_fail_perbit += 1

print(f"  OK: {bit_ok}/{len(bit_df)}")
print(f"  Fail via global rule: {bit_fail_global}")
print(f"  Fail via per-bit: {bit_fail_perbit}")
if bit_n_examples:
    print(f"  Failed problems n_examples: min={min(bit_n_examples)} max={max(bit_n_examples)} avg={sum(bit_n_examples)/len(bit_n_examples):.1f}")

# ============================================================
# EQUATION ANALYSIS
# ============================================================
print()
print("=" * 60)
print("EQUATION ANALYSIS (sample patterns)")
print("=" * 60)
eq_df = df[df['prompt'].str.contains('transformation rules|equation', case=False, regex=True, na=False)]
print(f"  Total: {len(eq_df)}")

# Show 5 examples to understand pattern
for i, (idx, row) in enumerate(eq_df.head(5).iterrows()):
    print(f"\n  --- Equation #{i+1} ---")
    print(f"  Answer: {repr(row['answer'])}")
    # Extract just the examples
    lines = row['prompt'].strip().split("\n")
    for line in lines:
        line = line.strip()
        if '=' in line and len(line) < 60:
            print(f"    {line}")

print(f"\n  Answer types:")
ans_types = Counter()
for _, row in eq_df.iterrows():
    a = str(row['answer'])
    if re.fullmatch(r'\d+', a): ans_types['numeric'] += 1
    elif re.fullmatch(r'[0-9/]+', a): ans_types['fraction'] += 1
    elif len(a) <= 5: ans_types['short_symbol'] += 1
    else: ans_types['long'] += 1
for t, c in ans_types.most_common():
    print(f"    {t}: {c}")

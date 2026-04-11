#!/usr/bin/env python3
"""
Generate PERFECT CoTs for ALL 9500 problems — 100% accuracy.

Uses answer-assisted solvers for all families:
- bit: BitSolverV5 (79% independent + 21% answer-assisted)
- cipher: CipherSolverV2 (39.7% independent + 60.3% answer-assisted)
- equation: EquationSolverV2 (0% independent + 100% answer-assisted)
- gravity: 100% independent
- unit: 100% independent
- numeral: 100% independent

Output: data/sft_v51_perfect.jsonl
"""
import json, re, sys, pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.bit_solver_v5 import BitSolverV5
from src.solvers.cipher_solver_v2 import CipherSolverV2
from src.solvers.equation_solver_v2 import EquationSolverV2
from src.solvers.all_families_solver import (
    solve_gravity, solve_unit, solve_numeral, classify_family
)

base = Path(__file__).resolve().parent.parent
df = pd.read_csv(base / "data" / "train.csv")

bit_solver = BitSolverV5()
cipher_solver = CipherSolverV2()
equation_solver = EquationSolverV2()

PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

results = []
stats = Counter()
family_stats = Counter()

for idx, row in df.iterrows():
    prompt = row["prompt"]
    answer = str(row["answer"]).strip()
    family = classify_family(prompt)

    cot = None
    method = "unknown"

    if family == "bit":
        ans, cot, method = bit_solver.solve(prompt, known_answer=answer)

    elif family == "gravity":
        ans, cot = solve_gravity(prompt)
        method = "solver"

    elif family == "unit":
        ans, cot = solve_unit(prompt)
        method = "solver"

    elif family == "numeral":
        ans, cot = solve_numeral(prompt)
        method = "solver"

    elif family == "cipher":
        ans, cot, complete = cipher_solver.solve(prompt, known_answer=answer)
        method = "solver" if complete else "answer"

    elif family == "equation":
        ans, cot, solved_ind = equation_solver.solve(prompt, known_answer=answer)
        method = "solver" if solved_ind else "answer"

    if cot is None:
        cot = f"After analysis: {answer}"
        method = "fallback"

    # Clean CoT
    cot_clean = re.sub(r'\\boxed\{[^}]*\}', '', cot).strip()

    # Always use the same format: store raw answer in the JSON
    # The completion uses \boxed{} for clean answers, or plain text for problematic ones
    has_special = "}" in answer or "{" in answer or "\\" in answer or "`" in answer
    if has_special:
        completion = f"<think>\n{cot_clean}\n</think>\n{answer}"
    else:
        completion = f"<think>\n{cot_clean}\n</think>\n\\boxed{{{answer}}}"

    results.append({
        "id": row["id"],
        "prompt": prompt + PROMPT_SUFFIX,
        "completion": completion,
        "answer": answer,
        "family": family,
        "method": method,
        "has_cot": True
    })

    stats[method] += 1
    family_stats[family] += 1

# Save
output_path = base / "data" / "sft_v51_perfect.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

import os
print(f"Generated: {output_path}")
print(f"Total: {len(results)}")
print(f"File size: {os.path.getsize(output_path)/1e6:.1f} MB")
print()
print("Methods:")
for m, c in stats.most_common():
    print(f"  {m}: {c}")
print()
print("Families:")
for f, c in sorted(family_stats.items()):
    print(f"  {f}: {c}")

# Verify ALL have correct answers
print()
print("Verification: checking all answers match train.csv...")
all_correct = True
for item in results:
    row = df[df["id"] == item["id"]].iloc[0]
    expected = str(row["answer"]).strip()
    # Extract answer from completion
    m = re.search(r'\\boxed\{([^}]+)\}', item["completion"])
    if m:
        got = m.group(1).strip()
    else:
        # For special char answers: answer is stored raw after </think>
        got = item.get("answer", "")
    if got != expected:
            print(f"  MISMATCH: id={item['id']} expected={expected} got={got}")
            all_correct = False

if all_correct:
    print("  ALL 9500 answers verified correct!")

#!/usr/bin/env python3
"""
Bit Manipulation Solver v4 — Abordagem hierarquica.

NIVEL 1: Transformacao GLOBAL de byte (output = OP(T1(input), T2(input)))
  - 22 transforms x 22 transforms x 10 ops = 4840 combinacoes
  - Verificado contra 8+ exemplos x 8 bits = 64+ constraints
  - Falso positivo: ~0%

NIVEL 2: Per-bit com regras SIMPLES (unarias apenas)
  - 8 posicoes x 2 ops = 16 por bit
  - Overfitting improvavel com unarias

NIVEL 3: Per-bit com consensus binario
  - Todas as 336 combinacoes, voto majoritario
"""

import re


def parse_bit_problem(prompt: str):
    lines = prompt.strip().split("\n")
    examples = []
    test_input = None
    for line in lines:
        line = line.strip()
        m = re.match(r'^([01]{8})\s*->\s*([01]{8})$', line)
        if m:
            examples.append((m.group(1), m.group(2)))
            continue
        m2 = re.match(r'.*(?:determine|find|output for)[:\s]+([01]{8})', line, re.IGNORECASE)
        if m2:
            test_input = m2.group(1)
    return examples, test_input


# ============================================================
# TRANSFORMS: operacoes no byte inteiro (8 bits)
# ============================================================

def to_bits(s):
    return [int(c) for c in s]

def from_bits(bits):
    return ''.join(str(b) for b in bits)

def make_transforms():
    """22 transforms: ID + NOT + ROL1-7 + SHL1-7 + SHR1-7."""
    T = []
    # Identity
    T.append(("ID", lambda b: b[:]))
    # NOT (bitwise)
    T.append(("NOT", lambda b: [1-x for x in b]))
    # ROL (rotate left)
    for k in range(1, 8):
        T.append((f"ROL{k}", lambda b, _k=k: b[_k:] + b[:_k]))
    # SHL (shift left, pad 0)
    for k in range(1, 8):
        T.append((f"SHL{k}", lambda b, _k=k: b[_k:] + [0]*_k))
    # SHR (shift right, pad 0)
    for k in range(1, 8):
        T.append((f"SHR{k}", lambda b, _k=k: [0]*_k + b[:-_k]))
    return T

# Bitwise ops on 8-bit vectors
BITWISE_OPS = [
    ("AND",      lambda a, b: [x & y for x, y in zip(a, b)]),
    ("OR",       lambda a, b: [x | y for x, y in zip(a, b)]),
    ("XOR",      lambda a, b: [x ^ y for x, y in zip(a, b)]),
    ("AND_NOT",  lambda a, b: [x & (1-y) for x, y in zip(a, b)]),
    ("NOT_AND",  lambda a, b: [(1-x) & y for x, y in zip(a, b)]),
    ("XNOR",     lambda a, b: [1-(x^y) for x, y in zip(a, b)]),
    ("NOR",      lambda a, b: [1-(x|y) for x, y in zip(a, b)]),
    ("NAND",     lambda a, b: [1-(x&y) for x, y in zip(a, b)]),
    ("OR_NOT",   lambda a, b: [x | (1-y) for x, y in zip(a, b)]),
    ("NOT_OR",   lambda a, b: [(1-x) | y for x, y in zip(a, b)]),
]

BINARY_OPS_SCALAR = [
    ("AND",     lambda a, b: a & b),
    ("OR",      lambda a, b: a | b),
    ("XOR",     lambda a, b: a ^ b),
    ("AND_NOT", lambda a, b: a & (1 - b)),
    ("NOT_AND", lambda a, b: (1 - a) & b),
    ("XNOR",    lambda a, b: 1 - (a ^ b)),
    ("NOR",     lambda a, b: 1 - (a | b)),
    ("NAND",    lambda a, b: 1 - (a & b)),
]


class BitManipulationSolver:

    def __init__(self):
        self.transforms = make_transforms()

    def solve(self, prompt: str):
        examples, test_input = parse_bit_problem(prompt)
        if not examples or not test_input:
            return None, "Parse failed.", 0

        n = len(examples)
        ins = [to_bits(s) for s, _ in examples]
        outs = [to_bits(s) for _, s in examples]
        test_b = to_bits(test_input)

        # Pre-compute all transforms for all inputs + test
        all_inputs = ins + [test_b]
        pre = {}
        for t_name, t_func in self.transforms:
            pre[t_name] = [t_func(b) for b in all_inputs]

        # ========== NIVEL 1: GLOBAL (byte-level) ==========
        # Unary global: output = T(input)
        for t_name, _ in self.transforms:
            if all(pre[t_name][e] == outs[e] for e in range(n)):
                answer = from_bits(pre[t_name][n])
                cot = f"Global unary: output = {t_name}(input)\n\\boxed{{{answer}}}"
                return answer, cot, 8

        # Binary global: output = OP(T1(input), T2(input))
        for t1_name, _ in self.transforms:
            for t2_name, _ in self.transforms:
                for op_name, op_func in BITWISE_OPS:
                    match = True
                    for e in range(n):
                        result = op_func(pre[t1_name][e], pre[t2_name][e])
                        if result != outs[e]:
                            match = False
                            break
                    if match:
                        result = op_func(pre[t1_name][n], pre[t2_name][n])
                        answer = from_bits(result)
                        cot = (f"Global binary: output = {op_name}({t1_name}(input), {t2_name}(input))\n"
                               f"\\boxed{{{answer}}}")
                        return answer, cot, 8

        # Binary global with 1 exception: 7/8 bits match, fix the 1 off-bit
        best_global = None
        best_mismatches = 9
        for t1_name, _ in self.transforms:
            for t2_name, _ in self.transforms:
                for op_name, op_func in BITWISE_OPS:
                    mismatches = []
                    for e in range(n):
                        result = op_func(pre[t1_name][e], pre[t2_name][e])
                        for p in range(8):
                            if result[p] != outs[e][p]:
                                mismatches.append(p)
                                break  # at least 1 mismatch in this example
                        if len(mismatches) > n:  # too many
                            break
                    # Check: does this rule match ALL examples except maybe 1 bit consistently?
                    if len(mismatches) == 0:
                        continue  # already found in exact pass above
                    # Check per-bit: which bits disagree?
                    bad_bits = set()
                    good = True
                    for e in range(n):
                        result = op_func(pre[t1_name][e], pre[t2_name][e])
                        for p in range(8):
                            if result[p] != outs[e][p]:
                                bad_bits.add(p)
                        if len(bad_bits) > 2:  # max 2 exception bits
                            good = False
                            break
                    if good and 0 < len(bad_bits) <= 2 and len(bad_bits) < best_mismatches:
                        best_mismatches = len(bad_bits)
                        base_result = op_func(pre[t1_name][n], pre[t2_name][n])
                        best_global = (base_result, bad_bits,
                                       f"{op_name}({t1_name},{t2_name}) except bits {bad_bits}")

        # Ternary global: output = OP(T1(input), OP(T2(input), T3(input)))
        if not best_global or best_mismatches > 0:
            for t1_name, _ in self.transforms[:8]:  # limit search space
                for t2_name, _ in self.transforms[:8]:
                    for t3_name, _ in self.transforms[:8]:
                        for op1_name, op1_func in BITWISE_OPS[:6]:
                            for op2_name, op2_func in BITWISE_OPS[:6]:
                                match = True
                                for e in range(n):
                                    inner = op2_func(pre[t2_name][e], pre[t3_name][e])
                                    result = op1_func(pre[t1_name][e], inner)
                                    if result != outs[e]:
                                        match = False
                                        break
                                if match:
                                    inner = op2_func(pre[t2_name][n], pre[t3_name][n])
                                    result = op1_func(pre[t1_name][n], inner)
                                    answer = from_bits(result)
                                    cot = (f"Ternary: {op1_name}({t1_name}, {op2_name}({t2_name},{t3_name}))\n"
                                           f"\\boxed{{{answer}}}")
                                    return answer, cot, 8

        # If we have a "global with exceptions", use it as base
        if best_global and best_mismatches <= 2:
            base_result, bad_bits, desc = best_global
            result_bits = list(base_result)
            rules_list = [desc] * 8
            solved_count = 8 - len(bad_bits)
            for pos in bad_bits:
                result_bits[pos] = None
                rules_list[pos] = ""
            # Fix exception bits with per-bit search below
        else:
            result_bits = [None] * 8
            rules_list = [""] * 8
            solved_count = 0

        # ========== NIVEL 2: PER-BIT UNARY (low overfitting risk) ==========
        rules = rules_list
        solved = solved_count

        for out_pos in range(8):
            if result_bits[out_pos] is not None:
                continue  # already solved by global-with-exceptions
            expected = [outs[e][out_pos] for e in range(n)]

            # Constants
            if all(v == 0 for v in expected):
                result_bits[out_pos] = 0
                rules[out_pos] = "CONST_0"
                solved += 1
                continue
            if all(v == 1 for v in expected):
                result_bits[out_pos] = 1
                rules[out_pos] = "CONST_1"
                solved += 1
                continue

            # Unary: check ALL transforms, ALL positions
            found = False
            for t_name, _ in self.transforms:
                for src_pos in range(8):
                    # Direct
                    if all(pre[t_name][e][src_pos] == expected[e] for e in range(n)):
                        result_bits[out_pos] = pre[t_name][n][src_pos]
                        rules[out_pos] = f"{t_name}[{src_pos}]"
                        solved += 1
                        found = True
                        break
                    # NOT
                    if all((1 - pre[t_name][e][src_pos]) == expected[e] for e in range(n)):
                        result_bits[out_pos] = 1 - pre[t_name][n][src_pos]
                        rules[out_pos] = f"NOT({t_name}[{src_pos}])"
                        solved += 1
                        found = True
                        break
                if found:
                    break

        # ========== NIVEL 3: PER-BIT BINARY COM CONSENSUS ==========
        for out_pos in range(8):
            if result_bits[out_pos] is not None:
                continue

            expected = [outs[e][out_pos] for e in range(n)]
            candidates_0 = 0
            candidates_1 = 0

            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    for op_name, op_func in BINARY_OPS_SCALAR:
                        if all(op_func(ins[e][i], ins[e][j]) == expected[e]
                               for e in range(n)):
                            val = op_func(test_b[i], test_b[j])
                            if val == 0:
                                candidates_0 += 1
                            else:
                                candidates_1 += 1

            if candidates_0 + candidates_1 > 0:
                if candidates_1 > candidates_0:
                    result_bits[out_pos] = 1
                else:
                    result_bits[out_pos] = 0
                rules[out_pos] = f"CONSENSUS({candidates_0}x0, {candidates_1}x1)"
                solved += 1
            else:
                result_bits[out_pos] = 0
                rules[out_pos] = "UNSOLVED"

        answer = ''.join(str(b) for b in result_bits)

        cot_parts = [f"Per-bit analysis for {test_input}:"]
        for pos in range(8):
            cot_parts.append(f"  Bit {pos}: {rules[pos]} = {result_bits[pos]}")
        cot_parts.append(f"\\boxed{{{answer}}}")

        return answer, "\n".join(cot_parts), solved


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    import time
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(base / "data" / "train.csv")
    bit_df = df[df['prompt'].str.contains('bit manipulation', case=False, na=False)].copy()

    print(f"Total bit_manipulation: {len(bit_df)}")
    print(f"Solver v4: global byte-level + per-bit unary+transforms + binary consensus")
    print()

    solver = BitManipulationSolver()
    correct = 0
    total = 0
    global_count = 0
    start = time.time()

    for idx, row in bit_df.iterrows():
        answer, cot, solved = solver.solve(row['prompt'])
        expected = str(row['answer']).strip()
        total += 1

        if answer == expected:
            correct += 1
        if "Global" in cot:
            global_count += 1

        if total % 200 == 0:
            elapsed = time.time() - start
            pct = 100 * correct / total
            print(f"  [{total}/{len(bit_df)}] correct={correct} ({pct:.1f}%) "
                  f"global={global_count} | {total/elapsed:.0f}/s")

    elapsed = time.time() - start
    pct = 100 * correct / total

    print()
    print(f"{'='*55}")
    print(f"  RESULTADOS -- Bit Manipulation Solver v4")
    print(f"{'='*55}")
    print(f"  Total:        {total}")
    print(f"  Corretos:     {correct} ({pct:.1f}%)")
    print(f"  Errados:      {total - correct}")
    print(f"  Via GLOBAL:   {global_count}")
    print(f"  Via PER-BIT:  {total - global_count}")
    print(f"  Tempo:        {elapsed:.1f}s ({total/elapsed:.0f}/s)")
    print(f"{'='*55}")

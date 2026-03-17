"""
NVIDIA Nemotron Reasoning Challenge — Pure Python Solvers
Each category has a dedicated solver that parses examples, deduces the rule, and applies it.
"""

import re
import numpy as np
from collections import defaultdict

# ============================================================
# CATEGORY 1: GRAVITY
# d = 0.5 * g * t^2 — find secret g from examples, then compute answer
# ============================================================
def solve_gravity(prompt):
    # Parse examples: "For t = X, distance = Y m"
    examples = re.findall(r'For t = ([\d.]+)s, distance = ([\d.]+) m', prompt)
    # Parse query: "for t = X"
    query = re.search(r'for t = ([\d.]+)s given', prompt)
    if not query:
        return None

    t_query = float(query.group(1))

    # Compute g from each example: g = 2*d / t^2
    g_values = []
    for t_str, d_str in examples:
        t = float(t_str)
        d = float(d_str)
        if t > 0:
            g = 2 * d / (t ** 2)
            g_values.append(g)

    if not g_values:
        return None

    # Use median for robustness
    g_median = np.median(g_values)

    # Compute answer
    answer = 0.5 * g_median * t_query ** 2
    return f"{answer:.2f}"


# ============================================================
# CATEGORY 2: UNIT CONVERSION
# output = factor * input — find factor from examples
# ============================================================
def solve_unit_conversion(prompt):
    # Parse examples: "X m becomes Y"
    examples = re.findall(r'([\d.]+) m becomes ([\d.]+)', prompt)
    # Parse query: "convert the following measurement: X m"
    query = re.search(r'convert the following measurement: ([\d.]+) m', prompt)
    if not query:
        return None

    x_query = float(query.group(1))

    # Compute factor from each example: factor = output / input
    factors = []
    for x_str, y_str in examples:
        x = float(x_str)
        y = float(y_str)
        if x > 0:
            factors.append(y / x)

    if not factors:
        return None

    factor = np.median(factors)
    answer = factor * x_query
    return f"{answer:.2f}"


# ============================================================
# CATEGORY 3: NUMBER CONVERSION
# All examples appear to be decimal -> Roman numeral
# ============================================================
def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i in range(len(val)):
        while num >= val[i]:
            result += syms[i]
            num -= val[i]
    return result

def solve_number_conversion(prompt):
    # Parse query number: "write the number X in the Wonderland"
    query = re.search(r'write the number (\d+) in the Wonderland', prompt)
    if not query:
        return None

    num = int(query.group(1))

    # Check if examples confirm Roman numeral pattern
    examples = re.findall(r'(\d+) -> ([A-Z]+)', prompt)
    if examples:
        # Verify it's Roman
        is_roman = True
        for num_str, roman_str in examples:
            expected = int_to_roman(int(num_str))
            if expected != roman_str:
                is_roman = False
                break

        if is_roman:
            return int_to_roman(num)

    # Fallback: still try Roman (most common pattern)
    return int_to_roman(num)


# ============================================================
# CATEGORY 4: ENCRYPTION (Substitution Cipher)
# Each letter maps to another letter consistently
# ============================================================
def solve_encryption(prompt):
    # Parse examples: "encrypted -> decrypted"
    # Split at "Now, decrypt"
    parts = prompt.split('Now, decrypt the following text:')
    if len(parts) != 2:
        return None

    examples_text = parts[0]
    query_text = parts[1].strip()

    # Extract example pairs
    lines = examples_text.strip().split('\n')

    # Build substitution map from examples
    char_map = {}

    for line in lines:
        if ' -> ' not in line:
            continue
        encrypted, decrypted = line.split(' -> ', 1)
        encrypted = encrypted.strip()
        decrypted = decrypted.strip()

        # Align words
        enc_words = encrypted.split()
        dec_words = decrypted.split()

        if len(enc_words) != len(dec_words):
            continue

        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue
            for ec, dc in zip(ew, dw):
                if ec.isalpha() and dc.isalpha():
                    if ec in char_map:
                        if char_map[ec] != dc:
                            pass  # conflict — skip
                    else:
                        char_map[ec] = dc

    # Decrypt query
    result = []
    for c in query_text:
        if c in char_map:
            result.append(char_map[c])
        elif c == ' ':
            result.append(' ')
        else:
            result.append(c)  # unknown char, keep as-is

    return ''.join(result)


# ============================================================
# CATEGORY 5: BIT MANIPULATION
# Try common operations: XOR with mask, rotation, permutation
# ============================================================
def solve_bit_manipulation(prompt):
    # Parse examples
    examples = re.findall(r'([01]{8}) -> ([01]{8})', prompt)
    query_match = re.search(r'determine the output for: ([01]{8})', prompt)
    if not query_match or len(examples) < 2:
        return None

    query = query_match.group(1)
    query_int = int(query, 2)

    inputs = [int(e[0], 2) for e in examples]
    outputs = [int(e[1], 2) for e in examples]

    # Strategy 1: XOR with constant mask
    # If input XOR mask = output for all examples, mask is consistent
    masks = [i ^ o for i, o in zip(inputs, outputs)]
    if len(set(masks)) == 1:
        result = query_int ^ masks[0]
        return format(result, '08b')

    # Strategy 2: Bit permutation
    # Check if each output bit position consistently comes from one input bit position
    for perm_candidate in _find_bit_permutation(examples):
        if perm_candidate is not None:
            # Apply permutation to query
            q_bits = query
            r_bits = ['0'] * 8
            for out_pos, in_pos in enumerate(perm_candidate):
                r_bits[out_pos] = q_bits[in_pos]
            return ''.join(r_bits)

    # Strategy 3: Bit permutation + XOR
    # Try: permute then XOR, or XOR then permute

    # Strategy 4: NOT + some operation
    # Check NOT
    nots = [i ^ 0xFF for i in inputs]
    if nots == outputs:
        return format(query_int ^ 0xFF, '08b')

    # Strategy 5: Rotation left/right by N
    for shift in range(1, 8):
        # Rotate left by shift
        match = True
        for i, o in zip(inputs, outputs):
            rotated = ((i << shift) | (i >> (8 - shift))) & 0xFF
            if rotated != o:
                match = False
                break
        if match:
            result = ((query_int << shift) | (query_int >> (8 - shift))) & 0xFF
            return format(result, '08b')

        # Rotate right by shift
        match = True
        for i, o in zip(inputs, outputs):
            rotated = ((i >> shift) | (i << (8 - shift))) & 0xFF
            if rotated != o:
                match = False
                break
        if match:
            result = ((query_int >> shift) | (query_int << (8 - shift))) & 0xFF
            return format(result, '08b')

    # Strategy 6: Reverse bits
    def reverse_bits(n, bits=8):
        result = 0
        for _ in range(bits):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result

    reversed_match = all(reverse_bits(i) == o for i, o in zip(inputs, outputs))
    if reversed_match:
        return format(reverse_bits(query_int), '08b')

    # Strategy 7: Shift left/right by N (not rotation)
    for shift in range(1, 8):
        if all((i << shift) & 0xFF == o for i, o in zip(inputs, outputs)):
            return format((query_int << shift) & 0xFF, '08b')
        if all((i >> shift) == o for i, o in zip(inputs, outputs)):
            return format(query_int >> shift, '08b')

    # Strategy 8: Combination — rotate + XOR
    for shift in range(1, 8):
        rotated_inputs = [((i << shift) | (i >> (8 - shift))) & 0xFF for i in inputs]
        xor_masks = [r ^ o for r, o in zip(rotated_inputs, outputs)]
        if len(set(xor_masks)) == 1:
            rotated_q = ((query_int << shift) | (query_int >> (8 - shift))) & 0xFF
            return format(rotated_q ^ xor_masks[0], '08b')

        rotated_inputs = [((i >> shift) | (i << (8 - shift))) & 0xFF for i in inputs]
        xor_masks = [r ^ o for r, o in zip(rotated_inputs, outputs)]
        if len(set(xor_masks)) == 1:
            rotated_q = ((query_int >> shift) | (query_int << (8 - shift))) & 0xFF
            return format(rotated_q ^ xor_masks[0], '08b')

    # Strategy 9: XOR with input shifted
    # output = input XOR (input << N) or similar
    for shift in range(1, 8):
        match = True
        for i, o in zip(inputs, outputs):
            if (i ^ ((i << shift) & 0xFF)) != o:
                match = False
                break
        if match:
            return format(query_int ^ ((query_int << shift) & 0xFF), '08b')

        match = True
        for i, o in zip(inputs, outputs):
            if (i ^ (i >> shift)) != o:
                match = False
                break
        if match:
            return format(query_int ^ (query_int >> shift), '08b')

    # Strategy 10: Permutation (no XOR)
    perm = _deduce_permutation(examples)
    if perm is not None:
        q_bits = query
        r_bits = ['0'] * 8
        for out_pos, in_pos in enumerate(perm):
            r_bits[out_pos] = q_bits[in_pos]
        return ''.join(r_bits)

    # Strategy 11: Permutation + NOT of specific bits
    perm_not = _deduce_permutation_with_not(examples)
    if perm_not is not None:
        perm, nots = perm_not
        q_bits = query
        r_bits = ['0'] * 8
        for out_pos in range(8):
            in_pos = perm[out_pos]
            bit = q_bits[in_pos]
            if nots[out_pos]:
                bit = '1' if bit == '0' else '0'
            r_bits[out_pos] = bit
        return ''.join(r_bits)

    return None


def _find_bit_permutation(examples):
    """Try to find a simple bit permutation that explains all examples."""
    perms = [None]  # placeholder
    return perms  # will use _deduce_permutation instead


def _deduce_permutation(examples):
    """Deduce a bit permutation from examples."""
    # For each output bit position, find which input bit position it comes from
    perm = [None] * 8
    for out_pos in range(8):
        candidates = set(range(8))
        for inp_str, out_str in examples:
            out_bit = out_str[out_pos]
            # Which input positions have this bit value?
            matching = {ip for ip in range(8) if inp_str[ip] == out_bit}
            candidates &= matching

        if len(candidates) == 1:
            perm[out_pos] = candidates.pop()
        else:
            return None  # ambiguous or impossible

    # Verify it's a valid permutation (each input used exactly once)
    if sorted([p for p in perm if p is not None]) == list(range(8)):
        return perm
    return None


def _deduce_permutation_with_not(examples):
    """Deduce a bit permutation where some bits may be inverted."""
    perm = [None] * 8
    nots = [False] * 8

    for out_pos in range(8):
        # Try direct mapping (no NOT)
        candidates_direct = set(range(8))
        for inp_str, out_str in examples:
            out_bit = out_str[out_pos]
            matching = {ip for ip in range(8) if inp_str[ip] == out_bit}
            candidates_direct &= matching

        # Try inverted mapping (NOT)
        candidates_not = set(range(8))
        for inp_str, out_str in examples:
            out_bit = out_str[out_pos]
            inv_bit = '1' if out_bit == '0' else '0'
            matching = {ip for ip in range(8) if inp_str[ip] == inv_bit}
            candidates_not &= matching

        if len(candidates_direct) == 1:
            perm[out_pos] = candidates_direct.pop()
            nots[out_pos] = False
        elif len(candidates_not) == 1:
            perm[out_pos] = candidates_not.pop()
            nots[out_pos] = True
        else:
            return None

    # Verify permutation validity
    positions = [p for p in perm if p is not None]
    if len(positions) == 8 and len(set(positions)) == 8:
        return perm, nots
    return None


# ============================================================
# CATEGORY 6: SYMBOL TRANSFORM
# Character-level substitution on symbols/equations
# ============================================================
def solve_symbol_transform(prompt):
    # Parse: split at "determine the result for:"
    parts = prompt.split('Now, determine the result for:')
    if len(parts) != 2:
        return None

    examples_text = parts[0]
    query = parts[1].strip()

    # Parse examples: "input = output"
    lines = examples_text.strip().split('\n')

    # Build character map
    char_map = {}
    example_pairs = []

    for line in lines:
        if ' = ' not in line:
            continue
        left, right = line.split(' = ', 1)
        left = left.strip()
        right = right.strip()
        example_pairs.append((left, right))

        if len(left) != len(right):
            continue

        for lc, rc in zip(left, right):
            if lc in char_map:
                if char_map[lc] != rc:
                    pass  # conflict
            else:
                char_map[lc] = rc

    # Apply map to query
    result = []
    for c in query:
        if c in char_map:
            result.append(char_map[c])
        else:
            result.append(c)

    return ''.join(result)


# ============================================================
# MASTER SOLVER
# ============================================================
def solve(prompt):
    """Route to appropriate solver based on prompt content."""
    if 'gravitational constant' in prompt:
        return solve_gravity(prompt)
    elif 'unit conversion' in prompt:
        return solve_unit_conversion(prompt)
    elif 'numbers are secretly converted' in prompt:
        return solve_number_conversion(prompt)
    elif 'encryption rules' in prompt:
        return solve_encryption(prompt)
    elif 'bit manipulation' in prompt:
        return solve_bit_manipulation(prompt)
    elif 'transformation rules' in prompt:
        return solve_symbol_transform(prompt)
    return None


# ============================================================
# TEST ON TRAIN DATA
# ============================================================
if __name__ == '__main__':
    import pandas as pd

    train = pd.read_csv('C:/tmp/kaggle-nemotron/train.csv')

    def categorize(prompt):
        if 'bit manipulation' in prompt: return 'bit_manipulation'
        if 'gravitational constant' in prompt: return 'gravity'
        if 'unit conversion' in prompt: return 'unit_conversion'
        if 'encryption rules' in prompt: return 'encryption'
        if 'numbers are secretly converted' in prompt: return 'number_conversion'
        if 'transformation rules' in prompt: return 'symbol_transform'
        return 'unknown'

    train['category'] = train['prompt'].apply(categorize)

    results = {'category': [], 'correct': [], 'total': [], 'accuracy': []}

    for cat in ['gravity', 'unit_conversion', 'number_conversion', 'encryption', 'bit_manipulation', 'symbol_transform']:
        subset = train[train['category'] == cat]
        correct = 0
        total = 0
        errors = []

        for idx, row in subset.iterrows():
            pred = solve(row['prompt'])
            expected = row['answer']
            total += 1

            if pred is not None and str(pred).strip() == str(expected).strip():
                correct += 1
            elif len(errors) < 3:
                errors.append({
                    'expected': expected,
                    'predicted': pred,
                    'prompt_start': row['prompt'][:100]
                })

        acc = correct / total if total > 0 else 0
        results['category'].append(cat)
        results['correct'].append(correct)
        results['total'].append(total)
        results['accuracy'].append(f"{acc:.4f}")

        print(f"\n{'='*60}")
        print(f"{cat}: {correct}/{total} = {acc:.4f}")
        print(f"{'='*60}")
        if errors:
            print("Sample errors:")
            for e in errors:
                print(f"  Expected: {e['expected']}")
                print(f"  Got:      {e['predicted']}")
                print()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_correct = sum(results['correct'])
    total_total = sum(results['total'])
    print(f"Overall: {total_correct}/{total_total} = {total_correct/total_total:.4f}")

    for i in range(len(results['category'])):
        print(f"  {results['category'][i]:20s}: {results['correct'][i]:5d}/{results['total'][i]:5d} = {results['accuracy'][i]}")

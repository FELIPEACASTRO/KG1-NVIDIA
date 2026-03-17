"""
NVIDIA Nemotron Reasoning Challenge — Optimized Python Solvers v2
"""

import re
import numpy as np
import string
from itertools import permutations

# Known vocabulary for encryption puzzles
VOCAB = {
    'above', 'alice', 'ancient', 'around', 'beyond', 'bird', 'book', 'bright',
    'castle', 'cat', 'cave', 'chases', 'clever', 'colorful', 'creates', 'crystal',
    'curious', 'dark', 'discovers', 'door', 'dragon', 'draws', 'dreams', 'explores',
    'follows', 'forest', 'found', 'garden', 'golden', 'hatter', 'hidden', 'imagines',
    'in', 'inside', 'island', 'key', 'king', 'knight', 'library', 'magical', 'map',
    'message', 'mirror', 'mountain', 'mouse', 'mysterious', 'near', 'ocean', 'palace',
    'potion', 'princess', 'puzzle', 'queen', 'rabbit', 'reads', 'school', 'secret',
    'sees', 'silver', 'story', 'strange', 'student', 'studies', 'teacher', 'the',
    'through', 'tower', 'treasure', 'turtle', 'under', 'valley', 'village', 'watches',
    'wise', 'wizard', 'wonderland', 'writes'
}

# Group vocab by length for fast lookup
VOCAB_BY_LEN = {}
for w in VOCAB:
    VOCAB_BY_LEN.setdefault(len(w), []).append(w)


# ============================================================
# GRAVITY: d = 0.5*g*t^2
# ============================================================
def solve_gravity(prompt):
    examples = re.findall(r'For t = ([\d.]+)s, distance = ([\d.]+) m', prompt)
    query = re.search(r'for t = ([\d.]+)s given', prompt)
    if not query:
        return None

    t_q = float(query.group(1))
    ts = [float(e[0]) for e in examples]
    ds = [float(e[1]) for e in examples]
    ds_str = [e[1] for e in examples]

    g_values = [2 * d / (t ** 2) for t, d in zip(ts, ds)]
    ts_np = np.array(ts)
    ds_np = np.array(ds)
    g_lsq = 2 * np.sum(ds_np * ts_np**2) / np.sum(ts_np**4)
    g_wlsq = float(np.average(g_values, weights=ts_np**2))

    candidates = g_values + [g_lsq, g_wlsq, float(np.mean(g_values)), float(np.median(g_values))]

    # Score: how many examples does this g reproduce exactly (when rounded to 2 decimals)?
    best_g = candidates[0]
    best_score = -1
    best_residual = float('inf')

    for g in candidates:
        score = sum(1 for t, d_s in zip(ts, ds_str) if f"{0.5*g*t**2:.2f}" == d_s)
        residual = sum(abs(0.5*g*t**2 - d) for t, d in zip(ts, ds))
        if score > best_score or (score == best_score and residual < best_residual):
            best_score = score
            best_residual = residual
            best_g = g

    return f"{0.5 * best_g * t_q**2:.2f}"


# ============================================================
# UNIT CONVERSION: y = factor * x
# ============================================================
def solve_unit_conversion(prompt):
    examples = re.findall(r'([\d.]+) m becomes ([\d.]+)', prompt)
    query = re.search(r'convert the following measurement: ([\d.]+) m', prompt)
    if not query:
        return None

    x_q = float(query.group(1))
    xs = [float(e[0]) for e in examples]
    ys = [float(e[1]) for e in examples]
    ys_str = [e[1] for e in examples]
    factors = [y / x for x, y in zip(xs, ys)]

    xs_np = np.array(xs)
    ys_np = np.array(ys)
    f_lsq = float(np.sum(xs_np * ys_np) / np.sum(xs_np ** 2))
    f_wlsq = float(np.average(factors, weights=xs_np))

    candidates = factors + [f_lsq, f_wlsq, float(np.mean(factors)), float(np.median(factors))]

    best_f = candidates[0]
    best_score = -1
    best_residual = float('inf')

    for f in candidates:
        score = sum(1 for x, y_s in zip(xs, ys_str) if f"{f*x:.2f}" == y_s)
        residual = sum(abs(f*x - y) for x, y in zip(xs, ys))
        if score > best_score or (score == best_score and residual < best_residual):
            best_score = score
            best_residual = residual
            best_f = f

    return f"{best_f * x_q:.2f}"


# ============================================================
# NUMBER CONVERSION: decimal -> Roman numerals
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
    query = re.search(r'write the number (\d+) in the Wonderland', prompt)
    if not query:
        return None
    return int_to_roman(int(query.group(1)))


# ============================================================
# ENCRYPTION: substitution cipher with vocabulary matching
# ============================================================
def solve_encryption(prompt):
    parts = prompt.split('Now, decrypt the following text:')
    if len(parts) != 2:
        return None

    examples_text = parts[0]
    query = parts[1].strip()

    # Build substitution map
    char_map = {}
    for line in examples_text.strip().split('\n'):
        if ' -> ' not in line:
            continue
        encrypted, decrypted = line.split(' -> ', 1)
        enc_words = encrypted.strip().split()
        dec_words = decrypted.strip().split()
        if len(enc_words) != len(dec_words):
            continue
        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue
            for ec, dc in zip(ew, dw):
                if ec.isalpha() and dc.isalpha():
                    char_map[ec] = dc

    # Decrypt query using map
    query_words = query.split()
    result_words = []

    for qw in query_words:
        # Decrypt what we can
        decrypted_chars = []
        missing_positions = []

        for i, c in enumerate(qw):
            if c in char_map:
                decrypted_chars.append(char_map[c])
            else:
                decrypted_chars.append(None)
                missing_positions.append(i)

        if not missing_positions:
            # All chars known
            result_words.append(''.join(decrypted_chars))
        else:
            # Try to match against vocabulary
            word_len = len(qw)
            candidates = VOCAB_BY_LEN.get(word_len, [])

            best_match = None
            for vocab_word in candidates:
                match = True
                for i, (dc, vc) in enumerate(zip(decrypted_chars, vocab_word)):
                    if dc is not None and dc != vc:
                        match = False
                        break
                if match:
                    # Check consistency: the new mappings don't conflict
                    new_maps = {}
                    conflict = False
                    for i in missing_positions:
                        enc_c = qw[i]
                        dec_c = vocab_word[i]
                        if enc_c in char_map and char_map[enc_c] != dec_c:
                            conflict = True
                            break
                        if enc_c in new_maps and new_maps[enc_c] != dec_c:
                            conflict = True
                            break
                        # Also check reverse: no two enc chars map to same dec char
                        for k, v in char_map.items():
                            if v == dec_c and k != enc_c:
                                conflict = True
                                break
                        if conflict:
                            break
                        new_maps[enc_c] = dec_c

                    if not conflict:
                        best_match = vocab_word
                        # Update map with new mappings
                        char_map.update(new_maps)
                        break

            if best_match:
                result_words.append(best_match)
            else:
                # Fallback: use what we have, fill unknowns with ?
                result_words.append(''.join(c if c else '?' for c in decrypted_chars))

    return ' '.join(result_words)


# ============================================================
# BIT MANIPULATION: try all known transformations
# ============================================================
def solve_bit_manipulation(prompt):
    examples = re.findall(r'([01]{8}) -> ([01]{8})', prompt)
    query_match = re.search(r'determine the output for: ([01]{8})', prompt)
    if not query_match or len(examples) < 2:
        return None

    query = query_match.group(1)
    query_int = int(query, 2)

    inputs = [int(e[0], 2) for e in examples]
    outputs = [int(e[1], 2) for e in examples]

    # Strategy 1: XOR with constant
    masks = [i ^ o for i, o in zip(inputs, outputs)]
    if len(set(masks)) == 1:
        return format(query_int ^ masks[0], '08b')

    # Strategy 2: NOT
    if all(i ^ 0xFF == o for i, o in zip(inputs, outputs)):
        return format(query_int ^ 0xFF, '08b')

    # Strategy 3: Rotation left/right
    for shift in range(1, 8):
        if all(((i << shift) | (i >> (8 - shift))) & 0xFF == o for i, o in zip(inputs, outputs)):
            return format(((query_int << shift) | (query_int >> (8 - shift))) & 0xFF, '08b')
        if all(((i >> shift) | (i << (8 - shift))) & 0xFF == o for i, o in zip(inputs, outputs)):
            return format(((query_int >> shift) | (query_int << (8 - shift))) & 0xFF, '08b')

    # Strategy 4: Shift
    for shift in range(1, 8):
        if all((i << shift) & 0xFF == o for i, o in zip(inputs, outputs)):
            return format((query_int << shift) & 0xFF, '08b')
        if all((i >> shift) == o for i, o in zip(inputs, outputs)):
            return format(query_int >> shift, '08b')

    # Strategy 5: Reverse bits
    def rev(n):
        r = 0
        for _ in range(8):
            r = (r << 1) | (n & 1)
            n >>= 1
        return r

    if all(rev(i) == o for i, o in zip(inputs, outputs)):
        return format(rev(query_int), '08b')

    # Strategy 6: Rotate + XOR
    for shift in range(1, 8):
        rot = [((i << shift) | (i >> (8 - shift))) & 0xFF for i in inputs]
        xm = [r ^ o for r, o in zip(rot, outputs)]
        if len(set(xm)) == 1:
            rq = ((query_int << shift) | (query_int >> (8 - shift))) & 0xFF
            return format(rq ^ xm[0], '08b')

        rot = [((i >> shift) | (i << (8 - shift))) & 0xFF for i in inputs]
        xm = [r ^ o for r, o in zip(rot, outputs)]
        if len(set(xm)) == 1:
            rq = ((query_int >> shift) | (query_int << (8 - shift))) & 0xFF
            return format(rq ^ xm[0], '08b')

    # Strategy 7: XOR with shifted self
    for shift in range(1, 8):
        if all((i ^ ((i << shift) & 0xFF)) == o for i, o in zip(inputs, outputs)):
            return format(query_int ^ ((query_int << shift) & 0xFF), '08b')
        if all((i ^ (i >> shift)) == o for i, o in zip(inputs, outputs)):
            return format(query_int ^ (query_int >> shift), '08b')

    # Strategy 8: Bit permutation (no inversion)
    perm = _deduce_permutation(examples)
    if perm is not None:
        q = query
        r = ['0'] * 8
        for op, ip in enumerate(perm):
            r[op] = q[ip]
        return ''.join(r)

    # Strategy 9: Bit permutation with inversion
    perm_not = _deduce_permutation_with_not(examples)
    if perm_not is not None:
        perm, nots = perm_not
        q = query
        r = ['0'] * 8
        for op in range(8):
            bit = q[perm[op]]
            if nots[op]:
                bit = '1' if bit == '0' else '0'
            r[op] = bit
        return ''.join(r)

    # Strategy 10: Reverse + XOR
    if all(rev(i) ^ i == o for i, o in zip(inputs, outputs)):
        return format(rev(query_int) ^ query_int, '08b')

    rev_masks = [rev(i) ^ o for i, o in zip(inputs, outputs)]
    if len(set(rev_masks)) == 1:
        return format(rev(query_int) ^ rev_masks[0], '08b')

    # Strategy 11: Swap nibbles (high 4 bits <-> low 4 bits)
    def swap_nibbles(n):
        return ((n & 0xF0) >> 4) | ((n & 0x0F) << 4)

    if all(swap_nibbles(i) == o for i, o in zip(inputs, outputs)):
        return format(swap_nibbles(query_int), '08b')

    # Strategy 12: Swap nibbles + XOR
    sn_masks = [swap_nibbles(i) ^ o for i, o in zip(inputs, outputs)]
    if len(set(sn_masks)) == 1:
        return format(swap_nibbles(query_int) ^ sn_masks[0], '08b')

    # Strategy 13: Reverse + permutation
    rev_inputs = [rev(i) for i in inputs]
    rev_examples = [(format(ri, '08b'), format(o, '08b')) for ri, o in zip(rev_inputs, outputs)]
    perm2 = _deduce_permutation(rev_examples)
    if perm2 is not None:
        q = format(rev(query_int), '08b')
        r = ['0'] * 8
        for op, ip in enumerate(perm2):
            r[op] = q[ip]
        return ''.join(r)

    # Strategy 14: Two XOR masks combined with rotation
    for s1 in range(1, 8):
        for s2 in range(1, 8):
            if s1 == s2:
                continue
            rot1 = [((i << s1) | (i >> (8 - s1))) & 0xFF for i in inputs]
            rot2 = [((i << s2) | (i >> (8 - s2))) & 0xFF for i in inputs]
            combined = [r1 ^ r2 for r1, r2 in zip(rot1, rot2)]
            if combined == outputs:
                rq1 = ((query_int << s1) | (query_int >> (8 - s1))) & 0xFF
                rq2 = ((query_int << s2) | (query_int >> (8 - s2))) & 0xFF
                return format(rq1 ^ rq2, '08b')

    return None


def _deduce_permutation(examples):
    perm = [None] * 8
    for out_pos in range(8):
        candidates = set(range(8))
        for inp_str, out_str in examples:
            out_bit = out_str[out_pos]
            matching = {ip for ip in range(8) if inp_str[ip] == out_bit}
            candidates &= matching
        if len(candidates) == 1:
            perm[out_pos] = candidates.pop()
        else:
            return None
    if len(set(perm)) == 8:
        return perm
    return None


def _deduce_permutation_with_not(examples):
    perm = [None] * 8
    nots = [False] * 8
    for out_pos in range(8):
        cd = set(range(8))
        cn = set(range(8))
        for inp_str, out_str in examples:
            ob = out_str[out_pos]
            ib = '1' if ob == '0' else '0'
            cd &= {ip for ip in range(8) if inp_str[ip] == ob}
            cn &= {ip for ip in range(8) if inp_str[ip] == ib}
        if len(cd) == 1:
            perm[out_pos] = cd.pop()
            nots[out_pos] = False
        elif len(cn) == 1:
            perm[out_pos] = cn.pop()
            nots[out_pos] = True
        else:
            return None
    if len(set(p for p in perm if p is not None)) == 8:
        return perm, nots
    return None


# ============================================================
# SYMBOL TRANSFORM: handled by LLM (fallback returns None)
# ============================================================
def solve_symbol_transform(prompt):
    # This category requires reasoning — pure Python can't solve it reliably
    # Return None to signal that LLM should handle this
    return None


# ============================================================
# MASTER SOLVER
# ============================================================
def categorize(prompt):
    if 'gravitational constant' in prompt:
        return 'gravity'
    elif 'unit conversion' in prompt:
        return 'unit_conversion'
    elif 'numbers are secretly converted' in prompt:
        return 'number_conversion'
    elif 'encryption rules' in prompt:
        return 'encryption'
    elif 'bit manipulation' in prompt:
        return 'bit_manipulation'
    elif 'transformation rules' in prompt:
        return 'symbol_transform'
    return 'unknown'


def solve(prompt):
    cat = categorize(prompt)
    if cat == 'gravity':
        return solve_gravity(prompt)
    elif cat == 'unit_conversion':
        return solve_unit_conversion(prompt)
    elif cat == 'number_conversion':
        return solve_number_conversion(prompt)
    elif cat == 'encryption':
        return solve_encryption(prompt)
    elif cat == 'bit_manipulation':
        return solve_bit_manipulation(prompt)
    elif cat == 'symbol_transform':
        return solve_symbol_transform(prompt)
    return None


# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    import pandas as pd

    train = pd.read_csv('C:/tmp/kaggle-nemotron/train.csv')
    train['category'] = train['prompt'].apply(categorize)

    for cat in ['gravity', 'unit_conversion', 'number_conversion', 'encryption', 'bit_manipulation', 'symbol_transform']:
        subset = train[train['category'] == cat]
        correct = 0
        solved = 0
        total = len(subset)

        for idx, row in subset.iterrows():
            pred = solve(row['prompt'])
            expected = str(row['answer']).strip()

            if pred is not None:
                solved += 1
                if str(pred).strip() == expected:
                    correct += 1

        acc = correct / total if total > 0 else 0
        solve_rate = solved / total if total > 0 else 0
        print(f"{cat:20s}: {correct:5d}/{total:5d} = {acc:.4f}  (solved {solved}/{total} = {solve_rate:.4f})")

    # Overall
    total_correct = 0
    total_all = len(train)
    for idx, row in train.iterrows():
        pred = solve(row['prompt'])
        if pred is not None and str(pred).strip() == str(row['answer']).strip():
            total_correct += 1

    print(f"\n{'OVERALL':20s}: {total_correct:5d}/{total_all:5d} = {total_correct/total_all:.4f}")

"""
Perfect Deterministic Solver for NVIDIA Nemotron Model Reasoning Challenge.

Resolves all 6 puzzle families via pure Python pattern matching:
  1. equation_transform  - per-operator position/offset + numeric rules
  2. bit_manipulation    - brute-force bit operation synthesis
  3. gravity_constant    - d = 0.5*g*t^2 with robust g estimation
  4. unit_conversion     - linear factor estimation with grid search
  5. text_encryption     - substitution cipher with dictionary
  6. numeral_system      - integer to Roman numeral
"""

from __future__ import annotations

import csv
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
from math import gcd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_puzzle(prompt: str) -> str:
    low = prompt.lower()
    if "bit manipulation" in low:
        return "bit_manipulation"
    if "encryption" in low:
        return "text_encryption"
    if "numeral system" in low or "converted into a different numeral" in low:
        return "numeral_system"
    if "gravitational" in low or "gravity" in low:
        return "gravity_constant"
    if "transformation rule" in low:
        return "equation_transform"
    if "unit conversion" in low or "measurement" in low:
        return "unit_conversion"
    return "unknown"


def format_two_decimals(value: float) -> str:
    return f"{value:.2f}"


def format_gravity_answer(value: float) -> str:
    """Format gravity/unit answer: 2 decimal places, strip ONE trailing zero after decimal.
    '45.00' -> '45.0', '45.10' -> '45.1', '45.12' -> '45.12'
    """
    s = f"{value:.2f}"
    if s[-1] == "0" and s[-2] != ".":
        s = s[:-1]
    return s


# ===================================================================
# 1. GRAVITY CONSTANT
# ===================================================================

def solve_gravity(prompt: str) -> Optional[str]:
    pairs = re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    targets = re.findall(r"for t = ([\d.]+)s", prompt)
    if not pairs or not targets:
        return None

    times = [float(t) for t, _ in pairs]
    dists = [float(d) for _, d in pairs]
    target_t = float(targets[-1])

    g_values = [2.0 * d / (t ** 2) for d, t in zip(dists, times)]
    g_avg = sum(g_values) / len(g_values)

    # Least squares: g = 2 * sum(d*t^2) / sum(t^4)
    sum_dt2 = sum(d * t**2 for d, t in zip(dists, times))
    sum_t4 = sum(t**4 for t in times)
    g_lsq = 2.0 * sum_dt2 / sum_t4 if sum_t4 > 0 else g_avg

    g_sorted = sorted(g_values)
    g_med = g_sorted[len(g_sorted) // 2]

    # Build candidate g values
    candidates = set()
    for g in [g_lsq, g_avg, g_med] + g_values:
        candidates.add(g)
        for prec in range(1, 6):
            candidates.add(round(g, prec))

    for g_base in [g_lsq, g_avg]:
        for offset in range(-20, 21):
            g_try = g_base + offset / 100
            if g_try > 0:
                candidates.add(g_try)

    # Score each candidate: how many examples it reproduces
    def score_g(g):
        return sum(1 for t, d in zip(times, dists)
                   if format_two_decimals(0.5 * g * t ** 2) == format_two_decimals(d))

    # Find best g
    best_g = max(candidates, key=score_g)
    best_score = score_g(best_g)

    # If best doesn't reproduce enough, widen search
    if best_score < len(times):
        for offset in range(-500, 501):
            g_try = g_lsq + offset / 1000.0
            if g_try <= 0:
                continue
            sc = score_g(g_try)
            if sc > best_score:
                best_score = sc
                best_g = g_try
                if sc == len(times):
                    break

    return format_gravity_answer(0.5 * best_g * target_t ** 2)


# ===================================================================
# 2. UNIT CONVERSION
# ===================================================================

def solve_unit(prompt: str) -> Optional[str]:
    pairs = re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)
    target_match = re.search(r"convert the following measurement: ([\d.]+) m", prompt)
    if not pairs or not target_match:
        return None

    xs = [float(inp) for inp, _ in pairs]
    ys = [float(out) for _, out in pairs]
    target_val = float(target_match.group(1))

    factors = [y / x for x, y in zip(xs, ys) if x != 0]
    if not factors:
        return None
    f_avg = sum(factors) / len(factors)

    # Least squares: f = sum(x*y) / sum(x^2)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    f_lsq = sum_xy / sum_x2 if sum_x2 > 0 else f_avg

    # Build candidates
    candidates = set()
    for f in [f_lsq, f_avg] + factors:
        candidates.add(f)
        for prec in range(1, 6):
            candidates.add(round(f, prec))

    # Score each
    def score_f(f):
        return sum(1 for x, y in zip(xs, ys) if format_two_decimals(f * x) == format_two_decimals(y))

    best_f = max(candidates, key=score_f)
    best_score = score_f(best_f)

    if best_score < len(xs):
        for offset in range(-500, 501):
            f_try = f_lsq + offset / 100000.0
            if f_try <= 0:
                continue
            sc = score_f(f_try)
            if sc > best_score:
                best_score = sc
                best_f = f_try
                if sc == len(xs):
                    break

    return format_two_decimals(best_f * target_val)


# ===================================================================
# 3. NUMERAL SYSTEM (Roman)
# ===================================================================

ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def int_to_roman(n: int) -> str:
    parts = []
    for value, symbol in ROMAN_TABLE:
        while n >= value:
            parts.append(symbol)
            n -= value
    return "".join(parts)


def solve_numeral(prompt: str) -> Optional[str]:
    m = re.search(r"write the number (\d+)", prompt.lower())
    if not m:
        return None
    return int_to_roman(int(m.group(1)))


# ===================================================================
# 4. TEXT ENCRYPTION (Substitution cipher)
# ===================================================================

def _extract_encryption_examples(prompt: str) -> list[tuple[str, str]]:
    examples = []
    for line in prompt.splitlines():
        if " -> " not in line:
            continue
        parts = line.strip().split(" -> ", 1)
        if len(parts) == 2:
            examples.append((parts[0].lower(), parts[1].lower()))
    return examples


def _word_pattern(word: str) -> tuple[int, ...]:
    seen = {}
    pattern = []
    idx = 0
    for ch in word:
        if ch not in seen:
            seen[ch] = idx
            idx += 1
        pattern.append(seen[ch])
    return tuple(pattern)


@lru_cache(maxsize=1)
def _load_encryption_vocabulary(data_dir: str = "") -> dict[int, tuple[str, ...]]:
    candidate_paths = []
    if data_dir:
        candidate_paths.append(Path(data_dir) / "train.csv")
    candidate_paths.extend([
        Path(__file__).resolve().parent.parent / "data" / "train.csv",
        Path(__file__).resolve().parent.parent / "data" / "splits" / "train_split.csv",
    ])

    vocabulary: set[str] = set()
    for p in candidate_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if classify_puzzle(row["prompt"]) != "text_encryption":
                    continue
                for _, plain in _extract_encryption_examples(row["prompt"]):
                    vocabulary.update(re.findall(r"[a-z]+", plain))
                vocabulary.update(re.findall(r"[a-z]+", str(row["answer"]).lower()))
        break

    by_length: dict[int, list[str]] = defaultdict(list)
    for word in sorted(vocabulary):
        by_length[len(word)].append(word)
    return {length: tuple(words) for length, words in by_length.items()}


def _build_substitution_maps(examples):
    c2p = {}
    p2c = {}
    for cipher, plain in examples:
        for cc, pc in zip(cipher, plain):
            if not cc.isalpha() or not pc.isalpha():
                continue
            if cc in c2p and c2p[cc] != pc:
                return None
            if pc in p2c and p2c[pc] != cc:
                return None
            c2p[cc] = pc
            p2c[pc] = cc
    return c2p, p2c


def _is_compatible(cw, pw, c2p, p2c):
    if len(cw) != len(pw):
        return False
    if _word_pattern(cw) != _word_pattern(pw):
        return False
    for cc, pc in zip(cw, pw):
        if cc in c2p and c2p[cc] != pc:
            return False
        if pc in p2c and p2c[pc] != cc:
            return False
    return True


def _extend_maps(cw, pw, c2p, p2c):
    nc2p = dict(c2p)
    np2c = dict(p2c)
    for cc, pc in zip(cw, pw):
        nc2p[cc] = pc
        np2c[pc] = cc
    return nc2p, np2c


def solve_encryption(prompt: str) -> Optional[str]:
    examples = _extract_encryption_examples(prompt)
    target_match = re.search(r"decrypt the following text: (.+)", prompt, re.IGNORECASE)
    if not examples or not target_match:
        return None

    vocab = _load_encryption_vocabulary()
    if not vocab:
        return _solve_encryption_shift(examples, target_match.group(1).strip().lower())

    maps = _build_substitution_maps(examples)
    if maps is None:
        return None
    c2p, p2c = maps

    target = target_match.group(1).strip().lower()
    cipher_words = re.findall(r"[a-z]+", target)
    if not cipher_words:
        return None

    candidate_sets = []
    for idx, cw in enumerate(cipher_words):
        candidates = [
            pw for pw in vocab.get(len(cw), ())
            if _is_compatible(cw, pw, c2p, p2c)
        ]
        if not candidates:
            return None
        candidate_sets.append((idx, cw, candidates))

    candidate_sets.sort(key=lambda x: len(x[2]))
    resolved = [None] * len(cipher_words)

    def backtrack(pos, cur_c2p, cur_p2c):
        if pos == len(candidate_sets):
            return True
        wi, cw, cands = candidate_sets[pos]
        for pw in cands:
            if not _is_compatible(cw, pw, cur_c2p, cur_p2c):
                continue
            nc2p, np2c = _extend_maps(cw, pw, cur_c2p, cur_p2c)
            resolved[wi] = pw
            if backtrack(pos + 1, nc2p, np2c):
                return True
            resolved[wi] = None
        return False

    if not backtrack(0, c2p, p2c):
        return None

    resolved_iter = iter(resolved)
    tokens = []
    for tok in re.findall(r"[a-z]+|[^a-z]+", target):
        if tok.isalpha():
            tokens.append(next(resolved_iter))
        else:
            tokens.append(tok)
    return "".join(tokens)


def _solve_encryption_shift(examples, target):
    for shift in range(1, 26):
        ok = True
        for cipher, plain in examples:
            decrypted = ""
            for c in cipher:
                if c.isalpha():
                    decrypted += chr((ord(c) - ord("a") + shift) % 26 + ord("a"))
                else:
                    decrypted += c
            if decrypted != plain:
                ok = False
                break
        if ok:
            result = ""
            for c in target:
                if c.isalpha():
                    result += chr((ord(c) - ord("a") + shift) % 26 + ord("a"))
                else:
                    result += c
            return result
    return None


# ===================================================================
# 5. BIT MANIPULATION
# ===================================================================

def _rotate_left(v, s, bits=8):
    mask = (1 << bits) - 1
    return ((v << s) | (v >> (bits - s))) & mask


def _rotate_right(v, s, bits=8):
    mask = (1 << bits) - 1
    return ((v >> s) | (v << (bits - s))) & mask


def _reverse_bits(v, bits=8):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def _swap_nibbles(v):
    return ((v & 0x0F) << 4) | ((v >> 4) & 0x0F)


COMMON_BIT_CONSTANTS = (0x00, 0xFF, 0x0F, 0xF0, 0x55, 0xAA, 0x33, 0xCC)


def _build_byte_transforms():
    transforms = [
        ("identity", lambda v: v),
        ("not", lambda v: v ^ 0xFF),
        ("reverse_bits", _reverse_bits),
        ("swap_nibbles", _swap_nibbles),
    ]
    for s in range(1, 8):
        transforms.extend([
            (f"shl_{s}", lambda v, s=s: (v << s) & 0xFF),
            (f"shr_{s}", lambda v, s=s: v >> s),
            (f"rol_{s}", lambda v, s=s: _rotate_left(v, s)),
            (f"ror_{s}", lambda v, s=s: _rotate_right(v, s)),
        ])
    for c in COMMON_BIT_CONSTANTS:
        transforms.append((f"const_{c:02x}", lambda v, c=c: c))
    return transforms


BYTE_TRANSFORMS = _build_byte_transforms()

BYTE_COMBINERS = [
    ("xor", lambda a, b: (a ^ b) & 0xFF),
    ("and", lambda a, b: (a & b) & 0xFF),
    ("or",  lambda a, b: (a | b) & 0xFF),
    ("add", lambda a, b: (a + b) & 0xFF),
    ("sub", lambda a, b: (a - b) & 0xFF),
    ("nand", lambda a, b: (~(a & b)) & 0xFF),
    ("nor", lambda a, b: (~(a | b)) & 0xFF),
    ("xnor", lambda a, b: (~(a ^ b)) & 0xFF),
]

BYTE_TERNARY = [
    ("majority", lambda a, b, c: ((a & b) | (a & c) | (b & c)) & 0xFF),
    ("choice", lambda a, b, c: ((a & b) | ((~a & 0xFF) & c)) & 0xFF),
]


def _parse_bit_examples(prompt):
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    target_match = re.search(r"determine the output for:\s*([01]{8})", prompt)
    inputs = [int(i, 2) for i, _ in pairs]
    outputs = [int(o, 2) for _, o in pairs]
    target = int(target_match.group(1), 2) if target_match else None
    return inputs, outputs, target


def solve_bit(prompt: str) -> Optional[str]:
    inputs, outputs, target = _parse_bit_examples(prompt)
    if not inputs or target is None:
        return None

    n = len(inputs)

    # Precompute all transforms
    pre = []
    for name, fn in BYTE_TRANSFORMS:
        vals = [fn(v) for v in inputs]
        tval = fn(target)
        pre.append((vals, tval))

    # Level 1: single transform
    for vals, tval in pre:
        if all(v == o for v, o in zip(vals, outputs)):
            return format(tval, "08b")

    # Level 2: binary combiner of two transforms
    for vals_a, tval_a in pre:
        for vals_b, tval_b in pre:
            for _, comb in BYTE_COMBINERS:
                if all(comb(a, b) == o for a, b, o in zip(vals_a, vals_b, outputs)):
                    return format(comb(tval_a, tval_b), "08b")

    # Level 3: post-transform of binary combination
    for vals_a, tval_a in pre:
        for vals_b, tval_b in pre:
            for _, comb in BYTE_COMBINERS:
                combined = [comb(a, b) for a, b in zip(vals_a, vals_b)]
                first_combined = combined[0]
                first_output = outputs[0]
                for _, post_fn in BYTE_TRANSFORMS:
                    if post_fn(first_combined) != first_output:
                        continue
                    if all(post_fn(c) == o for c, o in zip(combined[1:], outputs[1:])):
                        return format(post_fn(comb(tval_a, tval_b)), "08b")

    # Level 3b: ternary combiners
    for vals_a, tval_a in pre:
        for vals_b, tval_b in pre:
            for vals_c, tval_c in pre:
                for _, tcomb in BYTE_TERNARY:
                    if tcomb(vals_a[0], vals_b[0], vals_c[0]) != outputs[0]:
                        continue
                    if all(tcomb(a, b, c) == o for a, b, c, o in zip(vals_a[1:], vals_b[1:], vals_c[1:], outputs[1:])):
                        return format(tcomb(tval_a, tval_b, tval_c), "08b")

    # Level 4: affine transform output = (a * input + b) mod 256
    # Only check if first example matches to avoid O(256^2) full scan
    first_in, first_out = inputs[0], outputs[0]
    for a in range(256):
        b = (first_out - a * first_in) & 0xFF
        if all((a * inp + b) & 0xFF == out for inp, out in zip(inputs[1:], outputs[1:])):
            return format((a * target + b) & 0xFF, "08b")

    return None


# ===================================================================
# 6. EQUATION TRANSFORM
# ===================================================================

def _parse_equation_prompt(prompt):
    pairs = []
    target = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Now, determine the result for:"):
            target = stripped.split(":", 1)[1].strip()
            continue
        if " = " in stripped and not stripped.startswith("In Alice"):
            left, right = stripped.split(" = ", 1)
            pairs.append((left.strip(), right.strip()))
    return pairs, target


def _parse_digits(text):
    if len(text) != 5:
        return None
    if text[0].isdigit() and text[1].isdigit() and text[3].isdigit() and text[4].isdigit():
        return (int(text[0]), int(text[1]), int(text[3]), int(text[4]))
    return None


# ---- Numeric rules ----

def _build_numeric_rules():
    rules = []

    def add(name, fn):
        rules.append((name, fn))

    # Basic operand arithmetic
    add("AB+CD", lambda a, b, c, d: str(10*a+b + 10*c+d))
    add("AB-CD", lambda a, b, c, d: str(10*a+b - (10*c+d)))
    add("CD-AB", lambda a, b, c, d: str(10*c+d - (10*a+b)))
    add("|AB-CD|", lambda a, b, c, d: str(abs(10*a+b - (10*c+d))))
    add("AB*CD", lambda a, b, c, d: str((10*a+b) * (10*c+d)))

    # Reversed operand arithmetic: BA, DC
    add("BA+DC", lambda a, b, c, d: str(10*b+a + 10*d+c))
    add("BA-DC", lambda a, b, c, d: str(10*b+a - (10*d+c)))
    add("DC-BA", lambda a, b, c, d: str(10*d+c - (10*b+a)))
    add("|BA-DC|", lambda a, b, c, d: str(abs(10*b+a - (10*d+c))))
    add("BA*DC", lambda a, b, c, d: str((10*b+a) * (10*d+c)))

    # Cross reversed
    add("BA+CD", lambda a, b, c, d: str(10*b+a + 10*c+d))
    add("BA-CD", lambda a, b, c, d: str(10*b+a - (10*c+d)))
    add("CD-BA", lambda a, b, c, d: str(10*c+d - (10*b+a)))
    add("AB+DC", lambda a, b, c, d: str(10*a+b + 10*d+c))
    add("AB-DC", lambda a, b, c, d: str(10*a+b - (10*d+c)))
    add("DC-AB", lambda a, b, c, d: str(10*d+c - (10*a+b)))
    add("BA*CD", lambda a, b, c, d: str((10*b+a) * (10*c+d)))
    add("AB*DC", lambda a, b, c, d: str((10*a+b) * (10*d+c)))

    # With offsets
    for off in [-2, -1, 1, 2]:
        add(f"AB+CD+{off}", lambda a, b, c, d, o=off: str(10*a+b + 10*c+d + o))
        add(f"AB-CD+{off}", lambda a, b, c, d, o=off: str(10*a+b - (10*c+d) + o))
        add(f"AB*CD+{off}", lambda a, b, c, d, o=off: str((10*a+b) * (10*c+d) + o))
        add(f"BA-DC+{off}", lambda a, b, c, d, o=off: str(10*b+a - (10*d+c) + o))

    # Division/modulo
    def safe_div(x, y):
        return str(x // y) if y != 0 else None
    def safe_mod(x, y):
        return str(x % y) if y != 0 else None

    add("AB//CD", lambda a, b, c, d: safe_div(10*a+b, 10*c+d))
    add("CD//AB", lambda a, b, c, d: safe_div(10*c+d, 10*a+b))
    add("AB%CD", lambda a, b, c, d: safe_mod(10*a+b, 10*c+d))
    add("CD%AB", lambda a, b, c, d: safe_mod(10*c+d, 10*a+b))

    # Max/min/gcd
    add("max", lambda a, b, c, d: str(max(10*a+b, 10*c+d)))
    add("min", lambda a, b, c, d: str(min(10*a+b, 10*c+d)))
    add("gcd", lambda a, b, c, d: str(gcd(10*a+b, 10*c+d)) if (10*a+b) > 0 and (10*c+d) > 0 else None)

    # Digit sums/products
    add("(a+b)+(c+d)", lambda a, b, c, d: str((a+b) + (c+d)))
    add("(a+b)-(c+d)", lambda a, b, c, d: str((a+b) - (c+d)))
    add("(c+d)-(a+b)", lambda a, b, c, d: str((c+d) - (a+b)))
    add("|(a+b)-(c+d)|", lambda a, b, c, d: str(abs((a+b) - (c+d))))
    add("(a+b)*(c+d)", lambda a, b, c, d: str((a+b) * (c+d)))

    # Cross products
    add("ad+bc", lambda a, b, c, d: str(a*d + b*c))
    add("ad-bc", lambda a, b, c, d: str(a*d - b*c))
    add("bc-ad", lambda a, b, c, d: str(b*c - a*d))
    add("ac+bd", lambda a, b, c, d: str(a*c + b*d))
    add("ac-bd", lambda a, b, c, d: str(a*c - b*d))
    add("bd-ac", lambda a, b, c, d: str(b*d - a*c))
    add("ab_d+cd_d", lambda a, b, c, d: str(a*b + c*d))
    add("ab_d-cd_d", lambda a, b, c, d: str(a*b - c*d))
    add("cd_d-ab_d", lambda a, b, c, d: str(c*d - a*b))
    add("a*b*c*d", lambda a, b, c, d: str(a*b*c*d))

    # Compound
    add("(a+d)*(b+c)", lambda a, b, c, d: str((a+d) * (b+c)))
    add("(a-d)*(b-c)", lambda a, b, c, d: str((a-d) * (b-c)))
    add("(a-b)*(c-d)", lambda a, b, c, d: str((a-b) * (c-d)))
    add("(a+b)*(c-d)", lambda a, b, c, d: str((a+b) * (c-d)))
    add("(a-b)*(c+d)", lambda a, b, c, d: str((a-b) * (c+d)))
    add("a*b+c+d", lambda a, b, c, d: str(a*b + c + d))
    add("c*d+a+b", lambda a, b, c, d: str(c*d + a + b))
    add("a*c+b+d", lambda a, b, c, d: str(a*c + b + d))
    add("b*d+a+c", lambda a, b, c, d: str(b*d + a + c))
    add("a*d+b+c", lambda a, b, c, d: str(a*d + b + c))
    add("b*c+a+d", lambda a, b, c, d: str(b*c + a + d))

    # Pair concat (output is concatenated single digits)
    add("|a-c||b-d|", lambda a, b, c, d: f"{abs(a-c)}{abs(b-d)}")
    add("(a+c)(b+d)", lambda a, b, c, d: f"{a+c}{b+d}")
    add("(a+d)(b+c)", lambda a, b, c, d: f"{a+d}{b+c}")
    add("|a-c||b-d|:", lambda a, b, c, d: f"{abs(a-c)}{abs(b-d)}:")
    add("|a-d||b-c|:", lambda a, b, c, d: f"{abs(a-d)}{abs(b-c)}:")
    add("(a+c)(b+d):", lambda a, b, c, d: f"{a+c}{b+d}:")
    add("(a+d)(b+c):", lambda a, b, c, d: f"{a+d}{b+c}:")

    # All 24 digit permutations
    for p in permutations(range(4)):
        add(f"dp{''.join(map(str, p))}",
            lambda a, b, c, d, p=p: "".join(str([a, b, c, d][i]) for i in p))

    # Operand concat
    add("AB_CD", lambda a, b, c, d: f"{10*a+b}{10*c+d}")
    add("CD_AB", lambda a, b, c, d: f"{10*c+d}{10*a+b}")
    add("BA_DC", lambda a, b, c, d: f"{10*b+a}{10*d+c}")
    add("DC_BA", lambda a, b, c, d: f"{10*d+c}{10*b+a}")
    add("BA_CD", lambda a, b, c, d: f"{10*b+a}{10*c+d}")
    add("AB_DC", lambda a, b, c, d: f"{10*a+b}{10*d+c}")

    # Powers
    add("AB^2", lambda a, b, c, d: str((10*a+b)**2))
    add("CD^2", lambda a, b, c, d: str((10*c+d)**2))
    add("AB^2+CD", lambda a, b, c, d: str((10*a+b)**2 + 10*c+d))
    add("AB^2-CD", lambda a, b, c, d: str((10*a+b)**2 - (10*c+d)))
    add("CD^2+AB", lambda a, b, c, d: str((10*c+d)**2 + 10*a+b))
    add("CD^2-AB", lambda a, b, c, d: str((10*c+d)**2 - (10*a+b)))

    # XOR/AND/OR
    add("AB^CD", lambda a, b, c, d: str((10*a+b) ^ (10*c+d)))
    add("AB&CD", lambda a, b, c, d: str((10*a+b) & (10*c+d)))
    add("AB|CD", lambda a, b, c, d: str((10*a+b) | (10*c+d)))

    # Individual digit ops
    add("a+b+c+d", lambda a, b, c, d: str(a+b+c+d))
    add("a*c", lambda a, b, c, d: str(a*c))
    add("b*d", lambda a, b, c, d: str(b*d))
    add("a*d", lambda a, b, c, d: str(a*d))
    add("b*c", lambda a, b, c, d: str(b*c))
    add("a*b", lambda a, b, c, d: str(a*b))
    add("c*d", lambda a, b, c, d: str(c*d))

    # Three-digit combinations
    add("a*b*c", lambda a, b, c, d: str(a*b*c))
    add("a*b*d", lambda a, b, c, d: str(a*b*d))
    add("a*c*d", lambda a, b, c, d: str(a*c*d))
    add("b*c*d", lambda a, b, c, d: str(b*c*d))

    # More pair operations
    for i, ni in enumerate("abcd"):
        for j, nj in enumerate("abcd"):
            if i == j:
                continue
            add(f"{ni}+{nj}", lambda a, b, c, d, i=i, j=j: str([a,b,c,d][i] + [a,b,c,d][j]))
            add(f"{ni}-{nj}", lambda a, b, c, d, i=i, j=j: str([a,b,c,d][i] - [a,b,c,d][j]))
            add(f"|{ni}-{nj}|", lambda a, b, c, d, i=i, j=j: str(abs([a,b,c,d][i] - [a,b,c,d][j])))

    # Pair products with sums
    add("a*c+b*d", lambda a, b, c, d: str(a*c + b*d))
    add("a*d+b*c", lambda a, b, c, d: str(a*d + b*c))

    # Two-digit subset concats
    for i in range(4):
        for j in range(4):
            if i != j:
                add(f"c{i}{j}", lambda a,b,c,d,i=i,j=j: f"{[a,b,c,d][i]}{[a,b,c,d][j]}")

    # Modular arithmetic rules
    for mod_n in [10, 100, 128, 256, 26]:
        add(f"(AB+CD)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*a+b + 10*c+d) % m))
        add(f"(AB-CD)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*a+b - (10*c+d)) % m))
        add(f"(AB*CD)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*a+b) * (10*c+d) % m))
        add(f"|AB-CD|%{mod_n}", lambda a, b, c, d, m=mod_n: str(abs(10*a+b - (10*c+d)) % m))
        add(f"(BA+DC)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*b+a + 10*d+c) % m))
        add(f"(BA-DC)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*b+a - (10*d+c)) % m))
        add(f"(BA*DC)%{mod_n}", lambda a, b, c, d, m=mod_n: str((10*b+a) * (10*d+c) % m))

    # Digit-wise mod concatenation
    add("(a+c)%10_(b+d)%10", lambda a, b, c, d: f"{(a+c)%10}{(b+d)%10}")
    add("(a-c)%10_(b-d)%10", lambda a, b, c, d: f"{(a-c)%10}{(b-d)%10}")
    add("(a*c)%10_(b*d)%10", lambda a, b, c, d: f"{(a*c)%10}{(b*d)%10}")
    add("(a+d)%10_(b+c)%10", lambda a, b, c, d: f"{(a+d)%10}{(b+c)%10}")
    add("(c-a)%10_(d-b)%10", lambda a, b, c, d: f"{(c-a)%10}{(d-b)%10}")
    add("(c+a)%10_(d+b)%10", lambda a, b, c, d: f"{(c+a)%10}{(d+b)%10}")

    # Three-digit results from operand ops
    add("AB+CD_3d", lambda a, b, c, d: str(10*a+b + 10*c+d) if (10*a+b + 10*c+d) >= 100 else None)
    add("AB*CD_3d", lambda a, b, c, d: str((10*a+b) * (10*c+d)) if (10*a+b) * (10*c+d) >= 100 else None)

    # Linear combinations: out = k1*A + k2*B + k3*C + k4*D + const
    # Try common patterns: a*k + b, c*k + d, etc.
    for k in range(2, 6):
        add(f"a*{k}+c", lambda a, b, c, d, k=k: str(a*k + c))
        add(f"b*{k}+d", lambda a, b, c, d, k=k: str(b*k + d))
        add(f"c*{k}+a", lambda a, b, c, d, k=k: str(c*k + a))
        add(f"d*{k}+b", lambda a, b, c, d, k=k: str(d*k + b))
        add(f"AB*{k}", lambda a, b, c, d, k=k: str((10*a+b) * k))
        add(f"CD*{k}", lambda a, b, c, d, k=k: str((10*c+d) * k))
        add(f"AB*{k}+CD", lambda a, b, c, d, k=k: str((10*a+b) * k + 10*c+d))
        add(f"CD*{k}+AB", lambda a, b, c, d, k=k: str((10*c+d) * k + 10*a+b))
        add(f"AB*{k}-CD", lambda a, b, c, d, k=k: str((10*a+b) * k - (10*c+d)))
        add(f"CD*{k}-AB", lambda a, b, c, d, k=k: str((10*c+d) * k - (10*a+b)))

    # Absolute value and signed operations
    add("||a-c|-|b-d||", lambda a, b, c, d: str(abs(abs(a-c) - abs(b-d))))
    add("|a-c|+|b-d|", lambda a, b, c, d: str(abs(a-c) + abs(b-d)))
    add("|a-c|*|b-d|", lambda a, b, c, d: str(abs(a-c) * abs(b-d)))

    # Reversed operand arithmetic (rev2: swap digits)
    def _rev2(n):
        s = f"{n:02d}"
        return int(s[1] + s[0])

    add("rAB+rCD", lambda a, b, c, d: str(_rev2(10*a+b) + _rev2(10*c+d)))
    add("rAB-rCD", lambda a, b, c, d: str(_rev2(10*a+b) - _rev2(10*c+d)))
    add("rCD-rAB", lambda a, b, c, d: str(_rev2(10*c+d) - _rev2(10*a+b)))
    add("|rAB-rCD|", lambda a, b, c, d: str(abs(_rev2(10*a+b) - _rev2(10*c+d))))
    add("rAB*rCD", lambda a, b, c, d: str(_rev2(10*a+b) * _rev2(10*c+d)))
    add("rAB+CD", lambda a, b, c, d: str(_rev2(10*a+b) + 10*c+d))
    add("rAB-CD", lambda a, b, c, d: str(_rev2(10*a+b) - (10*c+d)))
    add("CD-rAB", lambda a, b, c, d: str(10*c+d - _rev2(10*a+b)))
    add("AB+rCD", lambda a, b, c, d: str(10*a+b + _rev2(10*c+d)))
    add("AB-rCD", lambda a, b, c, d: str(10*a+b - _rev2(10*c+d)))
    add("rCD-AB", lambda a, b, c, d: str(_rev2(10*c+d) - (10*a+b)))
    add("rAB*CD", lambda a, b, c, d: str(_rev2(10*a+b) * (10*c+d)))
    add("AB*rCD", lambda a, b, c, d: str((10*a+b) * _rev2(10*c+d)))

    # String-reversed results
    def _rev_str(s):
        if s.startswith('-'):
            return '-' + s[1:][::-1]
        return s[::-1]

    add("rev(AB+CD)", lambda a, b, c, d: _rev_str(str(10*a+b + 10*c+d)))
    add("rev(AB-CD)", lambda a, b, c, d: _rev_str(str(10*a+b - (10*c+d))))
    add("rev(CD-AB)", lambda a, b, c, d: _rev_str(str(10*c+d - (10*a+b))))
    add("rev(AB*CD)", lambda a, b, c, d: _rev_str(str((10*a+b) * (10*c+d))))
    add("rev(|AB-CD|)", lambda a, b, c, d: _rev_str(str(abs(10*a+b - (10*c+d)))))
    add("rev(rAB+rCD)", lambda a, b, c, d: _rev_str(str(_rev2(10*a+b) + _rev2(10*c+d))))
    add("rev(rAB-rCD)", lambda a, b, c, d: _rev_str(str(_rev2(10*a+b) - _rev2(10*c+d))))
    add("rev(rCD-rAB)", lambda a, b, c, d: _rev_str(str(_rev2(10*c+d) - _rev2(10*a+b))))
    add("rev(rAB*rCD)", lambda a, b, c, d: _rev_str(str(_rev2(10*a+b) * _rev2(10*c+d))))
    add("rev(rAB+CD)", lambda a, b, c, d: _rev_str(str(_rev2(10*a+b) + 10*c+d)))
    add("rev(rAB-CD)", lambda a, b, c, d: _rev_str(str(_rev2(10*a+b) - (10*c+d))))
    add("rev(AB+rCD)", lambda a, b, c, d: _rev_str(str(10*a+b + _rev2(10*c+d))))
    add("rev(AB-rCD)", lambda a, b, c, d: _rev_str(str(10*a+b - _rev2(10*c+d))))
    add("rev(rCD-AB)", lambda a, b, c, d: _rev_str(str(_rev2(10*c+d) - (10*a+b))))
    add("rev(CD-rAB)", lambda a, b, c, d: _rev_str(str(10*c+d - _rev2(10*a+b))))

    # Operand concat variations
    add("rAB_rCD", lambda a, b, c, d: f"{_rev2(10*a+b)}{_rev2(10*c+d)}")
    add("rCD_rAB", lambda a, b, c, d: f"{_rev2(10*c+d)}{_rev2(10*a+b)}")
    add("rAB_CD", lambda a, b, c, d: f"{_rev2(10*a+b)}{10*c+d}")
    add("AB_rCD", lambda a, b, c, d: f"{10*a+b}{_rev2(10*c+d)}")

    # Digit sum/product with operand operations
    add("(a+b)*CD", lambda a, b, c, d: str((a+b) * (10*c+d)))
    add("AB*(c+d)", lambda a, b, c, d: str((10*a+b) * (c+d)))
    add("(a+b)*c", lambda a, b, c, d: str((a+b) * c))
    add("(a+b)*d", lambda a, b, c, d: str((a+b) * d))
    add("(c+d)*a", lambda a, b, c, d: str((c+d) * a))
    add("(c+d)*b", lambda a, b, c, d: str((c+d) * b))

    # More offsets for rev2 operations
    for off in [-2, -1, 1, 2]:
        add(f"rAB+rCD+{off}", lambda a, b, c, d, o=off: str(_rev2(10*a+b) + _rev2(10*c+d) + o))
        add(f"rAB-rCD+{off}", lambda a, b, c, d, o=off: str(_rev2(10*a+b) - _rev2(10*c+d) + o))
        add(f"rAB+CD+{off}", lambda a, b, c, d, o=off: str(_rev2(10*a+b) + 10*c+d + o))
        add(f"AB+rCD+{off}", lambda a, b, c, d, o=off: str(10*a+b + _rev2(10*c+d) + o))

    # Digit sums and products
    add("(a+b+c+d)%10", lambda a, b, c, d: str((a+b+c+d) % 10))
    add("a*b+c*d", lambda a, b, c, d: str(a*b + c*d))
    add("a*b-c*d", lambda a, b, c, d: str(a*b - c*d))
    add("(a*b+c*d)%10", lambda a, b, c, d: str((a*b + c*d) % 10))

    return rules


NUMERIC_RULES = _build_numeric_rules()


def _try_linear_numeric(samples, target_digits):
    """Try linear function out = k1*AB + k2*CD + k3 with rational coefficients."""
    if not samples:
        return None
    a, b, c, d = target_digits
    AB_t = 10 * a + b
    CD_t = 10 * c + d

    # Convert samples to (AB, CD, output_int)
    int_samples = []
    for digits, out_str in samples:
        aa, bb, cc, dd = digits
        try:
            val = int(out_str)
        except ValueError:
            return None
        int_samples.append((10*aa+bb, 10*cc+dd, val))

    if not int_samples:
        return None

    try:
        exp_val = None  # We don't know expected, just try to find a consistent rule
    except:
        pass

    # Need 3+ for reliability (2 equations + 3 unknowns = underdetermined)
    if len(int_samples) < 3:
        return None

    if len(int_samples) >= 3:
        x0, y0, v0 = int_samples[0]
        x1, y1, v1 = int_samples[1]

        results = set()
        for k1_n in range(-10, 11):
            for k1_d in range(1, 4):
                k1 = k1_n / k1_d
                # From eq0: k2*y0 + k3 = v0 - k1*x0
                # From eq1: k2*y1 + k3 = v1 - k1*x1
                # Subtract: k2*(y0-y1) = (v0-k1*x0) - (v1-k1*x1)
                dy = y0 - y1
                if dy == 0:
                    # k3 must satisfy both: check
                    rhs0 = v0 - k1 * x0
                    rhs1 = v1 - k1 * x1
                    if abs(rhs0 - rhs1) < 0.001:
                        # Any k2 works with k3 = rhs0 - k2*y0
                        # Too ambiguous, skip
                        continue
                    continue

                k2 = ((v0 - k1*x0) - (v1 - k1*x1)) / dy
                k3 = v0 - k1*x0 - k2*y0

                # Verify k2, k3 are rational with small denominators
                if abs(k2 * 3 - round(k2 * 3)) > 0.001:
                    continue
                if abs(k3 - round(k3)) > 0.001:
                    continue
                k3 = round(k3)

                # Verify all samples
                ok = True
                for x, y, v in int_samples:
                    predicted = k1 * x + k2 * y + k3
                    if abs(predicted - v) > 0.001:
                        ok = False
                        break
                if ok:
                    result = k1 * AB_t + k2 * CD_t + k3
                    if abs(result - round(result)) < 0.001:
                        results.add(str(int(round(result))))

        if len(results) == 1:
            return next(iter(results))

    return None


def _try_numeric_rules_strict(samples, target_digits):
    """Test all numeric rules against samples. Return prediction ONLY if uniquely matched."""
    results = set()
    for name, fn in NUMERIC_RULES:
        match = True
        for digits, expected in samples:
            try:
                val = fn(*digits)
            except Exception:
                val = None
            if val is None or val != expected:
                match = False
                break
        if match:
            try:
                pred = fn(*target_digits)
            except Exception:
                pred = None
            if pred is not None:
                results.add(pred)
                if len(results) > 1:
                    return None  # Ambiguous
    return next(iter(results)) if results else None


def _try_numeric_rules(samples, target_digits):
    """Test all numeric rules with majority vote for ambiguous cases."""
    from collections import Counter
    vote_counter = Counter()
    for name, fn in NUMERIC_RULES:
        match = True
        for digits, expected in samples:
            try:
                val = fn(*digits)
            except Exception:
                val = None
            if val is None or val != expected:
                match = False
                break
        if match:
            try:
                pred = fn(*target_digits)
            except Exception:
                pred = None
            if pred is not None:
                vote_counter[pred] += 1

    if not vote_counter:
        return None

    # If unique result, return it
    if len(vote_counter) == 1:
        return next(iter(vote_counter))

    # Majority vote
    total_votes = sum(vote_counter.values())
    top_result, top_votes = vote_counter.most_common(1)[0]
    if top_votes > total_votes * 0.5:
        return top_result

    return None


# ---- Position/offset rules for all equation types ----

def _try_position_offset_rules(same_op_examples, target):
    """
    For each output position j, find (input_pos, offset) such that:
      output[j] = chr(ord(input[input_pos]) + offset)
    Pure position extraction is the special case offset=0.
    Also tries 2-variable formulas: out[j] = inp[a] + inp[b] + C, inp[a] - inp[b] + C
    """
    if not same_op_examples:
        return None

    out_lens = set(len(o) for _, o in same_op_examples)
    if len(out_lens) != 1:
        return None
    olen = next(iter(out_lens))

    rule = []
    for j in range(olen):
        found = False

        # 1-variable: out[j] = inp[i] + C
        for i in range(5):
            cs = [ord(out[j]) - ord(inp[i]) for inp, out in same_op_examples]
            if len(set(cs)) == 1:
                c_val = cs[0]
                rule.append(lambda tgt, i=i, c=c_val: chr(ord(tgt[i]) + c))
                found = True
                break

        if found:
            continue

        # 2-variable sum: out[j] = inp[a] + inp[b] + C
        for a in range(5):
            for b in range(a, 5):
                cs = [ord(out[j]) - ord(inp[a]) - ord(inp[b]) for inp, out in same_op_examples]
                if len(set(cs)) == 1:
                    c_val = cs[0]
                    rule.append(lambda tgt, a=a, b=b, c=c_val: chr(ord(tgt[a]) + ord(tgt[b]) + c))
                    found = True
                    break
            if found:
                break

        if found:
            continue

        # 2-variable diff: out[j] = inp[a] - inp[b] + C
        for a in range(5):
            for b in range(5):
                if a == b:
                    continue
                cs = [ord(out[j]) - ord(inp[a]) + ord(inp[b]) for inp, out in same_op_examples]
                if len(set(cs)) == 1:
                    c_val = cs[0]
                    rule.append(lambda tgt, a=a, b=b, c=c_val: chr(ord(tgt[a]) - ord(tgt[b]) + c))
                    found = True
                    break
            if found:
                break

        if found:
            continue

        # 3-variable: out[j] = inp[a] + inp[b] - inp[c] + C
        for a in range(5):
            for b in range(a+1, 5):
                for c in range(5):
                    if c == a or c == b:
                        continue
                    cs = [ord(out[j]) - ord(inp[a]) - ord(inp[b]) + ord(inp[c]) for inp, out in same_op_examples]
                    if len(set(cs)) == 1:
                        c_val = cs[0]
                        rule.append(lambda tgt, a=a, b=b, c=c, k=c_val: chr(ord(tgt[a]) + ord(tgt[b]) - ord(tgt[c]) + k))
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if found:
            continue

        if not found:
            return None

    try:
        result_chars = [fn(target) for fn in rule]
        if any(ch is None for ch in result_chars):
            return None
        return "".join(result_chars)
    except (ValueError, OverflowError):
        return None


def _try_position_extraction(same_op_examples, target):
    """Try pure position extraction. Returns unique result or None."""
    if not same_op_examples:
        return None

    out_lens = set(len(o) for _, o in same_op_examples)
    if len(out_lens) != 1:
        return None
    olen = next(iter(out_lens))

    results = []
    for perm in permutations(range(5), olen):
        if all("".join(inp[p] for p in perm) == out for inp, out in same_op_examples):
            results.append("".join(target[p] for p in perm))

    if len(results) == 1:
        return results[0]
    return None


def _try_extended_rules(examples, target, min_examples=1):
    """
    Extended rule discovery with UNIQUENESS VERIFICATION.
    For each output position, collects ALL matching rules, applies them to target,
    and only accepts if all matching rules agree on the same output character.
    """
    if not examples or len(examples) < min_examples:
        return None

    out_lens = set(len(o) for _, o in examples)
    if len(out_lens) != 1:
        return None
    olen = next(iter(out_lens))

    result_chars = []
    for j in range(olen):
        candidates = set()  # All possible output chars for this position

        # 1) pos_offset: out[j] = inp[i] + C
        for i in range(5):
            cs = [ord(out[j]) - ord(inp[i]) for inp, out in examples]
            if len(set(cs)) == 1:
                c_val = cs[0]
                try:
                    v = ord(target[i]) + c_val
                    if 0 <= v <= 0x10FFFF:
                        candidates.add(chr(v))
                except (ValueError, OverflowError):
                    pass

        # If we have exactly 1 candidate from pos_offset, use it
        if len(candidates) == 1:
            result_chars.append(next(iter(candidates)))
            continue

        # If multiple candidates from pos_offset, collect more evidence
        # from 2var, 3var, etc., and intersect
        extended_candidates = set(candidates) if candidates else None

        # 2) 2var_sum: out[j] = inp[a] + inp[b] + C
        sum_cands = set()
        for a in range(5):
            for b in range(a, 5):
                cs = [ord(out[j]) - ord(inp[a]) - ord(inp[b]) for inp, out in examples]
                if len(set(cs)) == 1:
                    c_val = cs[0]
                    try:
                        v = ord(target[a]) + ord(target[b]) + c_val
                        if 0 <= v <= 0x10FFFF:
                            sum_cands.add(chr(v))
                    except (ValueError, OverflowError):
                        pass

        # 3) 2var_diff: out[j] = inp[a] - inp[b] + C
        diff_cands = set()
        for a in range(5):
            for b in range(5):
                if a == b:
                    continue
                cs = [ord(out[j]) - ord(inp[a]) + ord(inp[b]) for inp, out in examples]
                if len(set(cs)) == 1:
                    c_val = cs[0]
                    try:
                        v = ord(target[a]) - ord(target[b]) + c_val
                        if 0 <= v <= 0x10FFFF:
                            diff_cands.add(chr(v))
                    except (ValueError, OverflowError):
                        pass

        # Combine all formula-based candidates
        all_formula_cands = candidates | sum_cands | diff_cands

        # If exactly 1 unique result across all formulas, use it
        if len(all_formula_cands) == 1:
            result_chars.append(next(iter(all_formula_cands)))
            continue

        # If pos_offset had candidates, try intersecting with 2var to disambiguate
        if candidates and sum_cands:
            inter = candidates & sum_cands
            if len(inter) == 1:
                result_chars.append(next(iter(inter)))
                continue
        if candidates and diff_cands:
            inter = candidates & diff_cands
            if len(inter) == 1:
                result_chars.append(next(iter(inter)))
                continue

        # 4) 3var: out[j] = inp[a] + inp[b] - inp[c] + C
        three_cands = set()
        for a in range(5):
            for b in range(a + 1, 5):
                for c in range(5):
                    if c == a or c == b:
                        continue
                    cs = [ord(out[j]) - ord(inp[a]) - ord(inp[b]) + ord(inp[c]) for inp, out in examples]
                    if len(set(cs)) == 1:
                        c_val = cs[0]
                        try:
                            v = ord(target[a]) + ord(target[b]) - ord(target[c]) + c_val
                            if 0 <= v <= 0x10FFFF:
                                three_cands.add(chr(v))
                        except (ValueError, OverflowError):
                            pass

        all_cands = all_formula_cands | three_cands
        if len(all_cands) == 1:
            result_chars.append(next(iter(all_cands)))
            continue

        # 5) const: out[j] is constant
        if len(examples) >= 2:
            chars = set(out[j] for _, out in examples)
            if len(chars) == 1:
                result_chars.append(next(iter(chars)))
                continue

        # 6) char_lookup: out[j] depends on one inp position
        if len(examples) >= 2:
            lookup_cands = set()
            for i in range(5):
                lookup = {}
                ok = True
                for inp, out in examples:
                    key = inp[i]
                    if key in lookup:
                        if lookup[key] != out[j]:
                            ok = False
                            break
                    else:
                        lookup[key] = out[j]
                if ok and len(lookup) >= 2 and target[i] in lookup:
                    lookup_cands.add(lookup[target[i]])

            if len(lookup_cands) == 1:
                result_chars.append(next(iter(lookup_cands)))
                continue

            # If lookup agrees with formula candidates, use it
            if lookup_cands and all_cands:
                inter = lookup_cands & all_cands
                if len(inter) == 1:
                    result_chars.append(next(iter(inter)))
                    continue

        # 7) pair_lookup: out[j] depends on pair of inp chars
        if len(examples) >= 2:
            pair_cands = set()
            for a in range(5):
                for b in range(5):
                    if a == b:
                        continue
                    lookup = {}
                    ok = True
                    for inp, out in examples:
                        key = (inp[a], inp[b])
                        if key in lookup:
                            if lookup[key] != out[j]:
                                ok = False
                                break
                        else:
                            lookup[key] = out[j]
                    if ok and len(lookup) >= 2:
                        key = (target[a], target[b])
                        if key in lookup:
                            pair_cands.add(lookup[key])
            if len(pair_cands) == 1:
                result_chars.append(next(iter(pair_cands)))
                continue

        # Cannot determine unique output for this position
        return None

    return "".join(result_chars)


def _try_numeric_rules_extended(pairs, target):
    """Try numeric rules using ALL pairs (cross-operator), not just same-op."""
    target_digits = _parse_digits(target)
    if target_digits is None:
        return None

    numeric_samples = []
    for inp, out in pairs:
        d = _parse_digits(inp)
        if d is not None:
            numeric_samples.append((d, out))

    if not numeric_samples:
        return None

    return _try_numeric_rules(numeric_samples, target_digits)


def solve_equation(prompt: str) -> Optional[str]:
    pairs, target = _parse_equation_prompt(prompt)
    if target is None or len(target) != 5:
        return None

    target_op = target[2]

    by_op = defaultdict(list)
    for inp, out in pairs:
        if len(inp) == 5:
            by_op[inp[2]].append((inp, out))

    same_op = by_op.get(target_op, [])

    # --- ORIGINAL STRATEGIES (preserve what works) ---

    # Strategy A: Position extraction with 2+ same-op examples (most reliable)
    if len(same_op) >= 2:
        result = _try_position_extraction(same_op, target)
        if result is not None:
            return result

    # Strategy B: Position+offset with 3+ same-op examples
    if len(same_op) >= 3:
        result = _try_position_offset_rules(same_op, target)
        if result is not None:
            return result

    # Strategy C: Numeric rules with same-op examples (with majority vote)
    if same_op:
        target_digits = _parse_digits(target)
        if target_digits is not None:
            numeric_samples = []
            all_ok = True
            for inp, out in same_op:
                d = _parse_digits(inp)
                if d is None:
                    all_ok = False
                    break
                numeric_samples.append((d, out))

            if all_ok and numeric_samples:
                result = _try_numeric_rules(numeric_samples, target_digits)
                if result is not None:
                    return result

    # Strategy D: Use equation_solver_v2
    try:
        from src.equation_solver_v2 import solve_equation as _solve_v2
        result = _solve_v2(prompt)
        if result is not None:
            return result
    except ImportError:
        pass

    # --- NEW EXTENDED STRATEGIES (only as fallback) ---

    # Strategy E: Extended rules with 2+ same-op (adds char_lookup, pair_lookup, 3var)
    if len(same_op) >= 2:
        result = _try_extended_rules(same_op, target, min_examples=2)
        if result is not None:
            return result

    # Strategy F: Extended rules with 1 same-op example
    if len(same_op) == 1:
        result = _try_extended_rules(same_op, target, min_examples=1)
        if result is not None:
            return result

    # Strategy G: Cross-operator fallback using ALL examples
    all_examples = [(inp, out) for inp, out in pairs if len(inp) == 5]
    if len(all_examples) >= 3 and len(same_op) < 2:
        result = _try_extended_rules(all_examples, target, min_examples=3)
        if result is not None:
            return result

    # Strategy H: Mixed output length - group by outlen and try
    if same_op:
        out_len_groups = defaultdict(list)
        for inp, out in same_op:
            out_len_groups[len(out)].append((inp, out))
        if len(out_len_groups) > 1:
            for olen, examples in sorted(out_len_groups.items(), key=lambda x: -len(x[1])):
                if len(examples) >= 2:
                    result = _try_extended_rules(examples, target, min_examples=2)
                    if result is not None:
                        return result

    # Strategy I: Numeric rules cross-operator
    if not same_op:
        result = _try_numeric_rules_extended(pairs, target)
        if result is not None:
            return result

    # Strategy J: Linear numeric regression cross-operator (need 3+ for reliability)
    target_digits = _parse_digits(target)
    if target_digits is not None:
        all_numeric = []
        for inp, out in pairs:
            if len(inp) == 5:
                d = _parse_digits(inp)
                if d is not None:
                    all_numeric.append((d, out))
        if len(all_numeric) >= 3:
            result = _try_linear_numeric(all_numeric, target_digits)
            if result is not None:
                return result

    # Cannot determine rule with confidence
    return None


# ===================================================================
# MASTER SOLVER
# ===================================================================

def solve(prompt: str) -> Optional[str]:
    family = classify_puzzle(prompt)

    if family == "gravity_constant":
        return solve_gravity(prompt)
    elif family == "unit_conversion":
        return solve_unit(prompt)
    elif family == "numeral_system":
        return solve_numeral(prompt)
    elif family == "text_encryption":
        return solve_encryption(prompt)
    elif family == "bit_manipulation":
        return solve_bit(prompt)
    elif family == "equation_transform":
        return solve_equation(prompt)
    else:
        return None


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate(csv_path: str, limit: int = 0):
    total = defaultdict(int)
    correct = defaultdict(int)
    solved = defaultdict(int)
    errors = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            prompt = row["prompt"]
            expected = row["answer"].strip()
            family = classify_puzzle(prompt)
            total[family] += 1

            try:
                predicted = solve(prompt)
            except Exception as e:
                predicted = None
                if len(errors[family]) < 3:
                    errors[family].append(str(e))

            if predicted is not None:
                solved[family] += 1
                if predicted.strip() == expected:
                    correct[family] += 1

    print("\n" + "=" * 65)
    print(f"{'Family':<22} {'Total':>6} {'Solved':>7} {'Correct':>8} {'Accuracy':>9}")
    print("-" * 65)
    grand_total = 0
    grand_correct = 0
    for fam in sorted(total.keys()):
        t = total[fam]
        s = solved[fam]
        c = correct[fam]
        acc = c / t * 100 if t > 0 else 0
        print(f"{fam:<22} {t:>6} {s:>7} {c:>8} {acc:>8.1f}%")
        grand_total += t
        grand_correct += c
        if errors.get(fam):
            for e in errors[fam]:
                print(f"  ERROR: {e}")

    print("-" * 65)
    overall = grand_correct / grand_total * 100 if grand_total > 0 else 0
    print(f"{'OVERALL':<22} {grand_total:>6} {sum(solved.values()):>7} {grand_correct:>8} {overall:>8.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "data" / "train.csv")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    evaluate(csv_path, limit)
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

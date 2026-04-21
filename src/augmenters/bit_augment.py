"""Bit manipulation augmenter — symmetry, rotation, NOT.

Baseline: 1602 rows. Target 3-5x expansion.

Kaggle category: `bit_manipulation`. Each example is a pair of 8-bit strings
(input, output) derived from a hidden boolean rule. This module produces
augmented pairs that remain consistent under three label-preserving
transformations of the underlying rule:

    1. Bit-complement symmetry (NOT both sides) —
       valid iff the rule is self-dual for ~x (e.g. NOT, XOR, XNOR).
       We guard with `_is_self_dual_sample` on a 4-row bundle.

    2. Column rotation —
       if all 8 columns share the same rule (pointwise),
       cyclic shift preserves the pair.

    3. Per-bit NOT of the output —
       generates the inverted rule. Used as hard-negative OR as
       a supervised twin with a modified rule indicator.

Usage:
    from src.augmenters.bit_augment import augment_bit
    aug = augment_bit([("10110100", "01001011"), ...], n_augs=3, seed=0)
"""
from __future__ import annotations

import random
from typing import Sequence

Pair = tuple[str, str]


def _complement(bits: str) -> str:
    return "".join("1" if c == "0" else "0" for c in bits)


def _rotate(bits: str, k: int) -> str:
    k = k % len(bits)
    return bits[k:] + bits[:k]


def _is_self_dual_sample(examples: Sequence[Pair]) -> bool:
    """Heuristic: pair is self-dual iff every row satisfies
    output == NOT(input) XOR constant(row), i.e. complementing input
    complements output coherently across all rows.
    """
    if not examples:
        return False
    # Check whether the bitwise rule is linear in the input (XOR-like):
    # f(~x) == ~f(x) for all rows
    for inp, out in examples:
        if len(inp) != len(out):
            return False
    # Conservative: allow only if input XOR output is constant across rows
    # (covers Identity, NOT, per-bit XOR with key) — these are exactly
    # the families where complement-symmetry is label-preserving.
    xors = set()
    for inp, out in examples:
        xors.add("".join("1" if a != b else "0" for a, b in zip(inp, out)))
    return len(xors) == 1


def _is_column_homogeneous(examples: Sequence[Pair]) -> bool:
    """Return True iff every column has the same map from input bit to
    output bit across rows — i.e. rotation is label-preserving.
    """
    if not examples:
        return False
    n = len(examples[0][0])
    # Column c is "pure identity-like" if for all rows, out[c] depends
    # only on inp[c]. Simplest check: projection per column has a
    # consistent 2-entry truth table identical across columns.
    tables = []
    for c in range(n):
        tt: dict[str, set[str]] = {"0": set(), "1": set()}
        for inp, out in examples:
            tt[inp[c]].add(out[c])
        # ambiguous column -> not column-wise
        if any(len(v) > 1 for v in tt.values()):
            return False
        tables.append((tuple(sorted(tt["0"])), tuple(sorted(tt["1"]))))
    return len(set(tables)) == 1


def augment_bit(
    examples: Sequence[Pair],
    n_augs: int = 3,
    seed: int = 0,
) -> list[Pair]:
    """Generate `n_augs` augmented pairs from a batch of bit examples.

    Args:
        examples: list of (input_bits, output_bits) 8-char strings.
                  Treated as a single "rule bundle" — transforms applied
                  coherently to all of them together.
        n_augs: number of augmented *bundles* to return (excluding the
                original, which is always prepended).
        seed: RNG seed for determinism.

    Returns:
        Flat list of pairs. First len(examples) are the originals.
    """
    out: list[Pair] = list(examples)
    if n_augs <= 0 or not examples:
        return out

    rng = random.Random(seed)
    self_dual = _is_self_dual_sample(examples)
    col_homog = _is_column_homogeneous(examples)

    strategies: list[str] = []
    if self_dual:
        strategies.append("complement")
    if col_homog:
        strategies.append("rotate")
    # NOT-twin is always safe as a *separate* rule label — we emit it
    # as a pure rule-variant so downstream can tag it.
    strategies.append("not_twin")

    produced = 0
    attempts = 0
    while produced < n_augs and attempts < n_augs * 4:
        attempts += 1
        strat = rng.choice(strategies)
        if strat == "complement":
            new = [(_complement(i), _complement(o)) for i, o in examples]
        elif strat == "rotate":
            k = rng.randint(1, len(examples[0][0]) - 1)
            new = [(_rotate(i, k), _rotate(o, k)) for i, o in examples]
        else:  # not_twin — complement only output
            new = [(i, _complement(o)) for i, o in examples]
        out.extend(new)
        produced += 1
    return out

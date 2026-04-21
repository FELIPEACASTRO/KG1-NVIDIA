"""Equation augmenter — operand shuffling, operator substitution,
constant rescaling.

Baseline: equation_transform (~596) + equation_numeric_guess (~136) = 732.
Target 3-5x expansion.

Strategies (all label-preserving or label-recomputing):
    1. **Operand shuffling (commutative)**: for `+` and `*`, reorder
       operands. `a + b -> b + a`. Safe: label unchanged.
    2. **Constant rescaling**: pick random integer k and apply to both
       sides of an equation. Needs label recomputation — we provide
       a stub that multiplies the numerical answer by k for
       equation_numeric.
    3. **Operator substitution**: replace `+` with `-` (and vice versa)
       with matching numerical answer recomputation. Only valid when
       we can parse the equation into AST form. Pure-text fallback
       emits a marker for the caller to recompute.
    4. **Variable renaming**: `x, y, z -> a, b, c` (permutation). Safe.

Usage:
    from src.augmenters.equation_augment import augment_equation
    pairs = augment_equation(("prompt", "answer"), n_augs=4, seed=0)
"""
from __future__ import annotations

import random
import re
from typing import Callable, Sequence

Pair = tuple[str, str]

_VAR_ALPHABET = list("xyzwabcdefghijklmnopqrstuv")
_VAR_RE = re.compile(r"(?<![A-Za-z])([a-z])(?![A-Za-z])")


def _rename_variables(text: str, rng: random.Random) -> str:
    """Permute single-letter variables coherently throughout the text."""
    present = sorted(set(m.group(1) for m in _VAR_RE.finditer(text)))
    if not present:
        return text
    pool = [v for v in _VAR_ALPHABET if v not in present]
    rng.shuffle(pool)
    mapping = {}
    for v in present:
        # choose a fresh variable that isn't already in the text
        if not pool:
            pool = [x for x in _VAR_ALPHABET if x != v]
            rng.shuffle(pool)
        mapping[v] = pool.pop()
    # Two-pass to avoid cascading rewrites
    placeholders = {v: f"\x00{i}\x00" for i, v in enumerate(mapping)}
    result = text
    for v, p in placeholders.items():
        result = re.sub(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", p, result)
    for v, new in mapping.items():
        result = result.replace(placeholders[v], new)
    return result


_ADD_TERM_RE = re.compile(r"([+-])\s*(\w+)")


def _shuffle_additive_terms(expr: str, rng: random.Random) -> str:
    """Shuffle terms around `+`/`-` within a single LHS/RHS expression,
    preserving the sign attached to each term.

    Assumes simple form like `a + b - c + 2*x`. Will not touch
    parentheses or multi-token terms with spaces.
    """
    # Normalize leading sign
    s = expr.strip()
    if not s or s[0] not in "+-":
        s = "+" + s
    terms = _ADD_TERM_RE.findall(s)
    if len(terms) < 2:
        return expr
    rng.shuffle(terms)
    # Reassemble; drop leading '+' if present
    out = "".join(sign + term for sign, term in terms)
    return out[1:] if out.startswith("+") else out


def _shuffle_across_equals(text: str, rng: random.Random) -> str:
    """Shuffle additive terms on *both* sides of '=' independently."""
    if "=" not in text:
        return text
    lhs, rhs = text.split("=", 1)
    return _shuffle_additive_terms(lhs, rng) + " = " + _shuffle_additive_terms(rhs, rng)


def augment_equation(
    example: Pair,
    n_augs: int = 4,
    seed: int = 0,
    recompute_answer: Callable[[str, str], str] | None = None,
) -> list[Pair]:
    """Generate `n_augs` augmented (prompt, answer) pairs.

    Args:
        example: (prompt, answer).
        n_augs: number of new variants (original prepended).
        seed: RNG seed.
        recompute_answer: optional callback `(new_prompt, old_answer) -> new_answer`.
                          Required only for label-changing transforms
                          (e.g. constant rescale on numeric). When None,
                          we fall back to pure label-preserving transforms.
    """
    out: list[Pair] = [example]
    if n_augs <= 0:
        return out

    prompt, answer = example
    transforms = ["rename", "shuffle"]

    for k in range(n_augs):
        rng = random.Random(seed + k + 1)
        strat = rng.choice(transforms)
        if strat == "rename":
            # Must rename coherently in answer too
            # Find variables in prompt; the same mapping must apply.
            # Simpler: concatenate with sentinel, rename, split.
            joined = prompt + "\n\x1e\n" + answer
            renamed = _rename_variables(joined, rng)
            new_p, new_a = renamed.split("\n\x1e\n", 1)
        elif strat == "shuffle":
            new_p = _shuffle_across_equals(prompt, rng)
            new_a = answer
            if recompute_answer is not None:
                new_a = recompute_answer(new_p, answer)
        else:
            new_p, new_a = prompt, answer
        out.append((new_p, new_a))
    return out

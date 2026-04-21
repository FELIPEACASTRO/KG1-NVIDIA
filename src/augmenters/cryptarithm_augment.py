"""Cryptarithm augmenter — symbol permutation + example reordering.

Baseline: cryptarithm_deduce (659), cryptarithm_guess (164) — 823 rows
total. Target 5-10x expansion (small categories benefit most).

Strategy:
    1. **Symbol permutation**: the cryptarithm uses 10 symbols mapped
       1-to-1 to digits 0-9. Any bijection f:Σ→Σ' produces a new valid
       problem with the same underlying arithmetic, only the surface
       glyphs change. This multiplies the rule-invariant training
       signal without leaking test distribution.
    2. **Example reordering**: the K few-shot demos inside the prompt
       are interchangeable — shuffling preserves correctness but
       breaks positional shortcut learning.
    3. **Digit-mapping rotation**: for `cryptarithm_guess` where the
       digit assignment is the target, we also rotate the digit->symbol
       mapping (10! options) to resample.

Usage:
    from src.augmenters.cryptarithm_augment import augment_cryptarithm
    pairs = augment_cryptarithm(("prompt...", "answer..."), n_augs=5, seed=0)
"""
from __future__ import annotations

import random
import re
from typing import Sequence

Pair = tuple[str, str]

# Default symbol alphabet for cryptarithm (covers kienngx + tong datasets).
# If problem uses a different set, pass it explicitly.
DEFAULT_SYMBOLS = list("!\"#$%&'()*+-./:;<>?@[\\]^`{|}")


def _collect_symbols(text: str, pool: Sequence[str]) -> list[str]:
    present = [s for s in pool if s in text]
    # Stable order (appearance order in text) so permutation is deterministic.
    present.sort(key=lambda s: text.find(s))
    return present


def _build_mapping(
    symbols: Sequence[str],
    pool: Sequence[str],
    rng: random.Random,
) -> dict[str, str]:
    """Map each `symbols[i]` to a distinct element of `pool` (not itself)."""
    # Candidates: anything from the pool not already matched to self
    pool_list = list(pool)
    rng.shuffle(pool_list)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    # Greedy; fallback to derangement by rotation
    for s in symbols:
        for cand in pool_list:
            if cand != s and cand not in used:
                mapping[s] = cand
                used.add(cand)
                break
    # Patch any missing (rare — only if pool smaller than symbols)
    for s in symbols:
        if s not in mapping:
            leftover = [p for p in pool_list if p not in used]
            mapping[s] = leftover[0] if leftover else s
            used.add(mapping[s])
    return mapping


def _apply_mapping(text: str, mapping: dict[str, str]) -> str:
    # Two-pass with placeholders to avoid re-mapping already-mapped chars
    placeholders = {s: f"\x00{i}\x00" for i, s in enumerate(mapping)}
    result = text
    for s, p in placeholders.items():
        result = result.replace(s, p)
    for s, new in mapping.items():
        result = result.replace(placeholders[s], new)
    return result


_DEMO_LINE = re.compile(r"^\s*\d{2}\s", re.MULTILINE)


def _shuffle_demo_rows(block: str, rng: random.Random) -> str:
    """Shuffle numbered demo lines within a block while keeping the '00 01 02...'
    numbering contiguous (renumber after shuffle).
    """
    lines = block.splitlines()
    demo_idx = [i for i, ln in enumerate(lines) if _DEMO_LINE.match(ln)]
    if len(demo_idx) < 2:
        return block
    demos = [lines[i] for i in demo_idx]
    rng.shuffle(demos)
    for new_pos, i in enumerate(demo_idx):
        # Replace leading "NN " with renumbered "MM "
        stripped = re.sub(r"^\s*\d{2}\s", "", demos[new_pos], count=1)
        lines[i] = f"{new_pos:02d} {stripped}"
    return "\n".join(lines)


def augment_cryptarithm(
    example: Pair,
    n_augs: int = 5,
    seed: int = 0,
    symbols_pool: Sequence[str] | None = None,
    shuffle_demos: bool = True,
) -> list[Pair]:
    """Generate `n_augs` augmented (prompt, answer) pairs.

    Args:
        example: (prompt_text, answer_text) — symbols appearing in both
                 are remapped consistently so correctness is preserved.
        n_augs: number of new variants (original prepended).
        seed: RNG seed.
        symbols_pool: override symbol alphabet. Defaults to
                      DEFAULT_SYMBOLS.
        shuffle_demos: also permute demo row order inside the prompt.
    """
    out: list[Pair] = [example]
    if n_augs <= 0:
        return out

    pool = list(symbols_pool) if symbols_pool is not None else DEFAULT_SYMBOLS
    prompt, answer = example
    present = _collect_symbols(prompt + answer, pool)
    if len(present) < 2:
        # Nothing to permute — fall back to demo-shuffle only
        if not shuffle_demos:
            return out
        rng = random.Random(seed)
        for k in range(n_augs):
            new_p = _shuffle_demo_rows(prompt, random.Random(seed + k + 1))
            out.append((new_p, answer))
        return out

    for k in range(n_augs):
        rng = random.Random(seed + k + 1)
        mapping = _build_mapping(present, pool, rng)
        new_prompt = _apply_mapping(prompt, mapping)
        new_answer = _apply_mapping(answer, mapping)
        if shuffle_demos:
            new_prompt = _shuffle_demo_rows(new_prompt, rng)
        out.append((new_prompt, new_answer))
    return out

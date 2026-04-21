"""Cipher augmenter — Caesar offset rotation, keyword shifts, case flip.

Baseline: 1576 rows. Target 2-3x expansion.

Strategies:
    1. **Caesar re-offset**: if the cipher is a Caesar shift of `k`,
       apply an additional shift `k'` to both plaintext-like and
       ciphertext-like tokens coherently. The *mapping* between them
       is preserved (both shifted by k'), so the rule "input->output
       via shift k" stays intact while surface bytes change.
    2. **Case flip**: uppercase the ciphertext, lowercase the plaintext,
       or vice versa — forces model to be case-invariant.
    3. **Substring reuse**: repeat the key/plaintext pair multiple times
       with minor length variations.

The safest augmenter is (1) — it works regardless of whether we know
`k`, because we never need to decode. We apply the *same* shift to
every alphabetic character in both sides of every demo row.

Usage:
    from src.augmenters.cipher_augment import augment_cipher
    pairs = augment_cipher(("prompt with demos", "answer"), n_augs=2, seed=0)
"""
from __future__ import annotations

import random
from typing import Sequence

Pair = tuple[str, str]


def _caesar_shift(text: str, k: int) -> str:
    """Shift every ASCII letter by k positions (preserving case, leaving
    non-letters untouched)."""
    out_chars: list[str] = []
    for ch in text:
        if "a" <= ch <= "z":
            out_chars.append(chr((ord(ch) - ord("a") + k) % 26 + ord("a")))
        elif "A" <= ch <= "Z":
            out_chars.append(chr((ord(ch) - ord("A") + k) % 26 + ord("A")))
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def _case_flip(text: str) -> str:
    return text.swapcase()


def _reverse_alphabet_map(text: str) -> str:
    """Atbash: a<->z, b<->y, ... . Useful as an additional rule-preserving
    transform only when the hidden rule is itself Caesar-like (reflection
    commutes with any shift). We keep it as an OPTIONAL strategy, opt-in."""
    out_chars: list[str] = []
    for ch in text:
        if "a" <= ch <= "z":
            out_chars.append(chr(ord("z") - (ord(ch) - ord("a"))))
        elif "A" <= ch <= "Z":
            out_chars.append(chr(ord("Z") - (ord(ch) - ord("A"))))
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def augment_cipher(
    example: Pair,
    n_augs: int = 2,
    seed: int = 0,
    strategies: Sequence[str] | None = None,
) -> list[Pair]:
    """Generate `n_augs` augmented cipher pairs.

    Args:
        example: (prompt, answer). Any Caesar-shift applied is applied
                 to BOTH — preserving the input->output mapping.
        n_augs: number of new variants (original prepended).
        seed: RNG seed.
        strategies: subset of {"caesar", "case_flip", "atbash"}.
                    Default: {"caesar", "case_flip"}. Atbash is opt-in
                    because it is only safe for strictly-Caesar hidden
                    rules — it will break keyword or Vigenère ciphers.
    """
    out: list[Pair] = [example]
    if n_augs <= 0:
        return out
    strat_pool = list(strategies) if strategies is not None else ["caesar", "case_flip"]
    if not strat_pool:
        return out

    prompt, answer = example

    for k in range(n_augs):
        rng = random.Random(seed + k + 1)
        strat = rng.choice(strat_pool)
        if strat == "caesar":
            shift = rng.randint(1, 25)
            new_p = _caesar_shift(prompt, shift)
            new_a = _caesar_shift(answer, shift)
        elif strat == "case_flip":
            new_p = _case_flip(prompt)
            new_a = _case_flip(answer)
        elif strat == "atbash":
            new_p = _reverse_alphabet_map(prompt)
            new_a = _reverse_alphabet_map(answer)
        else:
            new_p, new_a = prompt, answer
        out.append((new_p, new_a))
    return out

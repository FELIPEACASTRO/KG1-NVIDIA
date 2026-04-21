"""Data augmenters for the NVIDIA Nemotron Kaggle dataset.

Per-category augmenters that expand training rows deterministically
to improve robustness (temp=0 Kaggle inference demands stable models).

Modules:
    bit_augment        — symmetry / rotation / NOT for bit_manipulation
    cryptarithm_augment — symbol permutation for cryptarithm_deduce/guess
    equation_augment    — operand shuffling / operator substitution
    cipher_augment      — Caesar offset / keyword rotation

All augmenters follow the contract:
    augment_<name>(example, n_augs, seed=0) -> list[tuple[str, str]]

where example is (input, output) and the first item returned is always
the original example (idempotency) — the remaining n_augs are synthetic.
"""
from __future__ import annotations

from .bit_augment import augment_bit
from .cipher_augment import augment_cipher
from .cryptarithm_augment import augment_cryptarithm
from .equation_augment import augment_equation

__all__ = [
    "augment_bit",
    "augment_cipher",
    "augment_cryptarithm",
    "augment_equation",
]

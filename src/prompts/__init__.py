"""KG1 prompt builders — category-aware for V71+ training.

Empirically validated by Agent T3 (2026-04-21) on actual Nemotron tokenizer:
- Bits/digits already atomic (no spacing needed)
- Cipher letters MERGED (spacing helps +15-72pp per paper 2505.14178)
- Cryptarithm symbols MERGED (spacing helps)
"""
from .build_prompt import (  # noqa: F401
    build_prompt_v71,
    detect_category,
    BOXED_INSTRUCTION,
    CATEGORY_HINTS,
)

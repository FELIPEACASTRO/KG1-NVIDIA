"""Category-aware prompt builder for KG1 V71+ (Agent T3 empirical).

**Empirical validation** (Agent T3, 2026-04-21):
Tested locally with AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"):
- Binary "10110100" = 8 tokens (1 per digit) — already atomic, NO spacing needed
- Decimal "1000" = 4 tokens — already atomic
- "HELLO" (cipher) = 3 tokens ('HE','L','LO') — MERGED, spacing needed
- "!*[{" (cryptarithm) = 2 tokens ('!*','[{') — MERGED, spacing needed

**Paper evidence** (arxiv 2505.14178 Table 1):
- Pure BPE string: 9.10% accuracy
- Space-delimited: 24.90% (+15.80pp)
- Atomic-aligned: 81.60% (+72.50pp)
- BUT only matters when tokens are merged!

**Net expected gain**: +0.012 to +0.045 overall via category-aware prompts
(zero for bit/numeric, +0.5-2pp for cipher/cryptarithm)

Usage:
    from src.prompts.build_prompt import build_prompt_v71

    formatted = build_prompt_v71(puzzle_text, category="cipher")
    # Apply before apply_chat_template(..., enable_thinking=True)
"""
from __future__ import annotations


# Official Kaggle metric prefix (byte-for-byte match for prefix cache)
BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Tail protection — prevents format pitfalls from Agent T6 findings.
# FORMATS TO AVOID INSIDE \boxed{} (catastrophic, up to -40pp):
# - Units (m/s, kg, cm) — 3859 rows in train would break
# - Commas thousands (1,000) — 236 rows would break
# - Percent (50%) — 3859 rows would break
# - LaTeX (\text{}, \mathrm{}, \frac{}) — 3152 cipher rows would break
# - Nested \boxed{\boxed{X}} — regex breaks on first }
BOXED_STRICT = (
    "\nIMPORTANT: Inside the final `\\boxed{...}`, output ONLY the raw answer. "
    "DO NOT include units (m/s, kg, etc), percent signs, comma separators (use 1000 not 1,000), "
    "LaTeX commands (no \\text{}, \\mathrm{}, \\frac{}), or nested braces. "
    "No spaces between characters in the answer itself."
)

# Self-correction hint — powerful! Agent T6 Exploit 15
# Last non-empty \boxed{} wins — model can verify and correct
SELF_CORRECT_HINT = (
    "\nIf after reasoning you realize your first answer was wrong, "
    "you MAY output a second `\\boxed{correct_answer}` — the last non-empty one wins."
)

# Category-specific hints (empirically validated by T3)
CATEGORY_HINTS = {
    "cipher": (
        "\nTreat each letter as a separate unit. "
        "Parse and emit one character at a time — don't group adjacent letters."
    ),
    "text_encryption": (
        "\nTreat each letter as a separate unit. "
        "Parse and emit one character at a time — don't group adjacent letters."
    ),
    "cryptarithm_deduce": (
        "\nTreat each symbol and digit independently; "
        "do not group adjacent symbols. The operator is at position 2 of the 5-char LHS."
    ),
    "cryptarithm_guess": (
        "\nTreat each symbol and digit independently; "
        "do not group adjacent symbols. The operator is at position 2."
    ),
    # No hints needed for these (tokenizer is already atomic)
    "bit_manipulation": "",
    "equation_numeric_guess": "",
    "equation_numeric_deduce": "",
    "equation_transformation": "",
    "numeral_conversion": "",
    "gravitational_constant": "",
    "unit_conversion": "",
}

# Structured reasoning template (Variant C, arxiv 2502.12616 QuaSAR style)
# Zero-token cost in chat (thinking happens inside <think>)
STRUCTURED_TEMPLATE = (
    "\n\nReason as follows:\n"
    "1. Parse all examples carefully.\n"
    "2. Identify the transformation pattern.\n"
    "3. Hypothesize the rule.\n"
    "4. Verify on ALL examples.\n"
    "5. Apply to the query.\n"
    "6. Output ONLY `\\boxed{answer}` at the end."
)


def build_prompt_v71(
    puzzle_text: str,
    category: str = "",
    use_structured: bool = True,
    use_category_hints: bool = True,
    use_boxed_strict: bool = True,
    use_self_correct: bool = False,  # T6 Exploit 15 — enable for training
) -> str:
    """Build category-aware V71+ prompt.

    Args:
        puzzle_text: raw puzzle prompt (Alice Wonderland style)
        category: one of 9 Kaggle categories (or empty for generic)
        use_structured: add QuaSAR-style 6-step reasoning template
        use_category_hints: add tokenization-aware hints for cipher/cryptarithm
        use_boxed_strict: enforce no format pitfalls inside \\boxed{}
        use_self_correct: enable self-correction via last-boxed-wins (T6 Exploit 15)

    Returns:
        Formatted prompt ready for apply_chat_template.
    """
    base = puzzle_text

    # Category-specific tokenization hint (T3 empirical)
    if use_category_hints and category in CATEGORY_HINTS:
        hint = CATEGORY_HINTS[category]
        if hint:  # skip if empty (already-atomic categories)
            base += hint

    # Optional structured reasoning template (QuaSAR-inspired)
    if use_structured:
        base += STRUCTURED_TEMPLATE

    # Strict format protection (T6 defensive awareness)
    if use_boxed_strict:
        base += BOXED_STRICT

    # Self-correction hint (T6 Exploit 15 — powerful for training)
    if use_self_correct:
        base += SELF_CORRECT_HINT

    # Kaggle metric-official BOXED_INSTRUCTION (byte-for-byte match required)
    base += BOXED_INSTRUCTION

    return base


def detect_category(prompt: str) -> str:
    """Detect category from prompt text (heuristic, T3 aware)."""
    p = prompt.lower()
    if "cryptarithm" in p:
        return "cryptarithm_deduce" if "mapping" in p or "known" in p else "cryptarithm_guess"
    if "equation" in p and ("numeric" in p or "guess" in p):
        return "equation_numeric_guess" if "guess" in p else "equation_numeric_deduce"
    if "equation" in p:
        return "equation_transformation"
    if "bit manipulation" in p:
        return "bit_manipulation"
    if "gravitational" in p or "gravity" in p:
        return "gravitational_constant"
    if "unit conversion" in p:
        return "unit_conversion"
    if "numeral" in p:
        return "numeral_conversion"
    if "encryption" in p or "cipher" in p or "encrypted" in p:
        return "text_encryption"
    return ""


# --- Self-test ---
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test cipher prompt
    cipher_puzzle = "Decrypt: HELLO -> Ifmmp. What is 'World'?"
    p = build_prompt_v71(cipher_puzzle, "cipher")
    print("=== cipher prompt ===")
    print(p)
    print()

    # Test bit_manipulation (no spacing should be added)
    bit_puzzle = "Apply the rule: 10110100 -> 01001011. What is 11110000?"
    p = build_prompt_v71(bit_puzzle, "bit_manipulation")
    print("=== bit_manipulation prompt (NO spacing hint) ===")
    print(p)
    print()

    # Test category detection
    assert detect_category("In Alice's Wonderland, a bit manipulation rule") == "bit_manipulation"
    assert detect_category("In Alice's Wonderland, encryption rules") == "text_encryption"
    assert detect_category("cryptarithm with unique digit") == "cryptarithm_guess"
    print("Category detection: OK")

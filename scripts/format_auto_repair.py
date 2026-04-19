#!/usr/bin/env python3
"""Format auto-repair for Kaggle Nemotron scorer.

Claude-Opus-4-7 critical insight: "half the 178-team plateau is probably
format losses, not reasoning losses." Post-hoc regex fixup of common
\\boxed{} failures pairs with #689580 scorer fix.

Common failures to repair:
1. Leading zeros stripped ("0234" → "234") — #687798 HOST: binary is STRING now
2. Nested braces ("\\boxed{\\boxed{42}}") — scorer regex r'\\boxed\{([^}]*)(?:\}|$)' stops at first `}`
3. Unit strings ("5 meters" → "5") — many answers reject units
4. Extra whitespace ("\\boxed{ 42 }" → "\\boxed{42}")
5. LaTeX wrapping ("\\boxed{\\text{42}}")
6. Multiple \\boxed instances — scorer likely takes LAST one

Usage:
    python scripts/format_auto_repair.py < raw_model_output.txt > repaired.txt
    or as module: from format_auto_repair import repair_boxed_answer
"""
import re
import sys


def repair_boxed_answer(text: str,
                        expected_category: str = None,
                        examples: list = None,
                        preserve_leading_zeros: bool = True) -> str:
    """Repair common \\boxed{} format issues in model output.

    Args:
        text: raw model output
        expected_category: puzzle family (bit, cryptarithm, numeric, cipher, etc)
        examples: example I/O to infer expected format
        preserve_leading_zeros: if True, treat answers as strings preserving '0234'

    Returns:
        text with \\boxed{...} normalized for scorer.
    """
    if not text:
        return text

    # Find ALL \boxed{} spans (scorer takes first matching)
    # Use non-greedy to stop at first `}`
    matches = list(re.finditer(r"\\boxed\{([^{}]*)(?:\}|$)", text))

    if not matches:
        # No boxed found - try to rescue: look for "Answer: X" at end
        last_line = text.strip().split("\n")[-1]
        ans_match = re.search(r"(?:answer|final|result)[:\s]*\s*([a-zA-Z0-9\-\+\.\s]+?)$",
                              last_line, re.IGNORECASE)
        if ans_match:
            extracted = ans_match.group(1).strip()
            # Inject boxed at end
            text = text + f"\n\n\\boxed{{{extracted}}}"
            return text
        return text

    # Take LAST boxed (if scorer uses first/last varies) — both safe patterns
    # Per Claude: "scorer takes first" per #689580. Use FIRST.
    match = matches[0]
    raw_content = match.group(1)

    # Clean content
    cleaned = raw_content.strip()

    # Strip nested \boxed (common model mistake)
    cleaned = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", cleaned)
    # Strip \text{...}
    cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
    # Strip $ $ LaTeX delimiters
    cleaned = cleaned.replace("$", "").strip()
    # Strip whitespace
    cleaned = cleaned.strip()

    # Category-specific cleanup
    if expected_category:
        cat = expected_category.lower()

        if "bit" in cat or "binary" in cat:
            # Strip anything that's not 0/1 (common: "0b01110000" → "01110000")
            cleaned = re.sub(r"[^01]", "", cleaned)
            # Normalize length if examples suggest fixed width
            if examples:
                lens = [len(str(ex[2])) for ex in examples if len(ex) >= 3]
                if lens and max(lens) == min(lens):
                    # Fixed length - pad with leading zeros
                    target_len = lens[0]
                    cleaned = cleaned.zfill(target_len)[:target_len]

        elif "numeric" in cat or "numeral" in cat or "equation" in cat or "gravity" in cat or "unit" in cat:
            # Strip units, keep digits + signs + decimal
            # Handle negative: preserve leading "-"
            neg = cleaned.startswith("-")
            cleaned = re.sub(r"[^\d\.\-eE\+]", "", cleaned)
            if not cleaned:
                pass
            elif preserve_leading_zeros:
                # Keep as-is (Host confirmed string comparison for binary-like)
                pass
            else:
                # Try to convert to canonical number form
                try:
                    f = float(cleaned)
                    if f == int(f):
                        cleaned = str(int(f))
                    else:
                        cleaned = f"{f:g}"
                except ValueError:
                    pass

        elif "cipher" in cat or "text" in cat or "encryption" in cat:
            # Strip \text{} wrapping, keep alphanumeric
            cleaned = cleaned.strip('"').strip("'")
            # Don't normalize case - ciphers are case-sensitive

        elif "cryptarithm" in cat:
            # Preserve exactly (letters or digits)
            pass

    # Replace the first \boxed{} with cleaned content
    start, end = match.span()
    repaired = text[:start] + f"\\boxed{{{cleaned}}}" + text[end:]

    return repaired


def extract_scorer_answer(text: str) -> str:
    """Apply Kaggle scorer regex to extract answer (same as scorer).

    Per #689580: r'\\boxed\{([^}]*)(?:\}|$)' — stops at first `}`
    """
    match = re.search(r"\\boxed\{([^}]*)(?:\}|$)", text)
    return match.group(1).strip() if match else ""


def test():
    # Demo cases
    cases = [
        ("bit", "The answer is \\boxed{0b01110000}", [("1", "1", "01010101")]),
        ("numeric", "\\boxed{42 meters}", None),
        ("numeric", "\\boxed{\\text{100}}", None),
        ("bit", "\\boxed{ 1100 }", [("1", "1", "0011")]),
        ("cryptarithm", "So FINAL = \\boxed{10652}", None),
        ("numeric", "I think: 5. Final: \\boxed{5.0}", None),
        ("numeric", "Result: 20", None),  # no boxed - rescue case
    ]
    for cat, text, examples in cases:
        repaired = repair_boxed_answer(text, cat, examples)
        scored = extract_scorer_answer(repaired)
        print(f"[{cat}] input: {text!r}")
        print(f"  repaired: {repaired!r}")
        print(f"  scorer sees: {scored!r}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test()
    else:
        # Read from stdin, repair, write to stdout
        data = sys.stdin.read()
        print(repair_boxed_answer(data))

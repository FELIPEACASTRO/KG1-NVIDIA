#!/usr/bin/env python3
"""Kaggle-metric aware canonicalizer for Nemotron Reasoning Challenge outputs.

Reverse-engineered from the OFFICIAL metric at
``C:/Users/davis/AppData/Local/Temp/tc/metric_official_exact.py`` and the
ultra_consensus_report.md findings (2026-04-22).

Applies format-level fixes so the model's raw output maximizes the pass-rate of
``extract_final_answer`` + ``verify`` inside Kaggle's scorer:

1. Strip nested braces inside ``\\boxed{}`` (``\\frac``, ``\\text``, ``\\sqrt``).
2. Strip thousand separators (``,`` and ``_``) from numeric answers.
3. Convert scientific notation to decimal (``1e-3`` -> ``0.001``).
4. Strip unit suffixes (``m/s``, ``kg``, ``m/s^2``, etc).
5. Convert ``\\frac{a}{b}`` LaTeX fractions to decimal.
6. Strip trailing ``.0`` on integers (binary collision fix — BUG-2).
7. Force lowercase for cipher/cryptarithm (verify uses ``.lower()``).
8. Force exact 8-bit zero-fill for bit_manipulation.
9. For equation_transform, prefer ``Final answer is: X`` (bypass brace bug).
10. Ensure a single terminal ``\\boxed{CLEAN}`` is appended.

The module is intentionally dependency-light: only ``re`` and ``math`` + the
canonical answer helpers from ``src.competition_utils`` are used.

Usage (as a library)::

    from scripts.kg1_canonicalize_output import canonicalize_answer
    cleaned = canonicalize_answer(raw_text, family_hint="bit_manipulation")

Usage (as a CLI)::

    python scripts/kg1_canonicalize_output.py \
        --input raw_predictions.csv \
        --output predictions_canonical.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Soft import — canonicalizer MUST stay importable even if src changes.
try:
    from src.competition_utils import canonical_boxed_payload
except Exception:  # pragma: no cover - fallback for stripped environments
    def canonical_boxed_payload(text: object) -> str:
        return str(text or "").strip()


# ---------------------------------------------------------------------------
# Family detection (prompt-based).
# ---------------------------------------------------------------------------

_FAMILY_KEYWORDS = (
    ("bit_manipulation", ("bit manipulation", "binary", "xor the bits", "shift the bits")),
    ("text_encryption", ("decrypt the following text", "encryption", "cipher")),
    ("numeral_system", ("numeral system", "converted into a different numeral", "roman numerals")),
    ("gravity_constant", ("gravitational", "gravity constant", "acceleration due to gravity")),
    ("equation_transform", ("transformation rule", "transformation rules")),
    ("unit_conversion", ("unit conversion", "measurement", "convert the following")),
)


def detect_family(prompt: str) -> Optional[str]:
    """Return the canonical family derived from the prompt, or None."""
    if not prompt:
        return None
    low = prompt.lower()
    for family, keywords in _FAMILY_KEYWORDS:
        for kw in keywords:
            if kw in low:
                return family
    return None


# ---------------------------------------------------------------------------
# LaTeX stripping (regex-based, covers the common cases).
# ---------------------------------------------------------------------------

_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_LATEX_MATHRM_RE = re.compile(r"\\mathrm\s*\{([^{}]*)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_LATEX_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)")
_LATEX_FRAC_RE = re.compile(r"\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def _strip_latex_cosmetic(text: str) -> str:
    """Strip cosmetic LaTeX wrappers that would break verify()."""
    text = _LATEX_TEXT_RE.sub(r"\1", text)
    text = _LATEX_MATHRM_RE.sub(r"\1", text)
    text = _LATEX_SQRT_RE.sub(r"\1", text)
    text = _LATEX_LEFT_RIGHT_RE.sub("", text)
    return text


def _eval_latex_fractions(text: str) -> str:
    """Replace ``\\frac{a}{b}`` with the decimal value when a/b are numeric."""

    def _sub(match: re.Match) -> str:
        num_raw, den_raw = match.group(1), match.group(2)
        num_raw = num_raw.strip()
        den_raw = den_raw.strip()
        try:
            num = float(num_raw)
            den = float(den_raw)
            if den == 0:
                return match.group(0)
            value = num / den
            if value == int(value):
                return str(int(value))
            # Keep enough precision for rel_tol=1e-2 but strip trailing zeros.
            return format(value, ".10g")
        except ValueError:
            # Fall back to a/b textual form (still closer to something parseable
            # than the raw LaTeX).
            return f"{num_raw}/{den_raw}"

    for _ in range(3):  # iterate for nested fractions
        new_text = _LATEX_FRAC_RE.sub(_sub, text)
        if new_text == text:
            break
        text = new_text
    return text


# ---------------------------------------------------------------------------
# Unit / thousand separator stripping.
# ---------------------------------------------------------------------------

# Ordered from most specific to least to avoid partial matches (e.g. "m/s^2"
# must match before "m/s" before "m").
_UNIT_SUFFIXES = (
    "m/s^2", "m/s\u00b2", "m s^{-2}",
    "km/h", "km/s", "m/s",
    "kg\u00b7m/s", "kg*m/s",
    "\u00b0C", "\u00b0F", "\u00b0K",
    "kg", "mg", "g",
    "km", "cm", "mm", "nm", "\u00b5m", "um", "m",
    "ms", "\u00b5s", "us", "ns", "s",
    "Hz", "kHz", "MHz", "GHz",
    "J", "kJ", "MJ",
    "W", "kW", "MW",
    "N", "Pa", "kPa", "MPa",
    "V", "kV",
    "A", "mA",
    "\u03a9", "ohm", "Ohm",
)

_UNIT_RE = re.compile(
    r"\s*(?:" + "|".join(re.escape(u) for u in _UNIT_SUFFIXES) + r")\b\s*$",
    flags=re.IGNORECASE,
)


def _strip_units(text: str) -> str:
    """Strip a trailing unit suffix (if any)."""
    return _UNIT_RE.sub("", text).strip()


_THOUSAND_RE = re.compile(r"(?<=\d)[,_](?=\d{3}\b)")


def _strip_thousand_separators(text: str) -> str:
    """Strip ``,`` / ``_`` used as thousand separators (``1,234`` -> ``1234``)."""
    return _THOUSAND_RE.sub("", text)


# ---------------------------------------------------------------------------
# Scientific-notation -> decimal.
# ---------------------------------------------------------------------------

_SCI_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)[eE]([-+]?\d+)")


def _scientific_to_decimal(text: str) -> str:
    """Convert ``1e-3``, ``1.5e10`` etc to plain decimal form."""

    def _sub(match: re.Match) -> str:
        mantissa = match.group(1)
        exponent = int(match.group(2))
        try:
            value = float(mantissa) * (10 ** exponent)
        except (ValueError, OverflowError):
            return match.group(0)
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        return format(value, ".10g")

    return _SCI_RE.sub(_sub, text)


# ---------------------------------------------------------------------------
# Number cleanup (trailing .0 on integers — fixes BUG-2 binary-decimal).
# ---------------------------------------------------------------------------

_TRAILING_DOT_ZERO_RE = re.compile(r"^(-?\d+)\.0+$")


def _strip_trailing_dot_zero(text: str) -> str:
    """``100.0`` -> ``100``; required because verify() forces binary path on 0/1 strings."""
    m = _TRAILING_DOT_ZERO_RE.match(text.strip())
    return m.group(1) if m else text


# ---------------------------------------------------------------------------
# Family-specific enforcers.
# ---------------------------------------------------------------------------

_BINARY_RE = re.compile(r"[01]+")


def _enforce_bit_manipulation(text: str) -> str:
    """Force 8-bit zero-filled binary (BUG-5 protection)."""
    stripped = text.strip()
    # Extract the longest trailing run of 0/1 digits.
    match = re.search(r"[01]{1,16}$", stripped)
    if match:
        bits = match.group(0)
        if len(bits) < 8:
            bits = bits.zfill(8)
        elif len(bits) > 8:
            bits = bits[-8:]
        return bits
    # Try to cast from decimal form (e.g. "81" -> "01010001").
    try:
        value = int(stripped.split()[0])
        if 0 <= value < 256:
            return format(value, "08b")
    except (ValueError, IndexError):
        pass
    return stripped


def _enforce_cipher(text: str) -> str:
    """Cipher answers are lowercase and keep spaces (verify uses .lower()).

    Strips trailing punctuation (BUG-6) since verify() is strict on it.
    """
    return text.strip().rstrip(".!?;,").lower()


def _enforce_numeral(text: str) -> str:
    """Roman numerals: keep case as-is (verify falls back to case-insensitive)."""
    # Strip trailing punctuation that would break strict match (BUG-6).
    return text.strip().rstrip(".!?;,")


def _enforce_gravity_or_unit(text: str) -> str:
    """Float answers: strip units, separators, scientific notation."""
    text = _strip_units(text)
    text = _strip_thousand_separators(text)
    text = _scientific_to_decimal(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Boxed extraction/parsing for raw model output.
# ---------------------------------------------------------------------------

_BOXED_START_RE = re.compile(r"\\boxed\s*\{")


def _extract_last_boxed_body(raw: str) -> Optional[str]:
    """Return the content of the LAST top-level ``\\boxed{...}`` block.

    Uses a brace-depth walk so nested ``{`` are kept (unlike the Kaggle
    extractor which truncates at the first ``}``).
    """
    if not raw:
        return None
    last_body: Optional[str] = None
    for match in _BOXED_START_RE.finditer(raw):
        start = match.end()
        depth = 1
        i = start
        while i < len(raw) and depth > 0:
            ch = raw[i]
            if ch == "\\":
                i += 2  # skip escaped char
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_body = raw[start:i]
                    break
            i += 1
        if depth > 0 and last_body is None:
            # Unclosed box — take everything until EOS (matches metric fallback).
            last_body = raw[start:]
    return last_body


def _extract_final_answer_fallback(raw: str) -> Optional[str]:
    """Replicate the metric's fallback patterns if no boxed is present."""
    patterns = (
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:\uff1a]\s*([^\n]+)",
        r"final answer\s*[:\uff1a]\s*([^\n]+)",
    )
    for pat in patterns:
        matches = re.findall(pat, raw, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Main canonicalize entrypoint.
# ---------------------------------------------------------------------------


def canonicalize_answer(raw_output: str, family_hint: Optional[str] = None) -> str:
    """Return ``raw_output`` rewritten so it maximises pass-rate on Kaggle.

    Args:
        raw_output: the raw text generated by the LoRA (may contain reasoning,
            multiple boxed attempts, etc.)
        family_hint: optional canonical family name. If not provided, the
            returned text is only format-cleaned, not family-enforced.

    Returns:
        A string that ends with exactly one clean ``\\boxed{CLEAN}``. When the
        family is ``equation_transform``, a ``Final answer is: CLEAN`` line is
        prepended as well (bypasses the brace truncation bug).
    """
    if raw_output is None:
        return "\\boxed{NOT_FOUND}"

    text = str(raw_output)

    # 1. Extract candidate boxed payload.
    body = _extract_last_boxed_body(text)
    if body is None:
        body = _extract_final_answer_fallback(text)
    if body is None:
        # Fall back to the cleaned raw text (metric last-resort is the last line).
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        body = lines[-1] if lines else "NOT_FOUND"

    # 2. LaTeX cosmetic strip.
    body = _strip_latex_cosmetic(body)
    body = _eval_latex_fractions(body)

    # 3. Basic post-processing shared by all numeric families.
    cleaned = body.strip()
    cleaned = canonical_boxed_payload(cleaned)

    # 4. Family-specific normalisation.
    fam = (family_hint or "").strip().lower()
    if fam in {"bit_manipulation", "bits", "bit"}:
        cleaned = _enforce_bit_manipulation(cleaned)
    elif fam in {"text_encryption", "cipher", "encryption"}:
        cleaned = _enforce_cipher(cleaned)
    elif fam in {"numeral_system", "numeral", "roman"}:
        cleaned = _enforce_numeral(cleaned)
    elif fam in {"gravity_constant", "gravity", "unit_conversion", "unit"}:
        cleaned = _enforce_gravity_or_unit(cleaned)
    elif fam in {"equation_transform", "equation", "eq"}:
        cleaned = _strip_thousand_separators(cleaned)
        cleaned = _scientific_to_decimal(cleaned)
        cleaned = cleaned.strip()
    else:
        # Unknown family: apply the safest subset.
        cleaned = _strip_thousand_separators(cleaned)
        cleaned = _scientific_to_decimal(cleaned)
        cleaned = cleaned.strip()

    # 5. Strip trailing .0 on integers (binary collision fix).
    cleaned = _strip_trailing_dot_zero(cleaned)

    # 6. ASCII minus only.
    cleaned = cleaned.replace("\u2212", "-")

    # 7. Build final envelope. For equation_transform, prefer the "Final answer
    #    is: X" pattern which bypasses the brace-truncation bug on symbolic
    #    answers (94 of the 1555 equation rows contain a closing brace).
    if fam in {"equation_transform", "equation", "eq"}:
        return f"Final answer is: {cleaned}\n\\boxed{{{cleaned}}}"
    return f"\\boxed{{{cleaned}}}"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _canonicalize_csv(
    input_path: Path,
    output_path: Path,
    prediction_col: str = "prediction",
    family_col: Optional[str] = None,
    prompt_col: Optional[str] = None,
) -> int:
    """Apply canonicalize_answer to every row of ``input_path`` and write results.

    Returns the number of rows written.
    """
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if prediction_col not in fieldnames:
            raise ValueError(f"{input_path} is missing column '{prediction_col}'")
        rows = list(reader)

    for row in rows:
        family = None
        if family_col and row.get(family_col):
            family = row[family_col]
        elif prompt_col and row.get(prompt_col):
            family = detect_family(row[prompt_col])
        row[prediction_col] = canonicalize_answer(row.get(prediction_col, ""), family)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with raw predictions")
    parser.add_argument("--output", required=True, type=Path, help="CSV with canonicalised predictions")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--family-col", default=None, help="Optional column containing family labels")
    parser.add_argument("--prompt-col", default="prompt", help="Column used to auto-detect the family")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = _canonicalize_csv(
        args.input,
        args.output,
        prediction_col=args.prediction_col,
        family_col=args.family_col,
        prompt_col=args.prompt_col,
    )
    print(f"Canonicalised {written} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

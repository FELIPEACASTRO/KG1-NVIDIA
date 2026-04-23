#!/usr/bin/env python3
"""Rule-based CoT verifier (huikang trick #4 -> +0.06 LB).

Validates that every CoT in a training file actually produces the correct
boxed answer when the extractor runs over the completion. Any CoT whose
extracted answer does not ``verify()`` against the ground truth is dropped.

Per-type rules (matches `konbu17/nemotron-sft-lora-with-cot` md cell 0):

- **bit_manipulation**: exact 8-bit match.
- **cipher**: case-insensitive exact.
- **numeral**: case-insensitive exact (Roman).
- **gravity / unit**: numeric with ``rel_tol=5e-3`` (±0.5%, huikang's choice).
- **equation**: exact string (after whitespace strip).

Outputs a cleaned JSONL with only rows whose CoT passes verification.

Usage::

    python scripts/kg1_verify_cots.py --input solver_sft.jsonl \
        --output solver_sft_verified.jsonl --report-json runs/verify_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Extraction (mirrors scripts/kg1_canonicalize_output helpers for consistency).
# ---------------------------------------------------------------------------

_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_BINARY_RE = re.compile(r"^[01]+$")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

_FALLBACK_PATTERNS = (
    r"The final answer is:\s*([^\n]+)",
    r"Final answer is:\s*([^\n]+)",
    r"Final answer\s*[:\uff1a]\s*([^\n]+)",
)


def _extract_last_boxed(text: str) -> Optional[str]:
    """Brace-aware extractor that survives nested ``\\boxed{\\frac{1}{2}}``."""
    if not text:
        return None
    last: Optional[str] = None
    for match in _BOXED_START_RE.finditer(text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
            i += 1
        if depth > 0 and last is None:
            last = text[start:]
    return last


def _extract_answer(completion: str) -> Optional[str]:
    """Extract the final answer from a completion using boxed + fallbacks."""
    body = _extract_last_boxed(completion)
    if body is not None:
        return body.strip()
    for pat in _FALLBACK_PATTERNS:
        matches = re.findall(pat, completion, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Per-type verifiers (rule-based, matches huikang's correctness filter).
# ---------------------------------------------------------------------------


def _verify_bit(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    return pred.strip() == gt.strip() and bool(_BINARY_RE.match(gt))


def _verify_numeric(pred: Optional[str], gt: str, rel_tol: float = 5e-3, abs_tol: float = 0.05) -> bool:
    """Gravity + unit: ±0.5% relative or ±0.05 absolute (huikang tolerance)."""
    if pred is None:
        return False
    try:
        p = float(pred.strip())
        g = float(gt.strip())
    except ValueError:
        return False
    return math.isclose(p, g, rel_tol=rel_tol, abs_tol=abs_tol)


def _verify_case_insensitive(pred: Optional[str], gt: str) -> bool:
    """Cipher + numeral (Roman): case-insensitive exact."""
    if pred is None:
        return False
    return pred.strip().lower() == gt.strip().lower()


def _verify_equation(pred: Optional[str], gt: str) -> bool:
    """Equation: exact-string after whitespace strip."""
    if pred is None:
        return False
    return pred.strip() == gt.strip()


def _verify_cot(pred: Optional[str], gt: str, category: str) -> bool:
    if category == "bit_manipulation":
        return _verify_bit(pred, gt)
    if category in ("gravity", "unit_conversion"):
        return _verify_numeric(pred, gt)
    if category in ("cipher", "numeral"):
        return _verify_case_insensitive(pred, gt)
    if category.startswith("equation_") or category.startswith("cryptarithm_"):
        return _verify_equation(pred, gt)
    # Unknown family: fall back to the strictest (exact match after strip).
    return pred is not None and pred.strip() == gt.strip()


# ---------------------------------------------------------------------------
# Main filter pass.
# ---------------------------------------------------------------------------


def filter_verified(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    """Filter ``input_path`` keeping only rows whose CoT verifies.

    Returns ``(kept, total, per_category_stats)``.
    """
    kept = 0
    total = 0
    stats: Dict[str, Dict[str, int]] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            completion = row.get("completion", "") or row.get("cot", "")
            gt = row.get("answer", "")
            category = row.get("category") or "unknown"

            pred = _extract_answer(completion)
            ok = _verify_cot(pred, gt, category)

            bucket = stats.setdefault(category, {"kept": 0, "dropped": 0})
            if ok:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                bucket["kept"] += 1
                kept += 1
            else:
                bucket["dropped"] += 1

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "total": total,
                    "kept": kept,
                    "dropped": total - kept,
                    "kept_pct": round(100.0 * kept / max(total, 1), 2),
                    "per_category": stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return kept, total, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Solver JSONL to verify")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL (verified only)")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kept, total, stats = filter_verified(args.input, args.output, args.report_json)
    pct = 100.0 * kept / max(total, 1)
    print(f"Verified {kept}/{total} rows ({pct:.1f}%) -> {args.output}")
    for cat, bucket in sorted(stats.items()):
        c_total = bucket["kept"] + bucket["dropped"]
        print(
            f"  {cat}: {bucket['kept']}/{c_total} "
            f"({100.0 * bucket['kept'] / max(c_total, 1):.1f}% kept)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

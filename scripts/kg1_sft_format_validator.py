#!/usr/bin/env python3
"""Validate that SFT training data is in Kaggle-metric-optimal format.

Checks every row for:

- A single terminal ``\\boxed{...}`` with no trailing text.
- Boxed payload free of nested braces (``\\frac``, ``\\text``, ``\\sqrt``).
- No thousand separators or scientific notation in numeric answers.
- Family-specific rules:
    - bit_manipulation: exactly 8 bits, zero-padded.
    - text_encryption: lowercase ASCII + spaces.
    - equation_transform: ideally also emits ``Final answer is: X`` line.
    - gravity_constant/unit_conversion: plain decimal, no units.
- Prompt auto-detected family matches an explicit ``family`` column if present.
- Completion length within sane bounds (<= 8k characters).

Emits a JSON + CSV report listing every row that would benefit from
re-formatting and prints a summary table grouped by family.

Usage::

    python scripts/kg1_sft_format_validator.py \
        --input data/sft/v80_mega.jsonl \
        --report-json runs/sft_format_report.json \
        --report-csv runs/sft_format_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kg1_canonicalize_output import (  # noqa: E402
    canonicalize_answer,
    detect_family,
    _extract_last_boxed_body,
    _LATEX_FRAC_RE,
    _LATEX_TEXT_RE,
    _LATEX_MATHRM_RE,
    _LATEX_SQRT_RE,
    _SCI_RE,
)

# Hand-picked compatible issue codes — stable so downstream dashboards can
# aggregate across runs.
ISSUE_CODES = (
    "no_boxed",
    "multiple_boxed",
    "nested_braces",
    "latex_frac",
    "latex_text",
    "latex_mathrm",
    "latex_sqrt",
    "thousand_separator",
    "scientific_notation",
    "trailing_dot_zero",
    "bit_length_wrong",
    "cipher_uppercase",
    "cipher_non_ascii",
    "equation_no_final_answer_line",
    "unit_suffix",
    "trailing_punctuation",
    "completion_too_long",
    "family_mismatch",
    "empty_answer",
)

_BOXED_COUNT_RE = re.compile(r"\\boxed\s*\{")
_UNIT_LIKE_RE = re.compile(r"\b(kg|g|m|cm|km|s|ms|N|J|W|Pa|Hz|A|V|m/s|m/s\^2)\b", re.IGNORECASE)


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from either a JSONL or CSV file."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    else:
        raise ValueError(f"Unsupported SFT file format: {path.suffix}")


def _get_prompt(row: dict[str, Any]) -> str:
    """Best-effort prompt extraction across common SFT schemas."""
    for key in ("prompt", "question", "input", "instruction", "user"):
        if row.get(key):
            return str(row[key])
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
                return str(msg.get("content", ""))
    return ""


def _get_completion(row: dict[str, Any]) -> str:
    """Best-effort completion extraction across common SFT schemas."""
    for key in ("completion", "answer", "response", "output", "assistant", "target"):
        if row.get(key):
            return str(row[key])
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in {"assistant", "gpt", "bot"}:
                return str(msg.get("content", ""))
    return ""


def _get_family(row: dict[str, Any]) -> Optional[str]:
    for key in ("family", "category", "puzzle_family"):
        if row.get(key):
            return str(row[key])
    return None


def _check_row(row: dict[str, Any], max_completion_chars: int) -> dict[str, Any]:
    """Return per-row analysis including issue codes and suggested fix."""
    prompt = _get_prompt(row)
    completion = _get_completion(row)
    explicit_family = _get_family(row)
    detected_family = detect_family(prompt)
    family = detected_family or explicit_family or "unknown"

    issues: list[str] = []

    if not completion.strip():
        issues.append("empty_answer")

    boxed_count = len(_BOXED_COUNT_RE.findall(completion))
    if boxed_count == 0:
        issues.append("no_boxed")
    elif boxed_count > 1:
        issues.append("multiple_boxed")

    body = _extract_last_boxed_body(completion) or ""
    if body != body.strip():
        body = body.strip()

    if "{" in body or "}" in body:
        issues.append("nested_braces")

    if _LATEX_FRAC_RE.search(body):
        issues.append("latex_frac")
    if _LATEX_TEXT_RE.search(body):
        issues.append("latex_text")
    if _LATEX_MATHRM_RE.search(body):
        issues.append("latex_mathrm")
    if _LATEX_SQRT_RE.search(body):
        issues.append("latex_sqrt")
    if _SCI_RE.search(body):
        issues.append("scientific_notation")

    if re.search(r"\d,\d{3}", body) or re.search(r"\d_\d{3}", body):
        issues.append("thousand_separator")

    if re.fullmatch(r"-?\d+\.0+", body.strip()):
        issues.append("trailing_dot_zero")

    if _UNIT_LIKE_RE.search(body):
        issues.append("unit_suffix")

    if family == "bit_manipulation" and not re.fullmatch(r"[01]{8}", body.strip()):
        issues.append("bit_length_wrong")

    if family in {"text_encryption", "cipher"}:
        if body != body.lower():
            issues.append("cipher_uppercase")
        if any(ord(ch) > 127 for ch in body):
            issues.append("cipher_non_ascii")

    if family == "equation_transform":
        if "final answer is:" not in completion.lower():
            issues.append("equation_no_final_answer_line")

    if family in {"numeral_system", "text_encryption"}:
        if body.strip().endswith((".", ",", ";", ":")):
            issues.append("trailing_punctuation")

    if len(completion) > max_completion_chars:
        issues.append("completion_too_long")

    if explicit_family and detected_family and explicit_family != detected_family:
        issues.append("family_mismatch")

    suggested_fix = canonicalize_answer(completion, family_hint=family) if issues else completion

    return {
        "id": row.get("id") or row.get("row_id") or "",
        "family": family,
        "detected_family": detected_family,
        "explicit_family": explicit_family,
        "issues": issues,
        "body": body,
        "suggested_fix": suggested_fix,
        "completion_length": len(completion),
    }


def validate_sft(
    path: Path,
    max_completion_chars: int = 8192,
) -> dict[str, Any]:
    """Run validation over every row of ``path`` and return an aggregated report."""
    per_row: list[dict[str, Any]] = []
    family_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "clean": 0, "issues": defaultdict(int)}
    )
    issue_totals: dict[str, int] = defaultdict(int)

    for row in _iter_rows(path):
        analysis = _check_row(row, max_completion_chars=max_completion_chars)
        per_row.append(analysis)
        bucket = family_stats[analysis["family"]]
        bucket["total"] += 1
        if not analysis["issues"]:
            bucket["clean"] += 1
        for code in analysis["issues"]:
            bucket["issues"][code] += 1
            issue_totals[code] += 1

    total = len(per_row)
    summary_by_family = {
        fam: {
            "total": stats["total"],
            "clean": stats["clean"],
            "clean_rate": (stats["clean"] / stats["total"]) if stats["total"] else 0.0,
            "issues": dict(stats["issues"]),
        }
        for fam, stats in sorted(family_stats.items())
    }

    overall_clean = sum(stats["clean"] for stats in family_stats.values())
    return {
        "total_rows": total,
        "clean_rows": overall_clean,
        "clean_rate": overall_clean / total if total else 0.0,
        "issue_totals": dict(sorted(issue_totals.items())),
        "by_family": summary_by_family,
        "per_row": per_row,
    }


def _write_error_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "family", "issues", "body", "suggested_fix", "completion_length"],
        )
        writer.writeheader()
        for row in report["per_row"]:
            if not row["issues"]:
                continue
            writer.writerow({
                "id": row["id"],
                "family": row["family"],
                "issues": ",".join(row["issues"]),
                "body": row["body"],
                "suggested_fix": row["suggested_fix"],
                "completion_length": row["completion_length"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL or CSV SFT file")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--max-completion-chars", type=int, default=8192)
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_sft(args.input, max_completion_chars=args.max_completion_chars)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        # Drop per_row detail from the JSON to keep it small; keep summary.
        report_small = {k: v for k, v in report.items() if k != "per_row"}
        args.report_json.write_text(json.dumps(report_small, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.report_csv:
        _write_error_csv(report, args.report_csv)

    print(f"rows: {report['total_rows']}")
    print(f"clean_rate: {report['clean_rate']:.4f}")
    for fam, stats in report["by_family"].items():
        print(f"  {fam}: {stats['clean']}/{stats['total']} clean ({stats['clean_rate']:.4f})")
    print("issue_totals:")
    for code, count in report["issue_totals"].items():
        print(f"  {code}: {count}")

    if args.fail_on_issues and report["clean_rate"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

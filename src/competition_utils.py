"""Shared metric utilities for the NVIDIA Nemotron reasoning challenge.

The answer extraction and verification functions intentionally mirror the
public Kaggle metric path used by the Jiazhuang/Xduan local-CV notebooks:
extract the last boxed answer first, then fall back to final-answer phrases,
then the last number, then the last non-empty line.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"

OFFICIAL_INFERENCE_CONFIG: dict[str, Any] = {
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "max_lora_rank": 32,
    "max_tokens": 7680,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "trust_remote_code": True,
    "dtype": "auto",
}

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

FAMILIES = (
    "gravity_constant",
    "unit_conversion",
    "numeral_system",
    "text_encryption",
    "bit_manipulation",
    "equation_transform",
)

FAMILY_ALIASES = {
    "gravity": "gravity_constant",
    "grav": "gravity_constant",
    "gravity_constant": "gravity_constant",
    "unit": "unit_conversion",
    "units": "unit_conversion",
    "unit_conversion": "unit_conversion",
    "numeral": "numeral_system",
    "roman": "numeral_system",
    "roman_numeral": "numeral_system",
    "number_system": "numeral_system",
    "numeral_system": "numeral_system",
    "cipher": "text_encryption",
    "encryption": "text_encryption",
    "text": "text_encryption",
    "text_cipher": "text_encryption",
    "text_encryption": "text_encryption",
    "bit": "bit_manipulation",
    "bits": "bit_manipulation",
    "bit_manipulation": "bit_manipulation",
    "eq": "equation_transform",
    "equation": "equation_transform",
    "equation_rules": "equation_transform",
    "symbol_transform": "equation_transform",
    "equation_symbolic": "equation_transform",
    "equation_numeric": "equation_transform",
    "equation_numeric_deduce": "equation_transform",
    "equation_numeric_guess": "equation_transform",
    "cryptarithm_deduce": "equation_transform",
    "cryptarithm_guess": "equation_transform",
    "equation_transform": "equation_transform",
}


def _normalize_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[\s\-]+", "_", text)


def canonical_family(value: object) -> str:
    key = _normalize_key(value)
    return FAMILY_ALIASES.get(key, key or "unknown")


def classify_puzzle(prompt: str) -> str:
    low = str(prompt or "").lower()
    if "bit manipulation" in low or "8-bit binary" in low:
        return "bit_manipulation"
    if "encryption" in low or "decrypt the following text" in low or "cipher" in low:
        return "text_encryption"
    if "numeral system" in low or "converted into a different numeral" in low:
        return "numeral_system"
    if "gravitational" in low or "gravity" in low:
        return "gravity_constant"
    if "transformation rule" in low or "transformation rules" in low:
        return "equation_transform"
    if "unit conversion" in low or "measurement" in low:
        return "unit_conversion"
    return "unknown"


def extract_boxed_answers(text: str | None) -> list[str]:
    if text is None:
        return []
    value = str(text)
    starts = list(re.finditer(r"\\boxed\{", value))
    matches: list[str] = []
    for index, match in enumerate(starts):
        start = match.end()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
        segment = value[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    return matches


def extract_final_answer(text: str | None) -> str:
    """Extract the final answer with the public Kaggle fallback order."""

    if text is None:
        return "NOT_FOUND"
    value = str(text)

    matches = extract_boxed_answers(value)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:\uFF1A]\s*([^\n]+)",
        r"final answer\s*[:\uFF1A]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, value, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", value)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify_answer(stored_answer: object, predicted: object) -> bool:
    """Verify a prediction with the public Kaggle metric behavior."""

    expected = str(stored_answer).strip()
    observed = str(predicted).strip()
    if re.fullmatch(r"[01]+", expected):
        return observed.lower() == expected.lower()
    try:
        return math.isclose(float(expected), float(observed), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return observed.lower() == expected.lower()


def canonical_answer(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def escape_boxed_answer(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def unescape_latex_braces(value: object) -> str:
    return str(value).replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")


def canonical_boxed_payload(value: object) -> str:
    return canonical_answer(unescape_latex_braces(value))


def parse_finite_number(value: object) -> float | None:
    text = canonical_answer(value).replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def answers_equivalent(
    expected: object,
    observed: object,
    *,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-5,
    observed_is_boxed_payload: bool = False,
) -> bool:
    expected_text = canonical_answer(expected)
    observed_text = canonical_boxed_payload(observed) if observed_is_boxed_payload else canonical_answer(observed)
    expected_number = parse_finite_number(expected_text)
    observed_number = parse_finite_number(observed_text)
    if expected_number is not None and observed_number is not None:
        return math.isclose(expected_number, observed_number, rel_tol=rel_tol, abs_tol=abs_tol)
    return expected_text.lower() == observed_text.lower()


def box_answer(value: object) -> str:
    return f"\\boxed{{{escape_boxed_answer(value)}}}"

"""Shared metric utilities for the NVIDIA Nemotron reasoning challenge.

The score-facing answer extraction and verification functions intentionally
mirror the downloaded public Kaggle metric notebook. Diagnostic helpers such
as strict boxed extraction remain stricter than the official metric and must
not replace ``extract_final_answer`` for score/promotion decisions.
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


def ensure_prompt_suffix(prompt: object, suffix: str = PROMPT_SUFFIX) -> str:
    """Append the scoring prompt suffix exactly when it is absent."""

    text = str(prompt or "")
    if not suffix:
        return text
    if suffix.strip() in text:
        return text
    return text + suffix


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


def _extend_compact_symbolic_box_end(value: str, start: int, end: int) -> int:
    r"""Extend a boxed payload across literal close-braces in symbol answers."""

    cursor = end + 1
    if cursor >= len(value) or value[cursor].isspace() or value[cursor] == "{":
        return end

    index = cursor
    last_close: int | None = None
    while index < len(value):
        char = value[index]
        if char.isspace() or char == "{":
            break
        if char == "\\":
            if index + 1 < len(value) and not value[index + 1].isalnum() and not value[index + 1].isspace():
                index += 2
                continue
            break
        if char == "}":
            last_close = index
        elif char.isalpha():
            break
        index += 1

    if last_close is None or last_close <= end:
        return end
    candidate = value[start:last_close]
    if len(candidate) <= 64 and not any(item.isalpha() for item in candidate):
        return last_close
    return end


def extract_boxed_answers(text: str | None) -> list[str]:
    if text is None:
        return []
    value = str(text)
    matches: list[str] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        marker_pos = value.find(marker, cursor)
        if marker_pos == -1:
            break
        start = marker_pos + len(marker)
        minimal_end: int | None = None
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "}":
                minimal_end = index
                break
            index += 1

        depth = 1
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth == 0:
            end = _extend_compact_symbolic_box_end(value, start, index)
            while end + 1 < len(value) and value[end + 1] == "}":
                end += 1
            while (
                end + 3 < len(value)
                and value[end + 1] == "\\"
                and not value[end + 2].isalnum()
                and not value[end + 2].isspace()
                and value[end + 3] == "}"
            ):
                end += 3
            matches.append(value[start:end])
            cursor = end + 1
        else:
            if minimal_end is not None:
                matches.append(value[start:minimal_end])
                cursor = minimal_end + 1
            else:
                matches.append(value[start:])
                break
    return matches


def extract_closed_boxed_answers(text: str | None) -> list[str]:
    if text is None:
        return []
    value = str(text)
    matches: list[str] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        marker_pos = value.find(marker, cursor)
        if marker_pos == -1:
            break
        start = marker_pos + len(marker)
        minimal_end: int | None = None
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "}":
                minimal_end = index
                break
            index += 1

        depth = 1
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth == 0:
            end = _extend_compact_symbolic_box_end(value, start, index)
            while end + 1 < len(value) and value[end + 1] == "}":
                end += 1
            while (
                end + 3 < len(value)
                and value[end + 1] == "\\"
                and not value[end + 2].isalnum()
                and not value[end + 2].isspace()
                and value[end + 3] == "}"
            ):
                end += 3
            matches.append(value[start:end])
            cursor = end + 1
        elif minimal_end is not None:
            matches.append(value[start:minimal_end])
            cursor = minimal_end + 1
        else:
            break
    return matches


def has_unclosed_boxed_answer(text: str | None) -> bool:
    if text is None:
        return False
    value = str(text)
    marker = r"\boxed{"
    cursor = 0
    while True:
        marker_pos = value.find(marker, cursor)
        if marker_pos == -1:
            return False
        start = marker_pos + len(marker)
        minimal_end: int | None = None
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "}":
                minimal_end = index
                break
            index += 1

        depth = 1
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth != 0 and minimal_end is None:
            return True
        cursor = (index + 1) if depth == 0 else (minimal_end + 1 if minimal_end is not None else len(value))


def extract_final_boxed_answer(text: str | None) -> str:
    if has_unclosed_boxed_answer(text):
        return "NOT_FOUND"
    matches = extract_closed_boxed_answers(text)
    if not matches:
        return "NOT_FOUND"
    non_empty = [match.strip() for match in matches if match.strip()]
    if non_empty:
        return canonical_boxed_payload(non_empty[-1])
    return canonical_boxed_payload(matches[-1].strip())


def verify_strict_boxed_answer(expected: object, raw_output: object) -> bool:
    observed = extract_final_boxed_answer(str(raw_output or ""))
    if observed == "NOT_FOUND":
        return False
    expected_text = canonical_answer(expected)
    observed_text = canonical_answer(observed)
    if not expected_text or not observed_text:
        return False
    return expected_text.lower() == observed_text.lower()


def has_closed_boxed_answer(text: str | None) -> bool:
    if text is None:
        return False
    value = str(text)
    marker = r"\boxed{"
    cursor = 0
    while True:
        marker_pos = value.find(marker, cursor)
        if marker_pos == -1:
            return False
        start = marker_pos + len(marker)
        index = start
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                index += 2
                continue
            if char == "}":
                return True
            index += 1
        cursor = start


def extract_final_answer(text: str | None) -> str:
    """Extract the final answer with the official Kaggle fallback order."""

    if text is None:
        return "NOT_FOUND"
    value = str(text)

    boxed_starts = list(re.finditer(r"\\boxed\{", value))
    matches: list[str] = []
    for i, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(value)
        segment = value[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
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

#!/usr/bin/env python3
"""Pre-paid job integration gate for KG1 HF/Kaggle executions.

This gate is stricter than a static syntax check. It verifies that a launcher,
local dataset artifacts, audit manifests, hashes, target text, and FinOps
guards agree before we start a paid or long-running job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402

BLOCKED_DATASET_MARKERS = {
    "v447_v446_trace_dataset": "V447 traces contain hypothesis_formed contradictions.",
    "v461_synthetic_numeric_probe_pack": "V461 contains a full-reference exact prompt/answer seed.",
    "v463_v462_synthetic_numeric_hard_negative_audit": "V463 depends on the quarantined V461/V462 route.",
    "v464_v463_numeric_multirule_dataset": "V464 contains rejected_candidate == answer contamination.",
    "v468_v464_symbol_fix_dataset": "V468 still contains a full-reference exact prompt/answer seed.",
}
BLOCKED_ADAPTER_MARKERS = {
    "kg1-nemotron-lora-v448-nemo-h200-v447-clean-trace-v290ckpt6": "Adapter trained from quarantined V447 traces.",
    "kg1-nemotron-lora-v465-v464-numeric-multirule-v290ckpt6": "Adapter trained from quarantined V464 traces.",
    "kg1-nemotron-lora-v469-v468-symbol-fix-v290ckpt6": "Adapter trained from quarantined V468 traces.",
    "kg1-nemotron-lora-v499-nemo-h200-v498-numeric-teacher-v290ckpt6": (
        "V499 final eval regressed and answer-span weighting was inactive; forensics only."
    ),
    "kg1-nemotron-lora-v501-nemo-h200-v498-answer-span-v290ckpt6": (
        "V501 answer-span run was blocked by final eval regression; forensics only."
    ),
}

RESIDUAL_FIRST_MIN_TOTAL = 200
RESIDUAL_FIRST_MIN_BIT = 136
RESIDUAL_FIRST_MIN_EQUATION = 59
RESIDUAL_FIRST_MIN_COVERAGE = 0.70
PROTECTED_ROW_EXPECTED = ["8740ed31=01101000", "59bee375=10010101"]


@dataclass
class Finding:
    level: str
    code: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def require_text(text: str, snippet: str, code: str, findings: list[Finding]) -> None:
    if snippet not in text:
        findings.append(Finding("error", code, f"missing snippet: {snippet}"))


def require_regex(text: str, pattern: str, code: str, findings: list[Finding]) -> None:
    if not re.search(pattern, text, re.MULTILINE):
        findings.append(Finding("error", code, f"missing pattern: {pattern}"))


def parse_launcher_env_value(text: str, name: str) -> str:
    """Best-effort parser for launcher constants/env literals used by KG1 launchers."""
    patterns = [
        rf"{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']",
        rf"[\"']{re.escape(name)}[\"']\s*:\s*[\"']([^\"']+)[\"']",
        rf"export\s+{re.escape(name)}=([^\s\"']+)",
        rf"export\s+{re.escape(name)}=[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return str(match.group(1)).strip()
    return ""


def parse_launcher_env_float(text: str, name: str) -> float | None:
    value = parse_launcher_env_value(text, name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def audit_residual_first_gpu_gate(text: str, findings: list[Finding]) -> dict[str, Any]:
    """Require V540/V541 CPU evidence before any new paid training launcher."""
    required_exact = {
        "KG1_RESIDUAL_FIRST_GATE": "1",
        "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
        "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
        "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
        "KG1_V541_MISSMAP_GATE_STATUS": "passed",
        "KG1_V541_FLIP_LEDGER_STATUS": "passed",
        "KG1_EXPECTED_TRUNCATED": "0",
        "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
        "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
        "KG1_WEAK_LABEL_AWARE_SELECTION": "0",
        "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
    }
    observed: dict[str, Any] = {}
    for name, expected in required_exact.items():
        value = parse_launcher_env_value(text, name)
        observed[name] = value
        if value.lower() != expected.lower():
            findings.append(
                Finding(
                    "error",
                    "residual_first_gate_env_mismatch",
                    f"{name} expected {expected!r}, got {value or '<missing>'!r}",
                )
            )

    protected = parse_launcher_env_value(text, "KG1_PROTECTED_ID_ANSWERS")
    observed["KG1_PROTECTED_ID_ANSWERS"] = protected
    missing_protected = [item for item in PROTECTED_ROW_EXPECTED if item not in protected]
    if missing_protected:
        findings.append(
            Finding(
                "error",
                "residual_first_missing_protected_row_guard",
                f"KG1_PROTECTED_ID_ANSWERS must include {', '.join(PROTECTED_ROW_EXPECTED)}; missing {', '.join(missing_protected)}",
            )
        )

    numeric_thresholds = {
        "KG1_CPU_SIMULATED_TOTAL_CORRECT": RESIDUAL_FIRST_MIN_TOTAL,
        "KG1_CPU_SIMULATED_BIT_CORRECT": RESIDUAL_FIRST_MIN_BIT,
        "KG1_CPU_SIMULATED_EQUATION_CORRECT": RESIDUAL_FIRST_MIN_EQUATION,
        "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": RESIDUAL_FIRST_MIN_COVERAGE,
    }
    for name, threshold in numeric_thresholds.items():
        value = parse_launcher_env_float(text, name)
        observed[name] = value
        if value is None:
            findings.append(Finding("error", "residual_first_numeric_gate_missing", f"{name} is missing or non-numeric"))
            continue
        if value < float(threshold):
            findings.append(
                Finding(
                    "error",
                    "residual_first_numeric_gate_failed",
                    f"{name}={value} below required {threshold}",
                )
            )

    numeric_max_thresholds = {
        "KG1_CPU_SIMULATED_LOST_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_BIT_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": 0.0,
        "KG1_MAX_TOKEN_HEADROOM_RATIO": 0.90,
    }
    for name, threshold in numeric_max_thresholds.items():
        value = parse_launcher_env_float(text, name)
        observed[name] = value
        if value is None:
            findings.append(Finding("error", "residual_first_numeric_gate_missing", f"{name} is missing or non-numeric"))
            continue
        if value > float(threshold):
            findings.append(
                Finding(
                    "error",
                    "residual_first_numeric_max_gate_failed",
                    f"{name}={value} above required maximum {threshold}",
                )
            )

    return {
        "required": {
            "v540_extraction_gate_status": "passed",
            "cpu_extractor_parity_status": "passed",
            "prompt_template_parity_status": "passed",
            "v541_missmap_gate_status": "passed",
            "v541_flip_ledger_status": "passed",
            "cpu_simulated_total_min": RESIDUAL_FIRST_MIN_TOTAL,
            "cpu_simulated_bit_min": RESIDUAL_FIRST_MIN_BIT,
            "cpu_simulated_equation_min": RESIDUAL_FIRST_MIN_EQUATION,
            "cpu_miss_classification_coverage_min": RESIDUAL_FIRST_MIN_COVERAGE,
            "cpu_simulated_lost_rows_max": 0,
            "cpu_simulated_lost_bit_rows_max": 0,
            "cpu_simulated_lost_equation_rows_max": 0,
            "max_token_headroom_ratio_max": 0.90,
            "expected_truncated": 0,
            "protected_rows": PROTECTED_ROW_EXPECTED,
            "weak_label_aware_selection": "0",
            "cpu_simulation_uses_weak_labels": "0",
            "v536_val_stats_as_weak_evidence": "0",
        },
        "observed": observed,
    }


def audit_decoding_vs_adapter_drift_gate(text: str, findings: list[Finding]) -> dict[str, Any]:
    """Require the V568 decoding-vs-adapter drift evidence before paid training."""

    status_value = parse_launcher_env_value(text, "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS")
    if status_value.lower() == "deferred_post_checkpoint":
        observed = {
            "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": status_value,
            "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": parse_launcher_env_value(
                text, "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT"
            ),
            "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": parse_launcher_env_value(
                text, "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED"
            ),
            "KG1_EXPECTED_MAX_STEPS": parse_launcher_env_float(text, "KG1_EXPECTED_MAX_STEPS"),
            "MAX_STEPS": parse_launcher_env_float(text, "MAX_STEPS"),
            "SAVE_EVERY_STEPS": parse_launcher_env_float(text, "SAVE_EVERY_STEPS"),
            "EVAL_EVERY_STEPS": parse_launcher_env_float(text, "EVAL_EVERY_STEPS"),
        }
        allow_defer = str(observed["KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT"]).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        weak_required = str(observed["KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED"]).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        max_steps = observed["MAX_STEPS"] or observed["KG1_EXPECTED_MAX_STEPS"]
        if not allow_defer:
            findings.append(
                Finding(
                    "error",
                    "decoding_vs_adapter_drift_defer_not_allowed",
                    "deferred_post_checkpoint requires KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT=1",
                )
            )
        if not weak_required:
            findings.append(
                Finding(
                    "error",
                    "decoding_vs_adapter_drift_defer_without_weak_eval",
                    "deferred_post_checkpoint requires KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED=1",
                )
            )
        if max_steps is None or max_steps > 2:
            findings.append(
                Finding(
                    "error",
                    "decoding_vs_adapter_drift_defer_too_many_steps",
                    f"deferred_post_checkpoint requires MAX_STEPS<=2, got {max_steps}",
                )
            )
        for name in ("SAVE_EVERY_STEPS", "EVAL_EVERY_STEPS"):
            value = observed[name]
            if value is None or value > 2:
                findings.append(
                    Finding(
                        "error",
                        "decoding_vs_adapter_drift_defer_late_checkpoint",
                        f"deferred_post_checkpoint requires {name}<=2, got {value}",
                    )
                )
        return {
            "required": {
                "mode": "deferred_post_checkpoint",
                "max_steps_lte": 2,
                "first_checkpoint_weak_eval_required": True,
                "purpose": "allow one tiny smoke when V568 can only be measured after a new checkpoint exists",
            },
            "observed": observed,
        }

    required_exact = {
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "passed",
        "KG1_V568_LOGITS_NLL_GATE_STATUS": "passed",
        "KG1_V568_PROTECTED_MARGIN_STATUS": "passed",
    }
    observed: dict[str, Any] = {}
    for name, expected in required_exact.items():
        value = parse_launcher_env_value(text, name)
        observed[name] = value
        if value.lower() != expected:
            findings.append(
                Finding(
                    "error",
                    "decoding_vs_adapter_drift_env_mismatch",
                    f"{name} expected {expected!r}, got {value or '<missing>'!r}",
                )
            )

    numeric_values = {
        "KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION": parse_launcher_env_float(
            text, "KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION"
        ),
        "KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION": parse_launcher_env_float(
            text, "KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION"
        ),
        "KG1_V568_MISSING_LOGPROB_ROWS": parse_launcher_env_float(text, "KG1_V568_MISSING_LOGPROB_ROWS"),
        "KG1_V568_PROTECTED_ROWS_CHECKED": parse_launcher_env_float(text, "KG1_V568_PROTECTED_ROWS_CHECKED"),
    }
    observed.update(numeric_values)
    for name, value in numeric_values.items():
        if value is None:
            findings.append(Finding("error", "decoding_vs_adapter_drift_numeric_missing", f"{name} missing"))

    observed_regression = numeric_values["KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION"]
    allowed_regression = numeric_values["KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION"]
    if observed_regression is not None and allowed_regression is not None and observed_regression > allowed_regression:
        findings.append(
            Finding(
                "error",
                "decoding_vs_adapter_drift_margin_regression",
                f"observed_regression={observed_regression} > allowed={allowed_regression}",
            )
        )
    missing_logprob = numeric_values["KG1_V568_MISSING_LOGPROB_ROWS"]
    if missing_logprob is not None and missing_logprob != 0:
        findings.append(
            Finding(
                "error",
                "decoding_vs_adapter_drift_missing_logprobs",
                f"KG1_V568_MISSING_LOGPROB_ROWS={missing_logprob}",
            )
        )
    protected_checked = numeric_values["KG1_V568_PROTECTED_ROWS_CHECKED"]
    if protected_checked is not None and protected_checked < len(PROTECTED_ROW_EXPECTED):
        findings.append(
            Finding(
                "error",
                "decoding_vs_adapter_drift_incomplete_protected_rows",
                f"KG1_V568_PROTECTED_ROWS_CHECKED={protected_checked}",
            )
        )

    return {
        "required": {
            "statuses": required_exact,
            "max_observed_protected_margin_regression_lte_allowed": True,
            "missing_logprob_rows": 0,
            "protected_rows_min": len(PROTECTED_ROW_EXPECTED),
            "absolute_negative_margin": "warning_only_unless_hf_runtime_requires_strict",
        },
        "observed": observed,
    }


def block_quarantined_identity(text: str, findings: list[Finding], *, source: str) -> None:
    for marker, reason in BLOCKED_DATASET_MARKERS.items():
        if marker in text:
            findings.append(Finding("error", "quarantined_dataset_identity", f"{source}: {marker}: {reason}"))
    for marker, reason in BLOCKED_ADAPTER_MARKERS.items():
        if marker in text:
            findings.append(Finding("error", "quarantined_adapter_identity", f"{source}: {marker}: {reason}"))


def audit_launcher(args: argparse.Namespace, findings: list[Finding]) -> dict[str, Any]:
    launcher = args.launcher
    text = launcher.read_text(encoding="utf-8", errors="replace")
    block_quarantined_identity(text, findings, source=str(launcher))
    residual_first_report = (
        {"skipped": True, "reason": "explicitly_allowed_missing_residual_first_gates"}
        if args.allow_missing_residual_first_gates
        else audit_residual_first_gpu_gate(text, findings)
    )
    decoding_vs_adapter_drift_report = (
        {"skipped": True, "reason": "explicitly_allowed_missing_decoding_vs_adapter_drift_gate"}
        if args.allow_missing_decoding_drift_gate
        else audit_decoding_vs_adapter_drift_gate(text, findings)
    )
    if args.require_crisis_guards:
        require_regex(
            text,
            r"KG1_CRISIS_MODE_BACKFIRE_GUARD\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
            "launcher_missing_crisis_backfire_guard",
            findings,
        )
    if args.expected_data_repo:
        require_text(text, f'DATA_REPO = "{args.expected_data_repo}"', "launcher_data_repo_constant_mismatch", findings)
        command_export_forms = [
            f"export DATA_REPO='{args.expected_data_repo}'",
            'f"export DATA_REPO=\'{DATA_REPO}\'"',
            'f"export DATA_REPO=\'{base.DATA_REPO}\'"',
        ]
        if not any(snippet in text for snippet in command_export_forms):
            findings.append(
                Finding(
                    "error",
                    "launcher_command_data_repo_export_mismatch",
                    "missing command export for expected DATA_REPO",
                )
            )
    require_text(text, f'DATA_ROOT = "{args.expected_data_root}"', "launcher_data_root_mismatch", findings)
    require_text(text, f'PREF_TRAIN_SHA256 = "{args.expected_train_sha256}"', "launcher_train_sha_mismatch", findings)
    require_text(text, f'PREF_VAL_SHA256 = "{args.expected_val_sha256}"', "launcher_val_sha_mismatch", findings)
    require_text(text, f"PREF_TRAIN_ROWS = {args.expected_train_rows}", "launcher_train_rows_mismatch", findings)
    require_text(text, f"PREF_VAL_ROWS = {args.expected_val_rows}", "launcher_val_rows_mismatch", findings)
    require_text(text, f'OUTPUT_REPO = "{args.expected_output_repo}"', "launcher_output_repo_mismatch", findings)
    require_text(text, f'INIT_ADAPTER_REPO = "{args.expected_init_adapter_repo}"', "launcher_init_repo_mismatch", findings)
    require_text(
        text,
        f'INIT_ADAPTER_SUBFOLDER = "{args.expected_init_adapter_subfolder}"',
        "launcher_init_subfolder_mismatch",
        findings,
    )
    if args.dataset_schema == "preference":
        require_text(text, "export ALLOW_FORMAT_NEGATIVES=0", "launcher_allows_format_negatives", findings)
    declared_schema_match = re.search(r"KG1_DATASET_SCHEMA\s*[\"']?\s*:\s*[\"']([^\"']+)[\"']", text)
    declared_schema = declared_schema_match.group(1) if declared_schema_match else ""
    if declared_schema != args.dataset_schema:
        findings.append(
            Finding(
                "error",
                "launcher_dataset_schema_mismatch",
                f"launcher KG1_DATASET_SCHEMA={declared_schema or '<missing>'} cli_dataset_schema={args.dataset_schema}",
            )
        )
    require_text(text, "timeout=3600", "launcher_timeout_not_one_hour", findings)
    require_text(text, 'FLAVOR = "h200"', "launcher_not_h200", findings)
    require_text(text, 'KG1_HF_MAX_UNIT_COST_USD": "0.09"', "launcher_missing_cost_gate", findings)
    require_text(
        text,
        f"SAVE_EVERY_STEPS = {args.expected_save_every_steps}",
        "launcher_missing_first_checkpoint_save",
        findings,
    )
    require_text(
        text,
        f"EVAL_EVERY_STEPS = {args.expected_eval_every_steps}",
        "launcher_missing_first_checkpoint_eval",
        findings,
    )
    if args.expected_max_length:
        require_text(text, f"MAX_LENGTH = {args.expected_max_length}", "launcher_max_length_constant_mismatch", findings)
        max_length_export_forms = [
            f"export MAX_LENGTH={args.expected_max_length}",
            'f"export MAX_LENGTH={MAX_LENGTH}"',
            'f"export MAX_LENGTH={base.MAX_LENGTH}"',
        ]
        if not any(snippet in text for snippet in max_length_export_forms):
            findings.append(
                Finding(
                    "error",
                    "launcher_command_max_length_mismatch",
                    "missing command export for expected MAX_LENGTH",
                )
            )
        require_text(
            text,
            '"KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH)',
            "launcher_missing_expected_max_length_env",
            findings,
        )
    if args.expected_abort_max_reserved_gib:
        require_text(
            text,
            f"ABORT_MAX_RESERVED_GIB = {args.expected_abort_max_reserved_gib}",
            "launcher_abort_max_reserved_constant_mismatch",
            findings,
        )
        abort_export_forms = [
            f"export ABORT_MAX_RESERVED_GIB={args.expected_abort_max_reserved_gib}",
            'f"export ABORT_MAX_RESERVED_GIB={ABORT_MAX_RESERVED_GIB}"',
            'f"export ABORT_MAX_RESERVED_GIB={base.ABORT_MAX_RESERVED_GIB}"',
        ]
        if not any(snippet in text for snippet in abort_export_forms):
            findings.append(
                Finding(
                    "error",
                    "launcher_abort_max_reserved_export_mismatch",
                    "missing command export for expected ABORT_MAX_RESERVED_GIB",
                )
            )
    if args.expected_loss_normalization_mode:
        require_text(
            text,
            f'LOSS_NORMALIZATION_MODE = "{args.expected_loss_normalization_mode}"',
            "launcher_loss_normalization_mode_constant_mismatch",
            findings,
        )
        require_text(
            text,
            '"KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE',
            "launcher_missing_expected_loss_normalization_env",
            findings,
        )
        loss_export_forms = [
            f"export LOSS_NORMALIZATION_MODE={args.expected_loss_normalization_mode}",
            f"export LOSS_NORMALIZATION_MODE='{args.expected_loss_normalization_mode}'",
            f'export LOSS_NORMALIZATION_MODE="{args.expected_loss_normalization_mode}"',
            'export LOSS_NORMALIZATION_MODE="$KG1_LOSS_NORMALIZATION_MODE"',
        ]
        if not any(snippet in text for snippet in loss_export_forms):
            findings.append(
                Finding(
                    "error",
                    "launcher_loss_normalization_export_mismatch",
                    "missing command export for expected LOSS_NORMALIZATION_MODE",
                )
            )
    require_regex(text, r"MAX_STEPS\s*=\s*(?:[1-9]|1[0-2])\b", "launcher_max_steps_too_high", findings)
    if args.require_row_loss_weight:
        require_regex(
            text,
            r"USE_ROW_LOSS_WEIGHT\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
            "launcher_missing_row_loss_weight",
            findings,
        )
        require_regex(
            text,
            r"REQUIRE_ROW_LOSS_WEIGHT\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
            "launcher_missing_required_row_loss_weight",
            findings,
        )
    if args.expected_pair_score_mode:
        require_text(
            text,
            f"export PAIR_SCORE_MODE='{args.expected_pair_score_mode}'",
            "launcher_pair_score_mode_mismatch",
            findings,
        )
    if args.dataset_schema == "preference":
        require_text(
            text,
            "export PREFERENCE_SYSTEM_PROMPT='Solve the KG1 puzzle. End with exactly one final answer in \\boxed{}.'",
            "launcher_system_prompt_not_final_answer_only",
            findings,
        )
    else:
        require_text(text, "ANSWER_SPAN_LOSS_WEIGHT", "launcher_missing_answer_span_loss_control", findings)
        require_text(text, "ANSWER_SPAN_MIN_WEIGHTED_TOKENS", "launcher_missing_answer_span_min_token_gate", findings)
        weight_match = re.search(
            r"ANSWER_SPAN_LOSS_WEIGHT\s*=\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
            text,
        )
        min_tokens_match = re.search(
            r"ANSWER_SPAN_MIN_WEIGHTED_TOKENS\s*=\s*['\"]?([0-9]+)['\"]?",
            text,
        )
        if weight_match and min_tokens_match:
            answer_span_weight = float(weight_match.group(1))
            answer_span_min_tokens = int(min_tokens_match.group(1))
            if answer_span_weight <= 1.0 and answer_span_min_tokens > 0:
                findings.append(
                    Finding(
                        "error",
                        "launcher_answer_span_min_tokens_without_weighting",
                        "ANSWER_SPAN_MIN_WEIGHTED_TOKENS is positive but ANSWER_SPAN_LOSS_WEIGHT<=1.0; "
                        "this silently makes answer-span weighting inactive while the manifest looks gated.",
                    )
                )
            if answer_span_weight > 1.0 and answer_span_min_tokens <= 0:
                findings.append(
                    Finding(
                        "error",
                        "launcher_answer_span_weighting_without_min_token_floor",
                        "ANSWER_SPAN_LOSS_WEIGHT>1.0 requires a positive ANSWER_SPAN_MIN_WEIGHTED_TOKENS "
                        "floor so inactive answer-span weighting cannot pass pre-paid gates.",
                    )
                )
    require_text(text, "KG1_REQUIRED_TRAIN_FAMILIES", "launcher_missing_train_family_gate", findings)
    require_text(text, "KG1_REQUIRED_VAL_FAMILIES", "launcher_missing_val_family_gate", findings)
    require_text(text, "KG1_REQUIRED_TRAIN_SUBCATEGORIES", "launcher_missing_train_subcategory_gate", findings)
    require_text(text, "KG1_REQUIRED_VAL_SUBCATEGORIES", "launcher_missing_val_subcategory_gate", findings)
    blocked = [
        "data/v435e_adapter_probe_preference/20260515T_v435e_from_h200_probe",
        "7f5e11770ac09c15e695cf4690df2fe7b5985b4a8b5bf5f8a201e8b71fe8be81",
        "d66752ab8470e145744a8bf80bc9b8beab7a4a3479d9161d03d6cc61c8ff9d92",
    ]
    for marker in blocked:
        if marker in text:
            findings.append(Finding("error", "launcher_references_blocked_mixed_dataset", marker))
    return {
        "launcher": str(launcher),
        "expected_data_repo": args.expected_data_repo,
        "contains_h200": 'FLAVOR = "h200"' in text,
        "contains_timeout_3600": "timeout=3600" in text,
        "eval_prompt_requires_boxed_only_line": (
            "Return only one line" in text and "\\boxed" in text and "No reasoning" in text
        ),
        "expected_max_length": args.expected_max_length,
        "expected_abort_max_reserved_gib": args.expected_abort_max_reserved_gib,
        "expected_loss_normalization_mode": args.expected_loss_normalization_mode,
        "require_row_loss_weight": args.require_row_loss_weight,
        "declared_dataset_schema": declared_schema,
        "residual_first_gpu_gate": residual_first_report,
        "decoding_vs_adapter_drift_gate": decoding_vs_adapter_drift_report,
    }


def audit_tokenization_manifest(path: Path | None, args: argparse.Namespace, findings: list[Finding]) -> dict[str, Any]:
    if path is None:
        return {"skipped": True, "reason": "tokenization_manifest_not_provided"}
    manifest = read_json(path)
    if manifest.get("decision", {}).get("status") != "tokenization_gate_passed":
        findings.append(
            Finding("error", "tokenization_manifest_not_passed", str(manifest.get("decision", {}).get("status")))
        )
    tokenization = manifest.get("tokenization", {})
    train_max = int((tokenization.get("train") or {}).get("token_max", 0) or 0)
    val_max = int((tokenization.get("validation") or {}).get("token_max", 0) or 0)
    max_observed = max(train_max, val_max)
    manifest_max_length = int((manifest.get("config") or {}).get("max_length", 0) or 0)
    if args.expected_max_length and max_observed > args.expected_max_length:
        findings.append(
            Finding(
                "error",
                "runtime_max_length_below_tokenization_max",
                f"token_max={max_observed} expected_max_length={args.expected_max_length}",
            )
        )
    if args.expected_max_length and manifest_max_length and manifest_max_length < args.expected_max_length:
        findings.append(
            Finding(
                "error",
                "tokenization_gate_max_length_below_runtime",
                f"manifest_max_length={manifest_max_length} expected_max_length={args.expected_max_length}",
            )
        )
    return {
        "manifest": str(path),
        "status": manifest.get("decision", {}).get("status"),
        "manifest_max_length": manifest_max_length,
        "train_token_max": train_max,
        "validation_token_max": val_max,
        "runtime_expected_max_length": args.expected_max_length,
        "runtime_length_safe": (not args.expected_max_length) or max_observed <= args.expected_max_length,
    }


def _metadata_flag_is_false(row: dict[str, Any], metadata: dict[str, Any], flag: str) -> bool:
    if flag in row:
        return row.get(flag) is False
    if flag in metadata:
        return metadata.get(flag) is False
    return True


def percentile_int(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[int(ratio * (len(ordered) - 1))])


def audit_dataset_file(
    path: Path,
    expected_sha: str,
    expected_rows: int,
    split: str,
    findings: list[Finding],
    *,
    dataset_schema: str,
    max_assistant_chars_p95: int = 0,
    max_assistant_chars_max: int = 0,
) -> dict[str, Any]:
    block_quarantined_identity(str(path), findings, source=f"{split}_path")
    observed_sha = sha256_file(path)
    rows = read_jsonl(path)
    if observed_sha != expected_sha:
        findings.append(Finding("error", f"{split}_sha_mismatch", f"{observed_sha} != {expected_sha}"))
    if len(rows) != expected_rows:
        findings.append(Finding("error", f"{split}_row_count_mismatch", f"{len(rows)} != {expected_rows}"))
    ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()
    assistant_prefix_counts: Counter[str] = Counter()
    assistant_rule_prefix_rows = 0
    assistant_trace_rows = 0
    assistant_multiline_rows = 0
    assistant_final_answer_only_rows = 0
    assistant_boxed_only_rows = 0
    assistant_lengths_by_family: dict[str, list[int]] = {}
    bad_rows: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", ""))
        metadata = row.get("metadata") or {}
        if not row_id or row_id in ids:
            bad_rows.append(f"{index}:duplicate_or_missing_id:{row_id}")
        ids.add(row_id)
        family = str(row.get("family") or metadata.get("family") or "unknown")
        family_counts[family] += 1
        subcategory_counts[str(row.get("subcategory") or metadata.get("subcategory") or metadata.get("rule_class") or "unknown")] += 1
        messages = row.get("messages")
        if dataset_schema == "preference":
            chosen = str(row.get("chosen", ""))
            rejected = str(row.get("rejected", ""))
            negative_type = str(metadata.get("negative_type") or "unknown")
            negative_type_counts[negative_type] += 1
            if negative_type != "hard_negative_adapter_exact_wrong":
                bad_rows.append(f"{row_id}:negative_type:{negative_type}")
            if chosen == rejected:
                bad_rows.append(f"{row_id}:chosen_equals_rejected")
            if not chosen.startswith("Final answer: \\boxed{") or not chosen.endswith("}"):
                bad_rows.append(f"{row_id}:chosen_template")
            if not rejected.startswith("Final answer: \\boxed{") or not rejected.endswith("}"):
                bad_rows.append(f"{row_id}:rejected_template")
            if chosen.count("\\boxed{") != 1 or rejected.count("\\boxed{") != 1:
                bad_rows.append(f"{row_id}:box_count")
            for term in ("public-train label audit", "frozen adapter", "Rejected adapter"):
                if term in chosen:
                    bad_rows.append(f"{row_id}:chosen_forbidden_term:{term}")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                bad_rows.append(f"{row_id}:assistant_message_missing")
            elif messages[-1].get("content") != chosen:
                bad_rows.append(f"{row_id}:assistant_content_not_chosen")
        else:
            answer = str(row.get("answer", "")).strip()
            if not answer:
                bad_rows.append(f"{row_id}:missing_answer")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                bad_rows.append(f"{row_id}:assistant_message_missing")
            else:
                assistant_content = str(messages[-1].get("content", ""))
                assistant_stripped = assistant_content.strip()
                if assistant_stripped.startswith("RULE:"):
                    assistant_prefix_counts["RULE"] += 1
                    assistant_rule_prefix_rows += 1
                elif assistant_stripped.startswith("Final answer:"):
                    assistant_prefix_counts["Final answer"] += 1
                elif assistant_stripped.startswith("\\boxed{"):
                    assistant_prefix_counts["boxed"] += 1
                else:
                    assistant_prefix_counts["other"] += 1
                if re.fullmatch(r"Final answer:\s*\\boxed\{.*\}", assistant_stripped):
                    assistant_final_answer_only_rows += 1
                if re.fullmatch(r"\\boxed\{.*\}", assistant_stripped):
                    assistant_boxed_only_rows += 1
                if "Trace:" in assistant_content:
                    assistant_trace_rows += 1
                if "\n" in assistant_stripped:
                    assistant_multiline_rows += 1
                extracted = extract_final_answer(assistant_content)
                if not verify_answer(answer, extracted):
                    bad_rows.append(f"{row_id}:assistant_final_answer_mismatch:{extracted}")
                assistant_lengths_by_family.setdefault(family, []).append(len(assistant_content))
        for flag in ("gate_rows_used_for_training", "weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if not _metadata_flag_is_false(row, metadata, flag):
                bad_rows.append(f"{row_id}:{flag}_not_false")
        if len(bad_rows) >= 30:
            break
    assistant_length_stats: dict[str, dict[str, int]] = {}
    for family, lengths in sorted(assistant_lengths_by_family.items()):
        assistant_length_stats[family] = {
            "rows": len(lengths),
            "chars_p50": percentile_int(lengths, 0.50),
            "chars_p95": percentile_int(lengths, 0.95),
            "chars_max": max(lengths) if lengths else 0,
        }
    if dataset_schema == "sft" and max_assistant_chars_p95:
        offenders = {
            family: stats
            for family, stats in assistant_length_stats.items()
            if int(stats.get("chars_p95", 0)) > max_assistant_chars_p95
        }
        if offenders:
            findings.append(
                Finding(
                    "error",
                    f"{split}_assistant_chars_p95_too_high",
                    json.dumps({"max": max_assistant_chars_p95, "offenders": offenders}, sort_keys=True),
                )
            )
    if dataset_schema == "sft" and max_assistant_chars_max:
        offenders = {
            family: stats
            for family, stats in assistant_length_stats.items()
            if int(stats.get("chars_max", 0)) > max_assistant_chars_max
        }
        if offenders:
            findings.append(
                Finding(
                    "error",
                    f"{split}_assistant_chars_max_too_high",
                    json.dumps({"max": max_assistant_chars_max, "offenders": offenders}, sort_keys=True),
                )
            )
    if bad_rows:
        findings.append(Finding("error", f"{split}_dataset_content_invalid", json.dumps(bad_rows[:30], sort_keys=True)))
    return {
        "path": str(path),
        "sha256": observed_sha,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "assistant_prefix_counts": dict(sorted(assistant_prefix_counts.items())),
        "assistant_rule_prefix_rows": assistant_rule_prefix_rows,
        "assistant_trace_rows": assistant_trace_rows,
        "assistant_multiline_rows": assistant_multiline_rows,
        "assistant_final_answer_only_rows": assistant_final_answer_only_rows,
        "assistant_boxed_only_rows": assistant_boxed_only_rows,
        "assistant_length_stats": assistant_length_stats,
        "bad_rows_first30": bad_rows[:30],
    }


def audit_v438_manifest(path: Path, findings: list[Finding]) -> dict[str, Any]:
    manifest = read_json(path)
    total = manifest.get("total_summary") or {}
    flags = manifest.get("decision_flags") or {}
    required_zero = [
        "answer_box_mismatch_rows",
        "rejected_box_mismatch_rows",
        "chosen_mentions_adapter_prediction_rows",
        "chosen_mentions_public_train_label_audit_rows",
    ]
    for key in required_zero:
        if int(total.get(key, -1)) != 0:
            findings.append(Finding("error", "v438_audit_required_zero_failed", f"{key}={total.get(key)}"))
    if manifest.get("hf_gpu_allowed_for_same_objective") is not True:
        findings.append(Finding("error", "v438_audit_not_gpu_allowed", str(manifest.get("hf_gpu_allowed_for_same_objective"))))
    for key in ("answer_boxes_all_match", "rejected_boxes_all_match_adapter_prediction", "format_negatives_absent"):
        if flags.get(key) is not True:
            findings.append(Finding("error", "v438_audit_flag_false", f"{key}={flags.get(key)}"))
    if flags.get("chosen_leaks_adapter_wrong_answer_text_majority") is not False:
        findings.append(Finding("error", "v438_audit_chosen_leak_flag", str(flags.get("chosen_leaks_adapter_wrong_answer_text_majority"))))
    if flags.get("chosen_template_mentions_label_audit_majority") is not False:
        findings.append(Finding("error", "v438_audit_label_audit_flag", str(flags.get("chosen_template_mentions_label_audit_majority"))))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--launcher", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--v438-audit-manifest", type=Path, default=None)
    parser.add_argument("--dataset-schema", choices=["preference", "sft"], default="preference")
    parser.add_argument("--expected-save-every-steps", type=int, default=3)
    parser.add_argument("--expected-eval-every-steps", type=int, default=3)
    parser.add_argument("--expected-max-length", type=int, default=0)
    parser.add_argument("--expected-abort-max-reserved-gib", type=int, default=0)
    parser.add_argument("--expected-loss-normalization-mode", default="")
    parser.add_argument("--expected-eval-output-contract", choices=["", "one_line_boxed_no_reasoning"], default="")
    parser.add_argument("--max-assistant-chars-p95", type=int, default=0)
    parser.add_argument("--max-assistant-chars-max", type=int, default=0)
    parser.add_argument("--require-row-loss-weight", action="store_true")
    parser.add_argument("--tokenization-manifest-json", type=Path, default=None)
    parser.add_argument("--expected-data-repo", default="")
    parser.add_argument("--expected-data-root", default="")
    parser.add_argument("--expected-train-sha256", default="")
    parser.add_argument("--expected-val-sha256", default="")
    parser.add_argument("--expected-train-rows", type=int, default=0)
    parser.add_argument("--expected-val-rows", type=int, default=0)
    parser.add_argument("--expected-output-repo", default="")
    parser.add_argument("--expected-init-adapter-repo", default="")
    parser.add_argument("--expected-init-adapter-subfolder", default="")
    parser.add_argument("--expected-pair-score-mode", default="")
    parser.add_argument("--allow-missing-crisis-guards", action="store_true")
    parser.add_argument("--allow-missing-residual-first-gates", action="store_true")
    parser.add_argument("--allow-missing-decoding-drift-gate", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.self_test:
        required_values = {
            "--launcher": args.launcher,
            "--train-jsonl": args.train_jsonl,
            "--val-jsonl": args.val_jsonl,
            "--expected-data-root": args.expected_data_root,
            "--expected-train-sha256": args.expected_train_sha256,
            "--expected-val-sha256": args.expected_val_sha256,
            "--expected-train-rows": args.expected_train_rows,
            "--expected-val-rows": args.expected_val_rows,
            "--expected-output-repo": args.expected_output_repo,
            "--expected-init-adapter-repo": args.expected_init_adapter_repo,
            "--expected-init-adapter-subfolder": args.expected_init_adapter_subfolder,
        }
        missing = [name for name, value in required_values.items() if value in (None, "", 0)]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
    args.require_crisis_guards = not args.allow_missing_crisis_guards
    return args


def run_gate(args: argparse.Namespace, *, emit: bool = True) -> dict[str, Any]:
    findings: list[Finding] = []
    if emit:
        print("=== KG1 PRE PAID JOB INTEGRATION GATE START ===", flush=True)
    launcher_report = audit_launcher(args, findings)
    train_report = audit_dataset_file(
        args.train_jsonl,
        args.expected_train_sha256,
        args.expected_train_rows,
        "train",
        findings,
        dataset_schema=args.dataset_schema,
        max_assistant_chars_p95=args.max_assistant_chars_p95,
        max_assistant_chars_max=args.max_assistant_chars_max,
    )
    val_report = audit_dataset_file(
        args.val_jsonl,
        args.expected_val_sha256,
        args.expected_val_rows,
        "validation",
        findings,
        dataset_schema=args.dataset_schema,
        max_assistant_chars_p95=args.max_assistant_chars_p95,
        max_assistant_chars_max=args.max_assistant_chars_max,
    )
    boxed_only_contract = (
        args.expected_eval_output_contract == "one_line_boxed_no_reasoning"
        or bool(launcher_report.get("eval_prompt_requires_boxed_only_line"))
    )
    if args.dataset_schema == "sft" and boxed_only_contract:
        train_rows = int(train_report.get("rows", 0) or 0)
        val_rows = int(val_report.get("rows", 0) or 0)
        train_compatible_rows = int(train_report.get("assistant_final_answer_only_rows", 0) or 0) + int(
            train_report.get("assistant_boxed_only_rows", 0) or 0
        )
        val_compatible_rows = int(val_report.get("assistant_final_answer_only_rows", 0) or 0) + int(
            val_report.get("assistant_boxed_only_rows", 0) or 0
        )
        if int(train_report.get("assistant_rule_prefix_rows", 0) or 0) > 0:
            findings.append(
                Finding(
                    "error",
                    "sft_target_prefix_incompatible_with_eval_contract",
                    "launcher asks for one-line boxed output but train dataset contains assistant targets starting with RULE:",
                )
            )
        if int(train_report.get("assistant_trace_rows", 0) or 0) > 0 or int(
            val_report.get("assistant_trace_rows", 0) or 0
        ) > 0:
            findings.append(
                Finding(
                    "error",
                    "sft_trace_targets_incompatible_with_eval_contract",
                    "launcher asks for one-line boxed output but dataset contains Trace: assistant targets",
                )
            )
        if int(train_report.get("assistant_multiline_rows", 0) or 0) > 0 or int(
            val_report.get("assistant_multiline_rows", 0) or 0
        ) > 0:
            findings.append(
                Finding(
                    "error",
                    "sft_multiline_targets_incompatible_with_eval_contract",
                    "launcher asks for one-line boxed output but dataset contains multi-line assistant targets",
                )
            )
        if train_compatible_rows != train_rows or val_compatible_rows != val_rows:
            findings.append(
                Finding(
                    "error",
                    "sft_not_all_targets_match_eval_contract",
                    json.dumps(
                        {
                            "train_compatible_rows": train_compatible_rows,
                            "train_rows": train_rows,
                            "validation_compatible_rows": val_compatible_rows,
                            "validation_rows": val_rows,
                        },
                        sort_keys=True,
                    ),
                )
            )
        if (
            int(train_report.get("assistant_final_answer_only_rows", 0) or 0) == 0
            and int(train_report.get("assistant_boxed_only_rows", 0) or 0) == 0
        ):
            findings.append(
                Finding(
                    "error",
                    "sft_missing_eval_compatible_assistant_targets",
                    "launcher asks for one-line boxed output but train dataset has no boxed-only/final-answer-only targets",
                )
            )
    if args.dataset_schema == "preference":
        if args.v438_audit_manifest is None:
            findings.append(Finding("error", "v438_audit_manifest_missing", "preference schema requires --v438-audit-manifest"))
            v438_report = {}
        else:
            v438_report = audit_v438_manifest(args.v438_audit_manifest, findings)
    else:
        v438_report = {}
    v438_summary = (
        {
            "manifest": str(args.v438_audit_manifest),
            "rows": v438_report.get("rows"),
            "hf_gpu_allowed_for_same_objective": v438_report.get("hf_gpu_allowed_for_same_objective"),
            "total_summary": v438_report.get("total_summary"),
            "decision_flags": v438_report.get("decision_flags"),
        }
        if args.dataset_schema == "preference"
        else {"skipped": True, "reason": "sft_schema_does_not_use_v438_preference_audit"}
    )
    tokenization_summary = audit_tokenization_manifest(args.tokenization_manifest_json, args, findings)
    report = {
        "schema_version": "kg1_pre_paid_job_integration_gate_v2",
        "dataset_schema": args.dataset_schema,
        "expected_eval_output_contract": args.expected_eval_output_contract,
        "ok": not any(item.level == "error" for item in findings),
        "launcher": launcher_report,
        "train_dataset": train_report,
        "validation_dataset": val_report,
        "tokenization_manifest": tokenization_summary,
        "v438_audit": v438_summary,
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if emit:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print("=== KG1 PRE PAID JOB INTEGRATION GATE END ===", flush=True)
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _self_test_sft_rows(*, rule_prefix: bool = False, trace_prefix: bool = False) -> list[dict[str, Any]]:
    assistant_bit = "Final answer: \\boxed{01101000}"
    assistant_equation = "Final answer: \\boxed{42}"
    if rule_prefix:
        assistant_bit = "RULE: protected row must stay stable\n" + assistant_bit
    if trace_prefix:
        assistant_bit = "Trace: protected row must stay stable\n" + assistant_bit
    return [
        {
            "id": "8740ed31",
            "family": "bit_manipulation",
            "subcategory": "bit_pair_stride",
            "prompt": "self-test bit prompt",
            "answer": "01101000",
            "messages": [
                {"role": "user", "content": "self-test bit prompt"},
                {"role": "assistant", "content": assistant_bit},
            ],
            "metadata": {
                "family": "bit_manipulation",
                "subcategory": "bit_pair_stride",
                "gate_rows_used_for_training": False,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
            },
        },
        {
            "id": "518deb39",
            "family": "equation_transform",
            "subcategory": "equation_symbolic",
            "prompt": "self-test equation prompt",
            "answer": "42",
            "messages": [
                {"role": "user", "content": "self-test equation prompt"},
                {"role": "assistant", "content": assistant_equation},
            ],
            "metadata": {
                "family": "equation_transform",
                "subcategory": "equation_symbolic",
                "gate_rows_used_for_training": False,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
            },
        },
    ]


def _self_test_launcher_text(train_sha: str, val_sha: str, *, protected: str | None = None) -> str:
    protected_value = protected or "8740ed31=01101000,59bee375=10010101"
    return f'''
DATA_REPO = "kg1/self-test-data"
DATA_ROOT = "data/self-test"
PREF_TRAIN_SHA256 = "{train_sha}"
PREF_VAL_SHA256 = "{val_sha}"
PREF_TRAIN_ROWS = 2
PREF_VAL_ROWS = 2
OUTPUT_REPO = "felipesp1983/kg1-self-test-output"
INIT_ADAPTER_REPO = "felipesp1983/kg1-self-test-init"
INIT_ADAPTER_SUBFOLDER = "checkpoint-6"
SAVE_EVERY_STEPS = 2
EVAL_EVERY_STEPS = 2
MAX_STEPS = 4
MAX_LENGTH = 2048
ABORT_MAX_RESERVED_GIB = 70
LOSS_NORMALIZATION_MODE = "example_mean"
FLAVOR = "h200"
ANSWER_SPAN_LOSS_WEIGHT = 2.0
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = 1
USE_ROW_LOSS_WEIGHT = "1"
REQUIRE_ROW_LOSS_WEIGHT = "1"
KG1_RESIDUAL_FIRST_GATE = "1"
KG1_V540_EXTRACTION_GATE_STATUS = "passed"
KG1_CPU_EXTRACTOR_PARITY_STATUS = "passed"
KG1_PROMPT_TEMPLATE_PARITY_STATUS = "passed"
KG1_V541_MISSMAP_GATE_STATUS = "passed"
KG1_V541_FLIP_LEDGER_STATUS = "passed"
KG1_EXPECTED_TRUNCATED = "0"
KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS = "passed"
KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE = "0"
KG1_WEAK_LABEL_AWARE_SELECTION = "0"
KG1_CPU_SIMULATION_USES_WEAK_LABELS = "0"
KG1_PROTECTED_ID_ANSWERS = "{protected_value}"
KG1_CRISIS_MODE_BACKFIRE_GUARD = "1"
KG1_CPU_SIMULATED_TOTAL_CORRECT = "200"
KG1_CPU_SIMULATED_BIT_CORRECT = "136"
KG1_CPU_SIMULATED_EQUATION_CORRECT = "59"
KG1_CPU_MISS_CLASSIFICATION_COVERAGE = "0.70"
KG1_CPU_SIMULATED_LOST_ROWS = "0"
KG1_CPU_SIMULATED_LOST_BIT_ROWS = "0"
KG1_CPU_SIMULATED_LOST_EQUATION_ROWS = "0"
KG1_MAX_TOKEN_HEADROOM_RATIO = "0.90"
KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS = "passed"
KG1_V568_LOGITS_NLL_GATE_STATUS = "passed"
KG1_V568_PROTECTED_MARGIN_STATUS = "passed"
KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN = "-0.01"
KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION = "0.0"
KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION = "0.0"
KG1_V568_MISSING_LOGPROB_ROWS = "0"
KG1_V568_PROTECTED_ROWS_CHECKED = "2"
KG1_REQUIRED_TRAIN_FAMILIES = "bit_manipulation,equation_transform"
KG1_REQUIRED_VAL_FAMILIES = "bit_manipulation,equation_transform"
KG1_REQUIRED_TRAIN_SUBCATEGORIES = "bit_pair_stride,equation_symbolic"
KG1_REQUIRED_VAL_SUBCATEGORIES = "bit_pair_stride,equation_symbolic"
env = {{
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_HF_MAX_UNIT_COST_USD": "0.09",
    "KG1_EXPECTED_MAX_LENGTH": str(MAX_LENGTH),
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": LOSS_NORMALIZATION_MODE,
}}
cmd = """
export DATA_REPO='kg1/self-test-data'
export MAX_LENGTH=2048
export ABORT_MAX_RESERVED_GIB=70
export LOSS_NORMALIZATION_MODE=example_mean
timeout=3600
Return only one line with \\boxed{{...}}. No reasoning.
"""
'''


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_pre_paid_gate_self_test_") as temp_dir:
        root = Path(temp_dir)
        train = root / "train.jsonl"
        val = root / "val.jsonl"
        _write_jsonl(train, _self_test_sft_rows())
        _write_jsonl(val, _self_test_sft_rows())
        train_sha = sha256_file(train)
        val_sha = sha256_file(val)
        launcher = root / "launcher.py"
        launcher.write_text(_self_test_launcher_text(train_sha, val_sha), encoding="utf-8", newline="\n")
        common_args = [
            "--launcher",
            str(launcher),
            "--train-jsonl",
            str(train),
            "--val-jsonl",
            str(val),
            "--dataset-schema",
            "sft",
            "--expected-save-every-steps",
            "2",
            "--expected-eval-every-steps",
            "2",
            "--expected-max-length",
            "2048",
            "--expected-abort-max-reserved-gib",
            "70",
            "--expected-loss-normalization-mode",
            "example_mean",
            "--expected-eval-output-contract",
            "one_line_boxed_no_reasoning",
            "--max-assistant-chars-p95",
            "80",
            "--max-assistant-chars-max",
            "80",
            "--require-row-loss-weight",
            "--expected-data-repo",
            "kg1/self-test-data",
            "--expected-data-root",
            "data/self-test",
            "--expected-train-sha256",
            train_sha,
            "--expected-val-sha256",
            val_sha,
            "--expected-train-rows",
            "2",
            "--expected-val-rows",
            "2",
            "--expected-output-repo",
            "felipesp1983/kg1-self-test-output",
            "--expected-init-adapter-repo",
            "felipesp1983/kg1-self-test-init",
            "--expected-init-adapter-subfolder",
            "checkpoint-6",
        ]
        ok_report = run_gate(parse_args(common_args), emit=False)
        if ok_report["ok"] is not True:
            raise RuntimeError("self-test expected clean launcher/dataset to pass: " + json.dumps(ok_report["findings"]))

        missing_protected_launcher = root / "launcher_missing_protected.py"
        missing_protected_launcher.write_text(
            _self_test_launcher_text(train_sha, val_sha, protected="8740ed31=01101000"),
            encoding="utf-8",
            newline="\n",
        )
        bad_protected_args = common_args.copy()
        bad_protected_args[bad_protected_args.index(str(launcher))] = str(missing_protected_launcher)
        bad_protected_report = run_gate(parse_args(bad_protected_args), emit=False)
        bad_codes = {item["code"] for item in bad_protected_report["findings"]}
        if "residual_first_missing_protected_row_guard" not in bad_codes:
            raise RuntimeError("self-test expected missing second protected row to fail")

        drift_launcher = root / "launcher_bad_drift.py"
        drift_launcher.write_text(
            _self_test_launcher_text(train_sha, val_sha).replace(
                'KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION = "0.0"',
                'KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION = "0.01"',
            ),
            encoding="utf-8",
            newline="\n",
        )
        drift_args = common_args.copy()
        drift_args[drift_args.index(str(launcher))] = str(drift_launcher)
        drift_report = run_gate(parse_args(drift_args), emit=False)
        drift_codes = {item["code"] for item in drift_report["findings"]}
        if "decoding_vs_adapter_drift_margin_regression" not in drift_codes:
            raise RuntimeError("self-test expected protected margin regression to fail")

        rule_train = root / "train_rule_prefix.jsonl"
        _write_jsonl(rule_train, _self_test_sft_rows(rule_prefix=True))
        rule_train_sha = sha256_file(rule_train)
        rule_launcher = root / "launcher_rule_prefix.py"
        rule_launcher.write_text(_self_test_launcher_text(rule_train_sha, val_sha), encoding="utf-8", newline="\n")
        rule_args = common_args.copy()
        rule_args[rule_args.index(str(launcher))] = str(rule_launcher)
        rule_args[rule_args.index(str(train))] = str(rule_train)
        rule_args[rule_args.index(train_sha)] = rule_train_sha
        rule_report = run_gate(parse_args(rule_args), emit=False)
        rule_codes = {item["code"] for item in rule_report["findings"]}
        if "sft_target_prefix_incompatible_with_eval_contract" not in rule_codes:
            raise RuntimeError("self-test expected RULE-prefixed SFT target to fail boxed-only contract")

        trace_train = root / "train_trace_prefix.jsonl"
        _write_jsonl(trace_train, _self_test_sft_rows(trace_prefix=True))
        trace_train_sha = sha256_file(trace_train)
        trace_launcher = root / "launcher_trace_prefix.py"
        trace_launcher.write_text(_self_test_launcher_text(trace_train_sha, val_sha), encoding="utf-8", newline="\n")
        trace_args = common_args.copy()
        trace_args[trace_args.index(str(launcher))] = str(trace_launcher)
        trace_args[trace_args.index(str(train))] = str(trace_train)
        trace_args[trace_args.index(train_sha)] = trace_train_sha
        trace_report = run_gate(parse_args(trace_args), emit=False)
        trace_codes = {item["code"] for item in trace_report["findings"]}
        if "sft_trace_targets_incompatible_with_eval_contract" not in trace_codes:
            raise RuntimeError("self-test expected Trace-prefixed SFT target to fail boxed-only contract")
    print("kg1_pre_paid_job_integration_gate_self_test=ok", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    report = run_gate(args, emit=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

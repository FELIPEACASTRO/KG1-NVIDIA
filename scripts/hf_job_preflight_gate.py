#!/usr/bin/env python3
"""Cheap preflight gate for Hugging Face GPU jobs.

This script is intentionally separate from notebook_release_gate.py. The
notebook gate validates the launcher before it is pushed; this gate runs inside
the paid HF Job container before expensive model download/load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_PHASES = {"preinstall", "artifacts", "postinstall", "eval-preinstall", "eval-postinstall", "all"}
VALID_SAMPLING_MODES = {"shuffle", "weighted_replacement"}
BLOCKED_DATASET_MARKERS = {
    "v461_synthetic_numeric_probe_pack": (
        "V461 is quarantined: crisis audit found a synthetic prompt/answer seed "
        "matching the full reference set. Rebuild without full-reference seeds."
    ),
    "v463_v462_synthetic_numeric_hard_negative_audit": (
        "V463 is quarantined because it depends on the V461/V462 numeric route."
    ),
    "v464_v463_numeric_multirule_dataset": (
        "V464 is quarantined: crisis audit found rejected_candidate == answer "
        "in 24 train rows and 6 validation rows. Use a later corrected dataset."
    ),
    "v468_v464_symbol_fix_dataset": (
        "V468 is quarantined: crisis audit found it still contains a full-reference "
        "exact prompt/answer seed."
    ),
    "v447_v446_trace_dataset": (
        "Current V447 is quarantined: crisis audit found hypothesis_formed traces "
        "with contradictory boxed answers. Rebuild with rule_found-only traces."
    ),
    "v581_combined_teacher_distill_dataset": (
        "V581/V582 teacher distillation is quarantined for paid training: V589 V509 "
        "audit found exact weak/full reference overlap plus false anti-leakage flags. "
        "Use only as diagnostic evidence, not as an SFT dataset."
    ),
    "v582_combined_teacher_distill_dataset": (
        "V582 teacher distillation is quarantined for paid training: V589 V509 audit "
        "found exact weak/full reference overlap plus false anti-leakage flags. Use "
        "only as diagnostic evidence, not as an SFT dataset."
    ),
    "v573_v571_bitpair_v551_equation_reference_mix": (
        "V573/V574 reference mix is quarantined for paid training: V605 plateau audit "
        "measured 191/315, bit=135, equation=56, trunc=1, protected backfire=2."
    ),
    "v579_v571_bitpair_v551_equation_strictedge_mix": (
        "V579 strict-edge mix is quarantined as a paid-training source: downstream "
        "V591/V592 preserved no equation gains and triggered protected bit backfire."
    ),
    "v591_v579_symbolic_queryop_source_mix": (
        "V591 symbolic query-op source mix is quarantined for paid training: V605 "
        "measured 191/315, bit=135, equation=56, trunc=1, protected backfire=2."
    ),
    "v594_queryop_cryptarithm_preference_dataset": (
        "V594/V595 query-op preference route is quarantined for paid training: "
        "V597 weak eval stayed at equation=56 and regressed bit/protected rows."
    ),
    "v596_queryop_answer_only_preference_dataset": (
        "V596 answer-only query-op preference route is quarantined for paid training: "
        "V597/V602/V604 showed no equation transfer and no submit-safe total gain."
    ),
}

RESIDUAL_FIRST_MIN_TOTAL = 196
RESIDUAL_FIRST_MIN_BIT = 136
RESIDUAL_FIRST_MIN_EQUATION = 60
RESIDUAL_FIRST_MIN_COVERAGE = 0.70
PROTECTED_ROW_EXPECTED = [
    "8740ed31=01101000",
    "59bee375=10010101",
    "55d834d1=00111111",
]


def log_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label} = {json.dumps(payload, sort_keys=True)}", flush=True)


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int = 0) -> int:
    value = env_str(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float = 0.0) -> float:
    value = env_str(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {value!r}") from exc


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, "1" if default else "0").lower()
    return value in {"1", "true", "yes", "y", "on"}


def blocked_dataset_matches(data_identity: str) -> list[dict[str, str]]:
    return [
        {"marker": marker, "reason": reason}
        for marker, reason in BLOCKED_DATASET_MARKERS.items()
        if marker in data_identity
    ]


def require_env(names: list[str]) -> None:
    missing = [name for name in names if not env_str(name)]
    if missing:
        raise RuntimeError("Missing required HF job environment variables: " + ", ".join(missing))


def check_residual_first_gpu_gate() -> dict[str, Any]:
    """Block paid training unless V540/V541 CPU gates have already passed."""
    if env_bool("KG1_ALLOW_MISSING_RESIDUAL_FIRST_GATES", False):
        raise RuntimeError(
            "KG1_ALLOW_MISSING_RESIDUAL_FIRST_GATES is not allowed in paid HF training. "
            "Run V540 extraction/canonicalization and V541 miss-map CPU gates first."
        )

    required_exact = {
        "KG1_RESIDUAL_FIRST_GATE": "1",
        "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
        "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
        "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
        "KG1_V541_MISSMAP_GATE_STATUS": "passed",
        "KG1_V541_FLIP_LEDGER_STATUS": "passed",
        "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
        "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
        "KG1_EXPECTED_TRUNCATED": "0",
        "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
        "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
        "KG1_WEAK_LABEL_AWARE_SELECTION": "0",
        "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
    }
    observed: dict[str, Any] = {}
    for name, expected in required_exact.items():
        value = env_str(name)
        observed[name] = value
        if value.lower() != expected.lower():
            raise RuntimeError(f"{name} must be {expected!r} before paid training, got {value or '<missing>'!r}")

    protected = env_str("KG1_PROTECTED_ID_ANSWERS")
    observed["KG1_PROTECTED_ID_ANSWERS"] = protected
    missing_protected = [item for item in PROTECTED_ROW_EXPECTED if item not in protected]
    if missing_protected:
        raise RuntimeError(
            "KG1_PROTECTED_ID_ANSWERS must include "
            + ", ".join(PROTECTED_ROW_EXPECTED)
            + " before paid training; missing "
            + ", ".join(missing_protected)
        )

    total = env_int("KG1_CPU_SIMULATED_TOTAL_CORRECT", -1)
    bit = env_int("KG1_CPU_SIMULATED_BIT_CORRECT", -1)
    equation = env_int("KG1_CPU_SIMULATED_EQUATION_CORRECT", -1)
    coverage = env_float("KG1_CPU_MISS_CLASSIFICATION_COVERAGE", -1.0)
    observed.update(
        {
            "KG1_CPU_SIMULATED_TOTAL_CORRECT": total,
            "KG1_CPU_SIMULATED_BIT_CORRECT": bit,
            "KG1_CPU_SIMULATED_EQUATION_CORRECT": equation,
            "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": coverage,
        }
    )
    if total < RESIDUAL_FIRST_MIN_TOTAL:
        raise RuntimeError(f"CPU simulated total {total} below required {RESIDUAL_FIRST_MIN_TOTAL}")
    if bit < RESIDUAL_FIRST_MIN_BIT:
        raise RuntimeError(f"CPU simulated bit {bit} below required {RESIDUAL_FIRST_MIN_BIT}")
    if equation < RESIDUAL_FIRST_MIN_EQUATION:
        raise RuntimeError(f"CPU simulated equation {equation} below required {RESIDUAL_FIRST_MIN_EQUATION}")
    if coverage < RESIDUAL_FIRST_MIN_COVERAGE:
        raise RuntimeError(
            f"CPU miss classification coverage {coverage} below required {RESIDUAL_FIRST_MIN_COVERAGE}"
        )
    lost_rows = env_int("KG1_CPU_SIMULATED_LOST_ROWS", -1)
    lost_bit = env_int("KG1_CPU_SIMULATED_LOST_BIT_ROWS", -1)
    lost_equation = env_int("KG1_CPU_SIMULATED_LOST_EQUATION_ROWS", -1)
    max_token_headroom_ratio = env_float("KG1_MAX_TOKEN_HEADROOM_RATIO", 2.0)
    observed.update(
        {
            "KG1_CPU_SIMULATED_LOST_ROWS": lost_rows,
            "KG1_CPU_SIMULATED_LOST_BIT_ROWS": lost_bit,
            "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": lost_equation,
            "KG1_MAX_TOKEN_HEADROOM_RATIO": max_token_headroom_ratio,
        }
    )
    if lost_rows != 0:
        raise RuntimeError(f"CPU simulated lost rows must be 0 before paid training, got {lost_rows}")
    if lost_bit != 0:
        raise RuntimeError(f"CPU simulated lost bit rows must be 0 before paid training, got {lost_bit}")
    if lost_equation != 0:
        raise RuntimeError(f"CPU simulated lost equation rows must be 0 before paid training, got {lost_equation}")
    if max_token_headroom_ratio > 0.90:
        raise RuntimeError(
            f"token headroom ratio must be <=0.90 before paid training, got {max_token_headroom_ratio}"
        )
    return {
        "required": {
            "v540_extraction_gate_status": "passed",
            "cpu_extractor_parity_status": "passed",
            "prompt_template_parity_status": "passed",
            "v541_missmap_gate_status": "passed",
            "v541_flip_ledger_status": "passed",
            "v516_parser_current_baseline_status": "passed",
            "stale_prediction_parity_status": "passed",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not read git HEAD: {type(exc).__name__}: {exc}") from exc


def compile_repo_scripts() -> None:
    targets = [
        Path("scripts/hf_job_train_v90.py"),
        Path("scripts/hf_job_train_v315_preference.py"),
        Path("scripts/hf_job_preflight_gate.py"),
        Path("scripts/kg1_static_safety_gate.py"),
        Path("scripts/kg1_pre_paid_job_integration_gate.py"),
        Path("scripts/build_v243_training_mix.py"),
        Path("scripts/audit_jsonl_overlap.py"),
    ]
    missing = [str(path) for path in targets if not path.exists()]
    if missing:
        raise RuntimeError("Missing required repo scripts: " + ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "py_compile"] + [str(path) for path in targets])
    log_json("py_compile_ok", {"targets": [str(path) for path in targets]})


def torch_status() -> dict[str, Any]:
    import torch

    cuda_runtime = str(getattr(torch.version, "cuda", ""))
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": cuda_runtime,
        "cuda_runtime_major": int(cuda_runtime.split(".", 1)[0]) if re.match(r"^\d+", cuda_runtime) else 0,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(props.name if props else ""),
        "gpu_total_gib": float(props.total_memory / 1024**3 if props else 0.0),
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
    }


def check_torch_and_gpu(stage: str) -> None:
    status = torch_status()
    log_json(f"torch_{stage}", status)

    flavor = env_str("KG1_HF_FLAVOR")
    cuda_major = int(status.get("cuda_runtime_major") or 0)
    max_cuda_major = env_int("KG1_MAX_TORCH_CUDA_MAJOR", 0)
    if max_cuda_major and cuda_major > max_cuda_major:
        raise RuntimeError(
            "torch CUDA runtime is newer than this job allows: "
            f"runtime={status['cuda']} max_major={max_cuda_major} flavor={flavor}"
        )
    if (
        "a100" in flavor.lower()
        and cuda_major >= 13
        and not env_bool("KG1_ALLOW_CUDA13_ON_A100", False)
    ):
        raise RuntimeError(
            "Blocked CUDA 13 runtime on HF A100. Recent HF A100 jobs exposed a CUDA 12.x "
            "driver API, which made torch/vLLM fail after startup. Use H200 for this "
            "vLLM image, or use a CUDA 12-compatible image, or set KG1_ALLOW_CUDA13_ON_A100=1 "
            "only after a fresh driver gate proves compatibility."
        )

    expected_torch = env_str("KG1_EXPECTED_TORCH_VERSION")
    if expected_torch and status["torch"] != expected_torch:
        raise RuntimeError(
            f"torch changed unexpectedly at {stage}: expected {expected_torch}, got {status['torch']}"
        )

    if env_bool("KG1_REQUIRE_CUDA", True) and not status["cuda_available"]:
        raise RuntimeError("CUDA is required for this HF training job.")

    min_gpu_total_gib = env_float("KG1_MIN_GPU_TOTAL_GIB", 0.0)
    if min_gpu_total_gib and status["gpu_total_gib"] < min_gpu_total_gib:
        raise RuntimeError(
            f"GPU memory below floor: {status['gpu_total_gib']:.2f}GiB < {min_gpu_total_gib:.2f}GiB"
        )

    required_gpu_regex = env_str("KG1_REQUIRED_GPU_NAME_REGEX")
    if required_gpu_regex and not re.search(required_gpu_regex, status["gpu_name"], re.IGNORECASE):
        raise RuntimeError(
            f"GPU name mismatch: regex={required_gpu_regex!r}, observed={status['gpu_name']!r}"
        )


def check_hf_flavor_cost() -> None:
    flavor = env_str("KG1_HF_FLAVOR")
    unit_cost = env_float("KG1_HF_UNIT_COST_USD", 0.0)
    max_unit_cost = env_float("KG1_HF_MAX_UNIT_COST_USD", 0.0)
    allowed_flavors = {
        item.strip()
        for item in env_str("KG1_ALLOWED_HF_FLAVORS", "h200,h100,a100-large").split(",")
        if item.strip()
    }
    if flavor and allowed_flavors and flavor not in allowed_flavors:
        raise RuntimeError(f"HF flavor {flavor!r} not allowed; allowed={sorted(allowed_flavors)}")
    if max_unit_cost and unit_cost > max_unit_cost:
        raise RuntimeError(f"HF unit cost too high: {unit_cost} > {max_unit_cost}")
    log_json(
        "hf_flavor_cost_gate",
        {
            "flavor": flavor,
            "unit_cost_usd": unit_cost,
            "max_unit_cost_usd": max_unit_cost,
            "allowed_flavors": sorted(allowed_flavors),
        },
    )


def check_eval_prompt_contract_guard() -> dict[str, Any]:
    """Block weak-eval promotion settings proven unsafe by V563/V567.

    V567 showed that short/no-think evaluation is useful only as a diagnostic:
    it destroys protected rows and does not reproduce the V290 weak contract.
    Promotional weak evals must keep thinking enabled and keep the historical
    long generation budget unless the job explicitly marks itself diagnostic.
    """

    diagnostic_only = (
        env_bool("KG1_WEAK_EVAL_DIAGNOSTIC_ONLY", False)
        or env_bool("KG1_PROMPT_CONTRACT_DIAGNOSTIC_ONLY", False)
        or env_bool("KG1_ALLOW_PROMPT_CONTRACT_DIAGNOSTIC", False)
    )
    enforced = env_bool("KG1_ENFORCE_WEAK_PROMOTION_GATE", not diagnostic_only)
    disable_thinking = env_bool("KG1_DISABLE_THINKING", False)
    no_prompt_suffix = env_bool("KG1_NO_PROMPT_SUFFIX", False)
    prompt_suffix = env_str("KG1_PROMPT_SUFFIX")
    max_tokens = env_int("KG1_MAX_TOKENS", 7680)
    min_max_tokens = env_int("KG1_PROMOTIONAL_MIN_MAX_TOKENS", 7680)
    require_prompt_suffix = env_bool("KG1_REQUIRE_PROMPT_SUFFIX_FOR_PROMOTION", True)
    lower_suffix = prompt_suffix.lower()
    blockers: list[str] = []

    if enforced and not diagnostic_only:
        if disable_thinking:
            blockers.append("disable_thinking_promotional_eval_blocked")
        if max_tokens < min_max_tokens:
            blockers.append(f"max_tokens_lt_{min_max_tokens}")
        if require_prompt_suffix and no_prompt_suffix:
            blockers.append("no_prompt_suffix_promotional_eval_blocked")
        if "do not use <think>" in lower_suffix or "disable thinking" in lower_suffix:
            blockers.append("strict_no_think_prompt_suffix_promotional_eval_blocked")
        if (
            "no reasoning" in lower_suffix
            or "no explanation" in lower_suffix
            or "return only one line" in lower_suffix
        ):
            blockers.append("strict_no_reasoning_prompt_suffix_promotional_eval_blocked")

    payload = {
        "enforced": enforced,
        "diagnostic_only": diagnostic_only,
        "disable_thinking": disable_thinking,
        "no_prompt_suffix": no_prompt_suffix,
        "max_tokens": max_tokens,
        "min_max_tokens": min_max_tokens,
        "require_prompt_suffix": require_prompt_suffix,
        "prompt_suffix_sha256": hashlib.sha256(prompt_suffix.encode("utf-8")).hexdigest() if prompt_suffix else "",
        "blockers": blockers,
    }
    log_json("eval_prompt_contract_guard", payload)
    if blockers:
        raise RuntimeError(
            "Unsafe promotional weak-eval prompt contract blocked by V563/V567 guard: "
            + ", ".join(blockers)
        )
    return payload


def check_decoding_vs_adapter_drift_gate() -> dict[str, Any]:
    """Require the V568 diagnostic before paid promotional training.

    This gate prevents a recurring failure mode: treating every weak ACC drop as
    a decoding problem when the adapter may actually have moved probability mass
    toward a known wrong answer. A new paid train/promotion path must first show
    that protected rows did not regress against a baseline margin and that the
    logits/NLL probe is complete. Absolute negative margins are informative, but
    not fatal by default because V568 showed that the long CoT trajectory can
    still recover rows whose immediate short-answer margin is negative.
    """

    if not env_bool("KG1_REQUIRE_DECODING_VS_ADAPTER_DRIFT_GATE", True):
        payload = {"required": False, "reason": "KG1_REQUIRE_DECODING_VS_ADAPTER_DRIFT_GATE=0"}
        log_json("decoding_vs_adapter_drift_gate", payload)
        return payload

    status_value = env_str("KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS")
    if status_value.lower() == "deferred_post_checkpoint":
        max_steps = env_int("MAX_STEPS", env_int("KG1_EXPECTED_MAX_STEPS", 999999))
        save_every = env_int("SAVE_EVERY_STEPS", 999999)
        eval_every = env_int("EVAL_EVERY_STEPS", 999999)
        allow_defer = env_bool("KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT", False)
        first_checkpoint_eval_required = env_bool("KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED", False)
        blockers: list[str] = []
        if not allow_defer:
            blockers.append("deferred_gate_not_explicitly_allowed")
        if not first_checkpoint_eval_required:
            blockers.append("first_checkpoint_weak_eval_not_required")
        v618_surface = env_str("KG1_V618_MODULE_SURFACE_GATE_STATUS").lower() == "passed"
        max_steps_limit = 20 if v618_surface else 2
        checkpoint_limit = 10 if v618_surface else 2
        if max_steps > max_steps_limit:
            blockers.append(f"max_steps_gt_{max_steps_limit}:{max_steps}")
        if save_every > checkpoint_limit:
            blockers.append(f"save_every_gt_{checkpoint_limit}:{save_every}")
        if eval_every > checkpoint_limit:
            blockers.append(f"eval_every_gt_{checkpoint_limit}:{eval_every}")
        payload = {
            "required": True,
            "mode": "deferred_post_checkpoint",
            "purpose": "permit one bounded smoke when a new adapter checkpoint is required before V568 can be measured",
            "limits": {
                "max_steps_lte": max_steps_limit,
                "checkpoint_every_steps_lte": checkpoint_limit,
                "v618_surface_route": v618_surface,
            },
            "observed": {
                "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": status_value,
                "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": allow_defer,
                "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": first_checkpoint_eval_required,
                "KG1_V618_MODULE_SURFACE_GATE_STATUS": env_str("KG1_V618_MODULE_SURFACE_GATE_STATUS"),
                "MAX_STEPS": max_steps,
                "SAVE_EVERY_STEPS": save_every,
                "EVAL_EVERY_STEPS": eval_every,
            },
            "blockers": blockers,
        }
        log_json("decoding_vs_adapter_drift_gate", payload)
        if blockers:
            raise RuntimeError(
                "Deferred decoding-vs-adapter drift gate is unsafe: " + ", ".join(blockers)
            )
        return payload

    required_statuses = {
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "passed",
        "KG1_V568_LOGITS_NLL_GATE_STATUS": "passed",
        "KG1_V568_PROTECTED_MARGIN_STATUS": "passed",
    }
    observed: dict[str, Any] = {}
    blockers: list[str] = []

    for name, expected in required_statuses.items():
        value = env_str(name)
        observed[name] = value
        if value.lower() != expected:
            blockers.append(f"{name}_not_{expected}")

    min_wrong_minus_correct_margin = env_float(
        "KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN", -999.0
    )
    max_protected_margin_regression = env_float(
        "KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION", 999.0
    )
    allowed_protected_margin_regression = env_float(
        "KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION", 0.0
    )
    require_nonnegative_absolute_margin = env_bool(
        "KG1_V568_REQUIRE_NONNEGATIVE_ABSOLUTE_MARGIN", False
    )
    missing_logprob_rows = env_int("KG1_V568_MISSING_LOGPROB_ROWS", 999999)
    protected_rows_checked = env_int("KG1_V568_PROTECTED_ROWS_CHECKED", 0)
    observed.update(
        {
            "KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN": min_wrong_minus_correct_margin,
            "KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION": max_protected_margin_regression,
            "KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION": allowed_protected_margin_regression,
            "KG1_V568_REQUIRE_NONNEGATIVE_ABSOLUTE_MARGIN": require_nonnegative_absolute_margin,
            "KG1_V568_MISSING_LOGPROB_ROWS": missing_logprob_rows,
            "KG1_V568_PROTECTED_ROWS_CHECKED": protected_rows_checked,
        }
    )

    if require_nonnegative_absolute_margin and min_wrong_minus_correct_margin < 0.0:
        blockers.append("protected_wrong_answer_margin_negative")
    if max_protected_margin_regression > allowed_protected_margin_regression:
        blockers.append("protected_margin_regressed_vs_baseline")
    if missing_logprob_rows != 0:
        blockers.append("missing_logprob_rows_nonzero")
    if protected_rows_checked < len(PROTECTED_ROW_EXPECTED):
        blockers.append("protected_rows_checked_incomplete")

    payload = {
        "required": True,
        "purpose": "separate bad decoding from adapter drift toward known wrong answers",
        "required_statuses": required_statuses,
        "observed": observed,
        "blockers": blockers,
    }
    log_json("decoding_vs_adapter_drift_gate", payload)
    if blockers:
        raise RuntimeError(
            "Decoding-vs-adapter drift gate failed; do not spend paid GPU training: "
            + ", ".join(blockers)
        )
    return payload


def check_training_env() -> None:
    require_env(
        [
            "KG1_EXPECTED_COMMIT",
            "MODEL_NAME",
            "MODEL_REVISION",
            "DATA_REPO",
            "DATA_FILE",
            "VAL_FILE",
            "EXPECTED_TRAIN_SHA256",
            "EXPECTED_VAL_SHA256",
            "MIN_TRAIN_EXAMPLES",
            "MIN_VAL_EXAMPLES",
            "OUTPUT_REPO",
            "RUN_ID",
            "INIT_ADAPTER_REPO",
            "INIT_ADAPTER_SUBFOLDER",
            "LORA_R",
            "LORA_ALPHA",
            "LORA_TARGET_MODULES",
            "MAX_TRAINABLE_PARAM_RATIO",
            "MAX_LENGTH",
            "BATCH_SIZE",
            "MICRO_BATCH_SIZE",
            "LEARNING_RATE",
            "FINAL_LEARNING_RATE",
            "NUM_EPOCHS",
            "MAX_STEPS",
            "SAVE_EVERY_STEPS",
            "EVAL_EVERY_STEPS",
            "EVAL_MAX_EXAMPLES",
            "SAMPLING_MODE",
            "LOSS_NORMALIZATION_MODE",
            "MAX_PROMPT_TRUNCATION_RATE",
            "REQUIRE_OFFSET_MASK",
        ]
    )

    sampling_mode = env_str("SAMPLING_MODE")
    if sampling_mode not in VALID_SAMPLING_MODES:
        raise RuntimeError(
            f"SAMPLING_MODE must be one of {sorted(VALID_SAMPLING_MODES)}, got {sampling_mode!r}"
        )

    max_steps = env_int("MAX_STEPS")
    expected_max_steps = env_int("KG1_EXPECTED_MAX_STEPS", 0)
    if expected_max_steps and max_steps != expected_max_steps:
        raise RuntimeError(f"MAX_STEPS mismatch: expected {expected_max_steps}, got {max_steps}")
    max_length = env_int("MAX_LENGTH")
    expected_max_length = env_int("KG1_EXPECTED_MAX_LENGTH", 0)
    if expected_max_length and max_length != expected_max_length:
        raise RuntimeError(f"MAX_LENGTH mismatch: expected {expected_max_length}, got {max_length}")

    loss_normalization_mode = env_str("LOSS_NORMALIZATION_MODE")
    valid_loss_normalization_modes = {"token_mean", "example_mean"}
    if loss_normalization_mode not in valid_loss_normalization_modes:
        raise RuntimeError(
            "LOSS_NORMALIZATION_MODE must be one of "
            f"{sorted(valid_loss_normalization_modes)}, got {loss_normalization_mode!r}"
        )
    expected_loss_normalization_mode = env_str("KG1_EXPECTED_LOSS_NORMALIZATION_MODE")
    if expected_loss_normalization_mode and loss_normalization_mode != expected_loss_normalization_mode:
        raise RuntimeError(
            "LOSS_NORMALIZATION_MODE mismatch: "
            f"expected {expected_loss_normalization_mode!r}, got {loss_normalization_mode!r}"
        )

    if env_bool("KG1_REQUIRE_OFFSET_MASK", True) and not env_bool("REQUIRE_OFFSET_MASK", False):
        raise RuntimeError("REQUIRE_OFFSET_MASK must remain enabled for real training.")

    residual_first_gate = check_residual_first_gpu_gate()
    decoding_vs_adapter_drift_gate = check_decoding_vs_adapter_drift_gate()

    max_prompt_truncation_rate = env_float("MAX_PROMPT_TRUNCATION_RATE", 1.0)
    if max_prompt_truncation_rate > env_float("KG1_MAX_PROMPT_TRUNCATION_RATE", 0.0):
        raise RuntimeError(
            "MAX_PROMPT_TRUNCATION_RATE exceeds hard gate: "
            f"{max_prompt_truncation_rate} > {env_float('KG1_MAX_PROMPT_TRUNCATION_RATE', 0.0)}"
        )

    if env_bool("TOKENIZE_ONLY_DRY_RUN", False):
        raise RuntimeError("TOKENIZE_ONLY_DRY_RUN=1 is not a real HF training job.")
    if env_bool("DRY_RUN_VALIDATE_ONLY", False):
        raise RuntimeError("DRY_RUN_VALIDATE_ONLY=1 is not a real HF training job.")

    if env_bool("UPLOAD_TO_HF", True) and not env_str("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required when UPLOAD_TO_HF=1.")

    summary = {
        "model": env_str("MODEL_NAME"),
        "model_revision": env_str("MODEL_REVISION"),
        "data_repo": env_str("DATA_REPO"),
        "data_file": env_str("DATA_FILE"),
        "val_file": env_str("VAL_FILE"),
        "output_repo": env_str("OUTPUT_REPO"),
        "run_id": env_str("RUN_ID"),
        "max_steps": max_steps,
        "sampling_mode": sampling_mode,
        "loss_normalization_mode": loss_normalization_mode,
        "max_length": max_length,
        "lora_r": env_int("LORA_R"),
        "lora_alpha": env_int("LORA_ALPHA"),
        "target_modules": env_str("LORA_TARGET_MODULES"),
        "target_parameters": env_str("LORA_TARGET_PARAMETERS"),
        "trainable_lora_modules": env_str("TRAINABLE_LORA_MODULES"),
        "trainable_lora_name_substrings": env_str("TRAINABLE_LORA_NAME_SUBSTRINGS"),
        "require_lora_target_parameter_match": env_bool("REQUIRE_LORA_TARGET_PARAMETER_MATCH", False),
        "required_trainable_lora_name_substrings": env_str(
            "REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS"
        ),
        "residual_first_gpu_gate": residual_first_gate,
        "decoding_vs_adapter_drift_gate": decoding_vs_adapter_drift_gate,
    }
    log_json("training_env_gate", summary)

    data_identity = " ".join(
        [
            env_str("DATA_REPO"),
            env_str("DATA_FILE"),
            env_str("VAL_FILE"),
            env_str("EXPECTED_TRAIN_SHA256"),
            env_str("EXPECTED_VAL_SHA256"),
        ]
    )
    blocked = blocked_dataset_matches(data_identity)
    if blocked:
        raise RuntimeError("Blocked quarantined training dataset: " + json.dumps(blocked, sort_keys=True))


def parse_csv_env(name: str) -> list[str]:
    return [item.strip() for item in env_str(name).split(",") if item.strip()]


def percentile_int(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[int(ratio * (len(ordered) - 1))])


def boxed_payloads(text: str) -> list[str]:
    return re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))


def check_repo_gate() -> None:
    observed = run_git_head()
    expected = env_str("KG1_EXPECTED_COMMIT")
    log_json("repo_commit_gate", {"observed": observed, "expected": expected})
    if expected and observed != expected:
        raise RuntimeError(f"repo commit mismatch: expected {expected}, got {observed}")
    compile_repo_scripts()


def count_and_audit_jsonl(path: Path, label: str) -> dict[str, Any]:
    dataset_schema = env_str("KG1_DATASET_SCHEMA", "sft").strip().lower() or "sft"
    if dataset_schema not in {"sft", "preference"}:
        raise RuntimeError(f"KG1_DATASET_SCHEMA must be sft or preference, got {dataset_schema!r}")
    rows = 0
    bad_rows: list[dict[str, Any]] = []
    preference_bad_rows: list[dict[str, Any]] = []
    negative_types: dict[str, int] = {}
    families: dict[str, int] = {}
    subcategories: dict[str, int] = {}
    gate_row_flags = [
        "gate_rows_used_for_training",
        "weak_gate_rows_used_for_training",
        "full_gate_rows_used_for_training",
    ]
    gate_row_flag_counts: dict[str, int] = {flag: 0 for flag in gate_row_flags}
    gate_row_flag_missing_counts: dict[str, int] = {flag: 0 for flag in gate_row_flags}
    gate_row_flag_bad_rows: list[dict[str, Any]] = []
    weak_reference_signal_flags = [
        "expected_aware_teacher_signal",
        "label_audited_teacher_projection",
    ]
    weak_reference_signal_counts: dict[str, int] = {flag: 0 for flag in weak_reference_signal_flags}
    weak_reference_signal_bad_rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    duplicate_ids = 0
    assistant_missing = 0
    assistant_lengths_by_family: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
            except Exception as exc:
                bad_rows.append({"line": line_no, "error": repr(exc)})
                continue
            row_id = str(row.get("id", ""))
            if row_id:
                if row_id in ids:
                    duplicate_ids += 1
                ids.add(row_id)
            family = str(row.get("family") or row.get("category") or "unknown")
            families[family] = families.get(family, 0) + 1
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            for flag in gate_row_flags:
                if flag not in metadata:
                    gate_row_flag_missing_counts[flag] += 1
                    continue
                if metadata.get(flag) is False:
                    gate_row_flag_counts[flag] += 1
                else:
                    gate_row_flag_bad_rows.append(
                        {
                            "line": line_no,
                            "id": row_id,
                            "flag": flag,
                            "value": metadata.get(flag),
                        }
                    )
            for flag in weak_reference_signal_flags:
                value = row.get(flag) if flag in row else metadata.get(flag)
                if value is True:
                    weak_reference_signal_counts[flag] += 1
                    weak_reference_signal_bad_rows.append(
                        {
                            "line": line_no,
                            "id": row_id,
                            "flag": flag,
                            "value": value,
                        }
                    )
            subcategory = str(
                row.get("subcategory")
                or row.get("subtype")
                or metadata.get("subcategory")
                or metadata.get("subtype")
                or "unknown"
            )
            subcategories[subcategory] = subcategories.get(subcategory, 0) + 1
            if dataset_schema == "preference":
                prompt = str(row.get("prompt", ""))
                chosen = str(row.get("chosen", ""))
                rejected = str(row.get("rejected", ""))
                negative_type = str(metadata.get("negative_type", "unknown"))
                negative_types[negative_type] = negative_types.get(negative_type, 0) + 1
                chosen_boxes = boxed_payloads(chosen)
                rejected_boxes = boxed_payloads(rejected)
                if not prompt or not chosen or not rejected:
                    preference_bad_rows.append({"line": line_no, "id": row_id, "error": "missing_prompt_chosen_rejected"})
                if len(chosen_boxes) != 1:
                    preference_bad_rows.append({"line": line_no, "id": row_id, "error": "chosen_box_count"})
                if len(rejected_boxes) != 1:
                    preference_bad_rows.append({"line": line_no, "id": row_id, "error": "rejected_box_count"})
                if chosen_boxes and rejected_boxes and chosen_boxes[0] == rejected_boxes[0]:
                    preference_bad_rows.append({"line": line_no, "id": row_id, "error": "chosen_equals_rejected_payload"})
                if negative_type.startswith("format_negative_"):
                    preference_bad_rows.append({"line": line_no, "id": row_id, "error": "format_negative_blocked"})
                assistant_lengths_by_family.setdefault(family, []).append(len(chosen))
            else:
                messages = row.get("messages")
                if not isinstance(messages, list) or not any(
                    isinstance(item, dict) and item.get("role") == "assistant" for item in messages
                ):
                    assistant_missing += 1
                else:
                    assistant_text = ""
                    for item in reversed(messages):
                        if isinstance(item, dict) and item.get("role") == "assistant":
                            assistant_text = str(item.get("content", ""))
                            break
                    assistant_lengths_by_family.setdefault(family, []).append(len(assistant_text))
    assistant_length_stats: dict[str, dict[str, int]] = {}
    for family, lengths in sorted(assistant_lengths_by_family.items()):
        assistant_length_stats[family] = {
            "rows": len(lengths),
            "chars_p50": percentile_int(lengths, 0.50),
            "chars_p95": percentile_int(lengths, 0.95),
            "chars_max": max(lengths) if lengths else 0,
        }
    summary = {
        "label": label,
        "path": str(path),
        "dataset_schema": dataset_schema,
        "rows": rows,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "assistant_missing": assistant_missing,
        "bad_rows_first10": bad_rows[:10],
        "preference_bad_rows_first10": preference_bad_rows[:10],
        "negative_type_counts": dict(sorted(negative_types.items())),
        "gate_row_flag_counts": gate_row_flag_counts,
        "gate_row_flag_missing_counts": gate_row_flag_missing_counts,
        "gate_row_flag_bad_rows_first10": gate_row_flag_bad_rows[:10],
        "weak_reference_signal_counts": weak_reference_signal_counts,
        "weak_reference_signal_bad_rows_first10": weak_reference_signal_bad_rows[:10],
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "subcategory_counts_top20": dict(sorted(subcategories.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "assistant_length_stats": assistant_length_stats,
    }
    log_json(f"{label}_jsonl_audit", summary)
    if bad_rows:
        raise RuntimeError(f"{label} JSONL has invalid rows: {bad_rows[:3]}")
    if preference_bad_rows:
        raise RuntimeError(
            f"{label} preference JSONL has invalid rows: "
            + json.dumps(preference_bad_rows[:10], sort_keys=True)
        )
    if assistant_missing:
        raise RuntimeError(f"{label} has rows without assistant messages: {assistant_missing}")
    if gate_row_flag_bad_rows:
        raise RuntimeError(
            f"{label} has rows marked as gate/full/weak rows used for training: "
            + json.dumps(gate_row_flag_bad_rows[:10], sort_keys=True)
        )
    if any(count for count in gate_row_flag_missing_counts.values()):
        raise RuntimeError(
            f"{label} has rows missing required anti-leakage gate flags: "
            + json.dumps(gate_row_flag_missing_counts, sort_keys=True)
        )
    if weak_reference_signal_bad_rows:
        raise RuntimeError(
            f"{label} contains expected-aware/reference-derived teacher rows. "
            "Paid training must not use weak/full reference-derived labels: "
            + json.dumps(weak_reference_signal_bad_rows[:10], sort_keys=True)
        )
    max_p95 = env_int("KG1_MAX_ASSISTANT_CHARS_P95", 0)
    max_chars = env_int("KG1_MAX_ASSISTANT_CHARS_MAX", 0)
    if max_p95:
        offenders = {
            family: stats
            for family, stats in assistant_length_stats.items()
            if int(stats.get("chars_p95", 0)) > max_p95
        }
        if offenders:
            raise RuntimeError(
                f"{label} assistant chars p95 exceeds KG1_MAX_ASSISTANT_CHARS_P95={max_p95}: "
                + json.dumps(offenders, sort_keys=True)
            )
    if max_chars:
        offenders = {
            family: stats
            for family, stats in assistant_length_stats.items()
            if int(stats.get("chars_max", 0)) > max_chars
        }
        if offenders:
            raise RuntimeError(
                f"{label} assistant chars max exceeds KG1_MAX_ASSISTANT_CHARS_MAX={max_chars}: "
                + json.dumps(offenders, sort_keys=True)
            )
    return summary


def download_and_check_dataset(
    repo_id: str,
    filename: str,
    expected_sha: str,
    min_rows: int,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=env_str("HF_TOKEN") or None))
    observed_sha = sha256_file(path)
    log_json(
        f"{label}_dataset_file",
        {"repo_id": repo_id, "filename": filename, "path": str(path), "sha256": observed_sha, "expected_sha256": expected_sha},
    )
    if expected_sha and observed_sha.lower() != expected_sha.lower():
        raise RuntimeError(f"{label} dataset sha mismatch: observed={observed_sha} expected={expected_sha}")
    summary = count_and_audit_jsonl(path, label)
    if summary["rows"] < min_rows:
        raise RuntimeError(f"{label} row count below floor: {summary['rows']} < {min_rows}")
    return path, summary


def check_required_counts(summary: dict[str, Any], env_name: str, count_key: str) -> None:
    raw = env_str(env_name)
    if not raw:
        return
    observed = summary[count_key]
    missing: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in observed:
            missing.append(item)
    if missing:
        raise RuntimeError(f"{env_name} missing from {summary['label']}: {missing}")


def check_hub_artifacts() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = env_str("HF_TOKEN") or None
    api = HfApi(token=token)
    model_info = api.model_info(env_str("MODEL_NAME"), revision=env_str("MODEL_REVISION"))
    log_json(
        "model_revision_gate",
        {"model": env_str("MODEL_NAME"), "revision": env_str("MODEL_REVISION"), "sha": getattr(model_info, "sha", "")},
    )

    _train_path, train_summary = download_and_check_dataset(
        env_str("DATA_REPO"),
        env_str("DATA_FILE"),
        env_str("EXPECTED_TRAIN_SHA256"),
        env_int("MIN_TRAIN_EXAMPLES"),
        "train",
    )
    _val_path, val_summary = download_and_check_dataset(
        env_str("DATA_REPO"),
        env_str("VAL_FILE"),
        env_str("EXPECTED_VAL_SHA256"),
        env_int("MIN_VAL_EXAMPLES"),
        "validation",
    )
    check_required_counts(train_summary, "KG1_REQUIRED_TRAIN_FAMILIES", "family_counts")
    check_required_counts(train_summary, "KG1_REQUIRED_TRAIN_SUBCATEGORIES", "subcategory_counts")
    check_required_counts(val_summary, "KG1_REQUIRED_VAL_FAMILIES", "family_counts")
    check_required_counts(val_summary, "KG1_REQUIRED_VAL_SUBCATEGORIES", "subcategory_counts")

    adapter_repo = env_str("INIT_ADAPTER_REPO")
    adapter_subfolder = env_str("INIT_ADAPTER_SUBFOLDER").strip("/")
    files = set(api.list_repo_files(adapter_repo, repo_type="model"))
    adapter_config_name = f"{adapter_subfolder}/adapter_config.json" if adapter_subfolder else "adapter_config.json"
    adapter_weights_name = f"{adapter_subfolder}/adapter_model.safetensors" if adapter_subfolder else "adapter_model.safetensors"
    missing = [name for name in [adapter_config_name, adapter_weights_name] if name not in files]
    if missing:
        raise RuntimeError(f"Missing init adapter files in {adapter_repo}: {missing}")
    config_path = Path(
        hf_hub_download(
            repo_id=adapter_repo,
            filename="adapter_config.json",
            subfolder=adapter_subfolder or None,
            repo_type="model",
            token=token,
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = sorted(str(item) for item in (config.get("target_modules") or []))
    target_parameters = sorted(str(item) for item in (config.get("target_parameters") or []))
    modules_to_save = sorted(str(item) for item in (config.get("modules_to_save") or []))
    log_json(
        "init_adapter_gate",
        {
            "repo": adapter_repo,
            "subfolder": adapter_subfolder,
            "adapter_config": adapter_config_name,
            "adapter_weights": adapter_weights_name,
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": target_modules,
            "target_parameters": target_parameters,
            "modules_to_save": modules_to_save,
        },
    )
    if env_bool("KG1_STRICT_INIT_ADAPTER_CONFIG", False):
        if modules_to_save:
            raise RuntimeError(
                "Init adapter modules_to_save must be empty for KG1 adapter-only submit path: "
                + json.dumps(modules_to_save, sort_keys=True)
            )
        if int(config.get("r", -1)) != env_int("LORA_R"):
            raise RuntimeError(f"Init adapter r mismatch: {config.get('r')} != LORA_R={env_int('LORA_R')}")
        if int(config.get("lora_alpha", -1)) != env_int("LORA_ALPHA"):
            raise RuntimeError(
                f"Init adapter alpha mismatch: {config.get('lora_alpha')} != LORA_ALPHA={env_int('LORA_ALPHA')}"
            )
        configured_target_modules = sorted(parse_csv_env("LORA_TARGET_MODULES"))
        if target_modules and configured_target_modules != target_modules:
            raise RuntimeError(
                "Init adapter target_modules mismatch: "
                + json.dumps(
                    {"adapter": target_modules, "env": configured_target_modules},
                    sort_keys=True,
                )
            )
        configured_target_parameters = sorted(parse_csv_env("LORA_TARGET_PARAMETERS"))
        if configured_target_parameters != target_parameters:
            raise RuntimeError(
                "Init adapter target_parameters mismatch: "
                + json.dumps(
                    {"adapter": target_parameters, "env": configured_target_parameters},
                    sort_keys=True,
                )
            )
        if target_parameters and not env_bool("REQUIRE_LORA_TARGET_PARAMETER_MATCH", False):
            raise RuntimeError(
                "Init adapter has target_parameters but REQUIRE_LORA_TARGET_PARAMETER_MATCH is disabled."
            )
        if target_parameters and env_str("INIT_ADAPTER_LOAD_MODE", "peft").strip().lower() == "manual":
            raise RuntimeError(
                "INIT_ADAPTER_LOAD_MODE=manual is blocked for adapters with target_parameters. "
                "Use the PEFT-native PeftModel.from_pretrained path or run a dedicated CPU "
                "round-trip equivalence gate before allowing manual state_dict injection."
            )


def check_postinstall_imports() -> None:
    import importlib

    expected_version_env = {
        "huggingface_hub": "KG1_EXPECTED_HUGGINGFACE_HUB_VERSION",
        "transformers": "KG1_EXPECTED_TRANSFORMERS_VERSION",
        "peft": "KG1_EXPECTED_PEFT_VERSION",
        "accelerate": "KG1_EXPECTED_ACCELERATE_VERSION",
    }
    required = [
        "huggingface_hub",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
    ]
    if env_bool("KG1_REQUIRE_MAMBA_IMPORTS", True):
        required.extend(
            [
                "causal_conv1d",
                "mamba_ssm",
                "mamba_ssm.ops.triton.layernorm_gated",
                "mamba_ssm.ops.selective_scan_interface",
            ]
        )
    for module_name in required:
        module = importlib.import_module(module_name)
        version = str(getattr(module, "__version__", "unknown"))
        log_json(
            "postinstall_import_ok",
            {
                "module": module_name,
                "version": version,
            },
        )
        expected_env = expected_version_env.get(module_name, "")
        expected_version = env_str(expected_env) if expected_env else ""
        if expected_version and version != expected_version:
            raise RuntimeError(
                f"{module_name} changed unexpectedly after install: "
                f"expected {expected_version}, got {version}"
            )


def check_eval_postinstall_imports() -> None:
    import importlib

    for module_name in ["huggingface_hub", "pandas", "packaging", "safetensors", "vllm"]:
        module = importlib.import_module(module_name)
        log_json(
            "eval_postinstall_import_ok",
            {
                "module": module_name,
                "version": str(getattr(module, "__version__", "unknown")),
            },
        )


def run_phase(phase: str) -> None:
    if phase == "preinstall":
        check_repo_gate()
        check_torch_and_gpu("preinstall")
        check_hf_flavor_cost()
        check_training_env()
        return
    if phase == "artifacts":
        check_training_env()
        check_hub_artifacts()
        return
    if phase == "postinstall":
        check_torch_and_gpu("postinstall")
        check_postinstall_imports()
        check_training_env()
        return
    if phase == "eval-preinstall":
        check_repo_gate()
        check_torch_and_gpu("eval_preinstall")
        check_hf_flavor_cost()
        check_eval_prompt_contract_guard()
        return
    if phase == "eval-postinstall":
        check_torch_and_gpu("eval_postinstall")
        check_eval_postinstall_imports()
        check_hf_flavor_cost()
        check_eval_prompt_contract_guard()
        return
    if phase == "all":
        run_phase("preinstall")
        run_phase("artifacts")
        run_phase("postinstall")
        return
    raise RuntimeError(f"Unknown phase: {phase}")


def self_test() -> None:
    assert VALID_SAMPLING_MODES == {"shuffle", "weighted_replacement"}
    os.environ.setdefault("SAMPLING_MODE", "weighted_replacement")
    if env_str("SAMPLING_MODE") not in VALID_SAMPLING_MODES:
        raise RuntimeError("self-test failed")
    old_max_length = os.environ.get("MAX_LENGTH")
    old_expected_max_length = os.environ.get("KG1_EXPECTED_MAX_LENGTH")
    os.environ["MAX_LENGTH"] = "8192"
    os.environ["KG1_EXPECTED_MAX_LENGTH"] = "8192"
    if env_int("MAX_LENGTH") != env_int("KG1_EXPECTED_MAX_LENGTH"):
        raise RuntimeError("self-test max length equality failed")
    os.environ["KG1_EXPECTED_MAX_LENGTH"] = "4096"
    try:
        if env_int("MAX_LENGTH") == env_int("KG1_EXPECTED_MAX_LENGTH"):
            raise RuntimeError("self-test expected max length mismatch")
    finally:
        if old_max_length is None:
            os.environ.pop("MAX_LENGTH", None)
        else:
            os.environ["MAX_LENGTH"] = old_max_length
        if old_expected_max_length is None:
            os.environ.pop("KG1_EXPECTED_MAX_LENGTH", None)
        else:
            os.environ["KG1_EXPECTED_MAX_LENGTH"] = old_expected_max_length
    assert blocked_dataset_matches("repo data/v464_v463_numeric_multirule_dataset/train.jsonl")
    assert blocked_dataset_matches("repo data/v468_v464_symbol_fix_dataset/train.jsonl")
    assert blocked_dataset_matches("repo data/v447_v446_trace_dataset/train.jsonl")
    assert blocked_dataset_matches("repo data/v582_combined_teacher_distill_dataset/train.jsonl")
    assert blocked_dataset_matches("repo artifacts/v596_queryop_answer_only_preference_dataset/train.jsonl")
    assert not blocked_dataset_matches("repo data/v469_symbol_fix_rebuilt_clean/train.jsonl")
    old_env = dict(os.environ)
    try:
        os.environ["KG1_MAX_TOKENS"] = "7680"
        os.environ["KG1_DISABLE_THINKING"] = "0"
        os.environ["KG1_NO_PROMPT_SUFFIX"] = "0"
        os.environ["KG1_ENFORCE_WEAK_PROMOTION_GATE"] = "1"
        good_contract = check_eval_prompt_contract_guard()
        assert good_contract["blockers"] == []
        os.environ["KG1_MAX_TOKENS"] = "2048"
        try:
            check_eval_prompt_contract_guard()
        except RuntimeError as exc:
            if "max_tokens_lt_7680" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected short max_tokens contract failure")
        os.environ["KG1_MAX_TOKENS"] = "7680"
        os.environ["KG1_DISABLE_THINKING"] = "1"
        try:
            check_eval_prompt_contract_guard()
        except RuntimeError as exc:
            if "disable_thinking_promotional_eval_blocked" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected disable-thinking contract failure")
        os.environ["KG1_DISABLE_THINKING"] = "0"
        os.environ["KG1_PROMPT_SUFFIX"] = "\nReturn only one line: `\\boxed{answer}`. No reasoning. No explanation."
        try:
            check_eval_prompt_contract_guard()
        except RuntimeError as exc:
            if "strict_no_reasoning_prompt_suffix_promotional_eval_blocked" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected strict no-reasoning prompt suffix contract failure")
        os.environ["KG1_PROMPT_SUFFIX"] = ""
        os.environ["KG1_PROMPT_CONTRACT_DIAGNOSTIC_ONLY"] = "1"
        diagnostic_contract = check_eval_prompt_contract_guard()
        assert diagnostic_contract["diagnostic_only"] is True
        assert diagnostic_contract["blockers"] == []
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    old_env = dict(os.environ)
    try:
        os.environ["KG1_RESIDUAL_FIRST_GATE"] = "1"
        os.environ["KG1_V540_EXTRACTION_GATE_STATUS"] = "passed"
        os.environ["KG1_CPU_EXTRACTOR_PARITY_STATUS"] = "passed"
        os.environ["KG1_PROMPT_TEMPLATE_PARITY_STATUS"] = "passed"
        os.environ["KG1_V541_MISSMAP_GATE_STATUS"] = "passed"
        os.environ["KG1_V541_FLIP_LEDGER_STATUS"] = "passed"
        os.environ["KG1_V516_PARSER_CURRENT_BASELINE_STATUS"] = "passed"
        os.environ["KG1_STALE_PREDICTION_PARITY_STATUS"] = "passed"
        os.environ["KG1_EXPECTED_TRUNCATED"] = "0"
        os.environ["KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS"] = "passed"
        os.environ["KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE"] = "0"
        os.environ["KG1_WEAK_LABEL_AWARE_SELECTION"] = "0"
        os.environ["KG1_CPU_SIMULATION_USES_WEAK_LABELS"] = "0"
        os.environ["KG1_PROTECTED_ID_ANSWERS"] = ",".join(PROTECTED_ROW_EXPECTED)
        os.environ["KG1_CPU_SIMULATED_TOTAL_CORRECT"] = "196"
        os.environ["KG1_CPU_SIMULATED_BIT_CORRECT"] = "136"
        os.environ["KG1_CPU_SIMULATED_EQUATION_CORRECT"] = "60"
        os.environ["KG1_CPU_MISS_CLASSIFICATION_COVERAGE"] = "0.70"
        os.environ["KG1_CPU_SIMULATED_LOST_ROWS"] = "0"
        os.environ["KG1_CPU_SIMULATED_LOST_BIT_ROWS"] = "0"
        os.environ["KG1_CPU_SIMULATED_LOST_EQUATION_ROWS"] = "0"
        os.environ["KG1_MAX_TOKEN_HEADROOM_RATIO"] = "0.90"
        gate = check_residual_first_gpu_gate()
        assert gate["observed"]["KG1_CPU_SIMULATED_TOTAL_CORRECT"] == 196
        os.environ["KG1_CPU_SIMULATED_TOTAL_CORRECT"] = "195"
        try:
            check_residual_first_gpu_gate()
        except RuntimeError as exc:
            if "below required" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected residual-first CPU total failure")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    old_env = dict(os.environ)
    try:
        os.environ["KG1_REQUIRE_DECODING_VS_ADAPTER_DRIFT_GATE"] = "1"
        os.environ["KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS"] = "deferred_post_checkpoint"
        os.environ["KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT"] = "1"
        os.environ["KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED"] = "1"
        os.environ["KG1_V618_MODULE_SURFACE_GATE_STATUS"] = "passed"
        os.environ["MAX_STEPS"] = "20"
        os.environ["SAVE_EVERY_STEPS"] = "10"
        os.environ["EVAL_EVERY_STEPS"] = "10"
        deferred_gate = check_decoding_vs_adapter_drift_gate()
        assert deferred_gate["blockers"] == []
        assert deferred_gate["limits"]["max_steps_lte"] == 20
        assert deferred_gate["limits"]["checkpoint_every_steps_lte"] == 10
        os.environ["KG1_V618_MODULE_SURFACE_GATE_STATUS"] = ""
        try:
            check_decoding_vs_adapter_drift_gate()
        except RuntimeError as exc:
            if "max_steps_gt_2:20" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected legacy deferred drift limit failure")
        os.environ["KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS"] = "passed"
        os.environ["KG1_V568_LOGITS_NLL_GATE_STATUS"] = "passed"
        os.environ["KG1_V568_PROTECTED_MARGIN_STATUS"] = "passed"
        os.environ["KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN"] = "0.01"
        os.environ["KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION"] = "0.0"
        os.environ["KG1_V568_ALLOWED_PROTECTED_MARGIN_REGRESSION"] = "0.0"
        os.environ["KG1_V568_MISSING_LOGPROB_ROWS"] = "0"
        os.environ["KG1_V568_PROTECTED_ROWS_CHECKED"] = str(len(PROTECTED_ROW_EXPECTED))
        drift_gate = check_decoding_vs_adapter_drift_gate()
        assert drift_gate["blockers"] == []
        os.environ["KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN"] = "-0.01"
        negative_absolute_allowed = check_decoding_vs_adapter_drift_gate()
        assert negative_absolute_allowed["blockers"] == []
        os.environ["KG1_V568_REQUIRE_NONNEGATIVE_ABSOLUTE_MARGIN"] = "1"
        try:
            check_decoding_vs_adapter_drift_gate()
        except RuntimeError as exc:
            if "protected_wrong_answer_margin_negative" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected absolute protected margin failure")
        os.environ["KG1_V568_REQUIRE_NONNEGATIVE_ABSOLUTE_MARGIN"] = "0"
        os.environ["KG1_V568_MIN_WRONG_MINUS_CORRECT_MARGIN"] = "0.01"
        os.environ["KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION"] = "0.01"
        try:
            check_decoding_vs_adapter_drift_gate()
        except RuntimeError as exc:
            if "protected_margin_regressed_vs_baseline" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected protected margin regression failure")
        os.environ["KG1_V568_MAX_OBSERVED_PROTECTED_MARGIN_REGRESSION"] = "0.0"
        os.environ["KG1_V568_MISSING_LOGPROB_ROWS"] = "1"
        try:
            check_decoding_vs_adapter_drift_gate()
        except RuntimeError as exc:
            if "missing_logprob_rows_nonzero" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected missing logprob row failure")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        missing_flags = tmp / "missing_flags.jsonl"
        missing_flags.write_text(
            json.dumps(
                {
                    "id": "x",
                    "family": "equation_transform",
                    "messages": [{"role": "assistant", "content": "Final answer: \\boxed{1}"}],
                    "metadata": {},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            count_and_audit_jsonl(missing_flags, "missing_flags")
        except RuntimeError as exc:
            if "missing required anti-leakage gate flags" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected missing anti-leakage flags failure")

        clean_flags = tmp / "clean_flags.jsonl"
        clean_flags.write_text(
            json.dumps(
                {
                    "id": "x",
                    "family": "equation_transform",
                    "messages": [{"role": "assistant", "content": "Final answer: \\boxed{1}"}],
                    "metadata": {
                        "gate_rows_used_for_training": False,
                        "weak_gate_rows_used_for_training": False,
                        "full_gate_rows_used_for_training": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        clean_summary = count_and_audit_jsonl(clean_flags, "clean_flags")
        assert clean_summary["gate_row_flag_missing_counts"] == {
            "gate_rows_used_for_training": 0,
            "weak_gate_rows_used_for_training": 0,
            "full_gate_rows_used_for_training": 0,
        }

        old_schema = os.environ.get("KG1_DATASET_SCHEMA")
        os.environ["KG1_DATASET_SCHEMA"] = "preference"
        try:
            clean_preference = tmp / "clean_preference.jsonl"
            clean_preference.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "prompt": "prompt",
                        "chosen": "Final answer: \\boxed{12}",
                        "rejected": "Final answer: \\boxed{21}",
                        "family": "equation_transform",
                        "metadata": {
                            "negative_type": "hard_negative_symbol_flip",
                            "gate_rows_used_for_training": False,
                            "weak_gate_rows_used_for_training": False,
                            "full_gate_rows_used_for_training": False,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            clean_preference_summary = count_and_audit_jsonl(clean_preference, "clean_preference")
            if clean_preference_summary["dataset_schema"] != "preference":
                raise RuntimeError("self-test expected preference schema audit")
            blocked_preference = tmp / "blocked_preference.jsonl"
            blocked_preference.write_text(
                json.dumps(
                    {
                        "id": "p2",
                        "prompt": "prompt",
                        "chosen": "Final answer: \\boxed{12}",
                        "rejected": "Final answer: \\boxed{12}",
                        "family": "equation_transform",
                        "metadata": {
                            "negative_type": "format_negative_no_box",
                            "gate_rows_used_for_training": False,
                            "weak_gate_rows_used_for_training": False,
                            "full_gate_rows_used_for_training": False,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                count_and_audit_jsonl(blocked_preference, "blocked_preference")
            except RuntimeError as exc:
                if "preference JSONL has invalid rows" not in str(exc):
                    raise
            else:
                raise RuntimeError("self-test expected invalid preference rows to fail")
        finally:
            if old_schema is None:
                os.environ.pop("KG1_DATASET_SCHEMA", None)
            else:
                os.environ["KG1_DATASET_SCHEMA"] = old_schema

        expected_aware_flags = tmp / "expected_aware_flags.jsonl"
        expected_aware_flags.write_text(
            json.dumps(
                {
                    "id": "x",
                    "family": "equation_transform",
                    "messages": [{"role": "assistant", "content": "Final answer: \\boxed{1}"}],
                    "metadata": {
                        "expected_aware_teacher_signal": True,
                        "gate_rows_used_for_training": False,
                        "weak_gate_rows_used_for_training": False,
                        "full_gate_rows_used_for_training": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            count_and_audit_jsonl(expected_aware_flags, "expected_aware_flags")
        except RuntimeError as exc:
            if "expected-aware/reference-derived teacher rows" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test expected expected-aware teacher signal to fail")

        canonical_subcategory = tmp / "canonical_subcategory.jsonl"
        canonical_subcategory.write_text(
            json.dumps(
                {
                    "id": "x",
                    "family": "bit_manipulation",
                    "subcategory": "bit_bitpair_certified_source_only",
                    "messages": [{"role": "assistant", "content": "Final answer: \\boxed{00000000}"}],
                    "metadata": {
                        "subcategory": "bit_konbu_high_confidence_trace",
                        "gate_rows_used_for_training": False,
                        "weak_gate_rows_used_for_training": False,
                        "full_gate_rows_used_for_training": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        canonical_summary = count_and_audit_jsonl(canonical_subcategory, "canonical_subcategory")
        if canonical_summary["subcategory_counts"] != {"bit_bitpair_certified_source_only": 1}:
            raise RuntimeError(
                "self-test expected top-level canonical subcategory to override metadata subcategory: "
                + json.dumps(canonical_summary["subcategory_counts"], sort_keys=True)
            )
    print("hf_job_preflight_gate_self_test=ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(VALID_PHASES), default="preinstall")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    print("=== HF JOB PREFLIGHT GATE START ===", flush=True)
    print("generated_at_utc =", datetime.now(timezone.utc).isoformat(), flush=True)
    print("phase =", args.phase, flush=True)
    if args.self_test:
        self_test()
    else:
        run_phase(args.phase)
    print("=== HF JOB PREFLIGHT GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

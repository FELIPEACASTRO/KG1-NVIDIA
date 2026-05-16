#!/usr/bin/env python3
"""Static safety gate for KG1 scripts, HF job launchers, and notebooks.

This gate catches repository-level regressions that are cheaper to block before
running Colab, HF Jobs, or paid GPU work. It is intentionally conservative for
training/preference files: format-only negatives are allowed only in diagnostic
builders/gates, never in active HF jobs or notebooks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OLD_MIXED_V435E_PATH = "data/v435e_adapter_probe_preference/20260515T_v435e_from_h200_probe"
OLD_MIXED_V435E_TRAIN_SHA = "7f5e11770ac09c15e695cf4690df2fe7b5985b4a8b5bf5f8a201e8b71fe8be81"
OLD_MIXED_V435E_VAL_SHA = "d66752ab8470e145744a8bf80bc9b8beab7a4a3479d9161d03d6cc61c8ff9d92"

CRITICAL_SNIPPETS = {
    "scripts/build_v435e_adapter_probe_preference_dataset.py": {
        "correct rows excluded by default": "Correct adapter rows are not included by default",
        "diagnostic flag only": "--include-format-negatives",
        "include flag manifest": "\"include_format_negatives\": args.include_format_negatives",
        "format diagnostic warning": "format-only negatives are useful for a format audit",
    },
    "scripts/run_v435f_adapter_probe_preference_gate.py": {
        "format absence condition": "format_negatives_absent_for_preference",
        "allow flag": "--allow-format-negatives",
        "format row count": "format_negative_rows",
        "default hard-only path": "20260515T_v435e_hardneg_only",
    },
    "scripts/hf_job_train_v315_preference.py": {
        "format default false": "ALLOW_FORMAT_NEGATIVES = env_bool(\"ALLOW_FORMAT_NEGATIVES\", False)",
        "format rows blocked": "format_negative_blocked",
        "negative type accuracy": "negative_type_accuracy",
        "negative type from tokenized pair": "pair.get(\"negative_type\")",
        "boxed payload score modes": "BOXED_PAYLOAD_SCORE_MODES",
        "payload-only score mask": "build_boxed_payload_loss_mask",
        "score mask manifest": "\"score_mask_key\": score_mask_key()",
    },
    "scripts/kg1_pre_paid_job_integration_gate.py": {
        "dataset content audit": "audit_dataset_file",
        "target template check": "Final answer: \\\\boxed{",
        "blocked dataset marker gate": "BLOCKED_DATASET_MARKERS",
        "blocked adapter marker gate": "BLOCKED_ADAPTER_MARKERS",
        "data repo gate": "expected_data_repo",
        "command data repo export gate": "launcher_command_data_repo_export_mismatch",
        "crisis backfire guard": "launcher_missing_crisis_backfire_guard",
        "audit manifest gate": "hf_gpu_allowed_for_same_objective",
        "system prompt alignment gate": "launcher_system_prompt_not_final_answer_only",
        "h200 timeout gate": "launcher_timeout_not_one_hour",
        "first checkpoint eval gate": "launcher_missing_first_checkpoint_eval",
        "format negatives blocked": "launcher_allows_format_negatives",
    },
    "scripts/hf_job_preflight_gate.py": {
        "strict target modules check": "Init adapter target_modules mismatch",
        "strict target parameters check": "Init adapter target_parameters mismatch",
        "target parameter require check": "Init adapter has target_parameters but REQUIRE_LORA_TARGET_PARAMETER_MATCH is disabled",
        "gate row contamination flag": "weak_gate_rows_used_for_training",
        "gate row contamination fail": "gate/full/weak rows used for training",
        "missing gate flags fail": "missing required anti-leakage gate flags",
    },
    "scripts/hf_job_train_v90.py": {
        "target parameter alias matcher": "def target_parameter_name_matches",
        "gate-up alias target": "experts.gate_up_proj",
        "gate-up alias live name": ".up_proj.",
        "down alias live name": ".down_proj.",
        "target parameter trainability env": "REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE",
        "target parameter trainability tensors": "target_parameter_trainable_lora_tensors",
        "target parameter trainability mode": "target_parameters_trainability_mode",
        "manifest trainable filter report": "trainable_lora_module_filter",
        "default max length official": "MAX_LENGTH = env_int(\"MAX_LENGTH\", 8192)",
    },
    "scripts/package_hf_adapter_submission.py": {
        "official-like manifest schema required": "OFFICIAL_LIKE_SCHEMA_VERSION",
        "package threshold aligned": "--min-full-correct\", type=int, default=831",
        "immutable revision required": "missing immutable revision/resolved_revision",
        "adapter config hash check": "adapter_config sha mismatch",
        "adapter model hash check": "adapter_model sha mismatch",
        "official postprocessor rejected": "submission package cannot rely on external prediction postprocessor",
        "official-like control required": "full manifest missing official_like_control_gate",
        "official-like strict required": "official-like strict",
        "official gpu utilization required": "official-like gpu_memory_utilization",
        "manifest commit required": "full manifest missing repo_commit",
    },
    "src/competition_utils.py": {
        "expected-aware boxed extraction": "def extract_final_answer_for_expected",
        "literal closing brace guard": "immediately adjacent surplus braces",
        "escaped expected variant": "escaped_expected = escape_boxed_answer(expected_text)",
        "expected-aware debug warning": "submit-safe predictions",
        "expected-aware no prefix overcount": 'value[after] != "}"',
        "expected-aware uses strict verifier": "if verify_answer(expected_text, observed_text)",
    },
    "scripts/evaluate_lora_adapter.py": {
        "submit-safe extraction call": "prediction = extract_final_answer(raw_output)",
        "submit-safe prediction column": "submit_safe_label_free_prediction",
        "label-aware debug column": "label_aware_debug_prediction",
        "prediction metric mode": "\"prediction_metric_mode\": \"submit_safe_label_free\"",
    },
    "scripts/evaluate_lora_adapters_batch.py": {
        "submit-safe extraction call": "prediction = extract_final_answer(raw_output)",
        "submit-safe prediction column": "submit_safe_label_free_prediction",
        "label-aware debug column": "label_aware_debug_prediction",
        "prediction metric mode": "\"prediction_metric_mode\": \"submit_safe_label_free\"",
    },
    "scripts/audit_v505_label_free_candidate_revalidation.py": {
        "raw output label-free scoring": "extract_final_answer(raw_output)",
        "reference only blocked": "not_adapter_only_reference_solver_or_postprocessor",
        "adapter raw manifest field": "weak315_adapter_raw_scored",
    },
    "scripts/hf_job_weak_eval_v245.py": {
        "promotion equation floor": "KG1_WEAK_PROMOTE_EQUATION_MIN\", 60",
        "promotion total floor": "KG1_WEAK_PROMOTE_TOTAL_MIN\", 196",
        "promotion enforced by default": "not diagnostic_only",
        "official thinking default": "disable_thinking = env_bool(\"KG1_DISABLE_THINKING\", False)",
        "official token default": "KG1_MAX_TOKENS\", 7680",
        "official context default": "KG1_MAX_MODEL_LEN\", 8192",
    },
    "scripts/run_v286_generic_tokenization_gate.py": {
        "escaped boxed target": "box_answer(answer)",
        "expected-aware assistant extraction": "extract_final_answer_for_expected(assistant_content, answer)",
        "unescaped symbolic self-test": "unescaped symbolic boxed answer must fail",
    },
    "scripts/audit_v449_acc_metric_integrity.py": {
        "strict metric verifier": "verify_answer",
        "raw extraction audit": "raw_extraction_audit",
        "expected-aware delta": "expected_aware_minus_simple_correct",
        "no earlier boxed leakage self-test": "earlier_correct_later_wrong",
    },
    "scripts/hf_job_official_like_eval_gate_v284.py": {
        "failed gate exit hard": "official-like full eval gate failed; refusing successful exit",
        "failed gate override explicit": "KG1_ALLOW_FAILED_GATE_EXIT_0",
        "adapter config sha emitted": "adapter_config_sha256",
        "adapter model sha emitted": "adapter_model_sha256",
        "adapter resolved revision emitted": "resolved_revision",
        "official-like controls persisted": "\"official_like_control_gate\": official_like_control_gate",
    },
    "artifacts/v461_synthetic_numeric_probe_pack/build_v461_synthetic_numeric_probe_pack.py": {
        "raw probe fail closed": "\"hf_raw_probe_allowed\": False",
        "quarantine marker": "\"quarantined_after_v473\": True",
        "quarantine decision": "v461_quarantined_no_raw_probe",
    },
    "artifacts/v463_v462_synthetic_numeric_hard_negative_audit/build_v463_v462_synthetic_numeric_hard_negative_audit.py": {
        "dataset build fail closed": "v464_dataset_build_allowed = False",
        "quarantine condition": "\"route_not_quarantined_after_v473\": False",
        "quarantine decision": "v463_quarantined_signal_present_but_dataset_build_blocked",
    },
    "scripts/hf_job_full_eval_v276.py": {
        "failed gate exit hard": "full eval gate failed; refusing successful exit",
        "failed gate override explicit": "KG1_ALLOW_FAILED_GATE_EXIT_0",
    },
    "scripts/hf_job_weak_eval_v277_external_adapters.py": {
        "failed gate exit hard": "weak eval gate failed; refusing successful exit",
        "failed gate override explicit": "KG1_ALLOW_FAILED_GATE_EXIT_0",
    },
    "scripts/audit_v478_training_objective_alignment.py": {
        "effective family share": "effective_share_by_family",
        "bit effective floor": "min_bit_effective_share",
        "equation effective ceiling": "max_equation_effective_share",
        "gpu allowed decision": "hf_gpu_allowed",
    },
}

BLOCKED_TRAINING_DATASET_MARKERS = {
    "v461_synthetic_numeric_probe_pack": "V461 prompt pack contained a full-reference exact prompt/answer seed.",
    "v463_v462_synthetic_numeric_hard_negative_audit": "V463 depends on the quarantined V461/V462 numeric route.",
    "v464_v463_numeric_multirule_dataset": "V464 rejected candidates can equal the answer and is quarantined.",
    "v468_v464_symbol_fix_dataset": "V468 still contains a full-reference exact prompt/answer seed.",
    "v447_v446_trace_dataset": "Current V447 contains hypothesis_formed traces with contradictory boxed answers.",
}

BLOCKED_ADAPTER_MARKERS = {
    "kg1-nemotron-lora-v448-nemo-h200-v447-clean-trace-v290ckpt6": "Adapter was trained from quarantined V447 trace data.",
    "kg1-nemotron-lora-v465-v464-numeric-multirule-v290ckpt6": "Adapter was trained from quarantined V464 data.",
    "kg1-nemotron-lora-v469-v468-symbol-fix-v290ckpt6": "Adapter was trained from quarantined V468 data.",
    "kg1-nemotron-lora-v499-nemo-h200-v498-numeric-teacher-v290ckpt6": "V499 final eval regressed and answer-span weighting was inactive; forensics only.",
    "kg1-nemotron-lora-v501-nemo-h200-v498-answer-span-v290ckpt6": "V501 answer-span run was blocked by final eval regression; forensics only.",
}

TRUE_FORMAT_NEGATIVE_RE = re.compile(
    r"ALLOW_FORMAT_NEGATIVES\s*(?:=|:)\s*['\"]?(?:1|true|yes|on)['\"]?",
    re.IGNORECASE,
)
CLI_FORMAT_NEGATIVE_RE = re.compile(r"--(?:include|allow)-format-negatives\b")
EMPTY_LORA_TARGET_PARAMETERS_RE = re.compile(
    r"export\s+LORA_TARGET_PARAMETERS\s*=\s*(['\"]{2}|['\"]\s*['\"])",
    re.IGNORECASE,
)
DISABLED_TARGET_PARAMETER_MATCH_RE = re.compile(
    r"export\s+REQUIRE_LORA_TARGET_PARAMETER_MATCH\s*=\s*0\b",
    re.IGNORECASE,
)
EXPLICIT_TARGET_PARAMETER_TRAINABILITY_RE = re.compile(
    r"export\s+REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE\s*=\s*[01]\b",
    re.IGNORECASE,
)
ENABLED_TARGET_PARAMETER_TRAINABILITY_RE = re.compile(
    r"export\s+REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE\s*=\s*1\b",
    re.IGNORECASE,
)
TRAINABLE_LORA_MODULES_EXPORT_RE = re.compile(
    r"export\s+TRAINABLE_LORA_MODULES\s*=\s*['\"]([^'\"]*)['\"]",
    re.IGNORECASE,
)
HIGH_ANSWER_SPAN_LOSS_WEIGHT_RE = re.compile(
    r"ANSWER_SPAN_LOSS_WEIGHT\s*=\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
    re.IGNORECASE,
)
MANUAL_INIT_ADAPTER_LOAD_RE = re.compile(
    r"export\s+INIT_ADAPTER_LOAD_MODE\s*=\s*['\"]manual['\"]",
    re.IGNORECASE,
)
PRETOKENIZED_VAL_COPY_ONLY_TRUE_RE = re.compile(
    r"PRETOKENIZED_VAL_COPY_ONLY\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
    re.IGNORECASE,
)
CRISIS_BACKFIRE_GUARD_RE = re.compile(
    r"KG1_CRISIS_MODE_BACKFIRE_GUARD\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
    re.IGNORECASE,
)
WEAK_EVAL_DIAGNOSTIC_ONLY_RE = re.compile(
    r"KG1_WEAK_EVAL_DIAGNOSTIC_ONLY\s*(?:[\"']?\s*:\s*[\"']?(?:1|true|yes|on)|=\s*[\"']?(?:1|true|yes|on))",
    re.IGNORECASE,
)


@dataclass
class Finding:
    path: str
    level: str
    code: str
    detail: str


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def run_git(args: list[str], check: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stdout)
    return completed.stdout


def read_path_text(path: Path) -> str:
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8", errors="replace")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(str(item) for item in source))
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def is_scannable(path: Path) -> bool:
    rel = repo_rel(path)
    suffix = path.suffix.lower()
    if suffix not in {".py", ".ipynb", ".sh", ".yml", ".yaml"}:
        return False
    return (
        rel.startswith("scripts/")
        or rel.startswith("src/")
        or rel.startswith("notebooks/")
        or rel.startswith(".github/workflows/")
        or rel.startswith("artifacts/")
    )


def is_hf_job_or_notebook(path: Path, text: str) -> bool:
    rel = repo_rel(path)
    name = path.name.lower()
    if rel == "scripts/kg1_static_safety_gate.py":
        return False
    if path.suffix.lower() == ".ipynb":
        return True
    if "api.run_job(" in text or "HfApi(" in text or "huggingface_hub" in text:
        return True
    if rel.startswith("scripts/hf_job_"):
        return True
    if name.startswith("launch_") and "hf" in rel.lower():
        return True
    return False


def is_archived_fail_closed(text: str) -> bool:
    generic_archive = [
        "Archived KG1 launcher",
        "raise RuntimeError(",
        "quarantined",
        "fail-closed",
    ]
    if all(snippet in text for snippet in generic_archive):
        return True
    required = [
        "Archived V436 launcher",
        "raise RuntimeError(",
        "format-only negatives",
        "hard-negative-only V435E",
    ]
    return all(snippet in text for snippet in required)


def audit_text(path: Path, text: str) -> list[Finding]:
    rel = repo_rel(path)
    findings: list[Finding] = []
    job_or_notebook = is_hf_job_or_notebook(path, text)

    old_markers = [OLD_MIXED_V435E_PATH, OLD_MIXED_V435E_TRAIN_SHA, OLD_MIXED_V435E_VAL_SHA]
    if job_or_notebook and any(marker in text for marker in old_markers) and not is_archived_fail_closed(text):
        findings.append(
            Finding(
                rel,
                "error",
                "old_mixed_v435e_dataset_referenced",
                "Active job/notebook references archived V435E mixed preference data.",
            )
        )

    if job_or_notebook and TRUE_FORMAT_NEGATIVE_RE.search(text):
        findings.append(
            Finding(
                rel,
                "error",
                "allow_format_negatives_enabled",
                "Active job/notebook must not enable ALLOW_FORMAT_NEGATIVES.",
            )
        )

    if job_or_notebook and CLI_FORMAT_NEGATIVE_RE.search(text):
        findings.append(
            Finding(
                rel,
                "error",
                "format_negative_cli_in_active_job",
                "Active job/notebook must not pass --include-format-negatives or --allow-format-negatives.",
            )
        )

    if (
        job_or_notebook
        and not is_archived_fail_closed(text)
        and "INIT_ADAPTER_REPO" in text
        and EMPTY_LORA_TARGET_PARAMETERS_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "init_adapter_target_parameters_not_preserved",
                "HF training launchers with an init adapter must not blank LORA_TARGET_PARAMETERS; "
                "the V290/V291 lineage uses MoE target_parameters and losing them can change adapter behavior.",
            )
        )

    if (
        job_or_notebook
        and not is_archived_fail_closed(text)
        and "mlp.experts.gate_up_proj" in text
        and DISABLED_TARGET_PARAMETER_MATCH_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "lora_target_parameter_match_disabled",
                "Launchers that configure MoE target_parameters must keep REQUIRE_LORA_TARGET_PARAMETER_MATCH=1.",
            )
        )

    if (
        job_or_notebook
        and rel not in {"scripts/hf_job_train_v90.py", "scripts/kg1_static_safety_gate.py"}
        and not is_archived_fail_closed(text)
        and "mlp.experts.gate_up_proj" in text
        and ("TRAINABLE_LORA_MODULES" in text or "KG1_TRAINABLE_LORA_MODULES" in text)
        and not EXPLICIT_TARGET_PARAMETER_TRAINABILITY_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "target_parameter_trainability_not_explicit",
                "Launchers that combine MoE target_parameters with a trainable LoRA allowlist must set "
                "REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0 or 1 explicitly. This prevents confusing "
                "frozen-active target_parameters with actually trained target_parameters.",
            )
        )

    if (
        job_or_notebook
        and rel not in {"scripts/hf_job_train_v90.py", "scripts/kg1_static_safety_gate.py"}
        and not is_archived_fail_closed(text)
        and "mlp.experts.gate_up_proj" in text
        and ENABLED_TARGET_PARAMETER_TRAINABILITY_RE.search(text)
    ):
        trainable_module_exports = TRAINABLE_LORA_MODULES_EXPORT_RE.findall(text)
        trainable_modules_text = ",".join(trainable_module_exports)
        trainable_modules = {
            item.strip()
            for export in trainable_module_exports
            for item in export.split(",")
            if item.strip()
        }
        missing_moe_modules = sorted({"up_proj", "down_proj"} - trainable_modules)
        if missing_moe_modules:
            findings.append(
                Finding(
                    rel,
                    "error",
                    "trainable_target_parameters_missing_moe_modules",
                    "When REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1, TRAINABLE_LORA_MODULES must include "
                    f"up_proj and down_proj. Missing: {', '.join(missing_moe_modules)}. "
                    f"Observed TRAINABLE_LORA_MODULES={trainable_modules_text!r}",
                )
            )
        if "lm_head" in trainable_modules:
            findings.append(
                Finding(
                    rel,
                    "error",
                    "lm_head_trainable_in_moe_smoke",
                    "V491/V492 route requires lm_head frozen in the promotional MoE smoke; use a separate "
                    "documented ablation before re-enabling lm_head.",
                )
            )
        high_answer_span_match = HIGH_ANSWER_SPAN_LOSS_WEIGHT_RE.search(text)
        answer_span_weight = float(high_answer_span_match.group(1)) if high_answer_span_match else 1.0
        explicit_answer_span_route = (
            "answer_span_weighted" in text
            or "answer-span" in rel.lower()
            or "ANSWER_SPAN_MIN_WEIGHTED_TOKENS" in text
        )
        if high_answer_span_match and answer_span_weight != 1.0 and not explicit_answer_span_route:
            findings.append(
                Finding(
                    rel,
                    "error",
                    "high_answer_span_loss_weight_in_moe_smoke",
                    "Generic promotional MoE smokes require ANSWER_SPAN_LOSS_WEIGHT=1.0. "
                    "Use an explicitly named answer-span route with an answer-span min-token gate "
                    f"before raising it. Observed numeric value {high_answer_span_match.group(1)}.",
                )
            )
        if high_answer_span_match and answer_span_weight > 1.0 and explicit_answer_span_route:
            min_tokens_match = re.search(
                r"ANSWER_SPAN_MIN_WEIGHTED_TOKENS\s*=\s*['\"]?([0-9]+)['\"]?",
                text,
            )
            if not min_tokens_match or int(min_tokens_match.group(1)) <= 0:
                findings.append(
                    Finding(
                        rel,
                        "error",
                        "answer_span_weight_missing_min_token_gate",
                        "Explicit answer-span routes with ANSWER_SPAN_LOSS_WEIGHT>1.0 must set "
                        "ANSWER_SPAN_MIN_WEIGHTED_TOKENS to a positive value so inactive weighting "
                        "cannot silently pass.",
                    )
                )

    if (
        job_or_notebook
        and not is_archived_fail_closed(text)
        and ("mlp.experts.gate_up_proj" in text or "KG1_LORA_TARGET_PARAMETERS" in text)
        and MANUAL_INIT_ADAPTER_LOAD_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "manual_init_adapter_load_with_target_parameters",
                "Launchers with MoE LORA_TARGET_PARAMETERS must use the PEFT-native "
                "PeftModel.from_pretrained path unless a dedicated CPU round-trip gate proves "
                "manual state_dict injection is equivalent.",
            )
        )

    if (
        job_or_notebook
        and rel != "scripts/hf_job_train_v90.py"
        and not is_archived_fail_closed(text)
        and PRETOKENIZED_VAL_COPY_ONLY_TRUE_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "pretokenized_val_copy_only_enabled",
                "PRETOKENIZED_VAL_COPY_ONLY=1 reuses train rows as validation and makes eval_loss non-independent. "
                "It is diagnostic-only and must not appear in promotional HF jobs/notebooks.",
            )
        )

    if (
        job_or_notebook
        and rel not in {
            "scripts/hf_job_train_v90.py",
            "scripts/hf_job_weak_eval_v245.py",
            "scripts/hf_job_weak_eval_v277_external_adapters.py",
            "scripts/hf_job_full_eval_v276.py",
            "scripts/hf_job_official_like_eval_gate_v284.py",
            "scripts/hf_job_preflight_gate.py",
        }
        and not is_archived_fail_closed(text)
        and ("api.run_job(" in text or "HfApi(" in text)
        and ("FLAVOR = \"h200\"" in text or "FLAVOR = \"a100\"" in text or "FLAVOR = \"a10g" in text)
        and not CRISIS_BACKFIRE_GUARD_RE.search(text)
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "missing_crisis_mode_backfire_guard",
                "Paid HF/Kaggle launchers must set KG1_CRISIS_MODE_BACKFIRE_GUARD=1. "
                "This records that F2/backfire/silent-bug checks, blocked adapter/dataset lists, "
                "label-free metric gates, and FinOps kill-switches are intentional for this run.",
            )
        )

    if (
        job_or_notebook
        and rel != "scripts/hf_job_weak_eval_v245.py"
        and not is_archived_fail_closed(text)
        and "hf_job_weak_eval_v245.py" in text
        and not WEAK_EVAL_DIAGNOSTIC_ONLY_RE.search(text)
    ):
        weak_eval_required_controls = {
            "KG1_DISABLE_THINKING=0": r"[\"']KG1_DISABLE_THINKING[\"']\s*:\s*[\"']0[\"']|export\s+KG1_DISABLE_THINKING\s*=\s*0\b",
            "KG1_NO_PROMPT_SUFFIX=0": r"[\"']KG1_NO_PROMPT_SUFFIX[\"']\s*:\s*[\"']0[\"']|export\s+KG1_NO_PROMPT_SUFFIX\s*=\s*0\b",
            "KG1_MAX_TOKENS=7680": r"[\"']KG1_MAX_TOKENS[\"']\s*:\s*[\"']7680[\"']|export\s+KG1_MAX_TOKENS\s*=\s*7680\b",
            "KG1_MAX_MODEL_LEN=8192": r"[\"']KG1_MAX_MODEL_LEN[\"']\s*:\s*[\"']8192[\"']|export\s+KG1_MAX_MODEL_LEN\s*=\s*8192\b",
            "KG1_MAX_NUM_SEQS=64": r"[\"']KG1_MAX_NUM_SEQS[\"']\s*:\s*[\"']64[\"']|export\s+KG1_MAX_NUM_SEQS\s*=\s*64\b",
        }
        missing_controls = [
            name
            for name, pattern in weak_eval_required_controls.items()
            if not re.search(pattern, text)
        ]
        if missing_controls:
            findings.append(
                Finding(
                    rel,
                    "error",
                    "weak_eval_not_official_like",
                    "Promotional weak eval launchers must override hf_job_weak_eval_v245.py diagnostic defaults. "
                    "Missing controls: " + ", ".join(missing_controls) + ". "
                    "Set KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1 only for explicit non-promotional sweeps.",
                )
            )

    if (
        job_or_notebook
        and not is_archived_fail_closed(text)
        and rel != "scripts/hf_job_train_v90.py"
        and "bit_manipulation" in text
        and "equation_transform" in text
        and ("SOURCE_WEIGHTS" in text or "KG1_SOURCE_WEIGHTS" in text)
        and ("SUBCATEGORY_WEIGHTS" in text or "KG1_SUBCATEGORY_WEIGHTS" in text)
        and "audit_v478_training_objective_alignment.py" not in text
        and "objective_alignment" not in text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "missing_v478_objective_alignment_gate",
                "Weighted bit+equation HF job/notebook must run the V478 objective-alignment gate before GPU.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and "official_correct" in text and "answers_equivalent(" in text:
        findings.append(
            Finding(
                rel,
                "error",
                "permissive_metric_used_for_official_correct",
                "Official ACC diagnostics must use verify_answer, not answers_equivalent; numeric tolerance overcounts binary strings.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"\bprediction\s*=\s*\(?\s*extract_final_answer_for_expected\s*\(",
        text,
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "label_aware_prediction_used_for_submit_safe_metric",
                "Expected-aware extraction may be logged only as label_aware_debug_prediction; "
                "submit-safe prediction must use extract_final_answer(raw_output).",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"KG1_WEAK_PROMOTE_EQUATION_MIN[\"']?\s*,\s*57\b|KG1_WEAK_PROMOTE_EQUATION_MIN[\"']?\s*[,=]\s*57\b",
        text,
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_equation_env_gate",
                "Weak promotion env/default gate must use the current equation floor 60, not 57.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"KG1_WEAK_PROMOTE_TOTAL_MIN[\"']?\s*,\s*193\b|KG1_WEAK_PROMOTE_TOTAL_MIN[\"']?\s*[,=]\s*193\b",
        text,
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_total_env_gate",
                "Weak promotion env/default gate must use the submit-safe floor 196, not 193.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"\bWEAK_BIT_MIN_FOR_FULL\s*=\s*133\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_bit_gate",
                "Weak promotion gate must use the current no-regression bit floor: WEAK_BIT_MIN_FOR_FULL = 136.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"\bWEAK_MAX_TRUNC_FOR_FULL\s*=\s*3\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_trunc_gate",
                "Weak promotion gate must use current no-truncation floor: WEAK_MAX_TRUNC_FOR_FULL = 0.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"\bKG1_WEAK_(?:PROMOTE_)?BIT_MIN[\"']?\s*[,=]\s*133\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_bit_env_gate",
                "Weak promotion env/default gate must use bit floor 136, not 133.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"\bKG1_WEAK_(?:PROMOTE_)?TRUNC_MAX[\"']?\s*[,=]\s*3\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_trunc_env_gate",
                "Weak promotion env/default gate must use truncation cap 0, not 3.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"add_argument\(\s*[\"']--weak-bit-min[\"'][\s\S]{0,160}default\s*=\s*133\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_bit_argparse_default",
                "Argparse default for --weak-bit-min must be 136, not 133.",
            )
        )

    if rel != "scripts/kg1_static_safety_gate.py" and re.search(
        r"add_argument\(\s*[\"']--weak-trunc-max[\"'][\s\S]{0,160}default\s*=\s*3\b", text
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_weak_trunc_argparse_default",
                "Argparse default for --weak-trunc-max must be 0, not 3.",
            )
        )

    if rel == "scripts/package_hf_adapter_submission.py" and re.search(
        r"add_argument\(\s*[\"']--min-full-correct[\"'][\s\S]{0,160}default\s*=\s*(?:82[0-9]|830)\b",
        text,
    ):
        findings.append(
            Finding(
                rel,
                "error",
                "stale_package_full_correct_default",
                "Package default must require current official-like full floor 831, not an older 823/824 threshold.",
            )
        )

    if job_or_notebook and rel != "scripts/hf_job_preflight_gate.py":
        for marker, reason in BLOCKED_TRAINING_DATASET_MARKERS.items():
            if marker in text and not is_archived_fail_closed(text):
                findings.append(
                    Finding(
                        rel,
                        "error",
                        "blocked_training_dataset_referenced",
                        f"{marker}: {reason}",
                    )
                )
        for marker, reason in BLOCKED_ADAPTER_MARKERS.items():
            if marker in text and not is_archived_fail_closed(text):
                findings.append(
                    Finding(
                        rel,
                        "error",
                        "blocked_adapter_referenced",
                        f"{marker}: {reason}",
                    )
                )

    for critical_rel, snippets in CRITICAL_SNIPPETS.items():
        if rel != critical_rel:
            continue
        for name, snippet in snippets.items():
            if snippet not in text:
                findings.append(Finding(rel, "error", "critical_safety_snippet_missing", name))
    return findings


def discover_changed_paths(from_ref: str | None, to_ref: str) -> list[Path]:
    if from_ref:
        output = run_git(["diff", "--name-only", "--diff-filter=ACMRT", from_ref, to_ref], check=False)
        raw = [line.strip() for line in output.splitlines() if line.strip()]
    else:
        output = run_git(["status", "--short"], check=False)
        raw = []
        for line in output.splitlines():
            if not line.strip():
                continue
            raw.append(line[3:].strip())
    return sorted({ROOT / item for item in raw if (ROOT / item).exists()})


def load_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.paths_file:
        for line in args.paths_file.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                paths.append(ROOT / item if not Path(item).is_absolute() else Path(item))
    if args.paths:
        paths.extend(path if path.is_absolute() else ROOT / path for path in args.paths)
    if not paths:
        paths = discover_changed_paths(args.changed_from or None, args.changed_to)
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(item for item in path.rglob("*") if item.is_file())
        else:
            expanded.append(path)
    return sorted({path for path in expanded if path.exists() and is_scannable(path)})


def audit_paths(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        try:
            findings.extend(audit_text(path, read_path_text(path)))
        except Exception as exc:
            findings.append(Finding(repo_rel(path), "error", "static_safety_read_failed", repr(exc)))
    for critical_rel, snippets in CRITICAL_SNIPPETS.items():
        critical_path = ROOT / critical_rel
        if critical_path.exists() and critical_path not in paths:
            text = critical_path.read_text(encoding="utf-8", errors="replace")
            for name, snippet in snippets.items():
                if snippet not in text:
                    findings.append(Finding(critical_rel, "error", "critical_safety_snippet_missing", name))
    return findings


def run_self_test() -> int:
    print("=== KG1 STATIC SAFETY GATE SELF TEST START ===", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        bad = tmp / "launch_bad_hf.py"
        bad.write_text(
            "from huggingface_hub import HfApi\n"
            f"DATA_ROOT='{OLD_MIXED_V435E_PATH}'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        archived = tmp / "launch_archived_hf.py"
        archived.write_text(
            '"""Archived V436 launcher with format-only negatives and hard-negative-only V435E."""\n'
            f"DATA_ROOT='{OLD_MIXED_V435E_PATH}'\n"
            "def main():\n    raise RuntimeError('Archived launcher: hard-negative-only V435E required')\n",
            encoding="utf-8",
        )
        enabled = tmp / "job_enabled.py"
        enabled.write_text("from huggingface_hub import HfApi\nALLOW_FORMAT_NEGATIVES=1\n", encoding="utf-8")

        bad_findings = audit_text(bad, bad.read_text(encoding="utf-8"))
        if "old_mixed_v435e_dataset_referenced" not in {item.code for item in bad_findings}:
            print("missing old mixed dataset self-test finding", flush=True)
            return 1
        archived_findings = audit_text(archived, archived.read_text(encoding="utf-8"))
        if archived_findings:
            print(json.dumps([item.__dict__ for item in archived_findings], indent=2), flush=True)
            return 1
        enabled_findings = audit_text(enabled, enabled.read_text(encoding="utf-8"))
        if "allow_format_negatives_enabled" not in {item.code for item in enabled_findings}:
            print("missing ALLOW_FORMAT_NEGATIVES self-test finding", flush=True)
            return 1
        permissive_metric = tmp / "diag_metric.py"
        permissive_metric.write_text(
            "from src.competition_utils import answers_equivalent\n"
            "df['official_correct'] = df.apply(lambda row: answers_equivalent(row['answer'], row['prediction']), axis=1)\n",
            encoding="utf-8",
        )
        metric_findings = audit_text(permissive_metric, permissive_metric.read_text(encoding="utf-8"))
        if "permissive_metric_used_for_official_correct" not in {item.code for item in metric_findings}:
            print("missing permissive metric self-test finding", flush=True)
            return 1
        label_aware_metric = tmp / "label_aware_metric.py"
        label_aware_metric.write_text(
            "from src.competition_utils import extract_final_answer_for_expected\n"
            "prediction = extract_final_answer_for_expected(raw_output, expected)\n",
            encoding="utf-8",
        )
        label_aware_findings = audit_text(label_aware_metric, label_aware_metric.read_text(encoding="utf-8"))
        if "label_aware_prediction_used_for_submit_safe_metric" not in {
            item.code for item in label_aware_findings
        }:
            print("missing label-aware submit-safe metric self-test finding", flush=True)
            return 1
        stale_gate = tmp / "build_old_gate.py"
        stale_gate.write_text(
            "WEAK_BIT_MIN_FOR_FULL = 133\nWEAK_MAX_TRUNC_FOR_FULL = 3\n"
            "KG1_WEAK_BIT_MIN\", 133\nKG1_WEAK_TRUNC_MAX\", 3\n",
            encoding="utf-8",
        )
        stale_weak_promote = tmp / "run_old_weak_promote.py"
        stale_weak_promote.write_text(
            "equation_min = env_int(\"KG1_WEAK_PROMOTE_EQUATION_MIN\", 57)\n"
            "total_min = env_int(\"KG1_WEAK_PROMOTE_TOTAL_MIN\", 193)\n",
            encoding="utf-8",
        )
        stale_argparse = tmp / "run_old_argparse.py"
        stale_argparse.write_text(
            "parser.add_argument(\"--weak-bit-min\", type=int, default=133)\n"
            "parser.add_argument(\"--weak-trunc-max\", type=int, default=3)\n",
            encoding="utf-8",
        )
        stale_gate_findings = audit_text(stale_gate, stale_gate.read_text(encoding="utf-8"))
        stale_gate_findings.extend(audit_text(stale_weak_promote, stale_weak_promote.read_text(encoding="utf-8")))
        stale_gate_findings.extend(audit_text(stale_argparse, stale_argparse.read_text(encoding="utf-8")))
        stale_codes = {item.code for item in stale_gate_findings}
        if not {
            "stale_weak_bit_gate",
            "stale_weak_trunc_gate",
            "stale_weak_bit_env_gate",
            "stale_weak_trunc_env_gate",
            "stale_weak_equation_env_gate",
            "stale_weak_total_env_gate",
            "stale_weak_bit_argparse_default",
            "stale_weak_trunc_argparse_default",
        }.issubset(stale_codes):
            print("missing stale weak gate self-test finding", flush=True)
            return 1
        blocked_dataset = tmp / "job_blocked_dataset.py"
        blocked_dataset.write_text(
            "from huggingface_hub import HfApi\n"
            "DATA_FILE='data/v468_v464_symbol_fix_dataset/train.jsonl'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        blocked_findings = audit_text(blocked_dataset, blocked_dataset.read_text(encoding="utf-8"))
        if "blocked_training_dataset_referenced" not in {item.code for item in blocked_findings}:
            print("missing blocked training dataset self-test finding", flush=True)
            return 1
        blocked_adapter = tmp / "job_blocked_adapter.py"
        blocked_adapter.write_text(
            "from huggingface_hub import HfApi\n"
            "ADAPTER_REPO='felipesp1983/kg1-nemotron-lora-v469-v468-symbol-fix-v290ckpt6'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        blocked_adapter_findings = audit_text(blocked_adapter, blocked_adapter.read_text(encoding="utf-8"))
        if "blocked_adapter_referenced" not in {item.code for item in blocked_adapter_findings}:
            print("missing blocked adapter self-test finding", flush=True)
            return 1
        weighted_without_objective_gate = tmp / "launch_weighted_hf.py"
        weighted_without_objective_gate.write_text(
            "from huggingface_hub import HfApi\n"
            "KG1_REQUIRED_TRAIN_FAMILIES='bit_manipulation,equation_transform'\n"
            "KG1_SOURCE_WEIGHTS='equation=8,bit=1'\n"
            "KG1_SUBCATEGORY_WEIGHTS='equation_transform=12,bit_manipulation=1'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        weighted_gate_findings = audit_text(
            weighted_without_objective_gate,
            weighted_without_objective_gate.read_text(encoding="utf-8"),
        )
        if "missing_v478_objective_alignment_gate" not in {item.code for item in weighted_gate_findings}:
            print("missing V478 objective alignment self-test finding", flush=True)
            return 1
        weighted_with_objective_gate = tmp / "launch_weighted_hf_checked.py"
        weighted_with_objective_gate.write_text(
            "from huggingface_hub import HfApi\n"
            "KG1_REQUIRED_TRAIN_FAMILIES='bit_manipulation,equation_transform'\n"
            "KG1_SOURCE_WEIGHTS='equation=8,bit=1'\n"
            "KG1_SUBCATEGORY_WEIGHTS='equation_transform=12,bit_manipulation=1'\n"
            "OBJECTIVE_ALIGNMENT_GATE='scripts/audit_v478_training_objective_alignment.py'\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        weighted_checked_findings = audit_text(
            weighted_with_objective_gate,
            weighted_with_objective_gate.read_text(encoding="utf-8"),
        )
        if "missing_v478_objective_alignment_gate" in {item.code for item in weighted_checked_findings}:
            print("false positive V478 objective alignment self-test finding", flush=True)
            return 1
        missing_target_parameters = tmp / "launch_missing_target_parameters.py"
        missing_target_parameters.write_text(
            "from huggingface_hub import HfApi\n"
            "INIT_ADAPTER_REPO='felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke'\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS=''\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=0\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        target_findings = audit_text(
            missing_target_parameters,
            missing_target_parameters.read_text(encoding="utf-8"),
        )
        if "init_adapter_target_parameters_not_preserved" not in {item.code for item in target_findings}:
            print("missing init adapter target_parameters preservation self-test finding", flush=True)
            return 1
        disabled_target_match = tmp / "launch_disabled_target_match.py"
        disabled_target_match.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=0\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        disabled_match_findings = audit_text(
            disabled_target_match,
            disabled_target_match.read_text(encoding="utf-8"),
        )
        if "lora_target_parameter_match_disabled" not in {item.code for item in disabled_match_findings}:
            print("missing disabled target-parameter match self-test finding", flush=True)
            return 1
        implicit_target_trainability = tmp / "launch_implicit_target_trainability.py"
        implicit_target_trainability.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        implicit_trainability_findings = audit_text(
            implicit_target_trainability,
            implicit_target_trainability.read_text(encoding="utf-8"),
        )
        if "target_parameter_trainability_not_explicit" not in {
            item.code for item in implicit_trainability_findings
        }:
            print("missing target-parameter trainability self-test finding", flush=True)
            return 1
        explicit_target_trainability = tmp / "launch_explicit_target_trainability.py"
        explicit_target_trainability.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        explicit_trainability_findings = audit_text(
            explicit_target_trainability,
            explicit_target_trainability.read_text(encoding="utf-8"),
        )
        if "target_parameter_trainability_not_explicit" in {
            item.code for item in explicit_trainability_findings
        }:
            print("false positive target-parameter trainability self-test finding", flush=True)
            return 1
        p3_missing_moe_modules = tmp / "launch_p3_missing_moe_modules.py"
        p3_missing_moe_modules.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1\n"
            "export ANSWER_SPAN_LOSS_WEIGHT='1.0'\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        p3_missing_findings = audit_text(
            p3_missing_moe_modules,
            p3_missing_moe_modules.read_text(encoding="utf-8"),
        )
        p3_missing_codes = {item.code for item in p3_missing_findings}
        if "trainable_target_parameters_missing_moe_modules" not in p3_missing_codes:
            print("missing P3 trainable target-parameter module self-test finding", flush=True)
            return 1
        if "lm_head_trainable_in_moe_smoke" not in p3_missing_codes:
            print("missing P3 lm_head frozen self-test finding", flush=True)
            return 1
        p3_high_answer_span = tmp / "launch_p3_high_answer_span.py"
        p3_high_answer_span.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1\n"
            "export ANSWER_SPAN_LOSS_WEIGHT='12.0'\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        p3_high_findings = audit_text(
            p3_high_answer_span,
            p3_high_answer_span.read_text(encoding="utf-8"),
        )
        if "high_answer_span_loss_weight_in_moe_smoke" not in {item.code for item in p3_high_findings}:
            print("missing P3 answer-span loss weight self-test finding", flush=True)
            return 1
        p3_answer_span_allowed = tmp / "launch_answer_span_weighted.py"
        p3_answer_span_allowed.write_text(
            "from huggingface_hub import HfApi\n"
            "VERSION='v501_answer_span_weighted'\n"
            "ANSWER_SPAN_MIN_WEIGHTED_TOKENS='1000'\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1\n"
            "export ANSWER_SPAN_LOSS_WEIGHT='4.0'\n"
            "export ANSWER_SPAN_MIN_WEIGHTED_TOKENS='1000'\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        p3_answer_span_allowed_findings = audit_text(
            p3_answer_span_allowed,
            p3_answer_span_allowed.read_text(encoding="utf-8"),
        )
        p3_answer_span_allowed_codes = {item.code for item in p3_answer_span_allowed_findings}
        if {
            "high_answer_span_loss_weight_in_moe_smoke",
            "answer_span_weight_missing_min_token_gate",
        } & p3_answer_span_allowed_codes:
            print(
                "false positive answer-span weighted route self-test finding",
                json.dumps([item.__dict__ for item in p3_answer_span_allowed_findings], indent=2),
                flush=True,
            )
            return 1
        p3_answer_span_no_min = tmp / "launch_answer_span_weighted_no_min.py"
        p3_answer_span_no_min.write_text(
            "from huggingface_hub import HfApi\n"
            "VERSION='v501_answer_span_weighted'\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1\n"
            "export ANSWER_SPAN_LOSS_WEIGHT='4.0'\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        p3_answer_span_no_min_findings = audit_text(
            p3_answer_span_no_min,
            p3_answer_span_no_min.read_text(encoding="utf-8"),
        )
        if "answer_span_weight_missing_min_token_gate" not in {
            item.code for item in p3_answer_span_no_min_findings
        }:
            print("missing answer-span min-token gate self-test finding", flush=True)
            return 1
        p3_valid_moe_smoke = tmp / "launch_p3_valid_moe_smoke.py"
        p3_valid_moe_smoke.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "export REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1\n"
            "export ANSWER_SPAN_LOSS_WEIGHT='1.0'\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        p3_valid_findings = audit_text(
            p3_valid_moe_smoke,
            p3_valid_moe_smoke.read_text(encoding="utf-8"),
        )
        p3_forbidden_codes = {
            "trainable_target_parameters_missing_moe_modules",
            "lm_head_trainable_in_moe_smoke",
            "high_answer_span_loss_weight_in_moe_smoke",
        }
        if p3_forbidden_codes & {item.code for item in p3_valid_findings}:
            print(json.dumps([item.__dict__ for item in p3_valid_findings], indent=2), flush=True)
            return 1
        manual_target_parameter_load = tmp / "launch_manual_target_parameter_load.py"
        manual_target_parameter_load.write_text(
            "from huggingface_hub import HfApi\n"
            "KG1_LORA_TARGET_PARAMETERS='mlp.experts.gate_up_proj,mlp.experts.down_proj'\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export INIT_ADAPTER_LOAD_MODE='manual'\n"
            "export LORA_TARGET_PARAMETERS=\"$KG1_LORA_TARGET_PARAMETERS\"\n"
            "export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        manual_target_findings = audit_text(
            manual_target_parameter_load,
            manual_target_parameter_load.read_text(encoding="utf-8"),
        )
        if "manual_init_adapter_load_with_target_parameters" not in {item.code for item in manual_target_findings}:
            print("missing manual target-parameter load self-test finding", flush=True)
            return 1
        pretokenized_val_copy = tmp / "launch_pretokenized_copy.py"
        pretokenized_val_copy.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT=\"\"\"\n"
            "export PRETOKENIZED_VAL_COPY_ONLY=1\n"
            "\"\"\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        pretokenized_findings = audit_text(
            pretokenized_val_copy,
            pretokenized_val_copy.read_text(encoding="utf-8"),
        )
        if "pretokenized_val_copy_only_enabled" not in {item.code for item in pretokenized_findings}:
            print("missing pretokenized validation copy self-test finding", flush=True)
            return 1
        weak_eval_short = tmp / "launch_weak_short.py"
        weak_eval_short.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT='python3 scripts/hf_job_weak_eval_v245.py'\n"
            "job_env={'KG1_DISABLE_THINKING':'1','KG1_MAX_TOKENS':'96'}\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        weak_short_findings = audit_text(weak_eval_short, weak_eval_short.read_text(encoding="utf-8"))
        if "weak_eval_not_official_like" not in {item.code for item in weak_short_findings}:
            print("missing weak eval official-like self-test finding", flush=True)
            return 1
        weak_eval_diag = tmp / "launch_weak_diag.py"
        weak_eval_diag.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT='python3 scripts/hf_job_weak_eval_v245.py'\n"
            "KG1_WEAK_EVAL_DIAGNOSTIC_ONLY='1'\n"
            "job_env={'KG1_DISABLE_THINKING':'1','KG1_MAX_TOKENS':'96'}\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        weak_diag_findings = audit_text(weak_eval_diag, weak_eval_diag.read_text(encoding="utf-8"))
        if "weak_eval_not_official_like" in {item.code for item in weak_diag_findings}:
            print("false positive weak eval diagnostic-only self-test finding", flush=True)
            return 1
        weak_eval_official = tmp / "launch_weak_official.py"
        weak_eval_official.write_text(
            "from huggingface_hub import HfApi\n"
            "COMMAND_SCRIPT='python3 scripts/hf_job_weak_eval_v245.py'\n"
            "FLAVOR = \"h200\"\n"
            "KG1_CRISIS_MODE_BACKFIRE_GUARD=1\n"
            "job_env={"
            "'KG1_DISABLE_THINKING':'0',"
            "'KG1_NO_PROMPT_SUFFIX':'0',"
            "'KG1_MAX_TOKENS':'7680',"
            "'KG1_MAX_MODEL_LEN':'8192',"
            "'KG1_MAX_NUM_SEQS':'64'"
            "}\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        weak_official_findings = audit_text(weak_eval_official, weak_eval_official.read_text(encoding="utf-8"))
        if "weak_eval_not_official_like" in {item.code for item in weak_official_findings}:
            print(json.dumps([item.__dict__ for item in weak_official_findings], indent=2), flush=True)
            return 1
        missing_crisis_guard = tmp / "launch_missing_crisis_guard.py"
        missing_crisis_guard.write_text(
            "from huggingface_hub import HfApi\n"
            "FLAVOR = \"h200\"\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        missing_crisis_findings = audit_text(
            missing_crisis_guard,
            missing_crisis_guard.read_text(encoding="utf-8"),
        )
        if "missing_crisis_mode_backfire_guard" not in {item.code for item in missing_crisis_findings}:
            print("missing crisis-mode backfire guard self-test finding", flush=True)
            return 1
        with_crisis_guard = tmp / "launch_with_crisis_guard.py"
        with_crisis_guard.write_text(
            "from huggingface_hub import HfApi\n"
            "FLAVOR = \"h200\"\n"
            "KG1_CRISIS_MODE_BACKFIRE_GUARD=1\n"
            "HfApi().run_job(command=['true'])\n",
            encoding="utf-8",
        )
        with_crisis_findings = audit_text(with_crisis_guard, with_crisis_guard.read_text(encoding="utf-8"))
        if "missing_crisis_mode_backfire_guard" in {item.code for item in with_crisis_findings}:
            print("false positive crisis-mode backfire guard self-test finding", flush=True)
            return 1
        archived_quarantine = tmp / "launch_archived_quarantine.py"
        archived_quarantine.write_text(
            '"""Archived KG1 launcher for quarantined route; fail-closed."""\n'
            "def main():\n"
            "    raise RuntimeError('Archived KG1 launcher: quarantined route; fail-closed')\n",
            encoding="utf-8",
        )
        archived_quarantine_findings = audit_text(archived_quarantine, archived_quarantine.read_text(encoding="utf-8"))
        if archived_quarantine_findings:
            print(json.dumps([item.__dict__ for item in archived_quarantine_findings], indent=2), flush=True)
            return 1
    print("kg1_static_safety_gate_self_test=ok", flush=True)
    print("=== KG1 STATIC SAFETY GATE SELF TEST END ===", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files to audit.")
    parser.add_argument("--paths-file", type=Path, default=None, help="File containing repo-relative paths to audit.")
    parser.add_argument("--changed-from", default="", help="Git ref/sha to diff from.")
    parser.add_argument("--changed-to", default="HEAD", help="Git ref/sha to diff to.")
    parser.add_argument("--allow-empty", action="store_true", help="Return success when no scannable files are selected.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    paths = load_paths(args)
    findings = audit_paths(paths)
    if not paths and not args.allow_empty:
        findings.append(
            Finding(
                "",
                "error",
                "no_scannable_files_selected",
                "Pass files, --paths-file, --changed-from, or --allow-empty explicitly.",
            )
        )
    report: dict[str, Any] = {
        "schema_version": "kg1_static_safety_gate_v1",
        "ok": not any(item.level == "error" for item in findings),
        "file_count": len(paths),
        "files": [repo_rel(path) for path in paths],
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

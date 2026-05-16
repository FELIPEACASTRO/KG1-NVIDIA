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
    },
    "scripts/evaluate_lora_adapter.py": {
        "expected-aware extraction import": "extract_final_answer_for_expected",
        "expected-aware extraction call": "extract_final_answer_for_expected(raw_output, expected)",
    },
    "scripts/evaluate_lora_adapters_batch.py": {
        "expected-aware extraction import": "extract_final_answer_for_expected",
        "expected-aware extraction call": "extract_final_answer_for_expected(raw_output, expected)",
    },
    "scripts/run_v286_generic_tokenization_gate.py": {
        "escaped boxed target": "box_answer(answer)",
        "expected-aware assistant extraction": "extract_final_answer_for_expected(assistant_content, answer)",
        "unescaped symbolic self-test": "unescaped symbolic boxed answer must fail",
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
    "scripts/hf_job_train_v90.py": {
        "default max length official": "MAX_LENGTH = env_int(\"MAX_LENGTH\", 8192)",
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
        and not is_archived_fail_closed(text)
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
        stale_gate = tmp / "build_old_gate.py"
        stale_gate.write_text(
            "WEAK_BIT_MIN_FOR_FULL = 133\nWEAK_MAX_TRUNC_FOR_FULL = 3\n"
            "KG1_WEAK_BIT_MIN\", 133\nKG1_WEAK_TRUNC_MAX\", 3\n",
            encoding="utf-8",
        )
        stale_argparse = tmp / "run_old_argparse.py"
        stale_argparse.write_text(
            "parser.add_argument(\"--weak-bit-min\", type=int, default=133)\n"
            "parser.add_argument(\"--weak-trunc-max\", type=int, default=3)\n",
            encoding="utf-8",
        )
        stale_gate_findings = audit_text(stale_gate, stale_gate.read_text(encoding="utf-8"))
        stale_gate_findings.extend(audit_text(stale_argparse, stale_argparse.read_text(encoding="utf-8")))
        stale_codes = {item.code for item in stale_gate_findings}
        if not {
            "stale_weak_bit_gate",
            "stale_weak_trunc_gate",
            "stale_weak_bit_env_gate",
            "stale_weak_trunc_env_gate",
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

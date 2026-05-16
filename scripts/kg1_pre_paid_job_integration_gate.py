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
    require_regex(text, r"MAX_STEPS\s*=\s*(?:[1-9]|1[0-2])\b", "launcher_max_steps_too_high", findings)
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
        "contains_h200": 'FLAVOR = "h200"' in text,
        "contains_timeout_3600": "timeout=3600" in text,
    }


def _metadata_flag_is_false(row: dict[str, Any], metadata: dict[str, Any], flag: str) -> bool:
    if flag in row:
        return row.get(flag) is False
    if flag in metadata:
        return metadata.get(flag) is False
    return True


def audit_dataset_file(
    path: Path,
    expected_sha: str,
    expected_rows: int,
    split: str,
    findings: list[Finding],
    *,
    dataset_schema: str,
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
    bad_rows: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", ""))
        metadata = row.get("metadata") or {}
        if not row_id or row_id in ids:
            bad_rows.append(f"{index}:duplicate_or_missing_id:{row_id}")
        ids.add(row_id)
        family_counts[str(row.get("family") or metadata.get("family") or "unknown")] += 1
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
                extracted = extract_final_answer(assistant_content)
                if not verify_answer(answer, extracted):
                    bad_rows.append(f"{row_id}:assistant_final_answer_mismatch:{extracted}")
        for flag in ("gate_rows_used_for_training", "weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if not _metadata_flag_is_false(row, metadata, flag):
                bad_rows.append(f"{row_id}:{flag}_not_false")
        if len(bad_rows) >= 30:
            break
    if bad_rows:
        findings.append(Finding("error", f"{split}_dataset_content_invalid", json.dumps(bad_rows[:30], sort_keys=True)))
    return {
        "path": str(path),
        "sha256": observed_sha,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--v438-audit-manifest", type=Path, default=None)
    parser.add_argument("--dataset-schema", choices=["preference", "sft"], default="preference")
    parser.add_argument("--expected-save-every-steps", type=int, default=3)
    parser.add_argument("--expected-eval-every-steps", type=int, default=3)
    parser.add_argument("--expected-data-root", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-val-rows", type=int, required=True)
    parser.add_argument("--expected-output-repo", required=True)
    parser.add_argument("--expected-init-adapter-repo", required=True)
    parser.add_argument("--expected-init-adapter-subfolder", required=True)
    parser.add_argument("--expected-pair-score-mode", default="")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    print("=== KG1 PRE PAID JOB INTEGRATION GATE START ===", flush=True)
    launcher_report = audit_launcher(args, findings)
    train_report = audit_dataset_file(
        args.train_jsonl,
        args.expected_train_sha256,
        args.expected_train_rows,
        "train",
        findings,
        dataset_schema=args.dataset_schema,
    )
    val_report = audit_dataset_file(
        args.val_jsonl,
        args.expected_val_sha256,
        args.expected_val_rows,
        "validation",
        findings,
        dataset_schema=args.dataset_schema,
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
    report = {
        "schema_version": "kg1_pre_paid_job_integration_gate_v2",
        "dataset_schema": args.dataset_schema,
        "ok": not any(item.level == "error" for item in findings),
        "launcher": launcher_report,
        "train_dataset": train_report,
        "validation_dataset": val_report,
        "v438_audit": v438_summary,
        "findings": [item.__dict__ for item in findings],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("=== KG1 PRE PAID JOB INTEGRATION GATE END ===", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

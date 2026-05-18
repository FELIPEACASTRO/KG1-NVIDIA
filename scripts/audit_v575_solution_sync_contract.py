#!/usr/bin/env python3
"""V575 end-to-end contract audit for KG1 loss/ACC synchronization.

This is a cheap fail-closed audit for the active weak-eval/train path.  It does
not claim that lower loss implies higher ACC; it checks that the loss path,
generation path, extraction path, and promotion gate are using the same
contract so a run cannot be promoted by stale thresholds, label-aware parsing,
wrong files, or a silently different LoRA adapter.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    extract_final_answer,
    extract_final_answer_for_expected,
    verify_answer,
)


EXPECTED_WEAK_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
EXPECTED_WEAK_SHARED_ROW_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
EXPECTED_WEAK_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}
EXPECTED_WEAK_ROWS = 315
EXPECTED_TRAIN_SHA256 = "ba51538ddeedd16e9a5d5e2330b910616de79164a4d1c92bb705d6f2d664d1ae"
EXPECTED_VAL_SHA256 = "6957f0ac518050c4bb41272ccad5b514bc1d045e02778df7e7ef69101d5f2e75"
EXPECTED_TRAIN_COUNTS = {"bit_manipulation": 437, "equation_transform": 320}
EXPECTED_VAL_COUNTS = {"bit_manipulation": 79, "equation_transform": 80}
EXPECTED_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
EXPECTED_LORA_R = 32
EXPECTED_LORA_ALPHA = 32
EXPECTED_TARGET_PARAMETERS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
EXPECTED_PROMOTION = {"correct_min": 196, "equation_transform_min": 60, "bit_manipulation_min": 136, "truncated_max": 0}
OFFICIAL_PROMPT_SUFFIX = "Please put your final answer inside `\\boxed{}`"


@dataclass
class Finding:
    level: str
    code: str
    detail: str


def windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def path_exists(path: Path) -> bool:
    return os.path.exists(windows_long_path(path))


def read_text(path: Path) -> str:
    with open(windows_long_path(path), encoding="utf-8", errors="replace") as handle:
        return handle.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(windows_long_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_finding(findings: list[Finding], level: str, code: str, detail: str) -> None:
    findings.append(Finding(level=level, code=code, detail=detail))


def has_control(text: str) -> bool:
    return any((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127 for ch in text)


def non_ascii(text: str) -> list[str]:
    return sorted({f"U+{ord(ch):04X} {ch}" for ch in text if ord(ch) > 127})


def bool_cell(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def row_family(row: dict[str, Any]) -> str:
    return str(row.get("task_type") or row.get("family") or row.get("type") or row.get("pred_type") or "")


def jsonl_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("task_type") or metadata.get("family") or row.get("task_type") or row.get("family") or row.get("type") or "")


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(windows_long_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(windows_long_path(path), encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def audit_weak_csv(path: Path, findings: list[Finding]) -> dict[str, Any]:
    if not path_exists(path):
        add_finding(findings, "error", "weak_csv_missing", str(path))
        return {}
    rows = load_csv(path)
    observed_sha = sha256_file(path)
    counts = Counter(row_family(row) for row in rows)
    prompt_shas = Counter(hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest() for row in rows)
    ids = Counter(str(row.get("id", "")) for row in rows)
    report = {
        "path": str(path),
        "sha256": observed_sha,
        "rows": len(rows),
        "family_counts": dict(counts),
        "duplicate_ids": sorted([key for key, value in ids.items() if value > 1]),
        "duplicate_prompt_sha256_count": sum(1 for value in prompt_shas.values() if value > 1),
        "empty_prompt_rows": sum(1 for row in rows if not str(row.get("prompt", "")).strip()),
        "empty_answer_rows": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
        "control_char_rows": sum(1 for row in rows if has_control(str(row.get("prompt", "")) + str(row.get("answer", "")))),
        "non_ascii_answer_chars": sorted({char for row in rows for char in non_ascii(str(row.get("answer", "")))}),
    }
    if observed_sha != EXPECTED_WEAK_SHA256:
        add_finding(findings, "error", "weak_csv_sha_mismatch", f"{observed_sha} != {EXPECTED_WEAK_SHA256}")
    if len(rows) != EXPECTED_WEAK_ROWS:
        add_finding(findings, "error", "weak_csv_row_count_mismatch", f"{len(rows)} != {EXPECTED_WEAK_ROWS}")
    if dict(counts) != EXPECTED_WEAK_COUNTS:
        add_finding(findings, "error", "weak_csv_family_count_mismatch", f"{dict(counts)} != {EXPECTED_WEAK_COUNTS}")
    for key in ("duplicate_ids", "duplicate_prompt_sha256_count", "empty_prompt_rows", "empty_answer_rows", "control_char_rows", "non_ascii_answer_chars"):
        if report[key]:
            add_finding(findings, "error", f"weak_csv_{key}", json.dumps(report[key], ensure_ascii=False))
    return report


def message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def audit_jsonl_dataset(
    path: Path,
    *,
    expected_sha: str,
    expected_counts: dict[str, int],
    expected_rows: int,
    label: str,
    findings: list[Finding],
) -> dict[str, Any]:
    if not path_exists(path):
        add_finding(findings, "error", f"{label}_jsonl_missing", str(path))
        return {}
    rows = load_jsonl(path)
    counts = Counter(jsonl_family(row) for row in rows)
    top_sub = Counter(str(row.get("subcategory") or "") for row in rows)
    metadata_sub = Counter(str((row.get("metadata") or {}).get("subcategory") or "") for row in rows if isinstance(row.get("metadata"), dict))
    sub_mismatch = 0
    family_mismatch = 0
    assistant_mismatch: list[dict[str, str]] = []
    control_rows = 0
    non_ascii_chars: set[str] = set()
    row_loss_weight_counts: Counter[str] = Counter()
    missing_loss_weight = 0
    boxed_rows = 0
    plain_label_free_rows = 0
    raw_output_rows = 0
    empty_rows = 0
    for idx, row in enumerate(rows):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if row.get("family") and metadata.get("family") and str(row.get("family")) != str(metadata.get("family")):
            family_mismatch += 1
        if row.get("subcategory") and metadata.get("subcategory") and str(row.get("subcategory")) != str(metadata.get("subcategory")):
            sub_mismatch += 1
        loss_weight = metadata.get("loss_weight", row.get("loss_weight"))
        if loss_weight is None:
            missing_loss_weight += 1
        else:
            row_loss_weight_counts[str(loss_weight)] += 1
        if "raw_output" in row:
            raw_output_rows += 1
        prompt = str(row.get("prompt") or message_content(row, "user"))
        answer = str(row.get("answer") or metadata.get("answer") or "")
        assistant = message_content(row, "assistant")
        text = prompt + answer + assistant
        if not prompt.strip() or not answer.strip() or not assistant.strip():
            empty_rows += 1
        if has_control(text):
            control_rows += 1
        non_ascii_chars.update(non_ascii(text))
        extracted = extract_final_answer(assistant)
        if "\\boxed{" in assistant:
            boxed_rows += 1
        elif metadata.get("final_answer_format") == "final_answer_plain_label_free" and verify_answer(answer, extracted):
            plain_label_free_rows += 1
        if not verify_answer(answer, extracted):
            assistant_mismatch.append({"idx": str(idx), "id": str(row.get("id", "")), "answer": answer, "extracted": extracted})
    observed_sha = sha256_file(path)
    report = {
        "path": str(path),
        "sha256": observed_sha,
        "rows": len(rows),
        "family_counts": dict(counts),
        "top_level_subcategory_counts": dict(top_sub),
        "metadata_subcategory_counts": dict(metadata_sub),
        "subcategory_top_metadata_mismatch_rows": sub_mismatch,
        "family_top_metadata_mismatch_rows": family_mismatch,
        "loss_weight_counts": dict(row_loss_weight_counts),
        "missing_loss_weight_rows": missing_loss_weight,
        "boxed_assistant_rows": boxed_rows,
        "plain_label_free_assistant_rows": plain_label_free_rows,
        "assistant_answer_mismatch_count": len(assistant_mismatch),
        "assistant_answer_mismatch_first10": assistant_mismatch[:10],
        "raw_output_rows": raw_output_rows,
        "empty_rows": empty_rows,
        "control_char_rows": control_rows,
        "non_ascii_chars": sorted(non_ascii_chars),
    }
    if observed_sha != expected_sha:
        add_finding(findings, "error", f"{label}_sha_mismatch", f"{observed_sha} != {expected_sha}")
    if len(rows) != expected_rows:
        add_finding(findings, "error", f"{label}_row_count_mismatch", f"{len(rows)} != {expected_rows}")
    if dict(counts) != expected_counts:
        add_finding(findings, "error", f"{label}_family_count_mismatch", f"{dict(counts)} != {expected_counts}")
    if missing_loss_weight:
        add_finding(findings, "error", f"{label}_missing_loss_weight", str(missing_loss_weight))
    if raw_output_rows:
        add_finding(findings, "error", f"{label}_raw_output_leakage", str(raw_output_rows))
    if empty_rows:
        add_finding(findings, "error", f"{label}_empty_prompt_answer_or_assistant", str(empty_rows))
    if control_rows:
        add_finding(findings, "error", f"{label}_control_chars", str(control_rows))
    if non_ascii_chars:
        add_finding(findings, "error", f"{label}_non_ascii_chars", json.dumps(sorted(non_ascii_chars), ensure_ascii=False))
    if boxed_rows + plain_label_free_rows != len(rows):
        add_finding(
            findings,
            "error",
            f"{label}_non_label_free_assistant",
            f"boxed={boxed_rows}; plain_label_free={plain_label_free_rows}; rows={len(rows)}",
        )
    if assistant_mismatch:
        add_finding(findings, "error", f"{label}_assistant_answer_mismatch", json.dumps(assistant_mismatch[:5], ensure_ascii=False))
    if sub_mismatch:
        add_finding(
            findings,
            "info",
            f"{label}_subcategory_top_metadata_mismatch",
            "Top-level subcategory differs from metadata provenance; OK only because canonical_example_subcategory prioritizes top-level.",
        )
    return report


def audit_adapter_config(path: Path, findings: list[Finding]) -> dict[str, Any]:
    if not path_exists(path):
        add_finding(findings, "error", "adapter_config_missing", str(path))
        return {}
    config = json.loads(read_text(path))
    target_modules = sorted(config.get("target_modules") or [])
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "modules_to_save": config.get("modules_to_save"),
        "target_modules": target_modules,
        "target_parameters": config.get("target_parameters"),
    }
    if config.get("base_model_name_or_path") != EXPECTED_BASE_MODEL:
        add_finding(findings, "error", "adapter_base_model_mismatch", str(config.get("base_model_name_or_path")))
    if int(config.get("r", -1)) != EXPECTED_LORA_R:
        add_finding(findings, "error", "adapter_lora_r_mismatch", str(config.get("r")))
    if int(config.get("lora_alpha", -1)) != EXPECTED_LORA_ALPHA:
        add_finding(findings, "error", "adapter_lora_alpha_mismatch", str(config.get("lora_alpha")))
    if config.get("modules_to_save") not in (None, [], ""):
        add_finding(findings, "error", "adapter_modules_to_save_not_empty", str(config.get("modules_to_save")))
    if list(config.get("target_parameters") or []) != EXPECTED_TARGET_PARAMETERS:
        add_finding(findings, "error", "adapter_target_parameters_mismatch", str(config.get("target_parameters")))
    for required in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"):
        if required not in target_modules:
            add_finding(findings, "error", "adapter_target_module_missing", required)
    return report


def audit_eval_manifest(path: Path, findings: list[Finding]) -> dict[str, Any]:
    if not path_exists(path):
        add_finding(findings, "error", "weak_eval_manifest_missing", str(path))
        return {}
    manifest = json.loads(read_text(path))
    report = {
        "path": str(path),
        "repo_commit": manifest.get("repo_commit"),
        "blocked_actions": manifest.get("blocked_actions"),
        "weak_csv": manifest.get("weak_csv", {}),
        "eval_prompt_controls": manifest.get("eval_prompt_controls", {}),
        "weak_promotion_gate": manifest.get("weak_promotion_gate", {}),
        "protected_row_backfire_guard_passed": (manifest.get("protected_row_backfire_guard") or {}).get("passed"),
    }
    weak_csv = manifest.get("weak_csv") or {}
    if weak_csv.get("sha256") != EXPECTED_WEAK_SHA256:
        add_finding(findings, "error", "manifest_weak_sha_mismatch", str(weak_csv.get("sha256")))
    if weak_csv.get("observed_shared_row_contract_sha256") != EXPECTED_WEAK_SHARED_ROW_SHA256:
        add_finding(findings, "error", "manifest_shared_row_contract_mismatch", str(weak_csv.get("observed_shared_row_contract_sha256")))
    prompt_controls = manifest.get("eval_prompt_controls") or {}
    if prompt_controls.get("disable_thinking") is not False:
        add_finding(findings, "error", "manifest_disable_thinking_not_official", str(prompt_controls.get("disable_thinking")))
    if OFFICIAL_PROMPT_SUFFIX not in str(prompt_controls.get("prompt_suffix", "")):
        add_finding(findings, "error", "manifest_prompt_suffix_not_official", str(prompt_controls.get("prompt_suffix", "")))
    thresholds = (manifest.get("weak_promotion_gate") or {}).get("thresholds") or {}
    for key, value in EXPECTED_PROMOTION.items():
        if int(thresholds.get(key, -999)) != value:
            add_finding(findings, "error", f"manifest_promotion_{key}_mismatch", f"{thresholds.get(key)} != {value}")
    report["protected_guard_blocker"] = (manifest.get("protected_row_backfire_guard") or {}).get("passed") is False
    return report


def audit_prediction_csv(path: Path, findings: list[Finding], row_audit_csv: Path | None = None) -> dict[str, Any]:
    if not path_exists(path):
        add_finding(findings, "error", "prediction_csv_missing", str(path))
        return {}
    rows = load_csv(path)
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "stored_correct": 0, "raw_correct": 0, "label_aware_correct": 0, "truncated": 0})
    stored_mismatch: list[dict[str, str]] = []
    expected_delta_rows: list[dict[str, str]] = []
    row_audit_rows: list[dict[str, Any]] = []
    wrong_classes: Counter[str] = Counter()
    non_ascii_raw: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    for row in rows:
        family = row_family(row)
        answer = str(row.get("answer", ""))
        raw = str(row.get("raw_output", ""))
        stored_prediction = str(row.get("prediction", ""))
        raw_prediction = extract_final_answer(raw)
        expected_prediction = extract_final_answer_for_expected(raw, answer)
        stored_correct = bool_cell(row.get("correct"))
        raw_correct = verify_answer(answer, raw_prediction)
        expected_correct = verify_answer(answer, expected_prediction)
        truncated = bool_cell(row.get("truncated")) or bool_cell(row.get("truncated_bool"))
        finish_reasons[str(row.get("finish_reason", ""))] += 1
        by_family[family]["total"] += 1
        by_family[family]["stored_correct"] += int(stored_correct)
        by_family[family]["raw_correct"] += int(raw_correct)
        by_family[family]["label_aware_correct"] += int(expected_correct)
        by_family[family]["truncated"] += int(truncated)
        if stored_prediction != raw_prediction or stored_correct != raw_correct:
            stored_mismatch.append(
                {
                    "id": str(row.get("id", "")),
                    "family": family,
                    "answer": answer,
                    "stored_prediction": stored_prediction,
                    "raw_prediction": raw_prediction,
                    "stored_correct": str(stored_correct),
                    "raw_correct": str(raw_correct),
                }
            )
        if expected_correct and not raw_correct:
            expected_delta_rows.append(
                {
                    "id": str(row.get("id", "")),
                    "family": family,
                    "answer": answer,
                    "label_aware_prediction": expected_prediction,
                    "label_free_prediction": raw_prediction,
                }
            )
        wrong_class = ""
        if not raw_correct:
            if truncated:
                wrong_class = "decoding_truncated"
            elif family == "bit_manipulation" and re.fullmatch(r"[01]{8}", answer) and re.fullmatch(r"[01]{8}", raw_prediction or ""):
                wrong_class = "bit_binary_wrong"
            elif family == "bit_manipulation":
                wrong_class = "bit_other_wrong"
            else:
                wrong_class = "equation_wrong"
            wrong_classes[wrong_class] += 1
        raw_non_ascii = sorted(non_ascii(raw))
        non_ascii_raw.update(raw_non_ascii)
        row_audit_rows.append(
            {
                "id": str(row.get("id", "")),
                "family": family,
                "answer": answer,
                "stored_prediction": stored_prediction,
                "label_free_prediction": raw_prediction,
                "label_aware_debug_prediction": expected_prediction,
                "stored_correct": stored_correct,
                "label_free_correct": raw_correct,
                "label_aware_debug_correct": expected_correct,
                "wrong_class": wrong_class,
                "finish_reason": str(row.get("finish_reason", "")),
                "truncated": truncated,
                "completion_tokens": str(row.get("completion_tokens", "")),
                "prompt_sha256": hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest(),
                "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "raw_output_len": len(raw),
                "raw_non_ascii": " ".join(raw_non_ascii),
                "stored_vs_label_free_mismatch": stored_prediction != raw_prediction or stored_correct != raw_correct,
                "expected_aware_would_overcount": expected_correct and not raw_correct,
            }
        )
    if row_audit_csv is not None:
        row_audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with row_audit_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "id",
                "family",
                "answer",
                "stored_prediction",
                "label_free_prediction",
                "label_aware_debug_prediction",
                "stored_correct",
                "label_free_correct",
                "label_aware_debug_correct",
                "wrong_class",
                "finish_reason",
                "truncated",
                "completion_tokens",
                "prompt_sha256",
                "raw_output_sha256",
                "raw_output_len",
                "raw_non_ascii",
                "stored_vs_label_free_mismatch",
                "expected_aware_would_overcount",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_audit_rows)
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "row_audit_csv": str(row_audit_csv) if row_audit_csv is not None else "",
        "by_family": dict(by_family),
        "stored_vs_current_raw_parser_mismatch_count": len(stored_mismatch),
        "stored_vs_current_raw_parser_mismatch_first10": stored_mismatch[:10],
        "label_aware_minus_label_free_correct_count": len(expected_delta_rows),
        "label_aware_minus_label_free_first10": expected_delta_rows[:10],
        "wrong_classes": dict(wrong_classes),
        "finish_reasons": dict(finish_reasons),
        "non_ascii_raw_chars": sorted(non_ascii_raw),
    }
    totals = {family: values["raw_correct"] for family, values in by_family.items()}
    performance_blockers: list[str] = []
    if totals.get("equation_transform", 0) < EXPECTED_PROMOTION["equation_transform_min"]:
        performance_blockers.append(f"equation_transform={totals.get('equation_transform', 0)}<{EXPECTED_PROMOTION['equation_transform_min']}")
    if totals.get("bit_manipulation", 0) < EXPECTED_PROMOTION["bit_manipulation_min"]:
        performance_blockers.append(f"bit_manipulation={totals.get('bit_manipulation', 0)}<{EXPECTED_PROMOTION['bit_manipulation_min']}")
    truncation_total = sum(item["truncated"] for item in by_family.values())
    if truncation_total > EXPECTED_PROMOTION["truncated_max"]:
        performance_blockers.append(f"truncated={truncation_total}>{EXPECTED_PROMOTION['truncated_max']}")
    if stored_mismatch:
        performance_blockers.append(f"stored_prediction_stale_after_parser_fix={len(stored_mismatch)}")
    report["performance_blockers"] = performance_blockers
    if expected_delta_rows:
        add_finding(findings, "error", "expected_aware_would_overcount", json.dumps(expected_delta_rows[:5], ensure_ascii=False))
    return report


def dict_key_duplicates_in_assignment(path: Path, assignment_name: str) -> list[str]:
    tree = ast.parse(read_text(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if assignment_name not in names or not isinstance(node.value, ast.Dict):
                continue
            keys: list[str] = []
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
            return sorted([key for key, count in Counter(keys).items() if count > 1])
    return []


def require_snippet(text: str, snippet: str, findings: list[Finding], code: str) -> None:
    if snippet not in text:
        add_finding(findings, "error", code, f"missing snippet: {snippet}")


def audit_source_contract(findings: list[Finding]) -> dict[str, Any]:
    train_py = ROOT / "scripts/hf_job_train_v90.py"
    weak_py = ROOT / "scripts/hf_job_weak_eval_v245.py"
    eval_batch_py = ROOT / "scripts/evaluate_lora_adapters_batch.py"
    static_py = ROOT / "scripts/kg1_static_safety_gate.py"
    v509_py = ROOT / "scripts/audit_v509_training_dataset_integrity.py"
    v573_launcher = ROOT / "artifacts/v573_hf_h200_launch/launch_v573_hf_nemo_h200_v571bit_v551eq_refmix.py"
    notebook_gate = ROOT / "scripts/notebook_release_gate.py"
    files = [train_py, weak_py, eval_batch_py, static_py, v509_py, v573_launcher, notebook_gate]
    for path in files:
        if not path_exists(path):
            add_finding(findings, "error", "source_file_missing", str(path))
    train_text = read_text(train_py)
    weak_text = read_text(weak_py)
    eval_batch_text = read_text(eval_batch_py)
    static_text = read_text(static_py)
    v509_text = read_text(v509_py)
    launcher_text = read_text(v573_launcher)
    notebook_text = read_text(notebook_gate)
    require_snippet(train_text, "example.get(\"subcategory\")", findings, "train_canonical_subcategory_missing_top_level")
    require_snippet(train_text, "metadata.get(\"subcategory\")", findings, "train_canonical_subcategory_missing_metadata_fallback")
    require_snippet(train_text, "F.cross_entropy(flat_logits, flat_labels, reduction=\"none\")", findings, "train_masked_ce_missing")
    require_snippet(train_text, "LOSS_NORMALIZATION_MODE == \"example_mean\"", findings, "train_example_mean_missing")
    require_snippet(train_text, "parse_example_loss_weight", findings, "train_row_loss_weight_missing")
    require_snippet(weak_text, "evaluate_lora_adapters_batch.py", findings, "weak_eval_batch_evaluator_call_missing")
    require_snippet(weak_text, "--max-tokens", findings, "weak_eval_max_tokens_arg_missing")
    require_snippet(weak_text, "--prompt-suffix", findings, "weak_eval_prompt_suffix_arg_missing")
    require_snippet(eval_batch_text, "prediction = extract_final_answer(raw_output)", findings, "batch_eval_label_free_extractor_missing")
    require_snippet(eval_batch_text, "label_aware_debug_prediction", findings, "batch_eval_expected_aware_debug_missing")
    require_snippet(eval_batch_text, "extract_final_answer_for_expected(raw_output, expected)", findings, "batch_eval_expected_aware_extractor_missing")
    require_snippet(eval_batch_text, "\"prediction_metric_mode\": \"submit_safe_label_free\"", findings, "batch_eval_submit_safe_metric_mode_missing")
    require_snippet(weak_text, "KG1_WEAK_PROMOTE_TOTAL_MIN\", 196", findings, "weak_eval_total_gate_not_196")
    require_snippet(weak_text, "KG1_WEAK_PROMOTE_EQUATION_MIN\", 60", findings, "weak_eval_equation_gate_not_60")
    require_snippet(weak_text, "KG1_WEAK_PROMOTE_BIT_MIN\", 136", findings, "weak_eval_bit_gate_not_136")
    require_snippet(weak_text, "KG1_WEAK_PROMOTE_TRUNC_MAX\", 0", findings, "weak_eval_trunc_gate_not_zero")
    require_snippet(v509_text, "dataset_count=0; fail-closed", findings, "v509_dataset_count_fail_closed_missing")
    require_snippet(v509_text, "Explicit dataset JSONL path(s) not found", findings, "v509_missing_file_fail_closed_missing")
    require_snippet(v509_text, "windows_long_path(root)", findings, "v509_rglob_long_path_missing")
    require_snippet(static_text, "stale_weak_total_gate", findings, "static_gate_weak_min_for_full_193_block_missing")
    require_snippet(launcher_text, "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE", findings, "v573_target_parameter_trainability_env_missing")
    require_snippet(launcher_text, "export TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj'", findings, "v573_trainable_allowlist_missing")
    require_snippet(notebook_text, "WEAK_MIN_FOR_FULL = 196", findings, "notebook_gate_total_not_196")
    duplicates = dict_key_duplicates_in_assignment(static_py, "CRITICAL_SNIPPETS")
    if duplicates:
        add_finding(findings, "error", "static_gate_duplicate_critical_keys", json.dumps(duplicates))
    return {
        "checked_files": [str(path) for path in files],
        "critical_snippet_duplicate_keys": duplicates,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    report = {
        "schema_version": "kg1_v575_solution_sync_contract_v1",
        "weak_csv": audit_weak_csv(args.weak_csv, findings),
        "train_jsonl": audit_jsonl_dataset(
            args.train_jsonl,
            expected_sha=EXPECTED_TRAIN_SHA256,
            expected_counts=EXPECTED_TRAIN_COUNTS,
            expected_rows=sum(EXPECTED_TRAIN_COUNTS.values()),
            label="train",
            findings=findings,
        ),
        "val_jsonl": audit_jsonl_dataset(
            args.val_jsonl,
            expected_sha=EXPECTED_VAL_SHA256,
            expected_counts=EXPECTED_VAL_COUNTS,
            expected_rows=sum(EXPECTED_VAL_COUNTS.values()),
            label="val",
            findings=findings,
        ),
        "adapter_config": audit_adapter_config(args.adapter_config, findings),
        "eval_manifest": audit_eval_manifest(args.eval_manifest, findings),
        "prediction_csv": audit_prediction_csv(args.prediction_csv, findings, args.row_audit_csv),
        "source_contract": audit_source_contract(findings),
    }
    errors = [finding for finding in findings if finding.level == "error"]
    warnings = [finding for finding in findings if finding.level == "warning"]
    infos = [finding for finding in findings if finding.level == "info"]
    report["findings"] = [finding.__dict__ for finding in findings]
    report["finding_counts"] = {"error": len(errors), "warning": len(warnings), "info": len(infos)}
    report["ok"] = not errors
    report["decision"] = {
        "submit_safe_now": False,
        "reason": "This audit validates synchronization. Promotion still requires generated weak ACC >= 196/315 with equation>=60, bit>=136, trunc=0, protected rows held.",
        "performance_blockers": report["prediction_csv"].get("performance_blockers", []),
    }
    return report


def default_path(relative: str) -> Path:
    return ROOT / relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-csv", type=Path, default=default_path("artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"))
    parser.add_argument("--train-jsonl", type=Path, default=default_path("artifacts/v573_v571_bitpair_v551_equation_reference_mix/20260517T_v573_cpu_gate/v572_v571_bitpair_v551_equation_mix_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=default_path("artifacts/v573_v571_bitpair_v551_equation_reference_mix/20260517T_v573_cpu_gate/v572_v571_bitpair_v551_equation_mix_val.jsonl"))
    parser.add_argument("--adapter-config", type=Path, default=default_path("artifacts/v574_hf_h200_v573_weak_eval_launch/downloaded_final/checkpoint-2/adapter_config.json"))
    parser.add_argument("--eval-manifest", type=Path, default=default_path("artifacts/v574_hf_h200_v573_weak_eval_launch/downloaded_final/evals/v574-h200-officiallike-v573-checkpoint2-20260517T194619Z/v245_hf_weak_eval_manifest.json"))
    parser.add_argument("--prediction-csv", type=Path, default=default_path("artifacts/v574_hf_h200_v573_weak_eval_launch/downloaded_final/evals/v574-h200-officiallike-v573-checkpoint2-20260517T194619Z/eval/candidate_01_a9eaea99/v574_v573_checkpoint_2_official_like/v574_hf_weak_v574_v573_checkpoint_2_official_like_predictions.csv"))
    parser.add_argument("--output-json", type=Path, default=default_path("artifacts/v574_hf_h200_v573_weak_eval_launch/v575_solution_sync_contract_audit.json"))
    parser.add_argument("--row-audit-csv", type=Path, default=default_path("artifacts/v574_hf_h200_v573_weak_eval_launch/v575_prediction_row_audit.csv"))
    args = parser.parse_args()

    report = run_audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "finding_counts": report["finding_counts"], "output_json": str(args.output_json)}, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

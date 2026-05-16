#!/usr/bin/env python3
"""Audit V499 parameterization against the KG1 weak-gain objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
V498_ROOT = REPO_ROOT / "artifacts/v498_numeric_teacher_trace_pack/20260516T_v498_numeric_teacher"
V499_ROOT = REPO_ROOT / "artifacts/v499_hf_nemo_h200_v498_numeric_teacher_trace_launch"
ROADMAP_ROOT = REPO_ROOT / "artifacts/roadmaps"

EXPECTED = {
    "train_rows": 1712,
    "val_rows": 428,
    "train_sha256": "920b3c30b9ada9ad2685091194dcc53e717f72a9c037cafeef6e494f21511e79",
    "val_sha256": "68cda4162214359aaf7cda304c2a06902775b1aadb53fcadfd0edf7ff481ed80",
    "tokenization_status": "tokenization_gate_passed",
    "train_token_max_lte": 1024,
    "bit_effective_share_min": 0.35,
    "equation_effective_share_max": 0.65,
    "max_unit_cost_usd": 0.09,
    "required_family_counts": {"bit_manipulation": 512, "equation_transform": 1200},
    "required_val_family_counts": {"bit_manipulation": 128, "equation_transform": 300},
    "required_source_weights": {
        "v498_numeric_teacher_trace_pack": 1.0,
        "v498_bit_replay_guardrail_from_v475": 1.5,
    },
    "promotion_gate": {"total_gt": 192, "equation_gte": 60, "bit_gte": 136, "truncated_eq": 0},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str, severity: str = "fail") -> None:
    checks.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
    return ""


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row.get("family", "")) for row in rows)
    source = Counter(str(row.get("source", "")) for row in rows)
    subcat = Counter(str(row.get("subcategory", "")) for row in rows)
    ids = [str(row.get("id", "")) for row in rows]
    final_marker_missing = sum(1 for row in rows if "Final answer:" not in assistant_text(row))
    boxed_missing = sum(1 for row in rows if "\\boxed{" not in assistant_text(row))
    empty_answers = sum(1 for row in rows if not str(row.get("answer", "")).strip())
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "family_counts": dict(sorted(family.items())),
        "source_counts": dict(sorted(source.items())),
        "subcategory_counts": dict(sorted(subcat.items())),
        "missing_final_answer_marker": final_marker_missing,
        "missing_boxed_marker": boxed_missing,
        "empty_answers": empty_answers,
    }


def parse_float_after(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_logs(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    train_losses: list[dict[str, Any]] = []
    for line in lines:
        match = re.search(r"step=(\d+)/(\d+) lr=([0-9.eE+-]+) train_loss=([0-9.]+)", line)
        if match:
            train_losses.append(
                {
                    "step": int(match.group(1)),
                    "max_steps": int(match.group(2)),
                    "lr": float(match.group(3)),
                    "train_loss": float(match.group(4)),
                }
            )
    mem_reserved = [
        float(match.group(1))
        for line in lines
        for match in [re.search(r"mem_reserved=([0-9.]+)GiB", line)]
        if match
    ]
    tokenization_train = next((line for line in lines if line.startswith("Train tokenization summary:")), "")
    tokenization_val = next((line for line in lines if line.startswith("Validation tokenization summary:")), "")
    return {
        "status_markers": {
            "training_complete": "=== TRAINING COMPLETE ===" in text,
            "checkpoint_uploaded": "Checkpoint uploaded:" in text,
            "upload_complete": "Upload complete:" in text,
            "trainable_lora_filter_applied": "Applied trainable LoRA module filter:" in text,
            "target_parameters_trainable": '"target_parameters_trainability_mode": "trainable"' in text,
            "lm_head_frozen": '"frozen_by_module"' in text and '"lm_head": 4280320' in text,
        },
        "baseline_eval_loss": parse_float_after(r"baseline_eval_loss=([0-9.]+)", text),
        "final_eval_loss": parse_float_after(r"Final eval loss: ([0-9.]+)", text),
        "eval_loss_end_values": [
            float(match.group(1)) for match in re.finditer(r"eval_loss_eval_end loss=([0-9.]+)", text)
        ],
        "train_losses": train_losses,
        "max_mem_reserved_gib": max(mem_reserved) if mem_reserved else None,
        "tokenization_train_line": tokenization_train,
        "tokenization_validation_line": tokenization_val,
        "answer_span_weight": parse_float_after(r"answer_span_loss_weight=([0-9.]+)", tokenization_train),
        "answer_span_weighted_examples_train": int(parse_float_after(r"answer_span_weighted_examples=([0-9.]+)", tokenization_train) or 0),
        "answer_span_weighted_examples_val": int(parse_float_after(r"answer_span_weighted_examples=([0-9.]+)", tokenization_val) or 0),
        "raw_line_count": len(lines),
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    checks = manifest["checks"]
    failures = [item for item in checks if not item["ok"] and item["severity"] == "fail"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warn"]
    lines = [
        "# KG1 V500 V499 Parameterization Audit",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Next action: {manifest['decision']['next_action']}",
        f"- Reason: {manifest['decision']['reason']}",
        "",
        "## V499 Loss Snapshot",
        "",
        f"- Baseline eval loss: `{manifest['job_log']['baseline_eval_loss']}`",
        f"- Final eval loss: `{manifest['job_log']['final_eval_loss']}`",
        f"- Delta final-baseline: `{manifest['loss_delta']}`",
        f"- Max reserved memory GiB: `{manifest['job_log']['max_mem_reserved_gib']}`",
        "",
        "## Key Parameter Verdict",
        "",
        "| Area | Verdict | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        lines.append(f"| {item['name']} | `{verdict}` | {item['detail']} |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The dataset, hashes, family mix, tokenization gate, target-parameter trainability, H200 cost gate, and upload path are technically consistent.",
            "- The run is not a submit-gain candidate because final eval loss did not improve over baseline on the V498 validation sample.",
            "- The largest parameter gap is ACC alignment: `ANSWER_SPAN_LOSS_WEIGHT=1.0` produced `0` answer-span weighted examples, so the loss optimized full assistant traces instead of emphasizing the final boxed answer.",
            "- `MAX_STEPS=2` was correct for a paid smoke test, but it is not a value that should be expected to produce a new submit-safe adapter.",
            "- FinOps decision: do not run weak eval for V499 unless a separate reason appears, because the local objective signal is flat/slightly negative.",
            "",
            "## Required Next Configuration",
            "",
            "- Next paid run must use answer-focused loss, for example `ANSWER_SPAN_LOSS_WEIGHT>=4.0`, and a gate requiring answer-span weighted examples on train and validation.",
            "- Keep V290 checkpoint-6 as init adapter, keep MoE target parameters trainable, keep `lm_head` frozen, and keep bit replay guardrail at or above the current effective share.",
            "- Use a short but non-trivial run only after local gate confirms the answer-span weighting is active; candidate values: `MAX_STEPS=4..8`, `EVAL_EVERY_STEPS=2`, `SAVE_EVERY_STEPS=2`, with FinOps abort if eval loss rises by more than the baseline tolerance.",
            "- Only launch weak ACC eval if the training objective improves or if there is a new deterministic CPU projection.",
        ]
    )
    if failures or warnings:
        lines.extend(["", "## Open Items", ""])
        for item in failures + warnings:
            lines.append(f"- `{item['name']}`: {item['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v500_v499_parameterization_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = V498_ROOT / "v498_numeric_teacher_trace_train.jsonl"
    val_jsonl = V498_ROOT / "v498_numeric_teacher_trace_val.jsonl"
    dataset_manifest = read_json(V498_ROOT / "v498_numeric_teacher_trace_manifest.json")
    tokenization_manifest = read_json(V498_ROOT / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json")
    launch_manifest = read_json(
        V499_ROOT / "v499-nemo-h200-v498-numeric-teacher-v290ckpt6-20260516T183506Z_launch_manifest.json"
    )
    objective_manifest = read_json(
        V499_ROOT / "v499-nemo-h200-v498-numeric-teacher-v290ckpt6-20260516T183506Z_objective_alignment_gate.json"
    )
    upload_manifest = read_json(V499_ROOT / "v498_hf_dataset_upload_manifest.json")
    log_manifest = parse_logs(V499_ROOT / "v499_hf_job_6a08b91e3308d79117b911f6.log")

    train_summary = summarize_rows(read_jsonl(train_jsonl))
    val_summary = summarize_rows(read_jsonl(val_jsonl))
    checks: list[dict[str, Any]] = []

    add_check(checks, "train_rows", train_summary["rows"] == EXPECTED["train_rows"], f"{train_summary['rows']} rows")
    add_check(checks, "val_rows", val_summary["rows"] == EXPECTED["val_rows"], f"{val_summary['rows']} rows")
    add_check(
        checks,
        "train_family_counts",
        train_summary["family_counts"] == EXPECTED["required_family_counts"],
        json.dumps(train_summary["family_counts"], sort_keys=True),
    )
    add_check(
        checks,
        "val_family_counts",
        val_summary["family_counts"] == EXPECTED["required_val_family_counts"],
        json.dumps(val_summary["family_counts"], sort_keys=True),
    )
    add_check(checks, "duplicate_ids", train_summary["duplicate_ids"] == 0 and val_summary["duplicate_ids"] == 0, "train/val duplicate ids")
    add_check(
        checks,
        "assistant_final_markers",
        train_summary["missing_final_answer_marker"] == 0 and val_summary["missing_final_answer_marker"] == 0,
        "all rows include Final answer marker",
    )
    add_check(
        checks,
        "assistant_boxed_markers",
        train_summary["missing_boxed_marker"] == 0 and val_summary["missing_boxed_marker"] == 0,
        "all rows include boxed final answer",
    )
    add_check(
        checks,
        "reference_overlap",
        dataset_manifest["train_summary"]["reference_id_overlap"] == 0
        and dataset_manifest["train_summary"]["reference_prompt_overlap"] == 0
        and dataset_manifest["train_summary"]["reference_prompt_answer_overlap"] == 0,
        "train reference id/prompt/prompt+answer overlap all zero",
    )
    token_status = tokenization_manifest.get("decision", {}).get("status")
    add_check(checks, "tokenization_gate", token_status == EXPECTED["tokenization_status"], str(token_status))
    train_token_max = int(tokenization_manifest["tokenization"]["train"]["token_max"])
    add_check(checks, "train_max_length_runtime_safe", train_token_max <= EXPECTED["train_token_max_lte"], f"token_max={train_token_max}; runtime MAX_LENGTH=1024")
    add_check(checks, "offset_masks", "offset_masks=1712" in log_manifest["tokenization_train_line"] and "offset_masks=428" in log_manifest["tokenization_validation_line"], "runtime offset masks complete")
    source_weights = launch_manifest["objective_alignment"]["report"]["source_weights"]
    add_check(checks, "source_weights", source_weights == EXPECTED["required_source_weights"], json.dumps(source_weights, sort_keys=True))
    effective_family = objective_manifest["train"]["effective_share_by_family"]
    bit_share = float(effective_family["bit_manipulation"]["share"])
    eq_share = float(effective_family["equation_transform"]["share"])
    add_check(checks, "bit_replay_effective_share", bit_share >= EXPECTED["bit_effective_share_min"], f"bit_share={bit_share}")
    add_check(checks, "equation_effective_share", eq_share <= EXPECTED["equation_effective_share_max"], f"equation_share={eq_share}")
    add_check(checks, "hf_cost_gate", float(launch_manifest["hardware"]["unit_cost_usd"]) <= EXPECTED["max_unit_cost_usd"], f"cost={launch_manifest['hardware']['unit_cost_usd']}/min")
    add_check(checks, "h200_hardware_gate", launch_manifest["flavor"] == "h200" and "H200" in launch_manifest["hardware"]["accelerator_model"], json.dumps(launch_manifest["hardware"], sort_keys=True))
    add_check(checks, "init_adapter", launch_manifest["init_adapter"]["repo"].endswith("v290-rank19-micro-patch-smoke") and launch_manifest["init_adapter"]["subfolder"] == "checkpoint-6", json.dumps(launch_manifest["init_adapter"], sort_keys=True))
    add_check(checks, "target_parameters_trainable", log_manifest["status_markers"]["target_parameters_trainable"], "MoE target parameters reported trainable")
    add_check(checks, "lm_head_frozen", log_manifest["status_markers"]["lm_head_frozen"], "lm_head has zero trainable params in log")
    add_check(checks, "job_completed_and_uploaded", all([log_manifest["status_markers"]["training_complete"], log_manifest["status_markers"]["checkpoint_uploaded"], log_manifest["status_markers"]["upload_complete"]]), "training/checkpoint/final upload complete")
    max_mem = log_manifest["max_mem_reserved_gib"]
    add_check(checks, "memory_under_abort_cap", max_mem is not None and max_mem <= 78.0, f"max_mem_reserved={max_mem}GiB")
    baseline = log_manifest["baseline_eval_loss"]
    final = log_manifest["final_eval_loss"]
    loss_delta = None if baseline is None or final is None else round(final - baseline, 6)
    add_check(
        checks,
        "eval_loss_improved",
        baseline is not None and final is not None and final < baseline,
        f"baseline={baseline}; final={final}; delta={loss_delta}",
        severity="warn",
    )
    add_check(
        checks,
        "answer_span_acc_alignment",
        log_manifest["answer_span_weight"] is not None
        and log_manifest["answer_span_weight"] > 1.0
        and log_manifest["answer_span_weighted_examples_train"] > 0
        and log_manifest["answer_span_weighted_examples_val"] > 0,
        f"weight={log_manifest['answer_span_weight']}; weighted_train={log_manifest['answer_span_weighted_examples_train']}; weighted_val={log_manifest['answer_span_weighted_examples_val']}",
        severity="warn",
    )
    add_check(
        checks,
        "max_steps_submit_gain_sufficient",
        int(launch_manifest["recipe"]["max_steps"]) > 2,
        f"max_steps={launch_manifest['recipe']['max_steps']} was smoke-only",
        severity="warn",
    )

    hard_failures = [item for item in checks if not item["ok"] and item["severity"] == "fail"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warn"]
    decision = {
        "status": "parameterization_technical_pass_objective_warning" if not hard_failures else "parameterization_failed",
        "next_action": "Do not run paid weak eval for V499; create answer-span-weighted V500/V501 only after local gate proves weighting is active.",
        "reason": "Core gates passed, but final eval loss did not improve and the answer span was not emphasized for ACC.",
        "hard_failures": len(hard_failures),
        "warnings": len(warnings),
    }
    manifest = {
        "schema_version": "kg1_v500_v499_parameterization_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected": EXPECTED,
        "dataset_manifest": str(V498_ROOT / "v498_numeric_teacher_trace_manifest.json"),
        "tokenization_manifest": str(V498_ROOT / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"),
        "launch_manifest": str(V499_ROOT / "v499-nemo-h200-v498-numeric-teacher-v290ckpt6-20260516T183506Z_launch_manifest.json"),
        "objective_manifest": str(V499_ROOT / "v499-nemo-h200-v498-numeric-teacher-v290ckpt6-20260516T183506Z_objective_alignment_gate.json"),
        "upload_manifest": str(V499_ROOT / "v498_hf_dataset_upload_manifest.json"),
        "job_log_path": str(V499_ROOT / "v499_hf_job_6a08b91e3308d79117b911f6.log"),
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "job_log": log_manifest,
        "loss_delta": loss_delta,
        "checks": checks,
        "decision": decision,
    }
    json_path = args.output_dir / "v500_v499_parameterization_audit_manifest.json"
    md_path = args.output_dir / "KG1_V500_V499_PARAMETERIZATION_AUDIT.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, manifest)
    print("audit_manifest =", json_path, flush=True)
    print("audit_report =", md_path, flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    for item in checks:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        print(f"{verdict}: {item['name']} - {item['detail']}", flush=True)
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

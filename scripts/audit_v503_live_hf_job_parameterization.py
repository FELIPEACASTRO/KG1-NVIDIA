#!/usr/bin/env python3
"""Audit live HF job logs against KG1 V501 parameterization requirements."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts/v503_live_hf_job_parameterization"


def add(checks: list[dict[str, Any]], area: str, name: str, ok: bool, detail: str, severity: str = "fail") -> None:
    checks.append({"area": area, "name": name, "ok": bool(ok), "detail": detail, "severity": severity})


def extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def try_load_json_after(label: str, text: str) -> dict[str, Any]:
    idx = text.find(label)
    if idx < 0:
        return {}
    start = text.find("{", idx)
    if start < 0:
        return {}
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : pos + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def audit_log(log_text: str, job_id: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    lower = log_text.lower()
    train_weight = extract_float(r"Train tokenization summary:.*?answer_span_loss_weight=([0-9.]+)", log_text)
    train_weighted_examples = extract_int(r"Train tokenization summary:.*?answer_span_weighted_examples=(\d+)", log_text)
    train_weighted_tokens = extract_int(r"Train tokenization summary:.*?answer_span_weighted_tokens=(\d+)", log_text)
    val_weighted_examples = extract_int(r"Validation tokenization summary:.*?answer_span_weighted_examples=(\d+)", log_text)
    planned_steps = extract_int(r"Training: \d+ examples, planned_steps=(\d+)", log_text)
    trainable_pct = extract_float(r"trainable%: ([0-9.]+)", log_text)
    mem_values = [float(item) for item in re.findall(r"mem_reserved=([0-9.]+)GiB", log_text)]
    baseline_eval_loss = extract_float(r"baseline_eval_loss\s*=\s*([0-9.]+)", log_text)
    final_eval_loss = extract_float(r"final_eval_loss\s*=\s*([0-9.]+)", log_text)
    if final_eval_loss is None:
        final_eval_loss = extract_float(r"Final eval loss:\s*([0-9.]+)", log_text)
    controlled_final_regression_abort = (
        "RuntimeError: final_eval_regressed_vs_baseline" in log_text
        and baseline_eval_loss is not None
        and final_eval_loss is not None
        and final_eval_loss > baseline_eval_loss
    )
    baseline_complete = baseline_eval_loss is not None or "eval_loss_eval_progress 96/96" in log_text
    final_complete = final_eval_loss is not None or "Final eval loss" in log_text
    trainability = try_load_json_after("Applied trainable LoRA module filter:", log_text)
    sampling = try_load_json_after("Sampling:", log_text)

    target_trainable = trainability.get("target_parameter_trainable_lora_params", {}) if trainability else {}
    frozen_by_module = trainability.get("frozen_by_module", {}) if trainability else {}
    trainable_by_module = trainability.get("trainable_by_module", {}) if trainability else {}
    target_mode = trainability.get("target_parameters_trainability_mode") if trainability else None
    share_by_source = sampling.get("weighted_share_by_source", {}) if sampling else {}
    share_by_subcategory = sampling.get("weighted_share_by_subcategory", {}) if sampling else {}

    add(checks, "runtime", "job log contains V501 output repo", "kg1-nemotron-lora-v501" in log_text, "V501 repo marker present")
    add(checks, "runtime", "H200 context visible", "H200" in log_text or "NVIDIA H200" in log_text, "H200 marker present")
    add(
        checks,
        "runtime",
        "no unexpected python traceback",
        "traceback (most recent call last)" not in lower or controlled_final_regression_abort,
        "controlled final-eval regression abort" if controlled_final_regression_abort else "no traceback in log",
    )
    add(
        checks,
        "runtime",
        "no unexpected failure",
        ("runtimeerror" not in lower and "command failed" not in lower) or controlled_final_regression_abort,
        "controlled final-eval regression abort" if controlled_final_regression_abort else "no RuntimeError/command failed in log",
    )
    add(checks, "runtime", "reserved memory under cap", bool(mem_values) and max(mem_values) <= 78.0, f"max_mem_reserved={max(mem_values) if mem_values else None}")

    add(checks, "tokenization", "answer-span weight active", train_weight == 4.0, f"train_answer_span_loss_weight={train_weight}")
    add(checks, "tokenization", "train answer-span examples nonzero", (train_weighted_examples or 0) >= 1712, f"train_weighted_examples={train_weighted_examples}")
    add(checks, "tokenization", "train answer-span tokens above gate", (train_weighted_tokens or 0) >= 1000, f"train_weighted_tokens={train_weighted_tokens}")
    add(checks, "tokenization", "validation answer-span examples nonzero", (val_weighted_examples or 0) >= 428, f"val_weighted_examples={val_weighted_examples}")
    add(checks, "tokenization", "no truncation in summaries", "truncated=0" in log_text and "prompt_truncated=0" in log_text, "truncation markers present")

    add(checks, "training", "planned steps is short V501 smoke", planned_steps == 4, f"planned_steps={planned_steps}")
    add(checks, "training", "trainable percent bounded", trainable_pct is not None and 0.5 <= trainable_pct <= 3.5, f"trainable_pct={trainable_pct}")
    add(checks, "training", "target parameters trainable", target_mode == "trainable", f"target_parameters_trainability_mode={target_mode}")
    add(
        checks,
        "training",
        "MoE target params have trainable tensors",
        int(target_trainable.get("mlp.experts.down_proj", 0)) > 0 and int(target_trainable.get("mlp.experts.gate_up_proj", 0)) > 0,
        json.dumps(target_trainable, sort_keys=True),
    )
    add(checks, "training", "lm_head frozen", int(frozen_by_module.get("lm_head", 0)) > 0 and int(trainable_by_module.get("lm_head", 0)) == 0, f"frozen={frozen_by_module.get('lm_head')} trainable={trainable_by_module.get('lm_head', 0)}")

    add(checks, "sampling", "bit replay guardrail share active", float(share_by_source.get("v498_bit_replay_guardrail_from_v475", 0.0)) >= 0.35, json.dumps(share_by_source, sort_keys=True))
    add(checks, "sampling", "equation share active", float(share_by_source.get("v498_numeric_teacher_trace_pack", 0.0)) >= 0.55, json.dumps(share_by_source, sort_keys=True))
    add(checks, "sampling", "three equation subcategories present", len([k for k in share_by_subcategory if k.startswith("equation_numeric_")]) == 3, json.dumps(share_by_subcategory, sort_keys=True))

    add(checks, "eval", "baseline eval reached progress", "Baseline eval before training" in log_text and "eval_loss_eval_progress" in log_text, "baseline eval is running or complete")
    add(checks, "eval", "baseline eval complete", baseline_complete, f"baseline_eval_loss={baseline_eval_loss}", severity="warn")
    add(checks, "eval", "final eval available", final_complete, f"final_eval_loss={final_eval_loss}", severity="warn")
    if baseline_eval_loss is not None and final_eval_loss is not None:
        add(checks, "eval", "final eval does not regress", final_eval_loss <= baseline_eval_loss, f"baseline={baseline_eval_loss} final={final_eval_loss}")

    hard_failures = [item for item in checks if not item["ok"] and item["severity"] == "fail"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warn"]
    status = "runtime_parameterization_pass_pending_completion" if not hard_failures else "runtime_parameterization_failed"
    if final_eval_loss is not None and baseline_eval_loss is not None and final_eval_loss <= baseline_eval_loss and not hard_failures:
        status = "runtime_parameterization_pass_ready_for_weak_eval_decision"
    if controlled_final_regression_abort:
        status = "candidate_blocked_by_final_eval_regression"

    return {
        "schema_version": "kg1_v503_live_hf_job_parameterization_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "observed": {
            "train_answer_span_loss_weight": train_weight,
            "train_answer_span_weighted_examples": train_weighted_examples,
            "train_answer_span_weighted_tokens": train_weighted_tokens,
            "validation_answer_span_weighted_examples": val_weighted_examples,
            "planned_steps": planned_steps,
            "trainable_pct": trainable_pct,
            "target_parameters_trainability_mode": target_mode,
            "max_mem_reserved_gib": max(mem_values) if mem_values else None,
            "baseline_eval_loss": baseline_eval_loss,
            "final_eval_loss": final_eval_loss,
            "weighted_share_by_source": share_by_source,
            "weighted_share_by_subcategory": share_by_subcategory,
        },
        "checks": checks,
        "decision": {
            "status": status,
            "hard_failures": len(hard_failures),
            "warnings": len(warnings),
            "human_action_required": False,
            "next_action": (
                "Do not weak-eval or submit V501; keep V290 checkpoint-6 as submit-safe baseline and return to CPU/teacher search."
                if controlled_final_regression_abort
                else "Continue monitoring. Do not weak-eval until final eval is complete and non-regressive."
            ),
        },
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# KG1 V503 Live HF Job Parameterization Audit",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        f"Job ID: `{manifest['job_id']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Hard failures: `{manifest['decision']['hard_failures']}`",
        f"- Warnings: `{manifest['decision']['warnings']}`",
        f"- Next action: {manifest['decision']['next_action']}",
        "",
        "## Observed Runtime Parameters",
        "",
        "```json",
        json.dumps(manifest["observed"], indent=2, sort_keys=True),
        "```",
        "",
        "## Checks",
        "",
        "| Area | Check | Verdict | Detail |",
        "|---|---|---:|---|",
    ]
    for item in manifest["checks"]:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        lines.append(f"| {item['area']} | {item['name']} | `{verdict}` | {item['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_text = Path(args.log_path).read_text(encoding="utf-8", errors="replace")
    manifest = audit_log(log_text, args.job_id)
    manifest_path = out_dir / "v503_live_hf_job_parameterization_manifest.json"
    report_path = out_dir / "KG1_V503_LIVE_HF_JOB_PARAMETERIZATION.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, manifest)
    print("manifest =", manifest_path, flush=True)
    print("report =", report_path, flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    for item in manifest["checks"]:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        print(f"{verdict}: {item['area']}::{item['name']} - {item['detail']}", flush=True)
    return 1 if manifest["decision"]["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

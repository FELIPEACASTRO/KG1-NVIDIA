#!/usr/bin/env python3
"""Double-check KG1 parameterization across loss, ACC, data, LoRA and gates."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import box_answer, extract_final_answer, extract_final_answer_for_expected, verify_answer  # noqa: E402


V498_DIR = REPO_ROOT / "artifacts/v498_numeric_teacher_trace_pack/20260516T_v498_numeric_teacher"
V499_DIR = REPO_ROOT / "artifacts/v499_hf_nemo_h200_v498_numeric_teacher_trace_launch"
V500_DIR = REPO_ROOT / "artifacts/v500_v499_parameterization_audit"
V501_DIR = REPO_ROOT / "artifacts/v501_hf_nemo_h200_v498_answer_span_weighted_launch"
OUT_DIR = REPO_ROOT / "artifacts/v502_solution_parameterization_double_check"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add(checks: list[dict[str, Any]], area: str, name: str, ok: bool, detail: str, severity: str = "fail") -> None:
    checks.append({"area": area, "name": name, "ok": bool(ok), "detail": detail, "severity": severity})


def parse_constant(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return None
    return None


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    families = Counter(str(row.get("family", "")) for row in rows)
    sources = Counter(str(row.get("source", "")) for row in rows)
    subcats = Counter(str(row.get("subcategory", "")) for row in rows)
    final_extraction_fail = 0
    verify_fail = 0
    suspicious_answers = 0
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        text = assistant_text(row)
        extracted = extract_final_answer_for_expected(text, answer)
        if not extracted or extracted == "NOT_FOUND":
            final_extraction_fail += 1
        if not verify_answer(answer, extracted):
            verify_fail += 1
        if any(char in answer for char in ["\u2013", "\u2014", "\u2212", "\ufeff", "\u00a0"]):
            suspicious_answers += 1
    return {
        "rows": len(rows),
        "duplicate_ids": len(ids) - len(set(ids)),
        "families": dict(sorted(families.items())),
        "sources": dict(sorted(sources.items())),
        "subcategories": dict(sorted(subcats.items())),
        "final_extraction_fail": final_extraction_fail,
        "verify_fail": verify_fail,
        "suspicious_answers": suspicious_answers,
    }


def command_script_fragment(source: str) -> str:
    match = re.search(r"base\.COMMAND_SCRIPT = \(\s*base\.COMMAND_SCRIPT(?P<body>.*?)\n\s*\)", source, re.S)
    return match.group("body") if match else ""


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    checks = manifest["checks"]
    lines = [
        "# KG1 V502 Solution Parameterization Double Check",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Human action required: `{manifest['decision']['human_action_required']}`",
        f"- Next action: {manifest['decision']['next_action']}",
        f"- Reason: {manifest['decision']['reason']}",
        "",
        "## Checks",
        "",
        "| Area | Check | Verdict | Detail |",
        "|---|---|---:|---|",
    ]
    for item in checks:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        lines.append(f"| {item['area']} | {item['name']} | `{verdict}` | {item['detail']} |")
    lines.extend(
        [
            "",
            "## Practical Conclusion",
            "",
            "- V499 was structurally correct but not objective-correct for ACC: final eval loss was flat/slightly worse and answer-span weighting was inactive.",
            "- The next valid train path is V501: same verified V498 dataset, same V290 checkpoint-6, same MoE target parameters, frozen `lm_head`, but answer-span-weighted loss active before launch.",
            "- Weak ACC eval remains blocked until a local training objective improves or a deterministic CPU gate provides new no-loss evidence.",
            "- The current measurable target remains: `total>192`, `equation>=60`, `bit>=136`, `trunc=0`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    train_rows = read_jsonl(V498_DIR / "v498_numeric_teacher_trace_train.jsonl")
    val_rows = read_jsonl(V498_DIR / "v498_numeric_teacher_trace_val.jsonl")
    train_stats = row_stats(train_rows)
    val_stats = row_stats(val_rows)
    add(checks, "dataset", "train extraction verifies answers", train_stats["verify_fail"] == 0, json.dumps(train_stats, sort_keys=True))
    add(checks, "dataset", "val extraction verifies answers", val_stats["verify_fail"] == 0, json.dumps(val_stats, sort_keys=True))
    add(checks, "dataset", "no suspicious answer chars", train_stats["suspicious_answers"] == 0 and val_stats["suspicious_answers"] == 0, "checked minus/nbsp/bom/dashes")
    add(checks, "dataset", "family mix is intentional", train_stats["families"] == {"bit_manipulation": 512, "equation_transform": 1200}, json.dumps(train_stats["families"], sort_keys=True))

    token_manifest = read_json(V498_DIR / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json")
    add(checks, "tokenization", "gate passed", token_manifest.get("decision", {}).get("status") == "tokenization_gate_passed", str(token_manifest.get("decision", {})))
    add(checks, "tokenization", "runtime max length safe", int(token_manifest["tokenization"]["train"]["token_max"]) <= 1024, f"train token_max={token_manifest['tokenization']['train']['token_max']}")
    add(checks, "tokenization", "no prompt truncation", token_manifest["tokenization"]["train"]["prompt_truncation_rate"] == 0.0 and token_manifest["tokenization"]["validation"]["prompt_truncation_rate"] == 0.0, "train/val prompt truncation 0")
    add(checks, "tokenization", "offset masks complete", token_manifest["tokenization"]["train"]["offset_masks"] == 1712 and token_manifest["tokenization"]["validation"]["offset_masks"] == 428, "train=1712 val=428")

    v499_audit = read_json(V500_DIR / "v500_v499_parameterization_audit_manifest.json")
    add(checks, "v499", "technical gates passed", v499_audit["decision"]["hard_failures"] == 0, json.dumps(v499_audit["decision"], sort_keys=True))
    add(checks, "v499", "blocked by objective warning", v499_audit["decision"]["warnings"] >= 2, "expected warnings: eval loss, answer span, smoke steps", severity="warn")
    add(checks, "v499", "final eval did not improve", (v499_audit["loss_delta"] or 0) >= 0, f"loss_delta={v499_audit['loss_delta']}", severity="warn")

    v501_launcher = V501_DIR / "launch_v501_hf_nemo_h200_v498_answer_span_weighted.py"
    source = v501_launcher.read_text(encoding="utf-8")
    constants = {name: parse_constant(source, name) for name in ["VERSION", "RUN_ID", "OUTPUT_REPO", "MAX_STEPS", "ANSWER_SPAN_LOSS_WEIGHT", "ANSWER_SPAN_MIN_WEIGHTED_TOKENS", "SOURCE_WEIGHTS", "TRAINABLE_LORA_MODULES"]}
    fragment = command_script_fragment(source)
    add(checks, "v501", "version/output repo are v501", "v501" in str(constants["VERSION"]) and "v501" in str(constants["OUTPUT_REPO"]), json.dumps(constants, sort_keys=True))
    add(checks, "v501", "max steps non-smoke", constants["MAX_STEPS"] == 4, f"MAX_STEPS={constants['MAX_STEPS']}")
    add(checks, "v501", "answer span weight active", float(constants["ANSWER_SPAN_LOSS_WEIGHT"]) > 1.0, f"ANSWER_SPAN_LOSS_WEIGHT={constants['ANSWER_SPAN_LOSS_WEIGHT']}")
    add(checks, "v501", "answer span minimum nontrivial", int(constants["ANSWER_SPAN_MIN_WEIGHTED_TOKENS"]) >= 1000, f"ANSWER_SPAN_MIN_WEIGHTED_TOKENS={constants['ANSWER_SPAN_MIN_WEIGHTED_TOKENS']}")
    add(checks, "v501", "command overrides max steps", "export MAX_STEPS=2" in fragment and f"export MAX_STEPS={{MAX_STEPS}}" in fragment, "launcher rewrites inherited V493 MAX_STEPS")
    add(checks, "v501", "command enforces final eval baseline", "REQUIRE_FINAL_EVAL_LTE_BASELINE=1" in fragment, "final eval must not exceed baseline")
    add(checks, "v501", "bit replay still overweighted", "v498_bit_replay_guardrail_from_v475=1.50" in str(constants["SOURCE_WEIGHTS"]), str(constants["SOURCE_WEIGHTS"]))
    add(checks, "v501", "lm_head excluded from trainable filter", "lm_head" not in str(constants["TRAINABLE_LORA_MODULES"]).split(","), str(constants["TRAINABLE_LORA_MODULES"]))

    train_script = (REPO_ROOT / "scripts/hf_job_train_v90.py").read_text(encoding="utf-8")
    add(checks, "train_script", "answer span weighting implemented", "ANSWER_SPAN_LOSS_WEIGHT > 1.0" in train_script and "answer_span_weighted_examples" in train_script, "weighting branch and counters present")
    add(checks, "train_script", "min weighted token gate implemented", "ANSWER_SPAN_MIN_WEIGHTED_TOKENS" in train_script and "answer_span_weighted_tokens < ANSWER_SPAN_MIN_WEIGHTED_TOKENS" in train_script, "gate present")
    add(checks, "train_script", "masked CE normalizes by weighted tokens", "masked_loss.sum() / num_unmasked" in train_script, "masked loss denominator uses mask sum")
    add(checks, "train_script", "weighted replacement sampling implemented", "weighted_replacement" in train_script and "random.choices" in train_script and "weights=weights" in train_script, "weighted sampling path present")
    add(checks, "train_script", "final eval baseline abort implemented", "final_eval_loss > baseline_eval_loss + MAX_FINAL_EVAL_REGRESSION" in train_script, "baseline gate present")

    objective_script = (REPO_ROOT / "scripts/audit_v478_training_objective_alignment.py").read_text(encoding="utf-8")
    add(checks, "objective_gate", "bit/equation share gates present", "min-bit-effective-share" in objective_script and "max-equation-effective-share" in objective_script, "share gates present")

    extraction_cases = [
        ("00000101", r"Reasoning\nFinal answer: \boxed{00000101}", True),
        ("30", r"Final answer: \boxed{30}", True),
        ("-4", r"The final answer is: -4", True),
        ("a{b}\\c", r"Final answer: \boxed{a\{b\}\\c}", True),
    ]
    extraction_results = []
    for expected, predicted, expected_ok in extraction_cases:
        extracted = extract_final_answer_for_expected(predicted, expected)
        ok = verify_answer(expected, extracted)
        extraction_results.append({"expected": expected, "extracted": extracted, "ok": ok})
    add(checks, "acc_metric", "final-answer extraction unit cases", all(item["ok"] for item in extraction_results), json.dumps(extraction_results, sort_keys=True))
    add(checks, "acc_metric", "verify_answer requires extracted answer", not verify_answer("30", r"Final answer: \boxed{30}"), "raw CoT is not accepted directly; eval must extract first")
    add(checks, "acc_metric", "box_answer roundtrip", extract_final_answer(box_answer("abc")) == "abc", f"box_answer={box_answer('abc')}")

    hard_failures = [item for item in checks if not item["ok"] and item["severity"] == "fail"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warn"]
    decision = {
        "status": "double_check_pass_next_debug_v501" if not hard_failures else "double_check_failed",
        "human_action_required": False,
        "next_action": "Run V501 debug-only; launch H200 only if debug proves answer-span weighting and all gates are active.",
        "reason": "Core parameterization is now aligned for the next attempt; V499 remains blocked from weak eval.",
        "hard_failures": len(hard_failures),
        "warnings": len(warnings),
    }
    manifest = {
        "schema_version": "kg1_v502_solution_parameterization_double_check_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "constants_v501": constants,
        "train_stats": train_stats,
        "validation_stats": val_stats,
        "extraction_results": extraction_results,
        "checks": checks,
        "decision": decision,
    }
    manifest_path = OUT_DIR / "v502_solution_parameterization_double_check_manifest.json"
    report_path = OUT_DIR / "KG1_V502_SOLUTION_PARAMETERIZATION_DOUBLE_CHECK.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, manifest)
    print("manifest =", manifest_path, flush=True)
    print("report =", report_path, flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    for item in checks:
        verdict = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warn" else "FAIL")
        print(f"{verdict}: {item['area']}::{item['name']} - {item['detail']}", flush=True)
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

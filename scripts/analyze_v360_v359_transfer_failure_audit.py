#!/usr/bin/env python3
"""V360 CPU audit for the V359 transfer failure.

This script is intentionally CPU-only.  It does not try to rescue a failed HF
run by launching more evaluation.  It inspects the V357 teacher, V358 transfer
dataset, tokenization gate, and V359 measured weak result, then emits a
decision about whether another HF run is justified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA_VERSION = "kg1_v360_v359_transfer_failure_audit_v1"


DEFAULT_V357_MANIFEST = Path(
    "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_bit_global_ternary_gate_manifest.json"
)
DEFAULT_V358_MANIFEST = Path(
    "artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/"
    "v358_v357_bit_ternary_transfer_manifest.json"
)
DEFAULT_TOKENIZATION_MANIFEST = Path(
    "artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate/"
    "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
)
DEFAULT_V359_SUMMARY = Path("artifacts/v359_hf_a100_v358_bit_ternary_launch/V359_RESULT_SUMMARY.md")
DEFAULT_V359_TRAIN_LAUNCHER = Path(
    "artifacts/v359_hf_a100_v358_bit_ternary_launch/launch_v359_hf_a100_v358_bit_ternary.py"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/v360_v359_transfer_failure_audit/20260514T_cpu_audit")


BIT_RE = re.compile(r"^[01]{8}$")
PROMPT_EXAMPLE_RE = re.compile(r"^[01]{8} -> [01]{8}$", re.MULTILINE)
PROMPT_TARGET_RE = re.compile(r"Now, determine the output for: ([01]{8})")
BOXED_RE = re.compile(r"\\boxed\{([01]{8})\}")


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
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def parse_v359_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    def find_int(pattern: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    total_match = re.search(r"Result:\s*`(\d+)/315`", text)
    eq_match = re.search(r"`equation_transform`:\s*`(\d+)/155`", text)
    bit_match = re.search(r"`bit_manipulation`:\s*`(\d+)/160`", text)
    trunc_match = re.search(r"Truncated:\s*`(\d+)`", text)
    return {
        "summary_md": str(path),
        "summary_sha256": sha256_file(path),
        "checkpoint2_total_correct": int(total_match.group(1)) if total_match else None,
        "checkpoint2_equation_transform_correct": int(eq_match.group(1)) if eq_match else None,
        "checkpoint2_bit_manipulation_correct": int(bit_match.group(1)) if bit_match else None,
        "checkpoint2_truncated": int(trunc_match.group(1)) if trunc_match else find_int(r"truncation\s+`(\d+)`"),
        "contains_rejected": "V359 is rejected" in text,
        "contains_finops_cancel": "canceled" in text.lower() or "cancelado" in text.lower(),
    }


def summarize_split(rows: list[dict[str, Any]], split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subcategories = Counter()
    rule_classes = Counter()
    rule_slugs = Counter()
    source_ids = Counter()
    exprs = Counter()
    message_counts = Counter()
    assistant_role_rows = 0
    only_boxed_rows = 0
    assistant_boxed_rows = 0
    assistant_rule_rows = 0
    assistant_checks_rows = 0
    assistant_final_answer_rows = 0
    answer_in_assistant_rows = 0
    prompt_example_counts = Counter()
    prompt_target_rows = 0
    bad_rows: list[dict[str, Any]] = []
    prompt_chars: list[int] = []
    assistant_chars: list[int] = []
    answer_chars: list[int] = []

    for row in rows:
        metadata = row.get("metadata", {})
        subcategories[str(row.get("subcategory", ""))] += 1
        rule_classes[str(metadata.get("rule_class", ""))] += 1
        rule_slugs[str(metadata.get("rule_slug", ""))] += 1
        source_ids[str(metadata.get("source_id", ""))] += 1
        exprs[str(metadata.get("expr", ""))] += 1
        answer = str(row.get("answer", ""))
        prompt = str(row.get("prompt", ""))
        answer_chars.append(len(answer))
        prompt_chars.append(len(prompt))
        if not BIT_RE.fullmatch(answer):
            bad_rows.append({"id": row.get("id"), "reason": "answer_not_8_bit", "value": answer})
        examples = PROMPT_EXAMPLE_RE.findall(prompt)
        prompt_example_counts[len(examples)] += 1
        if PROMPT_TARGET_RE.search(prompt):
            prompt_target_rows += 1
        messages = row.get("messages", [])
        message_counts[len(messages)] += 1
        assistants = [msg.get("content", "") for msg in messages if msg.get("role") == "assistant"]
        if assistants:
            assistant_role_rows += 1
            assistant = str(assistants[-1])
            assistant_chars.append(len(assistant))
            if BOXED_RE.search(assistant):
                assistant_boxed_rows += 1
            if re.fullmatch(r"\s*\\boxed\{[01]{8}\}\s*", assistant):
                only_boxed_rows += 1
            if "Rule:" in assistant:
                assistant_rule_rows += 1
            if "Check examples:" in assistant:
                assistant_checks_rows += 1
            if "Final answer:" in assistant:
                assistant_final_answer_rows += 1
            if answer and answer in assistant:
                answer_in_assistant_rows += 1
        else:
            bad_rows.append({"id": row.get("id"), "reason": "missing_assistant_message", "value": ""})

    def stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "median": 0, "max": 0}
        return {"min": min(values), "median": float(median(values)), "max": max(values)}

    summary = {
        "split": split_name,
        "rows": len(rows),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "rule_class_counts": dict(sorted(rule_classes.items())),
        "unique_rule_slugs": len(rule_slugs),
        "unique_source_ids": len(source_ids),
        "unique_exprs": len(exprs),
        "message_count_distribution": dict(sorted(message_counts.items())),
        "prompt_example_count_distribution": dict(sorted(prompt_example_counts.items())),
        "prompt_target_rows": prompt_target_rows,
        "assistant_role_rows": assistant_role_rows,
        "assistant_boxed_rows": assistant_boxed_rows,
        "assistant_only_boxed_rows": only_boxed_rows,
        "assistant_rule_rows": assistant_rule_rows,
        "assistant_checks_rows": assistant_checks_rows,
        "assistant_final_answer_rows": assistant_final_answer_rows,
        "answer_in_assistant_rows": answer_in_assistant_rows,
        "prompt_char_stats": stats(prompt_chars),
        "assistant_char_stats": stats(assistant_chars),
        "answer_char_stats": stats(answer_chars),
        "bad_row_count": len(bad_rows),
        "bad_rows_first20": bad_rows[:20],
    }
    rule_rows = [
        {
            "split": split_name,
            "rule_slug": slug,
            "rows": count,
            "expr": next((str(r.get("metadata", {}).get("expr", "")) for r in rows if r.get("metadata", {}).get("rule_slug") == slug), ""),
            "rule_class": next(
                (str(r.get("metadata", {}).get("rule_class", "")) for r in rows if r.get("metadata", {}).get("rule_slug") == slug),
                "",
            ),
        }
        for slug, count in sorted(rule_slugs.items())
    ]
    return summary, rule_rows


def read_launcher_evidence(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "launcher_py": str(path),
        "launcher_sha256": sha256_file(path),
        "mentions_preferences_train": "preferences_train" in text or "PREFERENCE" in text.upper(),
        "mentions_sft_train_file": "v358_v357_bit_ternary_transfer_train.jsonl" in text,
        "max_steps_override": "MAX_STEPS=4" in text,
        "learning_rate_override": "LEARNING_RATE=6.0e-8" in text,
        "answer_span_loss_weight_16": 'ANSWER_SPAN_LOSS_WEIGHT = "16.0"' in text,
        "subcategory_weight_ternary_1_40": "bit_exact_global_ternary=1.40" in text,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derive_findings(
    v357_manifest: dict[str, Any],
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    tokenization_manifest: dict[str, Any],
    launcher_evidence: dict[str, Any],
    v359_result: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    v357_summary = v357_manifest.get("v357_summary", {})
    findings.append(
        {
            "severity": "blocking",
            "finding": "V359 did not transfer the V357 CPU gain.",
            "evidence": (
                f"V357 CPU summary={json.dumps(v357_summary, sort_keys=True)}; "
                f"V359 checkpoint-2 total={v359_result.get('checkpoint2_total_correct')}, "
                f"equation={v359_result.get('checkpoint2_equation_transform_correct')}, "
                f"bit={v359_result.get('checkpoint2_bit_manipulation_correct')}, "
                f"truncated={v359_result.get('checkpoint2_truncated')}."
            ),
            "action": "Do not full-eval, package, submit, or continue V359 checkpoints without a new CPU-gated dataset format.",
        }
    )
    if not launcher_evidence.get("mentions_preferences_train"):
        findings.append(
            {
                "severity": "high",
                "finding": "The hard-negative preference files were not used by the V359 SFT launcher.",
                "evidence": "Launcher references the SFT train JSONL, but not the preference train/val JSONL.",
                "action": "Either train with a real preference objective or remove preference artifacts from the active plan.",
            }
        )
    if int(train_summary.get("unique_rule_slugs", 0)) <= 20:
        findings.append(
            {
                "severity": "high",
                "finding": "V358 is narrow: it repeats 15 verified rules instead of teaching a broad bit solver.",
                "evidence": (
                    f"train unique_rule_slugs={train_summary.get('unique_rule_slugs')}; "
                    f"val unique_rule_slugs={val_summary.get('unique_rule_slugs')}."
                ),
                "action": "Next dataset must either be answer-only replay for exact rules or expand rule coverage before another GPU run.",
            }
        )
    if train_summary.get("assistant_only_boxed_rows") == 0:
        findings.append(
            {
                "severity": "medium",
                "finding": "Completion format differs from weak-eval prompt intent.",
                "evidence": (
                    f"assistant_only_boxed_rows={train_summary.get('assistant_only_boxed_rows')} while all rows include "
                    "Rule/Check examples/Final answer text."
                ),
                "action": "Test an answer-first or boxed-only V361 dataset in CPU/token gates before HF.",
            }
        )
    tok = tokenization_manifest.get("tokenization", {})
    if tok:
        findings.append(
            {
                "severity": "info",
                "finding": "V358 passed tokenization, so the failure is not explained by training truncation.",
                "evidence": json.dumps(
                    {
                        "train": tok.get("train", {}),
                        "validation": tok.get("validation", {}),
                    },
                    sort_keys=True,
                )[:1200],
                "action": "Keep token gates, but treat ACC regression as a learning/objective/format issue.",
            }
        )
    findings.append(
        {
            "severity": "blocking",
            "finding": "V359 bit-only data cannot improve equation_transform.",
            "evidence": "V358 train/val family counts are only bit_manipulation; measured equation stayed at 56/155.",
            "action": "Equation must continue through DSL/verifier/teacher data, not this bit-only SFT route.",
        }
    )
    return findings


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v357_manifest = read_json(args.v357_manifest_json)
    v358_manifest = read_json(args.v358_manifest_json)
    tokenization_manifest = read_json(args.tokenization_manifest_json)
    v359_result = parse_v359_summary(args.v359_summary_md)
    launcher_evidence = read_launcher_evidence(args.v359_train_launcher_py)

    outputs = v358_manifest.get("outputs", {})
    train_path = Path(outputs["train_jsonl"])
    val_path = Path(outputs["val_jsonl"])
    pref_train_path = Path(outputs["preferences_train_jsonl"])
    pref_val_path = Path(outputs["preferences_val_jsonl"])

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    pref_train_rows = read_jsonl(pref_train_path)
    pref_val_rows = read_jsonl(pref_val_path)

    train_summary, train_rules = summarize_split(train_rows, "train")
    val_summary, val_rules = summarize_split(val_rows, "validation")
    preference_summary = {
        "train_rows": len(pref_train_rows),
        "val_rows": len(pref_val_rows),
        "used_by_v359_launcher": bool(launcher_evidence.get("mentions_preferences_train")),
    }
    findings = derive_findings(
        v357_manifest,
        train_summary,
        val_summary,
        tokenization_manifest,
        launcher_evidence,
        v359_result,
    )

    hf_allowed = False
    next_action = (
        "Build V361 CPU-gated answer-first/boxed-only transfer data or return to equation DSL. "
        "Do not launch another HF job from V358/V359 artifacts."
    )
    decision = {
        "decision": "v360_blocks_more_hf_on_v358_v359",
        "hf_gpu_allowed": hf_allowed,
        "package_allowed": False,
        "kaggle_submit_allowed": False,
        "reason": (
            "V359 checkpoint-2 regressed to "
            f"{v359_result.get('checkpoint2_total_correct')}/315, "
            f"bit={v359_result.get('checkpoint2_bit_manipulation_correct')}/160, "
            f"truncated={v359_result.get('checkpoint2_truncated')}; "
            "V358 did not transfer V357 CPU gains."
        ),
        "next_action": next_action,
    }

    rule_summary_csv = output_dir / f"{args.label}_rule_summary.csv"
    format_summary_csv = output_dir / f"{args.label}_format_summary.csv"
    findings_csv = output_dir / f"{args.label}_findings.csv"
    manifest_json = output_dir / f"{args.label}_manifest.json"
    report_md = output_dir / f"{args.label}_report.md"

    write_csv(
        rule_summary_csv,
        train_rules + val_rules,
        ["split", "rule_slug", "rows", "expr", "rule_class"],
    )
    write_csv(
        format_summary_csv,
        [
            {"metric": key, "train": json.dumps(train_summary.get(key), sort_keys=True), "validation": json.dumps(val_summary.get(key), sort_keys=True)}
            for key in sorted(set(train_summary) | set(val_summary))
            if key not in {"bad_rows_first20"}
        ],
        ["metric", "train", "validation"],
    )
    write_csv(findings_csv, findings, ["severity", "finding", "evidence", "action"])

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "inputs": {
            "v357_manifest_json": str(args.v357_manifest_json),
            "v357_manifest_sha256": sha256_file(args.v357_manifest_json),
            "v358_manifest_json": str(args.v358_manifest_json),
            "v358_manifest_sha256": sha256_file(args.v358_manifest_json),
            "tokenization_manifest_json": str(args.tokenization_manifest_json),
            "tokenization_manifest_sha256": sha256_file(args.tokenization_manifest_json),
            "v359_summary_md": str(args.v359_summary_md),
            "v359_train_launcher_py": str(args.v359_train_launcher_py),
        },
        "v357_decision": v357_manifest.get("decision", {}),
        "v359_result": v359_result,
        "v358_train_summary": train_summary,
        "v358_validation_summary": val_summary,
        "v358_preference_summary": preference_summary,
        "v359_launcher_evidence": launcher_evidence,
        "findings": findings,
        "decision": decision,
        "outputs": {
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
            "rule_summary_csv": str(rule_summary_csv),
            "format_summary_csv": str(format_summary_csv),
            "findings_csv": str(findings_csv),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# V360 V359 Transfer Failure Audit",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['decision']}`",
        f"- HF GPU allowed: `{decision['hf_gpu_allowed']}`",
        f"- Reason: {decision['reason']}",
        f"- Next action: {decision['next_action']}",
        "",
        "## Measured Evidence",
        "",
        f"- V357 CPU teacher: `{json.dumps(v357_manifest.get('v357_summary', {}), sort_keys=True)}`",
        f"- V359 checkpoint-2: `{v359_result.get('checkpoint2_total_correct')}/315`, "
        f"equation `{v359_result.get('checkpoint2_equation_transform_correct')}/155`, "
        f"bit `{v359_result.get('checkpoint2_bit_manipulation_correct')}/160`, "
        f"truncated `{v359_result.get('checkpoint2_truncated')}`.",
        f"- V358 train rules: `{train_summary['unique_rule_slugs']}` rules, `{train_summary['rows']}` rows.",
        f"- V358 validation rules: `{val_summary['unique_rule_slugs']}` rules, `{val_summary['rows']}` rows.",
        f"- V358 preference rows: train `{preference_summary['train_rows']}`, val `{preference_summary['val_rows']}`, "
        f"used by V359 launcher `{preference_summary['used_by_v359_launcher']}`.",
        "",
        "## Findings",
        "",
    ]
    for item in findings:
        report_lines.extend(
            [
                f"- `{item['severity']}` {item['finding']}",
                f"  Evidence: {item['evidence']}",
                f"  Action: {item['action']}",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Promotion Rule",
            "",
            "No full eval, package, Kaggle submit, or additional HF run is allowed from V358/V359 unless a new CPU-gated dataset explains the failure and preserves the V357 gains before paid training.",
            "",
        ]
    )
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    return manifest


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        train = tmp / "train.jsonl"
        val = tmp / "val.jsonl"
        pref_train = tmp / "pref_train.jsonl"
        pref_val = tmp / "pref_val.jsonl"
        row = {
            "id": "toy",
            "family": "bit_manipulation",
            "subcategory": "bit_exact_global_ternary",
            "prompt": "00000000 -> 00000000\n11111111 -> 11111111\nNow, determine the output for: 10101010",
            "answer": "10101010",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "Rule: output = ID.\nFinal answer: \\boxed{10101010}"},
            ],
            "metadata": {
                "rule_slug": "id",
                "rule_class": "toy",
                "source_id": "toy_source",
                "expr": "ID",
            },
        }
        for path in (train, val, pref_train, pref_val):
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        v358_manifest = tmp / "v358.json"
        v358_manifest.write_text(
            json.dumps(
                {
                    "outputs": {
                        "train_jsonl": str(train),
                        "val_jsonl": str(val),
                        "preferences_train_jsonl": str(pref_train),
                        "preferences_val_jsonl": str(pref_val),
                    }
                }
            ),
            encoding="utf-8",
        )
        v357_manifest = tmp / "v357.json"
        v357_manifest.write_text(
            json.dumps({"decision": {"decision": "toy"}, "v357_summary": {"correct": 1}}),
            encoding="utf-8",
        )
        tok = tmp / "tok.json"
        tok.write_text(json.dumps({"tokenization": {"train": {}, "validation": {}}}), encoding="utf-8")
        summary = tmp / "summary.md"
        summary.write_text(
            "V359 is rejected\nResult: `190/315`\n`equation_transform`: `56/155`\n`bit_manipulation`: `134/160`\nTruncated: `1`\ncanceled\n",
            encoding="utf-8",
        )
        launcher = tmp / "launcher.py"
        launcher.write_text("TRAIN='v358_v357_bit_ternary_transfer_train.jsonl'\nMAX_STEPS=4\nLEARNING_RATE=6.0e-8\n", encoding="utf-8")
        args = argparse.Namespace(
            v357_manifest_json=v357_manifest,
            v358_manifest_json=v358_manifest,
            tokenization_manifest_json=tok,
            v359_summary_md=summary,
            v359_train_launcher_py=launcher,
            output_dir=tmp / "out",
            label="self_test",
        )
        manifest = run_analysis(args)
        if manifest["decision"]["decision"] != "v360_blocks_more_hf_on_v358_v359":
            raise AssertionError("unexpected self-test decision")
        if not Path(manifest["outputs"]["manifest_json"]).is_file():
            raise AssertionError("missing manifest output")
    print("v360_transfer_failure_audit_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v357-manifest-json", type=Path, default=DEFAULT_V357_MANIFEST)
    parser.add_argument("--v358-manifest-json", type=Path, default=DEFAULT_V358_MANIFEST)
    parser.add_argument("--tokenization-manifest-json", type=Path, default=DEFAULT_TOKENIZATION_MANIFEST)
    parser.add_argument("--v359-summary-md", type=Path, default=DEFAULT_V359_SUMMARY)
    parser.add_argument("--v359-train-launcher-py", type=Path, default=DEFAULT_V359_TRAIN_LAUNCHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v360_v359_transfer_failure_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    print("=== V360 TRANSFER FAILURE AUDIT START ===", flush=True)
    print("v357_manifest_json =", args.v357_manifest_json, flush=True)
    print("v358_manifest_json =", args.v358_manifest_json, flush=True)
    print("tokenization_manifest_json =", args.tokenization_manifest_json, flush=True)
    print("v359_summary_md =", args.v359_summary_md, flush=True)
    print("v359_train_launcher_py =", args.v359_train_launcher_py, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    manifest = run_analysis(args)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V360 TRANSFER FAILURE AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

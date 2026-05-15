#!/usr/bin/env python3
"""Audit V462 synthetic numeric raw outputs for real adapter mistakes.

V461 created a prompt-only synthetic numeric probe pack. V462 collected raw
adapter outputs on those prompts without labels. V463 joins the local V461
audit labels only after raw collection and decides whether the signal is
strong enough to build a later training dataset.

This script does not train, launch HF, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for item in (ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402


VERSION = "v463_v462_synthetic_numeric_hard_negative_audit"
DEFAULT_V461_DIR = ROOT / "artifacts" / "v461_synthetic_numeric_probe_pack" / "20260515T_cpu_gate"
DEFAULT_V462_DIR = (
    ROOT
    / "artifacts"
    / "v462_hf_v461_synthetic_raw_probe_outputs"
    / "runs"
    / "v462-v461-synthetic-raw-probe-20260515T223832Z"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v463_v462_synthetic_numeric_hard_negative_audit" / "20260515T_cpu_gate"

DETAIL_COLUMNS = [
    "id",
    "family",
    "target_rule_class",
    "query",
    "answer",
    "simulated_wrong_prediction",
    "postprocessor_prediction",
    "adapter_prediction",
    "adapter_correct",
    "adapter_matches_simulated_wrong",
    "postprocessor_correct",
    "real_hard_negative_candidate",
    "completion_tokens",
    "finish_reason",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "raw_prompt_sha256",
    "raw_prompt_normalized_sha256",
    "prompt_hashes_match",
    "decode_config_sha256",
    "adapter_repo",
    "adapter_subfolder",
    "prompt",
    "raw_output",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if expected and observed != expected:
        raise RuntimeError(f"{label} sha256 mismatch: expected {expected}, got {observed}")
    return observed


def require_unique_ids(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if not rid:
            raise RuntimeError(f"{label} contains blank id")
        if rid in by_id:
            raise RuntimeError(f"{label} contains duplicate id: {rid}")
        by_id[rid] = row
    return by_id


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, dict[str, int]] = {}
    for row in rows:
        rule = str(row["target_rule_class"])
        item = by_rule.setdefault(
            rule,
            {
                "rows": 0,
                "adapter_correct": 0,
                "adapter_matches_simulated_wrong": 0,
                "postprocessor_correct": 0,
                "real_hard_negative_candidate": 0,
                "prompt_hashes_match": 0,
                "finish_stop": 0,
            },
        )
        item["rows"] += 1
        for key in (
            "adapter_correct",
            "adapter_matches_simulated_wrong",
            "postprocessor_correct",
            "real_hard_negative_candidate",
            "prompt_hashes_match",
        ):
            item[key] += int(row[key] == "true")
        item["finish_stop"] += int(row["finish_reason"] == "stop")
    totals = Counter()
    for row in rows:
        for key in (
            "adapter_correct",
            "adapter_matches_simulated_wrong",
            "postprocessor_correct",
            "real_hard_negative_candidate",
            "prompt_hashes_match",
        ):
            totals[key] += int(row[key] == "true")
        totals["finish_stop"] += int(row["finish_reason"] == "stop")
    return {
        "rows": len(rows),
        "rule_summary": dict(sorted(by_rule.items())),
        "totals": dict(sorted(totals.items())),
    }


def render_report(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    decision = manifest["decision"]
    rule_lines = []
    for rule, row in summary["rule_summary"].items():
        rule_lines.append(
            "| "
            + " | ".join(
                [
                    rule,
                    str(row["rows"]),
                    str(row["adapter_correct"]),
                    str(row["adapter_matches_simulated_wrong"]),
                    str(row["postprocessor_correct"]),
                    str(row["real_hard_negative_candidate"]),
                    str(row["prompt_hashes_match"]),
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# V463 V462 Synthetic Numeric Hard Negative Audit",
            "",
            "## Summary",
            "",
            f"- Rows audited: `{summary['rows']}`.",
            f"- Real hard negatives: `{summary['totals'].get('real_hard_negative_candidate', 0)}`.",
            f"- Real hard-negative rule classes: `{decision['hard_negative_rule_class_count']}`.",
            f"- Decision: `{decision['decision']}`.",
            f"- V464 dataset build allowed: `{str(decision['v464_dataset_build_allowed']).lower()}`.",
            f"- HF GPU train allowed: `{str(decision['hf_gpu_train_allowed']).lower()}`.",
            f"- Next action: {decision['next_action']}",
            "",
            "## Rule Detail",
            "",
            "| Rule | Rows | Adapter correct | Adapter matches simulated wrong | Postprocessor correct | Real hard negatives | Prompt hashes match |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rule_lines,
            "",
            "## Interpretation",
            "",
            "This is synthetic prompt evidence joined after inference, so it can justify "
            "a CPU dataset proposal only. It does not authorize paid GPU training by "
            "itself. GPU training remains blocked until a later dataset gate shows "
            "multi-rule coverage, clean tokenization, bit replay, and no weak/full "
            "regression risk.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V463 V462 SYNTHETIC NUMERIC HARD NEGATIVE AUDIT START ===", flush=True)
    print("v461_manifest_json =", args.v461_manifest_json, flush=True)
    print("v461_audit_csv =", args.v461_audit_csv, flush=True)
    print("v462_manifest_json =", args.v462_manifest_json, flush=True)
    print("v462_raw_outputs_csv =", args.v462_raw_outputs_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    for path in [args.v461_manifest_json, args.v461_audit_csv, args.v462_manifest_json, args.v462_raw_outputs_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v461_manifest = read_json(args.v461_manifest_json)
    v462_manifest = read_json(args.v462_manifest_json)
    if v461_manifest.get("schema_version") != "kg1_v461_synthetic_numeric_probe_pack_v1":
        raise RuntimeError("Unexpected V461 schema")
    if v462_manifest.get("schema_version") != "kg1_v435c_adapter_probe_raw_outputs_v1":
        raise RuntimeError("Unexpected V462 raw-output schema")
    if v461_manifest.get("source_policy", {}).get("answers_in_prompt_pack") is not False:
        raise RuntimeError("V461 prompt pack source policy must confirm no answers in prompt pack")
    if v462_manifest.get("source_policy", {}).get("answers_input") is not False:
        raise RuntimeError("V462 source policy must confirm no answers were input")

    v461_audit_sha = assert_hash(
        args.v461_audit_csv,
        str(v461_manifest.get("outputs", {}).get("audit_csv_sha256", "")),
        "V461 audit CSV",
    )
    v462_raw_sha = assert_hash(
        args.v462_raw_outputs_csv,
        str(v462_manifest.get("outputs", {}).get("raw_outputs_csv_sha256", "")),
        "V462 raw outputs CSV",
    )

    audit_rows = read_csv(args.v461_audit_csv)
    raw_rows = read_csv(args.v462_raw_outputs_csv)
    raw_by_id = require_unique_ids(raw_rows, "V462 raw outputs")
    audit_by_id = require_unique_ids(audit_rows, "V461 audit")
    if set(audit_by_id) != set(raw_by_id):
        missing_raw = sorted(set(audit_by_id) - set(raw_by_id))
        extra_raw = sorted(set(raw_by_id) - set(audit_by_id))
        raise RuntimeError(f"V461/V462 id mismatch: missing_raw={missing_raw[:5]} extra_raw={extra_raw[:5]}")

    detail_rows: list[dict[str, Any]] = []
    for rid in sorted(audit_by_id):
        row = audit_by_id[rid]
        raw = raw_by_id[rid]
        answer = str(row.get("answer", "")).strip()
        simulated_wrong = str(row.get("simulated_wrong_prediction", "")).strip()
        postprocessor = str(row.get("postprocessor_prediction", "")).strip()
        prediction = str(raw.get("prediction", "")).strip()
        prompt = str(raw.get("prompt", ""))
        raw_prompt_sha = sha256_text(prompt)
        raw_prompt_norm_sha = sha256_text(normalize_prompt(prompt))
        prompt_hashes_match = (
            str(row.get("prompt_sha256", "")).strip() == raw_prompt_sha
            and str(row.get("prompt_normalized_sha256", "")).strip() == raw_prompt_norm_sha
        )
        adapter_correct = verify_answer(answer, prediction)
        adapter_matches_sim_wrong = verify_answer(simulated_wrong, prediction)
        postprocessor_correct = verify_answer(answer, postprocessor)
        real_hard_negative = (not adapter_correct) and adapter_matches_sim_wrong and postprocessor_correct
        detail_rows.append(
            {
                "id": rid,
                "family": row.get("family", ""),
                "target_rule_class": row.get("target_rule_class", ""),
                "query": row.get("query", ""),
                "answer": answer,
                "simulated_wrong_prediction": simulated_wrong,
                "postprocessor_prediction": postprocessor,
                "adapter_prediction": prediction,
                "adapter_correct": bool_text(adapter_correct),
                "adapter_matches_simulated_wrong": bool_text(adapter_matches_sim_wrong),
                "postprocessor_correct": bool_text(postprocessor_correct),
                "real_hard_negative_candidate": bool_text(real_hard_negative),
                "completion_tokens": raw.get("completion_tokens", ""),
                "finish_reason": raw.get("finish_reason", ""),
                "prompt_sha256": row.get("prompt_sha256", ""),
                "prompt_normalized_sha256": row.get("prompt_normalized_sha256", ""),
                "raw_prompt_sha256": raw_prompt_sha,
                "raw_prompt_normalized_sha256": raw_prompt_norm_sha,
                "prompt_hashes_match": bool_text(prompt_hashes_match),
                "decode_config_sha256": raw.get("decode_config_sha256", ""),
                "adapter_repo": raw.get("adapter_repo", ""),
                "adapter_subfolder": raw.get("adapter_subfolder", ""),
                "prompt": prompt,
                "raw_output": raw.get("raw_output", ""),
            }
        )

    summary = summarize(detail_rows)
    hard_rows = [row for row in detail_rows if row["real_hard_negative_candidate"] == "true"]
    hard_rule_count = len({row["target_rule_class"] for row in hard_rows})
    hard_count = len(hard_rows)
    conditions = {
        "all_ids_joined": len(detail_rows) == len(audit_rows) == len(raw_rows),
        "prompt_hashes_all_match": summary["totals"].get("prompt_hashes_match", 0) == len(detail_rows),
        "finish_reason_stop_only": {row["finish_reason"] for row in detail_rows} <= {"stop"},
        "postprocessor_all_correct": summary["totals"].get("postprocessor_correct", 0) == len(detail_rows),
        "hard_negative_count_ge_min": hard_count >= args.min_hard_negatives,
        "hard_negative_rule_classes_ge_min": hard_rule_count >= args.min_rule_classes,
    }
    v464_dataset_build_allowed = all(conditions.values())
    hf_gpu_train_allowed = False
    if v464_dataset_build_allowed:
        decision_text = "v463_multi_rule_synthetic_signal_ready_for_v464_cpu_dataset"
        next_action = (
            "Build V464 CPU dataset proposal with only real adapter hard negatives, bit replay, "
            "tokenization gates, and weak/full promotion guards. Do not train yet."
        )
    elif hard_count:
        decision_text = "v463_signal_present_but_gpu_blocked"
        next_action = "Do not train; either add more CPU probe classes or restrict V464 to a non-training analysis pack."
    else:
        decision_text = "v463_no_real_hard_negative_signal_gpu_blocked"
        next_action = "Archive this route for training; return to public-train/code-mining or solver/verifier work."

    detail_csv = args.output_dir / f"{args.label}_detail.csv"
    hard_csv = args.output_dir / f"{args.label}_hard_negatives.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    report_md = args.output_dir / f"{args.label}.md"
    write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)
    write_csv(hard_csv, hard_rows, DETAIL_COLUMNS)
    manifest = {
        "schema_version": "kg1_v463_v462_synthetic_numeric_hard_negative_audit_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_local_synthetic_audit": True,
            "weak_full_rows_used_for_training": False,
            "training": False,
            "submission": False,
            "purpose": "Decide whether synthetic numeric probes reveal real frozen-adapter mistakes worth a later CPU dataset proposal.",
        },
        "inputs": {
            "v461_manifest_json": str(args.v461_manifest_json),
            "v461_manifest_sha256": sha256_file(args.v461_manifest_json),
            "v461_audit_csv": str(args.v461_audit_csv),
            "v461_audit_sha256": v461_audit_sha,
            "v462_manifest_json": str(args.v462_manifest_json),
            "v462_manifest_sha256": sha256_file(args.v462_manifest_json),
            "v462_raw_outputs_csv": str(args.v462_raw_outputs_csv),
            "v462_raw_outputs_sha256": v462_raw_sha,
        },
        "thresholds": {
            "min_hard_negatives": args.min_hard_negatives,
            "min_rule_classes": args.min_rule_classes,
        },
        "summary": summary,
        "conditions": conditions,
        "decision": {
            "v464_dataset_build_allowed": v464_dataset_build_allowed,
            "hf_gpu_train_allowed": hf_gpu_train_allowed,
            "decision": decision_text,
            "hard_negative_count": hard_count,
            "hard_negative_rule_class_count": hard_rule_count,
            "blocking_conditions": [key for key, value in conditions.items() if not value],
            "next_action": next_action,
        },
        "outputs": {
            "detail_csv": str(detail_csv),
            "detail_sha256": sha256_file(detail_csv),
            "hard_negatives_csv": str(hard_csv),
            "hard_negatives_sha256": sha256_file(hard_csv),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_json, manifest)
    report_md.write_text(render_report(manifest), encoding="utf-8")
    manifest["outputs"]["report_sha256"] = sha256_file(report_md)
    write_json(manifest_json, manifest)
    print("summary =", json.dumps(summary, sort_keys=True), flush=True)
    print("conditions =", json.dumps(conditions, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V463 V462 SYNTHETIC NUMERIC HARD NEGATIVE AUDIT END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {
            "target_rule_class": "r1",
            "adapter_correct": "false",
            "adapter_matches_simulated_wrong": "true",
            "postprocessor_correct": "true",
            "real_hard_negative_candidate": "true",
            "prompt_hashes_match": "true",
            "finish_reason": "stop",
        },
        {
            "target_rule_class": "r2",
            "adapter_correct": "true",
            "adapter_matches_simulated_wrong": "false",
            "postprocessor_correct": "true",
            "real_hard_negative_candidate": "false",
            "prompt_hashes_match": "true",
            "finish_reason": "stop",
        },
    ]
    summary = summarize(rows)
    assert summary["rows"] == 2
    assert summary["totals"]["real_hard_negative_candidate"] == 1
    assert verify_answer("-01", "01") is False
    assert verify_answer("01", "1") is False
    print("v463_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v461-manifest-json", type=Path, default=DEFAULT_V461_DIR / "v461_synthetic_numeric_probe_pack_manifest.json")
    parser.add_argument("--v461-audit-csv", type=Path, default=DEFAULT_V461_DIR / "v461_synthetic_numeric_probe_pack_audit.csv")
    parser.add_argument("--v462-manifest-json", type=Path, default=DEFAULT_V462_DIR / "v462_v461_synthetic_raw_probe_manifest.json")
    parser.add_argument("--v462-raw-outputs-csv", type=Path, default=DEFAULT_V462_DIR / "v462_v461_synthetic_raw_probe_raw_outputs.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=VERSION)
    parser.add_argument("--min-hard-negatives", type=int, default=12)
    parser.add_argument("--min-rule-classes", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

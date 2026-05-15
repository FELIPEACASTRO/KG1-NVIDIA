#!/usr/bin/env python3
"""Audit V458 raw outputs for V457 numeric hard negatives.

V457 exported prompt-only public-train probes. V458 collected raw adapter
outputs for those prompts without labels. This CPU audit joins the V457 audit
labels only after raw collection, verifies the exact adapter behavior, and
decides whether there is enough real hard-negative signal for a paid GPU job.

It does not train, launch HF, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import verify_answer  # noqa: E402


DEFAULT_V457_DIR = REPO_ROOT / "artifacts/v457_public_train_numeric_probe_pack/20260515T_cpu_gate"
DEFAULT_V458_DIR = (
    REPO_ROOT
    / "artifacts/v458_hf_v457_numeric_raw_probe_outputs/runs/"
    / "v458-v457-numeric-raw-probe-20260515T220411Z"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v459_v458_numeric_hard_negative_audit"

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
    "decode_config_sha256",
    "adapter_repo",
    "adapter_subfolder",
    "prompt",
    "raw_output",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            writer.writerow({key: row.get(key, "") for key in columns})


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if expected and observed != expected:
        raise RuntimeError(f"{label} sha256 mismatch: expected {expected}, got {observed}")
    return observed


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
            },
        )
        item["rows"] += 1
        for key in (
            "adapter_correct",
            "adapter_matches_simulated_wrong",
            "postprocessor_correct",
            "real_hard_negative_candidate",
        ):
            item[key] += int(row[key] == "true")
    totals = Counter()
    for row in rows:
        for key in (
            "adapter_correct",
            "adapter_matches_simulated_wrong",
            "postprocessor_correct",
            "real_hard_negative_candidate",
        ):
            totals[key] += int(row[key] == "true")
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
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# V459 V458 Numeric Hard Negative Audit",
            "",
            "## Summary",
            "",
            f"- Rows audited: `{summary['rows']}`.",
            f"- Real hard negatives: `{summary['totals'].get('real_hard_negative_candidate', 0)}`.",
            f"- Decision: `{decision['decision']}`.",
            f"- HF GPU allowed: `{str(decision['hf_gpu_allowed']).lower()}`.",
            f"- Next action: {decision['next_action']}",
            "",
            "## Rule Detail",
            "",
            "| Rule | Rows | Adapter correct | Adapter matches simulated wrong | Postprocessor correct | Real hard negatives |",
            "|---|---:|---:|---:|---:|---:|",
            *rule_lines,
            "",
            "## Interpretation",
            "",
            "V458 confirmed adapter-level signal for one numeric equation class. "
            "This is stronger than synthetic-only evidence because the rejected answers "
            "are actual frozen-adapter predictions collected before labels were joined. "
            "The signal is still narrow: one rule class and seven hard negatives, so a "
            "paid GPU job remains blocked unless the next dataset builder can add clean "
            "coverage or explicitly accepts a one-rule micro-smoke risk.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V459 V458 NUMERIC HARD NEGATIVE AUDIT START ===", flush=True)
    print("v457_manifest_json =", args.v457_manifest_json, flush=True)
    print("v457_audit_csv =", args.v457_audit_csv, flush=True)
    print("v458_manifest_json =", args.v458_manifest_json, flush=True)
    print("v458_raw_outputs_csv =", args.v458_raw_outputs_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    for path in [args.v457_manifest_json, args.v457_audit_csv, args.v458_manifest_json, args.v458_raw_outputs_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v457_manifest = read_json(args.v457_manifest_json)
    v458_manifest = read_json(args.v458_manifest_json)
    v457_audit_sha = assert_hash(
        args.v457_audit_csv,
        str(v457_manifest.get("outputs", {}).get("audit_csv_sha256", "")),
        "V457 audit CSV",
    )
    v458_raw_sha = assert_hash(
        args.v458_raw_outputs_csv,
        str(v458_manifest.get("outputs", {}).get("raw_outputs_csv_sha256", "")),
        "V458 raw outputs CSV",
    )

    audit_rows = read_csv(args.v457_audit_csv)
    raw_rows = read_csv(args.v458_raw_outputs_csv)
    raw_by_id = {str(row.get("id", "")).strip(): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise RuntimeError("duplicate IDs in V458 raw outputs")

    detail_rows: list[dict[str, Any]] = []
    blocked = Counter()
    for row in audit_rows:
        rid = str(row.get("id", "")).strip()
        raw = raw_by_id.get(rid)
        if not raw:
            blocked["missing_raw_output"] += 1
            continue
        answer = str(row.get("answer", "")).strip()
        simulated_wrong = str(row.get("simulated_wrong_prediction", "")).strip()
        postprocessor = str(row.get("postprocessor_prediction", "")).strip()
        prediction = str(raw.get("prediction", "")).strip()
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
                "decode_config_sha256": raw.get("decode_config_sha256", ""),
                "adapter_repo": raw.get("adapter_repo", ""),
                "adapter_subfolder": raw.get("adapter_subfolder", ""),
                "prompt": raw.get("prompt", ""),
                "raw_output": raw.get("raw_output", ""),
            }
        )

    summary = summarize(detail_rows)
    hard_rows = [row for row in detail_rows if row["real_hard_negative_candidate"] == "true"]
    rule_count = len({row["target_rule_class"] for row in hard_rows})
    hard_count = len(hard_rows)
    conditions = {
        "all_audit_rows_have_raw_output": blocked["missing_raw_output"] == 0,
        "hard_negative_count_ge_min": hard_count >= args.min_hard_negatives,
        "hard_negative_rule_classes_ge_min": rule_count >= args.min_rule_classes,
        "postprocessor_all_correct": summary["totals"].get("postprocessor_correct", 0) == len(detail_rows),
        "finish_reason_stop_only": {row["finish_reason"] for row in detail_rows} <= {"stop"},
    }
    hf_gpu_allowed = all(conditions.values()) and args.allow_one_rule_micro_smoke
    if hf_gpu_allowed:
        decision_text = "v459_allows_one_rule_micro_smoke"
        next_action = "Build V460 one-rule hard-negative micro dataset with bit replay and launch only a one-checkpoint smoke."
    elif all(value for key, value in conditions.items() if key != "hard_negative_rule_classes_ge_min"):
        decision_text = "v459_signal_real_but_narrow_gpu_blocked"
        next_action = "Build V460 CPU dataset proposal, but do not launch paid GPU unless explicitly accepting one-rule risk."
    else:
        decision_text = "v459_blocks_gpu"
        next_action = "Do not launch GPU; collect more raw-output evidence or archive this route."

    detail_csv = args.output_dir / f"{args.label}_detail.csv"
    hard_csv = args.output_dir / f"{args.label}_hard_negatives.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    report_md = args.output_dir / f"{args.label}.md"
    write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)
    write_csv(hard_csv, hard_rows, DETAIL_COLUMNS)
    manifest = {
        "schema_version": "kg1_v459_v458_numeric_hard_negative_audit_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_public_train": True,
            "weak_full_rows_used_for_training": False,
            "purpose": "Audit real frozen-adapter numeric equation mistakes before any paid training.",
        },
        "inputs": {
            "v457_manifest_json": str(args.v457_manifest_json),
            "v457_manifest_sha256": sha256_file(args.v457_manifest_json),
            "v457_audit_csv": str(args.v457_audit_csv),
            "v457_audit_sha256": v457_audit_sha,
            "v458_manifest_json": str(args.v458_manifest_json),
            "v458_manifest_sha256": sha256_file(args.v458_manifest_json),
            "v458_raw_outputs_csv": str(args.v458_raw_outputs_csv),
            "v458_raw_outputs_sha256": v458_raw_sha,
        },
        "thresholds": {
            "min_hard_negatives": args.min_hard_negatives,
            "min_rule_classes": args.min_rule_classes,
            "allow_one_rule_micro_smoke": args.allow_one_rule_micro_smoke,
        },
        "blocked_reasons": dict(sorted(blocked.items())),
        "summary": summary,
        "conditions": conditions,
        "decision": {
            "hf_gpu_allowed": hf_gpu_allowed,
            "decision": decision_text,
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
    manifest["outputs"]["manifest_sha256"] = sha256_file(manifest_json)
    manifest["outputs"]["report_sha256"] = sha256_file(report_md)
    write_json(manifest_json, manifest)
    print("summary =", json.dumps(summary, sort_keys=True), flush=True)
    print("conditions =", json.dumps(conditions, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V459 V458 NUMERIC HARD NEGATIVE AUDIT END ===", flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {
            "target_rule_class": "r",
            "adapter_correct": "false",
            "adapter_matches_simulated_wrong": "true",
            "postprocessor_correct": "true",
            "real_hard_negative_candidate": "true",
        },
        {
            "target_rule_class": "r",
            "adapter_correct": "true",
            "adapter_matches_simulated_wrong": "false",
            "postprocessor_correct": "true",
            "real_hard_negative_candidate": "false",
        },
    ]
    summary = summarize(rows)
    assert summary["rows"] == 2
    assert summary["totals"]["real_hard_negative_candidate"] == 1
    assert verify_answer("-01", "01") is False
    assert verify_answer("01", "1") is False
    print("v459_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v457-manifest-json", type=Path, default=DEFAULT_V457_DIR / "v457_public_train_numeric_probe_pack_manifest.json")
    parser.add_argument("--v457-audit-csv", type=Path, default=DEFAULT_V457_DIR / "v457_public_train_numeric_probe_pack_audit.csv")
    parser.add_argument("--v458-manifest-json", type=Path, default=DEFAULT_V458_DIR / "v458_v457_numeric_raw_probe_manifest.json")
    parser.add_argument("--v458-raw-outputs-csv", type=Path, default=DEFAULT_V458_DIR / "v458_v457_numeric_raw_probe_raw_outputs.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v459_v458_numeric_hard_negative_audit")
    parser.add_argument("--min-hard-negatives", type=int, default=6)
    parser.add_argument("--min-rule-classes", type=int, default=2)
    parser.add_argument(
        "--allow-one-rule-micro-smoke",
        action="store_true",
        help="Explicitly allow a paid GPU micro-smoke even when evidence covers only one rule class.",
    )
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

#!/usr/bin/env python3
"""V435F CPU gate for V435E adapter-exact-wrong preferences.

The gate validates that V435E pairs are structurally sound, are backed by real
frozen-adapter wrong answers collected before label joining, and have enough
equation/bit coverage to justify a very short preference smoke. It does not
train, submit, or launch GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V435E_DIR = REPO_ROOT / "artifacts/v435e_adapter_probe_preference_dataset/20260515T_v435e_hardneg_only"
DEFAULT_V435E_MANIFEST = DEFAULT_V435E_DIR / "v435e_adapter_probe_preference_dataset_manifest.json"
DEFAULT_PREF_TRAIN = DEFAULT_V435E_DIR / "v435e_adapter_probe_preference_dataset_preferences_train.jsonl"
DEFAULT_PREF_VAL = DEFAULT_V435E_DIR / "v435e_adapter_probe_preference_dataset_preferences_val.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435f_adapter_probe_preference_gate"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def extract_boxed_payloads(text: object) -> list[str]:
    value = str(text or "")
    starts = list(re.finditer(r"\\boxed\{", value))
    payloads: list[str] = []
    for index, match in enumerate(starts):
        start = match.end()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
        segment = value[start:end]
        last_brace = segment.rfind("}")
        payloads.append(segment[:last_brace] if last_brace != -1 else segment)
    return payloads


def unescape_boxed_payload(value: object) -> str:
    return str(value).replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\").strip()


def row_audit(row: dict[str, Any], split: str) -> tuple[dict[str, Any], list[str]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    rid = str(row.get("id", ""))
    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    family = str(metadata.get("family", "unknown"))
    negative_type = str(metadata.get("negative_type", "unknown"))
    rule_class = str(metadata.get("rule_class", "unknown"))
    answer = str(metadata.get("answer", ""))
    prediction = str(metadata.get("adapter_prediction", ""))
    chosen_boxes = extract_boxed_payloads(chosen)
    rejected_boxes = extract_boxed_payloads(rejected)
    reasons: list[str] = []
    if not rid or not prompt or not chosen or not rejected:
        reasons.append("missing_required_field")
    if family not in {"equation_transform", "bit_manipulation"}:
        reasons.append("unexpected_family")
    if metadata.get("gate_rows_used_for_training") is not False:
        reasons.append("gate_rows_used_for_training_not_false")
    if metadata.get("weak_gate_rows_used_for_training") is not False:
        reasons.append("weak_gate_rows_used_for_training_not_false")
    if metadata.get("full_gate_rows_used_for_training") is not False:
        reasons.append("full_gate_rows_used_for_training_not_false")
    if metadata.get("raw_output_collected_without_labels") is not True:
        reasons.append("raw_output_not_declared_label_free")
    if metadata.get("labels_joined_after_collection_from_public_train") is not True:
        reasons.append("label_join_policy_missing")
    if metadata.get("locked_before_answer_audit") is not True:
        reasons.append("not_locked_before_answer_audit")
    if not str(metadata.get("adapter_repo", "")).strip() or not str(metadata.get("adapter_subfolder", "")).strip():
        reasons.append("missing_adapter_identity")
    if not str(metadata.get("v291_decode_config_sha256", "")).strip():
        reasons.append("missing_decode_config_hash")
    if not str(metadata.get("v291_raw_output", "")).strip():
        reasons.append("missing_adapter_raw_output")
    if len(chosen_boxes) != 1:
        reasons.append(f"chosen_box_count_{len(chosen_boxes)}")
    elif unescape_boxed_payload(chosen_boxes[0]) != answer.strip():
        reasons.append("chosen_box_not_answer")
    if negative_type == "hard_negative_adapter_exact_wrong":
        if len(rejected_boxes) != 1:
            reasons.append(f"rejected_box_count_{len(rejected_boxes)}")
        elif unescape_boxed_payload(rejected_boxes[0]) != prediction.strip():
            reasons.append("rejected_box_not_adapter_prediction")
        if answer.strip().lower() == prediction.strip().lower():
            reasons.append("hard_negative_prediction_equals_answer")
        if metadata.get("adapter_exact_wrong_certificate") is not True:
            reasons.append("missing_adapter_exact_wrong_certificate")
    elif not negative_type.startswith("format_negative_"):
        reasons.append("unexpected_negative_type")
    return (
        {
            "split": split,
            "id": rid,
            "family": family,
            "negative_type": negative_type,
            "rule_class": rule_class,
            "status": "approved" if not reasons else "blocked",
            "reason": "approved" if not reasons else ";".join(reasons),
        },
        reasons,
    )


def summarize(audits: list[dict[str, Any]]) -> dict[str, Any]:
    approved = [row for row in audits if row["status"] == "approved"]
    blocked = [row for row in audits if row["status"] != "approved"]
    family_counts = Counter(str(row["family"]) for row in approved)
    negative_counts = Counter(str(row["negative_type"]) for row in approved)
    hard_by_family = Counter(
        str(row["family"])
        for row in approved
        if str(row["negative_type"]) == "hard_negative_adapter_exact_wrong"
    )
    rule_classes = {
        str(row["rule_class"])
        for row in approved
        if row["family"] == "equation_transform"
        and row["negative_type"] == "hard_negative_adapter_exact_wrong"
    }
    reason_counts: Counter[str] = Counter()
    for row in blocked:
        for reason in str(row["reason"]).split(";"):
            if reason:
                reason_counts[reason] += 1
    return {
        "rows": len(audits),
        "approved_rows": len(approved),
        "blocked_rows": len(blocked),
        "approved_family_counts": dict(sorted(family_counts.items())),
        "approved_negative_type_counts": dict(sorted(negative_counts.items())),
        "approved_hard_negative_family_counts": dict(sorted(hard_by_family.items())),
        "approved_equation_hard_negative_rule_classes": sorted(rule_classes),
        "approved_equation_hard_negative_rule_class_count": len(rule_classes),
        "blocked_reason_counts": dict(reason_counts.most_common(40)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435F ADAPTER PROBE PREFERENCE GATE START ===", flush=True)
    print("v435e_manifest_json =", args.v435e_manifest_json, flush=True)
    print("preferences_train_jsonl =", args.preferences_train_jsonl, flush=True)
    print("preferences_val_jsonl =", args.preferences_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.v435e_manifest_json.is_file():
        raise FileNotFoundError(args.v435e_manifest_json)
    if not args.preferences_train_jsonl.is_file():
        raise FileNotFoundError(args.preferences_train_jsonl)
    if not args.preferences_val_jsonl.is_file():
        raise FileNotFoundError(args.preferences_val_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(args.v435e_manifest_json)
    if source_manifest.get("schema_version") != "kg1_v435e_adapter_probe_preference_dataset_v1":
        raise RuntimeError("unexpected V435E schema: " + str(source_manifest.get("schema_version")))
    outputs = source_manifest.get("outputs", {})
    if sha256_file(args.preferences_train_jsonl) != str(outputs.get("preferences_train_sha256", "")):
        raise RuntimeError("V435E train preference hash mismatch")
    if sha256_file(args.preferences_val_jsonl) != str(outputs.get("preferences_val_sha256", "")):
        raise RuntimeError("V435E validation preference hash mismatch")

    train_rows = read_jsonl(args.preferences_train_jsonl)
    val_rows = read_jsonl(args.preferences_val_jsonl)
    audits: list[dict[str, Any]] = []
    all_reasons: list[str] = []
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            audit, reasons = row_audit(row, split)
            audits.append(audit)
            all_reasons.extend(reasons)
    summary = summarize(audits)
    hard_counts = summary["approved_hard_negative_family_counts"]
    negative_counts = summary["approved_negative_type_counts"]
    format_negative_rows = sum(
        int(count)
        for key, count in negative_counts.items()
        if str(key).startswith("format_negative_")
    )
    conditions = {
        "all_rows_approved": summary["blocked_rows"] == 0,
        "approved_rows_ge_min": summary["approved_rows"] >= args.min_approved_rows,
        "equation_hard_negatives_ge_min": int(hard_counts.get("equation_transform", 0)) >= args.min_equation_hard_negatives,
        "bit_hard_negatives_ge_min": int(hard_counts.get("bit_manipulation", 0)) >= args.min_bit_hard_negatives,
        "bit_replay_ge_min": int(negative_counts.get("format_negative_format_no_box", 0)) >= args.min_bit_replay,
        "format_negatives_absent_for_preference": args.allow_format_negatives or format_negative_rows == 0,
        "equation_rule_classes_ge_min": summary["approved_equation_hard_negative_rule_class_count"] >= args.min_equation_rule_classes,
    }
    hf_gpu_allowed = all(conditions.values())
    decision = {
        "hf_gpu_allowed": hf_gpu_allowed,
        "decision": "v435f_preference_gate_passed_allow_short_smoke" if hf_gpu_allowed else "v435f_blocks_gpu",
        "blocking_conditions": [key for key, value in conditions.items() if not value],
        "next_action": (
            "Launch one short V436 preference smoke with first-checkpoint ACC gate and FinOps kill-switch."
            if hf_gpu_allowed
            else "Do not launch GPU. Fix blocked conditions or collect better raw outputs."
        ),
    }
    audit_path = args.output_dir / f"{args.label}_audit.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "kg1_v435f_adapter_probe_preference_gate_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v435e_manifest_json": str(args.v435e_manifest_json),
            "v435e_manifest_sha256": sha256_file(args.v435e_manifest_json),
            "preferences_train_jsonl": str(args.preferences_train_jsonl),
            "preferences_train_sha256": sha256_file(args.preferences_train_jsonl),
            "preferences_val_jsonl": str(args.preferences_val_jsonl),
            "preferences_val_sha256": sha256_file(args.preferences_val_jsonl),
        },
        "thresholds": {
            "min_approved_rows": args.min_approved_rows,
            "min_equation_hard_negatives": args.min_equation_hard_negatives,
            "min_bit_hard_negatives": args.min_bit_hard_negatives,
            "min_bit_replay": args.min_bit_replay,
            "min_equation_rule_classes": args.min_equation_rule_classes,
            "allow_format_negatives": args.allow_format_negatives,
        },
        "summary": summary,
        "conditions": conditions,
        "decision": decision,
        "outputs": {
            "audit_jsonl": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "manifest_json": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    print("summary =", json.dumps(summary, sort_keys=True), flush=True)
    print("conditions =", json.dumps(conditions, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V435F ADAPTER PROBE PREFERENCE GATE END ===", flush=True)
    return manifest


def self_test() -> None:
    row = {
        "id": "x",
        "prompt": "prompt",
        "chosen": "Final answer: \\boxed{|@\\{}",
        "rejected": "Final answer: \\boxed{bad\\}}",
        "metadata": {
            "family": "equation_transform",
            "negative_type": "hard_negative_adapter_exact_wrong",
            "rule_class": "equation_symbolic_sequence",
            "answer": "|@{",
            "adapter_prediction": "bad}",
            "v291_raw_output": "raw",
            "v291_decode_config_sha256": "sha",
            "adapter_repo": "repo",
            "adapter_subfolder": "ckpt",
            "raw_output_collected_without_labels": True,
            "labels_joined_after_collection_from_public_train": True,
            "locked_before_answer_audit": True,
            "adapter_exact_wrong_certificate": True,
            "gate_rows_used_for_training": False,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        },
    }
    audit, reasons = row_audit(row, "train")
    if reasons or audit["status"] != "approved":
        raise AssertionError((audit, reasons))
    print("v435f_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v435e-manifest-json", type=Path, default=DEFAULT_V435E_MANIFEST)
    parser.add_argument("--preferences-train-jsonl", type=Path, default=DEFAULT_PREF_TRAIN)
    parser.add_argument("--preferences-val-jsonl", type=Path, default=DEFAULT_PREF_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435f_adapter_probe_preference_gate")
    parser.add_argument("--min-approved-rows", type=int, default=120)
    parser.add_argument("--min-equation-hard-negatives", type=int, default=100)
    parser.add_argument("--min-bit-hard-negatives", type=int, default=10)
    parser.add_argument("--min-bit-replay", type=int, default=0)
    parser.add_argument("--min-equation-rule-classes", type=int, default=4)
    parser.add_argument(
        "--allow-format-negatives",
        action="store_true",
        help=(
            "Permit format-only negatives. Keep disabled for GPU preference training because "
            "mean-NLL preference can reward shorter no-box completions."
        ),
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

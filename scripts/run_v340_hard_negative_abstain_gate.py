#!/usr/bin/env python3
"""V340 CPU gate for hard-negative/abstain transfer.

This gate exists because V338B proved that lower eval_loss can fail to improve
family ACC. It validates whether the existing V337D hard-negative assets are
strong enough to justify another HF GPU job, and blocks GPU unless the next job
would actually use the preference/abstain signal instead of repeating SFT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V336A_MANIFEST = (
    REPO_ROOT
    / "artifacts/v336_integrated_no_loss_solver_gate/20260513T_cpu_gate/"
    / "v336a_integrated_no_loss_solver_gate_manifest.json"
)
DEFAULT_V337D_MANIFEST = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_manifest.json"
)
DEFAULT_TOKENIZATION_MANIFEST = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
)
DEFAULT_V336A_TRACE = (
    REPO_ROOT
    / "artifacts/v336_integrated_no_loss_solver_gate/20260513T_cpu_gate/"
    / "v336a_integrated_no_loss_candidate_trace.csv"
)
DEFAULT_TRAIN_JSONL = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_train.jsonl"
)
DEFAULT_VAL_JSONL = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_val.jsonl"
)
DEFAULT_PREF_TRAIN_JSONL = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_preferences_train.jsonl"
)
DEFAULT_PREF_VAL_JSONL = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_preferences_val.jsonl"
)

EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
}
DEFAULT_EXPECTED_V336A = {
    "correct": 197,
    "equation_transform_correct": 61,
    "bit_manipulation_correct": 136,
    "loss_count": 0,
}
V338B_OBSERVED_WEAK_EVAL = [
    {
        "checkpoint": "checkpoint-2",
        "correct": 190,
        "equation_transform_correct": 56,
        "bit_manipulation_correct": 134,
        "truncated": 1,
    },
    {
        "checkpoint": "checkpoint-4",
        "correct": 190,
        "equation_transform_correct": 56,
        "bit_manipulation_correct": 134,
        "truncated": 0,
    },
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def boxed_values(text: str) -> list[str]:
    return re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def prompt_hash_from_row(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        return sha256_text(normalize_prompt(str(messages[1].get("content", ""))))
    return sha256_text(normalize_prompt(str(row.get("prompt", ""))))


def normalize_rule_class(rule: str) -> str:
    value = str(rule or "").strip()
    for prefix in (
        "v274_guarded_numeric_",
        "v299_same_operator_unique_numeric_",
        "v329_",
    ):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    value = value.replace("symbolic_cryptarithm_", "cryptarithm_")
    return value


def assert_v336a(path: Path, expected: dict[str, int]) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v336a_integrated_no_loss_solver_gate_v1":
        raise RuntimeError("unexpected V336A schema: " + str(payload.get("schema_version")))
    if payload.get("observed_shared_row_contract_sha256") != EXPECTED_SHARED_ROW_CONTRACT_SHA256:
        raise RuntimeError("V336A shared row contract drift")
    integrated = payload.get("integrated", {})
    family = integrated.get("family_counts", {})
    observed = {
        "correct": int(integrated.get("correct", -1)),
        "equation_transform_correct": int(family.get("equation_transform", {}).get("correct", -1)),
        "bit_manipulation_correct": int(family.get("bit_manipulation", {}).get("correct", -1)),
        "loss_count": int(integrated.get("loss_count", -1)),
    }
    if observed != expected:
        raise RuntimeError(f"V336A expected {expected}, got {observed}")
    return payload


def assert_v337d(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("unexpected V337D schema: " + str(payload.get("schema_version")))
    if payload.get("source_policy", {}).get("weak_or_full_gate_rows_used_for_training") is not False:
        raise RuntimeError("V337D source policy drift: weak/full gate rows may be used for training")
    validation = payload.get("validation", {})
    train = validation.get("train", {})
    val = validation.get("validation", {})
    if int(train.get("reference_id_overlap", -1)) != 0 or int(train.get("reference_prompt_overlap", -1)) != 0:
        raise RuntimeError("V337D train anti-leakage drift")
    if int(val.get("reference_id_overlap", -1)) != 0 or int(val.get("reference_prompt_overlap", -1)) != 0:
        raise RuntimeError("V337D validation anti-leakage drift")
    return payload


def assert_tokenization(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v286_generic_tokenization_gate_v1":
        raise RuntimeError("unexpected tokenization gate schema: " + str(payload.get("schema_version")))
    decision = payload.get("decision", {})
    if decision.get("status") != "tokenization_gate_passed":
        raise RuntimeError("tokenization gate did not pass")
    tokenization = payload.get("tokenization", {})
    for split in ("train", "validation"):
        split_payload = tokenization.get(split, {})
        if int(split_payload.get("completion_tokens_dropped", -1)) != 0:
            raise RuntimeError(f"completion tokens dropped in tokenization {split}")
        if float(split_payload.get("prompt_truncation_rate", 1.0)) != 0.0:
            raise RuntimeError(f"prompt truncation in tokenization {split}")
        if int(split_payload.get("fallback_masks", -1)) != 0:
            raise RuntimeError(f"fallback masks in tokenization {split}")
    return payload


def assert_manifest_hashes(manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    outputs = manifest.get("outputs", {})
    key_by_name = {
        "train": "train_sha256",
        "validation": "val_sha256",
        "preferences_train": "preferences_train_sha256",
        "preferences_validation": "preferences_val_sha256",
    }
    for name, path in paths.items():
        expected = str(outputs.get(key_by_name[name], "")).strip()
        observed = sha256_file(path)
        if not expected:
            raise RuntimeError(f"V337D manifest missing hash for {name}")
        if observed != expected:
            raise RuntimeError(f"V337D {name} hash mismatch: expected {expected}, got {observed}")


def audit_candidate_trace(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    bad: list[str] = []
    accepted_rows = [row for row in rows if str(row.get("accepted", "")).lower() == "true"]
    gains = [row for row in accepted_rows if row.get("old_correct") == "False" and row.get("new_correct") == "True"]
    losses = [row for row in accepted_rows if row.get("old_correct") == "True" and row.get("new_correct") == "False"]
    for row in accepted_rows:
        if row.get("family") != "equation_transform":
            bad.append(f"{row.get('id')}:accepted_non_equation")
        if row.get("new_prediction") != row.get("answer"):
            bad.append(f"{row.get('id')}:accepted_new_prediction_not_answer")
        if int(row.get("conflict_count", "999")) != 0:
            bad.append(f"{row.get('id')}:accepted_conflict_count_nonzero")
        if int(row.get("candidate_count", "0")) < 1:
            bad.append(f"{row.get('id')}:accepted_candidate_count_lt_1")
    if bad:
        raise RuntimeError("candidate trace audit failed: " + json.dumps(bad[:20], sort_keys=True))
    return {
        "rows": len(rows),
        "accepted_rows": len(accepted_rows),
        "gain_rows": len(gains),
        "loss_rows": len(losses),
        "accepted_ids": [row.get("id") for row in accepted_rows],
        "accepted_rule_classes": dict(Counter(row.get("rule_class", "") for row in accepted_rows)),
        "accepted_subtypes": dict(Counter(row.get("subtype", "") for row in accepted_rows)),
        "sha256": sha256_file(path),
    }


def audit_sft_rows(path: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids: set[str] = set()
    prompts: set[str] = set()
    bad: list[str] = []
    family_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows, 1):
        rid = str(row.get("id", ""))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        messages = row.get("messages")
        if not rid:
            bad.append(f"{split}:{idx}:missing_id")
        if rid in ids:
            bad.append(f"{split}:{rid}:duplicate_id")
        ids.add(rid)
        psha = prompt_hash_from_row(row)
        if psha in prompts:
            bad.append(f"{split}:{rid}:duplicate_prompt")
        prompts.add(psha)
        family = str(row.get("family") or metadata.get("family") or "")
        family_counts[family] += 1
        component_counts[str(metadata.get("v337d_component", ""))] += 1
        source_counts[str(metadata.get("source_dataset", row.get("source", "")))] += 1
        rule_counts[str(metadata.get("rule_class", ""))] += 1
        subcategory_counts[str(row.get("subcategory", metadata.get("subcategory", "")))] += 1
        for flag in ("weak_gate_rows_used_for_training", "full_gate_rows_used_for_training", "gate_rows_used_for_training"):
            if metadata.get(flag) is not False:
                bad.append(f"{split}:{rid}:{flag}_not_false")
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{split}:{rid}:bad_messages")
            continue
        assistant = str(messages[2].get("content", ""))
        boxes = boxed_values(assistant)
        if len(boxes) != 1:
            bad.append(f"{split}:{rid}:assistant_box_count_{len(boxes)}")
        elif boxes[0] != str(row.get("answer", "")):
            bad.append(f"{split}:{rid}:assistant_box_answer_mismatch")
    if bad:
        raise RuntimeError("SFT audit failed: " + json.dumps(bad[:30], ensure_ascii=False))
    return {
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_prompt_hashes": len(prompts),
        "family_counts": dict(sorted(family_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "source_counts": dict(source_counts.most_common(40)),
        "rule_class_counts": dict(rule_counts.most_common(40)),
        "subcategory_counts": dict(subcategory_counts.most_common(40)),
        "sha256": sha256_file(path),
    }


def audit_preference_rows(path: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids: set[str] = set()
    bad: list[str] = []
    negative_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_row_ids: set[str] = set()
    chosen_box_counts: Counter[int] = Counter()
    rejected_box_counts: Counter[int] = Counter()
    hard_negative_wrong_box_count = 0
    hard_negative_same_box_count = 0
    format_negative_count = 0
    for idx, row in enumerate(rows, 1):
        rid = str(row.get("id", ""))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not rid:
            bad.append(f"{split}:{idx}:missing_id")
        if rid in ids:
            bad.append(f"{split}:{rid}:duplicate_id")
        ids.add(rid)
        family = str(metadata.get("family", ""))
        family_counts[family] += 1
        negative_type = str(metadata.get("negative_type", "unknown"))
        negative_counts[negative_type] += 1
        rule_counts[str(metadata.get("rule_class", ""))] += 1
        source_counts[str(metadata.get("source_dataset", metadata.get("source", "")))] += 1
        source_row_id = str(metadata.get("preference_source_row_id", ""))
        if source_row_id:
            source_row_ids.add(source_row_id)
        for flag in ("weak_gate_rows_used_for_training", "full_gate_rows_used_for_training", "gate_rows_used_for_training"):
            if metadata.get(flag) is not False:
                bad.append(f"{split}:{rid}:{flag}_not_false")
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if chosen == rejected:
            bad.append(f"{split}:{rid}:chosen_equals_rejected")
        chosen_boxes = boxed_values(chosen)
        rejected_boxes = boxed_values(rejected)
        chosen_box_counts[len(chosen_boxes)] += 1
        rejected_box_counts[len(rejected_boxes)] += 1
        if len(chosen_boxes) != 1:
            bad.append(f"{split}:{rid}:chosen_box_count_{len(chosen_boxes)}")
        if negative_type == "hard_negative_equation_near_miss":
            if len(rejected_boxes) != 1:
                bad.append(f"{split}:{rid}:hard_negative_rejected_box_count_{len(rejected_boxes)}")
            elif rejected_boxes[0] == chosen_boxes[0]:
                bad.append(f"{split}:{rid}:hard_negative_same_box")
                hard_negative_same_box_count += 1
            else:
                hard_negative_wrong_box_count += 1
        elif negative_type.startswith("format_negative_"):
            format_negative_count += 1
        else:
            bad.append(f"{split}:{rid}:unexpected_negative_type_{negative_type}")
    return {
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_source_rows": len(source_row_ids),
        "family_counts": dict(sorted(family_counts.items())),
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "rule_class_counts": dict(rule_counts.most_common(40)),
        "source_counts": dict(source_counts.most_common(40)),
        "chosen_box_count_distribution": dict(sorted(chosen_box_counts.items())),
        "rejected_box_count_distribution": dict(sorted(rejected_box_counts.items())),
        "hard_negative_wrong_box_count": hard_negative_wrong_box_count,
        "hard_negative_same_box_count": hard_negative_same_box_count,
        "format_negative_count": format_negative_count,
        "issue_count": len(bad),
        "issue_examples": bad[:30],
        "sha256": sha256_file(path),
    }


def compare_rule_coverage(trace_summary: dict[str, Any], sft_train: dict[str, Any], pref_train: dict[str, Any]) -> dict[str, Any]:
    accepted_rules = set(trace_summary["accepted_rule_classes"])
    normalized_accepted = {normalize_rule_class(rule) for rule in accepted_rules}
    sft_rules = set(sft_train["rule_class_counts"])
    pref_rules = set(pref_train["rule_class_counts"])
    normalized_sft = {normalize_rule_class(rule) for rule in sft_rules if rule}
    normalized_pref = {normalize_rule_class(rule) for rule in pref_rules if rule}
    return {
        "accepted_rules": sorted(accepted_rules),
        "normalized_accepted_rules": sorted(normalized_accepted),
        "sft_rules": sorted(sft_rules),
        "preference_rules": sorted(pref_rules),
        "accepted_rules_covered_by_sft": sorted(normalized_accepted & normalized_sft),
        "accepted_rules_missing_from_sft": sorted(normalized_accepted - normalized_sft),
        "accepted_rules_covered_by_preferences": sorted(normalized_accepted & normalized_pref),
        "accepted_rules_missing_from_preferences": sorted(normalized_accepted - normalized_pref),
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    trace = payload["candidate_trace"]
    pref = payload["preference_audit"]["train"]
    lines = [
        "# KG1 V340 - Hard Negative Abstain Gate",
        "",
        f"Generated at UTC: `{payload['generated_at_utc']}`",
        "",
        "## CPU Signal",
        "",
        f"- V336A integrated weak: `{payload['v336a_signal']['correct']}/315`.",
        f"- Equation: `{payload['v336a_signal']['equation_transform_correct']}/155`.",
        f"- Bit: `{payload['v336a_signal']['bit_manipulation_correct']}/160`.",
        f"- Accepted no-loss candidates: `{trace['accepted_rows']}`.",
        "",
        "## V337D Preference Assets",
        "",
        f"- Preference train rows: `{pref['rows']}`.",
        f"- Hard negative wrong-box rows: `{pref['hard_negative_wrong_box_count']}`.",
        f"- Negative types: `{json.dumps(pref['negative_type_counts'], sort_keys=True)}`.",
        "",
        "## V338B Evidence",
        "",
        "- `eval_loss` improved, but checkpoint weak eval regressed to `190/315`, equation `56`, bit `134`.",
        "",
        "## Decision",
        "",
        f"- `{decision['decision']}`",
        f"- HF GPU allowed: `{decision['hf_gpu_allowed']}`.",
        f"- Reason: {decision['reason']}",
        f"- Next action: {decision['next_action']}",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V340 HARD NEGATIVE ABSTAIN GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v336a_manifest_json =", args.v336a_manifest_json, flush=True)
    print("v337d_manifest_json =", args.v337d_manifest_json, flush=True)
    print("tokenization_manifest_json =", args.tokenization_manifest_json, flush=True)
    print("candidate_trace_csv =", args.candidate_trace_csv, flush=True)
    print("train_jsonl =", args.train_jsonl, flush=True)
    print("val_jsonl =", args.val_jsonl, flush=True)
    print("preferences_train_jsonl =", args.preferences_train_jsonl, flush=True)
    print("preferences_val_jsonl =", args.preferences_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_v336a = {
        "correct": args.expected_v336a_correct,
        "equation_transform_correct": args.expected_v336a_equation,
        "bit_manipulation_correct": args.expected_v336a_bit,
        "loss_count": args.expected_v336a_loss_count,
    }
    v336a = assert_v336a(args.v336a_manifest_json, expected_v336a)
    v337d = assert_v337d(args.v337d_manifest_json)
    tokenization = assert_tokenization(args.tokenization_manifest_json)
    manifest_hash_paths = {
        "train": args.train_jsonl,
        "validation": args.val_jsonl,
    }
    if not args.allow_derived_preferences:
        manifest_hash_paths.update(
            {
                "preferences_train": args.preferences_train_jsonl,
                "preferences_validation": args.preferences_val_jsonl,
            }
        )
    assert_manifest_hashes(v337d, manifest_hash_paths)

    trace_summary = audit_candidate_trace(args.candidate_trace_csv)
    train_summary = audit_sft_rows(args.train_jsonl, "train")
    val_summary = audit_sft_rows(args.val_jsonl, "validation")
    pref_train_summary = audit_preference_rows(args.preferences_train_jsonl, "preferences_train")
    pref_val_summary = audit_preference_rows(args.preferences_val_jsonl, "preferences_validation")
    coverage = compare_rule_coverage(trace_summary, train_summary, pref_train_summary)

    validation_issues: list[str] = []
    if (
        trace_summary["accepted_rows"] != args.expected_accepted_candidates
        or trace_summary["gain_rows"] != args.expected_accepted_candidates
        or trace_summary["loss_rows"] != 0
    ):
        validation_issues.append("unexpected_v336a_candidate_trace_counts")
    if train_summary["family_counts"].get("bit_manipulation") != args.expected_train_bit_rows:
        validation_issues.append("train_bit_replay_unexpected_count")
    if train_summary["family_counts"].get("equation_transform") != args.expected_train_equation_rows:
        validation_issues.append("train_equation_unexpected_count")
    if val_summary["family_counts"].get("bit_manipulation") != args.expected_val_bit_rows:
        validation_issues.append("validation_bit_replay_unexpected_count")
    if val_summary["family_counts"].get("equation_transform") != args.expected_val_equation_rows:
        validation_issues.append("validation_equation_unexpected_count")
    if pref_train_summary["negative_type_counts"].get("hard_negative_equation_near_miss", 0) < args.min_hard_negative_train:
        validation_issues.append("preference_train_hard_negative_below_min")
    if pref_val_summary["negative_type_counts"].get("hard_negative_equation_near_miss", 0) < args.min_hard_negative_val:
        validation_issues.append("preference_validation_hard_negative_below_min")
    if pref_train_summary["issue_count"]:
        validation_issues.append("preference_train_rows_have_invalid_pairs")
    if pref_val_summary["issue_count"]:
        validation_issues.append("preference_validation_rows_have_invalid_pairs")
    if pref_train_summary["hard_negative_same_box_count"]:
        validation_issues.append("preference_train_hard_negative_same_box")
    if pref_val_summary["hard_negative_same_box_count"]:
        validation_issues.append("preference_validation_hard_negative_same_box")
    if coverage["accepted_rules_missing_from_sft"]:
        validation_issues.append("accepted_rules_missing_from_sft")
    if coverage["accepted_rules_missing_from_preferences"]:
        validation_issues.append("accepted_rules_missing_from_preferences")

    assets_valid = not validation_issues
    preference_launcher_exists = args.preference_launcher.exists() if args.preference_launcher else False
    if not assets_valid:
        decision = {
            "decision": "v340_cpu_gate_failed_block_hf",
            "reason": "Hard-negative/abstain assets failed validation: " + ",".join(validation_issues),
            "next_action": "Fix CPU artifacts before any HF GPU job.",
            "hf_gpu_allowed": False,
            "preference_training_allowed": False,
        }
    elif not preference_launcher_exists:
        decision = {
            "decision": "v340_preference_assets_valid_but_gpu_blocked_until_preference_trainer",
            "reason": (
                "V337D SFT and preference assets are internally valid, but V338B already showed SFT loss "
                "improvement does not transfer to family ACC. No preference/abstain trainer launcher was "
                "provided, so another HF SFT job would repeat the falsified path."
            ),
            "next_action": (
                "Implement a DPO/ORPO/KTO-style or explicit abstain-selector trainer that consumes "
                "the preference rows, then run this gate again with --preference-launcher."
            ),
            "hf_gpu_allowed": False,
            "preference_training_allowed": False,
        }
    else:
        decision = {
            "decision": "v340_preference_assets_valid_preference_trainer_required_smoke_allowed",
            "reason": (
                "CPU assets passed and a preference trainer launcher exists. Only a tiny preference "
                "smoke is allowed; first-checkpoint kill-switch remains total>192, equation>56, bit>=136."
            ),
            "next_action": "Run a tiny preference-training smoke with strict first-checkpoint FinOps kill-switch.",
            "hf_gpu_allowed": True,
            "preference_training_allowed": True,
        }

    integrated = v336a["integrated"]
    family = integrated["family_counts"]
    payload = {
        "schema_version": "kg1_v340_hard_negative_abstain_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v336a_manifest_json": str(args.v336a_manifest_json),
            "v336a_manifest_sha256": sha256_file(args.v336a_manifest_json),
            "v337d_manifest_json": str(args.v337d_manifest_json),
            "v337d_manifest_sha256": sha256_file(args.v337d_manifest_json),
            "tokenization_manifest_json": str(args.tokenization_manifest_json),
            "tokenization_manifest_sha256": sha256_file(args.tokenization_manifest_json),
            "candidate_trace_csv": str(args.candidate_trace_csv),
            "candidate_trace_sha256": sha256_file(args.candidate_trace_csv),
            "train_jsonl": str(args.train_jsonl),
            "train_sha256": sha256_file(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "val_sha256": sha256_file(args.val_jsonl),
            "preferences_train_jsonl": str(args.preferences_train_jsonl),
            "preferences_train_sha256": sha256_file(args.preferences_train_jsonl),
            "preferences_val_jsonl": str(args.preferences_val_jsonl),
            "preferences_val_sha256": sha256_file(args.preferences_val_jsonl),
            "preference_launcher": str(args.preference_launcher) if args.preference_launcher else "",
            "preference_launcher_exists": preference_launcher_exists,
            "allow_derived_preferences": args.allow_derived_preferences,
            "min_hard_negative_train": args.min_hard_negative_train,
            "min_hard_negative_val": args.min_hard_negative_val,
            "expected_v336a": expected_v336a,
            "expected_accepted_candidates": args.expected_accepted_candidates,
            "expected_train_bit_rows": args.expected_train_bit_rows,
            "expected_train_equation_rows": args.expected_train_equation_rows,
            "expected_val_bit_rows": args.expected_val_bit_rows,
            "expected_val_equation_rows": args.expected_val_equation_rows,
        },
        "baseline": EXPECTED_BASELINE,
        "v336a_signal": {
            "correct": int(integrated["correct"]),
            "equation_transform_correct": int(family["equation_transform"]["correct"]),
            "bit_manipulation_correct": int(family["bit_manipulation"]["correct"]),
            "loss_count": int(integrated["loss_count"]),
            "accepted_candidate_count": int(integrated["accepted_candidate_count"]),
        },
        "v338b_observed_weak_eval": V338B_OBSERVED_WEAK_EVAL,
        "tokenization_gate": {
            "status": tokenization.get("decision", {}).get("status"),
            "train_rows": tokenization.get("tokenization", {}).get("train", {}).get("rows"),
            "validation_rows": tokenization.get("tokenization", {}).get("validation", {}).get("rows"),
            "train_prompt_truncation_rate": tokenization.get("tokenization", {}).get("train", {}).get(
                "prompt_truncation_rate"
            ),
            "validation_prompt_truncation_rate": tokenization.get("tokenization", {}).get("validation", {}).get(
                "prompt_truncation_rate"
            ),
        },
        "candidate_trace": trace_summary,
        "sft_audit": {"train": train_summary, "validation": val_summary},
        "preference_audit": {"train": pref_train_summary, "validation": pref_val_summary},
        "rule_coverage": coverage,
        "validation_issues": validation_issues,
        "assets_valid": assets_valid,
        "decision": decision,
        "outputs": {
            "manifest_json": str(args.output_dir / f"{args.label}_manifest.json"),
            "summary_md": str(args.output_dir / f"{args.label}_summary.md"),
        },
    }
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    summary_path = args.output_dir / f"{args.label}_summary.md"
    write_json(manifest_path, payload)
    summary_path.write_text(markdown_summary(payload), encoding="utf-8")

    print("assets_valid =", assets_valid, flush=True)
    print("preference_launcher_exists =", preference_launcher_exists, flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("candidate_trace_summary =", json.dumps(trace_summary, sort_keys=True), flush=True)
    print("preference_train_summary =", json.dumps(pref_train_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("summary_md =", summary_path, flush=True)
    print("=== V340 HARD NEGATIVE ABSTAIN GATE END ===", flush=True)
    return payload


def run_self_test() -> None:
    if boxed_values(r"Final answer: \boxed{42}") != ["42"]:
        raise AssertionError("boxed parser failed")
    if normalize_rule_class("v274_guarded_numeric_minus_signed_opposite_sign_guarded") != (
        "minus_signed_opposite_sign_guarded"
    ):
        raise AssertionError("rule normalizer failed")
    print("v340_hard_negative_abstain_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v336a-manifest-json", type=Path, default=DEFAULT_V336A_MANIFEST)
    parser.add_argument("--v337d-manifest-json", type=Path, default=DEFAULT_V337D_MANIFEST)
    parser.add_argument("--tokenization-manifest-json", type=Path, default=DEFAULT_TOKENIZATION_MANIFEST)
    parser.add_argument("--candidate-trace-csv", type=Path, default=DEFAULT_V336A_TRACE)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--preferences-train-jsonl", type=Path, default=DEFAULT_PREF_TRAIN_JSONL)
    parser.add_argument("--preferences-val-jsonl", type=Path, default=DEFAULT_PREF_VAL_JSONL)
    parser.add_argument("--preference-launcher", type=Path, default=None)
    parser.add_argument(
        "--allow-derived-preferences",
        action="store_true",
        help="Allow preference paths derived from V337D even when their hashes differ from the V337D manifest.",
    )
    parser.add_argument("--min-hard-negative-train", type=int, default=650)
    parser.add_argument("--min-hard-negative-val", type=int, default=170)
    parser.add_argument("--expected-v336a-correct", type=int, default=DEFAULT_EXPECTED_V336A["correct"])
    parser.add_argument(
        "--expected-v336a-equation",
        type=int,
        default=DEFAULT_EXPECTED_V336A["equation_transform_correct"],
    )
    parser.add_argument(
        "--expected-v336a-bit",
        type=int,
        default=DEFAULT_EXPECTED_V336A["bit_manipulation_correct"],
    )
    parser.add_argument("--expected-v336a-loss-count", type=int, default=DEFAULT_EXPECTED_V336A["loss_count"])
    parser.add_argument("--expected-accepted-candidates", type=int, default=5)
    parser.add_argument("--expected-train-bit-rows", type=int, default=720)
    parser.add_argument("--expected-train-equation-rows", type=int, default=720)
    parser.add_argument("--expected-val-bit-rows", type=int, default=160)
    parser.add_argument("--expected-val-equation-rows", type=int, default=180)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/v340_hard_negative_abstain_gate" / utc_compact(),
    )
    parser.add_argument("--label", default="v340_hard_negative_abstain_gate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V435 CPU adapter-level pair gate.

This gate intentionally does not train, submit, or launch GPU work. It audits
candidate chosen/rejected assets and only allows the next HF GPU smoke when the
pairs attack real V291/V290 adapter mistakes in permitted data.
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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V341_MANIFEST = (
    REPO_ROOT
    / "artifacts/v341_clean_preference_transfer_dataset/20260513T_cpu_gate/"
    / "v341_clean_preference_transfer_manifest.json"
)
DEFAULT_V341_PREF_TRAIN = (
    REPO_ROOT
    / "artifacts/v341_clean_preference_transfer_dataset/20260513T_cpu_gate/"
    / "v341_clean_preference_transfer_preferences_train.jsonl"
)
DEFAULT_V341_PREF_VAL = (
    REPO_ROOT
    / "artifacts/v341_clean_preference_transfer_dataset/20260513T_cpu_gate/"
    / "v341_clean_preference_transfer_preferences_val.jsonl"
)
DEFAULT_V337D_MANIFEST = (
    REPO_ROOT
    / "artifacts/v337d_minimal_transfer_dataset/20260513T_cpu_gate/"
    / "v337d_minimal_transfer_manifest.json"
)
DEFAULT_REFERENCE_WEAK_CSV = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_REFERENCE_FULL_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_BASELINE_WEAK_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435_adapter_level_pair_gate"

EXPECTED_BASELINE = {
    "total": 192,
    "equation_transform": 56,
    "bit_manipulation": 136,
    "truncated": 0,
}


PAIR_COLUMNS = [
    "split",
    "id",
    "family",
    "rule_class",
    "negative_type",
    "status",
    "reason",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "chosen_box",
    "rejected_box",
    "reference_id_overlap",
    "reference_prompt_overlap",
    "locked_before_answer_audit",
    "has_adapter_raw_output",
    "has_adapter_decode_config",
    "has_adapter_identity",
    "has_mdl",
    "has_loo",
    "has_renaming",
    "source_dataset",
]

RULE_COLUMNS = [
    "family",
    "rule_class",
    "candidate_pairs",
    "approved_pairs",
    "blocked_pairs",
    "hard_negative_pairs",
    "format_negative_pairs",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def boxed_values(text: Any) -> list[str]:
    return re.findall(r"\\boxed\{([^{}]*)\}", str(text or ""))


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
                raise RuntimeError(f"{path}:{line_no}: row is not a JSON object")
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and str(mapping.get(key, "")).strip():
            return mapping[key]
    return ""


def merged_lookup(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value = first_present(row, keys)
    if value:
        return value
    return first_present(metadata_of(row), keys)


def reference_set(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    rows = read_csv(path)
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            ids.add(rid)
        prompt = str(row.get("prompt") or row.get("generated_prompt") or "")
        if prompt:
            prompt_hashes.add(sha256_text(normalize_text(prompt)))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "ids": ids,
        "prompt_hashes": prompt_hashes,
    }


def assert_manifest_hash(manifest: dict[str, Any], path: Path, keys: tuple[str, ...]) -> None:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    expected = ""
    for key in keys:
        expected = str(outputs.get(key, "")).strip()
        if expected:
            break
    if not expected:
        raise RuntimeError(f"manifest missing hash for {path}: keys={keys}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch for {path}: expected {expected}, got {observed}")


def baseline_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0, "truncated": 0})
    for row in rows:
        family = str(row.get("family") or row.get("type") or row.get("pred_type") or "unknown")
        counts[family]["rows"] += 1
        counts[family]["correct"] += int(truthy(row.get("correct", "")))
        counts[family]["truncated"] += int(truthy(row.get("truncated", "")))
    total_correct = sum(item["correct"] for item in counts.values())
    total_truncated = sum(item["truncated"] for item in counts.values())
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "total_correct": total_correct,
        "total_truncated": total_truncated,
        "family_counts": dict(sorted(counts.items())),
    }
    expected = {
        "total": total_correct,
        "equation_transform": counts.get("equation_transform", {}).get("correct", -1),
        "bit_manipulation": counts.get("bit_manipulation", {}).get("correct", -1),
        "truncated": total_truncated,
    }
    summary["matches_expected_v291_v290"] = expected == EXPECTED_BASELINE
    summary["expected_baseline"] = EXPECTED_BASELINE
    summary["observed_baseline"] = expected
    return summary


def adapter_evidence_flags(row: dict[str, Any]) -> dict[str, bool]:
    raw_output = merged_lookup(row, ("v291_raw_output", "v290_raw_output", "adapter_raw_output", "baseline_raw_output"))
    decode_config = merged_lookup(row, ("decode_config", "adapter_decode_config", "v291_decode_config", "v290_decode_config"))
    adapter_identity = merged_lookup(
        row,
        (
            "adapter_commit",
            "adapter_path",
            "adapter_sha256",
            "adapter_name",
            "v291_adapter",
            "v290_adapter",
        ),
    )
    return {
        "has_adapter_raw_output": bool(str(raw_output).strip()),
        "has_adapter_decode_config": bool(str(decode_config).strip()),
        "has_adapter_identity": bool(str(adapter_identity).strip()),
    }


def certificate_flags(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "has_mdl": bool(str(merged_lookup(row, ("mdl", "mdl_score", "mdl_cost"))).strip()),
        "has_loo": bool(str(merged_lookup(row, ("loo", "leave_one_out", "leave_one_out_pass"))).strip()),
        "has_renaming": bool(str(merged_lookup(row, ("renaming", "renaming_stability", "renaming_pass"))).strip()),
    }


def audit_preference_row(
    row: dict[str, Any],
    *,
    split: str,
    reference_ids: set[str],
    reference_prompt_hashes: set[str],
) -> dict[str, Any]:
    metadata = metadata_of(row)
    rid = str(row.get("id", "")).strip()
    prompt = str(row.get("prompt", ""))
    prompt_hash = sha256_text(str(prompt).replace("\r\n", "\n"))
    prompt_norm_hash = sha256_text(normalize_text(prompt))
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    chosen_boxes = boxed_values(chosen)
    rejected_boxes = boxed_values(rejected)
    family = str(metadata.get("family") or row.get("family") or "")
    rule_class = str(metadata.get("rule_class") or row.get("rule_class") or "")
    negative_type = str(metadata.get("negative_type") or row.get("negative_type") or "")
    source_dataset = str(metadata.get("source_dataset") or metadata.get("source") or row.get("source_dataset") or "")
    locked = truthy(merged_lookup(row, ("locked_before_answer_audit",)))
    adapter_flags = adapter_evidence_flags(row)
    cert_flags = certificate_flags(row)
    id_overlap = rid in reference_ids
    prompt_overlap = prompt_norm_hash in reference_prompt_hashes
    reasons: list[str] = []

    if not rid:
        reasons.append("missing_id")
    if not prompt:
        reasons.append("missing_prompt")
    if not chosen or not rejected:
        reasons.append("missing_chosen_or_rejected")
    if len(chosen_boxes) != 1:
        reasons.append(f"chosen_box_count_{len(chosen_boxes)}")
    if len(rejected_boxes) != 1:
        reasons.append(f"rejected_box_count_{len(rejected_boxes)}")
    if len(chosen_boxes) == 1 and len(rejected_boxes) == 1 and chosen_boxes[0] == rejected_boxes[0]:
        reasons.append("same_chosen_rejected_box")
    if id_overlap:
        reasons.append("reference_id_overlap")
    if prompt_overlap:
        reasons.append("reference_prompt_overlap")
    if family != "equation_transform":
        reasons.append("not_equation_pair")
    if not negative_type.startswith("hard_negative"):
        reasons.append("not_hard_negative")
    if not adapter_flags["has_adapter_raw_output"]:
        reasons.append("missing_adapter_raw_output")
    if not adapter_flags["has_adapter_decode_config"]:
        reasons.append("missing_adapter_decode_config")
    if not adapter_flags["has_adapter_identity"]:
        reasons.append("missing_adapter_identity")
    if not locked:
        reasons.append("missing_locked_before_answer_audit")
    if not cert_flags["has_mdl"]:
        reasons.append("missing_mdl_certificate")
    if not cert_flags["has_loo"]:
        reasons.append("missing_loo_certificate")
    if not cert_flags["has_renaming"]:
        reasons.append("missing_renaming_certificate")

    approved = not reasons
    return {
        "split": split,
        "id": rid,
        "family": family,
        "rule_class": rule_class,
        "negative_type": negative_type,
        "status": "approved" if approved else "blocked",
        "reason": "approved_adapter_level_pair" if approved else ";".join(reasons),
        "prompt_sha256": prompt_hash,
        "prompt_normalized_sha256": prompt_norm_hash,
        "chosen_box": chosen_boxes[0] if len(chosen_boxes) == 1 else "",
        "rejected_box": rejected_boxes[0] if len(rejected_boxes) == 1 else "",
        "reference_id_overlap": id_overlap,
        "reference_prompt_overlap": prompt_overlap,
        "locked_before_answer_audit": locked,
        **adapter_flags,
        **cert_flags,
        "source_dataset": source_dataset,
    }


def summarize_pairs(pair_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status_counts = Counter(str(row["status"]) for row in pair_rows)
    reason_counts: Counter[str] = Counter()
    for row in pair_rows:
        if row["status"] == "approved":
            continue
        for reason in str(row["reason"]).split(";"):
            if reason:
                reason_counts[reason] += 1

    by_rule: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pair_rows:
        key = (str(row["family"]), str(row["rule_class"]))
        item = by_rule.setdefault(
            key,
            {
                "family": key[0],
                "rule_class": key[1],
                "candidate_pairs": 0,
                "approved_pairs": 0,
                "blocked_pairs": 0,
                "hard_negative_pairs": 0,
                "format_negative_pairs": 0,
            },
        )
        item["candidate_pairs"] += 1
        item["approved_pairs"] += int(row["status"] == "approved")
        item["blocked_pairs"] += int(row["status"] != "approved")
        if str(row["negative_type"]).startswith("hard_negative"):
            item["hard_negative_pairs"] += 1
        else:
            item["format_negative_pairs"] += 1

    approved_rule_modes = {
        str(row["rule_class"]) for row in pair_rows if row["status"] == "approved" and row["family"] == "equation_transform"
    }
    summary = {
        "candidate_pairs": len(pair_rows),
        "approved_pairs": int(status_counts.get("approved", 0)),
        "blocked_pairs": int(status_counts.get("blocked", 0)),
        "approved_equation_rule_modes": len(approved_rule_modes),
        "approved_equation_rule_classes": sorted(approved_rule_modes),
        "status_counts": dict(sorted(status_counts.items())),
        "blocked_reason_counts": dict(reason_counts.most_common(40)),
    }
    return summary, sorted(by_rule.values(), key=lambda row: (row["family"], row["rule_class"]))


def bit_guardrail_summary(v337d_manifest: dict[str, Any]) -> dict[str, Any]:
    normalization = v337d_manifest.get("component_normalization", {})
    bit_train = normalization.get("bit_train", {}) if isinstance(normalization, dict) else {}
    bit_validation = normalization.get("bit_validation", {}) if isinstance(normalization, dict) else {}
    preference_summary = v337d_manifest.get("preference_summary", {})
    pref_counts = {}
    if isinstance(preference_summary, dict):
        for split in ("train", "validation"):
            item = preference_summary.get(split, {})
            if isinstance(item, dict):
                pref_counts[split] = item.get("negative_type_counts", {})
    return {
        "bit_replay_train_rows": int(bit_train.get("selected_bit_rows", 0) or 0),
        "bit_replay_validation_rows": int(bit_validation.get("selected_bit_rows", 0) or 0),
        "bit_replay_available": int(bit_train.get("selected_bit_rows", 0) or 0) > 0
        and int(bit_validation.get("selected_bit_rows", 0) or 0) > 0,
        "programmatic_bit_hard_negative_rows": 0,
        "programmatic_bit_guardrail_ready": False,
        "preference_negative_type_counts": pref_counts,
        "decision": "partial_replay_only_missing_programmatic_bit_hard_negatives",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435 ADAPTER LEVEL PAIR GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("preference_manifest_json =", args.preference_manifest_json, flush=True)
    print("preferences_train_jsonl =", args.preferences_train_jsonl, flush=True)
    print("preferences_val_jsonl =", args.preferences_val_jsonl, flush=True)
    print("v337d_manifest_json =", args.v337d_manifest_json, flush=True)
    print("reference_weak_csv =", args.reference_weak_csv, flush=True)
    print("reference_full_csv =", args.reference_full_csv, flush=True)
    print("baseline_weak_csv =", args.baseline_weak_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pref_manifest = read_json(args.preference_manifest_json)
    if pref_manifest.get("schema_version") != "kg1_v341_clean_preference_transfer_dataset_v1":
        raise RuntimeError("unexpected preference manifest schema: " + str(pref_manifest.get("schema_version")))
    assert_manifest_hash(pref_manifest, args.preferences_train_jsonl, ("preferences_train_sha256",))
    assert_manifest_hash(pref_manifest, args.preferences_val_jsonl, ("preferences_val_sha256",))
    v337d_manifest = read_json(args.v337d_manifest_json)
    if v337d_manifest.get("schema_version") != "kg1_v337d_minimal_transfer_dataset_v1":
        raise RuntimeError("unexpected V337D manifest schema: " + str(v337d_manifest.get("schema_version")))

    weak_ref = reference_set(args.reference_weak_csv)
    full_ref = reference_set(args.reference_full_csv)
    reference_ids = set(weak_ref["ids"]) | set(full_ref["ids"])
    reference_prompt_hashes = set(weak_ref["prompt_hashes"]) | set(full_ref["prompt_hashes"])
    base_summary = baseline_summary(args.baseline_weak_csv)
    if not base_summary["matches_expected_v291_v290"]:
        raise RuntimeError("baseline weak CSV does not match expected V291/V290 metrics")

    train_rows = read_jsonl(args.preferences_train_jsonl)
    val_rows = read_jsonl(args.preferences_val_jsonl)
    pair_rows = [
        audit_preference_row(row, split="train", reference_ids=reference_ids, reference_prompt_hashes=reference_prompt_hashes)
        for row in train_rows
    ]
    pair_rows.extend(
        audit_preference_row(row, split="validation", reference_ids=reference_ids, reference_prompt_hashes=reference_prompt_hashes)
        for row in val_rows
    )
    pair_summary, rule_rows = summarize_pairs(pair_rows)
    bit_summary = bit_guardrail_summary(v337d_manifest)

    conditions = {
        "baseline_matches_expected": bool(base_summary["matches_expected_v291_v290"]),
        "approved_equation_rule_modes_ge_4": int(pair_summary["approved_equation_rule_modes"]) >= 4,
        "approved_pairs_gt_0": int(pair_summary["approved_pairs"]) > 0,
        "zero_reference_overlap_in_approved_pairs": not any(
            row["status"] == "approved" and (row["reference_id_overlap"] or row["reference_prompt_overlap"])
            for row in pair_rows
        ),
        "programmatic_bit_guardrail_ready": bool(bit_summary["programmatic_bit_guardrail_ready"]),
    }
    hf_gpu_allowed = all(conditions.values())
    decision = {
        "hf_gpu_allowed": hf_gpu_allowed,
        "decision": "v435_pair_gate_passed_allow_v436_smoke" if hf_gpu_allowed else "v435_pair_gate_blocks_gpu",
        "next_action": (
            "Launch V436 short adapter-only smoke with first-checkpoint kill-switch."
            if hf_gpu_allowed
            else "Do not launch GPU. Generate V291/V290 raw outputs on permitted train/synthetic rows or add certified adapter-level hard negatives."
        ),
        "blocking_conditions": [key for key, value in conditions.items() if not value],
    }

    pair_audit_csv = args.output_dir / f"{args.label}_pair_audit.csv"
    rule_summary_csv = args.output_dir / f"{args.label}_rule_summary.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    markdown_path = args.output_dir / f"{args.label}_decision.md"
    write_csv(pair_audit_csv, pair_rows, PAIR_COLUMNS)
    write_csv(rule_summary_csv, rule_rows, RULE_COLUMNS)

    manifest = {
        "schema_version": "kg1_v435_adapter_level_pair_gate_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "preference_manifest_json": str(args.preference_manifest_json),
            "preference_manifest_sha256": sha256_file(args.preference_manifest_json),
            "preferences_train_jsonl": str(args.preferences_train_jsonl),
            "preferences_train_sha256": sha256_file(args.preferences_train_jsonl),
            "preferences_val_jsonl": str(args.preferences_val_jsonl),
            "preferences_val_sha256": sha256_file(args.preferences_val_jsonl),
            "v337d_manifest_json": str(args.v337d_manifest_json),
            "v337d_manifest_sha256": sha256_file(args.v337d_manifest_json),
            "reference_weak_csv": str(args.reference_weak_csv),
            "reference_weak_sha256": weak_ref["sha256"],
            "reference_full_csv": str(args.reference_full_csv),
            "reference_full_sha256": full_ref["sha256"],
            "baseline_weak_csv": str(args.baseline_weak_csv),
            "baseline_weak_sha256": base_summary["sha256"],
        },
        "baseline": base_summary,
        "reference_summary": {
            "weak_rows": weak_ref["rows"],
            "full_rows": full_ref["rows"],
            "combined_reference_ids": len(reference_ids),
            "combined_reference_prompt_hashes": len(reference_prompt_hashes),
        },
        "pair_summary": pair_summary,
        "bit_guardrail": bit_summary,
        "promotion_conditions": conditions,
        "decision": decision,
        "outputs": {
            "pair_audit_csv": str(pair_audit_csv),
            "pair_audit_sha256": sha256_file(pair_audit_csv),
            "rule_summary_csv": str(rule_summary_csv),
            "rule_summary_sha256": sha256_file(rule_summary_csv),
            "decision_markdown": str(markdown_path),
            "manifest_json": str(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    markdown = [
        "# V435 Adapter-Level Pair Gate Decision",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Decision",
        "",
        f"- `hf_gpu_allowed`: `{str(hf_gpu_allowed).lower()}`",
        f"- decision: `{decision['decision']}`",
        f"- next action: {decision['next_action']}",
        "",
        "## Baseline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total weak | {base_summary['observed_baseline']['total']}/315 |",
        f"| equation_transform | {base_summary['observed_baseline']['equation_transform']}/155 |",
        f"| bit_manipulation | {base_summary['observed_baseline']['bit_manipulation']}/160 |",
        f"| truncated | {base_summary['observed_baseline']['truncated']} |",
        "",
        "## Pair Audit",
        "",
        f"- candidate pairs: `{pair_summary['candidate_pairs']}`",
        f"- approved pairs: `{pair_summary['approved_pairs']}`",
        f"- approved equation rule modes: `{pair_summary['approved_equation_rule_modes']}`",
        f"- top blocking reasons: `{json.dumps(pair_summary['blocked_reason_counts'], sort_keys=True)}`",
        "",
        "## Blocking Conditions",
        "",
    ]
    markdown.extend(f"- `{item}`" for item in decision["blocking_conditions"])
    markdown.append("")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    manifest["outputs"]["decision_markdown_sha256"] = sha256_file(markdown_path)
    write_json(manifest_json, manifest)

    print("pair_summary =", json.dumps(pair_summary, sort_keys=True), flush=True)
    print("bit_guardrail =", json.dumps(bit_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V435 ADAPTER LEVEL PAIR GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v435_selftest_") as temp_name:
        temp = Path(temp_name)
        row = {
            "id": "synthetic_train_1",
            "prompt": "Now determine: 12+05",
            "chosen": r"Final answer: \boxed{17}",
            "rejected": r"Final answer: \boxed{71}",
            "metadata": {
                "family": "equation_transform",
                "rule_class": "add_direct",
                "negative_type": "hard_negative_v291_raw_wrong",
                "source_dataset": "self_test",
                "locked_before_answer_audit": True,
                "v291_raw_output": r"Final answer: \boxed{71}",
                "v291_decode_config": {"temperature": 0.0},
                "adapter_commit": "selftest",
                "mdl_score": 3,
                "leave_one_out_pass": True,
                "renaming_stability": {"passed": True},
            },
        }
        ref_ids: set[str] = set()
        ref_hashes: set[str] = set()
        audited = audit_preference_row(row, split="train", reference_ids=ref_ids, reference_prompt_hashes=ref_hashes)
        if audited["status"] != "approved":
            raise AssertionError(audited)
        bad = dict(row)
        bad["metadata"] = dict(row["metadata"])
        bad["metadata"].pop("v291_raw_output")
        audited_bad = audit_preference_row(bad, split="train", reference_ids=ref_ids, reference_prompt_hashes=ref_hashes)
        if audited_bad["status"] != "blocked" or "missing_adapter_raw_output" not in audited_bad["reason"]:
            raise AssertionError(audited_bad)
        write_csv(temp / "pairs.csv", [audited, audited_bad], PAIR_COLUMNS)
    print("v435_adapter_level_pair_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preference-manifest-json", type=Path, default=DEFAULT_V341_MANIFEST)
    parser.add_argument("--preferences-train-jsonl", type=Path, default=DEFAULT_V341_PREF_TRAIN)
    parser.add_argument("--preferences-val-jsonl", type=Path, default=DEFAULT_V341_PREF_VAL)
    parser.add_argument("--v337d-manifest-json", type=Path, default=DEFAULT_V337D_MANIFEST)
    parser.add_argument("--reference-weak-csv", type=Path, default=DEFAULT_REFERENCE_WEAK_CSV)
    parser.add_argument("--reference-full-csv", type=Path, default=DEFAULT_REFERENCE_FULL_CSV)
    parser.add_argument("--baseline-weak-csv", type=Path, default=DEFAULT_BASELINE_WEAK_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435_adapter_level_pair_gate")
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

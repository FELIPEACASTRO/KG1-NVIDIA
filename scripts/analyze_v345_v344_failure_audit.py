#!/usr/bin/env python3
"""ACC-first failure audit for V344 preference/abstain transfer.

The goal is to explain why V344 did not improve weak family ACC despite using
V343 solver/verifier gains as teacher signal. This script does not use loss as
a promotion signal; it only compares exact-match behavior and dataset coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = ("bit_manipulation", "equation_transform")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_of(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or "").strip()


def keyed(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"{label}: row without id")
        if row_id in out:
            raise ValueError(f"{label}: duplicate id {row_id}")
        out[row_id] = row
    return out


def normalize_rule_class(rule_class: str) -> str:
    rule_class = rule_class.strip()
    prefix = "v274_guarded_numeric_"
    if rule_class.startswith(prefix):
        return rule_class[len(prefix) :]
    return rule_class


def boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    start = idx + len(marker)
    depth = 1
    chars: list[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        user_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
        return "\n".join(user_parts)
    return ""


def extract_metadata_seed_ids(metadata: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key, value in metadata.items():
        if "seed_ids" not in key and "candidate_ids" not in key:
            continue
        if isinstance(value, list):
            ids.extend(str(x) for x in value)
        elif isinstance(value, str):
            ids.append(value)
    return ids


def summarize_predictions(rows: list[dict[str, str]], correct_col: str) -> dict[str, Any]:
    family = {name: {"rows": 0, "correct": 0, "truncated": 0} for name in EXPECTED_FAMILIES}
    total = {"rows": 0, "correct": 0, "truncated": 0}
    for row in rows:
        fam = family_of(row)
        if fam not in family:
            continue
        correct = as_bool(row.get(correct_col, ""))
        truncated = as_bool(row.get("truncated", ""))
        total["rows"] += 1
        family[fam]["rows"] += 1
        if correct:
            total["correct"] += 1
            family[fam]["correct"] += 1
        if truncated:
            total["truncated"] += 1
            family[fam]["truncated"] += 1
    return {
        "rows": total["rows"],
        "correct": total["correct"],
        "accuracy": total["correct"] / total["rows"] if total["rows"] else 0.0,
        "truncated": total["truncated"],
        "family": family,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(paths: list[Path], weak_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    prompt_to_weak_id = {str(row.get("prompt", "")): row_id for row_id, row in weak_by_id.items()}
    direct_id_counts: Counter[str] = Counter()
    metadata_seed_counts: Counter[str] = Counter()
    prompt_overlap_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    preference_rows = 0
    boxed_chosen_counts: Counter[str] = Counter()
    boxed_rejected_counts: Counter[str] = Counter()

    for path in paths:
        for row in read_jsonl(path):
            row_id = str(row.get("id", "")).strip()
            if row_id in weak_by_id:
                direct_id_counts[row_id] += 1
            weak_prompt_id = prompt_to_weak_id.get(prompt_text(row))
            if weak_prompt_id:
                prompt_overlap_counts[weak_prompt_id] += 1

            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            rule = normalize_rule_class(str(metadata.get("rule_class", "")))
            if rule:
                rule_counts[rule] += 1
            for seed_id in extract_metadata_seed_ids(metadata):
                metadata_seed_counts[seed_id] += 1

            if "chosen" in row or "rejected" in row:
                preference_rows += 1
                boxed_chosen_counts[boxed_answer(str(row.get("chosen", "")))] += 1
                boxed_rejected_counts[boxed_answer(str(row.get("rejected", "")))] += 1

    return {
        "direct_id_counts": dict(direct_id_counts),
        "metadata_seed_counts": dict(metadata_seed_counts),
        "prompt_overlap_counts": dict(prompt_overlap_counts),
        "rule_counts": dict(rule_counts),
        "preference_rows": preference_rows,
        "boxed_chosen_counts": dict(boxed_chosen_counts),
        "boxed_rejected_counts": dict(boxed_rejected_counts),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_predictions_csv)
    v344_path = Path(args.v344_predictions_csv)
    v343_path = Path(args.v343_predictions_csv)
    trace_path = Path(args.v343_candidate_trace_csv)
    dataset_paths = [Path(p) for p in args.v344_dataset_jsonl]

    baseline_rows = read_csv_rows(baseline_path)
    v344_rows = read_csv_rows(v344_path)
    v343_rows = read_csv_rows(v343_path)
    trace_rows = read_csv_rows(trace_path) if trace_path.exists() else []

    baseline = keyed(baseline_rows, "baseline")
    v344 = keyed(v344_rows, "v344")
    v343 = keyed(v343_rows, "v343")
    if set(baseline) != set(v344) or set(baseline) != set(v343):
        raise ValueError("row contract mismatch among baseline, V344, and V343 predictions")
    if len(baseline) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} weak rows, got {len(baseline)}")

    trace_by_id = {row["id"]: row for row in trace_rows if row.get("id")}
    dataset_audit = audit_dataset(dataset_paths, baseline)

    rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    v343_gain_rows: list[dict[str, Any]] = []
    counts = Counter()
    counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)

    for row_id, base in baseline.items():
        current = v344[row_id]
        solver = v343[row_id]
        family = family_of(base)
        if family != family_of(current) or family != family_of(solver):
            raise ValueError(f"family mismatch for {row_id}")
        answer = str(base.get("answer", "")).strip()
        if answer != str(current.get("answer", "")).strip() or answer != str(solver.get("answer", "")).strip():
            raise ValueError(f"answer mismatch for {row_id}")

        baseline_correct = as_bool(base.get("correct", ""))
        v344_correct = as_bool(current.get("correct", ""))
        v343_correct = as_bool(solver.get("integrated_correct", ""))
        base_pred = str(base.get("prediction", "")).strip()
        v344_pred = str(current.get("prediction", "")).strip()
        v343_pred = str(solver.get("prediction", "")).strip()
        changed = base_pred != v344_pred
        v343_gain = (not baseline_correct) and v343_correct

        if (not baseline_correct) and v344_correct:
            bucket = "v344_gain_vs_baseline"
        elif baseline_correct and not v344_correct:
            bucket = "v344_loss_vs_baseline"
        elif v343_gain and not v344_correct:
            bucket = "v343_gain_not_transferred"
        elif changed:
            bucket = "v344_changed_no_accuracy_delta"
        else:
            bucket = "neutral"
        counts[bucket] += 1
        counts_by_family[family][bucket] += 1

        trace = trace_by_id.get(row_id, {})
        rule_class = str(trace.get("rule_class", "")).strip()
        normalized_rule = normalize_rule_class(rule_class)
        detail = {
            "id": row_id,
            "family": family,
            "answer": answer,
            "baseline_prediction": base_pred,
            "v344_prediction": v344_pred,
            "v343_prediction": v343_pred,
            "baseline_correct": baseline_correct,
            "v344_correct": v344_correct,
            "v343_integrated_correct": v343_correct,
            "v344_changed_prediction": changed,
            "bucket": bucket,
            "v343_rule_class": rule_class,
            "v343_rule_class_normalized": normalized_rule,
            "dataset_direct_id_count": dataset_audit["direct_id_counts"].get(row_id, 0),
            "dataset_prompt_overlap_count": dataset_audit["prompt_overlap_counts"].get(row_id, 0),
            "dataset_metadata_seed_count": dataset_audit["metadata_seed_counts"].get(row_id, 0),
            "dataset_rule_count": dataset_audit["rule_counts"].get(normalized_rule, 0),
            "v343_candidate_count": trace.get("candidate_count", ""),
            "v343_conflict_count": trace.get("conflict_count", ""),
            "v343_reason": trace.get("reason", ""),
        }
        rows.append(detail)
        if changed:
            changed_rows.append(detail)
        if v343_gain:
            v343_gain_rows.append(detail)

    outputs = {
        "row_comparison_csv": str(output_dir / "v345_v344_failure_row_comparison.csv"),
        "changed_predictions_csv": str(output_dir / "v345_v344_changed_predictions.csv"),
        "v343_gains_not_transferred_csv": str(output_dir / "v345_v343_gains_not_transferred.csv"),
        "manifest_json": str(output_dir / "v345_v344_failure_audit_manifest.json"),
    }
    write_csv(Path(outputs["row_comparison_csv"]), rows)
    write_csv(Path(outputs["changed_predictions_csv"]), changed_rows)
    write_csv(Path(outputs["v343_gains_not_transferred_csv"]), v343_gain_rows)

    v343_gain_coverage = {
        row["id"]: {
            "rule_class": row["v343_rule_class"],
            "normalized_rule": row["v343_rule_class_normalized"],
            "direct_id_count": row["dataset_direct_id_count"],
            "prompt_overlap_count": row["dataset_prompt_overlap_count"],
            "metadata_seed_count": row["dataset_metadata_seed_count"],
            "rule_count": row["dataset_rule_count"],
            "v344_changed_prediction": row["v344_changed_prediction"],
            "v344_correct": row["v344_correct"],
        }
        for row in v343_gain_rows
    }

    decision = {
        "decision": "block_repeated_v344_gpu_until_training_signal_changes",
        "reason": (
            "V344 checkpoint-2 preserved bit but produced 0 weak gains. "
            "The V343 rule classes are covered by synthetic transfer rows, but exact weak rows and prompts "
            "are intentionally absent; LR=1e-09 plus saturated preference accuracy did not transfer the rules."
        ),
        "next_action": (
            "Build an ACC-sensitive V346 trainer/gate: either answer-only hard-positive micro-overfit on "
            "rule-class synthetic rows with higher LR and immediate weak eval, or return to CPU solver DSL. "
            "Do not rerun the same preference objective."
        ),
    }

    manifest: dict[str, Any] = {
        "schema_version": "kg1_v345_v344_failure_audit_v1",
        "inputs": {
            "baseline_predictions_csv": str(baseline_path),
            "baseline_predictions_sha256": sha256_path(baseline_path),
            "v344_predictions_csv": str(v344_path),
            "v344_predictions_sha256": sha256_path(v344_path),
            "v343_predictions_csv": str(v343_path),
            "v343_predictions_sha256": sha256_path(v343_path),
            "v343_candidate_trace_csv": str(trace_path),
            "v343_candidate_trace_sha256": sha256_path(trace_path) if trace_path.exists() else "",
            "v344_dataset_jsonl": [str(p) for p in dataset_paths],
            "v344_dataset_sha256": {str(p): sha256_path(p) for p in dataset_paths},
        },
        "summaries": {
            "baseline_adapter_only": summarize_predictions(baseline_rows, "correct"),
            "v344_preference_abstain_checkpoint2": summarize_predictions(v344_rows, "correct"),
            "v343_cpu_solver_verifier": summarize_predictions(v343_rows, "integrated_correct"),
        },
        "delta_counts": dict(counts),
        "delta_counts_by_family": {family: dict(counter) for family, counter in counts_by_family.items()},
        "changed_prediction_count": len(changed_rows),
        "changed_prediction_ids": [row["id"] for row in changed_rows],
        "v343_gain_count": len(v343_gain_rows),
        "v343_gain_ids": [row["id"] for row in v343_gain_rows],
        "v343_gain_coverage": v343_gain_coverage,
        "dataset_audit": {
            "preference_rows": dataset_audit["preference_rows"],
            "rule_counts": dataset_audit["rule_counts"],
            "direct_id_counts_for_v343_gains": {
                row["id"]: row["dataset_direct_id_count"] for row in v343_gain_rows
            },
            "prompt_overlap_counts_for_v343_gains": {
                row["id"]: row["dataset_prompt_overlap_count"] for row in v343_gain_rows
            },
            "metadata_seed_counts_for_v343_gains": {
                row["id"]: row["dataset_metadata_seed_count"] for row in v343_gain_rows
            },
        },
        "decision": decision,
        "outputs": outputs,
    }
    Path(outputs["manifest_json"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("=== V345 V344 FAILURE AUDIT SUMMARY ===", flush=True)
    print("summaries =", json.dumps(manifest["summaries"], indent=2, sort_keys=True), flush=True)
    print("delta_counts =", json.dumps(manifest["delta_counts"], sort_keys=True), flush=True)
    print("changed_prediction_ids =", json.dumps(manifest["changed_prediction_ids"], sort_keys=True), flush=True)
    print("v343_gain_coverage =", json.dumps(v343_gain_coverage, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions-csv", required=True)
    parser.add_argument("--v344-predictions-csv", required=True)
    parser.add_argument("--v343-predictions-csv", required=True)
    parser.add_argument("--v343-candidate-trace-csv", required=True)
    parser.add_argument("--v344-dataset-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=315)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

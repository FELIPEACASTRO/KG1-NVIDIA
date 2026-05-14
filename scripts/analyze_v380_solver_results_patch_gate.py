#!/usr/bin/env python3
"""V380 CPU-only solver-results patch gate.

This gate audits whether the V378 solver_results signal can improve the V366
weak teacher without losses. It separates three levels:

1. oracle_solver_answer: directly use solver_answer from the audited parquet
   coverage CSV. This is diagnostic only.
2. reexecuted_solver_ops: recompute the answer from prompt + solver_ops +
   solver_mapping. This is stronger but still not fully independent when the
   mapping was conditioned on the answer.
3. strict_independent: reexecuted and not conditioned_on_answer. This is the
   only class that can directly unlock HF by itself.

The script does not train, package, submit, or call any remote service.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import parse_alice_prompt  # noqa: E402


DEFAULT_V366_CSV = (
    REPO_ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv"
)
DEFAULT_V375_RESIDUAL_CSV = (
    REPO_ROOT / "artifacts/v375_equation_residual_clustering/20260514T141424Z/v375_equation_residual_rows.csv"
)
DEFAULT_V378_SOLVER_COVERAGE_CSV = (
    REPO_ROOT / "artifacts/v378_nemotron_dataset_final_audit/v378_v375_solver_coverage.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v380_solver_results_patch_gate/20260514T_cpu_gate"


CANDIDATE_COLUMNS = [
    "id",
    "subtype",
    "solver_category",
    "solver_mode",
    "conditioned_on_answer",
    "old_prediction",
    "answer",
    "solver_answer",
    "oracle_solver_correct",
    "reexecuted_answer",
    "reexecuted_correct",
    "old_correct",
    "oracle_gain",
    "reexecuted_gain",
    "strict_independent_gain",
    "accepted_for_v381_teacher",
    "accepted_for_hf_unlock",
    "rejection_reason",
    "solver_ops",
    "solver_mapping",
]
SUMMARY_COLUMNS = [
    "strategy",
    "rows",
    "correct",
    "accuracy",
    "equation_transform_correct",
    "bit_manipulation_correct",
    "truncated",
    "delta_total_vs_v366",
    "delta_equation_vs_v366",
    "delta_bit_vs_v366",
]
CATEGORY_COLUMNS = [
    "solver_category",
    "conditioned_on_answer",
    "rows",
    "oracle_correct",
    "reexecuted_correct",
    "old_correct",
    "oracle_gains",
    "reexecuted_gains",
    "strict_independent_gains",
    "accepted_for_v381_teacher",
    "accepted_for_hf_unlock",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right) if left and right else 0


def apply_op(left: int, right: int, op_name: str) -> int | None:
    if op_name == "add":
        return left + right
    if op_name == "add_m1":
        return left + right - 1
    if op_name == "add_p1":
        return left + right + 1
    if op_name == "mul":
        return left * right
    if op_name == "mul_m1":
        return left * right - 1
    if op_name == "mul_p1":
        return left * right + 1
    if op_name == "absdiff":
        return abs(left - right)
    if op_name == "absdiff_m2":
        return abs(left - right) - 2
    if op_name == "sub_signed":
        return left - right
    if op_name == "rsub_signed":
        return right - left
    if op_name == "mod":
        return left % right if right else None
    if op_name == "gcd":
        return math.gcd(left, right)
    if op_name == "lcm":
        return safe_lcm(left, right)
    if op_name == "neg_absdiff":
        return -abs(left - right)
    if op_name == "concat_fwd":
        return int(f"{left}{right}")
    if op_name == "concat_rev":
        return int(f"{right}{left}")
    return None


def decode_number(text: str, mapping: dict[str, int], *, little_endian: bool) -> int | None:
    ordered = text[::-1] if little_endian else text
    try:
        return int("".join(str(mapping[ch]) for ch in ordered))
    except Exception:
        return None


def encode_number(value: int | None, inverse: dict[int, str], *, little_endian: bool) -> str:
    if value is None:
        return ""
    sign = "-" if value < 0 else ""
    digits = str(abs(int(value)))
    try:
        encoded = "".join(inverse[int(ch)] for ch in digits)
    except Exception:
        return ""
    if little_endian:
        encoded = encoded[::-1]
    return sign + encoded


def expression_candidates(
    expression: str,
    solver_ops: dict[str, str],
    mapping: dict[str, int],
    *,
    little_endian: bool,
) -> list[str]:
    inverse = {value: key for key, value in mapping.items()}
    out: list[str] = []
    for index, op_symbol in enumerate(expression):
        if op_symbol not in solver_ops or index == 0 or index == len(expression) - 1:
            continue
        left = decode_number(expression[:index], mapping, little_endian=little_endian)
        right = decode_number(expression[index + 1 :], mapping, little_endian=little_endian)
        if left is None or right is None:
            continue
        value = apply_op(left, right, solver_ops[op_symbol])
        encoded = encode_number(value, inverse, little_endian=little_endian)
        if encoded:
            out.append(encoded)
    return out


def reexecute_solver(prompt: str, solver_ops_text: str, mapping_text: str, solver_mode: str) -> tuple[str, str]:
    if not solver_ops_text or not mapping_text:
        return "", "missing_solver_ops_or_mapping"
    try:
        solver_ops = json.loads(solver_ops_text)
        mapping = {str(key): int(value) for key, value in json.loads(mapping_text).items()}
    except Exception as exc:
        return "", f"json_parse_failed:{exc}"
    _examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        return "", parse_status
    candidates = expression_candidates(
        query,
        solver_ops,
        mapping,
        little_endian=(solver_mode == "little_endian"),
    )
    if not candidates:
        return "", "no_query_candidate"
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        return unique[0], "multiple_query_candidates_first_used"
    return unique[0], "ok"


def summarize_predictions(rows: list[dict[str, Any]], prediction_key: str, baseline_summary: dict[str, Any]) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        total += 1
        families[family]["rows"] += 1
        ok = verify_answer(row["answer"], row[prediction_key])
        correct += int(ok)
        families[family]["correct"] += int(ok)
        is_truncated = truthy(row.get("truncated", row.get("truncated_bool", "")))
        truncated += int(is_truncated)
        families[family]["truncated"] += int(is_truncated)
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": {key: dict(value) for key, value in sorted(families.items())},
        "delta_total_vs_v366": correct - baseline_summary["correct"],
        "delta_equation_vs_v366": families["equation_transform"]["correct"]
        - baseline_summary["family"]["equation_transform"]["correct"],
        "delta_bit_vs_v366": families["bit_manipulation"]["correct"]
        - baseline_summary["family"]["bit_manipulation"]["correct"],
    }


def summary_row(strategy: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "rows": summary["rows"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "equation_transform_correct": summary["family"].get("equation_transform", {}).get("correct", 0),
        "bit_manipulation_correct": summary["family"].get("bit_manipulation", {}).get("correct", 0),
        "truncated": summary["truncated"],
        "delta_total_vs_v366": summary["delta_total_vs_v366"],
        "delta_equation_vs_v366": summary["delta_equation_vs_v366"],
        "delta_bit_vs_v366": summary["delta_bit_vs_v366"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.v366_csv)
    residual_by_id = {row["id"]: row for row in read_csv(args.v375_residual_csv)}
    solver_by_id = {row["id"]: row for row in read_csv(args.v378_solver_coverage_csv)}
    residual_ids = set(residual_by_id)

    for row in rows:
        row["v366_base_prediction"] = row.get("v366_prediction") or row.get("current_prediction") or row.get("prediction")
        row["oracle_solver_patch_prediction"] = row["v366_base_prediction"]
        row["reexecuted_solver_patch_prediction"] = row["v366_base_prediction"]
        row["strict_independent_patch_prediction"] = row["v366_base_prediction"]

    candidates: list[dict[str, Any]] = []
    category_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_id = {row["id"]: row for row in rows}
    for row_id in sorted(residual_ids & set(solver_by_id) & set(by_id)):
        base = by_id[row_id]
        residual = residual_by_id[row_id]
        solver = solver_by_id[row_id]
        old_prediction = base["v366_base_prediction"]
        solver_answer = str(solver.get("solver_answer", ""))
        answer = str(base["answer"])
        old_correct = verify_answer(answer, old_prediction)
        oracle_correct = truthy(solver.get("metric_correct"))
        conditioned = truthy(solver.get("conditioned_on_answer"))
        reexecuted, reexec_status = reexecute_solver(
            base["prompt"],
            solver.get("solver_ops", ""),
            solver.get("solver_mapping", ""),
            solver.get("solver_mode", ""),
        )
        reexecuted_correct = verify_answer(answer, reexecuted)
        oracle_gain = (not old_correct) and oracle_correct
        reexecuted_gain = (not old_correct) and reexecuted_correct
        strict_gain = reexecuted_gain and not conditioned
        accepted_for_v381 = reexecuted_gain
        accepted_for_hf = strict_gain
        if accepted_for_v381:
            base["reexecuted_solver_patch_prediction"] = reexecuted
        if oracle_gain:
            base["oracle_solver_patch_prediction"] = solver_answer
        if accepted_for_hf:
            base["strict_independent_patch_prediction"] = reexecuted
        if accepted_for_hf:
            reason = "accepted_strict_independent"
        elif accepted_for_v381:
            reason = "teacher_only_conditioned_mapping"
        elif oracle_gain:
            reason = "oracle_only_reexecution_failed"
        else:
            reason = "not_correct_or_no_gain"
        item = {
            "id": row_id,
            "subtype": residual.get("subtype", ""),
            "solver_category": solver.get("solver_category", "") or "None",
            "solver_mode": solver.get("solver_mode", "") or "None",
            "conditioned_on_answer": conditioned,
            "old_prediction": old_prediction,
            "answer": answer,
            "solver_answer": solver_answer,
            "oracle_solver_correct": oracle_correct,
            "reexecuted_answer": reexecuted,
            "reexecuted_correct": reexecuted_correct,
            "old_correct": old_correct,
            "oracle_gain": oracle_gain,
            "reexecuted_gain": reexecuted_gain,
            "strict_independent_gain": strict_gain,
            "accepted_for_v381_teacher": accepted_for_v381,
            "accepted_for_hf_unlock": accepted_for_hf,
            "rejection_reason": reason if reexec_status == "ok" else f"{reason};{reexec_status}",
            "solver_ops": solver.get("solver_ops", ""),
            "solver_mapping": solver.get("solver_mapping", ""),
        }
        candidates.append(item)
        key = (item["solver_category"], str(conditioned))
        category_counts[key]["rows"] += 1
        category_counts[key]["old_correct"] += int(old_correct)
        category_counts[key]["oracle_correct"] += int(oracle_correct)
        category_counts[key]["reexecuted_correct"] += int(reexecuted_correct)
        category_counts[key]["oracle_gains"] += int(oracle_gain)
        category_counts[key]["reexecuted_gains"] += int(reexecuted_gain)
        category_counts[key]["strict_independent_gains"] += int(strict_gain)
        category_counts[key]["accepted_for_v381_teacher"] += int(accepted_for_v381)
        category_counts[key]["accepted_for_hf_unlock"] += int(accepted_for_hf)

    baseline = summarize_predictions(rows, "v366_base_prediction", {"correct": 0, "family": defaultdict(Counter)})
    baseline["delta_total_vs_v366"] = 0
    baseline["delta_equation_vs_v366"] = 0
    baseline["delta_bit_vs_v366"] = 0
    oracle = summarize_predictions(rows, "oracle_solver_patch_prediction", baseline)
    reexecuted = summarize_predictions(rows, "reexecuted_solver_patch_prediction", baseline)
    strict = summarize_predictions(rows, "strict_independent_patch_prediction", baseline)
    summary_rows = [
        summary_row("v366_baseline", baseline),
        summary_row("oracle_solver_answer_patch_diagnostic_only", oracle),
        summary_row("reexecuted_solver_ops_teacher_patch", reexecuted),
        summary_row("strict_independent_patch", strict),
    ]
    category_rows = [
        {
            "solver_category": key[0],
            "conditioned_on_answer": key[1],
            **dict(counts),
        }
        for key, counts in sorted(category_counts.items())
    ]

    candidate_path = output_dir / "v380_solver_results_candidate_patch.csv"
    summary_path = output_dir / "v380_solver_results_patch_summary.csv"
    category_path = output_dir / "v380_solver_results_category_summary.csv"
    predictions_path = output_dir / "v380_reexecuted_teacher_predictions.csv"
    manifest_path = output_dir / "v380_solver_results_patch_gate_manifest.json"
    write_csv(candidate_path, candidates, CANDIDATE_COLUMNS)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(category_path, category_rows, CATEGORY_COLUMNS)
    write_csv(predictions_path, rows, list(rows[0].keys()) if rows else [])

    strict_gains = sum(int(row["accepted_for_hf_unlock"]) for row in candidates)
    teacher_gains = sum(int(row["accepted_for_v381_teacher"]) for row in candidates)
    oracle_gains = sum(int(row["oracle_gain"]) for row in candidates)
    decision = {
        "decision": "teacher_signal_only_no_hf_unlock" if teacher_gains and not strict_gains else "strict_independent_gain_found",
        "hf_gpu_allowed": bool(strict_gains),
        "v381_dataset_gate_allowed": bool(teacher_gains),
        "reason": (
            f"oracle_gains={oracle_gains}; reexecuted_teacher_gains={teacher_gains}; "
            f"strict_independent_gains={strict_gains}"
        ),
        "next_action": (
            "Build V381 cleaned teacher dataset from reexecuted rows, but keep HF blocked until tokenization/overlap gates pass."
            if teacher_gains
            else "No V381/HF; implement a stronger independent equation DSL."
        ),
    }
    manifest = {
        "schema_version": "kg1_v380_solver_results_patch_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v366_csv": str(args.v366_csv),
            "v375_residual_csv": str(args.v375_residual_csv),
            "v378_solver_coverage_csv": str(args.v378_solver_coverage_csv),
        },
        "counts": {
            "weak_rows": len(rows),
            "v375_residual_rows": len(residual_by_id),
            "solver_covered_residual_rows": len(candidates),
            "oracle_gains": oracle_gains,
            "reexecuted_teacher_gains": teacher_gains,
            "strict_independent_gains": strict_gains,
        },
        "summaries": {
            "v366_baseline": baseline,
            "oracle_solver_answer_patch_diagnostic_only": oracle,
            "reexecuted_solver_ops_teacher_patch": reexecuted,
            "strict_independent_patch": strict,
            "category_rows": category_rows,
        },
        "decision": decision,
        "outputs": {
            "candidate_patch_csv": str(candidate_path),
            "summary_csv": str(summary_path),
            "category_summary_csv": str(category_path),
            "reexecuted_teacher_predictions_csv": str(predictions_path),
            "manifest_json": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def self_test() -> int:
    assert apply_op(2, 3, "add") == 5
    assert apply_op(2, 3, "rsub_signed") == 1
    assert encode_number(-8, {8: "!"}, little_endian=False) == "-!"
    assert decode_number("$\"", {"$": 7, '"': 0}, little_endian=True) == 7
    print("v380_solver_results_patch_gate_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v366-csv", type=Path, default=DEFAULT_V366_CSV)
    parser.add_argument("--v375-residual-csv", type=Path, default=DEFAULT_V375_RESIDUAL_CSV)
    parser.add_argument("--v378-solver-coverage-csv", type=Path, default=DEFAULT_V378_SOLVER_COVERAGE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    print("=== V380 SOLVER RESULTS PATCH GATE START ===", flush=True)
    print("v366_csv =", args.v366_csv, flush=True)
    print("v375_residual_csv =", args.v375_residual_csv, flush=True)
    print("v378_solver_coverage_csv =", args.v378_solver_coverage_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    manifest = run(args)
    print("counts =", json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    print("summary_rows =", json.dumps(manifest["summaries"]["category_rows"], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V380 SOLVER RESULTS PATCH GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit broader numeric equation candidates after the V275 postprocessor.

V299 is CPU-only and diagnostic-only. It reads a labeled weak prediction CSV
and asks whether any label-free numeric candidate class adds equation coverage
without losses. Weak labels are used only for audit/gate decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import canonical_family, classify_puzzle, verify_answer
from kg1_v274_numeric_postprocessor import (
    group_examples_by_operator,
    normalize_payload,
    numeric_candidates,
    numeric_rule_functions,
    parse_alice_prompt,
    parse_numeric_token,
)

AUDIT_COLUMNS = [
    "id",
    "family",
    "query",
    "query_op",
    "answer",
    "baseline_prediction",
    "baseline_correct",
    "candidate_class",
    "status",
    "prediction",
    "candidate_rule_count",
    "unique_prediction_count",
    "candidate_rules",
    "candidate_correct",
    "gain",
    "loss",
    "proof",
]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    prediction = str(row.get("prediction", "")).strip()
    answer = str(row.get("answer", "")).strip()
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "family": family,
        "prediction": prediction,
        "answer": answer,
        "correct_bool": verify_answer(answer, prediction),
    }


def unique_prediction_candidate(
    candidate_class: str,
    candidates: list[dict[str, Any]],
    *,
    min_rule_count: int,
    max_rule_count: int,
) -> dict[str, Any]:
    predictions = sorted({normalize_payload(item.get("prediction", "")) for item in candidates if item.get("prediction") is not None})
    rule_names = sorted(
        {
            f"{item.get('name')}|revop={int(bool(item.get('reverse_operands')))}|revres={int(bool(item.get('reverse_result')))}"
            for item in candidates
        }
    )
    if len(predictions) == 1 and min_rule_count <= len(candidates) <= max_rule_count:
        return {
            "candidate_class": candidate_class,
            "status": "candidate",
            "prediction": predictions[0],
            "candidate_rule_count": len(candidates),
            "unique_prediction_count": len(predictions),
            "candidate_rules": ";".join(rule_names[:50]),
            "proof": f"unique_prediction={predictions[0]}; candidate_rules={len(candidates)}",
        }
    return {
        "candidate_class": candidate_class,
        "status": "abstain",
        "prediction": "",
        "candidate_rule_count": len(candidates),
        "unique_prediction_count": len(predictions),
        "candidate_rules": ";".join(rule_names[:50]),
        "proof": f"unique_prediction_count={len(predictions)}; candidate_rules={len(candidates)}",
    }


def conventional_operator_candidates(query: str, query_op: str) -> list[dict[str, Any]]:
    parsed = parse_numeric_token(query)
    if not parsed:
        return []
    left, _, right = parsed
    functions = numeric_rule_functions()
    name_map = {
        "+": {"add", "rev_add", "tens_add_ones_add"},
        "-": {"sub_ab", "abs_diff", "rev_sub_ab", "rev_abs_diff"},
        "*": {"mul", "rev_mul", "digit_mul_mod10_concat"},
        "/": {"abs_diff", "rev_abs_diff", "digit_absdiff_concat", "tens_absdiff_ones_absdiff_int"},
        ":": {"abs_diff", "rev_abs_diff", "digit_absdiff_concat", "tens_absdiff_ones_absdiff_int"},
    }
    names = name_map.get(query_op, set())
    rows: list[dict[str, Any]] = []
    for name in names:
        func = functions[name]
        for reverse_operands in (False, True):
            for reverse_result in (False, True):
                lhs = left[::-1] if reverse_operands else left
                rhs = right[::-1] if reverse_operands else right
                try:
                    raw = str(func(int(lhs), int(rhs)))
                except Exception:
                    continue
                prediction = raw[::-1] if reverse_result and not raw.startswith("-") else (
                    "-" + raw[1:][::-1] if reverse_result and raw.startswith("-") else raw
                )
                rows.append(
                    {
                        "name": name,
                        "reverse_operands": reverse_operands,
                        "reverse_result": reverse_result,
                        "prediction": prediction,
                    }
                )
    return rows


def audit_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row["family"] != "equation_transform":
        return []
    examples, query, parse_status = parse_alice_prompt(row["prompt"])
    parsed_query = parse_numeric_token(query)
    if parse_status != "ok" or not parsed_query:
        return []
    grouped_result = group_examples_by_operator(examples)
    if grouped_result is None:
        return []
    grouped, _ = grouped_result
    query_op = parsed_query[1]
    all_names = set(numeric_rule_functions())
    all_numeric_examples = [item for group in grouped.values() for item in group]

    candidates_by_class: list[dict[str, Any]] = []
    if query_op in grouped:
        candidates_by_class.append(
            unique_prediction_candidate(
                "same_operator_unique_numeric_dsl",
                numeric_candidates(grouped[query_op], query, all_names),
                min_rule_count=1,
                max_rule_count=4,
            )
        )
    else:
        candidates_by_class.append(
            {
                "candidate_class": "same_operator_unique_numeric_dsl",
                "status": "abstain",
                "prediction": "",
                "candidate_rule_count": 0,
                "unique_prediction_count": 0,
                "candidate_rules": "",
                "proof": f"query_op={query_op!r} absent from examples",
            }
        )

    candidates_by_class.append(
        unique_prediction_candidate(
            "all_numeric_examples_unique_dsl",
            numeric_candidates(all_numeric_examples, query, all_names),
            min_rule_count=1,
            max_rule_count=4,
        )
    )
    candidates_by_class.append(
        unique_prediction_candidate(
            "conventional_operator_prior_unique",
            conventional_operator_candidates(query, query_op),
            min_rule_count=1,
            max_rule_count=2,
        )
    )

    out: list[dict[str, Any]] = []
    for candidate in candidates_by_class:
        pred = str(candidate.get("prediction", ""))
        candidate_correct = bool(pred) and verify_answer(row["answer"], pred)
        baseline_correct = bool(row["correct_bool"])
        out.append(
            {
                "id": row["id"],
                "family": row["family"],
                "query": query,
                "query_op": query_op,
                "answer": row["answer"],
                "baseline_prediction": row["prediction"],
                "baseline_correct": baseline_correct,
                **candidate,
                "candidate_correct": candidate_correct,
                "gain": (not baseline_correct) and candidate_correct and candidate["status"] == "candidate",
                "loss": baseline_correct and (not candidate_correct) and candidate["status"] == "candidate",
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cls = str(row["candidate_class"])
        grouped[cls]["rows"] += 1
        if row["status"] == "candidate":
            grouped[cls]["candidate_rows"] += 1
        if row["candidate_correct"]:
            grouped[cls]["candidate_correct"] += 1
        if row["gain"]:
            grouped[cls]["gains"] += 1
        if row["loss"]:
            grouped[cls]["losses"] += 1
        if row["status"] == "candidate" and (not row["baseline_correct"]) and (not row["candidate_correct"]):
            grouped[cls]["wrong_on_baseline_miss"] += 1
    return [{"candidate_class": cls, **dict(counter)} for cls, counter in sorted(grouped.items())]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [normalize_row(row) for row in read_csv(args.input_csv)]
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        audit_rows.extend(audit_row(row))
    summary_rows = summarize(audit_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_dir / "v299_equation_numeric_candidate_audit.csv"
    summary_csv = args.output_dir / "v299_equation_numeric_candidate_summary.csv"
    write_csv(audit_csv, audit_rows, AUDIT_COLUMNS)
    write_csv(
        summary_csv,
        summary_rows,
        ["candidate_class", "rows", "candidate_rows", "candidate_correct", "gains", "losses", "wrong_on_baseline_miss"],
    )
    baseline_eq_correct = sum(1 for row in rows if row["family"] == "equation_transform" and row["correct_bool"])
    manifest = {
        "schema_version": "kg1_v299_equation_numeric_candidate_audit_v1",
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "rows": len(rows),
        "equation_rows": sum(1 for row in rows if row["family"] == "equation_transform"),
        "baseline_equation_correct": baseline_eq_correct,
        "numeric_audit_rows": len(audit_rows),
        "summary": summary_rows,
        "decision": {
            "decision": "no_numeric_candidate_promoted_until_zero_loss",
            "reason": "Candidate classes are diagnostic unless they have gains, zero losses, and no wrong_on_baseline_miss.",
        },
        "outputs": {
            "audit_csv": str(audit_csv),
            "summary_csv": str(summary_csv),
            "manifest_json": str(args.output_dir / "v299_equation_numeric_candidate_manifest.json"),
        },
    }
    for row in summary_rows:
        if int(row.get("gains", 0)) > 0 and int(row.get("losses", 0)) == 0 and int(row.get("wrong_on_baseline_miss", 0)) == 0:
            manifest["decision"] = {
                "decision": "numeric_candidate_class_needs_deployable_gate",
                "reason": f"{row['candidate_class']} has gains={row.get('gains', 0)} losses=0 wrong_on_baseline_miss=0; inspect for label-free deployability.",
            }
            break
    write_json(args.output_dir / "v299_equation_numeric_candidate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

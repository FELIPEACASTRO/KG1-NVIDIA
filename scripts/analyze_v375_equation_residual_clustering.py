#!/usr/bin/env python3
"""V375 CPU-only clustering for remaining equation_transform misses.

This script is diagnostic. It does not train, launch GPU jobs, package, or
submit. It clusters the equation misses that remain after the V366 CPU teacher
and the V374/V324 expanded equation gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    EXPECTED_ROW_CONTRACT_SHA256,
    classify_subtype,
    normalize_row,
    parse_alice_prompt,
    row_contract,
    sha256_file,
)


DEFAULT_V366_CSV = (
    REPO_ROOT
    / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv"
)
DEFAULT_V324_AUDIT_CSV = (
    REPO_ROOT
    / "artifacts/v374_cpu_residual_gate/20260514T_v374_cpu_gate/equation_v324_on_v366/"
    / "v324_equation_expanded_solver_audit.csv"
)

RESIDUAL_COLUMNS = [
    "id",
    "subtype",
    "answer",
    "current_prediction",
    "query",
    "examples_count",
    "query_len",
    "answer_len",
    "prediction_len",
    "query_charset",
    "answer_charset",
    "query_symbols",
    "example_input_symbols",
    "example_output_symbols",
    "query_symbols_seen_in_example_inputs",
    "answer_chars_subset_query",
    "answer_chars_subset_example_outputs",
    "answer_is_subsequence_of_query",
    "answer_is_substring_of_query",
    "answer_reverse_is_substring_of_query",
    "edit_distance_query_answer",
    "v324_candidate_rows",
    "v324_incorrect_candidates",
    "v324_abstain_rows",
    "cluster_key",
    "priority_reason",
]

CLUSTER_COLUMNS = [
    "cluster_key",
    "rows",
    "subtype",
    "answer_len",
    "query_len",
    "query_symbols_seen_in_example_inputs",
    "answer_chars_subset_query",
    "answer_chars_subset_example_outputs",
    "answer_is_subsequence_of_query",
    "v324_candidate_rows",
    "v324_incorrect_candidates",
    "priority_rows",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def symbols(text: str) -> str:
    return "".join(sorted({ch for ch in str(text) if not ch.isalnum()}))


def charset(text: str) -> str:
    return "".join(sorted(set(str(text))))


def is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    iterator = iter(haystack)
    return all(ch in iterator for ch in needle)


def edit_distance(a: str, b: str, cap: int = 12) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            value = min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + int(ca != cb))
            cur.append(value)
            row_min = min(row_min, value)
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def audit_counts_by_id(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        row_id = str(row.get("id", ""))
        status = str(row.get("candidate_status") or row.get("status") or "")
        verified = str(row.get("candidate_correct") or row.get("verified") or "").lower() in {"1", "true", "yes"}
        prediction = str(row.get("candidate_prediction") or row.get("prediction") or "")
        if prediction:
            out[row_id]["candidate_rows"] += 1
        if prediction and not verified:
            out[row_id]["incorrect_candidates"] += 1
        if status == "abstain" or not prediction:
            out[row_id]["abstain_rows"] += 1
    return out


def priority_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if row["subtype"] == "equation_numeric_operator":
        reasons.append("numeric_operator_residual")
    if row["answer_chars_subset_query"]:
        reasons.append("answer_chars_subset_query")
    if row["answer_is_subsequence_of_query"]:
        reasons.append("answer_subsequence_query")
    if row["answer_chars_subset_example_outputs"]:
        reasons.append("answer_chars_seen_in_outputs")
    if int(row["v324_candidate_rows"]) > 0:
        reasons.append("v324_had_candidate_but_failed")
    if not reasons:
        reasons.append("opaque_symbolic_residual")
    return "|".join(reasons)


def build_residual_rows(prediction_rows: list[dict[str, Any]], audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = audit_counts_by_id(audit_rows)
    residuals = []
    for raw in prediction_rows:
        row = normalize_row(raw)
        row.update(raw)
        if row.get("family") != "equation_transform":
            continue
        prediction = str(row.get("prediction") or row.get("v366_prediction") or "")
        if verify_answer(str(row.get("answer", "")), prediction):
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        if parse_status != "ok":
            subtype = "parse_" + parse_status
            examples = []
            query = ""
        else:
            subtype = classify_subtype(examples, query)
        example_inputs = [lhs for lhs, _rhs in examples]
        example_outputs = [rhs for _lhs, rhs in examples]
        all_example_inputs = "".join(example_inputs)
        all_example_outputs = "".join(example_outputs)
        answer = str(row.get("answer", ""))
        query_symbols = symbols(query)
        example_input_symbols = symbols(all_example_inputs)
        query_symbols_seen = set(query_symbols).issubset(set(example_input_symbols))
        item = {
            "id": row.get("id", ""),
            "subtype": subtype,
            "answer": answer,
            "current_prediction": prediction,
            "query": query,
            "examples_count": len(examples),
            "query_len": len(query),
            "answer_len": len(answer),
            "prediction_len": len(prediction),
            "query_charset": charset(query),
            "answer_charset": charset(answer),
            "query_symbols": query_symbols,
            "example_input_symbols": example_input_symbols,
            "example_output_symbols": symbols(all_example_outputs),
            "query_symbols_seen_in_example_inputs": query_symbols_seen,
            "answer_chars_subset_query": set(answer).issubset(set(query)),
            "answer_chars_subset_example_outputs": set(answer).issubset(set(all_example_outputs)),
            "answer_is_subsequence_of_query": is_subsequence(answer, query),
            "answer_is_substring_of_query": answer in query,
            "answer_reverse_is_substring_of_query": answer[::-1] in query,
            "edit_distance_query_answer": edit_distance(query, answer),
            "v324_candidate_rows": int(counts[str(row.get("id", ""))]["candidate_rows"]),
            "v324_incorrect_candidates": int(counts[str(row.get("id", ""))]["incorrect_candidates"]),
            "v324_abstain_rows": int(counts[str(row.get("id", ""))]["abstain_rows"]),
        }
        item["cluster_key"] = "|".join(
            [
                str(item["subtype"]),
                "alen=" + str(item["answer_len"]),
                "qlen=" + str(item["query_len"]),
                "qops_seen=" + str(int(bool(item["query_symbols_seen_in_example_inputs"]))),
                "ans_subset_q=" + str(int(bool(item["answer_chars_subset_query"]))),
                "ans_subset_out=" + str(int(bool(item["answer_chars_subset_example_outputs"]))),
                "ans_subseq_q=" + str(int(bool(item["answer_is_subsequence_of_query"]))),
            ]
        )
        item["priority_reason"] = priority_reason(item)
        residuals.append(item)
    return residuals


def cluster_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_key"])].append(row)
    out = []
    for key, items in grouped.items():
        first = items[0]
        out.append(
            {
                "cluster_key": key,
                "rows": len(items),
                "subtype": first["subtype"],
                "answer_len": first["answer_len"],
                "query_len": first["query_len"],
                "query_symbols_seen_in_example_inputs": first["query_symbols_seen_in_example_inputs"],
                "answer_chars_subset_query": first["answer_chars_subset_query"],
                "answer_chars_subset_example_outputs": first["answer_chars_subset_example_outputs"],
                "answer_is_subsequence_of_query": first["answer_is_subsequence_of_query"],
                "v324_candidate_rows": sum(int(row["v324_candidate_rows"]) for row in items),
                "v324_incorrect_candidates": sum(int(row["v324_incorrect_candidates"]) for row in items),
                "priority_rows": sum("opaque" not in str(row["priority_reason"]) for row in items),
            }
        )
    out.sort(key=lambda row: (int(row["priority_rows"]), int(row["rows"])), reverse=True)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V375 EQUATION RESIDUAL CLUSTERING START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v366_predictions_csv =", args.v366_predictions_csv, flush=True)
    print("v324_audit_csv =", args.v324_audit_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = read_csv(args.v366_predictions_csv)
    observed_contract = row_contract([normalize_row(row) for row in prediction_rows])
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    audit_rows = read_csv(args.v324_audit_csv)
    residuals = build_residual_rows(prediction_rows, audit_rows)
    clusters = cluster_rows(residuals)
    priority = [
        row
        for row in residuals
        if "opaque_symbolic_residual" not in str(row["priority_reason"]) or int(row["v324_candidate_rows"]) > 0
    ]
    priority.sort(key=lambda row: (row["subtype"], int(row["answer_len"]), row["id"]))

    outputs = {
        "residual_rows_csv": args.output_dir / "v375_equation_residual_rows.csv",
        "cluster_summary_csv": args.output_dir / "v375_equation_residual_cluster_summary.csv",
        "priority_rows_csv": args.output_dir / "v375_equation_residual_priority_rows.csv",
        "manifest_json": args.output_dir / "v375_equation_residual_clustering_manifest.json",
    }
    write_csv(outputs["residual_rows_csv"], residuals, RESIDUAL_COLUMNS)
    write_csv(outputs["cluster_summary_csv"], clusters, CLUSTER_COLUMNS)
    write_csv(outputs["priority_rows_csv"], priority, RESIDUAL_COLUMNS)

    subtype_counts = Counter(str(row["subtype"]) for row in residuals)
    priority_counts = Counter(str(row["priority_reason"]) for row in residuals)
    decision = {
        "decision": "diagnostic_only_no_hf",
        "reason": (
            f"residual_rows={len(residuals)}; clusters={len(clusters)}; "
            f"priority_rows={len(priority)}; no rule was promoted"
        ),
        "next_action": "Inspect top clusters manually or implement V376 only for a concrete low-ambiguity class.",
    }
    manifest = {
        "schema_version": "kg1_v375_equation_residual_clustering_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v366_predictions_csv": str(args.v366_predictions_csv),
            "v366_predictions_sha256": sha256_file(args.v366_predictions_csv),
            "v324_audit_csv": str(args.v324_audit_csv),
            "v324_audit_sha256": sha256_file(args.v324_audit_csv),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "residual_rows": len(residuals),
        "cluster_count": len(clusters),
        "priority_rows": len(priority),
        "subtype_counts": dict(subtype_counts),
        "priority_reason_counts": dict(priority_counts),
        "top_clusters": clusters[:20],
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("residual_rows =", len(residuals), flush=True)
    print("cluster_count =", len(clusters), flush=True)
    print("priority_rows =", len(priority), flush=True)
    print("subtype_counts =", json.dumps(dict(subtype_counts), sort_keys=True), flush=True)
    print("priority_reason_counts =", json.dumps(dict(priority_counts), sort_keys=True), flush=True)
    print("top_clusters =", json.dumps(clusters[:10], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V375 EQUATION RESIDUAL CLUSTERING END ===", flush=True)
    return manifest


def self_test() -> None:
    if not is_subsequence("abc", "axbyc"):
        raise AssertionError("subsequence failed")
    if is_subsequence("acb", "axbyc"):
        raise AssertionError("negative subsequence failed")
    if edit_distance("abc", "adc") != 1:
        raise AssertionError("edit distance failed")
    if symbols("ab+c*") != "*+":
        raise AssertionError("symbols failed")
    print("v375_equation_residual_clustering_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v366-predictions-csv", type=Path, default=DEFAULT_V366_CSV)
    parser.add_argument("--v324-audit-csv", type=Path, default=DEFAULT_V324_AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v375_equation_residual_clustering" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--self-test", action="store_true")
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

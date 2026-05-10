#!/usr/bin/env python3
"""Audit V238/V239 abstains with stricter symbolic and numeric rule candidates.

This is CPU-only and diagnostic-only. It does not train, run model generation,
score a model, package artifacts, download external payloads, or submit to
Kaggle. Candidate verification uses weak labels only as an audit signal; a rule
is deployable only when it is derived from prompt examples without ambiguity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from analyze_v238_alice_parser_probes import (
    EXPECTED_ROW_CONTRACT_SHA256,
    answers_equal,
    parse_alice_prompt,
    parse_numeric_token,
)


SYMBOLIC_COLUMNS = [
    "schema_version",
    "id",
    "query",
    "expected_answer",
    "baseline_prediction",
    "status",
    "prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "deployable_candidate",
    "proof",
    "mapping_count",
    "usable_mapping_count",
    "unique_prediction_count",
    "example_count",
]

NUMERIC_COLUMNS = [
    "schema_version",
    "id",
    "query",
    "query_op",
    "expected_answer",
    "baseline_prediction",
    "status",
    "prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "deployable_candidate",
    "proof",
    "same_operator_example_count",
    "candidate_rule_count",
    "unique_prediction_count",
    "candidate_rules",
    "example_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(int(value)):02d}"[-2:]
    return int(text[0]), int(text[1])


def reverse_int(value: int) -> int:
    return int(f"{abs(int(value)):02d}"[::-1])


def digit_sum(value: int) -> int:
    return sum(int(ch) for ch in str(abs(int(value))))


def numeric_rule_functions() -> dict[str, Callable[[int, int], str]]:
    return {
        "add": lambda a, b: str(a + b),
        "sub_ab": lambda a, b: str(a - b),
        "sub_ba": lambda a, b: str(b - a),
        "abs_diff": lambda a, b: str(abs(a - b)),
        "mul": lambda a, b: str(a * b),
        "concat_ab": lambda a, b: f"{abs(a)}{abs(b)}",
        "concat_ba": lambda a, b: f"{abs(b)}{abs(a)}",
        "sum_digits_all": lambda a, b: str(digit_sum(a) + digit_sum(b)),
        "rev_add": lambda a, b: str(reverse_int(a) + reverse_int(b)),
        "rev_sub_ab": lambda a, b: str(reverse_int(a) - reverse_int(b)),
        "rev_sub_ba": lambda a, b: str(reverse_int(b) - reverse_int(a)),
        "rev_abs_diff": lambda a, b: str(abs(reverse_int(a) - reverse_int(b))),
        "rev_mul": lambda a, b: str(reverse_int(a) * reverse_int(b)),
        "a_plus_b_plus1": lambda a, b: str(a + b + 1),
        "a_plus_b_minus1": lambda a, b: str(a + b - 1),
        "a_minus_b_plus1": lambda a, b: str(a - b + 1),
        "a_minus_b_minus1": lambda a, b: str(a - b - 1),
        "b_minus_a_plus1": lambda a, b: str(b - a + 1),
        "b_minus_a_minus1": lambda a, b: str(b - a - 1),
        "digit_absdiff_concat": lambda a, b: "".join(str(abs(x - y)) for x, y in zip(digits2(a), digits2(b))),
        "digit_add_mod10_concat": lambda a, b: "".join(str((x + y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ab_mod10_concat": lambda a, b: "".join(str((x - y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ba_mod10_concat": lambda a, b: "".join(str((y - x) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_mul_mod10_concat": lambda a, b: "".join(str((x * y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "tens_add_ones_add": lambda a, b: str((digits2(a)[0] + digits2(b)[0]) * 10 + digits2(a)[1] + digits2(b)[1]),
        "tens_absdiff_ones_absdiff_int": lambda a, b: str(
            abs(digits2(a)[0] - digits2(b)[0]) * 10 + abs(digits2(a)[1] - digits2(b)[1])
        ),
    }


def transducer_mappings_for_pair(lhs: str, rhs: str, cap: int) -> list[dict[str, str]]:
    mappings: list[dict[str, str]] = []

    def walk(i: int, j: int, mapping: dict[str, str]) -> None:
        if len(mappings) >= cap:
            return
        if i == len(lhs):
            if j == len(rhs):
                mappings.append(dict(mapping))
            return
        char = lhs[i]
        if char in mapping:
            value = mapping[char]
            if value == "":
                walk(i + 1, j, mapping)
            elif j < len(rhs) and rhs[j] == value:
                walk(i + 1, j + 1, mapping)
            return
        mapping[char] = ""
        walk(i + 1, j, mapping)
        del mapping[char]
        if j < len(rhs):
            mapping[char] = rhs[j]
            walk(i + 1, j + 1, mapping)
            del mapping[char]

    walk(0, 0, {})
    return mappings


def merge_mapping_sets(
    existing: list[dict[str, str]],
    pair_mappings: list[dict[str, str]],
    cap: int,
) -> list[dict[str, str]]:
    merged_rows: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for base in existing:
        for pair_mapping in pair_mappings:
            merged = dict(base)
            ok = True
            for key, value in pair_mapping.items():
                if key in merged and merged[key] != value:
                    ok = False
                    break
                merged[key] = value
            if not ok:
                continue
            marker = tuple(sorted(merged.items()))
            if marker in seen:
                continue
            seen.add(marker)
            merged_rows.append(merged)
            if len(merged_rows) >= cap:
                return merged_rows
    return merged_rows


def infer_symbolic_transducer(
    examples: list[tuple[str, str]],
    query: str,
    pair_cap: int,
    global_cap: int,
) -> dict[str, Any]:
    mappings: list[dict[str, str]] = [{}]
    for lhs, rhs in examples:
        pair_mappings = transducer_mappings_for_pair(lhs, rhs, pair_cap)
        if not pair_mappings:
            return {
                "status": "abstain",
                "prediction": "",
                "proof": "no_pair_mapping",
                "mapping_count": 0,
                "usable_mapping_count": 0,
                "unique_prediction_count": 0,
            }
        mappings = merge_mapping_sets(mappings, pair_mappings, global_cap)
        if not mappings:
            return {
                "status": "abstain",
                "prediction": "",
                "proof": "no_global_mapping",
                "mapping_count": 0,
                "usable_mapping_count": 0,
                "unique_prediction_count": 0,
            }
    predictions: list[str] = []
    usable = 0
    for mapping in mappings:
        if all(char in mapping for char in query):
            usable += 1
            predictions.append("".join(mapping[char] for char in query))
    unique_predictions = sorted(set(predictions))
    if len(unique_predictions) == 1:
        return {
            "status": "candidate",
            "prediction": unique_predictions[0],
            "proof": f"char_transducer mappings={len(mappings)} usable={usable}",
            "mapping_count": len(mappings),
            "usable_mapping_count": usable,
            "unique_prediction_count": 1,
        }
    return {
        "status": "abstain",
        "prediction": "",
        "proof": f"char_transducer mappings={len(mappings)} usable={usable} unique_predictions={len(unique_predictions)}",
        "mapping_count": len(mappings),
        "usable_mapping_count": usable,
        "unique_prediction_count": len(unique_predictions),
    }


def audit_symbolic_row(
    result: dict[str, str],
    item: dict[str, Any],
    pair_cap: int,
    global_cap: int,
) -> dict[str, Any]:
    examples, query, parse_status = parse_alice_prompt(str(item.get("prompt", "")))
    if parse_status != "ok":
        query = str(result.get("query", ""))
    audit = infer_symbolic_transducer(examples, query, pair_cap=pair_cap, global_cap=global_cap)
    verified = audit["status"] == "candidate" and answers_equal(audit["prediction"], result.get("expected_answer", ""))
    incorrect = audit["status"] == "candidate" and not verified
    deployable = audit["status"] == "candidate" and not incorrect
    return {
        "schema_version": "kg1_v241_symbolic_rule_candidate_audit_v1",
        "id": result.get("id", ""),
        "query": query,
        "expected_answer": result.get("expected_answer", ""),
        "baseline_prediction": result.get("baseline_prediction", ""),
        "status": audit["status"],
        "prediction": audit["prediction"],
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "deployable_candidate": deployable,
        "proof": audit["proof"],
        "mapping_count": audit["mapping_count"],
        "usable_mapping_count": audit["usable_mapping_count"],
        "unique_prediction_count": audit["unique_prediction_count"],
        "example_count": len(examples),
    }


def audit_numeric_row(result: dict[str, str], item: dict[str, Any], min_same_op_examples: int) -> dict[str, Any]:
    examples, query, parse_status = parse_alice_prompt(str(item.get("prompt", "")))
    if parse_status != "ok":
        query = str(result.get("query", ""))
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return {
            "schema_version": "kg1_v241_numeric_rule_candidate_audit_v1",
            "id": result.get("id", ""),
            "query": query,
            "query_op": "",
            "expected_answer": result.get("expected_answer", ""),
            "baseline_prediction": result.get("baseline_prediction", ""),
            "status": "abstain",
            "prediction": "",
            "verified_by_weak_label": False,
            "incorrect_by_weak_label": False,
            "deployable_candidate": False,
            "proof": "query_not_numeric_binary",
            "same_operator_example_count": 0,
            "candidate_rule_count": 0,
            "unique_prediction_count": 0,
            "candidate_rules": "",
            "example_count": len(examples),
        }
    same_operator: list[tuple[int, int, str]] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if parsed and parsed[1] == parsed_query[1]:
            same_operator.append((parsed[0], parsed[2], str(rhs)))
    candidates: list[tuple[str, str]] = []
    for name, func in numeric_rule_functions().items():
        ok = True
        for left, right, expected in same_operator:
            try:
                prediction = func(left, right)
            except Exception:
                ok = False
                break
            if prediction != expected:
                ok = False
                break
        if ok and same_operator:
            try:
                query_prediction = func(parsed_query[0], parsed_query[2])
            except Exception:
                continue
            candidates.append((name, query_prediction))
    unique_predictions = sorted(set(prediction for _, prediction in candidates))
    if not same_operator:
        status = "abstain"
        prediction = ""
        proof = "no_same_operator_examples"
    elif len(unique_predictions) != 1:
        status = "abstain"
        prediction = ""
        proof = f"candidate_rule_count={len(candidates)} unique_prediction_count={len(unique_predictions)}"
    elif len(same_operator) < min_same_op_examples:
        status = "under_evidenced_candidate"
        prediction = unique_predictions[0]
        proof = f"same_operator_examples={len(same_operator)} below_min={min_same_op_examples}"
    else:
        status = "candidate"
        prediction = unique_predictions[0]
        proof = "rules=" + ",".join(name for name, _ in candidates)
    verified = status == "candidate" and answers_equal(prediction, result.get("expected_answer", ""))
    incorrect = status == "candidate" and not verified
    return {
        "schema_version": "kg1_v241_numeric_rule_candidate_audit_v1",
        "id": result.get("id", ""),
        "query": query,
        "query_op": parsed_query[1],
        "expected_answer": result.get("expected_answer", ""),
        "baseline_prediction": result.get("baseline_prediction", ""),
        "status": status,
        "prediction": prediction,
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "deployable_candidate": status == "candidate" and not incorrect,
        "proof": proof,
        "same_operator_example_count": len(same_operator),
        "candidate_rule_count": len(candidates),
        "unique_prediction_count": len(unique_predictions),
        "candidate_rules": ",".join(name for name, _ in candidates),
        "example_count": len(examples),
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter(tuple(str(row.get(key, "")) for key in keys) for row in rows)
    return [
        {**{key: values[index] for index, key in enumerate(keys)}, "rows": count}
        for values, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V241 ABSTAIN RULE CANDIDATE AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v232_equation_workitems_jsonl =", args.v232_equation_workitems_jsonl, flush=True)
    print("v238_results_csv =", args.v238_results_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("min_same_operator_examples =", args.min_same_operator_examples, flush=True)
    print("pair_mapping_cap =", args.pair_mapping_cap, flush=True)
    print("global_mapping_cap =", args.global_mapping_cap, flush=True)

    equation_items = read_jsonl(args.v232_equation_workitems_jsonl)
    results = read_csv(args.v238_results_csv)
    item_by_id = {str(item.get("id", "")): item for item in equation_items}
    missing_items = sorted(row.get("id", "") for row in results if row.get("id", "") not in item_by_id)
    if missing_items:
        raise RuntimeError("V241 results reference missing equation workitems: " + json.dumps(missing_items[:20]))

    abstains = [row for row in results if row.get("status") == "abstain"]
    symbolic_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    for row in abstains:
        item = item_by_id[str(row.get("id", ""))]
        if row.get("prompt_kind") == "alice_symbolic_token_transform":
            symbolic_rows.append(audit_symbolic_row(row, item, args.pair_mapping_cap, args.global_mapping_cap))
        elif row.get("prompt_kind") == "alice_numeric_binary_operator":
            numeric_rows.append(audit_numeric_row(row, item, args.min_same_operator_examples))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    outputs = {
        "symbolic_rule_candidate_audit_csv": args.output_dir / f"{prefix}_symbolic_rule_candidate_audit.csv",
        "numeric_rule_candidate_audit_csv": args.output_dir / f"{prefix}_numeric_rule_candidate_audit.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_csv(outputs["symbolic_rule_candidate_audit_csv"], symbolic_rows, SYMBOLIC_COLUMNS)
    write_csv(outputs["numeric_rule_candidate_audit_csv"], numeric_rows, NUMERIC_COLUMNS)

    all_rows = [*symbolic_rows, *numeric_rows]
    deployable_verified = [
        row for row in all_rows if truthy(row.get("deployable_candidate")) and truthy(row.get("verified_by_weak_label"))
    ]
    deployable_incorrect = [
        row for row in all_rows if truthy(row.get("deployable_candidate")) and truthy(row.get("incorrect_by_weak_label"))
    ]
    under_evidenced = [row for row in all_rows if row.get("status") == "under_evidenced_candidate"]
    if deployable_verified and not deployable_incorrect:
        decision = {
            "decision": "prepare_parser_probe_with_v241_candidates",
            "reason": f"deployable_verified={len(deployable_verified)}; deployable_incorrect=0",
            "next_action": "Promote only deployable V241 candidates into a new parser probe with negative fixtures.",
        }
    else:
        decision = {
            "decision": "do_not_promote_v241_candidates",
            "reason": (
                f"deployable_verified={len(deployable_verified)}; "
                f"deployable_incorrect={len(deployable_incorrect)}; "
                f"under_evidenced={len(under_evidenced)}"
            ),
            "next_action": "Keep current Alice parser unchanged; use V239/V241 workpacks to design new examples or stricter DSL.",
        }

    manifest = {
        "schema_version": "kg1_v241_abstain_rule_candidate_audit_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
            "v232_equation_workitems_jsonl": str(args.v232_equation_workitems_jsonl),
            "v238_results_csv": str(args.v238_results_csv),
        },
        "input_artifact_hashes": {
            "v232_equation_workitems_jsonl": file_meta(args.v232_equation_workitems_jsonl),
            "v238_results_csv": file_meta(args.v238_results_csv),
        },
        "counts": {
            "v238_rows": len(results),
            "abstain_rows": len(abstains),
            "symbolic_rows": len(symbolic_rows),
            "numeric_rows": len(numeric_rows),
            "deployable_verified_candidates": len(deployable_verified),
            "deployable_incorrect_candidates": len(deployable_incorrect),
            "under_evidenced_candidates": len(under_evidenced),
        },
        "symbolic_status_summary": summarize(symbolic_rows, ["status", "proof"]),
        "numeric_status_summary": summarize(numeric_rows, ["status", "proof"]),
        "deployable_verified_preview": deployable_verified[:12],
        "deployable_incorrect_preview": deployable_incorrect[:12],
        "under_evidenced_preview": under_evidenced[:12],
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in outputs.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(outputs["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("symbolic_status_summary =", json.dumps(manifest["symbolic_status_summary"][:20], indent=2, sort_keys=True), flush=True)
    print("numeric_status_summary =", json.dumps(manifest["numeric_status_summary"][:20], indent=2, sort_keys=True), flush=True)
    print("deployable_verified_preview =", json.dumps(deployable_verified[:12], indent=2, sort_keys=True), flush=True)
    print("deployable_incorrect_preview =", json.dumps(deployable_incorrect[:12], indent=2, sort_keys=True), flush=True)
    print("under_evidenced_preview =", json.dumps(under_evidenced[:12], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V241 ABSTAIN RULE CANDIDATE AUDIT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-equation-workitems-jsonl", type=Path)
    parser.add_argument("--v238-results-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v241_abstain_rule_candidate_audit")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--pair-mapping-cap", type=int, default=2000)
    parser.add_argument("--global-mapping-cap", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workitems = root / "equation.jsonl"
        results = root / "results.csv"
        rows = [
            {
                "id": "sym_map",
                "expected_answer": "xy",
                "baseline_prediction": "ab",
                "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: ab = xy ac = xz Now, determine the result for: ab",
            },
            {
                "id": "num_add",
                "expected_answer": "11",
                "baseline_prediction": "0",
                "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 03#04 = 7 02#08 = 10 Now, determine the result for: 05#06",
            },
        ]
        with workitems.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "status", "prompt_kind", "query", "expected_answer", "baseline_prediction", "proof"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": "sym_map",
                    "status": "abstain",
                    "prompt_kind": "alice_symbolic_token_transform",
                    "query": "ab",
                    "expected_answer": "xy",
                    "baseline_prediction": "ab",
                    "proof": "selftest",
                }
            )
            writer.writerow(
                {
                    "id": "num_add",
                    "status": "abstain",
                    "prompt_kind": "alice_numeric_binary_operator",
                    "query": "05#06",
                    "expected_answer": "11",
                    "baseline_prediction": "0",
                    "proof": "selftest",
                }
            )
        args = argparse.Namespace(
            v232_equation_workitems_jsonl=workitems,
            v238_results_csv=results,
            output_dir=root / "out",
            label="v241_abstain_rule_candidate_audit",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            min_same_operator_examples=2,
            pair_mapping_cap=2000,
            global_mapping_cap=5000,
        )
        manifest = run_analysis(args)
        if manifest["counts"]["deployable_verified_candidates"] != 2:
            raise AssertionError("expected two verified self-test candidates")
        if manifest["counts"]["deployable_incorrect_candidates"] != 0:
            raise AssertionError("expected zero incorrect self-test candidates")
    print("v241_abstain_rule_candidate_audit_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.v232_equation_workitems_jsonl is None:
        parser.error("--v232-equation-workitems-jsonl is required unless --self-test is used")
    if args.v238_results_csv is None:
        parser.error("--v238-results-csv is required unless --self-test is used")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

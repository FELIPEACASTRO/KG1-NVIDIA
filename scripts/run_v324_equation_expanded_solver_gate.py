#!/usr/bin/env python3
"""CPU gate for expanded equation_transform solver candidates.

V324 is intentionally CPU-only and label-audited. It reads the current best
weak prediction CSV, validates the V221 shared row contract, and compares the
99 equation_transform baseline misses against:

* the existing V278 symbolic PBE DSL;
* the existing V299/V274 numeric DSL;
* the guarded V274 numeric postprocessor;
* one additional variable-operator symbolic template family.

Weak labels are used only as a brake. Nothing in this script trains, runs GPU
inference, packages, or submits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from kg1_v274_numeric_postprocessor import postprocess_numeric_prediction  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    AUDIT_COLUMNS as V278_AUDIT_COLUMNS,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_BASELINE_REPO,
    EXPECTED_ROW_CONTRACT_SHA256,
    build_audit_row,
    classify_subtype,
    download_file,
    normalize_row,
    numeric_candidate,
    parse_alice_prompt,
    row_contract,
    sha256_file,
    symbolic_candidates,
)
from run_v299_equation_numeric_candidate_audit import audit_row as audit_v299_numeric_row  # noqa: E402


EXTRA_AUDIT_COLUMNS = V278_AUDIT_COLUMNS + ["candidate_source"]
SUMMARY_COLUMNS = [
    "rule_class",
    "rows",
    "candidate_rows",
    "verified_candidates",
    "incorrect_candidates",
    "abstain_rows",
    "promotable_after_class_gate",
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


def family_counts(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        out[family]["rows"] += 1
        out[family]["correct"] += int(verify_answer(row["answer"], row.get(prediction_key, "")))
        out[family]["truncated"] += int(bool(row.get("truncated_bool", False)))
    return {family: dict(counter) for family, counter in sorted(out.items())}


def candidate_from_predictions(
    rule_class: str,
    predictions: list[str],
    proof: str,
    *,
    candidate_program_count: int,
) -> dict[str, Any]:
    unique = sorted({prediction for prediction in predictions if prediction != ""})
    if len(unique) == 1:
        return {
            "rule_class": rule_class,
            "status": "candidate",
            "prediction": unique[0],
            "proof": proof,
            "candidate_program_count": candidate_program_count,
            "unique_prediction_count": 1,
        }
    return {
        "rule_class": rule_class,
        "status": "abstain",
        "prediction": "",
        "proof": proof + f"; unique_prediction_count={len(unique)}",
        "candidate_program_count": candidate_program_count,
        "unique_prediction_count": len(unique),
    }


def is_operator_char(ch: str) -> bool:
    return bool(ch) and (not ch.isalnum())


def split_symbolic_token(token: str) -> list[tuple[str, str, str]]:
    text = str(token or "")
    splits: list[tuple[str, str, str]] = []
    for index in range(1, len(text) - 1):
        op = text[index]
        if is_operator_char(op):
            splits.append((text[:index], op, text[index + 1 :]))
    return splits


def apply_symbolic_transform(left: str, op: str, right: str, transform: str) -> str:
    joined = left + right
    if transform == "drop_operator":
        return joined
    if transform == "reverse_drop_operator":
        return joined[::-1]
    if transform == "left_only":
        return left
    if transform == "right_only":
        return right
    if transform == "right_left":
        return right + left
    if transform == "left_reverse_right":
        return left + right[::-1]
    if transform == "reverse_left_right":
        return left[::-1] + right
    if transform == "right_reverse_left":
        return right + left[::-1]
    if transform == "reverse_right_left":
        return right[::-1] + left
    if transform == "operator_between_reversed":
        return right + op + left
    raise KeyError(transform)


def symbolic_variable_operator_candidates(examples: list[tuple[str, str]], query: str) -> list[dict[str, Any]]:
    """Infer simple split-around-operator transforms without requiring fixed index.

    This catches symbolic cases where the operation character position varies
    across examples. It remains diagnostic unless the whole rule class has
    zero weak-label errors.
    """

    transforms = [
        "drop_operator",
        "reverse_drop_operator",
        "left_only",
        "right_only",
        "right_left",
        "left_reverse_right",
        "reverse_left_right",
        "right_reverse_left",
        "reverse_right_left",
        "operator_between_reversed",
    ]
    query_splits = split_symbolic_token(query)
    if not query_splits:
        return [
            {
                "rule_class": "symbolic_variable_operator_gate",
                "status": "abstain",
                "prediction": "",
                "proof": "query_has_no_symbolic_operator_split",
                "candidate_program_count": 0,
                "unique_prediction_count": 0,
            }
        ]

    results: list[dict[str, Any]] = []
    for transform in transforms:
        split_counts: list[int] = []
        ok = True
        for lhs, rhs in examples:
            matching = [
                (left, op, right)
                for left, op, right in split_symbolic_token(lhs)
                if apply_symbolic_transform(left, op, right, transform) == rhs
            ]
            split_counts.append(len(matching))
            if not matching:
                ok = False
                break
        if not ok:
            results.append(
                {
                    "rule_class": "symbolic_variable_operator_" + transform,
                    "status": "abstain",
                    "prediction": "",
                    "proof": "one_or_more_examples_have_no_matching_split; split_counts="
                    + ",".join(str(value) for value in split_counts),
                    "candidate_program_count": 0,
                    "unique_prediction_count": 0,
                }
            )
            continue
        predictions = [apply_symbolic_transform(left, op, right, transform) for left, op, right in query_splits]
        results.append(
            candidate_from_predictions(
                "symbolic_variable_operator_" + transform,
                predictions,
                "split_counts=" + ",".join(str(value) for value in split_counts),
                candidate_program_count=sum(split_counts),
            )
        )
    return results


def v274_guarded_candidate(row: dict[str, Any]) -> dict[str, Any]:
    decision = postprocess_numeric_prediction(
        str(row["prompt"]),
        str(row["prediction"]),
        family=str(row["family"]),
        truncated=bool(row.get("truncated_bool", False)),
    )
    if decision.applied:
        return {
            "rule_class": "v274_guarded_numeric_" + decision.rule,
            "status": "candidate",
            "prediction": decision.prediction,
            "proof": decision.proof,
            "candidate_program_count": 1,
            "unique_prediction_count": 1,
        }
    return {
        "rule_class": "v274_guarded_numeric_" + decision.rule,
        "status": "abstain",
        "prediction": "",
        "proof": decision.proof,
        "candidate_program_count": 0,
        "unique_prediction_count": 0,
    }


def v299_rows_as_v324(row: dict[str, Any], examples: list[tuple[str, str]], query: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for result in audit_v299_numeric_row(row):
        converted.append(
            build_audit_row(
                row,
                {
                    "rule_class": "v299_" + str(result.get("candidate_class", "")),
                    "status": result.get("status", "abstain"),
                    "prediction": result.get("prediction", ""),
                    "proof": result.get("proof", ""),
                    "candidate_program_count": result.get("candidate_rule_count", 0),
                    "unique_prediction_count": result.get("unique_prediction_count", 0),
                },
                examples,
                query,
            )
            | {"candidate_source": "v299_numeric_dsl"}
        )
    return converted


def summarize_rule_classes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rule_class"])].append(row)

    summaries: list[dict[str, Any]] = []
    for rule_class, items in sorted(grouped.items()):
        candidates = [row for row in items if row["status"] == "candidate"]
        verified = [row for row in candidates if row["verified_by_weak_label"]]
        incorrect = [row for row in candidates if row["incorrect_by_weak_label"]]
        promotable = bool(candidates) and bool(verified) and not incorrect
        for row in items:
            if row["status"] == "candidate":
                row["promotable_after_class_gate"] = promotable
        summaries.append(
            {
                "rule_class": rule_class,
                "rows": len(items),
                "candidate_rows": len(candidates),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "abstain_rows": len(items) - len(candidates),
                "promotable_after_class_gate": promotable,
            }
        )
    summaries.sort(
        key=lambda item: (
            int(item["verified_candidates"]),
            -int(item["incorrect_candidates"]),
            -int(item["candidate_rows"]),
            str(item["rule_class"]),
        ),
        reverse=True,
    )
    return summaries


def accepted_no_loss_candidates(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        row
        for row in audit_rows
        if row["status"] == "candidate" and row["verified_by_weak_label"] and row["promotable_after_class_gate"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["id"])].append(row)
    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for row_id, items in sorted(grouped.items()):
        predictions = sorted({str(row["prediction"]) for row in items})
        if len(predictions) == 1:
            row = dict(items[0])
            row["proof"] = "accepted_no_loss_classes=" + ";".join(sorted({str(item["rule_class"]) for item in items}))
            accepted.append(row)
        else:
            conflicts.append(
                {
                    "id": row_id,
                    "predictions": ";".join(predictions),
                    "rule_classes": ";".join(sorted({str(item["rule_class"]) for item in items})),
                }
            )
    return accepted, conflicts


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.input_csv:
        path = Path(args.input_csv)
        return [normalize_row(row) for row in read_csv(path)], {
            "source": "local_csv",
            "path": str(path),
            "sha256": sha256_file(path),
        }

    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v324_") as temp_name:
        path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(path)]
        meta = {
            "source": "hf_hub",
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": path.name,
            "sha256": sha256_file(path),
        }
    return rows, meta


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V324 EQUATION EXPANDED SOLVER GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("input_csv =", args.input_csv or "", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("target_equation_gain =", args.target_equation_gain, flush=True)
    print("bit_guardrail_min =", args.bit_guardrail_min, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, input_meta = load_rows(args)
    input_meta["rows"] = len(rows)

    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    baseline_family = family_counts(rows, "prediction")
    baseline_equation_correct = int(baseline_family.get("equation_transform", {}).get("correct", 0))
    baseline_bit_correct = int(baseline_family.get("bit_manipulation", {}).get("correct", 0))
    print("baseline_family_counts =", json.dumps(baseline_family, sort_keys=True), flush=True)

    equation_misses = [
        row
        for row in rows
        if row["family"] == "equation_transform" and not row["correct_bool"] and not row["truncated_bool"]
    ]
    print("equation_miss_rows =", len(equation_misses), flush=True)

    audit_rows: list[dict[str, Any]] = []
    parse_status_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    for index, row in enumerate(equation_misses, start=1):
        if index == 1 or index % 25 == 0 or index == len(equation_misses):
            print(f"equation_miss_audit_progress = {index}/{len(equation_misses)}", flush=True)
        examples, query, parse_status = parse_alice_prompt(str(row["prompt"]))
        parse_status_counts[parse_status] += 1
        if parse_status != "ok":
            audit_rows.append(
                build_audit_row(
                    row,
                    {
                        "rule_class": "alice_parse_gate",
                        "status": "abstain",
                        "prediction": "",
                        "proof": parse_status,
                    },
                    examples,
                    query,
                )
                | {"candidate_source": "parse_gate"}
            )
            continue

        subtype = classify_subtype(examples, query)
        subtype_counts[subtype] += 1

        v274_row = build_audit_row(row, v274_guarded_candidate(row), examples, query)
        v274_row["candidate_source"] = "v274_guarded_numeric_postprocessor"
        audit_rows.append(v274_row)

        if subtype == "equation_numeric_operator":
            v278_numeric = build_audit_row(
                row,
                numeric_candidate(examples, query, args.min_same_operator_examples),
                examples,
                query,
            )
            v278_numeric["candidate_source"] = "v278_numeric_same_operator"
            audit_rows.append(v278_numeric)
            audit_rows.extend(v299_rows_as_v324(row, examples, query))
        else:
            for result in symbolic_candidates(examples, query, args):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v278_symbolic_pbe"
                audit_rows.append(audit)
            for result in symbolic_variable_operator_candidates(examples, query):
                audit = build_audit_row(row, result, examples, query)
                audit["candidate_source"] = "v324_variable_operator_symbolic"
                audit_rows.append(audit)

    summary_rows = summarize_rule_classes(audit_rows)
    accepted, conflicts = accepted_no_loss_candidates(audit_rows)
    accepted_gain = len(accepted)
    projected_equation_correct = baseline_equation_correct + accepted_gain

    outputs = {
        "audit_csv": args.output_dir / "v324_equation_expanded_solver_audit.csv",
        "rule_summary_csv": args.output_dir / "v324_equation_expanded_solver_rule_summary.csv",
        "accepted_candidates_csv": args.output_dir / "v324_equation_expanded_solver_accepted_candidates.csv",
        "conflicts_csv": args.output_dir / "v324_equation_expanded_solver_conflicts.csv",
        "manifest_json": args.output_dir / "v324_equation_expanded_solver_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, EXTRA_AUDIT_COLUMNS)
    write_csv(outputs["rule_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["accepted_candidates_csv"], accepted, EXTRA_AUDIT_COLUMNS)
    write_csv(outputs["conflicts_csv"], conflicts, ["id", "predictions", "rule_classes"])

    if accepted_gain >= args.target_equation_gain and not conflicts and baseline_bit_correct >= args.bit_guardrail_min:
        decision = {
            "decision": "equation_cpu_gate_found_distillation_signal",
            "reason": (
                f"accepted_equation_gain={accepted_gain}; projected_equation_correct={projected_equation_correct}; "
                f"bit_correct_guardrail={baseline_bit_correct}>={args.bit_guardrail_min}; conflicts={len(conflicts)}"
            ),
            "next_action": "Convert accepted no-loss rows into short deterministic hard-negative traces; run a no-GPU dataset gate before HF.",
        }
    else:
        decision = {
            "decision": "no_new_equation_signal_for_hf_gpu",
            "reason": (
                f"accepted_equation_gain={accepted_gain}; target={args.target_equation_gain}; "
                f"projected_equation_correct={projected_equation_correct}; bit_correct={baseline_bit_correct}; conflicts={len(conflicts)}"
            ),
            "next_action": "Do not launch another HF GPU run from this route; expand CPU DSL or inspect locked source traces.",
        }

    manifest = {
        "schema_version": "kg1_v324_equation_expanded_solver_gate_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "baseline_family_counts": baseline_family,
        "equation_miss_rows": len(equation_misses),
        "parse_status_counts": dict(parse_status_counts),
        "subtype_counts": dict(subtype_counts),
        "rule_summary": summary_rows,
        "accepted_candidate_count": accepted_gain,
        "accepted_candidate_ids": [str(row["id"]) for row in accepted],
        "projected_equation_correct": projected_equation_correct,
        "projected_weak_correct_if_equation_only": sum(1 for row in rows if row["correct_bool"]) + accepted_gain,
        "conflict_count": len(conflicts),
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("parse_status_counts =", json.dumps(dict(parse_status_counts), sort_keys=True), flush=True)
    print("subtype_counts =", json.dumps(dict(subtype_counts), sort_keys=True), flush=True)
    print("rule_summary_top =", json.dumps(summary_rows[:12], indent=2, sort_keys=True), flush=True)
    print("accepted_candidate_count =", accepted_gain, flush=True)
    print("accepted_candidate_ids =", json.dumps(manifest["accepted_candidate_ids"], sort_keys=True), flush=True)
    print("projected_equation_correct =", projected_equation_correct, flush=True)
    print("projected_weak_correct_if_equation_only =", manifest["projected_weak_correct_if_equation_only"], flush=True)
    print("conflict_count =", len(conflicts), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V324 EQUATION EXPANDED SOLVER GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    examples = [("ab+c", "abc"), ("w*xy", "wxy"), ("12:3", "123")]
    results = symbolic_variable_operator_candidates(examples, "pq-r")
    by_class = {row["rule_class"]: row for row in results}
    result = by_class["symbolic_variable_operator_drop_operator"]
    if result["status"] != "candidate" or result["prediction"] != "pqr":
        raise AssertionError(result)
    reverse_examples = [("ab+c", "cba"), ("w*xy", "yxw")]
    reverse = {
        row["rule_class"]: row
        for row in symbolic_variable_operator_candidates(reverse_examples, "pq-r")
    }["symbolic_variable_operator_reverse_drop_operator"]
    if reverse["status"] != "candidate" or reverse["prediction"] != "rqp":
        raise AssertionError(reverse)
    print("v324_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path(f"artifacts/v324_equation_expanded_solver_gate/{utc_compact()}"))
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-equation-gain", type=int, default=4)
    parser.add_argument("--bit-guardrail-min", type=int, default=136)
    parser.add_argument("--pair-mapping-cap", type=int, default=3000)
    parser.add_argument("--global-mapping-cap", type=int, default=12000)
    parser.add_argument("--max-char-subset-size", type=int, default=4)
    parser.add_argument("--max-position-sources", type=int, default=7)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
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

#!/usr/bin/env python3
"""V364 CPU-only symbolic pair-table gate.

V363 showed that 70 remaining equation_transform misses have the query
operator present in same-position examples, but the existing same-op DSL could
not produce a unique no-loss candidate. V364 tests one new symbolic family:
per-output-character lookup tables keyed by operand positions around the
operator.

This is still CPU-only. It does not train, run GPU inference, package, or
submit. Weak labels are used only as a brake; a rule class is promotable only
with gains and zero losses.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "artifacts/v363_equation_residual_operator_support/20260514T_cpu_gate/v363_integrated_predictions.csv"
)

DECISION_COLUMNS = [
    "id",
    "family",
    "candidate_source",
    "rule_class",
    "old_prediction",
    "new_prediction",
    "answer",
    "old_correct",
    "new_correct",
    "accepted",
    "rejection_reason",
    "candidate_count",
    "conflict_count",
    "proof",
]

RULE_COLUMNS = [
    "family",
    "candidate_source",
    "rule_class",
    "candidate_rows",
    "changed_rows",
    "gains",
    "losses",
    "new_correct",
    "accepted_rows",
    "accepted",
    "reason",
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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def current_prediction(row: dict[str, Any]) -> str:
    for key in ("v364_prediction", "v363_prediction", "v355_prediction", "v350_prediction", "prediction", "baseline_prediction"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def normalize_prediction_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        base = normalize_row(row)
        base.update(row)
        base["current_prediction"] = current_prediction(row)
        base["current_correct"] = verify_answer(str(base.get("answer", "")), str(base["current_prediction"]))
        normalized.append(base)
    return normalized


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        fam = str(row["family"])
        ok = verify_answer(str(row["answer"]), str(row.get(prediction_key, "")))
        total["rows"] += 1
        total["correct"] += int(ok)
        family[fam]["rows"] += 1
        family[fam]["correct"] += int(ok)
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "accuracy": float(total["correct"] / total["rows"]) if total["rows"] else 0.0,
        "family": {key: dict(value) for key, value in sorted(family.items())},
    }


def source_specs() -> list[tuple[str, Callable[[str], tuple[str, ...]]]]:
    positions: list[tuple[str, Callable[[str], str]]] = [
        ("L0", lambda text: text[0]),
        ("L1", lambda text: text[1]),
        ("R0", lambda text: text[3]),
        ("R1", lambda text: text[4]),
    ]
    specs: list[tuple[str, Callable[[str], tuple[str, ...]]]] = []
    for name, func in positions:
        specs.append((name, lambda text, func=func: (func(text),)))
    for left_name, left_func in positions:
        for right_name, right_func in positions:
            specs.append(
                (
                    left_name + right_name,
                    lambda text, left_func=left_func, right_func=right_func: (left_func(text), right_func(text)),
                )
            )
    return specs


def pair_table_predictions(
    examples: list[tuple[str, str]],
    query: str,
    *,
    max_output_len: int,
    max_program_count: int,
) -> tuple[list[str], str, int]:
    if not examples or any(len(lhs) != 5 for lhs, _rhs in examples) or len(query) != 5:
        return [], "unsupported_token_length", 0
    rhs_lengths = {len(rhs) for _lhs, rhs in examples}
    if len(rhs_lengths) != 1:
        return [], "nonuniform_same_operator_rhs_lengths=" + ",".join(str(value) for value in sorted(rhs_lengths)), 0
    output_len = next(iter(rhs_lengths))
    if output_len <= 0 or output_len > max_output_len:
        return [], f"unsupported_output_len={output_len}", 0

    specs = source_specs()
    choices: list[list[tuple[str, str]]] = []
    proof_bits: list[str] = []
    for out_index in range(output_len):
        out_choices: list[tuple[str, str]] = []
        for name, func in specs:
            table: dict[tuple[str, ...], str] = {}
            ok = True
            for lhs, rhs in examples:
                key = func(lhs)
                value = rhs[out_index]
                if key in table and table[key] != value:
                    ok = False
                    break
                table[key] = value
            if not ok:
                continue
            query_key = func(query)
            if query_key not in table:
                continue
            out_choices.append((name, table[query_key]))
        if not out_choices:
            return [], f"no_source_for_output_index={out_index}", 0
        choices.append(out_choices)
        proof_bits.append(f"{out_index}:{','.join(name for name, _value in out_choices[:12])}")

    program_count = 1
    for choice in choices:
        program_count *= len(choice)
    if program_count > max_program_count:
        return [], f"program_count_above_cap={program_count}", program_count

    predictions = sorted({"".join(value for _name, value in combo) for combo in itertools.product(*choices)})
    return predictions, "choice_sources=" + ";".join(proof_bits), program_count


def pair_table_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("family")) != "equation_transform":
            continue
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        if parse_status != "ok" or classify_subtype(examples, query) != "equation_symbolic_punct":
            continue
        if len(query) != 5:
            continue
        query_op = query[2]
        same_examples = [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[2] == query_op]
        if len(same_examples) < args.min_same_operator_examples:
            continue
        predictions, proof, program_count = pair_table_predictions(
            same_examples,
            query,
            max_output_len=args.max_output_len,
            max_program_count=args.max_program_count,
        )
        if len(predictions) != 1:
            continue
        prediction = predictions[0]
        old_prediction = current_prediction(row)
        if not prediction or prediction == old_prediction:
            continue
        decisions.append(
            {
                "id": row["id"],
                "family": row["family"],
                "candidate_source": "v364_symbolic_pair_table",
                "rule_class": f"symbolic_pair_table_len_{len(prediction)}",
                "old_prediction": old_prediction,
                "new_prediction": prediction,
                "answer": row["answer"],
                "old_correct": verify_answer(str(row["answer"]), old_prediction),
                "new_correct": verify_answer(str(row["answer"]), prediction),
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": program_count,
                "conflict_count": 0,
                "proof": f"same_operator_examples={len(same_examples)}; {proof}",
            }
        )
    return decisions


def apply_rule_gates(decisions: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)
    for _key, items in grouped.items():
        gains = [row for row in items if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
        losses = [row for row in items if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
        can_accept = bool(gains) and not losses
        for row in items:
            if can_accept and row in gains:
                row["accepted"] = True
                row["rejection_reason"] = ""
            elif losses:
                row["rejection_reason"] = "reject_rule_class_has_losses"
            elif not gains:
                row["rejection_reason"] = "reject_rule_class_no_gain"
            else:
                row["rejection_reason"] = "reject_not_promotable"


def rule_summary(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["family"]), str(row["candidate_source"]), str(row["rule_class"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (family, source, rule_class), items in sorted(grouped.items()):
        changed = [row for row in items if row.get("new_prediction") != row.get("old_prediction")]
        gains = [row for row in changed if truthy(row.get("new_correct")) and not truthy(row.get("old_correct"))]
        losses = [row for row in changed if truthy(row.get("old_correct")) and not truthy(row.get("new_correct"))]
        accepted = [row for row in items if truthy(row.get("accepted"))]
        rows.append(
            {
                "family": family,
                "candidate_source": source,
                "rule_class": rule_class,
                "candidate_rows": len(items),
                "changed_rows": len(changed),
                "gains": len(gains),
                "losses": len(losses),
                "new_correct": sum(1 for row in items if truthy(row.get("new_correct"))),
                "accepted_rows": len(accepted),
                "accepted": bool(accepted),
                "reason": "accepted_no_loss" if accepted else "rejected_by_gate",
            }
        )
    rows.sort(key=lambda item: (int(item["accepted_rows"]), int(item["gains"]), -int(item["losses"])), reverse=True)
    return rows


def integrate(rows: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_id[str(row["id"])].append(row)
    integrated = []
    for row in rows:
        item = dict(row)
        item["v364_prediction"] = current_prediction(row)
        item["v364_source_rule"] = ""
        accepted_items = accepted_by_id.get(str(row["id"]), [])
        if len(accepted_items) == 1:
            item["v364_prediction"] = str(accepted_items[0]["new_prediction"])
            item["v364_source_rule"] = str(accepted_items[0]["rule_class"])
        elif len(accepted_items) > 1:
            raise RuntimeError(f"accepted conflict id={row['id']}")
        item["v364_correct"] = verify_answer(str(item["answer"]), str(item["v364_prediction"]))
        integrated.append(item)
    return integrated


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V364 SYMBOLIC PAIR TABLE GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("input_predictions_csv =", args.input_predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("min_same_operator_examples =", args.min_same_operator_examples, flush=True)
    print("max_output_len =", args.max_output_len, flush=True)
    print("max_program_count =", args.max_program_count, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_prediction_rows(read_csv(args.input_predictions_csv))
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    for row in rows:
        row["current_prediction_for_summary"] = current_prediction(row)
    baseline = summarize(rows, "current_prediction_for_summary")
    print("current_summary =", json.dumps(baseline, sort_keys=True), flush=True)

    decisions = pair_table_decisions(rows, args)
    apply_rule_gates(decisions)
    accepted = [row for row in decisions if truthy(row.get("accepted"))]
    rules = rule_summary(decisions)
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("accepted_candidate_count =", len(accepted), flush=True)
    print("candidate_rule_summary_top =", json.dumps(rules[:10], sort_keys=True), flush=True)

    integrated = integrate(rows, accepted)
    after = summarize(integrated, "v364_prediction")
    gains = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row["v364_prediction"]))
        and not verify_answer(str(row["answer"]), current_prediction(row))
    ]
    losses = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), current_prediction(row))
        and not verify_answer(str(row["answer"]), str(row["v364_prediction"]))
    ]
    eq_before = int(baseline["family"].get("equation_transform", {}).get("correct", 0))
    bit_before = int(baseline["family"].get("bit_manipulation", {}).get("correct", 0))
    eq_after = int(after["family"].get("equation_transform", {}).get("correct", 0))
    bit_after = int(after["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = bool(accepted) and not losses and (eq_after > eq_before or bit_after > bit_before)
    decision = {
        "decision": "v364_cpu_gate_passed" if gate_pass else "v364_cpu_gate_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            f"v364={after['correct']}/315; equation={eq_after}/155; bit={bit_after}/160; "
            f"gains={len(gains)}; losses={len(losses)}; accepted={len(accepted)}"
        ),
        "next_action": (
            "Build a transfer dataset only after tokenization/no-regression gates."
            if gate_pass
            else "Do not launch HF. Pair-table symbolic candidates caused losses or no gains."
        ),
    }
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "candidate_decisions_csv": args.output_dir / "v364_candidate_decisions.csv",
        "candidate_rules_csv": args.output_dir / "v364_candidate_rules.csv",
        "integrated_predictions_csv": args.output_dir / "v364_integrated_predictions.csv",
        "manifest_json": args.output_dir / "v364_symbolic_pair_table_gate_manifest.json",
    }
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["candidate_rules_csv"], rules, RULE_COLUMNS)
    write_csv(outputs["integrated_predictions_csv"], integrated, list(integrated[0]) if integrated else [])
    manifest = {
        "schema_version": "kg1_v364_symbolic_pair_table_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "input_predictions_csv": str(args.input_predictions_csv),
            "input_predictions_sha256": sha256_file(args.input_predictions_csv),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "current_summary": baseline,
        "v364_summary": after,
        "candidate_rule_summary_top": rules[:25],
        "accepted_candidate_count": len(accepted),
        "accepted_candidate_ids": [row["id"] for row in accepted],
        "gain_count": len(gains),
        "gain_ids": [row["id"] for row in gains],
        "loss_count": len(losses),
        "loss_ids": [row["id"] for row in losses],
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V364 SYMBOLIC PAIR TABLE GATE END ===", flush=True)
    return manifest


def self_test() -> None:
    examples = [("ab+cd", "x"), ("ab+ef", "y")]
    predictions, _proof, _program_count = pair_table_predictions(examples, "zz+cd", max_output_len=4, max_program_count=1000)
    assert predictions == ["x"]
    print("v364_symbolic_pair_table_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v364_symbolic_pair_table_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--max-output-len", type=int, default=4)
    parser.add_argument("--max-program-count", type=int, default=5000)
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

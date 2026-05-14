#!/usr/bin/env python3
"""V357 CPU gate for exact global ternary bit rules.

This is a strict label-audited teacher gate, not a submission path. It extends
V350's exact global unary/binary byte rules to exact ternary byte expressions:

    OP1(T1(input), OP2(T2(input), T3(input)))
    OP1(OP2(T2(input), T3(input)), T1(input))

The gate promotes only rows where the full expression search has exactly one
unique prediction, that prediction changes the current V350 answer, and the
class has zero losses over the entire weak bit contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import EXPECTED_ROW_CONTRACT_SHA256, normalize_row, row_contract, sha256_file  # noqa: E402
from solvers.bit_manipulation_solver import BITWISE_OPS, make_transforms, parse_bit_problem, to_bits, from_bits  # noqa: E402


DEFAULT_V350_INTEGRATED = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_integrated_predictions.csv"
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
    return str(row.get("v350_prediction") or row.get("prediction") or row.get("baseline_prediction") or "").strip()


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


def exact_global_ternary(prompt: str) -> dict[str, Any]:
    examples, test_input = parse_bit_problem(prompt)
    if not examples or not test_input:
        return {
            "status": "parse_failed",
            "prediction": "",
            "unique_prediction_count": 0,
            "candidate_rule_count": 0,
            "proof": "parse_failed",
        }

    transforms = make_transforms()
    inputs = [to_bits(left) for left, _right in examples]
    outputs = [to_bits(right) for _left, right in examples]
    test_bits = to_bits(test_input)
    all_inputs = inputs + [test_bits]
    n = len(inputs)
    pre = {name: [func(bits) for bits in all_inputs] for name, func in transforms}

    unique_predictions: dict[str, str] = {}
    candidate_rule_count = 0
    for t1_name, _t1_func in transforms:
        p1 = pre[t1_name]
        for t2_name, _t2_func in transforms:
            p2 = pre[t2_name]
            for t3_name, _t3_func in transforms:
                p3 = pre[t3_name]
                for op2_name, op2_func in BITWISE_OPS:
                    inner = [op2_func(p2[index], p3[index]) for index in range(n + 1)]
                    for op1_name, op1_func in BITWISE_OPS:
                        ok = True
                        for index in range(n):
                            if op1_func(p1[index], inner[index]) != outputs[index]:
                                ok = False
                                break
                        if ok:
                            pred = from_bits(op1_func(p1[n], inner[n]))
                            candidate_rule_count += 1
                            unique_predictions.setdefault(
                                pred,
                                f"{op1_name}({t1_name},{op2_name}({t2_name},{t3_name}))",
                            )

                        ok = True
                        for index in range(n):
                            if op1_func(inner[index], p1[index]) != outputs[index]:
                                ok = False
                                break
                        if ok:
                            pred = from_bits(op1_func(inner[n], p1[n]))
                            candidate_rule_count += 1
                            unique_predictions.setdefault(
                                pred,
                                f"{op1_name}({op2_name}({t2_name},{t3_name}),{t1_name})",
                            )

    if not unique_predictions:
        return {
            "status": "no_rule",
            "prediction": "",
            "unique_prediction_count": 0,
            "candidate_rule_count": 0,
            "proof": "no exact ternary rule",
        }
    if len(unique_predictions) != 1:
        return {
            "status": "ambiguous",
            "prediction": "",
            "unique_prediction_count": len(unique_predictions),
            "candidate_rule_count": candidate_rule_count,
            "proof": "predictions=" + ",".join(sorted(unique_predictions)[:10]),
        }
    prediction, proof = next(iter(unique_predictions.items()))
    return {
        "status": "unique",
        "prediction": prediction,
        "unique_prediction_count": 1,
        "candidate_rule_count": candidate_rule_count,
        "proof": proof,
    }


def build_decisions(rows: list[dict[str, Any]], progress_every: int) -> list[dict[str, Any]]:
    bit_rows = [row for row in rows if str(row["family"]) == "bit_manipulation"]
    decisions: list[dict[str, Any]] = []
    start = time.time()
    for index, row in enumerate(bit_rows, start=1):
        result = exact_global_ternary(str(row["prompt"]))
        if index == 1 or index % progress_every == 0 or index == len(bit_rows):
            print(
                "v357_bit_ternary_progress = "
                + f"{index}/{len(bit_rows)} decisions={len(decisions)} elapsed_s={time.time() - start:.1f}",
                flush=True,
            )
        old_prediction = current_prediction(row)
        if result["status"] != "unique" or result["prediction"] == old_prediction:
            continue
        decisions.append(
            {
                "id": row["id"],
                "family": "bit_manipulation",
                "candidate_source": "v357_bit_exact_global_ternary",
                "rule_class": "bit_exact_global_ternary_unique_prediction",
                "old_prediction": old_prediction,
                "new_prediction": result["prediction"],
                "answer": row["answer"],
                "old_correct": verify_answer(str(row["answer"]), old_prediction),
                "new_correct": verify_answer(str(row["answer"]), str(result["prediction"])),
                "accepted": False,
                "rejection_reason": "",
                "candidate_count": int(result["candidate_rule_count"]),
                "conflict_count": 0,
                "proof": result["proof"],
            }
        )
    return decisions


def apply_gate(decisions: list[dict[str, Any]]) -> None:
    gains = [row for row in decisions if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
    losses = [row for row in decisions if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
    if gains and not losses:
        for row in decisions:
            if row in gains:
                row["accepted"] = True
            else:
                row["rejection_reason"] = "reject_no_accuracy_gain"
    elif losses:
        for row in decisions:
            row["rejection_reason"] = "reject_bit_exact_global_ternary_has_losses"
    else:
        for row in decisions:
            row["rejection_reason"] = "reject_bit_exact_global_ternary_no_gain"


def rule_summary(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not decisions:
        return []
    changed = [row for row in decisions if row["new_prediction"] != row["old_prediction"]]
    gains = [row for row in changed if truthy(row["new_correct"]) and not truthy(row["old_correct"])]
    losses = [row for row in changed if truthy(row["old_correct"]) and not truthy(row["new_correct"])]
    accepted = [row for row in decisions if truthy(row["accepted"])]
    return [
        {
            "family": "bit_manipulation",
            "candidate_source": "v357_bit_exact_global_ternary",
            "rule_class": "bit_exact_global_ternary_unique_prediction",
            "candidate_rows": len(decisions),
            "changed_rows": len(changed),
            "gains": len(gains),
            "losses": len(losses),
            "accepted_rows": len(accepted),
            "accepted": bool(accepted),
            "reason": "accepted_no_loss" if accepted else "rejected_by_gate",
        }
    ]


def apply_accepted(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id = {str(row["id"]): row for row in decisions if truthy(row["accepted"])}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["v357_prediction"] = current_prediction(row)
        item["v357_source_rule"] = ""
        accepted = accepted_by_id.get(str(row["id"]))
        if accepted:
            item["v357_prediction"] = accepted["new_prediction"]
            item["v357_source_rule"] = accepted["rule_class"]
        item["v357_correct"] = verify_answer(str(item["answer"]), str(item["v357_prediction"]))
        # Keep prediction as integrated candidate for downstream tooling.
        item["prediction"] = item["v357_prediction"]
        out.append(item)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V357 BIT GLOBAL TERNARY GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v350_integrated_predictions_csv =", args.v350_integrated_predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(row) for row in read_csv(args.v350_integrated_predictions_csv)]
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    v350_summary = summarize(rows, "v350_prediction")
    print("v350_summary =", json.dumps(v350_summary, sort_keys=True), flush=True)

    decisions = build_decisions(rows, progress_every=args.progress_every)
    apply_gate(decisions)
    integrated = apply_accepted(rows, decisions)
    v357_summary = summarize(integrated, "v357_prediction")
    rules = rule_summary(decisions)
    gains = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), str(row["v357_prediction"]))
        and not verify_answer(str(row["answer"]), current_prediction(row))
    ]
    losses = [
        row
        for row in integrated
        if verify_answer(str(row["answer"]), current_prediction(row))
        and not verify_answer(str(row["answer"]), str(row["v357_prediction"]))
    ]
    eq_correct = int(v357_summary["family"].get("equation_transform", {}).get("correct", 0))
    bit_correct = int(v357_summary["family"].get("bit_manipulation", {}).get("correct", 0))
    gate_pass = not losses and bit_correct > args.expected_v350_bit_correct

    decision = {
        "decision": "v357_bit_global_ternary_gate_passed" if gate_pass else "v357_bit_global_ternary_gate_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            f"v357={v357_summary['correct']}/315; equation={eq_correct}/155; bit={bit_correct}/160; "
            f"gains={len(gains)}; losses={len(losses)}"
        ),
        "next_action": (
            "Build V358 transfer dataset from V357 bit gains with strong replay before any HF smoke."
            if gate_pass
            else "Do not launch HF. Continue CPU search."
        ),
    }
    print("candidate_decision_rows =", len(decisions), flush=True)
    print("candidate_rule_summary =", json.dumps(rules, sort_keys=True), flush=True)
    print("v357_summary =", json.dumps(v357_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)

    outputs = {
        "candidate_decisions_csv": args.output_dir / "v357_candidate_decisions.csv",
        "candidate_rules_csv": args.output_dir / "v357_candidate_rules.csv",
        "integrated_predictions_csv": args.output_dir / "v357_integrated_predictions.csv",
        "manifest_json": args.output_dir / "v357_bit_global_ternary_gate_manifest.json",
    }
    write_csv(outputs["candidate_decisions_csv"], decisions, DECISION_COLUMNS)
    write_csv(outputs["candidate_rules_csv"], rules, RULE_COLUMNS)
    write_csv(outputs["integrated_predictions_csv"], integrated, list(integrated[0]) if integrated else [])
    manifest = {
        "schema_version": "kg1_v357_bit_global_ternary_gate_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v350_integrated_predictions_csv": str(args.v350_integrated_predictions_csv),
            "v350_integrated_predictions_sha256": sha256_file(args.v350_integrated_predictions_csv),
        },
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "v350_summary": v350_summary,
        "v357_summary": v357_summary,
        "candidate_rule_summary": rules,
        "accepted_candidate_count": sum(1 for row in decisions if truthy(row["accepted"])),
        "accepted_candidate_ids": [row["id"] for row in decisions if truthy(row["accepted"])],
        "gain_count": len(gains),
        "gain_ids": [row["id"] for row in gains],
        "loss_count": len(losses),
        "loss_ids": [row["id"] for row in losses],
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V357 BIT GLOBAL TERNARY GATE END ===", flush=True)
    return manifest


def self_test() -> None:
    prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
10000000 -> 00000001
01000000 -> 10000000
00100000 -> 01000000

Now, determine the output for: 00010000"""
    result = exact_global_ternary(prompt)
    # This simple rotation is representable through ternary families, but the
    # exact unary/binary gate already owns it. The self-test only checks that
    # the search runs and returns a structured status.
    if result["status"] not in {"unique", "ambiguous", "no_rule"}:
        raise AssertionError(result)
    print("v357_bit_global_ternary_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v350-integrated-predictions-csv", type=Path, default=DEFAULT_V350_INTEGRATED)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v357_bit_global_ternary_gate" / utc_compact())
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--expected-v350-bit-correct", type=int, default=138)
    parser.add_argument("--progress-every", type=int, default=20)
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

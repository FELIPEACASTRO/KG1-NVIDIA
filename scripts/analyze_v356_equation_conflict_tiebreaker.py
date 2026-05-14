#!/usr/bin/env python3
"""V356 label-free tiebreaker audit for equation cryptarithm conflicts.

The goal is not to squeeze weak-label wins. It checks whether V350/V355
verified-but-rejected cryptarithm candidates have a defensible label-free
tiebreaker. In particular, it blocks candidates whose decisive operator appears
only in the query and never in the examples.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_v278_symbolic_pbe_dsl_audit_hf import parse_alice_prompt, sha256_file  # noqa: E402


DEFAULT_V350_INTEGRATED = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_integrated_predictions.csv"
)
DEFAULT_V350_DECISIONS = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_candidate_decisions.csv"
)

AUDIT_COLUMNS = [
    "id",
    "answer",
    "old_prediction",
    "new_prediction",
    "rule_class",
    "query",
    "ops_symbols",
    "query_only_ops",
    "example_seen_ops",
    "label_free_tiebreaker",
    "decision",
    "reason",
    "proof",
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


def extract_ops(proof: str) -> dict[str, str]:
    match = re.search(r"ops=(\{.*?\});\s*map=", str(proof))
    if not match:
        return {}
    text = match.group(1)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def audit_conflicts(integrated_rows: list[dict[str, str]], decision_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    prompt_by_id = {str(row["id"]): str(row.get("prompt", "")) for row in integrated_rows}
    out: list[dict[str, Any]] = []
    for row in decision_rows:
        if str(row.get("family")) != "equation_transform":
            continue
        if not truthy(row.get("new_correct")):
            continue
        if str(row.get("rejection_reason")) != "reject_conflicting_predictions":
            continue
        prompt = prompt_by_id.get(str(row["id"]), "")
        examples, query, parse_status = parse_alice_prompt(prompt)
        ops = extract_ops(str(row.get("proof", "")))
        example_text = "".join(lhs for lhs, _rhs in examples)
        example_seen_ops = sorted(symbol for symbol in ops if symbol in example_text)
        query_only_ops = sorted(symbol for symbol in ops if symbol in query and symbol not in example_text)
        if parse_status != "ok":
            decision = "reject"
            reason = "parse_failed"
            tiebreak = False
        elif query_only_ops:
            decision = "reject"
            reason = "query_only_operator_no_label_free_tiebreaker"
            tiebreak = False
        elif len(set(row.get("new_prediction", ""))) == 0:
            decision = "reject"
            reason = "empty_prediction"
            tiebreak = False
        else:
            decision = "needs_manual_math_proof"
            reason = "no_query_only_operator_detected_but_conflict_remains"
            tiebreak = False
        out.append(
            {
                "id": row["id"],
                "answer": row.get("answer", ""),
                "old_prediction": row.get("old_prediction", ""),
                "new_prediction": row.get("new_prediction", ""),
                "rule_class": row.get("rule_class", ""),
                "query": query,
                "ops_symbols": "".join(sorted(ops)),
                "query_only_ops": "".join(query_only_ops),
                "example_seen_ops": "".join(example_seen_ops),
                "label_free_tiebreaker": tiebreak,
                "decision": decision,
                "reason": reason,
                "proof": row.get("proof", ""),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V356 EQUATION CONFLICT TIEBREAKER START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v350_integrated_predictions_csv =", args.v350_integrated_predictions_csv, flush=True)
    print("v350_candidate_decisions_csv =", args.v350_candidate_decisions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    integrated = read_csv(args.v350_integrated_predictions_csv)
    decisions = read_csv(args.v350_candidate_decisions_csv)
    audit_rows = audit_conflicts(integrated, decisions)
    accepted = [row for row in audit_rows if truthy(row.get("label_free_tiebreaker"))]
    reason_counts = Counter(str(row["reason"]) for row in audit_rows)
    print("conflict_rows_audited =", len(audit_rows), flush=True)
    print("label_free_tiebreakers =", len(accepted), flush=True)
    print("reason_counts =", json.dumps(dict(reason_counts), sort_keys=True), flush=True)

    outputs = {
        "audit_csv": args.output_dir / "v356_equation_conflict_tiebreaker_audit.csv",
        "manifest_json": args.output_dir / "v356_equation_conflict_tiebreaker_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    decision = {
        "decision": "v356_conflict_tiebreaker_blocked",
        "hf_gpu_allowed": False,
        "reason": (
            "No label-free tiebreaker found. Verified conflict rows rely on query-only operators "
            "that never appear in examples."
        ),
        "next_action": "Do not promote these equation conflicts. Continue only with new CPU rules that remove ambiguity before weak labels.",
    }
    manifest = {
        "schema_version": "kg1_v356_equation_conflict_tiebreaker_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v350_integrated_predictions_csv": str(args.v350_integrated_predictions_csv),
            "v350_integrated_predictions_sha256": sha256_file(args.v350_integrated_predictions_csv),
            "v350_candidate_decisions_csv": str(args.v350_candidate_decisions_csv),
            "v350_candidate_decisions_sha256": sha256_file(args.v350_candidate_decisions_csv),
        },
        "conflict_rows_audited": len(audit_rows),
        "label_free_tiebreaker_count": len(accepted),
        "reason_counts": dict(reason_counts),
        "decision": decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V356 EQUATION CONFLICT TIEBREAKER END ===", flush=True)
    return manifest


def self_test() -> None:
    proof = 'ops={"$": "mul", "{": "sub_ab"}; map={"a": 1}'
    assert extract_ops(proof) == {"$": "mul", "{": "sub_ab"}
    print("v356_equation_conflict_tiebreaker_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v350-integrated-predictions-csv", type=Path, default=DEFAULT_V350_INTEGRATED)
    parser.add_argument("--v350-candidate-decisions-csv", type=Path, default=DEFAULT_V350_DECISIONS)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v356_equation_conflict_tiebreaker" / utc_compact())
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

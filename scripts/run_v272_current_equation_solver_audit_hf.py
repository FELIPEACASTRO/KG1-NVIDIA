#!/usr/bin/env python3
"""Audit deployable equation solver candidates on current-best misses.

V272 is CPU-only. It re-downloads the current-best weak prediction CSV, filters
the 99 current equation misses, and tests conservative symbolic/numeric rule
classes. Weak labels are used only as an audit brake: a rule class is never
promoted if it produces any incorrect candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import (  # noqa: E402
    answers_equal,
    deletion_positions_probe,
    parse_alice_prompt,
    prefix_suffix_probe,
    reverse_probe,
)
from analyze_v241_abstain_rule_candidate_audit import (  # noqa: E402
    infer_symbolic_transducer,
    numeric_rule_functions,
)


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_BASELINE_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_BASELINE_FILENAME = (
    "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
    "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
)

AUDIT_COLUMNS = [
    "id",
    "subtype",
    "answer",
    "baseline_prediction",
    "rule_class",
    "status",
    "prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "promotable_after_class_gate",
    "query",
    "example_count",
    "proof",
]

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download_file(repo_id: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V272") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "prediction": prediction,
        "family": family,
        "prompt_sha256": sha256_text(prompt),
        "correct_bool": verify_answer(answer, prediction),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def row_contract(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate ids in baseline predictions")
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: item["id"])
    )
    return sha256_text(payload)


def parse_numeric_token(value: str) -> tuple[int, str, int] | None:
    import re

    match = re.fullmatch(r"(-?\d+)([^\d\s])(-?\d+)", value)
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def classify_subtype(examples: list[tuple[str, str]], query: str) -> str:
    if examples and parse_numeric_token(query) and all(parse_numeric_token(lhs) for lhs, _ in examples):
        return "equation_numeric_operator"
    return "equation_symbolic_punct"


def audit_result(row: dict[str, Any], rule_class: str, result: dict[str, Any], examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    status = str(result.get("status", "abstain"))
    prediction = str(result.get("prediction", ""))
    verified = status in {"candidate", "under_evidenced_candidate"} and answers_equal(prediction, row["answer"])
    incorrect = status in {"candidate", "under_evidenced_candidate"} and not verified
    return {
        "id": row["id"],
        "subtype": classify_subtype(examples, query),
        "answer": row["answer"],
        "baseline_prediction": row["prediction"],
        "rule_class": rule_class,
        "status": status,
        "prediction": prediction,
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "promotable_after_class_gate": False,
        "query": query,
        "example_count": len(examples),
        "proof": str(result.get("proof", ""))[:500],
    }


def symbolic_audits(row: dict[str, Any], examples: list[tuple[str, str]], query: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    transducer = infer_symbolic_transducer(examples, query, pair_cap=args.pair_mapping_cap, global_cap=args.global_mapping_cap)
    outputs.append(audit_result(row, "symbolic_all_examples_char_transducer", transducer, examples, query))
    for rule_class, probe in (
        ("symbolic_reverse", reverse_probe),
        ("symbolic_prefix_suffix", prefix_suffix_probe),
        ("symbolic_positional_deletion_audit_only", deletion_positions_probe),
    ):
        outputs.append(audit_result(row, rule_class, probe(examples, query), examples, query))
    return outputs


def numeric_audits(row: dict[str, Any], examples: list[tuple[str, str]], query: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return [
            audit_result(
                row,
                "numeric_parse_gate",
                {"status": "abstain", "prediction": "", "proof": "query_not_numeric_binary"},
                examples,
                query,
            )
        ]
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
                candidates.append((name, func(parsed_query[0], parsed_query[2])))
            except Exception:
                pass
    unique_predictions = sorted(set(prediction for _, prediction in candidates))
    if not same_operator:
        result = {"status": "abstain", "prediction": "", "proof": "no_same_operator_examples"}
    elif len(unique_predictions) != 1:
        result = {
            "status": "abstain",
            "prediction": "",
            "proof": f"candidate_rule_count={len(candidates)} unique_prediction_count={len(unique_predictions)}",
        }
    elif len(same_operator) < args.min_same_operator_examples:
        result = {
            "status": "under_evidenced_candidate",
            "prediction": unique_predictions[0],
            "proof": f"same_operator_examples={len(same_operator)} below_min={args.min_same_operator_examples}",
        }
    else:
        result = {
            "status": "candidate",
            "prediction": unique_predictions[0],
            "proof": "rules=" + ",".join(name for name, _ in candidates),
        }
    return [audit_result(row, "numeric_same_operator_rule", result, examples, query)]


def summarize_rule_classes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[str(row["rule_class"])].append(row)
    summaries: list[dict[str, Any]] = []
    for rule_class, items in sorted(by_rule.items()):
        candidate_rows = [row for row in items if row["status"] in {"candidate", "under_evidenced_candidate"}]
        incorrect = [row for row in candidate_rows if row["incorrect_by_weak_label"]]
        verified = [row for row in candidate_rows if row["verified_by_weak_label"]]
        promotable = bool(candidate_rows) and not incorrect
        for row in items:
            if row["status"] in {"candidate", "under_evidenced_candidate"}:
                row["promotable_after_class_gate"] = promotable
        summaries.append(
            {
                "rule_class": rule_class,
                "rows": len(items),
                "candidate_rows": len(candidate_rows),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "abstain_rows": len(items) - len(candidate_rows),
                "promotable_after_class_gate": promotable,
            }
        )
    summaries.sort(key=lambda row: (int(row["verified_candidates"]), -int(row["incorrect_candidates"]), str(row["rule_class"])), reverse=True)
    return summaries


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V272 CURRENT EQUATION SOLVER AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("target_verified_gain =", args.target_verified_gain, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v272_") as temp_name:
        path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(path)]
        input_meta = {
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
        }

    observed = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed, flush=True)
    if observed != args.expected_shared_row_contract_sha256:
        raise RuntimeError(f"row contract mismatch: expected {args.expected_shared_row_contract_sha256}, got {observed}")

    equation_misses = [
        row
        for row in rows
        if row["family"] == "equation_transform" and not row["correct_bool"] and not row["truncated_bool"]
    ]
    audit_rows: list[dict[str, Any]] = []
    parse_status_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    for row in equation_misses:
        examples, query, parse_status = parse_alice_prompt(row["prompt"])
        parse_status_counts[parse_status] += 1
        if parse_status != "ok":
            audit_rows.append(
                audit_result(
                    row,
                    "alice_parse_gate",
                    {"status": "abstain", "prediction": "", "proof": parse_status},
                    examples,
                    query,
                )
            )
            continue
        subtype = classify_subtype(examples, query)
        subtype_counts[subtype] += 1
        if subtype == "equation_numeric_operator":
            audit_rows.extend(numeric_audits(row, examples, query, args))
        else:
            audit_rows.extend(symbolic_audits(row, examples, query, args))

    summary_rows = summarize_rule_classes(audit_rows)
    verified_promotable = [
        row
        for row in audit_rows
        if row["verified_by_weak_label"] and row["promotable_after_class_gate"] and row["status"] == "candidate"
    ]
    under_evidenced_verified = [
        row
        for row in audit_rows
        if row["verified_by_weak_label"] and row["promotable_after_class_gate"] and row["status"] == "under_evidenced_candidate"
    ]
    incorrect_promotable = [row for row in audit_rows if row["incorrect_by_weak_label"] and row["promotable_after_class_gate"]]

    outputs = {
        "audit_csv": args.output_dir / "v272_equation_solver_candidate_audit.csv",
        "rule_summary_csv": args.output_dir / "v272_rule_class_summary.csv",
        "verified_promotable_csv": args.output_dir / "v272_verified_promotable_candidates.csv",
        "manifest_json": args.output_dir / "v272_current_equation_solver_audit_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    write_csv(outputs["rule_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["verified_promotable_csv"], verified_promotable + under_evidenced_verified, AUDIT_COLUMNS)

    if len(verified_promotable) >= args.target_verified_gain and not incorrect_promotable:
        decision = "verified_solver_candidates_ready_for_eval"
        next_action = "Create a guarded override eval candidate with class-gated solver rules."
    elif verified_promotable or under_evidenced_verified:
        decision = "partial_solver_signal_needs_review"
        next_action = "Review verified/under-evidenced classes and expand only zero-incorrect rule families."
    else:
        decision = "no_deployable_solver_signal"
        next_action = "Unlock solver-guided external traces or design stronger symbolic-punctuation search."

    manifest = {
        "schema_version": "kg1_v272_current_equation_solver_audit_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed,
        "equation_miss_rows": len(equation_misses),
        "parse_status_counts": dict(parse_status_counts),
        "subtype_counts": dict(subtype_counts),
        "rule_summary": summary_rows,
        "verified_promotable_candidates": len(verified_promotable),
        "under_evidenced_verified_promotable_candidates": len(under_evidenced_verified),
        "incorrect_promotable_candidates": len(incorrect_promotable),
        "decision": {
            "decision": decision,
            "reason": (
                f"equation_misses={len(equation_misses)}; "
                f"verified_promotable={len(verified_promotable)}; "
                f"under_evidenced_verified={len(under_evidenced_verified)}; "
                f"incorrect_promotable={len(incorrect_promotable)}"
            ),
            "next_action": next_action,
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("parse_status_counts =", json.dumps(dict(parse_status_counts), sort_keys=True), flush=True)
    print("subtype_counts =", json.dumps(dict(subtype_counts), sort_keys=True), flush=True)
    print("rule_summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V272 CURRENT EQUATION SOLVER AUDIT END ===", flush=True)
    return manifest


def run_self_test() -> None:
    row = {
        "id": "x",
        "answer": "ba",
        "prediction": "aa",
        "prompt": "Below are a few examples:\nab = ba\ncd = dc\nNow, determine the result for: ef",
    }
    examples, query, status = parse_alice_prompt(row["prompt"])
    if status != "ok":
        raise AssertionError(status)
    result = reverse_probe(examples, query)
    audit = audit_result(row, "symbolic_reverse", result, examples, query)
    if audit["prediction"] != "fe":
        raise AssertionError(audit)
    print("v272_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v272_current_equation_solver_audit"))
    parser.add_argument("--run-id", default=f"v272-hf-cpu-equation-solver-audit-{utc_compact()}")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-verified-gain", type=int, default=4)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--pair-mapping-cap", type=int, default=3000)
    parser.add_argument("--global-mapping-cap", type=int, default=12000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

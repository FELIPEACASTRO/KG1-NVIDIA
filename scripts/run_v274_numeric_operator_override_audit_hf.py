#!/usr/bin/env python3
"""Audit guarded numeric equation overrides on the current best weak run.

V274 is CPU-only. It downloads the current-best weak prediction CSV and applies
only deployable, label-free post-processing rules that use the prompt plus the
model's own prediction. Weak labels are used only to audit gains/losses and to
block unsafe rule classes before any full-eval packaging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import (  # noqa: E402
    answers_equal,
    normalize_answer,
    parse_alice_prompt,
)
from analyze_v241_abstain_rule_candidate_audit import numeric_rule_functions  # noqa: E402


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_BASELINE_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_BASELINE_FILENAME = (
    "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
    "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
)

PREDICTION_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "prediction",
    "family",
    "task_type",
    "truncated",
    "override_rule",
    "baseline_prediction",
]

AUDIT_COLUMNS = [
    "id",
    "rule",
    "status",
    "family",
    "answer",
    "baseline_prediction",
    "override_prediction",
    "baseline_correct",
    "override_correct",
    "query",
    "example_count",
    "proof",
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
        raise RuntimeError("huggingface_hub is required for V274") from exc
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
        "baseline_correct_bool": verify_answer(answer, prediction),
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


def parse_numeric_token(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(\d+)(\D)(\d+)", value)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def reverse_text(value: str) -> str:
    text = str(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def reverse_normalized_keep_sign(value: str) -> str:
    text = normalize_answer(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def numeric_candidates(
    group: list[tuple[str, str, str]],
    query: str,
    names: set[str],
) -> list[dict[str, Any]]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return []
    query_left, _, query_right = parsed_query
    outputs: list[dict[str, Any]] = []
    functions: dict[str, Callable[[int, int], str]] = numeric_rule_functions()
    for name, func in functions.items():
        if name not in names:
            continue
        for reverse_operands in (False, True):
            for reverse_result in (False, True):
                ok = True
                for left, right, expected in group:
                    transformed_left = left[::-1] if reverse_operands else left
                    transformed_right = right[::-1] if reverse_operands else right
                    try:
                        raw = str(func(int(transformed_left), int(transformed_right)))
                    except Exception:
                        ok = False
                        break
                    prediction = reverse_text(raw) if reverse_result else raw
                    if prediction != expected:
                        ok = False
                        break
                if not ok:
                    continue
                transformed_left = query_left[::-1] if reverse_operands else query_left
                transformed_right = query_right[::-1] if reverse_operands else query_right
                try:
                    raw = str(func(int(transformed_left), int(transformed_right)))
                except Exception:
                    continue
                prediction = reverse_text(raw) if reverse_result else raw
                outputs.append(
                    {
                        "name": name,
                        "reverse_operands": reverse_operands,
                        "reverse_result": reverse_result,
                        "prediction": prediction,
                    }
                )
    return outputs


def group_examples_by_operator(examples: list[tuple[str, str]]) -> tuple[dict[str, list[tuple[str, str, str]]], list[str]] | None:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    op_sequence: list[str] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if not parsed:
            return None
        left, op, right = parsed
        grouped[op].append((left, right, str(rhs)))
        op_sequence.append(op)
    return dict(grouped), op_sequence


def choose_guarded_numeric_override(examples: list[tuple[str, str]], query: str, model_prediction: str) -> tuple[str | None, str, str]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return None, "not_numeric_query", "query is not a numeric binary expression"
    grouped_result = group_examples_by_operator(examples)
    if grouped_result is None:
        return None, "not_all_numeric_examples", "one or more examples are not numeric binary expressions"
    grouped, op_sequence = grouped_result
    query_op = parsed_query[1]
    if query_op not in grouped:
        return None, "query_operator_unseen", f"query_op={query_op!r} not present in examples"

    base = normalize_answer(model_prediction)
    group = grouped[query_op]

    if query_op == "-":
        signed = numeric_candidates(group, query, {"sub_ab", "rev_sub_ab"})
        predictions = sorted({str(item["prediction"]) for item in signed})
        if len(predictions) != 1:
            return None, "minus_signed_ambiguous", f"signed_predictions={predictions}"
        candidate = predictions[0]
        if set(op_sequence) == {"-"} and all(str(rhs).startswith("-") for _, rhs in examples):
            return None, "minus_guard_all_negative_examples", "single '-' rule with all negative examples was a known unsafe pattern"
        candidate_norm = normalize_answer(candidate)
        if base.lstrip("-") == candidate_norm.lstrip("-") and base != candidate_norm:
            return candidate, "minus_signed_opposite_sign_guarded", f"candidate={candidate}; baseline={model_prediction}"
        return None, "minus_model_not_opposite_sign", f"candidate={candidate}; baseline={model_prediction}"

    if query_op == ":":
        abs_family = numeric_candidates(
            group,
            query,
            {"abs_diff", "rev_abs_diff", "digit_absdiff_concat", "tens_absdiff_ones_absdiff_int"},
        )
        same_len_unreversed = [
            str(item["prediction"])
            for item in abs_family
            if not item["reverse_result"]
            and len(normalize_answer(item["prediction"])) == len(base)
            and base == reverse_normalized_keep_sign(str(item["prediction"]))
            and base != normalize_answer(item["prediction"])
        ]
        predictions = sorted(set(same_len_unreversed))
        if len(predictions) == 1:
            return predictions[0], "colon_absdiff_unreverse_same_len", f"candidate={predictions[0]}; baseline={model_prediction}"
        return None, "colon_no_unique_unreverse", f"candidate_predictions={predictions}"

    if query_op in {")", "+"}:
        add_family = numeric_candidates(group, query, {"add", "rev_add", "tens_add_ones_add"})
        direct_add = sorted(
            {
                str(item["prediction"])
                for item in add_family
                if not item["reverse_operands"]
                and not item["reverse_result"]
                and str(item["name"]) in {"add", "tens_add_ones_add"}
            }
        )
        add_predictions = {normalize_answer(item["prediction"]) for item in add_family}
        if len(direct_add) == 1 and base in add_predictions and base != normalize_answer(direct_add[0]):
            return direct_add[0], "add_direct_over_model_add_variant", f"candidate={direct_add[0]}; baseline={model_prediction}"
        return None, "add_no_unique_direct_variant", f"direct_add={direct_add}; baseline={model_prediction}"

    return None, "operator_not_guarded", f"query_op={query_op!r}"


def apply_overrides(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        out_row = dict(row)
        out_row["baseline_prediction"] = row["prediction"]
        out_row["override_rule"] = ""
        family = row["family"]
        rule = "not_attempted"
        status = "abstain"
        proof = "non-equation row"
        override_prediction: str | None = None
        query = ""
        examples: list[tuple[str, str]] = []
        if family == "equation_transform" and not row["truncated_bool"]:
            examples, query, parse_status = parse_alice_prompt(row["prompt"])
            if parse_status == "ok":
                override_prediction, rule, proof = choose_guarded_numeric_override(examples, query, row["prediction"])
                status = "candidate" if override_prediction is not None else "abstain"
            else:
                rule = "alice_parse_gate"
                proof = parse_status
        if override_prediction is not None:
            out_row["prediction"] = override_prediction
            out_row["override_rule"] = rule
        baseline_correct = bool(row["baseline_correct_bool"])
        override_correct = verify_answer(row["answer"], out_row["prediction"])
        audit_rows.append(
            {
                "id": row["id"],
                "rule": rule,
                "status": status,
                "family": family,
                "answer": row["answer"],
                "baseline_prediction": row["baseline_prediction"] if "baseline_prediction" in row else row["prediction"],
                "override_prediction": override_prediction or "",
                "baseline_correct": baseline_correct,
                "override_correct": override_correct,
                "query": query,
                "example_count": len(examples),
                "proof": proof,
            }
        )
        output_rows.append(out_row)
    return output_rows, audit_rows


def summarize(rows: list[dict[str, Any]], *, prediction_key: str) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    family_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0, "truncated": 0})
    for row in rows:
        family = row["family"]
        pred = str(row.get(prediction_key, ""))
        total += 1
        family_rows[family]["rows"] += 1
        if truthy(row.get("truncated", row.get("truncated_bool", "False"))):
            truncated += 1
            family_rows[family]["truncated"] += 1
        if verify_answer(row["answer"], pred):
            correct += 1
            family_rows[family]["correct"] += 1
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": dict(sorted(family_rows.items())),
    }


def summarize_audit(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit_rows:
        rule = str(row["rule"])
        grouped[rule]["rows"] += 1
        if row["status"] == "candidate":
            grouped[rule]["candidate_rows"] += 1
        if bool(row["baseline_correct"]):
            grouped[rule]["baseline_correct"] += 1
        if bool(row["override_correct"]):
            grouped[rule]["override_correct"] += 1
        if row["status"] == "candidate" and (not bool(row["baseline_correct"])) and bool(row["override_correct"]):
            grouped[rule]["gains"] += 1
        if row["status"] == "candidate" and bool(row["baseline_correct"]) and not bool(row["override_correct"]):
            grouped[rule]["losses"] += 1
        if row["status"] == "candidate" and (not bool(row["baseline_correct"])) and not bool(row["override_correct"]):
            grouped[rule]["wrong_on_baseline_miss"] += 1
    return [
        {"rule": rule, **dict(counts)}
        for rule, counts in sorted(grouped.items(), key=lambda item: (-item[1]["candidate_rows"], item[0]))
    ]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V274 NUMERIC OPERATOR OVERRIDE AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("expected_shared_row_contract_sha256 =", args.expected_shared_row_contract_sha256, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v274_") as temp_name:
        baseline_path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(baseline_path)]
        input_meta = {
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": baseline_path.name,
            "sha256": sha256_file(baseline_path),
            "rows": len(rows),
        }

    observed = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed, flush=True)
    if observed != args.expected_shared_row_contract_sha256:
        raise RuntimeError(f"row contract mismatch: expected {args.expected_shared_row_contract_sha256}, got {observed}")

    output_rows, audit_rows = apply_overrides(rows)
    baseline_summary = summarize(rows, prediction_key="prediction")
    override_summary = summarize(output_rows, prediction_key="prediction")
    audit_summary = summarize_audit(audit_rows)
    candidate_rows = [row for row in audit_rows if row["status"] == "candidate"]
    gains = [row for row in candidate_rows if (not bool(row["baseline_correct"])) and bool(row["override_correct"])]
    losses = [row for row in candidate_rows if bool(row["baseline_correct"]) and not bool(row["override_correct"])]
    wrong_on_misses = [row for row in candidate_rows if (not bool(row["baseline_correct"])) and not bool(row["override_correct"])]

    outputs = {
        "override_predictions_csv": args.output_dir / "v274_numeric_override_predictions.csv",
        "audit_csv": args.output_dir / "v274_numeric_override_audit.csv",
        "rule_summary_csv": args.output_dir / "v274_numeric_override_rule_summary.csv",
        "manifest_json": args.output_dir / "v274_numeric_operator_override_audit_manifest.json",
    }
    write_csv(outputs["override_predictions_csv"], output_rows, PREDICTION_COLUMNS)
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    summary_columns = [
        "rule",
        "rows",
        "candidate_rows",
        "baseline_correct",
        "override_correct",
        "gains",
        "losses",
        "wrong_on_baseline_miss",
    ]
    write_csv(outputs["rule_summary_csv"], audit_summary, summary_columns)

    eq_after = override_summary["family"].get("equation_transform", {})
    bit_after = override_summary["family"].get("bit_manipulation", {})
    weak_gate_pass = (
        int(override_summary["correct"]) >= args.weak_total_min
        and int(eq_after.get("correct", 0)) >= args.weak_eq_min
        and int(bit_after.get("correct", 0)) >= args.weak_bit_min
        and int(override_summary["truncated"]) <= args.weak_trunc_max
        and not losses
        and not wrong_on_misses
    )
    if weak_gate_pass:
        decision = "guarded_numeric_overrides_pass_weak_gate"
        next_action = "Package the same label-free postprocessor into an HF full-eval candidate with the V259 checkpoint-4 base predictions."
    else:
        decision = "guarded_numeric_overrides_do_not_pass_weak_gate"
        next_action = "Do not spend GPU on this postprocessor until the rule set has zero losses and reaches weak thresholds."

    manifest = {
        "schema_version": "kg1_v274_numeric_operator_override_audit_v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id or utc_compact(),
        "inputs": {
            "baseline": input_meta,
            "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
            "observed_shared_row_contract_sha256": observed,
        },
        "baseline_summary": baseline_summary,
        "override_summary": override_summary,
        "audit_summary": audit_summary,
        "candidate_rows": len(candidate_rows),
        "gains": len(gains),
        "losses": len(losses),
        "wrong_on_baseline_misses": len(wrong_on_misses),
        "weak_gate": {
            "pass": weak_gate_pass,
            "weak_total_min": args.weak_total_min,
            "weak_eq_min": args.weak_eq_min,
            "weak_bit_min": args.weak_bit_min,
            "weak_trunc_max": args.weak_trunc_max,
        },
        "decision": {
            "decision": decision,
            "next_action": next_action,
            "reason": (
                f"baseline={baseline_summary['correct']}; override={override_summary['correct']}; "
                f"eq={eq_after.get('correct', 0)}; bit={bit_after.get('correct', 0)}; "
                f"gains={len(gains)}; losses={len(losses)}; wrong_on_misses={len(wrong_on_misses)}"
            ),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    print("baseline_summary =", json.dumps(baseline_summary, sort_keys=True), flush=True)
    print("override_summary =", json.dumps(override_summary, sort_keys=True), flush=True)
    print("audit_summary =", json.dumps(audit_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("outputs =", json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V274 NUMERIC OPERATOR OVERRIDE AUDIT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf_cpu_runs/v274_numeric_operator_override_audit"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=133)
    parser.add_argument("--weak-trunc-max", type=int, default=3)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> None:
    examples = [
        ("06-63", "42"),
        ("96-32", "64"),
        ("87-15", "72"),
        ("58-64", "93"),
        ("87-63", "24"),
    ]
    prediction, rule, _ = choose_guarded_numeric_override(examples, "63-19", "55")
    assert prediction == "-55", (prediction, rule)
    unsafe_examples = [("02-23", "-21"), ("33-66", "-33"), ("75-58", "-82")]
    prediction, rule, _ = choose_guarded_numeric_override(unsafe_examples, "48-25", "-23")
    assert prediction is None and rule == "minus_guard_all_negative_examples", (prediction, rule)
    colon_examples = [("89$90", "8010"), ("88:77", "11"), ("10|87", "98"), ("41|87", "129")]
    prediction, rule, _ = choose_guarded_numeric_override(colon_examples, "37:67", "03")
    assert prediction == "30" and rule == "colon_absdiff_unreverse_same_len", (prediction, rule)
    add_examples = [("72)27", "99"), ("26#48", "22"), ("42#45", "3"), ("24#14", "10")]
    prediction, rule, _ = choose_guarded_numeric_override(add_examples, "94)40", "35")
    assert prediction == "134" and rule == "add_direct_over_model_add_variant", (prediction, rule)
    print("v274_numeric_operator_override_self_test=ok", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gate solver-derived signals before another HF training run.

V306 is CPU-only. It consolidates the strongest local signals found so far:

* V274 guarded numeric equation overrides.
* V300 full-byte bit grammar overrides.
* V296 stride-style bit relation solver as a diagnostic alternative.
* V299 broader numeric candidate classes as diagnostic alternatives.

Labels are used only for offline gate decisions. The output explicitly
separates deployable no-loss postprocessor signal from diagnostic candidates
that must not be promoted without a stricter label-free gate.
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
from kg1_v274_numeric_postprocessor import postprocess_numeric_prediction
from kg1_v300_bit_fullbyte_postprocessor import postprocess_bit_prediction

from run_v296_bit_stride_solver_audit import solve_stride
from run_v299_equation_numeric_candidate_audit import audit_row as audit_equation_candidate_row


AUDIT_COLUMNS = [
    "id",
    "family",
    "answer",
    "baseline_prediction",
    "combined_prediction",
    "baseline_correct",
    "combined_correct",
    "combined_gain",
    "combined_loss",
    "v274_eq_applied",
    "v274_eq_rule",
    "v300_bit_applied",
    "v300_bit_rule",
    "stride_prediction",
    "stride_status",
    "stride_default_bits",
    "stride_correct",
    "stride_gain_vs_baseline",
    "stride_loss_vs_baseline",
    "stride_agrees_with_v300",
    "truncated",
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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
        "baseline_correct_bool": verify_answer(answer, prediction),
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        ok = verify_answer(row["answer"], row.get(prediction_key, ""))
        total += 1
        correct += int(ok)
        truncated += int(row.get("truncated_bool", False))
        families[family]["rows"] += 1
        families[family]["correct"] += int(ok)
        families[family]["truncated"] += int(row.get("truncated_bool", False))
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": {key: dict(value) for key, value in sorted(families.items())},
    }


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row.get("family", "")), str(row.get("candidate_class", "")))
        grouped[key]["rows"] += 1
        if row.get("status") == "candidate":
            grouped[key]["candidate_rows"] += 1
        if row.get("candidate_correct"):
            grouped[key]["candidate_correct"] += 1
        if row.get("gain"):
            grouped[key]["gains"] += 1
        if row.get("loss"):
            grouped[key]["losses"] += 1
        if row.get("status") == "candidate" and (not row.get("baseline_correct")) and (not row.get("candidate_correct")):
            grouped[key]["wrong_on_baseline_miss"] += 1
    return [
        {"family": family, "candidate_class": candidate_class, **dict(counter)}
        for (family, candidate_class), counter in sorted(grouped.items())
    ]


def family_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_families = before.get("family", {})
    after_families = after.get("family", {})
    out: dict[str, Any] = {}
    for family in sorted(set(before_families) | set(after_families)):
        b = before_families.get(family, {})
        a = after_families.get(family, {})
        out[family] = {
            "rows": int(a.get("rows", b.get("rows", 0))),
            "baseline_correct": int(b.get("correct", 0)),
            "candidate_correct": int(a.get("correct", 0)),
            "delta_correct": int(a.get("correct", 0)) - int(b.get("correct", 0)),
            "baseline_truncated": int(b.get("truncated", 0)),
            "candidate_truncated": int(a.get("truncated", 0)),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = [normalize_row(row) for row in read_csv(args.input_csv)]
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    equation_candidate_rows_baseline: list[dict[str, Any]] = []
    equation_candidate_rows_combined: list[dict[str, Any]] = []

    for row in source_rows:
        baseline_prediction = str(row["prediction"])
        eq_decision = postprocess_numeric_prediction(
            row["prompt"],
            baseline_prediction,
            family=row["family"],
            truncated=bool(row["truncated_bool"]),
        )
        bit_decision = postprocess_bit_prediction(
            row["prompt"],
            eq_decision.prediction,
            family=row["family"],
            truncated=bool(row["truncated_bool"]),
        )
        combined_prediction = bit_decision.prediction
        baseline_correct = bool(row["baseline_correct_bool"])
        combined_correct = verify_answer(row["answer"], combined_prediction)

        stride_prediction = ""
        stride_status = "not_attempted"
        stride_default_bits = ""
        stride_correct = False
        stride_gain = False
        stride_loss = False
        stride_agrees_with_v300 = ""
        if row["family"] == "bit_manipulation":
            answer, meta = solve_stride(row["prompt"])
            stride_prediction = "" if answer is None else str(answer)
            stride_status = str(meta.get("status", "unknown"))
            stride_default_bits = str(meta.get("default_bits", ""))
            stride_correct = bool(stride_prediction) and verify_answer(row["answer"], stride_prediction)
            stride_gain = (not baseline_correct) and stride_correct
            stride_loss = baseline_correct and bool(stride_prediction) and (not stride_correct)
            if bit_decision.applied and stride_prediction:
                stride_agrees_with_v300 = str(stride_prediction == bit_decision.prediction)

        combined_row = dict(row)
        combined_row["baseline_prediction"] = baseline_prediction
        combined_row["prediction"] = combined_prediction
        combined_row["combined_prediction"] = combined_prediction
        combined_row["v274_eq_applied"] = eq_decision.applied
        combined_row["v274_eq_rule"] = eq_decision.rule
        combined_row["v274_eq_proof"] = eq_decision.proof
        combined_row["v300_bit_applied"] = bit_decision.applied
        combined_row["v300_bit_rule"] = bit_decision.rule
        combined_row["v300_bit_proof"] = bit_decision.proof
        output_rows.append(combined_row)

        audit_rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "answer": row["answer"],
                "baseline_prediction": baseline_prediction,
                "combined_prediction": combined_prediction,
                "baseline_correct": baseline_correct,
                "combined_correct": combined_correct,
                "combined_gain": (not baseline_correct) and combined_correct,
                "combined_loss": baseline_correct and (not combined_correct),
                "v274_eq_applied": eq_decision.applied,
                "v274_eq_rule": eq_decision.rule,
                "v300_bit_applied": bit_decision.applied,
                "v300_bit_rule": bit_decision.rule,
                "stride_prediction": stride_prediction,
                "stride_status": stride_status,
                "stride_default_bits": stride_default_bits,
                "stride_correct": stride_correct,
                "stride_gain_vs_baseline": stride_gain,
                "stride_loss_vs_baseline": stride_loss,
                "stride_agrees_with_v300": stride_agrees_with_v300,
                "truncated": row["truncated_bool"],
            }
        )

        if row["family"] == "equation_transform":
            baseline_candidate_row = dict(row)
            baseline_candidate_row["correct_bool"] = baseline_correct
            equation_candidate_rows_baseline.extend(audit_equation_candidate_row(baseline_candidate_row))

            combined_candidate_row = dict(row)
            combined_candidate_row["prediction"] = combined_prediction
            combined_candidate_row["correct_bool"] = combined_correct
            equation_candidate_rows_combined.extend(audit_equation_candidate_row(combined_candidate_row))

    baseline_summary = summarize(source_rows, "prediction")
    combined_summary = summarize(output_rows, "prediction")
    combined_gains = [row for row in audit_rows if row["combined_gain"]]
    combined_losses = [row for row in audit_rows if row["combined_loss"]]
    stride_gains = [row for row in audit_rows if row["stride_gain_vs_baseline"]]
    stride_losses = [row for row in audit_rows if row["stride_loss_vs_baseline"]]
    rule_applied = {
        "v274_eq": dict(Counter(row["v274_eq_rule"] for row in audit_rows if row["v274_eq_applied"])),
        "v300_bit": dict(Counter(row["v300_bit_rule"] for row in audit_rows if row["v300_bit_applied"])),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_dir / f"{args.label}_v306_solver_promotion_audit.csv"
    predictions_csv = args.output_dir / f"{args.label}_v306_combined_predictions.csv"
    equation_baseline_csv = args.output_dir / f"{args.label}_v306_equation_candidates_baseline.csv"
    equation_combined_csv = args.output_dir / f"{args.label}_v306_equation_candidates_after_combined.csv"
    summary_json = args.output_dir / f"{args.label}_v306_solver_promotion_manifest.json"

    write_csv(audit_csv, audit_rows, AUDIT_COLUMNS)
    if args.write_predictions_csv:
        write_csv(predictions_csv, output_rows, list(output_rows[0]) if output_rows else [])
    if equation_candidate_rows_baseline:
        write_csv(equation_baseline_csv, equation_candidate_rows_baseline, list(equation_candidate_rows_baseline[0]))
    if equation_candidate_rows_combined:
        write_csv(equation_combined_csv, equation_candidate_rows_combined, list(equation_candidate_rows_combined[0]))

    baseline_candidate_summary = summarize_candidate_rows(equation_candidate_rows_baseline)
    combined_candidate_summary = summarize_candidate_rows(equation_candidate_rows_combined)
    promotable_equation_after_combined = [
        row
        for row in combined_candidate_summary
        if int(row.get("gains", 0)) > 0
        and int(row.get("losses", 0)) == 0
        and int(row.get("wrong_on_baseline_miss", 0)) == 0
    ]

    decision = {
        "decision": "combined_solver_signal_ready_for_distillation"
        if combined_gains and not combined_losses
        else "combined_solver_signal_blocked",
        "reason": (
            f"combined gains={len(combined_gains)} losses={len(combined_losses)}; "
            f"stride gains={len(stride_gains)} losses={len(stride_losses)}; "
            f"equation_extra_promotable_after_combined={len(promotable_equation_after_combined)}"
        ),
        "submit_note": (
            "V302/V306 postprocessed predictions are local verifier outputs. They are not Kaggle-submit ready "
            "unless the official package allows custom inference; otherwise they must be distilled into an adapter."
        ),
    }

    manifest = {
        "schema_version": "kg1_v306_solver_promotion_gate_v1",
        "input_csv": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "label": args.label,
        "rows": len(source_rows),
        "baseline_summary": baseline_summary,
        "combined_summary": combined_summary,
        "family_delta": family_delta(baseline_summary, combined_summary),
        "combined_gains": len(combined_gains),
        "combined_losses": len(combined_losses),
        "combined_gain_ids": [row["id"] for row in combined_gains],
        "combined_loss_ids": [row["id"] for row in combined_losses],
        "rule_applied": rule_applied,
        "stride_diagnostic": {
            "gains_vs_baseline": len(stride_gains),
            "losses_vs_baseline": len(stride_losses),
            "gain_ids": [row["id"] for row in stride_gains],
            "loss_ids": [row["id"] for row in stride_losses],
            "decision": "diagnostic_only_lossy" if stride_losses else "needs_no_loss_recheck",
        },
        "equation_candidate_summary_baseline": baseline_candidate_summary,
        "equation_candidate_summary_after_combined": combined_candidate_summary,
        "equation_promotable_after_combined": promotable_equation_after_combined,
        "decision": decision,
        "outputs": {
            "audit_csv": str(audit_csv),
            "predictions_csv": str(predictions_csv) if args.write_predictions_csv else "",
            "equation_candidates_baseline_csv": str(equation_baseline_csv) if equation_candidate_rows_baseline else "",
            "equation_candidates_after_combined_csv": str(equation_combined_csv) if equation_candidate_rows_combined else "",
            "manifest_json": str(summary_json),
        },
    }
    write_json(summary_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def self_test() -> None:
    rows = [
        {
            "id": "eq_minus",
            "family": "equation_transform",
            "prompt": (
                "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
                "06-63 = 42\n96-32 = 64\n87-15 = 72\n58-64 = 93\n87-63 = 24\n"
                "Now, determine the result for: 63-19"
            ),
            "answer": "-55",
            "prediction": "55",
            "truncated": "False",
        },
        {
            "id": "other",
            "family": "unit_conversion",
            "prompt": "irrelevant",
            "answer": "1",
            "prediction": "1",
            "truncated": "False",
        },
    ]
    tmp = Path("_v306_self_test.csv")
    out = Path("_v306_self_test_out")
    try:
        write_csv(tmp, rows, list(rows[0]))
        args = argparse.Namespace(input_csv=tmp, output_dir=out, label="self_test", write_predictions_csv=False)
        manifest = run(args)
        if manifest["combined_gains"] != 1 or manifest["combined_losses"] != 0:
            raise AssertionError(f"unexpected self-test manifest: {manifest['decision']}")
    finally:
        if tmp.exists():
            tmp.unlink()
        if out.exists():
            for child in out.glob("*"):
                child.unlink()
            out.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v306_solver_promotion")
    parser.add_argument("--write-predictions-csv", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("v306_solver_promotion_gate_self_test=ok", flush=True)
        return 0
    if args.input_csv is None or args.output_dir is None:
        parser.error("--input-csv and --output-dir are required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

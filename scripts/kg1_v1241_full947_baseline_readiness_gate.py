#!/usr/bin/env python3
"""CPU-only readiness gate for the full947 086 baseline probes.

This gate exists to close the operational gap found after the V1241
baseline-identity probe was added: the probe is only meaningful when the local
workspace has a real full947 gold solution CSV plus two real 086 raw-output
CSVs, one generated with the natural prompt and one with the score-facing
answer-only prompt.

It never trains, uploads, packages adapters, calls external APIs, or submits to
Kaggle. When the three CSVs exist, it can invoke the V1241 baseline probe on
both 086 generations. The natural-prompt probe is diagnostic; the answer-only
probe is the blocking precondition, because it proves whether clean boxing keeps
the 086 public identity while producing strict-clean output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v1241_full947_baseline_readiness_gate"
SELFTEST_OUTPUT_DIR = ROOT / "artifacts" / "v1241_full947_baseline_readiness_gate_selftest"
TRANSFER_GATE = ROOT / "scripts" / "kg1_v1241_bit_equation_transfer_gate.py"
CANONICAL_FULL947_SOLUTION = (
    ROOT
    / "artifacts"
    / "v284_official_gate_worktree"
    / "artifacts"
    / "v1088_unicode_dataset_contract_audit"
    / "hf_cli_download"
    / "runtime_artifacts"
    / "v276_full_eval_bridge"
    / "v276-full947-bridge-20260511T1245Z"
    / "official_train_seed42_stratified10_val.csv"
)
CANONICAL_FULL947_SOLUTION_SHA256 = "84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935"
EXPECTED_ROWS = 947
EXPECTED_PUBLIC_CORRECT = 823
TARGET_089_CORRECT = 843
TARGET_090_CORRECT = 853
EXPECTED_BIT_PUBLIC_CORRECT = 135
EXPECTED_EQUATION_PUBLIC_CORRECT = 56
EXPECTED_PROTECTED_PUBLIC_CORRECT = 632
WEAK_FAMILIES = {"bit_manipulation", "equation_transform"}
ID_FIELDS = ("id", "row_id")
FAMILY_FIELDS = ("family", "type", "task_family")
TOKEN_FIELDS = ("completion_tokens", "output_tokens", "num_tokens", "tokens")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def first_present(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    fieldset = set(fieldnames)
    for candidate in candidates:
        if candidate in fieldset:
            return candidate
    return ""


def inspect_solution_csv(path: Path | None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if path is None:
        path = CANONICAL_FULL947_SOLUTION
        warnings.append("solution_csv_defaulted_to_canonical_full947")
    if not path.exists():
        return {"path": str(path), "exists": False, "errors": [f"solution CSV not found: {path}"], "warnings": warnings}

    try:
        rows, fieldnames = read_csv_rows(path)
    except Exception as exc:  # noqa: BLE001 - gate must report parse failures cleanly.
        return {"path": str(path), "exists": True, "errors": [f"cannot read solution CSV: {exc}"], "warnings": warnings}

    id_field = first_present(fieldnames, ID_FIELDS)
    family_field = first_present(fieldnames, FAMILY_FIELDS)
    if not id_field:
        errors.append("solution CSV must contain id or row_id")
    if "answer" not in fieldnames:
        errors.append("solution CSV must contain answer")
    if not family_field:
        errors.append("solution CSV must contain family/type/task_family for bit/equation/protected gates")
    actual_sha256 = sha256_file(path)
    if path.resolve() == CANONICAL_FULL947_SOLUTION.resolve() and actual_sha256 != CANONICAL_FULL947_SOLUTION_SHA256:
        errors.append(f"canonical solution sha256 mismatch:{actual_sha256}!={CANONICAL_FULL947_SOLUTION_SHA256}")
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"solution row_count_mismatch:{len(rows)}!={EXPECTED_ROWS}")

    ids: list[str] = []
    empty_answers = 0
    duplicate_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    seen: set[str] = set()
    if id_field:
        for row in rows:
            row_id = str(row.get(id_field, "")).strip()
            if not row_id:
                errors.append("solution CSV contains empty id")
                continue
            if row_id in seen:
                duplicate_ids.append(row_id)
            seen.add(row_id)
            ids.append(row_id)
            if not str(row.get("answer", "")).strip():
                empty_answers += 1
            family_counts[str(row.get(family_field, "unknown") or "unknown")] += 1

    if duplicate_ids:
        errors.append(f"solution duplicate ids: {duplicate_ids[:5]}")
    if empty_answers:
        errors.append(f"solution empty answers: {empty_answers}")
    protected_rows = sum(count for family, count in family_counts.items() if family not in WEAK_FAMILIES)
    if protected_rows < EXPECTED_PROTECTED_PUBLIC_CORRECT:
        errors.append(
            f"solution protected rows below 086 reference: {protected_rows}<{EXPECTED_PROTECTED_PUBLIC_CORRECT}"
        )

    return {
        "path": str(path),
        "exists": True,
        "sha256": actual_sha256,
        "canonical_path": str(CANONICAL_FULL947_SOLUTION),
        "canonical_sha256": CANONICAL_FULL947_SOLUTION_SHA256,
        "canonical_match": actual_sha256 == CANONICAL_FULL947_SOLUTION_SHA256,
        "rows": len(rows),
        "columns": fieldnames,
        "id_field": id_field,
        "family_field": family_field,
        "ids": ids,
        "family_counts": dict(sorted(family_counts.items())),
        "protected_rows": protected_rows,
        "errors": errors,
        "warnings": warnings,
    }


def inspect_prediction_csv(path: Path | None, label: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    label_cli = label.replace("_", "-")
    if path is None:
        return {
            "label": label,
            "path": "",
            "exists": False,
            "errors": [f"missing --{label_cli}-baseline-predictions"],
            "warnings": warnings,
        }
    if not path.exists():
        return {"label": label, "path": str(path), "exists": False, "errors": [f"{label} predictions not found: {path}"], "warnings": warnings}

    try:
        rows, fieldnames = read_csv_rows(path)
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "path": str(path), "exists": True, "errors": [f"cannot read {label} predictions: {exc}"], "warnings": warnings}

    id_field = first_present(fieldnames, ID_FIELDS)
    token_field = first_present(fieldnames, TOKEN_FIELDS)
    if not id_field:
        errors.append(f"{label} predictions must contain id or row_id")
    if "raw_output" not in fieldnames:
        errors.append(f"{label} predictions must contain raw_output; prediction-only CSV is not promotion-safe")
    if "finish_reason" not in fieldnames:
        errors.append(f"{label} predictions must contain finish_reason to make truncation gates auditable")
    if not token_field:
        errors.append(f"{label} predictions must contain completion token count for truncation gates")
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"{label} row_count_mismatch:{len(rows)}!={EXPECTED_ROWS}")

    ids: list[str] = []
    duplicate_ids: list[str] = []
    empty_raw_outputs = 0
    finish_reason_counts: Counter[str] = Counter()
    seen: set[str] = set()
    if id_field:
        for row in rows:
            row_id = str(row.get(id_field, "")).strip()
            if not row_id:
                errors.append(f"{label} predictions contain empty id")
                continue
            if row_id in seen:
                duplicate_ids.append(row_id)
            seen.add(row_id)
            ids.append(row_id)
            if not str(row.get("raw_output", "")).strip():
                empty_raw_outputs += 1
            finish_reason_counts[str(row.get("finish_reason", "") or "")] += 1

    if duplicate_ids:
        errors.append(f"{label} duplicate ids: {duplicate_ids[:5]}")
    if empty_raw_outputs:
        errors.append(f"{label} empty raw_output rows: {empty_raw_outputs}")

    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "columns": fieldnames,
        "id_field": id_field,
        "token_field": token_field,
        "ids": ids,
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def compare_id_sets(solution: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    if solution.get("errors") or prediction.get("errors"):
        return []
    solution_ids = set(solution.get("ids", []))
    prediction_ids = set(prediction.get("ids", []))
    missing = sorted(solution_ids - prediction_ids)
    extra = sorted(prediction_ids - solution_ids)
    errors: list[str] = []
    if missing:
        errors.append(f"{prediction['label']} missing solution ids: {missing[:5]}")
    if extra:
        errors.append(f"{prediction['label']} has extra ids not in solution: {extra[:5]}")
    return errors


def probe_command(
    *,
    solution_csv: Path,
    predictions_csv: Path,
    profile: str,
    max_tokens: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRANSFER_GATE),
        "--baseline-identity-probe",
        "--profile",
        profile,
        "--solution-csv",
        str(solution_csv),
        "--baseline-predictions",
        str(predictions_csv),
        "--max-tokens",
        str(max_tokens),
        "--output-dir",
        str(output_dir),
    ]


def run_probe(
    *,
    label: str,
    solution_csv: Path,
    predictions_csv: Path,
    profile: str,
    max_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    command = probe_command(
        solution_csv=solution_csv,
        predictions_csv=predictions_csv,
        profile=profile,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    report_path = output_dir / "kg1_v1241_baseline_identity_probe_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report = {"read_error": str(exc)}
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "report_path": str(report_path),
        "report": report,
        "passed": completed.returncode == 0 and report.get("decision") == "pass_baseline_strict_clean_identity_probe",
        "blockers": report.get("probe_blockers", []),
    }


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KG1 V1241 Full947 Baseline Readiness Gate",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "This CPU-only gate verifies the data needed before using full947 V1241 probes for V1243.",
        "It does not train, upload, submit, or prove leaderboard score.",
        "",
        "## Required Artifacts",
        "",
        f"- full947 solution CSV: `{report['inputs'].get('solution_csv', '')}`",
        f"- solution defaulted to canonical: `{report['inputs'].get('solution_csv_defaulted', False)}`",
        f"- 086 natural raw_output CSV: `{report['inputs'].get('natural_baseline_predictions', '')}`",
        f"- 086 answer-only raw_output CSV: `{report['inputs'].get('answer_only_baseline_predictions', '')}`",
        "",
        "## Score Objective",
        "",
        "- Target remains `>=0.89`, not `0.86`.",
        f"- Baseline reference: `{report['score_objective']['baseline_correct']}/947`.",
        f"- Minimum 0.89 proof: `{report['score_objective']['target_correct']}/947`.",
        f"- Required net gain after baseline readiness: `+{report['score_objective']['additional_correct_required_for_089']}`.",
        f"- Next required candidate gate: `{report['score_objective']['next_required_candidate_gate']}`.",
        "",
        "## Blocking Rule",
        "",
        "- The natural 086 probe is diagnostic; it may fail strict-clean and still be useful.",
        "- The answer-only 086 probe must pass public identity and strict-clean identity before any paid candidate train.",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers", [])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(
        [
            "",
            "## Probe Decisions",
            "",
            f"- Natural: `{report.get('probe_results', {}).get('natural', {}).get('report', {}).get('decision', 'not_run')}`",
            f"- Answer-only: `{report.get('probe_results', {}).get('answer_only', {}).get('report', {}).get('decision', 'not_run')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_csv = args.solution_csv or CANONICAL_FULL947_SOLUTION

    solution = inspect_solution_csv(solution_csv)
    natural = inspect_prediction_csv(args.natural_baseline_predictions, "natural")
    answer_only = inspect_prediction_csv(args.answer_only_baseline_predictions, "answer_only")

    blockers: list[str] = []
    for prefix, inspection in (("solution", solution), ("natural", natural), ("answer_only", answer_only)):
        blockers.extend(f"{prefix}:{error}" for error in inspection.get("errors", []))
    blockers.extend(compare_id_sets(solution, natural))
    blockers.extend(compare_id_sets(solution, answer_only))

    probe_results: dict[str, Any] = {}
    if not blockers and not args.inspect_only:
        probe_results["natural"] = run_probe(
            label="natural",
            solution_csv=solution_csv,
            predictions_csv=args.natural_baseline_predictions,
            profile=args.profile,
            max_tokens=int(args.max_tokens),
            output_dir=output_dir / "probe_086_natural",
        )
        probe_results["answer_only"] = run_probe(
            label="answer_only",
            solution_csv=solution_csv,
            predictions_csv=args.answer_only_baseline_predictions,
            profile=args.profile,
            max_tokens=int(args.max_tokens),
            output_dir=output_dir / "probe_086_answer_only",
        )
        if not probe_results["answer_only"]["passed"]:
            blockers.append("answer_only_086_baseline_identity_probe_not_pass")
    elif not blockers and args.inspect_only:
        blockers.append("probe_execution_skipped_by_inspect_only")

    if blockers:
        decision = "fail_full947_086_baseline_readiness"
    else:
        decision = "pass_full947_086_answer_only_baseline_ready"

    report = {
        "schema_version": "kg1_v1241_full947_baseline_readiness_gate_v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "blockers": blockers,
        "paid_candidate_train_precondition_satisfied": decision == "pass_full947_086_answer_only_baseline_ready",
        "score_claim_authorized": False,
        "kaggle_submit_authorized": False,
        "not_authorized": ["adapter_package", "kaggle_submit", "score_claim"],
        "expected_reference": {
            "rows": EXPECTED_ROWS,
            "public_correct": EXPECTED_PUBLIC_CORRECT,
            "bit_public_correct": EXPECTED_BIT_PUBLIC_CORRECT,
            "equation_public_correct": EXPECTED_EQUATION_PUBLIC_CORRECT,
            "protected_public_correct": EXPECTED_PROTECTED_PUBLIC_CORRECT,
        },
        "score_objective": {
            "target": ">=0.89",
            "target_correct": TARGET_089_CORRECT,
            "target_090_correct": TARGET_090_CORRECT,
            "baseline_correct": EXPECTED_PUBLIC_CORRECT,
            "additional_correct_required_for_089": TARGET_089_CORRECT - EXPECTED_PUBLIC_CORRECT,
            "baseline_readiness_is_not_score_success": True,
            "next_required_candidate_gate": "V1241 full947_089 with candidate >=843/947 and zero regressions",
        },
        "inputs": {
            "solution_csv": str(solution_csv),
            "solution_csv_defaulted": args.solution_csv is None,
            "natural_baseline_predictions": str(args.natural_baseline_predictions)
            if args.natural_baseline_predictions
            else "",
            "answer_only_baseline_predictions": str(args.answer_only_baseline_predictions)
            if args.answer_only_baseline_predictions
            else "",
            "profile": args.profile,
            "max_tokens": int(args.max_tokens),
            "inspect_only": bool(args.inspect_only),
        },
        "inspections": {
            "solution": {key: value for key, value in solution.items() if key != "ids"},
            "natural": {key: value for key, value in natural.items() if key != "ids"},
            "answer_only": {key: value for key, value in answer_only.items() if key != "ids"},
        },
        "probe_results": probe_results,
    }
    report_path = output_dir / "kg1_v1241_full947_baseline_readiness_gate.json"
    summary_path = output_dir / "KG1_V1241_FULL947_BASELINE_READINESS_GATE.md"
    write_json(report_path, report)
    write_markdown_summary(summary_path, report)
    print(json.dumps({"decision": decision, "blockers": blockers, "report": str(report_path)}, indent=2), flush=True)
    return (0 if decision == "pass_full947_086_answer_only_baseline_ready" else 2), report


def make_full947_solution() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(family: str, count: int, prefix: str) -> None:
        start = len(rows)
        for index in range(count):
            rows.append(
                {
                    "id": f"{prefix}_{index:04d}",
                    "prompt": f"{family} fixture {index}",
                    "answer": f"{prefix}_answer_{index:04d}",
                    "family": family,
                    "ordinal": str(start + index),
                }
            )

    add("bit_manipulation", 160, "bit")
    add("equation_transform", 155, "equation")
    add("gravity_constant", 140, "gravity")
    add("unit_conversion", 140, "unit")
    add("numeral_system", 120, "numeral")
    add("text_encryption", 120, "cipher")
    add("logic_grid", 112, "logic")
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"self-test full947 rows mismatch: {len(rows)}")
    return rows


def is_reference_correct_row(row: dict[str, str]) -> bool:
    family = row["family"]
    ordinal = int(row["ordinal"])
    if family == "bit_manipulation":
        return ordinal < EXPECTED_BIT_PUBLIC_CORRECT
    if family == "equation_transform":
        return ordinal < 160 + EXPECTED_EQUATION_PUBLIC_CORRECT
    return family not in WEAK_FAMILIES


def make_predictions(solution: list[dict[str, str]], *, clean_boxed: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in solution:
        value = row["answer"] if is_reference_correct_row(row) else f"wrong_{row['id']}"
        if clean_boxed:
            raw_output = f"</think>\n\\boxed{{{value}}}"
        else:
            raw_output = f"Final answer: {value}"
        rows.append(
            {
                "id": row["id"],
                "raw_output": raw_output,
                "finish_reason": "stop",
                "completion_tokens": "8",
            }
        )
    return rows


def run_self_test(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    fixtures = output_dir / "fixtures"
    solution = make_full947_solution()
    solution_csv = fixtures / "full947_solution.csv"
    natural_csv = fixtures / "086_natural_full947_raw_output.csv"
    answer_only_csv = fixtures / "086_answer_only_full947_raw_output.csv"
    write_csv(solution_csv, solution, ["id", "prompt", "answer", "family"])
    write_csv(natural_csv, make_predictions(solution, clean_boxed=False), ["id", "raw_output", "finish_reason", "completion_tokens"])
    write_csv(answer_only_csv, make_predictions(solution, clean_boxed=True), ["id", "raw_output", "finish_reason", "completion_tokens"])

    cases: list[dict[str, Any]] = []
    canonical_solution = inspect_solution_csv(None)
    canonical_pass = (
        canonical_solution.get("exists") is True
        and canonical_solution.get("rows") == EXPECTED_ROWS
        and canonical_solution.get("canonical_match") is True
        and canonical_solution.get("errors") == []
    )
    cases.append(
        {
            "case": "pass_canonical_full947_solution_inventory",
            "expected_pass": True,
            "observed_pass": canonical_pass,
            "path": canonical_solution.get("path", ""),
            "sha256": canonical_solution.get("sha256", ""),
            "rows": canonical_solution.get("rows", 0),
            "family_counts": canonical_solution.get("family_counts", {}),
            "errors": canonical_solution.get("errors", []),
        }
    )

    pass_args = SimpleNamespace(
        solution_csv=solution_csv,
        natural_baseline_predictions=natural_csv,
        answer_only_baseline_predictions=answer_only_csv,
        profile="full947_089",
        max_tokens=7680,
        output_dir=output_dir / "case_pass_dirty_natural_clean_answer_only",
        inspect_only=False,
    )
    pass_code, pass_report = run_gate(pass_args)
    cases.append(
        {
            "case": "pass_dirty_natural_clean_answer_only",
            "expected_pass": True,
            "observed_pass": pass_code == 0,
            "decision": pass_report["decision"],
            "natural_probe_passed": pass_report["probe_results"]["natural"]["passed"],
            "answer_only_probe_passed": pass_report["probe_results"]["answer_only"]["passed"],
            "blockers": pass_report["blockers"],
        }
    )

    missing_args = SimpleNamespace(
        solution_csv=solution_csv,
        natural_baseline_predictions=natural_csv,
        answer_only_baseline_predictions=fixtures / "missing_answer_only.csv",
        profile="full947_089",
        max_tokens=7680,
        output_dir=output_dir / "case_fail_missing_answer_only",
        inspect_only=False,
    )
    missing_code, missing_report = run_gate(missing_args)
    cases.append(
        {
            "case": "fail_missing_answer_only_artifact",
            "expected_pass": False,
            "observed_pass": missing_code == 0,
            "decision": missing_report["decision"],
            "blockers": missing_report["blockers"],
        }
    )

    decision = (
        "pass_v1241_full947_baseline_readiness_self_test"
        if all(case["expected_pass"] == case["observed_pass"] for case in cases)
        else "fail"
    )
    report = {
        "schema_version": "kg1_v1241_full947_baseline_readiness_selftest_v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "cases": cases,
    }
    report_path = output_dir / "kg1_v1241_full947_baseline_readiness_gate_selftest.json"
    write_json(report_path, report)
    print(json.dumps({"decision": decision, "report": str(report_path), "cases": cases}, indent=2), flush=True)
    return 0 if decision.startswith("pass_") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution-csv",
        type=Path,
        help="Defaults to the canonical local v276 full947 solution CSV when omitted.",
    )
    parser.add_argument("--natural-baseline-predictions", type=Path)
    parser.add_argument("--answer-only-baseline-predictions", type=Path)
    parser.add_argument("--profile", choices=("full947_089", "full947_090"), default="full947_089")
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inspect-only", action="store_true", help="Validate files but do not run V1241 probes.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic readiness self-tests.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_self_test(args.output_dir if args.output_dir != DEFAULT_OUTPUT_DIR else SELFTEST_OUTPUT_DIR)
    exit_code, _report = run_gate(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

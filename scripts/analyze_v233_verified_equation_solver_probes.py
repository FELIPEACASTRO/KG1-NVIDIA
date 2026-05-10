#!/usr/bin/env python3
"""Run conservative verified equation solver probes over V232 work items.

This script is CPU-only. It consumes the V232 verified solver workbench
manifest and tests deployable deterministic probes against the equation
workitems. It also records non-deployable oracle-candidate evidence separately.
It does not train, run model generation, run full scoring, package artifacts, or
submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
PROBE_RESULT_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "solver_route",
    "probe_name",
    "deployable",
    "status",
    "prediction",
    "expected_answer",
    "baseline_prediction",
    "correct_alternative_count",
    "prompt_sha256",
    "proof",
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
                raise ValueError(f"{path}:{line_no}: invalid jsonl row: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.strip("` ")
    text = re.sub(r"\s+", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text


def answers_equal(left: Any, right: Any) -> bool:
    lhs = normalize_answer(left)
    rhs = normalize_answer(right)
    if lhs == rhs:
        return True
    try:
        return Fraction(lhs) == Fraction(rhs)
    except Exception:
        return False


def format_number(value: Any) -> str:
    try:
        number = float(value)
        if math.isfinite(number) and abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
    except Exception:
        pass
    return str(value).strip()


def import_sympy() -> Any | None:
    try:
        import sympy as sp  # type: ignore

        return sp
    except Exception:
        return None


def extract_single_equation(prompt: str) -> tuple[str | None, str]:
    # Conservative extraction: accept only one short algebraic equality over a
    # single-letter variable. Anything noisier abstains.
    candidates = re.findall(
        r"([A-Za-z0-9_().+\-*/^ ]{1,80}=[A-Za-z0-9_().+\-*/^ ]{1,80})",
        prompt,
    )
    cleaned: list[str] = []
    for candidate in candidates:
        item = re.sub(r"\s+", " ", candidate).strip(" .,:;`")
        if not item or item.count("=") != 1:
            continue
        if re.search(r"[A-Za-z]", item) and re.search(r"\d", item):
            cleaned.append(item)
    unique = sorted(set(cleaned))
    if not unique:
        return None, "no_single_algebraic_equation_found"
    if len(unique) > 1:
        return None, "ambiguous_multiple_equations"
    return unique[0], "ok"


def solve_single_equation_with_sympy(prompt: str) -> dict[str, Any]:
    equation, reason = extract_single_equation(prompt)
    if not equation:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": reason,
        }
    sp = import_sympy()
    if sp is None:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "sympy_unavailable",
        }
    variables = sorted(set(re.findall(r"\b([a-zA-Z])\b", equation)))
    if len(variables) != 1:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "variable_count_not_one",
        }
    var = sp.symbols(variables[0])
    lhs_text, rhs_text = equation.split("=", 1)
    try:
        lhs = sp.sympify(lhs_text.replace("^", "**"))
        rhs = sp.sympify(rhs_text.replace("^", "**"))
        solutions = sp.solve(sp.Eq(lhs, rhs), var)
    except Exception as exc:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "sympy_parse_or_solve_failed: " + repr(exc),
        }
    if len(solutions) != 1:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "solution_count_not_one",
        }
    solution = solutions[0]
    try:
        verified = sp.simplify(lhs.subs(var, solution) - rhs.subs(var, solution)) == 0
    except Exception:
        verified = False
    if not verified:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "solution_failed_symbolic_verification",
        }
    return {
        "probe_name": "sympy_single_equation_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": format_number(solution),
        "proof": f"solved {equation} for {variables[0]} and verified substitution",
    }


def nondeployable_oracle_alt_probe(item: dict[str, Any]) -> dict[str, Any]:
    alternatives = as_list(item.get("pairwise_unique_gain_predictions"))
    expected = item.get("expected_answer", "")
    matches = [candidate for candidate in alternatives if answers_equal(candidate, expected)]
    if not matches:
        return {
            "probe_name": "oracle_alternative_candidate_probe",
            "deployable": False,
            "status": "abstain",
            "prediction": "",
            "proof": "no alternative candidate matched expected answer",
        }
    return {
        "probe_name": "oracle_alternative_candidate_probe",
        "deployable": False,
        "status": "verified",
        "prediction": matches[0],
        "proof": "non-deployable evidence from prior candidate prediction; labels are used only for analysis",
    }


def evaluate_probe_result(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected_answer", "")
    prediction = result.get("prediction", "")
    if result.get("status") == "candidate":
        status = "verified" if answers_equal(prediction, expected) else "incorrect"
    else:
        status = str(result.get("status", "abstain"))
    return {
        "schema_version": "kg1_v233_equation_probe_result_v1",
        "id": str(item.get("id", "")),
        "family": str(item.get("family", "")),
        "solver_route": str(item.get("solver_route", "")),
        "probe_name": str(result.get("probe_name", "")),
        "deployable": bool(result.get("deployable", False)),
        "status": status,
        "prediction": str(prediction),
        "expected_answer": str(expected),
        "baseline_prediction": str(item.get("baseline_prediction", "")),
        "correct_alternative_count": int(item.get("correct_alternative_count", 0) or 0),
        "prompt_sha256": str(item.get("prompt_sha256", "")),
        "proof": str(result.get("proof", "")),
    }


def run_probes(equation_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in equation_items:
        deployable = solve_single_equation_with_sympy(str(item.get("prompt", "")))
        results.append(evaluate_probe_result(item, deployable))
        oracle = nondeployable_oracle_alt_probe(item)
        results.append(evaluate_probe_result(item, oracle))
    return results


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    counts: Counter[tuple[str, bool, str]] = Counter()
    for row in results:
        counts[(str(row["probe_name"]), bool(row["deployable"]), str(row["status"]))] += 1
    rows = [
        {
            "probe_name": probe,
            "deployable": deployable,
            "status": status,
            "rows": count,
        }
        for (probe, deployable, status), count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def load_v232_paths(v232_manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = v232_manifest.get("outputs", {})
    required = {
        "equation_solver_workitems_jsonl",
        "bit_guardrail_workitems_jsonl",
        "acceptance_matrix_csv",
        "solver_contracts_json",
    }
    missing = sorted(name for name in required if not outputs.get(name))
    if missing:
        raise RuntimeError("V232 manifest missing outputs: " + json.dumps(missing))
    paths = {name: Path(str(outputs[name])) for name in required}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return paths


def shared_row_contract_from_v232_manifest(v232_manifest: dict[str, Any]) -> str:
    inputs = v232_manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    return str(
        inputs.get("observed_shared_row_contract_sha256")
        or v232_manifest.get("observed_shared_row_contract_sha256")
        or inputs.get("expected_shared_row_contract_sha256")
        or v232_manifest.get("expected_shared_row_contract_sha256")
        or ""
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V233 VERIFIED EQUATION SOLVER PROBES SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v232-analysis-manifest-json =", args.v232_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    v232_manifest = read_json(args.v232_analysis_manifest_json)
    observed_contract = shared_row_contract_from_v232_manifest(v232_manifest)
    if args.expected_shared_row_contract_sha256 and observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    paths = load_v232_paths(v232_manifest)
    equation_items = read_jsonl(paths["equation_solver_workitems_jsonl"])
    bit_items = read_jsonl(paths["bit_guardrail_workitems_jsonl"])
    if not equation_items:
        raise RuntimeError("V233 requires non-empty equation workitems")

    results = run_probes(equation_items)
    summary = summarize_results(results)
    deployable_verified = [
        row for row in results if bool(row["deployable"]) and row["status"] == "verified"
    ]
    nondeployable_verified = [
        row for row in results if not bool(row["deployable"]) and row["status"] == "verified"
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "equation_probe_results_jsonl": args.output_dir / f"{prefix}_equation_probe_results.jsonl",
        "equation_probe_summary_csv": args.output_dir / f"{prefix}_equation_probe_summary.csv",
        "equation_verified_overrides_csv": args.output_dir / f"{prefix}_equation_verified_overrides.csv",
        "equation_oracle_evidence_csv": args.output_dir / f"{prefix}_equation_oracle_evidence.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_jsonl(out_paths["equation_probe_results_jsonl"], results)
    summary.to_csv(out_paths["equation_probe_summary_csv"], index=False)
    pd.DataFrame(deployable_verified, columns=PROBE_RESULT_COLUMNS).to_csv(
        out_paths["equation_verified_overrides_csv"],
        index=False,
    )
    pd.DataFrame(nondeployable_verified, columns=PROBE_RESULT_COLUMNS).to_csv(
        out_paths["equation_oracle_evidence_csv"],
        index=False,
    )

    verified_count = len(deployable_verified)
    oracle_count = len(nondeployable_verified)
    if verified_count >= args.equation_target_gain:
        decision = {
            "decision": "prepare_gated_solver_rescue_eval",
            "reason": f"deployable_verified_equation_overrides={verified_count} >= target_gain={args.equation_target_gain}",
            "next_action": "Create a separate gated rescue-eval notebook that applies verified overrides and runs weak eval only.",
        }
    else:
        decision = {
            "decision": "improve_solver_parsers_before_eval",
            "reason": (
                f"deployable_verified_equation_overrides={verified_count}; "
                f"target_gain={args.equation_target_gain}; nondeployable_oracle_evidence={oracle_count}"
            ),
            "next_action": "Inspect V233 abstentions and extend route-specific parsers before any weak/full eval.",
        }

    manifest = {
        "schema_version": "kg1_v233_verified_equation_solver_probes_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v232_analysis_manifest_json": str(args.v232_analysis_manifest_json),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "observed_shared_row_contract_sha256": observed_contract,
            **{name: str(path) for name, path in sorted(paths.items())},
        },
        "input_artifact_hashes": {
            "v232_analysis_manifest_json": file_meta(args.v232_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(paths.items())},
        },
        "probe_counts": {
            "equation_workitems": int(len(equation_items)),
            "bit_guardrail_workitems": int(len(bit_items)),
            "deployable_verified_equation_overrides": int(verified_count),
            "nondeployable_oracle_evidence_rows": int(oracle_count),
            "target_gain": int(args.equation_target_gain),
        },
        "probe_summary": summary.to_dict(orient="records"),
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)

    print("probe_counts =", json.dumps(manifest["probe_counts"], sort_keys=True), flush=True)
    print("probe_summary =", summary.to_string(index=False), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V233 VERIFIED EQUATION SOLVER PROBES SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-analysis-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v233_verified_equation_solver_probes")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--equation-target-gain", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        equation_items = root / "equation.jsonl"
        bit_items = root / "bit.jsonl"
        acceptance = root / "acceptance.csv"
        contracts = root / "contracts.json"
        write_jsonl(
            equation_items,
            [
                {
                    "schema_version": "kg1_v232_solver_workitem_v1",
                    "id": "eq1",
                    "family": "equation_transform",
                    "solver_route": "sympy_symbolic_transform",
                    "expected_answer": "42",
                    "baseline_prediction": "41",
                    "correct_alternative_count": 1,
                    "pairwise_unique_gain_predictions": ["42"],
                    "prompt_sha256": "dummy",
                    "prompt": "Solve for x: x + 1 = 43.",
                }
            ],
        )
        write_jsonl(bit_items, [{"id": "bit1", "family": "bit_manipulation"}])
        write_csv(acceptance, [{"area": "equation_transform", "criterion": "minimum_verified_gain"}])
        write_json(contracts, {"schema_version": "kg1_v232_solver_contracts_v1"})
        v232_manifest = root / "v232_manifest.json"
        write_json(
            v232_manifest,
            {
                "expected_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "outputs": {
                    "equation_solver_workitems_jsonl": str(equation_items),
                    "bit_guardrail_workitems_jsonl": str(bit_items),
                    "acceptance_matrix_csv": str(acceptance),
                    "solver_contracts_json": str(contracts),
                },
            },
        )
        args = argparse.Namespace(
            v232_analysis_manifest_json=v232_manifest,
            output_dir=out_dir,
            label="v233_verified_equation_solver_probes",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            equation_target_gain=1,
        )
        manifest = run_analysis(args)
        if manifest["probe_counts"]["deployable_verified_equation_overrides"] != 1:
            raise AssertionError("expected one verified equation override")
        verified_columns = list(pd.read_csv(manifest["outputs"]["equation_verified_overrides_csv"]).columns)
        oracle_columns = list(pd.read_csv(manifest["outputs"]["equation_oracle_evidence_csv"]).columns)
        if verified_columns != PROBE_RESULT_COLUMNS:
            raise AssertionError("verified override CSV schema drifted")
        if oracle_columns != PROBE_RESULT_COLUMNS:
            raise AssertionError("oracle evidence CSV schema drifted")
    print("v233_verified_equation_solver_probes_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

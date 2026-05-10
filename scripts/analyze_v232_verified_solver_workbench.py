#!/usr/bin/env python3
"""Build a verified solver workbench from V231 miss-pack mining outputs.

This script is CPU-only. It consumes the V231 manifest, reloads the V230
source miss packs for full prompts, and emits work items and acceptance
contracts for deterministic equation/bit solver development. It does not train,
run model generation, run full scoring, package artifacts, or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def split_semicolon(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def compact_text(value: object, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def route_acceptance_contract(family: str, route: str) -> list[str]:
    if family == "equation_transform":
        if route == "sympy_symbolic_transform":
            return [
                "Parse every in-prompt example before proposing an override.",
                "Use symbolic simplification/solve only after variables and target are unambiguous.",
                "Normalize candidate and expected answer with an exact verifier.",
                "Abstain on parse ambiguity, multiple valid forms, or verifier disagreement.",
            ]
        if route == "constraint_cryptarithm":
            return [
                "Enforce digit-letter uniqueness and leading-zero constraints.",
                "Validate every equation or example from the prompt.",
                "Accept only a unique verified assignment.",
                "Abstain when more than one assignment is valid.",
            ]
        return [
            "Create a route-specific parser before proposing an override.",
            "Verify all prompt examples locally.",
            "Keep V226 baseline on ambiguity.",
        ]
    if route == "bitwise_named_operator_dsl":
        return [
            "Parse named operations before searching expressions.",
            "Enumerate only allowed bitwise operators from the prompt.",
            "Verify every in-prompt example with fixed bit width.",
            "Never override a V226-correct bit row without a proof.",
        ]
    return [
        "Use bit solver as guardrail first.",
        "Verify all examples locally.",
        "Keep V226 baseline on ambiguity.",
    ]


def build_workitems(
    taxonomy: pd.DataFrame,
    miss_pack: pd.DataFrame,
    family: str,
    target_gain: int,
) -> list[dict[str, Any]]:
    require_columns(
        taxonomy,
        {
            "id",
            "solver_route",
            "answer",
            "baseline_prediction",
            "correct_alternative_count",
            "correct_alternative_candidates",
            "pairwise_gain_count",
            "pairwise_gain_candidates",
            "pairwise_unique_gain_predictions",
        },
        f"{family}_taxonomy",
    )
    require_columns(
        miss_pack,
        {
            "id",
            "family",
            "answer",
            "baseline_prediction",
            "baseline_correct",
            "baseline_truncated",
            "correct_alternative_count",
            "correct_alternative_candidates",
            "prompt",
        },
        f"{family}_miss_pack",
    )
    bad_family = miss_pack[~miss_pack["family"].eq(family)]
    if len(bad_family):
        raise RuntimeError(f"{family}_miss_pack contains wrong family rows")

    full_by_id = {str(row.id): row for row in miss_pack.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for item in taxonomy.itertuples(index=False):
        row_id = str(getattr(item, "id"))
        if row_id not in full_by_id:
            raise KeyError(f"{family} taxonomy id missing from V230 miss pack: {row_id}")
        source = full_by_id[row_id]
        route = str(getattr(item, "solver_route"))
        alt_count = as_int(getattr(item, "correct_alternative_count"))
        gain_count = as_int(getattr(item, "pairwise_gain_count"))
        priority_score = alt_count * 100 + gain_count * 10
        if route in {"sympy_symbolic_transform", "bitwise_named_operator_dsl"}:
            priority_score += 5
        prompt = str(getattr(source, "prompt"))
        rows.append(
            {
                "schema_version": "kg1_v232_solver_workitem_v1",
                "id": row_id,
                "family": family,
                "solver_route": route,
                "priority_score": int(priority_score),
                "target_gain": int(target_gain),
                "expected_answer": str(getattr(source, "answer")),
                "baseline_prediction": str(getattr(source, "baseline_prediction")),
                "baseline_correct": str(getattr(source, "baseline_correct")),
                "baseline_truncated": str(getattr(source, "baseline_truncated")),
                "correct_alternative_count": alt_count,
                "correct_alternative_candidates": split_semicolon(getattr(item, "correct_alternative_candidates")),
                "pairwise_gain_count": gain_count,
                "pairwise_gain_candidates": split_semicolon(getattr(item, "pairwise_gain_candidates")),
                "pairwise_unique_gain_predictions": split_semicolon(getattr(item, "pairwise_unique_gain_predictions")),
                "prompt_sha256": prompt_sha256(prompt),
                "prompt_excerpt": compact_text(prompt),
                "prompt": prompt,
                "acceptance_contract": route_acceptance_contract(family, route),
                "verification_status": "not_implemented",
                "override_allowed_before_verifier": False,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["priority_score"]), row["solver_route"], row["id"]))


def acceptance_matrix_rows(
    equation_items: list[dict[str, Any]],
    bit_items: list[dict[str, Any]],
    equation_target_gain: int,
    bit_guardrail_min: int,
) -> list[dict[str, Any]]:
    equation_alt = sum(1 for row in equation_items if int(row["correct_alternative_count"]) > 0)
    bit_alt = sum(1 for row in bit_items if int(row["correct_alternative_count"]) > 0)
    return [
        {
            "area": "equation_transform",
            "criterion": "minimum_verified_gain",
            "required": equation_target_gain,
            "observed_candidate_rows": equation_alt,
            "status": "blocked_until_solver_verified",
            "evidence": "V231 taxonomy rows with correct alternative candidate.",
        },
        {
            "area": "equation_transform",
            "criterion": "no_unverified_override",
            "required": 1,
            "observed_candidate_rows": 0,
            "status": "mandatory",
            "evidence": "Override is forbidden until local parser/verifier proves the row.",
        },
        {
            "area": "bit_manipulation",
            "criterion": "guardrail_minimum",
            "required": bit_guardrail_min,
            "observed_candidate_rows": bit_alt,
            "status": "guardrail",
            "evidence": "Bit work items exist only to preserve or prove no-loss overrides.",
        },
        {
            "area": "release",
            "criterion": "train_full_package_submit",
            "required": 0,
            "observed_candidate_rows": 0,
            "status": "blocked",
            "evidence": "V232 is a CPU-only solver workbench.",
        },
    ]


def load_required_v231_outputs(v231_manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = v231_manifest.get("outputs", {})
    required = {
        "equation_miss_taxonomy_csv",
        "bit_miss_taxonomy_csv",
        "equation_solver_candidate_rules_json",
        "bit_guardrail_candidates_json",
    }
    missing = sorted(name for name in required if not outputs.get(name))
    if missing:
        raise RuntimeError("V231 manifest missing outputs: " + json.dumps(missing))
    paths = {name: Path(str(outputs[name])) for name in required}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return paths


def load_v230_miss_pack_paths(v231_manifest: dict[str, Any]) -> dict[str, Path]:
    inputs = v231_manifest.get("inputs", {})
    v230_path = Path(str(inputs.get("v230_analysis_manifest_json", "")))
    v230_manifest = read_json(v230_path)
    outputs = v230_manifest.get("outputs", {})
    required = {"equation_miss_pack_csv", "bit_miss_pack_csv", "pairwise_detail_csv"}
    missing = sorted(name for name in required if not outputs.get(name))
    if missing:
        raise RuntimeError("V230 manifest missing outputs: " + json.dumps(missing))
    paths = {name: Path(str(outputs[name])) for name in required}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return {"v230_analysis_manifest_json": v230_path, **paths}


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V232 VERIFIED SOLVER WORKBENCH SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v231_analysis_manifest_json =", args.v231_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    v231_manifest = read_json(args.v231_analysis_manifest_json)
    observed_contract = str(v231_manifest.get("expected_shared_row_contract_sha256", ""))
    if args.expected_shared_row_contract_sha256 and observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    v231_paths = load_required_v231_outputs(v231_manifest)
    v230_paths = load_v230_miss_pack_paths(v231_manifest)

    equation_taxonomy = read_csv(v231_paths["equation_miss_taxonomy_csv"])
    bit_taxonomy = read_csv(v231_paths["bit_miss_taxonomy_csv"])
    equation_miss_pack = read_csv(v230_paths["equation_miss_pack_csv"])
    bit_miss_pack = read_csv(v230_paths["bit_miss_pack_csv"])

    equation_items = build_workitems(
        equation_taxonomy,
        equation_miss_pack,
        "equation_transform",
        args.equation_target_gain,
    )
    bit_items = build_workitems(
        bit_taxonomy,
        bit_miss_pack,
        "bit_manipulation",
        0,
    )
    acceptance_rows = acceptance_matrix_rows(
        equation_items,
        bit_items,
        args.equation_target_gain,
        args.bit_guardrail_min,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "equation_solver_workitems_jsonl": args.output_dir / f"{prefix}_equation_solver_workitems.jsonl",
        "bit_guardrail_workitems_jsonl": args.output_dir / f"{prefix}_bit_guardrail_workitems.jsonl",
        "acceptance_matrix_csv": args.output_dir / f"{prefix}_acceptance_matrix.csv",
        "solver_contracts_json": args.output_dir / f"{prefix}_solver_contracts.json",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_jsonl(out_paths["equation_solver_workitems_jsonl"], equation_items)
    write_jsonl(out_paths["bit_guardrail_workitems_jsonl"], bit_items)
    pd.DataFrame(acceptance_rows).to_csv(out_paths["acceptance_matrix_csv"], index=False)

    equation_alt_rows = sum(1 for row in equation_items if int(row["correct_alternative_count"]) > 0)
    bit_alt_rows = sum(1 for row in bit_items if int(row["correct_alternative_count"]) > 0)
    contracts = {
        "schema_version": "kg1_v232_solver_contracts_v1",
        "generated_at_utc": utc_now(),
        "contracts": {
            "equation_transform": {
                "target_gain_minimum": int(args.equation_target_gain),
                "available_alternative_rows": int(equation_alt_rows),
                "must_abstain_without_verifier": True,
                "routes": sorted(set(row["solver_route"] for row in equation_items)),
                "acceptance_contract": route_acceptance_contract("equation_transform", "sympy_symbolic_transform"),
            },
            "bit_manipulation": {
                "guardrail_minimum": int(args.bit_guardrail_min),
                "available_alternative_rows": int(bit_alt_rows),
                "must_preserve_v226_correct_rows": True,
                "routes": sorted(set(row["solver_route"] for row in bit_items)),
                "acceptance_contract": route_acceptance_contract("bit_manipulation", "bitwise_named_operator_dsl"),
            },
        },
        "blocked_actions": ["train", "full_scoring", "package", "kaggle_submit"],
        "next_action": "Implement V233 verified equation solver probes against the V232 workitems; do not train yet.",
    }
    write_json(out_paths["solver_contracts_json"], contracts)

    decision = {
        "decision": "build_v233_verified_equation_solver_probes",
        "reason": (
            f"equation_workitems={len(equation_items)}; equation_alt_rows={equation_alt_rows}; "
            f"target_gain={args.equation_target_gain}; bit_workitems={len(bit_items)}; bit_alt_rows={bit_alt_rows}"
        ),
        "next_action": "Use V232 workitems to implement route-specific verified solver probes before any training or full scoring.",
    }
    manifest = {
        "schema_version": "kg1_v232_verified_solver_workbench_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v231_analysis_manifest_json": str(args.v231_analysis_manifest_json),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "observed_shared_row_contract_sha256": observed_contract,
            **{name: str(path) for name, path in sorted(v231_paths.items())},
            **{name: str(path) for name, path in sorted(v230_paths.items())},
        },
        "input_artifact_hashes": {
            "v231_analysis_manifest_json": file_meta(args.v231_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(v231_paths.items())},
            **{name: file_meta(path) for name, path in sorted(v230_paths.items())},
        },
        "workitem_counts": {
            "equation_solver_workitems": int(len(equation_items)),
            "equation_rows_with_correct_alternative": int(equation_alt_rows),
            "bit_guardrail_workitems": int(len(bit_items)),
            "bit_rows_with_correct_alternative": int(bit_alt_rows),
        },
        "top_equation_workitems": [
            {key: row[key] for key in ["id", "solver_route", "priority_score", "correct_alternative_count", "pairwise_gain_count", "prompt_sha256"]}
            for row in equation_items[:10]
        ],
        "top_bit_workitems": [
            {key: row[key] for key in ["id", "solver_route", "priority_score", "correct_alternative_count", "pairwise_gain_count", "prompt_sha256"]}
            for row in bit_items[:10]
        ],
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
    }
    write_json(out_paths["manifest_json"], manifest)

    print("workitem_counts =", json.dumps(manifest["workitem_counts"], sort_keys=True), flush=True)
    print("acceptance_matrix =", pd.DataFrame(acceptance_rows).to_string(index=False), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V232 VERIFIED SOLVER WORKBENCH SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v231-analysis-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v232_verified_solver_workbench")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--equation-target-gain", type=int, default=5)
    parser.add_argument("--bit-guardrail-min", type=int, default=136)
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
        v230_dir = root / "v230"
        v231_dir = root / "v231"
        out_dir = root / "out"
        v230_dir.mkdir()
        v231_dir.mkdir()

        equation_miss = v230_dir / "equation_miss.csv"
        bit_miss = v230_dir / "bit_miss.csv"
        pairwise = v230_dir / "pairwise.csv"
        write_csv(
            equation_miss,
            [
                {
                    "id": "eq1",
                    "family": "equation_transform",
                    "answer": "42",
                    "baseline_prediction": "41",
                    "baseline_correct": "False",
                    "baseline_truncated": "False",
                    "correct_alternative_count": "1",
                    "correct_alternative_candidates": "candidate_a",
                    "prompt": "Solve symbolic equation x + 1 = 43. Return x.",
                }
            ],
        )
        write_csv(
            bit_miss,
            [
                {
                    "id": "bit1",
                    "family": "bit_manipulation",
                    "answer": "7",
                    "baseline_prediction": "3",
                    "baseline_correct": "False",
                    "baseline_truncated": "False",
                    "correct_alternative_count": "1",
                    "correct_alternative_candidates": "candidate_b",
                    "prompt": "Apply xor and mask examples, then return the result.",
                }
            ],
        )
        write_csv(pairwise, [{"id": "eq1", "candidate": "candidate_a"}])
        v230_manifest = v230_dir / "v230_manifest.json"
        write_json(
            v230_manifest,
            {
                "outputs": {
                    "equation_miss_pack_csv": str(equation_miss),
                    "bit_miss_pack_csv": str(bit_miss),
                    "pairwise_detail_csv": str(pairwise),
                }
            },
        )

        equation_tax = v231_dir / "equation_tax.csv"
        bit_tax = v231_dir / "bit_tax.csv"
        write_csv(
            equation_tax,
            [
                {
                    "id": "eq1",
                    "solver_route": "sympy_symbolic_transform",
                    "answer": "42",
                    "baseline_prediction": "41",
                    "correct_alternative_count": "1",
                    "correct_alternative_candidates": "candidate_a",
                    "pairwise_gain_count": "1",
                    "pairwise_gain_candidates": "candidate_a",
                    "pairwise_unique_gain_predictions": "42",
                }
            ],
        )
        write_csv(
            bit_tax,
            [
                {
                    "id": "bit1",
                    "solver_route": "bitwise_named_operator_dsl",
                    "answer": "7",
                    "baseline_prediction": "3",
                    "correct_alternative_count": "1",
                    "correct_alternative_candidates": "candidate_b",
                    "pairwise_gain_count": "1",
                    "pairwise_gain_candidates": "candidate_b",
                    "pairwise_unique_gain_predictions": "7",
                }
            ],
        )
        eq_rules = v231_dir / "eq_rules.json"
        bit_rules = v231_dir / "bit_rules.json"
        write_json(eq_rules, {"schema_version": "kg1_v231_equation_solver_candidate_rules_v1"})
        write_json(bit_rules, {"schema_version": "kg1_v231_bit_guardrail_candidates_v1"})
        v231_manifest = v231_dir / "v231_manifest.json"
        write_json(
            v231_manifest,
            {
                "expected_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "inputs": {"v230_analysis_manifest_json": str(v230_manifest)},
                "outputs": {
                    "equation_miss_taxonomy_csv": str(equation_tax),
                    "bit_miss_taxonomy_csv": str(bit_tax),
                    "equation_solver_candidate_rules_json": str(eq_rules),
                    "bit_guardrail_candidates_json": str(bit_rules),
                },
            },
        )
        args = argparse.Namespace(
            v231_analysis_manifest_json=v231_manifest,
            output_dir=out_dir,
            label="v232_verified_solver_workbench",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            equation_target_gain=5,
            bit_guardrail_min=136,
        )
        manifest = run_analysis(args)
        if manifest["workitem_counts"]["equation_solver_workitems"] != 1:
            raise AssertionError("equation workitem count mismatch")
        if not Path(manifest["outputs"]["equation_solver_workitems_jsonl"]).exists():
            raise AssertionError("equation workitems missing")
    print("v232_verified_solver_workbench_self_test=ok", flush=True)
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

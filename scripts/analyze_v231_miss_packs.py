#!/usr/bin/env python3
"""Mine V230 miss packs for equation/bit rescue opportunities.

This script is CPU-only. It consumes the V230 complementarity manifest and its
CSV outputs, classifies baseline misses, and emits a concrete next-action
taxonomy for verified solver/router work. It does not train, evaluate a model,
package artifacts, or submit to Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def bool_text(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y", "t"}


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def compact_text(value: object, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def split_candidates(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def prompt_features(prompt: str) -> dict[str, Any]:
    lower = prompt.lower()
    examples = len(re.findall(r"(?i)\b(?:example|input|output|->|=>|maps? to|becomes)\b", prompt))
    return {
        "chars": len(prompt),
        "digits": len(re.findall(r"\d", prompt)),
        "operators": len(re.findall(r"[+\-*/^=<>%]", prompt)),
        "variables": len(re.findall(r"\b[a-zA-Z]\b", prompt)),
        "boxed_mentions": lower.count("boxed"),
        "example_markers": examples,
        "has_modular_language": any(token in lower for token in ["modulo", "mod ", "remainder", "xor", "binary"]),
        "has_symbolic_language": any(token in lower for token in ["symbolic", "expression", "polynomial", "equation", "formula"]),
        "has_table_language": any(token in lower for token in ["table", "mapping", "maps to", "pairs"]),
        "has_cryptarithm_language": any(token in lower for token in ["letter", "digit", "cryptarithm", "alphametic"]),
        "has_sequence_language": any(token in lower for token in ["sequence", "next", "pattern", "series"]),
    }


def classify_equation_route(prompt: str, answer: str) -> tuple[str, str]:
    features = prompt_features(prompt)
    lower = prompt.lower()
    answer_text = str(answer).strip()
    if features["has_cryptarithm_language"]:
        return "constraint_cryptarithm", "DFS/constraint solver with digit-letter uniqueness and prompt-example verification"
    if features["has_sequence_language"]:
        return "numeric_sequence_hypothesis", "integer sequence/formula search with holdout example verification"
    if features["has_table_language"]:
        return "program_by_example_mapping", "small DSL over observed input/output pairs, accept only if all pairs are explained"
    if features["has_symbolic_language"] or re.search(r"\b[x-z]\b", lower):
        return "sympy_symbolic_transform", "SymPy simplify/solve/expand/factor route with exact-string verifier"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer_text) or features["digits"] >= 8:
        return "numeric_formula_search", "enumerate arithmetic hypotheses and constants, accept only with full prompt consistency"
    return "equation_pattern_unknown", "manual taxonomy needed before automation"


def classify_bit_route(prompt: str) -> tuple[str, str]:
    lower = prompt.lower()
    if any(token in lower for token in ["xor", "and", "or", "shift", "rotate", "mask"]):
        return "bitwise_named_operator_dsl", "prioritize named operations from prompt, then verify all examples"
    if any(token in lower for token in ["binary", "bit", "bits"]):
        return "bitvector_superoptimizer", "enumerate xor/and/or/not/shift/rotate/mask expressions with simplicity ranking"
    return "bit_pattern_unknown", "manual taxonomy needed before bit override"


def aggregate_pairwise_gains(pairwise_detail: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"id", "candidate", "candidate_prediction", "candidate_correct", "candidate_truncated", "gained_vs_baseline"}
    missing = sorted(required - set(pairwise_detail.columns))
    if missing:
        raise ValueError(f"pairwise detail missing columns: {missing}")
    gains: dict[str, dict[str, Any]] = {}
    for row_id, group in pairwise_detail.groupby("id", sort=False):
        gained = group[
            group["gained_vs_baseline"].map(bool_text)
            & group["candidate_correct"].map(bool_text)
            & ~group["candidate_truncated"].map(bool_text)
        ].copy()
        gains[str(row_id)] = {
            "gain_candidates": gained["candidate"].astype(str).tolist(),
            "gain_predictions": gained["candidate_prediction"].astype(str).tolist(),
            "unique_gain_predictions": sorted(set(gained["candidate_prediction"].astype(str).tolist())),
            "gain_count": int(len(gained)),
        }
    return gains


def validate_miss_pack(frame: pd.DataFrame, expected_family: str, label: str) -> None:
    required = {
        "id",
        "family",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "baseline_truncated",
        "correct_alternative_count",
        "correct_alternative_candidates",
        "prompt",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    bad_family = frame[~frame["family"].eq(expected_family)]
    if len(bad_family):
        raise RuntimeError(f"{label} contains non-{expected_family} rows: {bad_family[['id', 'family']].head(5).to_dict(orient='records')}")
    duplicate_ids = int(frame["id"].duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(f"{label} has duplicate ids: {duplicate_ids}")


def taxonomy_rows(miss_pack: pd.DataFrame, pairwise_gains: dict[str, dict[str, Any]], family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in miss_pack.itertuples(index=False):
        prompt = str(getattr(item, "prompt"))
        answer = str(getattr(item, "answer"))
        row_id = str(getattr(item, "id"))
        features = prompt_features(prompt)
        if family == "equation_transform":
            route, action = classify_equation_route(prompt, answer)
        else:
            route, action = classify_bit_route(prompt)
        alt_candidates = split_candidates(getattr(item, "correct_alternative_candidates"))
        gains = pairwise_gains.get(row_id, {"gain_candidates": [], "gain_predictions": [], "unique_gain_predictions": [], "gain_count": 0})
        rows.append(
            {
                "id": row_id,
                "family": family,
                "answer": answer,
                "baseline_prediction": str(getattr(item, "baseline_prediction")),
                "baseline_correct": bool_text(getattr(item, "baseline_correct")),
                "baseline_truncated": bool_text(getattr(item, "baseline_truncated")),
                "correct_alternative_count": as_int(getattr(item, "correct_alternative_count")),
                "correct_alternative_candidates": ";".join(alt_candidates),
                "pairwise_gain_count": int(gains.get("gain_count", 0)),
                "pairwise_gain_candidates": ";".join(gains.get("gain_candidates", [])),
                "pairwise_unique_gain_predictions": ";".join(gains.get("unique_gain_predictions", [])),
                "solver_route": route,
                "recommended_action": action,
                "feature_digits": features["digits"],
                "feature_operators": features["operators"],
                "feature_variables": features["variables"],
                "feature_example_markers": features["example_markers"],
                "features_json": json.dumps(features, sort_keys=True),
                "prompt_excerpt": compact_text(prompt, 260),
            }
        )
    sort_cols = ["correct_alternative_count", "pairwise_gain_count", "solver_route", "id"]
    return pd.DataFrame(rows).sort_values(sort_cols, ascending=[False, False, True, True])


def route_summary(taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route, group in taxonomy.groupby("solver_route", sort=False):
        rows.append(
            {
                "solver_route": str(route),
                "rows": int(len(group)),
                "rows_with_alternative_candidate": int((group["correct_alternative_count"].astype(int) > 0).sum()),
                "rows_all_candidates_wrong": int((group["correct_alternative_count"].astype(int) == 0).sum()),
                "max_pairwise_gain_count": int(group["pairwise_gain_count"].astype(int).max()) if len(group) else 0,
                "sample_ids": ";".join(group["id"].astype(str).head(12).tolist()),
                "recommended_action": str(group["recommended_action"].iloc[0]) if len(group) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["rows_with_alternative_candidate", "rows"], ascending=[False, False])


def candidate_frequency(taxonomy: pd.DataFrame) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for value in taxonomy["correct_alternative_candidates"].astype(str):
        counts.update(split_candidates(value))
    return [{"candidate": candidate, "hit_rows": int(count)} for candidate, count in counts.most_common()]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if not args.v230_analysis_manifest_json.exists():
        raise FileNotFoundError(args.v230_analysis_manifest_json)
    if not args.v230_analysis_manifest_json.is_file():
        raise IsADirectoryError(
            "--v230-analysis-manifest-json must point to a JSON file, got: "
            + str(args.v230_analysis_manifest_json)
        )

    manifest = read_json(args.v230_analysis_manifest_json)
    observed_contract = str(manifest.get("observed_shared_row_contract_sha256", ""))
    if args.expected_shared_row_contract_sha256 and observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    outputs = manifest.get("outputs", {})
    required_outputs = {
        "baseline_miss_hits_csv",
        "equation_miss_pack_csv",
        "bit_miss_pack_csv",
        "pairwise_detail_csv",
        "candidate_summary_csv",
        "router_simulation_csv",
    }
    missing = sorted(name for name in required_outputs if not outputs.get(name))
    if missing:
        raise RuntimeError("V230 manifest missing output paths: " + json.dumps(missing))

    paths = {name: Path(str(outputs[name])) for name in required_outputs}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    baseline_misses = read_csv(paths["baseline_miss_hits_csv"])
    equation_misses = read_csv(paths["equation_miss_pack_csv"])
    bit_misses = read_csv(paths["bit_miss_pack_csv"])
    pairwise_detail = read_csv(paths["pairwise_detail_csv"])
    candidate_summary = read_csv(paths["candidate_summary_csv"])
    router_simulation = read_csv(paths["router_simulation_csv"])

    validate_miss_pack(equation_misses, "equation_transform", "equation_miss_pack")
    validate_miss_pack(bit_misses, "bit_manipulation", "bit_miss_pack")
    if len(baseline_misses) != len(equation_misses) + len(bit_misses):
        raise RuntimeError("baseline miss count does not equal equation+bit miss counts")

    pairwise_gains = aggregate_pairwise_gains(pairwise_detail)
    equation_taxonomy = taxonomy_rows(equation_misses, pairwise_gains, "equation_transform")
    bit_taxonomy = taxonomy_rows(bit_misses, pairwise_gains, "bit_manipulation")
    equation_route_summary = route_summary(equation_taxonomy)
    bit_route_summary = route_summary(bit_taxonomy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "equation_miss_taxonomy_csv": args.output_dir / f"{prefix}_equation_miss_taxonomy.csv",
        "equation_route_summary_csv": args.output_dir / f"{prefix}_equation_route_summary.csv",
        "bit_miss_taxonomy_csv": args.output_dir / f"{prefix}_bit_miss_taxonomy.csv",
        "bit_route_summary_csv": args.output_dir / f"{prefix}_bit_route_summary.csv",
        "equation_solver_candidate_rules_json": args.output_dir / f"{prefix}_equation_solver_candidate_rules.json",
        "bit_guardrail_candidates_json": args.output_dir / f"{prefix}_bit_guardrail_candidates.json",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }

    equation_taxonomy.to_csv(out_paths["equation_miss_taxonomy_csv"], index=False)
    equation_route_summary.to_csv(out_paths["equation_route_summary_csv"], index=False)
    bit_taxonomy.to_csv(out_paths["bit_miss_taxonomy_csv"], index=False)
    bit_route_summary.to_csv(out_paths["bit_route_summary_csv"], index=False)

    baseline_summary = manifest.get("baseline_summary", {})
    decision = manifest.get("decision", {})
    equation_alt_rows = int((equation_taxonomy["correct_alternative_count"].astype(int) > 0).sum())
    bit_alt_rows = int((bit_taxonomy["correct_alternative_count"].astype(int) > 0).sum())
    equation_rules = {
        "schema_version": "kg1_v231_equation_solver_candidate_rules_v1",
        "generated_at_utc": utc_now(),
        "target_family": "equation_transform",
        "baseline_correct": int(baseline_summary.get("equation_transform_correct", 0) or 0),
        "weak_gate_threshold": int(args.weak_eq_min),
        "baseline_gap": max(0, int(args.weak_eq_min) - int(baseline_summary.get("equation_transform_correct", 0) or 0)),
        "target_gain_minimum": int(args.equation_target_gain),
        "oracle_decision": decision,
        "taxonomy_rows": int(len(equation_taxonomy)),
        "rows_with_correct_alternative": equation_alt_rows,
        "candidate_hit_frequency": candidate_frequency(equation_taxonomy),
        "route_summary": equation_route_summary.to_dict(orient="records"),
        "acceptance_contract": [
            "No override without a local proof/verifier.",
            "Every in-prompt example must be parsed and satisfied.",
            "When parser/verifier is ambiguous, abstain and keep V226 baseline.",
            "A deployable V231/V232 candidate must recover at least the equation gap without reducing bit below guardrail.",
        ],
        "next_action": "Implement route-specific verified solvers for the highest-yield equation routes before any training run.",
    }
    bit_rules = {
        "schema_version": "kg1_v231_bit_guardrail_candidates_v1",
        "generated_at_utc": utc_now(),
        "target_family": "bit_manipulation",
        "baseline_correct": int(baseline_summary.get("bit_manipulation_correct", 0) or 0),
        "weak_gate_threshold": int(args.weak_bit_min),
        "guardrail_minimum": int(args.bit_guardrail_min),
        "taxonomy_rows": int(len(bit_taxonomy)),
        "rows_with_correct_alternative": bit_alt_rows,
        "candidate_hit_frequency": candidate_frequency(bit_taxonomy),
        "route_summary": bit_route_summary.to_dict(orient="records"),
        "acceptance_contract": [
            "Bit solver is a guardrail first: never replace correct V226 bit rows.",
            "Accept only expressions that satisfy all prompt examples.",
            "Keep V226 when a bit rule cannot be proven.",
            "Final candidate should preserve bit >= guardrail_minimum unless human explicitly accepts risk.",
        ],
        "next_action": "Use bit taxonomy to add no-loss bitvector checks after equation solver work.",
    }
    write_json(out_paths["equation_solver_candidate_rules_json"], equation_rules)
    write_json(out_paths["bit_guardrail_candidates_json"], bit_rules)

    manifest_out = {
        "schema_version": "kg1_v231_miss_pack_mining_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v230_analysis_manifest_json": str(args.v230_analysis_manifest_json),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "observed_shared_row_contract_sha256": observed_contract,
        },
        "input_artifact_hashes": {
            "v230_analysis_manifest_json": file_meta(args.v230_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(paths.items())},
        },
        "baseline_summary": baseline_summary,
        "v230_decision": decision,
        "candidate_summary_top": candidate_summary.head(12).to_dict(orient="records"),
        "router_simulation_top": router_simulation.head(8).to_dict(orient="records"),
        "miss_counts": {
            "baseline_misses": int(len(baseline_misses)),
            "equation_misses": int(len(equation_misses)),
            "bit_misses": int(len(bit_misses)),
            "equation_rows_with_correct_alternative": equation_alt_rows,
            "bit_rows_with_correct_alternative": bit_alt_rows,
        },
        "route_summary": {
            "equation_transform": equation_route_summary.to_dict(orient="records"),
            "bit_manipulation": bit_route_summary.to_dict(orient="records"),
        },
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": {
            "decision": "mine_equation_solvers_before_training",
            "reason": (
                f"equation_misses={len(equation_misses)}; "
                f"equation_alt_rows={equation_alt_rows}; "
                f"bit_misses={len(bit_misses)}; bit_alt_rows={bit_alt_rows}"
            ),
            "next_action": "Build verified equation solver candidates from taxonomy outputs; do not train or full-eval yet.",
        },
    }
    write_json(out_paths["manifest_json"], manifest_out)

    print("equation_route_summary =", equation_route_summary.to_string(index=False), flush=True)
    print("bit_route_summary =", bit_route_summary.to_string(index=False), flush=True)
    print("miss_counts =", json.dumps(manifest_out["miss_counts"], sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest_out["decision"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    return manifest_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v230-analysis-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v231_v230_miss_pack_mining")
    parser.add_argument("--expected-shared-row-contract-sha256", default="")
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=136)
    parser.add_argument("--equation-target-gain", type=int, default=5)
    parser.add_argument("--bit-guardrail-min", type=int, default=136)
    parser.add_argument("--self-test", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        equation = root / "equation.csv"
        bit = root / "bit.csv"
        baseline = root / "baseline.csv"
        pairwise = root / "pairwise.csv"
        candidates = root / "candidates.csv"
        router = root / "router.csv"
        rows_common = [
            {
                "id": "eq1",
                "family": "equation_transform",
                "answer": "42",
                "baseline_prediction": "41",
                "baseline_correct": "False",
                "baseline_truncated": "False",
                "correct_alternative_count": "1",
                "correct_alternative_candidates": "candidate_a",
                "prompt": "Given examples x=1 -> 2 and x=2 -> 4, solve the equation formula for x=21.",
            },
            {
                "id": "eq2",
                "family": "equation_transform",
                "answer": "AB",
                "baseline_prediction": "AA",
                "baseline_correct": "False",
                "baseline_truncated": "False",
                "correct_alternative_count": "0",
                "correct_alternative_candidates": "",
                "prompt": "Each letter is a digit in this cryptarithm. Find the result.",
            },
        ]
        bit_rows = [
            {
                "id": "bit1",
                "family": "bit_manipulation",
                "answer": "1010",
                "baseline_prediction": "1111",
                "baseline_correct": "False",
                "baseline_truncated": "False",
                "correct_alternative_count": "1",
                "correct_alternative_candidates": "candidate_b",
                "prompt": "Apply xor and shift to the binary input examples.",
            }
        ]
        write_csv(equation, rows_common)
        write_csv(bit, bit_rows)
        write_csv(baseline, rows_common + bit_rows)
        write_csv(
            pairwise,
            [
                {
                    "id": "eq1",
                    "candidate": "candidate_a",
                    "family": "equation_transform",
                    "answer": "42",
                    "prompt": rows_common[0]["prompt"],
                    "baseline_prediction": "41",
                    "baseline_correct": "False",
                    "baseline_truncated": "False",
                    "candidate_prediction": "42",
                    "candidate_correct": "True",
                    "candidate_truncated": "False",
                    "lost_vs_baseline": "False",
                    "gained_vs_baseline": "True",
                    "candidate_extra_truncated": "False",
                },
                {
                    "id": "bit1",
                    "candidate": "candidate_b",
                    "family": "bit_manipulation",
                    "answer": "1010",
                    "prompt": bit_rows[0]["prompt"],
                    "baseline_prediction": "1111",
                    "baseline_correct": "False",
                    "baseline_truncated": "False",
                    "candidate_prediction": "1010",
                    "candidate_correct": "True",
                    "candidate_truncated": "False",
                    "lost_vs_baseline": "False",
                    "gained_vs_baseline": "True",
                    "candidate_extra_truncated": "False",
                },
            ],
        )
        write_csv(candidates, [{"candidate": "baseline", "correct": "191"}])
        write_csv(router, [{"strategy": "baseline", "correct": "191"}])
        manifest = root / "manifest.json"
        write_json(
            manifest,
            {
                "observed_shared_row_contract_sha256": "abc",
                "baseline_summary": {"equation_transform_correct": 55, "bit_manipulation_correct": 136},
                "decision": {"decision": "self_test"},
                "outputs": {
                    "baseline_miss_hits_csv": str(baseline),
                    "equation_miss_pack_csv": str(equation),
                    "bit_miss_pack_csv": str(bit),
                    "pairwise_detail_csv": str(pairwise),
                    "candidate_summary_csv": str(candidates),
                    "router_simulation_csv": str(router),
                },
            },
        )
        args = build_parser().parse_args(
            [
                "--v230-analysis-manifest-json",
                str(manifest),
                "--output-dir",
                str(root / "out"),
                "--expected-shared-row-contract-sha256",
                "abc",
            ]
        )
        result = run_analysis(args)
        if result["miss_counts"]["equation_misses"] != 2:
            raise AssertionError("self-test equation miss count mismatch")
        if result["miss_counts"]["bit_rows_with_correct_alternative"] != 1:
            raise AssertionError("self-test bit alternative count mismatch")
    print("v231_miss_pack_mining_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    print("=== V231 MISS PACK MINING SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v230_analysis_manifest_json =", args.v230_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    run_analysis(args)
    print("=== V231 MISS PACK MINING SCRIPT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

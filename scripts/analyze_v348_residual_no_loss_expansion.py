#!/usr/bin/env python3
"""V348 CPU residual audit after the V343 no-loss solver/verifier gate.

This script does not train, run GPU inference, package, or submit. It records
which weak rows remain unsolved after V343 and classifies the residual search
space before any new HF spend is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import classify_subtype, parse_alice_prompt, sha256_file  # noqa: E402


DEFAULT_V343_PREDICTIONS = (
    REPO_ROOT
    / "artifacts/v343_equation_residual_solver_audit/20260513T_integrated_on_v290_v3/"
    / "v336a_integrated_no_loss_predictions.csv"
)
DEFAULT_TONG_DETAIL = (
    REPO_ROOT
    / "artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/"
    / "v333_tong_bit_reasoner_gate_tong_bit_detail.csv"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_of(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or "").strip()


def prediction_of(row: dict[str, Any]) -> str:
    for key in ("integrated_prediction", "prediction", "baseline_prediction"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def query_shape(query: str) -> str:
    shape = []
    for ch in str(query):
        if ch.isdigit():
            shape.append("D")
        elif ch.isalpha():
            shape.append("A")
        else:
            shape.append("P")
    return "".join(shape)


def hamming(a: str, b: str) -> tuple[int, str]:
    if len(a) != len(b):
        return max(len(a), len(b)), "length_mismatch"
    positions = [str(i) for i, (left, right) in enumerate(zip(a, b)) if left != right]
    return len(positions), ";".join(positions)


def summarize_counts(rows: list[dict[str, Any]], correct_key: str) -> dict[str, Any]:
    total = Counter()
    family: dict[str, Counter[str]] = {}
    for row in rows:
        fam = family_of(row)
        item = family.setdefault(fam, Counter())
        item["rows"] += 1
        total["rows"] += 1
        if truthy(row.get(correct_key, "")):
            item["correct"] += 1
            total["correct"] += 1
        if truthy(row.get("truncated", row.get("truncated_bool", ""))):
            item["truncated"] += 1
            total["truncated"] += 1
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "accuracy": float(total["correct"] / total["rows"]) if total["rows"] else 0.0,
        "truncated": int(total["truncated"]),
        "family": {key: dict(value) for key, value in sorted(family.items())},
    }


def load_tong_detail(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["id"]: row for row in read_csv(path) if row.get("id")}


def route_hint_for_equation(subtype: str, shape: str) -> str:
    if subtype == "equation_numeric_operator":
        return "expand_numeric_operator_dsl_only_if_unique_no_loss"
    if subtype == "equation_symbolic_punct" and shape == "PPPPP":
        return "symbolic_punctuation_transducer_or_cryptarithm_residual"
    if subtype == "equation_symbolic_punct":
        return "symbolic_punctuation_short_query_residual"
    return "parse_or_unknown_residual"


def route_hint_for_bit(tong: dict[str, str], hamming_distance: int) -> str:
    if truthy(tong.get("tong_gain_vs_baseline", "")):
        return "tong_teacher_gain_but_requires_no_loss_selector"
    if truthy(tong.get("tong_loss_vs_baseline", "")):
        return "tong_direct_route_blocked_loss_case"
    if hamming_distance <= 2:
        return "low_hamming_bit_residual_candidate_for_stride_confidence"
    return "complex_bit_residual_keep_baseline"


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V348 RESIDUAL NO-LOSS EXPANSION AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v343_predictions_csv =", args.v343_predictions_csv, flush=True)
    print("tong_detail_csv =", args.tong_detail_csv or "", flush=True)
    print("output_dir =", args.output_dir, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.v343_predictions_csv)
    tong_by_id = load_tong_detail(args.tong_detail_csv)

    normalized_rows: list[dict[str, Any]] = []
    equation_residuals: list[dict[str, Any]] = []
    bit_residuals: list[dict[str, Any]] = []
    equation_subtypes = Counter()
    equation_shapes = Counter()
    bit_hamming = Counter()

    for row in rows:
        fam = family_of(row)
        pred = prediction_of(row)
        integrated_correct = verify_answer(str(row.get("answer", "")).strip(), pred)
        normalized = {**row, "family": fam, "integrated_prediction": pred, "integrated_correct": integrated_correct}
        normalized_rows.append(normalized)
        if integrated_correct:
            continue

        if fam == "equation_transform":
            examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
            subtype = classify_subtype(examples, query) if parse_status == "ok" else parse_status
            shape = query_shape(query)
            equation_subtypes[subtype] += 1
            equation_shapes[shape] += 1
            equation_residuals.append(
                {
                    "id": row.get("id", ""),
                    "family": fam,
                    "answer": row.get("answer", ""),
                    "integrated_prediction": pred,
                    "baseline_prediction": row.get("baseline_prediction", ""),
                    "parse_status": parse_status,
                    "subtype": subtype,
                    "query": query,
                    "query_shape": shape,
                    "example_count": len(examples),
                    "example_preview": json.dumps(examples[:3], ensure_ascii=False),
                    "route_hint": route_hint_for_equation(subtype, shape),
                }
            )
        elif fam == "bit_manipulation":
            distance, positions = hamming(str(row.get("answer", "")).strip(), pred)
            bit_hamming[str(distance)] += 1
            tong = tong_by_id.get(str(row.get("id", "")), {})
            bit_residuals.append(
                {
                    "id": row.get("id", ""),
                    "family": fam,
                    "answer": row.get("answer", ""),
                    "integrated_prediction": pred,
                    "baseline_prediction": row.get("baseline_prediction", ""),
                    "hamming_distance": distance,
                    "hamming_positions": positions,
                    "tong_prediction": tong.get("tong_prediction", ""),
                    "tong_gain_vs_baseline": tong.get("tong_gain_vs_baseline", ""),
                    "tong_loss_vs_baseline": tong.get("tong_loss_vs_baseline", ""),
                    "tong_trace_sha256": tong.get("tong_trace_sha256", ""),
                    "route_hint": route_hint_for_bit(tong, distance),
                }
            )

    outputs = {
        "equation_residuals_csv": str(args.output_dir / "v348_equation_residuals.csv"),
        "bit_residuals_csv": str(args.output_dir / "v348_bit_residuals.csv"),
        "manifest_json": str(args.output_dir / "v348_residual_no_loss_expansion_manifest.json"),
    }
    write_csv(Path(outputs["equation_residuals_csv"]), equation_residuals)
    write_csv(Path(outputs["bit_residuals_csv"]), bit_residuals)

    summary = summarize_counts(normalized_rows, "integrated_correct")
    if int(summary["correct"]) != args.expected_correct:
        raise RuntimeError(f"expected correct={args.expected_correct}, got {summary['correct']}")
    eq_correct = int(summary["family"].get("equation_transform", {}).get("correct", 0))
    bit_correct = int(summary["family"].get("bit_manipulation", {}).get("correct", 0))
    if eq_correct != args.expected_equation_correct:
        raise RuntimeError(f"expected equation={args.expected_equation_correct}, got {eq_correct}")
    if bit_correct != args.expected_bit_correct:
        raise RuntimeError(f"expected bit={args.expected_bit_correct}, got {bit_correct}")

    decision = {
        "decision": "cpu_residual_map_ready_no_hf_gpu",
        "reason": (
            f"residual_equation={len(equation_residuals)}; residual_bit={len(bit_residuals)}; "
            "no new no-loss rule has been accepted by this audit"
        ),
        "next_action": (
            "Implement new label-free candidate rules only against these residual CSVs, then rerun a CPU "
            "no-loss gate. HF remains blocked until equation>63 or bit>136 with losses=0."
        ),
    }

    manifest = {
        "schema_version": "kg1_v348_residual_no_loss_expansion_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v343_predictions_csv": str(args.v343_predictions_csv),
            "v343_predictions_sha256": sha256_file(args.v343_predictions_csv),
            "tong_detail_csv": str(args.tong_detail_csv) if args.tong_detail_csv else "",
            "tong_detail_sha256": sha256_file(args.tong_detail_csv) if args.tong_detail_csv and args.tong_detail_csv.exists() else "",
        },
        "v343_summary": summary,
        "equation_residual_count": len(equation_residuals),
        "equation_residual_by_subtype": dict(equation_subtypes),
        "equation_residual_query_shapes": dict(equation_shapes),
        "bit_residual_count": len(bit_residuals),
        "bit_residual_hamming": dict(bit_hamming),
        "decision": decision,
        "outputs": outputs,
    }
    write_json(Path(outputs["manifest_json"]), manifest)

    print("v343_summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("equation_residual_by_subtype =", json.dumps(dict(equation_subtypes), sort_keys=True), flush=True)
    print("equation_residual_query_shapes =", json.dumps(dict(equation_shapes), sort_keys=True), flush=True)
    print("bit_residual_hamming =", json.dumps(dict(bit_hamming), sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V348 RESIDUAL NO-LOSS EXPANSION AUDIT END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v343-predictions-csv", type=Path, default=DEFAULT_V343_PREDICTIONS)
    parser.add_argument("--tong-detail-csv", type=Path, default=DEFAULT_TONG_DETAIL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-correct", type=int, default=199)
    parser.add_argument("--expected-equation-correct", type=int, default=63)
    parser.add_argument("--expected-bit-correct", type=int, default=136)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

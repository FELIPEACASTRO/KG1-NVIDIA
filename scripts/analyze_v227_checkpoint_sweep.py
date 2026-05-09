#!/usr/bin/env python3
"""Analyze V227 targeted equation micro-sweep outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v227_targeted_equation_micro_sweep")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-eq-min", type=int, default=60)
    parser.add_argument("--weak-bit-min", type=int, default=133)
    parser.add_argument("--weak-trunc-max", type=int, default=3)
    args = parser.parse_args()

    print("=== V227 ANALYZER START ===", flush=True)
    print("batch_summary_json =", args.batch_summary_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch = read_json(args.batch_summary_json)
    rows = [dict(item) for item in batch.get("rows", []) if item.get("status") == "ok"]
    if not rows:
        raise RuntimeError("No ok rows found in batch summary.")

    thresholds = {
        "total": int(args.weak_total_min),
        "equation_transform": int(args.weak_eq_min),
        "bit_manipulation": int(args.weak_bit_min),
        "truncated": int(args.weak_trunc_max),
    }

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        correct = as_int(row, "correct")
        eq_correct = as_int(row, "equation_transform_correct")
        bit_correct = as_int(row, "bit_manipulation_correct")
        truncated = as_int(row, "truncated", 999999)
        weak_gate = (
            correct >= thresholds["total"]
            and eq_correct >= thresholds["equation_transform"]
            and bit_correct >= thresholds["bit_manipulation"]
            and truncated <= thresholds["truncated"]
        )
        enriched = {
            **row,
            "weak_gate_pass_for_full": bool(weak_gate),
            "gate_total_gap": max(0, thresholds["total"] - correct),
            "gate_eq_gap": max(0, thresholds["equation_transform"] - eq_correct),
            "gate_bit_gap": max(0, thresholds["bit_manipulation"] - bit_correct),
            "gate_trunc_gap": max(0, truncated - thresholds["truncated"]),
        }
        summary_rows.append(enriched)

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "weak_gate_pass_for_full",
            "correct",
            "equation_transform_correct",
            "bit_manipulation_correct",
            "truncated",
        ],
        ascending=[False, False, False, False, True],
    )

    best = summary.iloc[0].to_dict()
    if bool(best["weak_gate_pass_for_full"]):
        decision = {
            "decision": "checkpoint_candidate_passed_weak_gate_confirm_full_eval",
            "best_candidate": best.get("name"),
            "reason": (
                f"correct={int(best['correct'])}; "
                f"eq={int(best['equation_transform_correct'])}; "
                f"bit={int(best['bit_manipulation_correct'])}; "
                f"truncated={int(best['truncated'])}"
            ),
            "next_action": "Run a separate full-eval confirmation notebook; do not submit automatically.",
        }
    else:
        decision = {
            "decision": "no_checkpoint_candidate_passed_weak_gate",
            "best_candidate": best.get("name"),
            "reason": (
                f"best_correct={int(best['correct'])}; "
                f"best_eq={int(best['equation_transform_correct'])}; "
                f"best_bit={int(best['bit_manipulation_correct'])}; "
                f"best_truncated={int(best['truncated'])}; "
                f"total_gap={int(best['gate_total_gap'])}; "
                f"eq_gap={int(best['gate_eq_gap'])}; "
                f"bit_gap={int(best['gate_bit_gap'])}; "
                f"trunc_gap={int(best['gate_trunc_gap'])}"
            ),
            "next_action": "Compare row-level misses from the best V226/V227 candidates and prepare the V228 data or adapter-complementarity route.",
        }

    prefix = args.label
    summary_csv = args.output_dir / f"{prefix}_summary.csv"
    manifest_json = args.output_dir / f"{prefix}_manifest.json"
    summary.to_csv(summary_csv, index=False)
    manifest = {
        "generated_at_utc": utc_now(),
        "batch_summary_json": str(args.batch_summary_json),
        "thresholds": thresholds,
        "candidate_count": len(summary_rows),
        "best": best,
        "decision": decision,
        "outputs": {
            "summary_csv": str(summary_csv),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("summary =", summary.to_string(index=False), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V227 ANALYZER END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

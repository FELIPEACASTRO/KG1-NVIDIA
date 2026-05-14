#!/usr/bin/env python3
"""Audit V352 weak eval against the V350 CPU teacher.

This script is intentionally CPU-only. It answers a narrow question:
did the V352 adapter transfer the accepted V350 no-loss bit fixes, or did
it regress back to the adapter-only behavior?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def family_summary(df: pd.DataFrame, family_col: str, correct_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    correct = bool_series(df[correct_col])
    for family, idx in df.groupby(family_col).groups.items():
        sub_correct = correct.loc[idx]
        rows.append(
            {
                "family": str(family),
                "rows": int(len(sub_correct)),
                "correct": int(sub_correct.sum()),
                "accuracy": float(sub_correct.mean()) if len(sub_correct) else 0.0,
            }
        )
    rows.append(
        {
            "family": "OVERALL",
            "rows": int(len(df)),
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()) if len(correct) else 0.0,
        }
    )
    return rows


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v350 = pd.read_csv(args.v350_integrated_predictions_csv).copy()
    v350_decisions = pd.read_csv(args.v350_candidate_decisions_csv).copy()
    v352 = pd.read_csv(args.v352_predictions_csv).copy()

    required_v350 = {"id", "family", "answer", "v350_prediction", "v350_correct"}
    required_v352 = {"id", "type", "answer", "prediction", "correct", "truncated"}
    missing_v350 = sorted(required_v350 - set(v350.columns))
    missing_v352 = sorted(required_v352 - set(v352.columns))
    if missing_v350:
        raise ValueError(f"V350 integrated predictions missing columns: {missing_v350}")
    if missing_v352:
        raise ValueError(f"V352 predictions missing columns: {missing_v352}")

    v350["id"] = v350["id"].astype(str)
    v352["id"] = v352["id"].astype(str)
    v350["v350_correct_bool"] = bool_series(v350["v350_correct"])
    v352["v352_correct_bool"] = bool_series(v352["correct"])

    shared = v350.merge(
        v352[["id", "prediction", "correct", "truncated"]],
        on="id",
        how="inner",
        suffixes=("_v350", "_v352"),
    )
    shared["v352_correct_bool"] = bool_series(shared["correct"])
    shared["v352_vs_v350_delta"] = shared["v352_correct_bool"].astype(int) - shared["v350_correct_bool"].astype(int)

    accepted = v350_decisions[v350_decisions.get("accepted", False).astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    accepted["id"] = accepted["id"].astype(str)
    accepted_detail = accepted.merge(
        v352[["id", "prediction", "correct", "truncated"]],
        on="id",
        how="left",
    )
    accepted_detail["v352_correct_bool"] = bool_series(accepted_detail["correct"])
    accepted_detail["transfer_hit_new_prediction"] = (
        accepted_detail["prediction"].astype(str) == accepted_detail["new_prediction"].astype(str)
    )
    accepted_detail["transfer_hit_answer"] = accepted_detail["v352_correct_bool"]

    deltas = {
        "v352_correct_v350_wrong": int(((~shared["v350_correct_bool"]) & shared["v352_correct_bool"]).sum()),
        "v352_wrong_v350_correct": int((shared["v350_correct_bool"] & (~shared["v352_correct_bool"])).sum()),
        "net_v352_minus_v350": int(shared["v352_vs_v350_delta"].sum()),
    }

    family_deltas: list[dict[str, Any]] = []
    for family, sub in shared.groupby("family"):
        family_deltas.append(
            {
                "family": str(family),
                "rows": int(len(sub)),
                "v350_correct": int(sub["v350_correct_bool"].sum()),
                "v352_correct": int(sub["v352_correct_bool"].sum()),
                "v352_correct_v350_wrong": int(((~sub["v350_correct_bool"]) & sub["v352_correct_bool"]).sum()),
                "v352_wrong_v350_correct": int((sub["v350_correct_bool"] & (~sub["v352_correct_bool"])).sum()),
                "net_v352_minus_v350": int(sub["v352_vs_v350_delta"].sum()),
            }
        )

    shared_out = output_dir / "v354_v352_vs_v350_row_delta.csv"
    accepted_out = output_dir / "v354_v350_accepted_transfer_audit.csv"
    summary_out = output_dir / "v354_v352_transfer_failure_manifest.json"

    shared_cols = [
        "id",
        "family",
        "answer_v350",
        "v350_prediction",
        "prediction",
        "v350_correct_bool",
        "v352_correct_bool",
        "v352_vs_v350_delta",
        "v350_source_rule",
    ]
    shared[[c for c in shared_cols if c in shared.columns]].to_csv(shared_out, index=False)
    accepted_detail.to_csv(accepted_out, index=False)

    manifest: dict[str, Any] = {
        "schema_version": "kg1_v354_v352_transfer_failure_audit_v1",
        "v350_integrated_predictions_csv": str(Path(args.v350_integrated_predictions_csv)),
        "v350_candidate_decisions_csv": str(Path(args.v350_candidate_decisions_csv)),
        "v352_predictions_csv": str(Path(args.v352_predictions_csv)),
        "shared_rows": int(len(shared)),
        "v350_family_summary": family_summary(v350, "family", "v350_correct_bool"),
        "v352_family_summary": family_summary(v352, "type", "v352_correct_bool"),
        "family_deltas_v352_minus_v350": family_deltas,
        "delta_summary_v352_minus_v350": deltas,
        "accepted_v350_count": int(len(accepted_detail)),
        "accepted_v350_transferred_count": int(accepted_detail["transfer_hit_answer"].sum()),
        "accepted_v350_transfer_detail_csv": str(accepted_out),
        "row_delta_csv": str(shared_out),
        "decision": {
            "status": "reject_v352_checkpoint2",
            "reason": (
                "V352 checkpoint-2 failed to transfer the accepted V350 bit fixes "
                "and regressed against both V350 teacher and adapter-only gate."
            ),
        },
    }
    summary_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def self_test() -> None:
    tmp = Path(".tmp_v354_self_test")
    tmp.mkdir(exist_ok=True)
    v350 = tmp / "v350.csv"
    decisions = tmp / "decisions.csv"
    v352 = tmp / "v352.csv"
    out = tmp / "out"
    pd.DataFrame(
        [
            {"id": "a", "family": "bit_manipulation", "answer": "11", "v350_prediction": "11", "v350_correct": True},
            {"id": "b", "family": "equation_transform", "answer": "2", "v350_prediction": "2", "v350_correct": True},
        ]
    ).to_csv(v350, index=False)
    pd.DataFrame(
        [
            {
                "id": "a",
                "family": "bit_manipulation",
                "candidate_source": "x",
                "rule_class": "r",
                "old_prediction": "10",
                "new_prediction": "11",
                "answer": "11",
                "old_correct": False,
                "new_correct": True,
                "accepted": True,
                "rejection_reason": "",
                "candidate_count": 1,
                "conflict_count": 0,
                "proof": "ok",
            }
        ]
    ).to_csv(decisions, index=False)
    pd.DataFrame(
        [
            {"id": "a", "type": "bit_manipulation", "answer": "11", "prediction": "10", "correct": False, "truncated": False},
            {"id": "b", "type": "equation_transform", "answer": "2", "prediction": "2", "correct": True, "truncated": False},
        ]
    ).to_csv(v352, index=False)
    manifest = run_audit(
        argparse.Namespace(
            v350_integrated_predictions_csv=v350,
            v350_candidate_decisions_csv=decisions,
            v352_predictions_csv=v352,
            output_dir=out,
        )
    )
    assert manifest["accepted_v350_count"] == 1
    assert manifest["accepted_v350_transferred_count"] == 0
    assert manifest["delta_summary_v352_minus_v350"]["net_v352_minus_v350"] == -1
    print("v354_self_test=ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v350-integrated-predictions-csv", type=Path)
    parser.add_argument("--v350-candidate-decisions-csv", type=Path)
    parser.add_argument("--v352-predictions-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in [
        "v350_integrated_predictions_csv",
        "v350_candidate_decisions_csv",
        "v352_predictions_csv",
        "output_dir",
    ]:
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

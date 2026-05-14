"""Audit why V368 failed to transfer V366 CPU teacher gains.

This is a CPU-only diagnostic. It compares:

- V290 checkpoint-6 adapter-only weak baseline.
- V366 CPU teacher/verifier integrated predictions.
- V368 checkpoint-1 adapter-only weak predictions.

The goal is not to create a new candidate. The goal is to decide whether any
further HF spend is justified on the V367/V368 route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def row_contract(df: pd.DataFrame) -> str:
    if len(df) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(df)}")
    if df["id"].nunique() != len(df):
        raise RuntimeError("duplicate ids in row contract input")
    payload = "\n".join(
        f"{row.id}\t{row.family}\t{row.answer}\t{hashlib.sha256(str(row.prompt).encode('utf-8')).hexdigest()}"
        for row in df.sort_values("id").itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(df: pd.DataFrame, col: str) -> dict[str, Any]:
    family: dict[str, Any] = {}
    for fam, grp in df.groupby("family", sort=True):
        family[str(fam)] = {"rows": int(len(grp)), "correct": int(grp[col].sum())}
    return {"rows": int(len(df)), "correct": int(df[col].sum()), "family": family}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-predictions-csv",
        type=Path,
        default=Path("artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"),
    )
    parser.add_argument(
        "--v366-predictions-csv",
        type=Path,
        default=Path("artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv"),
    )
    parser.add_argument(
        "--v366-manifest-json",
        type=Path,
        default=Path("artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_bit_fullbyte_ternary_op_gate_manifest.json"),
    )
    parser.add_argument(
        "--v368-predictions-csv",
        type=Path,
        default=Path("artifacts/v368_hf_a100_v367_bit_ternary_launch/eval_checkpoint1/predictions.csv"),
    )
    parser.add_argument(
        "--v368-eval-report-json",
        type=Path,
        default=Path("artifacts/v368_hf_a100_v367_bit_ternary_launch/eval_checkpoint1/eval_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v369_v368_transfer_failure_audit/20260514T_cpu_audit"),
    )
    parser.add_argument(
        "--expected-shared-row-contract-sha256",
        default=EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    )
    args = parser.parse_args()

    print("=== V369 V368 TRANSFER FAILURE AUDIT START ===", flush=True)
    print("baseline_predictions_csv =", args.baseline_predictions_csv, flush=True)
    print("v366_predictions_csv =", args.v366_predictions_csv, flush=True)
    print("v366_manifest_json =", args.v366_manifest_json, flush=True)
    print("v368_predictions_csv =", args.v368_predictions_csv, flush=True)
    print("v368_eval_report_json =", args.v368_eval_report_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    for path in [
        args.baseline_predictions_csv,
        args.v366_predictions_csv,
        args.v366_manifest_json,
        args.v368_predictions_csv,
        args.v368_eval_report_json,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline = pd.read_csv(args.baseline_predictions_csv)
    v366 = pd.read_csv(args.v366_predictions_csv)
    v368 = pd.read_csv(args.v368_predictions_csv)
    v366_manifest = read_json(args.v366_manifest_json)
    v368_report = read_json(args.v368_eval_report_json)

    merged = baseline[
        ["id", "prompt", "answer", "type", "prediction", "correct", "truncated"]
    ].rename(
        columns={
            "type": "family",
            "prediction": "baseline_prediction",
            "correct": "baseline_correct",
            "truncated": "baseline_truncated",
        }
    )
    merged = merged.merge(
        v368[["id", "prediction", "correct", "truncated"]].rename(
            columns={
                "prediction": "v368_prediction",
                "correct": "v368_correct",
                "truncated": "v368_truncated",
            }
        ),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        v366[
            [
                "id",
                "v366_prediction",
                "v366_correct",
                "v366_source_rule",
                "current_prediction",
                "current_correct",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 315:
        raise RuntimeError(f"Expected 315 shared rows, got {len(merged)}")

    for col in [
        "baseline_correct",
        "baseline_truncated",
        "v368_correct",
        "v368_truncated",
        "v366_correct",
        "current_correct",
    ]:
        merged[col] = bool_series(merged[col])

    observed_contract = row_contract(merged[["id", "prompt", "family", "answer"]])
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + str(args.expected_shared_row_contract_sha256)
            + ", got "
            + observed_contract
        )

    baseline_summary = summarize(merged, "baseline_correct")
    v366_summary = summarize(merged, "v366_correct")
    v368_summary = summarize(merged, "v368_correct")
    print("baseline_summary =", json.dumps(baseline_summary, sort_keys=True), flush=True)
    print("v366_summary =", json.dumps(v366_summary, sort_keys=True), flush=True)
    print("v368_summary =", json.dumps(v368_summary, sort_keys=True), flush=True)

    changed = merged[
        merged["baseline_prediction"].astype(str) != merged["v368_prediction"].astype(str)
    ].copy()
    changed["v368_gain_vs_baseline"] = (~changed["baseline_correct"]) & changed["v368_correct"]
    changed["v368_loss_vs_baseline"] = changed["baseline_correct"] & (~changed["v368_correct"])
    changed["v368_neutral_change"] = ~(changed["v368_gain_vs_baseline"] | changed["v368_loss_vs_baseline"])

    accepted_ids = [str(x) for x in v366_manifest.get("accepted_gain_ids", [])]
    transfer = merged[merged["id"].isin(accepted_ids)].copy()
    transfer["v366_gain_transferred_to_v368"] = transfer["v368_correct"]

    v368_unique_gains = merged[(~merged["baseline_correct"]) & merged["v368_correct"]].copy()
    v368_losses = merged[merged["baseline_correct"] & (~merged["v368_correct"])].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    changed_out = args.output_dir / "v369_v368_changed_vs_baseline.csv"
    transfer_out = args.output_dir / "v369_v366_gain_transfer.csv"
    unique_gains_out = args.output_dir / "v369_v368_unique_gains.csv"
    losses_out = args.output_dir / "v369_v368_losses.csv"
    family_out = args.output_dir / "v369_family_summary.csv"
    manifest_out = args.output_dir / "v369_v368_transfer_failure_manifest.json"
    summary_out = args.output_dir.parent / "V369_RESULT_SUMMARY.md"

    changed.to_csv(changed_out, index=False)
    transfer.to_csv(transfer_out, index=False)
    v368_unique_gains.to_csv(unique_gains_out, index=False)
    v368_losses.to_csv(losses_out, index=False)

    family_rows: list[dict[str, Any]] = []
    for fam, grp in merged.groupby("family", sort=True):
        family_rows.append(
            {
                "family": fam,
                "rows": int(len(grp)),
                "baseline_correct": int(grp["baseline_correct"].sum()),
                "v366_correct": int(grp["v366_correct"].sum()),
                "v368_correct": int(grp["v368_correct"].sum()),
                "v368_delta_vs_baseline": int(grp["v368_correct"].sum() - grp["baseline_correct"].sum()),
                "v368_delta_vs_v366": int(grp["v368_correct"].sum() - grp["v366_correct"].sum()),
            }
        )
    pd.DataFrame(family_rows).to_csv(family_out, index=False)

    decision = {
        "decision": "v368_route_blocked",
        "hf_gpu_allowed": False,
        "next_action": (
            "Do not continue V367/V368 bit-only SFT. Return to CPU-only diagnosis or "
            "a new equation/bit solver-to-adapter signal before any HF job."
        ),
        "reason": (
            f"V368={v368_summary['correct']}/315, baseline={baseline_summary['correct']}/315, "
            f"transferred_v366_gains={int(transfer['v368_correct'].sum())}/{len(transfer)}, "
            f"v368_unique_gains={len(v368_unique_gains)}, v368_losses={len(v368_losses)}"
        ),
    }
    manifest = {
        "schema_version": "kg1_v369_v368_transfer_failure_audit_v1",
        "observed_shared_row_contract_sha256": observed_contract,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "inputs": {
            "baseline_predictions_csv": str(args.baseline_predictions_csv),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions_csv),
            "v366_predictions_csv": str(args.v366_predictions_csv),
            "v366_predictions_sha256": sha256_file(args.v366_predictions_csv),
            "v366_manifest_json": str(args.v366_manifest_json),
            "v366_manifest_sha256": sha256_file(args.v366_manifest_json),
            "v368_predictions_csv": str(args.v368_predictions_csv),
            "v368_predictions_sha256": sha256_file(args.v368_predictions_csv),
            "v368_eval_report_json": str(args.v368_eval_report_json),
            "v368_eval_report_sha256": sha256_file(args.v368_eval_report_json),
        },
        "baseline_summary": baseline_summary,
        "v366_summary": v366_summary,
        "v368_summary": v368_summary,
        "v368_report_summary": {
            "candidate_name": v368_report.get("candidate_name", ""),
            "correct": v368_report.get("correct"),
            "accuracy": v368_report.get("accuracy"),
            "truncated": v368_report.get("truncated"),
            "tokens_per_second": v368_report.get("tokens_per_second"),
        },
        "changed_vs_baseline_count": int(len(changed)),
        "v368_gain_vs_baseline_count": int(changed["v368_gain_vs_baseline"].sum()),
        "v368_loss_vs_baseline_count": int(changed["v368_loss_vs_baseline"].sum()),
        "v368_neutral_change_count": int(changed["v368_neutral_change"].sum()),
        "v366_accepted_gain_count": int(len(transfer)),
        "v366_accepted_gains_transferred_to_v368": int(transfer["v368_correct"].sum()),
        "v366_accepted_gains_not_transferred_to_v368": int((~transfer["v368_correct"]).sum()),
        "v368_unique_gain_ids": v368_unique_gains["id"].astype(str).tolist(),
        "v368_loss_ids": v368_losses["id"].astype(str).tolist(),
        "decision": decision,
        "outputs": {
            "changed_vs_baseline_csv": str(changed_out),
            "v366_gain_transfer_csv": str(transfer_out),
            "v368_unique_gains_csv": str(unique_gains_out),
            "v368_losses_csv": str(losses_out),
            "family_summary_csv": str(family_out),
            "manifest_json": str(manifest_out),
            "summary_md": str(summary_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out.write_text(
        "\n".join(
            [
                "# V369 V368 transfer failure audit",
                "",
                "Generated: 2026-05-14",
                "",
                "## Result",
                "",
                f"- Baseline adapter-only: `{baseline_summary['correct']}/315`, equation `{baseline_summary['family']['equation_transform']['correct']}/155`, bit `{baseline_summary['family']['bit_manipulation']['correct']}/160`.",
                f"- V366 CPU teacher: `{v366_summary['correct']}/315`, equation `{v366_summary['family']['equation_transform']['correct']}/155`, bit `{v366_summary['family']['bit_manipulation']['correct']}/160`.",
                f"- V368 checkpoint-1 adapter-only: `{v368_summary['correct']}/315`, equation `{v368_summary['family']['equation_transform']['correct']}/155`, bit `{v368_summary['family']['bit_manipulation']['correct']}/160`.",
                "",
                "## Transfer check",
                "",
                f"- V366 accepted gains tested: `{len(transfer)}`.",
                f"- V366 gains transferred to V368: `{int(transfer['v368_correct'].sum())}/{len(transfer)}`.",
                f"- V368 changed `{len(changed)}` rows versus baseline: `{int(changed['v368_gain_vs_baseline'].sum())}` gain, `{int(changed['v368_loss_vs_baseline'].sum())}` losses, `{int(changed['v368_neutral_change'].sum())}` neutral changes.",
                f"- V368 unique gain IDs: `{', '.join(v368_unique_gains['id'].astype(str).tolist()) or 'none'}`.",
                f"- V368 loss IDs: `{', '.join(v368_losses['id'].astype(str).tolist()) or 'none'}`.",
                "",
                "## Decision",
                "",
                "Blocked. V368 does not justify more HF spend on the V367/V368 bit-only transfer route.",
                "",
                "Next action: CPU-only. Either diagnose a new solver-to-adapter signal with stronger evidence, or return to equation/bit DSL gates. No full eval, package, or Kaggle submit from V368.",
                "",
                "## Local artifacts",
                "",
                f"- Manifest: `{manifest_out}`",
                f"- V366 transfer detail: `{transfer_out}`",
                f"- Changed rows: `{changed_out}`",
                f"- Family summary: `{family_out}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], sort_keys=True), flush=True)
    print("=== V369 V368 TRANSFER FAILURE AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V399 CPU pairwise audit for V398 vs the locked V290/V392 weak baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import get_token, hf_hub_url


V398_REPO = "felipesp1983/kg1-nemotron-lora-v398-nemo-h200-sft-reconstructed-v290ckpt6"
V398_FILES = {
    "v398_checkpoint_2": "evals/v398-h200-v221contract-sft-reconstructed-20260514T225622Z/eval/v398_sft_reconstructed_checkpoint_2_v221_contract/v398_hf_weak_v398_sft_reconstructed_checkpoint_2_v221_contract_predictions.csv",
    "v398_checkpoint_4": "evals/v398-h200-v221contract-sft-reconstructed-20260514T225622Z/eval/v398_sft_reconstructed_checkpoint_4_v221_contract/v398_hf_weak_v398_sft_reconstructed_checkpoint_4_v221_contract_predictions.csv",
}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hf_file(repo: str, repo_file: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    url = hf_hub_url(repo_id=repo, filename=repo_file, repo_type="model")
    headers = {}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "prompt", "answer", "type", "prediction", "correct", "truncated"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    if len(df) != 315:
        raise ValueError(f"{label} expected 315 rows, got {len(df)}")
    if df["id"].duplicated().any():
        duplicated = df.loc[df["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"{label} duplicated ids: {duplicated}")
    out = df.copy()
    out["correct"] = out["correct"].astype(int)
    out["truncated"] = out["truncated"].astype(int)
    return out


def equation_signature(prompt: str) -> str:
    """Compact row feature for manual review; not used as a selector."""
    symbols: list[str] = []
    for line in str(prompt).splitlines():
        if "=" not in line:
            continue
        left = line.split("=", 1)[0]
        match = re.search(r"\d+\s*([^0-9\s]+)\s*\d+", left)
        if match:
            symbols.append(match.group(1))
    if not symbols:
        return "no_symbol_detected"
    counts = {sym: symbols.count(sym) for sym in sorted(set(symbols))}
    return ",".join(f"{sym}:{count}" for sym, count in counts.items())


def bit_signature(prompt: str) -> str:
    """Compact bit prompt feature for manual review; not used as a selector."""
    binaries = re.findall(r"\b[01]{8}\b", str(prompt))
    return f"bin8_count={len(binaries)}"


def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    features: list[str] = []
    for _, row in out.iterrows():
        if row["type"] == "equation_transform":
            features.append(equation_signature(row["prompt"]))
        elif row["type"] == "bit_manipulation":
            features.append(bit_signature(row["prompt"]))
        else:
            features.append("unknown")
    out["review_signature"] = features
    return out


def summarize_pairwise(merged: pd.DataFrame, candidate_name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family, g in merged.groupby("type", sort=True):
        base = g["baseline_correct"].astype(bool)
        cand = g["candidate_correct"].astype(bool)
        rows.append(
            {
                "candidate": candidate_name,
                "family": family,
                "rows": int(len(g)),
                "baseline_correct": int(base.sum()),
                "candidate_correct": int(cand.sum()),
                "both_correct": int((base & cand).sum()),
                "candidate_only_correct": int((~base & cand).sum()),
                "baseline_only_correct": int((base & ~cand).sum()),
                "both_wrong": int((~base & ~cand).sum()),
                "candidate_truncated": int(g["candidate_truncated"].sum()),
                "delta_correct": int(cand.sum() - base.sum()),
            }
        )
    base_all = merged["baseline_correct"].astype(bool)
    cand_all = merged["candidate_correct"].astype(bool)
    rows.append(
        {
            "candidate": candidate_name,
            "family": "OVERALL",
            "rows": int(len(merged)),
            "baseline_correct": int(base_all.sum()),
            "candidate_correct": int(cand_all.sum()),
            "both_correct": int((base_all & cand_all).sum()),
            "candidate_only_correct": int((~base_all & cand_all).sum()),
            "baseline_only_correct": int((base_all & ~cand_all).sum()),
            "both_wrong": int((~base_all & ~cand_all).sum()),
            "candidate_truncated": int(merged["candidate_truncated"].sum()),
            "delta_correct": int(cand_all.sum() - base_all.sum()),
        }
    )
    return {"candidate": candidate_name, "rows": rows}


def pairwise(base: pd.DataFrame, cand: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    cand_cols = [
        "id",
        "prediction",
        "raw_output",
        "correct",
        "truncated",
    ]
    optional_cols = [c for c in cand_cols if c in cand.columns]
    merged = base.merge(
        cand[optional_cols],
        on="id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != 315:
        raise ValueError(f"{candidate_name} id contract mismatch: merged {len(merged)} rows")
    merged = add_review_features(merged)
    merged = merged.rename(
        columns={
            "correct_baseline": "baseline_correct",
            "truncated_baseline": "baseline_truncated",
            "prediction_baseline": "baseline_prediction",
            "raw_output_baseline": "baseline_raw_output",
            "correct_candidate": "candidate_correct",
            "truncated_candidate": "candidate_truncated",
            "prediction_candidate": "candidate_prediction",
            "raw_output_candidate": "candidate_raw_output",
        }
    )
    merged["candidate"] = candidate_name
    merged["status_vs_baseline"] = "same_wrong"
    merged.loc[(merged["baseline_correct"] == 1) & (merged["candidate_correct"] == 1), "status_vs_baseline"] = "same_correct"
    merged.loc[(merged["baseline_correct"] == 0) & (merged["candidate_correct"] == 1), "status_vs_baseline"] = "candidate_only_correct"
    merged.loc[(merged["baseline_correct"] == 1) & (merged["candidate_correct"] == 0), "status_vs_baseline"] = "baseline_only_correct"
    return merged


def write_markdown(
    out_path: Path,
    baseline_path: Path,
    summaries: list[dict[str, Any]],
    decision: dict[str, Any],
    downloaded: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# V399 V398 Pairwise Complementarity Audit")
    lines.append("")
    lines.append(f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Baseline CSV: `{baseline_path.as_posix()}`")
    lines.append(f"- Baseline SHA256: `{sha256_path(baseline_path)}`")
    lines.append(f"- V398 repo: `{V398_REPO}`")
    lines.append("")
    lines.append("## Downloaded Inputs")
    lines.append("")
    for name, path in downloaded.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Candidate | Family | Baseline | Candidate | Candidate-only | Baseline-only | Delta | Trunc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        for row in summary["rows"]:
            lines.append(
                "| {candidate} | {family} | {baseline_correct} | {candidate_correct} | "
                "{candidate_only_correct} | {baseline_only_correct} | {delta_correct} | {candidate_truncated} |".format(**row)
            )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    for key, value in decision.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "V398 is only actionable if the candidate-only equation rows expose a simple deterministic selector "
        "with zero bit regression. Otherwise this branch must remain rejected for FinOps and ranking purposes."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    input_dir = out_dir / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, str] = {}
    candidate_paths: dict[str, Path] = {}
    for name, repo_file in V398_FILES.items():
        local = input_dir / name / Path(repo_file).name
        download_hf_file(V398_REPO, repo_file, local)
        downloaded[name] = local.as_posix()
        candidate_paths[name] = local

    baseline = load_predictions(args.baseline_predictions_csv, "baseline")
    if int(baseline["correct"].sum()) != 192:
        raise ValueError(f"baseline expected 192 correct, got {int(baseline['correct'].sum())}")
    expected_family = baseline.groupby("type")["correct"].sum().to_dict()
    if expected_family.get("equation_transform") != 56 or expected_family.get("bit_manipulation") != 136:
        raise ValueError(f"baseline family contract mismatch: {expected_family}")

    all_pairwise: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for name, path in candidate_paths.items():
        cand = load_predictions(path, name)
        merged = pairwise(baseline, cand, name)
        all_pairwise.append(merged)
        summaries.append(summarize_pairwise(merged, name))
        if args.write_full_rows:
            merged.to_csv(out_dir / f"{name}_pairwise_rows.csv", index=False)
        interesting = merged[merged["status_vs_baseline"].isin(["candidate_only_correct", "baseline_only_correct"])].copy()
        interesting.to_csv(out_dir / f"{name}_changed_rows.csv", index=False)

    summary_rows = [row for s in summaries for row in s["rows"]]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "v399_pairwise_summary.csv", index=False)

    best_overall = summary_df[summary_df["family"] == "OVERALL"].sort_values(
        ["candidate_correct", "candidate_only_correct", "candidate_truncated"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    best_equation = summary_df[summary_df["family"] == "equation_transform"].sort_values(
        ["candidate_only_correct", "delta_correct"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    best_bit = summary_df[summary_df["family"] == "bit_manipulation"].sort_values(
        ["candidate_correct", "candidate_truncated"],
        ascending=[False, True],
    ).iloc[0].to_dict()

    # This is intentionally conservative: row-level oracle hits are not a deployable selector.
    actionable = (
        int(best_equation["candidate_only_correct"]) >= 4
        and int(best_bit["baseline_only_correct"]) == 0
        and int(best_overall["candidate_truncated"]) == 0
    )
    decision = {
        "best_overall_candidate": str(best_overall["candidate"]),
        "best_overall_correct": int(best_overall["candidate_correct"]),
        "best_equation_candidate": str(best_equation["candidate"]),
        "best_equation_candidate_only_correct": int(best_equation["candidate_only_correct"]),
        "best_bit_candidate": str(best_bit["candidate"]),
        "best_bit_correct": int(best_bit["candidate_correct"]),
        "actionable_without_new_selector": bool(actionable),
        "decision": "mine_changed_rows_for_selector" if actionable else "close_v397_v398_training_branch",
        "next_action": (
            "Inspect candidate-only rows for a deterministic selector before any GPU job."
            if actionable
            else "Do not run more V397/V398 SFT; return to CPU solver/verifier DSL and baseline package."
        ),
    }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_predictions_csv": args.baseline_predictions_csv.as_posix(),
        "baseline_sha256": sha256_path(args.baseline_predictions_csv),
        "v398_repo": V398_REPO,
        "downloaded": downloaded,
        "summary_csv": (out_dir / "v399_pairwise_summary.csv").as_posix(),
        "summaries": summaries,
        "decision": decision,
    }
    (out_dir / "v399_pairwise_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_dir / "V399_V398_PAIRWISE_COMPLEMENTARITY.md", args.baseline_predictions_csv, summaries, decision, downloaded)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-predictions-csv",
        type=Path,
        default=Path("artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v399_v398_pairwise_complementarity/20260514T_v399_pairwise"),
    )
    parser.add_argument(
        "--write-full-rows",
        action="store_true",
        help="Write full pairwise rows including raw model outputs. Disabled by default to avoid large artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

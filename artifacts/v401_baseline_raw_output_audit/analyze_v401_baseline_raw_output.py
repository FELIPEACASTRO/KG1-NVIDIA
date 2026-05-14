#!/usr/bin/env python3
"""V401 CPU audit: can baseline misses be recovered from raw output text?"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def boxed_values(text: str) -> list[str]:
    values: list[str] = []
    # The local outputs usually contain simple \boxed{...}. Nested braces are
    # marked as partial and reviewed manually in the output CSV.
    for match in re.finditer(r"\\boxed\{([^{}]*)\}", text):
        values.append(match.group(1).strip())
    return values


def normalize_answer(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.strip()


def bit_standalone(raw: str, answer: str) -> bool:
    return bool(re.search(rf"(?<![01]){re.escape(answer)}(?![01])", raw))


def run(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.baseline_predictions_csv)
    required = {"id", "type", "answer", "prediction", "raw_output", "correct", "truncated"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(df) != 315:
        raise ValueError(f"expected 315 rows, got {len(df)}")
    misses = df[df["correct"].astype(int) == 0].copy()
    rows: list[dict[str, object]] = []
    for _, row in misses.iterrows():
        answer = normalize_answer(row["answer"])
        prediction = normalize_answer(row["prediction"])
        raw = "" if pd.isna(row["raw_output"]) else str(row["raw_output"])
        boxed = boxed_values(raw)
        answer_in_raw = answer in raw
        answer_in_boxed = answer in boxed
        if row["type"] == "bit_manipulation":
            answer_standalone = bit_standalone(raw, answer)
        else:
            answer_standalone = answer_in_raw
        rows.append(
            {
                "id": row["id"],
                "type": row["type"],
                "answer": answer,
                "prediction": prediction,
                "truncated": int(row["truncated"]),
                "answer_in_raw": bool(answer_in_raw),
                "answer_standalone_or_exact": bool(answer_standalone),
                "answer_in_any_simple_boxed": bool(answer_in_boxed),
                "simple_boxed_count": len(boxed),
                "simple_boxed_values": " | ".join(boxed[:10]),
                "raw_output_chars": len(raw),
                "raw_output_head": raw[:500].replace("\n", "\\n"),
            }
        )

    audit = pd.DataFrame(rows)
    audit_path = out_dir / "v401_baseline_miss_raw_output_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = []
    for family, group in audit.groupby("type", sort=True):
        summary.append(
            {
                "type": family,
                "miss_rows": int(len(group)),
                "answer_in_raw": int(group["answer_in_raw"].sum()),
                "answer_standalone_or_exact": int(group["answer_standalone_or_exact"].sum()),
                "answer_in_any_simple_boxed": int(group["answer_in_any_simple_boxed"].sum()),
                "truncated": int(group["truncated"].sum()),
            }
        )
    summary.append(
        {
            "type": "OVERALL",
            "miss_rows": int(len(audit)),
            "answer_in_raw": int(audit["answer_in_raw"].sum()),
            "answer_standalone_or_exact": int(audit["answer_standalone_or_exact"].sum()),
            "answer_in_any_simple_boxed": int(audit["answer_in_any_simple_boxed"].sum()),
            "truncated": int(audit["truncated"].sum()),
        }
    )
    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "v401_baseline_raw_output_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    decision = {
        "recoverable_simple_boxed_equation": int(
            audit[(audit["type"] == "equation_transform") & (audit["answer_in_any_simple_boxed"])]["id"].nunique()
        ),
        "recoverable_raw_equation": int(
            audit[(audit["type"] == "equation_transform") & (audit["answer_standalone_or_exact"])]["id"].nunique()
        ),
        "recoverable_simple_boxed_bit": int(
            audit[(audit["type"] == "bit_manipulation") & (audit["answer_in_any_simple_boxed"])]["id"].nunique()
        ),
        "recoverable_raw_bit": int(
            audit[(audit["type"] == "bit_manipulation") & (audit["answer_standalone_or_exact"])]["id"].nunique()
        ),
    }
    decision["actionable"] = bool(
        decision["recoverable_simple_boxed_equation"] >= 4 and decision["recoverable_raw_bit"] == 0
    )
    decision["decision"] = (
        "build_extractor_package_probe"
        if decision["actionable"]
        else "no_extractor_gain_adapter_generation_is_bottleneck"
    )
    manifest = {
        "schema_version": "kg1_v401_baseline_raw_output_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_predictions_csv": args.baseline_predictions_csv.as_posix(),
        "rows": int(len(df)),
        "miss_rows": int(len(misses)),
        "summary_csv": summary_path.as_posix(),
        "audit_csv": audit_path.as_posix(),
        "summary": summary,
        "decision": decision,
    }
    manifest_path = out_dir / "v401_baseline_raw_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# V401 Baseline Raw Output Audit",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Type | Miss rows | Answer in raw | Answer in simple boxed | Truncated |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        md.append(
            f"| `{item['type']}` | {item['miss_rows']} | {item['answer_standalone_or_exact']} | "
            f"{item['answer_in_any_simple_boxed']} | {item['truncated']} |"
        )
    md.extend(
        [
            "",
            "## Decision",
            "",
            f"- `decision`: `{decision['decision']}`",
            f"- `actionable`: `{decision['actionable']}`",
            "",
            "If the correct answer is not already present in simple boxed output, this cannot be fixed by extractor changes.",
        ]
    )
    (out_dir / "V401_BASELINE_RAW_OUTPUT_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
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
        default=Path("artifacts/v401_baseline_raw_output_audit/20260514T_v401_raw_output"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

#!/usr/bin/env python3
"""Build a local scoreboard from existing KG1 batch summaries and package manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("rows", [])
        if not rows and all(k in payload for k in ["correct", "equation_transform_correct", "bit_manipulation_correct"]):
            rows = [payload]
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "correct" not in row:
            continue
        copied = dict(row)
        copied["summary_path"] = path.as_posix()
        copied["summary_parent"] = path.parent.as_posix()
        out.append(copied)
    return out


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    def as_int(name: str, default: int = -1) -> int:
        try:
            return int(float(row.get(name, default)))
        except Exception:
            return default

    rows = as_int("rows", -1)
    correct = as_int("correct", -1)
    eq = as_int("equation_transform_correct", -1)
    bit = as_int("bit_manipulation_correct", -1)
    trunc = as_int("truncated", -1)
    if rows < 0:
        if correct > 315:
            rows = 947
        elif eq >= 0 and bit >= 0:
            rows = 315
    eval_kind = "unknown"
    if rows == 315 or (eq >= 0 and bit >= 0 and correct <= 315):
        eval_kind = "weak315"
    elif rows == 947 or correct > 315:
        eval_kind = "full947_or_full_like"
    return {
        "eval_kind": eval_kind,
        "name": str(row.get("name", "")),
        "adapter": str(row.get("adapter", "")),
        "status": str(row.get("status", "")),
        "rows": rows,
        "correct": correct,
        "accuracy": row.get("accuracy", ""),
        "equation_transform_correct": eq,
        "bit_manipulation_correct": bit,
        "truncated": trunc,
        "truncation_rate": row.get("truncation_rate", ""),
        "report_json": str(row.get("report_json", "")),
        "summary_path": str(row.get("summary_path", "")),
    }


def run(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in args.root.rglob("batch_candidate_summary.json"):
        # Avoid scanning hidden VCS and very old copied environments.
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        for row in iter_summary_rows(path):
            rows.append(normalize_summary_row(row))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no candidate summaries found")
    df = df.drop_duplicates(subset=["name", "adapter", "correct", "equation_transform_correct", "bit_manipulation_correct", "truncated", "summary_path"])
    df = df.sort_values(
        ["eval_kind", "correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"],
        ascending=[True, False, False, False, True],
    )
    scoreboard_path = out_dir / "v402_local_candidate_scoreboard.csv"
    df.to_csv(scoreboard_path, index=False)

    weak = df[df["eval_kind"] == "weak315"].copy()
    full = df[df["eval_kind"] == "full947_or_full_like"].copy()
    weak_top = weak.sort_values(["correct", "equation_transform_correct", "bit_manipulation_correct", "truncated"], ascending=[False, False, False, True]).head(20)
    full_top = full.sort_values(["correct", "truncated"], ascending=[False, True]).head(20)
    weak_top.to_csv(out_dir / "v402_top_weak315.csv", index=False)
    full_top.to_csv(out_dir / "v402_top_full_like.csv", index=False)

    best_weak = weak_top.iloc[0].to_dict() if not weak_top.empty else {}
    best_full = full_top.iloc[0].to_dict() if not full_top.empty else {}
    decision = {
        "best_weak_name": best_weak.get("name", ""),
        "best_weak_correct": int(best_weak.get("correct", -1)) if best_weak else -1,
        "best_weak_equation": int(best_weak.get("equation_transform_correct", -1)) if best_weak else -1,
        "best_weak_bit": int(best_weak.get("bit_manipulation_correct", -1)) if best_weak else -1,
        "best_weak_truncated": int(best_weak.get("truncated", -1)) if best_weak else -1,
        "best_full_name": best_full.get("name", ""),
        "best_full_correct": int(best_full.get("correct", -1)) if best_full else -1,
        "best_full_truncated": int(best_full.get("truncated", -1)) if best_full else -1,
        "found_local_weak_above_192": bool((weak["correct"] > 192).any()) if not weak.empty else False,
        "found_local_full_above_823": bool((full["correct"] > 823).any()) if not full.empty else False,
    }
    manifest = {
        "schema_version": "kg1_v402_local_candidate_scoreboard_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": args.root.as_posix(),
        "summary_count": int(len(df)),
        "scoreboard_csv": scoreboard_path.as_posix(),
        "top_weak_csv": (out_dir / "v402_top_weak315.csv").as_posix(),
        "top_full_csv": (out_dir / "v402_top_full_like.csv").as_posix(),
        "decision": decision,
    }
    (out_dir / "v402_local_candidate_scoreboard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md = [
        "# V402 Local Candidate Scoreboard",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- `found_local_weak_above_192`: `{decision['found_local_weak_above_192']}`",
        f"- `found_local_full_above_823`: `{decision['found_local_full_above_823']}`",
        "",
        "## Top Weak",
        "",
        "| Name | Correct | Equation | Bit | Truncated | Path |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in weak_top.head(10).iterrows():
        md.append(
            f"| `{row['name']}` | {row['correct']} | {row['equation_transform_correct']} | "
            f"{row['bit_manipulation_correct']} | {row['truncated']} | `{row['summary_path']}` |"
        )
    md.extend(["", "## Top Full-Like", "", "| Name | Correct | Truncated | Path |", "|---|---:|---:|---|"])
    for _, row in full_top.head(10).iterrows():
        md.append(f"| `{row['name']}` | {row['correct']} | {row['truncated']} | `{row['summary_path']}` |")
    (out_dir / "V402_LOCAL_CANDIDATE_SCOREBOARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v402_local_candidate_scoreboard/20260514T_v402_scoreboard"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

#!/usr/bin/env python3
"""V428 parser/raw-output rescue audit.

CPU-only audit for adapter-like raw outputs. It checks whether alternative
extraction strategies would recover weak ACC from existing raw generations.
This is diagnostic only; Kaggle scoring still owns the official parser.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, extract_boxed_answers, extract_final_answer, verify_answer  # noqa: E402


BASELINE_CSV = ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V425_ADAPTER_CSV = (
    ROOT
    / "artifacts/v425_global_prediction_archaeology/20260515T_v425_global_prediction_archaeology/"
    / "v425_adapter_like_candidates.csv"
)
OUT_DIR = ROOT / "artifacts/v428_parser_raw_output_rescue/20260515T_v428_parser_raw_output_rescue"

EXPECTED = {"total": 192, "equation": 56, "bit": 136, "truncated": 0}
RAW_COLUMNS = ("raw_output", "candidate_raw_output", "baseline_raw_output", "response", "output", "text", "prediction")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def family_for(row: dict[str, str]) -> str:
    return str(row.get("family") or row.get("type") or row.get("task_type") or classify_puzzle(row.get("prompt", "")))


def strategy_last_boxed(text: str) -> str:
    return extract_final_answer(text)


def strategy_first_boxed(text: str) -> str:
    values = [item.strip() for item in extract_boxed_answers(text) if item.strip()]
    return values[0] if values else extract_final_answer(text)


def strategy_last_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def strategy_first_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[0] if lines else ""


def strategy_last_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    return nums[-1] if nums else extract_final_answer(text)


def strategy_first_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    return nums[0] if nums else extract_final_answer(text)


STRATEGIES: dict[str, Callable[[str], str]] = {
    "official_last_boxed": strategy_last_boxed,
    "first_boxed": strategy_first_boxed,
    "last_line": strategy_last_line,
    "first_line": strategy_first_line,
    "last_number": strategy_last_number,
    "first_number": strategy_first_number,
}


def score(rows: list[dict[str, str]], ref: dict[str, dict[str, str]], column: str, strategy: str) -> dict[str, Any]:
    fn = STRATEGIES[strategy]
    total = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    gains: list[str] = []
    losses: list[str] = []
    for row in rows:
        row_id = row.get("id", "")
        if row_id not in ref:
            continue
        base = ref[row_id]
        pred = fn(row.get(column, ""))
        ok = verify_answer(base["answer"], pred)
        base_ok = verify_answer(base["answer"], base["prediction"])
        fam = family_for(base)
        total["rows"] += 1
        total["correct"] += int(ok)
        families[fam]["rows"] += 1
        families[fam]["correct"] += int(ok)
        if ok and not base_ok:
            gains.append(row_id)
        if base_ok and not ok:
            losses.append(row_id)
    return {
        "rows": int(total["rows"]),
        "correct": int(total["correct"]),
        "equation_transform_correct": int(families["equation_transform"]["correct"]),
        "bit_manipulation_correct": int(families["bit_manipulation"]["correct"]),
        "gains": gains,
        "losses": losses,
    }


def main() -> int:
    baseline = read_csv(BASELINE_CSV)
    ref = {row["id"]: row for row in baseline}
    adapter_paths = sorted({row["path"] for row in read_csv(V425_ADAPTER_CSV)})
    summary: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for rel in adapter_paths:
        path = ROOT / rel
        if not path.exists():
            continue
        rows = read_csv(path)
        columns = [column for column in rows[0].keys() if column in RAW_COLUMNS] if rows else []
        for column in columns:
            for strategy in STRATEGIES:
                result = score(rows, ref, column, strategy)
                if result["rows"] < 300:
                    continue
                item = {
                    "path": rel,
                    "column": column,
                    "strategy": strategy,
                    "rows": result["rows"],
                    "correct": result["correct"],
                    "equation_transform_correct": result["equation_transform_correct"],
                    "bit_manipulation_correct": result["bit_manipulation_correct"],
                    "delta_total": result["correct"] - EXPECTED["total"],
                    "delta_equation": result["equation_transform_correct"] - EXPECTED["equation"],
                    "delta_bit": result["bit_manipulation_correct"] - EXPECTED["bit"],
                    "gains_vs_baseline": len(result["gains"]),
                    "losses_vs_baseline": len(result["losses"]),
                    "gain_ids": ";".join(sorted(result["gains"])),
                    "loss_ids": ";".join(sorted(result["losses"])),
                }
                summary.append(item)
                if (
                    item["correct"] > EXPECTED["total"]
                    and item["equation_transform_correct"] > EXPECTED["equation"]
                    and item["bit_manipulation_correct"] >= EXPECTED["bit"]
                    and not result["losses"]
                ):
                    best_rows.append(item)
    summary.sort(
        key=lambda row: (
            -int(row["correct"]),
            -int(row["equation_transform_correct"]),
            -int(row["bit_manipulation_correct"]),
            int(row["losses_vs_baseline"]),
            row["path"],
            row["column"],
            row["strategy"],
        )
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "path",
        "column",
        "strategy",
        "rows",
        "correct",
        "equation_transform_correct",
        "bit_manipulation_correct",
        "delta_total",
        "delta_equation",
        "delta_bit",
        "gains_vs_baseline",
        "losses_vs_baseline",
        "gain_ids",
        "loss_ids",
    ]
    write_csv(OUT_DIR / "v428_parser_strategy_scores.csv", summary, cols)
    write_csv(OUT_DIR / "v428_promotable_parser_rescues.csv", best_rows, cols)
    decision = "v428_parser_rescue_found" if best_rows else "v428_no_parser_rescue"
    manifest = {
        "schema_version": "kg1_v428_parser_raw_output_rescue_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": EXPECTED,
        "adapter_paths_scanned": len(adapter_paths),
        "strategy_rows": len(summary),
        "promotable_rescue_count": len(best_rows),
        "best_strategy": summary[0] if summary else {},
        "decision": {
            "decision": decision,
            "hf_gpu_allowed": False,
            "reason": (
                "A parser/raw-output strategy beats baseline without losses."
                if best_rows
                else "No adapter-like raw output contains a no-loss parser rescue over V291/V290."
            ),
            "next_action": "Do not spend GPU from parser hypothesis unless a submit-compatible extraction path is proven.",
        },
        "outputs": {
            "manifest_json": str((OUT_DIR / "v428_parser_raw_output_rescue_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V428_PARSER_RAW_OUTPUT_RESCUE.md").relative_to(ROOT)),
            "scores_csv": str((OUT_DIR / "v428_parser_strategy_scores.csv").relative_to(ROOT)),
            "promotable_csv": str((OUT_DIR / "v428_promotable_parser_rescues.csv").relative_to(ROOT)),
        },
    }
    write_json(OUT_DIR / "v428_parser_raw_output_rescue_manifest.json", manifest)
    report = [
        "# V428 Parser Raw-Output Rescue",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        f"- Adapter-like paths scanned: `{len(adapter_paths)}`.",
        f"- Strategy rows scored: `{len(summary)}`.",
        f"- Promotable parser rescues: `{len(best_rows)}`.",
        "",
        "| Best strategy | Total | equation | bit | Gains | Losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if summary:
        best = summary[0]
        report.append(
            f"| `{best['path']}::{best['column']}::{best['strategy']}` | `{best['correct']}` | "
            f"`{best['equation_transform_correct']}` | `{best['bit_manipulation_correct']}` | "
            f"`{best['gains_vs_baseline']}` | `{best['losses_vs_baseline']}` |"
        )
    report += ["", f"Decision: `{decision}`. {manifest['decision']['reason']}"]
    (OUT_DIR / "V428_PARSER_RAW_OUTPUT_RESCUE.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

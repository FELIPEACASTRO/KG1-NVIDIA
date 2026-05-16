"""Revalidate historical KG1 prediction CSVs with submit-safe label-free scoring.

This audit is CPU-only.  It does not run inference, train, package, or submit.
It scans local artifacts for prediction CSVs, recomputes correctness using the
current submit-safe path, and flags candidates whose stored metrics differ from
the corrected metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import canonical_answer, extract_final_answer, verify_answer  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v505_label_free_candidate_revalidation"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_family(row: dict[str, str]) -> str:
    for key in ("family", "type", "task_type"):
        value = str(row.get(key) or "").strip()
        if value:
            if value == "equation_transform":
                return "equation_transform"
            if value == "bit_manipulation":
                return "bit_manipulation"
            return value
    prompt = str(row.get("prompt") or "").lower()
    if "alice" in prompt or "equation" in prompt:
        return "equation_transform"
    if "bit" in prompt or "binary" in prompt:
        return "bit_manipulation"
    return "unknown"


def bool_from_cell(value: object) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def score_row(row: dict[str, str]) -> dict[str, object]:
    answer = row.get("answer", "")
    stored_prediction = row.get("prediction", "")
    raw_output = row.get("raw_output", "")
    if raw_output:
        label_free_prediction = extract_final_answer(raw_output)
        metric_source = "raw_output_label_free"
    else:
        label_free_prediction = stored_prediction
        metric_source = "stored_prediction_only"
    label_free_correct = verify_answer(answer, label_free_prediction)
    stored_correct_cell = bool_from_cell(row.get("correct"))
    stored_prediction_correct = verify_answer(answer, stored_prediction)
    stored_correct = stored_correct_cell if stored_correct_cell is not None else stored_prediction_correct
    trunc = bool_from_cell(row.get("truncated"))
    return {
        "id": row.get("id", ""),
        "family": infer_family(row),
        "answer": canonical_answer(answer),
        "stored_prediction": stored_prediction,
        "label_free_prediction": label_free_prediction,
        "metric_source": metric_source,
        "stored_correct": bool(stored_correct),
        "stored_prediction_correct": bool(stored_prediction_correct),
        "label_free_correct": bool(label_free_correct),
        "stored_vs_label_free_diff": bool(stored_correct) != bool(label_free_correct),
        "prediction_changed": canonical_answer(stored_prediction) != canonical_answer(label_free_prediction),
        "truncated": bool(trunc) if trunc is not None else False,
    }


def summarize_scored_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    family = defaultdict(lambda: {"total": 0, "stored_correct": 0, "label_free_correct": 0})
    metric_sources = Counter()
    diffs = 0
    prediction_changes = 0
    truncated = 0
    for row in rows:
        fam = str(row["family"])
        family[fam]["total"] += 1
        family[fam]["stored_correct"] += int(bool(row["stored_correct"]))
        family[fam]["label_free_correct"] += int(bool(row["label_free_correct"]))
        metric_sources[str(row["metric_source"])] += 1
        diffs += int(bool(row["stored_vs_label_free_diff"]))
        prediction_changes += int(bool(row["prediction_changed"]))
        truncated += int(bool(row["truncated"]))
    total = len(rows)
    stored_correct = sum(int(bool(row["stored_correct"])) for row in rows)
    label_free_correct = sum(int(bool(row["label_free_correct"])) for row in rows)
    return {
        "total": total,
        "stored_correct": stored_correct,
        "label_free_correct": label_free_correct,
        "delta_label_free_minus_stored": label_free_correct - stored_correct,
        "stored_vs_label_free_diff_rows": diffs,
        "prediction_changed_rows": prediction_changes,
        "truncated": truncated,
        "metric_sources": dict(metric_sources),
        "family": dict(sorted(family.items())),
    }


def is_prediction_csv(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".csv"):
        return False
    return "prediction" in name or "predictions" in name


def audit_file(path: Path) -> dict[str, object] | None:
    rows = read_csv_rows(path)
    if not rows:
        return None
    columns = set(rows[0])
    if "answer" not in columns or "prediction" not in columns:
        return None
    scored = [score_row(row) for row in rows]
    summary = summarize_scored_rows(scored)
    summary.update(
        {
            "path": str(path),
            "name": path.stem,
            "has_raw_output": "raw_output" in columns,
            "columns": sorted(columns),
            "rows": len(rows),
        }
    )
    return {"summary": summary, "scored_rows": scored}


def decision_from_summary(summary: dict[str, object], baseline_total: int = 192) -> str:
    total = int(summary["total"])
    correct = int(summary["label_free_correct"])
    has_raw_output = bool(summary.get("has_raw_output"))
    family = summary.get("family", {})
    if not isinstance(family, dict):
        family = {}
    eq = int(family.get("equation_transform", {}).get("label_free_correct", 0)) if isinstance(family.get("equation_transform"), dict) else 0
    bit = int(family.get("bit_manipulation", {}).get("label_free_correct", 0)) if isinstance(family.get("bit_manipulation"), dict) else 0
    trunc = int(summary.get("truncated", 0))
    if total == 315 and not has_raw_output:
        return "not_adapter_only_reference_solver_or_postprocessor"
    if total == 315 and correct > baseline_total and eq >= 60 and bit >= 136 and trunc == 0:
        return "promote_for_official_like_full_eval"
    if total == 315 and correct > baseline_total and trunc == 0:
        return "needs_family_guard_review"
    if total == 315:
        return "do_not_promote_weak315"
    return "not_weak315_reference_only"


def run(args: argparse.Namespace) -> int:
    print("=== V505 LABEL-FREE CANDIDATE REVALIDATION START ===", flush=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    print("generated_at_utc =", generated_at, flush=True)
    print("scan_root =", args.scan_root, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(path for path in args.scan_root.rglob("*.csv") if is_prediction_csv(path))
    print("prediction_csv_candidate_count =", len(candidates), flush=True)

    summaries: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    for index, path in enumerate(candidates, start=1):
        if index % 25 == 0 or index == 1:
            print(f"auditing_file_index={index}/{len(candidates)} path={path}", flush=True)
        try:
            result = audit_file(path)
        except Exception as exc:  # noqa: BLE001 - audit should continue across historical files
            summaries.append(
                {
                    "path": str(path),
                    "name": path.stem,
                    "error": repr(exc),
                    "decision": "audit_error",
                }
            )
            continue
        if result is None:
            continue
        summary = dict(result["summary"])
        summary["decision"] = decision_from_summary(summary, baseline_total=args.baseline_total)
        summaries.append(summary)
        if int(summary["stored_vs_label_free_diff_rows"]) or int(summary["prediction_changed_rows"]):
            for row in result["scored_rows"]:
                if row["stored_vs_label_free_diff"] or row["prediction_changed"]:
                    diff_row = dict(row)
                    diff_row["source_csv"] = str(path)
                    diff_rows.append(diff_row)

    summaries.sort(
        key=lambda row: (
            int(row.get("label_free_correct", -1)) if str(row.get("rows", "")) == "315" else -1,
            int(row.get("label_free_correct", -1)),
        ),
        reverse=True,
    )

    summary_csv = args.output_dir / "v505_label_free_revalidation_summary.csv"
    write_csv(
        summary_csv,
        summaries,
        [
            "name",
            "path",
            "rows",
            "has_raw_output",
            "label_free_correct",
            "stored_correct",
            "delta_label_free_minus_stored",
            "truncated",
            "stored_vs_label_free_diff_rows",
            "prediction_changed_rows",
            "metric_sources",
            "family",
            "decision",
            "error",
        ],
    )
    diff_csv = args.output_dir / "v505_label_free_diff_rows.csv"
    write_csv(
        diff_csv,
        diff_rows,
        [
            "source_csv",
            "id",
            "family",
            "answer",
            "stored_prediction",
            "label_free_prediction",
            "metric_source",
            "stored_correct",
            "stored_prediction_correct",
            "label_free_correct",
            "stored_vs_label_free_diff",
            "prediction_changed",
            "truncated",
        ],
    )

    weak315 = [row for row in summaries if str(row.get("rows", "")) == "315" and not row.get("error")]
    weak315_adapter_raw = [row for row in weak315 if bool(row.get("has_raw_output"))]
    weak315_reference_only = [row for row in weak315 if not bool(row.get("has_raw_output"))]
    promotable = [row for row in weak315 if row.get("decision") == "promote_for_official_like_full_eval"]
    best = weak315_adapter_raw[0] if weak315_adapter_raw else None
    best_reference = weak315_reference_only[0] if weak315_reference_only else None
    manifest = {
        "schema_version": "kg1_v505_label_free_candidate_revalidation_v1",
        "generated_at_utc": generated_at,
        "scan_root": str(args.scan_root),
        "output_dir": str(args.output_dir),
        "files_scanned": len(candidates),
        "files_scored": len([row for row in summaries if not row.get("error")]),
        "weak315_scored": len(weak315),
        "weak315_adapter_raw_scored": len(weak315_adapter_raw),
        "weak315_reference_only_scored": len(weak315_reference_only),
        "promotable_count": len(promotable),
        "best_adapter_raw_weak315": best,
        "best_reference_only_weak315": best_reference,
        "summary_csv": str(summary_csv),
        "diff_csv": str(diff_csv),
        "decision": {
            "status": "promotable_candidate_found" if promotable else "no_label_free_candidate_above_gate",
            "next_action": (
                "Run official-like full eval for the top promotable candidate."
                if promotable
                else "Do not spend GPU; continue CPU teacher/verifier discovery or inspect label-free diffs only."
            ),
        },
    }
    manifest_path = args.output_dir / "v505_label_free_candidate_revalidation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("summary_csv =", summary_csv, flush=True)
    print("diff_csv =", diff_csv, flush=True)
    print("manifest_path =", manifest_path, flush=True)
    print("weak315_scored =", len(weak315), flush=True)
    print("promotable_count =", len(promotable), flush=True)
    print("weak315_adapter_raw_scored =", len(weak315_adapter_raw), flush=True)
    print("weak315_reference_only_scored =", len(weak315_reference_only), flush=True)
    if best:
        print("best_adapter_raw_weak315 =", json.dumps(best, sort_keys=True)[:2000], flush=True)
    if best_reference:
        print("best_reference_only_weak315 =", json.dumps(best_reference, sort_keys=True)[:2000], flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("=== V505 LABEL-FREE CANDIDATE REVALIDATION END ===", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-total", type=int, default=192)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

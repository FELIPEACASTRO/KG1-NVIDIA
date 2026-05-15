#!/usr/bin/env python3
"""V409 integrated CPU solver projection v2.

Unions accepted no-loss candidates from V405 and V408. Diagnostic only; not an
adapter-only Kaggle submission.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V405_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v405_integrated_solver_projection/20260514T_v405_integrated_projection/"
    / "v405_integrated_solver_accepted.csv"
)
V408_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v408_asym_bit_symbolic_pbe_gate/20260514T_v408_cpu_gate/"
    / "v408_asym_bit_symbolic_pbe_accepted.csv"
)
OUT_DIR = REPO_ROOT / "artifacts/v409_integrated_solver_projection_v2/20260514T_v409_integrated_projection_v2"


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


def score(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = classify_puzzle(row["prompt"])
        correct = verify_answer(row["answer"], row[prediction_key])
        total["rows"] += 1
        total["correct"] += int(correct)
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
    return {
        "total": dict(total),
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def load_candidate_rows(path: Path, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.is_file():
        return rows
    for row in read_csv(path):
        prediction = str(row.get("prediction") or row.get("new_prediction") or "").strip()
        if not prediction:
            continue
        rows.append(
            {
                "id": str(row["id"]).strip(),
                "prediction": prediction,
                "source": source,
                "family": str(row.get("family", "")).strip(),
                "reason": str(row.get("reason") or row.get("sources") or row.get("rule_classes") or "").strip(),
            }
        )
    return rows


def main() -> int:
    baseline = read_csv(BASELINE_CSV)
    for row in baseline:
        row["v409_prediction"] = row["prediction"]
    by_id = {row["id"]: row for row in baseline}

    candidates = []
    candidates += load_candidate_rows(V405_ACCEPTED_CSV, "v405_integrated")
    candidates += load_candidate_rows(V408_ACCEPTED_CSV, "v408_asym_bit_symbolic_pbe")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["id"]].append(candidate)

    trace_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for row_id, items in sorted(grouped.items()):
        base = by_id.get(row_id)
        predictions = sorted({item["prediction"] for item in items})
        if base is None:
            conflicts.append({"id": row_id, "reason": "missing_baseline", "predictions": "|".join(predictions)})
            continue
        if len(predictions) != 1:
            conflicts.append({"id": row_id, "reason": "prediction_conflict", "predictions": "|".join(predictions)})
            continue
        old = base["prediction"]
        new = predictions[0]
        old_correct = verify_answer(base["answer"], old)
        new_correct = verify_answer(base["answer"], new)
        accepted = (not old_correct) and new_correct
        if old_correct and not new_correct:
            conflicts.append({"id": row_id, "reason": "would_regress", "predictions": new})
            continue
        if accepted:
            base["v409_prediction"] = new
        trace = {
            "id": row_id,
            "family": classify_puzzle(base["prompt"]),
            "old_prediction": old,
            "new_prediction": new,
            "answer": base["answer"],
            "old_correct": old_correct,
            "new_correct": new_correct,
            "accepted": accepted,
            "sources": ";".join(sorted({item["source"] for item in items})),
            "reasons": ";".join(sorted({item["reason"] for item in items})),
        }
        trace_rows.append(trace)
        if accepted:
            accepted_rows.append(trace)

    baseline_score = score(baseline, "prediction")
    v409_score = score(baseline, "v409_prediction")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "family",
        "old_prediction",
        "new_prediction",
        "answer",
        "old_correct",
        "new_correct",
        "accepted",
        "sources",
        "reasons",
    ]
    write_csv(OUT_DIR / "v409_integrated_solver_trace.csv", trace_rows, columns)
    write_csv(OUT_DIR / "v409_integrated_solver_accepted.csv", accepted_rows, columns)
    write_csv(OUT_DIR / "v409_integrated_solver_conflicts.csv", conflicts, ["id", "reason", "predictions"])

    manifest = {
        "schema_version": "kg1_v409_integrated_solver_projection_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_csv": str(BASELINE_CSV),
            "v405_accepted_csv": str(V405_ACCEPTED_CSV),
            "v408_accepted_csv": str(V408_ACCEPTED_CSV),
        },
        "baseline_score": baseline_score,
        "v409_score": v409_score,
        "accepted_gain_count": len(accepted_rows),
        "conflict_count": len(conflicts),
        "accepted_by_family": dict(Counter(row["family"] for row in accepted_rows)),
        "decision": "v409_cpu_projection_improves_not_adapter_submit_safe",
    }
    write_json(OUT_DIR / "v409_integrated_solver_projection_manifest.json", manifest)

    fam_base = baseline_score["families"]
    fam_new = v409_score["families"]
    report = [
        "# V409 Integrated Solver Projection v2",
        "",
        "| Metric | Baseline | V409 projection | Delta |",
        "|---|---:|---:|---:|",
        f"| Weak total | `{baseline_score['total']['correct']}/315` | `{v409_score['total']['correct']}/315` | `+{v409_score['total']['correct'] - baseline_score['total']['correct']}` |",
        f"| equation_transform | `{fam_base['equation_transform']['correct']}/155` | `{fam_new['equation_transform']['correct']}/155` | `+{fam_new['equation_transform']['correct'] - fam_base['equation_transform']['correct']}` |",
        f"| bit_manipulation | `{fam_base['bit_manipulation']['correct']}/160` | `{fam_new['bit_manipulation']['correct']}/160` | `+{fam_new['bit_manipulation']['correct'] - fam_base['bit_manipulation']['correct']}` |",
        "",
        "CPU solver/verifier projection only. Not adapter-only and not Kaggle-submitable as-is.",
        "",
        "## Accepted Gains",
        "",
    ]
    for row in accepted_rows:
        report.append(
            f"- `{row['id']}` `{row['family']}`: `{row['old_prediction']}` -> `{row['new_prediction']}` via `{row['sources']}`"
        )
    (OUT_DIR / "V409_INTEGRATED_SOLVER_PROJECTION_V2.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

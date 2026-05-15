#!/usr/bin/env python3
"""V405 integrated CPU solver projection.

Combines only accepted no-loss CPU solver candidates from:
- V324 expanded equation numeric gate
- V329/V404 symbolic cryptarithm gate
- V403 exact global bit solver gate

This is diagnostic. It is not a Kaggle-submitable adapter package.
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
V324_CSV = (
    REPO_ROOT
    / "artifacts/v394_equation_row_level_inventory/20260514T_cpu_gate/"
    / "v324_on_v290_checkpoint6/v324_equation_expanded_solver_accepted_candidates.csv"
)
V329_CSV = (
    REPO_ROOT
    / "artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/"
    / "v329_symbolic_cryptarithm_accepted_candidates.csv"
)
V403_CSV = (
    REPO_ROOT
    / "artifacts/v403_formal_solver_abstain_audit/20260514T_v403_solver_abstain/"
    / "v403_accepted_formal_solver_candidates.csv"
)
OUT_DIR = REPO_ROOT / "artifacts/v405_integrated_solver_projection/20260514T_v405_integrated_projection"


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


def load_candidates(path: Path, source: str, prediction_keys: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.is_file():
        return rows
    for row in read_csv(path):
        prediction = ""
        for key in prediction_keys:
            if row.get(key):
                prediction = str(row[key]).strip()
                break
        if not prediction:
            continue
        rows.append(
            {
                "id": str(row["id"]).strip(),
                "prediction": prediction,
                "source": source,
                "rule_class": str(row.get("rule_class", "")).strip(),
                "proof": str(row.get("proof", "")).strip(),
            }
        )
    return rows


def score(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    for row in rows:
        family = classify_puzzle(row["prompt"])
        correct = verify_answer(row["answer"], row[prediction_key])
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
        total["rows"] += 1
        total["correct"] += int(correct)
    return {
        "total": dict(total),
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def main() -> int:
    baseline = read_csv(BASELINE_CSV)
    for row in baseline:
        row["integrated_prediction"] = row["prediction"]
    by_id = {row["id"]: dict(row) for row in baseline}
    candidates = []
    candidates += load_candidates(V324_CSV, "v324_equation_numeric", ("prediction",))
    candidates += load_candidates(V329_CSV, "v329_symbolic_cryptarithm", ("prediction",))
    candidates += load_candidates(V403_CSV, "v403_bit_global_exact", ("candidate_prediction", "prediction"))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["id"]].append(candidate)

    trace_rows: list[dict[str, Any]] = []

    conflicts = []
    for row_id, items in sorted(grouped.items()):
        predictions = sorted({item["prediction"] for item in items})
        base = by_id.get(row_id)
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
            by_id[row_id]["integrated_prediction"] = new
        trace_rows.append(
            {
                "id": row_id,
                "family": classify_puzzle(base["prompt"]),
                "old_prediction": old,
                "new_prediction": new,
                "answer": base["answer"],
                "old_correct": old_correct,
                "new_correct": new_correct,
                "accepted": accepted,
                "sources": ";".join(sorted({item["source"] for item in items})),
                "rule_classes": ";".join(sorted({item["rule_class"] for item in items})),
            }
        )

    integrated_rows = [by_id[row["id"]] for row in baseline]
    baseline_score = score(baseline, "prediction")
    integrated_score = score(integrated_rows, "integrated_prediction")
    accepted_rows = [row for row in trace_rows if row["accepted"]]

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
        "rule_classes",
    ]
    write_csv(OUT_DIR / "v405_integrated_solver_trace.csv", trace_rows, columns)
    write_csv(OUT_DIR / "v405_integrated_solver_accepted.csv", accepted_rows, columns)
    write_csv(OUT_DIR / "v405_integrated_solver_predictions.csv", integrated_rows, list(integrated_rows[0].keys()))
    write_csv(OUT_DIR / "v405_integrated_solver_conflicts.csv", conflicts, ["id", "reason", "predictions"])

    manifest = {
        "schema_version": "kg1_v405_integrated_solver_projection_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_score": baseline_score,
        "integrated_score": integrated_score,
        "accepted_gain_count": len(accepted_rows),
        "conflict_count": len(conflicts),
        "decision": "cpu_solver_projection_only_not_adapter_submit_safe",
        "inputs": {
            "baseline_csv": str(BASELINE_CSV),
            "v324_csv": str(V324_CSV),
            "v329_csv": str(V329_CSV),
            "v403_csv": str(V403_CSV),
        },
    }
    write_json(OUT_DIR / "v405_integrated_solver_projection_manifest.json", manifest)

    fam = integrated_score["families"]
    report = [
        "# V405 Integrated Solver Projection",
        "",
        "| Metric | Baseline | Integrated CPU solver | Delta |",
        "|---|---:|---:|---:|",
        f"| Total weak | `{baseline_score['total']['correct']}/315` | `{integrated_score['total']['correct']}/315` | `+{integrated_score['total']['correct'] - baseline_score['total']['correct']}` |",
        f"| equation_transform | `{baseline_score['families']['equation_transform']['correct']}/155` | `{fam['equation_transform']['correct']}/155` | `+{fam['equation_transform']['correct'] - baseline_score['families']['equation_transform']['correct']}` |",
        f"| bit_manipulation | `{baseline_score['families']['bit_manipulation']['correct']}/160` | `{fam['bit_manipulation']['correct']}/160` | `+{fam['bit_manipulation']['correct'] - baseline_score['families']['bit_manipulation']['correct']}` |",
        "",
        "This is a CPU solver/verifier projection only. It is not an adapter-only Kaggle submission.",
        "",
        "## Accepted Gains",
        "",
    ]
    for row in accepted_rows:
        report.append(
            f"- `{row['id']}` `{row['family']}`: `{row['old_prediction']}` -> `{row['new_prediction']}` via `{row['sources']}`"
        )
    (OUT_DIR / "V405_INTEGRATED_SOLVER_PROJECTION.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

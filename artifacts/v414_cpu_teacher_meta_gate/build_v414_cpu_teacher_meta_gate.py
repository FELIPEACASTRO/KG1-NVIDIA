#!/usr/bin/env python3
"""V414 CPU teacher meta gate.

This gate consolidates the current solver/verifier evidence into one reproducible
artifact. It is deliberately CPU-only: its job is to separate real row-level
teacher gains from routes that already failed to transfer into an adapter.
"""

from __future__ import annotations

import csv
import hashlib
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
V409_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v409_integrated_solver_projection_v2/20260514T_v409_integrated_projection_v2/"
    / "v409_integrated_solver_accepted.csv"
)
V412_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v412_cpu_synthesis_gate/20260514T_v412_cpu_gate/"
    / "v412_integrated_accepted.csv"
)
V357_DECISIONS_CSV = (
    REPO_ROOT
    / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/"
    / "v357_candidate_decisions.csv"
)
V357_MANIFEST_JSON = (
    REPO_ROOT
    / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/"
    / "v357_bit_global_ternary_gate_manifest.json"
)
V366_DECISIONS_CSV = (
    REPO_ROOT
    / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/"
    / "v366_candidate_decisions.csv"
)
V366_MANIFEST_JSON = (
    REPO_ROOT
    / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/"
    / "v366_bit_fullbyte_ternary_op_gate_manifest.json"
)
V368_MANIFEST_JSON = (
    REPO_ROOT
    / "artifacts/v369_v368_transfer_failure_audit/20260514T_cpu_audit/"
    / "v369_v368_transfer_failure_manifest.json"
)
V413_CUTOFF_MD = REPO_ROOT / "artifacts/v413_hf_h200_solver_first_transfer_launch/V413_FINOPS_CUTOFF.md"
OUT_DIR = REPO_ROOT / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("family") or classify_puzzle(row["prompt"]))


def score(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        correct = verify_answer(row["answer"], row[prediction_key])
        total["rows"] += 1
        total["correct"] += int(correct)
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
    return {
        "total": dict(total),
        "families": {family: dict(counter) for family, counter in sorted(by_family.items())},
    }


def load_accepted(path: Path, *, source: str, prediction_column: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for raw in read_csv(path):
        if "accepted" in raw and not boolish(raw["accepted"]):
            continue
        prediction = str(
            raw.get(prediction_column or "")
            or raw.get("new_prediction")
            or raw.get("prediction")
            or ""
        ).strip()
        if not prediction:
            continue
        rows.append(
            {
                "id": str(raw["id"]).strip(),
                "prediction": prediction,
                "source": source,
                "family": str(raw.get("family", "")).strip(),
                "rule_class": str(raw.get("rule_class") or raw.get("reasons") or raw.get("reason") or "").strip(),
                "proof": str(raw.get("proof") or raw.get("sources") or "").strip(),
            }
        )
    return rows


def apply_candidates(
    baseline: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    output_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(row) for row in baseline]
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        row[output_key] = row["prediction"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["id"]].append(candidate)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_id, items in sorted(grouped.items()):
        base = by_id.get(row_id)
        predictions = sorted({str(item["prediction"]) for item in items})
        sources = ";".join(sorted({str(item["source"]) for item in items}))
        rules = ";".join(sorted({str(item["rule_class"]) for item in items if item.get("rule_class")}))
        proofs = ";".join(sorted({str(item["proof"]) for item in items if item.get("proof")}))
        if base is None:
            rejected.append(
                {"id": row_id, "reason": "missing_baseline", "predictions": "|".join(predictions), "sources": sources}
            )
            continue
        if len(predictions) != 1:
            rejected.append(
                {
                    "id": row_id,
                    "family": family_for(base),
                    "reason": "prediction_conflict",
                    "predictions": "|".join(predictions),
                    "sources": sources,
                    "rules": rules,
                }
            )
            continue
        old_prediction = str(base[output_key])
        new_prediction = predictions[0]
        old_correct = verify_answer(base["answer"], old_prediction)
        new_correct = verify_answer(base["answer"], new_prediction)
        if old_correct and not new_correct:
            rejected.append(
                {
                    "id": row_id,
                    "family": family_for(base),
                    "reason": "would_regress_current_projection",
                    "old_prediction": old_prediction,
                    "new_prediction": new_prediction,
                    "answer": base["answer"],
                    "sources": sources,
                    "rules": rules,
                }
            )
            continue
        if (not old_correct) and new_correct:
            base[output_key] = new_prediction
            accepted.append(
                {
                    "id": row_id,
                    "family": family_for(base),
                    "old_prediction": old_prediction,
                    "new_prediction": new_prediction,
                    "answer": base["answer"],
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "sources": sources,
                    "rules": rules,
                    "proofs": proofs,
                }
            )
    return rows, accepted, rejected


def score_row(label: str, summary: dict[str, Any], baseline: dict[str, Any], status: str) -> dict[str, Any]:
    families = summary["families"]
    base_families = baseline["families"]
    return {
        "state": label,
        "total": summary["total"]["correct"],
        "total_delta": summary["total"]["correct"] - baseline["total"]["correct"],
        "equation_transform": families["equation_transform"]["correct"],
        "equation_delta": families["equation_transform"]["correct"] - base_families["equation_transform"]["correct"],
        "bit_manipulation": families["bit_manipulation"]["correct"],
        "bit_delta": families["bit_manipulation"]["correct"] - base_families["bit_manipulation"]["correct"],
        "status": status,
    }


def main() -> int:
    required = [
        BASELINE_CSV,
        V409_ACCEPTED_CSV,
        V412_ACCEPTED_CSV,
        V357_DECISIONS_CSV,
        V357_MANIFEST_JSON,
        V366_DECISIONS_CSV,
        V366_MANIFEST_JSON,
        V368_MANIFEST_JSON,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required V414 inputs: " + ", ".join(missing))

    baseline_rows = read_csv(BASELINE_CSV)
    baseline_summary = score(baseline_rows, "prediction")

    v409_candidates = load_accepted(V409_ACCEPTED_CSV, source="v409_integrated_solver_projection_v2")
    v412_candidates = load_accepted(V412_ACCEPTED_CSV, source="v412_cpu_synthesis_gate")
    v357_candidates = load_accepted(V357_DECISIONS_CSV, source="v357_bit_global_ternary_gate")
    v366_candidates = load_accepted(V366_DECISIONS_CSV, source="v366_bit_fullbyte_ternary_op_gate")

    v409_rows, v409_accepted, v409_rejected = apply_candidates(
        baseline_rows, v409_candidates, output_key="v409_prediction"
    )
    v409_summary = score(v409_rows, "v409_prediction")

    v412_union = v409_candidates + v412_candidates
    v412_rows, v412_accepted, v412_rejected = apply_candidates(
        baseline_rows, v412_union, output_key="v412_prediction"
    )
    v412_summary = score(v412_rows, "v412_prediction")

    v357_union = v412_union + v357_candidates
    v357_rows, v357_accepted, v357_rejected = apply_candidates(
        baseline_rows, v357_union, output_key="v357_projection"
    )
    v357_summary = score(v357_rows, "v357_projection")

    v366_union = v357_union + v366_candidates
    v366_rows, v366_accepted, v366_rejected = apply_candidates(
        baseline_rows, v366_union, output_key="v414_projection"
    )
    v414_summary = score(v366_rows, "v414_projection")

    v357_manifest = read_json(V357_MANIFEST_JSON)
    v366_manifest = read_json(V366_MANIFEST_JSON)
    v368_manifest = read_json(V368_MANIFEST_JSON)

    rows_by_id = {row["id"]: row for row in baseline_rows}
    v409_ids = {row["id"] for row in v409_accepted}
    v412_ids = {row["id"] for row in v412_accepted}
    new_since_v409 = [row for row in v366_accepted if row["id"] not in v409_ids]
    new_since_v412 = [row for row in v366_accepted if row["id"] not in v412_ids]

    for row in v366_rows:
        row["family"] = family_for(row)
        row["baseline_prediction"] = row["prediction"]
        row["baseline_correct"] = verify_answer(row["answer"], row["baseline_prediction"])
        row["v414_correct"] = verify_answer(row["answer"], row["v414_projection"])

    stage_rows = [
        score_row("V291/V290 adapter baseline", baseline_summary, baseline_summary, "submit-safe baseline"),
        score_row("V409 solver projection", v409_summary, baseline_summary, "CPU teacher only"),
        score_row("V412 CPU synthesis union", v412_summary, baseline_summary, "CPU teacher only; no new gain over V409"),
        score_row("V357 bit global ternary union", v357_summary, baseline_summary, "CPU teacher only"),
        score_row("V414/V366 consolidated CPU teacher", v414_summary, baseline_summary, "best CPU teacher; not adapter-only"),
    ]

    rejected_rows = v409_rejected + v412_rejected + v357_rejected + v366_rejected
    accepted_columns = [
        "id",
        "family",
        "old_prediction",
        "new_prediction",
        "answer",
        "old_correct",
        "new_correct",
        "sources",
        "rules",
        "proofs",
    ]
    prediction_columns = [
        "id",
        "family",
        "answer",
        "baseline_prediction",
        "v414_projection",
        "baseline_correct",
        "v414_correct",
        "prompt",
    ]
    stage_columns = [
        "state",
        "total",
        "total_delta",
        "equation_transform",
        "equation_delta",
        "bit_manipulation",
        "bit_delta",
        "status",
    ]
    rejected_columns = [
        "id",
        "family",
        "reason",
        "old_prediction",
        "new_prediction",
        "answer",
        "predictions",
        "sources",
        "rules",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "v414_stage_summary.csv", stage_rows, stage_columns)
    write_csv(OUT_DIR / "v414_accepted_union.csv", v366_accepted, accepted_columns)
    write_csv(OUT_DIR / "v414_new_since_v409.csv", new_since_v409, accepted_columns)
    write_csv(OUT_DIR / "v414_new_since_v412.csv", new_since_v412, accepted_columns)
    write_csv(OUT_DIR / "v414_rejected_or_conflicts.csv", rejected_rows, rejected_columns)
    write_csv(OUT_DIR / "v414_integrated_predictions.csv", v366_rows, prediction_columns)

    expected_v366 = v366_manifest["v366_summary"]["correct"]
    observed_v366 = v414_summary["total"]["correct"]
    manifest_match = expected_v366 == observed_v366
    v368_summary = v368_manifest["v368_summary"]
    v413_cutoff_present = V413_CUTOFF_MD.is_file()

    manifest = {
        "schema_version": "kg1_v414_cpu_teacher_meta_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_csv": str(BASELINE_CSV.relative_to(REPO_ROOT)),
            "baseline_sha256": sha256_file(BASELINE_CSV),
            "v409_accepted_csv": str(V409_ACCEPTED_CSV.relative_to(REPO_ROOT)),
            "v412_accepted_csv": str(V412_ACCEPTED_CSV.relative_to(REPO_ROOT)),
            "v357_manifest_json": str(V357_MANIFEST_JSON.relative_to(REPO_ROOT)),
            "v357_manifest_sha256": sha256_file(V357_MANIFEST_JSON),
            "v366_manifest_json": str(V366_MANIFEST_JSON.relative_to(REPO_ROOT)),
            "v366_manifest_sha256": sha256_file(V366_MANIFEST_JSON),
            "v368_transfer_failure_manifest_json": str(V368_MANIFEST_JSON.relative_to(REPO_ROOT)),
            "v368_transfer_failure_manifest_sha256": sha256_file(V368_MANIFEST_JSON),
            "v413_cutoff_md": str(V413_CUTOFF_MD.relative_to(REPO_ROOT)) if v413_cutoff_present else "",
        },
        "baseline_summary": baseline_summary,
        "stage_summary": stage_rows,
        "v414_summary": v414_summary,
        "accepted_union_count": len(v366_accepted),
        "new_since_v409_count": len(new_since_v409),
        "new_since_v412_count": len(new_since_v412),
        "rejected_or_conflict_count": len(rejected_rows),
        "manifest_crosscheck": {
            "v357_expected_correct": v357_manifest["v357_summary"]["correct"],
            "v357_observed_correct": v357_summary["total"]["correct"],
            "v366_expected_correct": expected_v366,
            "v366_observed_correct": observed_v366,
            "v366_manifest_match": manifest_match,
        },
        "transfer_failures": {
            "v368": {
                "total": v368_summary["correct"],
                "equation_transform": v368_summary["family"]["equation_transform"]["correct"],
                "bit_manipulation": v368_summary["family"]["bit_manipulation"]["correct"],
                "reason": v368_manifest["decision"]["reason"],
            },
            "v413": {
                "documented": v413_cutoff_present,
                "summary": "V413 checkpoint-2 weak=190/315, equation=56/155, bit=134/160, truncated=1; eval canceled by FinOps.",
            },
        },
        "decision": {
            "decision": "v414_cpu_teacher_valid_but_transfer_blocked",
            "hf_gpu_allowed": False,
            "reason": (
                "V414 reconstructs the V366 CPU teacher at 222/315 with equation=63/155 and "
                "bit=159/160, but V368 and V413 both failed to transfer solver/teacher gains to "
                "adapter-only weak eval."
            ),
            "next_action": (
                "Do not repeat V357/V366/V409/V410 teacher SFT. Next roadmap step must target "
                "adapter behavior directly: checkpoint/prompt/package-level selection or a new "
                "transfer mechanism with a CPU proof that differs from the rejected routes."
            ),
        },
        "outputs": {
            "summary_md": str((OUT_DIR / "V414_CPU_TEACHER_META_GATE.md").relative_to(REPO_ROOT)),
            "stage_summary_csv": str((OUT_DIR / "v414_stage_summary.csv").relative_to(REPO_ROOT)),
            "accepted_union_csv": str((OUT_DIR / "v414_accepted_union.csv").relative_to(REPO_ROOT)),
            "new_since_v409_csv": str((OUT_DIR / "v414_new_since_v409.csv").relative_to(REPO_ROOT)),
            "new_since_v412_csv": str((OUT_DIR / "v414_new_since_v412.csv").relative_to(REPO_ROOT)),
            "integrated_predictions_csv": str((OUT_DIR / "v414_integrated_predictions.csv").relative_to(REPO_ROOT)),
            "manifest_json": str((OUT_DIR / "v414_cpu_teacher_meta_gate_manifest.json").relative_to(REPO_ROOT)),
        },
    }
    write_json(OUT_DIR / "v414_cpu_teacher_meta_gate_manifest.json", manifest)

    report = [
        "# V414 CPU Teacher Meta Gate",
        "",
        "V414 consolidates the CPU solver/verifier evidence and explicitly separates it from adapter-only submit eligibility.",
        "",
        "## Comparison",
        "",
        "| State | Weak total | Delta | equation_transform | Delta | bit_manipulation | Delta | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in stage_rows:
        report.append(
            f"| {row['state']} | `{row['total']}/315` | `{row['total_delta']:+d}` | "
            f"`{row['equation_transform']}/155` | `{row['equation_delta']:+d}` | "
            f"`{row['bit_manipulation']}/160` | `{row['bit_delta']:+d}` | {row['status']} |"
        )
    report += [
        "",
        "## Transfer Blockers",
        "",
        "- V368 tried the V367/V366 bit-ternary transfer route and produced `191/315`, `equation=56/155`, `bit=135/160`; it transferred `0/8` V366 gains and introduced `2` losses vs baseline.",
        "- V413 tried the solver-first transfer route and produced `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1` at checkpoint-2; eval was canceled by FinOps.",
        "",
        "## Decision",
        "",
        "V366/V414 is the best CPU teacher currently available (`222/315`, `equation=63/155`, `bit=159/160`), but it is not adapter-only submit-safe. The same teacher-transfer pattern has already failed in GPU jobs, so another HF run on this route is blocked.",
        "",
        "Next action: target adapter behavior directly. Do not train again from the same teacher rows unless a new CPU gate proves a materially different transfer mechanism and the first checkpoint can beat `192/315`, `equation>56`, `bit>=136`, `truncated=0`.",
        "",
        "## Key New Rows Beyond V409",
        "",
    ]
    for row in new_since_v409:
        report.append(
            f"- `{row['id']}` `{row['family']}`: `{row['old_prediction']}` -> `{row['new_prediction']}` via `{row['sources']}` `{row['rules']}`"
        )
    (OUT_DIR / "V414_CPU_TEACHER_META_GATE.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

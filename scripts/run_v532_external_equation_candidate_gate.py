#!/usr/bin/env python3
"""CPU-only audit for public external equation candidate datasets.

This gate downloads small Kaggle datasets that expose equation candidate pools
and tests whether deployable, label-free selector signals can improve the
current weak equation plateau.  It never trains, packages, submits, or uses
`answer`/`expected_answer`/`competition_match` to choose a candidate.  Labels are
used only after selection to audit gains/losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_WEAK_CSV = (
    REPO_ROOT
    / "artifacts"
    / "v290_rank19_micro_patch_reference"
    / "runtime_artifacts"
    / "v245_weak_eval_bridge"
    / "v245-weak-bridge-hfonly-20260510T1950Z"
    / "v221_weak_315.csv"
)
DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "artifacts"
    / "v342_acc_first_diagnostic"
    / "v290_checkpoint6_baseline_predictions.csv"
)

DATASETS = {
    "critic_v2": "itskshivam/nemotron-equation-candidate-critic-v2",
    "router_v1": "itskshivam/nemotron-equation-candidate-critique-router-v1",
    "selection_v2": "sohamp13/nemotron-equation-candidate-selection-v2",
    "solver_swap_v1": "furkankesen/equation-solver-swap-v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        result = float(str(value).strip())
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def parse_rank(value: object, default: float = 9999.0) -> float:
    parsed = parse_float(value, default)
    return parsed


def no_failure(row: dict[str, str]) -> bool:
    failure = str(row.get("failure_reason", "")).strip().lower()
    status = str(row.get("execution_status", "")).strip().lower()
    if failure and failure not in {"none", "nan", "null"}:
        return False
    if status and any(token in status for token in ["fail", "error", "exception"]):
        return False
    return True


def canonical_ok(row: dict[str, str]) -> bool:
    status = str(row.get("canonicalization_status", "")).strip().lower()
    if not status:
        return True
    if any(token in status for token in ["fail", "error", "invalid"]):
        return False
    return True


def candidate_rank(row: dict[str, str], source: str) -> tuple[Any, ...]:
    """Rank without using label-only fields.

    Explicitly excluded: expected_answer, answer, competition_match, is_target,
    gold_label, winner_index, winner_position.
    """

    verifier_valid = truthy(row.get("verifier_valid"))
    verifier_score = parse_float(row.get("verifier_score"))
    sympy_ok = truthy(row.get("sympy_parse_success"))
    max_demo = parse_float(row.get("max_exact_demo_matches"))
    votes = parse_float(row.get("vote_count"))
    symbolic_bonus = parse_float(row.get("symbolic_context_bonus"))
    same_support = parse_float(row.get("same_canonical_answer_support"))
    policy_rank = parse_rank(row.get("base_policy_rank"))
    preference = -parse_rank(row.get("best_program_preference"), 9999.0)
    family_known = 1 if str(row.get("best_program_family", "")).strip() else 0
    source_bonus = {"router_v1": 2, "critic_v2": 1}.get(source, 0)
    return (
        1 if no_failure(row) else 0,
        1 if verifier_valid else 0,
        verifier_score,
        1 if canonical_ok(row) else 0,
        1 if sympy_ok else 0,
        max_demo,
        votes,
        same_support,
        symbolic_bonus,
        family_known,
        preference,
        -policy_rank,
        source_bonus,
    )


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: int = 900) -> tuple[int, str]:
    print("+ " + " ".join(cmd), flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.time() - started
    print(f"returncode = {proc.returncode} elapsed_s = {elapsed:.2f}", flush=True)
    tail = "\n".join(proc.stdout.splitlines()[-12:])
    if tail:
        print("tail =\n" + tail, flush=True)
    return proc.returncode, proc.stdout


def download_dataset(ref: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    rc, out = run_cmd(["kaggle", "datasets", "download", ref, "-p", str(target_dir), "-o", "-q"])
    if rc:
        raise RuntimeError(f"kaggle download failed for {ref}: {out[-1000:]}")
    zips = list(target_dir.glob("*.zip"))
    if len(zips) != 1:
        raise RuntimeError(f"expected one zip for {ref}, got {[p.name for p in zips]}")
    return zips[0]


def iter_zip_csv(zip_path: Path, member_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        target = None
        for name in names:
            if name.endswith(member_name):
                target = name
                break
        if target is None:
            return []
        with zf.open(target) as handle:
            text = handle.read().decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(text.splitlines()))


def baseline_label_free_prediction(row: dict[str, str]) -> str:
    raw = row.get("raw_output")
    if raw is not None and str(raw).strip():
        return extract_final_answer(raw)
    return str(row.get("prediction", "")).strip()


def score_selector(
    weak_rows: list[dict[str, str]],
    baseline_rows: dict[str, dict[str, str]],
    candidate_rows: list[dict[str, str]],
    selector_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    equation_rows = [row for row in weak_rows if row.get("type") == "equation_transform"]
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        rid = str(row.get("id", "")).strip()
        cand = str(row.get("candidate_answer", "")).strip()
        if not rid or not cand:
            continue
        by_id[rid].append(row)

    decisions: list[dict[str, Any]] = []
    counts = Counter()
    for weak in equation_rows:
        rid = weak["id"]
        answer = weak["answer"]
        base_row = baseline_rows.get(rid, {})
        baseline_pred = baseline_label_free_prediction(base_row)
        baseline_correct = verify_answer(answer, baseline_pred)
        rows = by_id.get(rid, [])
        chosen = None
        chosen_pred = baseline_pred
        if rows:
            chosen = max(rows, key=lambda row: candidate_rank(row, str(row.get("_source", ""))))
            chosen_pred = str(chosen.get("candidate_answer", "")).strip() or baseline_pred
        chosen_correct = verify_answer(answer, chosen_pred)
        if not baseline_correct and chosen_correct:
            outcome = "gain"
        elif baseline_correct and not chosen_correct:
            outcome = "loss"
        elif baseline_correct and chosen_correct:
            outcome = "preserve_correct"
        else:
            outcome = "still_wrong"
        counts[outcome] += 1
        counts["rows"] += 1
        if rows:
            counts["rows_with_candidates"] += 1
        if chosen_correct:
            counts["selected_correct"] += 1
        if baseline_correct:
            counts["baseline_correct"] += 1
        decisions.append(
            {
                "selector": selector_name,
                "id": rid,
                "answer": answer,
                "baseline_prediction": baseline_pred,
                "baseline_correct": baseline_correct,
                "selected_prediction": chosen_pred,
                "selected_correct": chosen_correct,
                "outcome": outcome,
                "candidate_count": len(rows),
                "source": chosen.get("_source", "") if chosen else "",
                "best_program_family": chosen.get("best_program_family", "") if chosen else "",
                "verifier_valid": chosen.get("verifier_valid", "") if chosen else "",
                "verifier_score": chosen.get("verifier_score", "") if chosen else "",
                "canonicalization_status": chosen.get("canonicalization_status", "") if chosen else "",
                "sympy_parse_success": chosen.get("sympy_parse_success", "") if chosen else "",
                "failure_reason": chosen.get("failure_reason", "") if chosen else "",
                "competition_match_audit_only": chosen.get("competition_match", "") if chosen else "",
            }
        )

    summary = {
        "selector": selector_name,
        "rows": counts["rows"],
        "rows_with_candidates": counts["rows_with_candidates"],
        "baseline_correct": counts["baseline_correct"],
        "selected_correct": counts["selected_correct"],
        "gains": counts["gain"],
        "losses": counts["loss"],
        "preserve_correct": counts["preserve_correct"],
        "still_wrong": counts["still_wrong"],
        "net": counts["gain"] - counts["loss"],
        "promotable": counts["gain"] > 0 and counts["loss"] == 0,
    }
    return summary, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "v532_external_equation_candidate_gate")
    parser.add_argument("--weak-csv", type=Path, default=DEFAULT_WEAK_CSV)
    parser.add_argument("--baseline-predictions-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    args = parser.parse_args()

    print("=== V532 EXTERNAL EQUATION CANDIDATE GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("weak_csv =", args.weak_csv, flush=True)
    print("baseline_predictions_csv =", args.baseline_predictions_csv, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weak_rows = read_csv(args.weak_csv)
    baseline_rows_list = read_csv(args.baseline_predictions_csv)
    baseline_by_id = {row["id"]: row for row in baseline_rows_list}
    print("weak_rows =", len(weak_rows), "baseline_rows =", len(baseline_rows_list), flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="kg1_v532_external_eq_"))
    print("tmp =", tmp, flush=True)
    all_candidates: list[dict[str, str]] = []
    dataset_summaries: list[dict[str, Any]] = []
    try:
        for source, ref in DATASETS.items():
            ds_dir = tmp / source
            zip_path = download_dataset(ref, ds_dir)
            rows: list[dict[str, str]] = []
            if source in {"critic_v2", "router_v1"}:
                rows = iter_zip_csv(zip_path, "candidate_pool.csv")
            elif source == "selection_v2":
                # This is not direct answer generation; keep as evidence only.
                rows = iter_zip_csv(zip_path, "train.csv") + iter_zip_csv(zip_path, "eval.csv")
            elif source == "solver_swap_v1":
                rows = iter_zip_csv(zip_path, "train.csv")

            weak_ids = {row["id"] for row in weak_rows}
            overlap = [row for row in rows if str(row.get("id", "")).strip() in weak_ids]
            for row in overlap:
                row["_source"] = source
            if source in {"critic_v2", "router_v1"}:
                all_candidates.extend(overlap)
            dataset_summaries.append(
                {
                    "source": source,
                    "ref": ref,
                    "zip_size": zip_path.stat().st_size,
                    "rows": len(rows),
                    "weak_overlap_rows": len(overlap),
                    "unique_weak_ids": len({str(row.get("id", "")).strip() for row in overlap}),
                    "direct_candidate_pool": source in {"critic_v2", "router_v1"},
                }
            )
            print("dataset_summary =", json.dumps(dataset_summaries[-1], sort_keys=True), flush=True)
    finally:
        resolved = tmp.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if str(resolved).lower().startswith(str(temp_root).lower()):
            shutil.rmtree(resolved, ignore_errors=True)
            print("tmp_deleted =", resolved, flush=True)

    selector_inputs = {
        "critic_v2_only": [row for row in all_candidates if row.get("_source") == "critic_v2"],
        "router_v1_only": [row for row in all_candidates if row.get("_source") == "router_v1"],
        "critic_router_union": all_candidates,
    }

    summaries: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    for name, rows in selector_inputs.items():
        summary, decisions = score_selector(weak_rows, baseline_by_id, rows, name)
        summaries.append(summary)
        all_decisions.extend(decisions)
        print("selector_summary =", json.dumps(summary, sort_keys=True), flush=True)

    decision_fields = [
        "selector",
        "id",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "selected_prediction",
        "selected_correct",
        "outcome",
        "candidate_count",
        "source",
        "best_program_family",
        "verifier_valid",
        "verifier_score",
        "canonicalization_status",
        "sympy_parse_success",
        "failure_reason",
        "competition_match_audit_only",
    ]
    write_csv(args.output_dir / "v532_external_equation_candidate_decisions.csv", all_decisions, decision_fields)
    write_csv(
        args.output_dir / "v532_external_equation_candidate_summary.csv",
        summaries,
        [
            "selector",
            "rows",
            "rows_with_candidates",
            "baseline_correct",
            "selected_correct",
            "gains",
            "losses",
            "preserve_correct",
            "still_wrong",
            "net",
            "promotable",
        ],
    )
    (args.output_dir / "v532_external_equation_dataset_summaries.json").write_text(
        json.dumps(dataset_summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "kg1_v532_external_equation_candidate_gate_v1",
        "generated_at_utc": utc_now(),
        "weak_csv": str(args.weak_csv),
        "baseline_predictions_csv": str(args.baseline_predictions_csv),
        "dataset_summaries": dataset_summaries,
        "selector_summaries": summaries,
        "decision": {
            "promotable_selectors": [row["selector"] for row in summaries if row["promotable"]],
            "next_action": "Only promote to training data if a selector has gains>0 and losses=0. Otherwise use as diagnosis/verifier feature source.",
        },
    }
    (args.output_dir / "v532_external_equation_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# V532 External Equation Candidate Gate",
        "",
        "CPU-only diagnostic. Candidate selection did not use labels, expected answers, or `competition_match`; those fields are audit-only.",
        "",
        "## Selector Summary",
        "",
        "| selector | selected_correct | baseline_correct | gains | losses | net | promotable |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['selector']}` | {row['selected_correct']}/{row['rows']} | "
            f"{row['baseline_correct']}/{row['rows']} | {row['gains']} | {row['losses']} | "
            f"{row['net']} | `{row['promotable']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- If no selector is promotable, these datasets are useful as verifier/canonicalization feature references, not direct submit-safe gains.",
            "- If a selector is promotable, convert only its gain rows into a guarded CPU rule or short hard-negative training pack and rerun weak gates.",
        ]
    )
    (args.output_dir / "KG1_V532_EXTERNAL_EQUATION_CANDIDATE_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("manifest_path =", args.output_dir / "v532_external_equation_candidate_manifest.json", flush=True)
    print("summary_path =", args.output_dir / "KG1_V532_EXTERNAL_EQUATION_CANDIDATE_GATE.md", flush=True)
    print("=== V532 EXTERNAL EQUATION CANDIDATE GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

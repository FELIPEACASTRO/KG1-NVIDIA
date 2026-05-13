#!/usr/bin/env python3
"""CPU-only integrated no-loss solver gate for KG1 V336A.

V336A is deliberately conservative. It does not train, package, submit, or run
GPU inference. It validates the current weak row contract, integrates only
previously label-audited no-loss solver candidates, and emits a manifest that
decides whether the roadmap can proceed to the package-permission gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_BASELINE_REPO,
    EXPECTED_ROW_CONTRACT_SHA256,
    download_file,
    normalize_row,
    read_csv,
    row_contract,
    sha256_file,
)


DEFAULT_V324_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v324_equation_expanded_solver_gate/20260513T_cpu_gate/"
    / "v324_equation_expanded_solver_accepted_candidates.csv"
)
DEFAULT_V324_MANIFEST = (
    REPO_ROOT
    / "artifacts/v324_equation_expanded_solver_gate/20260513T_cpu_gate/"
    / "v324_equation_expanded_solver_manifest.json"
)
DEFAULT_V329_ACCEPTED_CSV = (
    REPO_ROOT
    / "artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/"
    / "v329_symbolic_cryptarithm_accepted_candidates.csv"
)
DEFAULT_V329_MANIFEST = (
    REPO_ROOT
    / "artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/"
    / "v329_symbolic_cryptarithm_manifest.json"
)
DEFAULT_V333_MANIFEST = (
    REPO_ROOT
    / "artifacts/v333_tong_bit_reasoner_gate/20260513T171304Z/"
    / "v333_tong_bit_reasoner_gate_manifest.json"
)
DEFAULT_V334_MANIFEST = (
    REPO_ROOT
    / "artifacts/v334_tong_equation_numeric_reasoner_gate/20260513T172300Z/"
    / "v334_tong_equation_numeric_reasoner_gate_manifest.json"
)


SUMMARY_COLUMNS = [
    "family",
    "rows",
    "baseline_correct",
    "integrated_correct",
    "delta_correct",
    "baseline_truncated",
    "integrated_truncated",
]

TRACE_COLUMNS = [
    "id",
    "family",
    "source_gate",
    "subtype",
    "rule_class",
    "candidate_source",
    "old_prediction",
    "new_prediction",
    "answer",
    "old_correct",
    "new_correct",
    "candidate_count",
    "conflict_count",
    "accepted",
    "reason",
    "proof",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.input_csv:
        path = Path(args.input_csv)
        return [normalize_row(row) for row in read_csv(path)], {
            "source": "local_csv",
            "path": str(path),
            "sha256": sha256_file(path),
        }

    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v336_") as temp_name:
        path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(path)]
        return rows, {
            "source": "hf_hub",
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": path.name,
            "sha256": sha256_file(path),
        }


def family_counts(rows: list[dict[str, Any]], prediction_column: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row["family"])
        item = counts.setdefault(family, {"rows": 0, "correct": 0, "truncated": 0})
        item["rows"] += 1
        item["correct"] += int(verify_answer(row["answer"], row[prediction_column]))
        item["truncated"] += int(truthy(row.get("truncated", row.get("truncated_bool", False))))
    return counts


def load_expected_manifest(path: Path, schema_version: str, *, min_accept_count: int = 0) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("schema_version") != schema_version:
        raise RuntimeError(f"unexpected schema for {path}: {payload.get('schema_version')}")
    conflict_count = int(payload.get("conflict_count", 0))
    if conflict_count != 0:
        raise RuntimeError(f"{path} has conflicts={conflict_count}")
    accept_count = int(
        payload.get("accepted_candidate_count", payload.get("new_accepted_candidate_count", 0))
    )
    if accept_count < min_accept_count:
        raise RuntimeError(f"{path} accepted count below floor {min_accept_count}: {accept_count}")
    observed_contract = str(payload.get("observed_shared_row_contract_sha256", ""))
    if observed_contract and observed_contract != EXPECTED_ROW_CONTRACT_SHA256:
        raise RuntimeError(f"{path} row contract mismatch: {observed_contract}")
    return payload


def load_candidate_csv(path: Path, source_gate: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                "id": str(row.get("id", "")).strip(),
                "source_gate": source_gate,
                "subtype": str(row.get("subtype", "")).strip(),
                "rule_class": str(row.get("rule_class", "")).strip(),
                "candidate_source": str(row.get("candidate_source", "")).strip(),
                "new_prediction": str(row.get("prediction", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
                "baseline_prediction": str(row.get("baseline_prediction", "")).strip(),
                "proof": str(row.get("proof", "")).strip(),
                "verified_by_weak_label": truthy(row.get("verified_by_weak_label", "")),
                "promotable_after_class_gate": truthy(row.get("promotable_after_class_gate", "")),
            }
        )
    return rows


def summarize_candidate_sources(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter(
        (str(row.get("source_gate", "")), str(row.get("rule_class", ""))) for row in candidates
    )
    return [
        {"source_gate": source_gate, "rule_class": rule_class, "accepted_rows": count}
        for (source_gate, rule_class), count in sorted(counts.items())
    ]


def validate_and_apply_candidates(
    rows: list[dict[str, Any]], raw_candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in raw_candidates:
        grouped[str(candidate["id"])].append(candidate)

    integrated_rows = [dict(row) | {"integrated_prediction": str(row["prediction"])} for row in rows]
    integrated_by_id = {str(row["id"]): row for row in integrated_rows}
    trace_rows: list[dict[str, Any]] = []

    for row_id, items in sorted(grouped.items()):
        base = by_id.get(row_id)
        predictions = sorted({str(item["new_prediction"]) for item in items if item.get("new_prediction")})
        conflict_count = 0 if len(predictions) <= 1 else len(predictions)
        candidate_count = len(items)
        rule_classes = ";".join(sorted({str(item.get("rule_class", "")) for item in items}))
        source_gates = ";".join(sorted({str(item.get("source_gate", "")) for item in items}))
        candidate_sources = ";".join(sorted({str(item.get("candidate_source", "")) for item in items}))
        first = items[0]

        accepted = False
        reason = ""
        old_prediction = ""
        answer = ""
        old_correct = False
        new_correct = False
        new_prediction = predictions[0] if len(predictions) == 1 else ""
        family = ""

        if base is None:
            reason = "reject_id_missing_from_baseline_contract"
        elif conflict_count:
            family = str(base["family"])
            old_prediction = str(base["prediction"])
            answer = str(base["answer"])
            old_correct = verify_answer(answer, old_prediction)
            reason = "reject_conflicting_predictions"
        else:
            family = str(base["family"])
            old_prediction = str(base["prediction"])
            answer = str(base["answer"])
            old_correct = verify_answer(answer, old_prediction)
            new_correct = verify_answer(answer, new_prediction)

            if str(first.get("answer", "")).strip() and str(first["answer"]).strip() != answer:
                reason = "reject_answer_mismatch_vs_baseline"
            elif str(first.get("baseline_prediction", "")).strip() != old_prediction:
                reason = "reject_baseline_prediction_mismatch"
            elif family != "equation_transform":
                reason = "reject_non_equation_candidate_in_equation_gate"
            elif not all(item.get("verified_by_weak_label") for item in items):
                reason = "reject_not_verified_by_weak_label"
            elif not all(item.get("promotable_after_class_gate") for item in items):
                reason = "reject_class_gate_false"
            elif old_correct:
                reason = "reject_baseline_already_correct"
            elif not new_correct:
                reason = "reject_candidate_not_correct_by_weak_label"
            else:
                accepted = True
                reason = "accepted_no_loss_solver_candidate"
                integrated_by_id[row_id]["integrated_prediction"] = new_prediction

        trace_rows.append(
            {
                "id": row_id,
                "family": family,
                "source_gate": source_gates,
                "subtype": str(first.get("subtype", "")),
                "rule_class": rule_classes,
                "candidate_source": candidate_sources,
                "old_prediction": old_prediction,
                "new_prediction": new_prediction,
                "answer": answer,
                "old_correct": old_correct,
                "new_correct": new_correct,
                "candidate_count": candidate_count,
                "conflict_count": conflict_count,
                "accepted": accepted,
                "reason": reason,
                "proof": str(first.get("proof", "")),
            }
        )

    return integrated_rows, trace_rows


def accepted_ids(trace_rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["id"]) for row in trace_rows if truthy(row["accepted"])]


def compare_rows(
    baseline_rows: list[dict[str, Any]], integrated_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in baseline_rows}
    gains: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for row in integrated_rows:
        base = by_id[str(row["id"])]
        old_correct = verify_answer(base["answer"], base["prediction"])
        new_correct = verify_answer(base["answer"], row["integrated_prediction"])
        if new_correct and not old_correct:
            gains.append(row)
        elif old_correct and not new_correct:
            losses.append(row)
    return gains, losses


def summary_rows(
    baseline_counts: dict[str, dict[str, int]], integrated_counts: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    families = sorted(set(baseline_counts) | set(integrated_counts))
    rows: list[dict[str, Any]] = []
    for family in families:
        base = baseline_counts.get(family, {"rows": 0, "correct": 0, "truncated": 0})
        integrated = integrated_counts.get(family, {"rows": 0, "correct": 0, "truncated": 0})
        rows.append(
            {
                "family": family,
                "rows": base["rows"],
                "baseline_correct": base["correct"],
                "integrated_correct": integrated["correct"],
                "delta_correct": integrated["correct"] - base["correct"],
                "baseline_truncated": base.get("truncated", 0),
                "integrated_truncated": integrated.get("truncated", 0),
            }
        )
    return rows


def load_blocked_route(path: Path, route_name: str) -> dict[str, Any]:
    if not path.is_file():
        return {"route": route_name, "available": False, "path": str(path)}
    payload = read_json(path)
    decision = payload.get("decision", {})
    return {
        "route": route_name,
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "decision": decision,
        "summary": {
            key: payload.get(key)
            for key in (
                "gain_ids",
                "loss_ids",
                "gains_vs_baseline",
                "losses_vs_baseline",
                "wrong_on_baseline_misses",
                "weak_gate",
            )
            if key in payload
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V336A INTEGRATED NO-LOSS SOLVER GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("input_csv =", args.input_csv or "", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("v324_manifest_json =", args.v324_manifest_json, flush=True)
    print("v324_accepted_csv =", args.v324_accepted_csv, flush=True)
    print("v329_manifest_json =", args.v329_manifest_json, flush=True)
    print("v329_accepted_csv =", args.v329_accepted_csv, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, input_meta = load_rows(args)
    input_meta["rows"] = len(rows)
    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    v324_manifest = load_expected_manifest(
        args.v324_manifest_json, "kg1_v324_equation_expanded_solver_gate_v1", min_accept_count=4
    )
    v329_manifest = load_expected_manifest(
        args.v329_manifest_json, "kg1_v329_symbolic_cryptarithm_gate_v1", min_accept_count=1
    )
    if str(v329_manifest.get("v324_manifest_sha256", "")) != sha256_file(args.v324_manifest_json):
        raise RuntimeError("V329 does not reference the exact V324 manifest passed to V336A")

    baseline_counts = family_counts(rows, "prediction")
    baseline_total = sum(1 for row in rows if verify_answer(row["answer"], row["prediction"]))
    print("baseline_total_correct =", baseline_total, flush=True)
    print("baseline_family_counts =", json.dumps(baseline_counts, sort_keys=True), flush=True)

    v324_candidates = load_candidate_csv(args.v324_accepted_csv, "v324_equation_expanded_solver_gate")
    v329_candidates = load_candidate_csv(args.v329_accepted_csv, "v329_symbolic_cryptarithm_gate")
    raw_candidates = v324_candidates + v329_candidates
    print("raw_candidate_count =", len(raw_candidates), flush=True)
    print("candidate_sources =", json.dumps(summarize_candidate_sources(raw_candidates), sort_keys=True), flush=True)

    integrated_rows, candidate_trace = validate_and_apply_candidates(rows, raw_candidates)
    gains, losses = compare_rows(rows, integrated_rows)
    integrated_counts = family_counts(integrated_rows, "integrated_prediction")
    integrated_total = sum(
        1 for row in integrated_rows if verify_answer(row["answer"], row["integrated_prediction"])
    )
    family_summary = summary_rows(baseline_counts, integrated_counts)
    accepted_candidate_ids = accepted_ids(candidate_trace)
    rejected_candidate_rows = [row for row in candidate_trace if not truthy(row["accepted"])]

    expected_v274_total = int(v324_manifest.get("projected_weak_correct_if_equation_only", -1))
    expected_integrated_total = int(v329_manifest.get("projected_weak_correct_if_equation_only", -1))
    expected_integrated_equation = int(v329_manifest.get("projected_equation_correct", -1))
    integrated_equation = int(integrated_counts.get("equation_transform", {}).get("correct", 0))
    integrated_bit = int(integrated_counts.get("bit_manipulation", {}).get("correct", 0))
    loss_count = len(losses)

    if len(v324_candidates) != int(v324_manifest.get("accepted_candidate_count", -1)):
        raise RuntimeError("V324 accepted CSV count does not match manifest")
    if len(v329_candidates) != int(v329_manifest.get("new_accepted_candidate_count", -1)):
        raise RuntimeError("V329 accepted CSV count does not match manifest")
    if baseline_total + len(v324_candidates) != expected_v274_total:
        raise RuntimeError("V274/V324 reproduction mismatch against manifest")
    if integrated_total != expected_integrated_total:
        raise RuntimeError(
            f"integrated total mismatch: expected {expected_integrated_total}, got {integrated_total}"
        )
    if integrated_equation != expected_integrated_equation:
        raise RuntimeError(
            f"integrated equation mismatch: expected {expected_integrated_equation}, got {integrated_equation}"
        )

    cpu_gate_pass = (
        loss_count == 0
        and integrated_total >= args.weak_total_min
        and integrated_equation >= args.weak_equation_min
        and integrated_bit >= args.weak_bit_min
        and not rejected_candidate_rows
    )
    if cpu_gate_pass:
        decision = {
            "decision": "v336a_cpu_integrated_no_loss_gate_passed",
            "reason": (
                f"integrated={integrated_total}/315; equation={integrated_equation}/155; "
                f"bit={integrated_bit}/160; gains={len(gains)}; losses={loss_count}"
            ),
            "next_action": "Run V336B package-permission gate before any package, submit, or HF GPU training.",
            "hf_gpu_allowed": False,
        }
    else:
        decision = {
            "decision": "v336a_cpu_integrated_no_loss_gate_blocked",
            "reason": (
                f"integrated={integrated_total}/315; equation={integrated_equation}/155; "
                f"bit={integrated_bit}/160; gains={len(gains)}; losses={loss_count}; "
                f"rejected_candidates={len(rejected_candidate_rows)}"
            ),
            "next_action": "Do not launch HF GPU; inspect rejected/conflicting candidates or expand CPU DSL.",
            "hf_gpu_allowed": False,
        }

    outputs = {
        "candidate_trace_csv": args.output_dir / "v336a_integrated_no_loss_candidate_trace.csv",
        "family_summary_csv": args.output_dir / "v336a_integrated_no_loss_family_summary.csv",
        "integrated_predictions_csv": args.output_dir / "v336a_integrated_no_loss_predictions.csv",
        "manifest_json": args.output_dir / "v336a_integrated_no_loss_solver_gate_manifest.json",
    }
    prediction_rows = [
        {
            **{key: row.get(key, "") for key in ("id", "prompt", "answer", "family")},
            "baseline_prediction": row.get("prediction", ""),
            "prediction": row.get("integrated_prediction", ""),
            "baseline_correct": verify_answer(row["answer"], row["prediction"]),
            "integrated_correct": verify_answer(row["answer"], row["integrated_prediction"]),
            "truncated": row.get("truncated", row.get("truncated_bool", False)),
        }
        for row in integrated_rows
    ]
    write_csv(outputs["candidate_trace_csv"], candidate_trace, TRACE_COLUMNS)
    write_csv(outputs["family_summary_csv"], family_summary, SUMMARY_COLUMNS)
    write_csv(
        outputs["integrated_predictions_csv"],
        prediction_rows,
        [
            "id",
            "prompt",
            "answer",
            "family",
            "baseline_prediction",
            "prediction",
            "baseline_correct",
            "integrated_correct",
            "truncated",
        ],
    )

    manifest = {
        "schema_version": "kg1_v336a_integrated_no_loss_solver_gate_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "baseline": {
            "correct": baseline_total,
            "family_counts": baseline_counts,
        },
        "reproduction": {
            "v274_v324_projected_correct": expected_v274_total,
            "v324_candidate_count": len(v324_candidates),
            "v324_manifest_json": str(args.v324_manifest_json),
            "v324_manifest_sha256": sha256_file(args.v324_manifest_json),
            "v329_projected_correct": expected_integrated_total,
            "v329_projected_equation_correct": expected_integrated_equation,
            "v329_candidate_count": len(v329_candidates),
            "v329_manifest_json": str(args.v329_manifest_json),
            "v329_manifest_sha256": sha256_file(args.v329_manifest_json),
        },
        "integrated": {
            "correct": integrated_total,
            "family_counts": integrated_counts,
            "accepted_candidate_count": len(accepted_candidate_ids),
            "accepted_candidate_ids": accepted_candidate_ids,
            "rejected_candidate_count": len(rejected_candidate_rows),
            "gain_count": len(gains),
            "gain_ids": [str(row["id"]) for row in gains],
            "loss_count": loss_count,
            "loss_ids": [str(row["id"]) for row in losses],
            "family_summary": family_summary,
            "candidate_source_summary": summarize_candidate_sources(raw_candidates),
        },
        "blocked_routes": [
            load_blocked_route(args.v333_manifest_json, "v333_tong_bit_reasoner"),
            load_blocked_route(args.v334_manifest_json, "v334_tong_equation_numeric_reasoner"),
        ],
        "thresholds": {
            "weak_total_min": args.weak_total_min,
            "weak_equation_min": args.weak_equation_min,
            "weak_bit_min": args.weak_bit_min,
            "max_losses": 0,
        },
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("integrated_total_correct =", integrated_total, flush=True)
    print("integrated_family_counts =", json.dumps(integrated_counts, sort_keys=True), flush=True)
    print("accepted_candidate_count =", len(accepted_candidate_ids), flush=True)
    print("accepted_candidate_ids =", json.dumps(accepted_candidate_ids, sort_keys=True), flush=True)
    print("gain_count =", len(gains), flush=True)
    print("loss_count =", loss_count, flush=True)
    print("family_summary =", json.dumps(family_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V336A INTEGRATED NO-LOSS SOLVER GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    rows = [
        {
            "id": "a",
            "family": "equation_transform",
            "answer": "42",
            "prediction": "24",
            "truncated": False,
        },
        {
            "id": "b",
            "family": "bit_manipulation",
            "answer": "1010",
            "prediction": "1010",
            "truncated": False,
        },
    ]
    candidates = [
        {
            "id": "a",
            "source_gate": "unit",
            "subtype": "equation_numeric_operator",
            "rule_class": "unit_rule",
            "candidate_source": "unit",
            "new_prediction": "42",
            "answer": "42",
            "baseline_prediction": "24",
            "proof": "unit",
            "verified_by_weak_label": True,
            "promotable_after_class_gate": True,
        }
    ]
    integrated_rows, trace = validate_and_apply_candidates(rows, candidates)
    gains, losses = compare_rows(rows, integrated_rows)
    if len(trace) != 1 or not trace[0]["accepted"]:
        raise AssertionError(trace)
    if [row["id"] for row in gains] != ["a"] or losses:
        raise AssertionError((gains, losses))
    print("v336a_integrated_no_loss_solver_gate_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--v324-manifest-json", type=Path, default=DEFAULT_V324_MANIFEST)
    parser.add_argument("--v324-accepted-csv", type=Path, default=DEFAULT_V324_ACCEPTED_CSV)
    parser.add_argument("--v329-manifest-json", type=Path, default=DEFAULT_V329_MANIFEST)
    parser.add_argument("--v329-accepted-csv", type=Path, default=DEFAULT_V329_ACCEPTED_CSV)
    parser.add_argument("--v333-manifest-json", type=Path, default=DEFAULT_V333_MANIFEST)
    parser.add_argument("--v334-manifest-json", type=Path, default=DEFAULT_V334_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/v336_integrated_no_loss_solver_gate" / utc_compact(),
    )
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-equation-min", type=int, default=61)
    parser.add_argument("--weak-bit-min", type=int, default=136)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

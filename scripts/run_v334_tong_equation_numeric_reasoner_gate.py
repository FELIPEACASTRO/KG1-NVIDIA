#!/usr/bin/env python3
"""CPU gate for direct Tong equation_numeric reasoner use.

The Tong equation_numeric source is useful as a DSL reference, but a direct
runtime replacement must be label-audited before any training or submission
path. This script verifies that hypothesis against the current V221-contract
baseline and blocks it if it produces wrong candidates.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_BASELINE_REPO,
    EXPECTED_ROW_CONTRACT_SHA256,
    download_file,
    normalize_row,
    parse_alice_prompt,
    row_contract,
    sha256_file,
)
from run_v333_tong_bit_reasoner_gate import (  # noqa: E402
    DEFAULT_TONG_COMMIT,
    DEFAULT_TONG_REPO_URL,
    fetch_tong_source,
    sha256_text,
)


DETAIL_COLUMNS = [
    "id",
    "answer",
    "baseline_prediction",
    "tong_prediction",
    "baseline_correct",
    "tong_correct",
    "tong_gain_vs_baseline",
    "tong_loss_vs_baseline",
    "parse_status",
    "tong_status",
    "query",
    "trace_sha256",
    "trace_chars",
    "prompt_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def import_tong_equation_modules(repo_dir: Path) -> dict[str, Any]:
    for item in (repo_dir, repo_dir / "reasoners"):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    eq_module = importlib.import_module("reasoners.equation_numeric")
    store_module = importlib.import_module("reasoners.store_types")
    reasoning_module = importlib.import_module("reasoning")
    return {
        "reasoning_equation_numeric": eq_module.reasoning_equation_numeric,
        "Problem": store_module.Problem,
        "Example": store_module.Example,
        "extract_answer": reasoning_module.extract_answer,
    }


def tong_equation_predict(row: dict[str, Any], modules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    examples, query, parse_status = parse_alice_prompt(row["prompt"])
    if parse_status != "ok":
        return "", {
            "parse_status": parse_status,
            "tong_status": "not_attempted",
            "query": query,
            "trace_sha256": "",
            "trace_chars": 0,
        }
    problem = modules["Problem"](
        id=str(row["id"]),
        category="equation_numeric_deduce",
        examples=[modules["Example"](lhs, rhs) for lhs, rhs in examples],
        question=query,
        answer=str(row["answer"]),
        prompt=row["prompt"],
    )
    trace = modules["reasoning_equation_numeric"](problem)
    if not trace:
        return "", {
            "parse_status": parse_status,
            "tong_status": "no_trace",
            "query": query,
            "trace_sha256": "",
            "trace_chars": 0,
        }
    prediction = str(modules["extract_answer"](trace)).strip()
    return prediction, {
        "parse_status": parse_status,
        "tong_status": "ok" if prediction else "empty_prediction",
        "query": query,
        "trace_sha256": sha256_text(trace),
        "trace_chars": len(trace),
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    correct = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        ok = verify_answer(row["answer"], row.get(prediction_key, ""))
        total += 1
        correct += int(ok)
        families[family]["rows"] += 1
        families[family]["correct"] += int(ok)
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "family": {key: dict(value) for key, value in sorted(families.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V334 TONG EQUATION NUMERIC REASONER GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("tong_repo_url =", args.tong_repo_url, flush=True)
    print("tong_commit =", args.tong_commit, flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    work_dir = Path(tempfile.mkdtemp(prefix="kg1_v334_tong_equation_gate_"))
    try:
        tong_repo_dir, resolved_commit = fetch_tong_source(work_dir, args.tong_repo_url, args.tong_commit)
        modules = import_tong_equation_modules(tong_repo_dir)
        baseline_path = Path(args.baseline_csv) if args.baseline_csv else download_file(
            args.baseline_repo,
            args.baseline_filename,
            work_dir / "baseline_download",
            token,
        )
        baseline_source = (
            str(baseline_path)
            if args.baseline_csv
            else f"hf://{args.baseline_repo}/{args.baseline_filename.strip('/')}"
        )
        rows = [normalize_row(row) for row in read_csv(baseline_path)]
        observed_contract = row_contract(rows)
        print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
        if observed_contract != args.expected_shared_row_contract_sha256:
            raise RuntimeError(
                "row contract mismatch: expected "
                + args.expected_shared_row_contract_sha256
                + ", got "
                + observed_contract
            )

        output_rows: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            out = dict(row)
            if row["family"] == "equation_transform":
                prediction, meta = tong_equation_predict(row, modules)
                baseline_correct = bool(row["correct_bool"])
                tong_correct = bool(prediction) and verify_answer(row["answer"], prediction)
                if prediction:
                    out["prediction"] = prediction
                detail_rows.append(
                    {
                        "id": row["id"],
                        "answer": row["answer"],
                        "baseline_prediction": row["prediction"],
                        "tong_prediction": prediction,
                        "baseline_correct": baseline_correct,
                        "tong_correct": tong_correct,
                        "tong_gain_vs_baseline": (not baseline_correct) and tong_correct,
                        "tong_loss_vs_baseline": baseline_correct and bool(prediction) and (not tong_correct),
                        "parse_status": meta["parse_status"],
                        "tong_status": meta["tong_status"],
                        "query": meta["query"],
                        "trace_sha256": meta["trace_sha256"],
                        "trace_chars": meta["trace_chars"],
                        "prompt_sha256": row["prompt_sha256"],
                    }
                )
            output_rows.append(out)
            if index % 100 == 0:
                print(f"v334_progress_rows = {index}", flush=True)

        baseline_summary = summarize(rows, "prediction")
        tong_summary = summarize(output_rows, "prediction")
        gains = [row for row in detail_rows if row["tong_gain_vs_baseline"]]
        losses = [row for row in detail_rows if row["tong_loss_vs_baseline"]]
        wrong_on_misses = [
            row
            for row in detail_rows
            if (not row["baseline_correct"]) and row["tong_prediction"] and (not row["tong_correct"])
        ]
        status_counts = dict(Counter(row["tong_status"] for row in detail_rows))
        parse_counts = dict(Counter(row["parse_status"] for row in detail_rows))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = args.output_dir / f"{args.label}_tong_equation_replace_predictions.csv"
        detail_csv = args.output_dir / f"{args.label}_tong_equation_detail.csv"
        manifest_json = args.output_dir / f"{args.label}_manifest.json"
        write_csv(predictions_csv, output_rows, list(output_rows[0]) if output_rows else [])
        write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)

        direct_gate_pass = (
            int(tong_summary["correct"]) >= args.weak_total_min
            and len(gains) > 0
            and len(losses) <= args.max_losses
            and not wrong_on_misses
        )
        manifest = {
            "schema_version": "kg1_v334_tong_equation_numeric_reasoner_gate_v1",
            "generated_at_utc": utc_now(),
            "label": args.label,
            "inputs": {
                "tong_repo_url": args.tong_repo_url,
                "expected_tong_commit": args.tong_commit,
                "resolved_tong_commit": resolved_commit,
                "baseline_csv": baseline_source,
                "baseline_runtime_path": str(baseline_path),
                "baseline_csv_sha256": sha256_file(baseline_path),
                "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
                "observed_shared_row_contract_sha256": observed_contract,
            },
            "baseline_summary": baseline_summary,
            "tong_direct_replace_summary": tong_summary,
            "equation_rows": len(detail_rows),
            "tong_status_counts": status_counts,
            "parse_status_counts": parse_counts,
            "gains_vs_baseline": len(gains),
            "losses_vs_baseline": len(losses),
            "wrong_on_baseline_misses": len(wrong_on_misses),
            "gain_ids": [row["id"] for row in gains],
            "loss_ids": [row["id"] for row in losses],
            "direct_replace_gate": {
                "pass": direct_gate_pass,
                "weak_total_min": args.weak_total_min,
                "max_losses": args.max_losses,
            },
            "decision": {
                "decision": "tong_equation_direct_replace_promotable"
                if direct_gate_pass
                else "tong_equation_direct_reasoner_blocked",
                "reason": (
                    f"baseline={baseline_summary['correct']}; "
                    f"tong_replace={tong_summary['correct']}; "
                    f"gains={len(gains)}; losses={len(losses)}; "
                    f"wrong_on_misses={len(wrong_on_misses)}"
                ),
                "next_action": (
                    "Use Tong equation_numeric.py only as a DSL source for guarded no-loss classes; "
                    "do not use its direct predictions as overrides or training authorization."
                ),
            },
            "outputs": {
                "predictions_csv": str(predictions_csv),
                "detail_csv": str(detail_csv),
                "manifest_json": str(manifest_json),
            },
        }
        write_json(manifest_json, manifest)
        print("manifest_json =", manifest_json, flush=True)
        print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
        print("=== V334 TONG EQUATION NUMERIC REASONER GATE END ===", flush=True)
        return manifest
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    default_output_dir = REPO_ROOT / "artifacts" / "v334_tong_equation_numeric_reasoner_gate" / utc_compact()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tong-repo-url", default=DEFAULT_TONG_REPO_URL)
    parser.add_argument("--tong-commit", default=DEFAULT_TONG_COMMIT)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--label", default="v334_tong_equation_numeric_reasoner_gate")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--max-losses", type=int, default=0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

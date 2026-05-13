#!/usr/bin/env python3
"""CPU gate for Tong Hui Kang bit-reasoner signal.

This gate is diagnostic and FinOps-safe. It imports the public Tong reasoner
from a pinned GitHub commit, compares it against the current KG1 bit solver on
the official train bit rows, then audits whether replacing only weak
bit_manipulation predictions would improve the current V221-contract baseline.

Weak labels are used only for audit and promotion decisions. The script does
not train, run GPU inference, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
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

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_BASELINE_REPO,
    EXPECTED_ROW_CONTRACT_SHA256,
    download_file,
    normalize_row,
    row_contract,
    sha256_file,
)
from solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem  # noqa: E402


DEFAULT_TONG_REPO_URL = "https://github.com/tonghuikang/nemotron"
DEFAULT_TONG_COMMIT = "82bd1880aa8a8986ad572ccd17ae35b2b5c7da85"

DETAIL_COLUMNS = [
    "id",
    "family",
    "answer",
    "baseline_prediction",
    "tong_prediction",
    "current_solver_prediction",
    "baseline_correct",
    "tong_correct",
    "current_solver_correct",
    "tong_gain_vs_baseline",
    "tong_loss_vs_baseline",
    "current_solver_gain_vs_baseline",
    "current_solver_loss_vs_baseline",
    "tong_trace_sha256",
    "tong_trace_chars",
    "tong_status",
    "current_solver_status",
    "prompt_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout_s: int = 300) -> str:
    printable = " ".join(cmd)
    print(f"+ {printable}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    if proc.stdout.strip():
        print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {printable}")
    return proc.stdout


def fetch_tong_source(work_dir: Path, repo_url: str, commit: str) -> tuple[Path, str]:
    repo_dir = work_dir / "tonghuikang_nemotron"
    repo_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "init", str(repo_dir)], timeout_s=120)
    run_cmd(["git", "-C", str(repo_dir), "remote", "add", "origin", repo_url], timeout_s=120)
    try:
        run_cmd(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", commit], timeout_s=300)
    except Exception:
        run_cmd(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "main"], timeout_s=300)
    resolved = run_cmd(["git", "-C", str(repo_dir), "rev-parse", "FETCH_HEAD"], timeout_s=120).strip().splitlines()[-1]
    if commit and resolved != commit:
        raise RuntimeError(f"Tong repo commit mismatch: expected {commit}, got {resolved}")
    run_cmd(
        [
            "git",
            "-C",
            str(repo_dir),
            "checkout",
            "FETCH_HEAD",
            "--",
            "reasoning.py",
            "reasoners",
            "train.csv",
        ],
        timeout_s=300,
    )
    return repo_dir, resolved


def import_tong_modules(repo_dir: Path) -> dict[str, Any]:
    for item in (repo_dir, repo_dir / "reasoners"):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    bit_module = importlib.import_module("reasoners.bit_manipulation")
    store_module = importlib.import_module("reasoners.store_types")
    reasoning_module = importlib.import_module("reasoning")
    return {
        "reasoning_bit_manipulation": bit_module.reasoning_bit_manipulation,
        "Problem": store_module.Problem,
        "Example": store_module.Example,
        "extract_answer": reasoning_module.extract_answer,
    }


def tong_predict(prompt: str, row_id: str, answer: str, modules: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    examples, question = parse_bit_problem(prompt)
    if not examples or not question:
        return "", {"status": "parse_failed", "trace_sha256": "", "trace_chars": 0}
    problem = modules["Problem"](
        id=str(row_id),
        category="bit_manipulation",
        examples=[modules["Example"](lhs, rhs) for lhs, rhs in examples],
        question=question,
        answer=str(answer),
        prompt=prompt,
    )
    trace = modules["reasoning_bit_manipulation"](problem)
    if not trace:
        return "", {"status": "no_trace", "trace_sha256": "", "trace_chars": 0}
    prediction = str(modules["extract_answer"](trace)).strip()
    return prediction, {
        "status": "ok" if prediction else "empty_prediction",
        "trace_sha256": sha256_text(trace),
        "trace_chars": len(trace),
    }


def current_solver_predict(prompt: str, solver: BitManipulationSolver) -> tuple[str, str]:
    try:
        prediction, _trace, _confidence = solver.solve(prompt)
    except Exception as exc:
        return "", f"error:{type(exc).__name__}"
    if prediction is None:
        return "", "abstain"
    return str(prediction).strip(), "ok"


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = 0
    correct = 0
    truncated = 0
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["family"])
        ok = verify_answer(row["answer"], row.get(prediction_key, ""))
        total += 1
        correct += int(ok)
        truncated += int(bool(row.get("truncated_bool", False)))
        families[family]["rows"] += 1
        families[family]["correct"] += int(ok)
        families[family]["truncated"] += int(bool(row.get("truncated_bool", False)))
    return {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "truncated": truncated,
        "family": {family: dict(counter) for family, counter in sorted(families.items())},
    }


def bit_train_audit(train_csv: Path, modules: dict[str, Any], max_rows: int) -> dict[str, Any]:
    rows = read_csv(train_csv)
    bit_rows = [
        row
        for row in rows
        if canonical_family(row.get("family") or row.get("type") or classify_puzzle(str(row.get("prompt", ""))))
        == "bit_manipulation"
    ]
    if max_rows > 0:
        bit_rows = bit_rows[:max_rows]
    solver = BitManipulationSolver()
    tong_correct = 0
    current_correct = 0
    both_correct = 0
    tong_gain_vs_current = 0
    tong_loss_vs_current = 0
    parse_missing = 0
    for index, row in enumerate(bit_rows, start=1):
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", "")).strip()
        row_id = str(row.get("id", index))
        tong_pred, tong_meta = tong_predict(prompt, row_id, answer, modules)
        current_pred, _ = current_solver_predict(prompt, solver)
        if tong_meta["status"] == "parse_failed":
            parse_missing += 1
        tong_ok = verify_answer(answer, tong_pred)
        current_ok = verify_answer(answer, current_pred)
        tong_correct += int(tong_ok)
        current_correct += int(current_ok)
        both_correct += int(tong_ok and current_ok)
        tong_gain_vs_current += int(tong_ok and not current_ok)
        tong_loss_vs_current += int(current_ok and not tong_ok)
        if index % 200 == 0:
            print(
                "train_bit_progress = "
                + json.dumps(
                    {
                        "rows": index,
                        "tong_correct": tong_correct,
                        "current_correct": current_correct,
                        "tong_gain_vs_current": tong_gain_vs_current,
                        "tong_loss_vs_current": tong_loss_vs_current,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    rows_count = len(bit_rows)
    return {
        "rows": rows_count,
        "tong_correct": tong_correct,
        "tong_accuracy": tong_correct / rows_count if rows_count else 0.0,
        "current_solver_correct": current_correct,
        "current_solver_accuracy": current_correct / rows_count if rows_count else 0.0,
        "both_correct": both_correct,
        "tong_gain_vs_current": tong_gain_vs_current,
        "tong_loss_vs_current": tong_loss_vs_current,
        "parse_missing": parse_missing,
        "max_rows": max_rows,
    }


def weak_bit_gate(
    rows: list[dict[str, Any]],
    modules: dict[str, Any],
    *,
    weak_total_min: int,
    weak_bit_min: int,
    max_losses: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    solver = BitManipulationSolver()
    detail_rows: list[dict[str, Any]] = []
    tong_replace_rows: list[dict[str, Any]] = []
    current_replace_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        tong_row = dict(row)
        current_row = dict(row)
        if row["family"] == "bit_manipulation":
            tong_pred, tong_meta = tong_predict(row["prompt"], row["id"], row["answer"], modules)
            current_pred, current_status = current_solver_predict(row["prompt"], solver)
            baseline_ok = bool(row["correct_bool"])
            tong_ok = bool(tong_pred) and verify_answer(row["answer"], tong_pred)
            current_ok = bool(current_pred) and verify_answer(row["answer"], current_pred)
            if tong_pred:
                tong_row["prediction"] = tong_pred
            if current_pred:
                current_row["prediction"] = current_pred
            detail_rows.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "answer": row["answer"],
                    "baseline_prediction": row["prediction"],
                    "tong_prediction": tong_pred,
                    "current_solver_prediction": current_pred,
                    "baseline_correct": baseline_ok,
                    "tong_correct": tong_ok,
                    "current_solver_correct": current_ok,
                    "tong_gain_vs_baseline": (not baseline_ok) and tong_ok,
                    "tong_loss_vs_baseline": baseline_ok and bool(tong_pred) and (not tong_ok),
                    "current_solver_gain_vs_baseline": (not baseline_ok) and current_ok,
                    "current_solver_loss_vs_baseline": baseline_ok and bool(current_pred) and (not current_ok),
                    "tong_trace_sha256": tong_meta["trace_sha256"],
                    "tong_trace_chars": tong_meta["trace_chars"],
                    "tong_status": tong_meta["status"],
                    "current_solver_status": current_status,
                    "prompt_sha256": row["prompt_sha256"],
                }
            )
        tong_replace_rows.append(tong_row)
        current_replace_rows.append(current_row)
        if index % 100 == 0:
            print(f"weak_bit_progress_rows = {index}", flush=True)

    baseline_summary = summarize(rows, "prediction")
    tong_summary = summarize(tong_replace_rows, "prediction")
    current_summary = summarize(current_replace_rows, "prediction")
    tong_gains = [row for row in detail_rows if row["tong_gain_vs_baseline"]]
    tong_losses = [row for row in detail_rows if row["tong_loss_vs_baseline"]]
    current_gains = [row for row in detail_rows if row["current_solver_gain_vs_baseline"]]
    current_losses = [row for row in detail_rows if row["current_solver_loss_vs_baseline"]]
    tong_bit = tong_summary["family"].get("bit_manipulation", {})
    direct_replace_pass = (
        int(tong_summary["correct"]) >= weak_total_min
        and int(tong_bit.get("correct", 0)) >= weak_bit_min
        and len(tong_losses) <= max_losses
        and len(tong_gains) > 0
    )
    teacher_trace_signal = len(tong_gains) > 0
    summary = {
        "baseline_summary": baseline_summary,
        "tong_bit_replace_summary": tong_summary,
        "current_solver_bit_replace_summary": current_summary,
        "tong_gains_vs_baseline": len(tong_gains),
        "tong_losses_vs_baseline": len(tong_losses),
        "current_solver_gains_vs_baseline": len(current_gains),
        "current_solver_losses_vs_baseline": len(current_losses),
        "tong_gain_ids": [row["id"] for row in tong_gains],
        "tong_loss_ids": [row["id"] for row in tong_losses],
        "direct_replace_gate": {
            "pass": direct_replace_pass,
            "weak_total_min": weak_total_min,
            "weak_bit_min": weak_bit_min,
            "max_losses": max_losses,
            "note": "Deployable only if the label audit shows no losses and a weak total gain.",
        },
        "teacher_trace_gate": {
            "pass": teacher_trace_signal,
            "teacher_rows": len(tong_gains),
            "note": "Label-audited signal only; use for deterministic trace distillation, not direct routing.",
        },
    }
    return summary, detail_rows, tong_replace_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V333 TONG BIT REASONER GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("tong_repo_url =", args.tong_repo_url, flush=True)
    print("tong_commit =", args.tong_commit, flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    work_dir = Path(tempfile.mkdtemp(prefix="kg1_v333_tong_bit_gate_"))
    keep_temp = bool(args.keep_temp)
    try:
        tong_repo_dir, resolved_commit = fetch_tong_source(work_dir, args.tong_repo_url, args.tong_commit)
        modules = import_tong_modules(tong_repo_dir)
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

        train_summary = bit_train_audit(tong_repo_dir / "train.csv", modules, args.max_train_rows)
        weak_summary, detail_rows, tong_replace_rows = weak_bit_gate(
            rows,
            modules,
            weak_total_min=args.weak_total_min,
            weak_bit_min=args.weak_bit_min,
            max_losses=args.max_losses,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = args.output_dir / f"{args.label}_tong_bit_replace_predictions.csv"
        detail_csv = args.output_dir / f"{args.label}_tong_bit_detail.csv"
        train_json = args.output_dir / f"{args.label}_train_bit_comparison.json"
        manifest_json = args.output_dir / f"{args.label}_manifest.json"

        write_csv(predictions_csv, tong_replace_rows, list(tong_replace_rows[0]) if tong_replace_rows else [])
        write_csv(detail_csv, detail_rows, DETAIL_COLUMNS)
        write_json(train_json, train_summary)

        direct_pass = bool(weak_summary["direct_replace_gate"]["pass"])
        teacher_pass = bool(weak_summary["teacher_trace_gate"]["pass"])
        decision = (
            "tong_bit_replace_promotable"
            if direct_pass
            else "tong_bit_teacher_trace_signal_only"
            if teacher_pass
            else "tong_bit_signal_blocked"
        )
        manifest = {
            "schema_version": "kg1_v333_tong_bit_reasoner_gate_v1",
            "generated_at_utc": utc_now(),
            "label": args.label,
            "inputs": {
                "tong_repo_url": args.tong_repo_url,
                "expected_tong_commit": args.tong_commit,
                "resolved_tong_commit": resolved_commit,
                "baseline_repo": args.baseline_repo,
                "baseline_filename": args.baseline_filename,
                "baseline_csv": baseline_source,
                "baseline_runtime_path": str(baseline_path),
                "baseline_csv_sha256": sha256_file(baseline_path),
                "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
                "observed_shared_row_contract_sha256": observed_contract,
                "weak_total_min": args.weak_total_min,
                "weak_bit_min": args.weak_bit_min,
                "max_losses": args.max_losses,
            },
            "train_bit_comparison": train_summary,
            "weak_gate": weak_summary,
            "decision": {
                "decision": decision,
                "direct_replace_deployable": direct_pass,
                "teacher_trace_signal": teacher_pass,
                "reason": (
                    f"baseline={weak_summary['baseline_summary']['correct']}; "
                    f"tong_replace={weak_summary['tong_bit_replace_summary']['correct']}; "
                    f"tong_gains={weak_summary['tong_gains_vs_baseline']}; "
                    f"tong_losses={weak_summary['tong_losses_vs_baseline']}"
                ),
                "next_action": (
                    "Promote Tong bit replacement only if direct_replace_gate passes; otherwise use gain rows as "
                    "teacher-trace fixtures and keep searching for a label-free confidence rule."
                ),
            },
            "outputs": {
                "predictions_csv": str(predictions_csv),
                "detail_csv": str(detail_csv),
                "train_json": str(train_json),
                "manifest_json": str(manifest_json),
            },
        }
        write_json(manifest_json, manifest)
        print("manifest_json =", manifest_json, flush=True)
        print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
        print("=== V333 TONG BIT REASONER GATE END ===", flush=True)
        return manifest
    finally:
        if keep_temp:
            print("kept_temp_dir =", work_dir, flush=True)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    default_output_dir = REPO_ROOT / "artifacts" / "v333_tong_bit_reasoner_gate" / utc_compact()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tong-repo-url", default=DEFAULT_TONG_REPO_URL)
    parser.add_argument("--tong-commit", default=DEFAULT_TONG_COMMIT)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--label", default="v333_tong_bit_reasoner_gate")
    parser.add_argument("--weak-total-min", type=int, default=193)
    parser.add_argument("--weak-bit-min", type=int, default=136)
    parser.add_argument("--max-losses", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means all official train bit rows.")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

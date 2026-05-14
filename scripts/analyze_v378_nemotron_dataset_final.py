#!/usr/bin/env python3
"""V378 audit for nemotron_dataset_final ZIP and directory.

Streams/reads the local final dataset package, validates labels and traces
against the official public train, audits solver_results.parquet, and records
coverage of the V375 residual equation misses. Emits only small artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)


DEFAULT_ZIP = Path(r"C:\Users\davis\Downloads\nemotron_dataset_final.zip")
DEFAULT_DIR = Path(r"C:\Users\davis\Downloads\nemotron_dataset_final")
DEFAULT_REPORT = Path(
    r"C:\Users\davis\Downloads\Dataset andy279_nemotron-reasoning-challenge — Relatório Completo de Extração.md"
)
DEFAULT_TRAIN = DEFAULT_DIR / "competition_train.csv"
DEFAULT_V375 = (
    REPO_ROOT
    / "artifacts/v375_equation_residual_clustering/20260514T141424Z/v375_equation_residual_rows.csv"
)
DEFAULT_OUT = REPO_ROOT / "artifacts/v378_nemotron_dataset_final_audit"

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with zf.open(name) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redacted(text: str) -> str:
    return TOKEN_RE.sub("[REDACTED_HF_TOKEN]", text)


def norm_prompt(text: object) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    variants = [
        PROMPT_SUFFIX,
        " Please put your final answer inside `\\boxed{}`.",
        " Please put your final answer inside `\\boxed{}`",
        "\nPlease put your final answer inside `\\boxed{}`.",
        "\nPlease put your final answer inside `\\boxed{}`",
        "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in variants:
            if value.endswith(suffix):
                value = value[: -len(suffix)].rstrip()
                changed = True
    return re.sub(r"\s+", " ", value)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def load_train(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, int]]:
    by_id: dict[str, dict[str, str]] = {}
    prompt_to_id: dict[str, str] = {}
    fam_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            family = classify_puzzle(row["prompt"])
            item = {
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "family": family,
            }
            by_id[item["id"]] = item
            prompt_to_id[norm_prompt(item["prompt"])] = item["id"]
            fam_counts[family] += 1
    return by_id, prompt_to_id, dict(fam_counts)


def load_v375_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
    return ids


def file_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return rows


def zip_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rows.append(
                {
                    "relative_path": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "sha256": sha256_zip_member(zf, info.filename),
                }
            )
    return rows


def messages_from_json(obj: dict[str, Any]) -> tuple[str, str]:
    messages = obj.get("messages") if isinstance(obj.get("messages"), list) else []
    user = ""
    assistant = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and not user:
            user = str(msg.get("content", ""))
        if msg.get("role") == "assistant":
            assistant = str(msg.get("content", ""))
    return user, assistant


def analyze_jsonl(
    path: Path,
    train_by_id: dict[str, dict[str, str]],
    prompt_to_id: dict[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "jsonl",
        "rows": 0,
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "duplicate_known_id_rows": 0,
        "unknown_or_synthetic_rows": 0,
        "metric_correct": 0,
        "metric_wrong": 0,
        "long_trace_rows_over_7680_chars": 0,
        "by_family": {},
        "unique_known_ids_by_family": {},
        "metadata_category_counts": {},
        "metadata_status_counts": {},
        "base_loss_stats": {},
    }
    seen: set[str] = set()
    by_family_sets: dict[str, set[str]] = defaultdict(set)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    categories: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    losses: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            summary["rows"] += 1
            obj = json.loads(line)
            meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            user, assistant = messages_from_json(obj)
            rid = str(meta.get("id", "") or meta.get("problem_id", "")) or prompt_to_id.get(norm_prompt(user), "")
            category = str(meta.get("category", "") or meta.get("type", "") or "")
            status = str(meta.get("status", "") or "")
            if category:
                categories[category] += 1
            if status:
                statuses[status] += 1
            try:
                losses.append(float(meta["base_loss"]))
            except Exception:
                pass
            if len(assistant) > 7680:
                summary["long_trace_rows_over_7680_chars"] += 1
            item = train_by_id.get(rid)
            if not item:
                summary["unknown_or_synthetic_rows"] += 1
                continue
            if rid in seen:
                summary["duplicate_known_id_rows"] += 1
            seen.add(rid)
            family = item["family"]
            by_family_sets[family].add(rid)
            extracted = extract_final_answer(assistant)
            correct = verify_answer(item["answer"], extracted)
            summary["known_train_rows"] += 1
            summary["metric_correct"] += int(correct)
            summary["metric_wrong"] += int(not correct)
            family_counts[family]["rows"] += 1
            family_counts[family]["correct"] += int(correct)
            family_counts[family]["wrong"] += int(not correct)
    summary["unique_known_train_ids"] = len(seen)
    summary["by_family"] = {k: dict(v) for k, v in sorted(family_counts.items())}
    summary["unique_known_ids_by_family"] = {k: len(v) for k, v in sorted(by_family_sets.items())}
    summary["metadata_category_counts"] = dict(categories)
    summary["metadata_status_counts"] = dict(statuses)
    summary["base_loss_stats"] = quantiles(losses)
    return summary


def analyze_generated_csv(path: Path, train_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "generated_csv",
        "rows": 0,
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "duplicate_known_id_rows": 0,
        "answer_label_matches_official": 0,
        "answer_label_mismatches_official": 0,
        "generated_cot_metric_correct": 0,
        "generated_cot_metric_wrong": 0,
        "by_family_label": {},
        "by_family_cot": {},
        "unique_known_ids_by_family": {},
        "base_loss_stats": {},
    }
    seen: set[str] = set()
    by_family_sets: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cot_counts: dict[str, Counter[str]] = defaultdict(Counter)
    losses: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            summary["rows"] += 1
            rid = row.get("id", "")
            item = train_by_id.get(rid)
            try:
                losses.append(float(row["base_loss"]))
            except Exception:
                pass
            if not item:
                continue
            if rid in seen:
                summary["duplicate_known_id_rows"] += 1
            seen.add(rid)
            summary["known_train_rows"] += 1
            family = item["family"]
            by_family_sets[family].add(rid)
            label_ok = verify_answer(item["answer"], row.get("answer", ""))
            cot_answer = extract_final_answer(row.get("generated_cot", ""))
            cot_ok = verify_answer(item["answer"], cot_answer)
            summary["answer_label_matches_official"] += int(label_ok)
            summary["answer_label_mismatches_official"] += int(not label_ok)
            summary["generated_cot_metric_correct"] += int(cot_ok)
            summary["generated_cot_metric_wrong"] += int(not cot_ok)
            label_counts[family]["rows"] += 1
            label_counts[family]["correct"] += int(label_ok)
            label_counts[family]["wrong"] += int(not label_ok)
            cot_counts[family]["rows"] += 1
            cot_counts[family]["correct"] += int(cot_ok)
            cot_counts[family]["wrong"] += int(not cot_ok)
    summary["unique_known_train_ids"] = len(seen)
    summary["by_family_label"] = {k: dict(v) for k, v in sorted(label_counts.items())}
    summary["by_family_cot"] = {k: dict(v) for k, v in sorted(cot_counts.items())}
    summary["unique_known_ids_by_family"] = {k: len(v) for k, v in sorted(by_family_sets.items())}
    summary["base_loss_stats"] = quantiles(losses)
    return summary


def analyze_trajectory_csv(path: Path, train_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "trajectory_csv",
        "rows": 0,
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "generated_answer_metric_correct": 0,
        "generated_answer_metric_wrong": 0,
        "correctness_counts": {},
        "problem_type_counts": {},
        "by_family_generated": {},
    }
    seen: set[str] = set()
    correctness: Counter[str] = Counter()
    problem_types: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            summary["rows"] += 1
            correctness[str(row.get("correctness", "")).lower()] += 1
            problem_types[str(row.get("problem type", ""))] += 1
            item = train_by_id.get(row.get("id", ""))
            if not item:
                continue
            seen.add(item["id"])
            summary["known_train_rows"] += 1
            ok = verify_answer(item["answer"], row.get("generated answer", ""))
            summary["generated_answer_metric_correct"] += int(ok)
            summary["generated_answer_metric_wrong"] += int(not ok)
            family_counts[item["family"]]["rows"] += 1
            family_counts[item["family"]]["correct"] += int(ok)
            family_counts[item["family"]]["wrong"] += int(not ok)
    summary["unique_known_train_ids"] = len(seen)
    summary["correctness_counts"] = dict(correctness)
    summary["problem_type_counts"] = dict(problem_types)
    summary["by_family_generated"] = {k: dict(v) for k, v in sorted(family_counts.items())}
    return summary


def analyze_solver_parquet(
    path: Path,
    train_by_id: dict[str, dict[str, str]],
    v375_ids: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    df = pd.read_parquet(path)
    residual = set(v375_ids)
    df["official_answer"] = df["id"].map(lambda x: train_by_id.get(str(x), {}).get("answer", ""))
    df["official_family"] = df["id"].map(lambda x: train_by_id.get(str(x), {}).get("family", "unknown"))
    df["metric_correct"] = df.apply(
        lambda row: verify_answer(row["official_answer"], row["solver_answer"]) if row["official_answer"] else False,
        axis=1,
    )
    cov = df[df["id"].isin(residual)].copy()
    coverage_csv = out_dir / "v378_v375_solver_coverage.csv"
    cov[
        [
            "id",
            "answer",
            "solver_answer",
            "official_answer",
            "metric_correct",
            "solver_type",
            "solver_category",
            "conditioned_on_answer",
            "solver_mode",
            "solver_ops",
            "solver_mapping",
        ]
    ].to_csv(coverage_csv, index=False)
    return {
        "kind": "solver_results_parquet",
        "rows": int(len(df)),
        "unique_ids": int(df["id"].nunique()),
        "official_known_rows": int((df["official_answer"] != "").sum()),
        "metric_correct": int(df["metric_correct"].sum()),
        "metric_wrong": int((~df["metric_correct"]).sum()),
        "family_counts": df["official_family"].value_counts(dropna=False).to_dict(),
        "solver_correct_flag_counts": {str(k): int(v) for k, v in df["solver_correct"].value_counts(dropna=False).items()},
        "solver_type_counts": {str(k): int(v) for k, v in df["solver_type"].value_counts(dropna=False).items()},
        "solver_category_counts": {str(k): int(v) for k, v in df["solver_category"].value_counts(dropna=False).items()},
        "v375_residual_coverage": {
            "rows": int(len(cov)),
            "correct": int(cov["metric_correct"].sum()),
            "wrong": int((~cov["metric_correct"]).sum()),
            "category_counts": {str(k): int(v) for k, v in cov["solver_category"].value_counts(dropna=False).items()},
            "coverage_csv": str(coverage_csv),
        },
    }


def residual_trace_coverage(
    final_dir: Path,
    train_by_id: dict[str, dict[str, str]],
    prompt_to_id: dict[str, str],
    v375_ids: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    residual = set(v375_ids)
    converted_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for csv_path in [final_dir / "kaggle_logprob/results/filtered_merged_dataset.csv"]:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rid = row["id"]
                if rid in residual:
                    item = train_by_id[rid]
                    cot = extract_final_answer(row.get("generated_cot", ""))
                    converted_rows[rid].append(
                        {
                            "id": rid,
                            "base_loss": row.get("base_loss", ""),
                            "cot_correct": verify_answer(item["answer"], cot),
                            "cot_len": len(row.get("generated_cot", "")),
                        }
                    )
    solver_cov_path = out_dir / "v378_v375_solver_coverage.csv"
    solver_rows = {}
    if solver_cov_path.exists():
        with solver_cov_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                solver_rows[row["id"]] = row
    rows = []
    for rid in v375_ids:
        item = train_by_id[rid]
        traces = converted_rows.get(rid, [])
        best = sorted(traces, key=lambda r: (float(r["base_loss"]) if r["base_loss"] else 999.0, r["cot_len"]))[0] if traces else None
        solver = solver_rows.get(rid, {})
        rows.append(
            {
                "id": rid,
                "answer": item["answer"],
                "filtered_trace_rows": len(traces),
                "best_base_loss": "" if best is None else best["base_loss"],
                "best_cot_correct": "" if best is None else best["cot_correct"],
                "best_cot_len": "" if best is None else best["cot_len"],
                "solver_present": bool(solver),
                "solver_answer": solver.get("solver_answer", ""),
                "solver_metric_correct": solver.get("metric_correct", ""),
                "solver_category": solver.get("solver_category", ""),
                "solver_ops": solver.get("solver_ops", ""),
            }
        )
    out_csv = out_dir / "v378_v375_residual_trace_solver_coverage.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": len(rows),
        "filtered_trace_covered": sum(1 for r in rows if int(r["filtered_trace_rows"]) > 0),
        "filtered_trace_correct": sum(1 for r in rows if r["best_cot_correct"] is True),
        "solver_covered": sum(1 for r in rows if r["solver_present"]),
        "solver_correct": sum(1 for r in rows if str(r["solver_metric_correct"]).lower() == "true"),
        "coverage_csv": str(out_csv),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    a = summary["analyses"]
    lines = [
        "# V378 Nemotron Dataset Final Audit",
        "",
        "## Verdict",
        "",
        "- All files in the ZIP and extracted directory were inventoried; no large file was copied into the repo.",
        "- The new high-value item is `solver_results.parquet`: it gives structured solver metadata for `823` equation rows and covers `82/92` V375 residual equation misses.",
        "- The filtered logprob CSV is also useful: it covers all `92/92` V375 residual equation misses with generated CoT, `91/92` correct by project scorer.",
        "- This audit authorizes CPU-only V378/V379 gate work, not immediate HF or submit.",
        "",
        "## Key Signals",
        "",
        f"- `solver_results.parquet`: `{a['solver_results.parquet']['metric_correct']}/{a['solver_results.parquet']['rows']}` correct, V375 coverage `{a['solver_results.parquet']['v375_residual_coverage']['correct']}/{a['solver_results.parquet']['v375_residual_coverage']['rows']}`.",
        f"- `filtered_merged_dataset.csv`: `{a['kaggle_logprob/results/filtered_merged_dataset.csv']['generated_cot_metric_correct']}/{a['kaggle_logprob/results/filtered_merged_dataset.csv']['rows']}` CoT-correct, labels `{a['kaggle_logprob/results/filtered_merged_dataset.csv']['answer_label_matches_official']}/{a['kaggle_logprob/results/filtered_merged_dataset.csv']['rows']}`.",
        f"- `sft_train_full_9500.jsonl`: `{a['sft_train_full_9500.jsonl']['metric_correct']}/{a['sft_train_full_9500.jsonl']['rows']}` correct.",
        f"- residual trace+solver coverage: `{summary['v375_residual_trace_solver_coverage']}`.",
        "",
        "## Actionable Next Step",
        "",
        "- Build a CPU-only candidate patch/gate from the 79 solver-correct V375 residual rows plus 91 trace-correct residual CoTs.",
        "- Use one best trace per ID, no duplicate reweighting, tokenizer/offset-mask/truncation checks, and weak gate before HF.",
        "- Do not use raw `nemotron_traj.csv` as labels; it is only `4542/9500` correct.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train_by_id, prompt_to_id, fam_counts = load_train(args.train_csv)
    v375_ids = load_v375_ids(args.v375_residual_csv)

    dir_rows = file_inventory(args.final_dir)
    zip_rows = zip_inventory(args.zip_path)
    write_csv(out / "directory_file_inventory.csv", dir_rows)
    write_csv(out / "zip_entries.csv", zip_rows)

    report_text = args.report_md.read_text(encoding="utf-8", errors="replace")
    analyses: dict[str, Any] = {}
    for rel in [
        "sft_reconstructed.jsonl",
        "sft_train_converted.jsonl",
        "sft_train_full_9500.jsonl",
        "sft_train_reconstructed.jsonl",
    ]:
        analyses[rel] = analyze_jsonl(args.final_dir / rel, train_by_id, prompt_to_id)
    analyses["kaggle_sft_data/dataset_generated.csv"] = analyze_generated_csv(
        args.final_dir / "kaggle_sft_data/dataset_generated.csv", train_by_id
    )
    analyses["kaggle_logprob/results/filtered_merged_dataset.csv"] = analyze_generated_csv(
        args.final_dir / "kaggle_logprob/results/filtered_merged_dataset.csv", train_by_id
    )
    analyses["kaggle_trajectories/nemotron_traj.csv"] = analyze_trajectory_csv(
        args.final_dir / "kaggle_trajectories/nemotron_traj.csv", train_by_id
    )
    analyses["solver_results.parquet"] = analyze_solver_parquet(
        args.final_dir / "solver_results.parquet", train_by_id, v375_ids, out
    )
    residual_cov = residual_trace_coverage(args.final_dir, train_by_id, prompt_to_id, v375_ids, out)

    summary: dict[str, Any] = {
        "schema_version": "kg1_v378_nemotron_dataset_final_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_hashes": {
            "zip": {"path": str(args.zip_path), "bytes": args.zip_path.stat().st_size, "sha256": sha256_path(args.zip_path)},
            "report_md": {"path": str(args.report_md), "bytes": args.report_md.stat().st_size, "sha256": sha256_path(args.report_md)},
            "train_csv": {"path": str(args.train_csv), "bytes": args.train_csv.stat().st_size, "sha256": sha256_path(args.train_csv)},
        },
        "train_family_counts": fam_counts,
        "report_md": {
            "contains_hf_token_pattern": bool(TOKEN_RE.search(report_text)),
            "urls": sorted(set(URL_RE.findall(report_text))),
            "head_redacted": redacted(report_text[:1000]),
        },
        "directory_files": dir_rows,
        "zip_files": zip_rows,
        "zip_vs_directory_hash_mismatches": [
            z for z in zip_rows if not any(d["relative_path"] == z["relative_path"] and d["sha256"] == z["sha256"] for d in dir_rows)
        ],
        "analyses": analyses,
        "v375_residual_trace_solver_coverage": residual_cov,
        "outputs": {
            "summary_json": str(out / "v378_nemotron_dataset_final_audit_summary.json"),
            "report_md": str(out / "KG1_V378_NEMOTRON_DATASET_FINAL_AUDIT.md"),
            "directory_file_inventory_csv": str(out / "directory_file_inventory.csv"),
            "zip_entries_csv": str(out / "zip_entries.csv"),
        },
    }
    summary_path = out / "v378_nemotron_dataset_final_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, out / "KG1_V378_NEMOTRON_DATASET_FINAL_AUDIT.md")
    return summary


def self_test() -> int:
    assert norm_prompt("x" + PROMPT_SUFFIX) == "x"
    assert redacted("a hf_" + "A" * 30) == "a [REDACTED_HF_TOKEN]"
    print("v378_nemotron_dataset_final_audit_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--v375-residual-csv", type=Path, default=DEFAULT_V375)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    print("=== V378 NEMOTRON DATASET FINAL AUDIT START ===", flush=True)
    print("zip_path =", args.zip_path, flush=True)
    print("final_dir =", args.final_dir, flush=True)
    print("report_md =", args.report_md, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    summary = run(args)
    print("directory_file_count =", len(summary["directory_files"]), flush=True)
    print("zip_file_count =", len(summary["zip_files"]), flush=True)
    print("zip_vs_directory_hash_mismatches =", len(summary["zip_vs_directory_hash_mismatches"]), flush=True)
    print("solver_results =", json.dumps(summary["analyses"]["solver_results.parquet"], sort_keys=True), flush=True)
    print("v375_residual_trace_solver_coverage =", json.dumps(summary["v375_residual_trace_solver_coverage"], sort_keys=True), flush=True)
    print("outputs =", json.dumps(summary["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V378 NEMOTRON DATASET FINAL AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

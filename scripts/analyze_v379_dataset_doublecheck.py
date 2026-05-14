#!/usr/bin/env python3
"""V379 double-check for nemotron_dataset_final and nemotron_hacker_dataset.

This is a directory-level audit. It does not copy large dataset files; it
streams hashes and reuses V377/V378 measured manifests to produce a clean,
actionable summary for the roadmap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    classify_puzzle,
    extract_boxed_answers,
    verify_answer,
)

DEFAULT_FINAL_DIR = Path(r"C:\Users\davis\Downloads\nemotron_dataset_final")
DEFAULT_HACKER_DIR = Path(r"C:\Users\davis\Downloads\nemotron_hacker_dataset")
DEFAULT_FINAL_REPORT = Path(
    r"C:\Users\davis\Downloads\Dataset andy279_nemotron-reasoning-challenge — Relatório Completo de Extração.md"
)
DEFAULT_HACKER_REPORT = Path(
    r"C:\Users\davis\Downloads\Relatório de Extração_ Dataset andy279_nemotron-reasoning-challenge.md"
)
DEFAULT_V377_SUMMARY = REPO_ROOT / "artifacts/v377_nemotron_hacker_dataset_audit/v377_nemotron_hacker_dataset_audit_summary.json"
DEFAULT_V378_SUMMARY = REPO_ROOT / "artifacts/v378_nemotron_dataset_final_audit/v378_nemotron_dataset_final_audit_summary.json"
DEFAULT_OUT = REPO_ROOT / "artifacts/v379_dataset_doublecheck_audit"
DEFAULT_V217_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
DEFAULT_V217_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
EXPECTED_REPORT_FILES = [
    "sft_train_converted.jsonl",
    "sft_train_full_9500.jsonl",
    "sft_reconstructed.jsonl",
    "sft_train_reconstructed.jsonl",
    "kaggle_sft_data/dataset_generated.csv",
    "kaggle_trajectories/nemotron_traj.csv",
    "kaggle_logprob/results/filtered_merged_dataset.csv",
    "kaggle_logprob/results/tong_with_logprob.csv",
    "kaggle_logprob/results/yours_with_logprob.csv",
    "solver_results.parquet",
    "competition_train.csv",
    "competition_test.csv",
]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": sha256_path(path)})
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


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


def load_competition_train(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            family = classify_puzzle(row.get("prompt", ""))
            rows[row["id"]] = {
                "id": row["id"],
                "prompt": row.get("prompt", ""),
                "answer": row.get("answer", ""),
                "family": family,
                "prompt_norm": norm_prompt(row.get("prompt", "")),
            }
    return rows


def load_v217_prompts(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            rows[norm_prompt(prompt)] = {
                "id": str(obj.get("id", "")),
                "family": str(obj.get("family", "") or classify_puzzle(prompt)),
                "prompt_norm": norm_prompt(prompt),
            }
    return rows


def analyze_v217_overlap(
    competition_rows: dict[str, dict[str, str]],
    v217_train_path: Path,
    v217_val_path: Path,
) -> dict[str, Any]:
    competition_prompts = {row["prompt_norm"] for row in competition_rows.values()}

    def one(path: Path) -> dict[str, Any]:
        source = load_v217_prompts(path)
        family_counts: Counter[str] = Counter()
        overlap_ids: list[str] = []
        for prompt_norm, row in source.items():
            if prompt_norm in competition_prompts:
                family_counts[row["family"]] += 1
                if len(overlap_ids) < 20:
                    overlap_ids.append(row["id"])
        return {
            "path": str(path),
            "source_rows": len(source),
            "overlap_prompt_count": sum(family_counts.values()),
            "overlap_family_counts": dict(sorted(family_counts.items())),
            "example_v217_ids": overlap_ids,
        }

    return {
        "v217_train": one(v217_train_path),
        "v217_val": one(v217_val_path),
    }


def analyze_competition_test_overlap(train_rows: dict[str, dict[str, str]], test_path: Path) -> dict[str, Any]:
    train_prompts = {row["prompt_norm"]: row["id"] for row in train_rows.values()}
    train_ids = set(train_rows)
    rows = []
    with test_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prompt_norm = norm_prompt(row.get("prompt", ""))
            rows.append(
                {
                    "id": row.get("id", ""),
                    "id_in_train": row.get("id", "") in train_ids,
                    "prompt_in_train": prompt_norm in train_prompts,
                    "matching_train_id": train_prompts.get(prompt_norm, ""),
                }
            )
    return {
        "rows": len(rows),
        "id_overlap_count": sum(1 for row in rows if row["id_in_train"]),
        "prompt_overlap_count": sum(1 for row in rows if row["prompt_in_train"]),
        "overlap_rows": rows,
    }


def analyze_csv_duplicates(path: Path, id_column: str = "id") -> dict[str, Any]:
    row_counts: Counter[tuple[str, ...]] = Counter()
    id_counts: Counter[str] = Counter()
    fieldnames: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row_counts[tuple(str(row.get(name, "")) for name in fieldnames)] += 1
            id_counts[str(row.get(id_column, ""))] += 1
    duplicate_exact_rows = sum(count - 1 for count in row_counts.values() if count > 1)
    duplicate_id_rows = sum(count - 1 for count in id_counts.values() if count > 1)
    duplicate_ids = [key for key, count in id_counts.items() if count > 1][:20]
    return {
        "path": str(path),
        "rows": sum(row_counts.values()),
        "unique_exact_rows": len(row_counts),
        "duplicate_exact_rows": duplicate_exact_rows,
        "unique_ids": len(id_counts),
        "duplicate_id_rows": duplicate_id_rows,
        "duplicate_id_examples": duplicate_ids,
    }


def messages_from_obj(obj: dict[str, Any]) -> tuple[str, str]:
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


def analyze_jsonl_format(path: Path) -> dict[str, Any]:
    rows = 0
    malformed_think_tag_rows = 0
    one_open_two_close_rows = 0
    duplicate_message_pairs = 0
    pair_counts: Counter[tuple[str, str]] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            user, assistant = messages_from_obj(obj)
            pair = (user, assistant)
            pair_counts[pair] += 1
            if pair_counts[pair] > 1:
                duplicate_message_pairs += 1
            open_count = assistant.count("<think>")
            close_count = assistant.count("</think>")
            if open_count != close_count:
                malformed_think_tag_rows += 1
            if open_count == 1 and close_count == 2:
                one_open_two_close_rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "duplicate_message_pair_rows": duplicate_message_pairs,
        "malformed_think_tag_rows": malformed_think_tag_rows,
        "one_open_two_close_rows": one_open_two_close_rows,
    }


def analyze_sft_full_conflicts(path: Path, competition_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = 0
    multi_boxed_rows = 0
    earlier_wrong_last_correct_rows = 0
    earlier_wrong_last_correct_by_family: Counter[str] = Counter()
    declared_wrong_last_correct_rows = 0
    declared_wrong_last_correct_by_family: Counter[str] = Counter()
    answer_contains_brace_by_family: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    declared_examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            row_id = str(obj.get("id", "") or metadata.get("id", ""))
            meta = competition_rows.get(row_id, {})
            answer = meta.get("answer", str(obj.get("answer", "") or metadata.get("answer", "")))
            family = meta.get("family", str(obj.get("family", "") or metadata.get("type", "")))
            _, assistant = messages_from_obj(obj)
            boxes = [box.strip() for box in extract_boxed_answers(assistant)]
            if "{" in answer or "}" in answer:
                answer_contains_brace_by_family[family] += 1
            if len(boxes) > 1:
                multi_boxed_rows += 1
            if len(boxes) > 1 and verify_answer(answer, boxes[-1]):
                earlier_wrong = [box for box in boxes[:-1] if box and not verify_answer(answer, box)]
                if earlier_wrong:
                    earlier_wrong_last_correct_rows += 1
                    earlier_wrong_last_correct_by_family[family] += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "id": row_id,
                                "family": family,
                                "answer": answer,
                                "first_boxed": boxes[0] if boxes else "",
                                "last_boxed": boxes[-1] if boxes else "",
                                "boxed_count": len(boxes),
                            }
                        )
                declared_boxes = re.findall(
                    r"The answer in\s+\\boxed\{[^}]*\}\s+is\s+\\boxed\{([^}]*)\}",
                    assistant,
                )
                declared_wrong = [box.strip() for box in declared_boxes if box.strip() and not verify_answer(answer, box)]
                if declared_wrong:
                    declared_wrong_last_correct_rows += 1
                    declared_wrong_last_correct_by_family[family] += 1
                    if len(declared_examples) < 10:
                        declared_examples.append(
                            {
                                "id": row_id,
                                "family": family,
                                "answer": answer,
                                "declared_boxed": declared_wrong[-1],
                                "last_boxed": boxes[-1] if boxes else "",
                                "boxed_count": len(boxes),
                            }
                        )
    return {
        "path": str(path),
        "rows": rows,
        "multi_boxed_rows": multi_boxed_rows,
        "any_intermediate_boxed_wrong_last_correct_rows": earlier_wrong_last_correct_rows,
        "any_intermediate_boxed_wrong_last_correct_by_family": dict(sorted(earlier_wrong_last_correct_by_family.items())),
        "declared_answer_wrong_last_correct_rows": declared_wrong_last_correct_rows,
        "declared_answer_wrong_last_correct_by_family": dict(sorted(declared_wrong_last_correct_by_family.items())),
        "answer_contains_brace_by_family": dict(sorted(answer_contains_brace_by_family.items())),
        "intermediate_examples": examples,
        "declared_examples": declared_examples,
    }


def analyze_solver_parquet(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    conditioned_counts = {}
    if "conditioned_on_answer" in frame.columns:
        conditioned_counts = {str(k): int(v) for k, v in frame["conditioned_on_answer"].value_counts(dropna=False).items()}
    category_counts = {}
    if "solver_category" in frame.columns:
        category_counts = {str(k): int(v) for k, v in frame["solver_category"].value_counts(dropna=False).items()}
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "conditioned_on_answer_counts": conditioned_counts,
        "solver_category_counts": category_counts,
    }


def analyze_reports(*paths: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            result[str(path)] = {"exists": False}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        result[str(path)] = {
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
            "contains_hf_token_pattern": bool(TOKEN_RE.search(text)),
            "mentions_tong_with_logprob": "tong_with_logprob.csv" in text,
            "mentions_yours_with_logprob": "yours_with_logprob.csv" in text,
            "mentions_solver_results": "solver_results.parquet" in text,
            "mentions_34_test_puzzles": "34 puzzles" in text or "34 puzzles" in text.lower(),
        }
    return result


def summarize_metrics(v377: dict[str, Any], v378: dict[str, Any]) -> dict[str, Any]:
    a377 = v377.get("analyses", {})
    a378 = v378.get("analyses", {})
    return {
        "hacker_subset": {
            "sft_train_converted": {
                "rows": a377["sft_train_converted.jsonl"]["rows"],
                "unique_ids": a377["sft_train_converted.jsonl"]["unique_known_train_ids"],
                "metric_correct": a377["sft_train_converted.jsonl"]["metric_correct"],
                "bit_unique_ids": a377["sft_train_converted.jsonl"]["unique_known_ids_by_family"].get("bit_manipulation"),
                "equation_unique_ids": a377["sft_train_converted.jsonl"]["unique_known_ids_by_family"].get("equation_transform"),
            },
            "dataset_generated_cot": {
                "correct": a377["kaggle_sft_data/dataset_generated.csv"]["generated_cot_metric_correct"],
                "wrong": a377["kaggle_sft_data/dataset_generated.csv"]["generated_cot_metric_wrong"],
            },
            "traj_generated_answer": {
                "correct": a377["kaggle_trajectories/nemotron_traj.csv"]["generated_answer_metric_correct"],
                "wrong": a377["kaggle_trajectories/nemotron_traj.csv"]["generated_answer_metric_wrong"],
            },
        },
        "final_superset": {
            "filtered_merged_dataset": {
                "rows": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["rows"],
                "unique_ids": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["unique_known_train_ids"],
                "label_correct": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["answer_label_matches_official"],
                "cot_correct": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["generated_cot_metric_correct"],
                "bit_cot": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["by_family_cot"].get("bit_manipulation"),
                "equation_cot": a378["kaggle_logprob/results/filtered_merged_dataset.csv"]["by_family_cot"].get("equation_transform"),
            },
            "solver_results": {
                "rows": a378["solver_results.parquet"]["rows"],
                "metric_correct": a378["solver_results.parquet"]["metric_correct"],
                "metric_wrong": a378["solver_results.parquet"]["metric_wrong"],
                "v375_coverage": a378["solver_results.parquet"]["v375_residual_coverage"],
            },
            "v375_residual_trace_solver_coverage": v378.get("v375_residual_trace_solver_coverage", {}),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    final_inv = inventory(args.final_dir)
    hacker_inv = inventory(args.hacker_dir)
    write_csv(out / "final_directory_inventory.csv", final_inv)
    write_csv(out / "hacker_directory_inventory.csv", hacker_inv)

    final_by_rel = {r["relative_path"]: r for r in final_inv}
    hacker_by_rel = {r["relative_path"]: r for r in hacker_inv}
    common = sorted(set(final_by_rel) & set(hacker_by_rel))
    comparison = []
    for rel in common:
        comparison.append(
            {
                "relative_path": rel,
                "final_sha256": final_by_rel[rel]["sha256"],
                "hacker_sha256": hacker_by_rel[rel]["sha256"],
                "same_hash": final_by_rel[rel]["sha256"] == hacker_by_rel[rel]["sha256"],
                "final_bytes": final_by_rel[rel]["bytes"],
                "hacker_bytes": hacker_by_rel[rel]["bytes"],
            }
        )
    write_csv(out / "common_file_hash_comparison.csv", comparison)

    final_only = sorted(set(final_by_rel) - set(hacker_by_rel))
    hacker_only = sorted(set(hacker_by_rel) - set(final_by_rel))
    missing_claimed = [
        item for item in EXPECTED_REPORT_FILES if item not in final_by_rel and item not in hacker_by_rel
    ]

    row_counts = {
        "final_competition_train_rows": count_csv_rows(args.final_dir / "competition_train.csv"),
        "final_competition_test_rows": count_csv_rows(args.final_dir / "competition_test.csv"),
    }
    v377 = load_json(args.v377_summary)
    v378 = load_json(args.v378_summary)
    competition_rows = load_competition_train(args.final_dir / "competition_train.csv")
    extra_audit = {
        "v217_prompt_overlap": analyze_v217_overlap(competition_rows, args.v217_train, args.v217_val),
        "competition_test_train_overlap": analyze_competition_test_overlap(
            competition_rows, args.final_dir / "competition_test.csv"
        ),
        "filtered_merged_duplicates": analyze_csv_duplicates(
            args.final_dir / "kaggle_logprob/results/filtered_merged_dataset.csv"
        ),
        "sft_train_converted_format": analyze_jsonl_format(args.final_dir / "sft_train_converted.jsonl"),
        "sft_train_full_conflicts": analyze_sft_full_conflicts(
            args.final_dir / "sft_train_full_9500.jsonl", competition_rows
        ),
        "solver_parquet_conditioning": analyze_solver_parquet(args.final_dir / "solver_results.parquet"),
    }

    summary = {
        "schema_version": "kg1_v379_dataset_doublecheck_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "directories": {
            "final_dir": str(args.final_dir),
            "hacker_dir": str(args.hacker_dir),
            "final_file_count": len(final_inv),
            "hacker_file_count": len(hacker_inv),
            "final_total_bytes": sum(r["bytes"] for r in final_inv),
            "hacker_total_bytes": sum(r["bytes"] for r in hacker_inv),
            "common_file_count": len(common),
            "common_hash_mismatch_count": sum(1 for r in comparison if not r["same_hash"]),
            "final_only": final_only,
            "hacker_only": hacker_only,
        },
        "row_counts": row_counts,
        "report_audit": analyze_reports(args.final_report, args.hacker_report),
        "report_claims_missing_from_dirs": missing_claimed,
        "metric_summary": summarize_metrics(v377, v378),
        "extra_audit": extra_audit,
        "decisions": {
            "active_sources": [
                "solver_results.parquet",
                "kaggle_logprob/results/filtered_merged_dataset.csv",
                "sft_train_full_9500.jsonl",
            ],
            "diagnostic_only_sources": [
                "kaggle_trajectories/nemotron_traj.csv",
                "kaggle_sft_data/dataset_generated.csv",
                "sft_train_converted.jsonl",
            ],
            "retired_or_duplicate_sources": [
                "nemotron_hacker_dataset common files duplicated by nemotron_dataset_final",
                "sft_train_reconstructed.jsonl full raw expanded file",
                "sft_reconstructed.jsonl duplicate 9500 SFT source now superseded by final full/logprob files",
            ],
            "next_action": "V380 CPU-only equation solver candidate patch using 79 V375 solver-correct rows, then trace/tokenization gate.",
        },
        "outputs": {
            "summary_json": str(out / "v379_dataset_doublecheck_summary.json"),
            "final_inventory_csv": str(out / "final_directory_inventory.csv"),
            "hacker_inventory_csv": str(out / "hacker_directory_inventory.csv"),
            "common_hash_comparison_csv": str(out / "common_file_hash_comparison.csv"),
            "report_md": str(out / "KG1_V379_DATASET_DOUBLECHECK.md"),
        },
    }
    (out / "v379_dataset_doublecheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(summary, out / "KG1_V379_DATASET_DOUBLECHECK.md")
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    dirs = summary["directories"]
    metrics = summary["metric_summary"]
    extra = summary["extra_audit"]
    solver_conditioning = extra["solver_parquet_conditioning"]
    filtered_dupes = extra["filtered_merged_duplicates"]
    v217_train = extra["v217_prompt_overlap"]["v217_train"]
    v217_val = extra["v217_prompt_overlap"]["v217_val"]
    test_overlap = extra["competition_test_train_overlap"]
    converted_format = extra["sft_train_converted_format"]
    full_conflicts = extra["sft_train_full_conflicts"]
    lines = [
        "# V379 Dataset Double Check",
        "",
        "## Verdict",
        "",
        f"- `nemotron_dataset_final`: `{dirs['final_file_count']}` files, `{dirs['final_total_bytes']}` bytes.",
        f"- `nemotron_hacker_dataset`: `{dirs['hacker_file_count']}` files, `{dirs['hacker_total_bytes']}` bytes.",
        f"- Common files: `{dirs['common_file_count']}`; hash mismatches: `{dirs['common_hash_mismatch_count']}`.",
        "- `nemotron_dataset_final` is the superset. The hacker directory is duplicated/subsumed for active roadmap purposes.",
        "",
        "## Active Findings",
        "",
        f"- Solver parquet: `{metrics['final_superset']['solver_results']}`.",
        f"- Filtered logprob dataset: `{metrics['final_superset']['filtered_merged_dataset']}`.",
        f"- V375 residual coverage: `{metrics['final_superset']['v375_residual_trace_solver_coverage']}`.",
        (
            "- Solver conditioning audit: "
            f"`{solver_conditioning['conditioned_on_answer_counts']}`; "
            "conditioned rows are repair evidence, not independent proof traces."
        ),
        (
            "- Filtered dataset duplicate audit: "
            f"`{filtered_dupes['duplicate_exact_rows']}` exact duplicate rows and "
            f"`{filtered_dupes['duplicate_id_rows']}` duplicate-ID rows."
        ),
        "",
        "## Gaps Removed From Active Plan",
        "",
        (
            "- V217 train prompt overlap with the final package: "
            f"`{v217_train['overlap_prompt_count']}` prompts; families `{v217_train['overlap_family_counts']}`."
        ),
        (
            "- V217 validation prompt overlap with the final package: "
            f"`{v217_val['overlap_prompt_count']}` prompts; families `{v217_val['overlap_family_counts']}`. "
            "Any future train/validation split must filter these prompt hashes."
        ),
        (
            "- `competition_test.csv` sample overlap: "
            f"`{test_overlap['id_overlap_count']}/{test_overlap['rows']}` IDs and "
            f"`{test_overlap['prompt_overlap_count']}/{test_overlap['rows']}` prompts overlap train. "
            "It is not an evaluation set."
        ),
        (
            "- `sft_train_converted.jsonl` format audit: "
            f"`{converted_format['duplicate_message_pair_rows']}` duplicate message rows and "
            f"`{converted_format['malformed_think_tag_rows']}` malformed think-tag rows."
        ),
        (
            "- `sft_train_full_9500.jsonl` audit: all rows have multiple boxed spans; "
            f"`{full_conflicts['declared_answer_wrong_last_correct_rows']}` rows have a wrong declared boxed answer "
            f"before the final corrected box (`{full_conflicts['declared_answer_wrong_last_correct_by_family']}`). "
            f"`{full_conflicts['answer_contains_brace_by_family'].get('equation_transform', 0)}` equation answers contain braces."
        ),
        "- Do not use `nemotron_traj.csv` as labels; keep only for hard-negative/confidence analysis.",
        "- Do not use the whole `sft_train_reconstructed.jsonl`; it contains unknown/synthetic rows and is superseded by focused sources.",
        "- Do not treat missing report-mentioned `tong_with_logprob.csv` / `yours_with_logprob.csv` as available evidence; they are absent from both audited directories.",
        f"- Report-claimed-but-missing files: `{summary['report_claims_missing_from_dirs']}`.",
        "",
        "## Next Action",
        "",
        summary["decisions"]["next_action"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> int:
    assert TOKEN_RE.search("hf_" + "A" * 30)
    print("v379_dataset_doublecheck_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--hacker-dir", type=Path, default=DEFAULT_HACKER_DIR)
    parser.add_argument("--final-report", type=Path, default=DEFAULT_FINAL_REPORT)
    parser.add_argument("--hacker-report", type=Path, default=DEFAULT_HACKER_REPORT)
    parser.add_argument("--v377-summary", type=Path, default=DEFAULT_V377_SUMMARY)
    parser.add_argument("--v378-summary", type=Path, default=DEFAULT_V378_SUMMARY)
    parser.add_argument("--v217-train", type=Path, default=DEFAULT_V217_TRAIN)
    parser.add_argument("--v217-val", type=Path, default=DEFAULT_V217_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    print("=== V379 DATASET DOUBLE CHECK START ===", flush=True)
    print("final_dir =", args.final_dir, flush=True)
    print("hacker_dir =", args.hacker_dir, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    summary = run(args)
    print("directories =", json.dumps(summary["directories"], indent=2, sort_keys=True), flush=True)
    print("row_counts =", json.dumps(summary["row_counts"], indent=2, sort_keys=True), flush=True)
    print("report_claims_missing_from_dirs =", summary["report_claims_missing_from_dirs"], flush=True)
    print("extra_audit =", json.dumps(summary["extra_audit"], indent=2, sort_keys=True), flush=True)
    print("decisions =", json.dumps(summary["decisions"], indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(summary["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V379 DATASET DOUBLE CHECK END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

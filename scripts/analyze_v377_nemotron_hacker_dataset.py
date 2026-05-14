#!/usr/bin/env python3
"""V377 audit for the local nemotron_hacker_dataset.zip package.

The script streams ZIP members without extracting them to disk. It validates
rows against the official public-train labels with the project metric, records
provenance/security risks, and emits only small audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    canonical_family,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)


DEFAULT_ZIP = Path(r"C:\Users\davis\Downloads\nemotron_hacker_dataset.zip")
DEFAULT_REPORT = Path(r"C:\Users\davis\Downloads\Relatório de Extração_ Dataset andy279_nemotron-reasoning-challenge.md")
DEFAULT_TRAIN = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_OUT = REPO_ROOT / "artifacts/v377_nemotron_hacker_dataset_audit"

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_train(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    by_id: dict[str, dict[str, str]] = {}
    by_prompt: dict[str, dict[str, str]] = {}
    fam_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family = classify_puzzle(row.get("prompt", ""))
            item = {
                "id": str(row.get("id", "")),
                "prompt": str(row.get("prompt", "")),
                "answer": str(row.get("answer", "")),
                "family": family,
            }
            by_id[item["id"]] = item
            by_prompt[norm_prompt(item["prompt"])] = item
            fam_counts[family] += 1
    return by_id, by_prompt, dict(fam_counts)


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


def safe_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except Exception:
        return None
    return number if math.isfinite(number) else None


def get_messages(obj: dict[str, Any]) -> tuple[str, str, str]:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return "", "", ""
    roles = ",".join(str(m.get("role", "")) for m in messages if isinstance(m, dict))
    user = ""
    assistant = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        if role == "user" and not user:
            user = str(msg.get("content", ""))
        if role == "assistant":
            assistant = str(msg.get("content", ""))
    return roles, user, assistant


def resolve_train_item(
    metadata_id: str,
    prompt: str,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> tuple[dict[str, str] | None, str]:
    if metadata_id and metadata_id in train_by_id:
        return train_by_id[metadata_id], "id"
    key = norm_prompt(prompt)
    if key in train_by_prompt:
        return train_by_prompt[key], "prompt"
    return None, ""


def add_family_counts(target: dict[str, dict[str, int]], family: str, correct: bool | None) -> None:
    row = target.setdefault(family, {"rows": 0, "correct": 0, "wrong": 0, "unknown": 0})
    row["rows"] += 1
    if correct is True:
        row["correct"] += 1
    elif correct is False:
        row["wrong"] += 1
    else:
        row["unknown"] += 1


def analyze_jsonl_member(
    zf: zipfile.ZipFile,
    name: str,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "jsonl",
        "rows": 0,
        "malformed": 0,
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "duplicate_known_id_rows": 0,
        "unknown_or_synthetic_rows": 0,
        "matched_by": {},
        "metric_correct": 0,
        "metric_wrong": 0,
        "no_final_answer": 0,
        "role_shapes": {},
        "metadata_category_counts": {},
        "metadata_type_counts": {},
        "metadata_status_counts": {},
        "metadata_source_counts": {},
        "by_family": {},
        "unique_known_ids_by_family": {},
        "base_loss_stats": {},
        "assistant_char_stats": {},
        "long_trace_rows_over_7680_chars": 0,
        "wrong_first25": [],
    }
    matched_by: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    base_losses: list[float] = []
    assistant_lens: list[float] = []
    seen_known_ids: set[str] = set()
    unique_by_family: dict[str, set[str]] = defaultdict(set)

    with zf.open(name) as raw:
        for raw_line in raw:
            summary["rows"] += 1
            try:
                obj = json.loads(raw_line)
            except Exception:
                summary["malformed"] += 1
                continue
            if not isinstance(obj, dict):
                summary["malformed"] += 1
                continue
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            metadata_id = str(metadata.get("id", "") or metadata.get("problem_id", ""))
            category = str(metadata.get("category", "") or metadata.get("type", "") or "")
            status = str(metadata.get("status", "") or "")
            source = str(metadata.get("source", "") or "")
            if category:
                categories[category] += 1
                types[canonical_family(category)] += 1
            if status:
                statuses[status] += 1
            if source:
                sources[source] += 1
            loss = safe_float(metadata.get("base_loss"))
            if loss is not None:
                base_losses.append(loss)

            shape, user, assistant = get_messages(obj)
            roles[shape] += 1
            assistant_lens.append(float(len(assistant)))
            if len(assistant) > 7680:
                summary["long_trace_rows_over_7680_chars"] += 1
            item, matched = resolve_train_item(metadata_id, user, train_by_id, train_by_prompt)
            if item:
                if item["id"] in seen_known_ids:
                    summary["duplicate_known_id_rows"] += 1
                seen_known_ids.add(item["id"])
                unique_by_family[item["family"]].add(item["id"])
                matched_by[matched] += 1
                summary["known_train_rows"] += 1
                family = item["family"]
                extracted = extract_final_answer(assistant)
                if extracted == "NOT_FOUND":
                    summary["no_final_answer"] += 1
                correct = verify_answer(item["answer"], extracted)
                if correct:
                    summary["metric_correct"] += 1
                else:
                    summary["metric_wrong"] += 1
                    if len(summary["wrong_first25"]) < 25:
                        summary["wrong_first25"].append(
                            {
                                "id": item["id"],
                                "family": family,
                                "expected": item["answer"],
                                "extracted": extracted,
                                "metadata_id": metadata_id,
                                "category": category,
                                "status": status,
                            }
                        )
                add_family_counts(summary["by_family"], family, correct)
            else:
                summary["unknown_or_synthetic_rows"] += 1
                fam = canonical_family(category) if category else classify_puzzle(user)
                add_family_counts(summary["by_family"], fam or "unknown", None)

    summary["matched_by"] = dict(matched_by)
    summary["role_shapes"] = dict(roles)
    summary["metadata_category_counts"] = dict(categories)
    summary["metadata_type_counts"] = dict(types)
    summary["metadata_status_counts"] = dict(statuses)
    summary["metadata_source_counts"] = dict(sources)
    summary["base_loss_stats"] = quantiles(base_losses)
    summary["assistant_char_stats"] = quantiles(assistant_lens)
    summary["unique_known_train_ids"] = len(seen_known_ids)
    summary["unique_known_ids_by_family"] = {k: len(v) for k, v in sorted(unique_by_family.items())}
    return summary


def analyze_dataset_generated_csv(
    zf: zipfile.ZipFile,
    name: str,
    train_by_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "dataset_generated_csv",
        "rows": 0,
        "columns": [],
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "duplicate_known_id_rows": 0,
        "unknown_ids": 0,
        "answer_label_matches_official": 0,
        "answer_label_mismatches_official": 0,
        "by_family": {},
        "by_family_generated_cot": {},
        "unique_known_ids_by_family": {},
        "type_counts": {},
        "generated_cot_has_boxed": 0,
        "generated_cot_metric_correct": 0,
        "generated_cot_metric_wrong": 0,
        "mismatch_first25": [],
    }
    type_counts: Counter[str] = Counter()
    seen_known_ids: set[str] = set()
    unique_by_family: dict[str, set[str]] = defaultdict(set)
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        summary["columns"] = list(reader.fieldnames or [])
        for row in reader:
            summary["rows"] += 1
            rid = str(row.get("id", ""))
            rtype = str(row.get("type", ""))
            if rtype:
                type_counts[rtype] += 1
            item = train_by_id.get(rid)
            if not item:
                summary["unknown_ids"] += 1
                continue
            if item["id"] in seen_known_ids:
                summary["duplicate_known_id_rows"] += 1
            seen_known_ids.add(item["id"])
            unique_by_family[item["family"]].add(item["id"])
            summary["known_train_rows"] += 1
            label_ok = verify_answer(item["answer"], row.get("answer", ""))
            if label_ok:
                summary["answer_label_matches_official"] += 1
            else:
                summary["answer_label_mismatches_official"] += 1
            generated_cot = str(row.get("generated_cot", ""))
            extracted = extract_final_answer(generated_cot)
            cot_correct: bool | None = None
            if extracted != "NOT_FOUND":
                summary["generated_cot_has_boxed"] += 1
                cot_correct = verify_answer(item["answer"], extracted)
                if cot_correct:
                    summary["generated_cot_metric_correct"] += 1
                else:
                    summary["generated_cot_metric_wrong"] += 1
            add_family_counts(summary["by_family"], item["family"], label_ok)
            add_family_counts(summary["by_family_generated_cot"], item["family"], cot_correct)
            if (not label_ok or cot_correct is False) and len(summary["mismatch_first25"]) < 25:
                summary["mismatch_first25"].append(
                    {
                        "id": rid,
                        "family": item["family"],
                        "official": item["answer"],
                        "label": row.get("answer", ""),
                        "generated_cot_extracted": extracted,
                    }
                )
    summary["type_counts"] = dict(type_counts)
    summary["unique_known_train_ids"] = len(seen_known_ids)
    summary["unique_known_ids_by_family"] = {k: len(v) for k, v in sorted(unique_by_family.items())}
    return summary


def analyze_trajectory_csv(
    zf: zipfile.ZipFile,
    name: str,
    train_by_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "trajectory_csv",
        "rows": 0,
        "columns": [],
        "known_train_rows": 0,
        "unique_known_train_ids": 0,
        "duplicate_known_id_rows": 0,
        "unknown_ids": 0,
        "correctness_counts": {},
        "problem_type_counts": {},
        "correct_answer_label_matches_official": 0,
        "correct_answer_label_mismatches_official": 0,
        "generated_answer_metric_correct": 0,
        "generated_answer_metric_wrong": 0,
        "by_family_generated": {},
        "generated_text_char_stats": {},
        "wrong_first25": [],
    }
    correctness: Counter[str] = Counter()
    problem_types: Counter[str] = Counter()
    gen_lens: list[float] = []
    seen_known_ids: set[str] = set()
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        summary["columns"] = list(reader.fieldnames or [])
        for row in reader:
            summary["rows"] += 1
            rid = str(row.get("id", ""))
            correctness[str(row.get("correctness", "")).lower()] += 1
            ptype = str(row.get("problem type", ""))
            if ptype:
                problem_types[ptype] += 1
            gen_lens.append(float(len(str(row.get("generated", "")))))
            item = train_by_id.get(rid)
            if not item:
                summary["unknown_ids"] += 1
                continue
            if item["id"] in seen_known_ids:
                summary["duplicate_known_id_rows"] += 1
            seen_known_ids.add(item["id"])
            summary["known_train_rows"] += 1
            label_ok = verify_answer(item["answer"], row.get("correct answer", ""))
            if label_ok:
                summary["correct_answer_label_matches_official"] += 1
            else:
                summary["correct_answer_label_mismatches_official"] += 1
            generated_ok = verify_answer(item["answer"], row.get("generated answer", ""))
            if generated_ok:
                summary["generated_answer_metric_correct"] += 1
            else:
                summary["generated_answer_metric_wrong"] += 1
            add_family_counts(summary["by_family_generated"], item["family"], generated_ok)
            if not generated_ok and len(summary["wrong_first25"]) < 25:
                summary["wrong_first25"].append(
                    {
                        "id": rid,
                        "family": item["family"],
                        "official": item["answer"],
                        "generated_answer": row.get("generated answer", ""),
                        "correctness": row.get("correctness", ""),
                    }
                )
    summary["correctness_counts"] = dict(correctness)
    summary["problem_type_counts"] = dict(problem_types)
    summary["generated_text_char_stats"] = quantiles(gen_lens)
    summary["unique_known_train_ids"] = len(seen_known_ids)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    files = summary["file_hashes"]
    zip_members = summary["zip_members"]
    analyses = summary["analyses"]

    lines = [
        "# V377 Nemotron Hacker Dataset Audit",
        "",
        "## Verdict",
        "",
        "- ZIP was streamed in place; no large extraction was left on disk.",
        "- The package contains useful SFT/trajectory material, but the attached report describes gated-access bypass and 'leak' provenance. That makes the data compliance-sensitive.",
        "- Technically useful rows must remain blocked until legal/rules provenance is cleared and the CPU data gate proves no leakage/domain drift.",
        "- No HF GPU job, package, or Kaggle submit is authorized by this audit.",
        "",
        "## File Hashes",
        "",
    ]
    for key, row in files.items():
        lines.append(f"- `{key}`: bytes `{row['bytes']}`, sha256 `{row['sha256']}`")
    lines += ["", "## ZIP Contents", ""]
    for row in sorted(zip_members, key=lambda x: x["file_size"], reverse=True):
        lines.append(
            f"- `{row['filename']}`: bytes `{row['file_size']}`, sha256 `{row['sha256']}`"
        )
    lines += ["", "## Measured Dataset Signals", ""]

    for name, data in analyses.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- rows: `{data.get('rows')}`")
        if "known_train_rows" in data:
            lines.append(f"- known official train rows: `{data.get('known_train_rows')}`")
        if "unique_known_train_ids" in data:
            lines.append(
                f"- unique known official IDs: `{data.get('unique_known_train_ids')}`; "
                f"duplicate known-ID rows: `{data.get('duplicate_known_id_rows')}`"
            )
        if "unknown_or_synthetic_rows" in data:
            lines.append(f"- unknown/synthetic rows: `{data.get('unknown_or_synthetic_rows')}`")
        if "metric_correct" in data:
            lines.append(f"- project metric correct: `{data.get('metric_correct')}`; wrong: `{data.get('metric_wrong')}`")
        if "answer_label_matches_official" in data:
            lines.append(
                f"- answer labels vs official: matches `{data.get('answer_label_matches_official')}`, "
                f"mismatches `{data.get('answer_label_mismatches_official')}`"
            )
        if "generated_answer_metric_correct" in data:
            lines.append(
                f"- generated answer vs official: correct `{data.get('generated_answer_metric_correct')}`, "
                f"wrong `{data.get('generated_answer_metric_wrong')}`"
            )
        if "base_loss_stats" in data and data["base_loss_stats"]:
            lines.append(f"- base_loss stats: `{data['base_loss_stats']}`")
        if "by_family" in data:
            lines.append(f"- by family: `{data['by_family']}`")
        if "unique_known_ids_by_family" in data and data["unique_known_ids_by_family"]:
            lines.append(f"- unique IDs by family: `{data['unique_known_ids_by_family']}`")
        if "by_family_generated_cot" in data:
            lines.append(f"- generated CoT by family: `{data['by_family_generated_cot']}`")
        if "by_family_generated" in data:
            lines.append(f"- generated by family: `{data['by_family_generated']}`")
        if "metadata_status_counts" in data and data["metadata_status_counts"]:
            lines.append(f"- status counts: `{data['metadata_status_counts']}`")
        lines.append("")

    lines += [
        "## Actionable Decision",
        "",
        "- Treat `sft_train_full_9500.jsonl` and `sft_train_converted.jsonl` as candidate trace sources only after source/rules clearance.",
        "- `sft_train_converted.jsonl`/`dataset_generated.csv` are interesting because they are filtered/logprob-style and smaller, but they must not bypass anti-leakage and tokenizer gates.",
        "- `nemotron_traj.csv` is best used as hard-negative/confidence metadata, not as label source.",
        "- The direct next step is still CPU-only V377/V378 filtered trace gate: prove rows, hashes, families, length, loss buckets, and no-loss weak behavior before any HF spend.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_by_id, train_by_prompt, family_counts = load_train(args.train_csv)

    summary: dict[str, Any] = {
        "schema_version": "kg1_v377_nemotron_hacker_dataset_audit_v1",
        "generated_at_utc": utc_now(),
        "train_family_counts": family_counts,
        "file_hashes": {
            "zip": {
                "path": str(args.zip_path),
                "bytes": args.zip_path.stat().st_size,
                "sha256": sha256_path(args.zip_path),
            },
            "report_md": {
                "path": str(args.report_md),
                "bytes": args.report_md.stat().st_size,
                "sha256": sha256_path(args.report_md),
            },
            "train_csv": {
                "path": str(args.train_csv),
                "bytes": args.train_csv.stat().st_size,
                "sha256": sha256_path(args.train_csv),
            },
        },
        "report_md": {},
        "zip_members": [],
        "analyses": {},
        "outputs": {},
    }

    report_text = args.report_md.read_text(encoding="utf-8", errors="replace")
    summary["report_md"] = {
        "contains_hf_token_pattern": bool(TOKEN_RE.search(report_text)),
        "urls": sorted(set(URL_RE.findall(report_text))),
        "head_redacted": redacted(report_text[:1200]),
        "risk_flags": [
            "report_mentions_gated_manual_approval",
            "report_mentions_403_forbidden",
            "report_mentions_vazamento_or_leak",
            "report_mentions_bypass_or_attack_vectors",
        ],
    }

    with zipfile.ZipFile(args.zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = {
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "sha256": sha256_zip_member(zf, info.filename),
            }
            summary["zip_members"].append(member)

            if info.filename.endswith(".jsonl"):
                summary["analyses"][info.filename] = analyze_jsonl_member(
                    zf, info.filename, train_by_id, train_by_prompt
                )
            elif info.filename.endswith("dataset_generated.csv"):
                summary["analyses"][info.filename] = analyze_dataset_generated_csv(
                    zf, info.filename, train_by_id
                )
            elif info.filename.endswith("nemotron_traj.csv"):
                summary["analyses"][info.filename] = analyze_trajectory_csv(
                    zf, info.filename, train_by_id
                )
            elif info.filename.endswith(".md"):
                with zf.open(info.filename) as raw:
                    text = raw.read(min(info.file_size, 20000)).decode("utf-8", errors="replace")
                summary["analyses"][info.filename] = {
                    "kind": "markdown",
                    "bytes": info.file_size,
                    "contains_hf_token_pattern": bool(TOKEN_RE.search(text)),
                    "urls": sorted(set(URL_RE.findall(text))),
                    "head_redacted": redacted(text[:1200]),
                }

    zip_entries_csv = out_dir / "zip_entries.csv"
    with zip_entries_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["filename", "file_size", "compress_size", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["zip_members"])

    summary_json = out_dir / "v377_nemotron_hacker_dataset_audit_summary.json"
    report_md = out_dir / "KG1_V377_NEMOTRON_HACKER_DATASET_AUDIT.md"
    summary["outputs"] = {
        "summary_json": str(summary_json),
        "zip_entries_csv": str(zip_entries_csv),
        "report_md": str(report_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, report_md)
    return summary


def self_test() -> int:
    assert norm_prompt("abc" + PROMPT_SUFFIX) == "abc"
    assert redacted("x hf_" + "A" * 30 + " y") == "x [REDACTED_HF_TOKEN] y"
    print("v377_nemotron_hacker_dataset_audit_self_test=ok", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    print("=== V377 NEMOTRON HACKER DATASET AUDIT START ===", flush=True)
    print("zip_path =", args.zip_path, flush=True)
    print("report_md =", args.report_md, flush=True)
    print("train_csv =", args.train_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    summary = run(args)
    print("zip_member_count =", len(summary["zip_members"]), flush=True)
    for name, data in summary["analyses"].items():
        print("analysis =", json.dumps({"name": name, "rows": data.get("rows"), "kind": data.get("kind"), "known": data.get("known_train_rows"), "correct": data.get("metric_correct", data.get("generated_answer_metric_correct", "")), "wrong": data.get("metric_wrong", data.get("generated_answer_metric_wrong", ""))}, sort_keys=True), flush=True)
    print("outputs =", json.dumps(summary["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V377 NEMOTRON HACKER DATASET AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

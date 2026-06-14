#!/usr/bin/env python3
"""Pre-score KG1 predictions with the official Kaggle metric semantics.

This is CPU-only. It does not load Nemotron and does not submit to Kaggle.
It consumes a solution CSV plus a prediction CSV/JSONL produced by an eval job,
extracts final answers with the public metric logic, verifies them, and emits
score summaries by answer-shape bucket, family/status when available, and
failure taxonomy. It can also audit a submission.zip/adapter dir so the scored
artifact can be tied to concrete adapter hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

OFFICIAL_VISIBLE_SCORE_DEFAULTS = {
    "max_lora_rank": 32,
    "max_tokens": 7680,
    "top_p": 1.0,
    "temperature": 0.0,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
}


def fs_name(path: Path) -> str:
    """Return a Windows long-path-safe filesystem name."""

    text = str(path)
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            return "\\\\?\\" + resolved
        return resolved
    return text


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(fs_name(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_final_answer(text: str | None) -> str:
    r"""Extract final answer using the public Kaggle metric fallback order."""

    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [item.strip() for item in matches if item.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def extraction_source(text: str | None) -> str:
    if text is None:
        return "not_found"
    value = str(text)
    if re.search(r"\\boxed\{", value):
        extracted = extract_final_answer(value)
        return "boxed_empty" if extracted == "" else "boxed"
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    if any(re.findall(pattern, value, re.IGNORECASE) for pattern in patterns):
        return "final_answer_phrase"
    if re.findall(r"-?\d+(?:\.\d+)?", value):
        return "last_number"
    if [line.strip() for line in value.splitlines() if line.strip()]:
        return "last_line"
    return "not_found"


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify one prediction with the official public metric behavior."""

    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def answer_bucket(answer: object) -> str:
    text = str(answer).strip()
    if re.fullmatch(r"[01]+", text):
        return "BINARY_STRICT"
    try:
        float(text)
        return "NUMERIC_TOL"
    except Exception:
        return "STRING_EXACT"


def failure_reason(expected: str, extracted: str, source: str, correct: bool) -> str:
    if correct:
        return "CORRECT"
    if extracted == "NOT_FOUND" or source == "not_found":
        return "NOT_FOUND"
    bucket = answer_bucket(expected)
    observed = str(extracted).strip()
    exp = str(expected).strip()
    if bucket == "BINARY_STRICT":
        if re.fullmatch(r"[01]+", observed):
            if observed.lstrip("0") == exp.lstrip("0") and observed != exp:
                return "BINARY_WIDTH_OR_LEADING_ZERO"
            return "BINARY_WRONG_VALUE"
        return "BINARY_MALFORMED_OR_DECIMAL"
    if bucket == "NUMERIC_TOL":
        try:
            exp_num = float(exp)
            obs_num = float(observed)
            rel = abs(exp_num - obs_num) / max(abs(exp_num), 1e-12)
            return "NUMERIC_OUT_OF_TOL" if rel >= 1e-2 else "NUMERIC_ABS_TOL_FAIL"
        except Exception:
            return "NUMERIC_PARSE_FAIL"
    if source != "boxed":
        return "STRING_FALLBACK_SOURCE"
    if observed.lower() == exp.lower() and observed != exp:
        return "STRING_CASE_ONLY"
    if observed.strip().lower() == exp.strip().lower() and observed != exp:
        return "STRING_WHITESPACE_EDGE"
    return "STRING_WRONG_VALUE"


def read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(fs_name(path), encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("_line_no", line_no)
                    rows.append(row)
        return rows
    with open(fs_name(path), encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_name(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detect_prediction_text(row: dict[str, Any], raw_col: str | None, pred_col: str | None) -> tuple[str, str]:
    if raw_col:
        return str(row.get(raw_col, "")), raw_col
    if pred_col:
        return str(row.get(pred_col, "")), pred_col
    for candidate in ("raw_output", "raw_text", "output", "response", "completion"):
        if candidate in row:
            return str(row.get(candidate, "")), candidate
    for candidate in ("prediction", "extracted_prediction", "answer"):
        if candidate in row:
            return str(row.get(candidate, "")), candidate
    raise ValueError(
        "could not detect prediction column; pass --raw-output-col or --prediction-col"
    )


def is_preextracted_prediction_col(source_col: str, args: argparse.Namespace) -> bool:
    if args.raw_output_col and source_col == args.raw_output_col:
        return False
    if args.prediction_col and source_col == args.prediction_col:
        return True
    return source_col in {"prediction", "extracted_prediction", "answer"}


def load_family_map(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    mapping: dict[str, dict[str, str]] = {}
    for row in read_table(path):
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            continue
        mapping[row_id] = {
            "family": str(row.get("family") or row.get("category") or row.get("type") or ""),
            "status": str(row.get("status") or ""),
        }
    return mapping


def parse_threshold_specs(items: list[str] | None, flag_name: str) -> dict[str, float]:
    specs: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag_name} expects entries like family=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{flag_name} got an empty family name in {item!r}")
        specs[key] = float(value)
    return specs


def audit_adapter_path(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {"path": str(path), "exists": False, "error": "path_not_found"}
    result: dict[str, Any] = {"path": str(path), "exists": True}
    if path.is_file() and path.suffix.lower() == ".zip":
        result["kind"] = "submission_zip"
        result["zip_sha256"] = sha256_file(path)
        result["zip_size_bytes"] = path.stat().st_size
        with zipfile.ZipFile(fs_name(path)) as archive:
            names = archive.namelist()
            result["zip_entries"] = names
            config_entries = [name for name in names if name.replace("\\", "/").endswith("adapter_config.json")]
            model_entries = [name for name in names if name.replace("\\", "/").endswith("adapter_model.safetensors")]
            result["adapter_config_entries"] = config_entries
            result["adapter_model_entries"] = model_entries
            result["adapter_config_count"] = len(config_entries)
            result["adapter_model_count"] = len(model_entries)
            result["required_entries_missing"] = []
            if not config_entries:
                result["required_entries_missing"].append("adapter_config.json")
            if not model_entries:
                result["required_entries_missing"].append("adapter_model.safetensors")
            if len(config_entries) == 1:
                config_entry = config_entries[0]
                cfg = json.loads(archive.read(config_entry).decode("utf-8"))
                result["adapter_config"] = adapter_config_summary(cfg)
                result["adapter_config_sha256"] = hashlib.sha256(archive.read(config_entry)).hexdigest()
                config_dir = str(Path(config_entry).parent).replace("\\", "/")
                if config_dir == ".":
                    config_dir = ""
                expected_model_entry = (
                    f"{config_dir}/adapter_model.safetensors" if config_dir else "adapter_model.safetensors"
                )
                result["official_lora_dir_entry"] = config_dir or "."
                result["model_in_same_dir_as_config"] = expected_model_entry in names
            if len(model_entries) == 1:
                model_entry = model_entries[0]
                with archive.open(model_entry) as handle:
                    digest = hashlib.sha256()
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                    result["adapter_model_sha256"] = digest.hexdigest()
                    result["adapter_model_size_bytes"] = archive.getinfo(model_entry).file_size
        return result
    result["kind"] = "adapter_dir" if path.is_dir() else "file"
    cfg_path = path / "adapter_config.json" if path.is_dir() else None
    model_path = path / "adapter_model.safetensors" if path.is_dir() else None
    if cfg_path and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        result["adapter_config"] = adapter_config_summary(cfg)
        result["adapter_config_sha256"] = sha256_file(cfg_path)
    if model_path and model_path.exists():
        result["adapter_model_sha256"] = sha256_file(model_path)
        result["adapter_model_size_bytes"] = model_path.stat().st_size
    return result


def adapter_config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "base_model_name_or_path",
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "use_rslora",
        "bias",
        "target_modules",
        "target_parameters",
        "modules_to_save",
    ]
    return {field: cfg.get(field) for field in fields if field in cfg}


def score_predictions(args: argparse.Namespace) -> dict[str, Any]:
    solution_rows = read_table(args.solution_csv)
    prediction_rows = read_table(args.predictions)
    family_map = load_family_map(args.family_map)
    solution_ids = [str(row.get(args.id_col, "")).strip() for row in solution_rows]
    prediction_ids = [str(row.get(args.id_col, "")).strip() for row in prediction_rows]
    solution_id_counts = Counter(row_id for row_id in solution_ids if row_id)
    prediction_id_counts = Counter(row_id for row_id in prediction_ids if row_id)
    duplicate_solution_ids = sorted(row_id for row_id, count in solution_id_counts.items() if count > 1)
    duplicate_prediction_ids = sorted(row_id for row_id, count in prediction_id_counts.items() if count > 1)
    blank_solution_id_rows = sum(1 for row_id in solution_ids if not row_id)
    blank_prediction_id_rows = sum(1 for row_id in prediction_ids if not row_id)
    solution_by_id = {str(row[args.id_col]).strip(): row for row in solution_rows}
    prediction_by_id = {
        str(row[args.id_col]).strip(): row
        for row in prediction_rows
        if str(row.get(args.id_col, "")).strip()
    }
    missing_prediction_ids = sorted(set(solution_by_id) - set(prediction_by_id))
    extra_prediction_ids = sorted(set(prediction_by_id) - set(solution_by_id))

    details: list[dict[str, Any]] = []
    counters = Counter()
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_status: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    failure_counts = Counter()

    for row_id, solution in solution_by_id.items():
        expected = str(solution.get(args.answer_col, "")).strip()
        pred_row = prediction_by_id.get(row_id, {})
        raw_text, source_col = detect_prediction_text(pred_row, args.raw_output_col, args.prediction_col) if pred_row else ("", "")
        if pred_row and is_preextracted_prediction_col(source_col, args):
            source = "preextracted_prediction"
            extracted = raw_text.strip() if raw_text.strip() else "NOT_FOUND"
        else:
            source = extraction_source(raw_text)
            extracted = extract_final_answer(raw_text)
        correct = verify(expected, extracted)
        bucket = answer_bucket(expected)
        meta = family_map.get(row_id, {})
        family = str(solution.get("family") or solution.get("category") or solution.get("type") or meta.get("family") or "unknown")
        status = str(solution.get("status") or meta.get("status") or "unknown")
        reason = failure_reason(expected, extracted, source, correct)

        counters["rows"] += 1
        counters["correct" if correct else "incorrect"] += 1
        counters["missing_prediction_row"] += int(not pred_row)
        counters["no_box_fallback"] += int(source not in {"boxed", "preextracted_prediction"})
        counters["not_found"] += int(extracted == "NOT_FOUND")
        by_bucket[bucket]["rows"] += 1
        by_bucket[bucket]["correct"] += int(correct)
        by_family[family]["rows"] += 1
        by_family[family]["correct"] += int(correct)
        by_status[status]["rows"] += 1
        by_status[status]["correct"] += int(correct)
        by_source[source]["rows"] += 1
        by_source[source]["correct"] += int(correct)
        failure_counts[reason] += 1

        details.append(
            {
                "id": row_id,
                "expected": expected,
                "extracted": extracted,
                "correct": str(correct),
                "answer_bucket": bucket,
                "family": family,
                "status": status,
                "extract_source": source,
                "failure_reason": reason,
                "prediction_source_col": source_col,
                "raw_output_chars": len(raw_text),
            }
        )

    adapter_audit = audit_adapter_path(args.adapter_artifact)
    artifact_blockers = artifact_validation_blockers(args, adapter_audit)
    score = counters["correct"] / counters["rows"] if counters["rows"] else 0.0
    metric_blockers = metric_validation_blockers(
        args=args,
        score=score,
        counters=counters,
        by_family=by_family,
        duplicate_solution_ids=duplicate_solution_ids,
        duplicate_prediction_ids=duplicate_prediction_ids,
        blank_solution_id_rows=blank_solution_id_rows,
        blank_prediction_id_rows=blank_prediction_id_rows,
        missing_prediction_ids=missing_prediction_ids,
        extra_prediction_ids=extra_prediction_ids,
    )
    metric_pass = not metric_blockers
    summary = {
        "decision": "pass" if metric_pass and not artifact_blockers else "fail",
        "metric_decision": "pass" if metric_pass else "fail",
        "artifact_decision": "pass" if not artifact_blockers else "fail",
        "metric_blockers": metric_blockers,
        "artifact_blockers": artifact_blockers,
        "score": score,
        "rows": counters["rows"],
        "correct": counters["correct"],
        "incorrect": counters["incorrect"],
        "missing_prediction_rows": counters["missing_prediction_row"],
        "missing_prediction_ids_count": len(missing_prediction_ids),
        "extra_prediction_ids_count": len(extra_prediction_ids),
        "duplicate_solution_ids_count": len(duplicate_solution_ids),
        "duplicate_prediction_ids_count": len(duplicate_prediction_ids),
        "blank_solution_id_rows": blank_solution_id_rows,
        "blank_prediction_id_rows": blank_prediction_id_rows,
        "missing_prediction_ids_sample": missing_prediction_ids[:20],
        "extra_prediction_ids_sample": extra_prediction_ids[:20],
        "duplicate_solution_ids_sample": duplicate_solution_ids[:20],
        "duplicate_prediction_ids_sample": duplicate_prediction_ids[:20],
        "no_box_fallback_rows": counters["no_box_fallback"],
        "not_found_rows": counters["not_found"],
        "by_answer_bucket": summarize_group(by_bucket),
        "by_family": summarize_group(by_family),
        "by_status": summarize_group(by_status),
        "by_extract_source": summarize_group(by_source),
        "failure_counts": dict(failure_counts),
        "official_visible_score_defaults": OFFICIAL_VISIBLE_SCORE_DEFAULTS,
        "official_prompt_suffix": OFFICIAL_PROMPT_SUFFIX,
        "adapter_audit": adapter_audit,
        "inputs": {
            "solution_csv": str(args.solution_csv),
            "predictions": str(args.predictions),
            "family_map": str(args.family_map) if args.family_map else "",
            "id_col": args.id_col,
            "answer_col": args.answer_col,
            "raw_output_col": args.raw_output_col or "",
            "prediction_col": args.prediction_col or "",
        },
        "limitations": [
            "This pre-score is exact for extraction and verification on supplied predictions.",
            "It cannot know the hidden Kaggle test set; calibration depends on the validation set used.",
            "It does not generate model outputs; use an eval/vLLM job to produce raw_output first.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.output_dir / "official_prescore_details.csv"
    summary_json = args.output_dir / "official_prescore_summary.json"
    write_csv(
        detail_csv,
        details,
        [
            "id",
            "expected",
            "extracted",
            "correct",
            "answer_bucket",
            "family",
            "status",
            "extract_source",
            "failure_reason",
            "prediction_source_col",
            "raw_output_chars",
        ],
    )
    summary["outputs"] = {"details_csv": str(detail_csv), "summary_json": str(summary_json)}
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize_group(groups: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, counter in sorted(groups.items()):
        rows = counter["rows"]
        correct = counter["correct"]
        output[name] = {
            "rows": rows,
            "correct": correct,
            "incorrect": rows - correct,
            "accuracy": correct / rows if rows else 0.0,
        }
    return output


def metric_validation_blockers(
    *,
    args: argparse.Namespace,
    score: float,
    counters: Counter[str],
    by_family: dict[str, Counter[str]],
    duplicate_solution_ids: list[str],
    duplicate_prediction_ids: list[str],
    blank_solution_id_rows: int,
    blank_prediction_id_rows: int,
    missing_prediction_ids: list[str],
    extra_prediction_ids: list[str],
) -> list[str]:
    blockers: list[str] = []
    if blank_solution_id_rows:
        blockers.append(f"blank_solution_id_rows:{blank_solution_id_rows}")
    if blank_prediction_id_rows:
        blockers.append(f"blank_prediction_id_rows:{blank_prediction_id_rows}")
    if duplicate_solution_ids:
        blockers.append(f"duplicate_solution_ids:{len(duplicate_solution_ids)}")
    if duplicate_prediction_ids:
        blockers.append(f"duplicate_prediction_ids:{len(duplicate_prediction_ids)}")
    if args.strict_row_match:
        if missing_prediction_ids:
            blockers.append(f"missing_prediction_ids:{len(missing_prediction_ids)}")
        if extra_prediction_ids:
            blockers.append(f"extra_prediction_ids:{len(extra_prediction_ids)}")
    if args.min_score is not None and score < float(args.min_score):
        blockers.append(f"score_below_min:{score:.12f}<{float(args.min_score):.12f}")
    if args.require_score_above_baseline:
        if args.baseline_score is None:
            blockers.append("baseline_score_required")
        elif score <= float(args.baseline_score):
            blockers.append(f"score_not_above_baseline:{score:.12f}<={float(args.baseline_score):.12f}")
    if args.min_score is None and not args.require_score_above_baseline:
        if counters["correct"] != counters["rows"]:
            blockers.append(f"not_all_rows_correct:{counters['correct']}/{counters['rows']}")

    min_family_correct = parse_threshold_specs(args.min_family_correct, "--min-family-correct")
    min_family_accuracy = parse_threshold_specs(args.min_family_accuracy, "--min-family-accuracy")
    for family, min_correct in min_family_correct.items():
        observed = by_family.get(family, Counter())["correct"]
        if observed < int(min_correct):
            blockers.append(f"family_correct_below_min:{family}:{observed}<{int(min_correct)}")
    for family, min_accuracy in min_family_accuracy.items():
        group = by_family.get(family, Counter())
        rows = group["rows"]
        accuracy = group["correct"] / rows if rows else 0.0
        if accuracy < float(min_accuracy):
            blockers.append(f"family_accuracy_below_min:{family}:{accuracy:.12f}<{float(min_accuracy):.12f}")
    return blockers


def artifact_validation_blockers(args: argparse.Namespace, audit: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not args.strict_submission_artifact and not any(
        [
            args.expected_zip_sha256,
            args.expected_adapter_config_sha256,
            args.expected_adapter_model_sha256,
        ]
    ):
        return blockers
    if not audit:
        blockers.append("adapter_artifact_missing")
        return blockers
    if not audit.get("exists"):
        blockers.append("adapter_artifact_path_not_found")
        return blockers
    if args.expected_zip_sha256 and audit.get("zip_sha256") != args.expected_zip_sha256:
        blockers.append(
            f"zip_sha256_mismatch:{audit.get('zip_sha256')}!={args.expected_zip_sha256}"
        )
    if args.expected_adapter_config_sha256 and audit.get("adapter_config_sha256") != args.expected_adapter_config_sha256:
        blockers.append(
            "adapter_config_sha256_mismatch:"
            f"{audit.get('adapter_config_sha256')}!={args.expected_adapter_config_sha256}"
        )
    if args.expected_adapter_model_sha256 and audit.get("adapter_model_sha256") != args.expected_adapter_model_sha256:
        blockers.append(
            "adapter_model_sha256_mismatch:"
            f"{audit.get('adapter_model_sha256')}!={args.expected_adapter_model_sha256}"
        )
    if args.strict_submission_artifact:
        missing = audit.get("required_entries_missing") or []
        if missing:
            blockers.append("required_entries_missing:" + ",".join(missing))
        if audit.get("kind") == "submission_zip":
            if audit.get("adapter_config_count") != 1:
                blockers.append(f"adapter_config_count_not_one:{audit.get('adapter_config_count')}")
            if audit.get("adapter_model_count") != 1:
                blockers.append(f"adapter_model_count_not_one:{audit.get('adapter_model_count')}")
            if audit.get("model_in_same_dir_as_config") is False:
                blockers.append("adapter_model_not_in_same_dir_as_config")
        cfg = audit.get("adapter_config") or {}
        if cfg.get("r", 999) > 32:
            blockers.append(f"lora_rank_gt_32:{cfg.get('r')}")
        if cfg.get("peft_type") and str(cfg.get("peft_type")).upper() != "LORA":
            blockers.append(f"peft_type_not_lora:{cfg.get('peft_type')}")
        base = str(cfg.get("base_model_name_or_path") or "")
        if base and "Nemotron-3-Nano-30B-A3B-BF16" not in base:
            blockers.append(f"unexpected_base_model:{base}")
    return blockers


def run_self_test() -> None:
    tests = [
        ("boxed_last", extract_final_answer(r"a \boxed{wrong} b \boxed{right}"), "right"),
        ("boxed_nested", extract_final_answer(r"answer \boxed{\frac{1}{2}}"), r"\frac{1}{2}"),
        ("boxed_literal_brace", extract_final_answer(r"answer \boxed{}52}"), "}52"),
        ("phrase_last", extract_final_answer("Final answer: 1\nFinal answer: 2"), "2"),
        ("number_last", extract_final_answer("try 10 then 42"), "42"),
        ("last_line_text", extract_final_answer("reasoning\nhello world"), "hello world"),
        ("none", extract_final_answer(None), "NOT_FOUND"),
        ("binary_strict_leading_zero_fail", verify("00011011", "11011"), False),
        ("numeric_leading_zero_pass", verify("0271", "271"), True),
        ("numeric_tolerance_pass", verify("24.64", "24.6401"), True),
        ("numeric_1pct_boundary_fail", verify("100", "101"), False),
        ("roman_case_pass", verify("XLVII", "xlvii"), True),
        ("binary_like_decimal_fail", verify("10", "10.0"), False),
    ]
    failed = [(name, got, expected) for name, got, expected in tests if got != expected]
    if failed:
        for name, got, expected in failed:
            print(f"SELFTEST_FAIL {name}: got={got!r} expected={expected!r}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"self_test": "pass", "tests": len(tests)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-csv", type=Path, help="CSV with id and answer columns.")
    parser.add_argument("--predictions", type=Path, help="CSV/JSONL with id and raw_output or prediction.")
    parser.add_argument("--family-map", type=Path, help="Optional CSV/JSONL with id + family/category + status.")
    parser.add_argument("--adapter-artifact", type=Path, help="Optional submission.zip or adapter directory to hash/audit.")
    parser.add_argument("--strict-submission-artifact", action="store_true")
    parser.add_argument("--expected-zip-sha256")
    parser.add_argument("--expected-adapter-config-sha256")
    parser.add_argument("--expected-adapter-model-sha256")
    parser.add_argument("--strict-row-match", action="store_true", help="Fail if predictions miss or add ids.")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum acceptable official metric score.")
    parser.add_argument("--baseline-score", type=float, default=None, help="Baseline score for improvement gates.")
    parser.add_argument("--require-score-above-baseline", action="store_true")
    parser.add_argument(
        "--min-family-correct",
        action="append",
        default=[],
        help="Repeatable family=count blocker, e.g. bit_manipulation=136.",
    )
    parser.add_argument(
        "--min-family-accuracy",
        action="append",
        default=[],
        help="Repeatable family=accuracy blocker, e.g. gravity=1.0.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/official_metric_prescore"))
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--raw-output-col", default=None)
    parser.add_argument("--prediction-col", default=None)
    parser.add_argument("--allow-auto-column-detect", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.raw_output_col and not args.prediction_col and not args.allow_auto_column_detect:
        raise SystemExit(
            "pass --raw-output-col for raw model text or --prediction-col for already extracted answers "
            "(or --allow-auto-column-detect for legacy probing)"
        )
    if not args.solution_csv or not args.predictions:
        raise SystemExit("--solution-csv and --predictions are required unless --self-test is used")
    summary = score_predictions(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

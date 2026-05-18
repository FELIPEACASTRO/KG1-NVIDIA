#!/usr/bin/env python3
"""V586 CPU plateau diagnostics for KG1 weak-family adapters.

This gate compares a trusted row-level baseline CSV against one or more
candidate weak-eval CSVs using only submit-safe evidence:

raw_output -> label-free extract_final_answer -> verify_answer.

It does not train and it does not submit.  Its purpose is to stop blind GPU
loops by separating:

* true adapter gains from stale ``prediction`` or expected-aware columns;
* protected-row backfire from missing required gains;
* runaway/truncated decoding from a model actually preferring a wrong answer;
* answer-present-but-not-final cases from genuine reasoning misses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import (  # noqa: E402
    canonical_family,
    classify_puzzle,
    extract_boxed_answers,
    extract_final_answer,
    verify_answer,
)


DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "artifacts"
    / "v516_label_free_weak_baseline"
    / "v516_label_free_v290_checkpoint6_baseline.csv"
)
DEFAULT_CANDIDATE = [
    "v582_checkpoint2="
    + str(REPO_ROOT / "artifacts" / "v582_hf_h200_launch" / "v582_checkpoint2_predictions.csv")
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v586_plateau_row_diagnostics"
DEFAULT_PROTECTED = [
    "8740ed31=01101000",
    "59bee375=10010101",
    "55d834d1=00111111",
]
WINDOW_CHARS = (512, 1024, 2048, 4096)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def windows_long_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if sys.platform != "win32":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text.lstrip("\\"))
    return Path("\\\\?\\" + text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with windows_long_path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with windows_long_path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    windows_long_path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with windows_long_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def truthy(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y", "on"}


def as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty candidate name in {raw!r}")
    return name, Path(value.strip())


def parse_protected(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"protected spec must be ID=ANSWER, got {item!r}")
            rid, answer = item.split("=", 1)
            out[rid.strip()] = answer.strip()
    return out


def infer_family(row: dict[str, str]) -> str:
    raw = row.get("family") or row.get("type") or row.get("task_type") or row.get("pred_type") or ""
    family = canonical_family(raw)
    if family and family != "unknown":
        return family
    return canonical_family(classify_puzzle(str(row.get("prompt") or row.get("generated_prompt") or "")))


def stable_row_contract(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{hashlib.sha256(str(row.get('prompt','')).encode('utf-8', errors='replace')).hexdigest()}"
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def normalize_numeric_token(value: str) -> str:
    return value.replace(",", "").strip()


def answer_anywhere(raw_output: str, answer: str) -> tuple[bool, str, str]:
    text = str(raw_output or "")
    expected = str(answer or "").strip()
    if not text or not expected:
        return False, "", ""

    boxed = [item.strip() for item in extract_boxed_answers(text) if item.strip()]
    for payload in boxed:
        if verify_answer(expected, payload):
            return True, "boxed_payload", payload

    if re.fullmatch(r"[01]+", expected):
        pattern = rf"(?<![01]){re.escape(expected)}(?![01])"
        match = re.search(pattern, text)
        if match:
            return True, "binary_substring", match.group(0)
        return False, "", ""

    try:
        expected_float = float(expected)
    except Exception:
        expected_float = math.nan
    if math.isfinite(expected_float):
        for number in re.findall(r"-?\d+(?:\.\d+)?", text):
            if verify_answer(expected, normalize_numeric_token(number)):
                return True, "numeric_token", number
        return False, "", ""

    lowered_text = text.lower()
    lowered_expected = expected.lower()
    if lowered_expected in lowered_text:
        return True, "case_insensitive_substring", expected
    return False, "", ""


def final_position(raw_output: str, extracted: str) -> tuple[int, float]:
    text = str(raw_output or "")
    if not text:
        return -1, -1.0
    idx = text.rfind(str(extracted or ""))
    if idx < 0:
        return -1, -1.0
    return idx, idx / max(1, len(text))


def source_correct(row: dict[str, str], extracted_correct: bool) -> tuple[bool, bool, bool]:
    stored = row.get("stored_correct")
    correct = row.get("correct")
    label_free = row.get("label_free_correct")
    stored_ok = truthy(stored) if stored not in (None, "") else extracted_correct
    correct_ok = truthy(correct) if correct not in (None, "") else extracted_correct
    label_free_ok = truthy(label_free) if label_free not in (None, "") else extracted_correct
    return stored_ok, correct_ok, label_free_ok


RUN_ROW_COLUMNS = [
    "run_name",
    "id",
    "family",
    "answer",
    "extracted",
    "extracted_correct",
    "stored_prediction",
    "stored_prediction_matches_extracted",
    "correct_column_matches_extracted",
    "label_free_column_matches_extracted",
    "stored_correct_column_matches_extracted",
    "answer_anywhere",
    "answer_anywhere_source",
    "answer_anywhere_match",
    "final_answer_pos",
    "final_answer_pos_ratio",
    "boxed_count",
    "first_boxed",
    "last_boxed",
    "raw_chars",
    "completion_tokens",
    "finish_reason",
    "truncated",
    "early_512_correct",
    "early_1024_correct",
    "early_2048_correct",
    "early_4096_correct",
    "protected_role",
    "row_warnings",
    "row_blockers",
]


def normalize_run(name: str, path: Path, protected: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    ids = [str(row.get("id", "")).strip() for row in rows]
    duplicate_ids = sorted(rid for rid, count in Counter(ids).items() if rid and count > 1)
    normalized: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for row in rows:
        rid = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt") or row.get("generated_prompt") or "")
        answer = str(row.get("answer", "")).strip()
        family = infer_family(row)
        raw_output = str(row.get("raw_output", ""))
        extracted = extract_final_answer(raw_output) if raw_output.strip() else "NOT_FOUND"
        extracted_correct = verify_answer(answer, extracted)
        stored_prediction = str(row.get("prediction", "")).strip()
        stored_correct, correct_column, label_free_column = source_correct(row, extracted_correct)
        stored_prediction_matches = stored_prediction == extracted
        correct_column_matches = correct_column == extracted_correct
        label_free_matches = label_free_column == extracted_correct
        stored_correct_matches = stored_correct == extracted_correct
        truncated = truthy(row.get("truncated", row.get("truncated_bool", ""))) or str(row.get("finish_reason", "")).strip().lower() == "length"
        completion_tokens = as_int(row.get("completion_tokens", "0"))
        boxed = [item.strip() for item in extract_boxed_answers(raw_output) if item.strip()]
        any_ok, any_source, any_match = answer_anywhere(raw_output, answer)
        pos, ratio = final_position(raw_output, extracted)

        row_blockers: list[str] = []
        row_warnings: list[str] = []
        if not rid:
            row_blockers.append("missing_id")
        if not raw_output.strip():
            row_blockers.append("missing_raw_output")
        if raw_output.strip() and not stored_prediction_matches:
            row_blockers.append("stored_prediction_not_raw_extraction")
        if not correct_column_matches:
            row_blockers.append("correct_column_not_raw_extraction")
        if "label_free_correct" in row and not label_free_matches:
            row_warnings.append("label_free_column_not_raw_extraction")
        if "stored_correct" in row and not stored_correct_matches:
            row_warnings.append("stored_correct_column_not_raw_extraction")
        if truncated:
            row_blockers.append("truncated_row")

        protected_role = ""
        if rid in protected:
            if extracted_correct:
                protected_role = "protected_ok"
            else:
                protected_role = "protected_failed"
                row_blockers.append("protected_row_answer_mismatch")

        for blocker in row_blockers:
            blocker_counts[blocker] += 1
        for warning in row_warnings:
            warning_counts[warning] += 1

        early_values: dict[str, str] = {}
        for window in WINDOW_CHARS:
            early_extracted = extract_final_answer(raw_output[:window]) if raw_output.strip() else "NOT_FOUND"
            early_values[f"early_{window}_correct"] = bool_text(verify_answer(answer, early_extracted))

        normalized.append(
            {
                "run_name": name,
                "id": rid,
                "prompt": prompt,
                "family": family,
                "answer": answer,
                "raw_output": raw_output,
                "extracted": extracted,
                "extracted_correct_bool": extracted_correct,
                "extracted_correct": bool_text(extracted_correct),
                "stored_prediction": stored_prediction,
                "stored_prediction_matches_extracted": bool_text(stored_prediction_matches),
                "correct_column_matches_extracted": bool_text(correct_column_matches),
                "label_free_column_matches_extracted": bool_text(label_free_matches),
                "stored_correct_column_matches_extracted": bool_text(stored_correct_matches),
                "answer_anywhere": bool_text(any_ok),
                "answer_anywhere_bool": any_ok,
                "answer_anywhere_source": any_source,
                "answer_anywhere_match": any_match,
                "final_answer_pos": pos,
                "final_answer_pos_ratio": round(ratio, 6),
                "boxed_count": len(boxed),
                "first_boxed": boxed[0] if boxed else "",
                "last_boxed": boxed[-1] if boxed else "",
                "raw_chars": len(raw_output),
                "completion_tokens": completion_tokens,
                "finish_reason": str(row.get("finish_reason", "")),
                "truncated_bool": truncated,
                "truncated": bool_text(truncated),
                **early_values,
                "protected_role": protected_role,
                "row_warnings": ";".join(row_warnings),
                "row_blockers": ";".join(row_blockers),
            }
        )

    family_summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["family"])].append(row)
    for family, group in sorted(grouped.items()):
        tokens = [int(row["completion_tokens"]) for row in group]
        family_summary[family] = {
            "rows": len(group),
            "correct": sum(int(row["extracted_correct_bool"]) for row in group),
            "truncated": sum(int(row["truncated_bool"]) for row in group),
            "answer_anywhere_wrong_final": sum(
                int((not row["extracted_correct_bool"]) and bool(row["answer_anywhere_bool"])) for row in group
            ),
            "completion_tokens_mean": sum(tokens) / len(tokens) if tokens else 0.0,
            "completion_tokens_max": max(tokens) if tokens else 0,
        }
    all_tokens = [int(row["completion_tokens"]) for row in normalized]
    summary = {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(normalized),
        "duplicate_ids": duplicate_ids,
        "row_contract_sha256": stable_row_contract(normalized),
        "total_correct": sum(int(row["extracted_correct_bool"]) for row in normalized),
        "accuracy": (
            sum(int(row["extracted_correct_bool"]) for row in normalized) / len(normalized)
            if normalized
            else 0.0
        ),
        "truncated": sum(int(row["truncated_bool"]) for row in normalized),
        "answer_anywhere_wrong_final": sum(
            int((not row["extracted_correct_bool"]) and bool(row["answer_anywhere_bool"])) for row in normalized
        ),
        "completion_tokens_total": sum(all_tokens),
        "completion_tokens_mean": sum(all_tokens) / len(all_tokens) if all_tokens else 0.0,
        "completion_tokens_max": max(all_tokens) if all_tokens else 0,
        "per_family": family_summary,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }
    return normalized, summary


DELTA_COLUMNS = [
    "candidate",
    "id",
    "family",
    "answer",
    "baseline_extracted",
    "candidate_extracted",
    "baseline_correct",
    "candidate_correct",
    "delta_type",
    "protected_delta",
    "baseline_answer_anywhere",
    "candidate_answer_anywhere",
    "candidate_anywhere_wrong_final",
    "candidate_completion_tokens",
    "candidate_truncated",
    "candidate_finish_reason",
    "candidate_boxed_count",
    "candidate_final_answer_pos_ratio",
    "candidate_row_blockers",
]


def compare_runs(
    baseline: list[dict[str, Any]],
    candidate_name: str,
    candidate: list[dict[str, Any]],
    protected: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_by_id = {row["id"]: row for row in baseline}
    cand_by_id = {row["id"]: row for row in candidate}
    rows: list[dict[str, Any]] = []
    delta_counts: Counter[str] = Counter()
    protected_counts: Counter[str] = Counter()
    missing_in_candidate = sorted(set(base_by_id) - set(cand_by_id))
    extra_in_candidate = sorted(set(cand_by_id) - set(base_by_id))
    for rid, base in sorted(base_by_id.items()):
        cand = cand_by_id.get(rid)
        if cand is None:
            continue
        base_correct = bool(base["extracted_correct_bool"])
        cand_correct = bool(cand["extracted_correct_bool"])
        if base_correct and cand_correct:
            delta_type = "both_correct"
        elif base_correct and not cand_correct:
            delta_type = "flip_wrong"
        elif not base_correct and cand_correct:
            delta_type = "flip_correct"
        elif str(base["extracted"]) == str(cand["extracted"]):
            delta_type = "still_wrong_same"
        else:
            delta_type = "still_wrong_changed"
        delta_counts[delta_type] += 1

        protected_delta = ""
        if rid in protected:
            if base_correct and not cand_correct:
                protected_delta = "protected_id_backfire"
            elif not base_correct and cand_correct:
                protected_delta = "protected_id_gain"
            elif not base_correct and not cand_correct:
                protected_delta = "protected_id_missing_required_gain"
            else:
                protected_delta = "protected_id_preserved"
            protected_counts[protected_delta] += 1

        row = {
            "candidate": candidate_name,
            "id": rid,
            "family": base["family"],
            "answer": base["answer"],
            "baseline_extracted": base["extracted"],
            "candidate_extracted": cand["extracted"],
            "baseline_correct": bool_text(base_correct),
            "candidate_correct": bool_text(cand_correct),
            "delta_type": delta_type,
            "protected_delta": protected_delta,
            "baseline_answer_anywhere": base["answer_anywhere"],
            "candidate_answer_anywhere": cand["answer_anywhere"],
            "candidate_anywhere_wrong_final": bool_text((not cand_correct) and bool(cand["answer_anywhere_bool"])),
            "candidate_completion_tokens": cand["completion_tokens"],
            "candidate_truncated": cand["truncated"],
            "candidate_finish_reason": cand["finish_reason"],
            "candidate_boxed_count": cand["boxed_count"],
            "candidate_final_answer_pos_ratio": cand["final_answer_pos_ratio"],
            "candidate_row_blockers": cand["row_blockers"],
        }
        rows.append(row)

    summary = {
        "candidate": candidate_name,
        "rows_compared": len(rows),
        "missing_in_candidate": len(missing_in_candidate),
        "extra_in_candidate": len(extra_in_candidate),
        "delta_counts": dict(sorted(delta_counts.items())),
        "protected_delta_counts": dict(sorted(protected_counts.items())),
        "flip_correct_ids": [row["id"] for row in rows if row["delta_type"] == "flip_correct"],
        "flip_wrong_ids": [row["id"] for row in rows if row["delta_type"] == "flip_wrong"],
        "answer_anywhere_wrong_final_ids": [
            row["id"] for row in rows if row["candidate_anywhere_wrong_final"] == "true"
        ],
    }
    return rows, summary


def load_eval_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {"missing": str(path)}
    payload = json.loads(windows_long_path(path).read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "config": config,
        "config_sha256": sha256_json(config),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V586 Plateau Row Diagnostics",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- decision: `{summary['decision']}`",
        f"- baseline: `{summary['baseline']['name']}` `{summary['baseline']['total_correct']}/315`",
        "",
        "## Runs",
        "",
        "| run | total | bit | equation | truncated | token mean | token max | anywhere wrong-final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in [summary["baseline"], *summary["candidates"]]:
        per = run.get("per_family", {})
        bit = per.get("bit_manipulation", {}).get("correct", "")
        eq = per.get("equation_transform", {}).get("correct", "")
        lines.append(
            f"| `{run['name']}` | {run['total_correct']} | {bit} | {eq} | {run['truncated']} | "
            f"{run['completion_tokens_mean']:.2f} | {run['completion_tokens_max']} | "
            f"{run['answer_anywhere_wrong_final']} |"
        )
    lines.extend(["", "## Candidate Deltas", ""])
    for comp in summary["comparisons"]:
        lines.append(f"### `{comp['candidate']}`")
        lines.append("")
        lines.append(f"- delta_counts: `{json.dumps(comp['delta_counts'], sort_keys=True)}`")
        lines.append(f"- protected_delta_counts: `{json.dumps(comp['protected_delta_counts'], sort_keys=True)}`")
        lines.append(f"- flip_correct_ids: `{', '.join(comp['flip_correct_ids'][:20])}`")
        lines.append(f"- flip_wrong_ids: `{', '.join(comp['flip_wrong_ids'][:20])}`")
        lines.append(f"- answer_anywhere_wrong_final_ids: `{', '.join(comp['answer_anywhere_wrong_final_ids'][:20])}`")
        lines.append("")
    lines.extend(["## Blockers", ""])
    if summary["blockers"]:
        for blocker in summary["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Do not launch new paid training while any candidate has protected backfire, truncation, token runaway, or no true total gain.",
            "- If `answer_anywhere_wrong_final` is nonzero, fix decoding/stop/final-answer format before training.",
            "- If `flip_wrong` appears on protected rows, test adapter-scale attenuation before changing data.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V586 PLATEAU ROW DIAGNOSTICS START ===", flush=True)
    print("baseline_csv =", args.baseline_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = parse_protected(args.protected_id_answer)
    baseline_rows, baseline_summary = normalize_run("baseline", args.baseline_csv, protected)
    candidates: list[dict[str, Any]] = []
    run_rows = list(baseline_rows)
    all_delta_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    blockers: list[str] = []

    for spec in args.candidate_csv:
        name, path = parse_named_path(spec)
        rows, summary = normalize_run(name, path, protected)
        candidates.append(summary)
        run_rows.extend(rows)
        delta_rows, comp = compare_runs(baseline_rows, name, rows, protected)
        all_delta_rows.extend(delta_rows)
        comparisons.append(comp)
        if summary["rows"] != baseline_summary["rows"]:
            blockers.append(f"{name}:row_count_mismatch")
        if summary["truncated"] > 0:
            blockers.append(f"{name}:truncation_nonzero:{summary['truncated']}")
        if summary["completion_tokens_mean"] > args.max_completion_tokens_mean:
            blockers.append(f"{name}:completion_tokens_mean_gt_{args.max_completion_tokens_mean}:{summary['completion_tokens_mean']:.3f}")
        if summary["completion_tokens_max"] > args.max_completion_tokens_max:
            blockers.append(f"{name}:completion_tokens_max_gt_{args.max_completion_tokens_max}:{summary['completion_tokens_max']}")
        if comp["protected_delta_counts"].get("protected_id_backfire", 0):
            blockers.append(f"{name}:protected_id_backfire:{comp['protected_delta_counts']['protected_id_backfire']}")
        if summary["total_correct"] <= baseline_summary["total_correct"]:
            blockers.append(f"{name}:no_total_gain_vs_baseline:{summary['total_correct']}<= {baseline_summary['total_correct']}")
        base_bit = baseline_summary["per_family"].get("bit_manipulation", {}).get("correct", 0)
        cand_bit = summary["per_family"].get("bit_manipulation", {}).get("correct", 0)
        if cand_bit < base_bit:
            blockers.append(f"{name}:bit_regression_vs_baseline:{cand_bit}< {base_bit}")

    config_objects: dict[str, Any] = {}
    if args.baseline_report_json:
        config_objects["baseline"] = load_eval_config(args.baseline_report_json)
    for spec in args.candidate_report_json:
        name, path = parse_named_path(spec)
        config_objects[name] = load_eval_config(path)

    details_csv = args.output_dir / f"{args.label}_run_rows.csv"
    deltas_csv = args.output_dir / f"{args.label}_row_deltas.csv"
    summary_json = args.output_dir / f"{args.label}_summary.json"
    markdown_path = args.output_dir / f"KG1_{args.label.upper()}_SUMMARY.md"

    summary = {
        "schema_version": "kg1_v586_plateau_row_diagnostics_v1",
        "generated_at_utc": utc_now(),
        "decision": "blocked" if blockers else "passed",
        "protected_id_answers": protected,
        "thresholds": {
            "max_completion_tokens_mean": args.max_completion_tokens_mean,
            "max_completion_tokens_max": args.max_completion_tokens_max,
        },
        "baseline": baseline_summary,
        "candidates": candidates,
        "comparisons": comparisons,
        "generation_configs": config_objects,
        "blockers": blockers,
        "outputs": {
            "run_rows_csv": str(details_csv),
            "row_deltas_csv": str(deltas_csv),
            "summary_json": str(summary_json),
            "markdown": str(markdown_path),
        },
    }
    write_csv(details_csv, run_rows, RUN_ROW_COLUMNS)
    write_csv(deltas_csv, all_delta_rows, DELTA_COLUMNS)
    write_json(summary_json, summary)
    windows_long_path(markdown_path).write_text(build_markdown(summary), encoding="utf-8")
    print("decision =", summary["decision"], flush=True)
    print("baseline_total =", baseline_summary["total_correct"], flush=True)
    for cand in candidates:
        print(
            "candidate_total =",
            cand["name"],
            cand["total_correct"],
            "bit =",
            cand["per_family"].get("bit_manipulation", {}).get("correct", ""),
            "equation =",
            cand["per_family"].get("equation_transform", {}).get("correct", ""),
            "truncated =",
            cand["truncated"],
            flush=True,
        )
    print("blockers =", json.dumps(blockers, sort_keys=True), flush=True)
    print("summary_json =", summary_json, flush=True)
    print("markdown =", markdown_path, flush=True)
    print("=== V586 PLATEAU ROW DIAGNOSTICS END ===", flush=True)
    if args.fail_on_blocked and blockers:
        raise SystemExit(2)
    return summary


def self_test() -> None:
    raw = "first \\boxed{00000000}\nFinal answer: \\boxed{01101000}"
    ok, source, match = answer_anywhere(raw, "01101000")
    if not ok or source != "boxed_payload" or match != "01101000":
        raise AssertionError((ok, source, match))
    wrong_final_raw = "The answer may be 01101000, but final answer: \\boxed{01111000}"
    ok, source, match = answer_anywhere(wrong_final_raw, "01101000")
    if not ok or source != "binary_substring":
        raise AssertionError((ok, source, match))
    if extract_final_answer(wrong_final_raw) != "01111000":
        raise AssertionError("expected final answer to be wrong for self-test")
    print("analyze_v586_plateau_row_diagnostics_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--candidate-csv", action="append", default=[], help="name=path row-level candidate CSV")
    parser.add_argument("--baseline-report-json", type=Path, default=None)
    parser.add_argument("--candidate-report-json", action="append", default=[], help="name=path eval report JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v586_plateau_row_diagnostics")
    parser.add_argument("--protected-id-answer", action="append", default=list(DEFAULT_PROTECTED))
    parser.add_argument("--max-completion-tokens-mean", type=float, default=512.0)
    parser.add_argument("--max-completion-tokens-max", type=int, default=2048)
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.candidate_csv and not args.self_test:
        args.candidate_csv = list(DEFAULT_CANDIDATE)
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

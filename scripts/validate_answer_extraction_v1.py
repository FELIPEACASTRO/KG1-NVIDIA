#!/usr/bin/env python3
"""V540 CPU-only audit for answer extraction and weak ACC parity.

This gate exists because KG1 promotion decisions must be based on the same
label-free extraction path used during inference: raw model text -> public
extractor -> verify_answer.  It refuses to treat stored ``prediction`` or
summary-only weak results as enough evidence for a new paid job.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import (  # noqa: E402
    canonical_family,
    classify_puzzle,
    extract_final_answer,
    verify_answer,
)


EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
EXPECTED_WEAK_ROWS = 315
EXPECTED_WEAK_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}
DEFAULT_MAX_TOKENS = 7680
DEFAULT_MAX_TOKEN_HEADROOM_RATIO = 0.90
DEFAULT_PROTECTED_ID_ANSWER = ["8740ed31=01101000", "59bee375=10010101"]

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
    / "v516_label_free_weak_baseline"
    / "v516_label_free_v290_checkpoint6_baseline.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "v540_answer_extraction_audit"


DETAIL_COLUMNS = [
    "run_name",
    "id",
    "family",
    "answer",
    "prompt_sha256",
    "run_prompt_sha256",
    "prompt_hash_match",
    "stored_prediction",
    "extracted_answer",
    "stored_prediction_matches_extracted",
    "stored_correct_recomputed",
    "extracted_correct",
    "stored_correct_column",
    "correct_column",
    "stored_correct_column_matches",
    "correct_column_matches_extracted",
    "raw_output_present",
    "raw_output_chars",
    "boxed_count",
    "answer_span_start",
    "answer_span_end",
    "token_count",
    "token_counter",
    "token_headroom_ratio",
    "truncated",
    "protected_row",
    "warnings",
    "blockers",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def windows_long_path(path: Path) -> Path:
    """Return a Windows long-path-safe absolute path without changing POSIX behavior."""
    resolved = path.resolve(strict=False)
    if sys.platform != "win32":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text.lstrip("\\"))
    return Path("\\\\?\\" + text)


def path_exists(path: Path) -> bool:
    return windows_long_path(path).exists()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with windows_long_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with windows_long_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def truthy(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y", "on"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def infer_family(row: dict[str, str]) -> str:
    raw = str(row.get("family") or row.get("type") or row.get("task_type") or "").strip()
    if raw:
        return canonical_family(raw)
    return canonical_family(classify_puzzle(str(row.get("prompt", ""))))


def row_contract(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    return sha256_text(payload)


def load_weak_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        family = infer_family(row)
        normalized.append(
            {
                "id": str(row.get("id", "")).strip(),
                "prompt": prompt,
                "answer": str(row.get("answer", "")).strip(),
                "family": family,
                "prompt_sha256": sha256_text(prompt),
            }
        )
    blockers: list[str] = []
    if len(normalized) != EXPECTED_WEAK_ROWS:
        blockers.append(f"weak_rows_expected_{EXPECTED_WEAK_ROWS}_got_{len(normalized)}")
    ids = [str(row["id"]) for row in normalized]
    if any(not rid for rid in ids):
        blockers.append("weak_empty_id")
    duplicate_ids = sorted([rid for rid, count in Counter(ids).items() if count > 1])
    if duplicate_ids:
        blockers.append("weak_duplicate_ids:" + ",".join(duplicate_ids[:10]))
    if any(not str(row["answer"]).strip() for row in normalized):
        blockers.append("weak_empty_answer")
    counts = {str(k): int(v) for k, v in Counter(str(row["family"]) for row in normalized).items()}
    counts = dict(sorted(counts.items()))
    if counts != EXPECTED_WEAK_COUNTS:
        blockers.append(f"weak_family_counts_mismatch:{counts}")
    observed_contract = row_contract(normalized)
    if observed_contract != EXPECTED_SHARED_ROW_CONTRACT_SHA256:
        blockers.append(f"weak_contract_mismatch:{observed_contract}")
    return (
        normalized,
        {str(row["id"]): row for row in normalized},
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(normalized),
            "family_counts": counts,
            "observed_shared_row_contract_sha256": observed_contract,
            "expected_shared_row_contract_sha256": EXPECTED_SHARED_ROW_CONTRACT_SHA256,
            "blockers": blockers,
            "passed": not blockers,
        },
    )


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty name in spec {raw!r}")
    return name, Path(path.strip())


def parse_protected_pairs(values: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"protected id answer must be id=answer, got {item!r}")
            rid, answer = item.split("=", 1)
            pairs[rid.strip()] = answer.strip()
    return pairs


def find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    marker = r"\boxed{"
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        marker_pos = text.find(marker, cursor)
        if marker_pos == -1:
            break
        start = marker_pos + len(marker)
        minimal_end: int | None = None
        index = start
        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == "}":
                minimal_end = index
                break
            index += 1
        depth = 1
        index = start
        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth == 0:
            end = index
            while end + 1 < len(text) and text[end + 1] == "}":
                end += 1
            spans.append((start, end, text[start:end]))
            cursor = end + 1
        elif minimal_end is not None:
            spans.append((start, minimal_end, text[start:minimal_end]))
            cursor = minimal_end + 1
        else:
            spans.append((start, len(text), text[start:]))
            break
    return spans


def answer_span(text: str, extracted: str) -> tuple[int, int]:
    spans = [span for span in find_boxed_spans(text) if span[2].strip()]
    if spans:
        start, end, _payload = spans[-1]
        return start, end
    if extracted and extracted != "NOT_FOUND":
        pos = text.rfind(extracted)
        if pos >= 0:
            return pos, pos + len(extracted)
    return -1, -1


def token_count_approx(text: str) -> int:
    # Conservative deterministic proxy for CPU gates. Exact tokenizer parity is
    # enforced later inside the HF runtime before any paid train/eval.
    return len(re.findall(r"\S+", text))


def audit_row_level_run(
    *,
    name: str,
    path: Path,
    weak_rows: list[dict[str, Any]],
    weak_by_id: dict[str, dict[str, Any]],
    protected: dict[str, str],
    max_tokens: int,
    max_headroom_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blockers: list[str] = []
    if not path_exists(path):
        return [], {
            "name": name,
            "path": str(path),
            "kind": "row_level_csv",
            "status": "failed",
            "blockers": [f"run_csv_missing:{path}"],
        }
    source_rows = read_csv(path)
    run_by_id = {str(row.get("id", "")).strip(): row for row in source_rows if str(row.get("id", "")).strip()}
    duplicate_ids = sorted([rid for rid, count in Counter(str(row.get("id", "")).strip() for row in source_rows).items() if rid and count > 1])
    if duplicate_ids:
        blockers.append("run_duplicate_ids:" + ",".join(duplicate_ids[:10]))
    required = {"id", "raw_output", "prediction"}
    missing_columns = sorted(required - set(source_rows[0].keys() if source_rows else []))
    if missing_columns:
        blockers.append("run_missing_required_columns:" + ",".join(missing_columns))

    detail_rows: list[dict[str, Any]] = []
    family_summary: dict[str, Counter[str]] = defaultdict(Counter)
    max_seen_headroom = 0.0
    run_blocker_counter: Counter[str] = Counter()
    run_warning_counter: Counter[str] = Counter()
    for weak in weak_rows:
        rid = str(weak["id"])
        family = str(weak["family"])
        answer = str(weak["answer"])
        row = run_by_id.get(rid)
        row_blockers: list[str] = []
        row_warnings: list[str] = []
        if row is None:
            row_blockers.append("missing_run_row")
            row = {}
        raw_output = str(row.get("raw_output", ""))
        stored_prediction = str(row.get("prediction", "")).strip()
        extracted = extract_final_answer(raw_output) if raw_output.strip() else "NOT_FOUND"
        stored_correct_recomputed = verify_answer(answer, stored_prediction)
        extracted_correct = verify_answer(answer, extracted)
        run_prompt = str(row.get("prompt", "") or weak["prompt"])
        run_prompt_sha = str(row.get("prompt_sha256", "") or sha256_text(run_prompt))
        prompt_hash_match = run_prompt_sha == str(weak["prompt_sha256"])
        stored_matches_extracted = stored_prediction == extracted
        stored_correct_column = str(row.get("stored_correct", "")).strip()
        correct_column = str(row.get("correct", "")).strip()
        stored_correct_column_matches = True
        correct_column_matches_extracted = True
        if stored_correct_column:
            stored_correct_column_matches = truthy(stored_correct_column) == stored_correct_recomputed
        if correct_column:
            correct_column_matches_extracted = truthy(correct_column) == extracted_correct
        truncated = truthy(row.get("truncated", row.get("truncated_bool", ""))) or str(
            row.get("finish_reason", "")
        ).strip().lower() == "length"
        span_start, span_end = answer_span(raw_output, extracted)
        count_tokens = token_count_approx(raw_output)
        headroom = (count_tokens / max_tokens) if max_tokens else 0.0
        max_seen_headroom = max(max_seen_headroom, headroom)
        boxed_count = len(find_boxed_spans(raw_output))
        protected_row = rid in protected

        if not raw_output.strip():
            row_blockers.append("missing_raw_output")
        if raw_output.strip() and not stored_matches_extracted:
            if stored_correct_recomputed != extracted_correct:
                row_blockers.append("stored_prediction_changes_correct_vs_raw_extraction")
            else:
                row_warnings.append("stored_prediction_not_raw_extraction")
        if not prompt_hash_match:
            row_blockers.append("prompt_hash_mismatch")
        if not stored_correct_column_matches:
            # `stored_correct` is a historical diagnostic column.  Promotion
            # depends on `prediction` extracted from `raw_output` and the
            # current `correct` column only; stale stored_correct values are
            # recorded but must not block a clean label-free baseline.
            row_warnings.append("stored_correct_column_mismatch")
        if not correct_column_matches_extracted:
            row_blockers.append("correct_column_not_raw_extraction")
        if truncated:
            row_blockers.append("truncated_row")
        if headroom > max_headroom_ratio:
            row_blockers.append("token_headroom_gt_limit")
        if protected_row:
            expected_protected = protected[rid]
            if extracted != expected_protected or not verify_answer(expected_protected, extracted):
                row_blockers.append("protected_row_answer_mismatch")

        for blocker in row_blockers:
            run_blocker_counter[blocker] += 1
        for warning in row_warnings:
            run_warning_counter[warning] += 1
        family_summary[family]["rows"] += 1
        family_summary[family]["extracted_correct"] += int(extracted_correct)
        family_summary[family]["stored_correct_recomputed"] += int(stored_correct_recomputed)
        family_summary[family]["truncated"] += int(truncated)
        family_summary[family]["raw_output_missing"] += int(not raw_output.strip())
        detail_rows.append(
            {
                "run_name": name,
                "id": rid,
                "family": family,
                "answer": answer,
                "prompt_sha256": weak["prompt_sha256"],
                "run_prompt_sha256": run_prompt_sha,
                "prompt_hash_match": bool_text(prompt_hash_match),
                "stored_prediction": stored_prediction,
                "extracted_answer": extracted,
                "stored_prediction_matches_extracted": bool_text(stored_matches_extracted),
                "stored_correct_recomputed": bool_text(stored_correct_recomputed),
                "extracted_correct": bool_text(extracted_correct),
                "stored_correct_column": stored_correct_column,
                "correct_column": correct_column,
                "stored_correct_column_matches": bool_text(stored_correct_column_matches),
                "correct_column_matches_extracted": bool_text(correct_column_matches_extracted),
                "raw_output_present": bool_text(bool(raw_output.strip())),
                "raw_output_chars": len(raw_output),
                "boxed_count": boxed_count,
                "answer_span_start": span_start,
                "answer_span_end": span_end,
                "token_count": count_tokens,
                "token_counter": "approx_regex_nonspace",
                "token_headroom_ratio": round(headroom, 6),
                "truncated": bool_text(truncated),
                "protected_row": bool_text(protected_row),
                "warnings": ";".join(row_warnings),
                "blockers": ";".join(row_blockers),
            }
        )

    missing_weak_ids = sorted(set(weak_by_id) - set(run_by_id))
    extra_ids = sorted(set(run_by_id) - set(weak_by_id))
    if missing_weak_ids:
        blockers.append(f"missing_weak_ids:{len(missing_weak_ids)}")
    if extra_ids:
        blockers.append(f"extra_ids:{len(extra_ids)}")
    if run_blocker_counter:
        blockers.extend(f"{key}:{value}" for key, value in sorted(run_blocker_counter.items()))

    per_family = {family: dict(counter) for family, counter in sorted(family_summary.items())}
    total_correct = sum(int(counter["extracted_correct"]) for counter in family_summary.values())
    total_rows = sum(int(counter["rows"]) for counter in family_summary.values())
    summary = {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": "row_level_csv",
        "status": "passed" if not blockers else "failed",
        "rows": len(source_rows),
        "weak_rows_scored": total_rows,
        "total_correct": total_correct,
        "accuracy": (total_correct / total_rows) if total_rows else 0.0,
        "per_family": per_family,
        "max_token_headroom_ratio": round(max_seen_headroom, 6),
        "token_counter": "approx_regex_nonspace",
        "blocker_counts": dict(sorted(run_blocker_counter.items())),
        "warning_counts": dict(sorted(run_warning_counter.items())),
        "blockers": blockers,
        "protected": {
            rid: {
                "expected": expected,
                "extracted": next((row["extracted_answer"] for row in detail_rows if row["id"] == rid), ""),
                "passed": not any(
                    row["id"] == rid and "protected_row_answer_mismatch" in str(row["blockers"])
                    for row in detail_rows
                ),
            }
            for rid, expected in sorted(protected.items())
        },
    }
    return detail_rows, summary


def audit_summary_run(name: str, path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path_exists(path):
        return {"name": name, "path": str(path), "kind": "summary_only", "status": "failed", "blockers": [f"summary_missing:{path}"]}
    rows = read_csv(path) if path.suffix.lower() == ".csv" else []
    blockers.append("summary_only_missing_row_level_raw_outputs")
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": "summary_only",
        "status": "failed",
        "rows": rows,
        "blockers": blockers,
    }


def audit_manifest_summary(name: str, path: Path) -> dict[str, Any]:
    if not path_exists(path):
        return {"name": name, "path": str(path), "kind": "manifest_summary", "status": "failed", "blockers": [f"manifest_missing:{path}"]}
    payload = json.loads(windows_long_path(path).read_text(encoding="utf-8"))
    protected_guard = payload.get("protected_row_backfire_guard", {})
    weak_gate = payload.get("weak_promotion_gate", {})
    blockers = ["manifest_summary_missing_row_level_raw_outputs"]
    if protected_guard and not protected_guard.get("passed", False):
        blockers.append("manifest_protected_row_guard_failed")
    if weak_gate and weak_gate.get("passed_candidate_count", 0) == 0:
        blockers.append("manifest_weak_promotion_gate_blocked")
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": "manifest_summary",
        "status": "failed",
        "repo_commit": payload.get("repo_commit", ""),
        "weak_csv": payload.get("weak_csv", {}),
        "candidate_summary": payload.get("candidate_summary", {}),
        "protected_row_backfire_guard": protected_guard,
        "weak_promotion_gate": weak_gate,
        "blockers": blockers,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V540 Answer Extraction Audit",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- gate_status: `{summary['gate_status']}`",
        f"- cpu_extractor_parity_status: `{summary['cpu_extractor_parity_status']}`",
        f"- prompt_template_parity_status: `{summary['prompt_template_parity_status']}`",
        f"- overall_blockers: `{len(summary['blockers'])}`",
        "",
        "## Runs",
        "",
        "| run | kind | status | total | bit | equation | max headroom | blockers |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        per_family = run.get("per_family", {})
        bit = per_family.get("bit_manipulation", {}).get("extracted_correct", "")
        equation = per_family.get("equation_transform", {}).get("extracted_correct", "")
        lines.append(
            f"| `{run.get('name','')}` | `{run.get('kind','')}` | `{run.get('status','')}` | "
            f"{run.get('total_correct','')} | {bit} | {equation} | "
            f"{run.get('max_token_headroom_ratio','')} | {len(run.get('blockers', []))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Row-level runs may be used only when `status=passed` and they expose `raw_output`.",
            "- Summary-only V537 artifacts are diagnostic, not promotion evidence.",
            "- Any stored-prediction mismatch with `extract_final_answer(raw_output)` blocks downstream GPU.",
        ]
    )
    if summary["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in summary["blockers"][:80]:
            lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V540 ANSWER EXTRACTION AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("weak_csv =", args.weak_csv, flush=True)
    print("baseline_csv =", args.baseline_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weak_rows, weak_by_id, weak_summary = load_weak_rows(args.weak_csv)
    protected = parse_protected_pairs(args.protected_id_answer)
    print("weak_summary =", json.dumps(weak_summary, sort_keys=True), flush=True)
    print("protected_id_answers =", json.dumps(protected, sort_keys=True), flush=True)

    detail_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    row_specs = [("baseline", args.baseline_csv)]
    row_specs.extend(parse_named_path(item) for item in args.run_csv)
    for name, path in row_specs:
        rows, run_summary = audit_row_level_run(
            name=name,
            path=path,
            weak_rows=weak_rows,
            weak_by_id=weak_by_id,
            protected=protected,
            max_tokens=args.max_tokens,
            max_headroom_ratio=args.max_token_headroom_ratio,
        )
        detail_rows.extend(rows)
        run_summaries.append(run_summary)
        print("run_summary =", json.dumps({k: v for k, v in run_summary.items() if k != "blockers"}, sort_keys=True)[:4000], flush=True)
        print("run_blocker_count =", len(run_summary.get("blockers", [])), flush=True)

    for item in args.summary_csv:
        name, path = parse_named_path(item)
        run_summaries.append(audit_summary_run(name, path))
    for item in args.manifest_json:
        name, path = parse_named_path(item)
        run_summaries.append(audit_manifest_summary(name, path))

    blockers = list(weak_summary.get("blockers", []))
    for run in run_summaries:
        for blocker in run.get("blockers", []):
            blockers.append(f"{run.get('name')}:{blocker}")
    row_level_runs = [run for run in run_summaries if run.get("kind") == "row_level_csv"]
    extractor_parity_pass = all(
        not run.get("blocker_counts", {}).get("stored_prediction_changes_correct_vs_raw_extraction")
        and not run.get("blocker_counts", {}).get("correct_column_not_raw_extraction")
        for run in row_level_runs
    )
    prompt_parity_pass = all(not run.get("blocker_counts", {}).get("prompt_hash_mismatch") for run in row_level_runs)
    gate_pass = not blockers
    summary = {
        "schema_version": "kg1_v540_answer_extraction_audit_v1",
        "generated_at_utc": utc_now(),
        "weak_csv": weak_summary,
        "protected_id_answers": protected,
        "max_tokens": args.max_tokens,
        "max_token_headroom_ratio": args.max_token_headroom_ratio,
        "gate_status": "passed" if gate_pass else "failed",
        "cpu_extractor_parity_status": "passed" if extractor_parity_pass else "failed",
        "prompt_template_parity_status": "passed" if prompt_parity_pass else "failed",
        "runs": run_summaries,
        "blockers": blockers,
        "outputs": {
            "details_csv": str(args.output_dir / f"{args.label}_details.csv"),
            "summary_json": str(args.output_dir / f"{args.label}_summary.json"),
            "markdown": str(args.output_dir / f"KG1_{args.label.upper()}_SUMMARY.md"),
        },
    }
    write_csv(Path(summary["outputs"]["details_csv"]), detail_rows, DETAIL_COLUMNS)
    write_json(Path(summary["outputs"]["summary_json"]), summary)
    windows_long_path(Path(summary["outputs"]["markdown"])).write_text(build_markdown(summary), encoding="utf-8")
    print("gate_status =", summary["gate_status"], flush=True)
    print("cpu_extractor_parity_status =", summary["cpu_extractor_parity_status"], flush=True)
    print("prompt_template_parity_status =", summary["prompt_template_parity_status"], flush=True)
    print("details_csv =", summary["outputs"]["details_csv"], flush=True)
    print("summary_json =", summary["outputs"]["summary_json"], flush=True)
    print("markdown =", summary["outputs"]["markdown"], flush=True)
    print("=== V540 ANSWER EXTRACTION AUDIT END ===", flush=True)
    if args.fail_on_blocked and not gate_pass:
        raise SystemExit(2)
    return summary


def run_self_test() -> None:
    raw = "Reasoning\nThe answer is \\boxed{00101101}"
    if extract_final_answer(raw) != "00101101":
        raise AssertionError("extract_final_answer self-test failed")
    if extract_final_answer(r"\boxed{]}\!}") != r"]}\!":
        raise AssertionError("extract_final_answer symbolic escaped suffix self-test failed")
    if extract_final_answer(r"\boxed{$}{>}") != "$":
        raise AssertionError("extract_final_answer over-extension guard self-test failed")
    start, end = answer_span(raw, "00101101")
    if raw[start:end] != "00101101":
        raise AssertionError((start, end, raw[start:end]))
    rows = [
        {"id": "b", "family": "bit_manipulation", "answer": "1", "prompt_sha256": "h2"},
        {"id": "a", "family": "equation_transform", "answer": "2", "prompt_sha256": "h1"},
    ]
    observed = row_contract(rows)
    if observed != sha256_text("a\tequation_transform\t2\th1\nb\tbit_manipulation\t1\th2"):
        raise AssertionError("row_contract sorting failed")
    print("v540_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-csv", type=Path, default=DEFAULT_WEAK_CSV)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--run-csv", action="append", default=[], help="Additional row-level CSV as name=path.")
    parser.add_argument("--summary-csv", action="append", default=[], help="Summary-only CSV as name=path; always diagnostic-only.")
    parser.add_argument("--manifest-json", action="append", default=[], help="Summary manifest as name=path; always diagnostic-only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v540_answer_extraction_audit")
    parser.add_argument("--protected-id-answer", action="append", default=list(DEFAULT_PROTECTED_ID_ANSWER))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-token-headroom-ratio", type=float, default=DEFAULT_MAX_TOKEN_HEADROOM_RATIO)
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

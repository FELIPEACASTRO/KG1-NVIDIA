#!/usr/bin/env python3
"""CPU audit for KG1 prediction parsing and failure classes.

This script does not train, submit, or use weak/full labels as training data.
It audits an existing predictions CSV with raw model outputs and checks whether
any deterministic parser strategy can improve over the locked submit-safe
baseline without losing bit accuracy.
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
    canonical_answer,
    canonical_family,
    extract_boxed_answers,
    extract_final_answer,
    verify_answer,
)


DEFAULT_PREDICTIONS_CSV = REPO_ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v445_prediction_parse_audit"

BASELINE_TOTAL = 192
BASELINE_EQUATION = 56
BASELINE_BIT = 136
BASELINE_TRUNCATED = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nonempty_boxed(raw_output: str) -> list[str]:
    return [canonical_answer(item) for item in extract_boxed_answers(raw_output) if canonical_answer(item)]


def last_nonempty_line(raw_output: str) -> str:
    lines = [line.strip() for line in str(raw_output or "").splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def final_phrase_answer(raw_output: str) -> str:
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:\uFF1A]\s*([^\n]+)",
        r"final answer\s*[:\uFF1A]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, str(raw_output or ""), re.IGNORECASE)
        if matches:
            return canonical_answer(matches[-1])
    return "NOT_FOUND"


def last_number(raw_output: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(raw_output or ""))
    return matches[-1] if matches else "NOT_FOUND"


def early_extract(raw_output: str, chars: int) -> str:
    return extract_final_answer(str(raw_output or "")[:chars])


def answer_literal_in_raw(answer: str, raw_output: str) -> bool:
    expected = canonical_answer(answer)
    if not expected or expected == "NOT_FOUND":
        return False
    return expected.lower() in str(raw_output or "").lower()


def strategy_prediction(row: dict[str, Any], strategy: str) -> str:
    raw_output = str(row.get("raw_output", ""))
    boxes = nonempty_boxed(raw_output)
    if strategy == "current_prediction":
        return canonical_answer(row.get("prediction", ""))
    if strategy == "official_reextract_raw":
        return canonical_answer(extract_final_answer(raw_output))
    if strategy == "first_boxed":
        return boxes[0] if boxes else "NOT_FOUND"
    if strategy == "last_boxed":
        return boxes[-1] if boxes else "NOT_FOUND"
    if strategy == "first_line":
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        return lines[0] if lines else "NOT_FOUND"
    if strategy == "last_line":
        return last_nonempty_line(raw_output)
    if strategy == "final_phrase":
        return final_phrase_answer(raw_output)
    if strategy == "last_number":
        return last_number(raw_output)
    if strategy == "early_512":
        return early_extract(raw_output, 512)
    if strategy == "early_1024":
        return early_extract(raw_output, 1024)
    if strategy == "early_2048":
        return early_extract(raw_output, 2048)
    raise ValueError(f"unknown strategy: {strategy}")


STRATEGIES = [
    "current_prediction",
    "official_reextract_raw",
    "first_boxed",
    "last_boxed",
    "first_line",
    "last_line",
    "final_phrase",
    "last_number",
    "early_512",
    "early_1024",
    "early_2048",
]


def classify_failure(row: dict[str, Any]) -> str:
    if row["current_correct"]:
        return "correct"
    if row["truncated"]:
        return "truncated"
    if row["parser_mismatch"]:
        return "parser_mismatch"
    if row["boxed_count"] == 0:
        return "no_boxed"
    if row["last_boxed_correct"]:
        return "parser_should_have_rescued"
    if row["first_boxed_correct"]:
        return "first_boxed_only_correct"
    if row["answer_literal_in_raw"]:
        return "answer_literal_nonfinal_or_ambiguous"
    return "boxed_or_generation_wrong"


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    required = {"id", "answer", "prediction", "raw_output"}
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            raise RuntimeError("predictions csv missing columns: " + json.dumps(missing))

    for row in rows:
        raw_output = str(row.get("raw_output", ""))
        answer = canonical_answer(row.get("answer", ""))
        prediction = canonical_answer(row.get("prediction", ""))
        family = canonical_family(row.get("type") or row.get("task_type") or row.get("family") or "")
        boxes = nonempty_boxed(raw_output)
        official_reextract = canonical_answer(extract_final_answer(raw_output))
        first_box = boxes[0] if boxes else "NOT_FOUND"
        last_box = boxes[-1] if boxes else "NOT_FOUND"
        finish_reason = str(row.get("finish_reason", ""))
        current_correct = verify_answer(answer, prediction)
        item = {
            **row,
            "family": family,
            "answer": answer,
            "prediction": prediction,
            "current_correct": current_correct,
            "truncated": finish_reason == "length" or truthy(row.get("truncated")),
            "boxed_count": len(boxes),
            "first_boxed_prediction": first_box,
            "last_boxed_prediction": last_box,
            "official_reextract_prediction": official_reextract,
            "parser_mismatch": canonical_answer(official_reextract) != prediction,
            "first_boxed_correct": verify_answer(answer, first_box),
            "last_boxed_correct": verify_answer(answer, last_box),
            "official_reextract_correct": verify_answer(answer, official_reextract),
            "answer_literal_in_raw": answer_literal_in_raw(answer, raw_output),
            "raw_chars": len(raw_output),
        }
        item["failure_class"] = classify_failure(item)
        enriched.append(item)
    return enriched


def score_strategy(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    current_correct = {str(row["id"]): bool(row["current_correct"]) for row in rows}
    correct_by_family: Counter[str] = Counter()
    total = 0
    gains = 0
    losses = 0
    changed = 0
    for row in rows:
        pred = strategy_prediction(row, strategy)
        ok = verify_answer(row["answer"], pred)
        total += int(ok)
        correct_by_family[str(row["family"])] += int(ok)
        before = current_correct[str(row["id"])]
        gains += int(ok and not before)
        losses += int(before and not ok)
        changed += int(canonical_answer(pred) != canonical_answer(row["prediction"]))
    bit = correct_by_family["bit_manipulation"]
    equation = correct_by_family["equation_transform"]
    truncated = sum(int(row["truncated"]) for row in rows)
    return {
        "strategy": strategy,
        "rows": len(rows),
        "correct": total,
        "equation_transform_correct": equation,
        "bit_manipulation_correct": bit,
        "truncated": truncated,
        "changed_predictions": changed,
        "gains_vs_current": gains,
        "losses_vs_current": losses,
        "net_vs_current": gains - losses,
        "promotes": total > BASELINE_TOTAL and equation > BASELINE_EQUATION and bit >= BASELINE_BIT and truncated == BASELINE_TRUNCATED,
        "no_loss": losses == 0,
    }


def summarize_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"rows": 0, "misses": 0})
    for row in rows:
        key = (str(row["family"]), str(row["failure_class"]))
        counts[key]["rows"] += 1
        counts[key]["misses"] += int(not row["current_correct"])
    out = []
    for (family, failure_class), item in sorted(counts.items()):
        out.append({"family": family, "failure_class": failure_class, **item})
    return out


def family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "correct": 0,
            "misses": 0,
            "truncated": 0,
            "boxed_zero": 0,
            "boxed_multi": 0,
            "parser_mismatch": 0,
            "answer_literal_in_raw": 0,
        }
    )
    for row in rows:
        item = counts[str(row["family"])]
        item["rows"] += 1
        item["correct"] += int(row["current_correct"])
        item["misses"] += int(not row["current_correct"])
        item["truncated"] += int(row["truncated"])
        item["boxed_zero"] += int(row["boxed_count"] == 0)
        item["boxed_multi"] += int(row["boxed_count"] > 1)
        item["parser_mismatch"] += int(row["parser_mismatch"])
        item["answer_literal_in_raw"] += int(row["answer_literal_in_raw"] and not row["current_correct"])
    out = []
    for family, item in sorted(counts.items()):
        out.append(
            {
                "family": family,
                **item,
                "accuracy": item["correct"] / item["rows"] if item["rows"] else 0.0,
            }
        )
    total = defaultdict(int)
    for item in counts.values():
        for key, value in item.items():
            total[key] += value
    out.append(
        {
            "family": "OVERALL",
            **dict(total),
            "accuracy": total["correct"] / total["rows"] if total["rows"] else 0.0,
        }
    )
    return out


DETAIL_COLUMNS = [
    "id",
    "family",
    "answer",
    "prediction",
    "current_correct",
    "failure_class",
    "truncated",
    "boxed_count",
    "first_boxed_prediction",
    "last_boxed_prediction",
    "official_reextract_prediction",
    "parser_mismatch",
    "first_boxed_correct",
    "last_boxed_correct",
    "official_reextract_correct",
    "answer_literal_in_raw",
    "raw_chars",
    "finish_reason",
    "raw_output",
]

FAMILY_COLUMNS = [
    "family",
    "rows",
    "correct",
    "misses",
    "accuracy",
    "truncated",
    "boxed_zero",
    "boxed_multi",
    "parser_mismatch",
    "answer_literal_in_raw",
]

FAILURE_COLUMNS = ["family", "failure_class", "rows", "misses"]

STRATEGY_COLUMNS = [
    "strategy",
    "rows",
    "correct",
    "equation_transform_correct",
    "bit_manipulation_correct",
    "truncated",
    "changed_predictions",
    "gains_vs_current",
    "losses_vs_current",
    "net_vs_current",
    "no_loss",
    "promotes",
]


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V445 PREDICTION PARSE AUDIT START ===", flush=True)
    print("predictions_csv =", args.predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.predictions_csv.is_file():
        raise FileNotFoundError(args.predictions_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = enrich_rows(read_csv(args.predictions_csv))
    strategies = [score_strategy(rows, strategy) for strategy in STRATEGIES]
    families = family_summary(rows)
    failures = summarize_failures(rows)
    promotable = [row for row in strategies if row["promotes"]]

    detail_csv = args.output_dir / f"{args.label}_detail.csv"
    family_csv = args.output_dir / f"{args.label}_family_summary.csv"
    failure_csv = args.output_dir / f"{args.label}_failure_summary.csv"
    strategy_csv = args.output_dir / f"{args.label}_strategy_scores.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    write_csv(detail_csv, rows, DETAIL_COLUMNS)
    write_csv(family_csv, families, FAMILY_COLUMNS)
    write_csv(failure_csv, failures, FAILURE_COLUMNS)
    write_csv(strategy_csv, strategies, STRATEGY_COLUMNS)

    current = [row for row in strategies if row["strategy"] == "current_prediction"][0]
    decision = {
        "decision": "no_parser_or_extractor_gain",
        "hf_gpu_allowed": False,
        "reason": "No deterministic parser strategy beats the locked submit-safe gate.",
    }
    if promotable:
        decision = {
            "decision": "parser_strategy_promotes_cpu_only",
            "hf_gpu_allowed": False,
            "reason": "Parser strategy would improve CPU score, but parser/postprocessor is not adapter-only submit-safe.",
        }

    manifest = {
        "schema_version": "kg1_v445_prediction_parse_audit_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "predictions_csv": str(args.predictions_csv),
            "predictions_csv_sha256": sha256_file(args.predictions_csv),
        },
        "baseline_gate": {
            "total": BASELINE_TOTAL,
            "equation_transform": BASELINE_EQUATION,
            "bit_manipulation": BASELINE_BIT,
            "truncated": BASELINE_TRUNCATED,
        },
        "current": current,
        "promotable_strategy_count": len(promotable),
        "promotable_strategies": promotable,
        "decision": decision,
        "outputs": {
            "detail_csv": str(detail_csv),
            "detail_sha256": sha256_file(detail_csv),
            "family_summary_csv": str(family_csv),
            "family_summary_sha256": sha256_file(family_csv),
            "failure_summary_csv": str(failure_csv),
            "failure_summary_sha256": sha256_file(failure_csv),
            "strategy_scores_csv": str(strategy_csv),
            "strategy_scores_sha256": sha256_file(strategy_csv),
            "manifest_json": str(manifest_json),
        },
        "source_policy": {
            "weak_full_used_for_training": False,
            "purpose": "Audit parser/extractor failure classes only.",
            "adapter_only_submit_safe": False,
        },
    }
    write_json(manifest_json, manifest)
    print("current =", json.dumps(current, sort_keys=True), flush=True)
    print("promotable_strategy_count =", len(promotable), flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V445 PREDICTION PARSE AUDIT END ===", flush=True)
    return manifest


def self_test() -> None:
    sample = [
        {
            "id": "a",
            "type": "equation_transform",
            "answer": "42",
            "prediction": "41",
            "raw_output": "Reasoning 42\nFinal answer: \\boxed{41}",
            "finish_reason": "stop",
        },
        {
            "id": "b",
            "type": "bit_manipulation",
            "answer": "1010",
            "prediction": "1010",
            "raw_output": "\\boxed{1010}",
            "finish_reason": "stop",
        },
    ]
    enriched = enrich_rows(sample)
    assert enriched[0]["failure_class"] == "answer_literal_nonfinal_or_ambiguous"
    assert enriched[1]["current_correct"] is True
    current = score_strategy(enriched, "current_prediction")
    assert current["correct"] == 1 and current["bit_manipulation_correct"] == 1
    print("v445_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v445_prediction_parse_audit")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

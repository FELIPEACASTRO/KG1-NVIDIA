#!/usr/bin/env python3
"""KG1 local metric gate with Kaggle-style boxed answer extraction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import box_answer, canonical_answer, canonical_boxed_payload  # noqa: E402
from src.perfect_solver import classify_puzzle, solve  # noqa: E402


WEAK_FAMILIES = ("text_encryption", "equation_transform", "bit_manipulation")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def extract_boxed(text: str | None) -> str | None:
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None
    depth = 0
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch in "{}" and is_escaped(text, idx):
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:idx].strip()
    return None


def maybe_float(value: str) -> float | None:
    s = canonical_answer(value).replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s):
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def answers_match(expected: str, observed: str | None, rel_tol: float) -> bool:
    """Reproduz metric OFICIAL Kaggle (Agent D1 reverse engineering, 2026-04-21).

    CORRECOES vs versao anterior (bugs detectados em 2026-04-21):
    - REMOVIDO: re.fullmatch(r"[01]+", ...) strict — metric oficial NAO faz isso
    - MUDADO: math.isclose(rel_tol, abs_tol=1e-5) em vez de abs diff scaled

    Referencia: kaggle_research/extracted_kernel_code/huikang__*.txt L484-502
    """
    if observed is None:
        return False
    exp_s = canonical_answer(expected)
    obs_s = canonical_boxed_payload(observed)
    # Primeira tentativa: ambos numericos com math.isclose (comporta bit, numeral, gravity, unit)
    exp_f = maybe_float(exp_s)
    obs_f = maybe_float(obs_s)
    if exp_f is not None and obs_f is not None:
        return math.isclose(exp_f, obs_f, rel_tol=rel_tol, abs_tol=1e-5)
    # Fallback: comparacao case-insensitive (cipher, cryptarithm, etc)
    return obs_s.lower() == exp_s.lower()


def read_labeled_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        return list(reader)


def read_predictions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "id" not in fields:
            raise ValueError(f"{path} missing id column")
        prediction_col = "prediction" if "prediction" in fields else "answer" if "answer" in fields else None
        if prediction_col is None:
            raise ValueError(f"{path} missing prediction or answer column")
        return {row["id"]: row[prediction_col] for row in reader}


def score_rows(
    rows: list[dict[str, str]],
    predictions: dict[str, str] | None,
    families: set[str] | None,
    rel_tol: float,
    max_errors_per_family: int,
) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "solved": 0,
        "boxed": 0,
        "correct": 0,
        "errors": [],
    })

    for row in rows:
        prompt = row["prompt"]
        expected = row["answer"]
        family = classify_puzzle(prompt)
        if families and family not in families:
            continue

        if predictions is None:
            raw = solve(prompt)
            model_output = box_answer(raw) if raw is not None else ""
        else:
            model_output = predictions.get(row["id"], "")

        boxed = extract_boxed(model_output)
        matched = answers_match(expected, boxed, rel_tol)

        item = stats[family]
        item["total"] += 1
        item["boxed"] += 1 if boxed is not None else 0
        item["solved"] += 1 if boxed is not None else 0
        item["correct"] += 1 if matched else 0
        if not matched and len(item["errors"]) < max_errors_per_family:
            item["errors"].append({
                "id": row["id"],
                "expected": expected,
                "observed": boxed,
                "family": family,
            })

    by_family = {}
    total = correct = boxed_total = solved_total = 0
    for family, item in sorted(stats.items()):
        fam_total = item["total"]
        fam_correct = item["correct"]
        total += fam_total
        correct += fam_correct
        boxed_total += item["boxed"]
        solved_total += item["solved"]
        by_family[family] = {
            "total": fam_total,
            "solved": item["solved"],
            "boxed": item["boxed"],
            "correct": fam_correct,
            "accuracy": fam_correct / fam_total if fam_total else 0.0,
            "boxed_rate": item["boxed"] / fam_total if fam_total else 0.0,
            "sample_errors": item["errors"],
        }

    return {
        "total": total,
        "correct": correct,
        "solved": solved_total,
        "boxed": boxed_total,
        "overall_score": correct / total if total else 0.0,
        "boxed_rate": boxed_total / total if total else 0.0,
        "by_family": by_family,
    }


def write_errors_csv(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "id", "expected", "observed"])
        writer.writeheader()
        for family, item in metric["by_family"].items():
            for err in item.get("sample_errors", []):
                writer.writerow({
                    "family": family,
                    "id": err["id"],
                    "expected": err["expected"],
                    "observed": err["observed"],
                })


def parse_family_thresholds(values: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"family threshold must be FAMILY=VALUE, got: {value}")
        family, raw_threshold = value.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError(f"family threshold has empty family: {value}")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(f"family threshold is not numeric: {value}") from exc
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"family threshold must be between 0 and 1: {value}")
        thresholds[family] = threshold
    return thresholds


def build_decision(
    metric: dict[str, Any],
    min_score: float,
    min_boxed_rate: float,
    min_family_scores: dict[str, float],
) -> dict[str, Any]:
    reasons = []
    if metric["overall_score"] < min_score:
        reasons.append("overall_score_below_threshold")
    if metric["boxed_rate"] < min_boxed_rate:
        reasons.append("boxed_rate_below_threshold")
    for family, threshold in sorted(min_family_scores.items()):
        item = metric["by_family"].get(family)
        if item is None:
            reasons.append(f"family_missing:{family}")
            continue
        if item["accuracy"] < threshold:
            reasons.append(f"family_score_below_threshold:{family}")

    weak = {
        family: metric["by_family"][family]["accuracy"]
        for family in WEAK_FAMILIES
        if family in metric["by_family"]
    }
    return {
        "promotion_recommended": not reasons,
        "min_score": min_score,
        "min_boxed_rate": min_boxed_rate,
        "min_family_scores": min_family_scores,
        "reasons": reasons,
        "weak_family_scores": weak,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=REPO_ROOT / "data" / "kaggle" / "unzipped" / "train.csv")
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--families", nargs="*", help="Optional family filter")
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    parser.add_argument("--min-score", type=float, default=0.68)
    parser.add_argument("--min-boxed-rate", type=float, default=1.0)
    parser.add_argument(
        "--min-weak-family-score",
        type=float,
        default=None,
        help="Optional threshold applied to text_encryption, equation_transform, and bit_manipulation.",
    )
    parser.add_argument(
        "--min-family-score",
        action="append",
        default=[],
        metavar="FAMILY=VALUE",
        help="Optional per-family accuracy threshold. Can be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "runs" / "local_metric_gate.json")
    parser.add_argument("--errors-csv", type=Path)
    parser.add_argument("--max-errors-per-family", type=int, default=10)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_labeled_rows(args.train_csv)
    predictions = read_predictions(args.predictions_csv) if args.predictions_csv else None
    families = set(args.families) if args.families else None
    metric = score_rows(rows, predictions, families, args.rel_tol, args.max_errors_per_family)
    try:
        min_family_scores = parse_family_thresholds(args.min_family_score)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.min_weak_family_score is not None:
        if args.min_weak_family_score < 0.0 or args.min_weak_family_score > 1.0:
            print("[ERROR] --min-weak-family-score must be between 0 and 1", file=sys.stderr)
            return 2
        for family in WEAK_FAMILIES:
            min_family_scores.setdefault(family, args.min_weak_family_score)
    decision = build_decision(metric, args.min_score, args.min_boxed_rate, min_family_scores)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": "predictions_csv" if predictions is not None else "perfect_solver",
        "train_csv": str(args.train_csv),
        "predictions_csv": str(args.predictions_csv) if args.predictions_csv else None,
        "metric": metric,
        "decision": decision,
    }
    write_json(args.output_json, report)
    if args.errors_csv:
        write_errors_csv(args.errors_csv, metric)

    print(f"mode: {report['mode']}")
    print(f"rows: {metric['total']}")
    print(f"overall_score: {metric['overall_score']:.6f}")
    print(f"boxed_rate: {metric['boxed_rate']:.6f}")
    for family, item in metric["by_family"].items():
        print(f"{family}: {item['accuracy']:.6f} ({item['correct']}/{item['total']})")
    print(f"promotion recommended: {decision['promotion_recommended']}")
    print(f"report: {args.output_json}")
    if args.fail_on_block and not decision["promotion_recommended"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

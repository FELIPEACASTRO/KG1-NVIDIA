#!/usr/bin/env python3
"""KG1 V1241 strict transfer gate for bit/equation gains.

This gate turns the V1240 solver-verified weak-family artifacts into a
score-facing transfer contract. It is deliberately CPU-only: it prepares
solution/question CSVs and compares baseline vs candidate prediction CSVs from
real generation. A candidate passes only when gains in bit/equation survive
strict boxed-answer validation, truncation checks, and row-level regression
checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import (  # noqa: E402
    OFFICIAL_INFERENCE_CONFIG,
    canonical_answer,
    extract_closed_boxed_answers,
    extract_final_answer,
    extract_final_boxed_answer,
    has_unclosed_boxed_answer,
    verify_answer,
    verify_strict_boxed_answer,
)


DEFAULT_V1240_DIR = ROOT / "artifacts" / "v1240_solver_verified_weak_curriculum_gate"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "v1241_bit_equation_real_transfer_gate"
DEFAULT_SELF_TEST_DIR = ROOT / "artifacts" / "v1241_bit_equation_transfer_gate_selftest"

WEAK_FAMILIES = {"bit_manipulation", "equation_transform"}

PROFILE_DEFAULTS: dict[str, dict[str, float | int | None]] = {
    "tiny": {
        "expected_rows": 50,
        "min_bit_gain": 1,
        "min_equation_gain": 1,
        "min_total_gain": 2,
        "min_total_correct": None,
        "max_any_regressions": 0,
        "max_weak_regressions": 0,
        "max_protected_regressions": 0,
        "max_candidate_format_failures": 0,
        "max_candidate_public_false_gains": 0,
        "max_candidate_strict_false_public": 0,
        "max_candidate_truncated": 0,
    },
    "val170": {
        "expected_rows": 170,
        "min_bit_gain": 1,
        "min_equation_gain": 1,
        "min_total_gain": 2,
        "min_total_correct": None,
        "max_any_regressions": 0,
        "max_weak_regressions": 0,
        "max_protected_regressions": 0,
        "max_candidate_format_failures": 0,
        "max_candidate_public_false_gains": 0,
        "max_candidate_strict_false_public": 0,
        "max_candidate_truncated": 0,
    },
    "full947_089": {
        "expected_rows": 947,
        "min_bit_gain": 1,
        "min_equation_gain": 1,
        "min_total_gain": 20,
        "min_total_correct": 843,
        "max_any_regressions": 0,
        "max_weak_regressions": 0,
        "max_protected_regressions": 0,
        "max_candidate_format_failures": 0,
        "max_candidate_public_false_gains": 0,
        "max_candidate_strict_false_public": 0,
        "max_candidate_truncated": 0,
    },
    "full947_090": {
        "expected_rows": 947,
        "min_bit_gain": 1,
        "min_equation_gain": 1,
        "min_total_gain": 30,
        "min_total_correct": 853,
        "max_any_regressions": 0,
        "max_weak_regressions": 0,
        "max_protected_regressions": 0,
        "max_candidate_format_failures": 0,
        "max_candidate_public_false_gains": 0,
        "max_candidate_strict_false_public": 0,
        "max_candidate_truncated": 0,
    },
    "custom": {
        "expected_rows": None,
        "min_bit_gain": 0,
        "min_equation_gain": 0,
        "min_total_gain": 0,
        "min_total_correct": None,
        "max_any_regressions": 0,
        "max_weak_regressions": 0,
        "max_protected_regressions": 0,
        "max_candidate_format_failures": 0,
        "max_candidate_public_false_gains": 0,
        "max_candidate_strict_false_public": 0,
        "max_candidate_truncated": 0,
    },
}

V291_PUBLIC_REFERENCE = {
    "source": "validated_project_submission_history",
    "full947_correct": 823,
    "full947_rows": 947,
    "bit_manipulation_correct": 135,
    "bit_manipulation_rows": 160,
    "equation_transform_correct": 56,
    "equation_transform_rows": 155,
    "protected_families_correct": 632,
    "protected_families_rows": 632,
    "note": "Reference only. This gate must compare real raw_output CSVs, not reuse this constant as proof.",
}


@dataclass(frozen=True)
class Thresholds:
    expected_rows: int | None
    min_bit_gain: int
    min_equation_gain: int
    min_total_gain: int
    min_total_correct: int | None
    max_any_regressions: int
    max_weak_regressions: int
    max_protected_regressions: int
    max_candidate_format_failures: int
    max_candidate_public_false_gains: int
    max_candidate_strict_false_public: int
    max_candidate_truncated: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def first_message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def solution_row_from_jsonl(row: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = str(row.get("prompt") or first_message_content(row, "user"))
    metadata = row.get("metadata") or {}
    return {
        "id": str(row["id"]),
        "prompt": prompt,
        "answer": str(row["answer"]),
        "family": str(row.get("family", "")),
        "source": str(row.get("source", "")),
        "source_role": str(row.get("source_role", "")),
        "subcategory": str(row.get("subcategory", "")),
        "row_loss_weight": str(row.get("row_loss_weight", row.get("loss_weight", ""))),
        "split": split,
        "v1240_stability": str(metadata.get("v1240_stability", "")),
        "v1240_solver_rule_kind": str(metadata.get("v1240_solver_rule_kind", "")),
    }


def select_round_robin_by_role(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (str(item.get("source_role", "")), str(item.get("id", "")))):
        buckets[str(row.get("source_role", ""))].append(row)
    selected: list[dict[str, Any]] = []
    roles = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for role in roles:
            if len(selected) >= limit:
                break
            if buckets[role]:
                selected.append(buckets[role].pop(0))
    return selected


def build_contract(v1240_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = v1240_dir / "kg1_v1240_solver_verified_weak_curriculum_gate.json"
    train_path = v1240_dir / "v1240_solver_verified_train.jsonl"
    val_path = v1240_dir / "v1240_solver_verified_val.jsonl"
    audit_path = v1240_dir / "v1240_solver_verified_row_audit.csv"

    for path in (manifest_path, train_path, val_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(f"required V1240 artifact not found: {path}")

    v1240_manifest = read_json(manifest_path)
    train_rows = [solution_row_from_jsonl(row, "train") for row in load_jsonl(train_path)]
    val_rows = [solution_row_from_jsonl(row, "val") for row in load_jsonl(val_path)]

    train_family_counts = Counter(row["family"] for row in train_rows)
    val_family_counts = Counter(row["family"] for row in val_rows)
    if train_family_counts.get("bit_manipulation", 0) != 540:
        raise ValueError(f"unexpected V1240 train bit count: {train_family_counts.get('bit_manipulation', 0)}")
    if train_family_counts.get("equation_transform", 0) != 360:
        raise ValueError(f"unexpected V1240 train equation count: {train_family_counts.get('equation_transform', 0)}")
    if val_family_counts.get("bit_manipulation", 0) != 90:
        raise ValueError(f"unexpected V1240 val bit count: {val_family_counts.get('bit_manipulation', 0)}")
    if val_family_counts.get("equation_transform", 0) != 60:
        raise ValueError(f"unexpected V1240 val equation count: {val_family_counts.get('equation_transform', 0)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    solution_fields = [
        "id",
        "prompt",
        "answer",
        "family",
        "source",
        "source_role",
        "subcategory",
        "row_loss_weight",
        "split",
        "v1240_stability",
        "v1240_solver_rule_kind",
    ]
    question_fields = ["id", "prompt"]

    train_solution_csv = output_dir / "v1241_v1240_train_solution.csv"
    val_solution_csv = output_dir / "v1241_v1240_val170_solution.csv"
    val_questions_csv = output_dir / "v1241_v1240_val170_questions.csv"
    write_csv(train_solution_csv, train_rows, solution_fields)
    write_csv(val_solution_csv, val_rows, solution_fields)
    write_csv(val_questions_csv, [{"id": row["id"], "prompt": row["prompt"]} for row in val_rows], question_fields)

    bit_val = [row for row in val_rows if row["family"] == "bit_manipulation"]
    equation_val = [row for row in val_rows if row["family"] == "equation_transform"]
    protected_val = [row for row in val_rows if row["family"] not in WEAK_FAMILIES]
    tiny_rows = (
        select_round_robin_by_role(bit_val, 15)
        + select_round_robin_by_role(equation_val, 15)
        + sorted(protected_val, key=lambda item: (item["family"], item["id"]))
    )
    tiny_rows = sorted(tiny_rows, key=lambda item: (item["family"] not in WEAK_FAMILIES, item["family"], item["id"]))

    tiny_solution_csv = output_dir / "v1241_tiny_bit_equation_probe_solution.csv"
    tiny_questions_csv = output_dir / "v1241_tiny_bit_equation_probe_questions.csv"
    write_csv(tiny_solution_csv, tiny_rows, solution_fields)
    write_csv(tiny_questions_csv, [{"id": row["id"], "prompt": row["prompt"]} for row in tiny_rows], question_fields)

    readme_path = output_dir / "V1241_BIT_EQUATION_REAL_TRANSFER_GATE.md"
    readme = f"""# KG1 V1241 Bit/Equation Real Transfer Gate

Generated at UTC: `{utc_now()}`

Decision: `contract_created_cpu_only_no_train_no_submit`.

This artifact converts V1240 solver-verified weak-family rows into strict
transfer gates. It does not authorize Kaggle submit, paid GPU launch, package
creation, or leaderboard claims.

## What Counts As Real Gain

A candidate gain counts only when all of these are true:

- The row has real `raw_output` from generation.
- The candidate output has exactly one closed `\\boxed{{...}}`.
- The strict final boxed answer is correct.
- The public fallback extractor does not create a correctness gain that the
  strict boxed extractor rejects.
- The row is not truncated by `finish_reason`, explicit truncation flags, or
  `completion_tokens >= max_tokens`.
- No protected, weak-family, or total row regression exceeds the selected
  profile threshold.

## Produced CSVs

- `{val_solution_csv.name}`: all 170 V1240 validation rows.
- `{val_questions_csv.name}`: prompts for all 170 rows.
- `{tiny_solution_csv.name}`: 50-row balanced probe, with 15 bit, 15 equation,
  and all 20 protected anchors.
- `{tiny_questions_csv.name}`: prompts for the 50-row probe.

## Commands

Build/refresh this contract:

```bash
python scripts/kg1_v1241_bit_equation_transfer_gate.py --build-contract
```

Run the strict tiny transfer gate after generating baseline and candidate CSVs:

```bash
python scripts/kg1_v1241_bit_equation_transfer_gate.py \\
  --solution-csv artifacts/v1241_bit_equation_real_transfer_gate/v1241_tiny_bit_equation_probe_solution.csv \\
  --baseline-predictions artifacts/.../baseline_predictions.csv \\
  --candidate-predictions artifacts/.../candidate_predictions.csv \\
  --profile tiny \\
  --output-dir artifacts/.../v1241_tiny_transfer_report
```

Run the full V1240 validation gate:

```bash
python scripts/kg1_v1241_bit_equation_transfer_gate.py \\
  --solution-csv artifacts/v1241_bit_equation_real_transfer_gate/v1241_v1240_val170_solution.csv \\
  --baseline-predictions artifacts/.../baseline_predictions.csv \\
  --candidate-predictions artifacts/.../candidate_predictions.csv \\
  --profile val170 \\
  --output-dir artifacts/.../v1241_val170_transfer_report
```

For a leaderboard-facing 0.89 claim on 947 rows, use `--profile full947_089`
with the official 947-row solution/prediction CSVs. The hard threshold is
`843/947` correct, which is at least `+20` over the validated V291 reference
of `823/947`. For a 0.90 claim, use `--profile full947_090`, requiring
`853/947`, at least `+30`.
"""
    readme_path.write_text(readme, encoding="utf-8")

    manifest_path_out = output_dir / "kg1_v1241_bit_equation_real_transfer_gate.json"
    manifest = {
        "decision": "pass_v1241_contract_created_cpu_only_no_train_no_submit",
        "generated_at_utc": utc_now(),
        "v1240_decision": v1240_manifest.get("decision"),
        "inputs": {
            "v1240_manifest": str(manifest_path),
            "v1240_manifest_sha256": sha256_file(manifest_path),
            "v1240_train_jsonl": str(train_path),
            "v1240_train_sha256": sha256_file(train_path),
            "v1240_val_jsonl": str(val_path),
            "v1240_val_sha256": sha256_file(val_path),
            "v1240_audit_csv": str(audit_path),
            "v1240_audit_sha256": sha256_file(audit_path),
        },
        "counts": {
            "train_total": len(train_rows),
            "train_by_family": dict(sorted(train_family_counts.items())),
            "val_total": len(val_rows),
            "val_by_family": dict(sorted(val_family_counts.items())),
            "tiny_total": len(tiny_rows),
            "tiny_by_family": dict(sorted(Counter(row["family"] for row in tiny_rows).items())),
            "tiny_by_source_role": dict(sorted(Counter(row["source_role"] for row in tiny_rows).items())),
        },
        "strict_transfer_rules": [
            "candidate predictions must include raw_output",
            "candidate output must have exactly one closed boxed answer",
            "strict boxed correctness is the score used for promotion",
            "public-metric-only correctness is a blocker, not a gain",
            "strict-metric-only correctness is a blocker, not public-like proof",
            "truncated generations are blockers",
            "row-level regressions are blockers unless explicitly relaxed",
            "protected-family regressions are always tracked separately",
        ],
        "promotion_profiles": PROFILE_DEFAULTS,
        "v291_public_reference": V291_PUBLIC_REFERENCE,
        "blocked_actions": [
            "kaggle_submit",
            "paid_gpu_launch",
            "full947_package_creation",
            "leaderboard_score_claim_without_raw_output_gate",
        ],
        "outputs": {
            "train_solution_csv": str(train_solution_csv),
            "train_solution_sha256": sha256_file(train_solution_csv),
            "val_solution_csv": str(val_solution_csv),
            "val_solution_sha256": sha256_file(val_solution_csv),
            "val_questions_csv": str(val_questions_csv),
            "val_questions_sha256": sha256_file(val_questions_csv),
            "tiny_solution_csv": str(tiny_solution_csv),
            "tiny_solution_sha256": sha256_file(tiny_solution_csv),
            "tiny_questions_csv": str(tiny_questions_csv),
            "tiny_questions_sha256": sha256_file(tiny_questions_csv),
            "readme": str(readme_path),
            "readme_sha256": sha256_file(readme_path),
        },
    }
    write_json(manifest_path_out, manifest)
    return manifest


def parse_boolish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def parse_intish(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def row_id_field(fieldnames: list[str], label: str) -> str:
    for candidate in ("id", "row_id"):
        if candidate in fieldnames:
            return candidate
    if fieldnames:
        return fieldnames[0]
    raise ValueError(f"{label} CSV has no usable id column")


def normalize_solution_csv(path: Path) -> list[dict[str, str]]:
    rows, fieldnames = read_csv_rows(path)
    required = {"answer"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} missing required solution columns: {missing}")
    id_field = row_id_field(fieldnames, "solution")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get(id_field, "")).strip()
        if not row_id:
            raise ValueError(f"{path} contains a solution row without id")
        if row_id in seen:
            raise ValueError(f"{path} contains duplicate solution id: {row_id}")
        seen.add(row_id)
        normalized.append(
            {
                "id": row_id,
                "prompt": str(row.get("prompt", "")),
                "answer": str(row.get("answer", "")),
                "family": str(row.get("family") or row.get("type") or row.get("task_family") or "unknown"),
                "source_role": str(row.get("source_role", "")),
                "source": str(row.get("source", "")),
                "subcategory": str(row.get("subcategory", "")),
            }
        )
    return normalized


def normalize_predictions_csv(
    path: Path,
    *,
    label: str,
    allow_prediction_as_raw_output: bool,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows, fieldnames = read_csv_rows(path)
    id_field = row_id_field(fieldnames, label)
    if "raw_output" not in fieldnames:
        if allow_prediction_as_raw_output and "prediction" in fieldnames:
            pass
        else:
            raise ValueError(
                f"{label} predictions need raw_output. "
                "Use --allow-prediction-as-raw-output only for synthetic debugging, not promotion."
            )
    normalized: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get(id_field, "")).strip()
        if not row_id:
            raise ValueError(f"{path} contains a prediction row without id")
        if row_id in normalized:
            raise ValueError(f"{path} contains duplicate prediction id: {row_id}")
        raw_output = str(row.get("raw_output", ""))
        if not raw_output and allow_prediction_as_raw_output:
            raw_output = str(row.get("prediction", ""))
        normalized[row_id] = {key: str(value or "") for key, value in row.items()}
        normalized[row_id]["raw_output"] = raw_output
    return normalized, fieldnames


def score_one(
    solution: dict[str, str],
    prediction: dict[str, str] | None,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    raw_output = str((prediction or {}).get("raw_output", ""))
    answer = solution["answer"]
    closed_boxed_answers = extract_closed_boxed_answers(raw_output)
    boxed_marker_count = raw_output.count(r"\boxed{")
    unclosed_boxed = has_unclosed_boxed_answer(raw_output)
    final_boxed = extract_final_boxed_answer(raw_output)
    public_final = extract_final_answer(raw_output)
    strict_correct = verify_strict_boxed_answer(answer, raw_output)
    public_correct = verify_answer(answer, public_final)
    exact_one_closed_boxed = (
        boxed_marker_count == 1
        and len(closed_boxed_answers) == 1
        and not unclosed_boxed
        and final_boxed != "NOT_FOUND"
    )
    no_box = boxed_marker_count == 0
    multi_box = boxed_marker_count > 1 or len(closed_boxed_answers) > 1
    malformed_boxed = boxed_marker_count > len(closed_boxed_answers) or unclosed_boxed
    public_metric_only_false_gain = bool(public_correct and not strict_correct)
    strict_metric_only_false_positive = bool(strict_correct and not public_correct)

    finish_reason = str((prediction or {}).get("finish_reason", "")).strip().lower()
    completion_tokens = parse_intish((prediction or {}).get("completion_tokens"))
    explicit_truncated = any(
        parse_boolish((prediction or {}).get(key))
        for key in ("truncated", "truncated_bool", "was_truncated", "is_truncated")
    )
    length_finished = finish_reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "token_limit",
        "truncated",
    }
    max_token_hit = completion_tokens is not None and completion_tokens >= max_tokens
    truncated = bool(explicit_truncated or length_finished or max_token_hit)

    tags: list[str] = []
    if prediction is None:
        tags.append("missing_prediction")
    if no_box:
        tags.append("no_box")
    if multi_box:
        tags.append("multi_box")
    if malformed_boxed:
        tags.append("malformed_boxed")
    if public_metric_only_false_gain:
        tags.append("public_metric_only_false_gain")
    if strict_metric_only_false_positive:
        tags.append("strict_metric_only_false_positive")
    if truncated:
        tags.append("truncated")
    if not exact_one_closed_boxed:
        tags.append("not_exactly_one_closed_boxed")

    return {
        "id": solution["id"],
        "family": solution["family"],
        "source_role": solution.get("source_role", ""),
        "answer": answer,
        "public_final": public_final,
        "strict_final_boxed": final_boxed,
        "strict_final_canonical": canonical_answer(final_boxed) if final_boxed != "NOT_FOUND" else "NOT_FOUND",
        "public_correct": bool(public_correct),
        "strict_correct": bool(strict_correct),
        "boxed_marker_count": boxed_marker_count,
        "closed_boxed_count": len(closed_boxed_answers),
        "exact_one_closed_boxed": bool(exact_one_closed_boxed),
        "no_box": bool(no_box),
        "multi_box": bool(multi_box),
        "malformed_boxed": bool(malformed_boxed),
        "public_metric_only_false_gain": bool(public_metric_only_false_gain),
        "strict_metric_only_false_positive": bool(strict_metric_only_false_positive),
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens if completion_tokens is not None else "",
        "truncated": bool(truncated),
        "error_tags": ";".join(tags),
    }


def score_predictions(
    solutions: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    return [score_one(solution, predictions.get(solution["id"]), max_tokens=max_tokens) for solution in solutions]


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(scores),
        "public_correct": sum(1 for row in scores if row["public_correct"]),
        "strict_correct": sum(1 for row in scores if row["strict_correct"]),
        "exact_one_closed_boxed": sum(1 for row in scores if row["exact_one_closed_boxed"]),
        "no_box": sum(1 for row in scores if row["no_box"]),
        "multi_box": sum(1 for row in scores if row["multi_box"]),
        "malformed_boxed": sum(1 for row in scores if row["malformed_boxed"]),
        "public_metric_only_false_gain": sum(1 for row in scores if row["public_metric_only_false_gain"]),
        "strict_metric_only_false_positive": sum(
            1 for row in scores if row["strict_metric_only_false_positive"]
        ),
        "truncated": sum(1 for row in scores if row["truncated"]),
        "by_family": {},
    }
    for family in sorted({str(row["family"]) for row in scores}):
        family_rows = [row for row in scores if row["family"] == family]
        rows_count = len(family_rows)
        summary["by_family"][family] = {
            "rows": rows_count,
            "public_correct": sum(1 for row in family_rows if row["public_correct"]),
            "strict_correct": sum(1 for row in family_rows if row["strict_correct"]),
            "strict_accuracy": (
                sum(1 for row in family_rows if row["strict_correct"]) / rows_count if rows_count else 0.0
            ),
            "exact_one_closed_boxed": sum(1 for row in family_rows if row["exact_one_closed_boxed"]),
            "public_metric_only_false_gain": sum(
                1 for row in family_rows if row["public_metric_only_false_gain"]
            ),
            "strict_metric_only_false_positive": sum(
                1 for row in family_rows if row["strict_metric_only_false_positive"]
            ),
            "truncated": sum(1 for row in family_rows if row["truncated"]),
        }
    return summary


def int_threshold(args: argparse.Namespace, name: str, profile: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    default = PROFILE_DEFAULTS[profile][name]
    if default is None:
        return 0
    return int(default)


def optional_int_threshold(args: argparse.Namespace, name: str, profile: str) -> int | None:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    default = PROFILE_DEFAULTS[profile][name]
    return None if default is None else int(default)


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    profile = str(args.profile)
    return Thresholds(
        expected_rows=optional_int_threshold(args, "expected_rows", profile),
        min_bit_gain=int_threshold(args, "min_bit_gain", profile),
        min_equation_gain=int_threshold(args, "min_equation_gain", profile),
        min_total_gain=int_threshold(args, "min_total_gain", profile),
        min_total_correct=optional_int_threshold(args, "min_total_correct", profile),
        max_any_regressions=int_threshold(args, "max_any_regressions", profile),
        max_weak_regressions=int_threshold(args, "max_weak_regressions", profile),
        max_protected_regressions=int_threshold(args, "max_protected_regressions", profile),
        max_candidate_format_failures=int_threshold(args, "max_candidate_format_failures", profile),
        max_candidate_public_false_gains=int_threshold(args, "max_candidate_public_false_gains", profile),
        max_candidate_strict_false_public=int_threshold(args, "max_candidate_strict_false_public", profile),
        max_candidate_truncated=int_threshold(args, "max_candidate_truncated", profile),
    )


def compare_scores(
    solutions: list[dict[str, str]],
    baseline_scores: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    thresholds: Thresholds,
) -> tuple[bool, list[str], dict[str, Any], list[dict[str, Any]]]:
    baseline_by_id = {row["id"]: row for row in baseline_scores}
    candidate_by_id = {row["id"]: row for row in candidate_scores}
    solution_by_id = {row["id"]: row for row in solutions}

    row_deltas: list[dict[str, Any]] = []
    for row_id, solution in solution_by_id.items():
        baseline = baseline_by_id[row_id]
        candidate = candidate_by_id[row_id]
        improved = bool(candidate["strict_correct"] and not baseline["strict_correct"])
        regressed = bool(baseline["strict_correct"] and not candidate["strict_correct"])
        row_deltas.append(
            {
                "id": row_id,
                "family": solution["family"],
                "source_role": solution.get("source_role", ""),
                "answer": solution["answer"],
                "baseline_public_final": baseline["public_final"],
                "baseline_strict_final_boxed": baseline["strict_final_boxed"],
                "baseline_public_correct": baseline["public_correct"],
                "baseline_strict_correct": baseline["strict_correct"],
                "candidate_public_final": candidate["public_final"],
                "candidate_strict_final_boxed": candidate["strict_final_boxed"],
                "candidate_public_correct": candidate["public_correct"],
                "candidate_strict_correct": candidate["strict_correct"],
                "improved": improved,
                "regressed": regressed,
                "candidate_exact_one_closed_boxed": candidate["exact_one_closed_boxed"],
                "candidate_no_box": candidate["no_box"],
                "candidate_multi_box": candidate["multi_box"],
                "candidate_malformed_boxed": candidate["malformed_boxed"],
                "candidate_public_metric_only_false_gain": candidate["public_metric_only_false_gain"],
                "candidate_strict_metric_only_false_positive": candidate[
                    "strict_metric_only_false_positive"
                ],
                "candidate_truncated": candidate["truncated"],
                "candidate_finish_reason": candidate["finish_reason"],
                "candidate_completion_tokens": candidate["completion_tokens"],
                "candidate_error_tags": candidate["error_tags"],
            }
        )

    baseline_summary = summarize_scores(baseline_scores)
    candidate_summary = summarize_scores(candidate_scores)
    family_deltas: dict[str, dict[str, Any]] = {}
    for family in sorted({row["family"] for row in solutions}):
        family_rows = [row for row in row_deltas if row["family"] == family]
        baseline_public_correct = sum(1 for row in family_rows if row["baseline_public_correct"])
        baseline_correct = sum(1 for row in family_rows if row["baseline_strict_correct"])
        candidate_public_correct = sum(1 for row in family_rows if row["candidate_public_correct"])
        candidate_correct = sum(1 for row in family_rows if row["candidate_strict_correct"])
        family_deltas[family] = {
            "rows": len(family_rows),
            "baseline_public_correct": baseline_public_correct,
            "baseline_strict_correct": baseline_correct,
            "candidate_public_correct": candidate_public_correct,
            "candidate_strict_correct": candidate_correct,
            "public_correct_delta": candidate_public_correct - baseline_public_correct,
            "strict_correct_delta": candidate_correct - baseline_correct,
            "improved_rows": sum(1 for row in family_rows if row["improved"]),
            "regressed_rows": sum(1 for row in family_rows if row["regressed"]),
        }

    total_gain = int(candidate_summary["strict_correct"] - baseline_summary["strict_correct"])
    bit_gain = int(family_deltas.get("bit_manipulation", {}).get("strict_correct_delta", 0))
    equation_gain = int(family_deltas.get("equation_transform", {}).get("strict_correct_delta", 0))
    any_regressions = sum(1 for row in row_deltas if row["regressed"])
    weak_regressions = sum(
        1 for row in row_deltas if row["regressed"] and row["family"] in WEAK_FAMILIES
    )
    protected_regressions = sum(
        1 for row in row_deltas if row["regressed"] and row["family"] not in WEAK_FAMILIES
    )
    candidate_format_failures = sum(
        1
        for row in row_deltas
        if not row["candidate_exact_one_closed_boxed"]
        or row["candidate_no_box"]
        or row["candidate_multi_box"]
        or row["candidate_malformed_boxed"]
    )
    candidate_public_false_gains = sum(
        1 for row in row_deltas if row["candidate_public_metric_only_false_gain"]
    )
    candidate_strict_false_public = sum(
        1 for row in row_deltas if row["candidate_strict_metric_only_false_positive"]
    )
    candidate_truncated = sum(1 for row in row_deltas if row["candidate_truncated"])

    blockers: list[str] = []
    baseline_identity = {
        "required": bool(
            thresholds.expected_rows == V291_PUBLIC_REFERENCE["full947_rows"]
            and thresholds.min_total_correct is not None
        ),
        "reference_source": V291_PUBLIC_REFERENCE["source"],
        "reference_note": V291_PUBLIC_REFERENCE["note"],
        "require_strict_clean_baseline": True,
        "rationale": (
            "Full947 promotion protects the known V291 public baseline by comparing strict row deltas. "
            "Therefore the baseline CSV must be strict-clean: public identity and strict identity must both "
            "match the V291 reference before candidate gains are trusted."
        ),
    }
    if (
        thresholds.expected_rows == V291_PUBLIC_REFERENCE["full947_rows"]
        and thresholds.min_total_correct is not None
    ):
        protected_baseline_strict_correct = sum(
            int(values.get("baseline_strict_correct", 0))
            for family, values in family_deltas.items()
            if family not in WEAK_FAMILIES
        )
        protected_baseline_public_correct = sum(
            int(values.get("baseline_public_correct", 0))
            for family, values in family_deltas.items()
            if family not in WEAK_FAMILIES
        )
        expected_baseline = {
            "full947": V291_PUBLIC_REFERENCE["full947_correct"],
            "bit_manipulation": V291_PUBLIC_REFERENCE["bit_manipulation_correct"],
            "equation_transform": V291_PUBLIC_REFERENCE["equation_transform_correct"],
            "protected": V291_PUBLIC_REFERENCE["protected_families_correct"],
        }
        observed_baseline_public = {
            "full947": int(baseline_summary["public_correct"]),
            "bit_manipulation": int(family_deltas.get("bit_manipulation", {}).get("baseline_public_correct", 0)),
            "equation_transform": int(family_deltas.get("equation_transform", {}).get("baseline_public_correct", 0)),
            "protected": protected_baseline_public_correct,
        }
        observed_baseline_strict = {
            "full947": int(baseline_summary["strict_correct"]),
            "bit_manipulation": int(family_deltas.get("bit_manipulation", {}).get("baseline_strict_correct", 0)),
            "equation_transform": int(family_deltas.get("equation_transform", {}).get("baseline_strict_correct", 0)),
            "protected": protected_baseline_strict_correct,
        }
        public_mismatches: dict[str, dict[str, int]] = {}
        strict_clean_mismatches: dict[str, dict[str, int]] = {}
        for name, expected_value in expected_baseline.items():
            observed_public_value = observed_baseline_public[name]
            observed_strict_value = observed_baseline_strict[name]
            if observed_public_value != expected_value:
                public_mismatches[name] = {"observed": observed_public_value, "expected": expected_value}
                blockers.append(
                    f"baseline_public_identity_unverified:{name}:{observed_public_value}!={expected_value}"
                )
            if observed_strict_value != expected_value:
                strict_clean_mismatches[name] = {"observed": observed_strict_value, "expected": expected_value}
                blockers.append(
                    f"baseline_strict_clean_identity_unverified:{name}:{observed_strict_value}!={expected_value}"
                )
        baseline_identity.update(
            {
                "expected_public_reference": expected_baseline,
                "observed_public": observed_baseline_public,
                "observed_strict": observed_baseline_strict,
                "public_identity_pass": not public_mismatches,
                "strict_clean_identity_pass": not strict_clean_mismatches,
                "public_mismatches": public_mismatches,
                "strict_clean_mismatches": strict_clean_mismatches,
            }
        )
    if thresholds.expected_rows is not None and len(solutions) != thresholds.expected_rows:
        blockers.append(f"row_count_mismatch:{len(solutions)}!={thresholds.expected_rows}")
    if bit_gain < thresholds.min_bit_gain:
        blockers.append(f"bit_gain_below_threshold:{bit_gain}<{thresholds.min_bit_gain}")
    if equation_gain < thresholds.min_equation_gain:
        blockers.append(f"equation_gain_below_threshold:{equation_gain}<{thresholds.min_equation_gain}")
    if total_gain < thresholds.min_total_gain:
        blockers.append(f"total_gain_below_threshold:{total_gain}<{thresholds.min_total_gain}")
    if thresholds.min_total_correct is not None and candidate_summary["strict_correct"] < thresholds.min_total_correct:
        blockers.append(
            "candidate_total_correct_below_threshold:"
            f"{candidate_summary['strict_correct']}<{thresholds.min_total_correct}"
        )
    if any_regressions > thresholds.max_any_regressions:
        blockers.append(f"any_regressions_above_threshold:{any_regressions}>{thresholds.max_any_regressions}")
    if weak_regressions > thresholds.max_weak_regressions:
        blockers.append(f"weak_regressions_above_threshold:{weak_regressions}>{thresholds.max_weak_regressions}")
    if protected_regressions > thresholds.max_protected_regressions:
        blockers.append(
            f"protected_regressions_above_threshold:{protected_regressions}>{thresholds.max_protected_regressions}"
        )
    if candidate_format_failures > thresholds.max_candidate_format_failures:
        blockers.append(
            "candidate_format_failures_above_threshold:"
            f"{candidate_format_failures}>{thresholds.max_candidate_format_failures}"
        )
    if candidate_public_false_gains > thresholds.max_candidate_public_false_gains:
        blockers.append(
            "candidate_public_metric_only_false_gains_above_threshold:"
            f"{candidate_public_false_gains}>{thresholds.max_candidate_public_false_gains}"
        )
    if candidate_strict_false_public > thresholds.max_candidate_strict_false_public:
        blockers.append(
            "candidate_strict_metric_only_false_positive_above_threshold:"
            f"{candidate_strict_false_public}>{thresholds.max_candidate_strict_false_public}"
        )
    if candidate_truncated > thresholds.max_candidate_truncated:
        blockers.append(f"candidate_truncated_above_threshold:{candidate_truncated}>{thresholds.max_candidate_truncated}")

    comparison = {
        "decision": "pass" if not blockers else "fail",
        "generated_at_utc": utc_now(),
        "thresholds": thresholds.__dict__,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "family_deltas": family_deltas,
        "baseline_identity": baseline_identity,
        "gains": {
            "total_strict_correct_gain": total_gain,
            "bit_manipulation_strict_correct_gain": bit_gain,
            "equation_transform_strict_correct_gain": equation_gain,
        },
        "regressions": {
            "any": any_regressions,
            "weak_families": weak_regressions,
            "protected_families": protected_regressions,
        },
        "candidate_blocker_counts": {
            "format_failures": candidate_format_failures,
            "public_metric_only_false_gains": candidate_public_false_gains,
            "strict_metric_only_false_positives": candidate_strict_false_public,
            "truncated": candidate_truncated,
        },
        "blockers": blockers,
    }
    return not blockers, blockers, comparison, row_deltas


def write_transfer_outputs(output_dir: Path, comparison: dict[str, Any], row_deltas: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "kg1_v1241_bit_equation_transfer_gate_report.json"
    deltas_path = output_dir / "kg1_v1241_bit_equation_transfer_gate_row_deltas.csv"
    summary_path = output_dir / "KG1_V1241_BIT_EQUATION_TRANSFER_GATE_SUMMARY.md"

    write_json(report_path, comparison)
    fields = [
        "id",
        "family",
        "source_role",
        "answer",
        "baseline_public_final",
        "baseline_strict_final_boxed",
        "baseline_public_correct",
        "baseline_strict_correct",
        "candidate_public_final",
        "candidate_strict_final_boxed",
        "candidate_public_correct",
        "candidate_strict_correct",
        "improved",
        "regressed",
        "candidate_exact_one_closed_boxed",
        "candidate_no_box",
        "candidate_multi_box",
        "candidate_malformed_boxed",
        "candidate_public_metric_only_false_gain",
        "candidate_strict_metric_only_false_positive",
        "candidate_truncated",
        "candidate_finish_reason",
        "candidate_completion_tokens",
        "candidate_error_tags",
    ]
    write_csv(deltas_path, row_deltas, fields)

    status = comparison["decision"]
    blockers = comparison.get("blockers", [])
    lines = [
        "# KG1 V1241 Bit/Equation Transfer Gate Summary",
        "",
        f"Decision: `{status}`",
        f"Generated at UTC: `{comparison['generated_at_utc']}`",
        "",
        "## Strict Gains",
        "",
        f"- Total strict gain: `{comparison['gains']['total_strict_correct_gain']}`",
        f"- Bit strict gain: `{comparison['gains']['bit_manipulation_strict_correct_gain']}`",
        f"- Equation strict gain: `{comparison['gains']['equation_transform_strict_correct_gain']}`",
        "",
        "## Blocker Counts",
        "",
        f"- Any regressions: `{comparison['regressions']['any']}`",
        f"- Weak-family regressions: `{comparison['regressions']['weak_families']}`",
        f"- Protected-family regressions: `{comparison['regressions']['protected_families']}`",
        f"- Candidate format failures: `{comparison['candidate_blocker_counts']['format_failures']}`",
        (
            "- Candidate public-metric-only false gains: "
            f"`{comparison['candidate_blocker_counts']['public_metric_only_false_gains']}`"
        ),
        (
            "- Candidate strict-metric-only false positives: "
            f"`{comparison['candidate_blocker_counts']['strict_metric_only_false_positives']}`"
        ),
        f"- Candidate truncated rows: `{comparison['candidate_blocker_counts']['truncated']}`",
        "",
        "## Baseline Identity",
        "",
        f"- Required: `{comparison.get('baseline_identity', {}).get('required')}`",
        f"- Public identity pass: `{comparison.get('baseline_identity', {}).get('public_identity_pass', 'not_applicable')}`",
        f"- Strict-clean identity pass: `{comparison.get('baseline_identity', {}).get('strict_clean_identity_pass', 'not_applicable')}`",
        (
            "- Rationale: full947 promotion requires a strict-clean V291 baseline so strict row-delta "
            "regression checks protect the known public 823/947 baseline."
        ),
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Report JSON: `{report_path}`",
            f"- Row deltas CSV: `{deltas_path}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare(args: argparse.Namespace) -> int:
    if args.solution_csv is None or args.baseline_predictions is None or args.candidate_predictions is None:
        raise ValueError("--solution-csv, --baseline-predictions, and --candidate-predictions are required")
    thresholds = thresholds_from_args(args)
    solutions = normalize_solution_csv(args.solution_csv)
    baseline_predictions, _ = normalize_predictions_csv(
        args.baseline_predictions,
        label="baseline",
        allow_prediction_as_raw_output=bool(args.allow_prediction_as_raw_output),
    )
    candidate_predictions, _ = normalize_predictions_csv(
        args.candidate_predictions,
        label="candidate",
        allow_prediction_as_raw_output=bool(args.allow_prediction_as_raw_output),
    )
    baseline_scores = score_predictions(solutions, baseline_predictions, max_tokens=int(args.max_tokens))
    candidate_scores = score_predictions(solutions, candidate_predictions, max_tokens=int(args.max_tokens))
    passed, _blockers, comparison, row_deltas = compare_scores(
        solutions,
        baseline_scores,
        candidate_scores,
        thresholds,
    )
    output_dir = args.output_dir or (ROOT / "artifacts" / "v1241_bit_equation_transfer_gate_report")
    comparison["inputs"] = {
        "solution_csv": str(args.solution_csv),
        "solution_sha256": sha256_file(args.solution_csv),
        "baseline_predictions": str(args.baseline_predictions),
        "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
        "candidate_predictions": str(args.candidate_predictions),
        "candidate_predictions_sha256": sha256_file(args.candidate_predictions),
        "profile": args.profile,
        "max_tokens": int(args.max_tokens),
    }
    write_transfer_outputs(output_dir, comparison, row_deltas)
    print(json.dumps({"decision": comparison["decision"], "blockers": comparison["blockers"]}, indent=2))
    return 0 if passed else 2


def run_baseline_identity_probe(args: argparse.Namespace) -> int:
    if args.solution_csv is None or args.baseline_predictions is None:
        raise ValueError("--baseline-identity-probe requires --solution-csv and --baseline-predictions")
    thresholds = thresholds_from_args(args)
    solutions = normalize_solution_csv(args.solution_csv)
    baseline_predictions, _ = normalize_predictions_csv(
        args.baseline_predictions,
        label="baseline",
        allow_prediction_as_raw_output=bool(args.allow_prediction_as_raw_output),
    )
    baseline_scores = score_predictions(solutions, baseline_predictions, max_tokens=int(args.max_tokens))
    _passed, comparison_blockers, comparison, row_deltas = compare_scores(
        solutions,
        baseline_scores,
        baseline_scores,
        thresholds,
    )
    baseline_identity = comparison.get("baseline_identity", {})
    probe_blockers: list[str] = []
    if len(solutions) != V291_PUBLIC_REFERENCE["full947_rows"]:
        probe_blockers.append(f"row_count_mismatch:{len(solutions)}!={V291_PUBLIC_REFERENCE['full947_rows']}")
    if not baseline_identity.get("required"):
        probe_blockers.append("baseline_identity_not_required_profile_use_full947_089_or_full947_090")
    if baseline_identity.get("public_identity_pass") is not True:
        probe_blockers.append("baseline_public_identity_not_pass")
    if baseline_identity.get("strict_clean_identity_pass") is not True:
        probe_blockers.append("baseline_strict_clean_identity_not_pass")
    decision = "pass_baseline_strict_clean_identity_probe" if not probe_blockers else "fail"
    output_dir = args.output_dir or (ROOT / "artifacts" / "v1241_baseline_identity_probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "kg1_v1241_baseline_identity_probe_v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "probe_blockers": probe_blockers,
        "comparison_blockers_ignored_for_probe": comparison_blockers,
        "baseline": comparison["baseline"],
        "baseline_identity": baseline_identity,
        "family_deltas": comparison["family_deltas"],
        "inputs": {
            "solution_csv": str(args.solution_csv),
            "solution_sha256": sha256_file(args.solution_csv),
            "baseline_predictions": str(args.baseline_predictions),
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
            "profile": args.profile,
            "max_tokens": int(args.max_tokens),
        },
        "not_authorized": [
            "candidate_promotion",
            "adapter_package",
            "kaggle_submit",
            "score_claim",
        ],
    }
    report_path = output_dir / "kg1_v1241_baseline_identity_probe_report.json"
    row_deltas_path = output_dir / "kg1_v1241_baseline_identity_probe_row_deltas.csv"
    summary_path = output_dir / "KG1_V1241_BASELINE_IDENTITY_PROBE.md"
    write_json(report_path, report)
    write_csv(
        row_deltas_path,
        row_deltas,
        [
            "id",
            "family",
            "answer",
            "baseline_public_final",
            "baseline_strict_final_boxed",
            "baseline_public_correct",
            "baseline_strict_correct",
            "candidate_public_correct",
            "candidate_strict_correct",
            "candidate_exact_one_closed_boxed",
            "candidate_truncated",
        ],
    )
    lines = [
        "# KG1 V1241 Baseline Identity Probe",
        "",
        f"Decision: `{decision}`",
        "",
        "This CPU-only probe checks whether the full947 baseline CSV is strict-clean against the V291/086 public reference.",
        "It does not evaluate a candidate and does not authorize training, packaging, submission, or score claims.",
        "",
        "## Baseline Identity",
        "",
        f"- Public identity pass: `{baseline_identity.get('public_identity_pass')}`",
        f"- Strict-clean identity pass: `{baseline_identity.get('strict_clean_identity_pass')}`",
        f"- Observed public: `{baseline_identity.get('observed_public')}`",
        f"- Observed strict: `{baseline_identity.get('observed_strict')}`",
        "",
        "## Probe Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in probe_blockers) if probe_blockers else lines.append("- none")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Report JSON: `{report_path}`",
            f"- Row deltas CSV: `{row_deltas_path}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "blockers": probe_blockers, "report": str(report_path)}, indent=2))
    return 0 if not probe_blockers else 2


def make_self_test_solution() -> list[dict[str, str]]:
    return [
        {"id": "self_bit_a", "prompt": "bit a", "answer": "1010", "family": "bit_manipulation"},
        {"id": "self_bit_b", "prompt": "bit b", "answer": "1111", "family": "bit_manipulation"},
        {"id": "self_equation_a", "prompt": "equation a", "answer": "42", "family": "equation_transform"},
        {"id": "self_equation_b", "prompt": "equation b", "answer": "17", "family": "equation_transform"},
        {"id": "self_gravity", "prompt": "gravity", "answer": "24.64", "family": "gravity_constant"},
        {"id": "self_unit", "prompt": "unit", "answer": "12", "family": "unit_conversion"},
    ]


def boxed_prediction(row_id: str, value: str, *, finish_reason: str = "stop", tokens: int = 8) -> dict[str, str]:
    return {
        "id": row_id,
        "raw_output": f"</think>\n\\boxed{{{value}}}",
        "finish_reason": finish_reason,
        "completion_tokens": str(tokens),
    }


def public_only_prediction(row_id: str, value: str, *, finish_reason: str = "stop", tokens: int = 8) -> dict[str, str]:
    return {
        "id": row_id,
        "raw_output": f"Final answer: {value}",
        "finish_reason": finish_reason,
        "completion_tokens": str(tokens),
    }


def make_full947_identity_solution() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_rows(family: str, count: int, prefix: str) -> None:
        for index in range(count):
            answer = str(100000 + len(rows))
            rows.append(
                {
                    "id": f"{prefix}_{index:03d}",
                    "prompt": f"{family} identity fixture {index}",
                    "answer": answer,
                    "family": family,
                }
            )

    add_rows("bit_manipulation", V291_PUBLIC_REFERENCE["bit_manipulation_rows"], "full_bit")
    add_rows("equation_transform", V291_PUBLIC_REFERENCE["equation_transform_rows"], "full_eq")
    protected_each = V291_PUBLIC_REFERENCE["protected_families_rows"] // 4
    add_rows("gravity_constant", protected_each, "full_gravity")
    add_rows("numeral_system", protected_each, "full_numeral")
    add_rows("text_encryption", protected_each, "full_cipher")
    add_rows("unit_conversion", protected_each, "full_unit")
    if len(rows) != V291_PUBLIC_REFERENCE["full947_rows"]:
        raise AssertionError(f"full947 fixture row count mismatch: {len(rows)}")
    return rows


def make_full947_identity_predictions(
    solution: list[dict[str, str]],
    *,
    bit_correct: int,
    equation_correct: int,
    protected_correct: int,
    strict_clean: bool,
) -> list[dict[str, str]]:
    limits = {
        "bit_manipulation": bit_correct,
        "equation_transform": equation_correct,
        "protected": protected_correct,
    }
    seen = {"bit_manipulation": 0, "equation_transform": 0, "protected": 0}
    predictions: list[dict[str, str]] = []
    for row in solution:
        family = row["family"]
        bucket = family if family in WEAK_FAMILIES else "protected"
        seen[bucket] += 1
        is_correct = seen[bucket] <= limits[bucket]
        value = row["answer"] if is_correct else str(int(row["answer"]) + 9999)
        if strict_clean:
            predictions.append(boxed_prediction(row["id"], value))
        else:
            predictions.append(public_only_prediction(row["id"], value))
    return predictions


def write_self_test_csvs(
    output_dir: Path,
    case: str,
    solution: list[dict[str, str]],
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
) -> tuple[Path, Path, Path]:
    case_dir = output_dir / case
    solution_csv = case_dir / "solution.csv"
    baseline_csv = case_dir / "baseline.csv"
    candidate_csv = case_dir / "candidate.csv"
    write_csv(solution_csv, solution, ["id", "prompt", "answer", "family"])
    write_csv(baseline_csv, baseline, ["id", "raw_output", "finish_reason", "completion_tokens"])
    write_csv(candidate_csv, candidate, ["id", "raw_output", "finish_reason", "completion_tokens"])
    return solution_csv, baseline_csv, candidate_csv


def run_case(
    output_dir: Path,
    case: str,
    solution: list[dict[str, str]],
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
    *,
    expect_pass: bool,
    profile: str = "custom",
) -> dict[str, Any]:
    solution_csv, baseline_csv, candidate_csv = write_self_test_csvs(
        output_dir, case, solution, baseline, candidate
    )
    class Args:
        pass

    args = Args()
    args.solution_csv = solution_csv
    args.baseline_predictions = baseline_csv
    args.candidate_predictions = candidate_csv
    args.output_dir = output_dir / case / "report"
    args.profile = profile
    args.max_tokens = 64
    args.allow_prediction_as_raw_output = False
    args.expected_rows = None
    args.min_bit_gain = 1
    args.min_equation_gain = 1
    args.min_total_gain = 2
    args.min_total_correct = None
    args.max_any_regressions = None
    args.max_weak_regressions = None
    args.max_protected_regressions = None
    args.max_candidate_format_failures = None
    args.max_candidate_public_false_gains = None
    args.max_candidate_strict_false_public = None
    args.max_candidate_truncated = None

    exit_code = run_compare(args)  # writes report artifacts
    report = read_json(args.output_dir / "kg1_v1241_bit_equation_transfer_gate_report.json")
    passed = exit_code == 0
    if passed != expect_pass:
        raise AssertionError(f"self-test case {case} expected pass={expect_pass}, got pass={passed}: {report}")
    return {
        "case": case,
        "expected_pass": expect_pass,
        "observed_pass": passed,
        "blockers": report.get("blockers", []),
        "report": str(args.output_dir / "kg1_v1241_bit_equation_transfer_gate_report.json"),
    }


def run_baseline_probe_case(
    output_dir: Path,
    case: str,
    solution: list[dict[str, str]],
    baseline: list[dict[str, str]],
    *,
    expect_pass: bool,
    profile: str = "full947_089",
) -> dict[str, Any]:
    case_dir = output_dir / case
    solution_csv = case_dir / "solution.csv"
    baseline_csv = case_dir / "baseline.csv"
    write_csv(solution_csv, solution, ["id", "prompt", "answer", "family"])
    write_csv(baseline_csv, baseline, ["id", "raw_output", "finish_reason", "completion_tokens"])

    class Args:
        pass

    args = Args()
    args.solution_csv = solution_csv
    args.baseline_predictions = baseline_csv
    args.candidate_predictions = None
    args.output_dir = case_dir / "probe"
    args.profile = profile
    args.max_tokens = 64
    args.allow_prediction_as_raw_output = False
    args.expected_rows = None
    args.min_bit_gain = None
    args.min_equation_gain = None
    args.min_total_gain = None
    args.min_total_correct = None
    args.max_any_regressions = None
    args.max_weak_regressions = None
    args.max_protected_regressions = None
    args.max_candidate_format_failures = None
    args.max_candidate_public_false_gains = None
    args.max_candidate_strict_false_public = None
    args.max_candidate_truncated = None

    exit_code = run_baseline_identity_probe(args)
    report_path = args.output_dir / "kg1_v1241_baseline_identity_probe_report.json"
    report = read_json(report_path)
    passed = exit_code == 0
    if passed != expect_pass:
        raise AssertionError(f"baseline probe case {case} expected pass={expect_pass}, got pass={passed}: {report}")
    return {
        "case": case,
        "expected_pass": expect_pass,
        "observed_pass": passed,
        "blockers": report.get("probe_blockers", []),
        "report": str(report_path),
    }


def run_self_test(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solution = make_self_test_solution()
    baseline = [
        boxed_prediction("self_bit_a", "0000"),
        boxed_prediction("self_bit_b", "1111"),
        boxed_prediction("self_equation_a", "0"),
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "12"),
    ]
    candidate_pass = [
        boxed_prediction("self_bit_a", "1010"),
        boxed_prediction("self_bit_b", "1111"),
        boxed_prediction("self_equation_a", "42"),
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "12"),
    ]
    candidate_protected_regression = [
        boxed_prediction("self_bit_a", "1010"),
        boxed_prediction("self_bit_b", "1111"),
        boxed_prediction("self_equation_a", "42"),
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "13"),
    ]
    candidate_public_false_gain = [
        {"id": "self_bit_a", "raw_output": "Final answer: 1010", "finish_reason": "stop", "completion_tokens": "5"},
        boxed_prediction("self_bit_b", "1111"),
        boxed_prediction("self_equation_a", "42"),
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "12"),
    ]
    candidate_multi_box = [
        boxed_prediction("self_bit_a", "1010"),
        boxed_prediction("self_bit_b", "1111"),
        {"id": "self_equation_a", "raw_output": "\\boxed{1}\n\\boxed{42}", "finish_reason": "stop", "completion_tokens": "8"},
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "12"),
    ]
    candidate_truncated = [
        boxed_prediction("self_bit_a", "1010"),
        boxed_prediction("self_bit_b", "1111"),
        boxed_prediction("self_equation_a", "42", finish_reason="length"),
        boxed_prediction("self_equation_b", "17"),
        boxed_prediction("self_gravity", "24.64"),
        boxed_prediction("self_unit", "12"),
    ]
    symbolic_solution = solution + [
        {"id": "self_symbolic", "prompt": "symbolic", "answer": "{a}", "family": "text_encryption"},
    ]
    symbolic_baseline = baseline + [boxed_prediction("self_symbolic", "z")]
    candidate_strict_false_public = candidate_pass + [
        {
            "id": "self_symbolic",
            "raw_output": "</think>\n\\boxed{\\{a\\}}",
            "finish_reason": "stop",
            "completion_tokens": "8",
        }
    ]
    full947_solution = make_full947_identity_solution()
    full947_clean_baseline = make_full947_identity_predictions(
        full947_solution,
        bit_correct=V291_PUBLIC_REFERENCE["bit_manipulation_correct"],
        equation_correct=V291_PUBLIC_REFERENCE["equation_transform_correct"],
        protected_correct=V291_PUBLIC_REFERENCE["protected_families_correct"],
        strict_clean=True,
    )
    full947_dirty_baseline = make_full947_identity_predictions(
        full947_solution,
        bit_correct=V291_PUBLIC_REFERENCE["bit_manipulation_correct"],
        equation_correct=V291_PUBLIC_REFERENCE["equation_transform_correct"],
        protected_correct=V291_PUBLIC_REFERENCE["protected_families_correct"],
        strict_clean=False,
    )
    full947_candidate_089 = make_full947_identity_predictions(
        full947_solution,
        bit_correct=V291_PUBLIC_REFERENCE["bit_manipulation_correct"] + 10,
        equation_correct=V291_PUBLIC_REFERENCE["equation_transform_correct"] + 10,
        protected_correct=V291_PUBLIC_REFERENCE["protected_families_correct"],
        strict_clean=True,
    )

    results = [
        run_case(output_dir, "pass_real_gain", solution, baseline, candidate_pass, expect_pass=True),
        run_case(output_dir, "fail_row_count_profile_check", solution, baseline, candidate_pass, expect_pass=False, profile="tiny"),
        run_case(
            output_dir,
            "fail_protected_regression",
            solution,
            baseline,
            candidate_protected_regression,
            expect_pass=False,
        ),
        run_case(
            output_dir,
            "fail_public_metric_only_false_gain",
            solution,
            baseline,
            candidate_public_false_gain,
            expect_pass=False,
        ),
        run_case(output_dir, "fail_multi_box", solution, baseline, candidate_multi_box, expect_pass=False),
        run_case(output_dir, "fail_truncated", solution, baseline, candidate_truncated, expect_pass=False),
        run_case(
            output_dir,
            "fail_strict_metric_only_false_positive",
            symbolic_solution,
            symbolic_baseline,
            candidate_strict_false_public,
            expect_pass=False,
        ),
        run_case(
            output_dir,
            "pass_full947_clean_baseline_identity",
            full947_solution,
            full947_clean_baseline,
            full947_candidate_089,
            expect_pass=True,
            profile="full947_089",
        ),
        run_case(
            output_dir,
            "fail_full947_dirty_baseline_identity",
            full947_solution,
            full947_dirty_baseline,
            full947_candidate_089,
            expect_pass=False,
            profile="full947_089",
        ),
        run_baseline_probe_case(
            output_dir,
            "probe_pass_full947_clean_baseline_identity",
            full947_solution,
            full947_clean_baseline,
            expect_pass=True,
        ),
        run_baseline_probe_case(
            output_dir,
            "probe_fail_full947_dirty_baseline_identity",
            full947_solution,
            full947_dirty_baseline,
            expect_pass=False,
        ),
    ]
    manifest = {
        "decision": "pass_v1241_self_test",
        "generated_at_utc": utc_now(),
        "cases": results,
    }
    write_json(output_dir / "kg1_v1241_bit_equation_transfer_gate_selftest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-contract", action="store_true", help="Build V1241 contract artifacts from V1240.")
    parser.add_argument("--self-test", action="store_true", help="Run strict gate self-tests.")
    parser.add_argument(
        "--baseline-identity-probe",
        action="store_true",
        help="CPU-only check that a full947 baseline raw_output CSV is strict-clean against the V291/086 reference.",
    )
    parser.add_argument("--v1240-dir", type=Path, default=DEFAULT_V1240_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solution-csv", type=Path, default=None)
    parser.add_argument("--baseline-predictions", "--baseline-csv", type=Path, default=None)
    parser.add_argument("--candidate-predictions", "--candidate-csv", type=Path, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="tiny")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=int(OFFICIAL_INFERENCE_CONFIG["max_tokens"]))
    parser.add_argument(
        "--allow-prediction-as-raw-output",
        action="store_true",
        help="Debug only. Promotion gates require real raw_output.",
    )
    parser.add_argument("--min-bit-gain", type=int, default=None)
    parser.add_argument("--min-equation-gain", type=int, default=None)
    parser.add_argument("--min-total-gain", type=int, default=None)
    parser.add_argument("--min-total-correct", type=int, default=None)
    parser.add_argument("--max-any-regressions", type=int, default=None)
    parser.add_argument("--max-weak-regressions", type=int, default=None)
    parser.add_argument("--max-protected-regressions", type=int, default=None)
    parser.add_argument("--max-candidate-format-failures", type=int, default=None)
    parser.add_argument("--max-candidate-public-false-gains", type=int, default=None)
    parser.add_argument("--max-candidate-strict-false-public", type=int, default=None)
    parser.add_argument("--max-candidate-truncated", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.self_test:
        output_dir = args.output_dir or DEFAULT_SELF_TEST_DIR
        manifest = run_self_test(output_dir)
        print(json.dumps({"decision": manifest["decision"], "output_dir": str(output_dir)}, indent=2))
        return 0
    if args.build_contract:
        output_dir = args.output_dir or DEFAULT_OUT_DIR
        manifest = build_contract(args.v1240_dir, output_dir)
        print(json.dumps({"decision": manifest["decision"], "output_dir": str(output_dir)}, indent=2))
        return 0
    if args.baseline_identity_probe:
        return run_baseline_identity_probe(args)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

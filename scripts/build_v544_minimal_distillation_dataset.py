#!/usr/bin/env python3
"""Build the V544 minimal adapter-transfer distillation dataset.

V544 is intentionally narrow: it converts the verified V543 CPU teacher signal
into short boxed-answer supervised rows plus baseline-correct replay. It does
not train, launch HF, package, submit, or claim submit safety. GPU is blocked
until tokenization/objective/eval gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import sha256_file  # noqa: E402


DEFAULT_V543_PREDICTIONS = (
    REPO_ROOT / "artifacts/v542_cpu_equation_solver_gate/v543_symbolic_queryop_on_v350_v516_strict/v543_integrated_predictions.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v544_minimal_distillation_dataset/20260517T_v544_cpu_gate"

SYSTEM_MESSAGE = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalized_answer(value: Any) -> str:
    return str(value or "").strip()


def baseline_prediction(row: dict[str, Any]) -> str:
    for key in ("v343_prediction", "stored_prediction", "prediction"):
        value = normalized_answer(row.get(key))
        if value:
            return value
    return ""


def teacher_prediction(row: dict[str, Any]) -> str:
    for key in ("v543_prediction", "v350_prediction", "v343_prediction", "prediction"):
        value = normalized_answer(row.get(key))
        if value:
            return value
    return ""


def source_rule(row: dict[str, Any]) -> str:
    for key in ("v543_source_rule", "v350_source_rule"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "baseline_replay"


def final_answer_suffix(answer: str) -> tuple[str, str]:
    """Return a submit-like final answer suffix that verifies label-free.

    Symbolic equation answers may contain literal braces or backslashes. Escaping
    those characters inside ``\boxed{}`` teaches a payload that the public
    extractor reads literally with the backslashes, which is not equivalent to
    the target answer. Prefer raw boxed output when it round-trips through the
    label-free extractor; otherwise fall back to the supported ``Final answer:``
    extraction path without a boxed wrapper.
    """

    value = normalized_answer(answer)
    boxed = f"\\boxed{{{value}}}"
    if verify_answer(value, extract_final_answer(boxed)):
        return boxed, "boxed_raw_label_free"
    return value, "unboxed_label_free_fallback"


def trace_for(row: dict[str, Any], answer: str, role: str) -> tuple[str, str]:
    family = str(row.get("family", ""))
    rule = source_rule(row)
    suffix, suffix_format = final_answer_suffix(answer)
    if role == "teacher_gain":
        if family == "bit_manipulation":
            tag = "bit_teacher"
        elif "symbolic_cryptarithm" in rule:
            tag = "symbolic_queryop_teacher"
        else:
            tag = "guarded_numeric_teacher"
        return f"RULE: {tag}; {rule}. Final answer: {suffix}", suffix_format
    if family == "bit_manipulation":
        return f"RULE: preserve_bit_replay. Final answer: {suffix}", suffix_format
    return f"RULE: preserve_equation_replay. Final answer: {suffix}", suffix_format


def make_row(
    *,
    base: dict[str, Any],
    new_id: str,
    answer: str,
    role: str,
    weight: float,
    repeat_index: int,
) -> dict[str, Any]:
    prompt = str(base.get("prompt", ""))
    assistant, final_answer_format = trace_for(base, answer, role)
    extracted = extract_final_answer(assistant)
    if not verify_answer(answer, extracted):
        raise RuntimeError(f"assistant target does not verify for {new_id}: {assistant!r}")
    metadata = {
        "schema_version": "kg1_v544_minimal_distillation_dataset_v1",
        "source_dataset": "v544_minimal_distillation_dataset",
        "source_row_id": str(base.get("id", "")),
        "source_rule": source_rule(base),
        "role": role,
        "repeat_index": repeat_index,
        "loss_weight": weight,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "gate_rows_used_for_training": False,
        "weak_prompt_source_only_teacher_distill": role == "teacher_gain",
        "raw_output_required_for_promotion": True,
        "final_answer_format": final_answer_format,
    }
    return {
        "id": new_id,
        "prompt": prompt,
        "answer": answer,
        "family": str(base.get("family", "")),
        "subcategory": role,
        "source": "v544_minimal_distillation_dataset",
        "source_dataset": "v544_minimal_distillation_dataset",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def summarize_jsonl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row.get("family", "")) for row in rows)
    role = Counter(str(row.get("metadata", {}).get("role", "")) for row in rows)
    final_answer_format = Counter(str(row.get("metadata", {}).get("final_answer_format", "")) for row in rows)
    original_ids = Counter(str(row.get("metadata", {}).get("source_row_id", "")) for row in rows)
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family.items())),
        "role_counts": dict(sorted(role.items())),
        "final_answer_format_counts": dict(sorted(final_answer_format.items())),
        "unique_source_row_ids": len(original_ids),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V544 MINIMAL DISTILLATION DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v543_predictions_csv =", args.v543_predictions_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("teacher_repeat =", args.teacher_repeat, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.v543_predictions_csv)
    by_id = {str(row.get("id", "")): row for row in rows}
    baseline_correct = [
        row
        for row in rows
        if verify_answer(str(row.get("answer", "")), baseline_prediction(row))
    ]
    teacher_gain = [
        row
        for row in rows
        if not verify_answer(str(row.get("answer", "")), baseline_prediction(row))
        and verify_answer(str(row.get("answer", "")), teacher_prediction(row))
    ]
    teacher_gain_ids = sorted(str(row["id"]) for row in teacher_gain)
    print("baseline_correct_count =", len(baseline_correct), flush=True)
    print("teacher_gain_count =", len(teacher_gain), flush=True)
    print("teacher_gain_ids =", json.dumps(teacher_gain_ids), flush=True)
    if len(teacher_gain) != args.expected_teacher_gain_count:
        raise RuntimeError(f"expected {args.expected_teacher_gain_count} teacher gains, got {len(teacher_gain)}")
    protected = by_id.get(args.protected_id)
    if protected is None:
        raise RuntimeError(f"protected id missing: {args.protected_id}")
    if not verify_answer(args.protected_answer, baseline_prediction(protected)):
        raise RuntimeError(f"protected id baseline prediction is not preserved: {args.protected_id}")

    train_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in teacher_gain:
        answer = teacher_prediction(row)
        for repeat in range(args.teacher_repeat):
            train_rows.append(
                make_row(
                    base=row,
                    new_id=f"v544_teacher_{row['id']}_r{repeat:02d}",
                    answer=answer,
                    role="teacher_gain",
                    weight=args.teacher_weight,
                    repeat_index=repeat,
                )
            )
        audit_rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "baseline_prediction": baseline_prediction(row),
                "teacher_prediction": answer,
                "answer": row["answer"],
                "source_rule": source_rule(row),
                "role": "teacher_gain",
            }
        )
    for row in baseline_correct:
        answer = baseline_prediction(row)
        role = "bit_replay" if str(row.get("family")) == "bit_manipulation" else "equation_replay"
        weight = args.protected_weight if str(row["id"]) == args.protected_id else args.bit_replay_weight
        if role == "equation_replay":
            weight = args.equation_replay_weight
        train_rows.append(
            make_row(
                base=row,
                new_id=f"v544_replay_{row['id']}",
                answer=answer,
                role=role,
                weight=weight,
                repeat_index=0,
            )
        )
    train_source_ids = {str(row.get("metadata", {}).get("source_row_id", "")) for row in train_rows}
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("id", "")) in train_source_ids:
            continue
        val_rows.append(
            make_row(
                base=row,
                new_id=f"v544_val_{row['id']}",
                answer=str(row.get("answer", "")).strip(),
                role="weak_holdout_validation",
                weight=0.0,
                repeat_index=0,
            )
        )
    if len(val_rows) < args.min_val_rows:
        raise RuntimeError(f"validation rows below minimum: {len(val_rows)} < {args.min_val_rows}")

    prompt_hashes = [sha256_text(str(row.get("prompt", ""))) for row in train_rows]
    if len(prompt_hashes) != len(set((row["id"], prompt_hash) for row, prompt_hash in zip(train_rows, prompt_hashes))):
        raise RuntimeError("unexpected train duplicate id/prompt hash")
    train_prompt_set = {sha256_text(str(row.get("prompt", ""))) for row in train_rows}
    val_prompt_set = {sha256_text(str(row.get("prompt", ""))) for row in val_rows}
    overlap = sorted(train_prompt_set & val_prompt_set)
    if overlap:
        raise RuntimeError(f"train/val prompt overlap detected: {len(overlap)}")

    train_path = args.output_dir / "v544_minimal_distillation_train.jsonl"
    val_path = args.output_dir / "v544_minimal_distillation_val.jsonl"
    audit_path = args.output_dir / "v544_teacher_gain_audit.csv"
    manifest_path = args.output_dir / "v544_minimal_distillation_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_csv(
        audit_path,
        audit_rows,
        ["id", "family", "baseline_prediction", "teacher_prediction", "answer", "source_rule", "role"],
    )
    manifest = {
        "schema_version": "kg1_v544_minimal_distillation_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v543_predictions_csv": str(args.v543_predictions_csv),
            "v543_predictions_sha256": sha256_file(args.v543_predictions_csv),
        },
        "parameters": {
            "teacher_repeat": args.teacher_repeat,
            "teacher_weight": args.teacher_weight,
            "bit_replay_weight": args.bit_replay_weight,
            "equation_replay_weight": args.equation_replay_weight,
            "protected_id": args.protected_id,
            "protected_answer": args.protected_answer,
            "protected_weight": args.protected_weight,
        },
        "teacher_gain_ids": teacher_gain_ids,
        "baseline_correct_count": len(baseline_correct),
        "train_summary": summarize_jsonl(train_rows),
        "validation_summary": summarize_jsonl(val_rows),
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "teacher_gain_audit_csv": str(audit_path),
            "teacher_gain_audit_sha256": sha256_file(audit_path),
            "manifest_json": str(manifest_path),
        },
        "decision": {
            "status": "dataset_ready_for_v286_tokenization_gate",
            "gpu_allowed": False,
            "reason": (
                f"teacher_gains={len(teacher_gain)}; train_rows={len(train_rows)}; "
                f"val_rows={len(val_rows)}; protected={args.protected_id}"
            ),
            "next_action": "Run V286 tokenization gate with submit_safe_suffix and then objective/masking gate before any HF H200 launch.",
        },
    }
    write_json(manifest_path, manifest)
    print("train_summary =", json.dumps(manifest["train_summary"], sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(manifest["validation_summary"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V544 MINIMAL DISTILLATION DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        csv_path = tmp / "preds.csv"
        rows = [
            {
                "id": "g1",
                "prompt": "p gain",
                "answer": "B",
                "family": "equation_transform",
                "v343_prediction": "A",
                "v543_prediction": "B",
                "v543_source_rule": "toy_rule",
            },
            {
                "id": "r1",
                "prompt": "p replay",
                "answer": "1",
                "family": "bit_manipulation",
                "v343_prediction": "1",
                "v543_prediction": "1",
            },
            {
                "id": "hold",
                "prompt": "p holdout",
                "answer": "2",
                "family": "equation_transform",
                "v343_prediction": "x",
                "v543_prediction": "x",
            },
            {
                "id": "symbol_brace",
                "prompt": "p symbolic brace replay",
                "answer": "{]``",
                "family": "equation_transform",
                "v343_prediction": "{]``",
                "v543_prediction": "{]``",
            },
            {
                "id": "symbol_close_brace",
                "prompt": "p symbolic close brace holdout",
                "answer": "}/!}",
                "family": "equation_transform",
                "v343_prediction": "wrong",
                "v543_prediction": "wrong",
            },
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args = argparse.Namespace(
            v543_predictions_csv=csv_path,
            output_dir=tmp / "out",
            teacher_repeat=2,
            expected_teacher_gain_count=1,
            protected_id="r1",
            protected_answer="1",
            teacher_weight=2.0,
            bit_replay_weight=1.5,
            equation_replay_weight=1.0,
            protected_weight=3.0,
            min_val_rows=1,
        )
        manifest = build(args)
        assert manifest["train_summary"]["rows"] == 4
        assert manifest["validation_summary"]["rows"] == 2
        assert manifest["train_summary"]["final_answer_format_counts"]["boxed_raw_label_free"] >= 1
        assert manifest["validation_summary"]["final_answer_format_counts"]["unboxed_label_free_fallback"] >= 1
    print("v544_minimal_distillation_dataset_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v543-predictions-csv", type=Path, default=DEFAULT_V543_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-repeat", type=int, default=5)
    parser.add_argument("--expected-teacher-gain-count", type=int, default=9)
    parser.add_argument("--protected-id", default="8740ed31")
    parser.add_argument("--protected-answer", default="01101000")
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--bit-replay-weight", type=float, default=1.5)
    parser.add_argument("--equation-replay-weight", type=float, default=0.8)
    parser.add_argument("--protected-weight", type=float, default=3.0)
    parser.add_argument("--min-val-rows", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

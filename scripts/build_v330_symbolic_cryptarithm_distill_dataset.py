#!/usr/bin/env python3
"""Build V330 symbolic cryptarithm distillation data.

V330 converts the single V329 CPU-gated symbolic equation signal into
out-of-gate synthetic SFT rows. It must not train on weak/full rows directly.
Every generated row is re-validated by the V329 CP-SAT solver before it is
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
for item in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_v278_symbolic_pbe_dsl_audit_hf import parse_alice_prompt  # noqa: E402
from run_v329_symbolic_cryptarithm_gate import (  # noqa: E402
    prediction_from_solution,
    solve_cryptarithm_assignment,
)


DEFAULT_V329_MANIFEST = (
    REPO_ROOT
    / "artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/v329_symbolic_cryptarithm_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v330_symbolic_cryptarithm_distill_dataset"
DEFAULT_REFERENCE_PATHS = [
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv",
]

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify the candidate briefly, "
    "then end with exactly one final answer in \\boxed{}."
)
SAFE_DIGIT_SYMBOLS = list("!#$&()<>?@[]^_|~")
OPERATOR_SYMBOLS = list("%*/+-:")
SCHEMA_VERSION = "kg1_v330_symbolic_cryptarithm_distill_dataset_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"bad JSONL row {path}:{line_no}: {exc}") from exc
        return rows
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


def box(answer: str) -> str:
    text = str(answer).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "\\boxed{" + text + "}"


def final_answer_line(answer: str) -> str:
    return "Final answer: " + box(answer)


def make_messages(prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


def read_reference_fingerprints(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_rows(path)
        for row in rows:
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompt_hashes.add(sha256_text(prompt))
        summaries.append({"path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
    return ids, prompt_hashes, summaries


def validate_v329_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v329_symbolic_cryptarithm_gate_v1":
        raise RuntimeError("unexpected V329 schema: " + str(payload.get("schema_version")))
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    if decision.get("decision") != "symbolic_cryptarithm_cpu_gate_found_new_signal":
        raise RuntimeError("V329 decision does not authorize synthetic distillation")
    if int(payload.get("new_accepted_candidate_count", -1)) != 1:
        raise RuntimeError("V329 must have exactly one new accepted symbolic candidate")
    if list(payload.get("new_accepted_candidate_ids") or []) != ["99d6a3b5"]:
        raise RuntimeError("V329 accepted id drift: " + json.dumps(payload.get("new_accepted_candidate_ids")))
    if int(payload.get("projected_equation_correct", -1)) != 61:
        raise RuntimeError("V329 projected equation correct drift")
    if int(payload.get("conflict_count", -1)) != 0:
        raise RuntimeError("V329 conflicts must be zero")
    baseline = payload.get("baseline_family_counts") if isinstance(payload.get("baseline_family_counts"), dict) else {}
    bit = baseline.get("bit_manipulation") if isinstance(baseline.get("bit_manipulation"), dict) else {}
    if int(bit.get("correct", -1)) < 136:
        raise RuntimeError("V329 bit guardrail below 136")
    return payload


def symbol_number(value: int, digit_to_symbol: dict[int, str], *, width: int = 2) -> str:
    return "".join(digit_to_symbol[int(digit)] for digit in f"{value:0{width}d}")


def symbol_output(value: int, digit_to_symbol: dict[int, str]) -> str:
    return "".join(digit_to_symbol[int(digit)] for digit in str(value))


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
        "Below are a few examples:",
    ]
    for lhs, rhs in examples:
        lines.append(f"{lhs} = {rhs}")
    lines.append("")
    lines.append(f"Now, determine the result for: {query}")
    return "\n".join(lines)


def prove_with_v329_solver(prompt: str, answer: str, operator: str) -> tuple[bool, str]:
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        return False, "parse_status=" + parse_status
    assignments = solve_cryptarithm_assignment(
        examples,
        query,
        {operator: "mul"},
        max_solutions=20,
        solver_time_limit_s=0.5,
    )
    predictions = sorted({prediction_from_solution(solution, query, "mul") for solution in assignments})
    predictions = [prediction for prediction in predictions if prediction]
    if predictions == [answer]:
        return True, f"v329_cp_sat_unique_prediction={answer}; assignments={len(assignments)}"
    return False, f"v329_cp_sat_predictions={predictions}; assignments={len(assignments)}; answer={answer}"


def generate_case(rng: random.Random, *, split: str, index: int) -> dict[str, Any]:
    operator = rng.choice(OPERATOR_SYMBOLS)
    available_symbols = [symbol for symbol in SAFE_DIGIT_SYMBOLS if symbol != operator]
    digit_symbols = rng.sample(available_symbols, 10)
    digit_to_symbol = {digit: digit_symbols[digit] for digit in range(10)}
    symbol_to_digit = {symbol: digit for digit, symbol in digit_to_symbol.items()}

    used_pairs: set[tuple[int, int]] = set()
    examples: list[tuple[str, str]] = []
    trace_examples: list[str] = []
    example_count = 4 + (index % 3)
    for _ in range(example_count):
        while True:
            left = rng.randint(10, 99)
            right = rng.randint(10, 99)
            if (left, right) not in used_pairs:
                used_pairs.add((left, right))
                break
        lhs = symbol_number(left, digit_to_symbol) + operator + symbol_number(right, digit_to_symbol)
        rhs = symbol_output(left * right, digit_to_symbol)
        examples.append((lhs, rhs))
        trace_examples.append(f"- {lhs}: {left} * {right} = {left * right} -> {rhs}")

    seen_example_symbols = {char for lhs, rhs in examples for char in (lhs + rhs)}
    query_attempts = 0
    while True:
        query_attempts += 1
        if query_attempts > 500:
            raise RuntimeError("unable to sample a query with covered output symbols")
        q_left = rng.randint(10, 99)
        q_right = rng.randint(10, 99)
        query_candidate = symbol_number(q_left, digit_to_symbol) + operator + symbol_number(q_right, digit_to_symbol)
        answer_candidate = symbol_output(q_left * q_right, digit_to_symbol)
        query_symbols = set(query_candidate)
        if (q_left, q_right) not in used_pairs and set(answer_candidate).issubset(seen_example_symbols | query_symbols):
            break
    query = query_candidate
    answer_value = q_left * q_right
    answer = answer_candidate
    prompt = make_prompt(examples, query)

    ok, proof = prove_with_v329_solver(prompt, answer, operator)
    if not ok:
        raise RuntimeError("generated row failed V329 proof: " + proof)

    mapping_text = ", ".join(f"{symbol}={digit}" for symbol, digit in sorted(symbol_to_digit.items()))
    trace = "\n".join(
        [
            "Use the examples as a symbolic cryptarithm.",
            f"The operator {operator} is multiplication.",
            "Recovered digit map: " + mapping_text + ".",
            "Example checks:",
            *trace_examples[:3],
            f"Query: {query}: {q_left} * {q_right} = {answer_value} -> {answer}.",
            "Solver proof: " + proof + ".",
            final_answer_line(answer),
        ]
    )
    row_id = f"v330_{split}_symbolic_mul_{index:05d}_{sha256_text(prompt)[:10]}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": "v330_symbolic_cryptarithm_distill",
        "source": "v330_synthetic_from_v329_rule_class",
        "split": split,
        "family": "equation_transform",
        "subcategory": "equation_symbolic_cryptarithm_single_operator_mul",
        "rule_class": "symbolic_cryptarithm_single_operator_digits_mul",
        "operator": operator,
        "teacher": "v329_cp_sat_symbolic_cryptarithm",
        "v329_seed_ids_used_as_rule_evidence": ["99d6a3b5"],
        "gate_rows_used_for_training": False,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "v329_seed_rows_used_for_training": False,
        "loss_regime": "short_deterministic_trace_plus_final_box",
        "solver_proof": proof,
    }
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "equation_transform",
        "subcategory": "equation_symbolic_cryptarithm_single_operator_mul",
        "source": "v330_symbolic_cryptarithm_distill",
        "messages": make_messages(prompt, trace),
        "metadata": metadata,
    }


def build_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    count = args.train_rows if split == "train" else args.val_rows
    seed = args.seed if split == "train" else args.seed + 10000
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"unable to build {split} rows after {attempts} attempts")
        try:
            rows.append(generate_case(rng, split=split, index=len(rows)))
        except RuntimeError:
            continue
    rng.shuffle(rows)
    return rows


def validate_boxed_answer(text: str, answer: str) -> bool:
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return len(boxes) == 1 and boxes[0] == str(answer) and text.rstrip().endswith("\\boxed{" + str(answer) + "}")


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    ref_ids: set[str],
    ref_prompt_hashes: set[str],
    label: str,
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    bad: list[str] = []
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        messages = row.get("messages")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        prompt_hash = sha256_text(prompt)
        if not row_id:
            bad.append("missing_id")
        if row_id in seen_ids:
            bad.append(f"{row_id}:duplicate_id")
        if row_id in ref_ids:
            bad.append(f"{row_id}:reference_id_overlap")
        if not prompt:
            bad.append(f"{row_id}:missing_prompt")
        if prompt_hash in seen_prompts:
            bad.append(f"{row_id}:duplicate_prompt")
        if prompt_hash in ref_prompt_hashes:
            bad.append(f"{row_id}:reference_prompt_overlap")
        if not answer:
            bad.append(f"{row_id}:missing_answer")
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{row_id}:bad_messages")
        else:
            roles = [str(message.get("role", "")) for message in messages]
            if roles != ["system", "user", "assistant"]:
                bad.append(f"{row_id}:bad_message_roles")
            if str(messages[1].get("content", "")) != prompt:
                bad.append(f"{row_id}:user_prompt_mismatch")
            if not validate_boxed_answer(str(messages[-1].get("content", "")), answer):
                bad.append(f"{row_id}:assistant_not_exactly_one_final_box")
        for flag in (
            "gate_rows_used_for_training",
            "weak_gate_rows_used_for_training",
            "full_gate_rows_used_for_training",
            "v329_seed_rows_used_for_training",
        ):
            if metadata.get(flag) is not False:
                bad.append(f"{row_id}:{flag}_not_false")
        operator = str(metadata.get("operator", ""))
        ok, proof = prove_with_v329_solver(prompt, answer, operator)
        if not ok:
            bad.append(f"{row_id}:v329_solver_failed:{proof}")
        seen_ids.add(row_id)
        seen_prompts.add(prompt_hash)
        family_counts[str(row.get("family", ""))] += 1
        subcategory_counts[str(row.get("subcategory", ""))] += 1
        operator_counts[operator] += 1
    if bad:
        raise RuntimeError(f"{label} validation failed: " + json.dumps(bad[:20], ensure_ascii=False))
    return {
        "rows": len(rows),
        "unique_ids": len(seen_ids),
        "unique_prompt_hashes": len(seen_prompts),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "operator_counts": dict(sorted(operator_counts.items())),
        "reference_id_overlap": 0,
        "reference_prompt_overlap": 0,
        "solver_validated_rows": len(rows),
    }


def equation_near_miss(answer: str) -> str:
    text = str(answer)
    if len(text) > 1:
        return text[:-1]
    if text.isdigit():
        return str((int(text) + 1) % 10)
    return text + "?"


def format_negative(answer: str, mode: str) -> str:
    if mode == "no_box":
        return "Final answer: " + str(answer)
    if mode == "multiple_boxes":
        return "First guess: " + box(answer) + "\nFinal answer: " + box(equation_near_miss(answer))
    if mode == "trailing_text":
        return final_answer_line(answer) + "\nThis is the answer."
    raise ValueError("unknown negative mode: " + mode)


def build_preferences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        answer = str(row["answer"])
        chosen = str(row["messages"][-1]["content"])
        base_meta = {
            **dict(row.get("metadata") or {}),
            "schema_version": "kg1_v330_symbolic_cryptarithm_preference_v1",
            "preference_source_row_id": row["id"],
        }
        candidates = [
            ("hard_negative_equation_near_miss", final_answer_line(equation_near_miss(answer))),
            ("format_negative_no_box", format_negative(answer, "no_box")),
            ("format_negative_multiple_boxes", format_negative(answer, "multiple_boxes")),
            ("format_negative_trailing_text", format_negative(answer, "trailing_text")),
        ]
        for negative_type, rejected in candidates:
            out.append(
                {
                    "id": f"{row['id']}_{negative_type}",
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {**base_meta, "negative_type": negative_type},
                }
            )
    return out


def validate_preferences(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bad: list[str] = []
    negative_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        negative_type = str((row.get("metadata") or {}).get("negative_type", ""))
        if not row_id or not prompt or not chosen or not rejected:
            bad.append(f"{row_id}:missing_field")
        if chosen == rejected:
            bad.append(f"{row_id}:chosen_equals_rejected")
        if len(re.findall(r"\\boxed\{([^{}]*)\}", chosen)) != 1:
            bad.append(f"{row_id}:chosen_box_count")
        negative_counts[negative_type] += 1
    if bad:
        raise RuntimeError(f"{label} preference validation failed: " + json.dumps(bad[:20], ensure_ascii=False))
    return {"rows": len(rows), "negative_type_counts": dict(sorted(negative_counts.items()))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V330 SYMBOLIC CRYPTARITHM DISTILL DATASET START ===", flush=True)
    print("v329_manifest_json =", args.v329_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("seed =", args.seed, flush=True)
    print("train_rows =", args.train_rows, flush=True)
    print("val_rows =", args.val_rows, flush=True)
    v329_manifest = validate_v329_manifest(args.v329_manifest_json)
    ref_ids, ref_prompt_hashes, reference_summary = read_reference_fingerprints(args.reference_path)
    print("reference_id_count =", len(ref_ids), flush=True)
    print("reference_prompt_hash_count =", len(ref_prompt_hashes), flush=True)

    train_rows = build_split(args, "train")
    val_rows = build_split(args, "validation")
    train_preferences = build_preferences(train_rows)
    val_preferences = build_preferences(val_rows)
    train_summary = validate_rows(train_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="train")
    val_summary = validate_rows(val_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="validation")
    train_pref_summary = validate_preferences(train_preferences, "train")
    val_pref_summary = validate_preferences(val_preferences, "validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_sft_train.jsonl"
    val_path = args.output_dir / f"{args.label}_sft_val.jsonl"
    train_pref_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    val_pref_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(train_pref_path, train_preferences)
    write_jsonl(val_pref_path, val_preferences)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "v329_manifest_json": str(args.v329_manifest_json),
        "v329_manifest_sha256": sha256_file(args.v329_manifest_json),
        "v329_decision": v329_manifest.get("decision"),
        "source_gate": {
            "accepted_candidate_ids": v329_manifest.get("new_accepted_candidate_ids"),
            "projected_equation_correct": v329_manifest.get("projected_equation_correct"),
            "projected_weak_correct_if_equation_only": v329_manifest.get("projected_weak_correct_if_equation_only"),
        },
        "reference_summary": reference_summary,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "train_preference_summary": train_pref_summary,
        "validation_preference_summary": val_pref_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "preferences_train_jsonl": str(train_pref_path),
            "preferences_train_sha256": sha256_file(train_pref_path),
            "preferences_val_jsonl": str(val_pref_path),
            "preferences_val_sha256": sha256_file(val_pref_path),
            "manifest_json": str(manifest_path),
        },
        "training_authorization": "blocked_until_tokenization_gate_and_no_regression_mixed_dataset",
        "required_next_gate": [
            "scripts/run_v286_generic_tokenization_gate.py --assistant-final-answer-mode boxed_suffix",
            "build a mixed replay dataset with V304/V325/V330 before HF",
            "weak no-regression gate: bit>=136 equation>=61 total>=197",
        ],
    }
    write_json(manifest_path, manifest)
    print("train_jsonl =", train_path, flush=True)
    print("val_jsonl =", val_path, flush=True)
    print("preferences_train_jsonl =", train_pref_path, flush=True)
    print("preferences_val_jsonl =", val_pref_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("train_preference_summary =", json.dumps(train_pref_summary, sort_keys=True), flush=True)
    print("validation_preference_summary =", json.dumps(val_pref_summary, sort_keys=True), flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V330 SYMBOLIC CRYPTARITHM DISTILL DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        ref = tmp / "reference.csv"
        ref.write_text("id,prompt,answer\nref1,unused,0\n", encoding="utf-8")
        v329 = tmp / "v329_manifest.json"
        write_json(
            v329,
            {
                "schema_version": "kg1_v329_symbolic_cryptarithm_gate_v1",
                "decision": {"decision": "symbolic_cryptarithm_cpu_gate_found_new_signal"},
                "new_accepted_candidate_count": 1,
                "new_accepted_candidate_ids": ["99d6a3b5"],
                "projected_equation_correct": 61,
                "projected_weak_correct_if_equation_only": 197,
                "conflict_count": 0,
                "baseline_family_counts": {"bit_manipulation": {"correct": 136}},
            },
        )
        args = argparse.Namespace(
            v329_manifest_json=v329,
            output_dir=tmp / "out",
            label="v330_self_test",
            seed=330,
            train_rows=4,
            val_rows=2,
            reference_path=[ref],
        )
        manifest = run(args)
        if manifest["train_summary"]["rows"] != 4:
            raise AssertionError(manifest["train_summary"])
        if manifest["validation_summary"]["rows"] != 2:
            raise AssertionError(manifest["validation_summary"])
        if manifest["train_preference_summary"]["rows"] != 16:
            raise AssertionError(manifest["train_preference_summary"])
    print("v330_symbolic_cryptarithm_distill_dataset_self_test=ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v329-manifest-json", type=Path, default=DEFAULT_V329_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v330_symbolic_cryptarithm_distill")
    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--train-rows", type=int, default=240)
    parser.add_argument("--val-rows", type=int, default=60)
    parser.add_argument("--reference-path", type=Path, action="append", default=list(DEFAULT_REFERENCE_PATHS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

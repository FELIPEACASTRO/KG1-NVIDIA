#!/usr/bin/env python3
"""Build V498 focused numeric teacher trace pack.

V498 is a CPU-only dataset builder. It narrows the V475/V325 idea to the exact
numeric rule classes that V497 proved did not transfer into the adapter:

* signed minus: reject sign-stripped answers;
* colon absolute difference with trailing zero preserved;
* direct addition under symbolic/additive operators.

The weak/full rows are never used as training rows. They are only forbidden
reference fingerprints and diagnostic evidence. Bit rows are replay guardrails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import box_answer  # noqa: E402
import build_v282_rank19_micro_patch_dataset as v282  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v498_numeric_teacher_trace_pack/20260516T_v498_numeric_teacher"
DEFAULT_V497_MANIFEST = (
    REPO_ROOT
    / "artifacts/v497_cpu_residual_transfer_audit/20260516T_v497_cpu_audit/"
    / "v497_cpu_residual_transfer_audit_manifest.json"
)
DEFAULT_V475_TRAIN = (
    REPO_ROOT
    / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/"
    / "v475_equation_bit_replay_mix_train.jsonl"
)
DEFAULT_V475_VAL = (
    REPO_ROOT
    / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix/"
    / "v475_equation_bit_replay_mix_val.jsonl"
)
DEFAULT_WEAK_REF = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_FULL_REF = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"

TARGET_RULES = {
    "minus_signed_opposite_sign_guarded": {
        "builder": v282.build_minus_signed,
        "subcategory": "equation_numeric_minus_signed_hard_negative",
        "teacher": "reject_sign_stripped_candidate",
    },
    "colon_absdiff_restore_trailing_zero": {
        "builder": v282.build_colon_absdiff_restore_trailing_zero,
        "subcategory": "equation_numeric_colon_trailing_zero_hard_negative",
        "teacher": "reject_trailing_zero_drop",
    },
    "add_direct_over_model_add_variant": {
        "builder": v282.build_add_direct,
        "subcategory": "equation_numeric_add_direct_hard_negative",
        "teacher": "reject_distractor_operator",
    },
}
EXPECTED_V497_DECISION = "do_not_promote_v496_or_repeat_h200_sft"
EXPECTED_BIT_REPLAY_TRAIN_ROWS = 512
EXPECTED_BIT_REPLAY_VAL_ROWS = 128


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def normalize_answer(answer: Any) -> str:
    return re.sub(r"\s+", "", str(answer or "")).strip()


def prompt_answer_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")) + "\0" + normalize_answer(row.get("answer", "")))


def prompt_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


def read_reference(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompts: set[str] = set()
    prompt_answers: set[str] = set()
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "ids": ids, "prompt_hashes": prompts, "prompt_answer_hashes": prompt_answers}
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            rid = str(row.get("id", "") or row.get("row_id", "")).strip()
            prompt = str(row.get("prompt", "") or "")
            answer = str(row.get("answer", "") or "")
            if rid:
                ids.add(rid)
            if prompt:
                prompts.add(sha256_text(normalize_prompt(prompt)))
            if prompt and answer:
                prompt_answers.add(sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer)))
    return {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "sha256": sha256_file(path),
        "ids": ids,
        "prompt_hashes": prompts,
        "prompt_answer_hashes": prompt_answers,
    }


def query_from_prompt(prompt: str) -> str:
    marker = "Now, determine the result for:"
    if marker not in prompt:
        raise RuntimeError("prompt missing query marker")
    return prompt.rsplit(marker, 1)[1].strip()


def hard_negative_for(rule_name: str, query: str, answer: str) -> str:
    numbers = [int(value) for value in re.findall(r"\d+", query)]
    if rule_name == "minus_signed_opposite_sign_guarded":
        return answer[1:] if answer.startswith("-") else "-" + answer
    if rule_name == "colon_absdiff_restore_trailing_zero":
        if answer.endswith("0") and len(answer) > 1:
            return answer[:-1]
        return str(abs(numbers[0] - numbers[1])) if len(numbers) >= 2 else ""
    if rule_name == "add_direct_over_model_add_variant":
        return str(abs(numbers[0] - numbers[1])) if len(numbers) >= 2 else ""
    raise KeyError(rule_name)


def teacher_trace(rule_name: str, query: str, answer: str) -> str:
    wrong = hard_negative_for(rule_name, query, answer)
    boxed = box_answer(answer)
    if rule_name == "minus_signed_opposite_sign_guarded":
        return (
            "Rule: use the query operator and preserve the signed left-minus-right result.\n"
            f"Reject common wrong candidate {wrong}; it strips or flips the sign.\n"
            f"Query {query} gives {answer}.\n"
            f"Final answer: {boxed}"
        )
    if rule_name == "colon_absdiff_restore_trailing_zero":
        return (
            "Rule: for ':' compute the absolute difference and preserve the full decimal digits.\n"
            f"Reject common wrong candidate {wrong}; it drops a required trailing zero.\n"
            f"Query {query} gives {answer}.\n"
            f"Final answer: {boxed}"
        )
    if rule_name == "add_direct_over_model_add_variant":
        return (
            "Rule: use examples with the same additive query operator; ignore distractor operators.\n"
            f"Reject common wrong candidate {wrong}; it follows a distractor subtraction pattern.\n"
            f"Query {query} gives {answer}.\n"
            f"Final answer: {boxed}"
        )
    raise KeyError(rule_name)


def make_messages(prompt: str, trace: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
                "Infer the hidden rule from the examples, reject the common wrong candidate briefly, "
                "then end with exactly one final answer in \\boxed{}."
            ),
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": trace},
    ]


def build_equation_row(
    *,
    builder: Callable[[random.Random, int, str], dict[str, Any]],
    rng: random.Random,
    split: str,
    rule_name: str,
    index: int,
) -> dict[str, Any]:
    raw = builder(rng, index, split)
    prompt = str(raw["prompt"])
    answer = str(raw["answer"])
    query = query_from_prompt(prompt)
    trace = teacher_trace(rule_name, query, answer)
    row = {
        "id": f"v498_{split}_{rule_name}_{index:05d}",
        "prompt": prompt,
        "answer": answer,
        "family": "equation_transform",
        "subcategory": TARGET_RULES[rule_name]["subcategory"],
        "source": "v498_numeric_teacher_trace_pack",
        "messages": make_messages(prompt, trace),
        "metadata": {
            "schema_version": "kg1_v498_numeric_teacher_trace_pack_v1",
            "source": "v498_synthetic_numeric_teacher_trace",
            "split": split,
            "family": "equation_transform",
            "subcategory": TARGET_RULES[rule_name]["subcategory"],
            "rule_name": rule_name,
            "teacher": TARGET_RULES[rule_name]["teacher"],
            "hard_negative": hard_negative_for(rule_name, query, answer),
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "v497_cpu_residual_audit_used_as_rule_evidence_only": True,
        },
    }
    return row


def select_bit_replay(path: Path, *, split: str, count: int) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("family") == "bit_manipulation"]
    if len(rows) < count:
        raise RuntimeError(f"not enough bit replay rows in {path}: {len(rows)} < {count}")
    selected = []
    for index, row in enumerate(rows[:count]):
        out = json.loads(json.dumps(row))
        metadata = dict(out.get("metadata") or {})
        metadata.update(
            {
                "schema_version": "kg1_v498_numeric_teacher_trace_pack_v1",
                "source": "v498_bit_replay_guardrail_from_v475",
                "split": split,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "gate_rows_used_for_training": False,
                "v498_guardrail_role": "preserve_bit_manipulation_plateau",
            }
        )
        out["id"] = f"v498_{split}_bit_replay_{index:05d}_{out.get('id', '')}"
        out["source"] = "v498_bit_replay_guardrail_from_v475"
        out["metadata"] = metadata
        selected.append(out)
    return selected


def build_split(args: argparse.Namespace, *, split: str, seed: int, equation_per_rule: int, bit_rows: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for rule_name, spec in TARGET_RULES.items():
        builder = spec["builder"]
        for index in range(equation_per_rule):
            rows.append(build_equation_row(builder=builder, rng=rng, split=split, rule_name=rule_name, index=index))
    bit_path = args.v475_train_jsonl if split == "train" else args.v475_val_jsonl
    rows.extend(select_bit_replay(bit_path, split=split, count=bit_rows))
    rng.shuffle(rows)
    return rows


def audit_rows(rows: list[dict[str, Any]], *, references: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [prompt_hash(row) for row in rows]
    prompt_answers = [prompt_answer_hash(row) for row in rows]
    ref_ids = set().union(*(ref["ids"] for ref in references))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in references))
    ref_prompt_answers = set().union(*(ref["prompt_answer_hashes"] for ref in references))
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    rule_counts = Counter(str((row.get("metadata") or {}).get("rule_name", "")) for row in rows if row.get("family") == "equation_transform")
    missing_boxed_final = 0
    for row in rows:
        messages = row.get("messages") or []
        assistant = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"]
        if not assistant or ("Final answer: \\boxed{" not in str(assistant[-1].get("content", ""))):
            missing_boxed_final += 1
    summary = {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "equation_rule_counts": dict(sorted(rule_counts.items())),
        "missing_boxed_final": missing_boxed_final,
        "reference_id_overlap": len(set(ids) & ref_ids),
        "reference_prompt_overlap": len(set(prompts) & ref_prompts),
        "reference_prompt_answer_overlap": len(set(prompt_answers) & ref_prompt_answers),
    }
    if summary["duplicate_ids"] or summary["missing_boxed_final"]:
        raise RuntimeError(f"{label} row audit failed: {summary}")
    if summary["reference_id_overlap"] or summary["reference_prompt_overlap"] or summary["reference_prompt_answer_overlap"]:
        raise RuntimeError(f"{label} overlaps forbidden reference: {summary}")
    return summary


def validate_v497(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    decision = payload.get("decision") or {}
    if decision.get("decision") != EXPECTED_V497_DECISION:
        raise RuntimeError("unexpected V497 decision: " + str(decision))
    if int(payload.get("v324_verified_equation_gain", -1)) != 4:
        raise RuntimeError("V497 must show exactly the +4 CPU equation gain")
    if int(payload.get("v496_bit_loss_rows", -1)) != 2:
        raise RuntimeError("V497 must expose the two V496 bit losses")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V498 NUMERIC TEACHER TRACE PACK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("v497_manifest_json =", args.v497_manifest_json, flush=True)
    print("seed =", args.seed, flush=True)
    print("train_equation_per_rule =", args.train_equation_per_rule, flush=True)
    print("val_equation_per_rule =", args.val_equation_per_rule, flush=True)
    print("train_bit_rows =", args.train_bit_rows, flush=True)
    print("val_bit_rows =", args.val_bit_rows, flush=True)

    v497_manifest = validate_v497(args.v497_manifest_json)
    references = [read_reference(path) for path in args.reference_csv]
    train_rows = build_split(
        args,
        split="train",
        seed=args.seed,
        equation_per_rule=args.train_equation_per_rule,
        bit_rows=args.train_bit_rows,
    )
    val_rows = build_split(
        args,
        split="validation",
        seed=args.seed + 10000,
        equation_per_rule=args.val_equation_per_rule,
        bit_rows=args.val_bit_rows,
    )
    train_summary = audit_rows(train_rows, references=references, label="train")
    val_summary = audit_rows(val_rows, references=references, label="validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "v498_numeric_teacher_trace_train.jsonl"
    val_path = args.output_dir / "v498_numeric_teacher_trace_val.jsonl"
    manifest_path = args.output_dir / "v498_numeric_teacher_trace_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v498_numeric_teacher_trace_pack_v1",
        "generated_at_utc": utc_now(),
        "label": "v498_numeric_teacher_trace_pack",
        "v497_manifest_json": str(args.v497_manifest_json),
        "v497_manifest_sha256": sha256_file(args.v497_manifest_json),
        "v497_decision": v497_manifest.get("decision", {}),
        "rule_focus": sorted(TARGET_RULES),
        "reference_csvs": [str(path) for path in args.reference_csv],
        "reference_summary": [
            {key: value for key, value in ref.items() if key not in {"ids", "prompt_hashes", "prompt_answer_hashes"}}
            for ref in references
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "training_authorization": "blocked_until_tokenization_gate_and_v498_cpu_projection_gate",
        "required_next_gate": [
            "v286_tokenization_gate_boxed_suffix",
            "bit_guardrail_exact_binary",
            "first_checkpoint_kill_switch_equation_ge_60_bit_ge_136_trunc_0_total_gt_192",
        ],
    }
    write_json(manifest_path, manifest)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V498 NUMERIC TEACHER TRACE PACK END ===", flush=True)
    return manifest


def run_self_test() -> None:
    assert hard_negative_for("minus_signed_opposite_sign_guarded", "63-19", "-55") == "55"
    assert hard_negative_for("minus_signed_opposite_sign_guarded", "85-92", "92") == "-92"
    assert hard_negative_for("colon_absdiff_restore_trailing_zero", "37:67", "30") == "3"
    assert hard_negative_for("add_direct_over_model_add_variant", "94)40", "134") == "54"
    assert "Final answer: \\boxed{-55}" in teacher_trace("minus_signed_opposite_sign_guarded", "63-19", "-55")
    print("v498_numeric_teacher_trace_pack_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v497-manifest-json", type=Path, default=DEFAULT_V497_MANIFEST)
    parser.add_argument("--v475-train-jsonl", type=Path, default=DEFAULT_V475_TRAIN)
    parser.add_argument("--v475-val-jsonl", type=Path, default=DEFAULT_V475_VAL)
    parser.add_argument("--reference-csv", type=Path, action="append", default=[DEFAULT_WEAK_REF, DEFAULT_FULL_REF])
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--train-equation-per-rule", type=int, default=400)
    parser.add_argument("--val-equation-per-rule", type=int, default=100)
    parser.add_argument("--train-bit-rows", type=int, default=EXPECTED_BIT_REPLAY_TRAIN_ROWS)
    parser.add_argument("--val-bit-rows", type=int, default=EXPECTED_BIT_REPLAY_VAL_ROWS)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

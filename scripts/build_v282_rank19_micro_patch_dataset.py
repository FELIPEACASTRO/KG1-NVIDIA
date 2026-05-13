#!/usr/bin/env python3
"""Build V282 rank19 micro-patch training data.

Goal: start from the public-score 0.86 V194/rank19 adapter and teach the small
numeric equation patterns that V274/V275 fixed with a label-free verifier,
without training on the weak-gate rows themselves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"Final answer: {answer}"},
    ]


def make_row(
    *,
    row_id: str,
    prompt: str,
    answer: str,
    split: str,
    source: str,
    subcategory: str,
    rule_name: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": str(answer),
        "family": "equation_transform",
        "subcategory": subcategory,
        "source": source,
        "messages": make_messages(prompt, str(answer)),
        "metadata": {
            "source": source,
            "split": split,
            "family": "equation_transform",
            "subcategory": subcategory,
            "subtype": subcategory,
            "rule_name": rule_name,
            "v282_role": split,
            "v282_base_adapter": "v194_rank19_score086",
            "v282_training_intent": "small_verified_numeric_equation_patch_with_v217_replay",
            "answer_style": "final_answer_one_line_unboxed",
            "weak_gate_rows_used_for_training": False,
        },
    }


def alice_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
    return (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        f"{body}\n"
        f"Now, determine the result for: {query}"
    )


def two_digit(rng: random.Random) -> int:
    return rng.randint(1, 99)


def fmt2(value: int) -> str:
    return f"{value:02d}"


def rand_pair(rng: random.Random) -> tuple[int, int]:
    a = two_digit(rng)
    b = two_digit(rng)
    while a == b:
        b = two_digit(rng)
    return a, b


def build_minus_signed(rng: random.Random, index: int, split: str) -> dict[str, Any]:
    examples: list[tuple[str, str]] = []
    seen: set[str] = set()
    wanted_negative = index % 2 == 0
    while len(examples) < 5:
        a, b = rand_pair(rng)
        if wanted_negative and a > b:
            a, b = b, a
        if not wanted_negative and a < b:
            a, b = b, a
        lhs = f"{fmt2(a)}-{fmt2(b)}"
        if lhs in seen:
            continue
        seen.add(lhs)
        examples.append((lhs, str(a - b)))
    qa, qb = rand_pair(rng)
    if wanted_negative and qa > qb:
        qa, qb = qb, qa
    if not wanted_negative and qa < qb:
        qa, qb = qb, qa
    answer = str(qa - qb)
    prompt = alice_prompt(examples, f"{fmt2(qa)}-{fmt2(qb)}")
    return make_row(
        row_id=f"v282_{split}_minus_signed_{index:05d}",
        prompt=prompt,
        answer=answer,
        split=split,
        source="v282_v274_rule_synthetic",
        subcategory="equation_numeric_minus_signed",
        rule_name="minus_signed_opposite_sign_guarded",
    )


def build_minus_direct_negative_restore_sign(rng: random.Random, index: int, split: str) -> dict[str, Any]:
    examples: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(examples) < 4:
        a, b = rand_pair(rng)
        lhs = f"{fmt2(a)}-{fmt2(b)}"
        if lhs in seen:
            continue
        seen.add(lhs)
        examples.append((lhs, str(a - b)))
    qa, qb = rand_pair(rng)
    if qa > qb:
        qa, qb = qb, qa
    answer = str(qa - qb)
    prompt = alice_prompt(examples, f"{fmt2(qa)}-{fmt2(qb)}")
    return make_row(
        row_id=f"v282_{split}_minus_direct_negative_{index:05d}",
        prompt=prompt,
        answer=answer,
        split=split,
        source="v282_v343_rule_synthetic",
        subcategory="equation_numeric_minus_direct_negative",
        rule_name="minus_direct_negative_restore_sign",
    )


def build_colon_absdiff(rng: random.Random, index: int, split: str) -> dict[str, Any]:
    examples: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(examples) < 4:
        a, b = rand_pair(rng)
        lhs = f"{fmt2(a)}:{fmt2(b)}"
        if lhs in seen:
            continue
        seen.add(lhs)
        examples.append((lhs, str(abs(a - b))))
    qa, qb = rand_pair(rng)
    answer = str(abs(qa - qb))
    prompt = alice_prompt(examples, f"{fmt2(qa)}:{fmt2(qb)}")
    return make_row(
        row_id=f"v282_{split}_colon_absdiff_{index:05d}",
        prompt=prompt,
        answer=answer,
        split=split,
        source="v282_v274_rule_synthetic",
        subcategory="equation_numeric_colon_absdiff",
        rule_name="colon_absdiff_unreverse_same_len",
    )


def build_colon_absdiff_restore_trailing_zero(rng: random.Random, index: int, split: str) -> dict[str, Any]:
    examples: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(examples) < 4:
        a, b = rand_pair(rng)
        lhs = f"{fmt2(a)}:{fmt2(b)}"
        if lhs in seen:
            continue
        seen.add(lhs)
        examples.append((lhs, str(abs(a - b))))
    qa, qb = rand_pair(rng)
    while abs(qa - qb) % 10 != 0 or qa == qb:
        qa, qb = rand_pair(rng)
    answer = str(abs(qa - qb))
    prompt = alice_prompt(examples, f"{fmt2(qa)}:{fmt2(qb)}")
    return make_row(
        row_id=f"v282_{split}_colon_trailing_zero_{index:05d}",
        prompt=prompt,
        answer=answer,
        split=split,
        source="v282_v343_rule_synthetic",
        subcategory="equation_numeric_colon_trailing_zero",
        rule_name="colon_absdiff_restore_trailing_zero",
    )


def build_add_direct(rng: random.Random, index: int, split: str) -> dict[str, Any]:
    op = ")" if index % 2 == 0 else "+"
    distractor = "#" if op == ")" else "%"
    examples: list[tuple[str, str]] = []
    for _ in range(3):
        a, b = rand_pair(rng)
        examples.append((f"{fmt2(a)}{op}{fmt2(b)}", str(a + b)))
    for _ in range(2):
        a, b = rand_pair(rng)
        examples.append((f"{fmt2(a)}{distractor}{fmt2(b)}", str(abs(a - b))))
    rng.shuffle(examples)
    qa, qb = rand_pair(rng)
    answer = str(qa + qb)
    prompt = alice_prompt(examples, f"{fmt2(qa)}{op}{fmt2(qb)}")
    return make_row(
        row_id=f"v282_{split}_add_direct_{index:05d}",
        prompt=prompt,
        answer=answer,
        split=split,
        source="v282_v274_rule_synthetic",
        subcategory="equation_numeric_add_direct",
        rule_name="add_direct_over_model_add_variant",
    )


def generate_patch_rows(count_per_rule: int, split: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    builders = [build_minus_signed, build_colon_absdiff, build_add_direct]
    for builder in builders:
        for index in range(count_per_rule):
            rows.append(builder(rng, index, split))
    rng.shuffle(rows)
    return rows


def prompt_variants(row: dict[str, Any]) -> list[str]:
    values = [str(row.get("prompt", ""))]
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                values.append(str(message.get("content", "")))
    return [value.strip() for value in values if value and value.strip()]


def reference_from_csv(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompt_hashes.add(sha256_text(prompt))
    return ids, prompt_hashes


def assert_no_reference_overlap(rows: list[dict[str, Any]], ref_ids: set[str], ref_prompt_hashes: set[str], label: str) -> dict[str, Any]:
    id_hits: list[str] = []
    prompt_hits: list[str] = []
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid and rid in ref_ids:
            id_hits.append(rid)
        if any(sha256_text(variant) in ref_prompt_hashes for variant in prompt_variants(row)):
            prompt_hits.append(rid)
    report = {
        "label": label,
        "rows": len(rows),
        "reference_ids": len(ref_ids),
        "reference_prompt_hashes": len(ref_prompt_hashes),
        "id_overlap_count": len(id_hits),
        "prompt_overlap_count": len(prompt_hits),
        "id_overlap_sample": id_hits[:10],
        "prompt_overlap_sample": prompt_hits[:10],
    }
    if id_hits or prompt_hits:
        raise RuntimeError(f"{label} reference overlap detected: {json.dumps(report, sort_keys=True)}")
    return report


def dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicate_ids = 0
    duplicate_prompts = 0
    for row in rows:
        rid = str(row.get("id", ""))
        hashes = {sha256_text(variant) for variant in prompt_variants(row)}
        if rid in seen_ids:
            duplicate_ids += 1
            continue
        if seen_prompts.intersection(hashes):
            duplicate_prompts += 1
            continue
        seen_ids.add(rid)
        seen_prompts.update(hashes)
        kept.append(row)
    return kept, {"duplicate_ids_removed": duplicate_ids, "duplicate_prompts_removed": duplicate_prompts}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(Counter(str(row.get("family", "")) for row in rows).items())),
        "source_counts": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "subcategory_counts": dict(sorted(Counter(str(row.get("subcategory", "")) for row in rows).items())),
        "rule_counts": dict(sorted(Counter(str((row.get("metadata") or {}).get("rule_name", "")) for row in rows).items())),
    }


def validate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bad: list[str] = []
    for row in rows:
        messages = row.get("messages")
        answer = str(row.get("answer", ""))
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{row.get('id')}:bad_messages")
            continue
        if messages[-1].get("role") != "assistant":
            bad.append(f"{row.get('id')}:bad_assistant_role")
        if messages[-1].get("content") != f"Final answer: {answer}":
            bad.append(f"{row.get('id')}:assistant_answer_mismatch")
        if row.get("family") != "equation_transform" and str(row.get("id", "")).startswith("v282_"):
            bad.append(f"{row.get('id')}:patch_family_mismatch")
    if bad:
        raise RuntimeError(f"{label} row validation failed: {bad[:20]}")
    return {"label": label, "bad_rows": 0, **summarize(rows)}


def normalize_replay_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        prompt = str(item.get("prompt", ""))
        answer = str(item.get("answer", ""))
        item["messages"] = make_messages(prompt, answer)
        metadata = dict(item.get("metadata") or {})
        metadata["weak_gate_rows_used_for_training"] = False
        metadata.setdefault("source", str(item.get("source", "")))
        metadata.setdefault("split", split)
        metadata.setdefault("family", str(item.get("family", "")))
        metadata.setdefault("subcategory", str(item.get("subcategory", "")))
        metadata.setdefault("v282_replay_role", split)
        item["metadata"] = metadata
        normalized.append(item)
    return normalized


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V282 RANK19 MICRO PATCH DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("seed =", args.seed, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay_train = normalize_replay_rows(read_jsonl(args.replay_train_jsonl), "train")
    replay_val = normalize_replay_rows(read_jsonl(args.replay_val_jsonl), "validation")
    patch_train = generate_patch_rows(args.patch_train_per_rule, "train", args.seed)
    patch_val = generate_patch_rows(args.patch_val_per_rule, "validation", args.seed + 10000)

    ref_ids, ref_prompt_hashes = reference_from_csv([path for path in args.reference_csv if path])
    overlap_reports = [
        assert_no_reference_overlap(patch_train, ref_ids, ref_prompt_hashes, "patch_train_vs_reference"),
        assert_no_reference_overlap(patch_val, ref_ids, ref_prompt_hashes, "patch_val_vs_reference"),
    ]

    rng = random.Random(args.seed)
    train_rows, train_dedupe = dedupe(replay_train + patch_train)
    val_rows, val_dedupe = dedupe(replay_val + patch_val)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    if len(train_rows) < args.min_train_rows:
        raise RuntimeError(f"train rows below floor: {len(train_rows)} < {args.min_train_rows}")
    if len(val_rows) < args.min_val_rows:
        raise RuntimeError(f"validation rows below floor: {len(val_rows)} < {args.min_val_rows}")

    train_validation = validate_rows(train_rows, "train")
    val_validation = validate_rows(val_rows, "validation")

    train_path = args.output_dir / "v282_rank19_micro_patch_train.jsonl"
    val_path = args.output_dir / "v282_rank19_micro_patch_val.jsonl"
    manifest_path = args.output_dir / "v282_rank19_micro_patch_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v282_rank19_micro_patch_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "inputs": {
            "replay_train_jsonl": str(args.replay_train_jsonl),
            "replay_val_jsonl": str(args.replay_val_jsonl),
            "reference_csv": [str(path) for path in args.reference_csv],
            "patch_train_per_rule": args.patch_train_per_rule,
            "patch_val_per_rule": args.patch_val_per_rule,
        },
        "dedupe": {"train": train_dedupe, "validation": val_dedupe},
        "overlap_reports": overlap_reports,
        "validation": {"train": train_validation, "validation": val_validation},
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "decision": {
            "status": "dataset_ready_for_tokenization_gate",
            "next_action": "Run tokenize-only dry run, then one H200 micro-patch train from v194_protected if gate passes.",
        },
    }
    write_json(manifest_path, manifest)
    print("v282_dataset_manifest =", json.dumps(manifest, sort_keys=True), flush=True)
    print("=== V282 RANK19 MICRO PATCH DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf_cpu_runs/v282_rank19_micro_patch_dataset"))
    parser.add_argument("--label", default="v282_rank19_micro_patch")
    parser.add_argument("--replay-train-jsonl", type=Path, default=Path("data/v217/v217_short_answer_train.jsonl"))
    parser.add_argument("--replay-val-jsonl", type=Path, default=Path("data/v217/v217_short_answer_val.jsonl"))
    parser.add_argument(
        "--reference-csv",
        type=Path,
        action="append",
        default=[],
        help="CSV with id,prompt rows that must not overlap generated patch rows.",
    )
    parser.add_argument("--patch-train-per-rule", type=int, default=360)
    parser.add_argument("--patch-val-per-rule", type=int, default=40)
    parser.add_argument("--seed", type=int, default=282)
    parser.add_argument("--min-train-rows", type=int, default=11000)
    parser.add_argument("--min-val-rows", type=int, default=780)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        replay_train = tmp / "train.jsonl"
        replay_val = tmp / "val.jsonl"
        ref_csv = tmp / "ref.csv"
        write_jsonl(
            replay_train,
            [
                make_row(
                    row_id="replay_train_1",
                    prompt="p1",
                    answer="1",
                    split="train",
                    source="selftest_replay",
                    subcategory="equation_transform",
                    rule_name="replay",
                )
            ],
        )
        write_jsonl(
            replay_val,
            [
                make_row(
                    row_id="replay_val_1",
                    prompt="p2",
                    answer="2",
                    split="validation",
                    source="selftest_replay",
                    subcategory="equation_transform",
                    rule_name="replay",
                )
            ],
        )
        ref_csv.write_text("id,prompt\nweak1,not this prompt\n", encoding="utf-8")
        ns = argparse.Namespace(
            output_dir=tmp / "out",
            label="selftest",
            replay_train_jsonl=replay_train,
            replay_val_jsonl=replay_val,
            reference_csv=[ref_csv],
            patch_train_per_rule=2,
            patch_val_per_rule=1,
            seed=1,
            min_train_rows=7,
            min_val_rows=4,
        )
        manifest = build(ns)
        assert manifest["validation"]["train"]["rows"] == 7
        assert manifest["validation"]["validation"]["rows"] == 4
    print("v282_rank19_micro_patch_dataset_self_test=ok", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

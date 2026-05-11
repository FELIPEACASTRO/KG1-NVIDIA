#!/usr/bin/env python3
"""Render V285 verified ReasoningGym fixtures as KG1/Alice-style puzzles.

V287 is still CPU-only. It does not invent answers: every query and every
in-context example comes from V285 rows whose answers were already verified by
V282. The goal is to reduce distribution drift before any budgeted HF smoke
train by changing direct ReasoningGym prompts into KG1-like example/query
prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

DEFAULT_V285_MANIFEST = Path(
    "artifacts/v285_reasoninggym_auxiliary_dataset/20260511T1830Z/v285_reasoninggym_auxiliary_manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def normalize_answer(answer: Any) -> str:
    return re.sub(r"\s+", "", str(answer)).strip()


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Final answer: " + str(answer).strip()},
    ]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def source_dataset(row: dict[str, Any]) -> str:
    return str(metadata(row).get("source_dataset", ""))


def split_name(row: dict[str, Any]) -> str:
    return str(metadata(row).get("split", ""))


def input_for_row(row: dict[str, Any]) -> str:
    meta = metadata(row)
    source = source_dataset(row)
    if source == "bitwise_arithmetic":
        problem = str(meta.get("problem", "")).strip()
        if not problem:
            prompt = str(row.get("prompt", ""))
            problem = prompt.strip().splitlines()[-1].strip()
        return problem
    if source == "binary_alternation":
        value = meta.get("string", "")
        if not value:
            match = re.search(r"alternating:\s*([01]+)", str(row.get("prompt", "")), re.I)
            value = match.group(1) if match else ""
        return str(value).strip()
    if source == "count_bits":
        value = meta.get("number", meta.get("n", ""))
        if value == "":
            match = re.search(r"number\s+(-?\d+)", str(row.get("prompt", "")))
            value = match.group(1) if match else ""
        return str(value).strip()
    if source == "simple_equations":
        return str(meta.get("equation", row.get("prompt", ""))).strip()
    if source == "cryptarithm":
        words = meta.get("words_letters", [])
        result = str(meta.get("result_letters", "")).strip()
        if isinstance(words, list) and words and result:
            return " + ".join(str(word) for word in words) + " = " + result
        prompt = str(row.get("prompt", ""))
        block = re.findall(r"\b[A-Z]{2,}\b", prompt)
        if len(block) >= 2:
            return " + ".join(block[:-1]) + " = " + block[-1]
        return prompt.strip()
    raise RuntimeError("unsupported source_dataset for V287: " + source)


def family_intro(source: str) -> str:
    if source == "bitwise_arithmetic":
        return (
            "In Alice's Wonderland, a secret bit manipulation rule evaluates hexadecimal integer expressions. "
            "The examples below show expression -> output."
        )
    if source == "binary_alternation":
        return (
            "In Alice's Wonderland, a secret bit manipulation rule maps binary strings to the minimum swap count "
            "needed to make the string alternating, or -1 if impossible. The examples below show input -> output."
        )
    if source == "count_bits":
        return (
            "In Alice's Wonderland, a secret bit manipulation rule maps integers to the count of 1 bits in binary. "
            "The examples below show input -> output."
        )
    if source == "simple_equations":
        return (
            "In Alice's Wonderland, a secret equation transformation maps a linear equation to the value of its "
            "unknown. The examples below show equation -> output."
        )
    if source == "cryptarithm":
        return (
            "In Alice's Wonderland, a secret equation transformation maps an alphametic sum to the digit mapping "
            "that satisfies it. The examples below show equation -> mapping."
        )
    raise RuntimeError("unsupported source_dataset for intro: " + source)


def render_prompt(source: str, examples: list[dict[str, Any]], target: dict[str, Any]) -> str:
    lines = [family_intro(source), "", "Here are examples:"]
    for example in examples:
        lines.append(f"{input_for_row(example)} -> {str(example.get('answer', '')).strip()}")
    lines.append("")
    lines.append("Now, determine the output for:")
    lines.append(input_for_row(target))
    return "\n".join(lines)


def convert_rows(rows: list[dict[str, Any]], split: str, seed: int, examples_per_prompt: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed + (0 if split == "train" else 10000))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[source_dataset(row)].append(row)
    converted: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for source, items in sorted(by_source.items()):
        if len(items) <= examples_per_prompt:
            raise RuntimeError(f"not enough rows for source={source}: {len(items)} <= {examples_per_prompt}")
        for index, row in enumerate(items):
            candidates = [item for item in items if item.get("id") != row.get("id")]
            examples = rng.sample(candidates, examples_per_prompt)
            answer = str(row.get("answer", "")).strip()
            prompt = render_prompt(source, examples, row)
            original_id = str(row.get("id", ""))
            new_row = {
                "id": original_id.replace("v285_", "v287_"),
                "prompt": prompt,
                "answer": answer,
                "family": row.get("family", ""),
                "subcategory": str(row.get("subcategory", "")) + "_alice_style",
                "source": "v287_reasoninggym_alice_style_auxiliary",
                "messages": make_messages(prompt, answer),
                "metadata": {
                    **metadata(row),
                    "original_v285_id": original_id,
                    "source_dataset": source,
                    "split": split,
                    "v287_role": split,
                    "v287_rendering": "alice_style_examples",
                    "v287_examples_per_prompt": examples_per_prompt,
                    "v287_example_ids": [str(example.get("id", "")) for example in examples],
                    "v287_training_authorization": "blocked_until_v288_tokenization_gate_and_budget_decision",
                    "train_allowed": False,
                    "weak_gate_rows_used_for_training": False,
                },
            }
            converted.append(new_row)
            source_counts[source] += 1
    rng.shuffle(converted)
    return converted, {"rows": len(converted), "source_counts": dict(sorted(source_counts.items()))}


def validate_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    ids: list[str] = []
    prompt_hashes: list[str] = []
    prompt_answer_hashes: list[str] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    bad_rows: list[str] = []
    for row in rows:
        rid = str(row.get("id", ""))
        ids.append(rid)
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        prompt_hashes.append(sha256_text(normalize_prompt(prompt)))
        prompt_answer_hashes.append(sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer)))
        family_counts[str(row.get("family", ""))] += 1
        source_counts[source_dataset(row)] += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            bad_rows.append(rid)
            continue
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            bad_rows.append(rid)
            continue
        if messages[1].get("content") != prompt:
            bad_rows.append(rid)
            continue
        if messages[2].get("content") != "Final answer: " + answer:
            bad_rows.append(rid)
            continue
        if metadata(row).get("weak_gate_rows_used_for_training") is not False:
            bad_rows.append(rid)
    if bad_rows:
        raise RuntimeError(f"{split} bad rows: {bad_rows[:20]}")
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompt_hashes) - len(set(prompt_hashes))
    duplicate_prompt_answers = len(prompt_answer_hashes) - len(set(prompt_answer_hashes))
    if duplicate_ids or duplicate_prompts or duplicate_prompt_answers:
        raise RuntimeError(
            f"{split} duplicates detected: ids={duplicate_ids} prompts={duplicate_prompts} "
            f"prompt_answers={duplicate_prompt_answers}"
        )
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "prompt_hash_count": len(set(prompt_hashes)),
        "prompt_answer_hash_count": len(set(prompt_answer_hashes)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V287 REASONINGGYM ALICE STYLE DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v285_manifest_json =", args.v285_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("examples_per_prompt =", args.examples_per_prompt, flush=True)
    print("seed =", args.seed, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v285_manifest = read_json(args.v285_manifest_json)
    train_path = Path(v285_manifest["outputs"]["train_jsonl"])
    val_path = Path(v285_manifest["outputs"]["val_jsonl"])
    expected_train_sha = str(v285_manifest["outputs"].get("train_sha256", ""))
    expected_val_sha = str(v285_manifest["outputs"].get("val_sha256", ""))
    observed_train_sha = sha256_file(train_path)
    observed_val_sha = sha256_file(val_path)
    print("v285_train_jsonl =", train_path, flush=True)
    print("v285_val_jsonl =", val_path, flush=True)
    print("v285_train_sha256 =", observed_train_sha, flush=True)
    print("v285_val_sha256 =", observed_val_sha, flush=True)
    if expected_train_sha and observed_train_sha != expected_train_sha:
        raise RuntimeError(f"V285 train SHA mismatch: expected {expected_train_sha}, got {observed_train_sha}")
    if expected_val_sha and observed_val_sha != expected_val_sha:
        raise RuntimeError(f"V285 validation SHA mismatch: expected {expected_val_sha}, got {observed_val_sha}")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    converted_train, train_selection = convert_rows(train_rows, "train", args.seed, args.examples_per_prompt)
    converted_val, val_selection = convert_rows(val_rows, "validation", args.seed, args.examples_per_prompt)
    train_validation = validate_rows(converted_train, "train")
    val_validation = validate_rows(converted_val, "validation")
    train_prompt_answer = {
        sha256_text(normalize_prompt(row["prompt"]) + "\0" + normalize_answer(row["answer"]))
        for row in converted_train
    }
    val_prompt_answer = {
        sha256_text(normalize_prompt(row["prompt"]) + "\0" + normalize_answer(row["answer"]))
        for row in converted_val
    }
    if train_prompt_answer & val_prompt_answer:
        raise RuntimeError("V287 train/validation prompt+answer overlap detected")

    train_out = args.output_dir / "v287_reasoninggym_alice_style_train.jsonl"
    val_out = args.output_dir / "v287_reasoninggym_alice_style_val.jsonl"
    manifest_path = args.output_dir / "v287_reasoninggym_alice_style_manifest.json"
    write_jsonl(train_out, converted_train)
    write_jsonl(val_out, converted_val)
    manifest = {
        "schema_version": "kg1_v287_reasoninggym_alice_style_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_manifest": str(args.v285_manifest_json),
        "source_manifest_sha256": sha256_file(args.v285_manifest_json),
        "config": {
            "examples_per_prompt": args.examples_per_prompt,
            "seed": args.seed,
        },
        "selection": {
            "train": train_selection,
            "validation": val_selection,
        },
        "validation": {
            "train": train_validation,
            "validation": val_validation,
            "train_val_prompt_answer_overlap": 0,
        },
        "outputs": {
            "train_jsonl": str(train_out),
            "train_sha256": sha256_file(train_out),
            "val_jsonl": str(val_out),
            "val_sha256": sha256_file(val_out),
            "manifest_json": str(manifest_path),
        },
        "blocked_actions": [
            "gpu_train_without_v288_tokenization_gate",
            "model_generation",
            "full_eval",
            "package",
            "kaggle_submit",
        ],
        "decision": {
            "status": "alice_style_dataset_ready_for_tokenization_gate",
            "next_action": "Run V286/V288 tokenization gate before any HF smoke train.",
            "reason": (
                f"train={len(converted_train)}; validation={len(converted_val)}; "
                "answers inherited from V285/V282 verified fixtures"
            ),
        },
    }
    write_json(manifest_path, manifest)
    print("v287_manifest_json =", manifest_path, flush=True)
    print("v287_outputs =", json.dumps(manifest["outputs"], sort_keys=True), flush=True)
    print("v287_validation =", json.dumps(manifest["validation"], sort_keys=True), flush=True)
    print("v287_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("=== V287 REASONINGGYM ALICE STYLE DATASET END ===", flush=True)
    return manifest


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        train = tmp / "train.jsonl"
        val = tmp / "val.jsonl"
        manifest_path = tmp / "v285.json"
        out = tmp / "out"
        rows: list[dict[str, Any]] = []
        for idx in range(6):
            split = "train" if idx < 4 else "validation"
            row = {
                "id": f"v285_{split}_count_bits_u{idx}",
                "prompt": f"How many 1 bits are there in the binary representation of the number {idx + 3}?",
                "answer": str(bin(idx + 3).count("1")),
                "family": "bit_manipulation",
                "subcategory": "bit_counting",
                "messages": [],
                "metadata": {
                    "source_dataset": "count_bits",
                    "number": idx + 3,
                    "split": split,
                    "weak_gate_rows_used_for_training": False,
                },
            }
            rows.append(row)
        write_jsonl(train, [row for row in rows if metadata(row)["split"] == "train"])
        write_jsonl(val, [row for row in rows if metadata(row)["split"] == "validation"])
        write_json(
            manifest_path,
            {
                "outputs": {
                    "train_jsonl": str(train),
                    "train_sha256": sha256_file(train),
                    "val_jsonl": str(val),
                    "val_sha256": sha256_file(val),
                }
            },
        )
        args = argparse.Namespace(
            v285_manifest_json=manifest_path,
            output_dir=out,
            label="selftest",
            examples_per_prompt=1,
            seed=1,
        )
        manifest = build(args)
        assert manifest["validation"]["train"]["rows"] == 4
        assert manifest["validation"]["validation"]["rows"] == 2
    print("v287_reasoninggym_alice_style_dataset_self_test=ok", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v285-manifest-json", type=Path, default=DEFAULT_V285_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v287_reasoninggym_alice_style_dataset") / utc_compact())
    parser.add_argument("--label", default="v287_reasoninggym_alice_style")
    parser.add_argument("--examples-per-prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=287)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

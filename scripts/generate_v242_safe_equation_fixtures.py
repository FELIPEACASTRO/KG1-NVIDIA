#!/usr/bin/env python3
"""Generate leakage-guarded synthetic equation fixtures for KG1 V242.

This is CPU-only and data-generation-only. It does not train, run model
generation, score models, package submissions, download external payloads, or
submit to Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import string
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from audit_jsonl_overlap import build_reference, read_jsonl, sha256_text, prompt_variants


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)
SYMBOLS = list("!@#$%^&*()-_=+[]{}|;:'\",.<>?/`\\")
NUMERIC_OPS = list("!@#$%^&*()-_=+[]{}|;:'\",.<>?/`\\")
RULE_SUMMARY_COLUMNS = ["split", "subcategory", "rule_name", "rows"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def safe_answer(value: Any) -> str:
    return str(value).strip()


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
    return (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        f"{body}\n"
        f"Now, determine the result for: {query}"
    )


def row_id(prompt: str, split: str, index: int) -> str:
    digest = hashlib.sha256(f"v242|{split}|{index}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"v242_{split}_{digest}"


def make_row(
    *,
    split: str,
    index: int,
    prompt: str,
    answer: str,
    rule_name: str,
    subcategory: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rid = row_id(prompt, split, index)
    return {
        "answer": safe_answer(answer),
        "family": "equation_transform",
        "id": rid,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Final answer: " + safe_answer(answer)},
        ],
        "metadata": {
            "generator": "generate_v242_safe_equation_fixtures.py",
            "source": "v242_synthetic_safe_equation_fixtures",
            "subcategory": subcategory,
            "rule_name": rule_name,
            "split": split,
            "train_allowed": True,
            **metadata,
        },
        "prompt": prompt,
        "source": "v242_synthetic_safe_equation_fixtures",
        "subcategory": subcategory,
    }


def choose_symbols(rng: random.Random, count: int) -> list[str]:
    return rng.sample(SYMBOLS, count)


def random_token(rng: random.Random, alphabet: list[str], length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_symbolic_delete_selected(rng: random.Random) -> tuple[str, str, list[tuple[str, str]], str, dict[str, Any]]:
    alphabet = choose_symbols(rng, rng.randint(7, 11))
    delete = set(rng.sample(alphabet, rng.randint(1, 2)))

    def transform(token: str) -> str:
        return "".join(ch for ch in token if ch not in delete) or rng.choice([ch for ch in alphabet if ch not in delete])

    examples = []
    for _ in range(rng.randint(4, 6)):
        lhs = random_token(rng, alphabet, rng.randint(5, 7))
        examples.append((lhs, transform(lhs)))
    query = random_token(rng, alphabet, rng.randint(5, 7))
    answer = transform(query)
    return "symbolic_delete_selected_chars", answer, examples, query, {"delete_chars": "".join(sorted(delete))}


def generate_symbolic_keep_selected(rng: random.Random) -> tuple[str, str, list[tuple[str, str]], str, dict[str, Any]]:
    alphabet = choose_symbols(rng, rng.randint(7, 11))
    keep = set(rng.sample(alphabet, rng.randint(2, 4)))

    def transform(token: str) -> str:
        value = "".join(ch for ch in token if ch in keep)
        return value or rng.choice(sorted(keep))

    examples = []
    for _ in range(rng.randint(4, 6)):
        lhs = random_token(rng, alphabet, rng.randint(5, 8))
        examples.append((lhs, transform(lhs)))
    query = random_token(rng, alphabet, rng.randint(5, 8))
    answer = transform(query)
    return "symbolic_keep_selected_chars", answer, examples, query, {"keep_chars": "".join(sorted(keep))}


def generate_symbolic_char_transducer(rng: random.Random) -> tuple[str, str, list[tuple[str, str]], str, dict[str, Any]]:
    alphabet = choose_symbols(rng, rng.randint(6, 9))
    output_alphabet = choose_symbols(rng, rng.randint(6, 9))
    delete_chars = set(rng.sample(alphabet, rng.randint(1, 2)))
    mapping: dict[str, str] = {}
    for char in alphabet:
        mapping[char] = "" if char in delete_chars else rng.choice(output_alphabet)

    def transform(token: str) -> str:
        value = "".join(mapping[ch] for ch in token)
        return value or rng.choice(output_alphabet)

    examples = []
    for _ in range(rng.randint(5, 7)):
        lhs = random_token(rng, alphabet, rng.randint(5, 7))
        examples.append((lhs, transform(lhs)))
    query = random_token(rng, alphabet, rng.randint(5, 7))
    answer = transform(query)
    return "symbolic_char_transducer_with_deletion", answer, examples, query, {"mapping": mapping}


def generate_symbolic_until_marker(rng: random.Random) -> tuple[str, str, list[tuple[str, str]], str, dict[str, Any]]:
    alphabet = choose_symbols(rng, rng.randint(7, 10))
    marker = rng.choice(alphabet)
    filler = [ch for ch in alphabet if ch != marker]

    def make_lhs() -> str:
        prefix_len = rng.randint(1, 4)
        suffix_len = rng.randint(1, 4)
        return random_token(rng, filler, prefix_len) + marker + random_token(rng, filler, suffix_len)

    def transform(token: str) -> str:
        return token.split(marker, 1)[0]

    examples = []
    for _ in range(rng.randint(4, 6)):
        lhs = make_lhs()
        examples.append((lhs, transform(lhs)))
    query = make_lhs()
    answer = transform(query)
    return "symbolic_prefix_until_marker", answer, examples, query, {"marker": marker}


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(value):02d}"[-2:]
    return int(text[0]), int(text[1])


def reverse_int(value: int) -> int:
    return int(f"{abs(value):02d}"[::-1])


def digit_sum(value: int) -> int:
    return sum(int(ch) for ch in str(abs(value)))


def numeric_rules() -> dict[str, Callable[[int, int], str]]:
    return {
        "add": lambda a, b: str(a + b),
        "sub_ab": lambda a, b: str(a - b),
        "sub_ba": lambda a, b: str(b - a),
        "abs_diff": lambda a, b: str(abs(a - b)),
        "mul": lambda a, b: str(a * b),
        "concat_ab": lambda a, b: f"{a:02d}{b:02d}",
        "concat_ba": lambda a, b: f"{b:02d}{a:02d}",
        "digit_absdiff_concat": lambda a, b: "".join(str(abs(x - y)) for x, y in zip(digits2(a), digits2(b))),
        "digit_add_mod10_concat": lambda a, b: "".join(str((x + y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ab_mod10_concat": lambda a, b: "".join(str((x - y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "sum_digits_all": lambda a, b: str(digit_sum(a) + digit_sum(b)),
        "rev_abs_diff": lambda a, b: str(abs(reverse_int(a) - reverse_int(b))),
        "rev_add": lambda a, b: str(reverse_int(a) + reverse_int(b)),
    }


def numeric_expr(left: int, op: str, right: int) -> str:
    return f"{left:02d}{op}{right:02d}"


def generate_numeric_same_operator(rng: random.Random) -> tuple[str, str, list[tuple[str, str]], str, dict[str, Any]]:
    rule_name, func = rng.choice(list(numeric_rules().items()))
    op = rng.choice(NUMERIC_OPS)
    examples = []
    seen: set[tuple[int, int]] = set()
    for _ in range(rng.randint(4, 6)):
        while True:
            left = rng.randint(0, 99)
            right = rng.randint(0, 99)
            if (left, right) not in seen:
                seen.add((left, right))
                break
        examples.append((numeric_expr(left, op, right), func(left, right)))
    while True:
        left = rng.randint(0, 99)
        right = rng.randint(0, 99)
        if (left, right) not in seen:
            break
    query = numeric_expr(left, op, right)
    answer = func(left, right)
    return "numeric_same_operator_" + rule_name, answer, examples, query, {"operator": op}


SYMBOLIC_GENERATORS = [
    generate_symbolic_delete_selected,
    generate_symbolic_keep_selected,
    generate_symbolic_char_transducer,
    generate_symbolic_until_marker,
]


def generate_fixture(rng: random.Random, split: str, index: int) -> dict[str, Any]:
    if rng.random() < 0.62:
        rule_name, answer, examples, query, metadata = rng.choice(SYMBOLIC_GENERATORS)(rng)
        subcategory = "equation_symbolic_mixed_v242"
    else:
        rule_name, answer, examples, query, metadata = generate_numeric_same_operator(rng)
        subcategory = "equation_numeric_same_operator_v242"
    prompt = make_prompt(examples, query)
    return make_row(
        split=split,
        index=index,
        prompt=prompt,
        answer=answer,
        rule_name=rule_name,
        subcategory=subcategory,
        metadata={**metadata, "example_count": len(examples), "query": query},
    )


def dedupe_rows(rows: list[dict[str, Any]], existing_hashes: set[str]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for row in rows:
        variants = prompt_variants(row)
        digest = sha256_text(variants[0] if variants else row.get("prompt", ""))
        if digest in existing_hashes:
            continue
        existing_hashes.add(digest)
        deduped.append(row)
    return deduped


def generate_rows(split: str, count: int, seed: int, existing_hashes: set[str]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError(f"could not generate enough unique rows for {split}")
        candidate = generate_fixture(rng, split, len(rows))
        deduped = dedupe_rows([candidate], existing_hashes)
        if deduped:
            rows.extend(deduped)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_rule_summary(path: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[tuple[str, str, str]] = Counter()
    for split, rows in [("train", train_rows), ("validation", val_rows)]:
        for row in rows:
            meta = row.get("metadata", {})
            counter[(split, str(row.get("subcategory", "")), str(meta.get("rule_name", "")))] += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RULE_SUMMARY_COLUMNS)
        writer.writeheader()
        for (split, subcategory, rule_name), count in sorted(counter.items()):
            writer.writerow({"split": split, "subcategory": subcategory, "rule_name": rule_name, "rows": count})


def overlap_counts(rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    id_overlap = []
    prompt_overlap = []
    ref_ids: set[str] = reference["ids"]
    ref_hashes: dict[str, str] = reference["prompt_hashes"]
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in ref_ids:
            id_overlap.append({"candidate_id": row_id, "reference_id": row_id})
        for variant in prompt_variants(row):
            digest = sha256_text(variant)
            if digest in ref_hashes:
                prompt_overlap.append({"candidate_id": row_id, "reference_id": ref_hashes[digest], "prompt_sha256": digest})
    return {
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "id_overlap_preview": id_overlap[:10],
        "prompt_overlap_preview": prompt_overlap[:10],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V242 SAFE EQUATION FIXTURE GENERATOR START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("train_rows =", args.train_rows, flush=True)
    print("validation_rows =", args.validation_rows, flush=True)
    print("seed =", args.seed, flush=True)
    print("reference_jsonl =", args.reference_jsonl or "", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes: set[str] = set()
    train_rows = generate_rows("train", args.train_rows, args.seed * 10 + 1, existing_hashes)
    val_rows = generate_rows("validation", args.validation_rows, args.seed * 10 + 2, existing_hashes)

    overlap = {"reference_rows": 0, "train": {}, "validation": {}}
    if args.reference_jsonl:
        reference_rows = read_jsonl(args.reference_jsonl)
        reference = build_reference(reference_rows)
        overlap = {
            "reference_jsonl": str(args.reference_jsonl),
            "reference_rows": len(reference_rows),
            "reference_id_count": len(reference["ids"]),
            "reference_prompt_hash_count": len(reference["prompt_hashes"]),
            "train": overlap_counts(train_rows, reference),
            "validation": overlap_counts(val_rows, reference),
        }
        blocked = []
        for split in ["train", "validation"]:
            if overlap[split]["id_overlap_count"]:
                blocked.append(f"{split}: id_overlap_count={overlap[split]['id_overlap_count']}")
            if overlap[split]["prompt_overlap_count"]:
                blocked.append(f"{split}: prompt_overlap_count={overlap[split]['prompt_overlap_count']}")
        if blocked:
            raise RuntimeError("V242 generated fixture overlap blocked: " + "; ".join(blocked))

    outputs = {
        "train_jsonl": args.output_dir / f"{args.label}_train.jsonl",
        "validation_jsonl": args.output_dir / f"{args.label}_validation.jsonl",
        "rule_summary_csv": args.output_dir / f"{args.label}_rule_summary.csv",
        "manifest_json": args.output_dir / f"{args.label}_manifest.json",
    }
    write_jsonl(outputs["train_jsonl"], train_rows)
    write_jsonl(outputs["validation_jsonl"], val_rows)
    write_rule_summary(outputs["rule_summary_csv"], train_rows, val_rows)

    counts = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_symbolic_rows": sum(1 for row in train_rows if str(row.get("subcategory", "")).startswith("equation_symbolic")),
        "train_numeric_rows": sum(1 for row in train_rows if str(row.get("subcategory", "")).startswith("equation_numeric")),
        "validation_symbolic_rows": sum(1 for row in val_rows if str(row.get("subcategory", "")).startswith("equation_symbolic")),
        "validation_numeric_rows": sum(1 for row in val_rows if str(row.get("subcategory", "")).startswith("equation_numeric")),
    }
    manifest = {
        "schema_version": "kg1_v242_safe_equation_fixtures_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "counts": counts,
        "overlap_audit": overlap,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {name: file_meta(path) for name, path in outputs.items() if name != "manifest_json"},
        "decision": {
            "decision": "fixtures_ready_for_training_gate_review",
            "reason": "synthetic fixtures generated with zero weak-reference id/prompt overlap",
            "next_action": "Use only after a separate training notebook repeats overlap gates and training is explicitly authorized.",
        },
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(outputs["manifest_json"], manifest)
    print("counts =", json.dumps(counts, sort_keys=True), flush=True)
    print("overlap_audit =", json.dumps(overlap, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("=== V242 SAFE EQUATION FIXTURE GENERATOR END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v242_safe_equation_fixtures")
    parser.add_argument("--train-rows", type=int, default=1800)
    parser.add_argument("--validation-rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference = root / "reference.jsonl"
        reference.write_text(json.dumps({"id": "weak1", "prompt": "forbidden prompt"}) + "\n", encoding="utf-8")
        args = argparse.Namespace(
            output_dir=root / "out",
            label="v242_safe_equation_fixtures",
            train_rows=40,
            validation_rows=12,
            seed=242,
            reference_jsonl=reference,
        )
        manifest = run(args)
        if manifest["counts"]["train_rows"] != 40:
            raise AssertionError("unexpected self-test train row count")
        if manifest["counts"]["validation_rows"] != 12:
            raise AssertionError("unexpected self-test validation row count")
        if manifest["overlap_audit"]["train"]["id_overlap_count"] != 0:
            raise AssertionError("unexpected train id overlap")
        if manifest["overlap_audit"]["validation"]["prompt_overlap_count"] != 0:
            raise AssertionError("unexpected validation prompt overlap")
    print("v242_safe_equation_fixtures_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

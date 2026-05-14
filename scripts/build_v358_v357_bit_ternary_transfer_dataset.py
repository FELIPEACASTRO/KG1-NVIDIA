#!/usr/bin/env python3
"""Build V358 synthetic transfer data from V357 bit ternary rules.

V358 is CPU-only. It converts accepted V357 exact global ternary bit rules
plus the older V350 exact global binary rules into synthetic SFT/preference
rows. Weak/full evaluation rows are never used as training examples.
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

from src.solvers.bit_manipulation_solver import BITWISE_OPS, make_transforms, to_bits, from_bits  # noqa: E402


DEFAULT_V357_MANIFEST = REPO_ROOT / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_bit_global_ternary_gate_manifest.json"
DEFAULT_V350_MANIFEST = REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_no_loss_gate_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v358_v357_bit_ternary_transfer_dataset/20260514T_cpu_gate"
DEFAULT_REFERENCE_CSVS = [
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/v350_integrated_predictions.csv",
]

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden bit rule from the examples, verify it compactly, and end with exactly one boxed final answer."
)
SCHEMA_VERSION = "kg1_v358_v357_bit_ternary_transfer_dataset_v1"


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
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bits_from_int(value: int) -> str:
    return format(value & 0xFF, "08b")


def transform_lookup() -> dict[str, Callable[[list[int]], list[int]]]:
    return {name: func for name, func in make_transforms()}


def op_lookup() -> dict[str, Callable[[list[int], list[int]], list[int]]]:
    return {name: func for name, func in BITWISE_OPS}


class Expr:
    def __init__(self, name: str, children: list["Expr"] | None = None) -> None:
        self.name = name
        self.children = children or []

    def render(self) -> str:
        if not self.children:
            return self.name
        return self.name + "(" + ",".join(child.render() for child in self.children) + ")"


def split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args


def parse_expr(text: str) -> Expr:
    text = str(text).strip()
    match = re.fullmatch(r"([A-Z0-9_]+)\((.*)\)", text)
    if not match:
        return Expr(text)
    name = match.group(1)
    args = split_top_level_args(match.group(2))
    if len(args) != 2:
        raise ValueError(f"unsupported expression arity: {text}")
    return Expr(name, [parse_expr(args[0]), parse_expr(args[1])])


def parse_v350_binary_proof(proof: str) -> Expr:
    match = re.fullmatch(r"output=([A-Z_]+)\(([A-Z0-9]+)\(input\),([A-Z0-9]+)\(input\)\)", str(proof).strip())
    if not match:
        raise ValueError(f"unsupported V350 proof: {proof!r}")
    return Expr(match.group(1), [Expr(match.group(2)), Expr(match.group(3))])


def eval_expr(expr: Expr, input_bits: str) -> list[int]:
    transforms = transform_lookup()
    ops = op_lookup()
    bits = to_bits(input_bits)
    if not expr.children:
        if expr.name not in transforms:
            raise KeyError(expr.name)
        return transforms[expr.name](bits)
    if expr.name not in ops:
        raise KeyError(expr.name)
    return ops[expr.name](eval_expr(expr.children[0], input_bits), eval_expr(expr.children[1], input_bits))


def eval_rule(input_bits: str, expr: Expr) -> str:
    return from_bits(eval_expr(expr, input_bits))


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    lines.extend(f"{left} -> {right}" for left, right in examples)
    lines.extend(["", f"Now, determine the output for: {query}"])
    return "\n".join(lines)


def final_answer_line(answer: str) -> str:
    return "Final answer: " + r"\boxed{" + answer + "}"


def trace_for_rule(expr: Expr, examples: list[tuple[str, str]], query: str, answer: str) -> str:
    rule_text = expr.render()
    lines = [
        f"Rule: output = {rule_text}.",
        "Check examples:",
    ]
    for left, expected in examples[:8]:
        observed = eval_rule(left, expr)
        if observed != expected:
            raise RuntimeError(f"example mismatch: {left} -> {observed} != {expected}")
        lines.append(f"- {left} -> {observed} OK")
    observed_answer = eval_rule(query, expr)
    if observed_answer != answer:
        raise RuntimeError(f"target mismatch: {observed_answer} != {answer}")
    lines.extend(
        [
            "Target:",
            f"- input={query}",
            f"- output={answer}",
            final_answer_line(answer),
        ]
    )
    return "\n".join(lines)


def make_messages(prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


def flip_bit(answer: str, index: int) -> str:
    bits = list(answer)
    bits[index] = "0" if bits[index] == "1" else "1"
    return "".join(bits)


def make_case(
    *,
    rng: random.Random,
    split: str,
    rule: dict[str, Any],
    index: int,
    seen_prompts: set[str],
) -> dict[str, Any]:
    expr = parse_expr(rule["expr"])
    attempts = 0
    while True:
        attempts += 1
        values = rng.sample(range(256), 9)
        examples = [(bits_from_int(value), eval_rule(bits_from_int(value), expr)) for value in values[:8]]
        query = bits_from_int(values[8])
        answer = eval_rule(query, expr)
        prompt = make_prompt(examples, query)
        prompt_sha = sha256_text(" ".join(prompt.split()))
        if prompt_sha not in seen_prompts:
            seen_prompts.add(prompt_sha)
            break
        if attempts > 1000:
            raise RuntimeError("could not generate unique V358 prompt")

    assistant = trace_for_rule(expr, examples, query, answer)
    row_id = f"v358_{split}_{rule['source_version']}_{rule['rule_slug']}_{index:05d}_{sha256_text(prompt)[:10]}"
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "bit_manipulation",
        "subcategory": rule["subcategory"],
        "source": "v358_synthetic_from_verified_bit_rules",
        "messages": make_messages(prompt, assistant),
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "source": "v358_synthetic_from_verified_bit_rules",
            "teacher": rule["source_version"],
            "source_id": rule["id"],
            "rule_class": rule["rule_class"],
            "rule_slug": rule["rule_slug"],
            "expr": rule["expr"],
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "synthetic": True,
        },
    }


def make_preferences(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    preferences = []
    for index, row in enumerate(rows):
        answer = str(row["answer"])
        near_miss = flip_bit(answer, index % 8)
        chosen = str(row["messages"][-1]["content"])
        base_meta = dict(row["metadata"])
        for negative_type, rejected in (
            ("hard_negative_one_bit_flip", final_answer_line(near_miss)),
            ("format_negative_no_box", "Final answer: " + answer),
        ):
            meta = dict(base_meta)
            meta.update({"negative_type": negative_type, "preference_source_row_id": row["id"], "split": split})
            preferences.append(
                {
                    "id": f"{row['id']}_{negative_type}",
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": meta,
                }
            )
    return preferences


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [" ".join(str(row.get("prompt", "")).split()) for row in rows]
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    subcategory_counts = Counter(str(row.get("subcategory", "")) for row in rows)
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "unique_ids": len(set(ids)),
        "unique_prompts": len(set(prompts)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
    }


def read_reference_fingerprints(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    meta = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_csv(path)
        for row in rows:
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = " ".join(str(row.get("prompt", "")).split())
            if prompt:
                prompt_hashes.add(sha256_text(prompt))
        meta.append({"path": str(path), "rows": len(rows), "sha256": sha256_file(path)})
    return ids, prompt_hashes, meta


def validate_rows(rows: list[dict[str, Any]], split: str, reference_ids: set[str], reference_prompts: set[str]) -> dict[str, Any]:
    summary = summarize(rows)
    print(f"{split}_summary =", json.dumps(summary, sort_keys=True), flush=True)
    if summary["duplicate_ids"] or summary["duplicate_prompts"]:
        raise RuntimeError(f"{split} duplicates detected")
    ids = {str(row["id"]) for row in rows}
    prompts = {sha256_text(" ".join(str(row["prompt"]).split())) for row in rows}
    id_overlap = len(ids & reference_ids)
    prompt_overlap = len(prompts & reference_prompts)
    print(f"{split}_id_overlap_with_reference =", id_overlap, flush=True)
    print(f"{split}_prompt_sha256_overlap_with_reference =", prompt_overlap, flush=True)
    if id_overlap or prompt_overlap:
        raise RuntimeError(f"{split} anti-leakage failed: id={id_overlap} prompt={prompt_overlap}")
    return {**summary, "id_overlap_with_reference": id_overlap, "prompt_sha256_overlap_with_reference": prompt_overlap}


def slug_expr(expr: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", expr.lower()).strip("_")[:96]


def load_rules(v357_manifest_path: Path, v350_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    v357_manifest = read_json(v357_manifest_path)
    if v357_manifest.get("decision", {}).get("decision") != "v357_bit_global_ternary_gate_passed":
        raise RuntimeError("V357 did not pass")
    v350_manifest = read_json(v350_manifest_path)
    if v350_manifest.get("decision", {}).get("decision") != "v350_cpu_residual_no_loss_gate_passed":
        raise RuntimeError("V350 did not pass")

    rules: list[dict[str, Any]] = []
    for row in read_csv(Path(v357_manifest["outputs"]["candidate_decisions_csv"])):
        if str(row.get("accepted")).lower() == "true":
            expr = str(row["proof"])
            rules.append(
                {
                    "id": row["id"],
                    "source_version": "v357",
                    "rule_class": row["rule_class"],
                    "subcategory": "bit_exact_global_ternary",
                    "expr": expr,
                    "rule_slug": slug_expr(expr),
                }
            )

    for row in read_csv(Path(v350_manifest["outputs"]["candidate_decisions_csv"])):
        if str(row.get("family")) == "bit_manipulation" and str(row.get("accepted")).lower() == "true":
            expr = parse_v350_binary_proof(str(row["proof"])).render()
            rules.append(
                {
                    "id": row["id"],
                    "source_version": "v350_replay",
                    "rule_class": row["rule_class"],
                    "subcategory": "bit_exact_global_binary_replay",
                    "expr": expr,
                    "rule_slug": slug_expr(expr),
                }
            )
    if len(rules) < 15:
        raise RuntimeError(f"expected at least 15 V358 rules, got {len(rules)}")
    return v357_manifest, v350_manifest, rules


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V358 V357 BIT TERNARY TRANSFER DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v357_manifest_json =", args.v357_manifest_json, flush=True)
    print("v350_manifest_json =", args.v350_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("train_per_rule =", args.train_per_rule, flush=True)
    print("val_per_rule =", args.val_per_rule, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v357_manifest, v350_manifest, rules = load_rules(args.v357_manifest_json, args.v350_manifest_json)
    reference_ids, reference_prompts, reference_meta = read_reference_fingerprints(args.reference_csv)

    rng = random.Random(args.seed)
    seen_prompts: set[str] = set(reference_prompts)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for rule in rules:
        per_train = args.replay_train_per_rule if rule["source_version"] == "v350_replay" else args.train_per_rule
        per_val = args.replay_val_per_rule if rule["source_version"] == "v350_replay" else args.val_per_rule
        for index in range(per_train):
            train_rows.append(make_case(rng=rng, split="train", rule=rule, index=index, seen_prompts=seen_prompts))
        for index in range(per_val):
            val_rows.append(make_case(rng=rng, split="validation", rule=rule, index=index, seen_prompts=seen_prompts))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    train_validation = validate_rows(train_rows, "train", reference_ids, reference_prompts)
    val_validation = validate_rows(val_rows, "validation", reference_ids, reference_prompts)
    train_prompts = {sha256_text(" ".join(str(row["prompt"]).split())) for row in train_rows}
    val_prompts = {sha256_text(" ".join(str(row["prompt"]).split())) for row in val_rows}
    train_val_prompt_overlap = len(train_prompts & val_prompts)
    print("train_val_prompt_overlap =", train_val_prompt_overlap, flush=True)
    if train_val_prompt_overlap:
        raise RuntimeError("train/validation prompt overlap detected")

    pref_train = make_preferences(train_rows, "train")
    pref_val = make_preferences(val_rows, "validation")

    outputs = {
        "train_jsonl": args.output_dir / "v358_v357_bit_ternary_transfer_train.jsonl",
        "val_jsonl": args.output_dir / "v358_v357_bit_ternary_transfer_val.jsonl",
        "preferences_train_jsonl": args.output_dir / "v358_v357_bit_ternary_transfer_preferences_train.jsonl",
        "preferences_val_jsonl": args.output_dir / "v358_v357_bit_ternary_transfer_preferences_val.jsonl",
        "manifest_json": args.output_dir / "v358_v357_bit_ternary_transfer_manifest.json",
    }
    write_jsonl(outputs["train_jsonl"], train_rows)
    write_jsonl(outputs["val_jsonl"], val_rows)
    write_jsonl(outputs["preferences_train_jsonl"], pref_train)
    write_jsonl(outputs["preferences_val_jsonl"], pref_val)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "inputs": {
            "v357_manifest_json": str(args.v357_manifest_json),
            "v357_manifest_sha256": sha256_file(args.v357_manifest_json),
            "v350_manifest_json": str(args.v350_manifest_json),
            "v350_manifest_sha256": sha256_file(args.v350_manifest_json),
            "reference_csvs": reference_meta,
        },
        "source_v357": {
            "decision": v357_manifest.get("decision", {}),
            "accepted_candidate_ids": v357_manifest.get("accepted_candidate_ids", []),
        },
        "source_v350_replay": {
            "decision": v350_manifest.get("decision", {}),
            "accepted_candidate_ids": v350_manifest.get("accepted_candidate_ids", []),
        },
        "rules": rules,
        "policy": {
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "synthetic_from_verified_rules_only": True,
            "assistant_final_answer_mode": "boxed_suffix",
            "hf_gpu_allowed": False,
            "next_gate": "run_v286_generic_tokenization_gate before any HF launch",
        },
        "validation": {
            "train": train_validation,
            "validation": val_validation,
            "train_val_prompt_overlap": train_val_prompt_overlap,
            "preference_train_rows": len(pref_train),
            "preference_val_rows": len(pref_val),
        },
        "outputs": {
            "train_jsonl": str(outputs["train_jsonl"]),
            "train_sha256": sha256_file(outputs["train_jsonl"]),
            "val_jsonl": str(outputs["val_jsonl"]),
            "val_sha256": sha256_file(outputs["val_jsonl"]),
            "preferences_train_jsonl": str(outputs["preferences_train_jsonl"]),
            "preferences_train_sha256": sha256_file(outputs["preferences_train_jsonl"]),
            "preferences_val_jsonl": str(outputs["preferences_val_jsonl"]),
            "preferences_val_sha256": sha256_file(outputs["preferences_val_jsonl"]),
            "manifest_json": str(outputs["manifest_json"]),
        },
    }
    write_json(outputs["manifest_json"], manifest)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("=== V358 V357 BIT TERNARY TRANSFER DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    expr = parse_expr("XNOR(SHL2,AND_NOT(ROL5,SHR7))")
    answer = eval_rule("10001010", expr)
    if len(answer) != 8 or set(answer) - {"0", "1"}:
        raise AssertionError(answer)
    binary = parse_v350_binary_proof("output=XOR(SHL1(input),SHR4(input))")
    if binary.render() != "XOR(SHL1,SHR4)":
        raise AssertionError(binary.render())
    print("v358_v357_bit_ternary_transfer_dataset_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v357-manifest-json", type=Path, default=DEFAULT_V357_MANIFEST)
    parser.add_argument("--v350-manifest-json", type=Path, default=DEFAULT_V350_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-csv", type=Path, action="append", default=list(DEFAULT_REFERENCE_CSVS))
    parser.add_argument("--train-per-rule", type=int, default=64)
    parser.add_argument("--val-per-rule", type=int, default=16)
    parser.add_argument("--replay-train-per-rule", type=int, default=160)
    parser.add_argument("--replay-val-per-rule", type=int, default=40)
    parser.add_argument("--seed", type=int, default=358)
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

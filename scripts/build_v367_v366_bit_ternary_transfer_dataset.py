#!/usr/bin/env python3
"""Build V367 transfer data from the V366 full-byte ternary bit teacher.

V367 is CPU-only. It creates synthetic bit_manipulation rows from the V366
accepted CHO/MAJ3 rules, with smaller replay from the previous V357/V350 bit
rules. The supervised completion is exactly one boxed answer to avoid the
format mismatch observed in V359. Weak/full evaluation rows are never used as
training examples.
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


SCHEMA_VERSION = "kg1_v367_v366_bit_ternary_transfer_dataset_v1"
DEFAULT_V366_MANIFEST = (
    REPO_ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/"
    "v366_bit_fullbyte_ternary_op_gate_manifest.json"
)
DEFAULT_V357_MANIFEST = (
    REPO_ROOT / "artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/"
    "v357_bit_global_ternary_gate_manifest.json"
)
DEFAULT_V350_MANIFEST = (
    REPO_ROOT / "artifacts/v350_cpu_residual_no_loss_gate/20260514T_v350_cpu_gate/"
    "v350_no_loss_gate_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate"
DEFAULT_REFERENCE_CSVS = [
    REPO_ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv",
]

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden bit rule from the examples and return exactly one boxed answer."
)
BOXED_RE = re.compile(r"^\\boxed\{[01]{8}\}$")


class Expr:
    def __init__(self, name: str, children: list["Expr"] | None = None) -> None:
        self.name = name
        self.children = children or []

    def render(self) -> str:
        if not self.children:
            return self.name
        return self.name + "(" + ",".join(child.render() for child in self.children) + ")"


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
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def parse_expr(text: str) -> Expr:
    text = str(text).strip()
    match = re.fullmatch(r"([A-Z0-9_]+)\((.*)\)", text)
    if not match:
        return Expr(text)
    return Expr(match.group(1), [parse_expr(item) for item in split_top_level_args(match.group(2))])


def parse_v350_binary_proof(proof: str) -> Expr:
    match = re.fullmatch(r"output=([A-Z_]+)\(([A-Z0-9]+)\(input\),([A-Z0-9]+)\(input\)\)", str(proof).strip())
    if not match:
        raise ValueError(f"unsupported V350 proof: {proof!r}")
    return Expr(match.group(1), [Expr(match.group(2)), Expr(match.group(3))])


def transform_lookup() -> dict[str, Callable[[list[int]], list[int]]]:
    return {name: func for name, func in make_transforms()}


def op_lookup() -> dict[str, Callable[[list[int], list[int]], list[int]]]:
    return {name: func for name, func in BITWISE_OPS}


def majority3(a: list[int], b: list[int], c: list[int]) -> list[int]:
    return [1 if x + y + z >= 2 else 0 for x, y, z in zip(a, b, c)]


def choose(a: list[int], b: list[int], c: list[int]) -> list[int]:
    return [(x & y) | ((1 - x) & z) for x, y, z in zip(a, b, c)]


TERNARY_OPS: dict[str, Callable[[list[int], list[int], list[int]], list[int]]] = {
    "MAJ3": majority3,
    "CHO": choose,
}


def eval_expr(expr: Expr, input_bits: str) -> list[int]:
    transforms = transform_lookup()
    binary_ops = op_lookup()
    bits = to_bits(input_bits)
    if not expr.children:
        if expr.name not in transforms:
            raise KeyError(expr.name)
        return transforms[expr.name](bits)
    if len(expr.children) == 2:
        if expr.name not in binary_ops:
            raise KeyError(expr.name)
        return binary_ops[expr.name](eval_expr(expr.children[0], input_bits), eval_expr(expr.children[1], input_bits))
    if len(expr.children) == 3:
        if expr.name not in TERNARY_OPS:
            raise KeyError(expr.name)
        return TERNARY_OPS[expr.name](
            eval_expr(expr.children[0], input_bits),
            eval_expr(expr.children[1], input_bits),
            eval_expr(expr.children[2], input_bits),
        )
    raise ValueError(f"unsupported arity for {expr.render()}")


def eval_rule(input_bits: str, expr: Expr) -> str:
    return from_bits(eval_expr(expr, input_bits))


def make_prompt(examples: list[tuple[str, str]], query: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, majority, and choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    lines.extend(f"{left} -> {right}" for left, right in examples)
    lines.extend(["", f"Now, determine the output for: {query}"])
    return "\n".join(lines)


def boxed_answer(answer: str) -> str:
    return r"\boxed{" + answer + "}"


def make_messages(prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


def prompt_hash(prompt: str) -> str:
    return sha256_text(" ".join(str(prompt).split()))


def slug_expr(expr: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", expr.lower()).strip("_")[:96]


def flip_bit(answer: str, index: int) -> str:
    bits = list(answer)
    bits[index % len(bits)] = "0" if bits[index % len(bits)] == "1" else "1"
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
    for attempts in range(1, 1001):
        values = rng.sample(range(256), 9)
        examples = [(bits_from_int(value), eval_rule(bits_from_int(value), expr)) for value in values[:8]]
        query = bits_from_int(values[8])
        answer = eval_rule(query, expr)
        prompt = make_prompt(examples, query)
        key = prompt_hash(prompt)
        if key not in seen_prompts:
            seen_prompts.add(key)
            break
    else:
        raise RuntimeError("could not generate unique V367 prompt")

    assistant = boxed_answer(answer)
    row_id = f"v367_{split}_{rule['source_version']}_{rule['rule_slug']}_{index:05d}_{sha256_text(prompt)[:10]}"
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "bit_manipulation",
        "subcategory": rule["subcategory"],
        "source": "v367_synthetic_from_v366_teacher_with_replay",
        "messages": make_messages(prompt, assistant),
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "source": "v367_synthetic_from_v366_teacher_with_replay",
            "teacher": rule["source_version"],
            "source_id": rule["id"],
            "rule_class": rule["rule_class"],
            "rule_slug": rule["rule_slug"],
            "expr": rule["expr"],
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "synthetic": True,
            "completion_format": "boxed_only",
        },
    }


def make_preferences(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    preferences: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = str(row["answer"])
        wrong = flip_bit(answer, index)
        base_meta = dict(row["metadata"])
        for negative_type, rejected in (
            ("hard_negative_one_bit_flip_boxed_only", boxed_answer(wrong)),
            ("format_negative_raw_answer_no_box", answer),
        ):
            metadata = dict(base_meta)
            metadata.update(
                {
                    "negative_type": negative_type,
                    "preference_source_row_id": row["id"],
                    "split": split,
                }
            )
            preferences.append(
                {
                    "id": f"{row['id']}_{negative_type}",
                    "prompt": row["prompt"],
                    "chosen": boxed_answer(answer),
                    "rejected": rejected,
                    "metadata": metadata,
                }
            )
    return preferences


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


def summarize(rows: list[dict[str, Any]], split: str, reference_ids: set[str], reference_prompts: set[str]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [prompt_hash(str(row.get("prompt", ""))) for row in rows]
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    subcategory_counts = Counter(str(row.get("subcategory", "")) for row in rows)
    rule_counts = Counter(str(row.get("metadata", {}).get("rule_slug", "")) for row in rows)
    expr_counts = Counter(str(row.get("metadata", {}).get("expr", "")) for row in rows)
    teacher_counts = Counter(str(row.get("metadata", {}).get("teacher", "")) for row in rows)
    assistant_texts = [str(row["messages"][2]["content"]) for row in rows]
    bad_rows = []
    for row, assistant in zip(rows, assistant_texts):
        if not BOXED_RE.fullmatch(assistant):
            bad_rows.append({"id": row.get("id"), "reason": "assistant_not_boxed_only", "assistant": assistant})
        if row.get("metadata", {}).get("weak_gate_rows_used_for_training") is not False:
            bad_rows.append({"id": row.get("id"), "reason": "weak_gate_flag_not_false"})
        if row.get("metadata", {}).get("full_gate_rows_used_for_training") is not False:
            bad_rows.append({"id": row.get("id"), "reason": "full_gate_flag_not_false"})
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompts) - len(set(prompts))
    id_overlap = len(set(ids) & reference_ids)
    prompt_overlap = len(set(prompts) & reference_prompts)
    print(f"{split}_id_overlap_with_reference =", id_overlap, flush=True)
    print(f"{split}_prompt_sha256_overlap_with_reference =", prompt_overlap, flush=True)
    if duplicate_ids or duplicate_prompts or id_overlap or prompt_overlap or bad_rows:
        raise RuntimeError(
            json.dumps(
                {
                    "split": split,
                    "duplicate_ids": duplicate_ids,
                    "duplicate_prompts": duplicate_prompts,
                    "id_overlap": id_overlap,
                    "prompt_overlap": prompt_overlap,
                    "bad_rows": bad_rows[:20],
                },
                sort_keys=True,
            )
        )
    return {
        "split": split,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "unique_ids": len(set(ids)),
        "prompt_hash_count": len(set(prompts)),
        "duplicate_prompt_count_within_split": duplicate_prompts,
        "id_overlap_with_reference": id_overlap,
        "prompt_sha256_overlap_with_reference": prompt_overlap,
        "assistant_boxed_only_rows": sum(1 for text in assistant_texts if BOXED_RE.fullmatch(text)),
        "assistant_char_min": min(len(text) for text in assistant_texts) if assistant_texts else 0,
        "assistant_char_max": max(len(text) for text in assistant_texts) if assistant_texts else 0,
        "unique_rule_slugs": len(rule_counts),
        "unique_exprs": len(expr_counts),
    }


def load_rules(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    v366_manifest = read_json(args.v366_manifest_json)
    if v366_manifest.get("decision", {}).get("decision") != "v366_cpu_gate_passed":
        raise RuntimeError("V366 did not pass")
    v357_manifest = read_json(args.v357_manifest_json)
    if v357_manifest.get("decision", {}).get("decision") != "v357_bit_global_ternary_gate_passed":
        raise RuntimeError("V357 did not pass")
    v350_manifest = read_json(args.v350_manifest_json)
    if v350_manifest.get("decision", {}).get("decision") != "v350_cpu_residual_no_loss_gate_passed":
        raise RuntimeError("V350 did not pass")

    rules: list[dict[str, Any]] = []
    for row in read_csv(Path(v366_manifest["outputs"]["candidate_decisions_csv"])):
        if str(row.get("accepted")).lower() == "true":
            expr = str(row["proof"])
            rules.append(
                {
                    "id": row["id"],
                    "source_version": "v366_new",
                    "rule_class": row["rule_class"],
                    "subcategory": "bit_fullbyte_ternary_v366_new",
                    "expr": expr,
                    "rule_slug": slug_expr(expr),
                }
            )

    for row in read_csv(Path(v357_manifest["outputs"]["candidate_decisions_csv"])):
        if str(row.get("accepted")).lower() == "true":
            expr = str(row["proof"])
            rules.append(
                {
                    "id": row["id"],
                    "source_version": "v357_replay",
                    "rule_class": row["rule_class"],
                    "subcategory": "bit_exact_global_ternary_replay",
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
    new_count = sum(1 for rule in rules if rule["source_version"] == "v366_new")
    if new_count != 8:
        raise RuntimeError(f"expected 8 V366 rules, got {new_count}")
    return v366_manifest, v357_manifest, v350_manifest, rules


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V367 V366 BIT TERNARY TRANSFER DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v366_manifest_json =", args.v366_manifest_json, flush=True)
    print("v357_manifest_json =", args.v357_manifest_json, flush=True)
    print("v350_manifest_json =", args.v350_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("train_per_new_rule =", args.train_per_new_rule, flush=True)
    print("val_per_new_rule =", args.val_per_new_rule, flush=True)
    print("replay_train_per_rule =", args.replay_train_per_rule, flush=True)
    print("replay_val_per_rule =", args.replay_val_per_rule, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v366_manifest, v357_manifest, v350_manifest, rules = load_rules(args)
    reference_ids, reference_prompts, reference_meta = read_reference_fingerprints(args.reference_csv)
    rng = random.Random(args.seed)
    seen_prompts: set[str] = set(reference_prompts)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for rule in rules:
        if rule["source_version"] == "v366_new":
            train_count = args.train_per_new_rule
            val_count = args.val_per_new_rule
        else:
            train_count = args.replay_train_per_rule
            val_count = args.replay_val_per_rule
        for index in range(train_count):
            train_rows.append(make_case(rng=rng, split="train", rule=rule, index=index, seen_prompts=seen_prompts))
        for index in range(val_count):
            val_rows.append(make_case(rng=rng, split="validation", rule=rule, index=index, seen_prompts=seen_prompts))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    train_prompt_hashes = {prompt_hash(str(row["prompt"])) for row in train_rows}
    val_prompt_hashes = {prompt_hash(str(row["prompt"])) for row in val_rows}
    train_val_prompt_overlap = len(train_prompt_hashes & val_prompt_hashes)
    print("train_val_prompt_overlap =", train_val_prompt_overlap, flush=True)
    if train_val_prompt_overlap:
        raise RuntimeError("train/validation prompt overlap detected")

    train_summary = summarize(train_rows, "train", reference_ids, reference_prompts)
    val_summary = summarize(val_rows, "validation", reference_ids, reference_prompts)
    pref_train = make_preferences(train_rows, "train")
    pref_val = make_preferences(val_rows, "validation")

    outputs = {
        "train_jsonl": args.output_dir / "v367_v366_bit_ternary_transfer_train.jsonl",
        "val_jsonl": args.output_dir / "v367_v366_bit_ternary_transfer_val.jsonl",
        "preferences_train_jsonl": args.output_dir / "v367_v366_bit_ternary_transfer_preferences_train.jsonl",
        "preferences_val_jsonl": args.output_dir / "v367_v366_bit_ternary_transfer_preferences_val.jsonl",
        "manifest_json": args.output_dir / "v367_v366_bit_ternary_transfer_manifest.json",
    }
    write_jsonl(outputs["train_jsonl"], train_rows)
    write_jsonl(outputs["val_jsonl"], val_rows)
    write_jsonl(outputs["preferences_train_jsonl"], pref_train)
    write_jsonl(outputs["preferences_val_jsonl"], pref_val)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "inputs": {
            "v366_manifest_json": str(args.v366_manifest_json),
            "v366_manifest_sha256": sha256_file(args.v366_manifest_json),
            "v357_manifest_json": str(args.v357_manifest_json),
            "v357_manifest_sha256": sha256_file(args.v357_manifest_json),
            "v350_manifest_json": str(args.v350_manifest_json),
            "v350_manifest_sha256": sha256_file(args.v350_manifest_json),
            "reference_csvs": reference_meta,
        },
        "source_v366": {
            "decision": v366_manifest.get("decision", {}),
            "accepted_candidate_ids": v366_manifest.get("accepted_candidate_ids", []),
        },
        "source_v357_replay": {
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
            "assistant_final_answer_mode": "boxed_only",
            "hf_gpu_allowed": False,
            "next_gate": "run_v286_generic_tokenization_gate --assistant-final-answer-mode boxed_only",
            "finops": "HF blocked until tokenization passes; first checkpoint kill-switch required.",
        },
        "validation": {
            "train": train_summary,
            "validation": val_summary,
            "train_val_prompt_overlap": train_val_prompt_overlap,
            "preference": {
                "train_rows": len(pref_train),
                "val_rows": len(pref_val),
                "negative_types": {
                    "hard_negative_one_bit_flip_boxed_only": len(train_rows) + len(val_rows),
                    "format_negative_raw_answer_no_box": len(train_rows) + len(val_rows),
                },
            },
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
        "decision": {
            "decision": "v367_dataset_built_tokenization_required",
            "hf_gpu_allowed": False,
            "reason": (
                f"train_rows={len(train_rows)}; val_rows={len(val_rows)}; "
                f"v366_new_rules=8; total_rules={len(rules)}; boxed_only=true"
            ),
            "next_action": "Run V286 real tokenization gate before any HF upload or job.",
        },
    }
    write_json(outputs["manifest_json"], manifest)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("=== V367 V366 BIT TERNARY TRANSFER DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    for expr_text in ("MAJ3(ROL5,SHL1,SHR4)", "CHO(SHL2,SHR3,ROL1)", "XOR(SHL1,SHR4)"):
        expr = parse_expr(expr_text)
        answer = eval_rule("10001010", expr)
        if len(answer) != 8 or set(answer) - {"0", "1"}:
            raise AssertionError((expr_text, answer))
    if not BOXED_RE.fullmatch(boxed_answer("01010101")):
        raise AssertionError("boxed regex failed")
    print("v367_v366_bit_ternary_transfer_dataset_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v366-manifest-json", type=Path, default=DEFAULT_V366_MANIFEST)
    parser.add_argument("--v357-manifest-json", type=Path, default=DEFAULT_V357_MANIFEST)
    parser.add_argument("--v350-manifest-json", type=Path, default=DEFAULT_V350_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-csv", type=Path, action="append", default=list(DEFAULT_REFERENCE_CSVS))
    parser.add_argument("--train-per-new-rule", type=int, default=96)
    parser.add_argument("--val-per-new-rule", type=int, default=24)
    parser.add_argument("--replay-train-per-rule", type=int, default=24)
    parser.add_argument("--replay-val-per-rule", type=int, default=6)
    parser.add_argument("--seed", type=int, default=367)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

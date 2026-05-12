"""Build V303 synthetic bit full-byte distillation data.

The V300/V302 audits found label-free full-byte bit rules that recover weak/full
bit misses without losses when used as a postprocessor. This dataset converts
those rule families into synthetic short-answer training rows, without using
weak/full evaluation prompts as training examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

BITS = 8
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

EXACT_GAIN_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("CHO", ("SHL2", "SHR3", "ROL1")),
    ("MAJ3", ("ROL2", "SHL4", "SHR5")),
    ("CHO", ("SHL2", "SHR1", "ROL3")),
    ("MAJ3", ("ROL5", "SHL1", "SHR4")),
    ("MAJ3", ("ROL7", "SHL3", "SHR3")),
    ("MAJ3", ("ROL2", "SHL1", "SHR5")),
    ("MAJ3", ("ROL6", "SHL1", "SHR1")),
    ("MAJ3", ("ROL1", "SHL3", "SHR2")),
    ("XOR", ("SHL1", "SHR4")),
    ("CHO", ("SHL2", "SHR4", "ROL7")),
    ("CHO", ("SHL1", "SHR1", "ROL4")),
]

TERNARY_OPS = ("MAJ3", "CHO", "PAR3")
BINARY_OPS = ("XOR",)
TRANSFORM_POOL = (
    "ID",
    "NOT",
    "ROL1",
    "ROL2",
    "ROL3",
    "ROL4",
    "ROL5",
    "ROL6",
    "ROL7",
    "ROR1",
    "ROR2",
    "ROR3",
    "ROR4",
    "ROR5",
    "ROR6",
    "ROR7",
    "SHL1",
    "SHL2",
    "SHL3",
    "SHL4",
    "SHL5",
    "SHR1",
    "SHR2",
    "SHR3",
    "SHR4",
    "SHR5",
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"bad jsonl row {path}:{line_no}: {exc}") from exc
    return rows


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
        {"role": "assistant", "content": "Final answer: " + str(answer)},
    ]


def bit_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{lhs} -> {rhs}" for lhs, rhs in examples)
    return (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n\n"
        "Here are some examples of input -> output:\n"
        f"{body}\n\n"
        f"Now, determine the output for: {query}"
    )


def bits_to_text(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def text_to_bits(text: str) -> tuple[int, ...]:
    if len(text) != BITS or any(ch not in "01" for ch in text):
        raise ValueError(f"not an 8-bit value: {text!r}")
    return tuple(1 if ch == "1" else 0 for ch in text)


def not_bits(bits: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 - bit for bit in bits)


def apply_transform(bits: tuple[int, ...], transform: str) -> tuple[int, ...]:
    if transform == "ID":
        return bits
    if transform == "NOT":
        return not_bits(bits)
    prefix, value_text = transform[:3], transform[3:]
    if not value_text.isdigit():
        raise ValueError(f"bad transform: {transform}")
    k = int(value_text)
    if not 1 <= k < BITS:
        raise ValueError(f"bad transform offset: {transform}")
    if prefix == "ROL":
        return bits[k:] + bits[:k]
    if prefix == "ROR":
        return bits[-k:] + bits[:-k]
    if prefix == "SHL":
        return bits[k:] + tuple(0 for _ in range(k))
    if prefix == "SHR":
        return tuple(0 for _ in range(k)) + bits[:-k]
    raise ValueError(f"bad transform prefix: {transform}")


def apply_binary(op: str, left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    if op == "XOR":
        return tuple(a ^ b for a, b in zip(left, right))
    if op == "AND":
        return tuple(a & b for a, b in zip(left, right))
    if op == "OR":
        return tuple(a | b for a, b in zip(left, right))
    raise ValueError(f"bad binary op: {op}")


def apply_ternary(
    op: str,
    a_bits: tuple[int, ...],
    b_bits: tuple[int, ...],
    c_bits: tuple[int, ...],
) -> tuple[int, ...]:
    if op == "PAR3":
        return tuple(a ^ b ^ c for a, b, c in zip(a_bits, b_bits, c_bits))
    if op == "MAJ3":
        return tuple(1 if a + b + c >= 2 else 0 for a, b, c in zip(a_bits, b_bits, c_bits))
    if op == "CHO":
        return tuple((a & b) | ((1 - a) & c) for a, b, c in zip(a_bits, b_bits, c_bits))
    raise ValueError(f"bad ternary op: {op}")


def evaluate_expr(op: str, transforms: tuple[str, ...], input_bits: str) -> str:
    bits = text_to_bits(input_bits)
    transformed = [apply_transform(bits, transform) for transform in transforms]
    if len(transformed) == 2:
        return bits_to_text(apply_binary(op, transformed[0], transformed[1]))
    if len(transformed) == 3:
        return bits_to_text(apply_ternary(op, transformed[0], transformed[1], transformed[2]))
    raise ValueError(f"bad transform arity for {op}: {transforms}")


def random_bitstring(rng: random.Random) -> str:
    return f"{rng.randrange(0, 256):08b}"


def equivalent_transforms(left: str, right: str) -> bool:
    for value in range(256):
        bits = text_to_bits(f"{value:08b}")
        if apply_transform(bits, left) != apply_transform(bits, right):
            return False
    return True


def expr_name(op: str, transforms: tuple[str, ...]) -> str:
    return f"{op}({','.join(transforms)})"


def make_bit_row(
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
        "answer": answer,
        "family": "bit_manipulation",
        "subcategory": subcategory,
        "source": source,
        "messages": make_messages(prompt, answer),
        "metadata": {
            "source": source,
            "split": split,
            "family": "bit_manipulation",
            "subcategory": subcategory,
            "subtype": subcategory,
            "rule_name": rule_name,
            "answer_style": "final_answer_one_line_unboxed",
            "weak_gate_rows_used_for_training": False,
            "v303_role": split,
            "v303_training_intent": "distill_v300_fullbyte_bit_rules_without_eval_prompt_leakage",
        },
    }


def build_single_synthetic_row(
    rng: random.Random,
    *,
    op: str,
    transforms: tuple[str, ...],
    split: str,
    index: int,
    source: str,
    subcategory: str,
    example_count: int,
) -> dict[str, Any]:
    rule_name = expr_name(op, transforms)
    for _ in range(400):
        inputs: list[str] = []
        while len(inputs) < example_count + 1:
            value = random_bitstring(rng)
            if value not in inputs:
                inputs.append(value)
        examples = [(value, evaluate_expr(op, transforms, value)) for value in inputs[:-1]]
        query = inputs[-1]
        answer = evaluate_expr(op, transforms, query)
        example_outputs = {out for _, out in examples}
        query_popcounts = {item.count("1") for item in inputs}
        min_unique_outputs = 2 if len(transforms) == 2 else 3
        answer_gate = True if len(transforms) == 2 else answer not in {"00000000", "11111111"}
        if len(example_outputs) >= min_unique_outputs and len(query_popcounts) >= 3 and answer_gate:
            prompt = bit_prompt(examples, query)
            return make_bit_row(
                row_id=f"v303_{split}_{subcategory}_{index:06d}",
                prompt=prompt,
                answer=answer,
                split=split,
                source=source,
                subcategory=subcategory,
                rule_name=rule_name,
            )
    raise RuntimeError(f"failed to generate non-degenerate row for {rule_name}")


def sample_random_pattern(rng: random.Random, arity: int) -> tuple[str, tuple[str, ...]]:
    if arity == 2:
        op = rng.choice(BINARY_OPS)
        transforms = tuple(rng.sample(TRANSFORM_POOL, 2))
        while equivalent_transforms(transforms[0], transforms[1]):
            transforms = tuple(rng.sample(TRANSFORM_POOL, 2))
        return op, transforms
    if arity == 3:
        op = rng.choice(TERNARY_OPS)
        transforms = tuple(rng.sample(TRANSFORM_POOL, 3))
        return op, transforms
    raise ValueError(f"bad arity: {arity}")


def generate_patch_rows(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    if split == "train":
        seed = args.seed
        exact_per_pattern = args.train_exact_per_pattern
        random_ternary = args.train_random_ternary
        random_binary = args.train_random_binary
    else:
        seed = args.seed + 10000
        exact_per_pattern = args.val_exact_per_pattern
        random_ternary = args.val_random_ternary
        random_binary = args.val_random_binary

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    idx = 0
    for op, transforms in EXACT_GAIN_PATTERNS:
        for _ in range(exact_per_pattern):
            rows.append(
                build_single_synthetic_row(
                    rng,
                    op=op,
                    transforms=transforms,
                    split=split,
                    index=idx,
                    source="v303_v300_bit_fullbyte_distill_exact",
                    subcategory="bit_fullbyte_v300_gain_pattern",
                    example_count=args.example_count,
                )
            )
            idx += 1

    seen_random_patterns = set(EXACT_GAIN_PATTERNS)
    for _ in range(random_ternary):
        op, transforms = sample_random_pattern(rng, 3)
        while (op, transforms) in seen_random_patterns:
            op, transforms = sample_random_pattern(rng, 3)
        seen_random_patterns.add((op, transforms))
        rows.append(
            build_single_synthetic_row(
                rng,
                op=op,
                transforms=transforms,
                split=split,
                index=idx,
                source="v303_v300_bit_fullbyte_distill_random",
                subcategory="bit_fullbyte_safe_ternary",
                example_count=args.example_count,
            )
        )
        idx += 1

    for _ in range(random_binary):
        op, transforms = sample_random_pattern(rng, 2)
        while (op, transforms) in seen_random_patterns:
            op, transforms = sample_random_pattern(rng, 2)
        seen_random_patterns.add((op, transforms))
        rows.append(
            build_single_synthetic_row(
                rng,
                op=op,
                transforms=transforms,
                split=split,
                index=idx,
                source="v303_v300_bit_fullbyte_distill_random",
                subcategory="bit_fullbyte_binary",
                example_count=args.example_count,
            )
        )
        idx += 1

    rng.shuffle(rows)
    return rows


def normalize_base_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
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
        metadata.setdefault("v303_base_replay_role", split)
        item["metadata"] = metadata
        normalized.append(item)
    return normalized


def dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    out: list[dict[str, Any]] = []
    duplicate_ids = 0
    duplicate_prompts = 0
    for row in rows:
        row_id = str(row.get("id", ""))
        prompt_hash = sha256_text(str(row.get("prompt", "")))
        if row_id in seen_ids:
            duplicate_ids += 1
            continue
        if prompt_hash in seen_prompts:
            duplicate_prompts += 1
            continue
        seen_ids.add(row_id)
        seen_prompts.add(prompt_hash)
        out.append(row)
    return out, {"duplicate_ids_removed": duplicate_ids, "duplicate_prompts_removed": duplicate_prompts}


def reference_from_csv(paths: list[Path]) -> tuple[set[str], set[str]]:
    ref_ids: set[str] = set()
    ref_prompt_hashes: set[str] = set()
    for path in paths:
        if not path:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("id"):
                    ref_ids.add(str(row["id"]))
                if row.get("prompt"):
                    ref_prompt_hashes.add(sha256_text(str(row["prompt"])))
    return ref_ids, ref_prompt_hashes


def assert_no_reference_overlap(
    rows: list[dict[str, Any]],
    ref_ids: set[str],
    ref_prompt_hashes: set[str],
    label: str,
) -> dict[str, Any]:
    id_overlap = sorted(str(row.get("id", "")) for row in rows if str(row.get("id", "")) in ref_ids)
    prompt_overlap = sorted(
        str(row.get("id", ""))
        for row in rows
        if sha256_text(str(row.get("prompt", ""))) in ref_prompt_hashes
    )
    report = {
        "label": label,
        "rows": len(rows),
        "reference_ids": len(ref_ids),
        "reference_prompt_hashes": len(ref_prompt_hashes),
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "id_overlap_sample": id_overlap[:10],
        "prompt_overlap_sample": prompt_overlap[:10],
    }
    if id_overlap or prompt_overlap:
        raise RuntimeError(f"{label} overlaps reference rows: {report}")
    return report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(Counter(str(row.get("family", "")) for row in rows).items())),
        "source_counts": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "subcategory_counts": dict(sorted(Counter(str(row.get("subcategory", "")) for row in rows).items())),
        "rule_counts": dict(
            sorted(Counter(str((row.get("metadata") or {}).get("rule_name", "")) for row in rows).items())
        ),
    }


def validate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bad: list[str] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        metadata = row.get("metadata") or {}
        messages = row.get("messages")
        if not row_id:
            bad.append("missing_id")
        if not prompt:
            bad.append(f"{row_id}:missing_prompt")
        if not answer:
            bad.append(f"{row_id}:missing_answer")
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{row_id}:bad_messages")
            continue
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            bad.append(f"{row_id}:bad_message_roles")
        if messages[1].get("content") != prompt:
            bad.append(f"{row_id}:prompt_message_mismatch")
        if messages[2].get("content") != "Final answer: " + answer:
            bad.append(f"{row_id}:assistant_answer_mismatch")
        if metadata.get("weak_gate_rows_used_for_training") is not False:
            bad.append(f"{row_id}:weak_gate_flag_not_false")
        if row_id.startswith("v303_") and row.get("family") != "bit_manipulation":
            bad.append(f"{row_id}:v303_family_mismatch")
        if row_id.startswith("v303_") and not str(row.get("answer", "")).strip().count("0") + str(row.get("answer", "")).strip().count("1") == 8:
            bad.append(f"{row_id}:v303_answer_not_binary8")
    if bad:
        raise RuntimeError(f"{label} row validation failed: {bad[:25]}")
    return {"label": label, "bad_rows": 0, **summarize(rows)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V303 BIT FULLBYTE DISTILL DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("seed =", args.seed, flush=True)
    print("base_train_jsonl =", args.base_train_jsonl, flush=True)
    print("base_val_jsonl =", args.base_val_jsonl, flush=True)
    print("reference_csv =", [str(path) for path in args.reference_csv], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train = normalize_base_rows(read_jsonl(args.base_train_jsonl), "train")
    base_val = normalize_base_rows(read_jsonl(args.base_val_jsonl), "validation")
    patch_train = generate_patch_rows(args, "train")
    patch_val = generate_patch_rows(args, "validation")
    print("patch_train_summary =", json.dumps(summarize(patch_train), sort_keys=True), flush=True)
    print("patch_val_summary =", json.dumps(summarize(patch_val), sort_keys=True), flush=True)

    ref_ids, ref_prompt_hashes = reference_from_csv([path for path in args.reference_csv if path])
    overlap_reports = [
        assert_no_reference_overlap(patch_train, ref_ids, ref_prompt_hashes, "v303_patch_train_vs_reference"),
        assert_no_reference_overlap(patch_val, ref_ids, ref_prompt_hashes, "v303_patch_val_vs_reference"),
    ]

    train_rows, train_dedupe = dedupe(base_train + patch_train)
    val_rows, val_dedupe = dedupe(base_val + patch_val)
    rng = random.Random(args.seed + 303)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    if len(train_rows) < args.min_train_rows:
        raise RuntimeError(f"train rows below floor: {len(train_rows)} < {args.min_train_rows}")
    if len(val_rows) < args.min_val_rows:
        raise RuntimeError(f"validation rows below floor: {len(val_rows)} < {args.min_val_rows}")

    train_validation = validate_rows(train_rows, "train")
    val_validation = validate_rows(val_rows, "validation")
    print("train_validation =", json.dumps(train_validation, sort_keys=True), flush=True)
    print("val_validation =", json.dumps(val_validation, sort_keys=True), flush=True)

    train_path = args.output_dir / "v303_bit_fullbyte_distill_train.jsonl"
    val_path = args.output_dir / "v303_bit_fullbyte_distill_val.jsonl"
    manifest_path = args.output_dir / "v303_bit_fullbyte_distill_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v303_bit_fullbyte_distill_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "inputs": {
            "base_train_jsonl": str(args.base_train_jsonl),
            "base_val_jsonl": str(args.base_val_jsonl),
            "reference_csv": [str(path) for path in args.reference_csv],
            "example_count": args.example_count,
            "train_exact_per_pattern": args.train_exact_per_pattern,
            "train_random_ternary": args.train_random_ternary,
            "train_random_binary": args.train_random_binary,
            "val_exact_per_pattern": args.val_exact_per_pattern,
            "val_random_ternary": args.val_random_ternary,
            "val_random_binary": args.val_random_binary,
            "exact_gain_patterns": [expr_name(op, transforms) for op, transforms in EXACT_GAIN_PATTERNS],
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
            "next_action": (
                "Run V286 tokenization gate; if clean, launch one budget-capped HF LoRA job "
                "from the best adapter lineage to test whether V300 bit gains distill into adapter-only inference."
            ),
        },
    }
    write_json(manifest_path, manifest)
    print("v303_dataset_manifest =", json.dumps(manifest, sort_keys=True), flush=True)
    print("=== V303 BIT FULLBYTE DISTILL DATASET END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v303_bit_fullbyte_distill_dataset"))
    parser.add_argument("--label", default="v303_bit_fullbyte_distill")
    parser.add_argument(
        "--base-train-jsonl",
        type=Path,
        default=Path("artifacts/v290_rank19_micro_patch_dataset/20260511T1925Z/v282_rank19_micro_patch_train.jsonl"),
    )
    parser.add_argument(
        "--base-val-jsonl",
        type=Path,
        default=Path("artifacts/v290_rank19_micro_patch_dataset/20260511T1925Z/v282_rank19_micro_patch_val.jsonl"),
    )
    parser.add_argument("--reference-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--example-count", type=int, default=9)
    parser.add_argument("--train-exact-per-pattern", type=int, default=96)
    parser.add_argument("--train-random-ternary", type=int, default=360)
    parser.add_argument("--train-random-binary", type=int, default=120)
    parser.add_argument("--val-exact-per-pattern", type=int, default=8)
    parser.add_argument("--val-random-ternary", type=int, default=60)
    parser.add_argument("--val-random-binary", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-val-rows", type=int, default=880)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_text:
        tmp = Path(tmp_text)
        base_train = tmp / "base_train.jsonl"
        base_val = tmp / "base_val.jsonl"
        base_row = make_bit_row(
            row_id="base_bit_0001",
            prompt=bit_prompt([("00000000", "00000000"), ("11111111", "11111111")], "01010101"),
            answer="01010101",
            split="train",
            source="self_test",
            subcategory="bit_manipulation",
            rule_name="identity",
        )
        write_jsonl(base_train, [base_row])
        write_jsonl(base_val, [dict(base_row, id="base_bit_val_0001")])
        out = tmp / "out"
        args = build_parser().parse_args(
            [
                "--output-dir",
                str(out),
                "--base-train-jsonl",
                str(base_train),
                "--base-val-jsonl",
                str(base_val),
                "--seed",
                "303",
                "--train-exact-per-pattern",
                "1",
                "--train-random-ternary",
                "2",
                "--train-random-binary",
                "2",
                "--val-exact-per-pattern",
                "1",
                "--val-random-ternary",
                "1",
                "--val-random-binary",
                "1",
                "--min-train-rows",
                "10",
                "--min-val-rows",
                "5",
            ]
        )
        manifest = build(args)
        if Path(manifest["outputs"]["train_jsonl"]).exists() and Path(manifest["outputs"]["val_jsonl"]).exists():
            print("v303_bit_fullbyte_distill_dataset_self_test=ok", flush=True)
            return 0
    raise RuntimeError("self-test did not create outputs")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

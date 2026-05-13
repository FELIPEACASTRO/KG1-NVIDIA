#!/usr/bin/env python3
"""Build V304 solver-trace distillation data.

V303 used the right rule families but trained only one-line final answers. This
builder keeps the same leakage guards and replay data, while rewriting the
synthetic V300 bit rows and V282 numeric equation rows as short solver traces
that end with the same final-answer line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for item in (SCRIPT_DIR, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import build_v303_bit_fullbyte_distill_dataset as v303  # noqa: E402
from kg1_v274_numeric_postprocessor import parse_alice_prompt, parse_numeric_token  # noqa: E402
from kg1_v300_bit_fullbyte_postprocessor import parse_bit_problem  # noqa: E402


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, show a compact verification, "
    "then end with exactly one final answer line."
)


def final_answer_line(answer: str) -> str:
    return "Final answer: " + str(answer)


def exact_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": final_answer_line(answer)},
    ]


def trace_messages(prompt: str, answer: str, trace: str) -> list[dict[str, str]]:
    body = str(trace).rstrip()
    if not body.endswith(final_answer_line(answer)):
        body = body + "\n" + final_answer_line(answer)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": body},
    ]


def parse_rule_name(rule_name: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"([A-Z0-9_]+)\(([^()]+)\)", str(rule_name))
    if not match:
        raise RuntimeError(f"bad bit rule_name: {rule_name!r}")
    return match.group(1), tuple(part.strip() for part in match.group(2).split(",") if part.strip())


def bit_op_description(op: str) -> str:
    if op == "XOR":
        return "XOR(a,b)=a^b"
    if op == "MAJ3":
        return "MAJ3(a,b,c)=1 when at least two inputs are 1"
    if op == "CHO":
        return "CHO(a,b,c)=(a&b)|((1-a)&c)"
    if op == "PAR3":
        return "PAR3(a,b,c)=a^b^c"
    raise RuntimeError(f"unsupported bit op for trace: {op}")


def bit_op_value(op: str, values: tuple[int, ...]) -> int:
    if op == "XOR" and len(values) == 2:
        return values[0] ^ values[1]
    if op == "MAJ3" and len(values) == 3:
        return 1 if sum(values) >= 2 else 0
    if op == "CHO" and len(values) == 3:
        a, b, c = values
        return (a & b) | ((1 - a) & c)
    if op == "PAR3" and len(values) == 3:
        return values[0] ^ values[1] ^ values[2]
    raise RuntimeError(f"bad bit op/arity: {op} {values}")


def bit_serial_lines(input_bits: str, op: str, transforms: tuple[str, ...], *, prefix: str) -> tuple[str, list[str]]:
    bits = v303.text_to_bits(input_bits)
    transformed = [v303.apply_transform(bits, transform) for transform in transforms]
    output_bits: list[str] = []
    lines = [f"{prefix} input={input_bits}"]
    for idx, transform in enumerate(transforms, 1):
        lines.append(f"{prefix} T{idx}={transform}={v303.bits_to_text(transformed[idx - 1])}")
    for bit_idx in range(v303.BITS):
        values = tuple(vector[bit_idx] for vector in transformed)
        result = bit_op_value(op, values)
        output_bits.append(str(result))
        values_text = ",".join(str(value) for value in values)
        lines.append(f"{prefix} b{bit_idx}: {op}({values_text})={result}")
    output = "".join(output_bits)
    lines.append(f"{prefix} output={output}")
    return output, lines


def bit_trace(prompt: str, answer: str, rule_name: str) -> str:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        raise RuntimeError("bit trace prompt parse failed")
    op, transforms = parse_rule_name(rule_name)
    lines = [
        "Rule: solve output bits independently, then concatenate b0..b7.",
        f"Candidate expression: {rule_name}; {bit_op_description(op)}.",
        "Example verification:",
    ]
    for left, expected in examples[:9]:
        observed = v303.evaluate_expr(op, transforms, left)
        if observed != expected:
            raise RuntimeError(f"bit trace rule mismatch for {left}: {observed} != {expected}")
        lines.append(f"- {left} -> {observed} OK")
    observed_answer, serial_lines = bit_serial_lines(query, op, transforms, prefix="VER")
    if observed_answer != answer:
        raise RuntimeError(f"bit trace answer mismatch: {observed_answer} != {answer}")
    lines.append("Target bit-serial verification:")
    lines.extend(serial_lines)
    lines.append(final_answer_line(answer))
    return "\n".join(lines)


def equation_rule_value(rule_name: str, left: str, right: str) -> str:
    a = int(left)
    b = int(right)
    if rule_name in {"minus_signed_opposite_sign_guarded", "minus_direct_negative_restore_sign"}:
        return str(a - b)
    if rule_name in {"colon_absdiff_unreverse_same_len", "colon_absdiff_restore_trailing_zero"}:
        return str(abs(a - b))
    if rule_name == "add_direct_over_model_add_variant":
        return str(a + b)
    raise RuntimeError(f"unsupported V304 equation rule: {rule_name}")


def equation_rule_description(rule_name: str) -> str:
    if rule_name == "minus_signed_opposite_sign_guarded":
        return "For this '-' operator, compute left minus right and preserve the sign."
    if rule_name == "minus_direct_negative_restore_sign":
        return "For this '-' operator, compute left minus right; if the result is negative, keep the minus sign."
    if rule_name == "colon_absdiff_unreverse_same_len":
        return "For this ':' operator, compute the absolute difference and keep the natural digit order."
    if rule_name == "colon_absdiff_restore_trailing_zero":
        return "For this ':' operator, compute the absolute difference and keep any trailing zero."
    if rule_name == "add_direct_over_model_add_variant":
        return "For this additive operator, compute the direct sum of the two numbers."
    raise RuntimeError(f"unsupported V304 equation rule: {rule_name}")


def equation_trace(prompt: str, answer: str, rule_name: str) -> str:
    examples, query, status = parse_alice_prompt(prompt)
    if status != "ok":
        raise RuntimeError(f"equation trace prompt parse failed: {status}")
    parsed_query = parse_numeric_token(query)
    if parsed_query is None:
        raise RuntimeError(f"equation trace query parse failed: {query!r}")
    lines = [
        "Rule: " + equation_rule_description(rule_name),
        "Check the examples that use this operator:",
    ]
    query_op = parsed_query[1]
    checked = 0
    for lhs, expected in examples:
        parsed = parse_numeric_token(lhs)
        if parsed is None or parsed[1] != query_op:
            continue
        observed = equation_rule_value(rule_name, parsed[0], parsed[2])
        if observed != expected:
            raise RuntimeError(f"equation trace mismatch for {lhs}: {observed} != {expected}")
        lines.append(f"- {lhs} -> {observed}")
        checked += 1
    if checked <= 0:
        raise RuntimeError(f"no equation examples checked for operator {query_op!r}")
    observed_answer = equation_rule_value(rule_name, parsed_query[0], parsed_query[2])
    if observed_answer != answer:
        raise RuntimeError(f"equation trace answer mismatch: {observed_answer} != {answer}")
    lines.extend(
        [
            "Apply the same rule to the query:",
            f"- {query} -> {observed_answer}",
            final_answer_line(answer),
        ]
    )
    return "\n".join(lines)


def convert_v304_patch_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["id"] = str(item.get("id", "")).replace("v303_", "v304_", 1)
    source = str(item.get("source", "")).replace("v303_v300_bit_fullbyte", "v304_solver_trace_bit_fullbyte")
    item["source"] = source
    metadata = dict(item.get("metadata") or {})
    rule_name = str(metadata.get("rule_name", ""))
    metadata.update(
        {
            "source": source,
            "source_dataset": source,
            "v304_role": metadata.get("split", ""),
            "v304_trace_format": "bit_serial_target_verification_trace_v2",
            "v304_training_intent": "teach_solver_rule_not_only_final_answer",
            "weak_gate_rows_used_for_training": False,
        }
    )
    item["metadata"] = metadata
    item["messages"] = trace_messages(
        str(item.get("prompt", "")),
        str(item.get("answer", "")),
        bit_trace(str(item.get("prompt", "")), str(item.get("answer", "")), rule_name),
    )
    return item


def convert_base_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    metadata["weak_gate_rows_used_for_training"] = False
    metadata.setdefault("source", str(item.get("source", "")))
    metadata.setdefault("source_dataset", str(item.get("source", "")))
    metadata.setdefault("split", split)
    metadata.setdefault("family", str(item.get("family", "")))
    metadata.setdefault("subcategory", str(item.get("subcategory", "")))
    item["metadata"] = metadata
    prompt = str(item.get("prompt", ""))
    answer = str(item.get("answer", ""))
    if item.get("source") == "v282_v274_rule_synthetic":
        rule_name = str(metadata.get("rule_name", ""))
        metadata["v304_trace_format"] = "equation_numeric_rule_trace_v1"
        metadata["v304_training_intent"] = "teach_numeric_equation_rule_not_only_final_answer"
        item["messages"] = trace_messages(prompt, answer, equation_trace(prompt, answer, rule_name))
    else:
        item["messages"] = exact_messages(prompt, answer)
    return item


def validate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bad: list[str] = []
    trace_rows = 0
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
        assistant = str(messages[2].get("content", ""))
        if not assistant.rstrip().endswith(final_answer_line(answer)):
            bad.append(f"{row_id}:assistant_final_answer_suffix_mismatch")
        if "\n" in assistant:
            trace_rows += 1
        if metadata.get("weak_gate_rows_used_for_training") is not False:
            bad.append(f"{row_id}:weak_gate_flag_not_false")
        if row_id.startswith("v304_") and row.get("family") != "bit_manipulation":
            bad.append(f"{row_id}:v304_family_mismatch")
    if bad:
        raise RuntimeError(f"{label} row validation failed: {bad[:25]}")
    return {"label": label, "bad_rows": 0, "trace_rows": trace_rows, **v303.summarize(rows)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V304 SOLVER TRACE DISTILL DATASET START ===", flush=True)
    print("generated_at_utc =", v303.utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("seed =", args.seed, flush=True)
    print("base_train_jsonl =", args.base_train_jsonl, flush=True)
    print("base_val_jsonl =", args.base_val_jsonl, flush=True)
    print("reference_csv =", [str(path) for path in args.reference_csv], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train = [convert_base_row(row, "train") for row in v303.read_jsonl(args.base_train_jsonl)]
    base_val = [convert_base_row(row, "validation") for row in v303.read_jsonl(args.base_val_jsonl)]
    patch_train = [convert_v304_patch_row(row) for row in v303.generate_patch_rows(args, "train")]
    patch_val = [convert_v304_patch_row(row) for row in v303.generate_patch_rows(args, "validation")]
    print("patch_train_summary =", json.dumps(v303.summarize(patch_train), sort_keys=True), flush=True)
    print("patch_val_summary =", json.dumps(v303.summarize(patch_val), sort_keys=True), flush=True)

    ref_ids, ref_prompt_hashes = v303.reference_from_csv([path for path in args.reference_csv if path])
    overlap_reports = [
        v303.assert_no_reference_overlap(patch_train, ref_ids, ref_prompt_hashes, "v304_patch_train_vs_reference"),
        v303.assert_no_reference_overlap(patch_val, ref_ids, ref_prompt_hashes, "v304_patch_val_vs_reference"),
    ]

    train_rows, train_dedupe = v303.dedupe(base_train + patch_train)
    val_rows, val_dedupe = v303.dedupe(base_val + patch_val)
    rng = v303.random.Random(args.seed + 304)
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

    train_path = args.output_dir / "v304_solver_trace_distill_train.jsonl"
    val_path = args.output_dir / "v304_solver_trace_distill_val.jsonl"
    manifest_path = args.output_dir / "v304_solver_trace_distill_manifest.json"
    v303.write_jsonl(train_path, train_rows)
    v303.write_jsonl(val_path, val_rows)
    manifest = {
        "schema_version": "kg1_v304_solver_trace_distill_dataset_v1",
        "generated_at_utc": v303.utc_now(),
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
            "exact_gain_patterns": [v303.expr_name(op, transforms) for op, transforms in v303.EXACT_GAIN_PATTERNS],
        },
        "dedupe": {"train": train_dedupe, "validation": val_dedupe},
        "overlap_reports": overlap_reports,
        "validation": {"train": train_validation, "validation": val_validation},
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": v303.sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": v303.sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "decision": {
            "status": "dataset_ready_for_trace_tokenization_gate",
            "next_action": (
                "Run V286 tokenization gate with assistant-final-answer-mode=suffix. "
                "Only launch HF smoke if prompt truncation is zero and trace lengths stay inside budget."
            ),
            "reason": (
                f"trace_train_rows={train_validation['trace_rows']}; "
                f"trace_val_rows={val_validation['trace_rows']}"
            ),
        },
    }
    v303.write_json(manifest_path, manifest)
    print("v304_dataset_manifest =", json.dumps(manifest, sort_keys=True), flush=True)
    print("=== V304 SOLVER TRACE DISTILL DATASET END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v304_solver_trace_distill_dataset"))
    parser.add_argument("--label", default="v304_solver_trace_distill")
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
        eq_prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:\n"
            "28-88 = -60\n95-97 = -2\n33-37 = -4\n"
            "Now, determine the result for: 14-62"
        )
        eq_row = {
            "id": "v282_train_minus_signed_test",
            "prompt": eq_prompt,
            "answer": "-48",
            "family": "equation_transform",
            "subcategory": "equation_numeric_minus_signed",
            "source": "v282_v274_rule_synthetic",
            "messages": exact_messages(eq_prompt, "-48"),
            "metadata": {
                "source": "v282_v274_rule_synthetic",
                "rule_name": "minus_signed_opposite_sign_guarded",
                "weak_gate_rows_used_for_training": False,
            },
        }
        v303.write_jsonl(base_train, [eq_row])
        v303.write_jsonl(base_val, [dict(eq_row, id="v282_val_minus_signed_test")])
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
                "304",
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
        train_rows = v303.read_jsonl(Path(manifest["outputs"]["train_jsonl"]))
        if not any("\n" in str(row["messages"][-1]["content"]) for row in train_rows):
            raise RuntimeError("self-test expected trace rows")
    print("v304_solver_trace_distill_dataset_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify V281 ReasoningGym fixtures with local deterministic solvers.

V282 consumes the V281 selected rows and measures which source datasets have
trustworthy CPU verifiers. This is a gate for future data/probe use only. It
does not train, generate model predictions, package, or submit.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import operator
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}
ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int_expr(expr: str, names: dict[str, int] | None = None) -> int:
    names = names or {}
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return int(names[node.id])
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

    return int(visit(tree))


def to_base(value: int, base: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    out = []
    while value:
        value, rem = divmod(value, base)
        out.append(digits[rem])
    return sign + "".join(reversed(out))


def solve_base_conversion(question: str) -> str:
    base_names = {"binary": 2, "octal": 8, "decimal": 10, "hexadecimal": 16}
    match = re.search(r"convert the base-(\d+) number ([0-9a-zA-Z]+) to base-(\d+)", question, re.I)
    if not match:
        named = re.search(r"convert the (binary|octal|decimal|hexadecimal) number ([0-9a-zA-Z]+) to base-(\d+)", question, re.I)
        if named:
            src_base = base_names[named.group(1).lower()]
            number = named.group(2).lower()
            dst_base = int(named.group(3))
            return to_base(int(number, src_base), dst_base)
    if not match:
        raise ValueError("base conversion pattern not found")
    src_base = int(match.group(1))
    number = match.group(2).lower()
    dst_base = int(match.group(3))
    return to_base(int(number, src_base), dst_base)


def solve_count_bits(question: str) -> str:
    match = re.search(r"number\s+(-?\d+)", question)
    if not match:
        raise ValueError("count_bits number not found")
    value = int(match.group(1))
    if value < 0:
        raise ValueError("negative count_bits unsupported")
    return str(bin(value).count("1"))


def swaps_to_pattern(bits: str, first: str) -> int | None:
    target = "".join(first if i % 2 == 0 else ("1" if first == "0" else "0") for i in range(len(bits)))
    if target.count("0") != bits.count("0") or target.count("1") != bits.count("1"):
        return None
    mismatches = sum(a != b for a, b in zip(bits, target))
    return mismatches // 2


def solve_binary_alternation(question: str) -> str:
    match = re.search(r"binary string alternating:\s*([01]+)", question, re.I)
    if not match:
        raise ValueError("binary_alternation string not found")
    bits = match.group(1)
    if abs(bits.count("0") - bits.count("1")) > 1:
        return "-1"
    candidates = [value for value in (swaps_to_pattern(bits, "0"), swaps_to_pattern(bits, "1")) if value is not None]
    return str(min(candidates)) if candidates else "-1"


def solve_bitwise_arithmetic(question: str) -> str:
    expr = question.strip().splitlines()[-1].strip()
    value = safe_int_expr(expr)
    return ("-0x" + format(abs(value), "x")) if value < 0 else hex(value)


def eval_linear_expr(expr: str, variable: str, x: int) -> int:
    return safe_int_expr(expr.replace("^", "**"), {variable: x})


def solve_simple_equation(question: str) -> str:
    match = re.search(r"(?:equation|satisfies):\s*(.+?)\s*$", question, re.I | re.S)
    if not match:
        match = re.search(r"Solve for [a-zA-Z]:\s*(.+?)\s*$", question, re.I | re.S)
    if not match or "=" not in match.group(1):
        raise ValueError("simple equation not found")
    equation = match.group(1).strip()
    left, right = [part.strip() for part in equation.split("=", 1)]
    variables = sorted(set(re.findall(r"\b[a-zA-Z]\b", equation)))
    if len(variables) != 1:
        raise ValueError("expected one variable")
    variable = variables[0]
    f0 = Fraction(eval_linear_expr(left, variable, 0) - eval_linear_expr(right, variable, 0))
    f1 = Fraction(eval_linear_expr(left, variable, 1) - eval_linear_expr(right, variable, 1))
    slope = f1 - f0
    if slope == 0:
        raise ValueError("zero slope")
    answer = -f0 / slope
    return str(answer.numerator) if answer.denominator == 1 else str(float(answer))


def solve_number_format(question: str) -> str:
    mode_match = re.search(r"pick the (smallest|largest) number", question, re.I)
    cand_match = re.search(r"candidates:\s*(.+?)\s*$", question, re.I | re.S)
    if not mode_match or not cand_match:
        raise ValueError("number_format pattern not found")
    mode = mode_match.group(1).lower()
    raw_candidates = cand_match.group(1).strip().split()
    parsed = [(float(item.replace(",", "")), item.replace(",", "")) for item in raw_candidates]
    value, original = min(parsed) if mode == "smallest" else max(parsed)
    if "e" in original.lower():
        return str(value).rstrip("0").rstrip(".")
    return original


def parse_mapping(answer: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for part in answer.split(","):
        if "=" not in part:
            continue
        key, value = [item.strip() for item in part.split("=", 1)]
        if not re.fullmatch(r"[A-Z]", key) or not re.fullmatch(r"\d", value):
            raise ValueError("bad mapping token: " + part)
        mapping[key] = int(value)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("duplicate digit in mapping")
    return mapping


def word_value(word: str, mapping: dict[str, int]) -> int:
    if len(word) > 1 and mapping[word[0]] == 0:
        raise ValueError("leading zero")
    return int("".join(str(mapping[ch]) for ch in word))


def solve_cryptarithm_verify(question: str, answer: str) -> str:
    mapping = parse_mapping(answer)
    block = re.split(r"-{3,}", question, maxsplit=1)
    if len(block) != 2:
        raise ValueError("cryptarithm separator not found")
    lhs_words = re.findall(r"\b[A-Z]{2,}\b", block[0])
    rhs_words = re.findall(r"\b[A-Z]{2,}\b", block[1])
    if not lhs_words or not rhs_words:
        raise ValueError("cryptarithm words not found")
    lhs_sum = sum(word_value(word, mapping) for word in lhs_words)
    rhs_value = word_value(rhs_words[0], mapping)
    if lhs_sum != rhs_value:
        raise ValueError(f"mapping does not satisfy equation: {lhs_sum} != {rhs_value}")
    return ",".join(f"{key}={mapping[key]}" for key in sorted(mapping))


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def answers_match(expected: str, observed: str) -> bool:
    if normalize_answer(expected) == normalize_answer(observed):
        return True
    try:
        return math.isclose(float(str(expected).replace(",", "")), float(str(observed).replace(",", "")), rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        return False


def solve_row(row: dict[str, Any]) -> tuple[str, str, str]:
    source = str(row.get("source_dataset", ""))
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    if source == "base_conversion":
        return "verified", solve_base_conversion(question), "base conversion parser"
    if source == "binary_alternation":
        return "verified", solve_binary_alternation(question), "binary alternation swaps"
    if source == "bitwise_arithmetic":
        return "verified", solve_bitwise_arithmetic(question), "safe integer AST"
    if source == "count_bits":
        return "verified", solve_count_bits(question), "binary popcount"
    if source == "simple_equations":
        return "verified", solve_simple_equation(question), "linear equation by finite difference"
    if source == "number_format":
        return "verified", solve_number_format(question), "numeric candidate comparison"
    if source == "cryptarithm":
        return "verified", solve_cryptarithm_verify(question, answer), "provided mapping verifier"
    return "unsupported", "", "no local verifier for source"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V282 REASONINGGYM VERIFIER PROBES START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("selected_rows_jsonl =", args.selected_rows_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.selected_rows_jsonl)
    audit_rows: list[dict[str, Any]] = []
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = str(row.get("source_dataset", ""))
        try:
            status, prediction, proof = solve_row(row)
        except Exception as exc:
            status, prediction, proof = "error", "", repr(exc)[:500]
        expected = str(row.get("answer", ""))
        match = answers_match(expected, prediction) if status == "verified" else False
        final_status = "verified_match" if match else ("verified_mismatch" if status == "verified" else status)
        summary[source][final_status] += 1
        audit_rows.append(
            {
                "uuid": row.get("uuid", ""),
                "source_dataset": source,
                "kg1_relevance": row.get("kg1_relevance", ""),
                "status": final_status,
                "expected": expected,
                "prediction": prediction,
                "proof": proof,
                "question_sha256": row.get("question_sha256", ""),
            }
        )
    summary_rows = []
    for source in sorted(summary):
        counts = summary[source]
        total = sum(counts.values())
        verified_match = int(counts.get("verified_match", 0))
        summary_rows.append(
            {
                "source_dataset": source,
                "total": total,
                "verified_match": verified_match,
                "verified_mismatch": int(counts.get("verified_mismatch", 0)),
                "error": int(counts.get("error", 0)),
                "unsupported": int(counts.get("unsupported", 0)),
                "verified_match_rate": round(verified_match / total, 6) if total else 0.0,
            }
        )
    audit_path = args.output_dir / "v282_reasoninggym_verifier_audit.csv"
    summary_path = args.output_dir / "v282_reasoninggym_verifier_summary.csv"
    manifest_path = args.output_dir / "v282_reasoninggym_verifier_manifest.json"
    write_csv(audit_path, audit_rows, ["uuid", "source_dataset", "kg1_relevance", "status", "expected", "prediction", "proof", "question_sha256"])
    write_csv(summary_path, summary_rows, ["source_dataset", "total", "verified_match", "verified_mismatch", "error", "unsupported", "verified_match_rate"])
    total_verified = sum(row["verified_match"] for row in summary_rows)
    total_mismatch = sum(row["verified_mismatch"] for row in summary_rows)
    manifest = {
        "schema_version": "kg1_v282_reasoninggym_verifier_probes_v1",
        "generated_at_utc": utc_now(),
        "selected_rows_jsonl": str(args.selected_rows_jsonl),
        "rows": len(rows),
        "summary": summary_rows,
        "totals": {
            "verified_match": total_verified,
            "verified_mismatch": total_mismatch,
            "unsupported_or_error": len(rows) - total_verified - total_mismatch,
        },
        "decision": {
            "decision": "reasoninggym_verified_fixtures_ready_for_probe_design" if total_verified >= args.min_verified and total_mismatch == 0 else "reasoninggym_verifier_probe_not_ready",
            "reason": f"verified_match={total_verified}; verified_mismatch={total_mismatch}; rows={len(rows)}",
            "next_action": "Use verified source families only for CPU probes/dataset design; do not train directly.",
        },
        "outputs": {
            "audit_csv": str(audit_path),
            "summary_csv": str(summary_path),
            "manifest_json": str(manifest_path),
        },
        "blocked_actions": ["gpu_train", "model_generation", "full_eval", "package", "kaggle_submit"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("totals =", json.dumps(manifest["totals"], sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V282 REASONINGGYM VERIFIER PROBES END ===", flush=True)
    return manifest


def self_test() -> None:
    assert solve_base_conversion("Now, convert the base-6 number 4125 to base-5") == "12132"
    assert solve_count_bits("How many 1 bits are there in the binary representation of the number 1025593251?") == "16"
    assert solve_binary_alternation("Now, determine the minimum number of swaps to make the following binary string alternating: 1110110000101") == "2"
    assert solve_bitwise_arithmetic("Reply only with the final hexidecimal value.\n((0xd32d >> 0x0) << 0x1)") == "0x1a65a"
    assert solve_simple_equation("Find the value of z in the equation: 465*z + 806 = 37541") == "79"
    print("v282_reasoninggym_verifier_probes_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-rows-jsonl", type=Path, default=Path("artifacts/v281_reasoninggym_cpu_triage/20260511T1835Z/v281_reasoninggym_selected_rows.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v282_reasoninggym_verifier_probes") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--min-verified", type=int, default=900)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

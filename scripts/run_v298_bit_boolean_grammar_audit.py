#!/usr/bin/env python3
"""Audit an original bit boolean-grammar solver on labeled KG1 rows.

This is CPU-only evidence gathering. It intentionally does not copy external
solver code. The audit tests whether a conservative per-output-bit boolean
grammar can add verified coverage over the current local bit solver before any
training or HF GPU spend.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem

N_BITS = 8


@dataclass(frozen=True)
class Rule:
    level: str
    name: str
    positions: tuple[int, ...]
    signature: str
    query_value: str


def invert_sig(sig: str) -> str:
    return "".join("1" if ch == "0" else "0" for ch in sig)


def sig_from_func(columns: list[str], positions: tuple[int, ...], func: Callable[..., int]) -> str:
    return "".join(
        str(func(*(1 if columns[pos][row_idx] == "1" else 0 for pos in positions)))
        for row_idx in range(len(columns[0]))
    )


def value_from_func(query: str, positions: tuple[int, ...], func: Callable[..., int]) -> str:
    return str(func(*(1 if query[pos] == "1" else 0 for pos in positions)))


def row_family(row: pd.Series) -> str:
    direct = str(row.get("type", "") or row.get("family", "")).strip()
    if direct:
        return direct
    prompt = str(row.get("prompt", "")).lower()
    if "bit manipulation" in prompt or "binary" in prompt:
        return "bit_manipulation"
    return ""


UNARY_FUNCS: list[tuple[str, Callable[[int], int]]] = [
    ("ID", lambda a: a),
    ("NOT", lambda a: 1 - a),
]

BINARY_FUNCS: list[tuple[str, Callable[[int, int], int]]] = [
    ("AND", lambda a, b: a & b),
    ("OR", lambda a, b: a | b),
    ("XOR", lambda a, b: a ^ b),
    ("XNOR", lambda a, b: 1 - (a ^ b)),
    ("NAND", lambda a, b: 1 - (a & b)),
    ("NOR", lambda a, b: 1 - (a | b)),
    ("AND_NOT", lambda a, b: a & (1 - b)),
    ("NOT_AND", lambda a, b: (1 - a) & b),
    ("OR_NOT", lambda a, b: a | (1 - b)),
    ("NOT_OR", lambda a, b: (1 - a) | b),
]

TERNARY_FUNCS: list[tuple[str, Callable[[int, int, int], int]]] = [
    ("MAJ", lambda a, b, c: 1 if a + b + c >= 2 else 0),
    ("CHO", lambda a, b, c: (a & b) | ((1 - a) & c)),
    ("PAR3", lambda a, b, c: a ^ b ^ c),
    ("AO", lambda a, b, c: (a & b) | c),
    ("OA", lambda a, b, c: (a | b) & c),
    ("AX", lambda a, b, c: (a & b) ^ c),
    ("XA", lambda a, b, c: (a ^ b) & c),
    ("OX", lambda a, b, c: (a | b) ^ c),
    ("XO", lambda a, b, c: (a ^ b) | c),
]

QUATERNARY_FUNCS: list[tuple[str, Callable[[int, int, int, int], int]]] = [
    ("PAR4", lambda a, b, c, d: a ^ b ^ c ^ d),
    ("AOA", lambda a, b, c, d: (a & b) | (c & d)),
    ("OAO", lambda a, b, c, d: (a | b) & (c | d)),
    ("XX", lambda a, b, c, d: (a ^ b) ^ (c ^ d)),
    ("AXA", lambda a, b, c, d: (a & b) ^ (c & d)),
]


def find_rule_for_bit(columns: list[str], output_sig: str, query: str, out_pos: int, max_level: int) -> Rule:
    if output_sig == "0" * len(output_sig):
        return Rule("constant", "CONST0", (), output_sig, "0")
    if output_sig == "1" * len(output_sig):
        return Rule("constant", "CONST1", (), output_sig, "1")

    for pos in range(N_BITS):
        for name, func in UNARY_FUNCS:
            positions = (pos,)
            sig = sig_from_func(columns, positions, func)
            if sig == output_sig:
                return Rule("unary", name, positions, sig, value_from_func(query, positions, func))

    if max_level >= 2:
        for i, j in itertools.permutations(range(N_BITS), 2):
            for name, func in BINARY_FUNCS:
                positions = (i, j)
                sig = sig_from_func(columns, positions, func)
                if sig == output_sig:
                    return Rule("binary", name, positions, sig, value_from_func(query, positions, func))

    if max_level >= 3:
        for i, j, k in itertools.permutations(range(N_BITS), 3):
            for name, func in TERNARY_FUNCS:
                positions = (i, j, k)
                sig = sig_from_func(columns, positions, func)
                if sig == output_sig:
                    return Rule("ternary", name, positions, sig, value_from_func(query, positions, func))

    if max_level >= 4:
        for i, j, k, l in itertools.permutations(range(N_BITS), 4):
            for name, func in QUATERNARY_FUNCS:
                positions = (i, j, k, l)
                sig = sig_from_func(columns, positions, func)
                if sig == output_sig:
                    return Rule("quaternary", name, positions, sig, value_from_func(query, positions, func))

    return Rule("fallback", f"DEFAULT_OUT{out_pos}_0", (), "", "0")


def solve_boolean_grammar(prompt: str, max_level: int) -> tuple[str | None, dict[str, object]]:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        return None, {"status": "parse_failed", "rules": []}
    if any(len(inp) != N_BITS or len(out) != N_BITS for inp, out in examples) or len(query) != N_BITS:
        return None, {"status": "invalid_bit_width", "rules": []}

    input_columns = ["".join(inp[pos] for inp, _ in examples) for pos in range(N_BITS)]
    output_columns = ["".join(out[pos] for _, out in examples) for pos in range(N_BITS)]
    rules = [
        find_rule_for_bit(input_columns, output_columns[out_pos], query, out_pos, max_level)
        for out_pos in range(N_BITS)
    ]
    answer = "".join(rule.query_value for rule in rules)
    status = "ok" if all(rule.level != "fallback" for rule in rules) else "partial"
    return answer, {
        "status": status,
        "rules": [
            {
                "level": rule.level,
                "name": rule.name,
                "positions": rule.positions,
                "query_value": rule.query_value,
            }
            for rule in rules
        ],
        "fallback_bits": sum(1 for rule in rules if rule.level == "fallback"),
        "max_rule_level": max(("constant", "unary", "binary", "ternary", "quaternary", "fallback").index(rule.level) for rule in rules),
    }


def summarize_level_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for level in str(row.get("rule_levels", "")).split():
            if level:
                counts[level] = counts.get(level, 0) + 1
    return counts


def audit_csv(input_csv: Path, output_dir: Path, label: str, max_level: int, save_details: bool) -> dict[str, object]:
    df = pd.read_csv(input_csv)
    bit_df = df[df.apply(row_family, axis=1).eq("bit_manipulation")].copy()
    current_solver = BitManipulationSolver()
    rows: list[dict[str, object]] = []
    grammar_correct = 0
    current_correct = 0
    both_correct = 0
    grammar_gain = 0
    grammar_loss = 0
    parse_failed = 0
    partial = 0
    t0 = time.time()

    for idx, row in bit_df.iterrows():
        expected = str(row["answer"]).strip()
        grammar_answer, meta = solve_boolean_grammar(str(row["prompt"]), max_level)
        current_answer, _, _ = current_solver.solve(str(row["prompt"]))
        g_ok = grammar_answer == expected
        c_ok = current_answer == expected
        grammar_correct += int(g_ok)
        current_correct += int(c_ok)
        both_correct += int(g_ok and c_ok)
        grammar_gain += int(g_ok and not c_ok)
        grammar_loss += int(c_ok and not g_ok)
        parse_failed += int(meta.get("status") == "parse_failed")
        partial += int(meta.get("status") == "partial")
        rules = meta.get("rules", [])
        rule_levels = " ".join(str(rule.get("level")) for rule in rules) if isinstance(rules, list) else ""
        if save_details or g_ok != c_ok:
            rows.append(
                {
                    "id": row.get("id"),
                    "expected": expected,
                    "grammar_answer": grammar_answer,
                    "current_answer": current_answer,
                    "grammar_correct": g_ok,
                    "current_correct": c_ok,
                    "status": meta.get("status"),
                    "fallback_bits": meta.get("fallback_bits"),
                    "rule_levels": rule_levels,
                    "rule_names": " ".join(str(rule.get("name")) for rule in rules) if isinstance(rules, list) else "",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / f"{label}_v298_bit_boolean_grammar_details.csv"
    if rows:
        pd.DataFrame(rows).to_csv(detail_csv, index=False)
    summary = {
        "schema_version": "kg1_v298_bit_boolean_grammar_audit_v1",
        "input_csv": str(input_csv),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "label": label,
        "max_level": int(max_level),
        "rows_total": int(len(df)),
        "bit_rows": int(len(bit_df)),
        "grammar_correct": int(grammar_correct),
        "grammar_accuracy": grammar_correct / len(bit_df) if len(bit_df) else 0.0,
        "current_correct": int(current_correct),
        "current_accuracy": current_correct / len(bit_df) if len(bit_df) else 0.0,
        "both_correct": int(both_correct),
        "grammar_gain_vs_current": int(grammar_gain),
        "grammar_loss_vs_current": int(grammar_loss),
        "parse_failed": int(parse_failed),
        "partial_rows": int(partial),
        "detail_csv": str(detail_csv) if rows else "",
        "detail_rows": int(len(rows)),
        "rule_level_counts_in_details": summarize_level_counts(rows),
        "elapsed_s": round(time.time() - t0, 3),
    }
    (output_dir / f"{label}_v298_bit_boolean_grammar_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="bit_boolean_grammar")
    parser.add_argument("--max-level", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--save-details", action="store_true")
    args = parser.parse_args()
    summary = audit_csv(args.input_csv, args.output_dir, args.label, args.max_level, args.save_details)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

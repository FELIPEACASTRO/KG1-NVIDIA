#!/usr/bin/env python3
"""Audit conservative full-byte bit grammar overrides.

Unlike V298's free per-bit grammar, V300 only accepts one byte-level expression
that matches every output bit of every example. This follows the lower-divergence
direction identified in Kaggle discussions without copying external code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem

BITS = 8
Vec = tuple[int, ...]


def to_vec(text: str) -> Vec:
    return tuple(1 if ch == "1" else 0 for ch in str(text).strip())


def from_vec(vec: Vec) -> str:
    return "".join(str(bit) for bit in vec)


def not_vec(a: Vec) -> Vec:
    return tuple(1 - x for x in a)


def and_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x & y for x, y in zip(a, b))


def or_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x | y for x, y in zip(a, b))


def xor_vec(a: Vec, b: Vec) -> Vec:
    return tuple(x ^ y for x, y in zip(a, b))


def nand_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(and_vec(a, b))


def nor_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(or_vec(a, b))


def xnor_vec(a: Vec, b: Vec) -> Vec:
    return not_vec(xor_vec(a, b))


def and_not_vec(a: Vec, b: Vec) -> Vec:
    return and_vec(a, not_vec(b))


def not_and_vec(a: Vec, b: Vec) -> Vec:
    return and_vec(not_vec(a), b)


def or_not_vec(a: Vec, b: Vec) -> Vec:
    return or_vec(a, not_vec(b))


def not_or_vec(a: Vec, b: Vec) -> Vec:
    return or_vec(not_vec(a), b)


BINARY_OPS: list[tuple[str, Callable[[Vec, Vec], Vec]]] = [
    ("AND", and_vec),
    ("OR", or_vec),
    ("XOR", xor_vec),
    ("XNOR", xnor_vec),
    ("NAND", nand_vec),
    ("NOR", nor_vec),
    ("AND_NOT", and_not_vec),
    ("NOT_AND", not_and_vec),
    ("OR_NOT", or_not_vec),
    ("NOT_OR", not_or_vec),
]

TERNARY_OPS: list[tuple[str, Callable[[Vec, Vec, Vec], Vec]]] = [
    ("PAR3", lambda a, b, c: xor_vec(xor_vec(a, b), c)),
    ("MAJ3", lambda a, b, c: tuple(1 if x + y + z >= 2 else 0 for x, y, z in zip(a, b, c))),
    ("CHO", lambda a, b, c: or_vec(and_vec(a, b), and_vec(not_vec(a), c))),
    ("AND_OR", lambda a, b, c: or_vec(and_vec(a, b), c)),
    ("OR_AND", lambda a, b, c: and_vec(or_vec(a, b), c)),
    ("AND_XOR", lambda a, b, c: xor_vec(and_vec(a, b), c)),
    ("XOR_AND", lambda a, b, c: and_vec(xor_vec(a, b), c)),
    ("OR_XOR", lambda a, b, c: xor_vec(or_vec(a, b), c)),
    ("XOR_OR", lambda a, b, c: or_vec(xor_vec(a, b), c)),
]


def transforms(vec: Vec) -> list[tuple[str, Vec]]:
    rows: list[tuple[str, Vec]] = [("ID", vec), ("NOT", not_vec(vec))]
    for k in range(1, BITS):
        rows.append((f"ROL{k}", vec[k:] + vec[:k]))
    for k in range(1, BITS):
        rows.append((f"ROR{k}", vec[-k:] + vec[:-k]))
    for k in range(1, BITS):
        rows.append((f"SHL{k}", vec[k:] + tuple(0 for _ in range(k))))
    for k in range(1, BITS):
        rows.append((f"SHR{k}", tuple(0 for _ in range(k)) + vec[:-k]))
    return rows


def row_family(row: pd.Series) -> str:
    direct = str(row.get("type", "") or row.get("family", "")).strip()
    if direct:
        return direct
    prompt = str(row.get("prompt", "")).lower()
    if "bit manipulation" in prompt or "binary" in prompt:
        return "bit_manipulation"
    return ""


def solve_fullbyte(prompt: str, max_level: int, ternary_allowlist: set[str] | None = None) -> tuple[str | None, dict[str, object]]:
    examples, query = parse_bit_problem(prompt)
    if not examples or not query:
        return None, {"status": "parse_failed"}
    inputs = [to_vec(inp) for inp, _ in examples]
    outputs = [to_vec(out) for _, out in examples]
    query_vec = to_vec(query)
    if any(len(vec) != BITS for vec in inputs + outputs + [query_vec]):
        return None, {"status": "invalid_width"}

    all_inputs = inputs + [query_vec]
    per_transform: dict[str, list[Vec]] = {}
    for name, _ in transforms(query_vec):
        per_transform[name] = []
    for vec in all_inputs:
        for name, out in transforms(vec):
            per_transform[name].append(out)
    names = list(per_transform)
    n = len(inputs)

    for name in names:
        if all(per_transform[name][idx] == outputs[idx] for idx in range(n)):
            return from_vec(per_transform[name][n]), {"status": "ok", "level": "unary", "expr": name}

    if max_level >= 2:
        for left in names:
            for right in names:
                for op_name, op in BINARY_OPS:
                    if all(op(per_transform[left][idx], per_transform[right][idx]) == outputs[idx] for idx in range(n)):
                        pred = op(per_transform[left][n], per_transform[right][n])
                        return from_vec(pred), {"status": "ok", "level": "binary", "expr": f"{op_name}({left},{right})"}

    if max_level >= 3:
        for a in names:
            for b in names:
                for c in names:
                    for op_name, op in TERNARY_OPS:
                        if ternary_allowlist is not None and op_name not in ternary_allowlist:
                            continue
                        if all(
                            op(per_transform[a][idx], per_transform[b][idx], per_transform[c][idx]) == outputs[idx]
                            for idx in range(n)
                        ):
                            pred = op(per_transform[a][n], per_transform[b][n], per_transform[c][n])
                            return from_vec(pred), {"status": "ok", "level": "ternary", "expr": f"{op_name}({a},{b},{c})"}

    return None, {"status": "no_expression"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, object]:
    df = pd.read_csv(args.input_csv)
    bit_df = df[df.apply(row_family, axis=1).eq("bit_manipulation")].copy()
    current_solver = BitManipulationSolver()
    rows: list[dict[str, object]] = []
    full_correct = 0
    current_correct = 0
    baseline_correct = 0
    ensemble_correct = 0
    baseline_ensemble_correct = 0
    expression_rows = 0
    gains = 0
    losses = 0
    baseline_gains = 0
    baseline_losses = 0
    level_counts: dict[str, int] = {}
    t0 = time.time()
    ternary_allowlist = (
        {item.strip() for item in str(args.ternary_op_allowlist).split(",") if item.strip()}
        if args.ternary_op_allowlist
        else None
    )

    for _, row in bit_df.iterrows():
        expected = str(row["answer"]).strip()
        baseline_answer = str(row.get("prediction", "")).strip()
        full_answer, meta = solve_fullbyte(str(row["prompt"]), args.max_level, ternary_allowlist)
        current_answer, _, _ = current_solver.solve(str(row["prompt"]))
        chosen = full_answer if meta.get("status") == "ok" else current_answer
        baseline_chosen = full_answer if meta.get("status") == "ok" and args.compare_input_prediction else baseline_answer
        f_ok = full_answer == expected
        c_ok = current_answer == expected
        b_ok = baseline_answer == expected
        e_ok = chosen == expected
        be_ok = baseline_chosen == expected
        full_correct += int(f_ok)
        current_correct += int(c_ok)
        baseline_correct += int(b_ok)
        ensemble_correct += int(e_ok)
        baseline_ensemble_correct += int(be_ok)
        expression_rows += int(meta.get("status") == "ok")
        level = str(meta.get("level", "none"))
        level_counts[level] = level_counts.get(level, 0) + 1
        gains += int(e_ok and not c_ok)
        losses += int(c_ok and not e_ok)
        baseline_gains += int(be_ok and not b_ok)
        baseline_losses += int(b_ok and not be_ok)
        if args.save_details or f_ok != c_ok or chosen != current_answer or (args.compare_input_prediction and baseline_chosen != baseline_answer):
            rows.append(
                {
                    "id": row.get("id"),
                    "expected": expected,
                    "fullbyte_answer": full_answer,
                    "current_answer": current_answer,
                    "baseline_answer": baseline_answer,
                    "chosen_answer": chosen,
                    "baseline_chosen_answer": baseline_chosen,
                    "fullbyte_correct": f_ok,
                    "current_correct": c_ok,
                    "baseline_correct": b_ok,
                    "ensemble_correct": e_ok,
                    "baseline_ensemble_correct": be_ok,
                    "status": meta.get("status"),
                    "level": meta.get("level", ""),
                    "expr": meta.get("expr", ""),
                    "gain": e_ok and not c_ok,
                    "loss": c_ok and not e_ok,
                    "baseline_gain": be_ok and not b_ok,
                    "baseline_loss": b_ok and not be_ok,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.output_dir / f"{args.label}_v300_bit_fullbyte_grammar_details.csv"
    write_csv(detail_csv, rows)
    summary = {
        "schema_version": "kg1_v300_bit_fullbyte_grammar_audit_v1",
        "input_csv": str(args.input_csv),
        "input_sha256": hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        "label": args.label,
        "max_level": args.max_level,
        "rows_total": int(len(df)),
        "bit_rows": int(len(bit_df)),
        "expression_rows": int(expression_rows),
        "level_counts": level_counts,
        "fullbyte_correct": int(full_correct),
        "fullbyte_accuracy": full_correct / len(bit_df) if len(bit_df) else 0.0,
        "current_correct": int(current_correct),
        "current_accuracy": current_correct / len(bit_df) if len(bit_df) else 0.0,
        "baseline_prediction_correct": int(baseline_correct),
        "baseline_prediction_accuracy": baseline_correct / len(bit_df) if len(bit_df) else 0.0,
        "ensemble_correct": int(ensemble_correct),
        "ensemble_accuracy": ensemble_correct / len(bit_df) if len(bit_df) else 0.0,
        "ensemble_gain_vs_current": int(gains),
        "ensemble_loss_vs_current": int(losses),
        "baseline_ensemble_correct": int(baseline_ensemble_correct),
        "baseline_ensemble_accuracy": baseline_ensemble_correct / len(bit_df) if len(bit_df) else 0.0,
        "baseline_ensemble_gain_vs_input_prediction": int(baseline_gains),
        "baseline_ensemble_loss_vs_input_prediction": int(baseline_losses),
        "compare_input_prediction": bool(args.compare_input_prediction),
        "ternary_op_allowlist": sorted(ternary_allowlist) if ternary_allowlist is not None else [],
        "detail_csv": str(detail_csv) if rows else "",
        "detail_rows": len(rows),
        "elapsed_s": round(time.time() - t0, 3),
    }
    (args.output_dir / f"{args.label}_v300_bit_fullbyte_grammar_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="weak")
    parser.add_argument("--max-level", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--compare-input-prediction", action="store_true")
    parser.add_argument("--ternary-op-allowlist", default="")
    args = parser.parse_args()
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

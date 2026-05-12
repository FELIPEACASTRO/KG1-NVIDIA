#!/usr/bin/env python3
"""Audit a stride-based bit solver inspired by public Kaggle discussion 690307.

This is an audit/teacher-signal tool only. It does not authorize direct solver
submission. It measures whether the public per-output-bit relation strategy
adds verified coverage over the local solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem

N_BITS = 8
SECTION_ORDER = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)
PAIR_FAMILIES = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")


@dataclass(frozen=True)
class Candidate:
    family: str
    primary: Optional[int]
    secondary: Optional[int]
    expr: str


DEFAULT = Candidate("DEFAULT", None, None, "default 1")


def norm_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    return bits if len(bits) == N_BITS else ""


def invert(bits: str) -> str:
    return "".join("1" if b == "0" else "0" for b in bits)


def col(values: list[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def eval_pair(a: str, b: str, family: str) -> str:
    if family.endswith("-NOT"):
        b = "1" if b == "0" else "0"
        family = family.removesuffix("-NOT")
    if family == "AND":
        return "1" if a == "1" and b == "1" else "0"
    if family == "OR":
        return "1" if a == "1" or b == "1" else "0"
    if family == "XOR":
        return "1" if a != b else "0"
    raise ValueError(family)


def apply_candidate(bits: str, cand: Candidate) -> str:
    if cand.family == "DEFAULT":
        return "1"
    if cand.family == "0":
        return "0"
    if cand.family == "1":
        return "1"
    if cand.family == "I":
        return bits[int(cand.primary)]
    if cand.family == "NOT":
        return invert(bits[int(cand.primary)])
    return eval_pair(bits[int(cand.primary)], bits[int(cand.secondary)], cand.family)


def find_match(candidates: list[Candidate], family: str, p: Optional[int], s: Optional[int]) -> Optional[Candidate]:
    for cand in candidates:
        if cand.family != family:
            continue
        if cand.primary == p and (family not in PAIR_FAMILIES or cand.secondary == s):
            return cand
    return None


def left_runs(per_bit: list[list[Candidate]]) -> list[list[Candidate]]:
    if not per_bit or not per_bit[0]:
        return []
    runs: list[list[Candidate]] = []
    for start in per_bit[0]:
        run = [start]
        p, s = start.primary, start.secondary
        for bit in range(1, N_BITS):
            ep = (int(p) + 1) % N_BITS if p is not None else None
            es = (int(s) + 1) % N_BITS if s is not None else None
            found = find_match(per_bit[bit], start.family, ep, es)
            if found is None:
                break
            run.append(found)
            p, s = ep, es
        runs.append(run)
    return runs


def right_runs(per_bit: list[list[Candidate]]) -> list[list[Candidate]]:
    if not per_bit or not per_bit[-1]:
        return []
    runs: list[list[Candidate]] = []
    for end in per_bit[-1]:
        run = [end]
        p, s = end.primary, end.secondary
        for k in range(1, N_BITS):
            bit = N_BITS - 1 - k
            ep = (int(p) - 1) % N_BITS if p is not None else None
            es = (int(s) - 1) % N_BITS if s is not None else None
            found = find_match(per_bit[bit], end.family, ep, es)
            if found is None:
                break
            run.insert(0, found)
            p, s = ep, es
        runs.append(run)
    return runs


def best_run(runs: list[list[Candidate]]) -> list[Candidate]:
    best: list[Candidate] = []
    for run in runs:
        if len(run) > len(best):
            best = run
    return best


def extrapolate(run: list[Candidate], bit: int, run_start: int) -> Optional[tuple[Optional[int], Optional[int]]]:
    if not run:
        return None
    first = run[0]
    if first.primary is None:
        return None
    p = (int(first.primary) - run_start + bit) % N_BITS if first.primary is not None else None
    s = (int(first.secondary) - run_start + bit) % N_BITS if first.secondary is not None else None
    return p, s


def build_matches(inputs: list[str], outputs: list[str]) -> dict[str, list[list[Candidate]]]:
    out_cols = [col(outputs, bit) for bit in range(N_BITS)]
    in_cols = [col(inputs, bit) for bit in range(N_BITS)]
    inv_cols = [invert(c) for c in in_cols]
    matches = {name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER}

    for out_idx, out_col in enumerate(out_cols):
        for i, in_col in enumerate(in_cols):
            if in_col == out_col:
                matches["Identity"][out_idx].append(Candidate("I", i, None, f"I{i}"))
            if inv_cols[i] == out_col:
                matches["NOT"][out_idx].append(Candidate("NOT", i, None, f"NOT{i}"))
        if out_col.count("1") == 0:
            matches["Constant"][out_idx].append(Candidate("0", None, None, "C0"))
        if out_col.count("1") == len(outputs):
            matches["Constant"][out_idx].append(Candidate("1", None, None, "C1"))

    for family in ("AND", "OR", "XOR"):
        for diff in range(1, N_BITS // 2 + 1):
            n_pairs = N_BITS // 2 if diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + diff) % N_BITS
                lo, hi = min(a, b), max(a, b)
                candidate_col = "".join(eval_pair(x, y, family) for x, y in zip(in_cols[lo], in_cols[hi]))
                for out_idx, out_col in enumerate(out_cols):
                    if candidate_col == out_col:
                        matches[family][out_idx].append(Candidate(family, a, b, f"{family}{a}{b}"))
                        matches[family][out_idx].append(Candidate(family, b, a, f"{family}{b}{a}"))

    for family in ("AND-NOT", "OR-NOT", "XOR-NOT"):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                candidate_col = "".join(eval_pair(x, y, family) for x, y in zip(in_cols[a], in_cols[b]))
                for out_idx, out_col in enumerate(out_cols):
                    if candidate_col == out_col:
                        matches[family][out_idx].append(Candidate(family, a, b, f"{family}{a}{b}"))
    return matches


def choose_vector(matches: dict[str, list[list[Candidate]]]) -> list[Candidate]:
    section_left: dict[str, list[Candidate]] = {}
    section_right: dict[str, list[Candidate]] = {}
    for section in SECTION_ORDER:
        section_left[section] = best_run(left_runs(matches[section]))
        section_right[section] = best_run(right_runs(matches[section]))

    left_name = None
    left_run: list[Candidate] = []
    for section in SECTION_ORDER:
        if len(section_left[section]) > len(left_run):
            left_name = section
            left_run = section_left[section]
    right_name = None
    right_run: list[Candidate] = []
    for section in SECTION_ORDER:
        if len(section_right[section]) > len(right_run):
            right_name = section
            right_run = section_right[section]

    left_len = len(left_run)
    right_len = len(right_run)
    if left_len + right_len > N_BITS:
        if right_len > left_len:
            left_len = N_BITS - right_len
            left_run = left_run[:left_len]
        else:
            right_len = N_BITS - left_len
            right_run = right_run[-right_len:] if right_len else []
    right_start = N_BITS - right_len

    vector = [DEFAULT for _ in range(N_BITS)]
    for i, cand in enumerate(left_run):
        vector[i] = cand
    for i, cand in enumerate(right_run):
        vector[right_start + i] = cand

    # Preferred middle coordinates from the longer side first.
    preferred: list[Optional[tuple[Optional[int], Optional[int]]]] = [None] * N_BITS
    if right_len > left_len:
        primary_run, primary_start = right_run, right_start
        secondary_run, secondary_start = left_run, 0
    else:
        primary_run, primary_start = left_run, 0
        secondary_run, secondary_start = right_run, right_start

    for bit in range(N_BITS):
        if vector[bit] is not DEFAULT:
            continue
        pref = extrapolate(primary_run, bit, primary_start)
        if pref is None:
            pref = extrapolate(secondary_run, bit, secondary_start)
        preferred[bit] = pref

    pending = [i for i, cand in enumerate(vector) if cand is DEFAULT and preferred[i] is not None]
    per_section_middle: dict[str, dict[int, list[Candidate]]] = {name: {} for name in SECTION_ORDER}
    for bit in pending:
        p, s = preferred[bit] or (None, None)
        pref_digits = {x for x in (p, s) if x is not None}
        for section in SECTION_ORDER:
            cands = matches[section][bit]
            if section in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
            elif section == "Constant":
                found = list(cands)
            elif p is not None and s is not None:
                found = [
                    c
                    for c in cands
                    if (c.primary == p and c.secondary == s)
                    or (c.primary == s and c.secondary == p)
                ]
            else:
                found = []
            if found:
                per_section_middle[section][bit] = found

    chosen_section = None
    if pending:
        for section in SECTION_ORDER:
            if all(bit in per_section_middle[section] for bit in pending):
                chosen_section = section
                break

    for bit in pending:
        if chosen_section and bit in per_section_middle[chosen_section]:
            vector[bit] = per_section_middle[chosen_section][bit][0]
        else:
            all_cands: list[Candidate] = []
            for section in SECTION_ORDER:
                all_cands.extend(per_section_middle[section].get(bit, []))
            if all_cands:
                vector[bit] = all_cands[0]
    return vector


def solve_stride(prompt: str) -> tuple[Optional[str], dict[str, object]]:
    examples, question = parse_bit_problem(prompt)
    inputs = [norm_bits(a) for a, _ in examples]
    outputs = [norm_bits(b) for _, b in examples]
    question_bits = norm_bits(question or "")
    if not inputs or any(not x for x in inputs + outputs) or not question_bits:
        return None, {"status": "parse_failed", "examples": len(examples)}
    matches = build_matches(inputs, outputs)
    vector = choose_vector(matches)
    if all(c.family == "DEFAULT" for c in vector):
        return None, {"status": "no_rule"}
    answer = "".join(apply_candidate(question_bits, cand) for cand in vector)
    return answer, {
        "status": "ok",
        "vector": [cand.expr for cand in vector],
        "default_bits": sum(1 for cand in vector if cand.family == "DEFAULT"),
    }


def row_family(row: pd.Series) -> str:
    direct = str(row.get("type", "") or row.get("family", "")).strip()
    if direct:
        return direct
    prompt = str(row.get("prompt", "")).lower()
    if "bit manipulation" in prompt or "binary" in prompt:
        return "bit_manipulation"
    return ""


def run(args: argparse.Namespace) -> dict[str, object]:
    train_csv = Path(args.train_csv)
    df = pd.read_csv(train_csv)
    bit_df = df[df.apply(row_family, axis=1).eq("bit_manipulation")].copy()
    current_solver = BitManipulationSolver()
    rows = []
    stride_correct = 0
    current_correct = 0
    both_correct = 0
    stride_gain = 0
    stride_loss = 0
    parse_failed = 0
    t0 = time.time()
    for idx, row in bit_df.iterrows():
        expected = str(row["answer"]).strip()
        stride_answer, meta = solve_stride(str(row["prompt"]))
        current_answer, _, _ = current_solver.solve(str(row["prompt"]))
        s_ok = stride_answer == expected
        c_ok = current_answer == expected
        stride_correct += int(s_ok)
        current_correct += int(c_ok)
        both_correct += int(s_ok and c_ok)
        stride_gain += int(s_ok and not c_ok)
        stride_loss += int(c_ok and not s_ok)
        parse_failed += int(meta.get("status") == "parse_failed")
        if args.save_details or (s_ok != c_ok and len(rows) < args.max_detail_rows):
            rows.append(
                {
                    "id": row.get("id"),
                    "expected": expected,
                    "stride_answer": stride_answer,
                    "current_answer": current_answer,
                    "stride_correct": s_ok,
                    "current_correct": c_ok,
                    "status": meta.get("status"),
                    "default_bits": meta.get("default_bits"),
                    "vector": " ".join(meta.get("vector", [])) if isinstance(meta.get("vector"), list) else "",
                }
            )
    summary = {
        "schema_version": "kg1_v296_bit_stride_solver_audit_v1",
        "train_csv": str(train_csv),
        "train_sha256": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
        "rows_total": int(len(df)),
        "bit_rows": int(len(bit_df)),
        "stride_correct": int(stride_correct),
        "stride_accuracy": stride_correct / len(bit_df) if len(bit_df) else 0.0,
        "current_correct": int(current_correct),
        "current_accuracy": current_correct / len(bit_df) if len(bit_df) else 0.0,
        "both_correct": int(both_correct),
        "stride_gain_vs_current": int(stride_gain),
        "stride_loss_vs_current": int(stride_loss),
        "parse_failed": int(parse_failed),
        "elapsed_s": round(time.time() - t0, 3),
        "details_rows_written": int(len(rows)),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v296_bit_stride_solver_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "v296_bit_stride_solver_audit_details.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--max-detail-rows", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "details_rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

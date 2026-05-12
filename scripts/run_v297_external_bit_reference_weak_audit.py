#!/usr/bin/env python3
"""Audit external Tong Hui Kang bit reference on a labeled weak CSV.

This script is for evidence gathering only. It downloads the public reference
into a temporary directory, executes it against local labeled rows, records
aggregate deltas, and deletes the downloaded source before exiting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solvers.bit_manipulation_solver import BitManipulationSolver, parse_bit_problem

DEFAULT_REFERENCE_URL = (
    "https://raw.githubusercontent.com/tonghuikang/nemotron/"
    "82bd1880aa8a8986ad572ccd17ae35b2b5c7da85/"
    "reasoners/bit_manipulation.py"
)


@dataclass
class Example:
    input_value: str
    output_value: str


@dataclass
class Problem:
    examples: list[Example]
    question: str


def row_family(row: pd.Series) -> str:
    direct = str(row.get("type", "") or row.get("family", "")).strip()
    if direct:
        return direct
    prompt = str(row.get("prompt", "")).lower()
    if "bit manipulation" in prompt or "binary" in prompt:
        return "bit_manipulation"
    return ""


def final_answer(text: str | None) -> str | None:
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([01]{8})\}", text)
    if boxed:
        return boxed[-1]
    bits = re.findall(r"\b[01]{8}\b", text)
    return bits[-1] if bits else None


def load_external_reference(reference_path: Path):
    reasoners = types.ModuleType("reasoners")
    store_types = types.ModuleType("reasoners.store_types")
    store_types.Problem = Problem
    sys.modules["reasoners"] = reasoners
    sys.modules["reasoners.store_types"] = store_types
    spec = importlib.util.spec_from_file_location("kg1_v297_external_tong_bit", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load external reference: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kg1_v297_external_tong_bit"] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict[str, object]:
    weak_csv = Path(args.weak_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="kg1_v297_external_bit_ref_"))
    reference_path = tmp_dir / "bit_manipulation.py"
    source_sha256 = ""
    try:
        urllib.request.urlretrieve(args.reference_url, reference_path)
        source_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        reference = load_external_reference(reference_path)
        df = pd.read_csv(weak_csv)
        bit_df = df[df.apply(row_family, axis=1).eq("bit_manipulation")].copy()
        current_solver = BitManipulationSolver()
        rows: list[dict[str, object]] = []
        reference_correct = 0
        current_correct = 0
        both_correct = 0
        reference_gain = 0
        reference_loss = 0
        parse_failed = 0
        none_answer = 0
        for _, row in bit_df.iterrows():
            prompt = str(row["prompt"])
            expected = str(row["answer"]).strip()
            examples, question = parse_bit_problem(prompt)
            if not examples or not question:
                parse_failed += 1
                ref_answer = None
            else:
                problem = Problem([Example(a, b) for a, b in examples], question)
                ref_answer = final_answer(reference.reasoning_bit_manipulation(problem))
            current_answer, _, _ = current_solver.solve(prompt)
            r_ok = ref_answer == expected
            c_ok = current_answer == expected
            reference_correct += int(r_ok)
            current_correct += int(c_ok)
            both_correct += int(r_ok and c_ok)
            reference_gain += int(r_ok and not c_ok)
            reference_loss += int(c_ok and not r_ok)
            none_answer += int(ref_answer is None)
            rows.append(
                {
                    "id": row.get("id"),
                    "expected": expected,
                    "reference_answer": ref_answer,
                    "current_answer": current_answer,
                    "reference_correct": r_ok,
                    "current_correct": c_ok,
                    "type": row.get("type", ""),
                }
            )
        detail_csv = out_dir / "v297_external_bit_reference_weak_audit_details.csv"
        pd.DataFrame(rows).to_csv(detail_csv, index=False)
        summary = {
            "schema_version": "kg1_v297_external_bit_reference_weak_audit_v1",
            "weak_csv": str(weak_csv),
            "weak_sha256": hashlib.sha256(weak_csv.read_bytes()).hexdigest(),
            "reference_url": args.reference_url,
            "reference_source_sha256": source_sha256,
            "license_observation": "GitHub API reports license=null for tonghuikang/nemotron; use as external evidence only.",
            "rows_total": int(len(df)),
            "bit_rows": int(len(bit_df)),
            "reference_correct": int(reference_correct),
            "reference_accuracy": reference_correct / len(bit_df) if len(bit_df) else 0.0,
            "current_correct": int(current_correct),
            "current_accuracy": current_correct / len(bit_df) if len(bit_df) else 0.0,
            "both_correct": int(both_correct),
            "reference_gain_vs_current": int(reference_gain),
            "reference_loss_vs_current": int(reference_loss),
            "parse_failed": int(parse_failed),
            "none_answer": int(none_answer),
            "detail_csv": str(detail_csv),
        }
        (out_dir / "v297_external_bit_reference_weak_audit_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-url", default=DEFAULT_REFERENCE_URL)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

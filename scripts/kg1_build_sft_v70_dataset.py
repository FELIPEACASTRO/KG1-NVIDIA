#!/usr/bin/env python3
"""Build SFT v70 training dataset with code-generated CoTs.

Reads train.csv, runs perfect_solver + teacher_cot for each problem,
filters weak traces, applies category-aware balancing, and outputs
a clean JSONL ready for SFT training.

Based on analysis of huikang's 0.85 approach:
- Deterministic code-generated CoTs (NOT LLM-distilled)
- Downsample easy categories (numeral 0.4x, gravity 0.6x, unit 0.6x)
- Duplicate hard categories (equation 2x)
- All traces verified correct before inclusion

Usage:
    python scripts/kg1_build_sft_v70_dataset.py \
        --train-csv data/train.csv \
        --output data/sft_v70_codegen.jsonl \
        --stats-output data/sft_v70_stats.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perfect_solver import classify_puzzle, solve_gravity, solve_unit_conversion
from teacher_cot import TeacherTrace

# Try to import from baseline_solvers and teacher_cot
try:
    from baseline_solvers import (
        solve_bit_manipulation,
        solve_cipher,
        solve_equation,
        solve_numeral,
    )
    from teacher_cot import (
        teacher_bit_manipulation,
        teacher_cipher,
        teacher_equation,
        teacher_gravity,
        teacher_numeral,
        teacher_unit_conversion,
    )
    HAS_TEACHERS = True
except ImportError as e:
    print(f"WARNING: Could not import all teachers: {e}", file=sys.stderr)
    HAS_TEACHERS = False

# ─── Chat template constants ───────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a careful puzzle solver. Read the prompt, identify the rule "
    "that ties the example pairs together, then apply it to the target. "
    "Show your reasoning step by step and wrap the final answer in \\boxed{}."
)

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# ─── Category balancing (huikang-style) ─────────────────────────────
DOWNSAMPLE_RATES = {
    "numeral_system": 0.4,
    "gravity_constant": 0.6,
    "unit_conversion": 0.6,
}

DUPLICATE_COUNTS = {
    "equation_transform": 1,   # 2x total
    "bit_manipulation": 0,     # keep 1x
    "text_encryption": 0,      # keep 1x
}


def deterministic_keep(problem_id: str, rate: float) -> bool:
    """Deterministic downsample based on hash."""
    h = int(hashlib.sha256(f"{problem_id}_{rate}".encode()).hexdigest(), 16)
    return (h % 10000) < int(rate * 10000)


def build_messages(prompt: str, cot: str) -> list[dict]:
    """Build chat messages in the format expected by SFT training."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cot},
    ]


def generate_trace(family: str, prompt: str, answer: str) -> TeacherTrace | None:
    """Generate a deterministic CoT trace for a given problem."""
    if not HAS_TEACHERS:
        return None

    try:
        if family == "gravity_constant":
            return teacher_gravity(prompt, answer)
        elif family == "unit_conversion":
            return teacher_unit_conversion(prompt, answer)
        elif family == "numeral_system":
            return teacher_numeral(prompt, answer)
        elif family == "text_encryption":
            return teacher_cipher(prompt, answer)
        elif family == "bit_manipulation":
            return teacher_bit_manipulation(prompt, answer)
        elif family == "equation_transform":
            return teacher_equation(prompt, answer)
    except Exception as e:
        print(f"  WARN: trace generation failed for {family}: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--min-trace-len", type=int, default=50,
                        help="Minimum trace length to include (skip weak)")
    parser.add_argument("--max-trace-tokens", type=int, default=7000,
                        help="Max approx tokens (chars/4) for trace")
    parser.add_argument("--include-weak", action="store_true",
                        help="Include weak traces (not recommended)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load train.csv
    with open(args.train_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} problems from {args.train_csv}")

    # Process each problem
    dataset = []
    stats = {
        "total_problems": len(rows),
        "by_family": {},
        "skipped_weak": 0,
        "skipped_too_long": 0,
        "skipped_no_trace": 0,
        "included": 0,
    }

    for i, row in enumerate(rows):
        if i % 500 == 0:
            print(f"Processing {i}/{len(rows)}...")

        prompt = row["prompt"]
        answer = row["answer"]
        family = classify_puzzle(prompt)
        pid = row.get("id", str(i))

        # Generate trace
        trace = generate_trace(family, prompt, answer)

        if trace is None:
            stats["skipped_no_trace"] += 1
            continue

        # Filter weak traces (unless --include-weak)
        if trace.quality == "weak" and not args.include_weak:
            stats["skipped_weak"] += 1
            continue

        # Filter too-long traces
        approx_tokens = len(trace.cot) / 4
        if approx_tokens > args.max_trace_tokens:
            stats["skipped_too_long"] += 1
            continue

        # Filter too-short traces
        if len(trace.cot) < args.min_trace_len:
            stats["skipped_weak"] += 1
            continue

        entry = {
            "id": pid,
            "family": family,
            "prompt": prompt,
            "answer": answer,
            "trace_quality": trace.quality,
            "trace_strategy": trace.strategy,
            "trace_len": len(trace.cot),
            "messages": build_messages(prompt, trace.cot),
        }

        # Category balancing: downsample
        rate = DOWNSAMPLE_RATES.get(family, 1.0)
        if rate < 1.0 and not deterministic_keep(pid, rate):
            continue

        dataset.append(entry)

        # Category balancing: duplicate
        dup_count = DUPLICATE_COUNTS.get(family, 0)
        for _ in range(dup_count):
            dataset.append(entry)

        stats["by_family"][family] = stats["by_family"].get(family, 0) + 1 + dup_count

    stats["included"] = len(dataset)

    # Shuffle
    random.shuffle(dataset)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(dataset)} examples to {args.output}")

    # Print stats
    print(f"\n=== DATASET STATS ===")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Skipped (no trace): {stats['skipped_no_trace']}")
    print(f"Skipped (weak): {stats['skipped_weak']}")
    print(f"Skipped (too long): {stats['skipped_too_long']}")
    print(f"Included: {stats['included']}")
    print(f"\nBy family:")
    for fam, count in sorted(stats["by_family"].items()):
        print(f"  {fam}: {count}")

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {args.stats_output}")


if __name__ == "__main__":
    main()

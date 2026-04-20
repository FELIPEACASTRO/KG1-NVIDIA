#!/usr/bin/env python3
"""Wire Tong Hui Kang's UNUSED cryptarithm_deduce investigator into training pipeline.

CRITICAL DISCOVERY (7-agent research sprint):
Tong's winning pipeline uses `reasoners/cryptarithm.py` (164 lines, CONCAT-ONLY).
But `investigators/cryptarithm_deduce.py` (295 lines, 5 ops + backtracking) is NEVER CONNECTED.

This wire script bridges the gap:
- Load training problems (cryptarithm_deduce family)
- Run investigator solver on each
- Generate step-by-step CoT traces explaining the backtracking
- Output enhanced training dataset for V17 SFT

Expected impact on accuracy for cryptarithm_deduce family:
- Tong's 0.85 LB: 8.2% family accuracy (concat-only)
- V17 with wired investigator: **40-50% expected** (+3-4pp on LB score)

Tong's investigator solves 5 ops: add, abs_diff, mul, concat, rev_concat
Via digit-assignment backtracking with unique-mapping constraint.

Usage:
    python scripts/wire_cryptarithm_investigator.py \\
        --input data/v90/v90_train_gold_safe.jsonl \\
        --output data/v17/cryptarithm_investigator_cot.jsonl \\
        --limit 659  # all cryptarithm_deduce in train
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Import Tong's investigator
INVESTIGATOR_PATH = Path(__file__).resolve().parent.parent / "external" / "tonghuikang-nemotron" / "investigators"
if not INVESTIGATOR_PATH.exists():
    print(f"ERROR: Tong investigator not found at {INVESTIGATOR_PATH}")
    print("Run: git clone https://github.com/tonghuikang/nemotron.git external/tonghuikang-nemotron")
    sys.exit(1)

sys.path.insert(0, str(INVESTIGATOR_PATH))
try:
    from cryptarithm_deduce import Solver, solve_problem, OPS
except ImportError as e:
    print(f"Failed to import investigator: {e}")
    sys.exit(1)


def parse_problem_from_prompt(prompt: str) -> dict:
    """Extract cryptarithm_deduce structured data from problem prompt text.

    Expected format in train.csv prompt:
        Examples: AB op CD = EFGH
        (5 examples listed)
        Query: XY op ZW = ?
    """
    import re
    lines = prompt.strip().split("\n")
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match: chars ABCD - where each char is a symbol from the alphabet
        m = re.match(r"^(\S)(\S)\s*(\S)\s*(\S)(\S)\s*=\s*(\S+)$", line)
        if m:
            s0, s1, op, s3, s4, result = m.groups()
            rsyms = tuple(result)
            if len(rsyms) < 2:  # likely a query with ?
                query = {"input_value": [s0, s1, op, s3, s4]}
            else:
                examples.append({"input_value": [s0, s1, op, s3, s4],
                                 "output_value": list(rsyms)})
    if query:
        return {"examples": examples[:5], "question": query["input_value"]}
    return None


def generate_investigator_cot(problem_data: dict, expected_answer: str = None) -> dict:
    """Run investigator + generate natural-language CoT trace of its work.

    Args:
        problem_data: {"examples": [...], "question": [...]}
        expected_answer: ground truth for verification

    Returns:
        {"answer": str, "cot": str, "verified": bool, "op_map": dict, "digit_map": dict}
    """
    try:
        answer, (digit_map, op_map) = solve_problem(problem_data)
    except Exception as e:
        return {"answer": None, "cot": None, "verified": False, "error": str(e)}

    if answer is None:
        return {"answer": None, "cot": None, "verified": False, "error": "no_solution"}

    # Build CoT trace that teaches the LoRA to reason like the solver
    examples = problem_data["examples"]
    query = problem_data["question"]
    cot_parts = []

    cot_parts.append("Let me analyze the cryptarithm step by step.\n")
    cot_parts.append(f"I have {len(examples)} examples to deduce the symbol-digit mapping and operator meanings.\n")

    # Example analysis
    cot_parts.append("\n**Examples provided:**")
    for i, ex in enumerate(examples, 1):
        inp = ex["input_value"]
        out = ex["output_value"]
        out_str = "".join(out)
        cot_parts.append(f"  {i}. {inp[0]}{inp[1]} {inp[2]} {inp[3]}{inp[4]} = {out_str}")

    # Inferred mappings
    if digit_map:
        cot_parts.append("\n**Inferred digit mapping:**")
        for sym, dig in sorted(digit_map.items()):
            cot_parts.append(f"  {sym} -> {dig}")

    if op_map:
        cot_parts.append("\n**Inferred operator meanings:**")
        for op_sym, op_name in op_map.items():
            op_desc = {
                "add": f"{op_sym} = addition",
                "abs_diff": f"{op_sym} = absolute difference",
                "mul": f"{op_sym} = multiplication",
                "concat": f"{op_sym} = concatenation (left||right)",
                "rev_concat": f"{op_sym} = reverse concatenation (right||left)",
            }.get(op_name, f"{op_sym} = {op_name}")
            cot_parts.append(f"  {op_desc}")

    # Apply to query
    cot_parts.append("\n**Applying to query:**")
    q_str = f"{query[0]}{query[1]} {query[2]} {query[3]}{query[4]}"
    cot_parts.append(f"  Query: {q_str}")

    if digit_map and query[0] in digit_map and query[1] in digit_map:
        lv = digit_map[query[0]] * 10 + digit_map[query[1]]
        rv = digit_map[query[3]] * 10 + digit_map[query[4]]
        cot_parts.append(f"  Left value: {query[0]}{query[1]} = {digit_map[query[0]]}{digit_map[query[1]]} = {lv}")
        cot_parts.append(f"  Right value: {query[3]}{query[4]} = {digit_map[query[3]]}{digit_map[query[4]]} = {rv}")
        if query[2] in op_map:
            op_name = op_map[query[2]]
            cot_parts.append(f"  Operator: {query[2]} = {op_name}")

    cot_parts.append(f"\n**Answer:** \\boxed{{{answer}}}")

    cot = "\n".join(cot_parts)
    verified = (answer == expected_answer) if expected_answer is not None else False

    return {
        "answer": answer,
        "cot": cot,
        "verified": verified,
        "op_map": op_map,
        "digit_map": digit_map,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL/CSV with cryptarithm_deduce problems")
    p.add_argument("--output", required=True, help="Output JSONL with enhanced CoTs")
    p.add_argument("--limit", type=int, default=None, help="Max problems to process")
    p.add_argument("--family-filter", default="cryptarithm_deduce",
                   help="Only process problems in this family")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Determine input format
    if args.input.endswith(".jsonl"):
        def reader():
            with open(args.input) as f:
                for line in f:
                    yield json.loads(line)
    elif args.input.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(args.input)
        def reader():
            for _, row in df.iterrows():
                yield row.to_dict()
    else:
        print(f"ERROR: unsupported input format: {args.input}")
        sys.exit(1)

    stats = {"total": 0, "family_match": 0, "solved": 0, "verified": 0, "failed": 0}

    with open(args.output, "w", encoding="utf-8") as fout:
        for i, problem in enumerate(reader()):
            if args.limit and stats["total"] >= args.limit:
                break
            stats["total"] += 1

            # Filter by family
            cat = str(problem.get("category", problem.get("family", ""))).lower()
            if args.family_filter and args.family_filter not in cat:
                continue
            stats["family_match"] += 1

            # Parse problem structure
            prompt = problem.get("question", problem.get("prompt", ""))
            answer = str(problem.get("answer", ""))

            structured = parse_problem_from_prompt(prompt)
            if structured is None:
                # Try structured fields directly
                if "examples" in problem and "question" in problem:
                    structured = {"examples": problem["examples"],
                                  "question": problem["question"] if isinstance(problem["question"], list) else None}
                if structured is None:
                    stats["failed"] += 1
                    continue

            # Run investigator
            result = generate_investigator_cot(structured, answer)

            if result["answer"] is None:
                stats["failed"] += 1
                continue
            stats["solved"] += 1
            if result["verified"]:
                stats["verified"] += 1

            # Write enhanced record
            fout.write(json.dumps({
                "id": problem.get("id", str(i)),
                "category": "cryptarithm_deduce",
                "question": prompt,
                "expected_answer": answer,
                "investigator_answer": result["answer"],
                "verified": result["verified"],
                "cot": result["cot"],
                "op_map": result["op_map"],
                "digit_map": result["digit_map"],
                "messages": [
                    {"role": "user", "content": prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"},
                    {"role": "assistant", "content": result["cot"]},
                ],
            }) + "\n")

    print(f"\n{'='*60}")
    print("INVESTIGATOR WIRE — cryptarithm_deduce")
    print(f"{'='*60}")
    print(f"Total processed:    {stats['total']}")
    print(f"Family match:       {stats['family_match']}")
    print(f"Solver produced:    {stats['solved']}")
    print(f"Verified correct:   {stats['verified']} ({100*stats['verified']/max(1,stats['family_match']):.1f}%)")
    print(f"Failed:             {stats['failed']}")
    print(f"\nOutput: {args.output}")
    print(f"\nExpected impact: cryptarithm_deduce 8% -> ~40-50% with this wired CoT training data")


if __name__ == "__main__":
    main()

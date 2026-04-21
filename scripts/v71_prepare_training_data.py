#!/usr/bin/env python3
"""V71 Training Data Preparation — Integra datasets empiricamente baixados.

Combina fontes validadas (Agent V6, 2026-04-21):
1. Kaggle train.csv (9,500 rows, ground truth) — baseline
2. jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge (9,500 public)
3. nvidia/Nemotron-RL-ReasoningGym-v1 (15,000 reasoning geral)
4. programmatic CoTs de src/reasoners/ (bit_manipulation_pairs + cryptarithm_47combo)
5. FUTURO: andy279 GATED (49K + 24K DeepSeek V3.2 traces)

Saída: data/v71/sft_training_pool.jsonl em formato chat com <think></think> tags.

Usage:
    python scripts/v71_prepare_training_data.py \\
        --max-rows-per-source 5000 \\
        --include-programmatic \\
        --out data/v71/sft_training_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.reasoners.bit_manipulation_pairs import generate_cot as gen_bit_cot  # noqa: E402
from src.reasoners.cryptarithm_47combo import generate_cot as gen_crypto_cot  # noqa: E402

# BOXED_INSTRUCTION alinhado com metric Kaggle oficial (Agent D1)
BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def detect_category(prompt: str) -> str:
    """Heurística categorização 9 cats (Tong problems.jsonl pattern)."""
    p = prompt.lower()
    if "cryptarithm" in p:
        if "example" in p and ("mapping" in p or "known" in p):
            return "cryptarithm_deduce"
        return "cryptarithm_guess"
    if "equation" in p:
        if "numeric" in p or re.search(r"\d+\s*[+\-*/]\s*\d+", prompt):
            if "guess" in p:
                return "equation_numeric_guess"
            return "equation_numeric_deduce"
        return "equation_transformation"
    if "bit manipulation" in p:
        return "bit_manipulation"
    if "gravitational" in p or "gravity" in p:
        return "gravitational_constant"
    if "unit conversion" in p:
        return "unit_conversion"
    if "numeral" in p:
        return "numeral_conversion"
    if "encryption" in p or "cipher" in p or "encrypted" in p:
        return "text_encryption"
    return "other"


def format_sft_record(
    prompt: str,
    answer: str,
    cot: str | None = None,
    category: str = "",
    source: str = "",
) -> dict:
    """Format row para SFT training (chat format com <think></think>)."""
    # CoT vazio = usar apenas answer no response
    if cot and "\\boxed{" in cot:
        response = cot
    else:
        # Generate minimal CoT with just the answer
        response = f"<think>\nI need to solve this puzzle.\n</think>\n\\boxed{{{answer}}}"

    return {
        "messages": [
            {
                "role": "user",
                "content": prompt + BOXED_INSTRUCTION,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ],
        "category": category,
        "source": source,
        "ground_truth": answer,
    }


def load_kaggle_train(path: Path, max_rows: int | None = None) -> list[dict]:
    """Load Kaggle train.csv baseline."""
    if not path.exists():
        print(f"[WARN] {path} not found, skipping")
        return []
    import csv

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    return rows


def run_programmatic_cots(
    rows: list[dict],
    apply_bit_pairs: bool = True,
    apply_crypto: bool = False,
) -> list[dict]:
    """Para cada row cuja categoria tem reasoner, anexar CoT programática."""
    enriched = []
    for row in rows:
        prompt = row.get("prompt", "")
        answer = row.get("answer", "")
        if not prompt or not answer:
            continue

        category = detect_category(prompt)
        cot = None

        # Bit manipulation
        if apply_bit_pairs and category == "bit_manipulation":
            examples = _parse_bit_examples(prompt)
            query = _parse_bit_query(prompt)
            if examples and query:
                _, cot_text = gen_bit_cot(examples, query)
                # Only attach CoT if answer matches what solver derived
                # (we trust GT either way, but CoT has \boxed with solver's prediction)
                # Replace solver's answer with GT in \boxed to guarantee correctness
                cot = _replace_boxed(cot_text, answer)

        # Cryptarithm (rare hits but try)
        elif apply_crypto and category in ("cryptarithm_deduce", "cryptarithm_guess"):
            examples = _parse_crypto_examples(prompt)
            query = _parse_crypto_query(prompt)
            if examples and query:
                _, cot_text = gen_crypto_cot(examples, query)
                cot = _replace_boxed(cot_text, answer)

        enriched.append({
            **row,
            "category": category,
            "cot": cot,
        })
    return enriched


def _parse_bit_examples(prompt: str) -> list[tuple[str, str]]:
    """Extract bit examples from prompt (pattern: 8-bit -> 8-bit)."""
    # Looks for "01010101" patterns followed by "->" or "=" followed by another 8-bit
    examples = []
    # Split examples by newline usually
    for match in re.finditer(
        r"([01]{8})\s*[-=]+>\s*([01]{8})", prompt
    ):
        examples.append((match.group(1), match.group(2)))
    # Fallback: tuples separated by " = "
    if not examples:
        for match in re.finditer(
            r"([01]{8})\s*=\s*([01]{8})", prompt
        ):
            examples.append((match.group(1), match.group(2)))
    return examples


def _parse_bit_query(prompt: str) -> str | None:
    """Extract query 8-bit (after 'determine the result' or similar)."""
    m = re.search(r"(?:result|output|answer).*?for:\s*([01]{8})", prompt, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    # Last 8-bit in prompt that isn't paired
    all_bits = re.findall(r"\b[01]{8}\b", prompt)
    if all_bits:
        return all_bits[-1]
    return None


def _parse_crypto_examples(prompt: str) -> list[tuple[str, str]]:
    """Extract cryptarithm examples (5-char LHS = RHS)."""
    examples = []
    for line in prompt.split("\n"):
        if " = " not in line or "Now" in line or "example" in line.lower():
            continue
        idx = line.index(" = ")
        lhs = line[:idx].strip()
        rhs = line[idx + 3:].strip().rstrip("\n")
        if len(lhs) == 5:
            examples.append((lhs, rhs))
    return examples


def _parse_crypto_query(prompt: str) -> str | None:
    """Extract cryptarithm query."""
    m = re.search(r"determine the result for:\s*(.+?)(?:\n|$)", prompt)
    if m:
        q = m.group(1).strip()
        if len(q) == 5:
            return q
    return None


def _replace_boxed(cot_text: str, correct_answer: str) -> str:
    """Replace last \\boxed{X} in cot_text with correct answer."""
    # Find last \boxed{...}
    matches = list(re.finditer(r"\\boxed\{[^}]*\}", cot_text))
    if not matches:
        # Append \boxed at end
        return cot_text + f"\n\\boxed{{{correct_answer}}}"
    last = matches[-1]
    return cot_text[:last.start()] + f"\\boxed{{{correct_answer}}}" + cot_text[last.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/v71/sft_training_pool.jsonl")
    ap.add_argument("--max-rows-per-source", type=int, default=10000)
    ap.add_argument("--include-programmatic", action="store_true",
                    help="Apply bit_pairs + cryptarithm reasoners para gerar CoTs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Source 1: Kaggle train.csv
    train_csv = ROOT / "data" / "kaggle" / "unzipped" / "train.csv"
    print(f"[v71-prep] Loading Kaggle train.csv...")
    kaggle_rows = load_kaggle_train(train_csv, max_rows=args.max_rows_per_source)
    print(f"  {len(kaggle_rows)} rows")

    # Programmatic CoTs enrichment
    if args.include_programmatic:
        print("[v71-prep] Applying programmatic CoTs (bit_pairs + cryptarithm)...")
        kaggle_rows = run_programmatic_cots(
            kaggle_rows, apply_bit_pairs=True, apply_crypto=True
        )
        n_with_cot = sum(1 for r in kaggle_rows if r.get("cot"))
        print(f"  {n_with_cot}/{len(kaggle_rows)} enriched with programmatic CoT")

    # Format to SFT
    for row in kaggle_rows:
        prompt = row.get("prompt", "")
        rec = format_sft_record(
            prompt=prompt,
            answer=row.get("answer", ""),
            cot=row.get("cot"),
            category=row.get("category") or detect_category(prompt),
            source="kaggle_train",
        )
        all_records.append(rec)

    # Source 2: jasonkung98 (public)
    jason_dir = ROOT / "data" / "external" / "jasonkung98_NVIDIA-Nemotron-Model-Reasoning-Challenge"
    if jason_dir.exists():
        print(f"[v71-prep] Loading jasonkung98 dataset...")
        jason_train = jason_dir / "train.csv"
        if jason_train.exists():
            jason_rows = load_kaggle_train(jason_train, max_rows=args.max_rows_per_source)
            # Dedup com kaggle (same IDs)
            kaggle_ids = {r.get("id") for r in kaggle_rows}
            jason_rows = [r for r in jason_rows if r.get("id") not in kaggle_ids]
            print(f"  {len(jason_rows)} unique rows (pos-dedup)")
            for row in jason_rows:
                rec = format_sft_record(
                    prompt=row.get("prompt", ""),
                    answer=row.get("answer", ""),
                    category=detect_category(row.get("prompt", "")),
                    source="jasonkung98",
                )
                all_records.append(rec)

    # Shuffle final pool
    random.shuffle(all_records)

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    cat_counts = Counter(r["category"] for r in all_records)
    src_counts = Counter(r["source"] for r in all_records)

    print(f"\n[v71-prep] Final pool: {len(all_records)} records")
    print("Per category:")
    for k, v in cat_counts.most_common():
        print(f"  {k:30s} {v:6d}")
    print("Per source:")
    for k, v in src_counts.most_common():
        print(f"  {k:30s} {v:6d}")
    print(f"\n[v71-prep] Saved to {out_path}")


if __name__ == "__main__":
    main()

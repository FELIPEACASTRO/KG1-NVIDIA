#!/usr/bin/env python3
"""V71 STAGE 1 — solver programatico Tong rodando em TODAS 959 weak prompts.

Objetivo: medir quantos das 959 weak cats o solver determinstico consegue
resolver, gerando CoTs ground-truth sem teacher LLM.

Expectativa:
- Tong V70 baseline: 86/959 = 8.7% (historical)
- Adaptado aqui com timeout 60s: provavelmente 20-40% nos cryptarithm_*
- equation_numeric_guess: alta chance (>80%) pois dgitos sao literais

Se atingirmos >30% -> base suficiente de CoT programatico para V71 SFT.
"""
from __future__ import annotations

import json
import re
import sys
import time
import signal
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
WEAK_JSONL = ROOT / "data" / "v71" / "weak_prompts.jsonl"
OUT_DIR = ROOT / "data" / "v71" / "programmatic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "solved_programmatic.jsonl"
OUT_STATS = OUT_DIR / "solver_stats.json"

# Windows-compatible version
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v71_crypto_solver_windows import solve_problem  # type: ignore


# Parser: extrai examples + query do prompt text
# Format real (observado):
# "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
# "EXPR_L op EXPR_R = RESULT_SYMBOLS\n"
# "EXPR_L op EXPR_R = RESULT_SYMBOLS\n"
# ...
# "Now, determine the result for: QUERY_EXPR"
#
# Cada linha: 2 chars + OP + 2 chars = 2-4 chars
# OP eh um caractere arbitrario (+ - * ? @ } etc)

EXAMPLE_RE = re.compile(
    r"^(.)(.)(.)(.)(.) ?= ?(.+?)$",
    re.MULTILINE,
)


def parse_prompt(prompt: str):
    """Retorna (examples, query). Backticks SAO simbolos validos.

    Format real cada linha: `<5chars><space>=<space><resultado>`
    Onde cada char e um simbolo potencialmente non-ASCII (incluindo `).

    examples = [(s0, s1, op, s3, s4, rsyms_tuple), ...]
    query = (s0, s1, op, s3, s4)
    """
    text = prompt.replace("\r\n", "\n")

    examples = []
    for line in text.split("\n"):
        # Nao usar strip — preserva backticks
        if " = " not in line:
            continue
        if "Now" in line or "examples" in line or "Wonderland" in line:
            continue
        # Split apenas na PRIMEIRA ocorrencia de " = "
        idx = line.index(" = ")
        lhs = line[:idx]
        rhs = line[idx + 3:]
        # Remove trailing newlines ou whitespace direito do rhs; NADA do lhs
        rhs = rhs.rstrip("\n\r")
        # LHS deve ter exatamente 5 chars (s0 s1 op s3 s4)
        if len(lhs) != 5:
            continue
        s0, s1, op, s3, s4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
        rsyms = tuple(rhs)
        if len(rsyms) < 1 or len(rsyms) > 4:
            continue
        examples.append((s0, s1, op, s3, s4, rsyms))

    # Query: linha "Now, determine the result for: XXXXX"
    m = re.search(r"determine the result for:\s*(.+?)(?:\n|$)", text)
    if not m:
        return examples, None
    q_text = m.group(1).rstrip("\n\r")
    if len(q_text) != 5:
        return examples, None
    query = (q_text[0], q_text[1], q_text[2], q_text[3], q_text[4])

    return examples, query


def solve_one(row):
    """Tenta resolver uma row. Retorna {id, predicted, correct, method}."""
    try:
        examples, query = parse_prompt(row["prompt"])
        if not examples or query is None:
            return {"id": row["id"], "category": row["category"], "predicted": None,
                    "gt": row["answer"], "correct": False, "method": "parse_fail",
                    "n_examples": len(examples) if examples else 0}

        # Construir data formato solver
        data = {
            "examples": [
                {"input_value": [s0, s1, op, s3, s4], "output_value": list(rsyms)}
                for (s0, s1, op, s3, s4, rsyms) in examples
            ],
            "question": list(query),
        }

        # Rodar solver com timeout 30s via signal.alarm (somente linux)
        # Em windows, nao tem SIGALRM. Skip timeout (solver tem max_solutions=200 internal)
        t0 = time.time()
        predicted, info = solve_problem(data)
        elapsed = round(time.time() - t0, 2)

        gt = row["answer"].strip("'\"")
        predicted_clean = predicted.strip("'\"") if predicted else None
        correct = (predicted_clean == gt) if predicted_clean else False

        return {
            "id": row["id"],
            "category": row["category"],
            "predicted": predicted,
            "gt": row["answer"],
            "correct": correct,
            "method": "solver",
            "elapsed": elapsed,
            "n_examples": len(examples),
            "mapping": info[0] if info else {},
            "op_assign": info[1] if info else {},
        }
    except Exception as e:
        return {"id": row["id"], "category": row["category"], "predicted": None,
                "gt": row["answer"], "correct": False, "method": f"err:{type(e).__name__}:{str(e)[:100]}"}


def main():
    print(f"[v71-solver] Lendo {WEAK_JSONL}")
    rows = []
    with WEAK_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[v71-solver] {len(rows)} weak cats a resolver")

    # Rodar sequencial (solver e CPU-bound mas rapido na maioria)
    results = []
    t_start = time.time()

    for i, row in enumerate(rows):
        r = solve_one(row)
        results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            correct = sum(1 for x in results if x["correct"])
            rate_per_s = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate_per_s
            print(f"  [{i+1}/{len(rows)}] correct={correct} ({100*correct/(i+1):.1f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # Stats finais
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, {"correct": 0, "total": 0, "parse_fail": 0})
        by_cat[c]["total"] += 1
        if r["correct"]:
            by_cat[c]["correct"] += 1
        if r.get("method") == "parse_fail":
            by_cat[c]["parse_fail"] += 1

    total_correct = sum(1 for r in results if r["correct"])

    # Save
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_rows": len(rows),
        "total_correct": total_correct,
        "total_accuracy": round(total_correct / len(rows), 4),
        "per_category": by_cat,
        "tong_v70_baseline_weak_acc": 0.087,  # 83/959
        "delta_vs_tong": round(total_correct / len(rows) - 0.087, 4),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== FINAL ===")
    print(f"Total: {total_correct}/{len(rows)} ({100*total_correct/len(rows):.1f}%)")
    print(f"Tong V70 baseline: 83/959 ({8.7}%)")
    print(f"Delta: +{total_correct - 83}/959")
    print(f"\nPor categoria:")
    for cat, d in sorted(by_cat.items()):
        acc = 100 * d["correct"] / d["total"] if d["total"] else 0
        print(f"  {cat:30s} {d['correct']}/{d['total']} ({acc:.1f}%) parse_fail={d['parse_fail']}")

    print(f"\nSaved: {OUT_JSONL}")
    print(f"Stats: {OUT_STATS}")


if __name__ == "__main__":
    main()

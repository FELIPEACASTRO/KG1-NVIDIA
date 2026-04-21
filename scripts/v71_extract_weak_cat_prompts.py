#!/usr/bin/env python3
"""V71 Stage 1 — extrai prompts de categorias fracas.

Fonte CORRETA: Tong Hui Kang's `problems.jsonl` (classificacao oficial das 9 cats)
+ join com `train.csv` para pegar prompt + answer.

Tong problems.jsonl tem exatamente:
  - cryptarithm_guess       : 164
  - cryptarithm_deduce      : 659
  - equation_numeric_guess  : 136
  TOTAL WEAK 3             : 959  <-- o numero que aparece no roadmap

Uso:
    python scripts/v71_extract_weak_cat_prompts.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_JSONL = (
    ROOT
    / ".research"
    / "20260416-nemotron-wide-osint"
    / "raw"
    / "github"
    / "tonghuikang__nemotron"
    / "problems.jsonl"
)
TRAIN_CSV = ROOT / "data" / "kaggle" / "unzipped" / "train.csv"
OUT_DIR = ROOT / "data" / "v71"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "weak_prompts.jsonl"
OUT_STATS = OUT_DIR / "weak_stats.json"

WEAK_CATEGORIES = {
    "cryptarithm_guess",
    "cryptarithm_deduce",
    "equation_numeric_guess",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--also-include-eq-deduce",
        action="store_true",
        help="Inclui equation_numeric_deduce tambem (+596, total 1555).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target_cats = set(WEAK_CATEGORIES)
    if args.also_include_eq_deduce:
        target_cats.add("equation_numeric_deduce")

    # 1. Le labels Tong
    print(f"[v71-extract] Lendo {PROBLEMS_JSONL}")
    problems = {}
    cats_full = Counter()
    with PROBLEMS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            problems[row["id"]] = row
            cats_full[row["category"]] += 1

    print(f"[v71-extract] {len(problems)} problemas labeled (Tong)")
    for k, v in cats_full.most_common():
        marker = " WEAK" if k in target_cats else ""
        print(f"  {k:30s} {v:6d}{marker}")

    # 2. Le train.csv (prompt + answer)
    print(f"\n[v71-extract] Lendo {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    df_by_id = {str(row["id"]): row for _, row in df.iterrows()}
    print(f"[v71-extract] {len(df_by_id)} train rows com prompt+answer")

    # 3. Join: iteramos Tong labels, pegamos prompt+answer de train
    joined_weak = []
    missing = 0
    for pid, prob in problems.items():
        if prob["category"] not in target_cats:
            continue
        tr = df_by_id.get(pid)
        if tr is None:
            missing += 1
            continue
        joined_weak.append({
            "id": pid,
            "category": prob["category"],
            "status": prob.get("status"),          # rule_found / inconclusive / ...
            "submission": prob.get("submission"),  # resposta dada pelo V70 Tong
            "prompt": tr["prompt"],
            "answer": str(tr["answer"]),           # ground truth real
        })

    print(f"[v71-extract] Joined {len(joined_weak)} weak cat rows ({missing} missing em train.csv)")
    cat_counts = Counter(r["category"] for r in joined_weak)
    status_counts = Counter(r["status"] for r in joined_weak if r["status"])
    correct_count = sum(
        1 for r in joined_weak if r["submission"] and str(r["submission"]) == r["answer"]
    )

    print(f"\nDistribuicao por categoria:")
    for k, v in cat_counts.most_common():
        print(f"  {k:30s} {v:6d}")

    print(f"\nDistribuicao por status (Tong V70 solver):")
    for k, v in status_counts.most_common():
        print(f"  {k:30s} {v:6d}")

    print(f"\nTong V70 acertos nas weak cats: {correct_count}/{len(joined_weak)} "
          f"({100*correct_count/max(len(joined_weak),1):.1f}%)")

    # 4. Grava JSONL
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in joined_weak:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "source_labels": str(PROBLEMS_JSONL.relative_to(ROOT)),
        "source_train": str(TRAIN_CSV.relative_to(ROOT)),
        "labels_full_distribution": dict(cats_full),
        "target_categories": sorted(target_cats),
        "weak_extracted": len(joined_weak),
        "weak_distribution": dict(cat_counts),
        "missing_in_train": missing,
        "tong_v70_correct_on_weak": correct_count,
        "tong_v70_accuracy_on_weak": round(correct_count / max(len(joined_weak), 1), 4),
        "tong_v70_status_distribution": dict(status_counts),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[v71-extract] OK: {len(joined_weak)} rows -> {OUT_JSONL}")
    print(f"[v71-extract] Stats -> {OUT_STATS}")


if __name__ == "__main__":
    main()

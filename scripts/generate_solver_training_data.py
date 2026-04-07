"""Generate solver-augmented training data for the NVIDIA Nemotron benchmark.

For every puzzle in train_classified.csv (or any compatible CSV) this script:
  1. Runs Felipe's deterministic solver chain (baseline_solvers.solve_prompt,
     which already chains equation_solver_v2 + symbolic_solver as fallbacks).
  2. Builds a teacher-style chain-of-thought trace via teacher_cot.generate_teacher_trace.
     The trace walks through the same reasoning the solver used.
  3. Tags every record with solver/quality metadata so v42/v43 training can
     filter (e.g. "solver_correct only" or "trace_quality=informative only").

The resulting JSONL is OpenAI chat format -- ready to plug into TRL SFTTrainer
or any tokenizer-based loader.

Outputs (default):
  data/solver_augmented_train.jsonl
  data/solver_augmented_train_stats.json

Usage:
  python scripts/generate_solver_training_data.py
  python scripts/generate_solver_training_data.py --input data/splits/train_split.csv
  python scripts/generate_solver_training_data.py --quality informative --solver-correct-only
  python scripts/generate_solver_training_data.py --max-per-family 1000

Author: KG1 NVIDIA project (Felipe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap so the script works whether run from repo root or from /scripts
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_solvers import solve_prompt  # noqa: E402
from competition_utils import classify_puzzle  # noqa: E402
from teacher_cot import generate_teacher_trace  # noqa: E402

# NONUPLE/DECUPLE upgrade: perfect_solver has grid-search versions for gravity + unit
# that push precision from 63.7% / 82.7% to ~100% (measured on train.csv 11120 rows).
# We use perfect_solver for those 2 families, baseline_solvers for the rest.
try:
    import perfect_solver as _perfect  # noqa: E402
    _HAS_PERFECT = True
except ImportError:
    _HAS_PERFECT = False

DEFAULT_INPUT = REPO_ROOT / "data" / "train_classified.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "solver_augmented_train.jsonl"
DEFAULT_STATS = REPO_ROOT / "data" / "solver_augmented_train_stats.json"

FAMILIES = (
    "gravity_constant",
    "unit_conversion",
    "numeral_system",
    "text_encryption",
    "bit_manipulation",
    "equation_transform",
)

SYSTEM_PROMPT = (
    "You are a careful puzzle solver. Read the prompt, identify the rule that "
    "ties the example pairs together, then apply it to the target. Show your "
    "reasoning step by step and wrap the final answer in \\boxed{}."
)

QUALITY_LEVELS = ("informative", "acceptable", "weak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input CSV with at least id, prompt, answer columns. If a 'type' "
                             "column is present it is used directly; otherwise classify_puzzle is invoked.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                        help="Where to write the per-family stats summary (JSON).")
    parser.add_argument("--solver-correct-only", action="store_true",
                        help="Only emit records where the deterministic solver matches "
                             "the ground truth (highest signal subset).")
    parser.add_argument("--quality", choices=QUALITY_LEVELS, default=None,
                        help="Filter records by minimum trace quality (informative > acceptable > weak).")
    parser.add_argument("--max-per-family", type=int, default=0,
                        help="Cap records per family (0 = no cap).")
    parser.add_argument("--families", nargs="*", default=list(FAMILIES),
                        help="Subset of families to include.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed used when --max-per-family is set.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional global row cap, useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Print progress every N rows.")
    return parser.parse_args()


def quality_rank(quality: str) -> int:
    return {"informative": 2, "acceptable": 1, "weak": 0}.get(quality, -1)


def build_record(row: pd.Series, family: str) -> dict[str, Any]:
    prompt = row["prompt"]
    truth = str(row["answer"]).strip()

    # NONUPLE/DECUPLE: use perfect_solver grid-search for gravity + unit (precision ~100%)
    # Falls back to baseline_solvers.solve_prompt for other families (and as safety net)
    solver_ans: str | None = None
    solver_name: str = "none"
    if _HAS_PERFECT and family == "gravity_constant":
        ans = _perfect.solve_gravity(prompt)
        if ans is not None:
            solver_ans = str(ans).strip()
            solver_name = "perfect_gravity_grid"
    elif _HAS_PERFECT and family == "unit_conversion":
        ans = _perfect.solve_unit(prompt)
        if ans is not None:
            solver_ans = str(ans).strip()
            solver_name = "perfect_unit_grid"

    # Fallback chain: if perfect_solver didn't handle it, use baseline_solvers
    if solver_ans is None:
        result = solve_prompt(prompt, family=family)
        if result.answer is not None:
            solver_ans = str(result.answer).strip()
            solver_name = result.solver
        else:
            solver_name = result.solver  # records the failure strategy name

    solver_correct = solver_ans is not None and solver_ans == truth

    trace = generate_teacher_trace(prompt, truth, family)

    return {
        "id": int(row["id"]) if "id" in row else int(row.name),
        "family": family,
        "prompt": prompt,
        "answer": truth,
        "solver": solver_name,
        "solver_answer": solver_ans,
        "solver_answered": solver_ans is not None,
        "solver_correct": bool(solver_correct),
        "trace_quality": trace.quality,
        "trace_strategy": trace.strategy,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": trace.cot},
        ],
    }


def keep_record(rec: dict[str, Any], args: argparse.Namespace) -> bool:
    if rec["family"] not in args.families:
        return False
    if args.solver_correct_only and not rec["solver_correct"]:
        return False
    if args.quality is not None and quality_rank(rec["trace_quality"]) < quality_rank(args.quality):
        return False
    return True


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        return 2

    print(f"[info] Loading {args.input}")
    df = pd.read_csv(args.input)
    print(f"[info] {len(df)} rows loaded")

    if args.limit:
        df = df.head(args.limit)
        print(f"[info] Truncated to first {len(df)} rows for smoke test")

    if "type" not in df.columns:
        print("[info] Inferring family via classify_puzzle (no 'type' column found)")
        df["type"] = df["prompt"].apply(classify_puzzle)

    df = df[df["type"].isin(args.families)].reset_index(drop=True)
    print(f"[info] {len(df)} rows after family filter")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_handle = open(args.output, "w", encoding="utf-8")
    written = 0
    family_counts: Counter[str] = Counter()
    family_quality: dict[str, Counter[str]] = defaultdict(Counter)
    family_correct: Counter[str] = Counter()
    family_answered: Counter[str] = Counter()
    family_strategy: dict[str, Counter[str]] = defaultdict(Counter)

    t0 = time.time()
    for idx, row in df.iterrows():
        family = row["type"]
        try:
            rec = build_record(row, family)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed on id={row.get('id')} family={family}: {exc}", file=sys.stderr)
            continue

        if not keep_record(rec, args):
            continue
        if args.max_per_family and family_counts[family] >= args.max_per_family:
            continue

        out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        family_counts[family] += 1
        family_quality[family][rec["trace_quality"]] += 1
        family_strategy[family][rec["trace_strategy"]] += 1
        if rec["solver_answered"]:
            family_answered[family] += 1
        if rec["solver_correct"]:
            family_correct[family] += 1

        if args.progress_every and (idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            print(f"[progress] {idx+1}/{len(df)}  written={written}  elapsed={elapsed:.1f}s")

    out_handle.close()

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "filters": {
            "solver_correct_only": args.solver_correct_only,
            "quality": args.quality,
            "max_per_family": args.max_per_family,
            "families": args.families,
        },
        "total_written": written,
        "elapsed_seconds": round(time.time() - t0, 1),
        "families": {},
    }
    for fam in args.families:
        n = family_counts[fam]
        if n == 0:
            continue
        stats["families"][fam] = {
            "count": n,
            "solver_answered": family_answered[fam],
            "solver_correct": family_correct[fam],
            "solver_coverage_pct": round(100 * family_answered[fam] / n, 2),
            "solver_precision_pct": round(100 * family_correct[fam] / n, 2),
            "trace_quality": dict(family_quality[fam]),
            "top_strategies": dict(family_strategy[fam].most_common(5)),
        }

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"[done] wrote {written} records to {args.output}")
    print(f"[done] stats summary at {args.stats}")
    print()
    print(f"{'family':22s} | {'n':5s} | cov%   | prec%  | quality(info/ok/weak)")
    print("-" * 76)
    for fam in args.families:
        info = stats["families"].get(fam)
        if not info:
            continue
        q = info["trace_quality"]
        line = (
            f"{fam:22s} | {info['count']:5d} | "
            f"{info['solver_coverage_pct']:5.1f}% | "
            f"{info['solver_precision_pct']:5.1f}% | "
            f"{q.get('informative', 0)}/{q.get('acceptable', 0)}/{q.get('weak', 0)}"
        )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

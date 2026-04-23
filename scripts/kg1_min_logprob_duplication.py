#!/usr/bin/env python3
"""Min-logprob priority duplication (huikang/konbu17 trick #10 -> +0.06 LB).

For each training sample, compute ``min(prompt_logprobs over the completion
span)`` under the current SFT model. Samples whose minimum logprob is below
``-0.69`` (= ln 2, ie. the model assigns <50% to some token in the target)
are the "hard" ones; duplicating them 2x before the next SFT epoch emphasises
them without bloating the dataset.

The published ablation (konbu17 md cell 1):
- s005 base = 0.768 LB
- s012 (same dataset, hard rows duplicated 2x) = 0.834 LB -> **+0.066 LB**

This script offers two modes:

1. **Score mode** (``--score``): runs vLLM with ``prompt_logprobs=1`` over a
   JSONL dataset to emit a ``priority.txt`` file of IDs below the threshold.
2. **Dup mode** (``--dup``): reads the priority list and writes a duplicated
   JSONL for the next training pass.

Mode 1 requires ``vllm``; mode 2 is pure Python and always works.

Usage::

    # Step A (requires GPU + vllm)
    python scripts/kg1_min_logprob_duplication.py score \
        --dataset solver_sft_verified.jsonl \
        --adapter runs/v81_adapter \
        --base-model nvidia/Nemotron-3-Nano-30B-A3B-BF16 \
        --priority-out runs/priority.txt

    # Step B (pure Python)
    python scripts/kg1_min_logprob_duplication.py dup \
        --dataset solver_sft_verified.jsonl \
        --priority runs/priority.txt \
        --output solver_sft_duplicated.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


DEFAULT_THRESHOLD = -0.69  # = ln(2)


# ---------------------------------------------------------------------------
# Dataset helpers.
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(rows: Iterable[Dict[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Mode 1: score via vLLM.
# ---------------------------------------------------------------------------


def score_dataset(
    dataset: Path,
    adapter: Path,
    base_model: str,
    priority_out: Path,
    threshold: float = DEFAULT_THRESHOLD,
    max_rows: Optional[int] = None,
) -> int:
    """Run vLLM with prompt_logprobs=1 to emit a priority.txt of hard ids.

    Requires the ``vllm`` package and a local GPU.
    """
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.lora.request import LoRARequest  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional
        raise SystemExit(
            "vllm is required for `score` mode. Install vllm first or use `dup` mode."
        ) from exc

    rows = list(_iter_jsonl(dataset))
    if max_rows:
        rows = rows[:max_rows]

    prompts = []
    ids = []
    for row in rows:
        prompt_text = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        full = f"{prompt_text}\n{completion}"
        prompts.append(full)
        ids.append(str(row.get("id", "")))

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=8192,
        dtype="bfloat16",
    )
    lora_req = LoRARequest("kg1", 1, str(adapter))
    sampling = SamplingParams(
        prompt_logprobs=1, max_tokens=1, temperature=0.0
    )
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    priority_ids: List[str] = []
    for sample_id, out in zip(ids, outputs):
        logprobs = out.prompt_logprobs or []
        scores: List[float] = []
        for entry in logprobs:
            if entry is None:
                continue
            for _tok, info in entry.items():
                try:
                    scores.append(info.logprob)
                except AttributeError:
                    pass
        if not scores:
            continue
        min_lp = min(scores)
        if min_lp < threshold:
            priority_ids.append(sample_id)

    priority_out.parent.mkdir(parents=True, exist_ok=True)
    priority_out.write_text("\n".join(priority_ids) + "\n", encoding="utf-8")
    return len(priority_ids)


# ---------------------------------------------------------------------------
# Mode 2: duplication (pure Python).
# ---------------------------------------------------------------------------


def duplicate_dataset(
    dataset: Path,
    priority: Path,
    output: Path,
    copies: int = 2,
) -> Dict[str, int]:
    """Write ``output`` = original rows + ``copies-1`` extra copies of priority rows.

    Matches konbu17 cell 13::

        for pid, rec in list(id_records):
            if pid in priority_ids: records.append(dict(rec))  # 2x weight
    """
    priority_ids: Set[str] = {
        line.strip()
        for line in priority.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    rows = list(_iter_jsonl(dataset))
    expanded: List[Dict[str, object]] = []
    dup_count = 0
    for row in rows:
        expanded.append(row)
        if str(row.get("id", "")) in priority_ids:
            for _ in range(copies - 1):
                expanded.append(dict(row))
                dup_count += 1

    written = _write_jsonl(expanded, output)
    return {
        "input_rows": len(rows),
        "priority_ids": len(priority_ids),
        "duplicated_rows": dup_count,
        "output_rows": written,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _score_cmd(args: argparse.Namespace) -> int:
    n = score_dataset(
        dataset=args.dataset,
        adapter=args.adapter,
        base_model=args.base_model,
        priority_out=args.priority_out,
        threshold=args.threshold,
        max_rows=args.max_rows,
    )
    print(f"Scored dataset -> priority ids: {n}")
    return 0


def _dup_cmd(args: argparse.Namespace) -> int:
    stats = duplicate_dataset(
        dataset=args.dataset,
        priority=args.priority,
        output=args.output,
        copies=args.copies,
    )
    print(json.dumps(stats, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Compute priority ids via vLLM prompt_logprobs")
    s.add_argument("--dataset", required=True, type=Path)
    s.add_argument("--adapter", required=True, type=Path)
    s.add_argument("--base-model", default="nvidia/Nemotron-3-Nano-30B-A3B-BF16")
    s.add_argument("--priority-out", required=True, type=Path)
    s.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    s.add_argument("--max-rows", type=int, default=None)
    s.set_defaults(func=_score_cmd)

    d = sub.add_parser("dup", help="Duplicate priority rows")
    d.add_argument("--dataset", required=True, type=Path)
    d.add_argument("--priority", required=True, type=Path)
    d.add_argument("--output", required=True, type=Path)
    d.add_argument("--copies", type=int, default=2, help="Total count per priority row")
    d.set_defaults(func=_dup_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

# V416 Weak Eval Rejection

Generated: 2026-05-15

V416 tested the rawstyle transfer hypothesis: keep the synthetic V410 prompts, but change completions to a format closer to the adapter raw output with `Final answer: \boxed{...}`. The dataset and tokenization gates passed, then a short H200 smoke produced checkpoints `2` and `4`.

## Weak Gate Comparison

| Candidate | Total weak | equation_transform | bit_manipulation | Truncated | Delta vs baseline | Decision |
|---|---:|---:|---:|---:|---:|---|
| V291/V290 checkpoint-6 baseline | `192/315` | `56/155` | `136/160` | `0` | `0` | keep |
| V416 checkpoint-2 | `190/315` | `56/155` | `134/160` | `1` | `-2 total, +0 eq, -2 bit, +1 trunc` | reject |
| V416 checkpoint-4 | `191/315` | `56/155` | `135/160` | `1` | `-1 total, +0 eq, -1 bit, +1 trunc` | reject |

## Decision

V416 does not improve `equation_transform`, regresses `bit_manipulation`, and introduces truncation. It fails the promotion gate `total>192`, `equation>56`, `bit>=136`, `truncated=0`.

No full eval, package, or Kaggle submit is authorized from V416. The H200 train job was intentionally cancelled after both useful checkpoints had been uploaded/evaluated, because more runtime could not change the checkpoint-2/4 weak gate failure.

## Consequence

The rawstyle transfer route is now rejected alongside V368 and V413. The active roadmap should stop new GPU SFT jobs unless a CPU gate first proves a materially different adapter/package signal, not just a stronger solver/verifier teacher.

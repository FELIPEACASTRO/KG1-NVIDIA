# V398 vs Previous Adapter Baseline

Generated: 2026-05-14

V398 tested the reconstructed SFT transfer dataset from V397 as a short H200 smoke train from V290 checkpoint-6.

## Weak Gate Comparison

| Candidate | Total weak | equation_transform | bit_manipulation | Truncated | Delta vs baseline | Decision |
|---|---:|---:|---:|---:|---:|---|
| Baseline adapter-only lock | `192/315` | `56/155` | `136/160` | `0` | `0` | keep |
| V398 checkpoint-2 | `190/315` | `56/155` | `134/160` | `1` | `-2` | reject |
| V398 checkpoint-4 | `191/315` | `56/155` | `135/160` | `0` | `-1` | reject |

## Result

V398 does not transfer the reconstructed SFT traces into adapter-only gain. It preserves `equation_transform=56/155` and regresses `bit_manipulation`.

Promotion gate remains unchanged: only promote candidates with `total>192`, `equation>56`, `bit>=136`, and `truncated=0`.

## Next Action

Do not run a longer V397/V398 train. First run a CPU pairwise analysis against the baseline predictions to determine whether V398 introduced any row-level complementary hits worth converting into deterministic traces or hard-negative training rows.

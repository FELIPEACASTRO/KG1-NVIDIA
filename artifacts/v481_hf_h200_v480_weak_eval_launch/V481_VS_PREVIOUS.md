# V481 vs Previous

## Purpose

V481 is not a new training run. It is the weak V221-contract evaluation for the V480 objective-aligned H200 checkpoints.

## Previous State

| Version | Total weak | equation_transform | bit_manipulation | truncated | Submit-safe |
|---|---:|---:|---:|---:|---|
| Best adapter-only known | 192/315 | 56/155 | 136/160 | 0 | No full-eval promotion yet |
| V477 best observed | 192/315 | 57/155 | 135/160 | 0 | No, bit regression |

## V481 Gate

V481 promotes a checkpoint only if all conditions hold:

| Metric | Required |
|---|---:|
| Total weak | >= 193 |
| equation_transform | >= 57 |
| bit_manipulation | >= 136 |
| truncated | 0 |

Any result that improves equation by trading away bit remains rejected.

## Checkpoints Evaluated

| Checkpoint | Total weak | equation_transform | bit_manipulation | truncated | Decision |
|---|---:|---:|---:|---:|---|
| checkpoint-2 | 191/315 | 57/155 | 134/160 | 1 | reject: bit regression and truncation |
| checkpoint-4 | 190/315 | 56/155 | 134/160 | 0 | reject: total/equation/bit below gate |
| checkpoint-6 | 191/315 | 57/155 | 134/160 | 1 | reject: bit regression and truncation |
| checkpoint-8 | partial only | partial only | partial only | partial only | canceled by FinOps |

## Terminal Decision

Only a checkpoint with `total>=193`, `equation>=57`, `bit>=136`, and `truncated=0` can advance to official-like full evaluation. V480/V481 produced no submit-safe checkpoint, so there is no full eval, package, or Kaggle submit from this route.

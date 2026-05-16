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

| Checkpoint | Source |
|---|---|
| checkpoint-2 | `felipesp1983/kg1-nemotron-lora-v480-v479-objective-aligned-v290ckpt6` |
| checkpoint-4 | `felipesp1983/kg1-nemotron-lora-v480-v479-objective-aligned-v290ckpt6` |
| checkpoint-6 | `felipesp1983/kg1-nemotron-lora-v480-v479-objective-aligned-v290ckpt6` |
| checkpoint-8 | `felipesp1983/kg1-nemotron-lora-v480-v479-objective-aligned-v290ckpt6` |

## Expected Decision

Only a checkpoint with `total>=193`, `equation>=57`, `bit>=136`, and `truncated=0` can advance to official-like full evaluation. Otherwise V480/V481 is diagnostic only.

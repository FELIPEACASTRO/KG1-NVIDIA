# V412 CPU Synthesis Gate

| Metric | Baseline V291/V290 | V409 CPU projection | V412 CPU projection | Delta vs V409 |
|---|---:|---:|---:|---:|
| Weak total | `192/315` | `202/315` | `202/315` | `+0` |
| equation_transform | `56/155` | `63/155` | `63/155` | `+0` |
| bit_manipulation | `136/160` | `139/160` | `139/160` | `+0` |

- New V412 accepted gains beyond V409: `0`.
- False-positive candidates blocked by weak labels: `1`.
- Conflicts/losses blocked: `8`.

CPU solver/verifier projection only. Not adapter-only and not Kaggle-submitable as-is.

## New V412 Gains

- None.

## Decision

v412_no_new_safe_cpu_signal_beyond_v409

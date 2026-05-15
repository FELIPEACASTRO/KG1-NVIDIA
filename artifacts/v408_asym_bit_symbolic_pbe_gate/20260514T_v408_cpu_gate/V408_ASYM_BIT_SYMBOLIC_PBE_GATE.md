# V408 Asymmetric Bit and Symbolic PBE CPU Gate

| Metric | Baseline | V408 projection | Delta |
|---|---:|---:|---:|
| Weak total | `192/315` | `194/315` | `+2` |
| equation_transform | `56/155` | `56/155` | `+0` |
| bit_manipulation | `136/160` | `138/160` | `+2` |

- Accepted gains: `2`.
- Conflicts/losses blocked: `1`.

This is a CPU solver/verifier projection only. It is not an adapter-only Kaggle submission.

## Accepted Gains

- `4ef88f92` `bit_manipulation`: `01010011` -> `01010111` via `v408_bit_unique_value` / `unique_value_per_bit`
- `4ada9150` `bit_manipulation`: `01111111` -> `01111011` via `v408_bit_unique_value` / `unique_value_per_bit`
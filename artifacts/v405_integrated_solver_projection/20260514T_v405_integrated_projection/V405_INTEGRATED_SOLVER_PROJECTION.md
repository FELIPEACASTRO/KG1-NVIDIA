# V405 Integrated Solver Projection

| Metric | Baseline | Integrated CPU solver | Delta |
|---|---:|---:|---:|
| Total weak | `192/315` | `201/315` | `+9` |
| equation_transform | `56/155` | `63/155` | `+7` |
| bit_manipulation | `136/160` | `138/160` | `+2` |

This is a CPU solver/verifier projection only. It is not an adapter-only Kaggle submission.

## Accepted Gains

- `274def88` `equation_transform`: `-92` -> `92` via `v324_equation_numeric`
- `4ada9150` `bit_manipulation`: `01111111` -> `01111011` via `v403_bit_global_exact`
- `4c327b55` `bit_manipulation`: `11011110` -> `11011100` via `v403_bit_global_exact`
- `528ec0d8` `equation_transform`: `39` -> `-39` via `v324_equation_numeric`
- `7688e06e` `equation_transform`: `55` -> `-55` via `v324_equation_numeric`
- `99d6a3b5` `equation_transform`: `(<))` -> `?()<` via `v329_symbolic_cryptarithm`
- `c5b058d6` `equation_transform`: `35` -> `134` via `v324_equation_numeric`
- `d1bd7478` `equation_transform`: `3` -> `30` via `v324_equation_numeric`
- `fb623471` `equation_transform`: `21` -> `-21` via `v324_equation_numeric`
# V409 Integrated Solver Projection v2

| Metric | Baseline | V409 projection | Delta |
|---|---:|---:|---:|
| Weak total | `192/315` | `202/315` | `+10` |
| equation_transform | `56/155` | `63/155` | `+7` |
| bit_manipulation | `136/160` | `139/160` | `+3` |

CPU solver/verifier projection only. Not adapter-only and not Kaggle-submitable as-is.

## Accepted Gains

- `274def88` `equation_transform`: `-92` -> `92` via `v405_integrated`
- `4ada9150` `bit_manipulation`: `01111111` -> `01111011` via `v405_integrated;v408_asym_bit_symbolic_pbe`
- `4c327b55` `bit_manipulation`: `11011110` -> `11011100` via `v405_integrated`
- `4ef88f92` `bit_manipulation`: `01010011` -> `01010111` via `v408_asym_bit_symbolic_pbe`
- `528ec0d8` `equation_transform`: `39` -> `-39` via `v405_integrated`
- `7688e06e` `equation_transform`: `55` -> `-55` via `v405_integrated`
- `99d6a3b5` `equation_transform`: `(<))` -> `?()<` via `v405_integrated`
- `c5b058d6` `equation_transform`: `35` -> `134` via `v405_integrated`
- `d1bd7478` `equation_transform`: `3` -> `30` via `v405_integrated`
- `fb623471` `equation_transform`: `21` -> `-21` via `v405_integrated`
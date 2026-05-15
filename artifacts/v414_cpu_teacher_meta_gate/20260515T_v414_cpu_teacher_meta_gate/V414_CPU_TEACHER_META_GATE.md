# V414 CPU Teacher Meta Gate

V414 consolidates the CPU solver/verifier evidence and explicitly separates it from adapter-only submit eligibility.

## Comparison

| State | Weak total | Delta | equation_transform | Delta | bit_manipulation | Delta | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| V291/V290 adapter baseline | `192/315` | `+0` | `56/155` | `+0` | `136/160` | `+0` | submit-safe baseline |
| V409 solver projection | `202/315` | `+10` | `63/155` | `+7` | `139/160` | `+3` | CPU teacher only |
| V412 CPU synthesis union | `202/315` | `+10` | `63/155` | `+7` | `139/160` | `+3` | CPU teacher only; no new gain over V409 |
| V357 bit global ternary union | `214/315` | `+22` | `63/155` | `+7` | `151/160` | `+15` | CPU teacher only |
| V414/V366 consolidated CPU teacher | `222/315` | `+30` | `63/155` | `+7` | `159/160` | `+23` | best CPU teacher; not adapter-only |

## Transfer Blockers

- V368 tried the V367/V366 bit-ternary transfer route and produced `191/315`, `equation=56/155`, `bit=135/160`; it transferred `0/8` V366 gains and introduced `2` losses vs baseline.
- V413 tried the solver-first transfer route and produced `190/315`, `equation=56/155`, `bit=134/160`, `truncated=1` at checkpoint-2; eval was canceled by FinOps.

## Decision

V366/V414 is the best CPU teacher currently available (`222/315`, `equation=63/155`, `bit=159/160`), but it is not adapter-only submit-safe. The same teacher-transfer pattern has already failed in GPU jobs, so another HF run on this route is blocked.

Next action: target adapter behavior directly. Do not train again from the same teacher rows unless a new CPU gate proves a materially different transfer mechanism and the first checkpoint can beat `192/315`, `equation>56`, `bit>=136`, `truncated=0`.

## Key New Rows Beyond V409

- `048cc279` `bit_manipulation`: `01111000` -> `01010000` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_MAJ3`
- `05ca617c` `bit_manipulation`: `01011011` -> `11011011` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `06120e47` `bit_manipulation`: `11111110` -> `11110010` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `0e70c867` `bit_manipulation`: `01001100` -> `01000000` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `1a7c8520` `bit_manipulation`: `01100110` -> `01100000` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_MAJ3`
- `1abaffca` `bit_manipulation`: `01011000` -> `01000000` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_CHO`
- `202af98d` `bit_manipulation`: `11111111` -> `11111101` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `3a7dd604` `bit_manipulation`: `01001011` -> `01001010` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `3ace787f` `bit_manipulation`: `11111111` -> `11111011` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `55d834d1` `bit_manipulation`: `10111111` -> `00111111` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `5ba26f21` `bit_manipulation`: `01111110` -> `01011100` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_CHO`
- `7192535b` `bit_manipulation`: `00001010` -> `00000010` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_MAJ3`
- `78d02fc5` `bit_manipulation`: `11001011` -> `11001001` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `82ae858c` `bit_manipulation`: `11111101` -> `11001101` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `a6192d29` `bit_manipulation`: `01111110` -> `00001000` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_MAJ3`
- `a6704625` `bit_manipulation`: `00011111` -> `00001101` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `b8722d19` `bit_manipulation`: `11110100` -> `00100100` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_CHO`
- `b8aa3072` `bit_manipulation`: `00000111` -> `00000011` via `v366_bit_fullbyte_ternary_op_gate` `bit_fullbyte_ternary_op_CHO`
- `e1f3ffbb` `bit_manipulation`: `11111011` -> `11111010` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
- `e6d2a064` `bit_manipulation`: `11100010` -> `11100011` via `v357_bit_global_ternary_gate` `bit_exact_global_ternary_unique_prediction`
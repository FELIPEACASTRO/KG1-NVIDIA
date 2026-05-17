# V522 Source Target Alignment Audit

## Decision

- GPU allowed: `False`
- Dataset build allowed: `True`
- Status: `source_signal_found_dataset_build_only`
- Reason: Reference teacher has no-loss gains, but those gains are not adapter behavior. Use them only to choose source-side trace families; do not train on weak labels.
- Next action: Build V523 targeted source-only trace pack from permitted v304/v515-like sources: prioritize CHO/MAJ3/global ternary bit traces and current V516 label-free equation classes ['274def88', '7688e06e', 'c5b058d6', 'd1bd7478']; then run V286/V513/V521 before any GPU.

## Reference Signal

- No-loss teacher gains: `31`
- Teacher losses vs baseline: `0`
- Gain family counts: `{"bit_manipulation": 23, "equation_transform": 8}`

Top gain rules:

- `bit_manipulation:bit_exact_global_ternary_unique_prediction`: `13`
- `equation_transform:equation_reference_gain_untyped`: `8`
- `bit_manipulation:bit_fullbyte_ternary_op_CHO`: `4`
- `bit_manipulation:bit_fullbyte_ternary_op_MAJ3`: `4`
- `bit_manipulation:bit_exact_global_binary_OR`: `1`
- `bit_manipulation:bit_exact_global_binary_XOR`: `1`

## Source Coverage

| Source | Split | Rows | CHO | MAJ3 | PAR3 | XOR | OR | fullbyte | gain-pattern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v304_solver_trace_distill | train | 12822 | 506 | 709 | 105 | 5034 | 4873 | 1536 | 1056 |
| v304_solver_trace_distill | validation | 969 | 54 | 69 | 17 | 369 | 350 | 168 | 88 |
| v515_v514_fullbyte_residual | train | 2491 | 4 | 3 | 0 | 237 | 238 | 7 | 0 |
| v515_v514_fullbyte_residual | validation | 620 | 0 | 1 | 0 | 57 | 57 | 1 | 0 |

## Rule

The gain rows in this audit are diagnostic targets only. They cannot be copied into training labels. V523 must draw training rows from source-side synthetic/public/train data with no weak/full prompt overlap.

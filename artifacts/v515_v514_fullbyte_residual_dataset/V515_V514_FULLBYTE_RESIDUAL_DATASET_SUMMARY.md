# V515 V514 Fullbyte Residual Dataset Summary

- Generated UTC: `2026-05-16T22:13Z`
- Base dataset: V514 traceable bit V510 dataset
- Method: CPU-only residual full-byte solver
- Submit/package/train: not run

## Objective

Recover only the V514 bit rows that were dropped as unverified, without adding guesses.

Acceptance rule:

1. The row must be a V510 bit row not already converted by V514.
2. `solve_fullbyte(prompt)` must return `fullbyte_unique_prediction`.
3. The prediction must exactly verify against the row answer.
4. Ambiguous or no-expression rows remain excluded.

## Result

| Split | V514 rows | V515 rows | Added rows | Added bit rows |
|---|---:|---:|---:|---:|
| train | `2484` | `2491` | `+7` | `+7` |
| validation | `619` | `620` | `+1` | `+1` |

Family counts after V515:

| Split | equation_transform | bit_manipulation |
|---|---:|---:|
| train | `2018` | `473` |
| validation | `504` | `116` |

Residual search details:

| Split | residual seen | accepted | ambiguous | no expression |
|---|---:|---:|---:|---:|
| train | `143` | `7` | `0` | `136` |
| validation | `18` | `1` | `2` | `15` |

Accepted rule class: `fullbyte_unique_prediction`.

## Gates

V286 tokenization gate passed:

| Metric | Train | Validation |
|---|---:|---:|
| rows | `2491` | `620` |
| prompt truncation rate | `0.0` | `0.0` |
| completion tokens dropped | `0` | `0` |
| offset masks | `2491/2491` | `620/620` |
| token max | `553` | `541` |

V513 trace learnability recheck passed:

| Metric | Value |
|---|---:|
| projected rows | `3111` |
| blockers | `0` |
| warnings | `0` |
| info | `1` |

V478 objective alignment gate:

| Weights | bit effective share | equation effective share | Decision |
|---|---:|---:|---|
| equal source/subcategory weights | `18.99%` | `81.01%` | blocked |
| bit sources `1.5x`, other sources/subcategories `1.0x` | `26.01%` | `73.99%` | passed |

Artifacts:

- `v515_objective_alignment_equal_weights.json`
- `v515_objective_alignment_bit_source_1p5.json`

## Decision

V515 is a small verified coverage gain over V514, but still not a submit-safe ACC gain. It is eligible for HF CPU reproduction only. Paid GPU remains blocked until objective/pre-paid gates approve a tiny smoke and FinOps rules are satisfied.

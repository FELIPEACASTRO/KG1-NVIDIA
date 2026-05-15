# V431 Signed Cryptarithm Gate

Generated: 2026-05-15T10:35:07.369844+00:00

CPU-only gate for signed/padded symbolic cryptarithm rows. It is diagnostic only and not a submit artifact.

## Comparison

| Metric | Baseline V291/V290 | V431 projection | Delta |
|---|---:|---:|---:|
| Total weak correct | `192/315` | `193/315` | `1` |
| equation_transform | `56/155` | `57/155` | `1` |
| bit_manipulation | `136/160` | `136/160` | `0` |
| Truncated | `0` | `0` | `0` |

## Gate Counts

| Metric | Value |
|---|---:|
| Equation rows audited | `155` |
| Rows with unique candidate | `2` |
| Ambiguous candidate rows blocked | `4` |
| Accepted total gains vs baseline | `1` |
| Accepted new gains beyond V414 | `0` |
| Conflict rows blocked | `0` |

## Accepted Rows

| id | old_prediction | new_prediction | answer |
|---|---|---|---|
| `99d6a3b5` | `(<))` | `?()<` | `?()<` |

## Decision

`hf_gpu_allowed = false` unless this CPU gate beats the adapter-only baseline with no-loss rows.

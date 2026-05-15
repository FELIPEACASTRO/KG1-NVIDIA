# V429 Symbolic Edit-Transducer Gate

Generated: 2026-05-15T10:20:52.051200+00:00

CPU-only gate for a new symbolic punctuation rule class: ordered edit transducers with delete/copy/constant/table actions and constant insertions.

## Comparison

| Metric | Baseline V291/V290 | V429 projection | Delta |
|---|---:|---:|---:|
| Total weak correct | `192/315` | `192/315` | `0` |
| equation_transform | `56/155` | `56/155` | `0` |
| bit_manipulation | `136/160` | `136/160` | `0` |
| Truncated | `0` | `0` | `0` |

## Gate Counts

| Metric | Value |
|---|---:|
| Equation rows audited | `155` |
| Changed candidate rows | `1` |
| Accepted new gains | `0` |
| Conflict rows blocked | `0` |

## Accepted Rows

| id | old_prediction | new_prediction | answer |
|---|---|---|---|
| none | none | none | none |

## Decision

`hf_gpu_allowed = false` unless this CPU gate produces accepted gains that beat the adapter-only baseline. This run is evidence for/against the edit-transducer class only; it is not a submit artifact.

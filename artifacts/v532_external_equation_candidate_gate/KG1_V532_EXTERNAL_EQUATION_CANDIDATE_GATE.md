# V532 External Equation Candidate Gate

CPU-only diagnostic. Candidate selection did not use labels, expected answers, or `competition_match`; weak labels are used only after selection to audit aggregate gains/losses.

## Selector Summary

| selector | selected_correct | baseline_correct | gains | losses | net | diagnostic no-loss candidate |
|---|---:|---:|---:|---:|---:|---|
| `critic_v2_only` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |
| `router_v1_only` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |
| `critic_router_union` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |

## Dataset Scope

| source | rows | weak overlap rows | unique weak ids | direct candidate pool |
|---|---:|---:|---:|---|
| `critic_v2` | 51338 | 5833 | 155 | `True` |
| `router_v1` | 51334 | 5833 | 155 | `True` |
| `selection_v2` | 254 | 0 | 0 | `False` |
| `solver_swap_v1` | 2907 | 74 | 71 | `False` |

## Decision

- `blocked`: `True`.
- `hf_gpu_allowed`: `False`.
- No row-level weak decision file is written; row IDs/outcomes/correctness are not materialized.
- No weak gain row may be copied into training. This artifact is diagnostic-only.
- If no selector is a diagnostic no-loss candidate, these datasets are useful only as verifier/canonicalization feature references.
- `selection_v2` and `solver_swap_v1` were inventoried but not used as direct selectors in V532; they require a separate V535 source-only rule/canonicalization audit.
- If a selector becomes a diagnostic no-loss candidate in a future run, derive a source-only rule or hard-negative hypothesis and rerun weak/full anti-leakage gates before any training.

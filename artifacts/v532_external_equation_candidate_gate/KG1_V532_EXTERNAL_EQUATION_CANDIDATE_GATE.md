# V532 External Equation Candidate Gate

CPU-only diagnostic. Candidate selection did not use labels, expected answers, or `competition_match`; those fields are audit-only.

## Selector Summary

| selector | selected_correct | baseline_correct | gains | losses | net | promotable |
|---|---:|---:|---:|---:|---:|---|
| `critic_v2_only` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |
| `router_v1_only` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |
| `critic_router_union` | 29/155 | 55/155 | 2 | 28 | -26 | `False` |

## Decision

- If no selector is promotable, these datasets are useful as verifier/canonicalization feature references, not direct submit-safe gains.
- If a selector is promotable, convert only its gain rows into a guarded CPU rule or short hard-negative training pack and rerun weak gates.

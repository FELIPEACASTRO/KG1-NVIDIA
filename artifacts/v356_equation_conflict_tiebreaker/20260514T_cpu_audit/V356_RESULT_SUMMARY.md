# V356 Result Summary

Generated: 2026-05-14

## Scope

V356 audited the two `equation_transform` cryptarithm candidates that were correct by weak label but rejected by V350/V355 due to conflicting predictions.

## Result

| Metric | Value |
|---|---:|
| Conflicts audited | `2` |
| Label-free tiebreakers found | `0` |
| Query-only operator conflicts | `2` |

## Decision

Blocked.

Both apparent gains require choosing an operator that appears in the query but not in the examples. That is not a defensible label-free rule, so these rows remain `abstain`.

No HF job is authorized from V356.

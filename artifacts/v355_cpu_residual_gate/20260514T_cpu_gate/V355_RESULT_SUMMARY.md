# V355 Result Summary

Generated: 2026-05-14

## Scope

V355 audited the next CPU-only residual candidates after the V352 transfer failure:

- `bit_manipulation` stride/bit-pair solver classes;
- current bit solver high-coverage classes;
- `equation_transform` cryptarithm conflicts rejected by V350.

No GPU, no package, and no Kaggle submit were involved.

## Result

| State | Overall | Equation | Bit |
|---|---:|---:|---:|
| V350 baseline | `201/315` | `63/155` | `138/160` |
| V355 integrated | `201/315` | `63/155` | `138/160` |

## Gate Decision

Blocked.

Details:

- Candidate decisions audited: `49`.
- Accepted candidates: `0`.
- Bit stride had no safe gain and produced losses.
- Current bit solver had one gain in the best class, but also six losses.
- Two equation cryptarithm candidates matched the weak labels, but both were still ambiguous conflicts with no label-free tie-breaker.

Decision: do not launch HF. Continue CPU-only search.

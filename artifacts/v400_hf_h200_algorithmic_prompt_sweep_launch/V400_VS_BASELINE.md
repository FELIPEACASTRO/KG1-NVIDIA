# V400 Algorithmic Prompt Sweep vs Baseline

Date: 2026-05-14

Job: https://huggingface.co/jobs/felipesp1983/6a065b35e48bea4538b9d881

## Result

| Candidate | Total weak | equation_transform | bit_manipulation | Truncated | Delta vs baseline | Decision |
|---|---:|---:|---:|---:|---:|---|
| V392/V290 locked baseline | `192/315` | `56/155` | `136/160` | `0` | `0` | keep |
| V400 `symbolic_equation_first` | `175/315` | `40/155` | `135/160` | `27` | `-17` | reject |
| V400 `bit_stride_guarded` | `7/315` | `7/155` | `0/160` | `227` | `-185` | reject |

## Decision

V400 does not improve the ranking path. Explicit algorithmic instructions in the prompt caused long generations, truncation, and severe accuracy collapse. Do not run more broad algorithmic prompt suffixes on the locked adapter.

## Next Action

Return to CPU-only audits that can expose a submit-safe adapter/package change before any more GPU spend. The next low-cost check is whether baseline raw outputs already contain correct answers on missed rows but the final boxed extraction or formatting loses them. If not, the only safe package remains V291/V290 checkpoint-6.

# V413 FinOps Cutoff

V413 tested whether the V410 solver-first transfer dataset could move the
V291/V290 checkpoint-6 adapter beyond the current submit-safe weak baseline.

| Candidate | Weak total | equation_transform | bit_manipulation | Truncated | Decision |
|---|---:|---:|---:|---:|---|
| V291/V290 checkpoint-6 baseline | `192/315` | `56/155` | `136/160` | `0` | keep |
| V413 checkpoint-2 | `190/315` | `56/155` | `134/160` | `1` | reject |

The weak eval was canceled while checkpoint-4 was starting because checkpoint-2
already failed the hard promotion gate: total did not exceed baseline,
`equation_transform` stayed at `56`, `bit_manipulation` fell below `136`, and
truncation appeared.

Decision: no full eval, no package, no submit, and no longer V413 training.
Return to CPU row-level synthesis and require a new no-loss signal before any
additional GPU spend.

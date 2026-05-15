# V413 Weak Eval Comparison

V413 evaluates the two H200 solver-first transfer checkpoints against the same
V221 weak contract used for the current submit-safe baseline. Promotion remains
strict: no full eval or submit unless weak accuracy improves without bit
regression.

| Item | Previous submit-safe baseline | V413 candidate |
|---|---:|---:|
| Adapter source | V291/V290 checkpoint-6 | V413 checkpoint-2 and checkpoint-4 |
| Weak total gate | 192/315 | pending weak eval |
| equation_transform gate | 56/155 | pending weak eval |
| bit_manipulation gate | 136/160 | pending weak eval |
| Truncation gate | 0 | must remain 0 |
| Full eval trigger | already submit-safe | only if total > 192, equation > 56, bit >= 136 |
| Submit trigger | best known public score 0.86 | only if official-like full >= 824/947 |

V413 is intentionally a smoke transfer test, not a long training line. If weak
eval does not beat the baseline, the FinOps decision is to stop this line and
return to CPU solver/verifier synthesis.

# V459 V458 Numeric Hard Negative Audit

## Summary

- Rows audited: `22`.
- Real hard negatives: `7`.
- Decision: `v459_signal_real_but_narrow_gpu_blocked`.
- HF GPU allowed: `false`.
- Next action: Build V460 CPU dataset proposal, but do not launch paid GPU unless explicitly accepting one-rule risk.

## Rule Detail

| Rule | Rows | Adapter correct | Adapter matches simulated wrong | Postprocessor correct | Real hard negatives |
|---|---:|---:|---:|---:|---:|
| v274_guarded_numeric_minus_signed_opposite_sign_guarded | 22 | 15 | 7 | 22 | 7 |

## Interpretation

V458 confirmed adapter-level signal for one numeric equation class. This is stronger than synthetic-only evidence because the rejected answers are actual frozen-adapter predictions collected before labels were joined. The signal is still narrow: one rule class and seven hard negatives, so a paid GPU job remains blocked unless the next dataset builder can add clean coverage or explicitly accepts a one-rule micro-smoke risk.

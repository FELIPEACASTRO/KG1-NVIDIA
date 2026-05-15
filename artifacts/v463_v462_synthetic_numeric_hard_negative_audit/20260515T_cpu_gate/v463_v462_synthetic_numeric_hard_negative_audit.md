# V463 V462 Synthetic Numeric Hard Negative Audit

## Summary

- Rows audited: `56`.
- Real hard negatives: `26`.
- Real hard-negative rule classes: `3`.
- Decision: `v463_multi_rule_synthetic_signal_ready_for_v464_cpu_dataset`.
- V464 dataset build allowed: `true`.
- HF GPU train allowed: `false`.
- Next action: Build V464 CPU dataset proposal with only real adapter hard negatives, bit replay, tokenization gates, and weak/full promotion guards. Do not train yet.

## Rule Detail

| Rule | Rows | Adapter correct | Adapter matches simulated wrong | Postprocessor correct | Real hard negatives | Prompt hashes match |
|---|---:|---:|---:|---:|---:|---:|
| v274_guarded_numeric_add_direct_over_model_add_variant | 16 | 0 | 16 | 16 | 16 | 16 |
| v274_guarded_numeric_colon_absdiff_restore_trailing_zero | 16 | 16 | 0 | 16 | 0 | 16 |
| v274_guarded_numeric_minus_direct_negative_restore_sign | 8 | 0 | 8 | 8 | 8 | 8 |
| v274_guarded_numeric_minus_signed_opposite_sign_guarded | 16 | 14 | 2 | 16 | 2 | 16 |

## Interpretation

This is synthetic prompt evidence joined after inference, so it can justify a CPU dataset proposal only. It does not authorize paid GPU training by itself. GPU training remains blocked until a later dataset gate shows multi-rule coverage, clean tokenization, bit replay, and no weak/full regression risk.

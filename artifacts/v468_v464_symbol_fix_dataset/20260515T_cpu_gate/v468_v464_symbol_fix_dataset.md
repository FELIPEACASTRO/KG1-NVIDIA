# V464 V463 Numeric Multi-Rule Dataset

## Purpose

CPU dataset proposal from V463 multi-rule real adapter mistakes, plus V217 bit replay guardrail.

## Counts

- Train rows: `558`; families: `{'bit_manipulation': 512, 'equation_transform': 46}`.
- Validation rows: `138`; families: `{'bit_manipulation': 128, 'equation_transform': 10}`.
- Train hard negatives: `22`.
- Validation hard negatives: `4`.
- Rule classes in train hard negatives: `['v274_guarded_numeric_add_direct_over_model_add_variant', 'v274_guarded_numeric_minus_direct_negative_restore_sign', 'v274_guarded_numeric_minus_signed_opposite_sign_guarded']`.

## Decision

- Tokenization gate required: `true`.
- HF GPU allowed: `false`.
- Decision: `v464_dataset_ready_for_tokenization_gate`.
- Next action: Run V286 generic tokenization gate with boxed_suffix mode; only then consider a one-checkpoint HF smoke.

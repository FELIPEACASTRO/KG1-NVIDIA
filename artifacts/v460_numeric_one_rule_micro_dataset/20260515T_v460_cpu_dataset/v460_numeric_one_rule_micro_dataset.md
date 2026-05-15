# V460 Numeric One-Rule Micro Dataset

## Purpose

CPU proposal for a one-rule numeric equation smoke. It uses the V459 real adapter mistakes and bit replay guardrail.

## Counts

- Train rows: `146`; families: `{'bit_manipulation': 128, 'equation_transform': 18}`.
- Validation rows: `36`; families: `{'bit_manipulation': 32, 'equation_transform': 4}`.
- Real hard negatives in train: `7`.
- Equation validation rows are positive replay only: `True`.

## Decision

- HF GPU allowed: `false`.
- Decision: `v460_blocks_gpu_one_rule_risk_not_acknowledged`.
- Next action: Run tokenization gate, then require explicit one-rule micro-smoke risk acceptance before paid GPU.

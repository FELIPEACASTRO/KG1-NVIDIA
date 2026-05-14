# KG1 V374 CPU Residual Gate Summary

## Inputs

- Baseline/input predictions: `artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv`
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`
- Input baseline: `222/315`, `equation_transform=63/155`, `bit_manipulation=159/160`

## Equation Gate

Command route: `scripts/run_v324_equation_expanded_solver_gate.py` on top of V366 predictions.

Result:

- Equation miss rows audited: `92`
- Parse status: `92/92 ok`
- Subtypes: `10` numeric operator, `82` symbolic punctuation
- Accepted no-loss candidates: `0`
- Projected equation: `63/155`
- Decision: `no_new_equation_signal_for_hf_gpu`

## Bit Gate

Command route: `scripts/run_v333_tong_bit_reasoner_gate.py` with Tong commit `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85`.

Train-side audit:

- Tong train bit: `1364/1602 = 85.14%`
- Current local solver train bit: `1265/1602 = 78.96%`
- Tong gains vs current train solver: `157`
- Tong losses vs current train solver: `58`

Weak-side audit over V366 baseline:

- V366 baseline: `222/315`, `bit=159/160`, `equation=63/155`
- Tong direct bit replacement: `199/315`, `bit=136/160`, `equation=63/155`
- Tong gains vs V366 weak baseline: `0`
- Tong losses vs V366 weak baseline: `23`
- Decision: `tong_bit_signal_blocked`

## Decision

V374 does not authorize HF GPU, full eval, package, or submit.

Reason:

- Equation route found `0` accepted no-loss gains.
- Tong bit route confirms public train strength, but it is worse than V366 on the weak contract and has `23` weak losses.

Next action: CPU-only residual inspection of the `92` equation misses. Focus on clustering symbolic punctuation examples and proving a new rule class before any training.

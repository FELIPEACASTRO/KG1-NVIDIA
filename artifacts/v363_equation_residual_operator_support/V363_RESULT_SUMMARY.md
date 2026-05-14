# V363 Equation Residual Operator-Support Result

## Decision

Blocked. V363 does not allow HF GPU, full eval, package, or Kaggle submit.

## Evidence

- Script: `scripts/analyze_v363_equation_residual_operator_support.py`.
- Manifest: `artifacts/v363_equation_residual_operator_support/20260514T_cpu_gate/v363_equation_residual_operator_support_manifest.json`.
- Input predictions: `artifacts/v355_cpu_residual_gate/20260514T_cpu_gate/v355_integrated_predictions.csv`.
- Public train anti-leakage: `315` weak ids excluded before learning numeric operator priors.
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.

## Current Best

| State | Overall | Equation | Bit |
|---|---:|---:|---:|
| V355/V350 integrated CPU | `201/315` | `63/155` | `138/160` |
| V363 integrated CPU | `201/315` | `63/155` | `138/160` |

## Residual Map

Remaining `equation_transform` misses: `92`.

| Residual route | Rows |
|---|---:|
| numeric query operator unseen in examples | `10` |
| symbolic query operator seen, but same-op DSL has no unique candidate | `70` |
| symbolic query operator unseen in examples | `12` |

V363 tested public-train numeric operator priors after excluding weak ids. They were rejected because they caused losses on currently correct rows:

| Rule | Changed | Gains | Losses |
|---|---:|---:|---:|
| `numeric_operator_prior_45_sub_ab` | `8` | `0` | `7` |
| `numeric_operator_prior_123_concat_ab` | `3` | `0` | `3` |
| `numeric_operator_prior_92_abs_diff` | `1` | `0` | `1` |
| `numeric_operator_prior_93_abs_diff` | `1` | `0` | `1` |

Same-operator symbolic DSL produced no promotable no-loss candidate.

## Next Action

Do not launch HF from V363. The next useful CPU work must be a new symbolic-program family or a stronger label-free ambiguity resolver. More epochs, boxed-only replay, and public-train numeric priors are blocked by evidence.

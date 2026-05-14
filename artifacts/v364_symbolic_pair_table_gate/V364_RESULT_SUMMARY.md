# V364 Symbolic Pair-Table Gate Result

## Decision

Blocked. V364 does not allow HF GPU, full eval, package, or Kaggle submit.

## Evidence

- Script: `scripts/analyze_v364_symbolic_pair_table_gate.py`.
- Manifest: `artifacts/v364_symbolic_pair_table_gate/20260514T_cpu_gate/v364_symbolic_pair_table_gate_manifest.json`.
- Input predictions: `artifacts/v363_equation_residual_operator_support/20260514T_cpu_gate/v363_integrated_predictions.csv`.
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.

## Result

| State | Overall | Equation | Bit |
|---|---:|---:|---:|
| V363 input | `201/315` | `63/155` | `138/160` |
| V364 output | `201/315` | `63/155` | `138/160` |

V364 generated `12` candidate changes from the symbolic pair-table family, but none were correct gains.

| Rule | Changed | Gains | Losses |
|---|---:|---:|---:|
| `symbolic_pair_table_len_2` | `2` | `0` | `0` |
| `symbolic_pair_table_len_3` | `2` | `0` | `0` |
| `symbolic_pair_table_len_4` | `8` | `0` | `2` |

## Next Action

Do not train on V364. The pair-table hypothesis is rejected. The next CPU route must either add a genuinely new symbolic semantics family or switch to bit-only CPU search where V350/V357 showed measurable solver gains.

# V520 Local Candidate Mining

## Result

- Existing local CSVs were rescored against the V516 label-free weak contract.
- No adapter-only eval CSV found above `191/315` while preserving `bit>=136`, `trunc=0`, and `8740ed31=01101000`.
- High numbers such as `222/315`, `207/315`, and `201/315` are CPU solver/postprocessor/integrated projections, not direct Kaggle adapter submissions.

## Top Reference Signals (not submit adapter)

| total | equation | bit | protected | path |
|---:|---:|---:|---|---|
| 222 | 63 | 159 | True | `artifacts\v380_solver_results_patch_gate\20260514T_cpu_gate\v380_reexecuted_teacher_predictions.csv` |
| 222 | 63 | 159 | True | `artifacts\v366_bit_fullbyte_ternary_op_gate\20260514T_cpu_gate\v366_integrated_predictions.csv` |
| 214 | 63 | 151 | True | `artifacts\v365_bit_residual_boolean_grammar_gate\20260514T_cpu_gate\v365_integrated_predictions.csv` |
| 214 | 63 | 151 | True | `artifacts\v357_bit_global_ternary_gate\20260514T_cpu_gate\v357_integrated_predictions.csv` |
| 207 | 60 | 147 | True | `artifacts\v301_bit_postprocessor_gate\20260512T1130Z\v301_bit_postprocessed_predictions.csv` |
| 206 | 60 | 146 | True | `artifacts\v302_combined_postprocessor_gate\20260512T1200Z\v302_combined_postprocessed_predictions.csv` |

## Top Adapter Eval Candidates

| total | equation | bit | protected | path |
|---:|---:|---:|---|---|
| 191 | 56 | 135 | False | `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\predictions.csv` |
| 191 | 56 | 135 | False | `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\raw_predictions_pre_score.csv` |
| 191 | 56 | 135 | False | `artifacts\v352_hf_a100_v351_bit_transfer_launch\eval_v352_checkpoint2\predictions.csv` |
| 190 | 56 | 134 | False | `artifacts\v342_acc_first_diagnostic\v341_checkpoint2_predictions.csv` |

## Decision

Do not submit or rerun old candidates. The next useful work is to convert reference solver behavior into adapter behavior without losing `8740ed31`, then pass V519 guard plus weak promotion.

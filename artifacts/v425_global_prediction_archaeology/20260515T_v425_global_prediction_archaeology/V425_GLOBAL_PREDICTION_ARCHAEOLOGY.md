# V425 Global Prediction Archaeology

Generated: 2026-05-15T09:58:02.987378+00:00

## Baseline Contract

| Candidate | Total | equation_transform | bit_manipulation | Truncated |
|---|---:|---:|---:|---:|
| V291/V290 checkpoint-6 | `192/315` | `56/155` | `136/160` | `0` |

## Scan Summary

- CSV files considered: `303`.
- Scored prediction columns: `93`.
- Adapter-like scored columns: `12`.
- Promotable adapter-like candidates: `0`.

## Best Adapter-Like Candidates

| CSV | Col | Total | equation | bit | trunc | V414 hits | Losses | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `artifacts\v342_acc_first_diagnostic\v290_checkpoint6_baseline_predictions.csv` | `prediction` | `192` | `56` | `136` | `0` | `0` | `0` | reject |
| `artifacts\v342_acc_first_diagnostic\v290_checkpoint6_baseline_predictions.csv` | `raw_output` | `192` | `56` | `136` | `0` | `0` | `0` | reject |
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\predictions.csv` | `prediction` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\predictions.csv` | `raw_output` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\raw_predictions_pre_score.csv` | `prediction` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\raw_predictions_pre_score.csv` | `raw_output` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v347_v346_failure_audit\input\v346_checkpoint2_predictions.csv` | `prediction` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v347_v346_failure_audit\input\v346_checkpoint2_predictions.csv` | `raw_output` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v352_hf_a100_v351_bit_transfer_launch\eval_v352_checkpoint2\predictions.csv` | `prediction` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v352_hf_a100_v351_bit_transfer_launch\eval_v352_checkpoint2\predictions.csv` | `raw_output` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v342_acc_first_diagnostic\v341_checkpoint2_predictions.csv` | `prediction` | `190` | `56` | `134` | `1` | `0` | `2` | reject |
| `artifacts\v342_acc_first_diagnostic\v341_checkpoint2_predictions.csv` | `raw_output` | `190` | `56` | `134` | `1` | `0` | `2` | reject |

## Best Non-Adapter/Teacher Signals

| CSV | Col | Source | Total | equation | bit | trunc | V414 hits |
|---|---|---|---:|---:|---:|---:|---:|
| `artifacts\v366_bit_fullbyte_ternary_op_gate\20260514T_cpu_gate\v366_integrated_predictions.csv` | `prediction` | `teacher_solver_or_postprocessor` | `222` | `63` | `159` | `0` | `30` |
| `artifacts\v366_bit_fullbyte_ternary_op_gate\20260514T_cpu_gate\v366_integrated_predictions.csv` | `v366_prediction` | `teacher_solver_or_postprocessor` | `222` | `63` | `159` | `0` | `30` |
| `artifacts\v374_cpu_residual_gate\20260514T_v374_cpu_gate\bit_v333_tong_on_v366\v374_tong_bit_on_v366_tong_bit_replace_predictions.csv` | `v366_prediction` | `teacher_solver_or_postprocessor` | `222` | `63` | `159` | `0` | `30` |
| `artifacts\v380_solver_results_patch_gate\20260514T_cpu_gate\v380_reexecuted_teacher_predictions.csv` | `prediction` | `teacher_solver_or_postprocessor` | `222` | `63` | `159` | `0` | `30` |
| `artifacts\v380_solver_results_patch_gate\20260514T_cpu_gate\v380_reexecuted_teacher_predictions.csv` | `v366_prediction` | `teacher_solver_or_postprocessor` | `222` | `63` | `159` | `0` | `30` |
| `artifacts\v357_bit_global_ternary_gate\20260514T_cpu_gate\v357_integrated_predictions.csv` | `prediction` | `teacher_solver_or_postprocessor` | `214` | `63` | `151` | `0` | `22` |
| `artifacts\v357_bit_global_ternary_gate\20260514T_cpu_gate\v357_integrated_predictions.csv` | `v357_prediction` | `teacher_solver_or_postprocessor` | `214` | `63` | `151` | `0` | `22` |
| `artifacts\v365_bit_residual_boolean_grammar_gate\20260514T_cpu_gate\v365_integrated_predictions.csv` | `current_prediction` | `teacher_solver_or_postprocessor` | `214` | `63` | `151` | `0` | `22` |
| `artifacts\v365_bit_residual_boolean_grammar_gate\20260514T_cpu_gate\v365_integrated_predictions.csv` | `prediction` | `teacher_solver_or_postprocessor` | `214` | `63` | `151` | `0` | `22` |
| `artifacts\v365_bit_residual_boolean_grammar_gate\20260514T_cpu_gate\v365_integrated_predictions.csv` | `v357_prediction` | `teacher_solver_or_postprocessor` | `214` | `63` | `151` | `0` | `22` |

## Decision

`v425_no_promotable_adapter_candidate_found`: No existing adapter-like CSV beats total>192, equation>56, bit>=136, trunc=0. Do not spend GPU on archaeology result.

GPU spending remains blocked unless an adapter-like artifact or a new CPU gate proves a path that can beat the weak baseline without bit/truncation regression.
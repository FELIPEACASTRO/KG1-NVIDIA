# V415 Adapter Direct Audit

V415 scans existing row-level adapter eval CSVs and asks whether any candidate already captures V414 teacher gains without baseline regressions.

## Baseline

- V291/V290 weak: `192/315`, equation `56/155`, bit `136/160`.
- V414 teacher gain rows audited: `30`.

## Best Existing Adapter-Like Candidates By V414 Hits

| Candidate CSV | Correct | equation | bit | trunc | V414 hits | Losses vs baseline | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\predictions.csv` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v368_hf_a100_v367_bit_ternary_launch\eval_checkpoint1\raw_predictions_pre_score.csv` | `191` | `56` | `135` | `0` | `1` | `2` | reject |
| `artifacts\v342_acc_first_diagnostic\v290_checkpoint6_baseline_predictions.csv` | `192` | `56` | `136` | `0` | `0` | `0` | reject |
| `artifacts\v347_v346_failure_audit\input\v346_checkpoint2_predictions.csv` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v352_hf_a100_v351_bit_transfer_launch\eval_v352_checkpoint2\predictions.csv` | `191` | `56` | `135` | `0` | `0` | `1` | reject |
| `artifacts\v342_acc_first_diagnostic\v341_checkpoint2_predictions.csv` | `190` | `56` | `134` | `1` | `0` | `2` | reject |

## Decision

No existing adapter-like row-level eval passed the promotion screen. Some candidates may hit isolated V414 rows, but they also lose too many baseline rows or keep equation at `56`.
Next step remains a new transfer mechanism, not another broad SFT of the same teacher rows.
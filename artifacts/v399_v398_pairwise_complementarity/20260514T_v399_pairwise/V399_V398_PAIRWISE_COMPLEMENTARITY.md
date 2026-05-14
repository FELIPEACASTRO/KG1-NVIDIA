# V399 V398 Pairwise Complementarity Audit

- Generated UTC: `2026-05-14T23:23:20.861328+00:00`
- Baseline CSV: `artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv`
- Baseline SHA256: `910a051d8b8e652e37c0b0814ac59fe4a400b95cb432945b6a0244f97f5b31bf`
- V398 repo: `felipesp1983/kg1-nemotron-lora-v398-nemo-h200-sft-reconstructed-v290ckpt6`

## Downloaded Inputs

- Full V398 prediction CSVs were downloaded from HF for the audit and intentionally not kept in git because they are reproducible from `felipesp1983/kg1-nemotron-lora-v398-nemo-h200-sft-reconstructed-v290ckpt6`.
- Kept artifacts are the summary, manifest, script, and changed-row CSVs needed for the decision.

## Summary

| Candidate | Family | Baseline | Candidate | Candidate-only | Baseline-only | Delta | Trunc |
|---|---:|---:|---:|---:|---:|---:|---:|
| v398_checkpoint_2 | bit_manipulation | 136 | 134 | 0 | 2 | -2 | 1 |
| v398_checkpoint_2 | equation_transform | 56 | 56 | 0 | 0 | 0 | 0 |
| v398_checkpoint_2 | OVERALL | 192 | 190 | 0 | 2 | -2 | 1 |
| v398_checkpoint_4 | bit_manipulation | 136 | 135 | 0 | 1 | -1 | 0 |
| v398_checkpoint_4 | equation_transform | 56 | 56 | 0 | 0 | 0 | 0 |
| v398_checkpoint_4 | OVERALL | 192 | 191 | 0 | 1 | -1 | 0 |

## Decision

- `best_overall_candidate`: `v398_checkpoint_4`
- `best_overall_correct`: `191`
- `best_equation_candidate`: `v398_checkpoint_2`
- `best_equation_candidate_only_correct`: `0`
- `best_bit_candidate`: `v398_checkpoint_4`
- `best_bit_correct`: `135`
- `actionable_without_new_selector`: `False`
- `decision`: `close_v397_v398_training_branch`
- `next_action`: `Do not run more V397/V398 SFT; return to CPU solver/verifier DSL and baseline package.`

## Interpretation

V398 is only actionable if the candidate-only equation rows expose a simple deterministic selector with zero bit regression. Otherwise this branch must remain rejected for FinOps and ranking purposes.

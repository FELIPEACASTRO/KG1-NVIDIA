# KG1 V1241 Full947 Baseline Readiness Gate

Decision: `fail_full947_086_baseline_readiness`

This CPU-only gate verifies the data needed before using full947 V1241 probes for V1243.
It does not train, upload, submit, or prove leaderboard score.

## Required Artifacts

- full947 solution CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\fixtures\full947_solution.csv`
- solution defaulted to canonical: `False`
- 086 natural raw_output CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\fixtures\086_natural_full947_raw_output.csv`
- 086 answer-only raw_output CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\fixtures\missing_answer_only.csv`

## Score Objective

- Target remains `>=0.89`, not `0.86`.
- Baseline reference: `823/947`.
- Minimum 0.89 proof: `843/947`.
- Required net gain after baseline readiness: `+20`.
- Next required candidate gate: `V1241 full947_089 with candidate >=843/947 and zero regressions`.

## Blocking Rule

- The natural 086 probe is diagnostic; it may fail strict-clean and still be useful.
- The answer-only 086 probe must pass public identity and strict-clean identity before any paid candidate train.

## Blockers

- `answer_only:answer_only predictions not found: C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\fixtures\missing_answer_only.csv`

## Probe Decisions

- Natural: `not_run`
- Answer-only: `not_run`

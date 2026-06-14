# KG1 V1241 Bit/Equation Transfer Gate Summary

Decision: `fail`
Generated at UTC: `2026-06-13T22:25:49.078363+00:00`

## Strict Gains

- Total strict gain: `1`
- Bit strict gain: `1`
- Equation strict gain: `1`

## Blocker Counts

- Any regressions: `1`
- Weak-family regressions: `0`
- Protected-family regressions: `1`
- Candidate format failures: `0`
- Candidate public-metric-only false gains: `0`
- Candidate strict-metric-only false positives: `0`
- Candidate truncated rows: `0`

## Baseline Identity

- Required: `False`
- Public identity pass: `not_applicable`
- Strict-clean identity pass: `not_applicable`
- Rationale: full947 promotion requires a strict-clean V291 baseline so strict row-delta regression checks protect the known public 823/947 baseline.

## Blockers

- `total_gain_below_threshold:1<2`
- `any_regressions_above_threshold:1>0`
- `protected_regressions_above_threshold:1>0`

## Output Files

- Report JSON: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_bit_equation_transfer_gate_selftest\fail_protected_regression\report\kg1_v1241_bit_equation_transfer_gate_report.json`
- Row deltas CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_bit_equation_transfer_gate_selftest\fail_protected_regression\report\kg1_v1241_bit_equation_transfer_gate_row_deltas.csv`

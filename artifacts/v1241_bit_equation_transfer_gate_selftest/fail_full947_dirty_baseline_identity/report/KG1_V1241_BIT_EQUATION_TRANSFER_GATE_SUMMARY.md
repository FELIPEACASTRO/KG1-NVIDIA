# KG1 V1241 Bit/Equation Transfer Gate Summary

Decision: `fail`
Generated at UTC: `2026-06-13T22:25:49.309376+00:00`

## Strict Gains

- Total strict gain: `843`
- Bit strict gain: `145`
- Equation strict gain: `66`

## Blocker Counts

- Any regressions: `0`
- Weak-family regressions: `0`
- Protected-family regressions: `0`
- Candidate format failures: `0`
- Candidate public-metric-only false gains: `0`
- Candidate strict-metric-only false positives: `0`
- Candidate truncated rows: `0`

## Baseline Identity

- Required: `True`
- Public identity pass: `True`
- Strict-clean identity pass: `False`
- Rationale: full947 promotion requires a strict-clean V291 baseline so strict row-delta regression checks protect the known public 823/947 baseline.

## Blockers

- `baseline_strict_clean_identity_unverified:full947:0!=823`
- `baseline_strict_clean_identity_unverified:bit_manipulation:0!=135`
- `baseline_strict_clean_identity_unverified:equation_transform:0!=56`
- `baseline_strict_clean_identity_unverified:protected:0!=632`

## Output Files

- Report JSON: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_bit_equation_transfer_gate_selftest\fail_full947_dirty_baseline_identity\report\kg1_v1241_bit_equation_transfer_gate_report.json`
- Row deltas CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_bit_equation_transfer_gate_selftest\fail_full947_dirty_baseline_identity\report\kg1_v1241_bit_equation_transfer_gate_row_deltas.csv`

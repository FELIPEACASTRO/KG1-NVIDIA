# KG1 V1241 Bit/Equation Transfer Gate Summary

Decision: `fail`
Generated at UTC: `2026-06-13T22:25:49.168832+00:00`

## Strict Gains

- Total strict gain: `1`
- Bit strict gain: `0`
- Equation strict gain: `1`

## Blocker Counts

- Any regressions: `0`
- Weak-family regressions: `0`
- Protected-family regressions: `0`
- Candidate format failures: `1`
- Candidate public-metric-only false gains: `1`
- Candidate strict-metric-only false positives: `0`
- Candidate truncated rows: `0`

## Baseline Identity

- Required: `False`
- Public identity pass: `not_applicable`
- Strict-clean identity pass: `not_applicable`
- Rationale: full947 promotion requires a strict-clean V291 baseline so strict row-delta regression checks protect the known public 823/947 baseline.

## Blockers

- `bit_gain_below_threshold:0<1`
- `total_gain_below_threshold:1<2`
- `candidate_format_failures_above_threshold:1>0`
- `candidate_public_metric_only_false_gains_above_threshold:1>0`

## Output Files

- Report JSON: `artifacts\v1241_bit_equation_transfer_gate_selftest_legacy_alias\fail_public_metric_only_false_gain\report\kg1_v1241_bit_equation_transfer_gate_report.json`
- Row deltas CSV: `artifacts\v1241_bit_equation_transfer_gate_selftest_legacy_alias\fail_public_metric_only_false_gain\report\kg1_v1241_bit_equation_transfer_gate_row_deltas.csv`

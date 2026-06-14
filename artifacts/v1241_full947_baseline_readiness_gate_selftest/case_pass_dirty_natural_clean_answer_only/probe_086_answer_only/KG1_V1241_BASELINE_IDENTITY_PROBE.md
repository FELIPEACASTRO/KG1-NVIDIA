# KG1 V1241 Baseline Identity Probe

Decision: `pass_baseline_strict_clean_identity_probe`

This CPU-only probe checks whether the full947 baseline CSV is strict-clean against the V291/086 public reference.
It does not evaluate a candidate and does not authorize training, packaging, submission, or score claims.

## Baseline Identity

- Public identity pass: `True`
- Strict-clean identity pass: `True`
- Observed public: `{'full947': 823, 'bit_manipulation': 135, 'equation_transform': 56, 'protected': 632}`
- Observed strict: `{'full947': 823, 'bit_manipulation': 135, 'equation_transform': 56, 'protected': 632}`

## Probe Blockers

- none

## Files

- Report JSON: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\case_pass_dirty_natural_clean_answer_only\probe_086_answer_only\kg1_v1241_baseline_identity_probe_report.json`
- Row deltas CSV: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1241_full947_baseline_readiness_gate_selftest\case_pass_dirty_natural_clean_answer_only\probe_086_answer_only\kg1_v1241_baseline_identity_probe_row_deltas.csv`

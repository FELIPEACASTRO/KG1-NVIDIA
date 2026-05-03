# V198 micro-distillation dataset

Train rows: 1875
Validation rows: 720

## Train families

- bit_manipulation: 474
- equation_transform: 552
- gravity_constant: 220
- numeral_system: 220
- text_encryption: 189
- unit_conversion: 220

## Sources

- v198_v195_balanced_rehearsal: 1729
- v198_v196_wrong_anti_regression: 116
- v198_v197_strict_gain_distill: 30

## Roles

- balanced_rehearsal: 1729
- anti_regression: 78
- duplicate_id_fixed: 38
- strict_gain_distill: 30

## Safety

- V197 local validation anchors are not trained directly.
- V196 wrong cases are used as anti-regression rows.
- Stable families stay represented to reduce regression risk.

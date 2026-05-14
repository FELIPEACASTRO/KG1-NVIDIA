# V370 V367 format transfer audit

Generated: 2026-05-14

## Result

- V367 train assistant targets boxed-only: `1128/1128`.
- V367 validation assistant targets boxed-only: `282/282`.
- V368 raw outputs exact boxed-only: `0/315`.
- V368 bit rows containing old `We need to deduce` trace: `160/160`.
- V368 bit rows containing `Output bit columns`: `160/160`.
- V366 accepted gains had at least `96` train and `24` validation rows each in V367.
- V366 accepted gains transferred to V368: `0/8`.

## Decision

Blocked. The boxed-only objective did not control the actual bit inference format. V368 continued producing the old long bit-reasoning trace on every bit row.

Next action: do not spend HF on V367/V368. A future LoRA attempt must first be CPU-gated with targets that match the actual bit trace format, or the roadmap should return to equation DSL.

## Local artifacts

- Manifest: `artifacts\v370_v367_format_transfer_audit\20260514T_cpu_audit\v370_v367_format_transfer_manifest.json`
- Family format summary: `artifacts\v370_v367_format_transfer_audit\20260514T_cpu_audit\v370_family_format_summary.csv`
- V366 coverage detail: `artifacts\v370_v367_format_transfer_audit\20260514T_cpu_audit\v370_v366_gain_training_coverage.csv`
- Raw examples: `artifacts\v370_v367_format_transfer_audit\20260514T_cpu_audit\v370_raw_output_examples.csv`

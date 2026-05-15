# V435 Adapter-Level Pair Gate Decision

Generated: 2026-05-15T14:01:47.887858+00:00

## Decision

- `hf_gpu_allowed`: `false`
- decision: `v435_pair_gate_blocks_gpu`
- next action: Do not launch GPU. Generate V291/V290 raw outputs on permitted train/synthetic rows or add certified adapter-level hard negatives.

## Baseline

| Metric | Value |
|---|---:|
| total weak | 192/315 |
| equation_transform | 56/155 |
| bit_manipulation | 136/160 |
| truncated | 0 |

## Pair Audit

- candidate pairs: `3558`
- approved pairs: `0`
- approved equation rule modes: `0`
- top blocking reasons: `{"missing_adapter_decode_config": 3558, "missing_adapter_identity": 3558, "missing_adapter_raw_output": 3558, "missing_locked_before_answer_audit": 3558, "missing_loo_certificate": 3558, "missing_mdl_certificate": 3558, "missing_renaming_certificate": 3558, "not_hard_negative": 2700, "rejected_box_count_0": 900, "rejected_box_count_2": 900, "same_chosen_rejected_box": 900}`

## Blocking Conditions

- `approved_equation_rule_modes_ge_4`
- `approved_pairs_gt_0`
- `programmatic_bit_guardrail_ready`

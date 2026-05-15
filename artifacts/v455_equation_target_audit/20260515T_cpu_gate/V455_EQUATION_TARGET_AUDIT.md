# V455 Equation Target Audit

Generated: 2026-05-15T21:39:18.914534+00:00

## Result

| Item | Value |
|---|---:|
| Equation misses audited by V324 | `99` |
| Numeric misses | `16` |
| Symbolic/punctuation misses | `83` |
| V324 accepted no-loss candidates | `6` |
| V452 certified trainable pairs | `2` |
| Verified rows missing from builder | `4` |
| Missing verified classes | `3` |
| Symbolic verified candidates | `0` |
| `hf_gpu_allowed` | `false` |

## Class Gap

| Rule class | V324 verified | V452 promoted | Gap | Status |
|---|---:|---:|---:|---|
| `v274_guarded_numeric_add_direct_over_model_add_variant` | `1` | `0` | `1` | missing_builder_target |
| `v274_guarded_numeric_colon_absdiff_restore_trailing_zero` | `1` | `0` | `1` | missing_builder_target |
| `v274_guarded_numeric_minus_direct_negative_restore_sign` | `2` | `2` | `0` | covered |
| `v274_guarded_numeric_minus_signed_opposite_sign_guarded` | `2` | `0` | `2` | missing_builder_target |

## Decision

v324_accepted=6; v452_pairs=2; missing_verified=4; missing_classes=3; symbolic_verified=0

Implement V456 builder for the missing V324 verified numeric classes before any HF GPU.

No HF GPU is allowed from this audit until V456 closes the verified-class gap.

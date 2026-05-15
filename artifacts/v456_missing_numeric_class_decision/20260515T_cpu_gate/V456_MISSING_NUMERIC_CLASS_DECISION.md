# V456 Missing Numeric Class Decision

Generated: 2026-05-15T21:52:46.763970+00:00

## Result

| Item | Value |
|---|---:|
| Missing classes audited | `3` |
| Eligible for prepaid gate | `0` |
| Blocked by prior synthetic transfer failure | `2` |
| Needs public-train raw probe | `1` |
| Needs new builder | `0` |
| `hf_gpu_allowed` | `false` |

## Class Decisions

| Rule | Gap | Public promoted | Prior synthetic count | Builder | Action |
|---|---:|---:|---:|---|---|
| `add_direct_over_model_add_variant` | `1` | `0` | `6160` | `true` | blocked_prior_synthetic_transfer_failed |
| `colon_absdiff_restore_trailing_zero` | `1` | `0` | `0` | `true` | needs_public_train_raw_probe_before_gpu |
| `minus_signed_opposite_sign_guarded` | `2` | `0` | `6160` | `true` | blocked_prior_synthetic_transfer_failed |

## Decision

eligible=0; synthetic_failed=2; needs_probe=1; needs_builder=0

Build V457 public-train numeric raw-output probe pack for the missing classes; only train if it yields legal adapter-wrong pairs with zero leakage.

Do not open HF GPU from V456. The missing classes still need legal public-train adapter-wrong evidence.

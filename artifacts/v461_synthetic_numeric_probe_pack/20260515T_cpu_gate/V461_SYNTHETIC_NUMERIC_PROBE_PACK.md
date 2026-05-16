# V461 Synthetic Numeric Probe Pack

Generated: 2026-05-15T22:35:00.363347+00:00

| Item | Value |
|---|---:|
| Prompt rows | `56` |
| Rule classes | `4` |
| `hf_raw_probe_allowed` | `False` |
| `hf_gpu_train_allowed` | `False` |

V474 update: this route is quarantined and must not be used for new raw probes
or downstream training. Rebuild a clean V475+ pack if this idea is revisited.

## Rule Counts

| Rule class | Rows |
|---|---:|
| `v274_guarded_numeric_add_direct_over_model_add_variant` | `16` |
| `v274_guarded_numeric_colon_absdiff_restore_trailing_zero` | `16` |
| `v274_guarded_numeric_minus_direct_negative_restore_sign` | `8` |
| `v274_guarded_numeric_minus_signed_opposite_sign_guarded` | `16` |

Answers are present only in the local audit CSV. The prompt JSONL used by HF raw inference has no label-like fields.

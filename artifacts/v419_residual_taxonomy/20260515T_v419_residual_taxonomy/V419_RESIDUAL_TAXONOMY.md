# V419 Residual Taxonomy

Generated: 2026-05-15T03:53:09.146766+00:00

V419 analyzes the equation rows still unsolved after the best current CPU projection from V409/V412/V418.

| Metric | Value |
|---|---:|
| Equation residual rows after V418 | `92` |
| False positives blocked in V418 | `1` |
| Conflicts/losses blocked in V418 | `8` |

## Top Residual Buckets

| Bucket | Count |
|---|---:|
| punct_only::symbolic_punctuation_prompt::no_consistent_vsa_program | `80` |
| numeric_unsigned::numeric_operator_prompt::no_consistent_vsa_program | `9` |
| mixed_symbolic::numeric_operator_prompt::no_consistent_vsa_program | `1` |
| punct_only::symbolic_punctuation_prompt::ambiguous_near_top_vsa_predictions | `1` |
| punct_only::symbolic_punctuation_prompt::v412_vsa_ranked_unique_prediction | `1` |

## Decision

`hf_gpu_allowed = false`.

The next useful work is a new symbolic punctuation structural solver gate. Re-running V412 with wider caps already produced `0` new safe gains.

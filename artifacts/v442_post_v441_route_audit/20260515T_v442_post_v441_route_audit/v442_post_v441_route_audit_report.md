# V442 Post-V441 Route Audit

## Decision

- Decision: `same_preference_route_blocked_return_to_cpu_certified_builder`
- Reason: V441 tied baseline under boxed-payload scoring and V439 pairs have zero rule-certified rows.
- Next action: Implement CPU certified equation pair builder; do not launch another V435E/V439 preference GPU job.

## What V441 Proved

| Metric | Baseline | Checkpoint-3 | Delta |
|---|---:|---:|---:|
| Preference total | 7 | 7 | 0 |
| equation_transform | 6 | 6 | 0 |
| bit_manipulation | 1 | 1 | 0 |

V441 was technically healthy but did not move the validation signal. This blocks another
GPU relaunch on the same V439/V435E preference family.

## Pair Certification Audit

| Item | Value |
|---|---:|
| audited rows | 133 |
| existing source-ok rows | 133 |
| rule-certified rows | 0 |
| weak/full training rows | 0 |

The source is clean enough for diagnostics, but not enough for another paid ranking job.
The missing piece is a label-free rule certificate, not another loss variant.

## Residual Evidence

V419 residual taxonomy says the remaining hard equation work is mostly symbolic punctuation:

| Bucket | Count |
|---|---:|
| `punct_only::symbolic_punctuation_prompt::no_consistent_vsa_program` | 80 |
| `numeric_unsigned::numeric_operator_prompt::no_consistent_vsa_program` | 9 |
| `mixed_symbolic::numeric_operator_prompt::no_consistent_vsa_program` | 1 |
| `punct_only::symbolic_punctuation_prompt::ambiguous_near_top_vsa_predictions` | 1 |
| `punct_only::symbolic_punctuation_prompt::v412_vsa_ranked_unique_prediction` | 1 |

V433 found correct answers only inside ambiguous sets, not unique label-free gains:

| Metric | Value |
|---|---:|
| accepted new gains | 0 |
| ambiguous correct candidate rows | 4 |
| projected total | 192 |
| projected equation | 56 |
| projected bit | 136 |

## Active Implementation Order

1. Build a true CPU certified pair builder for `equation_symbolic_sequence` and `equation_symbolic_short`.
2. Freeze each candidate rule before looking at the public-train answer.
3. Require MDL, leave-one-out, renaming stability, and unique candidate count.
4. Only after at least four independent equation modes pass, regenerate preference rows.
5. Only then consider HF GPU; otherwise remain CPU-only.

## Outputs

- Pair audit CSV: `artifacts/v442_post_v441_route_audit/20260515T_v442_post_v441_route_audit/v442_post_v441_route_audit_pair_certification_audit.csv`
- Manifest: `artifacts/v442_post_v441_route_audit/20260515T_v442_post_v441_route_audit/v442_post_v441_route_audit_manifest.json`

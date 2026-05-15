# V432 V431 Label-Free Tiebreak Gate

Generated: `2026-05-15T11:36:42.978311+00:00`

CPU-only audit for whether V431 ambiguous/candidate rows can be promoted without using labels.

## Comparison

| Metric | Baseline V291/V290 | V431 | V432 label-free |
|---|---:|---:|---:|
| Total weak correct | `192/315` | `193/315` | `193/315` |
| equation_transform | `56/155` | `57/155` | `57/155` |
| bit_manipulation | `136/160` | `136/160` | `136/160` |
| Truncated | `0` | `0` | `0` |

## Tiebreak Result

- Audited V431 non-abstain rows: `6`.
- Label-free promotable new gains: `0`.
- Rows blocked by multiple candidates: `4`.
- False unique candidates blocked by verification: `1`.
- `hf_gpu_allowed = false`.

## Rows

| id | V431 status | candidates | answer | policy | decision |
|---|---|---|---|---|---|
| `99d6a3b5` | `accepted` | `?()<` | `?()<` | `unique_candidate_only` | `promotable_but_already_known_v414_not_new_signal` |
| `02902eb3` | `ambiguous` | `-]|/]&}|>}|]` | `>/` | `blocked_multiple_candidates` | `multiple_predictions_no_label_free_tiebreak` |
| `c43b5a13` | `candidate` | `/<` | `)|` | `unique_candidate_only` | `unique_candidate_false_positive` |
| `6cc5dafb` | `ambiguous` | `"#>#|%%)(|%%>>|%)(|%>>|)(|-%%>>|-%>>` | `)(` | `blocked_multiple_candidates` | `multiple_predictions_no_label_free_tiebreak` |
| `194695e8` | `ambiguous` | `%%{<|-%%{<|-{<|{<` | `%` | `blocked_multiple_candidates` | `multiple_predictions_no_label_free_tiebreak` |
| `5501c054` | `ambiguous` | `&&[|&[|-&&[|-&[|-[|[|[#>#|^)"` | `[#>#` | `blocked_multiple_candidates` | `multiple_predictions_no_label_free_tiebreak` |

## Decision

V431 does not create a new submit-safe signal. The only label-free unique correct row is already known by V414; ambiguous rows with correct candidates require a row-specific choice that is not available at submission time.

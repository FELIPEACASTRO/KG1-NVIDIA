# V513 Trace Learnability Gate

- Generated UTC: `2026-05-16T21:32:48.858884+00:00`
- Decision: `blocked_no_gpu`
- Reason: blockers found; do not launch paid GPU from this dataset
- Findings: blockers `2`, warnings `0`, info `1`
- Train rows: `2627`; validation rows: `637`

## Family And Style

| Split | Families | Assistant styles |
|---|---:|---:|
| train | `{"bit_manipulation": 609, "equation_transform": 2018}` | `{"bit_answer_only_boxed": 609, "equation_short_rule_reject_boxed": 2018}` |
| validation | `{"bit_manipulation": 133, "equation_transform": 504}` | `{"bit_answer_only_boxed": 133, "equation_short_rule_reject_boxed": 504}` |

## Lengths

| Split | Length summary |
|---|---|
| train | `{"bit_manipulation": {"assistant_line_max": 1, "assistant_line_p50": 1, "assistant_word_max": 3, "assistant_word_min": 3, "assistant_word_p50": 3, "assistant_word_p90": 3, "assistant_word_p99": 3}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 4, "assistant_word_max": 55, "assistant_word_min": 22, "assistant_word_p50": 31, "assistant_word_p90": 54, "assistant_word_p99": 55}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 1, "assistant_line_p50": 1, "assistant_word_max": 3, "assistant_word_min": 3, "assistant_word_p50": 3, "assistant_word_p90": 3, "assistant_word_p99": 3}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 4, "assistant_word_max": 55, "assistant_word_min": 22, "assistant_word_p50": 31, "assistant_word_p90": 54, "assistant_word_p99": 55}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 742 | 234 | `bit_manipulation` | `bit_guardrail_replay` | final answer: \boxed{<ans>} |
| 500 | 161 | `equation_transform` | `equation_numeric_minus_signed_hard_negative` | rule: use the query operator and preserve the signed left-minus-right result. reject common wrong candidate <num>; it strips or flips the sign. query <num><num> |
| 500 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero_hard_negative` | rule: for ':' compute the absolute difference and preserve the full decimal digits. reject common wrong candidate <num>; it drops a required trailing zero. quer |
| 250 | 132 | `equation_transform` | `equation_numeric_add_direct_hard_negative` | rule: use examples with the same additive query operator; ignore distractor operators. reject common wrong candidate <num>; it follows a distractor subtraction  |
| 250 | 129 | `equation_transform` | `equation_numeric_add_direct_hard_negative` | rule: use examples with the same additive query operator; ignore distractor operators. reject common wrong candidate <num>; it follows a distractor subtraction  |
| 200 | 112 | `equation_transform` | `equation_numeric_minus_signed` | rule: for this '-' operator, compute left minus right and preserve the sign. check the examples that use this operator: - <num><num> -> <num> - <num><num> -> <n |
| 200 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero` | rule: for this ':' operator, compute the absolute difference and keep any trailing zero. check the examples that use this operator: - <num>:<num> -> <num> - <nu |
| 200 | 78 | `equation_transform` | `equation_numeric_colon_absdiff` | rule: for this ':' operator, compute the absolute difference and keep the natural digit order. check the examples that use this operator: - <num>:<num> -> <num> |
| 200 | 72 | `equation_transform` | `equation_numeric_minus_direct_negative` | rule: for this '-' operator, compute left minus right; if the result is negative, keep the minus sign. check the examples that use this operator: - <num><num> - |
| 100 | 76 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>)<num> -> <num> - <num>)<num> ->  |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

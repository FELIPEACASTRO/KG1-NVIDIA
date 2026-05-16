# V513 Trace Learnability Gate

- Generated UTC: `2026-05-16T22:13:04.098643+00:00`
- Decision: `passed_cpu_structure_only`
- Reason: no structural blockers; still requires objective and FinOps gates before GPU
- Findings: blockers `0`, warnings `0`, info `1`
- Train rows: `2491`; validation rows: `620`

## Family And Style

| Split | Families | Assistant styles |
|---|---:|---:|
| train | `{"bit_manipulation": 473, "equation_transform": 2018}` | `{"bit_trace_with_rule_terms": 398, "boxed_other": 75, "equation_short_rule_reject_boxed": 2018}` |
| validation | `{"bit_manipulation": 116, "equation_transform": 504}` | `{"bit_trace_with_rule_terms": 95, "boxed_other": 21, "equation_short_rule_reject_boxed": 504}` |

## Lengths

| Split | Length summary |
|---|---|
| train | `{"bit_manipulation": {"assistant_line_max": 10, "assistant_line_p50": 10, "assistant_word_max": 73, "assistant_word_min": 8, "assistant_word_p50": 47, "assistant_word_p90": 52, "assistant_word_p99": 73}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 4, "assistant_word_max": 55, "assistant_word_min": 22, "assistant_word_p50": 31, "assistant_word_p90": 54, "assistant_word_p99": 55}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 10, "assistant_line_p50": 10, "assistant_word_max": 73, "assistant_word_min": 8, "assistant_word_p50": 47, "assistant_word_p90": 51, "assistant_word_p99": 73}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 4, "assistant_word_max": 55, "assistant_word_min": 22, "assistant_word_p50": 31, "assistant_word_p90": 54, "assistant_word_p99": 55}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 500 | 161 | `equation_transform` | `equation_numeric_minus_signed_hard_negative` | rule: use the query operator and preserve the signed left-minus-right result. reject common wrong candidate <num>; it strips or flips the sign. query <num><num> |
| 500 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero_hard_negative` | rule: for ':' compute the absolute difference and preserve the full decimal digits. reject common wrong candidate <num>; it drops a required trailing zero. quer |
| 250 | 132 | `equation_transform` | `equation_numeric_add_direct_hard_negative` | rule: use examples with the same additive query operator; ignore distractor operators. reject common wrong candidate <num>; it follows a distractor subtraction  |
| 250 | 129 | `equation_transform` | `equation_numeric_add_direct_hard_negative` | rule: use examples with the same additive query operator; ignore distractor operators. reject common wrong candidate <num>; it follows a distractor subtraction  |
| 200 | 112 | `equation_transform` | `equation_numeric_minus_signed` | rule: for this '-' operator, compute left minus right and preserve the sign. check the examples that use this operator: - <num><num> -> <num> - <num><num> -> <n |
| 200 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero` | rule: for this ':' operator, compute the absolute difference and keep any trailing zero. check the examples that use this operator: - <num>:<num> -> <num> - <nu |
| 200 | 78 | `equation_transform` | `equation_numeric_colon_absdiff` | rule: for this ':' operator, compute the absolute difference and keep the natural digit order. check the examples that use this operator: - <num>:<num> -> <num> |
| 200 | 72 | `equation_transform` | `equation_numeric_minus_direct_negative` | rule: for this '-' operator, compute left minus right; if the result is negative, keep the minus sign. check the examples that use this operator: - <num><num> - |
| 100 | 76 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>)<num> -> <num> - <num>)<num> ->  |
| 100 | 71 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>+<num> -> <num> - <num>+<num> ->  |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

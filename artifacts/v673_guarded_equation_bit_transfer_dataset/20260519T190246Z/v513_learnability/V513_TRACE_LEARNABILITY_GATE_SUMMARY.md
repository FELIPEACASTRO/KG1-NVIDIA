# V513 Trace Learnability Gate

- Generated UTC: `2026-05-19T19:04:20.799767+00:00`
- Decision: `passed_cpu_structure_only`
- Reason: no structural blockers; still requires objective and FinOps gates before GPU
- Findings: blockers `0`, warnings `0`, info `2`
- Train rows: `720`; validation rows: `180`

## Family And Style

| Split | Families | Assistant styles |
|---|---:|---:|
| train | `{"bit_manipulation": 240, "equation_transform": 480}` | `{"bit_trace_with_rule_terms": 240, "equation_short_rule_reject_boxed": 480}` |
| validation | `{"bit_manipulation": 60, "equation_transform": 120}` | `{"bit_trace_with_rule_terms": 60, "equation_short_rule_reject_boxed": 120}` |

## Lengths

| Split | Length summary |
|---|---|
| train | `{"bit_manipulation": {"assistant_line_max": 5, "assistant_line_p50": 5, "assistant_word_max": 26, "assistant_word_min": 26, "assistant_word_p50": 26, "assistant_word_p90": 26, "assistant_word_p99": 26}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 10, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 54, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 5, "assistant_line_p50": 5, "assistant_word_max": 26, "assistant_word_min": 26, "assistant_word_p50": 26, "assistant_word_p90": 26, "assistant_word_p99": 26}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 10, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 54, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 300 | 136 | `equation_transform` | `equation_numeric_minus_signed` | rule: for this '-' operator, compute left minus right and preserve the sign. check the examples that use this operator: - <num><num> -> <num> - <num><num> -> <n |
| 150 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero` | rule: for this ':' operator, compute the absolute difference and keep any trailing zero. check the examples that use this operator: - <num>:<num> -> <num> - <nu |
| 75 | 58 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>)<num> -> <num> - <num>)<num> ->  |
| 75 | 63 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>+<num> -> <num> - <num>+<num> ->  |
| 3 | 1 | `bit_manipulation` | `bit_exact_global_binary_replay` | rule: apply the <num>-bit operation or(rol2,shl4). rule class: bit_exact_global_binary_or. query byte code: qcg. use the same bit transformation shown by the pr |
| 2 | 1 | `bit_manipulation` | `bit_fullbyte_ternary_v366_new` | rule: apply the <num>-bit operation maj3(rol2,shl1,shr5). rule class: bit_fullbyte_ternary_op_maj3. query byte code: qkd. use the same bit transformation shown  |
| 2 | 1 | `bit_manipulation` | `bit_exact_global_binary_replay` | rule: apply the <num>-bit operation or(rol2,shl4). rule class: bit_exact_global_binary_or. query byte code: qgi. use the same bit transformation shown by the pr |
| 2 | 1 | `bit_manipulation` | `bit_exact_global_binary_replay` | rule: apply the <num>-bit operation xor(shl1,shr4). rule class: bit_exact_global_binary_xor. query byte code: qme. use the same bit transformation shown by the  |
| 2 | 1 | `bit_manipulation` | `bit_exact_global_binary_replay` | rule: apply the <num>-bit operation or(rol2,shl4). rule class: bit_exact_global_binary_or. query byte code: qci. use the same bit transformation shown by the pr |
| 2 | 1 | `bit_manipulation` | `bit_exact_global_ternary_replay` | rule: apply the <num>-bit operation xnor(rol7,or(rol1,shr1)). rule class: bit_exact_global_ternary_unique_prediction. query byte code: qei. use the same bit tra |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

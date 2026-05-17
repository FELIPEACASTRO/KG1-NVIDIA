# V513 Trace Learnability Gate

- Generated UTC: `2026-05-16T23:59:59.380702+00:00`
- Decision: `passed_cpu_structure_only`
- Reason: no structural blockers; still requires objective and FinOps gates before GPU
- Findings: blockers `0`, warnings `0`, info `1`
- Train rows: `1026`; validation rows: `219`

## Family And Style

| Split | Families | Assistant styles |
|---|---:|---:|
| train | `{"bit_manipulation": 706, "equation_transform": 320}` | `{"bit_trace_with_rule_terms": 706, "equation_short_rule_reject_boxed": 320}` |
| validation | `{"bit_manipulation": 139, "equation_transform": 80}` | `{"bit_trace_with_rule_terms": 139, "equation_short_rule_reject_boxed": 80}` |

## Lengths

| Split | Length summary |
|---|---|
| train | `{"bit_manipulation": {"assistant_line_max": 27, "assistant_line_p50": 27, "assistant_word_max": 106, "assistant_word_min": 97, "assistant_word_p50": 99, "assistant_word_p90": 106, "assistant_word_p99": 106}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 9, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 52, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 27, "assistant_line_p50": 27, "assistant_word_max": 106, "assistant_word_min": 97, "assistant_word_p50": 99, "assistant_word_p90": 106, "assistant_word_p99": 106}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 9, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 52, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 104 | 74 | `bit_manipulation` | `bit_v300_gain_pattern_other` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: xor(shl1,shr4); xor(a,b)=a^b. example verification: - <bin8> -> <bin8> ok  |
| 100 | 72 | `equation_transform` | `equation_numeric_minus_signed` | rule: for this '-' operator, compute left minus right and preserve the sign. check the examples that use this operator: - <num><num> -> <num> - <num><num> -> <n |
| 100 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero` | rule: for this ':' operator, compute the absolute difference and keep any trailing zero. check the examples that use this operator: - <num>:<num> -> <num> - <nu |
| 100 | 60 | `equation_transform` | `equation_numeric_colon_absdiff` | rule: for this ':' operator, compute the absolute difference and keep the natural digit order. check the examples that use this operator: - <num>:<num> -> <num> |
| 59 | 42 | `bit_manipulation` | `bit_cho_trace` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: cho(shl2,shr4,rol7); cho(a,b,c)=(a&b)\|((<num>-a)&c). example verification |
| 58 | 39 | `bit_manipulation` | `bit_cho_trace` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: cho(shl1,shr1,rol4); cho(a,b,c)=(a&b)\|((<num>-a)&c). example verification |
| 55 | 47 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>+<num> -> <num> - <num>+<num> ->  |
| 54 | 38 | `bit_manipulation` | `bit_cho_trace` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: cho(shl2,shr1,rol3); cho(a,b,c)=(a&b)\|((<num>-a)&c). example verification |
| 51 | 35 | `bit_manipulation` | `bit_cho_trace` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: cho(shl2,shr3,rol1); cho(a,b,c)=(a&b)\|((<num>-a)&c). example verification |
| 46 | 35 | `bit_manipulation` | `bit_maj3_trace` | rule: solve output bits independently, then concatenate b0..b7. candidate expression: maj3(rol6,shl1,shr1); maj3(a,b,c)=<num> when at least two inputs are <num> |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

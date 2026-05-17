# V513 Trace Learnability Gate

- Generated UTC: `2026-05-17T02:48:34.492121+00:00`
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
| train | `{"bit_manipulation": {"assistant_line_max": 2, "assistant_line_p50": 2, "assistant_word_max": 283, "assistant_word_min": 74, "assistant_word_p50": 76, "assistant_word_p90": 268, "assistant_word_p99": 277}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 9, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 52, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 2, "assistant_line_p50": 2, "assistant_word_max": 280, "assistant_word_min": 74, "assistant_word_p50": 76, "assistant_word_p90": 268, "assistant_word_p99": 275}, "equation_transform": {"assistant_line_max": 10, "assistant_line_p50": 9, "assistant_word_max": 54, "assistant_word_min": 46, "assistant_word_p50": 52, "assistant_word_p90": 54, "assistant_word_p99": 54}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 100 | 72 | `equation_transform` | `equation_numeric_minus_signed` | rule: for this '-' operator, compute left minus right and preserve the sign. check the examples that use this operator: - <num><num> -> <num> - <num><num> -> <n |
| 100 | 9 | `equation_transform` | `equation_numeric_colon_trailing_zero` | rule: for this ':' operator, compute the absolute difference and keep any trailing zero. check the examples that use this operator: - <num>:<num> -> <num> - <nu |
| 100 | 60 | `equation_transform` | `equation_numeric_colon_absdiff` | rule: for this ':' operator, compute the absolute difference and keep the natural digit order. check the examples that use this operator: - <num>:<num> -> <num> |
| 55 | 47 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>+<num> -> <num> - <num>+<num> ->  |
| 45 | 40 | `equation_transform` | `equation_numeric_add_direct` | rule: for this additive operator, compute the direct sum of the two numbers. check the examples that use this operator: - <num>)<num> -> <num> - <num>)<num> ->  |
| 22 | 22 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 22 | 19 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 14 | 14 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 13 | 13 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_rot, confidence=high, ambiguous_bits= |
| 13 | 12 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

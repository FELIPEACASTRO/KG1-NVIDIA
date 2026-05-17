# V513 Trace Learnability Gate

- Generated UTC: `2026-05-17T02:44:54.445917+00:00`
- Decision: `passed_cpu_structure_only`
- Reason: no structural blockers; still requires objective and FinOps gates before GPU
- Findings: blockers `0`, warnings `0`, info `1`
- Train rows: `1541`; validation rows: `255`

## Family And Style

| Split | Families | Assistant styles |
|---|---:|---:|
| train | `{"bit_manipulation": 1541}` | `{"bit_trace_with_rule_terms": 1541}` |
| validation | `{"bit_manipulation": 255}` | `{"bit_trace_with_rule_terms": 255}` |

## Lengths

| Split | Length summary |
|---|---|
| train | `{"bit_manipulation": {"assistant_line_max": 2, "assistant_line_p50": 2, "assistant_word_max": 283, "assistant_word_min": 74, "assistant_word_p50": 76, "assistant_word_p90": 268, "assistant_word_p99": 277}}` |
| validation | `{"bit_manipulation": {"assistant_line_max": 2, "assistant_line_p50": 2, "assistant_word_max": 286, "assistant_word_min": 74, "assistant_word_p50": 76, "assistant_word_p90": 270, "assistant_word_p99": 276}}` |

## Top Template Groups

| Count | Answers | Families | Subcategories | Preview |
|---:|---:|---|---|---|
| 52 | 46 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 46 | 42 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 34 | 32 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 32 | 29 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_mix, confidence=high, ambiguous_bits= |
| 20 | 20 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_rot, confidence=high, ambiguous_bits= |
| 18 | 18 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_rot, confidence=high, ambiguous_bits= |
| 18 | 18 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_rot, confidence=high, ambiguous_bits= |
| 12 | 12 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_rot, confidence=high, ambiguous_bits= |
| 9 | 7 | `bit_manipulation` | `bit_konbu_high_confidence_trace` | trace summary: infer the <num>-bit transformation from examples, then test candidate bit relations. source solver method=w_uni2, confidence=high, ambiguous_bits |
| 8 | 8 | `bit_manipulation` | `bit_huikang_synthetic_cho` | trace summary: infer a per-bit <num>-input transformation and verify it on all examples. rule: cho(not rot(<num>), shr(<num>), rot(<num>)). operator family: cho |

## Gate Meaning

- `blocked_no_gpu`: no HF GPU train should be launched from this dataset as-is.
- `passed_cpu_structure_only`: this still is not submit permission; it only authorizes a tiny paid smoke if other gates pass.
- V513 is CPU-only and does not package or submit.

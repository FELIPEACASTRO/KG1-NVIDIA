# V401 Baseline Raw Output Audit

Generated UTC: `2026-05-14T23:51:00.834098+00:00`

## Summary

| Type | Miss rows | Answer in raw | Answer in simple boxed | Truncated |
|---|---:|---:|---:|---:|
| `bit_manipulation` | 24 | 4 | 0 | 0 |
| `equation_transform` | 99 | 19 | 0 | 0 |
| `OVERALL` | 123 | 23 | 0 | 0 |

## Decision

- `decision`: `no_extractor_gain_adapter_generation_is_bottleneck`
- `actionable`: `False`

The `answer in raw` counts are diagnostic only. Manual spot-check shows these are not safe extractor wins: answers appear as characters/numbers inside reasoning text or intermediate examples, not as final simple boxed answers. If the correct answer is not already present in simple boxed output, this cannot be fixed by extractor changes.

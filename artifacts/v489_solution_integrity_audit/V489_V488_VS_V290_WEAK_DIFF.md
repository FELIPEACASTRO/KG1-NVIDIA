# V489 V488 vs V290 Weak Diff

| Metric | V290/V291 baseline | V488 ckpt-10 | Delta |
|---|---:|---:|---:|
| Total strict correct | 192/315 | 191/315 | -1 |
| equation_transform | 56/155 | 57/155 | +1 |
| bit_manipulation | 136/160 | 134/160 | -2 |
| Truncated rows | 0 | 1 | +1 |

## Row-Level Changes

- `equation_transform`: gains=1, regressions=0, both_correct=56, both_wrong=98, v488_truncated=0
- `bit_manipulation`: gains=0, regressions=2, both_correct=134, both_wrong=24, v488_truncated=1

## Decision

V488 is blocked. The +1 equation gain is outweighed by -2 bit and one new truncation. This confirms the current bottleneck is objective/format transfer, not only PEFT target-parameter continuity.

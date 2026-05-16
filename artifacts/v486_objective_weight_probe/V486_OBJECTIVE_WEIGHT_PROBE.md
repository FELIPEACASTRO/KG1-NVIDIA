# V486 Objective Weight Probe

Generated: 2026-05-16

## Summary

The failed V391 H200 launch proved that the previous `10.00` equation
subcategory weights were too aggressive. The V478 gate rejected the job before
training, which avoided paid GPU waste.

Local CPU objective probes on the exact V390/V475 train/validation JSONL files
found that `6.00` is the most aggressive equation subcategory weight that still
passes the train objective-balance gate.

## Train Effective Shares

| Equation subcategory weight | Bit share | Equation share | V478 train status |
|---:|---:|---:|---|
| 10.00 | 0.135975 | 0.864025 | reject |
| 8.00 | 0.164380 | 0.835620 | reject |
| 6.00 | 0.207788 | 0.792212 | pass |
| 4.00 | 0.282348 | 0.717652 | pass |
| 3.00 | 0.344081 | 0.655919 | pass |

## Decision

Use `6.00` for V486 because it keeps equation emphasis high while restoring bit
guardrail pressure above the `0.20` train effective-share floor.

The validation split remains equation-heavy by design, so V486 must still be
judged by weak micro-ACC and not by `eval_loss` alone.

# V516 Label-Free Equation Gate Summary

- Generated UTC: `2026-05-16T22:32Z`
- Input CSV: `artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv`
- Output CSV: `v516_label_free_v290_checkpoint6_baseline.csv`
- Submit/package/train: not run

## Baseline Correction

The V290 checkpoint-6 weak baseline must be treated as label-free:

| Metric | Stored | Label-free |
|---|---:|---:|
| Total weak | `192/315` | `191/315` |
| equation_transform | `56/155` | `55/155` |
| bit_manipulation | `136/160` | `136/160` |
| truncated | `0` | `0` |

The row contract matches the active weak contract:

`bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`

## Equation Gate Result

Running the expanded equation CPU gate on the label-free baseline found:

| Metric | Value |
|---|---:|
| equation misses | `100` |
| numeric operator misses | `16` |
| symbolic punctuation misses | `84` |
| accepted no-loss candidates | `4` |
| conflicts | `0` |
| projected equation if transferred | `59/155` |
| projected weak if transferred | `195/315` |

Accepted IDs:

- `274def88`
- `7688e06e`
- `c5b058d6`
- `d1bd7478`

## Decision

This is concrete CPU evidence, but it is not new equation data. These four
candidate classes already exist in the V475/V510/V515 training pool. The
failure mode is therefore transfer/learnability, not missing equation coverage.

Any next paid smoke must be justified by a changed mechanism. For V515, the
changed mechanism is the traceable bit replacement plus the corrected bit
objective weight, not a new equation rule.

## Guard Patch

`scripts/run_v324_equation_expanded_solver_gate.py` now fails closed when an
input CSV contains `raw_output` but `prediction` is not the label-free
extraction from that raw output. A negative test against the old V342 CSV was
blocked as expected, forcing future runs to use this V516 label-free bridge
first.

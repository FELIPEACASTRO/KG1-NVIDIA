# V514 Traceable Bit V510 Dataset

- Generated UTC: `2026-05-16T21:41:58Z`
- Base dataset: `V510 canonical active training pool`
- Purpose: replace bit answer-only rows with deterministic traces only when the local solver or V296 stride solver exactly reproduces the row answer.
- Status: `CPU gates passed locally`; this is not a submit or GPU approval by itself.

## Dataset Delta

| Split | Input rows | Output rows | Equation kept | Bit seen | Bit converted to trace | Bit dropped unverified |
|---|---:|---:|---:|---:|---:|---:|
| train | `2627` | `2484` | `2018` | `609` | `466` | `143` |
| validation | `637` | `619` | `504` | `133` | `115` | `18` |

## Bit Trace Methods

| Split | `bit_solver_v4` | `v296_stride_solver` |
|---|---:|---:|
| train | `431` | `35` |
| validation | `111` | `4` |

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Real tokenizer gate | `passed` | `0` prompt truncation, `0` completion tokens dropped, offset masks `2484/619`, token max `553/541` |
| V513 trace learnability recheck | `passed_cpu_structure_only` | `0` blockers, `0` warnings; bit trace p50 `47` words |

## Decision

V514 fixes the V510 structural blocker found by V513: V510 had `742/742` bit rows as answer-only targets. V514 has `581/742` bit rows converted into short deterministic traces and drops `161` unverified bit rows.

The next step is a CPU reproduction on HF. A paid GPU smoke is still blocked until the HF CPU result matches the local gates and the objective/pre-paid gates are updated for V514.

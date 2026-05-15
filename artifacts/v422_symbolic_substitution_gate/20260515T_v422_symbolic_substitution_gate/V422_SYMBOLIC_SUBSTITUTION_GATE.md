# V422 Symbolic Substitution Gate

Generated: 2026-05-15T09:26:44.397944+00:00

| Metric | Value |
|---|---:|
| Candidate rows | `16` |
| Accepted gains | `0` |
| Conflicts/losses | `5` |
| Projected weak total | `192/315` |
| Projected equation_transform | `56/155` |
| Projected bit_manipulation | `136/160` |

Decision: `hf_gpu_blocked_no_safe_gain`.

This CPU gate tests a new structural class: selecting positions from the 5-char
Alice expression and learning global or slot-specific character substitutions
from in-prompt examples. Weak labels are used only for audit.

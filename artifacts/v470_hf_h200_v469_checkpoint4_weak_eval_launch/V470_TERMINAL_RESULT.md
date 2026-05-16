# V470 Terminal Result

Job: `https://huggingface.co/jobs/felipesp1983/6a07bbe43308d79117b90e3b`

The launch manifest was written while the HF job was still `RUNNING`. The
terminal artifact audit later confirmed the job completed and uploaded weak eval
outputs.

| Candidate | Total weak | equation_transform | bit_manipulation | truncated | Decision |
|---|---:|---:|---:|---:|---|
| V469 checkpoint-4 via V470 | `190/315` | `56/155` | `134/160` | `1` | reject |

Baseline submit-safe gate:

| Metric | Required to promote | V470 |
|---|---:|---:|
| Total weak | `>=193` | `190` |
| equation_transform | `>=57` | `56` |
| bit_manipulation | `>=136` | `134` |
| truncated | `0` | `1` |

Decision: no full eval, no package, no Kaggle submit, and no more H200 spend on
this route.

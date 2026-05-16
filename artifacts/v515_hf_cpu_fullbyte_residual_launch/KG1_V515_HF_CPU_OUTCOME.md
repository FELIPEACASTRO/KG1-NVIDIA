# V515 HF CPU Fullbyte Residual Outcome

- Generated UTC: `2026-05-16T22:24Z`
- Job: `https://huggingface.co/jobs/felipesp1983/6a08edcf3308d79117b9167f`
- Repo commit: `bf259e368de90634cbeec4b1d8175f1787ba9ccc`
- Hardware: HF `cpu-upgrade`
- Output dataset repo: `felipesp1983/kg1-v515-v514-fullbyte-residual-artifacts`
- Output dataset path: `v515-hf-cpu-fullbyte-residual-20260516T221957Z`

## Result

Status: completed.

The HF CPU run reproduced the V515 local build and completed both required CPU gates:

| Check | Result |
|---|---|
| V515 build | passed |
| Train rows | `2491` |
| Validation rows | `620` |
| Train residual bit accepted | `7` |
| Validation residual bit accepted | `1` |
| V286 tokenization gate | passed |
| Prompt truncation | `0.0` |
| Completion tokens dropped | `0` |
| Train offset masks | `2491/2491` |
| Validation offset masks | `620/620` |
| V513 trace learnability recheck | passed |
| V513 blockers | `0` |
| V513 warnings | `0` |

## Decision

No training, package, full eval, or Kaggle submit was run in this job.

V515 is now the cleanest active dataset candidate for a future tiny smoke, but it is only a verified coverage improvement. It is not yet a submit-safe ACC gain.

Next required step: run objective/pre-paid gates for any proposed GPU smoke. If those gates cannot prove a route to `equation>=60`, `bit>=136`, `trunc=0`, keep work CPU-only.

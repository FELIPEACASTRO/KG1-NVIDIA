# V514 HF CPU Traceable Bit Dataset Outcome

- Generated UTC: `2026-05-16T22:12Z`
- Job: `https://huggingface.co/jobs/felipesp1983/6a08e9ad3308d79117b91609`
- Repo commit: `76c0b3567dab6cbb3fa5117396d6bb1ee13ecb82`
- Hardware: HF `cpu-upgrade`
- Output dataset repo: `felipesp1983/kg1-v514-traceable-bit-v510-artifacts`
- Output dataset path: `v514-hf-cpu-traceable-bit-20260516T220219Z`

## Result

Status: completed.

The HF CPU run reproduced the V514 local build and completed both required CPU gates:

| Check | Result |
|---|---|
| V514 build | passed |
| Train rows | `2484` |
| Validation rows | `619` |
| Train bit converted to trace | `466` |
| Train bit dropped unverified | `143` |
| Validation bit converted to trace | `115` |
| Validation bit dropped unverified | `18` |
| V286 tokenization gate | passed |
| Prompt truncation | `0.0` |
| Completion tokens dropped | `0` |
| Train offset masks | `2484/2484` |
| Validation offset masks | `619/619` |
| V513 trace learnability recheck | passed |
| V513 blockers | `0` |
| V513 warnings | `0` |

## Decision

No training, package, full eval, or Kaggle submit was run in this job.

V514 is now reproduced on HF CPU, but it is still not a submit-safe gain. The next step is CPU-only V515:

1. Reprocess the `161` V514 dropped bit rows with the full-byte/global solver.
2. Accept only `fullbyte_unique_prediction` rows that reproduce the known training answer with no ambiguity.
3. Append only verified short traces to V514.
4. Rerun V286 tokenization and V513 trace learnability gates before any paid GPU.

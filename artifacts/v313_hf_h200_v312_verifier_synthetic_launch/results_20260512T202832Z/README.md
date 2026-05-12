# V313 V312 verifier-synthetic weak eval result

V313 attempted to distill the V306 verifier/postprocessor gains into adapter-only behavior using the V312 synthetic SFT traces.

HF training job: `https://huggingface.co/jobs/felipesp1983/6a037e3a7618f125ee2b7955`

HF weak eval job: `https://huggingface.co/jobs/felipesp1983/6a03853d72518a06598ff981`

Output repo: `felipesp1983/kg1-nemotron-lora-v313-v312-verifier-synthetic-smoke`

Output path: `evals/v313-h200-v221contract-v312-verifier-synth-20260512T195231Z`

## Weak V221 contract results

| Candidate | Total | equation_transform | bit_manipulation | Truncated | Decision |
|---|---:|---:|---:|---:|---|
| `v313_checkpoint_3_v221_contract` | 191/315 | 56/155 | 135/160 | 0 | Reject: below V290 protected baseline |
| `v313_checkpoint_6_v221_contract` | 190/315 | 56/155 | 134/160 | 1 | Reject |
| `v313_checkpoint_9_v221_contract` | 190/315 | 56/155 | 134/160 | 1 | Reject |
| `v313_checkpoint_12_v221_contract` | 190/315 | 56/155 | 134/160 | 1 | Reject |

Protected adapter-only baseline remains V290 checkpoint-6 / V291 full-package lineage:
`192/315`, `equation_transform=56/155`, `bit_manipulation=136/160` on weak, and full `823/947`.

## Decision

Do not run full eval, package, or Kaggle submit for V313. The useful lesson is that low eval loss around the `8.55e-09` learning-rate region is not enough; the only promotion signal is per-family ACC under the V221/full contracts.

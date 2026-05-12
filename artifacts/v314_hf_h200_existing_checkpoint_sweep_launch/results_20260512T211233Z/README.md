# V314 H200 Existing Checkpoint Sweep Results

Purpose: evaluate existing V308 early checkpoints before spending more GPU on new training.

HF job: `https://huggingface.co/jobs/felipesp1983/6a038f8972518a06598ff9fd`

Output repo: `felipesp1983/kg1-nemotron-lora-v308-v304-attn-lmhead-v290ckpt6`

Output path: `evals/v314-h200-v221contract-v308-early-ckpt-sweep-20260512T203626Z`

Weak V221 contract results:

| Candidate | Total | Equation | Bit | Trunc |
| --- | ---: | ---: | ---: | ---: |
| `v308_checkpoint_6_v221_contract` | 190/315 | 56/155 | 134/160 | 0 |
| `v308_checkpoint_12_v221_contract` | 190/315 | 56/155 | 134/160 | 1 |
| `v308_checkpoint_18_v221_contract` | 190/315 | 56/155 | 134/160 | 0 |
| `v308_checkpoint_24_v221_contract` | 191/315 | 56/155 | 135/160 | 0 |

Decision:

- Reject V314 for full eval, package, and Kaggle submit.
- Best V314 checkpoint is `checkpoint-24`, but it remains below the operational adapter-only baseline V290 checkpoint-6: `192/315`, `equation=56/155`, `bit=136/160`, truncation `0`.
- This sweep confirms that continuing the V308/V313-style SFT path with more steps/epochs is not the most effective next action.
- Next effective path: preference or hard-negative distillation from verifier/postprocessor signals, or additional CPU verifier expansion, with promotion gates based on family ACC rather than eval loss.

Local artifact policy:

- Only small JSON/CSV eval artifacts were downloaded.
- No adapter checkpoints or large HF cache files are stored here.

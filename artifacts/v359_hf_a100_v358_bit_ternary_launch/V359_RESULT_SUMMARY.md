# V359 HF Smoke Result Summary

Generated: 2026-05-14

## Status

V359 is rejected.

## Inputs

- Dataset: V358 V357 bit ternary transfer.
- Dataset HF upload: `https://huggingface.co/datasets/felipesp1983/kg1-nemotron-training/commit/9292641bd005d8b1a0e70445fe8d3fe1464d0232`
- Init adapter: `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`
- Train job: `https://huggingface.co/jobs/felipesp1983/6a0598743308d79117b8f539`
- Output repo: `felipesp1983/kg1-nemotron-lora-v359-nemo-a100-v358-bit-ternary-v290ckpt6`

## Training

- Hardware: HF `a100-large`.
- Image: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`.
- Steps: `4`.
- Learning rate: `6.0e-8 -> 1.5e-8`.
- Trainable LoRA params: `8,015,872`.
- Completed checkpoints: `checkpoint-2`, `checkpoint-4`, `final`.
- Baseline eval loss: `0.4051`.
- Step 2 eval loss: `0.4051`.

## Weak Eval

- Eval job: `https://huggingface.co/jobs/felipesp1983/6a059efce48bea4538b9c865`
- Evaluated candidate: `checkpoint-2`.
- Result: `190/315`.
- `equation_transform`: `56/155`.
- `bit_manipulation`: `134/160`.
- Truncated: `1`.

## Decision

Checkpoint-2 violated all promotion guardrails:

- `total<=192`;
- `bit<136`;
- truncation regressed from `0` to `1`.

The H200 weak eval was canceled before evaluating checkpoint-4/final. No full eval, package, or Kaggle submit is allowed from V359.

## Next

Run V360 post-failure audit and redesign transfer format before any new HF job.

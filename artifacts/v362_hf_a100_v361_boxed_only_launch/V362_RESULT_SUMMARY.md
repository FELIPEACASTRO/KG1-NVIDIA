# V362 Boxed-Only HF Smoke Result

## Decision

Rejected. V362 does not release full eval, packaging, Kaggle submit, or checkpoint-2/final weak eval.

## Evidence

- Train job HF A100: `https://huggingface.co/jobs/felipesp1983/6a05a6dd3308d79117b8f574`.
- Train output repo: `felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6`.
- Train status: canceled after useful checkpoints were uploaded, for FinOps.
- Uploaded checkpoints before cancellation: `checkpoint-1`, `checkpoint-2`.
- Weak eval HF H200 checkpoint-1: `https://huggingface.co/jobs/felipesp1983/6a05aa47e48bea4538b9c8dc`.
- Weak eval output commit: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v362-nemo-a100-v361-boxed-only-v290ckpt6/commit/ed701d36d47ad2d80694d552b777a65b70e49a88`.
- Shared row contract: `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.

## Weak Results

| Candidate | Overall | Equation | Bit | Truncated |
|---|---:|---:|---:|---:|
| baseline adapter-only gate | `192/315` | `56/155` | `136/160` | `0` |
| V362 checkpoint-1 | `190/315` | `56/155` | `134/160` | `1` |

V362 checkpoint-1 regressed total by `-2`, bit by `-2`, and introduced truncation. Equation stayed flat.

## FinOps Action

Checkpoint-1 failed the gate, so checkpoint-2/final weak eval is blocked. The A100 train job was canceled after checkpoint-2 existed because final eval/upload was not needed for promotion.

## Local Evidence

- Eval manifest: `artifacts/v362_hf_a100_v361_boxed_only_launch/eval_cp1/cp1_weak_eval_manifest.json`.
- Batch summary: `artifacts/v362_hf_a100_v361_boxed_only_launch/eval_cp1/cp1_batch_candidate_summary.json`.
- Per-task summary: `artifacts/v362_hf_a100_v361_boxed_only_launch/eval_cp1/cp1_per_task.csv`.

## Next Action

Do not spend more GPU on V361/V362 boxed-only bit transfer. Return to CPU-only work:

1. Equation DSL/verifier expansion against the 99 equation misses.
2. Bit algorithm coverage expansion before any new SFT.
3. Only launch HF again if CPU gate proves new no-loss gains.

# V290 H200 Weak Eval Result

Run: `v290-h200-v221contract-rank19-micro-patch-20260511T200448Z`

HF eval job: `https://huggingface.co/jobs/felipesp1983/6a02369d317220dbbd1a7c03`

HF output commit: `https://huggingface.co/felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/commit/fee1eb26009eda5d9347365e661187616e62c6cf`

Output path: `evals/v290-h200-v221contract-rank19-micro-patch-20260511T200448Z`

## Gate

Required weak gate:

- Total `>=193/315`
- `equation_transform >=60/155`
- `bit_manipulation >=136/160`
- Truncation low enough for promotion

## Results

| Candidate | Total | Equation | Bit | Trunc | Decision |
|---|---:|---:|---:|---:|---|
| `v290_checkpoint_3_v221_contract` | 190 | 56 | 134 | 1 | Reject |
| `v290_checkpoint_6_v221_contract` | 192 | 56 | 136 | 0 | Reject for full/package; keep evidence |
| `v290_final_v221_contract` | 191 | 56 | 135 | 0 | Reject |

## Decision

V290 did not pass the weak gate. The best checkpoint preserved the bit target but did not move `equation_transform` beyond `56/155` and missed total `193/315`.

Do not run full eval, package, or Kaggle submit from V290. Keep `checkpoint-6` as evidence for future packageable rank-19 recipes.

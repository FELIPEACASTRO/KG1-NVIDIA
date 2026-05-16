# V481 Terminal Result

V481 evaluated the V480 objective-aligned H200 checkpoints against the V221 weak contract. The job was canceled by FinOps after checkpoint-6 because all completed checkpoints failed the promotion gate and checkpoint-6 was the best training eval-loss point.

Job: `https://huggingface.co/jobs/felipesp1983/6a07fcf1e48bea4538b9f6f8`

## Gate

| Metric | Required |
|---|---:|
| Total weak | `>=193/315` |
| equation_transform | `>=57/155` |
| bit_manipulation | `>=136/160` |
| truncated | `0` |

## Results

| Candidate | Total | equation_transform | bit_manipulation | Truncated | Decision |
|---|---:|---:|---:|---:|---|
| V480 checkpoint-2 | `191/315` | `57/155` | `134/160` | `1` | rejected: bit regression and truncation |
| V480 checkpoint-4 | `190/315` | `56/155` | `134/160` | `0` | rejected: total/bit/equation below gate |
| V480 checkpoint-6 | `191/315` | `57/155` | `134/160` | `1` | rejected: bit regression and truncation |
| V480 checkpoint-8 | partial only | partial only | partial only | partial only | canceled by FinOps |

## Interpretation

V480 corrected the V476 objective-weight bug, but the adapter still did not produce a submit-safe gain. The recurrent pattern is now clear:

- `equation_transform` can move from `56` to `57`.
- That movement is paid for by `bit_manipulation` dropping from `136` to `134`.
- The strongest equation checkpoints also reintroduced `1` truncation.
- Therefore the current LoRA objective is not preserving the deployed bit behavior while transferring equation fixes.

This is not an ACC evaluator bug. It is a model-behavior transfer failure under the current SFT recipe.

## Decision

No full eval, package, or Kaggle submit from V480/V481.

Next implementation should not repeat this recipe with more steps. The next gate must force either:

1. answer-only/final-token objective that does not rewrite bit reasoning format; or
2. a deterministic solver/verifier route outside LoRA; or
3. a training objective with hard negative pairs that explicitly penalizes the observed `bit=134` regressions.

Evidence:

- `artifacts/v481_hf_h200_v480_weak_eval_launch/V481_WEAK_RESULTS_SUMMARY.json`
- `artifacts/v481_hf_h200_v480_weak_eval_launch/v481_hf_job_6a07fcf1e48bea4538b9f6f8_canceled_logs.txt`

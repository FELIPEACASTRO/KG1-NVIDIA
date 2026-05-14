# KG1 V372 HF Eval Summary

## Jobs

- Train A100: https://huggingface.co/jobs/felipesp1983/6a05c8a53308d79117b8f840
- Weak eval H200 checkpoint-1: https://huggingface.co/jobs/felipesp1983/6a05cc44e48bea4538b9cca8
- Output repo: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v372-nemo-a100-v371-trace-style-v290ckpt6
- Eval upload commit: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v372-nemo-a100-v371-trace-style-v290ckpt6/commit/815d5bfb7b752e24afc77fdb9206b4745434a8f9

## Result

Checkpoint evaluated: `checkpoint-1`

| Metric | Value |
| --- | ---: |
| Rows | `315` |
| Correct | `191` |
| Accuracy | `0.6063492063492063` |
| `equation_transform` correct | `56/155` |
| `bit_manipulation` correct | `135/160` |
| Truncated | `0` |
| Tokens per second | `2940.4308` |

## Decision

Reject V372.

Reason:

- It does not beat the adapter-only baseline `192/315`.
- It regresses bit from `136/160` to `135/160`.
- `equation_transform` remains at `56/155`.

FinOps action: do not evaluate checkpoint-2, do not run full eval, do not package, and do not submit this route.

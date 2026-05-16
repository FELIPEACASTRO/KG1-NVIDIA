# V477 vs Previous

| Item | Previous | V477 |
|---|---|---|
| Purpose | V476 H200 training smoke on V475 mixed dataset | H200 weak eval sweep for V476 checkpoints |
| Adapter repo | `felipesp1983/kg1-nemotron-lora-v476-v475-equation-bit-replay-v290ckpt6` | same repo, checkpoints `2/4/6/8/10/12` when complete |
| Eval contract | not applicable during training | V221 weak 315 contract via `scripts/hf_job_weak_eval_v245.py` |
| Promotion gate | train only; no submit authority | promote only if `total>192`, `equation>56`, `bit>=136`, `truncated=0` |
| Full eval | blocked until weak gain | still blocked until weak gain is measured |
| FinOps rule | one-hour H200 train cap | H200 eval capped by timeout and cancelled/rejected if weak gate fails |

This version does not change model weights or datasets. It only measures whether
the V476 checkpoints converted the V475 CPU +4 equation signal into adapter-only
weak accuracy without losing the bit guardrail.

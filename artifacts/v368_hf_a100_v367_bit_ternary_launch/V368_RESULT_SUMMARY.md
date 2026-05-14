# V368 Result Summary

Generated: 2026-05-14

## Scope

V368 tested whether the V366 CPU teacher gains could transfer into a pure LoRA adapter using the V367 boxed-only bit ternary dataset.

## Jobs

- A100 train smoke: https://huggingface.co/jobs/felipesp1983/6a05bad7e48bea4538b9c997
- H200 weak eval checkpoint-1: https://huggingface.co/jobs/felipesp1983/6a05be653308d79117b8f5ce
- Eval artifact commit: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v368-nemo-a100-v367-bit-ternary-v290ckpt6/commit/ffbbbb3a77de65cbd87eb71a6ec9b1516507da68

## Weak Eval Result

| Metric | Value |
|---|---:|
| Overall | `191/315` |
| `equation_transform` | `56/155` |
| `bit_manipulation` | `135/160` |
| Truncated | `0` |
| Accuracy | `0.6063492063492063` |

## Decision

Rejected.

Reasons:

- Below adapter-only gate `>192/315`.
- `bit_manipulation` regressed below the required `136/160`.
- `equation_transform` stayed at `56/155`.
- V369 transfer audit showed V368 transferred `0/8` accepted V366 bit gains.
- V368 made `10` prediction changes versus baseline: `1` gain, `2` losses, `7` neutral changes.

FinOps decision: do not evaluate checkpoint-2, do not run full eval, do not package, and do not submit.

## Local Artifacts

- Downloaded eval artifacts: `artifacts/v368_hf_a100_v367_bit_ternary_launch/eval_checkpoint1/`
- Transfer audit: `artifacts/v369_v368_transfer_failure_audit/20260514T_cpu_audit/`

## Next Step

Return to CPU-only work. Do not continue V367/V368 bit-only SFT unless a new CPU gate produces a new adapter-transfer-specific signal. The practical next path is equation/bit DSL or a stronger solver-to-adapter conversion, not more epochs on V367.

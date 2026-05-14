# V352 Result Summary

Generated: 2026-05-14

## Scope

V352 tested whether the V350 CPU no-loss bit gains could transfer into a pure LoRA adapter using the V351 minimal bit-transfer dataset.

## Jobs

- A100 train smoke: https://huggingface.co/jobs/felipesp1983/6a0520b13308d79117b8f393
- H200 weak eval checkpoint-2: https://huggingface.co/jobs/felipesp1983/6a0524423308d79117b8f3a1
- Eval artifact commit: https://huggingface.co/felipesp1983/kg1-nemotron-lora-v352-nemo-a100-v351-bit-transfer-v290ckpt6/commit/3a36716ef7adad1fbaef176f276865bb7a52e55a

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
- V354 transfer audit showed V352 transferred `0/2` accepted V350 bit fixes.
- V352 is `-10` vs V350 CPU teacher: `-7` equation and `-3` bit.

FinOps decision: do not run V352 checkpoints 4/6/8, do not run full eval, do not package, do not submit.

## Local Artifacts

- Downloaded eval summaries: `artifacts/v352_hf_a100_v351_bit_transfer_launch/eval_v352_checkpoint2/`
- Transfer audit: `artifacts/v354_v352_transfer_failure_audit/20260514T_cpu_audit/`

## Next Step

Return to CPU-only residual search. Implement V355 and launch no HF job until CPU gate proves a new no-loss gain above V350.

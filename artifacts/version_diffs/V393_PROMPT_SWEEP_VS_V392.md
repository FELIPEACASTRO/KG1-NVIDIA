# V393 Prompt Sweep vs V392 Baseline Lock

Generated: 2026-05-14

## Objective

V393 is a no-training weak evaluation sweep over the locked V290 checkpoint-6 adapter. It tests whether prompt/template/generation controls can improve the weak gate before spending more HF GPU on LoRA training.

## Comparison Table

| Version | Action | Adapter | Weak total | equation_transform | bit_manipulation | Trunc | Decision |
|---|---|---|---:|---:|---:|---:|---|
| V392 lock | Current best baseline | V290 checkpoint-6 | 192/315 | 56/155 | 136/160 | 0 | Keep as submit baseline |
| V391 ckpt-4 | New train, rejected | V391 checkpoint-4 | 191/315 | 56/155 | 135/160 | 0 | Canceled by FinOps |
| V393 | No-training prompt sweep | V290 checkpoint-6 | pending | pending | pending | pending | Promote only if weak total > 192, equation > 56, bit >= 136, trunc = 0 |

## Prompt Variants

| Variant | Purpose |
|---|---|
| baseline_v290_repro | Reproduce the locked V290 weak contract before comparing variants. |
| v221_boxed_suffix | Test the older boxed-answer prompt that rescued some historical evaluations. |
| no_suffix | Check whether suffix instructions suppress correct equation-style answers. |
| strict_disable_thinking | Force no-thinking template while preserving strict boxed output. |
| strict_2048_tokens | Reduce answer drift with a smaller generation budget. |

## Gate

No full eval, packaging, or Kaggle submit is allowed from V393 unless a variant beats the locked weak baseline:

- total correct must be at least `193/315`;
- `equation_transform` must be at least `57/155`;
- `bit_manipulation` must remain at least `136/160`;
- truncation must remain `0`.

If no variant passes, the roadmap stays on CPU solver-gate work instead of another broad LoRA run.

# V477 Terminal Result

Run id: `v477-h200-v221contract-v476-v475-mix-20260516T040237Z`

HF job: `https://huggingface.co/jobs/felipesp1983/6a07eca03308d79117b90ee7`

Output adapter repo: `felipesp1983/kg1-nemotron-lora-v476-v475-equation-bit-replay-v290ckpt6`

Decision: canceled by FinOps after checkpoint-4 because the route did not produce a submit-safe gain and was already regressing the bit guardrail.

## Comparison

| Candidate | Total weak | equation_transform | bit_manipulation | truncated | Gate |
|---|---:|---:|---:|---:|---|
| Baseline V291/V290 checkpoint-6 | 192/315 | 56/155 | 136/160 | 0 | reference |
| V476 checkpoint-2 | 192/315 | 57/155 | 135/160 | 0 | fail: total not improved, bit regressed |
| V476 checkpoint-4 | 191/315 | 57/155 | 134/160 | 1 | fail: total, bit, truncation regressed |

## Interpretation

The V475 CPU equation signal did transfer partially to the adapter at checkpoint-2
(`equation 56 -> 57`), but it paid for that with `bit 136 -> 135`, leaving total
weak unchanged at `192/315`. Checkpoint-4 made the regression stronger
(`191/315`, `bit 134`, `truncated 1`).

This is not submit-safe. Do not package, full-evaluate, or submit V476
checkpoints from this run.

## Next Action

Return to CPU target construction and require a stronger no-loss signal before
another paid GPU run. The next GPU route must prove, before launch, that the
bit replay/anchor is strong enough to keep `bit>=136` while equation improves.

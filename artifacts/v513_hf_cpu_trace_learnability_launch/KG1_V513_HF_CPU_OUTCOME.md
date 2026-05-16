# V513 HF CPU Outcome

- Job: `https://huggingface.co/jobs/felipesp1983/6a08e383e48bea4538ba03ba`
- Commit: `10cd921023da2fd0431cf85fe45259abfd91705a`
- Flavor: `cpu-upgrade`
- Unit cost: `$0.0005/min`
- Status: `COMPLETED`
- Uploaded artifact repo: `felipesp1983/kg1-v513-trace-learnability-gate-artifacts`
- Uploaded path: `v513-hf-cpu-trace-learnability-20260516T213601Z`

## Result

The HF CPU run reproduced the local V513 gate:

- `status=blocked_no_gpu`
- blockers: `2`
- warnings: `0`
- info: `1`

Blockers:

- `bit_answer_only_trace_not_learnable_enough`: `742/742` bit rows are answer-only.
- `bit_trace_rows_below_gpu_floor`: `0` deterministic bit traces, required at least `32`.

## Decision

Do not launch another GPU job from V510 as-is. Replace answer-only bit replay
with deterministic bit-pair/bitsum/stride traces, rerun V513, then only consider
a tiny paid smoke if V513, objective alignment, tokenization, and FinOps gates
all pass.

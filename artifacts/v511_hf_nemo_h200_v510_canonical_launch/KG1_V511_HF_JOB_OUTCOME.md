# V511 HF Job Outcome

Generated: 2026-05-16

## Job

- HF job: `felipesp1983/6a08dc43e48bea4538ba02ce`
- URL: `https://huggingface.co/jobs/felipesp1983/6a08dc43e48bea4538ba02ce`
- Adapter repo: `felipesp1983/kg1-nemotron-lora-v511-nemo-h200-v510-canonical-v290ckpt6`
- Status: `COMPLETED`
- Runtime: `0.05h`

## Checks That Passed

- Dataset V510 was loaded from HF dataset commit
  `40e71a686d9970c3c842d26dcf89200fc4990a51`.
- Train rows: `2627`.
- Planned steps: `2`.
- MoE target parameters were trainable:
  - `target_parameters_trainability_mode="trainable"`
  - `mlp.experts.gate_up_proj`: `5934` trainable LoRA tensors
  - `mlp.experts.down_proj`: `5934` trainable LoRA tensors
- `lm_head` was frozen by module filter.
- Trainable LoRA params: `869,318,656 / 32,466,091,456 = 2.6776%`.
- Checkpoint and final adapter upload completed.

## Result

| Metric | Before | After | Decision |
|---|---:|---:|---|
| eval loss on 96 examples | `2.8125` | `2.8128` | blocked |

The smoke proved the H200 pipeline, V510 upload, MoE trainability, and adapter
upload path. It did not provide a positive loss signal. By FinOps rule, this
candidate should not receive a paid weak/full ACC evaluation unless a separate
cheap gate produces new evidence.

## Next Decision

- Do not package or submit V511.
- Do not launch a longer V511-style run.
- Return to CPU-first discovery/teacher validation before another paid GPU job.

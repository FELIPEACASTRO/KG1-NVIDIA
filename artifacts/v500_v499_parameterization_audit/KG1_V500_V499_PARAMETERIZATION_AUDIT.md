# KG1 V500 V499 Parameterization Audit

Generated UTC: `2026-05-16T18:54:35.476249+00:00`

## Decision

- Status: `parameterization_technical_pass_objective_warning`
- Next action: Do not run paid weak eval for V499; create answer-span-weighted V500/V501 only after local gate proves weighting is active.
- Reason: Core gates passed, but final eval loss did not improve and the answer span was not emphasized for ACC.

## V499 Loss Snapshot

- Baseline eval loss: `2.8125`
- Final eval loss: `2.8162`
- Delta final-baseline: `0.0037`
- Max reserved memory GiB: `76.1`

## Key Parameter Verdict

| Area | Verdict | Detail |
|---|---:|---|
| train_rows | `PASS` | 1712 rows |
| val_rows | `PASS` | 428 rows |
| train_family_counts | `PASS` | {"bit_manipulation": 512, "equation_transform": 1200} |
| val_family_counts | `PASS` | {"bit_manipulation": 128, "equation_transform": 300} |
| duplicate_ids | `PASS` | train/val duplicate ids |
| assistant_final_markers | `PASS` | all rows include Final answer marker |
| assistant_boxed_markers | `PASS` | all rows include boxed final answer |
| reference_overlap | `PASS` | train reference id/prompt/prompt+answer overlap all zero |
| tokenization_gate | `PASS` | tokenization_gate_passed |
| train_max_length_runtime_safe | `PASS` | token_max=331; runtime MAX_LENGTH=1024 |
| offset_masks | `PASS` | runtime offset masks complete |
| source_weights | `PASS` | {"v498_bit_replay_guardrail_from_v475": 1.5, "v498_numeric_teacher_trace_pack": 1.0} |
| bit_replay_effective_share | `PASS` | bit_share=0.390244 |
| equation_effective_share | `PASS` | equation_share=0.609756 |
| hf_cost_gate | `PASS` | cost=0.083333/min |
| h200_hardware_gate | `PASS` | {"accelerator_model": "H200", "accelerator_quantity": "1", "accelerator_vram": "141 GB", "cpu": "23 vCPU", "name": "h200", "pretty_name": "Nvidia H200", "ram": "256 GB", "unit_cost_usd": 0.083333, "unit_label": "minute"} |
| init_adapter | `PASS` | {"repo": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke", "subfolder": "checkpoint-6"} |
| target_parameters_trainable | `PASS` | MoE target parameters reported trainable |
| lm_head_frozen | `PASS` | lm_head has zero trainable params in log |
| job_completed_and_uploaded | `PASS` | training/checkpoint/final upload complete |
| memory_under_abort_cap | `PASS` | max_mem_reserved=76.1GiB |
| eval_loss_improved | `WARN` | baseline=2.8125; final=2.8162; delta=0.0037 |
| answer_span_acc_alignment | `WARN` | weight=1.0; weighted_train=0; weighted_val=0 |
| max_steps_submit_gain_sufficient | `WARN` | max_steps=2 was smoke-only |

## Findings

- The dataset, hashes, family mix, tokenization gate, target-parameter trainability, H200 cost gate, and upload path are technically consistent.
- The run is not a submit-gain candidate because final eval loss did not improve over baseline on the V498 validation sample.
- The largest parameter gap is ACC alignment: `ANSWER_SPAN_LOSS_WEIGHT=1.0` produced `0` answer-span weighted examples, so the loss optimized full assistant traces instead of emphasizing the final boxed answer.
- `MAX_STEPS=2` was correct for a paid smoke test, but it is not a value that should be expected to produce a new submit-safe adapter.
- FinOps decision: do not run weak eval for V499 unless a separate reason appears, because the local objective signal is flat/slightly negative.

## Required Next Configuration

- Next paid run must use answer-focused loss, for example `ANSWER_SPAN_LOSS_WEIGHT>=4.0`, and a gate requiring answer-span weighted examples on train and validation.
- Keep V290 checkpoint-6 as init adapter, keep MoE target parameters trainable, keep `lm_head` frozen, and keep bit replay guardrail at or above the current effective share.
- Use a short but non-trivial run only after local gate confirms the answer-span weighting is active; candidate values: `MAX_STEPS=4..8`, `EVAL_EVERY_STEPS=2`, `SAVE_EVERY_STEPS=2`, with FinOps abort if eval loss rises by more than the baseline tolerance.
- Only launch weak ACC eval if the training objective improves or if there is a new deterministic CPU projection.

## Open Items

- `eval_loss_improved`: baseline=2.8125; final=2.8162; delta=0.0037
- `answer_span_acc_alignment`: weight=1.0; weighted_train=0; weighted_val=0
- `max_steps_submit_gain_sufficient`: max_steps=2 was smoke-only

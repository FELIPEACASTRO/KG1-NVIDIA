# KG1 V503 Live HF Job Parameterization Audit

Generated UTC: `2026-05-16T19:35:30.201975+00:00`
Job ID: `6a08bf05e48bea4538b9ffb5`

## Decision

- Status: `candidate_blocked_by_final_eval_regression`
- Hard failures: `1`
- Warnings: `0`
- Next action: Do not weak-eval or submit V501; keep V290 checkpoint-6 as submit-safe baseline and return to CPU/teacher search.

## Observed Runtime Parameters

```json
{
  "baseline_eval_loss": 1.9919,
  "final_eval_loss": 1.9923,
  "max_mem_reserved_gib": 76.1,
  "planned_steps": 4,
  "target_parameters_trainability_mode": "trainable",
  "train_answer_span_loss_weight": 4.0,
  "train_answer_span_weighted_examples": 1712,
  "train_answer_span_weighted_tokens": 15197,
  "trainable_pct": 2.6776,
  "validation_answer_span_weighted_examples": 428,
  "weighted_share_by_source": {
    "v498_bit_replay_guardrail_from_v475": 0.390244,
    "v498_numeric_teacher_trace_pack": 0.609756
  },
  "weighted_share_by_subcategory": {
    "bit_guardrail_replay": 0.390244,
    "equation_numeric_add_direct_hard_negative": 0.203252,
    "equation_numeric_colon_trailing_zero_hard_negative": 0.203252,
    "equation_numeric_minus_signed_hard_negative": 0.203252
  }
}
```

## Checks

| Area | Check | Verdict | Detail |
|---|---|---:|---|
| runtime | job log contains V501 output repo | `PASS` | V501 repo marker present |
| runtime | H200 context visible | `PASS` | H200 marker present |
| runtime | no unexpected python traceback | `PASS` | controlled final-eval regression abort |
| runtime | no unexpected failure | `PASS` | controlled final-eval regression abort |
| runtime | reserved memory under cap | `PASS` | max_mem_reserved=76.1 |
| tokenization | answer-span weight active | `PASS` | train_answer_span_loss_weight=4.0 |
| tokenization | train answer-span examples nonzero | `PASS` | train_weighted_examples=1712 |
| tokenization | train answer-span tokens above gate | `PASS` | train_weighted_tokens=15197 |
| tokenization | validation answer-span examples nonzero | `PASS` | val_weighted_examples=428 |
| tokenization | no truncation in summaries | `PASS` | truncation markers present |
| training | planned steps is short V501 smoke | `PASS` | planned_steps=4 |
| training | trainable percent bounded | `PASS` | trainable_pct=2.6776 |
| training | target parameters trainable | `PASS` | target_parameters_trainability_mode=trainable |
| training | MoE target params have trainable tensors | `PASS` | {"mlp.experts.down_proj": 432791552, "mlp.experts.gate_up_proj": 432791552} |
| training | lm_head frozen | `PASS` | frozen=4280320 trainable=0 |
| sampling | bit replay guardrail share active | `PASS` | {"v498_bit_replay_guardrail_from_v475": 0.390244, "v498_numeric_teacher_trace_pack": 0.609756} |
| sampling | equation share active | `PASS` | {"v498_bit_replay_guardrail_from_v475": 0.390244, "v498_numeric_teacher_trace_pack": 0.609756} |
| sampling | three equation subcategories present | `PASS` | {"bit_guardrail_replay": 0.390244, "equation_numeric_add_direct_hard_negative": 0.203252, "equation_numeric_colon_trailing_zero_hard_negative": 0.203252, "equation_numeric_minus_signed_hard_negative": 0.203252} |
| eval | baseline eval reached progress | `PASS` | baseline eval is running or complete |
| eval | baseline eval complete | `PASS` | baseline_eval_loss=1.9919 |
| eval | final eval available | `PASS` | final_eval_loss=1.9923 |
| eval | final eval does not regress | `FAIL` | baseline=1.9919 final=1.9923 |

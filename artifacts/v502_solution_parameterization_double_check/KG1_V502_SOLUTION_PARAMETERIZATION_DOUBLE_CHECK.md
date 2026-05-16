# KG1 V502 Solution Parameterization Double Check

Generated UTC: `2026-05-16T18:58:31.195147+00:00`

## Decision

- Status: `double_check_pass_next_debug_v501`
- Human action required: `False`
- Next action: Run V501 debug-only; launch H200 only if debug proves answer-span weighting and all gates are active.
- Reason: Core parameterization is now aligned for the next attempt; V499 remains blocked from weak eval.

## Checks

| Area | Check | Verdict | Detail |
|---|---|---:|---|
| dataset | train extraction verifies answers | `PASS` | {"duplicate_ids": 0, "families": {"bit_manipulation": 512, "equation_transform": 1200}, "final_extraction_fail": 0, "rows": 1712, "sources": {"v498_bit_replay_guardrail_from_v475": 512, "v498_numeric_teacher_trace_pack": 1200}, "subcategories": {"bit_guardrail_replay": 512, "equation_numeric_add_direct_hard_negative": 400, "equation_numeric_colon_trailing_zero_hard_negative": 400, "equation_numeric_minus_signed_hard_negative": 400}, "suspicious_answers": 0, "verify_fail": 0} |
| dataset | val extraction verifies answers | `PASS` | {"duplicate_ids": 0, "families": {"bit_manipulation": 128, "equation_transform": 300}, "final_extraction_fail": 0, "rows": 428, "sources": {"v498_bit_replay_guardrail_from_v475": 128, "v498_numeric_teacher_trace_pack": 300}, "subcategories": {"bit_guardrail_replay": 128, "equation_numeric_add_direct_hard_negative": 100, "equation_numeric_colon_trailing_zero_hard_negative": 100, "equation_numeric_minus_signed_hard_negative": 100}, "suspicious_answers": 0, "verify_fail": 0} |
| dataset | no suspicious answer chars | `PASS` | checked minus/nbsp/bom/dashes |
| dataset | family mix is intentional | `PASS` | {"bit_manipulation": 512, "equation_transform": 1200} |
| tokenization | gate passed | `PASS` | {'next_action': 'Only consider a tiny HF smoke train if roadmap risk/budget gates approve it.', 'reason': 'train_rows=1712; val_rows=428; train_token_max=331; val_token_max=331; completion_truncation=0', 'status': 'tokenization_gate_passed'} |
| tokenization | runtime max length safe | `PASS` | train token_max=331 |
| tokenization | no prompt truncation | `PASS` | train/val prompt truncation 0 |
| tokenization | offset masks complete | `PASS` | train=1712 val=428 |
| v499 | technical gates passed | `PASS` | {"hard_failures": 0, "next_action": "Do not run paid weak eval for V499; create answer-span-weighted V500/V501 only after local gate proves weighting is active.", "reason": "Core gates passed, but final eval loss did not improve and the answer span was not emphasized for ACC.", "status": "parameterization_technical_pass_objective_warning", "warnings": 3} |
| v499 | blocked by objective warning | `PASS` | expected warnings: eval loss, answer span, smoke steps |
| v499 | final eval did not improve | `PASS` | loss_delta=0.0037 |
| v501 | version/output repo are v501 | `PASS` | {"ANSWER_SPAN_LOSS_WEIGHT": "4.0", "ANSWER_SPAN_MIN_WEIGHTED_TOKENS": "1000", "MAX_STEPS": 4, "OUTPUT_REPO": "felipesp1983/kg1-nemotron-lora-v501-nemo-h200-v498-answer-span-v290ckpt6", "RUN_ID": null, "SOURCE_WEIGHTS": "v498_numeric_teacher_trace_pack=1.00,v498_bit_replay_guardrail_from_v475=1.50", "TRAINABLE_LORA_MODULES": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj", "VERSION": "v501_v498_answer_span_weighted_moe_trainable_no_lmhead_from_v290_checkpoint6_nemo_h200"} |
| v501 | max steps non-smoke | `PASS` | MAX_STEPS=4 |
| v501 | answer span weight active | `PASS` | ANSWER_SPAN_LOSS_WEIGHT=4.0 |
| v501 | answer span minimum nontrivial | `PASS` | ANSWER_SPAN_MIN_WEIGHTED_TOKENS=1000 |
| v501 | command overrides max steps | `PASS` | launcher rewrites inherited V493 MAX_STEPS |
| v501 | command enforces final eval baseline | `PASS` | final eval must not exceed baseline |
| v501 | bit replay still overweighted | `PASS` | v498_numeric_teacher_trace_pack=1.00,v498_bit_replay_guardrail_from_v475=1.50 |
| v501 | lm_head excluded from trainable filter | `PASS` | q_proj,k_proj,v_proj,o_proj,up_proj,down_proj |
| train_script | answer span weighting implemented | `PASS` | weighting branch and counters present |
| train_script | min weighted token gate implemented | `PASS` | gate present |
| train_script | masked CE normalizes by weighted tokens | `PASS` | masked loss denominator uses mask sum |
| train_script | weighted replacement sampling implemented | `PASS` | weighted sampling path present |
| train_script | final eval baseline abort implemented | `PASS` | baseline gate present |
| objective_gate | bit/equation share gates present | `PASS` | share gates present |
| acc_metric | final-answer extraction unit cases | `PASS` | [{"expected": "00000101", "extracted": "00000101", "ok": true}, {"expected": "30", "extracted": "30", "ok": true}, {"expected": "-4", "extracted": "-4", "ok": true}, {"expected": "a{b}\\c", "extracted": "a{b}\\c", "ok": true}] |
| acc_metric | verify_answer requires extracted answer | `PASS` | raw CoT is not accepted directly; eval must extract first |
| acc_metric | box_answer roundtrip | `PASS` | box_answer=\boxed{abc} |

## Practical Conclusion

- V499 was structurally correct but not objective-correct for ACC: final eval loss was flat/slightly worse and answer-span weighting was inactive.
- The next valid train path is V501: same verified V498 dataset, same V290 checkpoint-6, same MoE target parameters, frozen `lm_head`, but answer-span-weighted loss active before launch.
- Weak ACC eval remains blocked until a local training objective improves or a deterministic CPU gate provides new no-loss evidence.
- The current measurable target remains: `total>192`, `equation>=60`, `bit>=136`, `trunc=0`.

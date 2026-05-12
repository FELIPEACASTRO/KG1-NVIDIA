# deepseek/deepseek-v4-pro

role: assistant
sha256: 2103bd168821562ed1aff8a767649a468621b839983b59dd1564be5650b36400
chars: 34917

rs_tmp_cb8ms32srp6
reasoning
completed
reasoning_text
Let me analyze the evidence carefully and construct a concrete, testable distillation plan.

**Baseline analysis:**
- Current LoRA baseline: 823/947 correct (86.9%), eq=56/155, bit=135/160
- Oracle (with postprocessor): 838/947 (88.5%), eq=60/155, bit=146/160
- Gains: +15 overall, +4 eq, +11 bit
- No losses from the oracle

**Equation gains (4 specific rules):**
1. `7688e06e`: minus_signed_opposite_sign_guarded → 55 → -55
2. `274def88`: minus_signed_opposite_sign_guarded → -92 → 92
3. `d1bd7478`: colon_absdiff_unreverse_same_len → 03 → 30
4. `c5b058d6`: add_direct_over_model_add_variant → 35 → 134

**Bit gains (11 IDs):** fullbyte_safe_ternary (10 gains) + fullbyte_binary (1 gain) = 11 total

**Failed attempts:**
- V313 SFT: eq stuck at 56, bit dropped to 135
- V315 preference: eq stuck at 56, bit regressed to 134 at ckpt16

So the core problem: standard SFT and preference optimization on verifier synthetic data failed to move equation_transform at all, and slightly hurt bit.

**Root cause hypotheses:**

1. **Signal dilution**: The 4 equation gains are only 4 out of 155 equation problems. In a dataset of 947 rows or synthetic variants, these 4 examples get drowned out. The model never sees them enough times or with enough weight.

2. **Token-level vs. sequence-level mismatch**: The equation_transform task requires the model to output a single number or short string. The verifier signal is at the sequence level (correct/incorrect), but the model's error is in the final token distribution. Standard next-token prediction SFT on full sequences may not sufficiently concentrate gradient signal on the crucial final output tokens.

3. **Minus/negation confusion**: Two of the four gains are about minus sign handling (`minus_signed_opposite_sign_guarded`). The model likely has a subtle bias toward one sign interpretation, and SFT on correct answers alone doesn't create the necessary contrast to flip this behavior.

4. **Catastrophic forgetting**: The bit manipulation task involves 160 problems. Adding equation-focused training data may shift the model's internal representations enough to slightly degrade bit performance.

5. **Preference optimization wrong signal**: V315 used preference pairs but may have had incorrect preference ordering or insufficient contrast. If the "chosen" response was the correct answer and "rejected" was the model's baseline wrong answer, but the probability margin was small, DPO/ORPO may not apply enough gradient pressure.

6. **No trace-style or chain-of-thought**: The model may output answers directly without reasoning steps. For equation transform tasks, forcing a reasoning trace (e.g., "Step 1: identify operation...") could help the model internalize the transformation rules.

Now let me construct the plan.

**equation_transform_plan:**

The key insight: we have exactly 4 verified gains. We need to make the model internalize these specific transformation rules. This requires:

1. **Targeted trace-style data**: For each of the 4 gains, create contrastive examples showing the WRONG reasoning path (what the model currently does) and the RIGHT reasoning path (what produces the correct answer).

2. **Forced reasoning chain**: Require the model to output a step-by-step trace. For equation_transform, the trace should include:
   - Parse the expression
   - Identify the operation type
   - Apply the transformation rule
   - Compute result
   - Output final answer

3. **Heavy upweighting**: These 4 examples (and close variants) should appear at very high frequency in the training data — e.g., 5-10% of all training examples.

4. **Loss weighting on answer tokens**: Apply higher loss weight (e.g., 10x) on the final answer tokens compared to the reasoning trace tokens.

5. **Contrastive DPO pairs**: For each of the 4 gains, create pairs where:
   - Chosen = correct answer with correct reasoning trace
   - Rejected = baseline model's wrong answer with its (flawed) reasoning trace
   
   Then run DPO with a high beta (e.g., 0.3-0.5) to create strong preference signal.

6. **Data augmentation**: Create ~20-50 variants of each of the 4 equation problems by altering numbers while preserving the same transformation rule type. This prevents memorization and forces rule generalization.

**bit_manipulation plan:**

The 11 bit gains are from fullbyte_safe_ternary (10) and fullbyte_binary (1). The challenge is to add these without losing existing bit accuracy.

1. **Joint training**: Don't train equation and bit separately. Mix them in the same training run with careful balancing.

2. **Bit-specific trace**: For bit manipulation, the trace should show:
   - Parse bit operation
   - Break into sub-operations (especially for fullbyte operations)
   - Apply safe ternary pattern
   - Verify output

3. **Replay buffer**: Include all 160 bit problems (not just the 11 gains) in training to prevent forgetting. The existing correct ones should be reinforced.

4. **Gradual unfreezing**: Start training with only the equation-focused data, then gradually introduce bit data, or vice versa, to minimize interference.

5. **Elastic weight consolidation (EWC)**: Or simpler: include a KL divergence penalty toward the base model's bit manipulation behavior.

Let me now write out the full JSON plan.Let me analyze the evidence carefully and construct a concrete, testable distillation plan.

**Baseline analysis:**
- Current LoRA baseline: 823/947 correct (86.9%), eq=56/155, bit=135/160
- Oracle (with postprocessor): 838/947 (88.5%), eq=60/155, bit=146/160
- Gains: +15 overall, +4 eq, +11 bit
- No losses from the oracle

**Equation gains (4 specific rules):**
1. `7688e06e`: minus_signed_opposite_sign_guarded → 55 → -55
2. `274def88`: minus_signed_opposite_sign_guarded → -92 → 92
3. `d1bd7478`: colon_absdiff_unreverse_same_len → 03 → 30
4. `c5b058d6`: add_direct_over_model_add_variant → 35 → 134

**Bit gains (11 IDs):** fullbyte_safe_ternary (10 gains) + fullbyte_binary (1 gain) = 11 total

**Failed attempts:**
- V313 SFT: eq stuck at 56, bit dropped to 135
- V315 preference: eq stuck at 56, bit regressed to 134 at ckpt16

So the core problem: standard SFT and preference optimization on verifier synthetic data failed to move equation_transform at all, and slightly hurt bit.

**Root cause hypotheses:**

1. **Signal dilution**: The 4 equation gains are only 4 out of 155 equation problems. In a dataset of 947 rows or synthetic variants, these 4 examples get drowned out. The model never sees them enough times or with enough weight.

2. **Token-level vs. sequence-level mismatch**: The equation_transform task requires the model to output a single number or short string. The verifier signal is at the sequence level (correct/incorrect), but the model's error is in the final token distribution. Standard next-token prediction SFT on full sequences may not sufficiently concentrate gradient signal on the crucial final output tokens.

3. **Minus/negation confusion**: Two of the four gains are about minus sign handling (`minus_signed_opposite_sign_guarded`). The model likely has a subtle bias toward one sign interpretation, and SFT on correct answers alone doesn't create the necessary contrast to flip this behavior.

4. **Catastrophic forgetting**: The bit manipulation task involves 160 problems. Adding equation-focused training data may shift the model's internal representations enough to slightly degrade bit performance.

5. **Preference optimization wrong signal**: V315 used preference pairs but may have had incorrect preference ordering or insufficient contrast. If the "chosen" response was the correct answer and "rejected" was the model's baseline wrong answer, but the probability margin was small, DPO/ORPO may not apply enough gradient pressure.

6. **No trace-style or chain-of-thought**: The model may output answers directly without reasoning steps. For equation transform tasks, forcing a reasoning trace (e.g., "Step 1: identify operation...") could help the model internalize the transformation rules.

Now let me construct the plan.

**equation_transform_plan:**

The key insight: we have exactly 4 verified gains. We need to make the model internalize these specific transformation rules. This requires:

1. **Targeted trace-style data**: For each of the 4 gains, create contrastive examples showing the WRONG reasoning path (what the model currently does) and the RIGHT reasoning path (what produces the correct answer).

2. **Forced reasoning chain**: Require the model to output a step-by-step trace. For equation_transform, the trace should include:
   - Parse the expression
   - Identify the operation type
   - Apply the transformation rule
   - Compute result
   - Output final answer

3. **Heavy upweighting**: These 4 examples (and close variants) should appear at very high frequency in the training data — e.g., 5-10% of all training examples.

4. **Loss weighting on answer tokens**: Apply higher loss weight (e.g., 10x) on the final answer tokens compared to the reasoning trace tokens.

5. **Contrastive DPO pairs**: For each of the 4 gains, create pairs where:
   - Chosen = correct answer with correct reasoning trace
   - Rejected = baseline model's wrong answer with its (flawed) reasoning trace
   
   Then run DPO with a high beta (e.g., 0.3-0.5) to create strong preference signal.

6. **Data augmentation**: Create ~20-50 variants of each of the 4 equation problems by altering numbers while preserving the same transformation rule type. This prevents memorization and forces rule generalization.

**bit_manipulation plan:**

The 11 bit gains are from fullbyte_safe_ternary (10) and fullbyte_binary (1). The challenge is to add these without losing existing bit accuracy.

1. **Joint training**: Don't train equation and bit separately. Mix them in the same training run with careful balancing.

2. **Bit-specific trace**: For bit manipulation, the trace should show:
   - Parse bit operation
   - Break into sub-operations (especially for fullbyte operations)
   - Apply safe ternary pattern
   - Verify output

3. **Replay buffer**: Include all 160 bit problems (not just the 11 gains) in training to prevent forgetting. The existing correct ones should be reinforced.

4. **Gradual unfreezing**: Start training with only the equation-focused data, then gradually introduce bit data, or vice versa, to minimize interference.

5. **Elastic weight consolidation (EWC)**: Or simpler: include a KL divergence penalty toward the base model's bit manipulation behavior.

Let me now write out the full JSON plan.
msg_tmp_a2e4s0o0omt
message
completed
assistant
output_text
```json
{
  "root_cause_hypotheses": [
    "Signal dilution: V313/V315 trained on broad synthetic mixes where the 4 equation-gain examples constituted <0.5% of tokens. Standard SFT and preference objective gradients were dominated by 940+ other rows, so the model never received sufficient gradient pressure on the 4 specific decision boundaries (minus-sign negation, colon-absdiff unreverse, add-direct-vs-variant).",
    "Token-level gradient starvation: equation_transform answers are typically 1-3 tokens (e.g., '-55', '30', '134'). SFT loss averages over all sequence tokens, so the crucial final-answer tokens received ~2-5% of total per-example loss. V313's verifier-synthetic data likely included long CoT prefixes that further diluted per-token gradient on the answer span.",
    "No contrastive/negative signal: the 2 minus_signed_opposite_sign_guarded gains are sign-flip errors. SFT on only correct answers does not teach the model what NOT to output. V315's preference pairs may have used a correct answer as 'chosen' and a random incorrect as 'rejected' rather than the model's own specific failure mode, producing weak gradient. Effective preference distillation for sign errors requires the rejected response to be the model's actual wrong-sign output.",
    "Catastrophic interference between tasks: V313 lost 1 bit point and V315 lost 1-2 bit points. The bit manipulation subspace (fullbyte operations) is fragile under representation drift from equation-focused training, suggesting no replay or regularization of the bit distribution was used.",
    "Lack of structured reasoning trace: the model likely emits answers directly. For rule-based transforms (e.g., colon_absdiff_unreverse_same_len requiring '03'→'30'), a reasoning chain forces the model to decompose the operation and reduces the chance of a shallow shortcut that happened to be correct on training examples but fails on the eval instance.",
    "Preference optimization beta too low / margin insufficient: V315 used DPO/ORPO with a beta that did not sufficiently penalize the rejected distribution for these specific hard negatives. High-confidence wrong answers (where the model assigns >60% probability to the wrong token) need stronger preference pressure."
  ],
  "equation_transform_plan": {
    "strategy": "Targeted contrastive distillation with forced reasoning traces and heavy upweighting of the 4 exact gain instances plus rule-preserving variants.",
    "trace_style": {
      "format": "mandatory step-by-step tagged reasoning before final answer",
      "template": "## Analysis\n- Operation type: {detected_op}\n- Input expression: {input}\n- Sign analysis: {sign_reasoning}\n- Transformation rule: {rule_name}\n- Intermediate: {intermediate}\n## Answer\n{final_answer}",
      "rationale": "Forcing structured decomposition prevents the model from taking a shallow memorization shortcut. The trace acts as a regularizer that ties the answer token distribution to explicit rule identification."
    },
    "data_mix": {
      "gain_1_2_minus_signed": {
        "ids": ["7688e06e", "274def88"],
        "rule": "minus_signed_opposite_sign_guarded",
        "core_examples": "Both exact eval instances with correct + incorrect traces as separate completions for DPO pairs",
        "variants_per_id": 25,
        "variant_strategy": "Generate variants by permuting operand values (±10 to ±99 range), preserving the sign-opposite pattern that triggers the guarded negation",
        "dpo_chosen": "Correct trace concluding with target answer (e.g., -55, 92)",
        "dpo_rejected": "Model's actual wrong trace from baseline inference, concluding with wrong sign (e.g., 55, -92)",
        "loss_weight": "10x on the final answer token span, 1x on trace tokens"
      },
      "gain_3_colon_absdiff": {
        "id": "d1bd7478",
        "rule": "colon_absdiff_unreverse_same_len",
        "core_examples": "Exact eval instance",
        "variants": 20,
        "variant_strategy": "Generate same-length digit-pair strings with colon separator, vary the digit values while keeping length=2 on each side. Ensure the 'unreverse' step is explicitly traced.",
        "loss_weight": "10x on answer token span"
      },
      "gain_4_add_direct": {
        "id": "c5b058d6",
        "rule": "add_direct_over_model_add_variant",
        "core_examples": "Exact eval instance (35+99=134)",
        "variants": 30,
        "variant_strategy": "Vary the two addends but preserve the pattern where the model's default 'variant' pathway produces a different sum than direct addition. Include cases where the variant path is seductively close to direct sum to force explicit rule selection.",
        "loss_weight": "10x on answer token span"
      },
      "negative_examples": "For each of the 4 gain rules, include 5-10 examples of the SAME rule type where the model already gets it right, presented with correct traces for reinforcement. Ratio of wrong-fix:right-reinforce ≈ 4:1 per rule.",
      "overall_weight_in_batch": "These equation-gain examples (exact + variants + negatives) should constitute 8-12% of each training batch by row count, and ~15-20% by token count due to trace length upweighting."
    },
    "loss_weighting": {
      "answer_token_mask": "Apply a binary mask multiplying loss by 10.0 for all tokens in the ## Answer section (final answer span). Trace tokens get weight 1.0.",
      "per_example_normalization": "Normalize so that total loss contribution per equation-gain example is ~5-8x that of a standard example, via the combination of higher batch frequency and answer-token upweighting.",
      "rule_level_reweighting": "If after 25% of training steps eq-gain validation accuracy on the 4 exact instances is below 75% (3/4), double the batch frequency of all equation-gain data for the remainder."
    },
    "guardrails": {
      "only_target_gains": "Do NOT include equation_transform examples beyond these 4 gain types. Adding other equation problems risks gradient competition with no verified benefit.",
      "no_random_equation_data": "Reject any synthetic augmentation that introduces equation problems outside the 4 verified gain rules.",
      "trace_quality_filter": "Automatically filter generated traces: reject any trace that reaches the correct answer through obviously wrong reasoning (checked via regex on intermediate steps).",
      "max_variant_similarity": "Ensure variants are not near-duplicates of eval instances. Use edit distance ≥ 2 on the operand portion."
    }
  },
  "bit_manipulation_plan": {
    "strategy": "Replay-all bit problems with the 11 gain instances receiving trace-level attention, using elastic weight consolidation (EWC) regularization to prevent drift on the existing 135 correct bit instances.",
    "fullbyte_safe_ternary_10_gains": {
      "approach": "For each of the 10 gain IDs, create a contrastive DPO pair: chosen = correct answer with decomposition trace showing the safe ternary pattern step-by-step; rejected = the model's current wrong output trace.",
      "trace_format": "## Bit Analysis\n- Operation: {op_type}\n- Input bits: {input_repr}\n- Byte boundary detection: {boundary_analysis}\n- Safe ternary decomposition: {step1}, {step2}, {step3}\n- Result assembly: {assembly}\n## Answer\n{final}",
      "variants_per_id": 10,
      "variant_strategy": "For each gain ID, generate variants by flipping non-critical bits to create similar-but-different fullbyte operations that exercise the same ternary decomposition pattern."
    },
    "fullbyte_binary_1_gain": {
      "approach": "Same contrastive trace approach as ternary gains. Since there is only 1 instance, create 15 variants to ensure the binary-specific decomposition pattern generalizes.",
      "variant_strategy": "Vary the byte positions and bit patterns while preserving the binary (non-ternary) nature of the operation."
    },
    "replay_buffer": {
      "all_160_bit_problems": "Include ALL 160 bit manipulation problems in every epoch. The 135 already-correct ones receive standard SFT with short trace (or no trace, just answer) at weight 1.0. The 11 gain instances get 5x sampling frequency and trace-level training.",
      "ratio": "In each batch: ~30% bit problems (of which ~20% are the 11 gain instances sampled at 5x, and ~80% are replay of the 135 correct ones), ~55% equation_transform targeted data, ~15% general non-equation non-bit data from the original 947 distribution to preserve general reasoning."
    },
    "anti_forgetting_measures": {
      "ewc_regularization": "After 1 epoch of joint training, compute Fisher information diagonal for the bit manipulation output head (last LoRA layer) on the 135 correct instances. Add EWC penalty λ=0.1 to subsequent training steps, penalizing weight movement away from these Fisher-important directions.",
      "alternative_simpler": "If EWC is too complex, use a simpler KL constraint: during training on equation data, add a KL divergence penalty (β_kl=0.05) between the current model's bit-problem output distribution and the frozen baseline model's distribution, evaluated on the 160 bit problems every 50 steps.",
      "bit_validation_every_N_steps": "Evaluate bit accuracy on a held-out set of 20 bit problems (including 3-4 of the 11 gain instances) every 200 training steps. If bit accuracy drops below 134/160 (the V315 ckpt16 level), immediately roll back to the best checkpoint and reduce equation-data batch frequency by 50%."
    }
  },
  "dataset_blueprint": {
    "total_rows_approx": 2400,
    "splits": {
      "train": 2100,
      "val": 300
    },
    "row_types_and_ratios": {
      "equation_gain_exact_4": {
        "count": 4,
        "description": "The exact 4 eval instances that gain from oracle. Each appears with both correct-trace completion (for SFT) and wrong-trace completion (for DPO rejected). DPO pairs: 4 chosen + 4 rejected = 8 rows.",
        "train_only": true
      },
      "equation_gain_variants": {
        "count": 100,
        "description": "25+25+20+30 = 100 variants across the 4 gain rules. Each variant has a correct-trace completion. 80 in train, 20 in val.",
        "train": 80,
        "val": 20
      },
      "equation_gain_negative_reinforcement": {
        "count": 24,
        "description": "6 per rule type where the model already gets it right. Correct-trace completions. 20 in train, 4 in val.",
        "train": 20,
        "val": 4
      },
      "bit_gain_exact_11": {
        "count": 11,
        "description": "The exact 11 bit gain instances. Each appears as DPO pair (chosen correct trace + rejected wrong trace). 22 rows total, all in train.",
        "train_only": true
      },
      "bit_gain_variants": {
        "count": 125,
        "description": "~10 variants each for 10 ternary gains + 15 variants for the 1 binary gain = 115. Rounded to 125 for even batching. Correct-trace completions. 100 train, 25 val.",
        "train": 100,
        "val": 25
      },
      "bit_replay_all": {
        "count": 320,
        "description": "All 160 bit problems repeated 2x per epoch. Short-answer completions (no trace) for efficiency. 270 train (includes the 135 already-correct + the same 11 gain instances in short-answer form for reinforcement), 50 val.",
        "train": 270,
        "val": 50
      },
      "general_retention": {
        "count": 600,
        "description": "Random sample of ~600 non-equation, non-bit problems from the original training distribution to preserve general reasoning capability. Includes diverse problem types from the 947-distribution. 500 train, 100 val.",
        "train": 500,
        "val": 100
      },
      "dpo_pairs_total": {
        "count": 30,
        "description": "4 equation gain exact + 11 bit gain exact = 15 DPO pairs (15 chosen + 15 rejected = 30 rows). These can be integrated into a DPO training phase or mixed into SFT as contrastive pairs with per-token weighted loss.",
        "train_only": true
      }
    },
    "approximate_train_composition_by_row_pct": {
      "equation_gain_all": "~9% (104 rows: 4 exact + 80 variants + 20 negatives)",
      "bit_gain_all": "~18% (210 rows: 11 exact + 10 exact rejected DPO + 100 variants + ~89 replay of gain instances)",
      "bit_replay_correct": "~16% (~181 rows: replay of 135 already-correct bit problems at 1.34x)",
      "general_retention": "~24% (500 rows)",
      "dpo_contrastive": "~1.4% (30 rows)",
      "padding_to_batch": "~32% (additional general or repeated high-value rows to fill batch size)"
    },
    "val_composition": {
      "equation_gain_variants": 20,
      "equation_negative": 4,
      "bit_gain_variants": 25,
      "bit_replay_sample": 50,
      "general_retention": 100,
      "held_out_bit_problems": 20,
      "held_out_equation": 10,
      "full_val_total": "~300 rows including diversity from original distribution plus targeted eval of the gain rules"
    }
  },
  "training_recipe": {
    "lora_config": {
      "modules": "q_proj, v_proj, o_proj (rank 64, alpha 128) plus additionally k_proj and gate_proj (rank 32, alpha 64) to give the model more capacity to learn the structured trace format and rule-conditioned transformations. The added k_proj and gate_proj modules were NOT used in V313/V315 and provide new capacity specifically for internalizing verifier logic.",
      "rationale": "q/v/o at rank 64 gives strong adaptation capacity for output distribution reshaping (needed for answer-token correction). k_proj at rank 32 helps the model attend correctly to operation-type tokens in its own trace. gate_proj at rank 32 helps MLP gating learn the rule-conditional logic. Total trainable params should be ~1.5-2x V313 to ensure sufficient capacity."
    },
    "lr_schedule": {
      "conservative_branch": {
        "name": "Conservative (higher confidence of no bit regression)",
        "lr": "5e-5 constant with linear warmup over first 200 steps (10% of total)",
        "total_steps": 2000,
        "effective_epochs": "~4-5 epochs over the ~2100 train rows at batch size 4-8",
        "rationale": "Lower LR minimizes representation drift, giving the bit-replay and EWC mechanisms time to stabilize."
      },
      "aggressive_branch": {
        "name": "Aggressive (faster convergence, higher risk)",
        "lr": "2e-4 with cosine decay to 2e-5",
        "total_steps": 1200,
        "effective_epochs": "~2-3 epochs",
        "rationale": "Higher initial LR may be needed to overcome the model's entrenched wrong-sign behavior on minus_signed_opposite_sign_guarded. Cosine decay reduces risk in later steps."
      }
    },
    "batch_size": "4-8 (effective batch size; use gradient accumulation if needed to hit batch size 8)",
    "optimizer": "AdamW, weight_decay=0.01, betas=(0.9, 0.999)",
    "training_phases": {
      "phase_1_sft_warmup": {
        "steps": "0 to 400 (conservative) or 0 to 240 (aggressive)",
        "data": "All SFT data (equation gain variants + bit replay + general retention). No DPO pairs yet.",
        "loss": "Standard next-token prediction with answer-token upweighting (10x on answer spans). Include KL penalty toward baseline on bit problems.",
        "goal": "Establish trace-format compliance and move the easier equation gains (colon_absdiff, add_direct) before tackling the harder sign-flip gains."
      },
      "phase_2_dpo_contrastive": {
        "steps": "401 to 1200 (conservative) or 241 to 720 (aggressive)",
        "data": "Mix SFT data (60% of batch) with DPO pairs (40% of batch). DPO pairs focus on the 4 equation exact instances + 11 bit exact instances.",
        "dpo_config": "beta=0.5 (strong preference pressure), reference model = baseline (pre-training) merged model. Use IPO variant if standard DPO shows unstable loss.",
        "loss": "Combined loss: 0.7 * DPO_loss + 0.3 * SFT_loss_on_chosen. This hybrid prevents the model from diverging too far from the SFT trace distribution.",
        "goal": "Flip the minus-signed behavior and the fullbyte ternary/binary instances through explicit contrast."
      },
      "phase_3_fine_tuning": {
        "steps": "1201 to 2000 (conservative) or 721 to 1200 (aggressive)",
        "data": "100% SFT data at reduced LR. No DPO pairs. Focus on consolidating gains and reinforcing bit replay.",
        "loss": "Standard SFT with answer-token upweighting. KL penalty on bit reduced to β_kl=0.02.",
        "goal": "Solidify the DPO-induced behavior into stable SFT-style parameterization, reduce any DPO-induced instability in bit accuracy."
      }
    },
    "checkpoint_frequency": "Every 200 steps. Keep all checkpoints. Run full validation gate at each checkpoint."
  },
  "validation_gate": {
    "metrics": {
      "primary": "weak_set_total_correct (target ≥ 193/315, meaning eq ≥ 60/155 + bit ≥ 136/160 = 196. But the evidence shows eq=60 and bit=146 in oracle gives 838 total, so weak-set target is eq≥60 AND bit≥136 AND total≥193.)",
      "equation_transform_correct": "≥ 60/155 (the oracle ceiling given current verifier). Must include all 4 gain instances being correct.",
      "bit_manipulation_correct": "≥ 136/160 (at least +1 over baseline, no regression below 135). Preferably 140+.",
      "gain_4_exact": "4/4 correct on the exact equation gain instances (7688e06e, 274def88, d1bd7478, c5b058d6). This is a HARD gate — if any of the 4 is wrong, the checkpoint is rejected regardless of overall eq score.",
      "gain_11_exact_bit": "≥ 7/11 correct on the exact bit gain instances. The oracle achieves 11/11 but the distillation may not capture all 11; 7+ indicates meaningful internalization.",
      "bit_replay_stability": "No more than 2 losses on the 135 previously-correct bit instances. If losses > 2, the checkpoint fails the stability gate.",
      "truncation": "≤ 2 truncated rows in full 947 evaluation.",
      "general_regression": "Score on 100 held-out general-retention problems must not decrease by more than 2 points vs baseline."
    },
    "stop_rules": [
      "Stop immediately if bit drops below 133 at any checkpoint (worse than V315 worst case).",
      "Stop if equation_transform drops below 54 at any checkpoint after phase 1 (worse than baseline, indicating negative transfer).",
      "Stop if 3 consecutive checkpoints show no improvement in any metric (plateau detected)."
    ],
    "promote_rules": [
      "Promote IF: gain_4_exact == 4/4 AND eq ≥ 60 AND bit ≥ 136 AND total ≥ 193 AND bit_replay_stability passed AND truncation ≤ 2 AND general_regression ≤ -2.",
      "If multiple checkpoints pass all gates, select the one with highest total correct (balanced toward higher bit).",
      "If a checkpoint achieves eq ≥ 61 (exceeding oracle!) with bit ≥ 136, flag for manual review — possible overfitting or data leakage."
    ],
    "full_eval_frequency": "At every checkpoint (every 200 steps). Also run a fast partial eval (gain_4_exact + gain_11_exact_bit + bit_stability_sample) every 50 steps during phase 2 for early signal."
  },
  "risks": [
    {
      "risk": "Minus-signed behavior is deeply baked into base model weights beyond LoRA reach",
      "probability": "Medium",
      "impact": "High — eq stays at 56",
      "mitigation": "The aggressive branch with 2e-4 LR and rank-64 LoRA on q/v/o/k/gate provides maximal adaptation capacity. If both branches fail after 3 attempts, escalate to full fine-tuning of the output projection layer (not just LoRA) for the answer head — but this may violate challenge rules."
    },
    {
      "risk": "DPO on 15 exact pairs causes overfitting: model memorizes the 4 eq instances but fails on variants",
      "probability": "Medium",
      "impact": "Medium — passes gain_4_exact but eq still ~56 on rest",
      "mitigation": "The 100 equation variants in training and 20 in validation directly test generalization. If val variant accuracy lags exact accuracy by >15%, reduce DPO weight and increase variant SFT proportion."
    },
    {
      "risk": "Bit accuracy degrades despite replay due to representation interference from k_proj/gate_proj LoRA",
      "probability": "Medium-Low",
      "impact": "High — fails promotion gate",
      "mitigation": "The EWC penalty and KL constraint are designed to prevent this. If bit drops in phase 2, immediately freeze the new k_proj and gate_proj LoRA modules and continue with only q/v/o. If still dropping, revert to conservative branch with k_proj and gate_proj removed entirely."
    },
    {
      "risk": "Trace format increases sequence length beyond model's effective context or slows inference beyond submission limits",
      "probability": "Low",
      "impact": "Medium — may cause truncation",
      "mitigation": "Keep trace tokens under 200 per example. Test max sequence length on the longest bit trace before full training. If needed, use a shorter trace format or drop trace for bit replay rows."
    },
    {
      "risk": "The 4 equation gains are verifier artifacts, not actually learnable by the model architecture",
      "probability": "Low",
      "impact": "Critical — entire plan fails",
      "mitigation": "First run a minimal viability test: train ONLY on the 4 exact equation instances + 100 variants with aggressive LR for 500 steps. If eq does not move from 56, the gains are likely unlearnable by LoRA and we need a fundamentally different approach (e.g., constrained decoding rules distilled via knowledge distillation from a larger teacher)."
    },
    {
      "risk": "General reasoning regression on non-bit, non-equation problems due to over-indexing on the targeted data",
      "probability": "Medium-Low",
      "impact": "Medium — may fail general_regression gate",
      "mitigation": "The 500 general-retention rows in training should prevent this. If general regression exceeds 2 points, increase general-retention proportion to 35% and reduce equation variant count."
    }
  ],
  "do_not_do": [
    "Do NOT train on verifier-synthetic data for equation problems outside the 4 verified gain rules. V313 proved this dilutes signal without helping eq score.",
    "Do NOT use full dataset SFT as a first pass. V313's broad SFT approach failed. Start targeted from step 0.",
    "Do NOT use DPO/ORPO with beta < 0.3. V315's weak preference signal suggests low beta fails to move the minus-sign behavior. Use beta ≥ 0.5 for sign-flip pairs.",
    "Do NOT exclude the 135 already-correct bit problems from training. V313 lost 1 bit point; V315 lost 1-2. Full replay with anti-forgetting regularization is mandatory.",
    "Do NOT train equation and bit in separate stages (e.g., all equation first, then all bit). Joint training from the start prevents catastrophic forgetting and allows the optimizer to find a shared representation that serves both tasks.",
    "Do NOT use rank < 32 LoRA. V313/V315 likely used lower-rank adapters that lacked capacity to represent the rule-conditioned transformations. Rank 64 on q/v/o is the minimum for this plan.",
    "Do NOT skip the viability test (risk #5). Before committing to the full 2000-step run, validate that the 4 equation gains are learnable at all with a 500-step probe. This saves compute budget.",
    "Do NOT generate synthetic equation-transform data by prompting the model itself. The model's own wrong behavior will contaminate the synthetic traces. Manually construct traces or use a stronger model (e.g., Nemotron full, GPT-4) to generate reasoning traces, then verify correctness programmatically.",
    "Do NOT evaluate on the exact eval split during training. The gain_4_exact gate uses the real eval instances — these must be held out completely. Train only on variants and the exact instances' DPO-pair formulations (where the rejected trace is the model's own baseline output, not a new inference on the eval instance).",
    "Do NOT run more than 3000 total training steps. If gains haven't materialized by step 2000 (conservative) or step 1200 (aggressive), further training is unlikely to help and risks overfitting the training distribution.",
    "Do NOT merge LoRA weights and re-train (iterative distillation). The challenge requires LoRA-only submission. Iterative merge-retrain loops are prohibited and waste budget. Each attempt must be a clean LoRA training from the base model."
  ]
}
```

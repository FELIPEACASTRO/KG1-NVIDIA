# USER

role: user
sha256: 324b9f47abc7b728985d479f2fa505169acd4a8d54db62ef2fd031684d516802
chars: 3034

message
user
input_text
You are advising an ML engineer. Use ONLY the evidence below; do not invent Kaggle private-test facts. We need to turn a verified local postprocessor/verifier signal into LoRA-only behavior for NVIDIA Nemotron reasoning challenge. The previous SFT/preference distillation failed to move equation_transform above 56 and slightly regressed bit. Provide a concrete, testable distillation plan.

Required output in JSON with keys:
- root_cause_hypotheses: list of why V313/V315 failed
- equation_transform_plan: steps to internalize exactly the 4 verified equation gains, including trace style, data mix, loss weighting, and guardrails
- bit_manipulation_plan: steps to internalize fullbyte_safe_ternary/fullbyte_binary gains without losing existing bit accuracy
- dataset_blueprint: exact row types and approximate ratios for next train/val
- training_recipe: LoRA modules, LR/steps/epochs suggestions, conservative vs aggressive branches
- validation_gate: metrics and stop/promote rules
- risks: concrete failure modes
- do_not_do: things likely to waste HF budget

Evidence:
{
  "task": "Kaggle NVIDIA Nemotron Model Reasoning Challenge; LoRA-only submit, no external postprocessor allowed unless distilled into adapter behavior.",
  "current_submit_baseline_full947": {
    "overall_correct": 823,
    "rows": 947,
    "accuracy": 0.8690601900739177,
    "bit_manipulation_correct": "135/160",
    "equation_transform_correct": "56/155",
    "truncated": 1
  },
  "verified_postprocessor_oracle_full947": {
    "overall_correct": 838,
    "accuracy": 0.8848996832101372,
    "bit_manipulation_correct": "146/160",
    "equation_transform_correct": "60/155",
    "gains": 15,
    "losses": 0
  },
  "equation_gain_rules": [
    {
      "id": "7688e06e",
      "rule": "minus_signed_opposite_sign_guarded",
      "baseline": "55",
      "target": "-55"
    },
    {
      "id": "274def88",
      "rule": "minus_signed_opposite_sign_guarded",
      "baseline": "-92",
      "target": "92"
    },
    {
      "id": "d1bd7478",
      "rule": "colon_absdiff_unreverse_same_len",
      "baseline": "03",
      "target": "30"
    },
    {
      "id": "c5b058d6",
      "rule": "add_direct_over_model_add_variant",
      "baseline": "35",
      "target": "134"
    }
  ],
  "bit_gain_rules": {
    "fullbyte_safe_ternary": 10,
    "fullbyte_binary": 1,
    "gain_ids": [
      "1abaffca",
      "0e70c867",
      "b8722d19",
      "7192535b",
      "8740ed31",
      "1a7c8520",
      "a6192d29",
      "048cc279",
      "4c327b55",
      "b8aa3072",
      "5ba26f21"
    ]
  },
  "failed_distillation_attempts": [
    {
      "version": "V313 SFT verifier synthetic",
      "best_weak": "191/315 eq=56 bit=135; no eq gain; bit -1"
    },
    {
      "version": "V315 preference on V312",
      "closed_checkpoints": "ckpt4/8/12 all 191 eq=56 bit=135; ckpt16 190 eq=56 bit=134"
    }
  ],
  "promotion_gate": "Promote only if weak eq>=60, bit>=136, total>=193, truncation acceptable; full eval no family regression."
}

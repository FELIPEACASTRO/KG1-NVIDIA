# openai/gpt-5.3-codex

variant: `openai/gpt-5.3-codex-20260224`

cost: `0.06937525` duration_ms: `72096` tokens: `4709`

```json
{
  "verdict": "Do not run another broad SFT or preference run. Your current gates are mostly correct. The next credible path is: (1) CPU-first executable equation teacher with much stronger DSL and metamorphic certification, then (2) a conservative, bit-protected adapter update using only certified data, and (3) if needed, contrastive decoy training built from executable near-misses. If CPU cannot produce high-precision certified equation supervision at meaningful coverage, stop and do not spend GPU.",
  "key_diagnosis": {
    "q1_are_finops_and_promotion_gates_correct": "Yes, mostly correct and appropriately strict. Keep them. I would add two gates: a pre-weak public-train holdout gate for equation lift with zero bit regression, and a sequence-length/truncation risk gate before GPU.",
    "q2_most_likely_missing_ingredient": "High-precision, execution-certified equation supervision with invariance checks (not raw/reconstructed text supervision), plus conservative update constraints to prevent bit-manipulation drift.",
    "q3_next_move_choice": "Primary: stronger symbolic DSL and data format change (certified answer-only + executable near-miss decoys). Secondary: conservative objective (bit-anchor replay / KL anchoring). Not more epochs/LR sweeps. Do not abandon LoRA yet, but abandon quickly if certified coverage/precision gates are not met."
  },
  "ranked_next_experiments": [
    {
      "rank": 1,
      "name": "CPU-only Equation DSL v2 with Metamorphic Certification",
      "hypothesis": "Equation transfer is failing because current supervision is noisy/underspecified; only executable, certified rules will produce signal that survives into adapter behavior.",
      "exact_data_source_allowed": [
        "Official/public train split only (inputs + labels).",
        "Synthetic variants generated from public-train rows via deterministic metamorphic transforms.",
        "No weak/full rows or labels at any stage."
      ],
      "cpu_gate_before_gpu": {
        "must_pass": [
          "Certified equation coverage >= 25% of public-train equation rows.",
          "Per-candidate rule passes original + >= 20 metamorphic checks with 100% consistency.",
          "Estimated label precision >= 98% on manually audited sample (stratified by rule family).",
          "Ambiguous/non-unique target rate <= 5%."
        ],
        "block_if": [
          "Coverage < 25%.",
          "Any audited precision failure below 98%.",
          "High ambiguity or inconsistent canonicalization."
        ]
      },
      "artifact_to_build": [
        "`eq_dsl_v2_program_bank.jsonl` (program + proof metadata).",
        "`eq_certified_silver_train.jsonl` (prompt, canonical final answer only, provenance).",
        "`eq_certified_holdout.jsonl` (public-train holdout for gating only)."
      ],
      "how_to_avoid_weak_full_leakage": [
        "Hard denylist of weak/full prompt hashes before any processing.",
        "Train/holdout split created only within public train.",
        "No selection, filtering, or thresholding based on weak/full outcomes."
      ],
      "success_fail_numbers_for_gpu": {
        "allow_gpu_if": "All CPU gates above pass.",
        "block_gpu_if": "Any CPU gate fails."
      },
      "how_it_could_improve_equation_without_hurting_bit": "It gives high-precision equation targets instead of noisy broad SFT labels; no bit data touched yet, so no bit risk at this stage.",
      "why_different_from_failed_sft_preference": "This is not more generic SFT or model-output preference. It is executable program-certified supervision with explicit correctness guarantees.",
      "expected_risk_cost": "Risk: medium-high technical risk (DSL may still fail). Cost: low-to-medium CPU only."
    },
    {
      "rank": 2,
      "name": "Single Conservative LoRA Run: Certified Equation + Bit Anchor Replay",
      "hypothesis": "You can gain equation accuracy if updates are narrowly targeted and regularized against bit drift.",
      "exact_data_source_allowed": [
        "Equation certified silver from Experiment 1.",
        "Public-train bit rows where baseline output is label-correct (anchor replay set).",
        "Public-train holdout only for gating."
      ],
      "cpu_gate_before_gpu": {
        "must_pass": [
          "Equation certified set size >= 400 examples (after dedupe).",
          "Bit anchor set >= 800 examples with verified exact-match correctness.",
          "Output formatting/canonicalizer emits short final-answer targets only; p99 target length below truncation-risk threshold used in your infra."
        ],
        "block_if": [
          "Certified equation examples < 400.",
          "Bit anchor conflicts/non-determinism found.",
          "Target format shows truncation risk."
        ]
      },
      "artifact_to_build": [
        "`train_eq_certified_plus_bit_anchor.jsonl`.",
        "`lora_recipe_conservative.yaml` (low LR, short run, strong regularization, early kill at first checkpoint).",
        "Adapter package with only `adapter_config.json` + `adapter_model.safetensors`."
      ],
      "how_to_avoid_weak_full_leakage": [
        "No weak/full rows in train or validation.",
        "Model selection for this run uses public-train holdout first, then your existing weak gate only after pass.",
        "No hand-tuning from weak/full feedback loops before public-train gate pass."
      ],
      "success_fail_numbers_for_gpu": {
        "first_checkpoint_kill_switch": [
          "Require weak gate: equation >= 58, bit >= 136, truncation = 0.",
          "If not met, kill immediately."
        ],
        "promotion_gate": [
          "Keep your existing strict gate: total > 192/315, equation > 56/155 (ideal >= 60), bit >= 136/160, truncation = 0.",
          "Then official-like full must be > 823/947."
        ]
      },
      "how_it_could_improve_equation_without_hurting_bit": "Equation gets new high-precision supervision; bit is protected by explicit replay anchors and conservative update magnitude.",
      "why_different_from_failed_sft_preference": "Not broad/raw SFT and not preference over noisy model outputs. It is precision-targeted, short-answer training with explicit anti-regression anchors for bit.",
      "expected_risk_cost": "Risk: medium. Cost: one short H200 run (< 1 hour if kill-switch at first checkpoint)."
    },
    {
      "rank": 3,
      "name": "Executable Near-Miss Contrastive Set (Only if Experiment 2 misses equation target)",
      "hypothesis": "Equation errors are often 'wrong transform but plausible'; training on executable near-miss decoys can sharpen decision boundaries.",
      "exact_data_source_allowed": [
        "Public-train equation rows only.",
        "Decoys generated by mutating certified DSL programs from Experiment 1 (guaranteed wrong outputs).",
        "No weak/full usage."
      ],
      "cpu_gate_before_gpu": {
        "must_pass": [
          "Each pair has 1 certified-correct target and >= 2 executable wrong decoys.",
          "Decoys are label-verified wrong and syntactically plausible.",
          "On public-train holdout, baseline confuses correct vs decoy in >= 30% of cases (proves training opportunity)."
        ],
        "block_if": [
          "Decoy quality not guaranteed wrong.",
          "Confusion opportunity < 30% (little signal)."
        ]
      },
      "artifact_to_build": [
        "`eq_contrastive_triplets.jsonl`.",
        "Training config for contrastive objective compatible with LoRA-only output adapter."
      ],
      "how_to_avoid_weak_full_leakage": [
        "Same hard denylist and hashing controls.",
        "No weak/full-derived negatives or thresholds."
      ],
      "success_fail_numbers_for_gpu": {
        "allow_gpu_if": "Experiment 2 failed to reach equation >= 58 but maintained bit >= 136 and truncation 0, and this CPU gate passes.",
        "block_gpu_if": "Any bit risk or weak/full leakage risk appears."
      },
      "how_it_could_improve_equation_without_hurting_bit": "Focuses only on equation decision boundaries while keeping bit anchors from Experiment 2 in mixed batches.",
      "why_different_from_failed_sft_preference": "Unlike prior preference data from model misses, these negatives are programmatically controlled, semantically close, and guaranteed wrong.",
      "expected_risk_cost": "Risk: medium-high. Cost: medium CPU + one additional short GPU run only if gated."
    }
  ],
  "required_cpu_gates": [
    "Hard leakage gate: zero weak/full row overlap by normalized hash before dataset build.",
    "Certification gate: executable consistency and metamorphic invariance for equation labels.",
    "Quality gate: audited precision >= 98% for certified equation targets.",
    "Coverage gate: enough certified equation examples to justify adaptation (>= 400 suggested).",
    "Bit safety gate: verified bit anchor replay set with no label conflicts.",
    "Truncation gate: short-target format and p99 length below your known safe threshold."
  ],
  "no_go_conditions": [
    "Any proposal requiring runtime solver/verifier/postprocessor/parser at inference.",
    "Any training selection or thresholding based on weak/full labels.",
    "Any broad SFT re-run without new certified CPU signal.",
    "Any GPU run when certified equation coverage/precision gates fail.",
    "Any checkpoint with bit < 136 or truncation > 0 at weak gate.",
    "Any package that is not adapter-only (`adapter_config.json` + `adapter_model.safetensors`)."
  ],
  "changes_to_current_gates": [
    "Keep current FinOps and promotion gates.",
    "Add mandatory public-train holdout gate before weak gate for every new recipe.",
    "Add mandatory leakage hash report artifact per run.",
    "Add mandatory sequence-length/truncation-risk report before GPU launch.",
    "Enforce 'one-shot' policy: no second GPU run of same recipe without new CPU evidence."
  ],
  "one_hour_h200_precondition": "Spend the final 1-hour H200 run only if you have a certified equation dataset with >= 400 high-precision examples, audited precision >= 98%, zero weak/full overlap, a verified bit anchor replay set >= 800 examples, and a pre-registered first-checkpoint kill rule requiring weak equation >= 58, bit >= 136, truncation = 0. If any precondition is missing, do not run."
}
```

# qwen/qwen3.5-plus-20260420

variant: `qwen/qwen3.5-plus-20260420`

cost: `0.0143115` duration_ms: `128411` tokens: `7523`

{
  "verdict": "Q1: FinOps and promotion gates are structurally correct and operationally necessary. They correctly decouple loss metrics from accuracy, enforce strict truncation limits, and prevent label leakage. The only adjustment needed is adding a pre-GPU data yield gate (see changes_to_current_gates). Q3: The next move is a different data format (structured step-wise traces with explicit intermediate grounding), not a new loss, not a rigid DSL, and not abandoning LoRA. The bottleneck is sequential rule application, not model capacity or optimization objective.",
  "key_diagnosis": "Q2: The most likely missing ingredient is explicit intermediate-state grounding. The failure distribution (equation boxed_or_generation_wrong=79) proves the model attempts end-to-end symbolic mapping without internal verification anchors. Standard SFT/pref fails because it trains on flat output distributions, inheriting the model's tendency to hallucinate algebraic steps or violate operator precedence. Distilling CPU-verified step-by-step transformation traces with structural delimiters forces the adapter to learn rule application conditionally rather than pattern-matching final forms.",
  "ranked_next_experiments": [
    {
      "rank": 1,
      "hypothesis": "Equation hallucination stems from missing explicit intermediate state grounding. Providing verified step-by-step transformation traces with structural markers will force sequential rule learning instead of end-to-end pattern matching.",
      "exact_data_source_allowed": "Public train + allowed rows. CPU solver/verifier pipeline generates step sequences. Only 100% step-verified traces retained. Bit rows included at 30% ratio with identical formatting to prevent regression.",
      "cpu_gate_before_gpu": "Verify trace coverage (>140 eq rows, >150 bit rows), step count >=3 per trace, 100% step-wise correctness, tokenization safe under max_seq_len, format deterministic. Zero weak/full labels used.",
      "artifact_to_build": "adapter_config.json + adapter_model.safetensors trained on [EQ_STEP]...[/EQ_STEP] formatted sequences",
      "how_to_avoid_weak_full_leakage": "Pipeline ingests only train/public splits. Verification uses only local row constraints. No leaderboard submission, score history, or global statistics used. Evaluation strictly isolated to local weak/full script.",
      "success_fail_numbers": "Pass: weak > 192, eq >= 60, bit >= 135, trunc = 0. Fail: eq <= 58 OR bit <= 134 OR trunc > 0. Kill at ckpt-1 if trend negative.",
      "how_it_could_realistically_improve_eq_without_reducing_bit_below_136": "Bit accuracy relies on local pattern recognition which is robust to formatting changes. By keeping bit rows at 30% with identical delimiters and focusing adapter capacity on eq step-logic, we protect bit while injecting explicit algebraic grounding for eq. The structured format reduces cross-family capacity interference.",
      "why_it_is_different_from_failed_SFT_preference_attempts": "Past SFT used flat/raw generation or filtered outputs without structural scaffolding. Preference failed due to ambiguous reward signals and noisy hard-negatives. This isolates intermediate states with absolute CPU verification, forcing the model to learn transformation rules as a conditional sequence rather than a single-shot mapping or relative ranking.",
      "expected_risk_cost": "Risk: Medium (formatting might confuse tokenizer or shift distribution). Cost: CPU ~3h, GPU 0.05 USD (1 epoch, early kill)"
    },
    {
      "rank": 2,
      "hypothesis": "Model generates syntactically correct but algebraically invalid steps. Generating multiple traces per row and training exclusively on fully verified ones will purge hallucinated rule pathways from the adapter distribution.",
      "exact_data_source_allowed": "Train/public rows. CPU generator produces 5-10 traces/row. Verifier checks each step. Keep only rows where >=1 trace is 100% verified. Format identical to Exp 1.",
      "cpu_gate_before_gpu": "Yield rate > 75% for eq rows, > 85% for bit rows. Avg verified steps >= 4. Tokenization stable. Zero leakage. Confirm no test-time script dependencies.",
      "artifact_to_build": "adapter_config.json + adapter_model.safetensors trained on verified-only traces with step delimiters",
      "how_to_avoid_weak_full_leakage": "Strict train/public isolation. Verification purely local. No score leakage or global metrics used in curation.",
      "success_fail_numbers": "Pass: weak > 193, eq >= 59, bit >= 136, trunc = 0. Fail: eq <= 57 OR bit <= 135 OR trunc > 0.",
      "how_it_could_realistically_improve_eq_without_reducing_bit_below_136": "Directly targets the 79-generation-error rows by removing incorrect step distributions. Bit preservation guaranteed by parallel sampling at stable ratio and identical training protocol. Rejection sampling acts as a hard filter, unlike preference which optimizes relatively.",
      "why_it_is_different_from_failed_SFT_preference_attempts": "Preference attempted to teach via relative NLL/answer ranking on noisy targets, which optimized for fluency over correctness. This uses hard truth filters to curate a clean, high-signal dataset. The adapter learns from positive examples only, avoiding the gradient dilution caused by ambiguous negatives.",
      "expected_risk_cost": "Risk: Low-Medium (data curation overhead, but training signal is cleaner). Cost: CPU ~4h, GPU 0.04 USD"
    },
    {
      "rank": 3,
      "hypothesis": "Syntactic variance and non-standard bracket/term ordering in equations introduces distribution shift. Teaching the adapter to normalize -> transform -> denormalize reduces noise and focuses learning on pure algebraic rules.",
      "exact_data_source_allowed": "Train/public. CPU canonicalizer (sympy-like normalization, bracket stripping, constant folding). Verified lossless mapping. Bit rows unchanged.",
      "cpu_gate_before_gpu": "Verify >95% rows map losslessly to canonical form. Tokenizer handles [CANON] tags. Zero leakage. Confirm canonical form does not alter test-relevant semantics.",
      "artifact_to_build": "adapter_config.json + adapter_model.safetensors trained on [CANON]input[END][TRANS]steps[OUT]output[END]",
      "how_to_avoid_weak_full_leakage": "Standard isolation. Transformations purely local. No external metrics.",
      "success_fail_numbers": "Pass: weak > 192, eq >= 58, bit >= 136, trunc = 0. Fail: eq <= 56 OR bit <= 135 OR trunc > 0.",
      "how_it_could_realistically_improve_eq_without_reducing_bit_below_136": "Bit manipulation is mostly flag/shift based and less affected by canonical algebraic normalization. Balanced dataset prevents drift. By standardizing input syntax, the adapter reduces token-level variance, allocating capacity to learning the actual transformation rules.",
      "why_it_is_different_from_failed_SFT_preference_attempts": "Past attempts didn't separate structural formatting from rule learning. This explicitly teaches a three-stage pipeline (normalize, transform, denormalize), which mimics symbolic computation workflows. If the model can't handle canonical forms, it confirms the bottleneck is structural, not rule-based.",
      "expected_risk_cost": "Risk: High (canonical form might diverge from test distribution or confuse tokenizer). Cost: CPU ~2h, GPU 0.03 USD"
    }
  ],
  "required_cpu_gates": [
    "Any training data generation must pass 100% step-wise verification on CPU before GPU allocation.",
    "Trace formatting must be strictly delimited and tokenization-checked to prevent offset-mask drift or special token collision.",
    "Bit manipulation traces must be sampled at stable ratio (>=30%) to protect baseline 136/160 from capacity overwrite.",
    "No weak/full leaderboard scores, submission history, or global statistics used in data curation, metric calculation, or row selection.",
    "CPU yield/verification rate must be logged; if <70% for equation rows, abort experiment (indicates solver mismatch or unsolvable rows in current format, not model capacity)."
  ],
  "no_go_conditions": [
    "GPU allocation if CPU gate data yield < 70% for equation_transform or < 80% for bit_manipulation.",
    "Any experiment introducing runtime python, postprocessor, parser override, constrained decoding, or logit processor at inference.",
    "Training with weak/full labels, leaderboard metrics, or test-time adaptation signals.",
    "If ckpt-1 evaluation shows weak <= 192 OR eq <= 56 OR bit <= 134 OR trunc > 0, kill immediately.",
    "Adapter config or safetensors missing standard keys (peft_type, base_model_name_or_path, target_modules, r, alpha, etc.) or exceeding max_rank/context."
  ],
  "changes_to_current_gates": "Q1: Current gates are operationally correct for FinOps and promotion. Recommended addition: Enforce a 'step-verification yield' gate before GPU. If CPU solver/verifier yields <70% valid traces for equation rows, do not train (indicates format/solver mismatch rather than model capacity). Also, explicitly cap bit_manipulation ratio in training data to 30-40% to prevent the observed 136->134 regression from capacity overwrite. Add a tokenizer safety check for any new delimiters before GPU spend.",
  "one_hour_h200_precondition": "Q4: I will spend exactly one H200 hour only if the CPU pipeline produces a curated dataset with: (1) >= 140 equation_transform rows with 100% verified step-wise traces, (2) >= 150 bit_manipulation rows with identical formatting, (3) tokenization verified safe and offset-mask aligned, and (4) a dry-run of the training script confirms exact offset-mask alignment, standard LoRA config compatibility, and zero runtime dependencies. If these 4 conditions are met, the H200 run uses 1 epoch, lr=2e-4, batch=16, with a hard kill at ckpt-1 if weak eval fails >192 or eq fails >=58."
}

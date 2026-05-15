# qwen/qwen3.6-max-preview

variant: `qwen/qwen3.6-max-preview-20260420`

cost: `0.051818` duration_ms: `227769` tokens: `7878`

```json
{
  "verdict": "PROCEED WITH EXTREME CAUTION. Current distillation failures indicate a structural mismatch between answer-level training and the multi-step symbolic reasoning required for equation_transform. Blanket LoRA updates are causing feature interference (bit regression) and failing to encode rewrite rules. The next phase must shift from final-answer distillation to process-level supervision with strict interference isolation. Any run that repeats answer-only SFT, preference on hard negatives, or blanket layer updates is predicted to fail and should be blocked.",
  "key_diagnosis": [
    "Equation_transform requires multi-step symbolic rewriting; final-answer SFT only teaches superficial input-output mapping, not the underlying operator rules.",
    "Blanket LoRA across all transformer layers causes catastrophic interference with low-level features responsible for bit_manipulation and formatting, explaining consistent bit drops to 134/135.",
    "Preference/hard-negative training failed because it punishes wrong outputs without providing the correct procedural pathway. The adapter learns what not to do, but not how to transform.",
    "CPU solvers succeed because they execute explicit symbolic rules; LoRA fails because the autoregressive decoder never observed intermediate rule applications during training.",
    "Loss/eval metrics are decoupled from family ACC because the model optimizes token likelihood on superficial patterns rather than structural rule correctness."
  ],
  "ranked_next_experiments": [
    {
      "id": "EXP-1",
      "title": "Process-Explicit SFT with Symbolic Derivation Traces & Selective Loss Masking",
      "hypothesis": "Training on CPU-generated step-by-step derivation traces forces the adapter to internalize transformation rules. Masking loss on intermediate reasoning tokens prevents overfitting to trace syntax while preserving rule learning in the weight update.",
      "data_source_allowed": "Public train equation rows only. SymPy-generated step-by-step derivation traces formatted to exactly match competition output style. Zero weak/full labels used.",
      "cpu_gate_before_gpu": "1. SymPy verifies 100% trace mathematical correctness. 2. Tokenized sequence length strictly < max_context - prompt_length. 3. Exact tokenization/offset alignment with baseline. 4. Public proxy eval (held-out public split) shows >= +3 equation ACC over baseline. 5. vLLM offline load dry-run passes.",
      "artifact_to_build": "SFT dataset with explicit intermediate steps. PEFT adapter trained with `train_on_inputs=False` and custom masking that computes loss only on rule-transition tokens and the final answer block.",
      "leakage_avoidance": "Only public train rows used. Synthetic traces generated via deterministic symbolic engine. Row hashes logged and intersected against weak/full hashes to guarantee zero overlap.",
      "success_fail_numbers": "SUCCESS: Public proxy equation >= 60, bit >= 136, trunc=0. FAIL: Public proxy equation <= 57, bit < 134, or trunc > 0. Blocks GPU immediately if gate fails.",
      "realistic_improvement_path": "Teaches the procedural pathway (rule application sequence) instead of just the destination. Selective masking prevents the adapter from overwriting bit/formatting weights with trace-syntax noise.",
      "difference_from_failed_attempts": "Past SFT used raw/reconstructed answers or high-confidence rows without intermediate steps. This provides the missing procedural signal that CPU solvers use, directly addressing the 'boxed_or_generation_wrong' failure class.",
      "expected_risk_cost": "CPU trace generation: ~$2-5. GPU training: ~$15-25 (1h H200). Risk: Trace format mismatch causing generation drift. Mitigation: Strict format alignment, dry-run generation test, and 70/30 real/synthetic mix."
    },
    {
      "id": "EXP-2",
      "title": "Layer-Selective LoRA Targeting Mid-Reasoning Blocks",
      "hypothesis": "MoE models encode symbolic manipulation in mid-to-late transformer layers. Restricting LoRA to these layers prevents overwriting early-layer features responsible for bit_manipulation and token routing, preserving bit>=136 while allowing equation learning.",
      "data_source_allowed": "Same process-trace dataset from EXP-1, or filtered high-confidence public equation rows if EXP-1 data pipeline is delayed.",
      "cpu_gate_before_gpu": "1. vLLM dry-run confirms `layers_to_transform` or layer-restricted `target_modules` loads and generates without error. 2. Parameter count <= baseline adapter. 3. Config matches official vLLM LoRA schema exactly.",
      "artifact_to_build": "PEFT config with `layers_to_transform: [12, 13, ..., 22]` (exact range mapped to Nemotron-3-Nano architecture). Standard SFT objective on process data.",
      "leakage_avoidance": "Identical to EXP-1. No label-based filtering. Hash-audited.",
      "success_fail_numbers": "SUCCESS: Weak equation > 58, bit >= 136, trunc=0. FAIL: Bit < 135 or equation <= 56. Immediate kill if trunc > 0 at checkpoint-1.",
      "realistic_improvement_path": "Reduces update interference. Bit manipulation relies on low-level pattern routing; equation transform relies on mid-level symbolic abstraction. Isolating updates protects bit features from gradient noise.",
      "difference_from_failed_attempts": "All past runs used blanket LoRA across all layers. This explicitly constrains the update subspace to prevent the consistent bit regression observed in V398/V416/V444.",
      "expected_risk_cost": "CPU config validation: $0. GPU training: ~$10-20. Risk: vLLM LoRA path may reject layer-restricted configs depending on version. Mitigation: Mandatory offline load test before any training."
    },
    {
      "id": "EXP-3",
      "title": "Rule-Invariant Synthetic Clustering (Structural Augmentation)",
      "hypothesis": "Equation failures stem from poor rule generalization. Training on structurally isomorphic variants (same operator AST, different constants/variables) forces the adapter to learn abstract transformation rules rather than row-specific patterns.",
      "data_source_allowed": "Public equation rows. CPU generates 3-5 structurally identical variants per row via symbolic substitution. All variants CPU-verified.",
      "cpu_gate_before_gpu": "1. Structural isomorphism verified via AST comparison. 2. 100% mathematical correctness. 3. Distribution shift check: token overlap with baseline < 15%. 4. Public proxy shows improved generalization on held-out variants.",
      "artifact_to_build": "Clustered SFT dataset. Standard causal LM loss. No preference/DPO. Mixed 70% real public / 30% synthetic variants to anchor generation style.",
      "leakage_avoidance": "Strictly public/synthetic. No weak/full labels used for selection or training. Hashes tracked and audited.",
      "success_fail_numbers": "SUCCESS: Public proxy equation >= 59, bit >= 136. FAIL: Equation <= 56 or bit < 135. Blocks GPU.",
      "realistic_improvement_path": "Forces rule abstraction by exposing the model to multiple instances of the same underlying transformation. Directly targets the 'answer_literal_nonfinal_or_ambiguous' failure class by stabilizing rule application across variants.",
      "difference_from_failed_attempts": "Past SFT lacked structural diversity. V443's certified builder failed because it looked for exact string/slot matches. This uses symbolic AST substitution to guarantee rule consistency without string fragility.",
      "expected_risk_cost": "CPU generation: ~$5-8. GPU: ~$15. Risk: Synthetic distribution mismatch causing generation style drift. Mitigation: Strict style anchoring and proxy gate."
    }
  ],
  "required_cpu_gates": [
    "1. 100% symbolic/mathematical verification of all training targets via SymPy or equivalent deterministic engine.",
    "2. Tokenized sequence length strictly < max_context - prompt_length. Zero truncation allowed in training data.",
    "3. vLLM offline load dry-run: adapter_config.json + adapter_model.safetensors must load and run 5 dummy prompts without error before any weak eval.",
    "4. Public proxy evaluation gate: held-out public split must show equation >= +3 over baseline AND bit >= 136 before touching weak set.",
    "5. Exact row hash/family count audit. Zero weak/full set overlap. Leakage check via set intersection.",
    "6. Tokenization/offset-mask alignment verified against baseline V290/V291 pipeline."
  ],
  "no_go_conditions": [
    "Any experiment using weak/full labels for row selection, filtering, weighting, or training.",
    "Any runtime verifier, postprocessor, parser, logit processor, constrained decoding, or test-time script.",
    "Training on final answers without intermediate process traces (repeats proven failures).",
    "Blanket LoRA across all layers without interference isolation (high probability of bit regression).",
    "Preference/DPO/hard-negative training without explicit positive procedural signal.",
    "GPU launch without passing public proxy ACC gate and vLLM load dry-run.",
    "Any adapter config not strictly compliant with official vLLM LoRA schema (must load via standard path).",
    "Continuing past checkpoint-1 if trunc > 0 or bit < 135, regardless of loss."
  ],
  "changes_to_current_gates": [
    "Add 'Public Proxy ACC Gate': Require >= +3 equation improvement on held-out public data before weak evaluation. Prevents weak-set overfitting and saves GPU.",
    "Add 'vLLM Offline Load Dry-Run': Mandatory config+tensor load test with dummy generation before any evaluation. Eliminates submission-format failures.",
    "Replace loss-based early stopping with 'Truncation/Bit Floor Kill-Switch': If checkpoint-1 shows trunc > 0 or bit < 135, kill immediately. Loss is decoupled from ACC and must be ignored.",
    "Explicitly ban blanket LoRA configs; mandate layer/module targeting or process-masked SFT for all future runs to prevent interference."
  ],
  "one_hour_h200_precondition": "A CPU-verified process-trace dataset of >= 500 equation rows with 100% SymPy correctness, tokenized length strictly within context limits, and a successful vLLM offline load dry-run of the exact adapter config. Additionally, a public proxy evaluation must demonstrate equation ACC >= 59 and bit ACC >= 136 on a held-out public split. Without these three conditions met, the H200 hour will likely repeat the distillation failure and is not justified under FinOps constraints."
}
```

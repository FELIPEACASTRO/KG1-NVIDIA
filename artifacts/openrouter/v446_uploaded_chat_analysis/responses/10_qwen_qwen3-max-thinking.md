# qwen/qwen3-max-thinking

variant: `qwen/qwen3-max-thinking-20260123`

cost: `0.00891852` duration_ms: `46925` tokens: `1796`

{
  "verdict": "Proceed with extreme caution—only CPU-gated, symbolically grounded, minimal-distillation experiments are viable.",
  "key_diagnosis": "The equation_transform family fails due to lack of compositional symbolic generalization in the LoRA. Current SFT/preference data lacks explicit, verifiable, minimal rule traces that mirror how humans solve such equations (e.g., operator precedence, variable elimination, distributive simplification). Bit manipulation succeeds because patterns are local and low-entropy; equations require global algebraic awareness. Distillation fails because we train on *answers*, not *provable reasoning traces*. The missing ingredient is not more data—it is data with **certified, stepwise symbolic justifications** aligned with the model’s token generation path.",
  "ranked_next_experiments": [
    {
      "name": "V446A: Certified minimal-equation trace distillation (LoRA-only)",
      "hypothesis": "If we distill only from examples where a lightweight CPU solver can produce a **minimal, deterministic, token-aligned symbolic trace** (e.g., 'a(b+c) → ab+ac' with explicit token offsets matching model generation), then LoRA can learn equation manipulation without harming bit tasks.",
      "exact_data_source_allowed": "Only 120 equation rows missed by V290 ckpt-6 (from V435D), filtered to those where a new CPU trace builder (V446B) produces a **single, verifiable, non-ambiguous symbolic rewrite sequence** that aligns with the model’s generation window (≤5 tokens of deviation). Max 40 rows.",
      "cpu_gate_before_gpu": "V446B CPU trace builder must output ≥15 rows with trace_certainty=1.0 (no ambiguity, one canonical path), and when injected as SFT targets, must yield **offline adapter simulation ≥60/155 on equation weak** (via vLLM dry-run with adapter weights frozen) **without dropping bit below 136 or introducing truncation**.",
      "artifact_to_build": "adapter_config.json + adapter_model.safetensors trained only on the certified trace rows, using the same tokenization/offset-mask as baseline. No new template—use official prompt format verbatim.",
      "avoid_weak_full_leakage": "Data sourced exclusively from public train rows already in weak set (no val/test leakage). Labels derived only from CPU-certified symbolic rules, not from ground truth or oracle answers.",
      "success_fail_numbers": "CPU gate success: equation ≥60/155, bit ≥136, trunc=0 in dry-run. Fail: any truncation, bit<136, or equation≤56.",
      "why_different_from_failed_attempts": "Prior SFT used final answers or high-confidence projections without stepwise symbolic grounding. This uses **provably minimal rewrite paths** aligned to token generation, mimicking how the model must incrementally produce correct output. Not answer distillation—**process distillation**.",
      "expected_risk_cost": "CPU cost: <0.5 USD. GPU cost: 0 if CPU gate fails. If passes, H200 ≤45 min (~4 USD). Risk: medium—trace builder may yield <15 rows."
    },
    {
      "name": "V447A: Bit-equation invariant preservation fine-tuning",
      "hypothesis": "Equation performance is harmed because LoRA updates interfere with shared low-rank subspaces used for bit tasks. By constraining gradient updates on layers critical for bit (via CPU-identified sensitivity map), we can safely improve equation without bit regression.",
      "exact_data_source_allowed": "Same 40-row certified trace set from V446A (if available), or fallback to 20 highest-certainty equation rows from V435D with answer confidence >0.99 from CPU solver.",
      "cpu_gate_before_gpu": "Layer sensitivity audit: run per-layer LoRA ablation on bit family using V290 ckpt-6. Identify top 3 layers whose modification causes bit drop. CPU gate passes only if proposed training freezes those layers and offline dry-run shows bit ≥136, equation ≥58.",
      "artifact_to_build": "LoRA adapter with layer-specific rank/r=0 on sensitive layers (i.e., no update). Standard config otherwise.",
      "avoid_weak_full_leakage": "All data from public weak train rows. No use of full set labels.",
      "success_fail_numbers": "CPU dry-run: equation ≥58, bit ≥136, trunc=0. Fail: bit<136 or trunc>0.",
      "why_different_from_failed_attempts": "Previous runs updated all layers uniformly. This uses **structural protection** of bit-critical pathways, decoupling interference.",
      "expected_risk_cost": "CPU: ~1 USD for sensitivity map. GPU: ≤30 min if gate passes (~2.7 USD). Risk: high—may still not move equation if core issue is representational, not interference."
    },
    {
      "name": "V448A: Equation-first LoRA warm start from bit-frozen base",
      "hypothesis": "Starting LoRA training from a checkpoint where bit performance is locked (via LoRA merging or freezing) creates a stable foundation for equation learning.",
      "exact_data_source_allowed": "Only certified equation traces (≤30 rows). Base model = V290 ckpt-6 with bit-performing LoRA merged into weights (CPU-verified bit=136/160). Then apply new LoRA only on equation data.",
      "cpu_gate_before_gpu": "Merged base must pass bit=136 offline. New LoRA dry-run must show equation ≥57 without bit drop.",
      "artifact_to_build": "Two-stage: (1) merged base (not submitted), (2) small LoRA on top (submitted). Only (2) is in adapter package.",
      "avoid_weak_full_leakage": "Merging uses only adapter weights + base model—no data leakage.",
      "success_fail_numbers": "Equation ≥57, bit ≥136, trunc=0 in end-to-end dry-run with two-stage sim.",
      "why_different_from_failed_attempts": "Decouples representation stabilization (bit) from task learning (equation). Prior attempts trained jointly, causing destructive interference.",
      "expected_risk_cost": "CPU: ~1.5 USD. GPU: ≤40 min (~3.6 USD). Risk: very high—vLLM may not support two-stage LoRA loading as per rules."
    }
  ],
  "required_cpu_gates": [
    "All candidate rows must originate from public train set already in weak evaluation",
    "All labels must be derived from deterministic CPU-certified symbolic traces—not ground truth, not confidence scores",
    "Offline dry-run using official vLLM LoRA loading path must show: total weak >192, equation >56, bit ≥136, trunc=0",
    "Tokenization and offset-mask must match baseline exactly—verified by hash",
    "No use of full set (947 rows) in any CPU or GPU step"
  ],
  "no_go_conditions": [
    "Any experiment proposing training on full set or using full labels",
    "Any SFT using final answers without stepwise symbolic justification",
    "Any preference/hard-negative data without certified trace alignment",
    "Any change to prompt template or decoding logic",
    "Any GPU launch without passing CPU dry-run gate",
    "Any artifact that cannot be loaded by standard vLLM LoRA path as two files"
  ],
  "changes_to_current_gates": "Strengthen CPU gate to require **symbolic trace certification**, not just answer correctness. Add dry-run verification of adapter-only decode path before GPU. Keep all other FinOps rules.",
  "one_hour_h200_precondition": "Only one H200 hour remains if and only if: (1) CPU trace builder produces ≥15 certified equation rows with unique, minimal symbolic paths; (2) offline vLLM dry-run with proposed adapter shows equation ≥60/155, bit ≥136/160, total weak ≥195/315, trunc=0; (3) tokenization/offset-mask matches baseline hash; (4) adapter loads in vLLM without error. No exceptions."
}

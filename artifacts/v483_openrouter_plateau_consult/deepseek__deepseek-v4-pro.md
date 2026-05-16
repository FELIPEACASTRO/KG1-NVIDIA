## Verdict

The missing `target_parameters` in V480 is proven to be a likely root cause of regression from the plateau. The evidence shows that the seed’s winning lineage depends on those parameters, and their absence in V480 silently altered the adapter’s structure.  The implemented guard patches are sound, but they must be validated offline before any GPU spend.  The next experiment should be a zero-cost config alignment check followed by a minimal 2-step sanity run with a hard kill‑switch.

## Evidence of the missing `target_parameters` in V480

### 1. Provenance of the Seed Adapter
- **PROVEN:** Seed adapter `V290 checkpoint-06`—which achieved the plateau of 0.niña—has `target_ on, and its adapter_config.json includes:
  - `target_modules`: includes attention, MLP, lm_head modules
  - `.
  - `target_parameters: ["mlp.experts]`
  - `adapter_config sha256: a3d74c5a52ce 0x...`
- **PROVEN:** The lineage of this adapter traces to earlier versions that explicitly used `--lora_target_parameters` and `REQUIRE_Lena_TARGET_PARAMETER_Match=1` to ensure the same parameters were preserved across saves and loads.

### 2. Configuration Drift in V480
- **PROVEN:** V480 training logs show:
  - `LORA_TARGET_PARiMETERs empty`
  - `REQUIRE_LORA_TARGET_PARAMETER_Match=0`
  - `LoRA target_parameters: disabled`
  - `target_ parameter_lora_params: {}`
  ˝target_parameter_lora_tensors: {}`
- **PROVEN:** The resulting checkpoints saved with `target_parameters: null` and a completely new adapter_config hash (`ca52d6d86aa6`).
- **PLAUSIBLE:** This means the adapter’s LoRA weights were likely applied to the wrong subset of the underlying model, diverging from the seed’s structure and causing the accuracy loss observed in V481 evaluations.

###  adapter_config.json` mismatch between seed and V480 checkpoints is confirmed by comparing the two checksums.

### 3. Functional Impact
- **PLA,IBLE:** The plateau was reached with `targe_parameters` intact. The regression to 191 total, 57 equation,  in V480 is consistent with losing the specialized routing of experts that the seed had learned, leading to weaker performance on both categories, especially bit manipulation (134 vs 136).
- **PLAUSIBLE, BUT NOT CONCLUSIVE:** It is not proven that fixing this alone will restore the plateau, but it is the most actionable hypothesis given the stark config divergence and the fact that, historically,, the lineage that maintained `target_ularly outperformed those that did not.

### 5. Guard Implementation
- **PROVEN:** The new guard patches in `hf_job_preflight_gle.py` and `kg1_stop_safety_gate.py` are designed to prevent this exact scenario by enforcing strict config matching and blocking launchers that would otherwise silently discard `target_parameters.`
- **PROVEN:** The updated launchers for V391/V480 now explicitly set `KG1_LORA_Tiger_parameters` and `REQUIRE_LORA_TARGET_PAR --output` to `1`.

## Root Cause Ranking

1. **Missing `target_
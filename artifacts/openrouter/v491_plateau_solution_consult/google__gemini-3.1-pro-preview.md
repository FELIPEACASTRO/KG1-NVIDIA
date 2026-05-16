## 1. Verdict
The plateau is caused by **gradient interference and capacity bottlenecking**, compounded by `lm_head` corruption. V487 proved that training only attention + `lm_head` (`q/k/v/o/lm_head`) forces the model to overwrite established `bit_manipulation` pathways to memorize `equation_transform` traces, while shifting `lm_head` causes EOS failure (truncation). 

Equation-first is the correct strategic route, but it cannot succeed structurally until the MoE experts (`up_proj`, `down_proj`) are explicitly trainable to provide isolated capacity for the new reasoning traces. The immediate next step is a highly constrained, 4-step micro-SFT that freezes `lm_head` and trains the MoE experts, gated by a step-2 kill switch.

## 2. Root Cause Ranking
1. **Catastrophic Forgetting via Capacity Limits (85%)**
   * *Evidence*: V488 gained 1 equation row but lost 2 bit rows. V487 launcher (lines 131-133) explicitly excluded `up_proj` and `down_proj` from `TRAINABLE_LORA_MODULES`, leaving MoE parameters `frozen_active`.
   * *Falsification*: Train with `up_proj,down_proj` in the allowlist. If bit still regresses, the interference is in the dataset traces, not the parameter capacity.
2. **EOS Corruption via `lm_head` Training (75%)**
   * *Evidence*: V488 introduced 1 truncation. Tokenization gate confirms `completion_truncation=0` in the dataset, meaning the model forgot how to stop generating. `lm_head` was trainable in V487.
   * *Falsification*: Freeze `lm_head`. If truncation drops back to 0, `lm_head` was the culprit.
3. **Over-weighted Answer Span Masking (60%)**
   * *Evidence*: Roadmap notes `answer_span_loss_weight=12.0` as a recurring risk. Code lines 1268-1274 apply this multiplier. High weights force the model to memorize the final token rather than the reasoning trace, lowering `eval_loss` without improving ACC.
   * *Falsification*: Set `ANSWER_SPAN_LOSS_WEIGHT=1.0`. If `eval_loss` rises but ACC stabilizes, the multiplier was causing format collapse.
4. **Dataset Trace Generalization Failure (40%)**
   * *Evidence*: V390/V326 contains 800 equation rows distilled without loss. The model may be memorizing these specific 800 traces rather than learning the underlying rule.
   * *Falsification*: If the MoE-enabled smoke test fails to generalize to the validation set (equation stays at 56), the traces themselves lack the necessary intermediate logic.
5. **MoE Router Representation Drift (UNKNOWN)**
   * *Evidence*: Not explicitly detailed in the provided manifests whether the Nemotron router is frozen or trainable, but standard PEFT freezes it. If experts update but the router doesn't, routing distribution may drift.
   * *Falsification*: Cannot easily falsify without custom router logging, but standard LoRA practice assumes frozen routers are acceptable for small `r`.

## 3. Implementation Bugs or Gaps To Check
* **Gap in V487 Launcher `TRAINABLE_LORA_MODULES`**: Lines 131-133 of the V487 launcher set `TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'`. Even though `LORA_TARGET_PARAMETERS` included the MoE experts, the `apply_trainable_lora_module_filter` (lines 704-731) requires the module name to be in the CSV to set `requires_grad_(True)`. This is why V487 reported them as `frozen_active`.
* **Truncation Risk from `lm_head`**: Training `lm_head` in PEFT frequently degrades the EOS token probability unless the dataset perfectly matches the base model's EOS distribution.
* **Answer Span Weighting Logic**: Lines 1258-1274 apply `ANSWER_SPAN_LOSS_WEIGHT`. If this is set to 10.0 or 12.0, the gradients from the reasoning steps (weight 1.0) are dwarfed by the final answer, destroying the Chain-of-Thought transfer.

## 4. Exact Next Experiment
Run a minimal HF smoke test (V491) to prove MoE trainability without `lm_head` corruption.

* **Trainable Modules**: `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj` *(Remove `lm_head`)*
* **Target Parameters**: `mlp.experts.gate_up_proj,mlp.experts.down_proj`
* **Env Flags**: `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
* **Hyperparameters**: 
  * `LEARNING_RATE=2.0e-8` *(Lower than V487 to protect bit)*
  * `MAX_STEPS=4`
  * `SAVE_EVERY_STEPS=2`
  * `EVAL_EVERY_STEPS=2`
* **Loss Weight**: `ANSWER_SPAN_LOSS_WEIGHT=1.0` *(Flat weighting to force reasoning trace learning)*
* **Dataset**: V390/V326 (Same as V487 to isolate the variable).
* **First-Checkpoint Kill-Switch (Step 2)**: 
  * Abort immediately if `bit_manipulation < 136`.
  * Abort immediately if `truncated > 0`.
  * Abort immediately if `equation_transform < 57`.
* **Expected Cost/Risk**: Very low. Maximum 4 steps on H200.
* **Expected Outcome**: 
  * *Best-case*: Total 193, eq 57, bit 136, trunc 0. Proves MoE capacity solves the interference.
  * *Worst-case*: Total 191, eq 56, bit 135. Proves the dataset traces themselves cause forgetting, killing the broad SFT route.

## 5. Alternative If That Experiment Fails
If V491 fails the step-2 kill switch, **abandon broad SFT**. 
1. Extract the exact 3 rows that flipped in V488 (`518deb39` [eq gain], `8740ed31` [bit loss], `59bee375` [bit loss/trunc]).
2. Generate 10-20 CPU-only traces for these specific IDs using the V475 CPU solver projection.
3. Construct a micro-dataset (under 50 rows) of hard-negatives and verified chosen traces for just these edge cases.
4. Run a targeted overfit test to see if the adapter *can* physically represent the correct bit and equation answers simultaneously without OOM or metric collapse.

## 6. Stop Doing
* **Stop training `lm_head`**. It is the most likely cause of the `truncated=1` regression in V488.
* **Stop using `ANSWER_SPAN_LOSS_WEIGHT > 1.0`**. It masks whether the model is actually learning the reasoning trace and artificially lowers `eval_loss`.
* **Stop running jobs beyond 2-4 steps without a passing micro-ACC gate**. `eval_loss` is completely decoupled from strict string-match ACC.

## 7. Roadmap Patch
Insert these exact bullets into the active roadmap:
* **MoE Trainability Requirement**: Any job using `LORA_TARGET_PARAMETERS` must explicitly include `up_proj,down_proj` in `TRAINABLE_LORA_MODULES` and `REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS`.
* **Freeze `lm_head`**: `lm_head` must be removed from all trainable allowlists to prevent EOS token corruption and truncation regressions.
* **Flat Loss Weighting**: `ANSWER_SPAN_LOSS_WEIGHT` must be locked to `1.0`. Do not artificially inflate final-answer gradients until reasoning traces are proven to transfer.
* **Micro-ACC Kill Switch**: All future GPU jobs must evaluate weak ACC at step 2. The job must automatically terminate if `bit < 136` or `truncated > 0`.
* **Dataset Pivot Trigger**: If the MoE-enabled smoke test fails to hold `bit=136` while gaining `equation`, broad SFT is permanently blocked. Pivot exclusively to micro-batch hard-negative trace rewrites.

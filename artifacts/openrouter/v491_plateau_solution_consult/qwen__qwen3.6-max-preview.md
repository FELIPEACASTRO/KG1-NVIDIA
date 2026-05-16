## 1. Verdict
The primary blocker is **frozen MoE `target_parameters` during training combined with excessive answer-span loss weighting**. V487/V488 trained only attention/head modules (`q/k/v/o/lm_head`) while MoE projections (`up_proj`/`down_proj` aliases) remained `frozen_active`. This forces gradients to push against frozen expert features, destabilizing precise bit outputs (causing -2 regression and truncation) while artificially lowering loss on the weighted answer span. Equation-first is the correct direction, but only if MoE LoRA tensors are explicitly trainable and the loss weight is reduced to prevent gradient conflict. Run a tightly gated micro-smoke with MoE trainability enabled, reduced span weight, and a hard first-checkpoint kill-switch. If bit regresses or truncation appears, abort immediately.

## 2. Root Cause Ranking
1. **Frozen MoE `target_parameters` during training** (Confidence: 85%). Evidence: V487 env used `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` and excluded `up_proj/down_proj` from `TRAINABLE_LORA_MODULES`. V489 audit confirms `target_parameters_trainability_mode=frozen_active`. Falsify: Enable MoE trainability in micro-smoke; if equation improves without bit regression, confirmed.
2. **Excessive `ANSWER_SPAN_LOSS_WEIGHT` masking ACC** (Confidence: 75%). Evidence: Roadmap flags `12.0` as a recurring risk. High weight on final tokens with frozen MoE features creates gradient conflict in attention layers, directly correlating with V488's truncation and bit regression. Falsify: Reduce to `3.0`; monitor if truncation disappears and bit stabilizes at ckpt-2.
3. **Attention-only updates destabilizing bit precision** (Confidence: 65%). Evidence: V488 lost rows `8740ed31` and `59bee375` (bit) while gaining 1 equation. Bit manipulation requires exact token-level control; perturbing attention/routing without adapting expert transformations breaks it. Falsify: Compare gradient norms per module; if attention norms dominate while MoE stays zero, bit drift is expected.
4. **Generation distribution shift causing truncation** (Confidence: 50%). Evidence: V488 `truncated=1` on `59bee375`. High loss weight + frozen features can disrupt EOS probability or reasoning length. Falsify: Check raw token length for `59bee375` in V488 vs baseline. If length increased >20%, it's generation drift, not data truncation.
5. **Dataset objective imbalance despite V486 probe** (Confidence: 30%). Evidence: V486 showed effective bit share ~0.21 vs equation ~0.79. Equation dominance may starve bit gradients. Falsify: Switch to uniform sampling or cap equation weight at 0.5; check if bit ACC holds.

## 3. Implementation Bugs or Gaps To Check
- **Critical Gap:** `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE` is a validation flag only. In `apply_trainable_lora_module_filter` (lines 714-733), `requires_grad_(True)` is toggled **only** if a parameter matches `TRAINABLE_LORA_MODULES` or `TRAINABLE_LORA_NAME_SUBSTRINGS`. Setting the flag to `1` without adding `up_proj,down_proj` to those lists will either crash the job or leave MoE frozen. The code does not auto-enable trainability for `target_parameters`.
- **Loss Weight Risk:** If `ANSWER_SPAN_LOSS_WEIGHT=12.0` was active in V488, it applies 12x gradient to final tokens. With frozen MoE, this forces unstable updates in attention/head layers, explaining the truncation and bit regression. Verify the exact env value used.
- **Truncation Guard Missing:** Weak eval counts truncation but does not log raw output length or EOS token probability. Add a length delta check to distinguish format drift from hard token limits.
- **Matcher Alias Verification:** V487 fixed `mlp.experts.gate_up_proj` → `mixer.experts.<id>.up_proj` aliasing. Verify `target_parameter_name_matches` correctly resolves all expert IDs. A silent partial match would leave some experts frozen, causing inconsistent equation behavior.
- **Sampling Mode:** `SAMPLING_MODE='weighted_replacement'` with high equation weights reproduces the V391 imbalance pattern. Confirm effective loss share per family before GPU allocation.

## 4. Exact Next Experiment
- **Name:** V491 MoE-Trainable Micro-Smoke
- **Trainable Config:** `TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head,up_proj,down_proj'`. Set `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`. Patch launcher to explicitly log `target_parameters_trainability_mode=trainable` before step 1.
- **LR/Steps:** `LEARNING_RATE=2.0e-8` (lower to stabilize MoE), `MAX_STEPS=6`, `SAVE_EVERY_STEPS=2`, `EVAL_EVERY_STEPS=2`.
- **Loss Weight:** `ANSWER_SPAN_LOSS_WEIGHT=3.0`. Log gradient norms per module to detect conflict.
- **Dataset Mix:** Use V390/V326 but switch to `SAMPLING_MODE='uniform'` or cap equation effective weight at `0.5`. Prevent equation dominance from starving bit gradients.
- **First-Checkpoint Kill-Switch (Checkpoint-2):** Run weak eval on 315 rows. Gate: `total > 192`, `equation > 56`, `bit >= 136`, `truncated == 0`. If ANY condition fails, `sys.exit(1)` immediately. Do not proceed to step 4.
- **Cost/Risk:** Low. ~15-20 mins on H200. Risk: VRAM pressure from trainable MoE; mitigate with `MICRO_BATCH_SIZE=1` and gradient checkpointing (already enabled).
- **Expected Outcomes:** Best-case: equation 57-58, bit 136, trunc 0, total 193+. Worst-case: bit <136 or trunc >0 at ckpt-2 → immediate abort, confirming MoE trainability alone requires different LR/data balance or rank scaling.

## 5. Alternative If That Experiment Fails
- **Path:** CPU hard-negative distillation + targeted micro-adapter.
- **Steps:**
  1. Extract exact failure rows: V488 regressions (`8740ed31`, `59bee375`) and baseline equation misses.
  2. Run CPU teacher to generate minimal, verified reasoning traces (max 3 steps) + exact boxed answers for these rows only.
  3. Construct a tiny dataset (~50-80 rows) of these hard negatives + 50 bit replay anchors.
  4. Run a CPU probe: freeze all LoRA except `lm_head` and one expert routing parameter; test if 10-20 gradient steps flip the specific token probabilities without broad SFT.
  5. If CPU probe shows correct token probability shift >15% with zero bit regression, package as targeted micro-adapter. If not, the plateau is architectural/rank-limited for this adapter; consider `r=64` only if VRAM allows, or accept baseline. No further broad SFT.

## 6. Stop Doing
- Stop running H200 jobs with `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` or frozen MoE `target_parameters`.
- Stop using `ANSWER_SPAN_LOSS_WEIGHT > 5.0` without proof it improves discrete ACC; it is causing gradient masking and truncation.
- Stop evaluating only `eval_loss` or waiting until checkpoint-10 for weak ACC. Enforce first-checkpoint micro-ACC gate.
- Stop broad SFT/replay on the full 5k dataset without a proven transfer mechanism. It burns GPU and regresses bit.
- Stop assuming `target_parameters` become trainable automatically via config flags. The code requires explicit module/substring matching to toggle `requires_grad`.

## 7. Roadmap Patch
- **P3 Update:** Mandate `TRAINABLE_LORA_MODULES` must include `up_proj,down_proj` (or equivalent substrings) when `LORA_TARGET_PARAMETERS` is set. `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1` is now a hard preflight gate.
- **Code Fix:** Patch `apply_trainable_lora_module_filter` to explicitly toggle `requires_grad_(True)` for parameters matching `LORA_TARGET_PARAMETERS` when `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`, independent of `TRAINABLE_LORA_MODULES`.
- **Loss Weight Rule:** Cap `ANSWER_SPAN_LOSS_WEIGHT` at 3.0 for all future smokes until discrete ACC gain is proven. Log gradient norms per module to detect conflict.
- **Kill-Switch Enforcement:** All HF launchers must run `weak_promotion_gate` at `SAVE_EVERY_STEPS` interval. If `bit < 136` or `truncated > 0`, job auto-terminates with `FINOPS_ABORT`.
- **Truncation Guard:** Add raw output length check to weak eval. If any row exceeds baseline token length by >20%, flag as `format_drift` and block promotion.
- **Dataset Sampling:** Switch from weighted replacement to uniform or stratified sampling for smokes to prevent equation dominance from starving bit gradients.
- **FinOps Rule:** No job exceeds 6 steps without passing checkpoint-2 weak gate. H200 usage capped at 30 mins per smoke variant. Cancel immediately if gate fails.

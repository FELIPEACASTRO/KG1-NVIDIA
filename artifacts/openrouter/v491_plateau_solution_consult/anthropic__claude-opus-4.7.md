## 1. Verdict

The most likely blocker is **mechanism, not data or metric**: V487 trained only attention+lm_head LoRA while the MoE `target_parameters` (`mlp.experts.gate_up_proj` / `down_proj`) were `frozen_active`. The optimizer therefore had no path to change MoE expert routing, which is exactly where bit_manipulation byte-pattern memorization lives. Lowering loss on attention LoRA without touching experts produces the observed pattern: `eval_loss` moves, ACC drifts ±1–2 with bit regressions because attention-only updates perturb decoding without giving the experts a corrective gradient.

**Equation-first is partially correct but mis-scoped.** The submit-safe gain ceiling we need is `+1 equation` AND `bit >= 136` AND `trunc = 0`. V477 and V488 both proved that pure equation pressure costs bit. The next experiment should be **bit-preserving with a small, targeted equation lift**, not equation-first. Concretely: enable `up_proj/down_proj` LoRA as trainable but with a **much smaller LR for those parameters** (or via brief warmup on bit replay only), keep `answer_span_loss_weight` unchanged, and gate on the first checkpoint with a hard bit floor.

UNKNOWN: whether the Nemotron LoRA wrapping of `target_parameters` actually exposes independent param groups for differential LRs — must be verified before relying on per-group LR.

## 2. Root Cause Ranking

1. **MoE `target_parameters` frozen_active (confidence 70%)**
   - Evidence: V487 launcher sets `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,lm_head` and `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`. V485 manifest shows 5934 LoRA tensors each for `mlp.experts.gate_up_proj` and `down_proj` exist but were not in the trainable allowlist. V490 confirms `frozen_active`.
   - Falsify quickly: rerun smoke with `up_proj,down_proj` added to `TRAINABLE_LORA_MODULES` and `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`, max_steps=2, eval at step 2. If weak ACC still plateaus with bit=136, this is not the blocker.

2. **Attention-only LoRA updates corrupt bit byte patterns (confidence 60%)**
   - Evidence: V477 (eq=57, bit=135) and V488 (eq=57, bit=134, trunc=1) both lost bit rows while changing only attention+lm_head LoRA. Bit answers are 8-char exact-match `[01]+`; one flipped token = wrong row. The V489 disagreement table shows V488 bit predictions differ from V290 by single-bit flips (`11100011` vs `11100010`, `01101000` vs `01111000`).
   - Falsify quickly: run V487 config but freeze `lm_head` LoRA only (keep q/k/v/o). If bit stops regressing, `lm_head` LoRA was the proximate corrupter.

3. **`lm_head` LoRA trainable on tiny dataset (confidence 55%)**
   - Evidence: `lm_head` has only 2 LoRA tensors / 4.28M params but is in `TRAINABLE_LORA_MODULES`. Even tiny LR drift on `lm_head` changes the distribution over `0`/`1` tokens directly, which is exactly what bit single-bit flips would look like.
   - Falsify quickly: same as #2. Single ablation: drop `lm_head` from trainable set; rerun smoke. If bit holds at 136 and equation does not collapse, lm_head was the leak.

4. **LR=4e-8 is in a regime that only nudges attention numerics (confidence 40%)**
   - Evidence: V487 uses `LEARNING_RATE=4.0e-8`, `FINAL_LEARNING_RATE=1.0e-8`, `MAX_STEPS=12`. At that LR, attention LoRA produces sub-rounding perturbations that change argmax on borderline tokens (bit flips) but cannot move MoE-controlled equation logic. This explains "loss moves, ACC doesn't" — it's noise-floor decoding.
   - Falsify quickly: probe with LR=1e-7 on q/k/v/o frozen, up/down trainable only, max_steps=4. If still no movement, LR is not the issue.

5. **`answer_span_loss_weight` > 1 amplifying noise on bit completions (confidence 25%)**
   - Evidence: UNKNOWN — `KG1_ANSWER_SPAN_LOSS_WEIGHT` is variable-driven and not pinned in the V487 snippet. V484 audit flagged 12.0 as a recurrent concern. The masking code at line 1273 multiplies loss on the answer span; on 8-token binary answers this is a huge gradient signal on `lm_head`.
   - Falsify quickly: log the actual `ANSWER_SPAN_LOSS_WEIGHT` and `answer_span_weighted_tokens` from V487's tokenization summary. If >1.0, run one ablation with weight=1.0.

## 3. Implementation Bugs or Gaps To Check

- **`apply_trainable_lora_module_filter` matching on `.{module}.`**: The substring match `f".{module}." in name` (line 704) may not match Nemotron's actual LoRA tensor names for MoE experts, which V485 shows arrive as `mixer.experts.<id>.up_proj` rather than `mlp.experts.gate_up_proj`. Add `up_proj`/`down_proj` to `TRAINABLE_LORA_MODULES` and confirm via `trainable_by_module["up_proj"] > 0` in the manifest before training. If it stays 0, the matcher is wrong for MoE aliases.
- **`REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` in V487**: explicit gap, not bug. Flip to 1 for the next run so the job fail-closes if MoE experts are not trainable.
- **`lm_head` in trainable set with `MAX_TRAINABLE_PARAM_RATIO=0.035`**: Confirm whether enabling `up_proj/down_proj` (864M LoRA params combined) would breach the ratio cap and silently re-freeze something via the cap path. Inspect the cap enforcement code (not shown).
- **`ANSWER_SPAN_LOSS_WEIGHT` indirection**: `KG1_ANSWER_SPAN_LOSS_WEIGHT` value is UNKNOWN in evidence. Pin and log it explicitly per run; reject if >1.0 unless deliberately ablating.
- **Expected-aware extraction last-boxed correctness**: code at lines 232–248 walks all `\boxed{` markers but only attempts variant match on the **last** position, then falls back to `extract_final_answer`. The fallback is not shown — must verify `extract_final_answer` also takes the last boxed, otherwise the diagnostic-only delta of +1 equation (`4bb8c6cd`) could be either legitimate or a fallback selecting earlier boxed.
- **`verify_answer` symbol path (line 257–262)**: when expected is non-binary and `float(expected)` succeeds, numeric tolerance applies. Equation answers like `"101"` and `"100"` (from V489 audit) are binary-like and pass `[01]+`, so they go strict — correct. But equation answers like `"100.0"` vs `"100"` would pass numeric tolerance. UNKNOWN whether weak labels contain such cases; audit.
- **V488 truncation row `59bee375`**: truncation=1 means generation hit max_new_tokens, not training tokenization. Check the eval `max_new_tokens` and whether the trained adapter started emitting longer chain-of-thought. If yes, this is a decoding budget issue, not a knowledge issue.
- **Bit replay weighting**: V326 weights are `v304_bit_replay_only=1.0`, `v325_equation_no_loss_distill=4.0`. Equation is upweighted 4x. With MoE frozen, the model has no capacity to localize equation learning, so attention LoRA absorbs the 4x pressure and degrades bit. Re-evaluate the 4:1 ratio.

## 4. Exact Next Experiment

**Goal**: prove MoE LoRA trainability transfers equation while preserving bit, with smallest possible footprint.

- **Trainable modules**: `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj` (drop `lm_head`)
- **target_parameters**: keep `mlp.experts.down_proj,mlp.experts.gate_up_proj`; set `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
- **LR**: `LEARNING_RATE=2.0e-8`, `FINAL_LEARNING_RATE=5.0e-9` (lower than V487 because trainable param count rises ~150x with MoE experts on)
- **Schedule**: `MAX_STEPS=6`, `SAVE_EVERY_STEPS=2`, `EVAL_EVERY_STEPS=2`, eval at step 2 first
- **answer_span_loss_weight**: `1.0` (force; do not amplify). If a previous run used >1, this is the controlled comparison.
- **Dataset mix**: keep V390/V326 but invert the source weights to **bit-protective**: `v304_bit_replay_only=2.0`, `v325_equation_no_loss_distill=1.0`. Rationale: V488 proved current 1:4 ratio sacrifices bit.
- **First-checkpoint (step 2) kill-switch** (hard, not advisory):
  - abort if `bit_manipulation_correct < 136`
  - abort if `truncated > 0`
  - abort if `total < 192`
  - continue only if `total >= 192` AND `equation >= 56` AND `bit >= 136` AND `trunc == 0`
  - promote-eligible at later checkpoint only if `total > 192`, `equation > 56`, `bit >= 136`, `trunc == 0`
- **Eval `max_new_tokens`**: pin to V290/V291 baseline value; log explicitly. UNKNOWN current value.
- **Pre-flight assertions** (must log in manifest before step 1):
  - `target_parameters_trainability_mode == "trainable"`
  - `trainable_by_module["up_proj"] > 0` and `trainable_by_module["down_proj"] > 0`
  - `ANSWER_SPAN_LOSS_WEIGHT == 1.0`
- **Cost / risk**: H200, max ~15 min, FinOps-cancel at step 2 on any guardrail miss. Risk: OOM from ~864M trainable LoRA — mitigate by keeping `BATCH_SIZE=4`, `MICRO_BATCH_SIZE=1`, `GRADIENT_CHECKPOINTING=1` (already on), `MAX_LENGTH=1024`.
- **Expected best case**: total 193–195, equation 57–58, bit 136, trunc 0 → promotion candidate.
- **Expected worst case**: bit drops below 136 again at step 2 → kill-switch fires, total cost <10 min H200. Confirms attention+experts together cannot be co-tuned at this LR and we must move to per-group LR or freeze attention.

## 5. Alternative If That Experiment Fails

If step-2 kill-switch fires (bit < 136 again):

1. **CPU-only ablation map (no GPU)**: regenerate predictions for the V290 baseline adapter with two extractor variants — `simple` (last `\boxed`) vs `expected_aware`. The V489 audit already shows one equation row (`4bb8c6cd`) differs. Hand-audit whether that delta is legal under rule #4. UNKNOWN whether it leaks; if it does, we have been chasing phantom gains.
2. **Per-group LR probe (cheap GPU, max 6 min)**: same trainable set as the main experiment, but set attention LoRA LR to `0` (effectively freeze) and only train `up_proj/down_proj`. This tests whether MoE-only updates can move equation without touching bit decoding at all. Requires confirming the trainer supports param groups for LoRA — UNKNOWN from evidence.
3. **Bit-row hard-negative CPU dataset** (no broad SFT): mine exactly the rows V488 regressed on (`8740ed31`, `59bee375`) plus the V488 equation gain (`518deb39`) and any V477 regressions. Build a <50-row targeted dataset where chosen = baseline-correct output, rejected = V488-style flip. Use as bit replay anchor in the next run. This is dataset surgery, not broad SFT — explicitly aligned with roadmap rule on no weak/full row training (these come from the predictions diff, not from gate labels; **verify this distinction holds** before use, otherwise it violates constraint #2).
4. **Stop adapter modifications, freeze V290/V291 as final submission** if no CPU/GPU probe in 3 attempts can show `equation > 56` with `bit >= 136` and `trunc = 0`. The competition baseline 192/315 is the safe floor; an unpromotable plateau is not worth further H200.

## 6. Stop Doing

- Stop running H200 with `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0` when `LORA_TARGET_PARAMETERS` is set. Always fail-closed.
- Stop including `lm_head` in `TRAINABLE_LORA_MODULES` for any bit-sensitive run until isolated as harmless.
- Stop using 4:1 equation:bit source weights. V488 proved it costs bit.
- Stop interpreting `eval_loss` improvements as progress. Roadmap already says this; enforce by removing `eval_loss` from any promotion log line.
- Stop launching any run >6 steps before the first weak micro-ACC checkpoint passes.
- Stop reusing the V487 launcher template without diffing `TRAINABLE_LORA_MODULES` and `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE` explicitly.
- Stop attempts to "fix" equation in isolation. The submit-safe constraint is joint; isolated equation gains have failed twice in the same way.

## 7. Roadmap Patch

- P3 smoke must set `TRAINABLE_LORA_MODULES=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj` (no `lm_head`) and `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`; reject any launcher that omits either.
- P3 smoke must pin `ANSWER_SPAN_LOSS_WEIGHT=1.0` and log the resolved value in the pre-train manifest; any value >1.0 requires an explicit ablation entry in the error ledger before launch.
- P3 source-weight policy: invert to bit-protective `v304_bit_replay_only=2.0`, `v325_equation_no_loss_distill=1.0` until a run demonstrates `bit >= 136` is preserved at higher equation weights.
- P3 first-checkpoint (step 2) hard kill-switch: abort on `bit < 136` OR `trunc > 0` OR `total < 192`; promotion requires `total > 192` AND `equation > 56` AND `bit >= 136` AND `trunc == 0`.
- Pre-flight gate must assert and log `target_parameters_trainability_mode == "trainable"` and `trainable_by_module["up_proj"]`, `trainable_by_module["down_proj"]` both > 0; fail-closed otherwise.
- Pin eval `max_new_tokens` to the V290/V291 value across all weak gates; record in every manifest. Truncation root-cause must be investigated before any further equation push.
- Add a CPU regression gate that compares `simple_extracted` vs `expected_aware_extracted` per family on every weak prediction CSV; block promotion if expected-aware selects a non-last boxed payload anywhere.
- Add a per-row regression gate: any new run that regresses a previously-correct V290/V291 bit row by more than 1 row must auto-fail FinOps regardless of equation gain.
- If two consecutive P3 smokes fail the bit floor at step 2, escalate to alternative #2 (per-group LR with attention LR=0); if that also fails, freeze V290/V291 as final submission and stop GPU spend.
- Archive `lm_head` in `TRAINABLE_LORA_MODULES` as a known risk in `KG1_ERROR_LEDGER_2026_05_15.md` with V477/V488 row evidence; require explicit ablation justification to re-enable.

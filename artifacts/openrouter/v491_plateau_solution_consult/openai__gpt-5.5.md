## 1. Verdict

Most likely blocker: the current adapter training recipe is changing behavior in the wrong subspace/objective. V487/V488 proved PEFT continuity was no longer the only issue: it loaded/ran, loss moved, and equation gained +1, but bit regressed -2 and truncation appeared.

Next action: run exactly one short MoE-target-parameter trainability smoke, not another broad SFT repeat.

Equation_transform-first is correct only in the narrow sense: try to recover `equation_transform >56` while treating bit and truncation as hard kill-switches. It is not correct to run equation-heavy training if bit falls below 136 or truncation appears. The first checkpoint must pass:

- total `>=193`
- equation `>=57`
- bit `>=136`
- truncated `0`

If not, cancel.

## 2. Root Cause Ranking

| Rank | Likely blocker | Confidence | Evidence | Fast falsification |
|---:|---|---:|---|---|
| 1 | `target_parameters` MoE LoRA were active but not trainable in V487/V488, so the tested mechanism was incomplete. | 35% | V490 says V487 trained only `q_proj,k_proj,v_proj,o_proj,lm_head`; `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` were frozen-active. V488 failed despite PEFT continuity. | CPU dry-run + first checkpoint manifest must show `target_parameters_trainability_mode="trainable"` and `target_parameter_trainable_lora_tensors` nonzero for both target parameters. If ckpt-2 still gives `bit<136`, `truncated>0`, or total `<=192`, falsified for this route. |
| 2 | Objective balance still trades equation for bit. | 25% | V477: equation 57, bit 135. V488: equation 57, bit 134, trunc 1. V391 was blocked because equation effective share was 0.864. V486 still had equation share 0.792 and later V488 regressed bit. | Before GPU, run objective-weight probe. Block if bit effective pressure is below the configured floor. On ckpt-2, cancel if the same pattern appears: equation +1 with bit -1/-2. |
| 3 | Loss/eval_loss is misaligned with strict row-level ACC. | 18% | Roadmap and V490 state loss moves but ACC plateau remains. V487 checkpoint-10 had best `eval_loss=1.3519` but weak 191. Binary-like answers require exact strings. | Treat `eval_loss` as diagnostic only. If ckpt-2 loss improves but weak ACC gate fails, cancel immediately. |
| 4 | Answer formatting / truncation fragility is causing real row losses. | 12% | V488 introduced one truncation on bit row `59bee375`. V489 found extraction risks and fixed expected-aware extraction. V490 says adapter is sensitive to small format/decoding changes. | For every new weak eval, emit raw extraction audit: `simple_extracted`, `expected_aware_extracted`, truncation count, and row IDs. Any `truncated>0` kills the run. |
| 5 | Remaining PEFT/filter observability bug. | 10% | Previous continuity bug existed. V489 found F2 observability gap: frozen-active target parameters were not visible in old manifest. Code has now added trainability counters, but this has not yet been exercised in a successful GPU run. | Pre-GPU CPU gate must fail-closed if `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1` and any target parameter has zero trainable tensors. Compare manifest counts to V485 expected `5934/5934`. |

## 3. Implementation Bugs or Gaps To Check

1. **Trainable filter must prove MoE target parameters are actually trainable.**  
   File/logic: `apply_trainable_lora_module_filter`, lines 0602-0814.  
   Required test:
   - launch dry-run with `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`
   - `LORA_TARGET_PARAMETERS="mlp.experts.down_proj,mlp.experts.gate_up_proj"`
   - `TRAINABLE_LORA_MODULES` or substrings must match `up_proj/down_proj`
   - manifest must show:
     - `target_parameters_trainability_mode="trainable"`
     - `target_parameter_trainable_lora_tensors["mlp.experts.down_proj"] > 0`
     - `target_parameter_trainable_lora_tensors["mlp.experts.gate_up_proj"] > 0`
   - expected structural count from V485: `5934` tensors each.

2. **Do not accidentally repeat V487.**  
   V487 env had:
   - `TRAINABLE_LORA_MODULES='q_proj,k_proj,v_proj,o_proj,lm_head'`
   - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=0`  
   That recipe is already falsified by V488 weak 191. Any new launcher with that combination should be blocked.

3. **Objective probe must block V391/V486-style equation dominance.**  
   Evidence:
   - V391 bit share `0.135975`, equation share `0.864025`, rejected.
   - V486 bit share `0.207788`, equation share `0.792212`, then V488 regressed bit.  
   Gap: current evidence does not prove a safe exact target share. Therefore the next launcher must emit the effective family share and be blocked if bit pressure is not deliberately raised above V486 or explicitly justified. UNKNOWN optimal share.

4. **Answer-span loss weight may be masking ACC.**  
   Evidence says `answer_span_loss_weight=12.0` is a recurring risk but not proven.  
   Code lines 1236-1418 correctly logs weighted examples/tokens and raises if weighting is configured but no spans are found.  
   Gap: no evidence that high answer-span weight improves strict ACC. For the next mechanism test, do not increase it.

5. **Strict metric path must remain locked.**  
   File/logic: `verify_answer`, lines 0252-0262.  
   Correct behavior:
   - `[01]+` expected answers require exact string match.
   - `answers_equivalent` is diagnostic-only.  
   Required check: every weak eval manifest must state it used `verify_answer`, not `answers_equivalent`.

6. **Expected-aware extraction must stay last-boxed only.**  
   File/logic: `extract_final_answer_for_expected`, lines 0210-0249.  
   Current code uses only `marker_positions[-1]`, which satisfies the rule.  
   Required regression test: earlier boxed correct answer + final boxed wrong answer must score wrong.

7. **Truncation must be treated as behavioral failure, not parser noise.**  
   V488 had `truncated=1` and lost bit row `59bee375`.  
   Required check: weak eval output must include row-level truncation IDs. Any new truncation kills the run.

8. **`modules_to_save` must remain empty.**  
   V485 seed has `modules_to_save=[]`.  
   Required package/preflight check: no full saved modules. `lm_head` may exist only as LoRA target, not full module save.

## 4. Exact Next Experiment

Run one minimal smoke: **V491 MoE-target-trainable smoke**.

### Trainable configuration

Use the V290/V291 checkpoint-6 seed that passed V485.

Set:

```bash
INIT_ADAPTER_LOAD_MODE='peft'
FAIL_ON_MISSING_ADAPTER_KEYS=1

LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.0
LORA_TARGET_MODULES='down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj'
LORA_TARGET_PARAMETERS='mlp.experts.down_proj,mlp.experts.gate_up_proj'

TRAINABLE_LORA_MODULES='up_proj,down_proj,lm_head'
TRAINABLE_LORA_NAME_SUBSTRINGS=''
REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='up_proj,down_proj,lm_head'

REQUIRE_LORA_TARGET_PARAMETER_MATCH=1
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1
```

Rationale:

- This is structurally different from V487.
- It directly tests the unfalsified hypothesis: train MoE target-parameter LoRA.
- It avoids repeating `q/k/v/o/lm_head`-only training.
- `lm_head` stays LoRA-only, not `modules_to_save`.

If this OOMs before first checkpoint, retry once with:

```bash
TRAINABLE_LORA_MODULES='up_proj,down_proj'
REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS='up_proj,down_proj'
```

Do not fall back to V487 modules.

### LR / steps / checkpoints

Use a short, conservative run:

```bash
LEARNING_RATE=1.0e-8
FINAL_LEARNING_RATE=1.0e-8
NUM_EPOCHS=1
MAX_STEPS=4
SAVE_EVERY_STEPS=2
EVAL_EVERY_STEPS=2
BASELINE_EVAL_BEFORE_TRAIN=1
```

Reason: V488 regressed bit with `4.0e-8 -> 1.0e-8` while training fewer modules. More trainable MoE parameters increases risk. UNKNOWN optimal LR; this is a falsification smoke, not a production run.

### Answer-span loss weight

Set:

```bash
ANSWER_SPAN_LOSS_WEIGHT=1.0
```

Do not use or increase `12.0` in this smoke. Evidence only supports it as a risk, not as an ACC win.

### Dataset mix / weights

Use the clean V390/V326 dataset only.

Before GPU, run the existing objective-weight probe and require:

- no gate/weak/full rows used for training
- train/val hashes match V390/V326 manifests
- tokenization gate passes with:
  - offset masks OK
  - fallback masks `0`
  - prompt truncation `0`
  - completion truncation `0`

Do not use V391 weights. Do not accept equation share like V391. Because V486 still led to bit regression, raise bit pressure relative to V486 before launch. Exact safe share is UNKNOWN; the gate should record the effective bit/equation shares and block any unreviewed equation-dominant config.

### First-checkpoint kill-switch

At checkpoint-2, run strict weak eval on all 315 weak rows.

Continue to checkpoint-4 only if checkpoint-2 satisfies all:

```text
status == ok
total >= 193
equation_transform >= 57
bit_manipulation >= 136
truncated == 0
```

Cancel immediately if any are true:

```text
total <= 192
equation_transform <= 56
bit_manipulation < 136
truncated > 0
```

At checkpoint-4, promote only if the same gate passes. Full official-like eval/package only if weak passes.

### Required manifests

The checkpoint-2 manifest must include:

- `target_parameters_trainability_mode`
- `target_parameter_lora_tensors`
- `target_parameter_trainable_lora_tensors`
- `target_parameter_trainable_lora_params`
- `trainable_by_module`
- `frozen_by_module`
- weak row diff vs V290/V291
- truncation row IDs
- `simple_extracted` vs `expected_aware_extracted` audit
- strict `verify_answer` metric confirmation

### Cost / risk

Cost: low-to-moderate H200 smoke, capped at first checkpoint unless it passes.  
Risk: high OOM/regression risk because `up_proj/down_proj` LoRA are large; V485 shows each target-parameter group has `5934` tensors and `432,791,552` LoRA params.

### Expected weak outcomes

Best case:

```text
total 193-195
equation 57-59
bit 136
truncated 0
```

Worst case:

```text
OOM before ckpt-2, or weak <=191, equation 57, bit 134-135, truncated 0-1
```

If worst case appears, this route is falsified. Do not extend steps.

## 5. Alternative If That Experiment Fails

Do not repeat broad SFT. Move to CPU/cheap probes and targeted synthetic fixtures.

1. **Row-diff forensic only, not training on weak rows.**  
   Analyze weak IDs `518deb39`, `8740ed31`, `59bee375` only to classify failure modes. Do not use their prompts/answers as train labels, chosen/rejected pairs, or selection targets.

2. **Generate non-gate synthetic fixtures from the failure modes.**  
   Use CPU rules/DSL to create new prompts with no `id`, prompt hash, normalized prompt, or n-gram overlap with weak/full rows. Labels must come from the rule generator, not weak/full answers.

3. **Build a tiny hard-negative dataset.**  
   Scope:
   - equation final-answer-only examples
   - bit exact-string preservation examples
   - truncation/boxed-format fixtures
   - no broad traces unless tokenization and leakage gates pass

4. **CPU-only gates before any GPU:**
   - anti-leakage by `id`, `prompt_sha256`, normalized prompt, and n-gram
   - tokenization gate
   - objective-weight probe
   - strict extraction regression tests
   - trainability manifest check

5. **Only then run a 2-step adapter smoke.**  
   Same weak kill-switch: total `>=193`, equation `>=57`, bit `>=136`, trunc `0`.

## 6. Stop Doing

- Stop using `eval_loss` or lower loss as promotion evidence.
- Stop running H200 jobs beyond first checkpoint without weak micro-ACC passing.
- Stop repeating V487-style `q/k/v/o/lm_head` training with MoE target parameters frozen-active.
- Stop using V391-like equation-dominant objectives.
- Stop accepting equation `57` if bit drops below `136`.
- Stop packaging/submitting unless weak and full-like gates both improve.
- Stop any metric path using `answers_equivalent` for official ACC.
- Stop any expected-aware extraction that can select an earlier boxed answer.
- Stop training from weak/full rows, weak/full labels, or row-specific chosen/rejected pairs.
- Stop considering runtime solvers/verifiers/postprocessors/logit masks/prompt hacks as submit-safe.

## 7. Roadmap Patch

Insert these bullets:

- Next GPU job is limited to one MoE-target-trainable smoke with `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`; V487-style frozen-active MoE recipes are blocked.
- The smoke must show `target_parameters_trainability_mode="trainable"` and nonzero trainable tensor counts for both `mlp.experts.down_proj` and `mlp.experts.gate_up_proj` before training.
- First checkpoint weak eval is mandatory at checkpoint-2; continue only if total `>=193`, equation `>=57`, bit `>=136`, and truncated `0`.
- `ANSWER_SPAN_LOSS_WEIGHT` must not be increased; next mechanism smoke uses `1.0` unless a separate CPU gate proves otherwise.
- Every new weak eval must emit row-level diff vs V290/V291, truncation IDs, and `simple_extracted` vs `expected_aware_extracted` audit.
- Objective-weight probe must run before GPU and must reject V391/V486-style unreviewed equation dominance.
- If MoE-target-trainable smoke fails, broad SFT is blocked; fallback is CPU-generated, anti-leakage hard-negative fixtures only.
- Weak/full rows remain eval-only and may be used only for error taxonomy, never for labels, chosen/rejected pairs, or cherry-pick selection.
- Package/submit remains blocked unless weak passes promotion and full official-like exceeds `823/947`.
- Any launcher with `INIT_AD

## Verdict
Relaunch is justified, but only as a **minimal, tightly gated smoke run**: the V480 `target_parameters` drop is a **real configuration regression** and is the strongest submit-safe root-cause candidate for your bit regressions/plateau. Do **not** run another full expensive job until CPU preflight proves config/state coverage parity with the V290 lineage and the first checkpoint clears your weak-gate kill-switch.

## Evidence Assessment
- **PROVEN**: V290 seed adapter config includes non-empty `target_parameters` (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`), while V480 run logs show `LORA_TARGET_PARAMETERS` empty and checkpoints saved with `target_parameters = null`.
- **PROVEN**: Your launch path allowed this mismatch (`REQUIRE_LORA_TARGET_PARAMETER_MATCH=0`), so continuation semantics differed from seed lineage.
- **PLAUSIBLE**: Missing `target_parameters` changed trainable coverage in MoE MLP paths enough to cause bit regressions and no real weak-gate gain despite similar eval_loss.
- **PROVEN**: Loss improved/changed in prior runs without weak accuracy gains; eval_loss alone is not predictive for your gate objective.
- **UNKNOWN**: Whether restoring `target_parameters` alone recovers +1 to +4 weak rows (needs controlled A/B smoke).
- **PLAUSIBLE**: Some prior equation+1 / bit-1 outcomes are objective tradeoff plus adapter coverage drift, not pure data hash issues (your dataset checks look stricter now).
- **UNKNOWN**: “accuracy=0.0000” custom logging bug root cause (likely metric path/parsing issue, but not proven from current artifacts).

## Root Cause Ranking
1. **LoRA config drift in continuation (missing `target_parameters`)** — **70%**
2. **Continuation loading method not preserving seed adapter semantics exactly** (new config + state load vs true PEFT load path) — **55%**
3. **Objective/mixture still slightly misaligned for weak-gate metric despite fixed gross bug** — **40%**
4. **Eval/inference pipeline mismatch (template/decode/truncation/offset-mask differences)** — **35%**
5. **Metric logging/aggregation defects masking true behavior in some runs** — **25%**

## Required Code/Gate Patches
1. **Hard fail on seed/config mismatch (already started; keep strict by default)**
   ```python
   # preflight: compare normalized config fields
   must_match = ["peft_type","task_type","r","lora_alpha","lora_dropout",
                 "bias","target_modules","target_parameters"]
   for k in must_match:
       if norm(seed_cfg.get(k)) != norm(job_cfg.get(k)):
           raise RuntimeError(f"INIT_ADAPTER_CONFIG_MISMATCH:{k}")
   if seed_cfg.get("target_parameters") and not os.getenv("REQUIRE_LORA_TARGET_PARAMETER_MATCH") == "1":
       raise RuntimeError("target_parameters present in seed but match disabled")
   if is_moe_model and not job_cfg.get("target_parameters"):
       raise RuntimeError("MoE adapter requires non-empty target_parameters")
   ```

2. **Use PEFT-native continuation load path**
   - Preferred:
     ```python
     base = AutoModelForCausalLM.from_pretrained(base_id, ...)
     model = PeftModel.from_pretrained(base, seed_adapter_dir, is_trainable=True)
     # continue training same adapter
     ```
   - Avoid “fresh `LoraConfig` + `set_peft_model_state_dict`” unless you also assert exact config equivalence and key coverage.

3. **Trainable coverage check (CPU)**
   ```python
   trainable = [n for n,p in model.named_parameters() if p.requires_grad]
   assert any("gate_up_proj" in n for n in trainable)
   assert any("down_proj" in n for n in trainable)
   # compare counts vs seed-expected snapshot
   assert len(trainable) == expected_trainable_count
   ```

4. **Safetensors key/shape/dtype parity checks (not size-only)**
   ```python
   sd = safe_load(adapter_model)
   cfg = json.load(open(adapter_config))
   assert cfg["target_parameters"]  # non-empty for this lineage
   assert all(any(tp in k for k in sd.keys()) for tp in cfg["target_parameters"])
   # check key count and per-key shape against seed tolerance
   ```

5. **Post-save checkpoint integrity gate**
   - After each checkpoint save:
     - `adapter_config.json` exact-match fields to init adapter (except allowed metadata).
     - non-empty target_parameters.
     - expected tensor-key family present.

6. **Inference/template consistency gate (CPU)**
   - Single script emits canonical prompt for fixed row IDs from baseline and candidate.
   - Diff must be zero except allowed whitespace.
   - Decode params lock file (`temperature/top_p/max_new_tokens/stop`) hash-checked.

## Minimal Next Experiment
1. **CPU-only preflight A/B (no GPU)**
   - Load base + V290 seed via `PeftModel.from_pretrained(..., is_trainable=True)`.
   - Assert config parity, trainable coverage, safetensor key coverage.
   - Simulate one save cycle; re-open checkpoint; re-run same assertions.
   - **Fail immediately** if any mismatch.

2. **Micro-train smoke on HF GPU (max 2 checkpoints)**
   - Same data as V480 objective-aligned set.
   - Very short run, evaluate weak set at ckpt-1 and ckpt-2.
   - Keep decode/inference identical to baseline evaluator.

3. **Kill-switch criteria**
   - At first evaluated checkpoint:
     - `truncated == 0` (hard fail otherwise),
     - `bit >= 136` (hard fail),
     - `total >= 193` (hard fail).
   - If fail at ckpt-1, stop run; do not proceed to full training.

4. **Promotion criteria (unchanged)**
   - Keep only if submit-safe gate passed; otherwise discard regardless of eval_loss.

## Stop Doing
- Full-cost H200 runs without strict init/save config parity gates.
- Any experiment judged primarily by eval_loss.
- Training paths that rely on solver/verifier/postprocess signals for adapter-only promotion decisions.
- Multi-variable changes in one run (data mix + objective + loader + decode), which destroys attribution.
- Using file size similarity as evidence of adapter equivalence.

## Missing Evidence
- V290 vs V480 full `adapter_config.json` files (raw, not summaries).
- V290 and V480 `adapter_model.safetensors` key listings with shapes/dtypes.
- Continuation loader code path used in V480 (exact function calls around PEFT model creation/loading).
- `print_trainable_parameters()` (or equivalent named-parameter dump) for V290-init run and V480-init run.
- Weak-eval inference config/logs (prompt template hash, decode params, truncation handling) for baseline vs V481 evaluator.
- The metric script/log segment that produced `accuracy=0.0000` to isolate logging vs true scoring bug.
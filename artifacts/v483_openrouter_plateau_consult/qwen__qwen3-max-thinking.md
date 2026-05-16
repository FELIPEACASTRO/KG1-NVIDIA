## Verdict  
Relaunch a minimal continuation job from V290 checkpoint-6 **with strict target_parameters preservation and preflight validation**, as the missing target_parameters in V480 is a **plausible and high-impact root cause** for the accuracy plateau and bit regression. The evidence strongly suggests V480 trained a non-MoE-aware LoRA (effectively freezing expert paths), which explains loss stagnation and weak accuracy decoupling. A 2-step smoke job with CPU preflight checks is low-cost and necessary before further GPU spend.

---

## Evidence Assessment  
- **V480 training logs show `LORA_TARGET_PARAMETERS` empty and `REQUIRE_LORA_TARGET_PARAMETER_MATCH=0`**: PROVEN  
- **V290 seed adapter has non-null `target_parameters` targeting MoE expert layers**: PROVEN  
- **V480 adapter_config.json has `target_parameters = null`**: PROVEN  
- **V480 eval shows bit accuracy regressed by 2 despite lower eval_loss**: PROVEN  
- **Equation +1 paired with bit -1/-2 is consistent with MoE path dropout**: PLAUSIBLE  
- **Loss decreased but weak accuracy unchanged → misaligned optimization**: PLAUSIBLE  
- **`accuracy=0.0000` logs indicate possible output masking or tokenizer misalignment**: UNKNOWN (needs generation debug logs)  

---

## Root Cause Ranking  
1. **Missing `target_parameters` in V480 → LoRA not applied to MoE expert layers** (Confidence: 85%)  
   - Explains bit regression, loss/accuracy decoupling, and failure to improve on weak set.  
2. **Incorrect PEFT state loading → adapter weights not properly initialized from seed** (Confidence: 60%)  
   - If `set_peft_model_state_dict` used without config alignment, MoE tensors may be ignored.  
3. **Tokenizer or prompt template drift in V479+ data pipeline** (Confidence: 40%)  
   - Could cause silent truncation or misalignment, but `truncated=0` in most runs argues against this.  
4. **Seed adapter corruption or safetensors key mismatch** (Confidence: 30%)  
   - File size similarity doesn’t guarantee tensor correctness; key coverage needed.  
5. **Overfitting to equation subset due to residual objective imbalance** (Confidence: 25%)  
   - V477 showed equation+1/bit-1, but V480 shows no equation gain → less likely primary cause.

---

## Required Code/Gate Patches  
Add to `hf_job_preflight_gate.py` (run on CPU before GPU job):

```python
def validate_adapter_continuation(init_adapter_path: str, job_config: dict):
    # 1. Load seed adapter config
    seed_config = json.load(open(f"{init_adapter_path}/adapter_config.json"))
    
    # 2. Enforce target_parameters match if present in seed
    if seed_config.get("target_parameters"):
        assert job_config.get("LORA_TARGET_PARAMETERS") == ",".join(seed_config["target_parameters"]), \
            "LORA_TARGET_PARAMETERS must match seed adapter"
        assert job_config.get("REQUIRE_LORA_TARGET_PARAMETER_MATCH") == "1", \
            "REQUIRE_LORA_TARGET_PARAMETER_MATCH must be 1 for MoE adapters"
    
    # 3. Validate safetensors keys cover all target_parameters
    from safetensors import safe_open
    with safe_open(f"{init_adapter_path}/adapter_model.safetensors", framework="pt") as f:
        keys = set(f.keys())
    expected_keys = set()
    for param in seed_config["target_parameters"]:
        expected_keys.update([f"{param}.lora_A.weight", f"{param}.lora_B.weight"])
    assert expected_keys.issubset(keys), f"Missing LoRA tensors: {expected_keys - keys}"
    
    # 4. Confirm new config will preserve target_parameters
    new_config = build_lora_config(job_config)  # your config builder
    assert new_config.target_parameters == seed_config["target_parameters"], \
        "New adapter config must preserve target_parameters"
```

Also enforce in launcher:
```bash
export KG1_LORA_TARGET_PARAMETERS="mlp.experts.gate_up_proj,mlp.experts.down_proj"
export REQUIRE_LORA_TARGET_PARAMETER_MATCH=1
```

---

## Minimal Next Experiment  
**Goal**: Validate that correct MoE LoRA continuation recovers baseline and shows equation gain.  

**Steps**:  
1. **CPU preflight**: Run `hf_job_preflight_gate.py` on V290 checkpoint-6 with V480+ launcher env. Must pass.  
2. **Launch 2-step job** on H200:  
   - Seed: V290 checkpoint-6  
   - Data: V479 objective-aligned (equation:bit = 50:50)  
   - Config: `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`, `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`  
   - Steps: 2 (save checkpoint-2 only)  
3. **Evaluate checkpoint-2 on weak set**:  
   - **Pass**: total ≥ 193, equation ≥ 57, bit ≥ 136, truncated = 0  
   - **Fail**: any metric below → kill job, no further steps.  

**Budget**: < 0.5 H200-hour. If pass, proceed to 8-step; if fail, debug loading.

---

## Stop Doing  
- **Training without `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1` for MoE adapters** → silent regression risk.  
- **Using `set_peft_model_state_dict` without config parity checks** → state may ignore MoE tensors.  
- **Chasing eval_loss reductions without weak accuracy gating** → loss ≠ submit-safe gain.  
- **Launching jobs without CPU preflight config/tensor validation** → wastes GPU on broken setups.  
- **Replay/SFT on full data without hard-negative focus on equation_transform** → low ROI.

---

## Missing Evidence  
- **V290 checkpoint-6 `adapter_model.safetensors` key list** → confirm MoE LoRA tensors exist.  
- **V480 training step debug logs showing which modules received gradients** → verify MoE layers frozen.  
- **Weak eval generation logs for V480 checkpoint-2** → inspect bit/equation failure patterns (e.g., truncation, format errors).  
- **PEFT loading trace from V480 job** → confirm whether `PeftModel.from_pretrained` was used vs manual state dict.
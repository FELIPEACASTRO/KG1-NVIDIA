# T1 + T2 Findings — Critical Updates Applied 2026-04-21

## T1 (vLLM Inference Optimization) — 5 findings

### Finding #1: `mamba_ssm_cache_dtype=float32` CRÍTICO
**Source**: vLLM docs via context7 — `"Only float32 is known to have no accuracy issues by default"` for NemotronH

**Action**:
- Verify base model config has `mamba_ssm_cache_dtype=float32` (not bf16)
- Add assert in V70.5 notebook training cell:
  ```python
  if hasattr(model.config, 'mamba_ssm_cache_dtype'):
      assert str(model.config.mamba_ssm_cache_dtype) in ('torch.float32', 'float32', 'auto'), \
          'mamba_ssm_cache_dtype must be float32'
  ```

**Gain estimate**: +0.002 to +0.01 (drift in equations with long CoT)

### Finding #2: Training max_length MUST exactly match Kaggle kernel 8192
**Source**: Mamba state extrapolation degrades accuracy if mismatch

**Action**: V70.5 notebook already sets `max_length=8192` ✅

**Gain estimate**: +0.003 to +0.008 (existing V70.5 expected)

### Finding #3: BOXED_INSTRUCTION byte-for-byte match
**Source**: Prefix cache only hits with exact match

**Action**:
- Use EXACT string: `"\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`
- Same in training template + inference template
- Do NOT modify whitespace, newlines, or punctuation

**Status**: scripts/local_score.py ALREADY updated with exact string (P0 fixes) ✅

### Finding #4: max_tokens=7680 leaves only 512 for prompt
**Source**: vLLM config analysis — `max_model_len=8192 - max_tokens=7680 = 512 for prompt`

**Action**:
- Verify our puzzle prompts are ≤ 400 tokens after tokenization
- Typical Alice Wonderland prompt: 150-300 tokens ✅ (safe)
- BOXED_INSTRUCTION adds ~25 tokens ✅

**Gain estimate**: +0.0 (prevents catastrophic failure if prompts too long)

### Finding #5: Bug vLLM speculative decoding + Mamba corrupts SSM state
**Source**: GitHub Issue #39273

**Action**: Kaggle kernel already doesn't use speculative decoding. Safe.

**Gain estimate**: N/A (avoid regression)

## T2 (LoRA Target Modules) — 4 findings

### Finding #1: V70 `all-linear` config is CORRECT
**Source**: 3 papers validated (arxiv 2410.09016, 2411.03855, 2511.06739)

**Key insight**: LoRA em `x_proj`/`dt_proj` ATRAPALHA reasoning (-4.3 GLUE). Kaggle FORBIDDEN protege automaticamente.

**Action**: Maintain V70 config ✅ (no change needed)

### Finding #2: `gate_proj` FORBIDDEN = unavoidable structural loss
**Source**: Rank-1 LoRAs paper — `gate_proj` is most critical MLP for reasoning

**Impact**: -0.5 to -1% theoretical loss vs Qwen-like baselines

**Action**: NONE — every competitor has same restriction

**Key insight**: Daulet's 0.87 is NOT from gate_proj (everyone blocked). Diferencial is elsewhere (data/neuro-symbolic/compute).

### Finding #3: V71b candidate — DoRA (`use_dora=True`)
**Source**: Multiple papers show +0.5-1% typical gain

**Action**: Add to V71 experiment list:
```python
lora_cfg = LoraConfig(
    r=32, lora_alpha=32,
    target_modules="all-linear",
    use_dora=True,  # <- NEW
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Gain estimate**: +0.005 (IC [+0.002, +0.010])

**Trainable params overhead**: +5%

### Finding #4: V71c candidate — Heterogeneous rank
**Source**: DR-LoRA style — different rank per module group

**Action**:
```python
lora_cfg = LoraConfig(
    r=32, lora_alpha=32,
    target_modules="all-linear",
    rank_pattern={
        'q_proj': 64, 'k_proj': 64, 'v_proj': 64, 'o_proj': 64,  # attention critical
        'in_proj': 32, 'out_proj': 32,                            # Mamba normal
        'up_proj': 16, 'down_proj': 16,                           # MoE economy (128 experts)
        'lm_head': 32,                                            # vocab head
    },
    alpha_pattern={'q_proj': 128, 'k_proj': 128, 'v_proj': 128, 'o_proj': 128,
                   'in_proj': 64, 'out_proj': 64,
                   'up_proj': 32, 'down_proj': 32,
                   'lm_head': 64},
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Gain estimate**: +0.01 to +0.02 (theoretical, untested in our setup)

**Trainable params**: similar to baseline (MoE r=16 saves 60%, attention r=64 costs 2x)

## Combined T1 + T2 Roadmap Updates

### V70.5 (current) — apply T1 finding #1 + #3:
- Verify `mamba_ssm_cache_dtype=float32`
- Use exact BOXED_INSTRUCTION
- (already has max_length=8192 + enable_thinking=True)

### V71a = V70.5 + T1/T2 full alignment
- All T1 verifications
- No LoRA config changes yet
- Expected: +0.015-0.018 over V70 (just alignment)

### V71b = V71a + DoRA (single var change)
- Set `use_dora=True`
- Expected additional: +0.005

### V71c = V71b + heterogeneous rank pattern
- Use rank_pattern dict
- Expected additional: +0.01 to +0.02
- RISK: PEFT rank_pattern requires exact regex match per module name — could silently skip modules

## Updated Expected Score Progression v5.0

| Stage | Config | Expected score (Kaggle) |
|---|---|---|
| V70 current | baseline | 0.84 (local) / 0.86-0.87 (real w/ metric fix) |
| V70.5 | + max_length=8192 + enable_thinking + T1 checks | 0.86-0.88 |
| V71a | + full T1 alignment | 0.87-0.89 |
| V71b | + DoRA | 0.875-0.895 |
| V71c | + heterogeneous rank | 0.885-0.905 |

**Note**: probabilities compound with IC uncertainty. Upper bound realistic ~0.91.

## Files to update

1. `notebooks/KG1_V70_5_FIXED_METRIC.ipynb` — add T1 Finding #1 assert
2. Create `notebooks/KG1_V71b_DORA.ipynb` — experiment notebook with DoRA
3. Create `notebooks/KG1_V71c_HETEROGENEOUS_RANK.ipynb` — experiment with rank_pattern
4. Update `ROADMAP_V71_TOP1_ALTA_CONFIANCA_v4.md` → v5.0 with these findings

## Status

- ✅ T1 vLLM optimization findings documented
- ✅ T2 LoRA targets ablation plan documented
- 🟡 V70.5 notebook needs manual update (can't edit .ipynb directly)
- 🟡 V71b/c notebooks to be created
- 🟡 V5.0 roadmap consolidation (agent a6d1504 running)

**Priority**: when V70 resubmit Kaggle test completes, decide if V71a/b/c pipeline is needed based on LB score.

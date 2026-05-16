# V483 OpenRouter Plateau Debug Prompt

## System Prompt

You are an external senior ML systems auditor.

Rules:
- Do not invent facts. If evidence is insufficient, say exactly what file/log is needed.
- Focus on the NVIDIA Nemotron Model Reasoning Challenge adapter-only Kaggle submission path.
- We cannot use a runtime solver, verifier, postprocessor, or row-label oracle in the Kaggle submission.
- The objective is a real submit-safe gain on weak/full gates, not lower eval_loss alone.
- Be concrete: cite checks, failure modes, code-level patches, and a minimal next experiment.
- Prefer actions that can be validated on CPU before paid HF GPU jobs.


## User Prompt

We need an independent technical audit and decision recommendation.

Context:
- Competition: NVIDIA Nemotron Model Reasoning Challenge.
- Submission type we can use: adapter-only LoRA/PEFT package. No runtime solver, verifier, external postprocessor, row-label oracle, or hand-coded test-time solver is allowed in our intended submit path.
- Weak validation contract: 315 rows, split into bit_manipulation 160 and equation_transform 155.
- Current submit-safe baseline:
  - total = 192/315
  - equation_transform = 56/155
  - bit_manipulation = 136/160
  - truncated = 0
- Recent non-submit-safe solver/verifier work can reach about 196/315 with equation 60 and bit 136, but that relies on postprocessor/verifier logic and cannot be directly submitted as adapter-only.
- Promotion gate for any new adapter:
  - total >= 193
  - equation_transform >= 57, ideally 60
  - bit_manipulation >= 136
  - truncated = 0
- Full public/leaderboard plateau is around 0.86. We need small real gains, even +1 to +4 weak rows, but they must be adapter-only and submit-safe.

What we tried:
1. Broad SFT and replay training repeatedly lowered train/eval loss but did not improve weak accuracy. Often bit regressed.
2. Hard-negative and solver-trace training routes improved local solver/verifier signals but did not transfer to LoRA-only predictions.
3. Objective-alignment bug was found in V476:
   - equation weight accidentally about 99.0508 percent
   - bit weight about 0.9492 percent
   - V477 corrected part of that but produced only equation +1 with bit -1, not submit-safe.
4. V479 made dataset/objective checks stricter:
   - exact family/subcategory counts
   - dedupe and prompt/id checks
   - tokenization checks
   - no known dataset row-count/hash issue found.
5. V480 trained on H200 with V479 objective-aligned data. It failed:
   - baseline eval_loss before training: 0.9725
   - checkpoint 2 eval_loss: 0.9761
   - checkpoint 4 eval_loss: 0.9752
   - checkpoint 6 eval_loss: 0.9738
   - checkpoint 8/final eval_loss: 0.9739
   - Weak eval V481:
     - checkpoint 2: total 191, equation 57, bit 134, truncated 1
     - checkpoint 4: total 190, equation 56, bit 134, truncated 0
     - checkpoint 6: total 191, equation 57, bit 134, truncated 1
   - No submit-safe candidate.

Strong current suspected bug:
- Seed adapter V290 checkpoint-6 has adapter_config.json:
  - target_modules includes attention/MLP/lm_head modules
  - target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
  - adapter_config sha256 = a3d74c5a52ce75f71a8406222d877b9760ea18a40a772bcf407686c8ea19f11d
- V480 training logs show:
  - LORA_TARGET_PARAMETERS empty
  - REQUIRE_LORA_TARGET_PARAMETER_MATCH=0
  - "LoRA target_parameters: disabled"
  - target_parameter_lora_params: {}
  - target_parameter_lora_tensors: {}
- V480 checkpoints were saved with target_parameters = null:
  - adapter_config sha256 = ca52d6d86aa6be727be6af3b7ce1d8c7a1743c429034a7e3e742f0ec3e8fefe7
- Therefore V480 likely did not preserve the PEFT target_parameters of the seed adapter lineage that reached the 0.86 plateau.

Implemented local guard patches now:
- hf_job_preflight_gate.py:
  - if KG1_STRICT_INIT_ADAPTER_CONFIG=1, compare target_modules and target_parameters between init adapter and job env.
  - block if init adapter has target_parameters but REQUIRE_LORA_TARGET_PARAMETER_MATCH=0.
- kg1_static_safety_gate.py:
  - block active launchers that clear LORA_TARGET_PARAMETERS or disable target-parameter match for MoE adapters.
- V391/V480 launchers:
  - should now set KG1_LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj
  - should now set REQUIRE_LORA_TARGET_PARAMETER_MATCH=1

Key questions for you:
1. Is the missing target_parameters in V480 a plausible root cause for the plateau/regression? Is it proven enough to relaunch a minimal smoke, or could it be incidental?
2. How exactly should a PEFT/LoRA continuation job load a seed adapter that has target_parameters? Compare:
   - create a new LoraConfig and then set_peft_model_state_dict
   - PeftModel.from_pretrained on the seed adapter
   - manually cloning adapter_config.json fields into the new config
   - any other safer approach
3. What concrete preflight checks should run before paid HF GPU:
   - config equality
   - trainable tensor coverage
   - safetensors key coverage
   - target_parameters non-empty
   - output checkpoint config equality
   - any generation/template/offset-mask checks
4. Is same or similar adapter_model.safetensors size with different adapter_config dangerous here? What exactly should we compare besides file size?
5. What is the single best next experiment under FinOps constraints?
   - Prefer max 2 training steps or CPU-only validation first.
   - Must have a kill-switch if first checkpoint does not maintain bit>=136, total>=193, trunc=0.
6. Are there any other likely silent bugs that explain:
   - eval_loss changes but weak accuracy not moving
   - equation +1 paired with bit -1 or bit -2
   - accuracy=0.0000 bugs seen in some custom logging
7. What should we explicitly stop doing?

Required output format:
Return Markdown with these exact sections:

## Verdict
One paragraph with your bottom-line recommendation.

## Evidence Assessment
Bullet list. Mark each point as PROVEN, PLAUSIBLE, or UNKNOWN.

## Root Cause Ranking
Rank 1 to 5 likely causes. Include confidence percentages.

## Required Code/Gate Patches
Concrete checks or pseudocode. Keep it implementation-focused.

## Minimal Next Experiment
A step-by-step experiment with exact pass/fail criteria.

## Stop Doing
Actions that waste money or add noise.

## Missing Evidence
List only files/logs needed to decide better.


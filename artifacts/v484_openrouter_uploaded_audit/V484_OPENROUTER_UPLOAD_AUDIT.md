# V484 OpenRouter Uploaded Audit

Generated: 2026-05-16

Inputs:

| File | Bytes | Messages | Extracted strings | Unique URLs |
|---|---:|---:|---:|---:|
| `C:\Users\davis\Downloads\OpenRouter Chat Sat May 16 2026.json` | 316625 | 14 | 1738 | 42 |
| `C:\Users\davis\Downloads\OpenRouter Chat Sat May 16 2026 (1).json` | 761499 | 15 | 3770 | 118 |

## Executive Finding

The uploaded OpenRouter responses do not provide a new submit-ready dataset,
adapter, or public shortcut. Their useful contribution is narrower and more
important: they strengthen the technical case that the current plateau must be
debugged at the PEFT/LoRA adapter-continuity layer before any additional paid
training.

The actionable consensus is:

1. The adapter-only baseline is still `192/315`, `equation_transform=56/155`,
   `bit_manipulation=136/160`, `truncated=0`.
2. V477/V480-style jobs that improve loss but not ACC should not continue.
3. The most likely implementation risk is PEFT config/load drift around
   `target_parameters`, especially when a manual state_dict path is used for a
   MoE adapter lineage.
4. The next paid job must be blocked until a CPU round-trip gate proves that
   `adapter_config.json`, `adapter_model.safetensors`, trainable LoRA tensors,
   and gradients are structurally correct.

## Models/Services Observed

The files include responses from free and paid OpenRouter models, including
Qwen, DeepSeek, Claude, GPT, NVIDIA Nemotron, OpenRouter Owl, Baidu Cobuddy,
Poolside, GLM, GPT-OSS, and others. The high-signal answers converged on the
same sequence:

- fix/verify PEFT load path first;
- do not treat `eval_loss` as a promotion metric;
- run CPU gates before H200;
- only then attempt a tiny smoke train.

Zero responses supplied a measured adapter-only improvement above
`192/56/136/0`.

## High-Signal Technical Evidence

| Evidence | Interpretation for KG1 |
|---|---|
| PEFT LoRA docs define `target_parameters` for MoE `nn.Parameter` targets, not only `target_modules`. | KG1 V290/V291 lineage must preserve `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`. |
| PEFT low-level adapter-injection docs warn that `target_parameters` adapters require the correct PEFT config; state_dict injection alone is not reliable. | Manual `set_peft_model_state_dict`/direct-load paths cannot be trusted without a round-trip equivalence gate. |
| PEFT target-parameter tests include GPT-OSS/Llama4 expert params. | The expected structure is testable; KG1 should assert it directly. |
| PEFT PR `#2710` fixed multiple `target_parameters` issues and explicitly preferred strict failure over silent errors for unsupported multi-adapter behavior. | KG1 should be fail-closed, not warning-only, around target-parameter load. |
| Public Nemotron challenge repos/models mostly describe generic SFT/QLoRA recipes. | They are useful for reference only; they do not override KG1's measured plateau or submit-only constraints. |

## URLs Reviewed Or Classified

Useful references:

- https://huggingface.co/docs/peft/en/developer_guides/low_level_api
- https://huggingface.co/docs/peft/main/package_reference/lora
- https://github.com/huggingface/peft/blob/261366de/tests/test_target_parameters.py
- https://github.com/huggingface/peft/blob/v0.19.0/src/peft/utils/save_and_load.py
- https://github.com/huggingface/peft/pull/2710
- https://github.com/huggingface/peft/issues/3016
- https://github.com/Ayman-Sabek/NVIDIA_Kaggle_Nemotron
- https://huggingface.co/GaryNENE/nemotron-nano-8b-reasoning-lora
- https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces

Noise or non-actionable for ACC:

- OpenRouter provider pages, legal pages, pricing/navigation pages.
- Generic model cards without measured KG1 adapter-only family deltas.
- Advice to use more epochs, broad SFT, larger LR sweeps, or longer H200 runs
  without a new CPU gate.

## Decisions Applied

Implemented in this cleanup:

- `scripts/hf_job_train_v90.py` now defaults `INIT_ADAPTER_LOAD_MODE` to
  `peft`, so the normal path is `PeftModel.from_pretrained`.
- `scripts/hf_job_preflight_gate.py` blocks `INIT_ADAPTER_LOAD_MODE=manual`
  when the init adapter has `target_parameters`.
- `scripts/kg1_static_safety_gate.py` blocks active launchers that export
  `INIT_ADAPTER_LOAD_MODE='manual'` while using MoE target parameters.
- V391 launcher was updated from manual load mode to PEFT-native load mode.
- The roadmap was rewritten to keep only the active plan and archive old
  history/noise.

## Required Next Work

Create V485 CPU PEFT round-trip gate:

1. Load base model scaffold and seed adapter with `PeftModel.from_pretrained`.
2. Assert `target_parameters` equality with V290/V291 config.
3. Assert LoRA keys, shapes, dtypes, and counts before/after save/reload.
4. Assert non-empty target-parameter LoRA tensors.
5. Run a tiny forward/backward probe and assert gradients on expected LoRA
   weights.
6. Emit `hf_gpu_allowed=true` only if all checks pass.

Only after V485 passes should any paid HF GPU smoke run.


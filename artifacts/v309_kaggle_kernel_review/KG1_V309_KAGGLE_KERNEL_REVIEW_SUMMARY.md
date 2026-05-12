# KG1 V309 Kaggle Kernel Review Summary

Generated: 2026-05-12

Scope:

- `sorokin/aimo2-tir-rm`
- `huikang/end-to-end-finetuning-for-lb-0-85`
- `huikang/tinker-submission-notebook`
- `aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo`
- `siddhvr/lmsys-cahpp-llama3-8b-inference-baseline`
- `kishanvavdara/inference-llama-3-8b`
- `emiz6413/inference-gemma-2-9b-4-bit-qlora`

The requested Kaggle writeup/discussion/comment pages were attempted through the public page and Kaggle CLI-accessible paths. The CLI does not expose writeup/discussion bodies, and the web pages returned JS shells rather than stable article/comment content. Therefore no claim from those pages is accepted here unless supported by a pulled notebook, metadata, or existing local KG1 artifact.

## Directly Useful Findings

### Huikang end-to-end finetuning for LB 0.85

Evidence from pulled notebook:

- Competition source: `nvidia-nemotron-model-reasoning-challenge`.
- Model source: `metric/nemotron-3-nano-30b-a3b-bf16`.
- Machine: `NvidiaRtxPro6000`.
- `LORA_RANK=32`, `LORA_ALPHA=32`, `LORA_DROPOUT=0.0`.
- `MAX_SEQ_LEN=8192`, `NUM_STEPS=1000`, `BATCH_SIZE=32`, `MICRO_BATCH_SIZE=4`.
- `LEARNING_RATE=2e-4`.
- `RESET_WEIGHTS=True`.
- `MOE_TIE_WEIGHTS=True`.
- Target modules include attention, Mamba linear projections, MLP projections, and `lm_head`: `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,in_proj,out_proj,lm_head`.
- Uses token-level weights/masks and Cut Cross Entropy style `linear_cross_entropy`.
- Manually adds LoRA to `lm_head`.
- Implements Tinker-style MoE expert tying for expert-side LoRA updates.

KG1 impact:

- This is a separate high-risk/high-upside training stack, not a small continuation of V290/V308.
- The current PEFT continuation path does not truly exercise the expert `target_parameters` path; V308 logs show the expert target parameters were configured but not matched.
- Next usable action is a gated Huikang-style smoke: verify loss mask corpus, MoE tying, expert tensor conversion, final adapter key compatibility, and then run a very short HF job before any long H200 spend.

### Huikang tinker submission notebook

Evidence from pulled notebook:

- Competition source: `nvidia-nemotron-model-reasoning-challenge`.
- Model sources include `metric/nemotron-3-nano-30b-a3b-bf16` and Huikang adapter versions.
- Rewrites adapter config `target_modules` into the official linear module set.
- Renames trained adapter keys from `base_model.model.model` style into `base_model.model.backbone`.
- Handles expert unfusing: `w1` to per-expert `up_proj`, `w2` to per-expert `down_proj`.
- Converts Mamba `gate_proj + x_proj` into `in_proj` via SVD.
- Keeps `lm_head` in the output path.
- Compares final adapter keys against reference keys before packaging.

KG1 impact:

- Any Huikang-style train must add a packaging gate, not only `PeftModel.save_pretrained`.
- The gate must validate tensor count, expected official key names, expert unfusing, Mamba merge, `lm_head`, and shape parity against a known accepted adapter.

### Sorokin AIMO2 TIR/RM

Evidence from pulled notebook:

- Uses vLLM generation, executable code snippets, boxed-answer extraction, repeated sampling, majority vote, and reward-model rescoring.
- Reward model: Qwen2.5 Math RM 72B served through vLLM with bitsandbytes quantization.
- Generates and executes code snippets with timeout, then chooses boxed answers by reward/majority.

KG1 impact:

- Not directly submit-ready for KG1 adapter-only packaging.
- Useful as a teacher/verifier pattern for `equation_transform`: generate multiple candidate traces, execute/check deterministic pieces, then distill verified answers into adapter training data.

### EEDI Qwen32B logits-processor notebook

Evidence from pulled notebook:

- Uses retrieval + Qwen2.5 32B AWQ via vLLM.
- Uses `MultipleChoiceLogitsProcessor` to force choices with `temperature=0`, `top_k=1`, `max_tokens=1`, `seed=777`.

KG1 impact:

- Not directly compatible with official adapter-only KG1 submission if inference is fixed.
- Useful for internal teacher/eval probes where answer format is discrete and malformed outputs must be eliminated.

## Low Direct Value Findings

- LMSYS Llama3/Gemma notebooks show useful engineering patterns: dual-GPU inference, 8-bit/4-bit loading, unmerged LoRA to avoid quantization merge error, TTA/blending.
- These are not KG1-domain solvers and should not drive immediate training decisions.
- Keep as engineering references only.

## Operational Decision

Priority order after this review:

1. Finish and weak-evaluate V308 because it is already running and passed HF gates.
2. If V308 checkpoint-30 or final improves weak family accuracy, promote only through the existing weak/full gates.
3. If V308 does not improve weak accuracy, stop simple continuation runs.
4. Implement a V309/V310 Huikang-style smoke only after adding the expert/Mamba/lm_head packaging gate.
5. Use AIMO2 TIR/RM and logits-processor patterns only to produce verified teacher traces, not as direct KG1 submissions.

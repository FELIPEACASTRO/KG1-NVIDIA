# V194 Adapter Audit - 2026-05-06

Fonte Drive:

- Pasta: `init_adapter_v194_rank19_build/adapter`
- `adapter_config.json`
- `adapter_model.safetensors`

Copia local:

- `artifacts/drive_exports/v194_adapter/adapter_config.json`

## Campos Relevantes

- `peft_type`: `LORA`
- `task_type`: `CAUSAL_LM`
- `r`: `32`
- `lora_alpha`: `32`
- `lora_dropout`: `0.0`
- `bias`: `none`
- `modules_to_save`: `null`
- `target_parameters`:
  - `mlp.experts.gate_up_proj`
  - `mlp.experts.down_proj`
- `target_modules`:
  - `k_proj`
  - `up_proj`
  - `down_proj`
  - `out_proj`
  - `v_proj`
  - `q_proj`
  - `lm_head`
  - `o_proj`
  - `in_proj`
- `base_model_name_or_path`: `metric/nemotron-3-nano-30b-a3b-bf16`
- eval report base model used: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Gate Interpretation

- Rank gate passes: `r=32`, equal to the competition max.
- Dropout continuation default should stay `0.0`.
- Alpha continuation default should stay `32`.
- Target modules should not be guessed; continuation must preserve this config unless a separate compatibility test proves otherwise.
- `lm_head` is present in `target_modules`.

## Risk

Earlier roadmap gates preferred no `lm_head`/`embed_tokens` for new submit candidates. V194 reproduces with `lm_head` in `target_modules`, so this is not an automatic rejection for the protected baseline. It is a continuation/surgery risk:

- Do not strip `lm_head` from V194 unless a full solve-rate gate proves no regression.
- Do not add new `modules_to_save`.
- Do not change rank/target modules during the first continuation branch.
- Any new candidate derived from V194 must record this exception in the adapter audit.

## Decision

Continuation from V194 remains possible, but only after:

1. dataset gate passes;
2. tokenizer/chat-template audit is recorded;
3. training config keeps `r=32`, `lora_alpha=32`, `dropout=0.0`, and the audited target modules;
4. weak eval improves over `190/315`;
5. strong families remain `632/632`.

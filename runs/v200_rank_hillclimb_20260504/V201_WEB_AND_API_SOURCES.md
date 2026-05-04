# V201 Web And API Sources

Generated: 2026-05-04

## Sources Used

- NVIDIA Megatron Bridge Nemotron 3 Nano docs:
  - https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html
  - Used for LoRA support and target-module discipline around `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`, `in_proj`, and `out_proj`.
- OpenRouter chat completion docs:
  - https://openrouter.ai/docs/api-reference/chat-completion
  - Used to run the multi-model roadmap panel through `/api/v1/chat/completions`.
- OpenRouter quickstart:
  - https://openrouter.ai/docs/quickstart
  - Used to confirm standard OpenAI-compatible request shape.
- Prometheus public Nemotron Kaggle GRPO/SFT lead:
  - https://huggingface.co/datasets/prometheus04/nvidia-kaggle/blob/main/train_grpo.py
  - Used only as a research lead. It supports solver/reward-driven data verification as an offline idea, not direct submission.
- Kaggle competition reference:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
  - Used as the competition identity/reference.

## Local Evidence Used

- `best_baseline_registry.json`
- `old_notebook_audit_raw.json`
- `openrouter_v201_panel_results.json`
- `WEB_RESEARCH_087_REPORT.md` from `runs/web_research_087_20260502`

## Decision Impact

The sources support a conservative plan:

- keep LoRA target modules disciplined;
- use OpenRouter models only for offline review;
- avoid direct use of public adapters without local gates;
- build one micro experiment from the exact V194 baseline.

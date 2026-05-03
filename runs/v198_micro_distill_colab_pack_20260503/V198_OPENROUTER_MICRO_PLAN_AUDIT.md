# V198 OpenRouter Micro Plan Audit

- ok_models: `5/5`
- estimated_cost: `$0.038196`
- decisions: `{'proceed': 5}`

## Model verdicts

- `openai/gpt-5.5` decision=`proceed` confidence=`0.82` risk=`Micro-distillation may overfit narrow V197 gains or cause stable-family regression despite anchor exclusion; duplicate prompts and anti-regression repeats could` change=`Use early checkpoint selection with a hard no-regression gate; keep max_steps 45 but evaluate 15/30/45.`
- `anthropic/claude-sonnet-4.6` decision=`proceed` confidence=`82` risk=`30-row strict_gain_distill signal is extremely sparse relative to 1729-row rehearsal mass; attention-only fine-tuning on this imbalance may wash out the two val` change=`Add a batch-level dedup filter keyed on prompt hash before SFT loop to eliminate any residual duplicate_prompts risk, and log per-category validation deltas exp`
- `google/gemini-2.5-flash` decision=`proceed` confidence=`0.9` risk=`Regression in stable families due to limited anti-regression data for all families, or insufficient micro-distillation for significant gains.` change=`Consider adding more diverse anti-regression data for all families, not just 'bit_manipulation' and 'equation_transform', if regression becomes an issue. Also, `
- `qwen/qwen3.6-flash` decision=`proceed` confidence=`0.85` risk=`Overfitting to rehearsal data or duplicate prompts causing minor local score fluctuation` change=`Implement early stopping at step 30 if validation loss plateaus; audit duplicate prompts for potential label leakage`
- `deepseek/deepseek-v4-flash` decision=`proceed` confidence=`0.85` risk=`V198 includes 116 V196 wrong anti-regression rows and 30 V197 strict gain distill rows; if any of those cause subtle regression on stable families (e.g., numera` change=`none`

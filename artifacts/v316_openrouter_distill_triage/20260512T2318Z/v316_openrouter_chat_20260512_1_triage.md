# V316 OpenRouter Chat Triage - 2026-05-12

Source: `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026 (1).json`
Source bytes: `199244`

## deepseek/deepseek-r1-distill-qwen-32b

- chars: `7090`
- sha256: `6554453f80db02bd9299c50cee5d38cf1eb4dfc0c27bcbedc7b84b91d91bb2fc`
- file: `02_deepseek__deepseek-r1-distill-qwen-32b.md`

## deepseek/deepseek-r1-distill-llama-70b

- chars: `7251`
- sha256: `e5ae9ab651172010271db32d1b01fbb4f2e2c68141b6c52ecd0992bf6495c587`
- file: `03_deepseek__deepseek-r1-distill-llama-70b.md`

## deepseek/deepseek-v4-pro

- chars: `34917`
- sha256: `2103bd168821562ed1aff8a767649a468621b839983b59dd1564be5650b36400`
- file: `04_deepseek__deepseek-v4-pro.md`

## qwen/qwen3.6-max-preview

- chars: `16593`
- sha256: `3d9e4debbaf0404c734f30535d7d415c015b8a1fe3257a02d42b6a6a405aea72`
- file: `05_qwen__qwen3.6-max-preview.md`

## qwen/qwen3.6-plus

- chars: `21922`
- sha256: `dc731af2977eb467709307cabb7d28c684a25ecf98dafdf21f6dd835ffa3193f`
- file: `06_qwen__qwen3.6-plus.md`

## openai/gpt-5.3-codex

- chars: `10835`
- sha256: `9904231eda97071800f8d329c7b74571170cb66a679406ba72f1f59c41596df1`
- file: `07_openai__gpt-5.3-codex.md`

## Consolidated Evidence-Bound Triage

### Signal dilution is the dominant plausible cause of V313/V315 failure.

- supporting_models: `deepseek-v4-pro, qwen3.6-max-preview, qwen3.6-plus, gpt-5.3-codex`
- evidence_status: `consistent_with_observed_plateau`
- action: Keep V316/V317 training sets small, targeted, and heavily weighted toward the 4 equation and 11 bit verified gains plus keepers.

### Final-answer tokens are likely underweighted relative to trace tokens.

- supporting_models: `deepseek-v4-pro, gpt-5.3-codex`
- evidence_status: `hypothesis_consistent_with_short_numeric_targets`
- action: Add V317 option for answer-span weighted SFT or short final-answer-aligned completions if V316 does not move weak equation above 56.

### Hard negatives should mirror exact observed wrong answers.

- supporting_models: `deepseek-v4-pro, qwen3.6-max-preview, qwen3.6-plus, gpt-5.3-codex`
- evidence_status: `directly_actionable_from_verified_pairs`
- action: For equation, contrast 55 vs -55, -92 vs 92, 03 vs 30, 35 vs 134; for bit, contrast safe ternary/binary guard outcomes against baseline wrong outputs.

### Bit regression is catastrophic-interference risk, not a solved secondary issue.

- supporting_models: `deepseek-v4-pro, qwen3.6-max-preview, qwen3.6-plus, gpt-5.3-codex`
- evidence_status: `supported_by_V315_ckpt16_bit_134`
- action: Keep bit keeper replay and reject checkpoints below bit=135 immediately; promote only bit>=136.

### Eval loss / train loss is not sufficient as a decision metric.

- supporting_models: `gpt-5.3-codex, qwen3.6-max-preview`
- evidence_status: `supported_by_repeated_low_loss_runs_without_family_gain`
- action: Continue using weak family gates, targeted probes, truncation checks, and full non-regression gates.

## Usable Now

- Proceed with V316 MLP/expert-targeted LoRA because attention-heavy V308 and V312-derived SFT/preference did not acquire gains.
- Keep the V316 promotion gate strict: total>=192/193, eq>=56/60, bit>=136, truncation non-regression.
- Add roadmap V317: answer-span weighted SFT / compact-output distillation if V316 plateaus.

## Rejected Or Deferred

- High learning rates such as 8e-5 to 1.5e-4 or 10k+ steps from generic OpenRouter suggestions. Reason: Contradicts local tiny-LR PEFT lineage and would likely overwrite the good 0.86+ adapter under HF budget.
- Training on impoverished inputs only, such as bare 55 -> -55 pairs. Reason: Mismatch with Kaggle prompt distribution; can memorize isolated strings and fail inference prompt format.
- Sequential 100% equation curriculum without bit replay. Reason: High risk of bit regression; V315 already showed bit can drop under targeted training.
- Treating OpenRouter advice as empirical evidence. Reason: The file contains model hypotheses, not new measured Kaggle/HF results.

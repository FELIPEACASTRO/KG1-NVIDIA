# V202B long-context pretokenized dry-run/smoke

- Notebook: `notebooks/KG1_V202B_H100_A100_LONG_CONTEXT_PRETOKENIZED_COLAB_PRO.ipynb`.
- Starts only from exact V194 rank-19 adapter SHA `01259fef943bc16c31d8f7907be076cc987381a6a1bbe732b1b33c2d9f2ea95f` and zip SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- Uses full Kaggle dataset `atahalam/tonghuikang-0-87-nemotron-dataset` zip SHA `461776d6bc44d482988d23c4e584128b66a93d2500fe7c428f4e895ab42e9eb8`.
- Reads Tong/Huikang token/mask segments directly from the zip via `PRETOKENIZED_ARCHIVE_ZIP`; no chat-template retokenization.
- Enforces `MAX_LENGTH=8192`, `LORA_R=32`, attention-only trainable filter, `BATCH_SIZE=1`, `MICRO_BATCH_SIZE=1`.
- Default run performs model-load dry-run only and writes `dry_run_model_recipe_report.json`.
- Optional one-step smoke train is disabled by default and remains no-submit with final eval no-regression gate.
- Raw `generation.jsonl` and `nemotron_traj.csv` are deliberately not used as SFT because the V202 audit showed many false/partial outputs.
- No Kaggle submit cell exists.

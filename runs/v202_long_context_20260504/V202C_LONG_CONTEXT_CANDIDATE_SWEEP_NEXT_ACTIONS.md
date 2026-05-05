# V202C long-context candidate sweep

- Notebook: `notebooks/KG1_V202C_H100_A100_LONG_CONTEXT_CANDIDATE_SWEEP_COLAB_PRO.ipynb`.
- Requires a passed V202B smoke summary in Drive before running.
- Runs three independent candidates from exact V194 rank-19, not chained from V202B smoke.
- Candidate A: all Tong categories, 3 steps, LR `2e-8`.
- Candidate B: official-category-only subset, 3 steps, LR `2e-8`.
- Candidate C: all Tong categories, 5 steps, LR `1e-8`.
- All candidates enforce `MAX_LENGTH=8192`, `BATCH_SIZE=1`, attention-only trainable filter, final eval <= baseline.
- No Kaggle submit cell exists; passed candidates still require vLLM/Kaggle-layout gates.

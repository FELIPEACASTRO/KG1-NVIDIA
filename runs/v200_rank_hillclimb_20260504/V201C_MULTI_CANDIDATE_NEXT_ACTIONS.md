# V201C three-candidate micro-train

- Notebook: `notebooks/KG1_V201C_H100_A100_MULTI_CANDIDATE_MICRO_COLAB_PRO.ipynb`.
- Runs three independent candidates from exact V194 rank-19, not sequential phases on one adapter.
- Candidate A: neutral shuffle, 3 steps, LR `2e-7 -> 1e-7`.
- Candidate B: light equation/cryptarithm weighting, 2 steps, LR `1e-7 -> 5e-8`.
- Candidate C: light bit/cipher weighting, 2 steps, LR `1e-7 -> 5e-8`.
- Each candidate has baseline eval before training and final eval no-regression gate.
- Only passed candidates are converted and preflighted; no Kaggle submit cell exists.
- H100 or A100 80GB High-RAM is required; A100 40GB is blocked.

# V199 conservative continuation

- Notebook: `notebooks/KG1_V199_CONSERVATIVE_CONTINUE_COLAB_PRO.ipynb`
- Starts from exact V194 rank-19 adapter SHA `01259fef...`.
- V194 evidence: public score `0.86`, rank `19/2613`, zip SHA `49886191...`.
- Hard rule: every next training/adjustment must start from the best-known Kaggle ranking baseline, not the latest submission.
- Rebuilds V194 from `aaitdads/my-0p86-adapter` plus kernel output `51997779`, then blocks on SHA mismatch.
- Runs 20 steps at LR `3e-6 -> 8e-7`.
- Saves checkpoints every 10 steps and evaluates every 10 steps.
- Converts and gates final/checkpoint candidates.
- Does not submit to Kaggle automatically.

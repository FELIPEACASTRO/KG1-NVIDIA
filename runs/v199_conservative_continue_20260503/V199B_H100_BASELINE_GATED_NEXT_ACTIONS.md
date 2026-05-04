# V199B H100 baseline-gated conservative continuation

- Notebook: `notebooks/KG1_V199B_H100_BASELINE_GATED_COLAB_PRO.ipynb`
- Requires H100 with at least 75 GiB GPU memory, High-RAM runtime, and at least 100 GiB free on `/content`.
- Starts from exact V194 rank-19 adapter SHA `01259fef...` extracted from the validated `submission.zip` SHA `49886191...`.
- Evaluates the exact V194 baseline before training on the same validation split.
- Runs 10 steps at LR `1e-6 -> 3e-7`.
- Evaluates every 5 steps and aborts if eval loss exceeds baseline by more than `0.02`.
- Blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.
- Converts and gates candidates only after the baseline gate passes; does not submit to Kaggle automatically.

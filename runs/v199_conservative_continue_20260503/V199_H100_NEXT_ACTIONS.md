# V199 H100 High-RAM conservative continuation

- Notebook: `notebooks/KG1_V199_H100_HIGH_RAM_COLAB_PRO.ipynb`
- Requires H100 with at least 75 GiB GPU memory, High-RAM runtime, and at least 100 GiB free on `/content`.
- Starts from exact V194 rank-19 adapter SHA `01259fef...`.
- V194 evidence: public score `0.86`, rank `19/2613`, zip SHA `49886191...`.
- Primary init path: exact V194 rank-19 `submission.zip` from Drive/env, validated by zip/model/config SHA before extraction.
- Automatic Tinker reconstruction is disabled unless `ALLOW_V194_REBUILD_FALLBACK=1` is explicitly set.
- Runs 20 steps at LR `3e-6 -> 8e-7`.
- Converts and gates final/checkpoint candidates; does not submit to Kaggle automatically.

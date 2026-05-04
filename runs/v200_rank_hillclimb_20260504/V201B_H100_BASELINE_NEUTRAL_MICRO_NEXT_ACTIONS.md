# V201B H100/A100 baseline-neutral micro-train

- Notebook: `notebooks/KG1_V201B_H100_BASELINE_NEUTRAL_MICRO_COLAB_PRO.ipynb`.
- V201A is blocked and must not be used as init because final eval regressed: `1.1222 > 1.1205`.
- Production baseline remains V194/ref `52275052`, public score `0.86`, rank `19/2613`.
- Starts only from exact V194 `submission.zip` SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- Runs 3 steps at LR `2e-7 -> 1e-7` with normal shuffled sampling, no weighted replacement and no custom weight maps.
- Trains attention LoRA modules only: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Evaluates V194 baseline before training and blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.
- Converts with `kg1_v201b_posttrain_gate.py`; no Kaggle submit cell is present.
- Submit only if the posttrain gate is `READY`, local eval is non-regressive, and Kaggle authorization is explicit.

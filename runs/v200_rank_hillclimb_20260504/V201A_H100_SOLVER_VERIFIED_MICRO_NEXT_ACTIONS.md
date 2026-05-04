# V201A H100 solver-verified weak-category micro-train

- Notebook: `notebooks/KG1_V201A_H100_SOLVER_VERIFIED_MICRO_COLAB_PRO.ipynb`.
- Production baseline remains V194/ref `52275052`, public score `0.86`, rank `19/2613`.
- Uses Drive root `/content/drive/MyDrive/KG1_NVIDIA_V201` for V201A outputs, while still accepting the V194 zip from `/content/drive/MyDrive/Submit` and legacy `/content/drive/MyDrive/KG1_NVIDIA_V199` paths.
- Starts only from exact V194 `submission.zip` SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- Runs 5 steps at LR `3e-7 -> 1e-7`.
- Uses weighted weak-category sampling with `key=value` maps: bit manipulation, cipher, cryptarithm, and equation numeric subcategories.
- Trains attention LoRA modules only: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Evaluates V194 baseline before training and blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.
- Converts with `kg1_v201a_posttrain_gate.py`; no Kaggle submit cell is present.
- After training, submit only if the posttrain gate is `READY` and Kaggle authorization is explicit.

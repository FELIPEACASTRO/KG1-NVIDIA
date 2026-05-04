# V200A H100 micro attention-only hillclimb

- Notebook: `notebooks/KG1_V200A_H100_MICRO_ATTENTION_COLAB_PRO.ipynb`.
- Production baseline remains V194/ref `52275052`, public score `0.86`, rank `19/2613`.
- Starts only from exact V194 `submission.zip` SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- Runs 5 steps at LR `5e-7 -> 2e-7`.
- Trains attention LoRA modules only: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Evaluates V194 baseline before training and blocks final promotion unless `final_eval_loss <= baseline_eval_loss`.
- Converts with `kg1_v200a_posttrain_gate.py`; no Kaggle submit cell is present.
- After training, compare the gate report and ZIP SHA before deciding whether to submit.

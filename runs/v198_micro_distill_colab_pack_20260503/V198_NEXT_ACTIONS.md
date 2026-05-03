# V198 next actions

Status: Colab pack generated locally. No Kaggle submission was created.

## Run order

1. Copy `kg1_v198_colab_pack.zip` to `/content/drive/MyDrive/KG1_NVIDIA_V198/` or push the notebook/pack branch.
2. Run `notebooks/KG1_V198_MICRO_DISTILL_COLAB_PRO.ipynb` on Colab Pro H100.
3. Prefer V195 `final_adapter`; if unavailable, checkpoint-110/75/55; fallback baseline is allowed but weaker.
4. After conversion, run local inference/prescore before any Kaggle submit.

## Stop rule

- If eval loss at step 30/45 is worse than the V195 continuation trend, stop and keep V195.
- If local validation has any stable-family regression, do not submit.

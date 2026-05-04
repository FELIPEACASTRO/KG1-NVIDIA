# V201 >=0.87 Master Roadmap

Generated: 2026-05-04

## Baseline Rule

Current production baseline:

- Version: V194.
- Kaggle ref: `52275052`.
- Public score: `0.86`.
- User-confirmed rank: `19/2613`.
- Zip SHA256: `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.

Every new training, merge, or package attempt must start from this baseline unless a later submission beats it on Kaggle rank/score.

## What The Research Changed

The web/API/OpenRouter audit did not find a proven public adapter above `0.86`.

It did find a consistent strategy:

- Keep lineage fixed to V194.
- Avoid broad adapter soups.
- Avoid high-learning-rate continuation.
- Use API/model panels only offline for review and data verification, not in the Kaggle artifact.
- Try one small weak-category update at a time.

## Candidate V201A

Purpose: push weak categories without disturbing the 0.86 baseline.

- Notebook: `notebooks/KG1_V201A_H100_SOLVER_VERIFIED_MICRO_COLAB_PRO.ipynb`.
- Output root: `/content/drive/MyDrive/KG1_NVIDIA_V201/output_v201a_h100_solver_verified_micro_5`.
- Init: exact V194 rank-19 zip only.
- Steps: `5`.
- LR: `3e-7 -> 1e-7`.
- Trainable modules: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Sampling mode: `weighted_replacement`.
- Weak-category emphasis:
  - `bit_manipulation:2.5`
  - `cipher:2.0`
  - `cryptarithm_deduce:3.0`
  - `cryptarithm_guess:2.0`
  - `equation_numeric_deduce:3.0`
  - `equation_numeric_guess:2.0`
  - `equation_transform:1.5`
- Rehearsal source mix remains active so strong categories are not dropped.

## Hard Gates

Before training:

- V194 zip SHA must match exactly.
- Adapter model/config SHA must match the rank-19 baseline after extraction.
- Notebook must have no auto-submit code.
- H100/high-RAM runtime should be used by default.

During training:

- Baseline eval must run before any training step.
- Eval at step 5 must not exceed baseline by more than `0.005`.
- Final eval must be `<= baseline_eval_loss`.

After training:

- Posttrain conversion must produce a root-only adapter zip.
- Candidate label must be `final`.
- Doublecheck JSON must exist.
- Human authorization is required before Kaggle submit.

After Kaggle score:

- `< 0.86`: discard.
- `= 0.86`: quarantine unless rank/selection improves over V194 rank 19.
- `> 0.86`: promote as the new best baseline.

## Rejected Paths

- Any V198-derived candidate.
- Any unknown Drive adapter source.
- Any notebook with Kaggle auto-submit.
- Any private Kaggle kernel output dependency.
- Any broad soup/high-alpha merge before a better source is found.
- Any candidate that passes packaging but fails baseline-eval no-regression.

## Next Experiments If V201A Scores 0.86

Do not promote V201A automatically.

Then run V201B as a separate micro-probe:

- Same exact V194 init.
- Same gates.
- 3 steps instead of 5.
- LR `2e-7 -> 1e-7`.
- Only equation/cryptarithm weighting.

If V201A scores below `0.86`, discard it and do not reuse it as an init.

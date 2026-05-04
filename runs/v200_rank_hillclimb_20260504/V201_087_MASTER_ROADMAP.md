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
  - `bit_manipulation=2.5`
  - `cipher=2.0`
  - `cryptarithm_deduce=3.0`
  - `cryptarithm_guess=2.0`
  - `equation_numeric_deduce=3.0`
  - `equation_numeric_guess=2.0`
  - `equation_transform=1.5`
- Rehearsal source mix remains active so strong categories are not dropped.

Status after execution:

- Baseline eval: `1.1205`.
- Final eval: `1.1222`.
- Decision: blocked by `final_eval_loss <= baseline_eval_loss`.
- Action: do not submit and do not reuse V201A as init.

## Candidate V201B

Purpose: retry with less movement after V201A regressed locally.

- Notebook: `notebooks/KG1_V201B_H100_BASELINE_NEUTRAL_MICRO_COLAB_PRO.ipynb`.
- Output root: `/content/drive/MyDrive/KG1_NVIDIA_V201/output_v201b_h100_baseline_neutral_micro_3`.
- Init: exact V194 rank-19 zip only.
- Steps: `3`.
- LR: `2e-7 -> 1e-7`.
- Trainable modules: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Sampling mode: `shuffle`.
- Custom source/subcategory weights: disabled.
- Gate: final eval must be `<= baseline_eval_loss`; otherwise the adapter is forensic only.

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

## Next Experiments If V201B Scores 0.86

Do not promote V201B automatically.

Then run V201C as a separate targeted probe:

- Same exact V194 init.
- Same gates.
- 2 or 3 steps.
- LR `1e-7 -> 5e-8`.
- Only one category at a time, starting with equation/cryptarithm.

If V201B scores below `0.86` or fails local no-regression, discard it and do not reuse it as an init.

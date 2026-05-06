# V207B External Adapter Triage Runbook - 2026-05-06

## Purpose

Continue the roadmap after the V207A/V207B evidence rejected V206B/V206C.

This stage evaluates external/current adapters against the fixed V207A official-like ACC harness. It is designed to spend full 947-row H100/A100 time only if a candidate first beats V194 on the weak families:

- `equation_transform`
- `bit_manipulation`

## Generated Notebook

Local notebook:

`notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb`

Builder:

`scripts/build_v207b_external_adapter_triage_colab.py`

Expected Colab URL after the notebook is pushed to branch `v207b-external-triage`:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v207b-external-triage/notebooks/KG1_V207B_EXTERNAL_ADAPTER_TRIAGE_COLAB.ipynb`

Note: the URL above will work only after the notebook is pushed to the referenced GitHub branch.

## Safety Properties

- No Kaggle submission command.
- `ALLOW_KAGGLE_SUBMIT = False`.
- No training.
- Uses existing V207A validation artifacts from Drive.
- Audits adapter structure before evaluation.
- Skips known rejected V206B/V206C paths by default.
- Runs full 947-row evaluation only when `weak_delta > 0`.

## Required Existing Drive Inputs

The notebook expects V207A outputs already produced under:

`/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate`

Required files:

- `validation/official_train_seed42_stratified10_val.csv`
- `v194_baseline_eval/v194_baseline_predictions.csv`
- `v194_baseline_eval/v194_baseline_per_task.csv`
- `v194_baseline_eval/v194_baseline_eval_report.json`

Baseline used:

- V194: `822/947 = 0.868004`
- weak-family baseline: `190/315`

## Candidate Discovery

Manual defaults included:

- V194 duplicate sanity:
  - `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter`
  - `KG1_NVIDIA_V202D/final_v194_keep_no_submit/adapter`
- V199B candidate locations, if present:
  - `KG1_NVIDIA_V199B/final_adapter`
  - `KG1_NVIDIA_V199B/adapter`
- Public/external landing-zone examples:
  - `KG1_PUBLIC_ADAPTERS/aaitdads_my_0p86_adapter`
  - `KG1_PUBLIC_ADAPTERS/huikang_tinker_v27/adapter`
  - `KG1_PUBLIC_ADAPTERS/huikang_tinker_v26/adapter`
  - `KG1_PUBLIC_ADAPTERS/huikang_tinker_v20/adapter`
  - `KG1_PUBLIC_ADAPTERS/kien_variant/adapter`
  - `KG1_PUBLIC_ADAPTERS/bugkeeper_v20/adapter`
  - `KG1_PUBLIC_ADAPTERS/dgxchen_trained_adapter`

Automatic scan roots:

- `KG1_PUBLIC_ADAPTERS`
- `KG1_NVIDIA_PUBLIC_ADAPTERS`
- `KG1_NVIDIA_EXTERNAL`
- `KG1_NVIDIA_V199B`
- `KG1_NVIDIA_V202D`
- `KG1_NVIDIA_V204`
- `KG1_NVIDIA_TINKER`
- `KG1_NVIDIA_KIEN`

## Output Files

All outputs are written to:

`/content/drive/MyDrive/KG1_NVIDIA_V207B/output_v207b_external_adapter_triage`

Key outputs:

- `v207b_discovered_candidates.json`
- `v207b_adapter_structure_audit.csv`
- `v207b_adapter_structure_audit.json`
- `v207b_weak_screen_results.csv`
- `v207b_weak_screen_results.json`
- `v207b_full_gate_results.csv`
- `v207b_full_gate_results.json`
- `V207B_FINAL_RUN_SUMMARY.json`

## Decision Rule

Reject immediately if:

- adapter rank is missing or `>32`;
- required adapter files are missing;
- bad historical lm_head namespace appears;
- weak-family delta is `<=0`;
- truncation is materially worse with no correct-answer gain.

Promote to full 947-row gate only if:

- structure audit passes;
- weak-family rows equal `315`;
- weak correct is greater than V194's `190/315`.

Submission is still blocked after full approval until a human explicitly approves Kaggle submission.

## Local Validation Performed

- Generated notebook successfully.
- Notebook JSON parsed successfully.
- Notebook has 13 cells.
- Required progress markers found.
- No Kaggle submit command found.
- `python -m py_compile` passed for the builder and critical gate/eval scripts.
- `python test_scoring.py` passed `29/29`.

## Human Intervention Point

The next required human action is to run this notebook in Colab after pushing it, or open the local notebook manually in Colab with Drive mounted.

No further local CPU-only work can determine whether external adapters beat V194; the next signal requires A100/H100 execution with the Drive adapter files.


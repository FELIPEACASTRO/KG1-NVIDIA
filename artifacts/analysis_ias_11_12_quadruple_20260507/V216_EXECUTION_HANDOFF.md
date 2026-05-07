# V216 Execution Handoff

Status: **READY_FOR_COLAB_AFTER_PUSH**

## What Changed

- Built `data/v216/v216_score_push_train.jsonl`.
- Built `data/v216/v216_score_push_val.jsonl`.
- Built `data/v216/v216_score_push_manifest.json`.
- Built `notebooks/KG1_V216_EQUATION_SCORE_PUSH_COLAB.ipynb`.
- Added reproducible builders:
  - `scripts/build_v216_score_push_dataset.py`
  - `scripts/build_v216_equation_score_push_colab.py`

## Critical Correction

The raw V216 focused files contained empty-answer rows:

- train empty-answer rows removed: `249`
- val empty-answer rows removed: `17`

The score-push dataset excludes those rows. Do **not** train directly on
`v216_equation_symbolic_focus_train.jsonl`.

## Dataset

- train rows: `10210`
- validation rows: `681`
- train sha256: `8cfd065c102187b12c131aae7475c35e28073721175b4e6108004b0afc4d5d03`
- val sha256: `80efe71260c8589b998699543c85aff3ff140bc90e431dfa0ec33bce3e0921c0`
- weak validation leaks: `0`
- extraction mismatches: `0`
- empty answers: `0`

Train family counts:

- `equation_transform`: `6935`
- `bit_manipulation`: `2699`
- `gravity_constant`: `144`
- `numeral_system`: `144`
- `text_encryption`: `144`
- `unit_conversion`: `144`

## Colab

Colab URL:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v216-equation-score-push/notebooks/KG1_V216_EQUATION_SCORE_PUSH_COLAB.ipynb`

This URL works only after the notebook and data are pushed to the
`v216-equation-score-push` branch.

## Default Train Recipe

- initial adapter: protected V194
- dataset: V216 score-push
- LR: `3e-8`
- max steps: `24`
- trainable modules: `q_proj,k_proj,v_proj,o_proj,out_proj,in_proj`
- sampling mode: `weighted_replacement`
- source weights:
  `v216_synthetic_kg1_symbolic_rules=0.55,v216_synthetic_kg1_numeric_rules=0.45,v216_synthetic_kg1_bit_rules=0.25,v216_base_clean_safe_strict_equation=0.85,v216_base_clean_safe_strict_bit=0.75,v215_replay_anchor=0.9`
- subcategory weights:
  `equation_symbolic_binary=1.2,equation_symbolic_unary=1.2,equation_numeric=0.85,bit_manipulation=0.75`

## Gates

Baseline assumptions:

- weak total: `190/315`
- equation_transform: `55/155`
- bit_manipulation: `135/160`

Promote to full validation only if:

- weak total >= `193`
- equation_transform >= `60`
- bit_manipulation >= `133`
- weak truncation <= `3`

Package only if:

- full validation >= `831`
- full truncation <= `4`

The notebook does not submit to Kaggle.

## Roadmap

1. Push this local work to branch `v216-equation-score-push`.
2. Open the Colab URL above on H100.
3. Run all cells.
4. If dry-run fails on trainable parameter ratio, reduce
   `KG1_V216_TRAINABLE_MODULES` to `q_proj,k_proj,v_proj,o_proj`.
5. If weak gate fails, reject the adapter and do not spend full validation time.
6. If weak gate passes, run full validation through the notebook.
7. If full gate passes, review the generated package manually before using one of
   today's Kaggle submissions.

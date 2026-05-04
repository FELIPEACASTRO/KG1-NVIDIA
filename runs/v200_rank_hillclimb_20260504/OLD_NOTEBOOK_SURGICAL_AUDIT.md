# Old Notebook Surgical Audit

Generated: 2026-05-04

Raw audit file: `old_notebook_audit_raw.json`.

## Scope

- Notebooks scanned: 134.
- Main purpose: identify repeatable failure modes before building the next >=0.87 candidate.

## Findings

- `auto_submit`: 42 notebooks.
  - Risk: a notebook can submit a candidate before the rank-baseline rule is checked.
- `kaggle_api_in_notebook`: 12 notebooks.
  - Risk: side effects and credential/path failures inside training notebooks.
- `private_kernel_output`: 2 notebooks.
  - Risk: Kaggle permission failures and non-reproducible lineage reconstruction.
- `files_upload`: 2 notebooks.
  - Risk: stalled Colab upload flow and ambiguous source zip.
- `h100_gate`: 50 notebooks.
  - This is useful, but many of these did not also include the current best-baseline rule.
- `best_baseline_rule`: 8 notebooks.
  - Only late notebooks encode the rule that all new work must start from the best ranked submission.
- `baseline_eval_gate`: 4 notebooks.
  - Only V199B/V200-class notebooks block local eval regression before packaging.

## Root Causes Seen In Regressions

- Wrong lineage: V198 became the de facto source in some flows and scored `0.84`.
- Broad update/soup risk: V191 scored `0.78`.
- Focal overtraining risk: V174 scored `0.41`.
- Packaging/stripping drift: historical attempts produced `0.50-0.54`.
- External Kaggle kernel output dependency: denied permissions made lineage reconstruction non-deterministic.

## Fix Applied In V201A

- Exact V194 zip SHA is mandatory before training.
- Best-baseline rule is embedded in the notebook.
- Training starts from V194 rank-19 only.
- No Kaggle submit cell exists.
- Training is 5-step, low-LR, attention-only.
- The trainer evaluates V194 before any update.
- Final adapter is blocked if `final_eval_loss > baseline_eval_loss`.
- Posttrain gate checks adapter conversion and zip layout before any human submit decision.

# OpenRouter V201 Panel Summary

Generated: 2026-05-04

## Execution

- API key was found in local environment/config material and validated without printing it.
- Model catalog snapshot: `openrouter_models_snapshot.json`.
- Panel result file: `openrouter_v201_panel_results.json`.
- Selected models: 12.
- Successful API responses: 11.
- Rate-limited/upstream failure: 1 (`google/gemma-4-26b-a4b-it:free`).
- Useful non-empty responses: 9.

## Consensus

The useful models converged on the same operational conclusion:

- Do not start from V198, V195, V199B, or an unknown Drive adapter.
- Do not use broad soups or high-alpha interpolation.
- Do not submit automatically from a notebook.
- The only defensible next experiment is a micro candidate from the exact V194 rank-19 zip.
- The update should be tiny: 3-8 steps, very low LR, attention-only trainable modules, weak-category weighted sampling, and strong rehearsal.
- The candidate can only be promoted if it passes local baseline-eval gates and then beats the current public/rank baseline.

## Recommended Candidate

`V201A`:

- Init: exact V194 `submission.zip`.
- Required zip SHA256: `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- Trainable modules: `in_proj,out_proj,q_proj,k_proj,v_proj,o_proj`.
- Steps: `5`.
- LR: `3e-7 -> 1e-7`.
- Sampling: weighted replacement toward weak subcategories:
  - `bit_manipulation=2.5`
  - `cipher=2.0`
  - `cryptarithm_deduce=3.0`
  - `cryptarithm_guess=2.0`
  - `equation_numeric_deduce=3.0`
  - `equation_numeric_guess=2.0`
- Gates:
  - exact V194 SHA before training;
  - baseline eval before training;
  - final eval loss must be `<= baseline_eval_loss`;
  - Kaggle zip must be root-only adapter layout;
  - no notebook auto-submit.

## Probability Estimate

The panel estimates the probability of public `>=0.87` as low-to-moderate, roughly `20-35%`.

That estimate is important: this is not a guaranteed improvement. It is the smallest high-rigor attempt that has a plausible path to gain while keeping regression blocked by promotion policy.

## Promotion Policy

- Public score `< 0.86`: discard.
- Public score `= 0.86`: quarantine unless rank/selection improves over V194 rank 19.
- Public score `> 0.86`: promote as the new baseline.

No local gate can guarantee the hidden/public Kaggle score. The notebook instead prevents blind regression by refusing to package or promote locally regressed candidates and by requiring explicit submit authorization.

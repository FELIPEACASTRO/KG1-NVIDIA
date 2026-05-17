# V536 H200 Attempt 1 Failure And Fix

## Failed Job

- Job: `felipesp1983/6a0930223308d79117b9181a`
- URL: `https://huggingface.co/jobs/felipesp1983/6a0930223308d79117b9181a`
- Result: failed before training steps started.

## Root Cause

The V536 dataset passed the local V286 tokenization gate with:

- `max_length=8192`
- train `token_max=1123`
- validation `token_max=1123`
- `prompt_truncation_rate=0.0`

The H200 launcher inherited `MAX_LENGTH=1024` from the base V493 command script.
At runtime, `hf_job_train_v90.py` correctly stopped with:

- `prompt_truncated=78/1026`
- `prompt_truncation_rate=7.6023%`
- required max prompt truncation rate: `0.0%`

This was a launch-parameter gap, not an adapter, dataset, or metric failure.

## Fix

- Set V536 runtime `MAX_LENGTH=2048`.
- Export `KG1_EXPECTED_MAX_LENGTH=2048` so remote preflight checks the value.
- Extended `scripts/kg1_pre_paid_job_integration_gate.py` with:
  - `--expected-max-length`
  - `--tokenization-manifest-json`
  - fail-closed check that `token_max <= runtime MAX_LENGTH`.

## Current Decision

The first attempt is blocked for training/eval/package/submit. A retry is allowed
only after commit/push of this fix and successful local launcher debug.

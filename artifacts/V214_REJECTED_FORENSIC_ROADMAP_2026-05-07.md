# V214 Rejected - Forensic Roadmap - 2026-05-07

## Decision

V214 is rejected and must not be submitted or used as a continuation base.

Weak eval result from the completed Colab run:

- V214 weak: `137/315 = 0.434921`
- Protected V194 weak baseline: `190/315`
- Weak delta: `-53`
- V214 truncation: `55/315 = 0.174603`
- Gate: full eval allowed only if weak `>=191/315` and truncation `<=3`
- Result: `weak_gate_pass_for_full = False`

Even with strong families staying perfect at `632/632`, the maximum possible
full score would be `137 + 632 = 769/947`, below both V194 `822/947` and the
strict gate `828/947`.

## Immediate Actions

1. Do not run full eval for approval.
2. Do not submit V214.
3. Do not use this adapter as the base for V215 or any later training:
   `/content/drive/MyDrive/KG1_NVIDIA_V214/output_v214_micro_replay/train_v214_v194_cont_lr3e7_s1/final_adapter`
4. Preserve the V214 weak artifacts:
   - `v214_micro_weak_predictions.csv`
   - `v214_micro_weak_raw_predictions_pre_score.csv`
   - `v214_micro_weak_per_task.csv`
   - `v214_micro_weak_eval_report.json`

## New Forensic Notebook

Colab URL after push:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_WEAK_FAILURE_FORENSIC_COLAB.ipynb`

Purpose:

- CPU-only forensic analysis of V214 weak failure.
- No training.
- No submit.
- No LLM calls.
- Reads the completed weak-eval CSV/JSON from Google Drive.

Expected outputs in Drive:

- `v214_micro_weak_rejected_forensic_rows.csv`
- `v214_micro_weak_rejected_forensic_per_type.csv`
- `v214_micro_weak_rejected_forensic_failure_buckets.csv`
- `v214_micro_weak_rejected_forensic_top_truncated.csv`
- `v214_micro_weak_rejected_forensic_summary.json`
- `v214_micro_weak_rejected_forensic_report.md`
- `v214_weak_failure_forensic_manifest.json`

## Why This Forensic Comes Before Any New GPU

The V194 boxed rewrite probe already showed parser/format recovery is not the
main bottleneck:

- Safe extractor recovery on weak errors: `0`
- Semantic tail recovery on weak errors: `0`
- Weak error buckets: `ALGEBRA_MANIP=100`, `ARITHM_BOUNDARY=24`, `LOOP_TRUNC=1`
- Decision: `reasoning_first_solvers_dataset`

But V214 showed a separate failure mode:

- continuation training can push the model into verbose/truncated outputs;
- the failure is large enough to invalidate the training recipe itself;
- V215 uses the same continuation pattern and is therefore blocked until this
  failure is diagnosed.

## Gate For Resuming GPU Work

GPU work remains blocked until the forensic report confirms at least:

1. Which task families caused the 55 truncations.
2. Whether failures are mostly `TRUNCATED`, `LOOP_OR_VERBOSITY`,
   `FORMAT_NO_BOXED`, or `WRONG_NON_TRUNCATED`.
3. Whether the train data shape caused answer-length inflation.
4. Whether any future experiment can be made strictly smaller than V214.

## Current Recommended Roadmap

### Step A - Run forensic notebook

Run:

`KG1_V214_WEAK_FAILURE_FORENSIC_COLAB.ipynb`

Stop condition:

- Report decision must be `REJECT_V214_NO_FULL_EVAL`.
- If it says anything else, stop and manually inspect gates.

### Step B - Review forensic outputs

Required checks:

- Per-type truncation counts.
- Failure bucket shares.
- Top 20 longest/truncated rows.
- Whether `bit_manipulation` or non-weak families drove the verbosity.

### Step C - Decide next experiment

Allowed only after Step B:

- If truncation is concentrated in one family: design a no-train prompt/output
  length mitigation probe first.
- If truncation is broad: abandon continuation training recipe and redesign the
  train target style.
- If V214 errors are mostly wrong but non-truncated: do not do more template
  training; return to solver-verified data only.
- If bit remains the only high-confidence opportunity: use V215 dataset only
  after a fresh dry-run proves the new recipe cannot reproduce V214 truncation.

### Step D - V215 remains blocked

V215 bit-focused data is prepared, but not authorized for GPU yet.

Reason:

- V215 depends on the same V194 continuation pathway.
- V214 proved that pathway can catastrophically regress weak/truncation.
- Running V215 before diagnosing V214 would risk repeating the same failure.

No Kaggle submit is authorized by this roadmap.

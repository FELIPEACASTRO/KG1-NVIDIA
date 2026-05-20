# KG1 V713 V712 Import Failure Consult

Updated: 2026-05-20 23:17 UTC

## Inputs

- V712 job:
  `https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06`
- Failure summary:
  `artifacts/openrouter/v713_v712_import_failure_consult/v712_failure_summary.json`
- Prompt:
  `artifacts/openrouter/v713_v712_import_failure_consult/KG1_POST_TRAIN_OPENROUTER_PROMPT.md`
- Responses:
  `artifacts/openrouter/v713_v712_import_failure_consult/openrouter_responses.md`
- Hugging Face metadata track:
  `artifacts/openrouter/v713_v712_import_failure_consult/huggingface_metadata_track.json`

## Panel Result

OpenRouter returned `5/5` usable responses. Hugging Face metadata returned
`6/6` model/provider records.

Consensus:

- V712 does not have weak ACC evidence. The job failed in weak eval, not in
  training.
- The next action should be weak-eval-only reuse of uploaded V712 checkpoints,
  not another paid training run.
- Evaluate `checkpoint-10` first because it had the best validation loss:
  baseline `2.6376`, checkpoint-10 `2.6363`, checkpoint-20/final `2.6373`.
- The import fix must be pushed/deployed before any remote eval:
  `scripts/__init__.py`, single evaluator compile, and import preflight.
- Keep A100-only and keep H200 forbidden.
- Keep label-free extraction, `verify_answer`, protected-row guard, zero
  truncation, boxed-rate, no-box fallback, and thresholds unchanged.

## Accepted

1. **Push/import fix before remote work.**
   The local gates now prove the launcher catches the exact failure mode:
   `scripts_package_gate`, `weak_eval_import_gate_ok`, static safety, pre-paid
   integration, and both self-tests passed.

2. **No retraining now.**
   V712 uploaded usable checkpoints. A weak-eval-only job is cheaper and is the
   missing artifact. Paying for another 20-step train before measuring ACC would
   be guesswork.

3. **Checkpoint priority.**
   `checkpoint-10` is first because it is the best-loss checkpoint. Evaluate
   `checkpoint-20` only if checkpoint-10 leaves a plausible path or if the
   comparison is needed to understand loss/ACC divergence.

4. **If weak eval fails score/backfire, do not continue GPU blindly.**
   Use the row-level weak manifest to decide whether the next route is
   decoding, protected-row replay, or a stronger training objective.

## Rejected Or Deferred

- **Changing `KG1_MAX_TOKENS` to `1024` for promotion.**
  Rejected for the next promotional weak eval because the active official-like
  contract is `7680`, and previous bounded decoding at `512` collapsed ACC.
  A `1024` run can only be diagnostic-only after the official-like checkpoint-10
  weak eval exists.

- **Removing `KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=512`.**
  Rejected. This is an anti-runaway promotion guard; deleting it would allow
  false gains and contradict current roadmap rules.

- **Increasing max steps/LR or adding `o_proj` now.**
  Deferred. These are future training experiments only after the missing weak
  ACC artifact proves V712 checkpoint behavior.

- **Treating broad `adapter_config.target_modules` as root cause.**
  Rejected for V712/V708. V711 and V712 trainability reports show effective
  trainable modules were `q_proj,v_proj`, with `lm_head` absent and MoE target
  parameters frozen.

## Local Decision

Proceed only with a cheap weak-eval-only A100 job after the import fix is on the
remote GitHub branch used by HF Jobs.

Required candidate:

- Adapter repo: `felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal`
- Adapter subfolder: `checkpoint-10`
- Candidate name: `v712_checkpoint_10`
- Decode: official-like, `KG1_DISABLE_THINKING=0`, `KG1_MAX_TOKENS=7680`,
  `KG1_MAX_MODEL_LEN=8192`, `KG1_MAX_NUM_SEQS=64`
- Gates: total `>=196`, bit `>=136`, equation `>=60`, truncation `0`,
  boxed rate `1.0`, no-box fallback `0`, protected backfire `0`

Blocked:

- new train;
- H200;
- package;
- full eval;
- Kaggle submit;
- threshold weakening.

# KG1 V494 Loss/ACC Sync Audit

Generated: 2026-05-16

## Scope

This audit checks whether the components that produce training loss, eval loss,
weak accuracy, family counts, and promotion gates are internally synchronized.
It also records the independent review requested by the user.

## Skills Used

- `senior-ml-engineer`: production gate, FinOps, and promotion criteria.
- `senior-data-scientist`: metric validity and proxy-vs-primary metric check.
- `senior-data-engineer`: data contracts, hashes, row counts, and family schema.
- `pytorch-lightning`: training-loop sanity checklist, even though the job uses
  raw PyTorch instead of Lightning.
- `skill-tester`: self-test/gate coverage mindset.

## Findings

| Area | Status | Evidence | Decision |
|---|---|---|---|
| Train loss | Correct as teacher-forced masked CE | `scripts/hf_job_train_v90.py` masks prompt tokens, shifts logits/labels, and fails if supervised tokens would be truncated | Use only as numerical health signal |
| Eval loss | Internally correct but not independent ACC proxy | Same masked CE over validation traces; V493 moved `1.9233 -> 1.9152`, too small to imply generation gain | Never promote from eval_loss alone |
| Weak ACC | Correct strict scoring path | `evaluate_lora_adapters_batch.py` merges by `id` and scores with `verify_answer` | Promotion remains weak/full ACC only |
| Weak data contract | Correct | `hf_job_weak_eval_v245.py` validates SHA, 315 rows, 160 bit, 155 equation, duplicate IDs, empty answers, and shared row contract | Keep fail-closed |
| Family counts | Correct | weak eval canonicalizes `bit_manipulation` and `equation_transform`; mismatch with prompt classifier fails | No action |
| Expected-aware extraction | Restricted but must be audited on marginal gains | Current code only disambiguates the last boxed payload and verifies it strictly | Any +1/+2 gain needs raw/simple/expected-aware diff |
| `PRETOKENIZED_VAL_COPY_ONLY` | Dangerous if enabled | It can make validation a copy of training rows, invalidating eval_loss independence | Static gate now blocks it in promotional jobs |
| Weak eval controls | Must be explicit | V245 defaults are short diagnostic settings; promotional eval must override them | Static gate now requires long-context controls unless diagnostic-only |
| V493 target pieces | Implemented in logs | `target_parameters_trainability_mode=trainable`; `up_proj/down_proj` trainable; `lm_head` frozen; `ANSWER_SPAN_LOSS_WEIGHT=1.0` | Proceed to checkpoint-2 weak eval |

## Critical Clarification

Loss and ACC are synchronized at the data-contract level, not mathematically
equivalent:

- loss = teacher-forced token-level cross entropy over supervised assistant
  tokens in JSONL train/validation traces;
- ACC = autoregressive generation through vLLM, answer extraction, exact
  verifier, and weak/full row contracts.

Therefore a lower loss can still leave ACC unchanged or worse if the generation
policy, prompt rendering, final-answer formatting, or family-specific rule
selection does not improve. The correct gate is still:

`total > 192`, `equation_transform > 56`, `bit_manipulation >= 136`,
`truncated = 0`.

## New Gate Changes

`scripts/kg1_static_safety_gate.py` now blocks:

1. `PRETOKENIZED_VAL_COPY_ONLY=1` in promotional HF jobs/notebooks.
2. Promotional `hf_job_weak_eval_v245.py` launchers that omit:
   - `KG1_DISABLE_THINKING=0`
   - `KG1_NO_PROMPT_SUFFIX=0`
   - `KG1_MAX_TOKENS=7680`
   - `KG1_MAX_MODEL_LEN=8192`
   - `KG1_MAX_NUM_SEQS=64`

Diagnostic sweeps can opt out only with `KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=1`.

## V493 Runtime Observation

The H200 smoke completed without runtime error:

- baseline eval loss: `1.9233`
- final eval loss: `1.9152`
- checkpoint uploaded: `checkpoint-2`
- adapter repo:
  `felipesp1983/kg1-nemotron-lora-v493-nemo-h200-moe-trainable-no-lmhead-v290ckpt6`

This is not enough to claim improvement. The only valid next action is the V494
weak eval on `checkpoint-2`.

# V478 Plateau Component Audit

Generated: 2026-05-16

## Executive finding

The plateau is not currently explained by an ACC verifier bug or by corrupt
V475 data. The strongest confirmed gap is objective mismatch in the training
piece: V475 physically contains a bit guardrail, but the V476 weighted sampler
made bit almost absent from the effective loss.

This explains the observed V477 weak result:

| Candidate | Total weak | equation_transform | bit_manipulation | truncated | Decision |
|---|---:|---:|---:|---:|---|
| Baseline V290/V291 checkpoint-6 | 192/315 | 56/155 | 136/160 | 0 | reference |
| V476 checkpoint-2 | 192/315 | 57/155 | 135/160 | 0 | fail: equation +1 paid by bit -1 |
| V476 checkpoint-4 | 191/315 | 57/155 | 134/160 | 1 | fail: bit/truncation regressed |

## Audited pieces

| Piece | Finding | Status |
|---|---|---|
| Weak ACC evaluator | Uses weak row contract, expected family counts, adapter config checks, and `verify_answer` path. No evidence found that the screenshot value is a metric bug. | OK |
| V475 dataset integrity | Train `1312`, val `328`, no duplicate ids/prompts, no weak/full reference overlap. | OK |
| V286 tokenization gate | Passed with prompt truncation `0`, offset masks complete, max token length `331`. | OK |
| Loss function | `masked_cross_entropy_loss` is valid masked CE over assistant completion tokens. It optimizes synthetic validation CE, not weak ACC. | OK but not promotion metric |
| Eval loss sampling | `select_eval_sample` balances validation by category and is not the weak gate. Loss can move without improving weak ACC. | Expected limitation |
| V324 CPU target signal | Four accepted weak equation candidates project `equation 56 -> 60`, weak `192 -> 196`, but those rows are evidence only, not training rows. | Real CPU signal |
| V325 synthetic target | Five synthetic numeric rule classes are generated uniformly; this is broader than the four accepted V324 rows. | Transfer-risk |
| V476 trainable adapter slice | Updates `q_proj,k_proj,v_proj,o_proj,lm_head`; MLP expert LoRA tensors remain active but frozen. | Conservative, may limit equation transfer |
| V476 weighted sampler | Fails objective alignment: bit effective share is only `0.9492%` despite physical bit share `39.0244%`. | Root gap |

## Objective alignment numbers

V476 launcher weights:

- `SOURCE_WEIGHTS`: `v475_v325_equation_no_loss_distill=8.00`, `v475_v217_bit_replay_guardrail=1.25`
- `SUBCATEGORY_WEIGHTS`: equation rules `12.00`, bit replay `1.15`

Effective train objective:

| Family | Physical rows | Physical share | Effective weight | Effective share |
|---|---:|---:|---:|---:|
| equation_transform | 800 | 60.9756% | 76,800.0 | 99.0508% |
| bit_manipulation | 512 | 39.0244% | 736.0 | 0.9492% |

The bit guardrail was present in the file but effectively missing from the
loss. With `weighted_replacement`, the model saw equation-style examples almost
all the time, which is consistent with bit dropping from `136` to `135` and then
`134`.

## New guardrail

Added `scripts/audit_v478_training_objective_alignment.py`.

The gate computes physical and effective family/source/subcategory shares from
the exact JSONL plus launcher weights. With default safety thresholds:

- `min_bit_effective_share >= 0.20`
- `max_equation_effective_share <= 0.80`
- `max_any_family_effective_share <= 0.80`

V476 fails the gate:

- `bit_effective_share_below_floor`: `0.009492 < 0.200000`
- `equation_effective_share_above_ceiling`: `0.990508 > 0.800000`
- `one_family_dominates_effective_objective`: `0.990508 > 0.800000`

JSON audit:

- `artifacts/v478_plateau_component_audit/v478_training_objective_alignment_gate_v476.json`

Static safety was also hardened: future weighted HF launchers/notebooks that
mix `bit_manipulation` and `equation_transform` must reference this objective
alignment gate before GPU.

## Operational decision

Do not relaunch V476 with more epochs, more steps, H200, or the same weights.
That would likely continue the same trade: small equation movement, bit
regression, and no submit-safe total gain.

Next paid GPU route must pass objective alignment before launch. For an
adapter-only attempt that must preserve bit, the bit effective share must be a
real training objective, not a token presence check.

## Next experiment shape

Before any GPU spend:

1. Build a V478/V479 CPU-only candidate dataset using only rule classes directly
   supported by V324 accepted rows, or explicitly label extra classes as
   exploratory.
2. Use family-balanced or capped weighted sampling. Do not let equation exceed
   `80%` effective share while bit is a hard guardrail.
3. Run `scripts/audit_v478_training_objective_alignment.py` and V286
   tokenization gate.
4. Launch GPU only if:
   - objective alignment passes;
   - no prompt/id overlap appears;
   - weak-eval kill switch is `total>192`, `equation>56`, `bit>=136`,
     `truncated=0`.

Loss remains diagnostic only. Submit decisions remain ACC-first.

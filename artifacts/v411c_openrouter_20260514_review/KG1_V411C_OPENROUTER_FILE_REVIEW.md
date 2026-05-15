# KG1 V411C OpenRouter 2026-05-14 File Review

Source file: `C:\Users\davis\Downloads\OpenRouter Chat Thu May 14 2026 (1).json`

## Scope

This review audited the OpenRouter export for concrete changes that can improve `bit_manipulation` or `equation_transform` under the current KG1 gates.

Inventory:

- Assistant outputs reviewed: `9`.
- Relevant technical URLs extracted: `252`.
- Clean URLs requested with lightweight HEAD/streamed GET: `221`.
- Reachable URLs: `198`.
- Broken/blocked/malformed URLs: `23`.

The URL access audit is saved in `url_access_audit.json`. It is an availability audit only; it does not mean every linked method is accepted.

## Current Measured Baseline

| Candidate state | Weak total | equation_transform | bit_manipulation | Submit-safe |
|---|---:|---:|---:|---|
| Best adapter-only V291/V290 checkpoint-6 | `192/315` | `56/155` | `136/160` | yes |
| V409 solver/verifier projection | `202/315` | `63/155` | `139/160` | no |

No new adapter-only gain was measured from this file. The file changes the next CPU gate design, not the submit status.

## Accepted Findings

### 1. Equation: FlashFill/VSA Ranking Must Be Added to V412

Accepted into roadmap.

Useful source:

- `https://people.csail.mit.edu/rishabh/papers/cacm12.pdf`

Why it matters:

- The file repeatedly converges on Programming by Example with a small DSL, version-space representation, intersection across examples, and ranking.
- This is directly aligned with the current `equation_transform` bottleneck: many candidate programs can fit the examples, so the acceptance rule needs to rank and reject ambiguous programs, not just enumerate more operators.

Implementation change for V412:

- Build a VSA/DAG per row for `equation_transform`.
- Intersect candidates across all provided examples.
- Rank by:
  - lower AST depth;
  - fewer nodes;
  - fewer arbitrary literals/constants;
  - penalty for undefined behavior;
  - penalty for output format outside observed examples;
  - preference for reused substrings/digits over memorized literals.
- Add leave-one-example-out checks where enough examples exist.
- Add synthetic stability probes: leading zero, sign flip, digit length change, delimiter swap, and `+1/-1` perturbation.
- Accept only if the top candidate is unique or all top candidates predict the same target answer.

Acceptance threshold:

- `100%` consistency on prompt examples.
- No undefined output on probes.
- Stable target prediction across probes or abstain.
- No loss against baseline weak rows.

### 2. Bit: Manthan-Style SAT Repair Is Useful, But Only After LUT

Accepted as P2/V412B, not as the first V412 task.

Useful source:

- `https://arxiv.org/abs/2005.06922`

Why it matters:

- Manthan supports data-driven Boolean synthesis with proof-guided refinement.
- This matches the bit solver direction after LUT k=2/k=3: when a per-bit Boolean candidate nearly fits, a SAT/MaxSAT-style repair can identify a small correcting predicate.

Critical limitation:

- In this challenge we do not know the hidden true function for all 256 byte inputs. We only know the prompt examples and the target row answer during local weak audits.
- Therefore, claims like "verify the true target on all 256 inputs" are not valid for test-time inference unless the underlying rule is recovered from examples.

Implementation change:

- Keep V412 focused on exact example verification plus abstention.
- Add V412B optional repair only for rows where LUT k=2/k=3 has near-fit and multiple candidates disagree.
- Do not accept repaired bit programs unless they fit every prompt example and the final predicted output is unique/consensual.

### 3. E-Graphs / Equality Saturation Are Search Optimizers, Not Acceptance Criteria

Accepted only as implementation optimization.

Useful sources:

- `https://egraphs-good.github.io/`
- `https://cap.ecn.purdue.edu/dryadsynth`
- `https://arxiv.org/abs/2304.10768`

Why it matters:

- E-graphs, DryadSynth-style enhanced enumeration, and abstract-interpretation pruning can reduce search cost.
- They do not by themselves prove that an answer is submit-safe.

Implementation change:

- Use these only if V412 enumeration is too slow.
- Acceptance remains: exact verifier, uniqueness/consensus, no-loss weak audit, bit guardrail, truncation guardrail.

## Rejected Or Deferred Findings

| Finding | Decision | Reason |
|---|---|---|
| Train more epochs / lower loss | rejected | Already failed repeatedly; loss moved without ACC gain. |
| Broad GRPO/RL from solver/verifier | deferred | Expensive and no proof it transfers to adapter-only. |
| ChopChop / constrained decoding | deferred | Potentially useful only if runtime rules allow custom inference constraints; not adapter-only today. |
| Teacher-only or verifier postprocessor gains | rejected for submit | V409/V405 are diagnostic, not current submit-safe package behavior. |
| Guaranteed `+15/+20` bit gain from literature | rejected | Unsupported without local weak/full gate measurement. |
| "All 256 input verification" as proof of hidden bit rule | rejected | Hidden target function is unknown; only prompt examples are available at inference. |

## Roadmap Impact

V411C does not authorize a submit, full eval, or HF/Kaggle GPU job.

It updates the next implementation:

1. V412 equation must include VSA/ranking/probe stability, not just a larger DSL.
2. V412 bit keeps LUT k=2/k=3 and adds optional Manthan-style repair only after the base LUT gate.
3. Any new GPU job remains blocked until CPU gate shows a new no-loss signal beyond V409 or a direct adapter/package transfer signal.


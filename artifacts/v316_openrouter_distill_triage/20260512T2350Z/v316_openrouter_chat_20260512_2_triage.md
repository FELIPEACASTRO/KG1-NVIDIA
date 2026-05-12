# V316 OpenRouter Distillation Triage - Updated 2026-05-12

Source: `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026 (2).json`

Local extraction: `artifacts/v316_openrouter_distill_triage/20260512T2350Z/`

Scope: evaluate multi-model distillation advice for turning verified V302/V306/V312 postprocessor/verifier gains into raw LoRA behavior for `bit_manipulation` and `equation_transform`.

## Evidence Baseline

- Current practical adapter baseline under discussion: `823/947`, `bit_manipulation=135/160`, `equation_transform=56/155`.
- Verified postprocessor/verifier ceiling under discussion: `838/947`, `bit_manipulation=146/160`, `equation_transform=60/155`.
- Target delta: `+11` bit rows and `+4` equation rows, with zero or near-zero family regression.
- Failed/insufficient attempts already observed:
  - V313 SFT verifier synthetic: no equation gain, bit regression.
  - V315 preference on V312: checkpoints stayed at `191/315`, `equation=56`; late checkpoint regressed bit to `134`.
  - V308/V304 axis with q/k/v/o/lm_head did not promote.

## Accepted Findings

1. Signal dilution is still the most likely failure mode.
   - The target equation signal is only 4 rows out of 155 local equation rows and 947 full rows.
   - Generic SFT/preference can show low loss while never moving those four decisions.
   - Future data should keep the 4 equation gain rules and 11 bit gain rules highly visible, not buried in broad synthetic corpora.

2. Final-answer token starvation is a concrete implementation risk.
   - The target changes are often only a few output tokens: `55 -> -55`, `-92 -> 92`, `03 -> 30`, `35 -> 134`, and binary strings.
   - Long traces can dilute the gradient on the final answer.
   - V317 should use short structured traces plus answer-span/token weighting, or answer-aligned completions where the answer receives materially more loss weight.

3. Hard negatives must be the frozen model's actual wrong outputs.
   - Random wrong answers are too easy and may not push the model off its current failure mode.
   - DPO/preference rows should contrast `chosen = verified correct raw output` against `rejected = exact frozen-adapter wrong output` for each target row or verified variant.

4. Bit anti-regression must be explicit and large.
   - Several responses independently called out the same practical guardrail: include all 135 currently-correct bit rows as keepers.
   - A stronger future design should cover all 160 bit rows: 135 correct keepers, 11 gain rows oversampled, and 14 still-wrong rows retained at low weight to preserve distribution.
   - A simple, low-risk fallback to KL regularization is hard batch mixing: keep at least 30-35% bit rows in every batch.

5. Fast probes should be mandatory before expensive weak/full eval.
   - Every checkpoint should be checked against:
     - 4 exact equation gain IDs;
     - 11 exact bit gain IDs;
     - bit keeper set;
     - equation keeper set;
     - truncation.
   - If two or three consecutive checkpoints remain `equation=56` and `bit<=135`, stop instead of extending the run.

6. V316 remains a valid current experiment, but V317 needs a different trainer/data objective if V316 fails.
   - V316 is testing `up_proj/down_proj` LoRA capacity, which differs from earlier q/k/v/o/lm_head attempts.
   - If V316 stays flat, the next change should not be "more of the same"; it should be answer-span weighted SFT plus hard-negative preference/contrastive rows and full bit keeper replay.

## Rejected Or Deferred Findings

1. High LR recipes such as `1e-4`, `1.5e-4`, `3e-5`, `5e-5`, or `8e-5` are rejected for the current PEFT lineage.
   - They are not backed by our local adapter behavior.
   - Existing successful lineage uses extremely low LR; high LR is likely to destroy the 0.86+ adapter behavior.

2. Exact target rows as the only SFT targets are not enough.
   - They may memorize local weak/full probes without learning a deployable rule.
   - Use exact rows for probes and hard-negative contrast, but rely on verified programmatic variants for SFT signal.

3. Sequential equation-only training is too risky.
   - It conflicts with observed bit regression in V313/V315.
   - Joint/interleaved training with bit keeper replay is safer.

4. Dual/separate LoRA routers are deferred.
   - The challenge path requires a single raw adapter behavior for submission.
   - Separate LoRA routing adds merge/compatibility risk and should be considered only after a simpler single-adapter V317 fails.

5. KL regularization is useful but not first implementation priority.
   - It may be effective, but it adds implementation complexity and GPU cost.
   - First low-cost version: enforce bit replay percentage and hard reject checkpoints that drop bit keepers.

## Concrete Impact On Roadmap

If V316 promotes, do not change course until full eval confirms no family regression.

If V316 does not promote, launch V317 with:

- compact trace format ending in a clear `Answer:` span;
- answer-span loss multiplier around `3x` initially, not speculative `15x` until tested;
- per-example weights:
  - equation gain variants and exact hard negatives highest;
  - bit gain variants high;
  - bit keepers required and unskippable;
  - global keepers moderate;
- DPO/contrastive pairs only where rejected output is the frozen adapter's observed wrong answer;
- batch mix with at least `30-35%` bit rows;
- checkpoint gate that rejects `bit<135`, promotes only with `bit>=136`, and requires `equation>56` or `total>=193`.

## Bottom Line

The updated OpenRouter file contains no new measured ACC, but it does add a stronger implementation direction: do not chase lower eval_loss. The next useful experiment after V316 must make the final answer tokens louder, make negatives match the model's real mistakes, and make bit keeper replay mandatory.

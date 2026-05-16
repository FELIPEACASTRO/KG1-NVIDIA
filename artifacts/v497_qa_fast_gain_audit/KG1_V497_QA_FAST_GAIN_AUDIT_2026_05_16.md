# KG1 V497 QA Fast-Gain Audit

Date: 2026-05-16

Scope: loss, eval loss, weak ACC, strict scoring, V475/V495/V496 data flow,
H200 performance, FinOps gates, and the fastest submit-safe route.

## Executive Decision

The current plateau is not caused by an obvious ACC calculation bug. The metric
path is correct, and the V496 result is a real adapter regression:

- V290 checkpoint-6: `192/315`, equation `56/155`, bit `136/160`, trunc `0`.
- V496 checkpoint-2: `191/315`, equation `57/155`, bit `134/160`, trunc `1`.

The fastest responsible route is now CPU-first equation teacher refinement plus
bit preservation checks. More H200 SFT without a new CPU signal is blocked.

## QA Findings

| Area | Finding | Impact | Required Action |
|---|---|---|---|
| ACC metric | `audit_v449_acc_metric_integrity.py` passes; `verify_answer` strict path is used for promotion | no metric inflation bug found | keep strict verifier; never use permissive bit scoring |
| Expected-aware extraction | Adds one known row `4bb8c6cd` in both baseline and V496 | not a new adapter gain | every marginal gain needs simple vs expected-aware diff |
| V475 dataset | correct train/val hashes, 1312/328 rows, no objective alignment findings | dataset is not the execution bug | do not repeat unchanged V475 SFT |
| V495 training | MoE params trainable, `lm_head` frozen, answer-span weight 1.0 | mechanism is technically correct | loss drop alone remains non-promotional |
| V496 weak eval | +1 equation, -2 bit, +1 truncation | not submit-safe | block full eval/package/submit |
| H200 speed | 1.5M completion tokens dominate runtime | hardware is not the main bottleneck | reduce paid eval frequency; keep official-like settings only for finalists |
| FinOps | V496 took under 1 hour and uploaded diagnostics before gate failure | spend was controlled | no more H200 until CPU gate predicts real promotion |
| Regression pattern | repeated `equation=57` attempts lose bit or truncation | broad SFT is trading errors | pivot to row-family constrained teacher/guardrail |

## Why Loss Does Not Track ACC Here

Loss is teacher-forced masked cross entropy over the supervised assistant tokens
in synthetic traces. ACC is autoregressive generation plus answer extraction and
strict verifier over weak/full rows. These are synchronized by data contracts,
but they are not equivalent objectives.

Observed proof:

- V493 loss improved and weak ACC regressed to `190/315`.
- V495 loss improved slightly and weak ACC regressed to `191/315`.
- The only V496 gain is one equation row, while bit loses two rows.

## H200 Performance Notes

The H200 is doing the work; the eval is long because the model emits long
reasoning despite the strict one-line suffix.

Official-like weak eval must keep:

- `KG1_DISABLE_THINKING=0`
- `KG1_NO_PROMPT_SUFFIX=0`
- `KG1_MAX_TOKENS=7680`
- `KG1_MAX_MODEL_LEN=8192`
- `KG1_MAX_NUM_SEQS=64`

Changing these can speed up eval, but it changes the benchmark and must be
diagnostic-only unless re-baselined against V290 checkpoint-6.

## Fastest Submit-Safe Plan

1. Freeze V290 checkpoint-6 as the only submit-safe adapter.
2. Stop broad H200 SFT until CPU finds a new equation teacher signal.
3. Build CPU residual audit for the remaining equation misses:
   - symbolic punctuation transformations;
   - colon/quote/backslash escaping cases;
   - numeric operator variants already represented by V324/V475;
   - raw-output cases where simple extraction differs from expected-aware.
4. For every proposed rule, require:
   - no weak/full labels used for training;
   - no bit rows changed in CPU projection;
   - no non-binary bit output in any synthetic/weak-like probe;
   - projected weak target at least `equation>=60`, `bit>=136`, `trunc=0`.
5. Only then create one short adapter-transfer run with:
   - final-answer-only traces;
   - hard negatives from solver-generated wrong answers, not weak labels;
   - bit replay guardrail;
   - first-checkpoint kill-switch.

## Blocked Routes

- More epochs on V475 as-is.
- Repeating V390/V326 or V475 broad SFT.
- Increasing `ANSWER_SPAN_LOSS_WEIGHT` to chase loss.
- Re-enabling `lm_head` in the promotional smoke.
- Using weak IDs as train labels.
- Submitting V496 or any candidate with `bit<136` or `trunc>0`.

# KG1 V504 Crisis Root Cause Audit

Generated: 2026-05-16

## Decision

V501 is blocked. Do not weak-eval, full-eval, package, or submit it.

The active submit-safe baseline remains V290/V291 checkpoint-6:

| Metric | Submit-safe baseline |
|---|---:|
| Weak total | 192/315 |
| equation_transform | 56/155 |
| bit_manipulation | 136/160 |
| truncated | 0 |

## Exact Problems Found

1. **ACC promotion could be label-aware.**
   `extract_final_answer_for_expected()` was being used to create the scored
   `prediction` column. That helper uses the expected answer, so it is valid for
   labeled debug audits but not for submit-safe prediction. This could turn
   parser improvements into apparent ACC gains. Fixed by making scored
   predictions use `extract_final_answer(raw_output)` and writing expected-aware
   extraction only as `label_aware_debug_prediction`.

2. **Expected-aware extraction accepted unsafe prefixes.**
   A payload like `\boxed{30 wrong}` could match expected `30` because the helper
   accepted spaces and punctuation after the expected prefix. Fixed by accepting
   only a real closing `}` delimiter for expected-aware debug extraction.

3. **Weak promotion defaults were stale and diagnostic.**
   The weak gate defaulted to `equation>=57`, `total>=193`, `max_tokens=96`,
   `max_model_len=4096`, and thinking disabled. That is useful for cheap
   triage, but it is not the current submit-safe promotion contract. Fixed by
   defaulting promotional weak eval to official-like controls and
   `total>=196`, `equation>=60`, `bit>=136`, `trunc=0`; diagnostic mode must be
   explicit.

4. **Full eval best-candidate selection could fail incorrectly.**
   The full gate selected highest `correct` before considering truncation. A
   candidate with more correct rows but too many truncations could hide another
   candidate that passed both criteria. Fixed by ranking passing candidates
   first.

5. **V501 proved the mechanism still does not produce a safe gain.**
   V501 did activate answer-span weighting and trainable MoE target parameters:
   train answer-span weighted examples `1712`, tokens `15197`; MoE
   `gate_up/down` trainable tensors present; `lm_head` frozen. It still
   regressed final eval loss from `1.9919` to `1.9923`, so the FinOps
   kill-switch correctly blocked promotion.

6. **Answer-span weighting was active but not answer-dominant.**
   In V498, final answer text is only about `2.68%` of assistant characters.
   With `ANSWER_SPAN_LOSS_WEIGHT=4.0`, the answer span accounts for only about
   `36%` of weighted completion loss. This explains why loss can move while
   weak ACC stays flat or regresses. The next trainable route must be treated as
   a new objective experiment, not a promotion path.

7. **Pre-paid job integration gate was still preference-oriented.**
   It expected `chosen/rejected`, `SAVE_EVERY_STEPS=3`, `EVAL_EVERY_STEPS=3`,
   and the preference system prompt. V498/V501 is SFT `messages/answer`, so this
   gate was not aligned to the actual route. Fixed by adding SFT schema support
   and configurable first checkpoint/eval steps.

8. **Anti-leakage flags could be absent.**
   `hf_job_preflight_gate.py` counted missing
   `gate_rows_used_for_training`, `weak_gate_rows_used_for_training`, and
   `full_gate_rows_used_for_training`, but only failed when flags were present
   and non-false. Fixed by failing promotional dataset preflight when those
   flags are absent.

## What Is Not The Root Cause

| Area | Verdict |
|---|---|
| Dataset hashes for V498/V501 | Passed |
| Row counts and family/subcategory balance | Passed |
| Tokenization and offset masks | Passed |
| Truncation in training data | Passed |
| PEFT target parameter presence | Passed |
| MoE `gate_up/down` trainability in V501 | Passed |
| `lm_head` frozen in V501 | Passed |
| H200 memory/cost guard | Passed |

## Implemented Fixes

| File | Change |
|---|---|
| `src/competition_utils.py` | Expected-aware extraction made debug-only and delimiter-strict |
| `scripts/evaluate_lora_adapter.py` | Scored `prediction` is label-free; debug prediction separated |
| `scripts/evaluate_lora_adapters_batch.py` | Same label-free scoring fix for batch eval |
| `scripts/analyze_eval_predictions.py` | `official_correct` now reextracts from raw output label-free |
| `scripts/hf_job_weak_eval_v245.py` | Official-like defaults and stricter promotion thresholds |
| `scripts/hf_job_official_like_eval_gate_v284.py` | Best candidate chosen among rows that pass full gate first |
| `scripts/hf_job_full_eval_v276.py` | Same best-candidate full-gate fix |
| `scripts/hf_job_preflight_gate.py` | Missing anti-leakage flags now fail |
| `scripts/kg1_pre_paid_job_integration_gate.py` | Added SFT schema support and blocked V499/V501 adapters |
| `scripts/kg1_static_safety_gate.py` | New guards for label-aware scoring, stale weak thresholds, blocked adapters |

## Next Action

Do not launch another H200 job from V498/V501. The next work item is CPU-first:

1. Re-score previous weak/full outputs with label-free extraction only.
2. Rebuild the candidate inventory using the corrected metric.
3. If no submit-safe `>192/315` candidate remains, stop paid training and
   return to deterministic CPU teacher discovery.
4. Only open a new GPU job after a CPU gate shows a new label-free gain with
   `equation>=60`, `bit>=136`, and `trunc=0`.


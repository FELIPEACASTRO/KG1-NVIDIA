# V468 Crisis Mode Audit

## Finding

V465/V466 were trained/evaluated on the old V464 dataset, which contained silent contradictory supervision in `equation_transform` traces.

The failing pattern was:

- assistant text says a candidate is rejected;
- that candidate is equal to the gold answer under `verify_answer`;
- the same trace then ends with that same value in `\boxed{}`.

This is a silent training bug because `eval_loss` can still decrease while the model receives inconsistent semantic supervision.

## Impact

Old V464 contradiction count:

| Split | equation rows | rejected candidate equals answer |
|---|---:|---:|
| train | 46 | 24 |
| validation | 10 | 6 |

The affected rule-replay classes were primarily:

- `v274_guarded_numeric_colon_absdiff_restore_trailing_zero`;
- `v274_guarded_numeric_minus_signed_opposite_sign_guarded`.

V466 weak eval confirmed no deployable gain from the contaminated V465 adapter:

| Checkpoint | Total | equation_transform | bit_manipulation | truncated | Decision |
|---|---:|---:|---:|---:|---|
| checkpoint-4 | 189/315 | 56/155 | 133/160 | 1 | reject |
| checkpoint-8 | 192/315 | 56/155 | 136/160 | 1 | reject |

V466 was canceled before spending more on checkpoint-12/16/final.

## Fix

`scripts/build_v464_v463_numeric_multirule_dataset.py` now selects a rejected candidate that is actually different from the answer:

- for real hard negatives, prefer the real `adapter_prediction` only when it is wrong;
- for rule replay, prefer `simulated_wrong_prediction`;
- if no non-answer rejected candidate exists, fail the CPU dataset build.

The builder now records:

- `metadata.rejected_candidate`;
- `metadata.rejected_candidate_source`;
- rejected candidate source counts in the manifest.

`scripts/run_v286_generic_tokenization_gate.py` now blocks any dataset where:

- `metadata.rejected_candidate` verifies equal to `answer`;
- assistant text contains `candidate 'X' is rejected` and `X` verifies equal to `answer`.

## Fixed Dataset

V468 rebuild:

- train: `558` rows = `46` equation + `512` bit replay;
- validation: `138` rows = `10` equation + `128` bit replay;
- hard negatives in train: `22` across `3` rule classes;
- contradictory rejected candidates: `0`;
- train/validation prompt overlap: `0`;
- tokenization gate: passed with real Nemotron tokenizer.

Token gate result:

| Metric | Value |
|---|---:|
| train rows | 558 |
| validation rows | 138 |
| train token max | 327 |
| validation token max | 356 |
| prompt truncation rate | 0.0 |
| completion tokens dropped | 0 |
| fallback masks | 0 |
| offset masks | 696/696 |

## Decision

Old V464 and V465 are blocked for further training decisions.

Next allowed action is a new tiny HF smoke train from V468 only, with the same hard gate:

- weak total `>192/315`;
- `equation_transform >56/155`;
- `bit_manipulation >=136/160`;
- `truncated = 0`.

If checkpoint-4 misses any of these, cancel immediately by FinOps.

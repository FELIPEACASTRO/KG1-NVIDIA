# V471 Crisis Triple-Check Result

Generated: 2026-05-16

## Decision

V469/V470 is rejected. It is not submit-safe and must not receive more H200
time.

| Route | Total weak | equation_transform | bit_manipulation | truncated | Delta vs locked submit-safe |
|---|---:|---:|---:|---:|---:|
| Locked submit-safe baseline | `192/315` | `56/155` | `136/160` | `0` | `0` |
| V470 eval of V469 checkpoint-4 | `190/315` | `56/155` | `134/160` | `1` | `-2` |

Promotion gate result: blocked. V470 did not improve equation, lost two bit
rows, and reintroduced truncation.

## Bugs And Silent Failure Modes Found

| Issue | Risk | Fix |
|---|---|---|
| V464 contradictory hard negatives | Loss could fall while teaching a false rejection rule | V468 already rebuilds with `0` rejected-candidate contradictions; V286 now blocks this class |
| Evaluation CSV without `answer` | Could emit misleading `accuracy=0.0000` instead of failing | `evaluate_lora_adapter.py` and batch evaluator now hard-fail |
| vLLM output count mismatch | Predictions could silently misalign with prompts | single and batch evaluators now hard-fail if counts differ |
| Completion-token truncation in training | Loss/ACC could be optimized on amputated completions | `hf_job_train_v90.py` now fails if truncation drops supervised loss tokens |
| Symbolic boxed answer not metric-extractable | Dataset can look syntactically valid but fail KG1 extraction | V286 now verifies `extract_final_answer(...)` against `answer`; V447 builder rejects non-extractable traces |
| Weak eval promotion was diagnostic-only | Bad checkpoint could still look operationally successful | `hf_job_weak_eval_v245.py` now supports enforced promotion gate after diagnostics upload |
| Notebook release gate weak thresholds stale | Future notebooks could allow bit regression/truncation | release gate now requires `WEAK_BIT_MIN_FOR_FULL = 136` and `WEAK_MAX_TRUNC_FOR_FULL = 0` |

## Metric Integrity Check

`artifacts/v471_crisis_solution_audit/v470_metric_integrity/v470_metric_integrity_manifest.json`
confirms the V470 metric path is strict:

- strict correct: `190/315`;
- permissive numeric diagnostic would falsely count `205/315`;
- all `15` strict/permissive disagreements are `bit_manipulation`;
- therefore promotion must always use `src.competition_utils.verify_answer`.

## Parser Rescue Check

`artifacts/v471_crisis_solution_audit/v470_parse_audit/v470_parse_audit_manifest.json`
found no deterministic parser/extractor rescue:

- current V470 remains `190/315`;
- no strategy beats the locked submit-safe gate;
- no parser-only gain is available for this checkpoint.

## Valid Dataset State

V468 corrected dataset passed the CPU metric-extraction/tokenization gate:

- train rows: `558`;
- validation rows: `138`;
- contradiction gate: `0`;
- prompt truncation: `0`;
- completion tokens dropped: `0`;
- fallback masks: `0`.

This makes V468 a valid artifact for analysis, but V469/V470 proves this
particular SFT route does not transfer into weak ACC.

## Operational Rule

Do not run another GPU job from this route. The next route must be CPU-first:

1. mine new deterministic equation classes against the `99` weak equation
   misses;
2. require at least `+4` equation rows with `0` losses and `bit>=136`;
3. only then build a small verified dataset;
4. only then run an HF smoke with enforced gate:
   `total>=193`, `equation>=57`, `bit>=136`, `truncated=0`.

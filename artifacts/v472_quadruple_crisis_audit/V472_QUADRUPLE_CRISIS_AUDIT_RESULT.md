# V472 Quadruple Crisis Audit Result

Generated: 2026-05-16

## Scope

This audit rechecked the KG1 pieces that directly affect loss, eval, ACC,
promotion gates, data integrity, symbols, and FinOps.

## Findings

| Finding | Evidence | Action |
|---|---|---|
| Historical Colab builders still used stale weak gates | `WEAK_BIT_MIN_FOR_FULL = 133` and `WEAK_MAX_TRUNC_FOR_FULL = 3` in V217-V231/V244/V245 builders | Updated builders to `136` and `0`; added static safety rule to block stale values |
| Quarantined V464 dataset still exists as tracked evidence | V472 data scan found `24` train and `6` validation rows with rejected candidate equal to answer | Kept tracked evidence, but HF preflight now blocks any training identity containing `v464_v463_numeric_multirule_dataset` |
| V468 still inherited a full-reference exact prompt/answer seed | `63-19 -> -55` matched the full reference row `7688e06e`; source path traced back to V461/V463 | V461/V463/V464/V468 are blocked for training until rebuilt under a new clean dataset version |
| V447 trace dataset contains contradictory `hypothesis_formed` rows | `141` rows append the official final answer after an internal boxed answer that differs | V446 now excludes `hypothesis_formed`; V447 builder now keeps `rule_found` only |
| Eval postprocessor could run before truncation metadata existed | V274 checked `truncated`, but eval code set it only after postprocessing | Eval now sets `truncated`/`truncated_bool` before any postprocessor |
| CSV eval readers could coerce ids/answers | `pd.read_csv` without `dtype=str` risks losing leading zeroes or converting blanks | Eval/analyze/gate CSV readers now use string mode and reject empty critical fields |
| V300 bit full-byte postprocessor selected first compatible program | Ambiguous prompts could have multiple matching programs with different query predictions | V300 now applies only when all compatible programs produce one unique prediction |
| Prompt/raw-output JSONL are not training datasets | V452/V457/V458/V461/V462 have empty answers or no assistant by design | Not promoted to training; kept as diagnostic/probe artifacts only |
| Cache/debug residue existed | many `__pycache__` directories and obsolete untracked V445/V447/V449 local artifacts | Removed safe untracked/cache files only |

## Current Gate Policy

Any future GPU route must pass:

- strict metric path: `verify_answer`, not `answers_equivalent`;
- weak promotion: `total>=193`, `equation>=57`, `bit>=136`, `truncated=0`;
- no supervised-token truncation;
- no missing `answer` in eval solution CSV;
- vLLM output count equals prompt count;
- CSV ids/prompts/answers preserved as strings with no empty critical fields;
- assistant final answer extractable by `extract_final_answer` and correct by
  `verify_answer`;
- no quarantined V461/V463/V464/V468/V447 data.

## Files

- Data integrity manifest:
  `artifacts/v472_quadruple_crisis_audit/v472_repository_data_integrity_manifest.json`
- Static gate updated:
  `scripts/kg1_static_safety_gate.py`
- HF preflight gate updated:
  `scripts/hf_job_preflight_gate.py`
- Generic tokenization/leakage gate updated:
  `scripts/run_v286_generic_tokenization_gate.py`

## Decision

No new HF job should run from V447, V461, V463, V464, V468, V469, or V470
transfer routes. The only valid next path remains CPU-first mining of new
equation classes with zero-loss proof before GPU, using a new clean dataset
version and V286 with forbidden reference CSVs when available.

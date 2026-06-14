# KG1 Score Determinants Audit - 2026-06-13

## Decision

This audit found and fixed multiple score-determining silent gaps in the active KG1 path.

No new Kaggle score is claimed here. The validated baseline remains `823/947 = 0.869060`.
The current work improves the correctness of the training/evaluation/gating pipeline before GPU
training and submission.

## Macro Score Model

One full947 row is worth:

```text
1 / 947 = +0.001055966
```

Known baseline:

| family | baseline |
| --- | ---: |
| gravity_constant | 159/159 |
| unit_conversion | 159/159 |
| numeral_system | 157/157 |
| text_encryption | 157/157 |
| bit_manipulation | 135/160 |
| equation_transform | 56/155 |
| overall | 823/947 |

Implication:

- Protected families are already `632/632`; they have no public headroom and only regression risk.
- Useful public-full947 headroom is concentrated in `bit_manipulation` and `equation_transform`.
- `>=0.89` requires `843/947`, i.e. `+20` rows from baseline.
- `>=0.90` requires `853/947`, i.e. `+30` rows from baseline.

## Determinants Of Score

### 1. Official Metric Semantics

Score is determined by `extract_final_answer(raw_output)` followed by `verify(answer, extracted)`.

Validated behavior:

- Last non-empty `\boxed{...}` payload wins under the public extractor.
- If no boxed answer exists, fallback can use final-answer phrase, last number, or last non-empty line.
- Binary answers require exact binary string shape; leading zero loss is fatal.
- Numeric answers use `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`.
- String answers are case-insensitive after edge stripping; internal whitespace still matters.

Risk:

- Public fallback can create false confidence. Promotion must require real `raw_output`, one closed boxed answer, no malformed box, no truncation, and no regression.

Validation:

- `python scripts/kg1_official_metric_prescore_gate.py` passed.
- Official notebook SHA matched.
- Prescore parity: `22` cases, `0` failures.
- Local utility parity: `12` cases, `0` failures.

### 2. Official Inference Surface

Active score-facing config:

```text
model = nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
revision = cbd3fa9f933d55ef16a84236559f4ee2a0526848
max_lora_rank = 32
max_tokens = 7680
temperature = 0.0
top_p = 1.0
max_model_len = 8192
max_num_seqs = 64
gpu_memory_utilization = 0.85
enable_prefix_caching = true
enable_chunked_prefill = true
enable_thinking = true
```

Risk:

- Old defaults `3584/4096/temp=1.0/max_num_seqs=128` distort truncation, decoding, and score estimates.

Fixes applied:

- `scripts/kg1_official_metric_prescore.py`: visible defaults corrected to `7680/8192/temp=0.0/64`.
- `scripts/kg1_official_metric_prescore_gate.py`: now gates prescore visible defaults too.
- `scripts/evaluate_lora_adapter.py`: docstring and defensive fallbacks corrected.
- `scripts/evaluate_lora_adapters_batch.py`: defensive fallbacks corrected.
- `scripts/notebook_release_gate.py`: stale snippets changed from `4096/3584` to `8192/7680`.

Additional double-check fix:

- `scripts/evaluate_lora_adapter.py`: prompt rendering now appends the official suffix only when it is absent. This prevents V1241/V1243 CSVs, which already include the suffix, from being evaluated with duplicated `Please put your final answer...` instructions.

### 3. Raw Output Contract

Score-relevant generated rows must preserve:

- `raw_output` column from real generation.
- `finish_reason`.
- `completion_tokens`.
- final boxed payload.

Risk:

- A pre-extracted `prediction` can hide no-box fallback, truncation, malformed boxes, and multiple-box behavior.

Fixes applied:

- `scripts/solve_rate_gate.py`: prediction-only CSVs are now blocked by default.
- `--allow-prediction-only` exists only for explicit diagnostics, not promotion.
- Solve-rate self-test now verifies `prediction_only_blocked=true`.
- V1241 remains the strict promotion gate for bit/equation raw-output comparisons.

### 4. Truncation Accounting

Truncation is score-determining because many failures are not semantic; the model can fail simply by never reaching the box.

Strict blocker sources:

- `finish_reason in {length,max_tokens,max_output_tokens,token_limit,truncated}`
- explicit truncation flags
- `completion_tokens >= max_tokens`

Fix applied:

- `scripts/kg1_v1241_bit_equation_transfer_gate.py`: default `--max-tokens` now uses `OFFICIAL_INFERENCE_CONFIG["max_tokens"] = 7680`, not stale `2048`.

Validation:

- `python scripts/kg1_v1241_bit_equation_transfer_gate.py --self-test` passed.

### 5. Family Headroom And Regression Risk

Protected families:

- `gravity_constant`: no gain path; protect `159/159`.
- `unit_conversion`: no gain path; protect `159/159`.
- `numeral_system`: no gain path; protect `157/157`.
- `text_encryption`: no gain path; high symbolic regression risk; protect `157/157`.

Weak families:

- `bit_manipulation`: 25 misses; strongest validated projection signal.
- `equation_transform`: 99 misses; largest headroom, but high symbolic/boxed risk.

Promotion rule:

- A candidate that improves bit/equation but regresses protected rows is not a score-safe candidate.

### 6. LoRA Scope

Active specialist scope:

```text
down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj
```

Forbidden in V1243 specialist probes:

```text
out_proj,lm_head,embed_tokens
```

Risk found:

- V1243 preview previously let the trainer create LoRA modules using the trainer default, which included `lm_head` and `out_proj`, while the trainable filter wanted only 7 modules. That mismatch could confuse model-load reports and silently leave extra inactive LoRA tensors.

Fixes applied:

- `scripts/kg1_v1243_solver_to_lora_graft_builder.py`: `LORA_TARGET_MODULES` and `TRAINABLE_LORA_MODULES` are both locked to the same 7 modules.
- `scripts/kg1_v1243_graft_trainer_contract_gate.py`: gate now requires the exact 7-module target.
- `scripts/hf_job_train_v90.py`: default `DEFAULT_LORA_TARGET_MODULES` now uses the same 7-module safe target, so an operator who forgets the V1243 env no longer silently creates `lm_head/out_proj` LoRA tensors by default.

Validated dry-run:

- `bit`: parsed target modules are exactly the 7 safe modules.
- `equation`: parsed target modules are exactly the 7 safe modules.

### 7. Payload-First Loss

Score only sees the final extracted answer. Long reasoning is mostly risk.

Active V1243 trainer behavior:

- Completion-only loss.
- `BOXED_PAYLOAD_LOSS_WEIGHT=5.0`.
- `REQUIRE_BOXED_PAYLOAD_WEIGHT=1`.
- `REQUIRE_OFFSET_MASK=1`.
- `TOKENIZE_ONLY_DRY_RUN=1` for first validation.

Validated dry-runs:

| phase | rows | rows with boxed payload weight | boosted boxed tokens | truncation | fallback masks |
| --- | ---: | ---: | ---: | ---: | ---: |
| bit train | 724 | 724 | 8690 | 0 | 0 |
| equation train | 544 | 544 | 4422 | 0 | 0 |
| val170 | 170 | 170 | 1798 | 0 | 0 |

### 8. Row-Level Sampling Weights

Risk found:

- V1243 builder wrote `row_loss_weight`/`loss_weight`, but the trainer discarded those fields during tokenization.
- Before the fix, weighted replacement sampling used raw row counts:
  - bit specialist share: `0.745856`
  - equation specialist share: `0.661765`
- That under-trained the score headroom families relative to the intended plan.

Fixes applied:

- `scripts/hf_job_train_v90.py` now preserves `row_loss_weight` through tokenization.
- `example_sampling_weight()` now multiplies by `row_sampling_weight()`.
- `scripts/kg1_v1243_graft_trainer_contract_gate.py` now checks the source snippets and `weight_by_family`.

Validated after fix:

| phase | specialist share | protected replay share |
| --- | ---: | ---: |
| bit | 0.823357 | 0.176643 |
| equation | 0.772633 | 0.227367 |

### 9. Dataset Integrity

V1243 artifacts:

- `v1243_bit_specialist_train.jsonl`: 724 rows.
- `v1243_equation_specialist_train.jsonl`: 544 rows.
- `v1243_protected_replay_train.jsonl`: 184 rows.
- `v1243_micro_consolidation_train.jsonl`: 1084 rows.
- `v1243_val170.jsonl`: 170 rows.

Contract gate validation:

- `python scripts/kg1_v1243_graft_trainer_contract_gate.py` passed.
- `errors=0`.
- `train_val_prompt_overlap=0`.
- Dataset hashes and row counts match preview.
- All rows have top-level answers.
- All targets have exactly one closed boxed answer.
- All V1243 prompts must contain exactly one official prompt suffix and no control characters.
- `v1243_val170.jsonl` is now summarized as a true holdout: `source_role=v1243_holdout`, top-level sampling weight `0.0` by family.
- No GPU/package/submission/score claim is authorized by this artifact.

### 10. Promotion Gates

Current strict gates:

- `tiny`: 50 rows, requires bit gain, equation gain, total gain, no regressions.
- `val170`: 170 rows, same strict pattern.
- `full947_089`: requires `>=843/947`, `+20`, `+1 bit`, `+1 equation`, no regressions, no truncation, no format failure.
- `full947_090`: requires `>=853/947`, `+30`, `+1 bit`, `+1 equation`, no regressions, no truncation, no format failure.

No score claim is valid without real generated `raw_output` CSVs passing these gates.

Operational command alignment:

- V1243 generated commands now use the actual V1241 CLI flags: `--baseline-predictions` and `--candidate-predictions`.
- V1241 still accepts the old `compare --baseline-csv --candidate-csv` form as a compatibility alias, but the V1243 contract gate rejects new manifests that emit that obsolete command shape.

## Corrections Applied In This Audit

1. Corrected prescore visible defaults to the official current surface.
2. Added prescore-default validation to the official metric gate.
3. Corrected evaluator and batch evaluator stale generation fallbacks.
4. Corrected notebook release gate stale `4096/3584` checks.
5. Corrected V1241 default max-token truncation threshold to `7680`.
6. Blocked prediction-only solve-rate CSVs by default.
7. Added solve-rate self-test coverage for prediction-only blocking.
8. Locked V1243 LoRA target modules to the exact 7 safe modules.
9. Preserved V1243 row-level sampling weights through trainer tokenization.
10. Made trainer weighted replacement consume row-level weights.
11. Added V1243 contract checks for trainer row-weight snippets.
12. Added V1243 contract checks for `weight_by_family`.
13. Regenerated/validated V1241 and V1243 gate artifacts where needed.
14. Fixed prompt rendering so official suffix is not duplicated when CSV prompts already include it.
15. Added shared `ensure_prompt_suffix()` helper.
16. Added V1243 prompt-suffix count and control-character checks.
17. Corrected V1243 generated V1241 gate commands.
18. Added V1241 CLI compatibility aliases for old command artifacts.
19. Added V1243 manifest-command audit to block obsolete command shapes.
20. Changed trainer default LoRA modules to the 7-module safe target.
21. Corrected V1243 holdout summary so zero sampling weights remain zero.
22. Added V1243 contract checks for holdout role and zero holdout weights.
23. Added CPU-only score-path operational audit gate.
24. Corrected historical V206 builder recipes that still referenced `lm_head/out_proj`.
25. Updated stale ROADMAP score guidance from historical `3500/128` settings to the official active `7680/64` contract.
26. Added robust truncation detection to the single-adapter evaluator: `length`, `max_tokens`, `max_output_tokens`, `token_limit`, `truncated`, and `completion_tokens >= max_tokens`.
27. Applied the same truncation detector to the batch evaluator.
28. Hardened `solve_rate_gate.py` so approval uses exact-one-closed-boxed format, blocks multi-box false positives, and blocks candidate truncation by default.
29. Preserved `finish_reason` and `completion_tokens` through adapter-mode solve-rate scoring.
30. Upgraded the score-path operational audit to v2 with direct in-memory fixtures for truncation, multi-box rejection, and raw-output-centered score logic.
31. Added `scripts/kg1_active_gate_registry_audit.py` to satisfy the active-gate registry requirement for the no-train/no-paid/no-submit state.
32. Added a historical/superseded banner to `docs/ROADMAP.md`; it must not be used as active runbook or submit authorization.
33. Added registry checks for documentation status, active no-submit source scan, V1243 no-authorization flags, and required green reports.

## Current Validated State

Validated commands:

```powershell
python -m py_compile scripts\hf_job_train_v90.py scripts\kg1_v1243_graft_trainer_contract_gate.py scripts\kg1_v1243_solver_to_lora_graft_builder.py scripts\kg1_v1241_bit_equation_transfer_gate.py scripts\evaluate_lora_adapter.py scripts\evaluate_lora_adapters_batch.py scripts\kg1_official_metric_prescore.py scripts\kg1_official_metric_prescore_gate.py scripts\notebook_release_gate.py scripts\solve_rate_gate.py
python scripts\kg1_official_metric_prescore_gate.py
python scripts\kg1_v1243_graft_trainer_contract_gate.py
python scripts\kg1_v1241_bit_equation_transfer_gate.py --self-test
python scripts\solve_rate_gate.py --self-test
python scripts\kg1_score_path_operational_audit.py
python scripts\kg1_active_gate_registry_audit.py
```

All listed validations passed.

Additional double-check validations passed:

```powershell
python -m py_compile src\competition_utils.py scripts\kg1_score_path_operational_audit.py scripts\build_v206a_h100_loss_gated_colab.py scripts\build_v206c_h100_delta_scale_colab.py
python scripts\kg1_v1241_bit_equation_transfer_gate.py compare --self-test --output-dir artifacts\v1241_bit_equation_transfer_gate_selftest_legacy_alias
$preview = Get-Content -Raw artifacts\v1243_solver_to_lora_graft\v1243_hf_env_preview.json | ConvertFrom-Json; foreach ($p in $preview.bit_specialist.PSObject.Properties) { Set-Item -Path ("Env:" + $p.Name) -Value ([string]$p.Value) }; $env:OUTPUT_DIR='artifacts\v1243_solver_to_lora_graft_tokenize_dryrun\bit'; python scripts\hf_job_train_v90.py
$preview = Get-Content -Raw artifacts\v1243_solver_to_lora_graft\v1243_hf_env_preview.json | ConvertFrom-Json; foreach ($p in $preview.equation_specialist.PSObject.Properties) { Set-Item -Path ("Env:" + $p.Name) -Value ([string]$p.Value) }; $env:OUTPUT_DIR='artifacts\v1243_solver_to_lora_graft_tokenize_dryrun\equation'; python scripts\hf_job_train_v90.py
```

Score-path operational audit result:

- `schema_version = kg1_score_path_operational_audit_v2`
- `decision = pass_score_path_operational_audit_no_gpu_no_submit`
- `errors = 0`
- `warnings = 0`
- report: `artifacts/score_path_operational_audit/kg1_score_path_operational_audit.json`

Score-logic desk-test result:

- truncation reasons `length`, `max_tokens`, `max_output_tokens`, `token_limit`, and `truncated` all detected;
- `completion_tokens == 7680` detected as truncation;
- `completion_tokens == 7679` not flagged;
- public-metric-correct multi-box output rejected by promotion logic;
- public-metric-correct truncated output rejected by promotion logic.

Active gate registry result:

- `decision = pass_active_gate_registry_no_train_no_paid_no_submit`
- `errors = 0`
- `active_train_path_authorized = false`
- `paid_gpu_launch_authorized = false`
- `adapter_package_authorized = false`
- `kaggle_submit_authorized = false`
- report: `artifacts/active_gate_registry_audit/kg1_active_gate_registry_audit.json`

Prompt desk-test result:

- Prompt without suffix renders with exactly one suffix.
- Prompt already containing the official suffix also renders with exactly one suffix.

Tokenization dry-run results:

| phase | train rows | val rows | truncation | fallback masks | boxed weighted rows | weighted specialist share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bit specialist | 724/724 | 170/170 | 0 | 0 | 724 | 0.823357 |
| equation specialist | 544/544 | 170/170 | 0 | 0 | 544 | 0.772633 |

## Runtime Score-Contract Indicator

`scripts/hf_job_train_v90.py` now emits a score-contract runtime report in the job log:

- `KG1_SCORE_CONTRACT_RUNTIME_JSON_BEGIN`
- JSON payload with `schema_version = kg1_score_contract_runtime_v1`
- `KG1_SCORE_CONTRACT_RUNTIME_JSON_END`
- one-line summary: `KG1_SCORE_CONTRACT_STATUS=PASS|FAIL`

The V1243 HF env preview now enables:

- `SCORE_CONTRACT_RUNTIME_CHECK=1`
- `REQUIRE_SCORE_CONTRACT_PASS=1`
- `SCORE_CONTRACT_TARGET_ACCURACY=0.89`
- `SCORE_CONTRACT_FULL_ROWS=947`
- `SCORE_CONTRACT_BASELINE_CORRECT=823`
- `SCORE_CONTRACT_EXPECTED_TARGET_MODULES=down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj`
- `SCORE_CONTRACT_REQUIRE_TRAINABLE_FILTER=1`
- `SCORE_CONTRACT_MAX_PROMPT_TRUNCATION_RATE=0.0`

The report checks the score-facing contract while the job runs:

- model name and revision;
- official inference constants;
- LoRA rank and target modules;
- trainable LoRA module filter after model load;
- one official prompt suffix per row;
- assistant targets with one terminal boxed answer;
- boxed payload verified against top-level answer;
- no gate rows marked as training rows;
- offset masks, prefix mismatches, prompt truncation, and boxed-payload loss coverage;
- target math for `0.89`: `843/947`, i.e. `+20` rows from `823/947`.

This indicator does not prove a Kaggle score. It proves that the training job is still aligned with the score-generation contract. A score claim still requires real vLLM `raw_output` CSVs and V1241 full947 comparison.

## Runtime Score-Proxy Eval Telemetry

`scripts/hf_job_train_v90.py` also emits a cheaper training-direction signal at baseline, eval checkpoints, and final eval:

- `KG1_SCORE_PROXY_EVAL_JSON_BEGIN`
- JSON payload with `schema_version = kg1_score_proxy_eval_v1`
- `KG1_SCORE_PROXY_EVAL_JSON_END`
- one-line summary: `KG1_SCORE_PROXY_STATUS=<label>`

The V1243 HF env preview now enables:

- `SCORE_PROXY_EVAL_CHECK=1`
- `SCORE_PROXY_EVAL_MAX_EXAMPLES=170`
- `BASELINE_EVAL_BEFORE_TRAIN=1`
- `EVAL_MAX_EXAMPLES=170`
- `FRIENDLY_REALTIME_LOGS=1`
- `FRIENDLY_LOG_SCORE_HINTS=1`

The score-proxy report measures, by family:

- validation loss on the same completion mask used for training;
- token accuracy on all active target tokens;
- loss only on boosted final `\boxed{...}` tail tokens;
- token accuracy only on the boosted final boxed tail;
- exact boxed-tail token match rate per row;
- delta versus the pre-training baseline when baseline eval is enabled.

Use this to reject jobs where generic loss looks good but the score-facing tail does not improve. A healthy run should show falling `boxed_tail_loss` and rising `boxed_tail_token_accuracy` / `boxed_tail_exact_rate`, especially for `bit_manipulation` and `equation_transform`. This is still teacher-forced telemetry; it is not a Kaggle score claim. Real score proof still requires greedy vLLM generation, `raw_output`, extraction, verification, and V1241 full947 comparison.

## Runtime Score-Trajectory Telemetry

`scripts/hf_job_train_v90.py` now emits one more signal so a job is not judged by loss alone:

- `KG1_SCORE_TRAJECTORY_JSON_BEGIN`
- JSON payload with `schema_version = kg1_score_trajectory_v1`
- `KG1_SCORE_TRAJECTORY_JSON_END`
- one-line summary: `KG1_SCORE_TRAJECTORY_STATUS=<label>`

The V1243 HF env preview enables:

- `SCORE_TRAJECTORY_CHECK=1`
- `REQUIRE_SCORE_TRAJECTORY_PASS=0`
- `SCORE_TRAJECTORY_MIN_WEAK_EXACT_DELTA=0.0`
- `SCORE_TRAJECTORY_MAX_PROTECTED_EXACT_DROP=0.0`
- `SCORE_TRAJECTORY_MAX_OVERALL_EXACT_DROP=0.0`
- `SCORE_TRAJECTORY_MAX_BOXED_LOSS_REGRESSION=0.0`

This trajectory signal compares each checkpoint against the pre-train baseline and reports:

- `target_correct_required=843/947`, with `additional_rows_required=20`;
- weak-family exact-tail delta across `bit_manipulation` and `equation_transform`;
- separate `bit_manipulation` and `equation_transform` exact-tail deltas;
- protected-family exact-tail delta across gravity, numeral, cipher, and unit;
- overall exact-tail delta;
- boxed-tail loss delta;
- `score_trajectory_alignment`.

Interpretation:

- `BASELINE`: reference captured; no direction yet.
- `OK`: bit/equation exact-tail signal improved and global/protected families did not regress.
- `WATCH`: evidence is mixed or still insufficient.
- `RISK`: loss/proxy can look comfortable, but the score-facing trajectory is not moving correctly.
- `STOP`: global or protected-family exact-tail regression exceeded tolerance.

This is still not a score claim. It is a real-time cost-control and direction-control indicator. A candidate can only claim `>=0.89` after raw-output full947 comparison reaches at least `843/947` with strict-clean boxed outputs.

## Human-Friendly Real-Time Job Logs

`scripts/hf_job_train_v90.py` now emits human-readable `KG1-TEACH` cards in addition to machine-readable JSON markers.

Each card follows the same structure:

- `O que e`: what just happened.
- `Por que importa`: why it matters for score or cost.
- `Como ler`: how to interpret the signal.
- `Numeros-chave`: the small set of numbers to watch.
- `Proxima acao`: what to do next.

Important cards:

- `[KG1-TEACH][...][RUN_START][START]`
- `[KG1-TEACH][...][DATA_TOKENIZATION][OK]`
- `[KG1-TEACH][...][SCORE_CONTRACT][OK|STOP]`
- `[KG1-TEACH][...][MODEL_LOAD][START]`
- `[KG1-TEACH][...][LORA_TRAINABLE][CHECK]`
- `[KG1-TEACH][...][TRAIN_LOOP][START]`
- `[KG1-TEACH][...][TRAIN_PULSE][RUNNING]`
- `[KG1-TEACH][...][SCORE_PROXY][BASELINE|OK|WATCH|RISK|STOP]`
- `[KG1-TEACH][...][SCORE_TRAJECTORY][BASELINE|OK|WATCH|RISK|STOP]`
- `[KG1-TEACH][...][ABORT][STOP]`
- `[KG1-TEACH][...][JOB_DONE][OK]`

Operational rule:

- `SCORE_CONTRACT=STOP`: stop; score path is structurally unsafe.
- `SCORE_PROXY=RISK`: loss may be improving but score-facing boxed tail is not; inspect before continuing expensive GPU.
- `SCORE_TRAJECTORY=RISK/STOP`: bit/equation are not gaining safely toward the `843/947` target; do not promote.
- `SCORE_PROXY=OK`: cheap directional signal is healthy, but still not a score claim.
- `JOB_DONE=OK`: adapter and manifest exist; next proof remains raw-output evaluation.

## Remaining Non-Validated Items

These are not yet proven:

- GPU model-load dry-run with the real adapter path.
- Real V1243 specialist training.
- Real raw-output generation.
- Tiny/val170/full947 V1241 comparison against baseline.
- Any actual ACC gain.
- Any Kaggle submission authorization.

## Practical Score Forecast After Fixes

This is a pipeline-quality forecast, not a measured score:

- More credible than the previous V1243 forecast because row weights now actually affect sampling.
- First trained specialist attempt still likely lands below `0.89` unless bit transfer is strong and equation avoids format regressions.
- Realistic first-pass range remains roughly `+10` to `+20` rows if GPU training behaves.
- `0.89` requires the high end of that range.
- `0.90` still likely requires additional equation discovery/transfer beyond this V1243 pass.

## Final Audit Verdict

The score path is now materially cleaner:

- metric parity is gated;
- official inference defaults are aligned;
- raw-output promotion is stricter;
- truncation threshold matches the official surface;
- LoRA modules are exact;
- boxed payload loss is active;
- row-level sampling weights now really affect training;
- protected-family regression gates remain mandatory.

The next valid step is not a Kaggle submit. It is a GPU model-load dry-run, followed by a small real generation probe and V1241 strict comparison.

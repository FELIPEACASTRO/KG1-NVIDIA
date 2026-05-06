# Current Roadmap - NVIDIA Nemotron Kaggle - 2026-05-05

## Objective

Move from the current public score `0.86` to `0.87+` while protecting the existing rank-19 baseline.

Current live state:

- Competition: `nvidia-nemotron-model-reasoning-challenge`
- Deadline: `2026-06-15 23:59:00`
- Teams: `2711`
- Public top score: `0.87`
- Our team: `Felipe Angelo`
- Our visible public score: `0.86`
- Gap to top: `0.01`
- Latest submission: `V199B exact-sha final candidate sha 19fa057a local-audit`

## Non-Negotiable Rules

1. Do not auto-submit to Kaggle. Submission needs explicit human approval.
2. Promote only by official-score proxy: solve-rate versus V194, not eval loss alone.
3. Keep LoRA rank `<=32`, compatible with `NVIDIA Nemotron-3-Nano-30B`.
4. Submission package must be `submission.zip` with the adapter layout expected by Kaggle/vLLM.
5. No hand labeling or human prediction of validation/test rows.
6. Do not reveal API keys or sensitive tokens in logs, notebooks, docs, or prompts.

## Official Metric Implications

Kaggle evaluates final-answer accuracy. The official inference uses:

- `max_lora_rank = 32`
- `max_tokens = 7680`
- `top_p = 1.0`
- `temperature = 0.0`
- `max_num_seqs = 64`
- `gpu_memory_utilization = 0.85`
- `max_model_len = 8192`

The parser prioritizes `\boxed{}` answers and falls back to heuristic patterns / last numeric value. Training and inference must avoid malformed boxed answers, unit suffixes, nested-brace truncation, and conflicting final answers.

## What Changed After Double Check

Previous V202D/V202C work used strict loss gates. That is not enough. Drive logs show:

| Candidate | all720 loss | official360 loss | Decision |
|---|---:|---:|---|
| V194 baseline | 0.1633525139 | 0.1217182323 | Reference |
| A_all_shuffle_3s_lr2e8 | 0.1630450838 | 0.1218027311 | Reject |
| B_official_only_3s_lr2e8 | 0.1635006218 | 0.1218149149 | Reject |
| C_all_shuffle_5s_lr1e8 | 0.1632457744 | 0.1217938514 | Reject |

Even where `all720` loss improved, the official-like split regressed. These are not promotion candidates.

## 2026-05-06 Roadmap Override

`ANALISE_DESAFIO_IAS_2.txt` was reviewed after the V207A/V207B evidence. The useful parts are now folded into this roadmap as stricter execution gates.

New confirmed state:

- V194 official-like local gate: `822/947 = 0.868004`.
- V194 weak-family baseline: `190/315`.
- Current bottlenecks: `bit_manipulation` and `equation_transform`.
- Saturated strong families must be protected: `632/632`.
- V206B answer-only and V206C delta scaling are rejected by weak-family score, not merely by loss.

Measured weak-family rejects:

| Candidate | Weak Correct | Total | Delta vs V194 | Decision |
|---|---:|---:|---:|---|
| `v206c_s0p020_weak` | 157 | 315 | -33 | Reject |
| `v206c_s0p050_weak` | 123 | 315 | -67 | Reject |
| `v206c_s0p100_weak` | 158 | 315 | -32 | Reject |
| `v206b_answer_only_weak` | 150 | 315 | -40 | Reject |

V206B truncation was `43/315 = 13.65%`, so any candidate with materially elevated truncation is rejected unless it also produces a verified net correctness gain.

Roadmap change:

1. Run V207B external adapter triage first or in parallel with forensics.
2. Add a V194 row-level forensic dump before any new training branch.
3. Build deterministic solvers/verifiers for `equation_transform` and `bit_manipulation`.
4. Generate synthetic CoT only from verified solver output and only for diagnosed weak clusters.
5. Micro-training or adapter surgery is conditional, not the default next action.

Detailed integration artifact:

- `artifacts/analysis_ias_2_roadmap/ANALISE_IAS_2_ROADMAP_INTEGRATION_2026-05-06.md`

Additional `ANALISE_DESAFIO_IAS_3.txt` refinement:

- `artifacts/analysis_ias_3_roadmap/ANALISE_IAS_3_ROADMAP_INTEGRATION_2026-05-06.md`

This second refinement does not change the core roadmap. It adds detailed weak-error subtypes, step-level SymPy verification, safer bitwise parsing, filtered ReasoningGym usage, replay-ratio experiments, target-module audit, and reproducibility/license artifacts.

Additional `ANALISE_DESAFIO_IAS_4.txt` refinement:

- `artifacts/analysis_ias_4_roadmap/ANALISE_IAS_4_ROADMAP_INTEGRATION_2026-05-06.md`

This third refinement changes priority: V194 row-level forensics is now the immediate local priority. V207B external adapter triage remains useful, but should run in parallel/background when Colab is available.

Additional `ANALISE_DESAFIO_IAS_5.txt` refinement:

- `artifacts/analysis_ias_5_roadmap/ANALISE_IAS_5_ROADMAP_INTEGRATION_2026-05-06.md`

This fourth refinement keeps V209, but adds extractor sensitivity, V194 determinism diff, stronger solver gates, prompt-stability checks, bounded V207B triage, and taxonomy-driven training entry conditions.

Additional `ANALISE_DESAFIO_IAS_6.txt` refinement:

- `artifacts/analysis_ias_6_roadmap/ANALISE_IAS_6_ROADMAP_INTEGRATION_2026-05-06.md`

This fifth refinement keeps V210, but downweights over-granular taxonomy, adds semantic raw-output search, makes prompt-stability diagnostic rather than a hard gate, increases V207B smoke size, and requires V194 trainability/template audits before continuation training.

Additional `ANALISE_DESAFIO_IAS_7.txt` refinement:

- `artifacts/analysis_ias_7_roadmap/ANALISE_IAS_7_ROADMAP_INTEGRATION_2026-05-06.md`

This sixth refinement keeps V211, but makes the next action more precise: run a tail-restricted extractor/semantic `boxed_rewrite_probe` before any GPU training. It also replaces global semantic raw-output matching with final-block matching, defines a minimum 24h forensic schema, tightens V194 adapter/template audits, and keeps strong-family submit gates strict.

Additional `ANALISE_DESAFIO_IAS_8.txt` refinement:

- `artifacts/analysis_ias_8_roadmap/ANALISE_IAS_8_BOXED_REWRITE_PROBE_INTEGRATION_2026-05-06.md`

This seventh refinement approves the `boxed_rewrite_probe`, but changes how it is judged: the primary metric is now net change after preserving the 190 weak successes, not gross recovery on the 125 weak errors. It also requires balanced `\boxed{...}` parsing, stricter `fallback_last_number`, SymPy timeouts, restricted numeric tolerance, and parser-proxy disagreement checks before GPU.

## Current Assets

Validated local artifacts:

- `artifacts/competition_state.json`
- `artifacts/kaggle_live_audit/kaggle_live_audit_report.json`
- `artifacts/github_notebook_audit/github_notebooks_audit_report.json`
- `artifacts/hf_drive_audit/hf_drive_audit_report.json`
- `artifacts/google_drive_audit/google_drive_findings_2026-05-05.md`
- `artifacts/v206_data_audit/v206_data_audit_report.json`
- `data/v206/v206_curated_train.jsonl`
- `data/v206/v206_curated_manifest.json`

Drive source of truth:

- `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/submission.zip`
- `KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter`
- `KG1_NVIDIA_V202D/data/tonghuikang-0-87-nemotron-dataset.zip`

Local blocker:

- Exact V194 binary was not found in the local workspace by hash scan.
- Local GPU is not suitable for full 30B training/evaluation.
- H100/A100/Colab execution is required for training and solve-rate evaluation.

## V206 Dataset Decision

Use `data/v206/v206_curated_train.jsonl` as the next training pack.

Manifest:

- Rows: `6548`
- SHA256: `65a810d54da73fd3859d7ee9a9edc0c35a3f89231c0033ea74f26e55f254f9f0`
- Families:
  - bit_manipulation: `2050`
  - equation_transform: `2109`
  - gravity_constant: `584`
  - numeral_system: `594`
  - text_encryption: `633`
  - unit_conversion: `578`

Use:

- Core: verified v100, v95, v92, row-filtered v94.
- Rehearsal: v90 sampled per family.
- Avoid by default: v198 and unfiltered trace archives.
- Use public trace consensus only as a targeted gain queue, not bulk SFT.

## Training Plan

Phase 1 - Baseline Integrity And V194 Forensics

- Run from exact Drive V194 adapter.
- Verify adapter loads with zero missing/unexpected LoRA tensors.
- If possible, export or hash Drive `submission.zip` before training.
- Recompute V194 solve-rate on the same proxy used for candidates.
- If compute allows, repeat V194 or run duplicate sanity checks to detect vLLM/batching drift at temperature `0.0`.
- If compute allows, run two full V194 passes and save row-level diffs. If diff is greater than `3` rows, stop and debug harness/vLLM determinism before promotion decisions.
- Save raw row-level V194 outputs before new training:
  - raw output;
  - extracted answer;
  - strict boxed answer;
  - fallback extracted answer;
  - boxed count/status;
  - nested boxed status;
  - final boxed answer position;
  - tokens after final boxed answer;
  - truncation status;
  - hit-max-tokens status;
  - EOS seen status;
  - estimated prompt tokens;
  - completion tokens;
  - output template cluster;
  - failure category.
- Build completion-token histograms by family and compare strict-boxed extraction against the official/fallback parser.
- Run extractor sensitivity before training:
  - `strict_last_boxed`;
  - `strict_first_boxed`;
  - `latex_normalized_boxed`;
  - `fallback_regex`;
  - `fallback_last_number`;
  - current V207A extractor.
- If extractor disagreement changes `>=3` strong-family rows, audit the harness/parser before training.
- Add forensic fields where available: `raw_output_hash`, `adapter_sha`, `model_version`, `inference_seed`, `prompt_tokens`, `total_tokens`, `stop_reason`, `reasoning_tokens`, `estimated_context_tokens`, `tokens_remaining_budget`, `truncation_mode`, `repetition_ngram_max`, `boxed_contents_list`, `last_boxed_is_final`, `canonical_gold`, `canonical_pred`, `numeric_match_1e2`, `symbolic_match`, `extract_path`, `format_violations`, `num_bracket_pairs`, and optionally `loss_per_token`.
- Add semantic raw-output search fields where possible: `sympy_semantic_raw_match`, `bitwise_ast_raw_match`, `semantic_match_anywhere`, and `semantic_match_location`.
- Do not treat global `semantic_match_anywhere` as a primary decision signal. It can match intermediate steps and produce false positives. The primary signal should be tail/final-candidate matching:
  - `semantic_match_tail`;
  - `semantic_match_final_block`;
  - `semantic_match_scope`;
  - `semantic_false_positive_audit`.
- Run a 24h `boxed_rewrite_probe` before GPU training:
  - compare current V207A extractor, `strict_last_boxed_raw`, `latex_normalized_last_boxed`, `sympy_canonical_last_boxed`, `fallback_last_number`, numeric/base-normalized match, and tail-restricted semantic final-candidate match;
  - input must include all `315` weak rows when available: the `125` weak V194 errors plus the `190` weak V194 successes;
  - also run the same schema on the `632` strong successes if raw outputs are available;
  - compute `gross_recovery_E = new correct rows among the 125 weak errors`;
  - compute `preservation_E = rows preserved among the 190 weak successes`;
  - compute `net_change_E = gross_recovery_E - (190 - preservation_E)`;
  - make branch decisions from `net_change_best`, not gross recovery alone;
  - `net_change_best >=15` means parser/format is material enough for format-first investigation;
  - `net_change_best 6-14` means hybrid format-first plus reasoning later;
  - `net_change_best 2-5` means parser is marginal and reasoning-first remains likely;
  - `net_change_best <=1` means proceed toward solver-verified reasoning fixes;
  - `net_change_best <0` means reject the extractor and continue reasoning-first;
  - `>40%` FORMAT/EXTRACT among weak errors inverts the roadmap to format-first;
  - `<=20%` FORMAT/EXTRACT and `>=60%` reasoning/bit/algebra keeps the solver-first path.
- `boxed_rewrite_probe` extractor requirements:
  - use a balanced parser for the last `\boxed{...}`; do not rely on a simple `\\boxed{([^}]*)}` regex;
  - add `strict_last_boxed_balanced`;
  - keep LaTeX normalization conservative: strip whitespace, `$`, `\left`, `\right`, spacing commands, and normalize `\dfrac`/`\tfrac` to `\frac`; do not rewrite fractions, roots, signs, grouping braces, exponents, or domain-bearing text;
  - apply `fallback_last_number` only if no boxed candidate exists or all boxed candidates fail, and only in the last non-empty line or final marker block;
  - reject `fallback_last_number` candidates preceded by structural words such as `step`, `rule`, `index`, `line`, `example`, or `case`;
  - reject final blocks with multiple distinct numeric candidates unless a final-answer marker disambiguates;
  - apply `rtol=1e-2` only to explicitly decimal/float-like equation answers; never use tolerance for bit rows, integers, exact fractions, masks, widths, or base-normalized answers;
  - wrap SymPy parse/match in a timeout, target `<=1s` per candidate, and record `sympy_timeout`.
- `boxed_rewrite_probe` parser-proxy check:
  - implement `reference_public_extract` if a public/reference parser can be verified;
  - if none is available, create a synthetic extraction test set covering nested boxed answers, multiple boxed answers, text after boxed, missing boxed, fractions, decimal/fraction equivalence, hex/bin/decimal bit answers, and scientific notation;
  - if V207A vs reference parser differs by `>=5` strong rows, stop and align parser/harness;
  - if V207A vs reference parser differs by `>=3` weak rows, run decisions under both parsers and proceed only if the selected branch is identical.
- Minimum 24h forensic schema should not wait for every optional field. Required first-pass fields are: `id`, `task_type`, `prompt`, `gold`, `raw_output`, `raw_output_hash`, `extracted_answer`, `current_v207a_extract`, `strict_last_boxed`, `latex_normalized_last_boxed`, `fallback_last_number`, `correct`, `completion_tokens`, `prompt_tokens`, `total_tokens`, `stop_reason` or `vllm_finish_reason`, `eos_seen`, `hit_max_tokens`, `truncation_mode`, `boxed_count`, `last_boxed_is_final`, `tokens_after_final_box`, `repetition_ngram_max`, `raw_contains_gold`, `semantic_match_tail`, `solver_answer`, `template_cluster_id`, and `failure_bucket`.
- Required probe artifacts before GPU: `probe_v194_weak315.csv` or `.parquet`, optional `probe_v194_strong632.csv` or `.parquet`, `probe_summary.json`, `probe_bucket_suggestions.csv`, `probe_decision.md`, and `extractor_disagreement.csv`.
- Add token-position fields: `tokens_to_first_box`, `tokens_from_last_box_to_end`, `raw_output_len_chars`, `raw_output_len_tokens`, and `longest_repeated_ngram`.
- Add template/tokenizer drift fields before training: `tokenized_prompt_sha`, `train_template_sha`, `eval_template_sha`, `tokenizer_name_or_sha`, and `special_tokens_map_hash`.

Phase 2 - V207B External Adapter Triage

Do not launch another V206-style training branch by default. Evaluate existing candidate adapters with the V207B weak-family screen, but do not let this replace V194 forensics. V207B is a parallel/background signal; V194 forensics dictates any new data or training branch.

V207B budget for this stage:

- structural audit first;
- optional smoke screen: about `100` rows when budget allows, e.g. `40` strong + `60` weak;
- reject smoke candidates with more than `2/40` strong failures;
- weak `315` only for smoke survivors;
- full `947` only if weak is `>195/315`;
- stop after about `6h` GPU without a weak-positive candidate.

Priority candidates:

- V194 duplicate sanity paths.
- V199B sanity paths if present.
- `huikang/nemotron-adapter` variants.
- `kienngx/nemotron-nano-30b-trained` variants.
- `bugkeeper/nemotron-adapter-v20`.
- Any public adapter with valid PEFT files and rank `<=32`.

Reject immediately if:

- adapter rank is missing or `>32`;
- required adapter files are missing;
- bad historical `lm_head` namespace appears;
- weak-family score is `<=190/315`;
- truncation is materially worse with no correctness gain.

Promote to full 947-row gate only if weak-family score is `>190/315`, preferably `>=195/315`.

Phase 3 - Row-Level Weak Error Taxonomy

Classify the `125` V194 weak-family errors before generating data:

- parse/extraction failure;
- malformed or missing `\boxed{}`;
- raw output contains gold but extractor missed it;
- wrong arithmetic/calculation;
- wrong algebra/symbolic transform;
- wrong bitwise interpretation;
- ambiguity/canonicalization mismatch;
- truncation or no final answer;
- long-loop/babbling.

`bit_manipulation` subtypes to track:

- XOR/operator-precedence confusion;
- signed vs unsigned interpretation;
- binary/hex/decimal normalization;
- off-by-one shift and shift-direction confusion;
- arithmetic vs logical shift;
- mask-width and overflow masking;
- carry propagation;
- two's-complement negation;
- popcount/bit_count errors.
- sign-extension mismatch;
- shift-out-of-bounds;
- word-size inference error;
- Python negative-bitwise-semantics mismatch.
- rotate vs shift;
- leading/trailing zero count;
- bit reverse or Gray-code error;
- chained operation order;
- output base mismatch.

`equation_transform` subtypes to track:

- algebraic rearrangement error;
- variable-isolation error;
- sign inversion;
- fraction-to-decimal loss;
- symbolic vs numeric mix;
- distributive/expand/factor error;
- exponentiation error;
- equivalent expression not canonicalized;
- correct answer with wrong format;
- missing or multiple `\boxed{}` markers;
- truncation before final answer.
- bracket unbalance in polynomial expressions;
- cross-multiplication sign inversion;
- spurious-root inclusion;
- domain-validity error;
- infinite substitution loop.
- partial solve in systems of equations;
- radical/nth-root handling;
- log/exp identity misuse;
- implicit variable rename;
- incomplete multi-solution answer;
- constant vs variable confusion.

Taxonomy gate:

- cover at least `90%` of the 125 weak errors before dataset generation;
- target `>=95%`;
- if `other` is greater than `10%`, expand taxonomy before training.
- assign an operational macro bucket first: `FORMAT_EXTRACT`, `ALGEBRA_MANIP`, `ARITHM_BOUNDARY`, `LOOP_TRUNC`, `CANONICALIZATION`, `AMBIGUOUS_PROMPT`, or `OTHER`;
- detailed subtypes should inform generators but should not block progress if the macro action is clear.
- Assign macro buckets in ordered priority:
  1. `LOOP_TRUNC`;
  2. `FORMAT_EXTRACT`;
  3. `CANONICALIZATION`;
  4. `ALGEBRA_MANIP`;
  5. `ARITHM_BOUNDARY`;
  6. `AMBIGUOUS_PROMPT`;
  7. `OTHER`.
- If `ARITHM_BOUNDARY` has fewer than about `6/125` rows, fold it into `ALGEBRA_MANIP` for first training decisions and keep it only as metadata.
- `AMBIGUOUS_PROMPT` and `OTHER` are excluded from the first fixes pool.
- `OTHER >15%` is a hard stop for training.
- Revised macro branch triggers after IAS8:
  - `FORMAT_EXTRACT >=35-40%` means format-first;
  - `ALGEBRA_MANIP + ARITHM_BOUNDARY >=55-60%` means reasoning-first;
  - `LOOP_TRUNC >=20-30%` means truncation-first;
  - if these conflict, prioritize no-train/parser alignment first, then truncation-first, then format-first, then reasoning-first.

Phase 4 - Verified Data And Conditional Micro-Training

Build deterministic solvers/verifiers before training:

- SymPy-backed verifier for `equation_transform`.
- Python bitwise verifier for `bit_manipulation`.
- Synthetic data accepted only if verifier passes.
- Every training row must have provenance and a final single `\boxed{...}` answer.
- Equation CoT must pass step-level verification, not final-answer-only verification.
- Bitwise parsing must use a safe parser or AST whitelist, not raw `eval`.
- ReasoningGym can be tested only as a filtered, license-checked candidate source for observed weak clusters; never as bulk SFT.
- Preserve the shortest successful V194 output template for each family where possible; do not introduce a new verbose `Thought:` style unless V194 successful outputs already use it.
- Track `tokens_after_final_box` and reject data with trailing reasoning after the final answer.
- Preferred synthetic CoT length is under `2500` generated tokens; hard review threshold is under `4096`; reject anything that leaves less than a 1024-token safety margin inside `max_model_len=8192`.
- Add `truncation_mode` values: `none`, `hit_max_tokens_without_boxed`, `hit_max_tokens_with_boxed_earlier`, `repetitive_loop_truncation`, `long_legitimate_reasoning`, and `under_generation_no_boxed`.
- If parse/format buckets are `>=40%`, train only short template/format corrections, consider loss masking around the final `\boxed{...}`, and use heavier replay.
- If arithmetic/algebra/bitwise buckets are `>=60%`, train solver-verified concise CoT/fixes.
- If buckets are mixed or `other >10%`, do not train yet; refine taxonomy.
- Default SFT loss is normal next-token loss on completion with prompt masked. Final-box-only loss masking is allowed only in a tiny diagnostic branch if row-level evidence proves format-only failure dominates.
- Final-box-only loss masking remains banned by default for `equation_transform`. It is only allowed in a tiny diagnostic branch if `FORMAT_EXTRACT` dominates and row-level evidence shows the reasoning/final candidate is already correct.
- Training completion target should be bucket-dependent:
  - format-only rows may use short V194-style final-answer/template examples;
  - algebra rows require solver-verified concise CoT;
  - bit rows should use concise deterministic steps or direct final answers only if that matches successful V194 templates.
- Training completion length policy:
  - median target `500-1500` tokens;
  - default hard cap `2500` tokens;
  - review/reject above `4096` tokens;
  - reject any example leaving less than `1024` context tokens under `max_model_len=8192`.

Training remains conditional:

Candidate defaults:

- start with rank 8 or 16 before rank 32.
- alpha matched conservatively to rank.
- dropout 0 unless validation proves otherwise.
- max length 8192 where compute allows.
- very low LR and short runs first.
- use weak-family rows plus strong-family replay anchors.
- treat replay ratio as a measured axis: `80/20`, `50/50`, `1:4`, or `1:8`, depending on weak gain vs strong-family preservation.
- add a `1 weak fix : 7 V194-success replay` micro-replay branch if solvers validate enough weak fixes; initial scale is about `125` verified weak fixes plus about `875` V194-success replay examples.
- first conservative mix should use at least `60%` strong-success replay, `20-25%` weak-success replay, and only `15-20%` solver-verified weak fixes.
- if weak does not move and strong remains intact, add a measured alternate branch around `70%` replay total / `30%` fixes.
- do not use `1 weak fix : 7 replay` as the only branch when equation reasoning dominates, because it may dilute the correction signal.
- do not go below `50%` strong-success replay in the first training branch.
- replay examples should include strong-family successes and weak-family successes with low/medium completion tokens.
- for a V194-continuation branch, keep V194's existing rank and target modules; do not assume rank 32 if the protected adapter is rank 19 or otherwise configured.
- V194-continuation LR grid: `5e-7` and `1e-6`, one epoch first, optional second pass only if weak improves and strong rows remain intact.
- extended LR grid if needed: `3e-7`, `5e-7`, `1e-6`, starting at `5e-7`.
- add solver fuzzing/unit-test gates before data generation; no synthetic training data if solver coverage is below `95%`, with `>=99%` preferred.
- before V194 continuation, create `v194_adapter_trainability_audit.json` confirming exact rank/alpha/dropout/target modules, trainable reload with no missing/unexpected PEFT tensors, no `lm_head`/`embed_tokens`, tokenizer compatibility, and template hashes.
- audit `base_model_name_or_path`, `peft_type`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `task_type`, `target_modules`, `modules_to_save`, tokenizer revision, special tokens map, prompt prefix, chat template, and EOS behavior.
- if compute permits, run a one-step trainability/loss sanity check after loading V194 with `is_trainable=True`; reject continuation if the starting loss is unexpectedly high or reload changes PEFT tensors.
- record and audit `target_modules`; prefer valid attention/MLP modules matching the adapter layout (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- avoid embeddings/lm_head unless compatibility is proven; reject `lm_head` for submit candidates by default.
- avoid bulk training unless row-level proof shows net gain.

Phase 5 - Promotion Gate

Use `scripts/solve_rate_gate.py` or the equivalent Colab path.

Promote only if all hold:

- Full local score is at least `825/947` for submit consideration.
- `825-827/947` is review-only; prefer `828/947+` for a strict submit candidate because the public `0.86` tie is saturated.
- Weak-family score is greater than V194's `190/315`.
- Preferred weak-family score remains `>=195/315`; strict submit target is `>=198/315`.
- No strong-family regression; target remains `632/632`.
- A `631/632` strong-family review band is allowed only if full score is `>=828/947`, weak score is `>=198/315`, and the lost strong row is proven isolated/non-systematic. This band is not automatic-submit.
- `630/632` strong is diagnostic-only unless there is explicit human override after row-level proof; it is not a normal submit path.
- Strong-family truncation must be `0`.
- Weak-family truncation should be `<=1` row for strict submit; `<=3` rows remains review-only.
- Full truncation above `0.5%` requires investigation before promotion.
- Candidate prompt-stability test should pass before submit if feasible: original prompt, whitespace-normalized prompt, and harmless newline perturbation should differ by no more than `2` rows; diff greater than `3` rejects the candidate unless human override is explicit.
- Treat prompt-stability as a diagnostic flag, not a hard rejection gate by itself, unless instability is severe and overlaps with known brittle failure rows.
- Boxed/final-answer extraction rate remains near perfect.
- Candidate package rank and layout pass Kaggle preflight.
- Public leaderboard submission is manually approved.

Recommended threshold for a real submission candidate:

- Strict target: at least `+6` net correct over V194 (`>=828/947`) with zero strong-family regression.
- Preferred strict target: `>=830/947` with weak `>=198/315`.
- Review-only minimum: at least `+3` net correct over V194 (`>=825/947`) with exceptional row-level evidence.
- Never submit a candidate that only improves eval loss.

Phase 6 - Manual Submission Packet

Before any submission:

- Create `submission.zip`.
- Verify `adapter_config.json` and `adapter_model.safetensors`.
- Verify rank `<=32`.
- Save SHA256 for zip and adapter.
- Save solve-rate report and row deltas.
- Save `adapter_audit.json`.
- Save `full_gate_report.json` and `per_family_report.csv`.
- Save `data_lineage.json`.
- Save `license_manifest.md`.
- Save `reproducibility.md`.
- Save extractor sensitivity and determinism reports if generated.
- Human go/no-go.

## Rejected Paths

- V202C A/B/C as currently logged.
- V202D loss-only strict-promotion artifacts.
- V206B answer-only as a promotion path.
- V206C delta scaling as a promotion path.
- NF4 adapter paths that historically regressed.
- `strip_lm_head` paths.
- Bulk v198 micro-distill without solve-rate proof.
- Raw public traces without strict verifier filtering.
- Generic adapter soup before a weak-positive external adapter exists.
- Unverified LLM-generated labels.
- LLM-generated CoT used as labels without deterministic verification.
- Bulk ReasoningGym or bulk train.csv SFT without weak-cluster filtering.
- Post-processing/canonicalization hooks as final strategy; they are allowed only for diagnosis or training-data verification.
- Aggressive LR changes such as `1e-4` or `2e-4` as default training settings.
- Any submit candidate touching `lm_head` without a documented, passing structural and family gate.
- Synthetic CoT that is too long, teaches loops, contains intermediate boxed answers, or leaves trailing text after the final boxed answer.
- New answer templates that diverge from V194 successful output style without row-level proof.
- Training before extractor sensitivity and weak-error taxonomy are complete.
- Bulk weak-heavy replay mixes such as `80% weak / 20% strong` as the initial branch.
- Continuing V194 without a trainability/template/tokenizer audit.
- Final-box-only loss masking as a default strategy.

## Next Action

V214 update: the V194 raw outputs were found in Google Drive and downloaded locally. `boxed_rewrite_probe` reproduced V194 at `822/947`, weak `190/315`, strong `632/632`, and found `0` safe parser/format recoveries among the `125` weak errors. Buckets are now `ALGEBRA_MANIP=100`, `ARITHM_BOUNDARY=24`, `LOOP_TRUNC=1`, `FORMAT_EXTRACT=0`, so the active branch is `reasoning_first_solvers_dataset`.

The legacy solver probe found `bit_manipulation` solver `159/160`, correcting `24/25` V194 bit errors with zero V194-success losses. It also found `equation_transform` solver `57/155`, only `3` gains and `1` loss, so equation remains diagnostic-only.

V214 execution update: the bit preview and non-validation micro-replay candidate were built. The trainable dataset is `data/v214/v214_micro_replay_candidate.jsonl` with `880` verified single-boxed rows and zero V194 validation overlap. A deterministic internal loss split was also created:

- `data/v214/v214_micro_train.jsonl`: `792` rows.
- `data/v214/v214_micro_val.jsonl`: `88` rows.
- `data/v214/v214_micro_split_manifest.json`: train/val overlap `0`.

Next action: run the V214 Colab dry-run gate first. Training is blocked unless the Colab environment explicitly sets `KG1_V214_RUN_TRAIN=1`. If training runs, it is a one-step V194 continuation at `3e-7` with a trainable LoRA module filter, followed by weak eval first. Full `947` eval is blocked unless weak improves over V194 (`>=191/315`) with acceptable truncation. No Kaggle submit is allowed.

Current V214 Colab execution URL:

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

Use the published branch URL unless the notebook is later merged to `master`.

No Kaggle submission is allowed without explicit human approval.

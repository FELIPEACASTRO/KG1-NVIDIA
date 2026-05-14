# V379 Dataset Double Check

## Verdict

- `nemotron_dataset_final`: `13` files, `509113679` bytes.
- `nemotron_hacker_dataset`: `7` files, `401052223` bytes.
- Common files: `6`; hash mismatches: `0`.
- `nemotron_dataset_final` is the superset. The hacker directory is duplicated/subsumed for active roadmap purposes.

## Active Findings

- Solver parquet: `{'rows': 823, 'metric_correct': 800, 'metric_wrong': 23, 'v375_coverage': {'category_counts': {'None': 3, 'arithmetic': 36, 'little_endian': 29, 'mixed_concat': 14}, 'correct': 79, 'coverage_csv': 'C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v378_nemotron_dataset_final_audit\\v378_v375_solver_coverage.csv', 'rows': 82, 'wrong': 3}}`.
- Filtered logprob dataset: `{'rows': 8703, 'unique_ids': 7044, 'label_correct': 8703, 'cot_correct': 8691, 'bit_cot': {'correct': 1752, 'rows': 1754, 'wrong': 2}, 'equation_cot': {'correct': 2428, 'rows': 2438, 'wrong': 10}}`.
- V375 residual coverage: `{'coverage_csv': 'C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v378_nemotron_dataset_final_audit\\v378_v375_residual_trace_solver_coverage.csv', 'filtered_trace_correct': 91, 'filtered_trace_covered': 92, 'rows': 92, 'solver_correct': 79, 'solver_covered': 82}`.
- Solver conditioning audit: `{'True': 741, 'False': 82}`; conditioned rows are repair evidence, not independent proof traces.
- Filtered dataset duplicate audit: `821` exact duplicate rows and `1659` duplicate-ID rows.

## Gaps Removed From Active Plan

- V217 train prompt overlap with the final package: `1476` prompts; families `{'bit_manipulation': 654, 'equation_transform': 246, 'gravity_constant': 144, 'numeral_system': 144, 'text_encryption': 144, 'unit_conversion': 144}`.
- V217 validation prompt overlap with the final package: `103` prompts; families `{'bit_manipulation': 30, 'equation_transform': 9, 'gravity_constant': 16, 'numeral_system': 16, 'text_encryption': 16, 'unit_conversion': 16}`. Any future train/validation split must filter these prompt hashes.
- `competition_test.csv` sample overlap: `3/3` IDs and `3/3` prompts overlap train. It is not an evaluation set.
- `sft_train_converted.jsonl` format audit: `1659` duplicate message rows and `6923` malformed think-tag rows.
- `sft_train_full_9500.jsonl` audit: all rows have multiple boxed spans; `364` rows have a wrong declared boxed answer before the final corrected box (`{'bit_manipulation': 238, 'equation_transform': 126}`). `173` equation answers contain braces.
- Do not use `nemotron_traj.csv` as labels; keep only for hard-negative/confidence analysis.
- Do not use the whole `sft_train_reconstructed.jsonl`; it contains unknown/synthetic rows and is superseded by focused sources.
- Do not treat missing report-mentioned `tong_with_logprob.csv` / `yours_with_logprob.csv` as available evidence; they are absent from both audited directories.
- Report-claimed-but-missing files: `['kaggle_logprob/results/tong_with_logprob.csv', 'kaggle_logprob/results/yours_with_logprob.csv']`.

## Next Action

V380 CPU-only equation solver candidate patch using 79 V375 solver-correct rows, then trace/tokenization gate.

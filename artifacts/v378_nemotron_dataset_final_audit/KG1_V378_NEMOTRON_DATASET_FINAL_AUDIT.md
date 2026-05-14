# V378 Nemotron Dataset Final Audit

## Verdict

- All files in the ZIP and extracted directory were inventoried; no large file was copied into the repo.
- The new high-value item is `solver_results.parquet`: it gives structured solver metadata for `823` equation rows and covers `82/92` V375 residual equation misses.
- The filtered logprob CSV is also useful: it covers all `92/92` V375 residual equation misses with generated CoT, `91/92` correct by project scorer.
- This audit authorizes CPU-only V378/V379 gate work, not immediate HF or submit.

## Key Signals

- `solver_results.parquet`: `800/823` correct, V375 coverage `79/82`.
- `filtered_merged_dataset.csv`: `8691/8703` CoT-correct, labels `8703/8703`.
- `sft_train_full_9500.jsonl`: `9500/9500` correct.
- residual trace+solver coverage: `{'rows': 92, 'filtered_trace_covered': 92, 'filtered_trace_correct': 91, 'solver_covered': 82, 'solver_correct': 79, 'coverage_csv': 'C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v378_nemotron_dataset_final_audit\\v378_v375_residual_trace_solver_coverage.csv'}`.

## Actionable Next Step

- Build a CPU-only candidate patch/gate from the 79 solver-correct V375 residual rows plus 91 trace-correct residual CoTs.
- Use one best trace per ID, no duplicate reweighting, tokenizer/offset-mask/truncation checks, and weak gate before HF.
- Do not use raw `nemotron_traj.csv` as labels; it is only `4542/9500` correct.

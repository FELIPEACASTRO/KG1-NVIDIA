# V379 Attached Reports Triple Check

## Verdict

- The attached reports add useful process rules, not a new measured gain.
- Attached `competition_train.csv` matches final package: `True`.
- Official train rows: `9500`; family counts: `{'bit_manipulation': 1602, 'equation_transform': 1555, 'gravity_constant': 1597, 'numeral_system': 1576, 'text_encryption': 1576, 'unit_conversion': 1594}`.
- Raw attached reports are not committed because at least one contains an HF token-like string; only redacted metadata is versioned.

## Actionable Findings

### Original andy279 SFT train has 49,290 examples / 7,200 unique puzzles

- Evidence: `mentioned_in_reports=3; local_original_files_available=false`.
- Roadmap decision: not active training input until actual sft_train.jsonl is approved/downloaded and audited.

### Original validation has 1,165 examples / 1,123 puzzles, including 399 unsolved transformation rows

- Evidence: `mentioned_in_reports=3; transformation_unsolved_mentions=2`.
- Roadmap decision: supports equation_transform as solver/DSL problem; no direct adapter gain without data access.

### Original train contains heavy bit/equation signal

- Evidence: `bit_17285_mentions=3; transformation_10741_mentions=3; solver_bit_1602_mentions=3; solver_transformation_1101_mentions=3`.
- Roadmap decision: if access is granted later, mine only after strict V381-style gates; do not assume immediate gain.

### SFT README quality recipe cleans boxed LaTeX, reextracts answers, recomputes correctness, keeps only correct attempts

- Evidence: `quality_recipe_mentions=4`.
- Roadmap decision: promote this as mandatory V381 cleaning rule for any local trace source.

### competition_test.csv has 34 puzzles

- Evidence: `report_mentions=2; audited_final_test_rows=3`.
- Roadmap decision: claim contradicted locally; active roadmap keeps competition_test.csv retired as eval.

### Downloaded competition_train.csv is new data

- Evidence: `sha256=d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa; rows=9500; unique_ids=9500`.
- Roadmap decision: not new; identical to final package train and useful only as official prompt/answer reference.

### Reports mention tong_with_logprob/yours_with_logprob

- Evidence: `tong_mentions=1; yours_mentions=1; V379 confirmed files absent locally`.
- Roadmap decision: not evidence until files exist and pass hash/metric audit.

### Reports mention raw multi-attempt trace files

- Evidence: `raw_trace_mentions=2; V379 inventories do not contain these raw trace JSONLs`.
- Roadmap decision: not active input; if acquired later, audit as a new source before any training.

## Gain Assessment

- New measured adapter gain: `0`.
- New measured CPU gain: `0`.
- Actionable route: No direct gain from the attached files. The only actionable improvement is better gating/cleaning for V380/V381. Expected gain remains conditional on V380 accepting solver categories with zero losses, then V381 transferring without bit regression.

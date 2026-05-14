# V347 V346 Failure Audit

This folder records the ACC-first audit for V346 checkpoint-2.

Result:

- Baseline V290 checkpoint-6: `192/315`, `equation_transform=56/155`, `bit_manipulation=136/160`.
- V346 answer exact-match checkpoint-2: `191/315`, `equation_transform=56/155`, `bit_manipulation=135/160`.
- V343 CPU solver/verifier reference: `199/315`, `equation_transform=63/155`, `bit_manipulation=136/160`.

Decision:

- V346 has `0` gains versus baseline and `1` bit regression.
- The `7` V343 solver gains were not transferred to the adapter.
- Do not run full eval, package, Kaggle submit, or more H200 evals for V346 checkpoint-4/6 without new CPU evidence.

Note: the audit reuses `scripts/analyze_v345_v344_failure_audit.py`, so some generated CSV/JSON field names still say `v344`; in this folder those fields refer to the V346 checkpoint-2 predictions.

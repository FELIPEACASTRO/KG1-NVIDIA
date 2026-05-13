# KG1 V340 - Hard Negative Abstain Gate

Generated at UTC: `2026-05-13T21:00:36.915023+00:00`

## CPU Signal

- V336A integrated weak: `197/315`.
- Equation: `61/155`.
- Bit: `136/160`.
- Accepted no-loss candidates: `5`.

## V337D Preference Assets

- Preference train rows: `2880`.
- Hard negative wrong-box rows: `683`.
- Negative types: `{"format_negative_multiple_boxes": 720, "format_negative_no_box": 720, "format_negative_trailing_text": 720, "hard_negative_equation_near_miss": 720}`.

## V338B Evidence

- `eval_loss` improved, but checkpoint weak eval regressed to `190/315`, equation `56`, bit `134`.

## Decision

- `v340_cpu_gate_failed_block_hf`
- HF GPU allowed: `False`.
- Reason: Hard-negative/abstain assets failed validation: preference_train_rows_have_invalid_pairs,preference_validation_rows_have_invalid_pairs,preference_train_hard_negative_same_box,preference_validation_hard_negative_same_box
- Next action: Fix CPU artifacts before any HF GPU job.

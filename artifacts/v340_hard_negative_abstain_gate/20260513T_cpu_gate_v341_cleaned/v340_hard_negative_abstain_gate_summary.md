# KG1 V340 - Hard Negative Abstain Gate

Generated at UTC: `2026-05-13T21:04:35.506545+00:00`

## CPU Signal

- V336A integrated weak: `197/315`.
- Equation: `61/155`.
- Bit: `136/160`.
- Accepted no-loss candidates: `5`.

## V337D Preference Assets

- Preference train rows: `2843`.
- Hard negative wrong-box rows: `683`.
- Negative types: `{"format_negative_multiple_boxes": 720, "format_negative_no_box": 720, "format_negative_trailing_text": 720, "hard_negative_equation_near_miss": 683}`.

## V338B Evidence

- `eval_loss` improved, but checkpoint weak eval regressed to `190/315`, equation `56`, bit `134`.

## Decision

- `v340_preference_assets_valid_preference_trainer_required_smoke_allowed`
- HF GPU allowed: `True`.
- Reason: CPU assets passed and a preference trainer launcher exists. Only a tiny preference smoke is allowed; first-checkpoint kill-switch remains total>192, equation>56, bit>=136.
- Next action: Run a tiny preference-training smoke with strict first-checkpoint FinOps kill-switch.

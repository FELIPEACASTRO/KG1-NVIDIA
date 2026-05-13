# KG1 V340 - Hard Negative Abstain Gate

Generated at UTC: `2026-05-13T22:31:24.179991+00:00`

## CPU Signal

- V336A integrated weak: `199/315`.
- Equation: `63/155`.
- Bit: `136/160`.
- Accepted no-loss candidates: `7`.

## V337D Preference Assets

- Preference train rows: `4160`.
- Hard negative wrong-box rows: `1040`.
- Negative types: `{"format_negative_multiple_boxes": 1040, "format_negative_no_box": 1040, "format_negative_trailing_text": 1040, "hard_negative_equation_near_miss": 1040}`.

## V338B Evidence

- `eval_loss` improved, but checkpoint weak eval regressed to `190/315`, equation `56`, bit `134`.

## Decision

- `v340_preference_assets_valid_but_gpu_blocked_until_preference_trainer`
- HF GPU allowed: `False`.
- Reason: V337D SFT and preference assets are internally valid, but V338B already showed SFT loss improvement does not transfer to family ACC. No preference/abstain trainer launcher was provided, so another HF SFT job would repeat the falsified path.
- Next action: Implement a DPO/ORPO/KTO-style or explicit abstain-selector trainer that consumes the preference rows, then run this gate again with --preference-launcher.

# KG1 V526 Example Mean Objective Dry Run

generated_at_utc: 2026-05-17T02:50:29.388798+00:00

## Decision

- status: `example_mean_dry_run_passed`
- gpu_allowed: `True`
- scope: `one_short_h200_smoke_only`
- reason: example_mean keeps the row-normalized family mix close to the V522 reference
- next_action: Create a V523 example_mean smoke launcher with first-checkpoint ACC kill-switch

## Objective Mix

| Metric | Value |
|---|---:|
| reference bit share | 0.741935 |
| token_mean bit share | 0.819409 |
| example_mean bit share | 0.688109 |
| token_mean bit/equation ratio | 4.537382 |
| example_mean bit/equation ratio | 2.20625 |
| reference bit/equation ratio | 2.874993 |
| example_mean delta from reference | 0.053826 |

## Dataset Checks

- train boxed missing: `0`
- train answer mismatch: `0`
- train control chars: `0`
- train forbidden training flags: `0`
- validation boxed missing: `0`
- validation answer mismatch: `0`
- validation control chars: `0`
- validation forbidden training flags: `0`

## Blockers

- none

# V369 V368 transfer failure audit

Generated: 2026-05-14

## Result

- Baseline adapter-only: `192/315`, equation `56/155`, bit `136/160`.
- V366 CPU teacher: `222/315`, equation `63/155`, bit `159/160`.
- V368 checkpoint-1 adapter-only: `191/315`, equation `56/155`, bit `135/160`.

## Transfer check

- V366 accepted gains tested: `8`.
- V366 gains transferred to V368: `0/8`.
- V368 changed `10` rows versus baseline: `1` gain, `2` losses, `7` neutral changes.
- V368 unique gain IDs: `4ef88f92`.
- V368 loss IDs: `8740ed31, 59bee375`.

## Decision

Blocked. V368 does not justify more HF spend on the V367/V368 bit-only transfer route.

Next action: CPU-only. Either diagnose a new solver-to-adapter signal with stronger evidence, or return to equation/bit DSL gates. No full eval, package, or Kaggle submit from V368.

## Local artifacts

- Manifest: `artifacts\v369_v368_transfer_failure_audit\20260514T_cpu_audit\v369_v368_transfer_failure_manifest.json`
- V366 transfer detail: `artifacts\v369_v368_transfer_failure_audit\20260514T_cpu_audit\v369_v366_gain_transfer.csv`
- Changed rows: `artifacts\v369_v368_transfer_failure_audit\20260514T_cpu_audit\v369_v368_changed_vs_baseline.csv`
- Family summary: `artifacts\v369_v368_transfer_failure_audit\20260514T_cpu_audit\v369_family_summary.csv`

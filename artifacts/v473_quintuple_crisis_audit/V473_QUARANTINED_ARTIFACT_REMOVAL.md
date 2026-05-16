# V473 Quarantined Artifact Removal

These tracked artifacts were removed from the active worktree because they are
not valid training/eval/package inputs after the crisis audits:

- `artifacts/v447_v446_trace_dataset/**`: V447 contained `hypothesis_formed`
  traces with contradictory internal boxed answers.
- `artifacts/v464_v463_numeric_multirule_dataset/**`: V464 contained rows where
  the rejected candidate verified equal to the gold answer.
- `artifacts/v468_v464_symbol_fix_dataset/**`: V468 fixed the rejected-candidate
  contradiction but still inherited a full-reference exact prompt/answer seed.

The operational record is preserved in:

- `artifacts/roadmaps/KG1_ERROR_LEDGER_2026_05_15.md`
- `artifacts/roadmaps/KG1_SCORE_IMPROVEMENT_ROADMAP_2026_05_10.md`
- `artifacts/v473_quintuple_crisis_audit/V473_QUINTUPLE_CRISIS_AUDIT_RESULT.md`

Rule: future work must rebuild a new clean dataset/version with forbidden
reference CSV checks, contradiction checks, symbol/boxed extraction checks, and
fresh V286 output. These deleted paths must not be restored as active inputs.

# V653 V618 Applicability Note

Generated: 2026-05-18.

V618 was executed after the V653 launcher correction from `MAX_STEPS=2`,
`LR=2e-8` to `MAX_STEPS=20`, `LR=1e-6 -> 1e-7`.

Result:

- cleared prior route blockers:
  - `command_export_audit_missing`;
  - `learning_rate_too_low_for_output_policy_route`;
  - `output_policy_steps_lt_20`;
  - `final_learning_rate_extremely_low_for_output_policy_route`.
- remaining V618 blockers:
  - `dataset_eos_policy_not_declared`;
  - `dataset_multiline_targets_nonzero`;
  - `dataset_non_ascii_targets_nonzero`;
  - `dataset_prompt_contract_not_official_like`;
  - `weak_csv_sha_mismatch`.

Interpretation:

- The remaining blockers come from applying the V618 official-template
  single-line output-policy gate to a V653 compact-trace SFT dataset manifest.
- V653 is not an answer-only/official-template dataset. It intentionally uses
  short multiline rule traces plus boxed suffix to satisfy V513 learnability
  after V652 answer-only was blocked.
- Therefore V618 is not the promotion gate for V653. The relevant V653 gates
  are V509, V286, V513, V478, V524, V526, static safety, pre-paid integration,
  HF LFS preflight, checkpoint weak eval, anti-runaway/protected-row eval and
  label-free `verify_answer`.

Action:

- Do not treat the residual V618 schema blockers as permission to launch or as
  a hard block on V653.
- Do treat the V618 LR/steps finding as implemented: V653 now uses a non-trivial
  20-step route with checkpoint/eval every 2 steps.

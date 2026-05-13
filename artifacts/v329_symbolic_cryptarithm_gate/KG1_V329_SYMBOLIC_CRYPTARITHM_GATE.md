# KG1 V329 Symbolic Cryptarithm Gate

Date: 2026-05-13

## Scope

V329 is a CPU-only gate for the remaining `equation_transform` symbolic/punctuation misses.
It tests one narrow, label-free hypothesis: some 5-character Alice expressions encode two
two-digit numbers around one symbolic operator, and the RHS encodes the decimal result.

It does not train, run GPU inference, package, or submit.

## Result

- Baseline weak contract: `192/315`, `equation_transform=56/155`, `bit_manipulation=136/160`.
- Existing V324 accepted gain: `+4 equation`, projected `196/315`, `equation=60/155`.
- New V329 accepted gain: `+1 equation_symbolic_punct`.
- Combined V324+V329 projected weak: `197/315`.
- Combined projected `equation_transform`: `61/155`.
- `bit_manipulation` guardrail remains `136/160`.
- Conflicts: `0`.

Accepted row:

| id | subtype | rule_class | baseline | prediction | answer |
|---|---|---|---|---|---|
| `99d6a3b5` | `equation_symbolic_punct` | `symbolic_cryptarithm_single_operator_digits_mul` | `(<))` | `?()<` | `?()<` |

## Critical QA Finding

The first broad class was blocked correctly:

- `symbolic_cryptarithm_operator_digits_mul` produced correct rows and wrong rows.
- After splitting by prompt-derived structure, only `single_operator_digits_mul` was promotable.
- Multi-operator classes remain blocked because they have incorrect candidates.

Sensitivity check:

- Re-ran the gate with `solver_time_limit_s=0.2` and `max_solutions_per_assignment=10`.
- Accepted candidates CSV stayed identical.
- Rule summary CSV stayed identical.
- Therefore the accepted `+1` is stable against a less aggressive CP-SAT timeout in this audit.

This is a useful positive signal, but it is not Kaggle-submit-ready as a postprocessor. It must be
distilled into adapter behavior or allowed by an official inference path before any submit.

## Files

- Script: `scripts/run_v329_symbolic_cryptarithm_gate.py`
- Manifest: `artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/v329_symbolic_cryptarithm_manifest.json`
- Sensitivity manifest: `artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_sensitivity/v329_symbolic_cryptarithm_manifest.json`
- Audit CSV: `artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/v329_symbolic_cryptarithm_audit.csv`
- Accepted candidates: `artifacts/v329_symbolic_cryptarithm_gate/20260513T_cpu_gate_r2/v329_symbolic_cryptarithm_accepted_candidates.csv`

## Next Action

Build V330 distillation rows for this exact rule class, then run tokenization and no-regression gates.
Do not launch HF GPU until the V330 dataset gate proves:

- no weak/full prompt leakage;
- no answer conflicts;
- offset masks present;
- `bit>=136` guardrail remains protected;
- first weak checkpoint must beat `equation=56` without reducing bit.

# V357 Bit Global Ternary Gate Summary

Generated: 2026-05-14

## Status

V357 passed the CPU no-loss gate.

## Result

| Metric | V350 baseline | V357 integrated | Delta |
|---|---:|---:|---:|
| Weak total | `201/315` | `214/315` | `+13` |
| `equation_transform` | `63/155` | `63/155` | `0` |
| `bit_manipulation` | `138/160` | `151/160` | `+13` |
| Losses | `0` | `0` | `0` |

## Accepted IDs

`e6d2a064`, `0e70c867`, `05ca617c`, `a6704625`, `78d02fc5`, `55d834d1`, `4ef88f92`, `202af98d`, `3ace787f`, `3a7dd604`, `06120e47`, `e1f3ffbb`, `82ae858c`.

## Artifacts

- Manifest: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_bit_global_ternary_gate_manifest.json`
- Integrated predictions: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_integrated_predictions.csv`
- Candidate decisions: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_candidate_decisions.csv`
- Candidate rules: `artifacts/v357_bit_global_ternary_gate/20260514T_cpu_gate/v357_candidate_rules.csv`

## Decision

V357 is a CPU teacher/verifier only. It is not a valid Kaggle package path by itself. It authorizes V358 dataset construction and one short V359 HF smoke run with FinOps kill-switch.

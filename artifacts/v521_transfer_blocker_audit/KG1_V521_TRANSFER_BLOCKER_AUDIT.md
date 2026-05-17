# V521 Transfer Blocker Audit

## Decision

- GPU allowed: `False`
- Status: `blocked_until_new_cpu_transfer_signal`
- Reason: V518 showed loss/ACC divergence and V520 found zero submit-safe adapter candidates above baseline. The active datasets are either already failed as-is or have insufficient new bit transfer signal.
- Next action: Build V522 CPU source-target alignment/learnability audit: mine only train/public solver traces, prove new coverage over the protected bit backfire class and at least one equation rule class, then permit GPU only if the no-GPU gate predicts a real label-free gain with bit>=136, trunc=0, and 8740ed31 preserved.

## Why this matters

- V517 reduced loss, but V518 did not improve submit-safe ACC.
- V518 gained one equation row and lost the protected bit row `8740ed31=01101000`.
- V520 found zero local adapter-only CSVs above the label-free baseline without backfire.
- Therefore, another paid job is blocked until a CPU-only transfer gate proves new signal.

## Dataset Summary

| Dataset | Split | Rows | Family counts | Bit traces | Equation traces | Finding |
|---|---:|---:|---|---:|---:|---|
| v390_equation_no_loss_distill | train | 800 | `{"equation_transform": 800}` | 0/0 | 800/800 | blocks direct GPU: equation-only dataset has no bit guardrail rows |
| v390_equation_no_loss_distill | validation | 200 | `{"equation_transform": 200}` | 0/0 | 200/200 | blocks direct GPU: equation-only dataset has no bit guardrail rows |
| v475_equation_bit_replay_mix | train | 1312 | `{"bit_manipulation": 512, "equation_transform": 800}` | 0/512 | 800/800 | already tested: V495/V496 gained equation but lost bit and truncation |
| v475_equation_bit_replay_mix | validation | 328 | `{"bit_manipulation": 128, "equation_transform": 200}` | 0/128 | 200/200 | already tested: V495/V496 gained equation but lost bit and truncation |
| v510_canonical_active_training_pool | train | 2627 | `{"bit_manipulation": 609, "equation_transform": 2018}` | 0/609 | 2018/2018 | blocks as-is: bit trace ratio below 80 percent; already tested: V511/V513 showed no transferable bit trace signal as built |
| v510_canonical_active_training_pool | validation | 637 | `{"bit_manipulation": 133, "equation_transform": 504}` | 0/133 | 504/504 | blocks as-is: bit trace ratio below 80 percent; already tested: V511/V513 showed no transferable bit trace signal as built |
| v515_v514_fullbyte_residual | train | 2491 | `{"bit_manipulation": 473, "equation_transform": 2018}` | 406/473 | 2018/2018 | already tested: V517/V518 lower loss still lost protected bit row; risk: unweighted bit share below 25 percent |
| v515_v514_fullbyte_residual | validation | 620 | `{"bit_manipulation": 116, "equation_transform": 504}` | 97/116 | 504/504 | already tested: V517/V518 lower loss still lost protected bit row; risk: unweighted bit share below 25 percent |
| v304_solver_trace_distill | train | 12822 | `{"bit_manipulation": 4231, "equation_transform": 8015, "gravity_constant": 144, "numeral_system": 144, "text_encryption": 144, "unit_conversion": 144}` | 1536/4231 | 1081/8015 | no structural blocker found by V521 |
| v304_solver_trace_distill | validation | 969 | `{"bit_manipulation": 332, "equation_transform": 573, "gravity_constant": 16, "numeral_system": 16, "text_encryption": 16, "unit_conversion": 16}` | 168/332 | 120/573 | no structural blocker found by V521 |

## Operational Rule

Do not run H200/A100/HF GPU from these datasets as-is. A new job needs a V522-style CPU gate that proves:

1. no exact prompt overlap with weak/full rows;
2. no weak/full training flags;
3. protected row `8740ed31` remains correct in weak eval;
4. label-free total improves beyond baseline;
5. `bit_manipulation>=136`, `equation_transform>55`, and `truncated=0`.

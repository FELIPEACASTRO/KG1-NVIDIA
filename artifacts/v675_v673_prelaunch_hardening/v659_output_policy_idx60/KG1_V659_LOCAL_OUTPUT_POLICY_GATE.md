# KG1 V659 Local Output Policy Gate

- generated_at_utc: `2026-05-19T20:06:21.327897+00:00`
- label: `v675_v673_output_policy_after_no_lmhead_idx60`
- status: `passed`
- submit_allowed: `False`
- train_or_eval_allowed: `True`
- blocker_count: `0`
- warning_count: `2`

## Inputs

- `train`: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v284_official_gate_worktree\artifacts\v673_guarded_equation_bit_transfer_dataset\20260519T190246Z\v673_guarded_equation_bit_transfer_train.jsonl` sha256 `69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f`
- `validation`: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v284_official_gate_worktree\artifacts\v673_guarded_equation_bit_transfer_dataset\20260519T190246Z\v673_guarded_equation_bit_transfer_val.jsonl` sha256 `df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b`
- weak_reference_csv: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v284_official_gate_worktree\artifacts\v290_rank19_micro_patch_reference\runtime_artifacts\v245_weak_eval_bridge\v245-weak-bridge-hfonly-20260510T1950Z\v221_weak_315.csv`
- weak_reference_sha256: `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`

## Dataset Summaries

### train

- rows: `720`
- family_counts: `{"bit_manipulation": 240, "equation_transform": 480}`
- effective_family_weights: `{"bit_manipulation": 0.14893617021276573, "equation_transform": 0.8510638297872342}`
- assistant_word_stats: `{"count": 720, "max": 54.0, "min": 26.0, "p50": 48.5, "p90": 54.0, "p95": 54.0, "p99": 54.0}`
- first_box_word_idx_stats: `{"count": 720, "max": 53.0, "min": 25.0, "p50": 47.5, "p90": 53.0, "p95": 53.0, "p99": 53.0}`
- bit_op_counts: `{"AND": 26, "CHO": 47, "NOT": 52, "OR": 104, "XOR": 60}`
- issue_counts: `{}`

### validation

- rows: `180`
- family_counts: `{"bit_manipulation": 60, "equation_transform": 120}`
- effective_family_weights: `{"bit_manipulation": 0.14893617021276603, "equation_transform": 0.851063829787234}`
- assistant_word_stats: `{"count": 180, "max": 54.0, "min": 26.0, "p50": 48.5, "p90": 54.0, "p95": 54.0, "p99": 54.0}`
- first_box_word_idx_stats: `{"count": 180, "max": 53.0, "min": 25.0, "p50": 47.5, "p90": 53.0, "p95": 53.0, "p99": 53.0}`
- bit_op_counts: `{"AND": 5, "CHO": 16, "NOT": 15, "OR": 25, "XOR": 16}`
- issue_counts: `{}`

## Cross Split

`{"train_prompt_hashes": 720, "train_val_prompt_answer_overlap": 0, "train_val_prompt_overlap": 0, "val_prompt_hashes": 180}`

## Top Blockers

- none

## Top Warnings

- `{"code": "bit_required_op_absent_from_dataset", "missing_ops": ["ROT", "SHL", "SHR"], "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v673_guarded_equation_bit_transfer_train.jsonl", "split": "train"}`
- `{"code": "bit_required_op_absent_from_dataset", "missing_ops": ["ROT", "SHL", "SHR"], "path": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v673_guarded_equation_bit_transfer_dataset\\20260519T190246Z\\v673_guarded_equation_bit_transfer_val.jsonl", "split": "validation"}`

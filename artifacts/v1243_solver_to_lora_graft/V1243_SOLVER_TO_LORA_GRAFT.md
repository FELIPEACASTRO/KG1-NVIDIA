# KG1 V1243 Solver-to-LoRA GRAFT

Generated UTC: `2026-06-14T16:36:21+00:00`

## Verdict

- Decision: `pass_v1243_cpu_dataset_graft_no_gpu_no_submit`
- This is CPU-only data preparation. It does not authorize GPU, package, or Kaggle submission.

## Algorithm

`GRAFT` = Gate-verified Replay Answer-Focused Transfer.

The method builds family-specialist LoRA training packs from V1240 solver-verified rows:

- bit specialist: bit rows plus protected replay.
- equation specialist: equation rows plus protected replay.
- micro consolidation: bit + equation + protected replay, only after specialists pass gates.
- validation: V1240 val170 held out for raw-output gate checks.

Targets stay short and score-facing: one terminal boxed answer. The intended trainer
contract is completion-only loss with priority on the final boxed payload. The generated
HF preview locks `TOKENIZE_ONLY_DRY_RUN=1`, `REQUIRE_OFFSET_MASK=1`,
`REQUIRE_BOXED_PAYLOAD_WEIGHT=1`, `BOXED_PAYLOAD_LOSS_WEIGHT=5.0`,
the runtime score-contract indicator in hard-fail mode, and score-proxy
evaluation logs for the boxed answer tail. It also enables `SCORE_TRAJECTORY_CHECK=1`,
which emits `KG1_SCORE_TRAJECTORY_STATUS` so loss is never used as the only
directional signal toward `>=0.89`. Human-friendly `KG1-TEACH` logs stay
enabled for real-time job monitoring.

## Outputs

- Bit specialist train: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_bit_specialist_train.jsonl`
- Equation specialist train: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_equation_specialist_train.jsonl`
- Protected replay train: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_protected_replay_train.jsonl`
- Micro consolidation train: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_micro_consolidation_train.jsonl`
- Val170 holdout: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_val170.jsonl`
- HF env preview: `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1243_solver_to_lora_graft\v1243_hf_env_preview.json`

## Trainer Contract

- The first validation run is tokenize-only and cannot upload to HF.
- Dataset hashes and minimum row counts are pinned in the env preview.
- The full 086 adapter is loaded with `LORA_TARGET_MODULES=down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj`.
- The graft delta is restricted with `TRAINABLE_LORA_MODULES=down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj`.
- Row-level sampling weights must be preserved through tokenization before weighted replacement sampling.
- The trainer must boost final boxed-payload tokens before any real GPU run.
- Run `python scripts/kg1_v1243_graft_trainer_contract_gate.py` after regeneration.

## Mandatory Gate Commands

Run these only after real generation CSVs exist. They require raw_output columns.
The canonical full947 solution CSV is `C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v284_official_gate_worktree\artifacts\v1088_unicode_dataset_contract_audit\hf_cli_download\runtime_artifacts\v276_full_eval_bridge\v276-full947-bridge-20260511T1245Z\official_train_seed42_stratified10_val.csv`
with expected sha256 `84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935`.
The builder hard-fails if V1243 train or val170 prompts overlap this full947 judge.
Before comparing any candidate, generate two 086/V291 raw_output CSVs:
one with the natural prompt and one with the same answer-only
prompt family used by the candidate. The natural probe is diagnostic; the answer-only probe
must pass strict-clean identity before any paid candidate train or full947 comparison.
The objective remains `>=0.89`: full947_089 requires `843/947`, i.e. `+20` over the 086 baseline.

```powershell
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_full947_baseline_readiness_gate.py' --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v284_official_gate_worktree\artifacts\v1088_unicode_dataset_contract_audit\hf_cli_download\runtime_artifacts\v276_full_eval_bridge\v276-full947-bridge-20260511T1245Z\official_train_seed42_stratified10_val.csv' --natural-baseline-predictions 'path\to\086_natural_full947_raw_output.csv' --answer-only-baseline-predictions 'path\to\086_answer_only_full947_raw_output.csv'
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_bit_equation_transfer_gate.py' --baseline-identity-probe --profile full947_089 --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v284_official_gate_worktree\artifacts\v1088_unicode_dataset_contract_audit\hf_cli_download\runtime_artifacts\v276_full_eval_bridge\v276-full947-bridge-20260511T1245Z\official_train_seed42_stratified10_val.csv' --baseline-predictions 'path\to\086_natural_full947_raw_output.csv'
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_bit_equation_transfer_gate.py' --baseline-identity-probe --profile full947_089 --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v284_official_gate_worktree\artifacts\v1088_unicode_dataset_contract_audit\hf_cli_download\runtime_artifacts\v276_full_eval_bridge\v276-full947-bridge-20260511T1245Z\official_train_seed42_stratified10_val.csv' --baseline-predictions 'path\to\086_answer_only_full947_raw_output.csv'
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_bit_equation_transfer_gate.py' --profile tiny --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1241_bit_equation_real_transfer_gate\v1241_tiny_bit_equation_probe_solution.csv' --baseline-predictions 'path\to\baseline_predictions_with_raw_output.csv' --candidate-predictions 'path\to\candidate_predictions_with_raw_output.csv'
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_bit_equation_transfer_gate.py' --profile val170 --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v1241_bit_equation_real_transfer_gate\v1241_v1240_val170_solution.csv' --baseline-predictions 'path\to\baseline_predictions_with_raw_output.csv' --candidate-predictions 'path\to\candidate_predictions_with_raw_output.csv'
python 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\scripts\kg1_v1241_bit_equation_transfer_gate.py' --profile full947_089 --solution-csv 'C:\Users\davis\Workspace\KG1 -NVIDIA\artifacts\v1243_final_publish_worktree\artifacts\v284_official_gate_worktree\artifacts\v1088_unicode_dataset_contract_audit\hf_cli_download\runtime_artifacts\v276_full_eval_bridge\v276-full947-bridge-20260511T1245Z\official_train_seed42_stratified10_val.csv' --baseline-predictions 'path\to\086_answer_only_full947_raw_output.csv' --candidate-predictions 'path\to\candidate_predictions_with_raw_output.csv'
```

## Do Not Do

- Do not use this artifact as score proof.
- Do not launch paid GPU until dry-run/tokenization/objective gates pass.
- Do not submit unless V1241 full947 passes with raw outputs.
- Do not train on FASE5/s140/fase5_mix.

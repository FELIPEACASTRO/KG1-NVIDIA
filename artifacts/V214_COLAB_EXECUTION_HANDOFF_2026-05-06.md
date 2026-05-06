# V214 Colab Execution Handoff - 2026-05-06

## Notebook

Local path:

- `notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

Colab URL after push:

- `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

This URL uses the published branch `v214-h100-micro-replay`.

## Purpose

Run the next V214 gate:

1. Mount Google Drive.
2. Bootstrap V214 data/scripts into `/content/kg1`.
3. Audit dependency versions and fresh subprocess imports.
4. Check H100/high-RAM sizing before model load.
5. Audit V214 dataset hashes and row counts.
6. Audit the protected V194 adapter in Drive.
7. Build weak/full/strong validation CSVs from the protected V194 validation file.
8. Run trainability dry-run.
9. Optionally run one-step V194 continuation.
10. Evaluate weak first.
11. Evaluate full only if weak improves over V194.

The notebook never packages and never submits to Kaggle.

## Required Drive Inputs

- `/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter`
- `/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`

The adapter directory must contain:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`

## Default Mode

By default:

- `KG1_V214_RUN_DRY_RUN=1`
- `KG1_V214_RUN_TRAIN=0`
- `KG1_V214_RUN_EVAL=1`

This means the notebook will run audits and dry-run checks, but will not train
unless explicitly enabled.

## To Enable Training

Before running the training cell in Colab, set:

```python
import os
os.environ["KG1_V214_RUN_TRAIN"] = "1"
```

Training design:

- starts from V194 adapter;
- `INIT_ADAPTER_LOAD_MODE=peft`;
- LR `3e-7`;
- `MAX_STEPS=1`;
- `MAX_LENGTH=4096`;
- batch size `4`, micro batch `1`;
- trainable LoRA filter: `q_proj,k_proj,v_proj,o_proj,in_proj,out_proj`;
- no Hugging Face upload;
- no Kaggle submit.

## Runtime Size Gate

The notebook blocks model load if the runtime is too small:

- GPU total memory must be at least `70 GiB`;
- system RAM total must be at least `45 GiB`;
- system RAM available must be at least `20 GiB`;
- `/content` free disk must be at least `80 GiB`.

An H100 name is expected and logged. A non-H100 GPU can proceed only if the
memory gate passes, but it prints a warning because the intended runtime is
H100 high-RAM.

## Anti-Stall Logs

The command wrapper prints:

- command start/end;
- working directory;
- full command line;
- log path;
- return code;
- elapsed seconds.

During silent long-running commands it emits a `[V214 heartbeat]` every 60
seconds with:

- elapsed seconds;
- seconds since last command output;
- system RAM total/available;
- `/content` disk free/total;
- `nvidia-smi` GPU name, memory used/total, and utilization.

## Output Paths

Root:

- `/content/drive/MyDrive/KG1_NVIDIA_V214/output_v214_micro_replay`

Dry-run:

- `dry_run_v214_v194_cont_lr3e7_s1/dry_run.log`
- `dry_run_v214_v194_cont_lr3e7_s1/dry_run_model_recipe_report.json`

Training:

- `train_v214_v194_cont_lr3e7_s1/train.log`
- `train_v214_v194_cont_lr3e7_s1/final_adapter`
- `train_v214_v194_cont_lr3e7_s1/final_adapter/v90_training_manifest.json`

Eval:

- `eval_v214_v194_cont_lr3e7_s1/weak_eval/v214_micro_weak_eval_report.json`
- `eval_v214_v194_cont_lr3e7_s1/weak_eval/v214_micro_weak_per_task.csv`
- `eval_v214_v194_cont_lr3e7_s1/full_eval/v214_micro_full_eval_report.json`
- `eval_v214_v194_cont_lr3e7_s1/full_eval/v214_micro_full_per_task.csv`
- `v214_colab_run_manifest.json`

## Gates

Weak gate to run full eval:

- weak `>=191/315`;
- weak truncation `<=3`.

Strict candidate after human review:

- full `>=828/947`;
- weak `>=198/315`;
- strong `632/632`;
- full truncation within review threshold.

Preferred candidate:

- full `>=830/947`;
- weak `>=198/315`;
- strong `632/632`.

Any candidate below these thresholds is diagnostic-only.

## Current Local Dataset State

- `data/v214/v214_micro_replay_candidate.jsonl`: `880` rows.
- `data/v214/v214_micro_train.jsonl`: `792` rows.
- `data/v214/v214_micro_val.jsonl`: `88` rows.
- train/val overlap: `0`.
- V194 validation overlap: `0`.
- all candidate rows are verified and single-boxed.

## Human Approval Boundary

Human approval is required for:

- launching paid/limited H100 compute;
- setting `KG1_V214_RUN_TRAIN=1`;
- any Kaggle submission.

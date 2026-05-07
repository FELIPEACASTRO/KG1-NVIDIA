# V214 Pre-Execution Audit - 2026-05-06

## Verdict

The V214 Colab is ready for the first Colab execution gate.

Run first in default mode (`KG1_V214_RUN_TRAIN=0`) to execute audits and
dry-run checks. Enable training only after those pass.

## Published Notebook

- URL: `https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`
- Branch: `v214-h100-micro-replay`
- Validated commit: `438b3a158720b55d98f904a85343bfd1ed8da3b0`
- GitHub API status: `200`
- Notebook blob size: `2883146`
- Notebook blob sha: `099d4750af2b4cd8fe5320a39594d5169d8ed1db`
- Commit-pinned raw sha256: `6d20d21595b021293c3b1865c93b2a34a3b62bd01450069fcb09f557db22cdb2`

Static checks:

- notebook JSON parses;
- all code cells parse via Python AST;
- all operational code cells contain V214 progress markers;
- notebook contains PEFT minimum-version check;
- notebook contains fresh subprocess import check;
- notebook contains H100/high-RAM sizing gate;
- notebook contains 60-second heartbeat resource logs.
- notebook contains GPU-first model placement controls;
- notebook contains TF32/matmul precision controls;
- notebook contains optional `hf_transfer` and `bitsandbytes` setup.

## Dataset Checks

Candidate:

- `data/v214/v214_micro_replay_candidate.jsonl`
- rows: `880`
- sha256: `6bd1f4727345fce59c9697cfc5ea84bdaae821fedd27a786edc7418e98e223e4`
- family mix:
  - `bit_manipulation`: `160`
  - `equation_transform`: `120`
  - `gravity_constant`: `150`
  - `numeral_system`: `150`
  - `text_encryption`: `150`
  - `unit_conversion`: `150`
- verified rows: `880/880`
- single-boxed rows: `880/880`
- unique prompt+answer signatures: `880/880`

Internal training split:

- train: `data/v214/v214_micro_train.jsonl`
- train rows: `792`
- train sha256: `da601695ca59e6e981638a1105c4b9750d077fb9d15717bb0474d95a85e552a7`
- val: `data/v214/v214_micro_val.jsonl`
- val rows: `88`
- val sha256: `ace511e400542241f3ed6bdba35c5b5d4852c72410d2b1504c88330e89482183`
- train/val overlap: `0`
- train verified: `792/792`
- val verified: `88/88`

## Embedded Notebook Files

Remote notebook embeds the required files with expected hashes:

- `data/v214/v214_micro_train.jsonl`: `da601695ca59e6e981638a1105c4b9750d077fb9d15717bb0474d95a85e552a7`
- `data/v214/v214_micro_val.jsonl`: `ace511e400542241f3ed6bdba35c5b5d4852c72410d2b1504c88330e89482183`
- `scripts/hf_job_train_v90.py`: `2af64e812a7c1f56232ba5841bc9c2a78f18a069853c2d444f8ad22c6c57c90c`
- `scripts/evaluate_lora_adapter.py`: `b56764960629cf4c43c7dab09e8f2dc8b338b284d2ed1d65a325ca80c4463168`
- `src/competition_utils.py`: `b6aa77804681ddf29478815627aad023d9a8e673208369a930d97f13b8e15a13`

## Dependency Strategy

Notebook checks/imports:

- `pandas`
- `huggingface_hub`
- `hf_transfer` when `HF_HUB_ENABLE_HF_TRANSFER=1`
- `transformers`
- `peft>=0.18.1`
- `torch`
- `vllm`
- `bitsandbytes` optional, with fallback to torch Adam if unavailable
- `packaging`

After installing `vllm`, it runs a fresh Python subprocess import check for:

- `torch`
- `transformers`
- `peft`
- `vllm`

This matters because Colab package installs can change dependency versions in a
way that the current kernel does not fully reflect.

## Performance Controls

The notebook/training script uses conservative performance settings:

- `MODEL_DEVICE_MAP=cuda` by default, to avoid slow CPU offload on H100;
- override via `KG1_V214_MODEL_DEVICE_MAP` if diagnostics require `auto`;
- `ATTN_IMPLEMENTATION=eager` by default for compatibility with the existing
  V194/V206 path;
- override via `KG1_V214_ATTN_IMPLEMENTATION` only after dry-run proof;
- TF32 enabled for supported CUDA matmul/CUDNN operations;
- `torch.set_float32_matmul_precision("high")`;
- `HF_HUB_ENABLE_HF_TRANSFER=1`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- `TOKENIZERS_PARALLELISM=false`;
- `bitsandbytes` attempted for `PagedAdam8bit`;
- `MAX_LENGTH=4096`;
- `BATCH_SIZE=4`;
- `MICRO_BATCH_SIZE=1`;
- `ABORT_MAX_RESERVED_GIB=78`.

These settings prioritize fast failure on inadequate runtimes and avoid silent
CPU offload. The solve-rate gates remain unchanged.

## Runtime Size Gate

The notebook blocks model load unless:

- GPU total memory `>=70 GiB`;
- system RAM total `>=45 GiB`;
- system RAM available `>=20 GiB`;
- `/content` free disk `>=55 GiB` after safe cleanup.

The notebook warns, but does not block, when `/content` free disk is below
`65 GiB`.

Safe cleanup before the disk check removes:

- `/content/sample_data`;
- `/root/.cache/pip`;
- `/tmp/pip-*`.

Expected H100 high-RAM verdict:

- H100 80GB should pass the GPU memory gate.
- Colab high-RAM should pass the system RAM gate.
- The observed H100 runtime with about `59.5 GiB` free on `/content` should
  pass after this gate adjustment, while logging a disk-warning.
- A 40GB GPU should fail before model load.
- Standard low-RAM runtime should fail before model load.

If the runtime is not named H100 but has enough memory, it may proceed with a
warning. The intended runtime remains H100 high-RAM.

## Anti-Stall Logging

Every shell command prints:

- command start/end;
- working directory;
- exact command line;
- log path;
- return code;
- elapsed seconds.

If a command is silent for a long time, the wrapper prints `[V214 heartbeat]`
every 60 seconds with:

- elapsed seconds;
- seconds since last output;
- system RAM total/available;
- `/content` disk free/total;
- `nvidia-smi` GPU name, memory used/total, and utilization.

This is designed to distinguish a real stall from a long model download/load.

## Required Google Drive Inputs

The Drive connector confirmed the root folders exist:

- `KG1_NVIDIA_V202D`
- `KG1_NVIDIA_V207A`

Confirmed under `KG1_NVIDIA_V202D`:

- `init_adapter_v194_rank19_build`
- `init_adapter_v194_rank19_build/adapter`
- `init_adapter_v194_rank19_build/adapter/adapter_config.json`
- `init_adapter_v194_rank19_build/adapter/adapter_model.safetensors`
- `submission.zip`

Confirmed under `KG1_NVIDIA_V207A/output_v207a_acc_gate/validation`:

- `official_train_seed42_stratified10_val.csv`
- `official_train_seed42_stratified10_val_weak_families.csv`
- `official_train.csv`

The notebook itself will still perform the definitive Colab-side checks for:

- `/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter/adapter_config.json`
- `/content/drive/MyDrive/KG1_NVIDIA_V202D/init_adapter_v194_rank19_build/adapter/adapter_model.safetensors` or `.bin`
- `/content/drive/MyDrive/KG1_NVIDIA_V207A/output_v207a_acc_gate/validation/official_train_seed42_stratified10_val.csv`

## Execution Decision

Proceed to Colab execution in this order:

1. Open the published notebook.
2. Select H100 high-RAM runtime.
3. Run default mode first; do not set `KG1_V214_RUN_TRAIN=1`.
4. Confirm dependency audit, size gate, dataset audit, Drive audit, and dry-run.
5. Only if all pass, set `KG1_V214_RUN_TRAIN=1` and run the training/eval cells.

No Kaggle submission is allowed from this notebook.

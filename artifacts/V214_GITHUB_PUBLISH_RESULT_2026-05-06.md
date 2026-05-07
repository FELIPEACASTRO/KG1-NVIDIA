# V214 GitHub Publish Result - 2026-05-06

## Published Branch

- Repository: `https://github.com/FELIPEACASTRO/KG1-NVIDIA`
- Branch: `v214-h100-micro-replay`
- Notebook validation commit: `8b238831e748c866aa6e0c64e146bb88a71a185c`
- Note: this publish-result document may appear in a later commit on the same branch.

## Colab URL

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

## GitHub Validation

Validated through GitHub API:

- endpoint: `https://api.github.com/repos/FELIPEACASTRO/KG1-NVIDIA/contents/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb?ref=v214-h100-micro-replay`
- status: `200`
- file: `KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`
- size: `2889519`
- blob sha: `0954c26e8b245dd64eb6d670f24195be35ba83fb`
- commit-pinned raw sha256: `32cfc6e4dab0f207d1bda3fb4b409406416740a76901e1b9c61db79872df447c`
- validated markers:
  - `mamba-ssm[causal-conv1d]`
  - `mamba_ssm_postinstall`
  - `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn`
  - `from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined`
  - `from causal_conv1d import causal_conv1d_fn, causal_conv1d_update`

Validated through `git ls-remote`:

- `8b238831e748c866aa6e0c64e146bb88a71a185c refs/heads/v214-h100-micro-replay`

## Pull Request URL

`https://github.com/FELIPEACASTRO/KG1-NVIDIA/pull/new/v214-h100-micro-replay`

## Operational Status

- Notebook published.
- Notebook includes H100/high-RAM sizing gate before model load.
- Notebook emits `[V214 heartbeat]` resource logs every 60 seconds during silent commands.
- Notebook includes GPU-first placement, TF32/matmul precision controls, optional
  `hf_transfer`, required `mamba-ssm` validation, optional `bitsandbytes`, and
  a standalone published training script.
- Notebook relaxes the Colab H100 disk gate to `55 GiB` minimum after safe cleanup
  and warns below `65 GiB`.
- Submit remains disabled.
- Training remains disabled by default.
- To train in Colab, set `KG1_V214_RUN_TRAIN=1`.
- Human approval is still required for H100 execution and any Kaggle submission.

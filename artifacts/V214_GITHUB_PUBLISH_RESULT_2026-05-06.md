# V214 GitHub Publish Result - 2026-05-06

## Published Branch

- Repository: `https://github.com/FELIPEACASTRO/KG1-NVIDIA`
- Branch: `v214-h100-micro-replay`
- Notebook validation commit: `13d5a9faf148ba78e9a9e5766423f20d71b5749d`
- Note: this publish-result document may appear in a later commit on the same branch.

## Colab URL

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

## GitHub Validation

Validated through GitHub API:

- endpoint: `https://api.github.com/repos/FELIPEACASTRO/KG1-NVIDIA/contents/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb?ref=v214-h100-micro-replay`
- status: `200`
- file: `KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`
- size: `2889047`
- blob sha: `4f53dd42887b109b478fadc39173ec81c7d374b6`
- commit-pinned raw sha256: `fe5ef5347b9c90189d6dc94045fed77d68ad51d1e07248eef7dcdf22c4fde775`
- validated markers:
  - `mamba-ssm[causal-conv1d]`
  - `mamba_ssm_postinstall`
  - `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn`

Validated through `git ls-remote`:

- `13d5a9faf148ba78e9a9e5766423f20d71b5749d refs/heads/v214-h100-micro-replay`

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

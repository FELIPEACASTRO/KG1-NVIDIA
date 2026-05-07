# V214 GitHub Publish Result - 2026-05-06

## Published Branch

- Repository: `https://github.com/FELIPEACASTRO/KG1-NVIDIA`
- Branch: `v214-h100-micro-replay`
- Notebook validation commit: `5987a455ba0c7f69ac31c98e502d675d11e0e072`
- Note: this publish-result document may appear in a later commit on the same branch.

## Colab URL

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

## GitHub Validation

Validated through GitHub API:

- endpoint: `https://api.github.com/repos/FELIPEACASTRO/KG1-NVIDIA/contents/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb?ref=v214-h100-micro-replay`
- status: `200`
- file: `KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`
- size: `2886859`
- blob sha: `e03cde2f0decbff282dd9a28519b7433a4851c16`
- commit-pinned raw sha256: `cb7b6b13506d8bf449e015d417aa1fb5a8a19462b02904d823e6696fd2e08daf`

Validated through `git ls-remote`:

- `5987a455ba0c7f69ac31c98e502d675d11e0e072 refs/heads/v214-h100-micro-replay`

## Pull Request URL

`https://github.com/FELIPEACASTRO/KG1-NVIDIA/pull/new/v214-h100-micro-replay`

## Operational Status

- Notebook published.
- Notebook includes H100/high-RAM sizing gate before model load.
- Notebook emits `[V214 heartbeat]` resource logs every 60 seconds during silent commands.
- Notebook includes GPU-first placement, TF32/matmul precision controls, optional
  `hf_transfer`, optional `bitsandbytes`, and a standalone published training script.
- Notebook relaxes the Colab H100 disk gate to `55 GiB` minimum after safe cleanup
  and warns below `65 GiB`.
- Submit remains disabled.
- Training remains disabled by default.
- To train in Colab, set `KG1_V214_RUN_TRAIN=1`.
- Human approval is still required for H100 execution and any Kaggle submission.

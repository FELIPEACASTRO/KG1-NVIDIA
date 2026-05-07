# V214 GitHub Publish Result - 2026-05-06

## Published Branch

- Repository: `https://github.com/FELIPEACASTRO/KG1-NVIDIA`
- Branch: `v214-h100-micro-replay`
- Notebook validation commit: `2bcceda55e83f7ca8774d4faea2a60ad0d6a5c3e`
- Note: this publish-result document may appear in a later commit on the same branch.

## Colab URL

`https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/v214-h100-micro-replay/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`

## GitHub Validation

Validated through GitHub API:

- endpoint: `https://api.github.com/repos/FELIPEACASTRO/KG1-NVIDIA/contents/notebooks/KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb?ref=v214-h100-micro-replay`
- status: `200`
- file: `KG1_V214_H100_MICRO_REPLAY_COLAB.ipynb`
- size: `2898460`
- blob sha: `cc7274842e694028e2d6a2b06f8d30060a69efb3`
- commit-pinned raw sha256: `261b3d7c41c180ef569aa571dd5e173a2fd6b213f928dc5df749247aa13df581`
- validated markers:
  - `mamba-ssm[causal-conv1d]`
  - `mamba_ssm_postinstall`
  - `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn`
  - `from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined`
  - `from causal_conv1d import causal_conv1d_fn, causal_conv1d_update`
  - `AGGRESSIVE_DISK_CLEANUP`
  - `USE_BITSANDBYTES`
  - `Optimizer: torch Adam (USE_BITSANDBYTES=0)`
  - `projected_content_free_after_model_cache_gib`

Validated through `git ls-remote`:

- `2bcceda55e83f7ca8774d4faea2a60ad0d6a5c3e refs/heads/v214-h100-micro-replay`

## Pull Request URL

`https://github.com/FELIPEACASTRO/KG1-NVIDIA/pull/new/v214-h100-micro-replay`

## Operational Status

- Notebook published.
- Notebook includes H100/high-RAM sizing gate before model load.
- Notebook emits `[V214 heartbeat]` resource logs every 60 seconds during silent commands.
- Notebook includes GPU-first placement, TF32/matmul precision controls, optional
  `hf_transfer`, required `mamba-ssm` validation, default-disabled
  `bitsandbytes`, and a standalone published training script.
- Notebook requires `/content >=60 GiB` free after cleanup, projects a `42 GiB`
  model cache, and requires at least `15 GiB` projected post-cache free space.
- Notebook warns below `/content <70 GiB`.
- Notebook logs top disk users before/after cleanup and removes partial Nemotron
  HF cache plus large unused preinstalled packages in the temporary Colab
  runtime.
- Submit remains disabled.
- Training remains disabled by default.
- To train in Colab, set `KG1_V214_RUN_TRAIN=1`.
- Human approval is still required for H100 execution and any Kaggle submission.

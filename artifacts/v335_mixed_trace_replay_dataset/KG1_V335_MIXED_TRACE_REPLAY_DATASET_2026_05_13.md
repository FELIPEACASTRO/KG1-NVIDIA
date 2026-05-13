# KG1 V335 Mixed Trace Replay Dataset - 2026-05-13

## Objective

Build a guarded SFT dataset that combines the strongest verified trace sources without leaking weak/full gate rows:

- V304 broad solver-trace replay for bit protection and general family coverage.
- V325 V324 numeric equation no-loss traces.
- V330 V329 symbolic/cryptarithm no-loss traces.

The goal is not to authorize a submit. The goal is to authorize a small HF smoke only if local gates prove the dataset is clean, tokenizable, and aligned with the current weak bottleneck.

## Artifacts

- Builder: `scripts/build_v335_mixed_trace_replay_dataset.py`
- Manifest: `artifacts/v335_mixed_trace_replay_dataset/20260513T_cpu_gate/v335_mixed_trace_replay_manifest.json`
- Train: `artifacts/v335_mixed_trace_replay_dataset/20260513T_cpu_gate/v335_mixed_trace_replay_train.jsonl`
- Validation: `artifacts/v335_mixed_trace_replay_dataset/20260513T_cpu_gate/v335_mixed_trace_replay_val.jsonl`
- Real tokenizer gate: `artifacts/v335_mixed_trace_replay_dataset/20260513T_cpu_gate/tokenization_gate_real/v286_generic_tokenization_gate_manifest.json`

## Dataset Contract

- Train rows: `13542`
- Validation rows: `1149`
- Train SHA256: `fed84002b6f9104869c743cce816a81e279400c8031ac3545846871fecc50654`
- Validation SHA256: `1af6a221d3539294163cd684ded1a0de49d3631d2357d8a8aa0f560de1f1866d`
- Train family counts:
  - `bit_manipulation=4231`
  - `equation_transform=8735`
  - `gravity_constant=144`
  - `numeral_system=144`
  - `text_encryption=144`
  - `unit_conversion=144`
- Validation family counts:
  - `bit_manipulation=332`
  - `equation_transform=753`
  - `gravity_constant=16`
  - `numeral_system=16`
  - `text_encryption=16`
  - `unit_conversion=16`

## Gates Passed

- Python compile passed for `scripts/build_v335_mixed_trace_replay_dataset.py`.
- Anti-leakage gate: `0` id overlap and `0` prompt overlap with V221 weak reference and V291 full reference.
- Duplicate gate: `0` duplicate ids and `0` duplicate normalized prompt hashes inside train/validation merges.
- Format normalization: all `13542` train rows and `1149` validation rows were normalized to `Final answer: \boxed{answer}` suffix.
- Real Nemotron tokenizer gate passed with `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Tokenization: `0` prompt truncation, `0` completion tokens dropped, offset masks present for all rows.
- Real tokenizer max length observed:
  - train max tokens: `749`
  - validation max tokens: `748`

## Skill-Based Review

- Senior data scientist: the dataset is only a training hypothesis. The measurable success criterion remains weak gate improvement over `192/315`, not lower train loss.
- Senior data engineer: data contracts are explicit via row counts, hashes, duplicate checks, source counts, family counts, and anti-leakage references.
- Senior ML engineer: a paid HF job is only justified as a short smoke with first-checkpoint kill-switch. No full eval or package is allowed from this dataset alone.
- PyTorch/Lightning lens: sequence lengths are safe for the planned LoRA path; the critical runtime risks are environment, adapter compatibility, checkpoint cadence, and evaluation at the first checkpoint.
- QA lens: current artifacts pass local syntax and tokenization gates; notebook gate is not applicable because no `.ipynb` was changed.

## Decision

V335 is approved for a short HF smoke after the dataset upload, launcher debug, commit, and push gates.

HF upload and launcher debug status:

- Dataset upload commit: `ec42d6afb1d6a6b1f8243e7ea776fa3d496a8e9f`
- Tokenization gate upload commit: `2b8bb3b6edf1dc2d3a9cfc445a1d06a9c201a97e`
- Launcher: `artifacts/v335_hf_nemo_a100_mixed_trace_replay_launch/launch_v335_hf_nemo_a100_mixed_trace_replay.py`
- Local debug: passed
- Selected HF flavor: `a100-large`
- Hardware: `Nvidia A100 - large`, `80 GB`, `142 GB RAM`
- Unit cost: `0.041667 USD/min`, below the `0.05 USD/min` gate
- Init adapter check: `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6` has required PEFT files

Required kill-switch for continuing beyond the first checkpoint:

- `total > 192`
- `equation_transform > 56`
- `bit_manipulation >= 136`
- no tokenization/truncation regression
- no adapter/config mismatch

If these conditions fail, cancel the HF job and do not spend additional GPU time.

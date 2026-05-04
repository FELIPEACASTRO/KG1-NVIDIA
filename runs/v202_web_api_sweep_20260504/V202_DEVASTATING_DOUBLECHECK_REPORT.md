# V202 Devastating Doublecheck Report

Generated: 2026-05-04

## Verdict

Do not submit V201A, V201B, or V201C outputs. All recent micro-continuation attempts regressed against their own baselines, including the 1-step ultralow candidates.

The next safe route is V202 audit-first, then a separate V202B long-context training notebook only if the audit passes.

## New High-Signal Source

The strongest source found in the second sweep is the official public repository:

https://github.com/tonghuikang/nemotron

Key audited recipe facts:

- `corpus.py` sets `TOKEN_LIMIT = 8192`.
- `train_sft.py` defaults to `max_length = 8192`.
- `train_sft.py` uses `lora_rank = 32`.
- `train_sft.py` uses `batch_size = 64`, `num_epochs = 1`, `micro_batch_size = 16`.
- The recipe tracks token-level loss/logprob and masked/unmasked training tokens.

This confirms that the V199/V201 2048-token continuation route is misaligned with the public 0.87-style route.

## Corrections Applied

- Hardened V202 notebook ZIP extraction against path traversal.
- Added required SHA256 checks for `corpus.jsonl`, `problems.jsonl`, `generation.jsonl`, and `nemotron_traj.csv`.
- Replaced submit guard `assert` with explicit `RuntimeError`.
- Added `REQUIRED_TRAINING_CONTRACT` for 8192-token future training.
- Made the final cell fail clearly if `RUN_TRAINING=True` is set without a real training cell.
- Removed transient `/root/.kaggle/kaggle.json` at notebook stop.
- Updated `hf_job_train_v90.py` to accept both `key=value` and `key:value` weight maps.
- Updated `kg1_sft_format_validator.py` so malformed JSONL rows count as `parse_error`.
- Updated `kg1_sft_format_validator.py` to use the last assistant message in multi-turn rows.
- Updated `kg1_submission_gate.py` to reject ambiguous recursive adapter ZIPs with multiple configs or safetensors.

## Hard Rejects

- Any V201C final adapter.
- Any adapter that regresses eval versus its own baseline.
- Any 2048-token training on the Tong/Huikang long-context corpus without distillation.
- Any public adapter copied without local vLLM gate versus V194.
- Any submit from the V202 audit notebook.

## Next Execution

Run:

https://colab.research.google.com/github/FELIPEACASTRO/KG1-NVIDIA/blob/claude/competent-shamir/notebooks/KG1_V202_H100_A100_LONG_CONTEXT_EVAL_GATE_COLAB_PRO.ipynb

Expected behavior:

- It downloads only selected required files.
- It verifies hashes.
- It writes reports under `/content/drive/MyDrive/KG1_NVIDIA_V202/reports`.
- It stops by design.
- It does not train.
- It does not submit.

## Roadmap

1. Run V202 audit notebook on H100 or A100 80GB.
2. Review `v202_download_manifest.json`, `v202_data_audit.json`, and `v202_plan_manifest.json`.
3. Build V202B only after audit passes.
4. V202B must enforce `max_length=8192`, rank <=32, and the vLLM gate settings:
   - `max_lora_rank=32`
   - `max_model_len=8192`
   - `max_tokens=7680`
   - `temperature=0.0`
   - `top_p=1.0`
5. Promote only if V202B beats V194 and has zero anchor/category regression.

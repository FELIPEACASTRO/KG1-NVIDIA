# V202 Triplecheck Final Report

Generated: 2026-05-04

## Verdict

Do not submit V201A, V201B, or V201C. All recent micro-continuation attempts regressed on the local no-regression gate, including one-step ultralow runs.

The next safe execution is the V202 audit-only notebook. It must not train and must not submit. Its job is to verify the long-context data route before a separate V202B 8192-token training notebook is built.

## Corrections Applied In This Triplecheck

- V202 notebook now reads Kaggle credentials from Colab Secrets via `google.colab.userdata`, environment variables, or Drive `kaggle.json`.
- V202 notebook now registers `atexit` cleanup for `/root/.kaggle/kaggle.json` and also cleans it in the final stop cell.
- V202 notebook now has deterministic cell IDs and passes `nbformat.validate`.
- `hf_job_train_v90.py` weight maps now accept `key:value` as a compatibility fallback but reject empty keys, duplicate keys, non-finite values, zero, and negative weights.
- `kg1_colab_ipynb_execution_gate.py` now detects more accidental submit paths, including `competition_submit`, `KaggleApi()`, and subprocess-based Kaggle submit commands.
- `kg1_training_data_gate.py` now writes a structured report when a reference CSV is missing instead of crashing with an unstructured traceback.

## Verification Results

- `python -m py_compile` passed for the V202 builder and all critical training/gate scripts.
- V202 notebook JSON parsed successfully.
- All V202 code cells compile.
- `nbformat.validate` passed.
- Notebook has 11 cells, 9 code cells, and no missing cell IDs.
- Published raw V202 notebook SHA256 after fixes: `77f0cde6ca44b10565c29cab1e747c51903a40617d88d3583799105df4380b70`.
- V202 notebook contains no `kaggle competitions submit` or `competition_submit` command.
- `RUN_TRAINING=False`, `RUN_VLLM_GATE=False`, and `ALLOW_KAGGLE_SUBMIT=False` remain the defaults.

## API And Web Findings

High-signal route:

- `tonghuikang/nemotron` remains the strongest public recipe signal. It points to `TOKEN_LIMIT=8192`, `max_length=8192`, LoRA rank 32, and long-context SFT rather than 2048-token micro-continuation.

Fresh HF checks:

- `andy279/nemotron-reasoning-challenge`: gated dataset, 4 files, includes `sft_train.jsonl` and `sft_val.jsonl`.
- `andy279/nemotron-reasoning-challenge-raw-traces`: gated dataset, 10 files, includes solver/trace JSONL files.
- `prometheus04/nvidia-kaggle`: public dataset with SFT scripts, a small adapter, and synthetic data. Useful for mining patterns only; base/adapter contract must be gated before any reuse.
- `GaryNENE/nemotron-nano-8b-reasoning-lora`: public model repo with scripts only, no adapter weights found in the tree.

Kaggle search additions:

- `kishanvavdara/nemotron-reasoning-traj` appeared as a reasoning trajectory dataset. Prior local analysis says trajectory-style data must be filtered for correctness before SFT.
- Recent public notebooks from `croftadams/*` appeared in search, but no evidence yet that they beat V194. Treat as mining material only.

## Blocking Issues Still Enforced

- Existing `kg1_colab_ipynb_execution_gate.py`, `kg1_v198_final_submit_doublecheck.py`, and `kg1_v199_posttrain_gate.py` are V198/V199-specific. They must not be treated as a V202 submit gate.
- A V202-specific posttrain and final-submit gate must be created before any V202B candidate can be submitted.
- Metric direction is lower-is-better for eval loss. A candidate passes only if it is less than or equal to V194 on the same harness and does not lose category anchors.
- Public adapters must not be copied into submission. They can only inform topology/data choices after structural and vLLM gates.

## Roadmap

1. Run the V202 audit-only notebook on H100 or A100 80GB High-RAM.
2. Confirm SHA and parse health for `corpus.jsonl`, `problems.jsonl`, `generation.jsonl`, and `nemotron_traj.csv`.
3. Build V202B only after the audit passes: `max_length=8192`, `max_model_len=8192`, `max_tokens=7680`, `max_lora_rank=32`, `temperature=0`, `top_p=1`.
4. Use V194 as immutable floor. No submit unless local vLLM/harness beats or ties V194 globally and shows net gains by category.
5. Mix data conservatively: V194 rehearsal plus verified long-context Tong/Huikang-style data, small bit 3-input boost, and solver-verified cryptarithm/equation examples.
6. Exclude unverified false/partial/public synthetic rows from direct SFT.
7. Create V202-specific posttrain and submission gates before packaging.
8. Submit only after explicit authorization.

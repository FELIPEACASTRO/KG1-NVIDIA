# KG1 V712 A100 Equation Signal Plan

Updated: 2026-05-20 23:10 UTC

This plan has been converted into a bounded A100 execution. The V712 training
phase completed; no H200 was launched. The job failed in weak eval before ACC
because of a remote Python package/import issue.

Job:
`https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06`

Launcher:
`artifacts/v712_hf_a100_equation_signal_launch/launch_v712_hf_a100_equation_signal.py`

## V712 Result

Training completed:

- Baseline eval loss: `2.6376`.
- `checkpoint-10` eval loss: `2.6363` and best.
- `checkpoint-20` / final eval loss: `2.6373`.
- Train tokenized rows: `852/852`.
- Validation tokenized rows: `195/195`.
- Truncation: `0`.
- Fallback masks: `0`.
- Effective trainable LoRA params: `1,867,776`, only `q_proj,v_proj`.
- MoE target parameters: frozen, `0` trainable params.

Weak eval did not run to ACC:

- Failure:
  `ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'`.
- Location: `scripts/evaluate_lora_adapters_batch.py` import path.
- Root cause: `scripts/` was a namespace package without `__init__.py`; remote
  images can contain unrelated third-party packages named `scripts`.

Fix applied:

- Added `scripts/__init__.py`.
- Launcher now compiles `scripts/evaluate_lora_adapter.py` and
  `scripts/evaluate_lora_adapters_batch.py`.
- Launcher now runs `scripts_package_gate` and `weak_eval_import_gate_ok`.
- Static and pre-paid gates now block missing weak-eval import preflight.
- Validation after fix:
  - `v712_static_safety_gate_after_importfix.json`: `ok=true`.
  - `v712_pre_paid_job_integration_gate_after_importfix.json`: `ok=true`.

Operational decision: do not rerun V712 training. The next paid action, after
the import fix is available remotely, must be weak-eval-only on the already
uploaded checkpoints, starting with `checkpoint-10` because it had the best
loss.

## Why V712 Exists

V710 proved V708 checkpoint-5 is not promotable:

- total `191/315`
- bit `135/160`
- equation `56/155`
- truncation `1`
- protected backfire present

V711 resolved the main false suspicion:

- broad `adapter_config.target_modules` is active inherited adapter surface;
- effective trainable surface was `q_proj,v_proj`;
- MoE target parameters were frozen;
- `lm_head` was absent;
- trainability gate passed.

The remaining likely training-side problem is weak signal:

- V708 used only `5` steps;
- batch size `2`;
- LR `5e-7 -> 1e-7`;
- final eval loss regressed by `+0.0018`;
- equation weak ACC stayed `56/155`.

## Proposed V712 Direction

Do not change the adapter contract. Increase signal while keeping strict gates.

Candidate settings to prepare, not launch yet:

- GPU: A100 only, preferably `a100-large`.
- Image: keep CUDA12/A100-safe image unless a local runtime gate approves a
  better one.
- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Initial adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- Dataset: same V708 equation-only dataset.
- Train rows: `852`.
- Validation rows: `195`.
- LoRA active target modules:
  `down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj`.
- Effective trainable modules: `q_proj,v_proj` only.
- MoE target parameters: loaded but frozen.
- `lm_head`: excluded.
- `r=32`, `alpha=32`, dropout `0.0`.
- `max_length=1024`.
- `loss_normalization_mode=example_mean`.
- row-loss weight enabled.

Training signal actually launched:

- `max_steps=20`.
- `save_every_steps=10`.
- `eval_every_steps=10`.
- `batch_size=2`.
- `micro_batch_size=1`.
- LR: `2e-6 -> 5e-7`.

The original draft `50` steps was rejected by the existing pre-paid drift gate:
`deferred_post_checkpoint` is allowed only up to `MAX_STEPS<=20` with
checkpoint/eval intervals `<=10`. We kept the gate intact instead of weakening
it for a larger paid run.

Weak eval:

- Keep official-like decode:
  `disable_thinking=0`, `require_disable_thinking=0`, `max_tokens=7680`,
  `max_model_len=8192`, `max_num_seqs=64`.
- Evaluate a selected checkpoint only; do not package/full eval/submit.
- Required checkpoint: `checkpoint-20`.

## Mandatory Pre-Paid Gates

All passed before launch:

- `hf jobs ps` showed no active jobs before launch.
- Static safety gate:
  `artifacts/v712_hf_a100_equation_signal_launch/v712_static_safety_gate.json`,
  `ok=true`, `findings=[]`.
- Pre-paid integration gate:
  `artifacts/v712_hf_a100_equation_signal_launch/v712_pre_paid_job_integration_gate.json`,
  `ok=true`, `findings=[]`.
- Runtime image gate: A100-only, no H200.
- Dataset hash and row-count gate:
  - train SHA256 `a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d`;
  - validation SHA256 `f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba`.
- Tokenization/mask gate:
  - zero truncation;
  - zero fallback masks;
  - zero completion-token drops;
  - response-token cross-entropy mask only.
- LoRA trainability gate:
  - `q_proj,v_proj` trainable;
  - MoE target parameters frozen;
  - `lm_head` absent;
  - modules_to_save empty/null.
- Remote command must execute weak eval after training, not just compile it.
- Weak promotion thresholds unchanged:
  - total `>=196`;
  - bit `>=136`;
  - equation `>=60`;
  - truncation `0`;
  - boxed rate `1.0`;
  - no-box fallback `0`;
  - protected backfire `0`;
  - label-aware delta `0`.

## Stop Conditions

Do not continue to another paid run if V712:

- regresses protected rows;
- stays at equation `56/155`;
- produces truncation/no-box rows;
- only improves label-aware debug;
- improves one row but remains below thresholds;
- has final eval loss substantially worse without weak ACC gain.

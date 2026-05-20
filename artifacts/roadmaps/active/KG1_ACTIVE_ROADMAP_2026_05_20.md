# KG1 Active Roadmap

Updated: 2026-05-20 23:10 UTC

This file is the current executable source of truth. Older roadmap sections are
archive only unless this file references them explicitly.

## Current Decision

Do not package, run full eval, submit to Kaggle, or launch another paid GPU job.

V710 completed the required weak-eval-only measurement for V708 `checkpoint-5`
and failed the promotion gate. V712 completed the A100 training phase but failed
before weak ACC because the weak-eval batch harness could not import
`scripts.evaluate_lora_adapter` in the remote container. This is a packaging
bug, not an ACC result. No H200 is allowed.

Current next action: do not rerun training. Fix/push the import-package guard,
then run the cheapest possible A100 weak-eval-only job against the already
uploaded V712 checkpoints, starting with `checkpoint-10` because it had the best
eval loss. Only consider `checkpoint-20` after checkpoint-10 is measured.

V710 job:
`https://huggingface.co/jobs/felipesp1983/6a0e25bc3fef25139d6c9fd0`

Failure ledger:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/KG1_V710_FAILURE_ANALYSIS.md`

V711 parameter audit:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/KG1_V711_PARAMETER_AUDIT.md`

V711 OpenRouter/Hugging Face decision:
`artifacts/openrouter/v711_v710_failure_consult/KG1_V711_CONSENSUS_AND_LOCAL_DECISION.md`

V712 no-GPU plan:
`artifacts/v712_a100_equation_signal_plan/KG1_V712_A100_EQUATION_SIGNAL_PLAN.md`

V712 launcher:
`artifacts/v712_hf_a100_equation_signal_launch/launch_v712_hf_a100_equation_signal.py`

V712 failed job:
`https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06`

V712 launch manifest:
`artifacts/v712_hf_a100_equation_signal_launch/v712-a100-equation-signal-v290ckpt6-20260520T222638Z_launch_manifest.json`

V712 static gate:
`artifacts/v712_hf_a100_equation_signal_launch/v712_static_safety_gate.json`

V712 pre-paid integration gate:
`artifacts/v712_hf_a100_equation_signal_launch/v712_pre_paid_job_integration_gate.json`

V712 import-fix gates:
`artifacts/v712_hf_a100_equation_signal_launch/v712_static_safety_gate_after_importfix.json`
and
`artifacts/v712_hf_a100_equation_signal_launch/v712_pre_paid_job_integration_gate_after_importfix.json`

V713 post-failure OpenRouter/HF consult:
`artifacts/openrouter/v713_v712_import_failure_consult/KG1_V713_CONSENSUS_AND_LOCAL_DECISION.md`

## Confirmed V712 Outcome

Training completed on A100. Weak ACC was not produced because the job failed in
the weak-eval stage:

- HF job status: `ERROR`.
- Failure: `ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'`.
- Root cause: remote namespace/package collision risk because repo `scripts/`
  had no `__init__.py`, while `evaluate_lora_adapters_batch.py` imports
  `scripts.evaluate_lora_adapter`.
- Fix applied locally:
  - added `scripts/__init__.py`;
  - added the same file in the official gate worktree;
  - launcher now compiles `scripts/evaluate_lora_adapter.py`;
  - launcher now runs `scripts_package_gate` and `weak_eval_import_gate_ok`
    before any train/eval command;
  - static and pre-paid gates now block missing weak-eval import preflight.

V712 training metrics:

- Baseline validation loss: `2.6376`.
- `checkpoint-10` validation loss: `2.6363` and marked best.
- `checkpoint-20` validation loss: `2.6373`.
- Final validation loss: `2.6373`.
- Train raw/tokenized rows: `852/852`.
- Validation raw/tokenized rows: `195/195`.
- Train and validation truncation: `0`.
- Fallback masks: `0`.
- Offset masks: train `852`, validation `195`.
- Effective trainable LoRA params: `1,867,776`.
- Frozen LoRA params: `882,006,016`.
- Trainable surface: `q_proj,v_proj`.
- MoE target parameter trainability: frozen, `0` trainable params.

Decision: V712 is not promotable yet because no weak ACC/backfire result exists.
The only clean continuation is weak-eval-only reuse of uploaded checkpoints after
the import fix is available to the remote job.

V713 external consult accepted the same decision: push/deploy the import fix and
run weak-eval-only on `checkpoint-10` first. Rejected for promotion: changing
`KG1_MAX_TOKENS` to `1024`, removing avg-completion-token guard, adding a new
training run, or weakening any threshold before the missing weak ACC artifact.

## Confirmed V710 Outcome

Adapter evaluated:
`felipesp1983/kg1-nemotron-lora-v708-a100-equation-single-family/checkpoint-5`

Manifest:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/downloaded_job_artifacts/repo_snapshot/evals/v710-a100-v708-ckpt5-weak-20260520T211950Z/v245_hf_weak_eval_manifest.json`

Result:

- Total correct: `191/315`.
- Accuracy: `0.606349`.
- `bit_manipulation`: `135/160`.
- `equation_transform`: `56/155`.
- Truncated rows: `1`.
- Boxed rows: `314/315`.
- No-box fallback rows: `1`.
- Avg completion tokens: `4775.58`.
- Max completion tokens: `7680`.

Promotion blockers:

- `correct_lt_196`.
- `equation_lt_60`.
- `bit_lt_136`.
- `truncated_gt_0`.
- `avg_completion_tokens_gt_512`.
- `no_box_fallback_gt_0`.
- `boxed_rate_lt_1.0`.
- `protected_row_backfire_guard_failed`.

Protected-row failures:

- `8740ed31`: expected `01101000`, baseline `01101000`, candidate
  `01111000`; real backfire from correct baseline.
- `59bee375`: expected `10010101`, baseline `10010101`, candidate `2`; real
  backfire from correct baseline and the truncated/no-box row.
- `55d834d1`: expected `00111111`, baseline `10111111`, candidate
  `10111111`; missing required gain.

Row-level delta:

- V703 was `190/315`, bit `134/160`, equation `56/155`.
- V710 is `191/315`, bit `135/160`, equation `56/155`.
- Only one row improved vs V703: `4ada9150`.
- There were no V710 regressions vs V703, but V710 still regresses two
  protected V516-correct rows: `59bee375` and `8740ed31`.

Decision: V710 is not promotable or submittable.

## Active Parameter Contract

Dataset:

- V708 single-family train rows: `852`.
- V708 single-family validation rows: `195`.
- Family: `equation_transform` only.
- Train hash:
  `a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d`.
- Validation hash:
  `f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba`.

Training/loss active candidate:

- V712: `max_steps=20`, `save_every_steps=10`, `eval_every_steps=10`.
- V712 LR: `2e-6 -> 5e-7`.
- V708 historical failed probe: `max_steps=5`, `save_every_steps=5`,
  `eval_every_steps=5`, LR `5e-7 -> 1e-7`.
- `max_length=1024`.
- `LOSS_NORMALIZATION_MODE=example_mean`.
- Cross-entropy is masked to response tokens.
- Row-loss weight is required.
- EOS mask must stop after EOS.
- Tokenization gates require zero truncation, zero fallback masks, and zero
  completion-token drops.

LoRA:

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Initial adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke/checkpoint-6`.
- V712 output adapter repo:
  `felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal`.
- V708 failed output adapter repo:
  `felipesp1983/kg1-nemotron-lora-v708-a100-equation-single-family`.
- LoRA `r=32`, `alpha=32`.
- Intended trainable modules: `q_proj,v_proj`.
- `lm_head` excluded.
- MoE `target_parameters` frozen.

Important audit note: the evaluated adapter config reports the broad active
inherited adapter surface, but V711 proved the effective trainable surface was
correctly filtered:

- active target modules:
  `down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj`;
- trainable filter: `q_proj,v_proj`;
- trainable LoRA params: `1,867,776`;
- frozen LoRA params: `882,006,016`;
- MoE target parameter trainable params: `0`;
- `lm_head` absent;
- gate:
  `artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/v711_lora_trainability_manifest_gate.json`,
  `passed=true`.

Therefore broad `adapter_config.target_modules` is not, by itself, the current
root cause. The likely remaining causes are insufficient training signal and
runaway decoding/protected-row backfire.

Weak eval:

- Weak CSV:
  `runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv`.
- Weak CSV SHA256:
  `85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6`.
- Shared row contract SHA256:
  `bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff`.
- `KG1_DISABLE_THINKING=0`.
- `KG1_REQUIRE_DISABLE_THINKING=0`.
- `KG1_NO_PROMPT_SUFFIX=0`.
- `KG1_MAX_TOKENS=7680`.
- `KG1_MAX_MODEL_LEN=8192`.
- `KG1_MAX_NUM_SEQS=64`.
- Promotion thresholds: total `>=196`, bit `>=136`, equation `>=60`,
  truncation `0`, boxed rate `1.0`, no no-box fallback, protected-row backfire
  `0`.

Finance/runtime:

- H200 forbidden.
- Use A100 only.
- HF A100-large observed cost: `0.041667 USD/min`.
- Remaining budget is tight. Do not launch if any paid KG1 job is active.

## Confirmed V708 Outcome

HF job:
`https://huggingface.co/jobs/felipesp1983/6a0e1a7b3fef25139d6c9f1c`

Result:

- Job completed.
- Baseline eval loss: `2.6376`.
- Final step-5 eval loss: `2.6394`.
- Loss delta: `+0.0018`, a slight regression.
- Adapter uploaded to:
  `felipesp1983/kg1-nemotron-lora-v708-a100-equation-single-family/checkpoint-5`.
- Final adapter uploaded to:
  `felipesp1983/kg1-nemotron-lora-v708-a100-equation-single-family/final_adapter`.

Blocker found after the run:

- The launch manifest required first-checkpoint weak eval.
- The remote command only compiled `scripts/hf_job_weak_eval_v245.py`.
- It did not execute weak eval.

This false-gain pattern was fixed in launch/gate logic, but V710 proved the
resulting adapter is still below promotion thresholds.

## Accepted External Review So Far

V709 OpenRouter + Hugging Face consult:
`artifacts/openrouter/v709_v708_post_train_orchestration_bug/`

Accepted consensus:

- Block promotion of V708 current result.
- Do not retrain before real weak eval.
- Run weak-eval-only A100 on `checkpoint-5`.
- Keep official-like decoding: `disable_thinking=0`, `max_tokens=7680`.
- Keep protected-row and promotion gates.

Rejected noisy suggestions:

- `disable_thinking=1` or `max_tokens=512`: rejected because V706 collapsed ACC.
- Lowering equation threshold to `57`: rejected. Active threshold remains `60`.
- Broad LoRA/rank/module changes before real weak eval: rejected.

## V712 Execution Contract

V712 was reduced from the draft `50` steps to `20` steps because the active
pre-paid gate allows `deferred_post_checkpoint` only when `MAX_STEPS<=20` and
`SAVE/EVAL_EVERY_STEPS<=10`. This avoids weakening the drift/backfire policy
just to run a larger experiment.

Confirmed before launch:

- Static safety gate: `ok=true`, `findings=[]`.
- Pre-paid integration gate: `ok=true`, `findings=[]`.
- HF dataset validation: train/validation hashes match the local dataset.
- HF active job blocker before launch: none.
- Runtime target: `a100-large`, A100 80GB, no H200.
- Dataset: train `852`, validation `195`, all `equation_transform`.
- Tokenization gate: train/val truncation `0`, fallback masks `0`,
  offset masks complete.
- LoRA effective trainability required by inline post-train gate:
  `q_proj,v_proj` trainable only, MoE target parameters frozen, `lm_head`
  absent, modules_to_save empty/null.
- Weak eval after training is mandatory on `checkpoint-20`.

V712 is not promotable unless the job weak eval passes unchanged thresholds:
total `>=196`, bit `>=136`, equation `>=60`, truncation `0`, boxed rate `1.0`,
no no-box fallback, protected-row backfire `0`.

## Next Action

1. Monitor V712 job `6a0e3565ac8efd7fbbb2aa06`.
2. If V712 fails, download logs/artifacts, run a failure consult prompt via
   OpenRouter/Hugging Face, and update this roadmap before any new paid run.
3. If V712 passes weak thresholds, still do not submit/package until the
   promotion artifacts and protected-row guard are locally downloaded and
   verified.
4. Any future eval script must print progress checkpoints; V710 was too silent
   during long generation.

Hard stop rules:

- No H200.
- No submit/package/full eval from V710.
- No threshold weakening to make a false pass.
- No second paid job while V712 is active.
- No new paid job while another paid KG1 job is active.

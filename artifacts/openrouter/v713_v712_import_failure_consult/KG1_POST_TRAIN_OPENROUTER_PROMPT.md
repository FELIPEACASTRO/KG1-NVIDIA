# KG1 post-training crisis consult

You are an external ML/MLOps/code-review panel for the KG1 NVIDIA Nemotron Model Reasoning Challenge solution.
Return only actionable findings. Do not invent files, metrics, or leaderboard results.
If evidence is insufficient, say exactly which local artifact or metric is missing.

## Hard rules
- Kaggle score is final-answer accuracy, not eval_loss alone.
- False gains are forbidden: no promotion without label-free extraction, verify_answer, zero truncation, and protected-row backfire guards.
- Distinguish bad decoding from adapter weights pushing the model toward wrong answers.
- Loss must be cross-entropy on masked assistant/answer tokens and must stay aligned with accuracy gates.
- If row_loss_weight is used in train, validation eval_loss must use the same row-weight contract.
- No new paid GPU run should be recommended unless CPU/gate evidence predicts an accuracy gain and protects current correct rows.
- Prefer A100-large. H200 is allowed only when A100 cannot run the stack or memory requirement is objectively proven.
- After any failed train/eval/job/gate and after any correction for that failure, require a new artifact-backed OpenRouter review plus a Hugging Face metadata/provider track before more GPU spend.

## Required response format
1. Verdict: proceed / block / needs artifact.
2. Top 5 concrete bugs or gaps, each tied to evidence in the prompt.
3. Exact next experiment that is cheapest and most likely to improve weak ACC.
4. Parameters to change or freeze, with values.
5. Gates that must pass before another paid GPU job.
6. Anything in the current plan that should be deleted because it is noise.


## Current run metadata
- run_id: `v713-v712-import-failure-after-a100-train`
- generated_at_utc: `2026-05-20T23:13:08.840953+00:00`
- Current observed plateau: weak ACC has not exceeded the deployable baseline. V664 reached only 192/315.
- Best actionable weak target remains at least 196/315 without protected-row regression, with bit >= 136 and equation >= 60.
- V664 weak result: total 192/315, bit 136/160, equation 56/155, truncated 0, boxed_rate 1.0, but completions were extremely long and protected bit row 8740ed31 backfired.
- V664 training moved only q_proj/v_proj LoRA tensors from V290 checkpoint-6; non-q/v tensors were unchanged.
- V664 train loss decreased in 2 steps, but the generation behavior stayed long and unsafe. Loss movement alone is not acceptable evidence.


## Hugging Face metadata/provider track

This section is not a leaderboard signal. It records which relevant HF models/providers are live enough to be used as an external review track.

```json
{
  "generated_at_utc": "2026-05-20T23:13:06.534176+00:00",
  "models": [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  ],
  "ok_count": 6,
  "result_count": 6,
  "results": [
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "cerebras",
          "provider_id": "gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "novita",
          "provider_id": "openai/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "together",
          "provider_id": "openai/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "fireworks-ai",
          "provider_id": "accounts/fireworks/models/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "groq",
          "provider_id": "openai/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "nscale",
          "provider_id": "openai/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "featherless-ai",
          "provider_id": "openai/gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "sambanova",
          "provider_id": "gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "scaleway",
          "provider_id": "gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "openai/gpt-oss-120b",
          "isModelAuthor": false,
          "provider": "ovhcloud",
          "provider_id": "gpt-oss-120b",
          "status": "live",
          "task": "conversational",
          "type": null
        }
      ],
      "library_name": null,
      "likes": null,
      "model_id": "openai/gpt-oss-120b",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    },
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen3-32B",
          "isModelAuthor": false,
          "provider": "novita",
          "provider_id": "qwen/qwen3-32b-fp8",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen3-32B",
          "isModelAuthor": false,
          "provider": "groq",
          "provider_id": "qwen/qwen3-32b",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen3-32B",
          "isModelAuthor": false,
          "provider": "nscale",
          "provider_id": "Qwen/Qwen3-32B",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen3-32B",
          "isModelAuthor": false,
          "provider": "featherless-ai",
          "provider_id": "Qwen/Qwen3-32B",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen3-32B",
          "isModelAuthor": false,
          "provider": "ovhcloud",
          "provider_id": "Qwen3-32B",
          "status": "live",
          "task": "conversational",
          "type": null
        }
      ],
      "library_name": null,
      "likes": null,
      "model_id": "Qwen/Qwen3-32B",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    },
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/QwQ-32B",
          "isModelAuthor": false,
          "provider": "nscale",
          "provider_id": "Qwen/QwQ-32B",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/QwQ-32B",
          "isModelAuthor": false,
          "provider": "featherless-ai",
          "provider_id": "Qwen/QwQ-32B",
          "status": "live",
          "task": "conversational",
          "type": null
        }
      ],
      "library_name": null,
      "likes": null,
      "model_id": "Qwen/QwQ-32B",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    },
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
          "isModelAuthor": false,
          "provider": "nscale",
          "provider_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
          "isModelAuthor": false,
          "provider": "featherless-ai",
          "provider_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
          "isModelAuthor": false,
          "provider": "scaleway",
          "provider_id": "qwen2.5-coder-32b-instruct",
          "status": "live",
          "task": "conversational",
          "type": null
        }
      ],
      "library_name": null,
      "likes": null,
      "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    },
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "deepseek-ai/DeepSeek-R1-0528",
          "isModelAuthor": false,
          "provider": "novita",
          "provider_id": "deepseek/deepseek-r1-0528",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "deepseek-ai/DeepSeek-R1-0528",
          "isModelAuthor": false,
          "provider": "featherless-ai",
          "provider_id": "deepseek-ai/DeepSeek-R1-0528",
          "status": "live",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "deepseek-ai/DeepSeek-R1-0528",
          "isModelAuthor": false,
          "provider": "together",
          "provider_id": "deepseek-ai/DeepSeek-R1-0528",
          "status": "error",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "deepseek-ai/DeepSeek-R1-0528",
          "isModelAuthor": false,
          "provider": "hyperbolic",
          "provider_id": "deepseek-ai/DeepSeek-R1-0528",
          "status": "error",
          "task": "conversational",
          "type": null
        },
        {
          "adapter": null,
          "adapter_weights_path": null,
          "hf_model_id": "deepseek-ai/DeepSeek-R1-0528",
          "isModelAuthor": false,
          "provider": "sambanova",
          "provider_id": "DeepSeek-R1-0528",
          "status": "error",
          "task": "conversational",
          "type": null
        }
      ],
      "library_name": null,
      "likes": null,
      "model_id": "deepseek-ai/DeepSeek-R1-0528",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    },
    {
      "downloads": null,
      "gated": null,
      "inference_provider_mapping": [],
      "library_name": null,
      "likes": null,
      "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
      "pipeline_tag": null,
      "private": null,
      "sha": null,
      "status": "ok",
      "tags": []
    }
  ],
  "schema_version": "kg1_hf_metadata_review_track_v1",
  "skip": false,
  "status": "ok"
}
```


## Roadmap
Path: `artifacts\roadmaps\active\KG1_ACTIVE_ROADMAP_2026_05_20.md`

```text
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

```


## Training or launch manifest
Path: `artifacts\v712_hf_a100_equation_signal_launch\v712-a100-equation-signal-v290ckpt6-20260520T222638Z_launch_manifest.json`

```text
{
  "active_job_blockers": [],
  "blocked_actions": [
    "kaggle_submit",
    "package"
  ],
  "branch": "v230-v226-complementarity",
  "dataset": {
    "data_repo": "felipesp1983/kg1-v708-equation-single-family-dataset",
    "data_root": "v708-equation-single-family-20260520T",
    "dataset_upload_commit": "b85f1a49625c2cec585b203015097fc27d1d7c72",
    "families": [
      "equation_transform"
    ],
    "hf_dataset": {
      "hf_train_file": "v708-equation-single-family-20260520T/v708_equation_single_family_train.jsonl",
      "hf_val_file": "v708-equation-single-family-20260520T/v708_equation_single_family_val.jsonl",
      "train_sha256": "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d",
      "validation_sha256": "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba"
    },
    "local_dataset": {
      "manifest_file": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v708_equation_single_family_manifest.json",
      "train_file": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v708_equation_single_family_train.jsonl",
      "train_rows": 852,
      "train_sha256": "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d",
      "validation_file": "C:\\Users\\davis\\Workspace\\KG1 -NVIDIA\\artifacts\\v284_official_gate_worktree\\artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v708_equation_single_family_val.jsonl",
      "validation_rows": 195,
      "validation_sha256": "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba"
    },
    "subcategories": [
      "equation_numeric_add_direct_low_support",
      "equation_numeric_colon_absdiff_unreverse_low_support",
      "equation_numeric_minus_signed_reverse_high_support",
      "equation_numeric_minus_signed_reverse_low_support",
      "equation_symbolic_cryptarithm_single_operator_mul",
      "symbolic_cryptarithm_multi_operator_digits_add",
      "symbolic_cryptarithm_multi_operator_digits_mul",
      "symbolic_cryptarithm_single_operator_digits_mul",
      "v640_lkevin_equation_symbolic_trace"
    ],
    "train_file": "v708-equation-single-family-20260520T/v708_equation_single_family_train.jsonl",
    "train_rows": 852,
    "train_sha256": "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d",
    "val_file": "v708-equation-single-family-20260520T/v708_equation_single_family_val.jsonl",
    "val_rows": 195,
    "val_sha256": "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba"
  },
  "expected_commit": "67a27bcb2ed6a4e9856adb61ece63516a0b29637",
  "flavor": "a100-large",
  "generated_at_utc": "2026-05-20T22:26:40.517790+00:00",
  "hardware": {
    "accelerator_model": "A100",
    "accelerator_quantity": "1",
    "accelerator_vram": "80 GB",
    "cpu": "12 vCPU",
    "name": "a100-large",
    "pretty_name": "Nvidia A100 - large",
    "ram": "142 GB",
    "unit_cost_usd": 0.041667,
    "unit_label": "minute"
  },
  "image": "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano",
  "init_adapter": {
    "contract": "V290 r=32 alpha=32; adapter-only modules exclude lm_head; MoE target_parameters loaded but frozen",
    "repo": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
    "subfolder": "checkpoint-6"
  },
  "job_env": {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "KG1_ABORT_MAX_RESERVED_GIB": "78",
    "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
    "KG1_ALLOWED_HF_FLAVORS": "a100-large",
    "KG1_ALLOW_CUDA13_ON_A100": "1",
    "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
    "KG1_ALLOW_MANUAL_TARGET_PARAMETERS_LOAD": "1",
    "KG1_ANSWER_SPAN_LOSS_WEIGHT": "1.0",
    "KG1_ANSWER_SPAN_MIN_WEIGHTED_TOKENS": "0",
    "KG1_BRANCH": "v230-v226-complementarity",
    "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
    "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": "1.0",
    "KG1_CPU_SIMULATED_BIT_CORRECT": "136",
    "KG1_CPU_SIMULATED_EQUATION_CORRECT": "60",
    "KG1_CPU_SIMULATED_LOST_BIT_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": "0",
    "KG1_CPU_SIMULATED_LOST_ROWS": "0",
    "KG1_CPU_SIMULATED_TOTAL_CORRECT": "196",
    "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
    "KG1_CRISIS_MODE_BACKFIRE_GUARD": "1",
    "KG1_CUDA13_A100_DRIVER_GATE_STATUS": "inline_smoke_required",
    "KG1_DATASET_SCHEMA": "sft",
    "KG1_DATA_REPO": "felipesp1983/kg1-v708-equation-single-family-dataset",
    "KG1_DATA_ROOT": "v708-equation-single-family-20260520T",
    "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
    "KG1_DISABLE_THINKING": "0",
    "KG1_DROP_INIT_ADAPTER_TARGET_MODULES": "lm_head",
    "KG1_ENFORCE_WEAK_RUNTIME_POLICY": "1",
    "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
    "KG1_EVAL_TIMEOUT_S": "4200",
    "KG1_EXPECTED_COMMIT": "67a27bcb2ed6a4e9856adb61ece63516a0b29637",
    "KG1_EXPECTED_LOSS_NORMALIZATION_MODE": "example_mean",
    "KG1_EXPECTED_MAX_LENGTH": "1024",
    "KG1_EXPECTED_MAX_STEPS": "20",
    "KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
    "KG1_EXPECTED_TRUNCATED": "0",
    "KG1_EXPECTED_WEAK_CSV_SHA256": "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6",
    "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
    "KG1_FREEZE_LORA_TARGET_PARAMETERS": "1",
    "KG1_GENERATION_TIMEOUT_S": "900",
    "KG1_HF_FLAVOR": "a100-large",
    "KG1_HF_MAX_UNIT_COST_USD": "0.05",
    "KG1_HF_UNIT_COST_USD": "0.041667",
    "KG1_INIT_ADAPTER_REPO": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
    "KG1_INIT_ADAPTER_SUBFOLDER": "checkpoint-6",
    "KG1_LORA_TARGET_PARAMETERS": "mlp.experts.gate_up_proj,mlp.experts.down_proj",
    "KG1_LOSS_MASK_STOP_AFTER_EOS": "1",
    "KG1_LOSS_NORMALIZATION_MODE": "example_mean",
    "KG1_MAX_MODEL_LEN": "8192",
    "KG1_MAX_NEW_TOKENS": "7680",
    "KG1_MAX_NUM_SEQS": "64",
    "KG1_MAX_PROMPT_TRUNCATION_RATE": "0.0",
    "KG1_MAX_TOKENS": "7680",
    "KG1_MAX_TOKEN_HEADROOM_RATIO": "0.371",
    "KG1_MIN_GPU_TOTAL_GIB": "70",
    "KG1_NO_PROMPT_SUFFIX": "0",
    "KG1_OUTPUT_REPO": "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal",
    "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
    "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
    "KG1_PROTECTED_ROW_GUARD": "1",
    "KG1_REQUIRED_GPU_NAME_REGEX": "A100",
    "KG1_REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS": "q_proj,v_proj",
    "KG1_REQUIRED_TRAIN_FAMILIES": "equation_transform",
    "KG1_REQUIRED_TRAIN_SUBCATEGORIES": "equation_numeric_add_direct_low_support,equation_numeric_colon_absdiff_unreverse_low_support,equation_numeric_minus_signed_reverse_high_support,equation_numeric_minus_signed_reverse_low_support,equation_symbolic_cryptarithm_single_operator_mul,symbolic_cryptarithm_multi_operator_digits_add,symbolic_cryptarithm_multi_operator_digits_mul,symbolic_cryptarithm_single_operator_digits_mul,v640_lkevin_equation_symbolic_trace",
    "KG1_REQUIRED_VAL_FAMILIES": "equation_transform",
    "KG1_REQUIRED_VAL_SUBCATEGORIES": "equation_numeric_add_direct_low_support,equation_numeric_colon_absdiff_unreverse_low_support,equation_numeric_minus_signed_reverse_high_support,equation_numeric_minus_signed_reverse_low_support,equation_symbolic_cryptarithm_single_operator_mul,symbolic_cryptarithm_multi_operator_digits_add,symbolic_cryptarithm_multi_operator_digits_mul,symbolic_cryptarithm_single_operator_digits_mul,v640_lkevin_equation_symbolic_trace",
    "KG1_REQUIRE_CUDA": "1",
    "KG1_REQUIRE_DISABLE_THINKING": "0",
    "KG1_REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": "0",
    "KG1_REQUIRE_MAMBA_IMPORTS": "1",
    "KG1_REQUIRE_OFFSET_MASK": "1",
    "KG1_REQUIRE_ROW_LOSS_WEIGHT": "1",
    "KG1_RESIDUAL_FIRST_GATE": "1",
    "KG1_ROW_LOSS_WEIGHT_REDUCTION": "scale_mean",
    "KG1_RUN_ID": "v712-a100-equation-signal-v290ckpt6-20260520T222638Z",
    "KG1_SAVE_EMBEDDING_LAYERS": "0",
    "KG1_SOURCE_WEIGHTS": "v708_equation_single_family_dataset=1.00",
    "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
    "KG1_STOP_ON_PROTECTED_BACKFIRE": "1",
    "KG1_STRICT_INIT_ADAPTER_CONFIG": "1",
    "KG1_SUBCATEGORY_WEIGHTS": "equation_numeric_add_direct_low_support=1.00,equation_numeric_colon_absdiff_unreverse_low_support=1.00,equation_numeric_minus_signed_reverse_high_support=1.00,equation_numeric_minus_signed_reverse_low_support=1.00,equation_symbolic_cryptarithm_single_operator_mul=1.00,symbolic_cryptarithm_multi_operator_digits_add=1.00,symbolic_cryptarithm_multi_operator_digits_mul=1.00,symbolic_cryptarithm_single_operator_digits_mul=1.00,v640_lkevin_equation_symbolic_trace=1.00",
    "KG1_TRAINABLE_LORA_MODULES": "q_proj,v_proj",
    "KG1_TRAINABLE_LORA_NAME_SUBSTRINGS": "",
    "KG1_TRAIN_FILE": "v708-equation-single-family-20260520T/v708_equation_single_family_train.jsonl",
    "KG1_TRAIN_ROWS": "852",
    "KG1_TRAIN_SHA": "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d",
    "KG1_USE_ROW_LOSS_WEIGHT": "1",
    "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
    "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
    "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
    "KG1_V541_FLIP_LEDGER_STATUS": "passed",
    "KG1_V541_MISSMAP_GATE_STATUS": "passed",
    "KG1_V618_MODULE_SURFACE_GATE_STATUS": "passed",
    "KG1_V619_MODULE_SURFACE_GATE_STATUS": "passed",
    "KG1_V666_CPU_GATE_STACK_REPORT": "artifacts/v708_hf_a100_launch/v708_cpu_gate_stack.json",
    "KG1_V666_CPU_GATE_STACK_STATUS": "passed",
    "KG1_VAL_FILE": "v708-equation-single-family-20260520T/v708_equation_single_family_val.jsonl",
    "KG1_VAL_ROWS": "195",
    "KG1_VAL_SHA": "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba",
    "KG1_WEAK_CSV_FILE": "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    "KG1_WEAK_EVAL_DATA_REPO": "felipesp1983/kg1-nemotron-training",
    "KG1_WEAK_EVAL_HARNESS": "scripts/hf_job_weak_eval_v245.py",
    "KG1_WEAK_EVAL_REQUIRED_CHECKPOINT": "checkpoint-20",
    "KG1_WEAK_LABEL_AWARE_SELECTION": "0",
    "KG1_WEAK_MANIFEST_FILE": "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json",
    "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "512",
    "KG1_WEAK_PROMOTE_BIT_MIN": "136",
    "KG1_WEAK_PROMOTE_BOXED_RATE_MIN": "1.0",
    "KG1_WEAK_PROMOTE_EQUATION_MIN": "60",
    "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "7680",
    "KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX": "0",
    "KG1_WEAK_PROMOTE_TOTAL_MIN": "196",
    "KG1_WEAK_PROMOTE_TRUNC_MAX": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8"
  },
  "job_id": "6a0e3565ac8efd7fbbb2aa06",
  "job_status": "RUNNING",
  "job_url": "https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06",
  "mode": "launched",
  "namespace": "felipesp1983",
  "next_action": "run pre-paid integration gate; launch only one A100-large 20-step signal probe after all gates pass",
  "output_repo": "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal",
  "recipe": {
    "abort_max_reserved_gib": 78,
    "batch_size": 2,
    "dropped_init_adapter_target_modules": "lm_head",
    "eval_every_steps": 10,
    "eval_max_examples": 195,
    "final_learning_rate": "5.0e-7",
    "learning_rate": "2.0e-6",
    "lora_alpha": 32,
    "lora_r": 32,
    "loss_mask_stop_after_eos": "1",
    "loss_normalization_mode": "example_mean",
    "max_length": 1024,
    "max_steps": 20,
    "micro_batch_size": 1,
    "promotion_gate": "first checkpoint weak eval required with total>=196, bit>=136, equation>=60, truncation=0, boxed rate=1.0, no protected backfire",
    "row_loss_weight_reduction": "scale_mean",
    "save_every_steps": 10,
    "source_weights": "v708_equation_single_family_dataset=1.00",
    "subcategory_weights": "equation_numeric_add_direct_low_support=1.00,equation_numeric_colon_absdiff_unreverse_low_support=1.00,equation_numeric_minus_signed_reverse_high_support=1.00,equation_numeric_minus_signed_reverse_low_support=1.00,equation_symbolic_cryptarithm_single_operator_mul=1.00,symbolic_cryptarithm_multi_operator_digits_add=1.00,symbolic_cryptarithm_multi_operator_digits_mul=1.00,symbolic_cryptarithm_single_operator_digits_mul=1.00,v640_lkevin_equation_symbolic_trace=1.00",
    "target_modules": "down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj",
    "target_parameters": "mlp.experts.gate_up_proj,mlp.experts.down_proj",
    "target_parameters_trainability": "frozen",
    "trainable_lora_modules": "q_proj,v_proj",
    "use_row_loss_weight": "1",
    "weak_eval_command": "scripts/hf_job_weak_eval_v245.py on checkpoint-20 after hf_job_train_v90.py and inline trainability gate"
  },
  "run_id": "v712-a100-equation-signal-v290ckpt6-20260520T222638Z",
  "version": "v712_a100_equation_signal_v290ckpt6"
}
```


## Failure analysis summary
Path: `artifacts\openrouter\v713_v712_import_failure_consult\v712_failure_summary.json`

```text
{
  "adapter_contract": {
    "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "base_revision": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    "effective_trainable_modules": [
      "q_proj",
      "v_proj"
    ],
    "frozen_lora_params": 882006016,
    "init_adapter_repo": "felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke",
    "init_adapter_subfolder": "checkpoint-6",
    "lm_head_excluded": true,
    "lora_alpha": 32,
    "lora_r": 32,
    "modules_to_save_expected": [],
    "target_parameter_trainable_lora_params": 0,
    "target_parameters": [
      "mlp.experts.gate_up_proj",
      "mlp.experts.down_proj"
    ],
    "target_parameters_trainability_mode": "frozen_active",
    "trainable_lora_params": 1867776
  },
  "failure": {
    "classification": "silent packaging/import bug; not model or loss failure; no ACC evidence produced",
    "exception_type": "ModuleNotFoundError",
    "importing_script": "scripts/evaluate_lora_adapters_batch.py",
    "message": "No module named 'scripts.evaluate_lora_adapter'",
    "phase": "weak_eval_after_training",
    "root_cause_hypothesis": "Remote container resolved `scripts` as a different regular package or could not treat repo scripts as a regular package because repo scripts/ had no __init__.py."
  },
  "fix_applied_locally": {
    "files_changed": [
      "scripts/__init__.py",
      "artifacts/v284_official_gate_worktree/scripts/__init__.py",
      "artifacts/v284_official_gate_worktree/artifacts/v712_hf_a100_equation_signal_launch/launch_v712_hf_a100_equation_signal.py",
      "artifacts/v284_official_gate_worktree/scripts/kg1_pre_paid_job_integration_gate.py",
      "artifacts/v284_official_gate_worktree/scripts/kg1_static_safety_gate.py"
    ],
    "launcher_import_preflight": [
      "scripts_package_gate",
      "weak_eval_import_gate_ok",
      "py_compile scripts/evaluate_lora_adapter.py",
      "py_compile scripts/evaluate_lora_adapters_batch.py"
    ],
    "validated": {
      "local_import_official_worktree": true,
      "local_import_root_workspace": true,
      "pre_paid_integration_gate_ok": true,
      "pre_paid_integration_gate_self_test_ok": true,
      "py_compile": true,
      "static_safety_gate_ok": true,
      "static_safety_gate_self_test_ok": true
    }
  },
  "generated_at_utc": "2026-05-20T23:10:00Z",
  "job": {
    "flavor": "a100-large",
    "h200_used": false,
    "id": "6a0e3565ac8efd7fbbb2aa06",
    "output_repo": "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal",
    "status": "ERROR",
    "url": "https://huggingface.co/jobs/felipesp1983/6a0e3565ac8efd7fbbb2aa06"
  },
  "requested_panel_decision": {
    "budget_policy": "Do not rerun training if weak-eval-only can reuse uploaded checkpoints.",
    "candidate_next_actions": [
      "Launch a weak-eval-only A100 job for V712 checkpoint-10 after the import fix is available remotely.",
      "If checkpoint-10 passes import and weak eval but fails score/backfire, consult again before more GPU spend.",
      "Only evaluate checkpoint-20 after checkpoint-10 if checkpoint-10 result shows a plausible ACC path."
    ],
    "hardware_policy": "Use A100-large only; H200 forbidden.",
    "promotion_thresholds": {
      "bit_manipulation_min": 136,
      "boxed_rate_min": 1.0,
      "equation_transform_min": 60,
      "no_box_fallback_max": 0,
      "protected_backfire_max": 0,
      "total_correct_min": 196,
      "truncated_max": 0
    }
  },
  "schema_version": "kg1_v713_v712_import_failure_summary_v1",
  "training_result": {
    "baseline_eval_loss": 2.6376,
    "checkpoint_10_eval_loss": 2.6363,
    "checkpoint_10_is_best_loss": true,
    "checkpoint_20_eval_loss": 2.6373,
    "fallback_masks": 0,
    "final_eval_loss": 2.6373,
    "phase": "completed",
    "train_offset_masks": 852,
    "train_rows_raw": 852,
    "train_rows_tokenized": 852,
    "train_truncated": 0,
    "validation_offset_masks": 195,
    "validation_rows_raw": 195,
    "validation_rows_tokenized": 195,
    "validation_truncated": 0,
    "weak_accuracy_available": false
  }
}
```


## Previous OpenRouter consensus
Path: `artifacts\openrouter\v711_v710_failure_consult\KG1_V711_CONSENSUS_AND_LOCAL_DECISION.md`

```text
# KG1 V711 Consensus And Local Decision

Updated: 2026-05-20 22:55 UTC

## Inputs

- V710 failure ledger:
  `artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/KG1_V710_FAILURE_ANALYSIS.md`
- Full prompt:
  `artifacts/openrouter/v711_v710_failure_consult/KG1_V711_OPENROUTER_HF_PROMPT.md`
- Compact prompt:
  `artifacts/openrouter/v711_v710_failure_consult/KG1_V711_COMPACT_RETRY_PROMPT.md`
- Hugging Face metadata track:
  `artifacts/openrouter/v711_v710_failure_consult/KG1_V711_HUGGINGFACE_METADATA_TRACK.md`
- OpenRouter results:
  `artifacts/openrouter/v711_v710_failure_consult/openrouter_results.jsonl`
- OpenRouter compact retry:
  `artifacts/openrouter/v711_v710_failure_consult/openrouter_compact_retry_results.jsonl`

## External Signals

Models consulted:

- `deepseek/deepseek-v4-pro`
- `qwen/qwen3-max-thinking`
- `google/gemini-3.1-pro-preview`
- `anthropic/claude-sonnet-4.6`
- `openai/gpt-5.5`
- compact retry: `qwen/qwen3-235b-a22b-2507`,
  `google/gemini-3.1-flash-lite`, `anthropic/claude-haiku-4.5`,
  `openai/gpt-oss-120b:free`
- `deepseek/deepseek-v4-flash:free` was rate-limited upstream (`429`)

Consensus points worth keeping:

- Do not launch another paid GPU job before a local audit.
- The adapter/LoRA trainability contract must be proven explicitly; do not infer
  trainability from `adapter_config.json` alone.
- The V710 output still has runaway decoding: `59bee375` hit `7680` tokens and
  produced no `\boxed{}`.
- V708 had too little learning signal: `5` steps, batch size `2`, LR
  `5e-7 -> 1e-7`, and only `1,867,776` trainable LoRA params.
- Protected-row labels must not be added to train data. They stay as backfire
  guards only.

Rejected or unsafe suggestions:

- Lower promotion thresholds. This would create a false gain.
- Use H200. User policy forbids it.
- Use label-aware scoring or protected labels for promotion/training.
- Unfreeze `lm_head`. Current contract excludes `lm_head`; changing it would be
  a new high-risk experiment.
- Treat `modules_to_save=["equation_transform"]` as meaningful. This is not a
  valid model module in this code path.
- Cap official-like generation to `512` tokens as a promotion path. V706 already
  showed short/no-thinking decode collapse, and V710's only gained row used
  `6987` completion tokens. Short-decode can be a diagnostic only.

## Local Resolution Of The Main Suspicion

OpenRouter repeatedly flagged "LoRA config drift" because `adapter_config.json`
lists a broad active target surface:

`down_proj,in_proj,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj`

Local audit showed this is not automatically a trainability bug. The V708 final
training manifest records the effective trainable filter:

- trainable modules: `q_proj,v_proj`
- trainable LoRA params: `1,867,776`
- frozen LoRA params: `882,006,016`
- MoE target parameters:
  `mlp.experts.gate_up_proj,mlp.experts.down_proj`
- MoE trainable LoRA params: `0`
- target parameter trainability mode: `frozen_active`
- `lm_head` absent from active target modules

New gate:
`artifacts/v710_hf_a100_v708_checkpoint5_weak_eval/v711_lora_trainability_manifest_gate.json`

Result: `passed=true`, `blockers=[]`.

Interpretation: the adapter config's broad active surface is inherited/loading
surface, not proof that V708 trained all modules. The gate must stay so we do
not repeat this ambiguity.

## Updated Root Cause Ranking

1. Insufficient training signal is now the highest-confidence training-side
   cause. V708 trained only `5` steps at LR `5e-7 -> 1e-7`; equation stayed
   exactly `56/155`, and eval loss regressed slightly.
2. Runaway decoding remains a real eval-side failure. The row `59bee375`
   generated repeated text until the token limit, no box, and prediction `2`.
3. Protected bit backfire remains real. Rows `8740ed31` and `59bee375` are
   baseline-correct but adapter-wrong.
4. LoRA trainability drift is resolved as a false suspicion for V708, but it is
   now a required pre-paid gate for any future run.

## Decision

Do not launch immediately.

Next implementation step is a local V712 readiness gate stack:

1. Keep the new LoRA trainability manifest gate.
2. Add the gate result to any future paid launch manifest.
3. Run a local row-output pathology audit for V710 protected rows.
4. Prepare, but do not launch yet, an A100-only V712 candidate with more
   meaningful learning signal than V708.

## Candidate V712 Direction If Gates Pass

This is not authorized to launch yet.

- GPU: A100 only.
- No H200 fallback.
- Keep official-like weak eval.
- Keep protected-row guard and thresholds unchanged.
- Keep LoRA active adapter surface compatible with V290 loading, but effective
  trainable filter must be exactly `q_proj,v_proj`.
- Keep MoE target parameters frozen.
- Keep `lm_head` excluded.
- Increase training signal from V708:
  - `max_steps`: propose `50` for the next experiment.
  - `save_every_steps`: `10` or `25`.
  - `eval_every_steps`: `10` or `25`.
  - LR must be reviewed; V708 `5e-7 -> 1e-7` was likely too weak.
- Weak eval only after a selected checkpoint, not package/full eval/submit.

Required signal to continue:

- total `>=196/315`
- bit `>=136/160`
- equation `>=60/155`
- no protected backfire
- truncation `0`
- boxed rate `1.0`
- no label-aware-only gain

```


## Extra artifact 1
Path: `artifacts\openrouter\v713_v712_import_failure_consult\v712_hf_job_tail.log`

```text
+ KG1_CANDIDATE_NAMES=v712_checkpoint_20
+ export KG1_CANDIDATE_NAME=v712_checkpoint_20
+ KG1_CANDIDATE_NAME=v712_checkpoint_20
+ export KG1_OUTPUT_DIR=/tmp/kg1_v712_weak_eval
+ KG1_OUTPUT_DIR=/tmp/kg1_v712_weak_eval
+ export KG1_OUTPUT_PATH_IN_REPO=evals/v712-a100-equation-signal-v290ckpt6-20260520T222638Z
+ KG1_OUTPUT_PATH_IN_REPO=evals/v712-a100-equation-signal-v290ckpt6-20260520T222638Z
+ export KG1_LABEL_PREFIX=v712_hf_weak
+ KG1_LABEL_PREFIX=v712_hf_weak
+ export KG1_UPLOAD_TO_HF=1
+ KG1_UPLOAD_TO_HF=1
+ export KG1_ENFORCE_WEAK_PROMOTION_GATE=1
+ KG1_ENFORCE_WEAK_PROMOTION_GATE=1
+ export KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=0
+ KG1_WEAK_EVAL_DIAGNOSTIC_ONLY=0
+ export KG1_EXPECTED_LORA_R=32
+ KG1_EXPECTED_LORA_R=32
+ export KG1_EXPECTED_LORA_ALPHA=32
+ KG1_EXPECTED_LORA_ALPHA=32
+ export KG1_EXPECTED_ADAPTER_BASE_MODEL_NAME_OR_PATH=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
+ KG1_EXPECTED_ADAPTER_BASE_MODEL_NAME_OR_PATH=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
+ export KG1_VLLM_GPU_MEMORY_UTILIZATION=0.86
+ KG1_VLLM_GPU_MEMORY_UTILIZATION=0.86
+ export KG1_ALLOW_VLLM_DEEP_GEMM=0
+ KG1_ALLOW_VLLM_DEEP_GEMM=0
+ export VLLM_USE_DEEP_GEMM=0
+ VLLM_USE_DEEP_GEMM=0
+ export VLLM_MOE_USE_DEEP_GEMM=0
+ VLLM_MOE_USE_DEEP_GEMM=0
+ export VLLM_USE_DEEP_GEMM_E8M0=0
+ VLLM_USE_DEEP_GEMM_E8M0=0
+ export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
+ VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
+ export VLLM_DEEP_GEMM_WARMUP=skip
+ VLLM_DEEP_GEMM_WARMUP=skip
+ export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
+ VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
+ /opt/venv/bin/python scripts/hf_job_weak_eval_v245.py
=== V245 HF WEAK EVAL JOB START ===
generated_at_utc = 2026-05-20T23:02:49.880012+00:00
torch_gpu_status = {
  "cuda": "13.0",
  "cuda_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_total_gib": 79.250732421875,
  "torch": "2.9.0a0+50eac811a6.nv25.09"
}
repo_commit = 67a27bcb2ed6a4e9856adb61ece63516a0b29637
expected_repo_commit = 67a27bcb2ed6a4e9856adb61ece63516a0b29637
import_ok = vllm
data_repo = felipesp1983/kg1-nemotron-training
weak_csv_file = runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv
weak_manifest_file = runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v245_weak_eval_bridge_manifest.json
adapter_repo = felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal
adapter_specs = [
  {
    "name": "v712_checkpoint_20",
    "repo": "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal",
    "subfolder": "checkpoint-20"
  }
]
output_repo = felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal
run_id = v712-a100-equation-signal-v290ckpt6-20260520T222638Z
output_dir = /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z
weak_csv_gate = {
  "family_counts": {
    "bit_manipulation": 160,
    "equation_transform": 155
  },
  "observed_shared_row_contract_sha256": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
  "path": "/root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
  "rows": 315,
  "sha256": "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
}
weak_manifest_gate = {
  "canonical_weak_csv": {
    "bytes": 118669,
    "family_counts": {
      "bit_manipulation": 160,
      "equation_transform": 155
    },
    "observed_shared_row_contract_sha256": "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff",
    "rows": 315,
    "sha256": "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
  },
  "path_in_repo": "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z",
  "schema_version": "kg1_v245_weak_eval_bridge_manifest_v1"
}
eval_id_subset_gate = {
  "enabled": false,
  "eval_csv": "/root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
  "schema_version": "kg1_v245_eval_id_subset_v1",
  "source_weak_csv": "/root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
}
snapshot_adapter_repo = felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal
snapshot_allow_patterns = ["checkpoint-20/*"]
adapter_gates = {
  "adapters": [
    {
      "adapter_config": "/tmp/kg1_v245_adapter_snapshots/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/02f8793bdef8/checkpoint-20/adapter_config.json",
      "adapter_dir": "/tmp/kg1_v245_adapter_snapshots/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/02f8793bdef8/checkpoint-20",
      "adapter_weights": "/tmp/kg1_v245_adapter_snapshots/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/02f8793bdef8/checkpoint-20/adapter_model.safetensors",
      "adapter_weights_bytes": 3537299144,
      "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
      "candidate_name": "v712_checkpoint_20",
      "lora_alpha": 32,
      "modules_to_save": [],
      "r": 32,
      "repo": "felipesp1983/kg1-nemotron-lora-v712-a100-equation-signal",
      "subfolder": "checkpoint-20",
      "target_modules": [
        "down_proj",
        "out_proj",
        "o_proj",
        "in_proj",
        "up_proj",
        "v_proj",
        "k_proj",
        "q_proj"
      ],
      "target_parameters": [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj"
      ]
    }
  ],
  "count": 1
}
weak_runtime_policy_gate = {
  "blockers": [],
  "disable_thinking": false,
  "enforced": true,
  "eval_timeout_s": 4200,
  "generation_timeout_s": 900,
  "max_tokens": 7680,
  "passed": true,
  "promote_max_completion_tokens": 7680,
  "require_disable_thinking": false,
  "schema_version": "kg1_v245_weak_runtime_policy_gate_v1"
}
eval_prompt_controls = {
  "disable_thinking": false,
  "no_prompt_suffix": false,
  "prompt_suffix": "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
}
eval_runtime_controls = {
  "eval_limit": 0,
  "generation_timeout_s": 900,
  "gpu_memory_utilization": 0.86,
  "llm_init_timeout_s": 0,
  "vllm_enable_chunked_prefill": "",
  "vllm_enable_prefix_caching": "",
  "vllm_enforce_eager": "",
  "vllm_use_v1": ""
}
candidate_by_candidate_eval = true
--- COMMAND START ---
cwd = /tmp/kg1
+ /opt/venv/bin/python /tmp/kg1/scripts/evaluate_lora_adapters_batch.py --solution-csv /root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv --questions-csv /root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv --candidates-json /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/candidate_01_4264e906.json --base-model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --label-prefix v712_hf_weak --seed 42 --limit 0 --output-dir /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/eval/candidate_01_4264e906 --max-tokens 7680 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.86 --llm-init-timeout-s 0 --generation-timeout-s 900 --warmup-rows 0 --continue-on-error --prompt-suffix 
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
timeout_s = 4200
log_path = /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/candidate_01_4264e906_weak_eval.log
Traceback (most recent call last):
  File "/tmp/kg1/scripts/evaluate_lora_adapters_batch.py", line 28, in <module>
    from scripts.evaluate_lora_adapter import (  # noqa: E402
ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'
returncode = 1
--- COMMAND END ---
Traceback (most recent call last):
  File "/tmp/kg1/scripts/hf_job_weak_eval_v245.py", line 1640, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/tmp/kg1/scripts/hf_job_weak_eval_v245.py", line 1635, in main
    run_eval(args)
  File "/tmp/kg1/scripts/hf_job_weak_eval_v245.py", line 1063, in run_eval
    run_cmd(
  File "/tmp/kg1/scripts/hf_job_weak_eval_v245.py", line 814, in run_cmd
    raise RuntimeError(f"command failed rc={rc}: {printable}")
RuntimeError: command failed rc=1: /opt/venv/bin/python /tmp/kg1/scripts/evaluate_lora_adapters_batch.py --solution-csv /root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv --questions-csv /root/.cache/huggingface/hub/datasets--felipesp1983--kg1-nemotron-training/snapshots/af532d34c147545b67d15994931bda865ac6eb74/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv --candidates-json /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/candidate_01_4264e906.json --base-model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --label-prefix v712_hf_weak --seed 42 --limit 0 --output-dir /tmp/kg1_v712_weak_eval/v712-a100-equation-signal-v290ckpt6-20260520T222638Z/eval/candidate_01_4264e906 --max-tokens 7680 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.86 --llm-init-timeout-s 0 --generation-timeout-s 900 --warmup-rows 0 --continue-on-error --prompt-suffix 
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`

```


## Extra artifact 2
Path: `artifacts\v712_hf_a100_equation_signal_launch\v712_pre_paid_job_integration_gate_after_importfix.json`

```text
{
  "dataset_schema": "sft",
  "expected_eval_output_contract": "",
  "findings": [],
  "launcher": {
    "contains_expected_flavor": true,
    "contains_timeout_3600": true,
    "declared_dataset_schema": "sft",
    "decoding_vs_adapter_drift_gate": {
      "observed": {
        "EVAL_EVERY_STEPS": 10.0,
        "KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT": "1",
        "KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS": "deferred_post_checkpoint",
        "KG1_EXPECTED_MAX_STEPS": 20.0,
        "KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED": "1",
        "MAX_STEPS": 20.0,
        "SAVE_EVERY_STEPS": 10.0
      },
      "required": {
        "checkpoint_every_steps_lte": 10,
        "first_checkpoint_weak_eval_required": true,
        "max_steps_lte": 20,
        "mode": "deferred_post_checkpoint",
        "purpose": "allow one tiny smoke only when checkpoint-5 weak eval and protected-row guards are mandatory",
        "v618_surface_route": true
      }
    },
    "eval_prompt_requires_boxed_only_line": false,
    "expected_abort_max_reserved_gib": 78,
    "expected_data_repo": "felipesp1983/kg1-v708-equation-single-family-dataset",
    "expected_flavor": "a100-large",
    "expected_loss_normalization_mode": "example_mean",
    "expected_max_length": 1024,
    "first_checkpoint_weak_eval_controls": {
      "KG1_DISABLE_THINKING": "0",
      "KG1_ENFORCE_WEAK_RUNTIME_POLICY": "1",
      "KG1_EVAL_CANDIDATE_BY_CANDIDATE": "1",
      "KG1_EVAL_TIMEOUT_S": "4200",
      "KG1_GENERATION_TIMEOUT_S": "900",
      "KG1_MAX_MODEL_LEN": "8192",
      "KG1_MAX_NUM_SEQS": "64",
      "KG1_MAX_TOKENS": "7680",
      "KG1_NO_PROMPT_SUFFIX": "0",
      "KG1_PROTECTED_ROW_GUARD": "1",
      "KG1_REQUIRE_DISABLE_THINKING": "0",
      "KG1_STOP_ON_PROTECTED_BACKFIRE": "1",
      "KG1_WEAK_EVAL_HARNESS": "scripts/hf_job_weak_eval_v245.py",
      "KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX": "512",
      "KG1_WEAK_PROMOTE_MAX_COMPLETION_TOKENS_MAX": "7680"
    },
    "first_checkpoint_weak_eval_required": true,
    "launcher": "artifacts\\v712_hf_a100_equation_signal_launch\\launch_v712_hf_a100_equation_signal.py",
    "lora_target_parameters_contract": {
      "freeze_lora_target_parameters": true,
      "has_lora_target_parameters": true,
      "require_lora_target_parameters_trainable": false,
      "target_parameters_literal": "mlp.experts.gate_up_proj,mlp.experts.down_proj"
    },
    "require_row_loss_weight": true,
    "residual_first_gpu_gate": {
      "observed": {
        "KG1_ADAPTER_CPU_FORMAT_PARITY_STATUS": "passed",
        "KG1_CPU_EXTRACTOR_PARITY_STATUS": "passed",
        "KG1_CPU_MISS_CLASSIFICATION_COVERAGE": 1.0,
        "KG1_CPU_SIMULATED_BIT_CORRECT": 136.0,
        "KG1_CPU_SIMULATED_EQUATION_CORRECT": 60.0,
        "KG1_CPU_SIMULATED_LOST_BIT_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_EQUATION_ROWS": 0.0,
        "KG1_CPU_SIMULATED_LOST_ROWS": 0.0,
        "KG1_CPU_SIMULATED_TOTAL_CORRECT": 196.0,
        "KG1_CPU_SIMULATION_USES_WEAK_LABELS": "0",
        "KG1_EXPECTED_TRUNCATED": "0",
        "KG1_MAX_TOKEN_HEADROOM_RATIO": 0.371,
        "KG1_PROMPT_TEMPLATE_PARITY_STATUS": "passed",
        "KG1_PROTECTED_ID_ANSWERS": "8740ed31=01101000,59bee375=10010101,55d834d1=00111111",
        "KG1_RESIDUAL_FIRST_GATE": "1",
        "KG1_STALE_PREDICTION_PARITY_STATUS": "passed",
        "KG1_V516_PARSER_CURRENT_BASELINE_STATUS": "passed",
        "KG1_V536_VAL_STATS_AS_WEAK_EVIDENCE": "0",
        "KG1_V540_EXTRACTION_GATE_STATUS": "passed",
        "KG1_V541_FLIP_LEDGER_STATUS": "passed",
        "KG1_V541_MISSMAP_GATE_STATUS": "passed",
        "KG1_WEAK_LABEL_AWARE_SELECTION": "0"
      },
      "required": {
        "cpu_extractor_parity_status": "passed",
        "cpu_miss_classification_coverage_min": 0.7,
        "cpu_simulated_bit_min": 136,
        "cpu_simulated_equation_min": 60,
        "cpu_simulated_lost_bit_rows_max": 0,
        "cpu_simulated_lost_equation_rows_max": 0,
        "cpu_simulated_lost_rows_max": 0,
        "cpu_simulated_total_min": 196,
        "cpu_simulation_uses_weak_labels": "0",
        "expected_truncated": 0,
        "max_token_headroom_ratio_max": 0.9,
        "prompt_template_parity_status": "passed",
        "protected_rows": [
          "8740ed31=01101000",
          "59bee375=10010101",
          "55d834d1=00111111"
        ],
        "stale_prediction_parity_status": "passed",
        "v516_parser_current_baseline_status": "passed",
        "v536_val_stats_as_weak_evidence": "0",
        "v540_extraction_gate_status": "passed",
        "v541_flip_ledger_status": "passed",
        "v541_missmap_gate_status": "passed",
        "weak_label_aware_selection": "0"
      }
    },
    "row_loss_weight_flag_counts": {
      "--require-row-loss-weight": 3,
      "--require-validation-row-loss-weight": 3,
      "--use-row-loss-weight": 3
    },
    "v666_cpu_gate_launcher_contract": {
      "declared_report": "artifacts/v708_hf_a100_launch/v708_cpu_gate_stack.json",
      "observed_status": "passed",
      "required_status": "passed"
    }
  },
  "learnability_manifest": {
    "reason": "learnability_manifest_not_provided",
    "skipped": true
  },
  "ok": true,
  "preference_manifest": {
    "reason": "sft_schema_does_not_use_preference_audit",
    "skipped": true
  },
  "schema_version": "kg1_pre_paid_job_integration_gate_v2",
  "tokenization_manifest": {
    "manifest": "artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v286_tokenization_real\\v286_generic_tokenization_gate_manifest.json",
    "manifest_max_length": 8192,
    "runtime_expected_max_length": 1024,
    "runtime_length_safe": true,
    "status": "tokenization_gate_passed",
    "train_token_max": 379,
    "validation_token_max": 371
  },
  "train_dataset": {
    "assistant_boxed_only_rows": 0,
    "assistant_final_answer_only_rows": 0,
    "assistant_length_stats": {
      "equation_transform": {
        "chars_max": 505,
        "chars_p50": 388,
        "chars_p95": 501,
        "rows": 852
      }
    },
    "assistant_multiline_rows": 852,
    "assistant_prefix_counts": {
      "other": 852
    },
    "assistant_rule_prefix_rows": 0,
    "assistant_trace_rows": 120,
    "bad_rows_first30": [],
    "expected_aware_signal_rows_first30": [],
    "family_counts": {
      "equation_transform": 852
    },
    "negative_type_counts": {},
    "path": "artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v708_equation_single_family_train.jsonl",
    "rows": 852,
    "sha256": "a329115d11cd9dc708822d8978f1e6b68711c1c01c63df3440de336ab16edc5d",
    "subcategory_counts": {
      "equation_numeric_add_direct_low_support": 80,
      "equation_numeric_colon_absdiff_unreverse_low_support": 80,
      "equation_numeric_minus_signed_reverse_high_support": 80,
      "equation_numeric_minus_signed_reverse_low_support": 120,
      "equation_symbolic_cryptarithm_single_operator_mul": 240,
      "symbolic_cryptarithm_multi_operator_digits_add": 40,
      "symbolic_cryptarithm_multi_operator_digits_mul": 40,
      "symbolic_cryptarithm_single_operator_digits_mul": 40,
      "v640_lkevin_equation_symbolic_trace": 132
    }
  },
  "v438_audit": {
    "skipped": true
  },
  "v666_cpu_gate": {
    "blockers": [],
    "check_count": 8,
    "decision": "gpu_allowed",
    "failed_checks": [],
    "gpu_allowed": true,
    "ok": true,
    "path": "artifacts\\v708_hf_a100_launch\\v708_cpu_gate_stack.json",
    "schema_version": "kg1_v708_cpu_gate_stack_v1"
  },
  "validation_dataset": {
    "assistant_boxed_only_rows": 0,
    "assistant_final_answer_only_rows": 0,
    "assistant_length_stats": {
      "equation_transform": {
        "chars_max": 504,
        "chars_p50": 388,
        "chars_p95": 501,
        "rows": 195
      }
    },
    "assistant_multiline_rows": 195,
    "assistant_prefix_counts": {
      "other": 195
    },
    "assistant_rule_prefix_rows": 0,
    "assistant_trace_rows": 30,
    "bad_rows_first30": [],
    "expected_aware_signal_rows_first30": [],
    "family_counts": {
      "equation_transform": 195
    },
    "negative_type_counts": {},
    "path": "artifacts\\v708_equation_single_family_dataset\\20260520T_v708_cpu_gate\\v708_equation_single_family_val.jsonl",
    "rows": 195,
    "sha256": "f3c8160c982283b42f5930f2cf1fad87e7d644112d792cc5fb6f92e0843b2bba",
    "subcategory_counts": {
      "equation_numeric_add_direct_low_support": 20,
      "equation_numeric_colon_absdiff_unreverse_low_support": 20,
      "equation_numeric_minus_signed_reverse_high_support": 20,
      "equation_numeric_minus_signed_reverse_low_support": 30,
      "equation_symbolic_cryptarithm_single_operator_mul": 60,
      "symbolic_cryptarithm_multi_operator_digits_add": 10,
      "symbolic_cryptarithm_multi_operator_digits_mul": 10,
      "symbolic_cryptarithm_single_operator_digits_mul": 10,
      "v640_lkevin_equation_symbolic_trace": 15
    }
  }
}
```


## Extra artifact 3
Path: `artifacts\v712_hf_a100_equation_signal_launch\v712_static_safety_gate_after_importfix.json`

```text
{
  "file_count": 3,
  "files": [
    "artifacts/v712_hf_a100_equation_signal_launch/launch_v712_hf_a100_equation_signal.py",
    "scripts/kg1_pre_paid_job_integration_gate.py",
    "scripts/kg1_static_safety_gate.py"
  ],
  "findings": [],
  "ok": true,
  "schema_version": "kg1_static_safety_gate_v1"
}
```


## Final instruction
Give a surgical answer that can change the next roadmap step. Do not repeat generic ML advice.
Focus on concrete implementation, data, masking, decoding, LoRA contract, validation, and gate changes that can improve ACC safely.

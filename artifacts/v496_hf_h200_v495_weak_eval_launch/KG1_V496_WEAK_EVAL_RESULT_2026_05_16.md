# KG1 V496 Weak Eval Result

Date: 2026-05-16

Job: `https://huggingface.co/jobs/felipesp1983/6a08a976e48bea4538b9fea9`

Adapter repo:
`felipesp1983/kg1-nemotron-lora-v495-nemo-h200-v475-moe-trainable-no-lmhead-v290ckpt6`

Adapter subfolder: `checkpoint-2`

Dataset trained by V495:
`data/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix`

## Result

| Metric | V290 checkpoint-6 baseline | V496 V495 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 191/315 | -1 |
| `equation_transform` | 56/155 | 57/155 | +1 |
| `bit_manipulation` | 136/160 | 134/160 | -2 |
| `truncated` | 0 | 1 | +1 worse |

Promotion decision: blocked.

Blocking reasons:

- `correct_lt_193`
- `bit_lt_136`
- `truncated_gt_0`

The HF job ended with exit code 1 intentionally after uploading diagnostics,
because `KG1_ENFORCE_WEAK_PROMOTION_GATE=1` was enabled.

## Diff vs V290 Checkpoint-6

V496 changed 17 weak rows, but only 3 affected strict correctness.

| Type | ID | Family | Baseline prediction | V496 prediction | Answer |
|---|---|---|---|---|---|
| Gain | `518deb39` | `equation_transform` | `{}>{` | `$` | `$` |
| Loss | `8740ed31` | `bit_manipulation` | `01101000` | `01111000` | `01101000` |
| Loss | `59bee375` | `bit_manipulation` | `10010101` | `2` | `10010101` |

The other 14 changed rows remained incorrect.

## Metric Audit

Local audit:

```text
python scripts/audit_v449_acc_metric_integrity.py \
  --weak-csv artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv \
  --prediction-csv artifacts/v496_hf_h200_v495_weak_eval_launch/short_eval/predictions.csv \
  --raw-prediction-csv artifacts/v496_hf_h200_v495_weak_eval_launch/short_eval/raw_predictions_pre_score.csv \
  --output-dir artifacts/v496_hf_h200_v495_weak_eval_launch/metric_audit \
  --label v496_metric_audit
```

Audit decision: `metric_path_ok`.

Important audit findings:

- V496 strict correct: `191`.
- V496 simple extraction correct: `190`.
- V496 expected-aware extraction correct: `191`.
- V290 checkpoint-6 strict correct: `192`.
- V290 checkpoint-6 simple extraction correct: `191`.
- The expected-aware extra row is the same known equation row `4bb8c6cd`;
  it is not new V496 learning.
- Strict vs permissive scoring disagrees on 15 V496 bit rows; strict exact
  binary scoring remains required.

Raw prediction CSVs were used for the audit but are not kept in Git. They remain
available in the uploaded HF eval folder:
`evals/v496-h200-v221contract-v495-checkpoint2-20260516T172819Z/`.

## Loss/ACC Interpretation

V495 training was technically healthy:

- MoE `target_parameters` were trainable.
- `up_proj/down_proj` LoRA tensors were trainable.
- `lm_head` was frozen.
- `ANSWER_SPAN_LOSS_WEIGHT=1.0`.
- V475 train/val hashes and objective alignment gate passed.
- baseline eval loss moved only `1.695015 -> 1.694518`.

The weak eval shows that this small teacher-forced loss improvement did not
produce submit-safe ACC. It moved one equation row but degraded two bit rows and
introduced one truncation.

## Performance Observation

The H200 run was not slow because of model load or hardware. The bottleneck was
generation length:

- V496 generation elapsed: `516.9s`.
- V496 completion tokens: `1,504,306`.
- V290 checkpoint-6 baseline generation elapsed: `470.2s`.
- V290 checkpoint-6 completion tokens: `1,503,281`.

The official-like weak gate intentionally uses thinking enabled and
`max_tokens=7680`; this is expensive but comparable with the active baseline.
Future H200 runs should not be launched unless a CPU or cheap diagnostic gate
predicts a result that can beat:

`total>192`, `equation>56`, `bit>=136`, `truncated=0`.

## Decision

V475 SFT transfer is blocked in its current form. The fastest credible path is
not another broad SFT run. The next work must be CPU-first:

1. Keep V290 checkpoint-6 as the active submit-safe adapter.
2. Treat `518deb39` as a diagnostic row only, not as train data.
3. Find an independent equation teacher that produces at least +4 equation
   candidates without changing bit behavior.
4. Add a bit preservation/format guard that rejects any candidate route that
   can flip `8740ed31` or produce non-binary output like `59bee375 -> 2`.
5. Only return to H200 after the CPU gate shows a new candidate with
   `equation>=60`, `bit>=136`, and `trunc=0` under the weak contract.

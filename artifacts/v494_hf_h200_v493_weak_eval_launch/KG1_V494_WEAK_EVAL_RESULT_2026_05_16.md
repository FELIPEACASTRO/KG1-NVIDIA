# KG1 V494 Weak Eval Result

Date: 2026-05-16

Job: `https://huggingface.co/jobs/felipesp1983/6a089f4fe48bea4538b9fe35`

Adapter repo:
`felipesp1983/kg1-nemotron-lora-v493-nemo-h200-moe-trainable-no-lmhead-v290ckpt6`

Adapter subfolder: `checkpoint-2`

## Result

| Metric | V290 checkpoint-6 baseline | V494 checkpoint-2 | Delta |
|---|---:|---:|---:|
| Total weak | 192/315 | 190/315 | -2 |
| `equation_transform` | 56/155 | 57/155 | +1 |
| `bit_manipulation` | 136/160 | 133/160 | -3 |
| `truncated` | 0 | 1 | +1 worse |

Promotion decision: blocked.

Blocking reasons:

- `correct_lt_193`
- `bit_lt_136`
- `truncated_gt_0`

The HF job ended with exit code 1 intentionally because
`KG1_ENFORCE_WEAK_PROMOTION_GATE=1` was enabled and no candidate passed.

## Loss/ACC Interpretation

V493 training had a small teacher-forced masked-CE improvement:

- baseline eval loss: `1.9233`
- final eval loss: `1.9152`

That did not transfer to ACC. This confirms the V494 audit rule: `eval_loss`
is diagnostic only and cannot be used as a promotion proxy. Promotion must use
generation, extraction, strict `verify_answer`, truncation, and per-family weak
or full gates.

## Metric Audit

Local audit:

```text
python scripts/audit_v449_acc_metric_integrity.py \
  --weak-csv artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv \
  --prediction-csv artifacts/v494_hf_h200_v493_weak_eval_launch/short_eval/predictions.csv \
  --raw-prediction-csv artifacts/v494_hf_h200_v493_weak_eval_launch/short_eval/raw_predictions_pre_score.csv \
  --output-dir artifacts/v494_hf_h200_v493_weak_eval_launch/metric_audit \
  --label v494_metric_audit
```

The raw prediction CSVs used by this audit are not duplicated in Git because
they contain multi-MB raw generations. They remain available in the uploaded HF
eval folder:
`evals/v494-h200-v221contract-v493-checkpoint2-20260516T164501Z/`.

Audit decision: `metric_path_ok`.

Important audit findings:

- `strict_correct=190`.
- `simple_correct=189`.
- `expected_aware_correct=190`.
- `expected_aware_minus_simple_correct=1`.
- The extra expected-aware row is `4bb8c6cd`, family `equation_transform`.
- Strict vs permissive scoring disagrees on 15 bit rows, all in
  `bit_manipulation`; strict exact binary scoring is required.

## Diff vs V290 Checkpoint-6

V494 has one real gain and three real losses vs the active submit-safe weak
baseline:

| Type | ID | Family | Baseline prediction | V494 prediction | Answer |
|---|---|---|---|---|---|
| Gain | `518deb39` | `equation_transform` | `{}>{` | `$` | `$` |
| Loss | `5b9964c7` | `bit_manipulation` | `00011011` | `00001011` | `00011011` |
| Loss | `8740ed31` | `bit_manipulation` | `01101000` | `01111000` | `01101000` |
| Loss | `59bee375` | `bit_manipulation` | `10010101` | `2` | `10010101` |

## Decision

The V493/V494 mechanism test succeeded technically but failed competitively:

- MoE `target_parameters` were trainable.
- `lm_head` was frozen.
- answer-span loss weight was conservative.
- long-context weak eval ran with the intended controls.
- the result still regressed bit and introduced truncation.

Therefore the active plan must not spend more GPU on broad SFT over the V290
lineage until a CPU teacher/verifier shows a new, non-leaking signal that keeps
`bit_manipulation>=136` and `truncated=0`.

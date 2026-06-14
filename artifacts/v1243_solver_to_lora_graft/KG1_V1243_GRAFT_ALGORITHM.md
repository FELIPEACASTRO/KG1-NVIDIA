# KG1 V1243 GRAFT Algorithm

Generated: 2026-06-13

## Name

`GRAFT`: Gate-verified Replay Answer-Focused Transfer.

## Goal

Transform verified `bit_manipulation` and `equation_transform` solver knowledge into a LoRA
adapter update, without using solver/runtime postprocessing at submission time and without false
positive promotion.

The adapter must learn to emit the score-facing final answer under the official prompt format.

Baseline reference:

- Full947: `823/947`.
- Bit: `135/160`.
- Equation: `56/155`.
- Protected families: `632/632`.

Claim targets:

- `0.89`: V1241 `full947_089`, `>=843/947`.
- `0.90`: V1241 `full947_090`, `>=853/947`.

## Core Invention

The solver should not be copied into runtime. It should be projected into LoRA through a narrow
gradient signal:

1. The input remains the original puzzle prompt.
2. The target is a short score-facing continuation:

```text
</think>
\boxed{answer}
```

3. The loss should focus on the final boxed payload rather than long teacher reasoning.
4. Bit and equation are trained as specialists first, with protected replay in every phase.
5. No specialist is accepted unless real generations pass raw-output gates.

This intentionally differs from FASE5:

- no missing top-level gold;
- no disabled gate as proof;
- no broad noisy mix;
- no logprob-only promotion;
- no Kaggle submit from weak gates.

## Mathematical Objective

For a training row `x_i, y_i, f_i`, where:

- `x_i` is the official puzzle prompt;
- `y_i` is the verified target continuation;
- `f_i` is the family;
- `M_i(t)` is the completion mask;
- `B_i(t)` is the boxed-payload mask;
- `w_f` is the family/phase weight;
- `theta_0` is the frozen base+baseline adapter;
- `Delta_phi` is the trainable LoRA delta.

The preferred objective is:

```text
L(phi) =
  mean_i w_f(i) *
  sum_t M_i(t) * (1 + lambda_box * B_i(t)) *
  CE(p_{theta_0 + Delta_phi}(token_t | x_i, y_<t), token_t)
  /
  sum_t M_i(t) * (1 + lambda_box * B_i(t))
```

Recommended first values:

- `lambda_box = 4` for bit/equation.
- `lambda_box = 2` for protected replay.
- If the trainer cannot apply token weights yet, approximate this using short targets and weighted
  replacement sampling.

Promotion metric is not this loss. Promotion metric is V1241 strict raw-output accuracy.

## Dataset Construction

The implemented builder creates:

- `v1243_bit_specialist_train.jsonl`
- `v1243_equation_specialist_train.jsonl`
- `v1243_protected_replay_train.jsonl`
- `v1243_micro_consolidation_train.jsonl`
- `v1243_val170.jsonl`

### Bit Specialist

Rows:

- 540 V1240 bit rows.
- 184 protected replay anchors.
- Total: 724.

Purpose:

- Teach answer emission for byte-rule inference.
- Preserve solved families through replay.

Weights:

- Bit rows: `1.35`.
- Protected replay: `0.85`.

### Equation Specialist

Rows:

- 360 V1240 equation rows.
- 184 protected replay anchors.
- Total: 544.

Purpose:

- Teach string-faithful numeric/reversal equation behavior.
- Keep symbolic/parser-fragile material out of first risky expansion.

Weights:

- Equation rows: `1.65`.
- Protected replay: `0.95`.

### Micro Consolidation

Rows:

- 540 bit.
- 360 equation.
- 184 protected replay.
- Total: 1084.

Purpose:

- Only after specialists pass, run a very small low-LR consolidation if needed.

Weights:

- Bit: `1.05`.
- Equation: `1.20`.
- Protected: `1.0`.

## Hyperparameter Ladder

### Bit Probe

- Warm-start: best proven V291/submit086/s160-grade adapter only.
- Rank: `8`.
- Alpha: `16`.
- Dropout: `0.0`.
- LR: `3e-6` start, decay to `1e-6`.
- Max steps: `60`.
- Eval every: `10`.
- Max length: `2048`.
- Sampling: weighted replacement.
- Upload: off for dry-run.
- GPU authorization: false until all dry-run gates pass.

### Equation Probe

- Warm-start: baseline, or bit specialist only if bit passed gates.
- Rank: `8`.
- Alpha: `16`.
- Dropout: `0.0`.
- LR: `2e-6` start, decay to `8e-7`.
- Max steps: `50`.
- Eval every: `10`.
- Max length: `2048`.
- Sampling: weighted replacement.
- Upload: off for dry-run.

### Trainable Modules

Initial conservative allowlist:

```text
down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj
```

Do not widen target modules until trainability and raw-output gates show this is necessary.

## Merge Policy

Default: do not merge.

Preferred order:

1. Select the best passing specialist checkpoint.
2. If both specialists pass independently, try micro consolidation.
3. If a merge is necessary, sweep small coefficients and gate each candidate:

```text
Delta = alpha_bit * Delta_bit + alpha_eq * Delta_eq
alpha_bit, alpha_eq in {0.25, 0.50, 0.75, 1.00}
```

Reject any merge with:

- bit loss;
- equation loss;
- protected loss;
- formatting loss;
- truncation;
- public-metric-only gain.

## Gates

### Pre-GPU

- V1243 builder passes.
- V1240 manifest decision is valid.
- All rows have top-level `answer`.
- Exactly one closed boxed target per row.
- Train/val prompt hash overlap is zero.
- Tokenization/mask dry-run proves completion tokens are unmasked.
- Adapter load contract passes.
- `UPLOAD_TO_HF=0` and `DRY_RUN_VALIDATE_ONLY=1` for first validation job.

### After Bit Specialist

- Real raw generations exist.
- V1241 `tiny` passes.
- V1241 `val170` passes.
- Bit improves by at least +1 on strict gate.
- Equation does not regress.
- Protected does not regress.

### After Equation Specialist

- Real raw generations exist.
- V1241 `tiny` passes.
- V1241 `val170` passes.
- Equation improves by at least +1.
- Bit does not regress.
- Protected does not regress.

### Before 0.89 or 0.90 Claim

- Full947 raw generations exist.
- Every row has `raw_output`.
- Candidate headline score equals strict closed-boxed score.
- V1241 `full947_089` or `full947_090` passes.
- Protected families remain `632/632`.

## Silent Bugs This Algorithm Blocks

- Missing top-level `answer`.
- Multiple boxed answers.
- Unclosed boxed answers.
- Truncated generations.
- Public extractor accepts but strict extractor rejects.
- Symbolic equation braces/backslashes escaping incorrectly.
- Train/val overlap by prompt hash.
- Eval loss improves while exact raw-output ACC drops.
- Adapter partially loads and silently loses LoRA tensors.
- Protected family behavior erodes during weak-family transfer.

## First Implementation Status

Implemented:

- CPU builder: `scripts/kg1_v1243_solver_to_lora_graft_builder.py`.
- Generated packs under `artifacts/v1243_solver_to_lora_graft/`.
- Manifest: `kg1_v1243_solver_to_lora_graft_manifest.json`.
- HF env preview: `v1243_hf_env_preview.json`.
- Trainer token-level boxed-payload weighting: `scripts/hf_job_train_v90.py`.
- CPU trainer contract gate: `scripts/kg1_v1243_graft_trainer_contract_gate.py`.
- Contract report: `artifacts/v1243_solver_to_lora_graft_contract_gate/kg1_v1243_graft_trainer_contract_gate.json`.
- Specialist LoRA target modules are locked to `down_proj,in_proj,k_proj,o_proj,q_proj,up_proj,v_proj`
  in both `LORA_TARGET_MODULES` and `TRAINABLE_LORA_MODULES`; `lm_head` and `out_proj`
  are intentionally excluded for V1243 specialist probes.
- Row-level sampling weights are now preserved through tokenization and consumed by
  `weighted_replacement`: bit specialist share is `0.823357` and equation specialist
  share is `0.772633` in the latest tokenize-only dry-runs.
- Bit tokenize-only dry-run: `artifacts/v1243_solver_to_lora_graft_tokenize_dryrun/bit/dry_run_model_recipe_report.json`.
- Equation tokenize-only dry-run: `artifacts/v1243_solver_to_lora_graft_tokenize_dryrun/equation/dry_run_model_recipe_report.json`.

Not implemented yet:

- In-training V1241 callback.
- Real GPU training.
- Real raw-output generation for V1243 candidates.
- Any submit authorization.

## Decision

`GRAFT` is ready as a CPU data-preparation and training-contract artifact.

It is not yet a trained adapter and not score evidence.

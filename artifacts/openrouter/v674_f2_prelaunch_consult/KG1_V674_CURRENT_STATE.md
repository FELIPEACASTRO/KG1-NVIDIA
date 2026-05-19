# KG1 V674 Current State For OpenRouter F2 Prelaunch Review

Generated for the 2026-05-19 prelaunch decision. This file is an evidence
block for the OpenRouter prompt; it is not a promotion gate by itself.

## Competition And Objective

- Competition: Kaggle NVIDIA Nemotron Model Reasoning Challenge.
- Submission goal: adapter-only, submit-safe accuracy improvement.
- Metric that matters: final-answer accuracy through label-free extraction and
  `verify_answer`, not eval_loss alone.
- Current weak plateau: `192/315`.
- Active family floors:
  - `bit_manipulation >= 136/160`.
  - `equation_transform >= 60/155`.
  - `total >= 196/315`.
  - `truncated = 0`.
  - `no_box_fallback = 0`.
  - `boxed_rate = 1.0`.
  - protected-row backfire = `0`.
- Current actionable CPU target for V673: simulated `196/315`, with
  `bit=136/160`, `equation=60/155`, lost rows `0`, and no weak-label training.

## Active Operational Rules

- H200 is blocked. Use `a100-large` only.
- No Kaggle submit unless explicitly requested after gates pass.
- Keep `official_like` intact.
- False gains are forbidden. A candidate must pass label-free extraction,
  `verify_answer`, protected-row guard, truncation/box gates, hash checks, and
  LoRA adapter-only checks.
- If a paid job fails or finishes, build a complete OpenRouter prompt with the
  current result before authorizing another paid route.
- Launch code must be committed and pushed before HF job launch because the
  launcher records the current `HEAD` as `EXPECTED_COMMIT` and the remote job
  checks out that commit.

## Recent F2 Bugs Found And Fixed

1. Adapter-only save risk:
   - Canceled A100 job `6a0cada33aba298b21d14304` because PEFT warned that
     `save_embedding_layers=True` would be used automatically due to `lm_head`
     appearing in `target_modules`.
   - This was a real silent packaging bug risk: the adapter could include base
     embeddings/lm_head and stop being clean adapter-only.
   - Fix: `scripts/hf_job_train_v90.py` now saves all checkpoints and final
     output through `save_adapter_only(model, output_dir)`, which calls
     `model.save_pretrained(..., save_embedding_layers=SAVE_EMBEDDING_LAYERS)`.
   - Active default and launcher value: `SAVE_EMBEDDING_LAYERS=0`.

2. Row-loss weighting was silently canceled:
   - With `MICRO_BATCH_SIZE=1`, `LOSS_NORMALIZATION_MODE=example_mean`, and the
     old weighted mean `(per_example_loss * weight).sum() / weight.sum()`, the
     row weight canceled out in every microbatch.
   - This means equation/bit row weights could look configured but have no
     effect on gradient scale.
   - Fix: `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`. Under `example_mean`, the
     denominator is the unweighted active example count, so a single example
     with `loss_weight=2.0` doubles the loss.
   - Self-test in `hf_job_train_v90.py` proves this behavior.

3. Validation loss alignment:
   - Train and validation now both use row-loss weights when row-loss weighting
     is active. This prevents best-loss decisions from measuring a different
     distribution than the optimizer.
   - Loss is still not a promotion metric by itself; it is only useful if ACC
     and protected-row gates agree.

4. Stale threshold:
   - `RESIDUAL_FIRST_MIN_EQUATION` and all active V673 promotion gates are now
     `60`, not `59`.
   - Required floors: total `196`, bit `136`, equation `60`, truncation `0`.

5. Dataset metadata contamination:
   - V673 dataset assistant targets are trace-plus-final-boxed, not boxed-only.
   - Builder fixed `metadata.completion_format` to
     `trace_plus_final_boxed`.
   - New active dataset has `bad_boxed_only_trace=0`.

6. A100 memory guard:
   - Previous A100 reserved memory went above the old 72 GiB abort line without
     proving true OOM.
   - Active launcher uses `ABORT_MAX_RESERVED_GIB=78`.

7. Dry-run report observability:
   - Tokenize-only dry-run report now includes structured
     `tokenization.train` and `tokenization.validation` counters:
     `prompt_truncated`, `fallback_masks`, `completion_tokens_dropped`,
     `offset_masks`, `row_loss_weight_*`.

## Active Dataset V673

- Local root:
  `artifacts/v673_guarded_equation_bit_transfer_dataset/20260519T190246Z`.
- Train JSONL:
  `v673_guarded_equation_bit_transfer_train.jsonl`.
- Validation JSONL:
  `v673_guarded_equation_bit_transfer_val.jsonl`.
- Train rows: `720`.
- Validation rows: `180`.
- Train family counts:
  - `equation_transform`: `480`.
  - `bit_manipulation`: `240`.
- Validation family counts:
  - `equation_transform`: `120`.
  - `bit_manipulation`: `60`.
- Train subcategories:
  - `equation_numeric_minus_signed`: `240`.
  - `equation_numeric_add_direct`: `120`.
  - `equation_numeric_colon_trailing_zero`: `120`.
  - `bit_exact_global_ternary_replay`: `96`.
  - `bit_fullbyte_ternary_v366_new`: `96`.
  - `bit_exact_global_binary_replay`: `48`.
- Validation subcategories:
  - `equation_numeric_minus_signed`: `60`.
  - `equation_numeric_add_direct`: `30`.
  - `equation_numeric_colon_trailing_zero`: `30`.
  - `bit_exact_global_ternary_replay`: `24`.
  - `bit_fullbyte_ternary_v366_new`: `24`.
  - `bit_exact_global_binary_replay`: `12`.
- Train SHA256:
  `69f76195e2a004de5c01c919038210da0987b67476911ca706e7ba9b4160477f`.
- Validation SHA256:
  `df2d44e334de65cb91da935768db93f4727f700edd762dd9fd6d48b3d5d8d14b`.
- HF dataset repo:
  `felipesp1983/kg1-v673-guarded-equation-bit-transfer-artifacts`.
- HF dataset root:
  `v673-guarded-equation-bit-transfer-20260519T190246Z`.
- HF upload commit:
  `f729f85dded2bc0a680b85059b60f2e267ae4c6e`.

## Active Training Recipe V673

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Base revision: `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.
- Parent adapter:
  `felipesp1983/kg1-nemotron-lora-v290-rank19-micro-patch-smoke`,
  subfolder `checkpoint-6`.
- Output repo:
  `felipesp1983/kg1-nemotron-lora-v673-a100-guarded-eqbit-v290ckpt6`.
- HF flavor: `a100-large`.
- Max job time: one hour.
- Train max steps: `20`.
- Save/eval every: `10`.
- `MAX_LENGTH=1024` for training.
- Eval official-like contract remains `max_tokens=7680`,
  `temperature=0`, `top_p=1`.
- Loss normalization: `example_mean`.
- `USE_ROW_LOSS_WEIGHT=1`.
- `REQUIRE_ROW_LOSS_WEIGHT=1`.
- `REQUIRE_VALIDATION_ROW_LOSS_WEIGHT=1`.
- `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`.
- `SAVE_EMBEDDING_LAYERS=0`.
- `LOSS_MASK_STOP_AFTER_EOS=1`.
- `SAMPLING_MODE=weighted_replacement`.
- LoRA:
  - `r=32`.
  - `alpha=32`.
  - dropout `0`.
  - target modules include attention and MLP adapter surfaces as declared by
    the V673 launcher.
  - MoE target parameters:
    `mlp.experts.gate_up_proj,mlp.experts.down_proj`.
  - `REQUIRE_LORA_TARGET_PARAMETER_MATCH=1`.
  - `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE=1`.
- Protected rows:
  - `8740ed31=01101000`.
  - `59bee375=10010101`.
  - `55d834d1=00111111`.

## Gates That Passed After Fixes

- V286 tokenizer-real gate:
  - `status=tokenization_gate_passed`.
  - `prompt_truncated=0`.
  - `fallback_masks=0`.
  - `completion_tokens_dropped=0`.
  - max token length `335`.
- V509 dataset integrity:
  - `status=datasets_pass_integrity_audit`.
  - `dataset_count=2`.
  - `blocked_dataset_count=0`.
- V513 learnability:
  - `status=passed_cpu_structure_only`.
  - `blocker=0`.
  - `warning=0`.
- V478 objective:
  - `hf_gpu_allowed=true`.
  - bit effective share `0.148936`.
  - equation effective share `0.851064`.
- EOS/loss mask:
  - final-loss EOS rate `1.0`.
  - no no-loss rows.
- Static safety gate, active files only:
  - `ok=true`.
  - `findings=[]`.
- Pre-paid job integration:
  - `ok=true`.
  - `findings=[]`.
  - confirms `a100-large`, dataset hashes, `ABORT_MAX_RESERVED_GIB=78`,
    `SAVE_EMBEDDING_LAYERS=0`, and
    `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`.
- V666 CPU gate stack:
  - `gpu_allowed=true`.
  - `blockers=[]`.

## Known Failure History To Keep In Mind

- V664 reached only `192/315`, `bit=136/160`, `equation=56/155`, with long
  completions and protected backfire. Loss movement alone did not imply ACC.
- V661 checkpoint-2 regressed to `191/315`, `bit=135/160`,
  `equation=56/155`, `truncated=1`, `no_box_fallback=1`, and protected row
  backfire.
- V653/V660/V661/V662/V663/V664 are frozen as promotional routes.
- Broad H200 exploratory training is removed from the plan.
- Runtime solver/verifier/postprocessor is not acceptable as an adapter-only
  submission route.

## Questions For The External Model

Return only concrete, falsifiable engineering guidance:

1. Is there any remaining mechanism by which the V673 route could produce a
   false gain or hidden backfire despite the gates above?
2. Does `ROW_LOSS_WEIGHT_REDUCTION=scale_mean` correctly fix the microbatch-1
   cancellation issue, or should the weighted objective be implemented
   differently for train and validation?
3. Does `SAVE_EMBEDDING_LAYERS=0` fully close the adapter-only packaging risk
   given that `lm_head` appears in `target_modules`, or should `lm_head` be
   removed from target modules as an additional safety step?
4. Are the active thresholds (`total>=196`, `bit>=136`, `equation>=60`,
   truncation/fallback/protected=0) sufficient, or is any gate missing to
   distinguish bad decoding from adapter drift toward wrong answers?
5. Given the budget constraint and H200 ban, should we launch the bounded
   A100 V673 smoke now, or block and run another CPU-only diagnostic first?

Required response format:

```json
{
  "verdict": "proceed|block|needs_one_more_cpu_gate",
  "top_risks": [
    {
      "risk": "...",
      "evidence": "...",
      "required_fix_or_gate": "...",
      "blocks_a100_launch": true
    }
  ],
  "parameters_to_freeze": {
    "SAVE_EMBEDDING_LAYERS": "0",
    "ROW_LOSS_WEIGHT_REDUCTION": "scale_mean"
  },
  "parameters_to_change_before_launch": {},
  "cheapest_next_action": "...",
  "delete_from_roadmap": ["..."]
}
```

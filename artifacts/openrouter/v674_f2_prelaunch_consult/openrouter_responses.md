# KG1 Post-Train OpenRouter Responses

## openai/gpt-5.5

- status: `ok`

1. **Verdict: block**

Block the current V673 A100 launch **as configured**. Unblock only after a no-cost launch/LoRA-contract patch and rerun of the existing static/pre-paid gates. The CPU gate evidence is good enough for **one bounded A100-large smoke**, but not with `lm_head` risk and broad MLP/MoE trainability still active.

---

2. **Top 5 concrete bugs or gaps**

1. **`lm_head` still appears in `target_modules`; `SAVE_EMBEDDING_LAYERS=0` is not enough by itself.**
   - Evidence: Extra artifact asks whether `SAVE_EMBEDDING_LAYERS=0` fully closes risk “given that `lm_head` appears in `target_modules`.” Earlier A100 was canceled because PEFT warned `save_embedding_layers=True` automatically due to `lm_head`.
   - Action: remove `lm_head` from LoRA target modules before launch. Add a hard gate that fails if `adapter_config.json.target_modules` or safetensors keys contain `lm_head`, `embed_tokens`, `word_embeddings`, or full embedding/lm-head tensors.

2. **Current LoRA surface is too broad for a micro-rescue and conflicts with the failure history.**
   - Evidence: active recipe trains `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj` plus MoE target parameters `mlp.experts.gate_up_proj,mlp.experts.down_proj`; previous broad/MoE route V661 caused drift/backfire, and V672 consensus said “sem MoE” / `q_proj/v_proj` first. Runtime trainable fraction was `2.6776%`.
   - Action: for this rescue, freeze MLP/MoE. Use attention-only. Recommended safest launch: `q_proj,v_proj` only. If you choose `q,k,v,o`, require an explicit trainable-fraction gate and protected-row generated smoke before full weak eval.

3. **Dataset surface still teaches multiline trace before the final box, while the dominant failure mode is runaway generation.**
   - Evidence: manifest says `assistant_multiline_rows=720/720`, `assistant_prefix_counts.other=720`, `assistant_boxed_only_rows=0`, `assistant_final_answer_only_rows=0`; V661/V664 failures had extremely long completions and protected backfire.
   - Action: add a dataset/output-policy gate reporting: exactly one usable `\boxed{...}` per assistant target, payload byte-equal to `answer`, no text after final boxed payload, EOS immediately after final boxed close brace, and first-box token position stats. If this gate is missing, the local artifact missing is:
     `v673 dataset surface/output-policy manifest with exactly_one_boxed, boxed_payload_byte_equal, text_after_boxed, eos_after_boxed, first_box_token_index stats`.

4. **Decoding-vs-adapter-drift gate is deferred, but first checkpoint is too late/large for a drift-prone route.**
   - Evidence: manifest explicitly has `KG1_DECODING_VS_ADAPTER_DRIFT_GATE_STATUS=deferred_post_checkpoint`; previous failures show raw adapter drift/backfire, not extractor error. Current save/eval is every `10` with `MAX_STEPS=20`.
   - Action: make first checkpoint cheaper and earlier: `MAX_STEPS=10`, `SAVE_EVERY_STEPS=5`, `EVAL_EVERY_STEPS=5`, or even `MAX_STEPS=6`, `SAVE/EVAL=2` if the launcher supports it. Do not run to step 20 before seeing generated outputs.

5. **Launch manifest shown does not expose enough rendered training parameters to audit reproducibility.**
   - Evidence: provided pre-paid manifest includes max length, row-loss flags, dataset hashes, and some launcher fields, but not the fully rendered LR, scheduler, warmup, microbatch, grad accumulation, exact `LORA_TARGET_MODULES`, exact `LORA_TARGET_PARAMETERS`, or final PEFT adapter config.
   - Action: before paid launch, produce a rendered-env artifact. Missing local artifact:
     `rendered HF job environment / resolved launcher manifest containing LR, scheduler, warmup, MICRO_BATCH_SIZE, GRAD_ACCUM, LORA_TARGET_MODULES, LORA_TARGET_PARAMETERS, target_modules after PEFT construction, trainable tensor-name sample, trainable fraction, and SAVE_EMBEDDING_LAYERS`.

---

3. **Exact next experiment cheapest and most likely to improve weak ACC**

Do **not** run another CPU diagnostic. The V666 CPU gate already predicts the needed `196/315` with zero simulated losses and no weak-label training.

Run one

## anthropic/claude-sonnet-4.6

- status: `ok`

## Panel Verdict: **PROCEED with one pre-commit verification**

All 8 CPU gates pass. The two canceled jobs (6a0ca9f5, 6a0cada3) were correctly aborted. The fixes are real and the evidence chain is coherent. One unverified contract item (see Bug #1) must be confirmed locally before the HF push, but it does not require a new GPU run.

---

## Top 5 Bugs / Gaps

### Bug 1 — `lm_head` in `target_modules` with `SAVE_EMBEDDING_LAYERS=0` is an unresolved contradiction
**Evidence:** The roadmap states `lm_head` appears in `target_modules` (triggering the PEFT `save_embedding_layers=True` auto-warning that caused the cancellation of job `6a0cada3`). The fix sets `SAVE_EMBEDDING_LAYERS=0`, but the launcher manifest does not confirm that `lm_head` was *removed* from `target_modules`. If `lm_head` remains in `target_modules` and `save_embedding_layers=0` is passed, PEFT ≥0.19 will silently skip saving the lm_head LoRA adapter weights — meaning those weights are trained but not saved, producing a corrupt adapter.

**Required fix:** Either (a) remove `lm_head` from `TRAINABLE_LORA_MODULES` and `REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS` entirely, or (b) confirm via `adapter_config.json` dry-run that `target_modules` in the saved adapter does not contain `lm_head`. This is a CPU-only check. The active launcher file `launch_v673_hf_a100_guarded_eqbit.py` must be inspected for the literal `lm_head` string before commit.

---

### Bug 2 — CPU simulation predicts exactly `196/315` with zero headroom; any single protected-row backfire drops below the gate
**Evidence:** `KG1_CPU_SIMULATED_TOTAL_CORRECT=196`, `KG1_CPU_SIMULATED_LOST_ROWS=0`. The promotion floor is `total>=196`. The V673 dataset contains `bit_exact_global_binary_replay` (48 train rows) targeting row `8740ed31` which backfired in V664. If the adapter moves that row's output even one token, the total drops to 195 and the gate fails.

**Required fix:** The protected-row smoke must be run *on the checkpoint-10 output* before any continuation beyond step 10. The gate must explicitly check `8740ed31`, `59bee375`, `55d834d1` against their locked answers (`01101000`, `10010101`, `00111111`) with zero tolerance. This is already declared but the zero-headroom situation means the first checkpoint eval is not a "nice to have" — it is the only signal that matters.

---

### Bug 3 — `ROW_LOSS_WEIGHT_REDUCTION=scale_mean` fixes the microbatch-1 cancellation, but the validation loss is now on a *different scale* than any prior run's eval_loss
**Evidence:** Prior runs (V290, V664) reported eval_loss in the range 5.82–5.86 under unweighted reduction. V673 validation now applies `scale_mean` with equation weight dominating at 85% share. The absolute eval_loss value will be higher (equation rows have more loss tokens: p50=110 vs bit p50=72). Comparing V673 checkpoint eval_loss to V664's `5.8231` to decide "is this better?" is invalid — the distributions are incomparable.

**Required fix:** Do not use eval_loss magnitude to compare V673 checkpoints against V664 or V290. The only valid checkpoint selector is holdout non-weak ACC (starts_boxed_rate, avg_completion_tokens, protected-row guard). Add an explicit note to the gate: `eval_loss_cross_run_comparison_blocked=true`.

---

### Bug 4 — `MAX_LENGTH=1024` for training but `max_tokens=7680` for official-like eval creates a decoding regime mismatch that is not gated
**Evidence:** Train token max is 335 (well within 1024). But the model is evaluated with `max_tokens=7680`. The adapter is trained on completions of ≤115 loss tokens (equation p95). At inference the model can generate up to 7680 tokens. V664 showed `avg_completion_tokens=4772`, `p99=7350` — the adapter did not learn to stop. V673's `LOSS_MASK_STOP_AFTER_EOS=1` is the intended fix, but there is no gate that verifies the checkpoint-10 output has `avg_completion_tokens<=512` before continuation.

**Required fix:** Add a hard gate at checkpoint-10: `avg_completion_tokens_p95 <= 512` on the non-weak holdout. If this fails, the route is decoding drift (not adapter weight error) and the fix is decoding parameters (`repetition_penalty`, `stop_sequences`), not more training steps.

---

### Bug 5 — The V672 ledger shows only `4` directly usable `equation_numeric` candidates, not `12`
**Evidence:** `v666_cpu_gate` check `v672_residual_miss_ledger` reports `direct_usable_counts: {equation_numeric_miss: 4, bit_residual_miss: 10}` but `trainability_decision_counts: {trainable: 12, trainable_guarded: 2, needs_rule_proof: 15, drop: 7}`. The CPU simulation reaches `equation=60` (the floor), but only 4 equation rows are "direct usable." The remaining 8 "trainable" equation rows require rule proof that is not yet in the ledger. If the adapter generalizes to those 8 rows incorrectly, it can flip currently-correct rows.

**Required fix:** Before claiming `equation>=60` is safe, verify that the 8 non-direct trainable equation rows in the dataset do not overlap with currently-correct V290 baseline rows. Missing artifact: the ledger must show per-row `baseline_correct` flag for all 12 trainable equation rows. If any trainable row is currently correct in V290, it must be moved to `protected-only` or `drop`.

---

## Exact Next Experiment (Cheapest, Highest ACC Probability)

**CPU-only, ~10 minutes, no GPU spend:**

Run `grep -i lm_head artifacts/v673_hf_a100_launch/launch_v673_hf_a100_guarded_eqbit.py` and confirm `lm_head` is absent from `TRAINABLE_LORA_MODULES`. Then open the V672 residual miss ledger and add a `baseline_v290_correct` boolean column for each of the 12 trainable equation rows. Any row where `baseline_v290_correct=true` must be reclassified as `drop` or `protected-only`.

If both checks pass: commit, push, launch A100 V673 with `max_steps=20`, `save/eval=10`. Evaluate checkpoint-10 on non-weak holdout only. Do not run full weak eval until checkpoint-10 passes: `starts_boxed_rate>=0.95`, `avg_completion_tokens_p95<=512`, `protected_backfire=0`, `equation_gain>=2_lines_vs_V290`.

---

## Parameters to Change or Freeze

| Parameter | Value | Action |
|---|---|---|
| `lm_head` in `target_modules` | **remove** | Change before commit |
| `SAVE_EMBEDDING_LAYERS` | `0` | Freeze |
| `ROW_LOSS_WEIGHT_REDUCTION` | `scale_mean` | Freeze |
| `LOSS_MASK_STOP_AFTER_EOS` | `1` | Freeze |
| `ABORT_MAX_RESERVED_GIB` | `78` | Freeze |
| `MAX_STEPS` | `20` | Freeze (do not extend before checkpoint-10 gate) |
| `SAVE_EVERY_STEPS` / `EVAL_EVERY_STEPS` | `10` | Freeze |
| `r` / `alpha` | `32` / `32` | Freeze |
| `LR` | `5e-7` | Freeze for this smoke |
| `MICRO_BATCH_SIZE` | `1` | Freeze (required for `scale_mean` correctness) |
| eval_loss as checkpoint selector | **blocked** | Do not use cross-run |

---

## Gates That Must Pass Before Another Paid GPU Job

In order:

1. **`lm_head` absent from `target_modules`** — CPU grep, 1 minute. Blocks launch if present.
2. **V672 ledger `baseline_v290_correct` column** — all 12 trainable equation rows must have `baseline_v290_correct=false`, or be reclassified. Missing artifact: this column does not exist in the current ledger manifest.
3. **Checkpoint-10 non-weak holdout**: `starts_boxed_rate>=0.95`, `avg_completion_tokens_p95<=512`, `protected_backfire=0`, `equation_gain>=2` vs V290 baseline, `bit>=136`.
4. **Checkpoint-10 protected-row smoke**: exact string match for `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111`.
5. **Label-free == label-aware** on checkpoint-10 holdout (delta must be 0).

Only after all 5 pass: run full weak eval. Only after full weak eval passes all promotion thresholds: submit.

---

## Delete From Roadmap (Noise)

1. **Any reference to comparing V673 eval_loss magnitude against V664/V290 eval_loss** — the `scale_mean` change makes these incomparable. Remove all loss-based cross-run comparisons from the promotion decision tree.
2. **The `equation_symbolic` audit as a today-target** — the ledger shows 87 symbolic misses with no deterministic rule proof. It is correctly listed as "taxonomy only" but still appears in the roadmap narrative as a potential today-gain. Delete it from the active plan entirely; it is noise that adds decision overhead.
3. **The `RESIDUAL_FIRST_MIN_EQUATION=59` threshold** — already replaced by `60` but still referenced in older gate descriptions. Remove all `59` references from active gate configs to prevent a stale threshold from passing a future gate incorrectly.
4. **H200 memory justification analysis** — the A100 abort was at `>72 GiB` without true OOM. The fix (`ABORT_MAX_RESERVED_GIB=78`) is correct. Any further H200 feasibility analysis is wasted effort until A100 proves technically impossible with a logged OOM, not a soft abort.
5. **The `AND_OR` V366 item** — already listed as "out of plan" but still appears in the V672 consensus narrative. Remove from all active documents to prevent it from being re-evaluated in a future OpenRouter prompt as a live option.

## google/gemini-3.1-pro-preview

- status: `ok`

```json
{
  "verdict": "block",
  "top_risks": [
    {
      "risk": "Adapter drift and protected-row backfire due to MoE/MLP LoRA targets.",
      "evidence": "V673 launcher includes 'up_proj,down_proj' and 'mlp.experts.gate_up_proj,mlp.experts.down_proj'. The V662 consensus explicitly identified broad LoRA with MLP/MoE as the root cause of V661's drift, long generations, and protected-row backfire.",
      "required_fix_or_gate": "Remove all MLP and MoE targets from TRAINABLE_LORA_MODULES and LORA_TARGET_PARAMETERS. Restrict to attention-only (q_proj, k_proj, v_proj, o_proj) to safely transfer the short trace format without deep reasoning drift.",
      "blocks_a100_launch": true
    },
    {
      "risk": "Vocabulary distribution shift and adapter-only packaging violation from lm_head.",
      "evidence": "The prompt notes PEFT warned about save_embedding_layers=True because 'lm_head' appeared in target_modules. Even with SAVE_EMBEDDING_LAYERS=0, training LoRA on lm_head risks severe generation length drift and may violate strict adapter-only Kaggle constraints.",
      "required_fix_or_gate": "Explicitly strip 'lm_head' from target_modules in the PEFT config. Add a preflight gate asserting 'lm_head' is not in adapter_config.json.",
      "blocks_a100_launch": true
    },
    {
      "risk": "Parent adapter (V290 checkpoint-6) contamination.",
      "evidence": "V673 uses V290 checkpoint-6 as the base. If V290 was trained with lm_head or MoE modules, continuing it might inherit the exact weight drift that causes runaway generation.",
      "required_fix_or_gate": "Audit V290 checkpoint-6 adapter_config.json. If it contains lm_head or MoE, you must either drop those weights during load or restart from a clean attention-only parent.",
      "blocks_a100_launch": true
    },
    {
      "risk": "Generation length runaway despite short dataset tokens.",
      "evidence": "V664 failed with avg_completion_tokens=4772 despite short targets. V673 dataset has token_max=335, but without strict decoding bounds during the weak eval gate, the model can still ignore the EOS.",
      "required_fix_or_gate": "Enforce avg_completion_tokens <= 512 and max_completion_tokens <= 1024 in the checkpoint-10 weak eval gate. If it exceeds this, the adapter is drifting and must be killed.",
      "blocks_a100_launch": false
    },
    {
      "risk": "Microbatch scale_mean validation mismatch.",
      "evidence": "ROW_LOSS_WEIGHT_REDUCTION=scale_mean correctly fixes the microbatch-1 cancellation for training. However, if validation loss is calculated over the whole validation set at once (or with a different batch size), the scale_mean denominator will differ, making train/val loss magnitudes incomparable.",
      "required_fix_or_gate": "Ensure validation loss reduction uses the exact same scale_mean logic per-batch as training, or normalize both by the sum of weights globally for logging.",
      "blocks_a100_launch": false
    }
  ],
  "parameters_to_freeze": {
    "SAVE_EMBEDDING_LAYERS": "0",
    "ROW_LOSS_WEIGHT_REDUCTION": "scale_mean",
    "LOSS_MASK_STOP_AFTER_EOS": "1",
    "ABORT_MAX_RESERVED_GIB": "78",
    "EXPECTED_FLAVOR": "a100-large"
  },
  "parameters_to_change_before_launch": {
    "TRAINABLE_LORA_MODULES": "q_proj,k_proj,v_proj,o_proj",
    "REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS": "q_proj,k_proj,v_proj,o_proj",
    "LORA_TARGET_PARAMETERS": "",
    "REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE": "0"
  },
  "cheapest_next_action": "Run a local script to strip 'lm_head', 'up_proj', 'down_proj', and 'mlp.experts.*' from the V673 launch config and verify V290 checkpoint-6 does not force them. Once the LoRA config is strictly attention-only, launch the bounded V673 A100 smoke test (max_steps=20).",
  "delete_from_roadmap": [
    "Training MoE/MLP parameters (mlp.experts.gate_up_proj, mlp.experts.down_proj, up_proj, down_proj) in any short-term transfer probe.",
    "Including lm_head in target_modules for adapter-only submissions.",
    "Relying on loss movement alone without the avg_completion_tokens <= 512 gate."
  ]
}
```

## deepseek/deepseek-v4-pro

- status: `ok`

```json
{
  "verdict": "proceed",
  "top_risks": [
    {
      "risk": "No post-training completion‑length cap in promotion gates – long completions can cause timeouts, hidden truncation, and protected‑row backfire.",
      "evidence": "V661 avg_completion_tokens=4775, p99=7350, truncated=1. Current promotion gates only check truncated=0, no avg/max limit.",
      "required_fix_or_gate": "Add mandatory `avg_completion_tokens ≤ 1024` and `max_completion_tokens ≤ 4096` to weak‑eval gate before any submit.",
      "blocks_a100_launch": false
    },
    {
      "risk": "Trace‑plus‑final‑boxed target may still encourage long reasoning, even with training max_length=1024, because LoRA adapters shift distribution toward verbose traces and base model tends to continue.",
      "evidence": "V661/664 trained with trace‑style targets and generated 4k–7k tokens; training max_length alone did not constrain inference length.",
      "required_fix_or_gate": "No change required pre‑launch, but the first checkpoint must pass a strict length gate (see above) before any continuation.",
      "blocks_a100_launch": false
    },
    {
      "risk": "Equation‑heavy objective (85.1% effective share) may regress bit rows below floor (136/160) – protected rows are only 3 of 160 bit rows.",
      "evidence": "V664 reached bit=136 exactly, V673 bit effective share only 0.149; no explicit bit‑preservation gate beyond the three protected rows and overall bit floor.",
      "required_fix_or_gate": "Accept that this is a deliberate trade‑off; gating on total bit≥136 in checkpoint‑10 weak eval is sufficient. No change pre‑launch.",
      "blocks_a100_launch": false
    },
    {
      "risk": "ROW_LOSS_WEIGHT_REDUCTION=scale_mean with microbatch_size=1 and example_mean normalisation is mathematically correct, but the row weights themselves (1.0 for equation, 0.35 for bit) were chosen heuristically – no evidence they minimise bit regression.",
      "evidence": "V666 gate shows effective shares: equation 0.851, bit 0.149, but the per‑row weight ratio was not validated on a holdout bit mix.",
      "required_fix_or_gate": "No change for this first smoke; if bit falls ≤135, next iteration must re‑tune row_loss_weight via a short CPU sweep.",
      "blocks_a100_launch": false
    },
    {
      "risk": "No automated verification that the trace inside each training example leads to the correct final answer – a bug in the rule‑based builder could inject wrong reasoning.",
      "evidence": "V513 learnability gate only checks structure; V509 integrity checks duplicates/hashes. There is no manifest proving that the trace→answer mapping is consistent.",
      "required_fix_or_gate": "Before launch, run a one‑shot CPU audit: extract the final boxed answer from the assistant field of 50 random training rows and compare byte‑equal to the expected answer. If any mismatch, block launch and rebuild dataset.",
      "blocks_a100_launch": true
    }
  ],
  "parameters_to_freeze": {
    "SAVE_EMBEDDING_LAYERS": "0",
    "ROW_LOSS_WEIGHT_REDUCTION": "scale_mean",
    "LOSS_MASK_STOP_AFTER_EOS": "1",
    "MAX_LENGTH": 1024,
    "r": 32,
    "alpha": 32,
    "LR": "5e-7 (as declared, confirm launcher value)",
    "MAX_STEPS": 20,
    "EVAL_EVERY_STEPS": 10,
    "SAVE_EVERY_STEPS": 10
  },
  "parameters_to_change_before_launch": {},
  "cheapest_next_action": "Run a single‑shot CPU script that parses V673 train.jsonl, extracts the final boxed answer from the assistant text (using the official boxed regex), and compares it with the `answer` field. If 100% match in a random sample of 50 rows, launch the A100 smoke immediately. If any mismatch, abort and fix builder. Cost: <1 minute CPU. Then launch V673 smoke as per launcher.",
  "gates_before_next_paid_gpu_after_smoke": [
    "Checkpoint‑10 passes full weak eval with total ≥ 196/315, bit ≥ 136/160, equation ≥ 60/155.",
    "truncated = 0, no_box_fallback = 0, boxed_rate = 1.0.",
    "0 protected‑row backfire.",
    "avg_completion_tokens ≤ 1024 and max_completion_tokens ≤ 4096 (new gate).",
    "label‑free extraction correctness verified by label‑aware‑minus‑label‑free == 0.",
    "V667 surface gate (LoRA only, no embedding/lm_head in adapter) passed on the saved checkpoint.",
    "All findings in weak‑eval manifest warnings = 0."
  ],
  "delete_from_roadmap": [
    "All references to V662 as a potential active route (already superceded).",
    "Any retention of V653, V660, V661, V663 as fallback routes – keep only V290 checkpoint‑6 as parent.",
    "Mention of H200 even for future exploration – permanently block.",
    "The stale ‘equation=59’ threshold still lingering in any nested gate (ensure only equation=60 is active)."
  ]
}
```

## qwen/qwen3.6-max-preview

- status: `ok`

1. Verdict: block

2. Top 5 concrete bugs or gaps, each tied to evidence in the prompt
   1. **Runaway decoding risk from trace targets:** V673 dataset metadata declares `completion_format=trace_plus_final_boxed` and `assistant_multiline_rows=720`, yet `assistant_trace_rows=0`. V664/V661 failed with `avg_completion_tokens=4772`, `truncated=1`, and protected backfire because the adapter learned to emit long reasoning before `\boxed{}`. Loss masking (`LOSS_MASK_STOP_AFTER_EOS=1`) prevents gradient flow past EOS but does not constrain autoregressive sampling. Without a pre-launch generation gate, V673 will likely repeat the exact same truncation/backfire failure.
   2. **`lm_head` in `target_modules` vs adapter-only contract:** The PEFT warning that triggered `SAVE_EMBEDDING_LAYERS=0` occurred because `lm_head` is in `target_modules`. Setting `SAVE_EMBEDDING_LAYERS=0` suppresses weight serialization but LoRA still trains adapters on `lm_head`. This creates a hidden base-weight dependency, risks silent logit drift, and violates the strict adapter-only packaging contract. `lm_head` must be removed from `target_modules` entirely.
   3. **MoE `target_parameters` + `target_modules` gradient collision:** Launcher declares both `target_modules=...,up_proj,down_proj` and `LORA_TARGET_PARAMETERS=mlp.experts.gate_up_proj,mlp.experts.down_proj`. PEFT routes these differently; activating both risks double-adapter registration or misaligned expert routing, causing unstable MLP updates. V664 already showed MLP/MoE weight drift without ACC gain.
   4. **`scale_mean` loss scale vs LR mismatch:** `ROW_LOSS_WEIGHT_REDUCTION=scale_mean` correctly fixes the microbatch=1 cancellation, but increases gradient magnitude proportionally to row weights (equation effective share `0.851`). The scheduled `LR=5e-7 -> 1e-7` was tuned for the old canceled/sum reduction. Gradients will now be larger, risking early-step overshoot or checkpoint-2 instability without LR adjustment or gradient norm verification.
   5. **Missing label-free extractor dry-run on new format:** Gates verify tokenization counts and CPU simulation, but no artifact proves the official label-free extractor + `verify_answer` successfully parses the new `trace_plus_final_boxed` JSONL. If trace newlines/whitespace break the regex or `verify_answer`, the job will waste GPU on unparseable targets. Missing artifact: `cpu_extractor_parity_on_trace_format.json`.

3. Exact next experiment that is cheapest and most likely to improve weak ACC
   Run a **CPU-only generation dry-run** using V290 `checkpoint-6` + the V673 prompt template on a 20-row stratified sample (10 equation_numeric, 10 bit_residual). Use exact official-like decoding (`max_tokens=7680`, `temperature=0`, `top_p=1`). Pipe raw outputs through the label-free extractor + `verify_answer`. Measure `avg_completion_tokens`, `boxed_rate`, `starts_boxed_rate`, and `verify_answer` pass rate. This isolates prompt/template decoding behavior from adapter weights, proves whether the trace format causes runaway generation, and validates extractor compatibility before any GPU spend.

4. Parameters to change or freeze, with values
   - **Change:** `TARGET_MODULES` -> `q_proj,k_proj,v_proj,o_proj` (remove `lm_head`, `up_proj`, `down_proj`). If MLP adaptation is required, use only `target_modules=gate_up_proj,down_proj` and drop `LORA_TARGET_PARAMETERS` entirely to avoid PEFT collision.
   - **Change:** `LR` -> `2e-7` constant for steps 1-10 (compensates for `scale_mean` gradient increase; avoids premature decay to 1e-7 before checkpoint-2 evaluation).
   - **Change:** Add `MAX_NEW_TOKENS=256` for all weak eval/generation runs to hard-cap runaway traces and guarantee `truncated=0`.
   - **Freeze:** `SAVE_EMBEDDING_LAYERS=0`, `ROW_LOSS_WEIGHT_REDUCTION=scale_mean`, `LOSS_MASK_STOP_AFTER_EOS=1`, `MAX_LENGTH=1024`, `ABORT_MAX_RESERVED_GIB=78`, `USE_ROW_LOSS_WEIGHT=1`, `REQUIRE_VALIDATION_ROW_LOSS_WEIGHT=1`.

5. Gates that must pass before another paid GPU job
   - **CPU Decoding Format Gate:** 20-row dry-run must yield `avg_completion_tokens <= 128`, `boxed_rate = 1.0`, `starts_boxed_rate >= 0.9`, and `verify_answer` pass rate >= 0.95. If failed, switch dataset to `boxed_only` or `short_trace_prefix` before launch.
   - **PEFT Module Collision Gate:** Local `get_peft_model` dry-run must show exactly one LoRA adapter per targeted weight, zero `lm_head` adapters, and zero duplicate expert parameter registrations.
   - **Extractor Parity Gate:** Label-free extractor + `verify_answer` must parse 100% of the V673 validation JSONL targets without fallback, regex mismatch, or whitespace-induced failure. Missing artifact: `v673_extractor_parity_on_trace_format.json`.
   - **Protected Row Smoke:** V290 `checkpoint-6` + V673 prompt must reproduce baseline answers for `8740ed31=01101000`, `59bee375=10010101`, `55d834d1=00111111` with zero deviation before adapter training begins.
   - **Gradient Norm Check:** First 2 training steps must log `grad_norm <= 1.0` under `scale_mean`. If exceeded, clamp LR to `1e-7` or enable `max_grad_norm=0.5`.

6. Anything in the current plan that should be deleted because it is noise
   - Delete `LORA_TARGET_PARAMETERS` / `REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE` contract. It conflicts with standard `target_modules`, adds unverified PEFT routing complexity, and has no proven ACC gain over standard module targeting.
   - Delete `LR=5e-7 -> 1e-7` decay schedule for a 20-step probe. It is noise; use constant LR for short diagnostic runs.
   - Delete `KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT=1`. Deferring decoding drift checks guarantees repeating V664's backfire. Decoding format must be gated pre-launch.
   - Delete all active references to `equation_symbolic` or the `87 symbolic punctuation misses`. They are explicitly out of scope for the +4 equation gain target, dilute row-weight focus, and violate the "no broad SFT" rule.

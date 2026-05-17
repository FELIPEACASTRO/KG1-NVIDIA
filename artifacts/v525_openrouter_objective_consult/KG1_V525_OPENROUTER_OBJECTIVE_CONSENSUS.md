# KG1 V525 OpenRouter Objective Consult Consensus

generated_at_utc: 2026-05-17T00:14:53.412967+00:00
prompt_path: `C:/Users/davis/Workspace/KG1 -NVIDIA/artifacts/v284_official_gate_worktree/artifacts/v525_openrouter_objective_consult/KG1_V525_OPENROUTER_OBJECTIVE_PROMPT.md`
responses_path: `C:/Users/davis/Workspace/KG1 -NVIDIA/artifacts/v284_official_gate_worktree/artifacts/v525_openrouter_objective_consult/v525_openrouter_model_responses.json`
models_requested: 5
models_ok: 5

## Consensus

- Use `example_mean` / per-example normalization before another token-mean run: 5/5 model responses mention/support related terms.
- Avoid broad SFT / more epochs without new CPU signal: 3/5 model responses mention/support related terms.
- Prefer short H200 smoke only behind strict first-checkpoint gates: 5/5 model responses mention/support related terms.
- Use family/row balance or cap token mass so bit traces do not drown equation: 5/5 model responses mention/support related terms.
- Protect bit row 8740ed31 and require no regression before promotion: 5/5 model responses mention/support related terms.
- Do not submit unless weak/full gate shows real no-regression gain: 4/5 model responses mention/support related terms.

## Per-Model Key Answer Excerpts

### openai/gpt-5.5
- usage: `{'prompt_tokens': 12425, 'completion_tokens': 3000, 'total_tokens': 15425, 'cost': 0.152125, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.152125, 'upstream_inference_prompt_cost': 0.062125, 'upstream_inference_completions_cost': 0.09}, 'completion_tokens_details': {'reasoning_tokens': 2588, 'image_tokens': 0, 'audio_tokens': 0}}`

## Verdict

V523/V524 is technically directionally correct: V524 correctly identified that V523’s token-level objective is dominated by long bit traces, and `example_mean` is the fastest responsible correction because it removes the most obvious token-length bias without rebuilding data.

However, V523 is **not justified for a full run**. It is justified only for a **short H200 smoke** if preflight proves that `LOSS_NORMALIZATION_MODE=example_mean` is actually active and the first checkpoint is weak-gate evaluated immediately.

Decision: **GO for one short V523 + `example_mean` smoke, not for extended training.** If the first checkpoint does not beat the adapter-only baseline without bit regression, stop and build V525.

---

## Ranked Next Actions

1. **Run V523 with `LOSS_NORMALIZATION_MODE=example_mean` as a short smoke.**
   - Evidence: V523 is source-only, no weak/full overlap, tokenization passed, trace learnability passed.
   - Evidence: V524 shows token loss mass is badly skewed: bit token share `0.9067` vs reference gain bit share `0.7419`.
   - `example_mean` directly corrects the token-mass skew with minimal new moving parts.
   - This is the fastest responsible test.

2. **If V523 smoke fails, build V525 with shorter bit traces and reference-mix quotas.**
   - Goal: keep bit signal but prevent bit CoT length from dominating.
   - Use V522 no-loss gain mix as the target: roughly `74% bit / 26% equation` by rows, but cap bit token mass near that range too.

3. **Add reference-mix family row weights only after the example-mean smoke.**
   - Hypothesis: mild bit upweighting could preserve `8740ed31` while allowing equation improvement.
   - Do **not** use naive 50/50 family-balanced loss; it would overweight equation relative to

### anthropic/claude-opus-4.7
- usage: `{'prompt_tokens': 21852, 'completion_tokens': 3000, 'total_tokens': 24852, 'cost': 0.18426, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.18426, 'upstream_inference_prompt_cost': 0.10926, 'upstream_inference_completions_cost': 0.075}, 'completion_tokens_details': {'reasoning_tokens': 0, 'image_tokens': 0, 'audio_tokens': 0}}`

# Verdict

V523 + `example_mean` is a **necessary but insufficient** correction. The token-level dominance (90.7% bit token mass for 68.8% bit rows) is real, but `example_mean` alone treats a 5-token equation answer as equally weighted to a 500-token bit CoT — which is what you want for ACC, but it will also dilute the **per-token gradient signal on bit answers** and may destabilize the protected bit row `8740ed31`. The safer path is **`example_mean` + an answer-span loss weight that ignores the long CoT prefix**, combined with a small rebuild (V525) that caps bit trace length. Going to GPU with `example_mean` on V523 *as-is* is a coin-flip on the protected bit row. **No-go on V523 H200; build V525 first, then 1 short smoke.**

# Ranked Next Actions

1. **Build V525** (CPU only, 2–4h): same source pool as V523, but cap bit trace length and rebalance row counts so token-mass ratio matches reference gain share (~74/26).
2. **Add answer-span loss weighting** (`ANSWER_SPAN_LOSS_WEIGHT=4.0–8.0` on the `\boxed{...}` payload) in addition to `LOSS_NORMALIZATION_MODE=example_mean`. This is what actually matters for exact-answer ACC.
3. **Run V525 H200 smoke** ≤30min, eval at step 25/50 with the kill-switch in §4.
4. *(Hypothesis)* If V525 still backfires on `8740ed31`, add that row's rule family (full-byte ternary, position-3 sensitive) as an explicit hard-mined source trace — not the row itself.
5. *(Hypothesis, last resort)* Family-balanced loss (per-batch family reweighting) — defer; harder to verify and easier to silently break.

**Ranking of objective corrections (Q1):**
1. **Answer-span loss weight on `\boxed{...}` payload** (highest leverage; directly aligned with exact-match metric).
2. **example_mean normalization** (correct direction, partial fix).
3. **Shorter bit traces in V525** (structural; reduces the problem at the source).
4. Family-balanced row weights (works but adds a hyperparameter to tune blind).
5. Hard negatives (rejected — already failed in V436/V440/V441 lineage).

The right answer is **(1)+(2)+(3) combined**, not any single one.

# Exact V525/V526 Configuration

**V525 dataset quotas** (derived from V522 reference: 74.2% bit / 25.8% equation gain share; V524 token mass 90.7% bit → must drop to ~74%):

| Field | Target |
|---|---:|
| Train rows total | ~900 |
| Bit rows | 540 (60%) |
| Equation rows | 360 (40%) |
| Val rows total | 200 |
| Val bit | 120 |
| Val equation | 80 |
| **Max bit trace tokens per row** | **256** (V523 had effectiv

[excerpt truncated; see JSON response file]

### google/gemini-3.1-pro-preview
- usage: `{'prompt_tokens': 15509, 'completion_tokens': 2996, 'total_tokens': 18505, 'cost': 0.06697, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.06697, 'upstream_inference_prompt_cost': 0.031018, 'upstream_inference_completions_cost': 0.035952}, 'completion_tokens_details': {'reasoning_tokens': 1882, 'image_tokens': 0, 'audio_tokens': 0}}`

**VERDICT**
The V523/V524 plan is technically correct in identifying token-mass bias as the primary blocker. The 90.7% bit token share in V523 mathematically drowns out the 31.2% equation row share under standard `token_mean` loss. `example_mean` is the correct, immediate mathematical fix to realign the gradient with the row distribution. V523 is safe for a strictly gated 1-hour H200 smoke test.

**RANKED NEXT ACTIONS**

1. **(Fastest Path) Run V523 H200 Smoke with `example_mean`**: Execute a <=1 hour job on V523 using `LOSS_NORMALIZATION_MODE=example_mean`. This directly tests the hypothesis that token-mass bias caused the V518 plateau, without requiring a new dataset build.
2. **(Fallback Path) Build V525 Shorter Traces**: If V523 + `example_mean` fails to produce a submit-safe gain at checkpoint 1, the traces themselves lack transferable signal. Immediately build V525 by truncating bit traces to their core logic and increasing equation row mass, then rerun CPU gates.
3. **(Hypothesis) Payload-Only Score Masking**: If `example_mean` still results in loss/ACC divergence, apply the V441-style `boxed_payload_loss_mask` to force the adapter to focus exclusively on the final answer tokens rather than the trace boilerplate.

**EXACT V525 CONFIGURATION (IF REBUILD REQUIRED)**
If V523 fails, V525 must strictly enforce the V522 reference gain distribution (74.2% bit, 25.8% equation) at *both* the row and token levels.
*   **Train Rows**: ~1,000 total (742 `bit_manipulation`, 258 `equation_transform`).
*   **Val Rows**: ~200 total (148 bit, 52 equation).
*   **Token Mass Limits**: Cap bit traces at ~450 tokens max. Do not allow 1000+ token bit traces. Target a token mass ratio of roughly 75% bit / 25% equation.
*   **Hard Negatives**: Exclude format-only negatives. Use only verified solver traces (CHO/MAJ3/global ternary) that do not overlap with weak/full prompts.

**KILL-SWITCH / GATES**
Cancel the H200 job immediately at the first evaluation checkpoint (e.g., step 50) if the local weak gate does not meet **ALL** of the following exact thresholds:
*   `overall >= 192/315` (Must beat or tie baseline immediately)
*   `bit_manipulation >= 136/160` (Zero tolerance for bit regression)
*   `equation_transform >= 56/155` (Must show at least baseline stability, ideally 57+)
*   `truncation == 0`
*   Protected row `8740ed31` prediction must exactly match `01101000` (No backfire).

**SILENT BUG CHECKLIST (PRE-GPU)**
1. **Offset-Mask**: Verify `REQUIRE_OFFSET_MASK=True` an

[excerpt truncated; see JSON response file]

### qwen/qwen3.6-max-preview
- usage: `{'prompt_tokens': 14163, 'completion_tokens': 5383, 'total_tokens': 19546, 'cost': 0.04831944, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.04831944, 'upstream_inference_prompt_cost': 0.01472952, 'upstream_inference_completions_cost': 0.03358992}, 'completion_tokens_details': {'reasoning_tokens': 3647, 'image_tokens': 0, 'audio_tokens': 0}}`

**Verdict**
V523 is structurally sound and correctly targeted to V522 no-loss gain rules, but V524 proves it is optimization-biased: bit CoT traces consume 90.7% of loss tokens despite representing only 68.8% of rows. This length dominance drowns equation signal and destabilizes fragile bit boundaries (evidenced by V518's backfire on `8740ed31`). `LOSS_NORMALIZATION_MODE=example_mean` is the correct, lowest-latency correction. A short, strictly gated H200 smoke is justified. If it fails, a V525 rebuild with explicit token-mass quotas is mandatory. All projected gains are hypotheses until gate-validated.

**Ranked Next Actions**
1. **Dry-run verify `example_mean` implementation:** Run a CPU tokenization + forward pass on 10 V523 rows (5 long bit, 5 short equation). Confirm per-example loss is computed as `CE_sum / active_label_tokens` and then averaged across the batch. Verify bit vs equation loss contributions are within 1.5x, not 9x.
2. **Launch ≤1-hour H200 smoke on V523:** Use `LOSS_NORMALIZATION_MODE=example_mean`, `MAX_STEPS=50`, `EVAL_EVERY_STEPS=25`. Attach the exact kill-switch below. Cancel immediately if checkpoint-1 fails.
3. **If smoke fails, build V525:** Rebalance dataset to cap bit token mass at ≤70% of total loss tokens. Strip verbose CoT boilerplate from bit traces while preserving rule-critical tokens (CHO/MAJ3/PAR3 operations). Rerun V286/V513/V524 CPU gates before any GPU.
4. **Post-smoke analysis (if thresholds met):** Run full weak gate (315 rows). If `bit>=136`, `equation>=57`, `total>=193`, `trunc=0`, and protected rows hold, package adapter for submit. Do not extend training without a second gate validation.

**Exact V525/V526 Configuration**
*(Use only if V523 smoke fails or shows bit regression)*
- **Train row ratio:** 60% bit / 40% equation (~600 bit, ~400 equation). Compensates for inherent bit trace length.
- **Validation row ratio:** 50% bit / 50% equation (100/100). Ensures early signal isn't masked by family imbalance.
- **Token-mass limits:** Bit loss-token share must be ≤0.70. Equation loss-token share ≥0.30. Enforce via `example_mean` + explicit row weights if needed (`eq_weight=1.3`, `bit_weight=1.0`).
- **Trace length caps:** Max bit trace tokens = 450. Truncate repetitive CoT narration; retain only operation steps, intermediate states, and final `\boxed{}`. Equation traces remain untruncated (already short).
- **Rule coverage:** Maintain V522 priority: `CHO≥200`, `MAJ3≥200`, `PAR3≥80`, `global_ternary≥100`, equation c

[excerpt truncated; see JSON response file]

### deepseek/deepseek-v4-pro
- usage: `{'prompt_tokens': 13562, 'completion_tokens': 4936, 'total_tokens': 18498, 'cost': 0.01903264, 'is_byok': False, 'prompt_tokens_details': {'cached_tokens': 13184, 'cache_write_tokens': 0, 'audio_tokens': 0, 'video_tokens': 0}, 'cost_details': {'upstream_inference_cost': 0.01903264, 'upstream_inference_prompt_cost': 0.00234896, 'upstream_inference_completions_cost': 0.01668368}, 'completion_tokens_details': {'reasoning_tokens': 3001, 'image_tokens': 0, 'audio_tokens': 0}}`

Verdict
The V523/V524 plan is technically correct in its diagnosis but incomplete for a paid GPU run. The token-level loss dominance by bit traces (90.7% of loss tokens vs. 74.2% reference gain share) is a real blocker. `example_mean` is a necessary but insufficient fix—it equalizes per-example weight but does not rebalance the signal content (long bit traces may still dominate gradient updates through token count within each example). A V525 dataset rebuild with shorter bit traces and higher equation token mass is required before any GPU smoke. The current V523 dataset is not safe for H200 even with `example_mean` because the token-mass imbalance will still bias early checkpoint learning toward bit patterns at the expense of equation signal.

Ranked Next Actions
1. **Build V525 dataset with token-mass quotas** (highest priority, CPU-only). This directly addresses the V524 warning and is the only path that can produce a submit-safe gain without risking another backfire. Must be completed and pass V286/V513/V524 before any GPU.
2. **Implement and validate `example_mean` loss normalization** (already coded, needs dry-run validation with V525). This is the correct objective correction for the remaining row-count imbalance after token-mass is rebalanced.
3. **Run V525 + `example_mean` on H200 with strict kill-switch** (only after V525 passes all CPU gates). This is the first GPU action.
4. **If V525 fails kill-switch, build V526 with family-balanced loss weights** (hypothesis: explicit per-family loss weights may be needed if `example_mean` alone does not sufficiently boost equation gradient signal). This is a fallback, not the primary plan.
5. **Investigate shorter bit trace formats** (hypothesis: bit traces can be compressed to essential operations only, reducing token count by 50-70% while preserving rule coverage). This is a medium-term optimization, not for immediate smoke.

Exact V525/V526 Configuration
**V525 dataset quotas (derived from V522 no-loss gain distribution and V524 token dominance):**
- Train rows: 800-1000 total
  - bit_manipulation: 400-500 rows (50-62.5% row share, down from 68.8%)
  - equation_transform: 400-500 rows (37.5-50% row share, up from 31.1%)
- Target token-mass ratio: bit ≤ 70% of total loss tokens (down from 90.7%)
- Bit trace construction:
  - CHO/MAJ3 traces: limit to ≤ 200 tokens per example (current V523 bit traces average ~467 tokens)
  - PAR3 traces: limit to ≤ 150 tokens
  - Remove verbose CoT explanations; keep only o

[excerpt truncated; see JSON response file]

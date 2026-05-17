# KG1 V525 OpenRouter Actionable Findings

## openai/gpt-5.5
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

## anthropic/claude-opus-4.7
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
| **Max bit trace tokens per row** | **256** (V523 had effectively ~470 avg implied by 329702/706) |
| **Max equation trace tokens per row** | 128 |
| Target loss-token bit share | 0.70–0.78 (within ±5pp of reference 0.742) |
| Target loss-token mass bit | ~140k |
| Target loss-token mass equation | ~46k |
**Bucket allocation (mirrors V522 no-loss gain rule distribution):**
- `bit_exact_global_ternary_unique_prediction`: 220 rows (was top gain rule, 13/23)
- `bit_fullbyte_ternary_op_CHO`: 80
- `bit_fullbyte_ternary_op_MAJ3`: 80
- `bit_fullbyte_ternary_op_PAR3`: 60 (covers position-sensitive class likely related to `8740ed31`)
- `bit_exact_global_binary_OR/XOR`: 50 each
- Equation 4 classes × 90 rows = 360
**Trainer env for V525 smoke (V526 = the run):**
LOSS_NORMALIZATION_MODE=example_mean
# Kill-Switch / Gates
**Pre-GPU (must all pass on V525):**
- V286 tokenization gate: trunc=0, offset masks present 100%
- V513 trace learnability: 0 blockers
- V521 transfer blocker re-run: weak/full prompt overlap = 0
- V524 quota re-run: `loss_token_bit_share ∈ [0.70, 0.78]`
- New gate: protected row `8740ed31` rule family present in train ≥ 8 rows
**During GPU (eval at step 20 and step 40):**
| Metric | Kill threshold |
|---|---|
| `prompt_sha256=8740ed31` predicted answer | **Must == `01101000`. Any other value → abort.** |
| `bit_manipulation` weak | **< 136 → abort** |
| `equation_transform` weak | < 56 at step 40 → abort (allow temporary <56 at step 20) |
| weak total | < 191 at step 40 → abort |
| `truncated` | > 0 → abort |
| `eval_loss` vs baseline | rise > +0.02 → abort |
| reserved GPU mem | > 75 GiB → abort |
**Promote checkpoint only if:** bit ≥ 136 AND equation ≥ 58 AND total ≥ 194 AND `8740ed31` preserved AND trunc = 0. Anything weaker → no promotion, no submit.
# Silent Bug Checklist
1. **`\boxed{}` literal escape**: re-grep the V525 builder for `\b` → backspace bug (you caught this in V523; re-verify hexdump of 20 random target strings).
3. **Answer-span mask non-empty**: for every row, assert `sum(answer_span_mask) >= ANSWER_SPAN_MIN_WEIGHTED_TOKENS`. V441 had non-empty payload masks but check the new combined mode.
4. **Prompt template parity**: diff V525 prompt template byte-for-byte against the weak gate's inference prompt template. Single trailing newline mismatches have silently caused ACC drops before.
7. **Duplicate leakage**: hash-join V525 train prompts against (a) weak 315 rows, (b) full gate rows, (c) test set. Assert 0 overlap on all three.
8. **Protected row regression precheck**: before any training, run baseline eval and confirm V516 adapter still gets `8740ed31=01101000`. If not, you're measuring against a moved baseline.
9. **Loss-mode unit test**: assert that with `example_mean`, a batch of [1 row × 500 tokens + 1 row × 10 tokens] produces loss = mean of two per-example CEs, not token-weighted. Add as a `pytest`.
11. **Gradient checkpointing × answer-span mask**: confirm the answer-span weighting tensor isn't being recomputed/lost across the checkpoint boundary (this is subtle; add a hash assertion).
12. **`gate_rows_used_for_training=false`** in V525 manifest (you fixed this in V523

## google/gemini-3.1-pro-preview
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
*   `overall >= 192/315` (Must beat or tie baseline immediately)
*   `bit_manipulation >= 136/160` (Zero tolerance for bit regression)
*   `equation_transform >= 56/155` (Must show at least baseline stability, ideally 57+)
*   `truncation == 0`
*   Protected row `8740ed31` prediction must exactly match `01101000` (No backfire).
**SILENT BUG CHECKLIST (PRE-GPU)**
2. **Prompt Template**: Verify the Python `\b` backspace bug is permanently fixed and the template strictly emits `Final answer: \boxed{`.
3. **Answer Extraction**: Verify the regex correctly extracts the payload from `\boxed{...}` without capturing trailing braces or whitespace.
7. **Protected Row Regression**: Explicitly assert that row `8740ed31` is present in the eval suite and tracked.
**EXPECTED GAIN / RISK**
*   **Expected Gain**: +1 to +3 rows in `equation_transform` (reaching 57-59) by allowing the equation loss to actually influence the adapter weights without being drowned by bit tokens.
*   **Risk**: `example_mean` upweights short equation traces, which might cause the model to forget long-context bit tracking, risking a drop in `bit_manipulation` below 136. The kill-switch mitigates this.
*   **Diagnostic for Loss/ACC Divergence**: If eval_loss drops but exact ACC is flat/worse, the model is memorizing the *trace format* (e.g., outputting CHO/MAJ3 steps

## qwen/qwen3.6-max-preview
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
- **Rule coverage:** Maintain V522 priority: `CHO≥200`, `MAJ3≥200`, `PAR3≥80`, `global_ternary≥100`, equation classes `['274def88','7688e06e','c5b058d6','d1bd7478']` at ≥80 each.
- **Metadata:** `gate_rows_used_for_training=false`, `loss_normalization=example_mean`, `max_bit_trace_tokens=450`.
**Kill-Switch / Gates**
- `bit_manipulation >= 136` (no regression from V516 floor)
- `equation_transform >= 57` (minimum +1 net gain vs V516/V518)
- `weak_total >= 193`
- `truncation_rate == 0`
- Protected row `8740ed31` prediction == `01101000` (exact string match)
- Reference gain row `518deb39` prediction == `$` (exact string match)
- `eval_loss <= baseline_eval_loss` (sanity check; not promotional)
- FinOps hard stop: walltime ≤60 minutes. No checkpoint-2 unless checkpoint-1 passes all above.
**Silent Bug Checklist**
- [ ] **Offset-mask alignment:** Verify loss mask covers ONLY CoT + answer tokens. Prompt tokens must have weight 0. Spot-check 3 long bit rows and 3 short equation rows.
- [ ] **`\boxed` tokenization:** Confirm the backspace fix propagates to tokenized tensors. Decode a sample target sequence; must contain literal `\boxed{` and `}` with no control characters.
- [ ] **Prompt template parity:** Training system prompt and few-shot format must byte-match inference packaging. Mismatches cause silent distribution shift.
- [ ] **Answer extraction regex:** Test on V523 val set. Must handle whitespace, newlines, and trailing punctuation. Fail if extraction rate <98%.
- [ ] **Family mapping integrity:** Cross-check `prompt_sha256` routing against `bit_manipulation`/`equation_transform` tags. Zero cross-family contamination allowed.
- [ ] **Duplicate/leakage scan:** SHA256 hash all V523 train prompts against weak/full gate prompts. Overlap must be exactly 0.
- [ ] **Protected row regression guard:** Explicitly run `8740ed31` and `518deb39` through the untrained adapter and checkpoint-1. Log exact token probabilities at the flip position (bit index 3 for `8740ed31`).
**Expected Gain / Risk**
- **Hypothesis:** `example_mean` will reduce bit token dominance from 90.7% to ~65-70%, allowing equation gradient signal to register. Expected gain: +1 to +3 equation rows, bit stable at 136. Total weak: 193-195/315.
- **Risk:** `example_mean` may overcorrect, causing gradient instability on long bit traces or slight bit regression (135/160). Mitigated by hard kill-switch and protected row guard. If equation loss drops but exact ACC stays flat, the model is learning format/boilerplate, not rule logic (requires V525 trace shortening).
- **Diagnostic for low eval_loss / flat ACC:** Compute per-family eval loss and per-row logprob of the exact target answer. If bit loss drops >10% but equation loss is flat, token bias persists. If fragile rows show probability mass shifting to adjacent tokens (e.g., bit position 3 flips `0`→`1`), the decision boundary is destabilizing. Stop training; adjust family weights or trace length. Do not trust aggregate loss.
**What Not To Do**
- Do not train on weak/full gate labels or test prompts.
- Do not run another broad SFT or preference job without a new CPU transfer signal.
- Do not extend training past checkpoint-1 if kill-switch thresholds are not met.
- Do not submit unless the 315-row weak gate shows measured gain with zero regression on protected rows and `trunc=0`.
- Do not rely on hidden verifiers, postprocessors, or solver routing for submit-safe packaging.
**One-Sentence Decision**
Run a single ≤1-hour H200 smoke on V523 with `LOSS_NORMALIZATION_MODE=example_mean` and a hard checkpoint-1 kill-switch; abort and rebuild V525 with strict token-mass quotas if thresholds are not met.

## deepseek/deepseek-v4-pro
The V523/V524 plan is technically correct in its diagnosis but incomplete for a paid GPU run. The token-level loss dominance by bit traces (90.7% of loss tokens vs. 74.2% reference gain share) is a real blocker. `example_mean` is a necessary but insufficient fix—it equalizes per-example weight but does not rebalance the signal content (long bit traces may still dominate gradient updates through token count within each example). A V525 dataset rebuild with shorter bit traces and higher equation token mass is required before any GPU smoke. The current V523 dataset is not safe for H200 even with `example_mean` because the token-mass imbalance will still bias early checkpoint learning toward bit patterns at the expense of equation signal.
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
  - Remove verbose CoT explanations; keep only operational steps
- Equation trace construction:
  - Maintain current format (short rule reject boxed, ~106 tokens avg)
  - Include all 4 V516 label-free equation classes equally
- Validation: 200-250 rows with same family ratio as train
- Source: v304_solver_trace_distill only, filtered to V522 no-loss gain rule families
- Hard constraint: zero overlap with weak/full prompts, protected row 8740ed31 must be in validation
**V526 fallback (if V525 kill-switch triggers):**
- Same row quotas as V525
- Add `LOSS_FAMILY_WEIGHTS` env var: `{"bit_manipulation": 0.4, "equation_transform": 0.6}`
- Implement as multiplicative weight on per-example loss before averaging
Kill-Switch / Gates
**Pre-GPU gates (must all pass before H200 launch):**
- V286 tokenization gate: max tokens ≤ 749, truncation = 0, offset masks present for all rows
- V513 trace learnability gate: 0 blockers, 0 warnings
- V524 quota/token objective audit: bit token share ≤ 0.70, equation token mass ≥ 30% of total
- V521 transfer blocker audit: no weak/full overlap, protected row 8740ed31 in validation
- Static gate: no gate_rows_used_for_training=true, no blocked dataset markers
**GPU kill-switch (abort at first checkpoint if ANY condition fails):**
- Checkpoint evaluation at step 50 (or earliest save step):
  - bit_manipulation ≥ 136 (no regression from V516 baseline)
  - equation_transform ≥ 57 (minimum +1 gain over V516's 56)
  - weak total ≥ 193 (minimum +1 gain over V516's 191)
  - Protected row 8740ed31 prediction = `01101000` (no backfire)
  - Truncation = 0
- If any condition fails: immediate abort, no further checkpoints, no upload
- Additional abort conditions:
  - eval_loss > 3.30 (should not increase significantly from V518's 3.2720)
  - GPU memory reservation > 80% of available
**Before any GPU, verify:**
2. **Prompt template consistency**: Verify V525 uses identical prompt template as V516 baseline (no extra spaces, newlines, or system prompts). Compare tokenized prompt lengths for same raw prompts.
3. **Answer extraction**: Confirm `\boxed{...}` extraction regex matches exactly between training data construction and weak gate evaluation. Test on protected row 8740ed31.
4. **Family mapping**: Verify bit_manipulation/equation_transform family labels are assigned correctly in V525. Cross-reference 10 random examples against source dataset family tags.
6. **Duplicate leakage**: Check for exact prompt hash collisions between V525 train and weak/full gate prompts. Also check for near-duplicates (edit distance < 10 chars).
7. **Protected row regression**: Run V516 baseline adapter on V525 validation set, confirm 8740ed31 is correct. If not, V525 validation construction is flawed.
8. **`example_mean` implementation**: Verify loss normalization divides by per-example active label token count, not total tokens. Test with a batch of 2 examples with different label lengths—loss should be equal if both are 100% correct.
9. **Gradient accumulation with `example_mean`**: Confirm loss is averaged over examples within each micro-batch, then gradients are accumulated correctly across micro-batches. Test with batch_size=4, micro_batch_size=2.
**Expected gain (if V525 + example_mean works):**
- equation_transform: +2-4 rows (to 58-60/155), based on V522 teacher showing 8 equation gains from these rule families
- bit_manipulation: 0 to +2 rows (to 136-138/160), preserving baseline while potentially gaining from CHO/MAJ3 traces
- weak total: 193-196/315
- Probability of success: 40-50% (first targeted dataset with token-mass rebalancing; previous failures were with un-balanced objectives)
**Risk:**
- High: Another backfire on protected row 8740ed31 (mitigated by kill-switch)
- Medium: equation_transform stays at 56 despite objective fix (signal may still be too weak for LoRA rank 32)
- Low: bit_manipulation drops below 136 (mitigated by kill-switch and bit trace preservation)
- Low: New silent bug in `example_mean` causes training instability (mitigated by pre-GPU dry-run)
- Do NOT run V523 on H200 even with `example_mean`—the token-mass imbalance is fatal
- Do NOT train another broad SFT without CPU gate signal (V518 already proved this fails)
- Do NOT use V515/V510/V475 datasets as-is (already tested and failed)
- Do NOT implement family-balanced loss weights before trying `example_mean` with rebalanced token mass (over-engineering before simpler fix is tested)
- Do NOT submit to Kaggle without weak/full gate showing measured gain with zero regression
- Do NOT extend training beyond first checkpoint if kill-switch triggers (FinOps rules)
- Do NOT add more bit traces to "preserve" bit—the current bit score is stable; the risk is drowning equation, not losing bit
Build V525 with bit token-mass ≤ 70% and equation rows ≥ 400, validate through all CPU gates, then run a 1-hour H200 smoke with `example_mean` loss and abort at first checkpoint unless equation ≥ 57, bit ≥ 136, and protected row 8740ed31 is preserved.

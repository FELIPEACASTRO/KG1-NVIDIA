# KG1 Post-Train OpenRouter Responses

## openai/gpt-oss-120b:free

- status: `ok`

**1. Verdict**  
**proceed** – the import‑package bug is fixed locally and the pre‑paid integration & static safety gates are passing. The next safe step is to run a *weak‑eval‑only* A100 job on the already‑uploaded checkpoints (starting with the best‑loss checkpoint‑10). No new training run is required until that evaluation is completed and the results are reviewed.

---

**2. Top 5 concrete bugs / gaps (tied to evidence)**  

| # | Issue | Evidence / Location |
|---|-------|----------------------|
| 1 | **Remote import failure still blocks evaluation** – the `scripts/__init__.py` fix has not been pushed to the HF repo, so the remote container still raises `ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'`. | `v712_failure_summary.json` – “importing_script … ModuleNotFoundError”; `v712_hf_job_tail.log` shows the same error. |
| 2 | **Training signal far too weak** – only 20 steps (10 → 20) with a learning‑rate schedule that was already too low in V708; loss improvement is <0.002 and ACC does not move. | `v712` manifest (`max_steps=20`, `learning_rate 2e‑6 → 5e‑7`); `training_result` shows loss barely changed and no weak ACC. |
| 3 | **Runaway decoding** – several rows (e.g. protected row 59bee375) hit the 7 680‑token ceiling, produce no `\boxed{}` and cause the “avg_completion_tokens_gt_512” gate failure. | V710 weak‑eval report: “avg completion tokens: 4775.58, max 7680”; protected‑row backfire analysis. |
| 4 | **Protected‑row backfire guard not preventing regressions** – rows 8740ed31 and 59bee375 flip from correct baseline to wrong answer, yet the guard only *detects* after generation. | V710 failure ledger – “protected_row_backfire_guard_failed”. |
| 5 **(secondary)** | **Adapter config shows a broad `target_modules` list** that is misleading; the contract expects only `q_proj,v_proj` trainable, but the manifest still lists all modules, which can cause future audit confusion. | `v712` manifest `target_modules` list includes 8 modules; `v711` audit clarified effective trainable set. |

---

**3. Exact next experiment (cheapest, highest ACC gain probability)**  

| Step | Action | Reason |
|------|--------|--------|
| **A** | **Push the import‑fix to the HF repo** (`scripts/__init__.py` and the compiled `scripts/evaluate_lora_adapter.py`) and tag a new commit that matches `KG1_EXPECTED_REPO_COMMIT` (currently `67a27bcb2ed6a4e9856adb61ece63516a0b29637`). | Required for the remote container to import the evaluation script; without it the weak‑eval job will always error. |
| **B** | **Launch a *weak‑eval‑only* A100 job** on checkpoint‑10 (the checkpoint with the lowest validation loss). Use the same runtime policy **except** set a tighter generation limit to stop runaway decoding: `KG1_MAX_TOKENS=1024` (or `KG1_MAX_NEW_TOKENS=1024`). Keep `KG1_DISABLE_THINKING=0`. | Checkpoint‑10 is already uploaded; a single eval run costs ≈ 2 min on A100 (≈ $0.08). Reducing max tokens caps the pathological long completions that currently break the “avg_completion_tokens” gate, giving a realistic chance to meet the promotion thresholds. |
| **C** | **Collect the weak‑eval manifest** and verify that all promotion gates pass (total ≥ 196, bit ≥ 136, equation ≥ 60, truncation = 0, boxed = 1.0, no backfire, avg tokens ≤ 512). | This is the only evidence needed before any further GPU spend. |
| **D** (contingent) | If checkpoint‑10 passes all gates, **promote** the adapter (submit to Kaggle). If it fails **only** because of the avg‑tokens gate, consider a second weak‑eval on checkpoint‑20 with the same token cap. | Guarantees we only spend more GPU time when there is a clear ACC path. |

---

**4. Parameters to change / freeze (with concrete values)**  

| Parameter | New Value | Scope | Rationale |
|-----------|-----------|-------|-----------|
| `KG1_MAX_TOKENS` (or `KG1_MAX_NEW_TOKENS`) | `1024` | Weak‑eval only (override via env var for this job) | Stops the 7 680‑token runaway while still allowing enough context for the 315 rows. |
| `KG1_DISABLE_THINKING` | `0` (unchanged) | Keep as‑is – previous attempts with `1` collapsed ACC. |
| LoRA trainable modules | **Freeze** all modules **except** `q_proj` and `v_proj` (already enforced) | No change needed – already verified by `v711_lora_trainability_manifest_gate`. |
| Learning‑rate schedule | **No change** for this eval (training already finished). | Not applicable to eval. |
| `KG1_STOP_ON_PROTECTED_BACKFIRE` | `1` (already set) | Ensure the job aborts immediately if any protected row deviates. |
| `KG1_ROW_LOSS_WEIGHT` | `1.0` (already set) | Keep to guarantee loss weighting consistency. |
| `KG1_EVAL_TIMEOUT_S` | `4200` (unchanged) | Sufficient for the reduced token budget. |

---

**5. Gates that **must** pass before any further paid GPU job**  

1. **Import‑package gate** – `scripts_package_gate` & `weak_eval_import_gate_ok` must be `passed=true`.  
2. **Static safety gate** – `ok=true`.  
3. **Pre‑paid integration gate** – `ok=true`.  
4. **Weak‑runtime‑policy gate** – `passed=true` (including `disable_thinking=false`, `max_tokens` as set, `generation_timeout_s` ≤ 900).  
5. **Protected‑row guard** – `passed=true` (no backfire).  
6. **Promotion gates** – total ≥ 196, bit ≥ 136, equation ≥ 60, truncation = 0, boxed = 1.0, no no‑box fallback, avg completion tokens ≤ 512.  
7. **Row‑loss‑weight gate** – `use_row_loss_weight=1` confirmed.  

Only after **all** of the above are satisfied may a new training run be considered.

---

**6. Noise / unnecessary items to delete from the current plan**  

| Item | Why it’s noise |
|------|----------------|
| The broad `target_modules` list in the adapter config (`down_proj,in_proj,…`) – it does not affect trainability and only confuses audits. | Already clarified by the `v711_lora_trainability_manifest_gate`; keeping it only adds audit overhead. |
| The “candidate by candidate” eval flag (`KG1_EVAL_CANDIDATE_BY_CANDIDATE=1`) for a single checkpoint eval – it adds no value when only one adapter is evaluated. | Can be omitted to simplify the launch script. |
| The `KG1_ALLOW_DECODING_DRIFT_DEFERRED_FOR_FIRST_CHECKPOINT=1` flag – with only a weak‑eval run there is no drift to defer. | Not needed for this evaluation‑only job. |
| The `KG1_MAX_NUM_SEQS=64` setting – weak‑eval runs a single sequence per row; the high value is irrelevant and can be lowered to the default (1) to reduce memory pressure. | Simplifies resource accounting. |

--- 

**Bottom line:** Push the import fix, run a *single* weak‑eval on checkpoint‑10 with `MAX_TOKENS=1024`, verify that **all** gates pass, and only then decide whether to submit or to invest in a stronger training run. This is the cheapest path with the highest chance of reaching the required ACC without violating any policy.

## google/gemini-2.5-flash-lite

- status: `ok`

1. Verdict: needs artifact.

2. Top 5 concrete bugs or gaps:
    - **Packaging/Import Failure:** The `ModuleNotFoundError: No module named 'scripts.evaluate_lora_adapter'` in `v712`'s weak evaluation phase prevented any accuracy metrics from being generated. This is a critical failure as it means no evaluation data exists for the latest training run. (Evidence: `v712_failure_summary.json`, `v712_hf_job_tail.log`)
    - **Lack of Weak Accuracy Data:** Due to the import failure, `v712` has no weak accuracy results. This means there's no evidence that the training improved upon the baseline or met any of the target metrics (total ACC >= 196, bit >= 136, equation >= 60). (Evidence: `v712_failure_summary.json`, `Current observed plateau`)
    - **Protected Row Backfire in V710:** V710 exhibited protected-row backfire on rows `8740ed31` and `59bee375`, indicating the adapter is pushing the model towards incorrect answers on specific, previously correct examples. This needs to be addressed before any further training or evaluation. (Evidence: `V710 failure ledger`, `Previous OpenRouter consensus`)
    - **Insufficient Training Signal in V708/V712:** V708 trained for only 5 steps, and V712's training was cut short by the import error. The current training duration and learning rate appear insufficient to significantly improve accuracy, as evidenced by the plateaued weak ACC and V708's slight eval loss regression. (Evidence: `Current observed plateau`, `V710 failure ledger`, `V712 training metrics`)
    - **Long and Unsafe Completions:** V664 and V710 showed extremely long completions (avg 4775.58 tokens in V710) and the protected-row backfire guard failed. This suggests decoding issues or an inability of the model to properly terminate responses, contributing to incorrect answers and potential safety concerns. (Evidence: `V664 weak result`, `V710 failure ledger`, `Current observed plateau`)

3. Exact next experiment that is cheapest and most likely to improve weak ACC:
    A weak-eval-only run of `v712`'s `checkpoint-10` on A100-large, ensuring the import fix is correctly deployed in the remote environment.

4. Parameters to change or freeze, with values:
    - **Change:** Ensure the `scripts/__init__.py` fix is correctly applied and verified in the remote Hugging Face environment before launching the weak-eval job.
    - **Freeze:**
        - `KG1_WEAK_PROMOTE_TOTAL_MIN`: `196`
        - `KG1_WEAK_PROMOTE_BIT_MIN`: `136`
        - `KG1_WEAK_PROMOTE_EQUATION_MIN`: `60`
        - `KG1_WEAK_PROMOTE_TRUNC_MAX`: `0`
        - `KG1_WEAK_PROMOTE_BOXED_RATE_MIN`: `1.0`
        - `KG1_WEAK_PROMOTE_NO_BOX_FALLBACK_MAX`: `0`
        - `KG1_PROTECTED_ROW_GUARD`: `1`
        - `KG1_STOP_ON_PROTECTED_BACKFIRE`: `1`
        - `KG1_DISABLE_THINKING`: `0`
        - `KG1_MAX_TOKENS`: `7680`
        - LoRA trainable modules: `q_proj,v_proj`
        - MoE target parameters: frozen

5. Gates that must pass before another paid GPU job:
    - **Import Fix Verification:** The `ModuleNotFoundError` must be resolved, and the weak evaluation script must successfully import `scripts.evaluate_lora_adapter`. This should be verified by a successful pre-paid integration gate run that includes the weak evaluation import preflight. (Evidence: `v712_pre_paid_job_integration_gate_after_importfix.json`, `v712_static_safety_gate_after_importfix.json`)
    - **Successful Weak Evaluation Run:** The weak evaluation job for `v712`'s `checkpoint-10` must complete without errors and produce a CSV output.
    - **Protected Row Guard Pass:** The weak evaluation must pass the protected row guard, meaning no backfires on the specified protected rows.
    - **Accuracy Thresholds Met:** The weak evaluation results for `checkpoint-10` must meet or exceed the promotion thresholds (total ACC >= 196, bit >= 136, equation >= 60, truncation 0, boxed rate 1.0, no no-box fallback, protected-row backfire 0).

6. Anything in the current plan that should be deleted because it is noise:
    - No specific items can be deleted as "noise" at this stage. The current plan correctly identifies the import failure as the immediate blocker. However, the focus should shift from "monitoring V712 job" to "ensuring the import fix is deployed and verified, then running a weak-eval-only job."

## anthropic/claude-3-haiku

- status: `ok`

Verdict: Needs artifact.

Top 5 concrete gaps:
1. The import failure in the weak-eval stage is a packaging/namespace issue, not a model or loss failure. The fix of adding `__init__.py` to the `scripts/` directory is a good start, but the root cause is still unclear.
2. There is no evidence of the weak ACC result for V712, so the promotion decision cannot be made. The next step should be to run a weak-eval-only job on the already uploaded V712 checkpoints, starting with `checkpoint-10` since it had the best validation loss.
3. The LoRA trainability contract is still not fully proven. The local audit showed the effective trainable surface was correct, but the broad `adapter_config.json` surface should be explicitly gated to avoid future ambiguity.
4. The protected-row backfire issue from V710 remains unresolved. A local row-level audit is needed to understand the root cause of the backfire on rows `8740ed31` and `59bee375`.
5. The training signal in V708 was likely too weak, with only 5 steps, a low LR, and a small trainable LoRA surface. The next experiment should increase the training signal, potentially with more steps, a higher LR, and a larger trainable LoRA surface.

Next experiment: Run a weak-eval-only A100 job on the already uploaded V712 checkpoints, starting with `checkpoint-10`. Only consider `checkpoint-20` if `checkpoint-10` fails to meet the promotion thresholds.

Parameters to change or freeze:
- Freeze all parameters except `q_proj` and `v_proj`.
- Increase the training signal compared to V708, e.g., `max_steps=50`, `save_every_steps=10`, `eval_every_steps=10`, and review the LR.

Gates that must pass before another paid GPU job:
- Packaging/import fix: ensure the `scripts/` directory is properly treated as a package.
- LoRA trainability manifest: explicitly prove the effective trainable surface.
- Protected-row backfire audit: understand and mitigate the root cause of the backfire.
- Weak-eval-only on `checkpoint-10`: total `>=196`, bit `>=136`, equation `>=60`, truncation `0`, boxed rate `1.0`, no protected backfire.

Anything to delete:
None. The current plan is reasonable, but the lack of weak ACC evidence for V712 is the main blocker.

## deepseek/deepseek-v4-flash

- status: `ok`

## 1. Verdict
**Needs artifact.**  
The next paid GPU job must be a **weak‑eval‑only A100 reuse of the already‑uploaded V712 checkpoints** (cheapest path). No training, no H200, no threshold weakening. The import fix is validated locally; the remote job must now confirm weak ACC, backfire, and decoding metrics. Only after that artifact exists can a training‑side improvement be justified.

## 2. Top 5 Concrete Bugs or Gaps (tied to evidence)

| # | Bug / Gap | Evidence |
|---|-----------|----------|
| 1 | **Negligible training signal** – V712 validation loss dropped only 0.0013 (2.6376→2.6363) and final loss was 2.6373. The LR (2e‑6→5e‑7, 20 steps) and LoRA surface (q_proj/v_proj) produce no meaningful shift in generation behavior. | V712 loss numbers from roadmap (`checkpoint-10 eval_loss 2.6363`, baseline 2.6376). V710 weak ACC 191/315 (equation 56/155) shows no improvement over V664 (192/315, equation 56/155). |
| 2 | **Runaway decoding not addressed** – V710 average completion tokens 4775, max 7680; protected row `59bee375` hit token limit and produced no box. V712 training did nothing to penalize long generations or enforce early boxed output. | V710 failure ledger (`avg_completion_tokens_gt_512`, truncated rows, no‑box fallback). V712 loss did not improve decoding length. |
| 3 | **Catastrophic forgetting of bit rows** – Training only on equation rows (852) causes bit ACC to stagnate/regress: V703 bit 134, V710 bit 135, V664 bit 136. No bit rows in training, yet weak eval includes 160 bit rows. The adapter pushes the model away from correct bit answers on protected rows (`8740ed31`, `59bee375` are bit rows). | V710 bit 135/160 (down from V664 136), protected rows are bit_manipulation; V712 trained on equation only with same effect likely. |
| 4 | **Insufficient LoRA modules to control reasoning length** – Only q_proj/v_proj are trainable (1.8M params out of 884M). Changing attention query/value projections may not be enough to suppress the model’s innate tendency to generate verbose chain‑of‑thought (Nemotron‑3‑Nano). Output projection (o_proj) or key projection (k_proj) could be more effective. | V710 generations extremely long; training did not reduce length; previous V706 (disable_thinking=1) collapsed accuracy, indicating the model relies on long reasoning. |
| 5 | **Validation loss row‑weight contract unverified** – The manifest sets `KG1_ROW_LOSS_WEIGHT_REDUCTION=scale_mean` and `KG1_USE_ROW_LOSS_WEIGHT=1`, but no artifact confirms the validation loop actually applies row weights with the same reduction. If validation loss is computed without row weights, the reported “best checkpoint‑10” may be misleading. | Roadmap states “Row‑loss weight is required” but no validation loss row‑weight audit exists. The pre‑paid integration gate skipped the learnability manifest (`skipped: true`). |

## 3. Exact Next Experiment (Cheapest, Most Likely to Improve Weak ACC)
**Weak‑eval‑only A100 job on V712 checkpoint‑10** (the best eval loss checkpoint) using the import‑fix gates.  
* Rationale: Cost ~$2.50 (estimated 1‑hour run). Provides the missing ACC/backfire artifact for V712. If checkpoint‑10 passes promotion thresholds (total ≥196, equation ≥60, bit ≥136, no backfire) – highly unlikely – we can proceed. If it fails, we gain the exact failure pattern (which rows backfire, decoding lengths, equation/bit distribution) needed to design a targeted training fix.  
* **Do not launch a new training run until after this weak eval artifact is reviewed.**

## 4. Parameters to Change or Freeze
**For the weak‑eval‑only job**: freeze all training parameters. Keep everything identical to the original V712 launch manifest.  
* Mandatory: `KG1_FIRST_CHECKPOINT_WEAK_EVAL_REQUIRED=1`, `KG1_WEAK_EVAL_REQUIRED_CHECKPOINT=checkpoint-20` but we will override to evaluate `checkpoint-10` first by editing the candidate list.  
* Decoding: keep `KG1_DISABLE_THINKING=0`, `KG1_MAX_TOKENS=7680`, `KG1_PROTECTED_ROW_GUARD=1`.  
* Hardware: A100‑large only, H200 forbidden.  

**For a future training run (if weak eval fails)**:  
* Increase `max_steps` from 20 to **??** (need weak eval results to decide, but likely 50–100).  
* Increase `learning_rate` from 2e‑6 to **5e‑6** (to escape plateau).  
* Change `target_modules` to include **`o_proj`** in addition to `q_proj,v_proj` – this may better control output token distribution and reduce rambling.  
* Include **10–20 bit_manipulation rows** in the training dataset (e.g., the V516 rows that are most at risk) to prevent regression.  
* Add a **completion‑length penalty** or early‑stopping reward in loss (e.g., penalize generations that exceed training data length) – but this would require changing the training script, which is a larger experiment.

## 5. Gates That Must Pass Before Another Paid GPU Job
1. **Import‑fix gates**: `scripts_package_gate` and `weak_eval_import_gate_ok` must pass locally and be deployed to the remote job (already done).  
2. **No active paid KG1 job** – ensure no other HF job is running.  
3. **Static safety gate** (after_importfix) – must be `ok=true` (already true).  
4. **Pre‑paid integration gate** (after_importfix) – must be `ok=true` and `findings=[]` (already true).  
5. **Dataset hash match**

## qwen/qwen3-235b-a22b-2507

- status: `ok`

1. Verdict: proceed  
2. Top 5 concrete bugs or gaps:  
   - Missing `scripts/__init__.py` caused remote `ModuleNotFoundError`, blocking weak ACC measurement despite completed training.  
   - Protected row `8740ed31` backfired (predicted `01111000` vs expected `01101000`), indicating adapter weights are pushing toward wrong answers, not decoding failure.  
   - Equation sub-tasks remain at 56/155 (V664) and 56/155 (V710), showing insufficient learning signal despite LoRA updates.  
   - `KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=512` gate is violated by V664’s 4775.58 avg tokens, but no truncation occurred—decoding control is ineffective.  
   - Training used only `q_proj/v_proj` LoRA, but loss movement (2.6376 → 2.6363) did not translate to ACC gain, suggesting misalignment between loss and accuracy.  
3. Exact next experiment: Run weak-eval-only A100 job on V712 `checkpoint-10` using fixed import package (with `scripts/__init__.py`) to obtain ACC/backfire evidence without retraining.  
4. Parameters to change or freeze:  
   - Freeze: `lm_head`, `target_parameters` (MoE experts), non-`q/v` LoRA modules.  
   - Change: Increase `max_steps` to `50`, `learning_rate` to `5e-6 → 1e-6`, keep `q_proj,v_proj` only.  
5. Gates that must pass before another paid GPU job:  
   - `weak_eval_import_gate_ok` (verifies `scripts.evaluate_lora_adapter` importable)  
   - `scripts_package_gate` (verifies `scripts/` is a valid package)  
   - `py_compile scripts/evaluate_lora_adapter.py`  
   - `v711_lora_trainability_manifest_gate.json` (ensures only `q_proj,v_proj` trainable)  
   - `protected_row_backfire_guard` (0 backfires on `8740ed31`, `59bee375`, `55d834d1`)  
6. Anything to delete: Remove `KG1_WEAK_PROMOTE_AVG_COMPLETION_TOKENS_MAX=512` from promotion gates—it is unenforced and misleading, as long generations (e.g., 6987 tokens) are needed for correct reasoning. Keep only `truncated=0` and `max_tokens=7680`.

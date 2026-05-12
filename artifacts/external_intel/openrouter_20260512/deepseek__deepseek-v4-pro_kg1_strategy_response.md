**1. Ranked next experiments (adapter-only compliant, low-cost priority)**

| Rank | Experiment | Expected Benefit | Risk | Cost | Evidence Basis |
|------|------------|------------------|------|------|----------------|
| 1 | **Full-adapter SFT on high-quality bit+equation data** (Konbu17 synthetic bit 3000 rows, Kienngx bitwise 1489 rows filtered for correctness, ReasoningGym simple_equations/cryptarithm ~2000 rows, plus 10% side-family replay) | +2–4 correct on bit/equation; potential to reach 0.87 public | Side-family degradation if replay insufficient | Low: single GPU (RTX 6000), few hours, existing datasets | Konbu17 synthetic bit has `true_rule`/`solver_correct`; Kienngx has 1489 bitwise rows; ReasoningGym provides verified equation tasks; V291 already 0.86, so small targeted gains plausible. |
| 2 | **Distill deterministic postprocessor corrections into full adapter via CoT generation** | +4 equation rows (matching V274/V275 gain) | CoT quality may be low; model may not internalize rule | Medium: requires generating CoT for ~4 corrected equation problems + similar cases, then SFT | V274/V275 recovered +4 equation rows on weak set; V293 lm_head-only distill failed, but full-adapter SFT with CoT could succeed. |
| 3 | **DPO on adapter using model completions scored by verifier** | +1–3 on bit/equation if reward signal clean | DPO instability; need careful hyperparameters; risk of side-family drift | Medium: multiple completions per problem, DPO training | ReasoningGym supports algorithmic scoring; Konbu17 data has true answers; can create preference pairs from correct vs. incorrect model outputs. |
| 4 | **Procedural data generation via ReasoningGym for equation/bit sub-types** | Unlimited targeted data; could fix specific error patterns | Distribution shift from competition test set | Low–medium: generation code exists, but need to validate quality | ReasoningGym has 104 tasks including simple_equations, cryptarithm; supports procedural generation. |

**Hypothesis (explicit):** Full-adapter SFT on verified reasoning traces can internalize the missing rules, unlike lm_head-only patching (V293). DPO may be unnecessary if SFT data is sufficiently targeted.

**2. Data generation/filtering recipe**
- **Bit manipulation:**
  - Use Konbu17 `3000 synthetic bit rows` with `generated_cot` and `true_rule`/`solver_correct` as positive examples.
  - From Kienngx CoT+labels, extract `bitwise` (1489 rows). Filter: keep only rows where the final answer matches a programmatic check (if verifiable) or where `solver_correct` is true. (Hypothesis: many are correct; exact proportion unknown but worth filtering.)
  - Optional: generate additional bit problems using ReasoningGym’s bitwise tasks or custom templates, ensuring answer verifiability.
- **Equation transformation:**
  - Use ReasoningGym `simple_equations` and `cryptarithm` tasks (verified correct by environment). Extract ~2000 examples with CoT (if available) or generate CoT using a strong model (e.g., Nemotron itself with high-temperature sampling, keeping only correct completions).
  - From Kienngx `symbolic/algebraic` (1022 rows), filter for correctness similarly.
  - **Do not use** kishanvavdara dataset for equation/bit: 821/823 equation_symbolic false, 1449/1602 bit false.
- **Side-family replay:** Sample 500–1000 rows from side families (gravity, unit, numeral, cipher) from Kienngx or kishanvavdara where correctness is true, to prevent catastrophic forgetting.
- **Final dataset:** ~6000–8000 rows, balanced with ~60% bit+equation, 40% side replay.

**3. Training recipe changes after V293 failure**
- **Abandon lm_head-only training.** V293 proved that concentrating a patch into the lm_head (even with multiple checkpoints) fails to internalize rule-based corrections; loss flatlined at 2.5447, no accuracy gain.
- **Train full adapter** (all adapter layers: attention, FFN). Use LoRA rank 16–32, alpha 16–32, target all linear layers in the adapter.
- **Learning rate:** 2e-5 with cosine decay, batch size 4–8, 2–3 epochs. Monitor eval loss and accuracy on a held-out set of bit/equation problems.
- **Loss masking:** Only compute loss on answer/target tokens (not full CoT) to focus on final answer correctness, or use standard next-token prediction on full CoT if CoT is high-quality.
- **Regularization:** Use a small KL penalty towards the base V291 adapter to prevent side-family drift (optional, if degradation observed).

**4. Gates/metrics to prevent wasted HF spend**
- **Local eval set:** Use the 315-row weak candidate set (from V230) as a quick proxy. It correlates with full 947-row set: V291 gets 197/315 (bit 140, eq 56). Track:
  - Overall accuracy on 315-set.
  - Bit subset accuracy (target ≥142/160).
  - Equation subset accuracy (target ≥60/155).
  - Side-family accuracy (must not drop below 100% on the 315-set’s side rows, or monitor full side accuracy if available).
- **Training gates:**
  - After 1 epoch, if bit+equation accuracy on 315-set does not improve by ≥1 point, adjust data mix or learning rate.
  - If side-family accuracy drops >1%, increase replay proportion.
  - If eval loss diverges, stop early.
- **Pre-submission check:** Run full 947-row official-like evaluation. Only submit if full accuracy ≥828/947 (current V291: 823) and bit ≥136/160, equation ≥57/155, side families 100%.

**5. Exact stop/go criteria for submission**
- **Go:** Full-adapter SFT (Experiment 1) yields ≥828/947 on local full set, with bit ≥136, equation ≥57, side families unchanged. Public LB expected to move from 0.86 to 0.87.
- **Stop/Iterate:** If after 2 training attempts with varied data mixes, no improvement >1 row on equation or bit, switch to Experiment 2 (distill postprocessor corrections) or Experiment 3 (DPO).
- **Hard stop:** If any training causes side-family accuracy to drop below 99% on full set, discard that adapter and revert to data balancing.
- **Submission decision:** Submit only the single best adapter that passes the local gate; do not submit multiple similar adapters to avoid public score ties without gain.
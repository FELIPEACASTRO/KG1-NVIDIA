## 1) Ranked next experiments

### 1) Small LoRA on upper attention+MLP, targeted only at bit/equation
**Why:** V293 showed `lm_head`-only distillation did not internalize the deterministic rule gains: weak stayed ~190–192/315, equation stuck at 56, eval loss flat at 2.5447. The strongest evidence is that the gain exists in postprocessing (+4 equation on V291 full, to 827/947) but was not absorbed by `lm_head` alone.
**Hypothesis:** the missing capacity is in hidden representations, not output biasing.

- **Expected benefit:** +1 to +3 overall, mostly from equation; possible +0 to +1 bit.
- **Risk:** moderate risk of side-family drift if trained too broadly.
- **Cost:** low to medium if adapter remains small and training set is tightly filtered.
- **Adapter-only compliant:** yes.

**Recommended form**
- LoRA on top 20–30% transformer blocks.
- Target both attention projections and MLP projections, not `lm_head`.
- Keep rank modest first.

---

### 2) Targeted SFT on high-quality synthetic equation transformations plus hard bit rows
**Why:** your only verified deployable opportunity is to make the adapter behave like the deterministic fixer. Since the public trajectory dataset is very weak for target families (equation_symbolic false 821/823, equation_numeric false 543/732, bit false 1449/1602), raw use is likely harmful.
**Hypothesis:** filtered/generated target-family data with exact verifiable labels is better than larger noisy Nemotron-CoT corpora.

- **Expected benefit:** +1 to +3, mostly equation, maybe bit.
- **Risk:** low if held-out side-family mix is preserved in training.
- **Cost:** low if procedurally generated and auto-verified.
- **Adapter-only compliant:** yes.

---

### 3) Two-stage adapter training: preserve base first, then targeted patch
**Why:** current side families are already 100%; the challenge is adding a few target-family fixes without regressions.
**Hypothesis:** a short “stability” stage on side families plus a very short targeted patch stage will reduce collateral damage better than one mixed run.

- **Expected benefit:** not direct gain; reduces regression risk while chasing +1 to +3.
- **Risk:** low.
- **Cost:** low.
- **Adapter-only compliant:** yes.

---

### 4) Bit-only focused adapter variant, only if equation-focused run plateaus
**Why:** V291 is already 135/160 bit; V230 oracle shows 140 is theoretically recoverable in known candidates, but adapter-free deterministic bit gains were not demonstrated on full like equation gains were. So bit seems less evidence-backed than equation for immediate ROI.
- **Expected benefit:** +0 to +2.
- **Risk:** medium; bit training can be brittle and may hurt non-bit tasks.
- **Cost:** low to medium.
- **Adapter-only compliant:** yes.

---

### 5) Use public weak CoT datasets only as mined hard-negative source, not primary supervision
**Why:** the supplied evidence says target-family correctness is poor in the public 9500-row trajectory set, especially equation_symbolic.
**Hypothesis:** these datasets can still help identify prompt formats and error modes, but training on their labels directly will cap or hurt target-family accuracy.

- **Expected benefit:** indirect only.
- **Risk:** high if used naively.
- **Cost:** low.

---

## 2) Data generation / filtering recipe

## A. Equation transformation dataset: primary source
Most evidence-grounded path.

**Goal:** generate exact-answer supervised examples for transformation-equation / equation_transform style tasks.

**Recipe**
1. **Procedurally generate algebraic transformation pairs**
   - Start from simple linear and multi-step equations with one variable.
   - Create equivalent transformed equations by applying one or more operations:
     - add/subtract constant both sides
     - multiply/divide both sides
     - combine like terms
     - move terms across equality
     - simplify fractions/signs/parentheses
   - Also generate distractor candidate transformations with one subtle error:
     - sign flip
     - operation applied to one side only
     - coefficient mishandling
     - invalid cancellation
2. **Auto-verify**
   - Symbolically or numerically verify equivalence between original and transformed forms.
   - Keep only examples with exact verification.
3. **Format like competition outputs**
   - If task is multiple-choice style, train on selecting the verified candidate.
   - If task is direct answer style, train on exact final answer token only.
4. **Difficulty mix**
   - 60% easy/medium one-step and two-step.
   - 30% multi-step with negatives/fractions/parentheses.
   - 10% adversarial near-miss transformations.
5. **Size**
   - Start with 2k–6k examples, not huge.
6. **Holdout**
   - Reserve 300–500 generated examples with the same template families for exact-match validation.

**Why this is grounded**
- ReasoningGym includes simple_equations and procedural generation capability.
- Your deterministic postprocessor already found +4 equation rows, so target behavior is learnable in principle.

---

## B. Bit manipulation dataset: secondary source
Use only verified rows.

**Recipe**
1. Start from **Konbu17 bit datasets**:
   - keep rows with `solver_correct=true` or equivalent verified correctness.
   - prioritize failed rows from the base model if available as hard cases.
2. Generate additional synthetic bit examples:
   - binary shifts
   - masking
   - xor/and/or
   - parity/count-bit style if they resemble observed competition bit tasks
3. Auto-verify all labels with exact bitwise execution.
4. Add “confusable” negatives differing by one operation or one bit position.
5. Keep dataset modest: 1k–3k high-quality rows.

**Why this is grounded**
- Konbu17 provides verified synthetic bit rows and metadata like `true_rule/solver_correct`.
- Public trajectory data is weak on bit correctness, so quality filtering is important.

---

## C. Side-family preservation set
Since side families are currently 100%, explicitly preserve them.

**Recipe**
- Sample a small stable set from gravity/unit/numeral/cipher families using only rows with trusted labels.
- Keep 300–800 total examples.
- Use these in every run with low weight.
- Purpose is not improvement; it is regression detection and regularization.

---

## D. Do not use as primary labels
- `kishanvavdara/nemotron-reasoning-traj` target-family rows, especially equation_symbolic/numeric, except maybe as prompt-text source after verification.
- Kienngx labels unless you independently verify answer correctness.

---

## 3) Training recipe changes after V293 failed

## Main change: stop `lm_head`-only
That failure is directly evidenced.

### Recommended adapter config
**Hypothesis:** small hidden-layer LoRA is enough for +1 to +4 without overfitting.

- Target modules:
  - attention projections + MLP projections in upper layers
- Exclude:
  - `lm_head`-only training as main strategy
- Rank:
  - start small/moderate
- Alpha/dropout:
  - conservative, to avoid side-family drift

### Recommended curriculum
**Stage 1: preservation warmup**
- Mixed side-family preservation set + a small amount of target-family clean data.
- Short run.

**Stage 2: targeted patch**
- Equation-heavy mix with some bit.
- Oversample hard verified examples.
- Keep side-family examples at 10–20% of batches.

### Loss shaping
If feasible:
- Upweight exact final answer tokens versus long rationale.
- Prefer short-answer SFT over verbose CoT if evaluator scores final answer only.
- If training format requires rationale text, keep it concise and deterministic.

### Data mixing suggestion
- 50–60% equation verified synthetic
- 20–30% bit verified
- 10–20% side-family preservation
- Optional 5–10% mined hard negatives converted to corrected examples

### Hard-example mining
From your own local eval logs:
- collect rows where V291 misses and deterministic logic would fix them
- create synthetic neighbors around those patterns
- do not train on row-level oracle outputs directly; train on generalized pattern families

### Checkpoint strategy
- Save frequent checkpoints early.
- V293’s best weak was only 192/315 at ckpt12; so use short runs with early branch/kill, not long expensive runs.

---

## 4) Gates / metrics to prevent wasted HF spend

## Pre-training gate
Run before any serious training job:
1. **Data quality**
   - target-family labels must be auto-verified at >99.5%.
   - if not, do not train on them.
2. **Format sanity**
   - ensure train target matches submission decoding style exactly.
3. **Small overfit probe**
   - can adapter overfit 200 clean target examples?
   - if no, config is wrong; stop.

## During-training gates
Evaluate every checkpoint on:
1. **Official-like full local set**
   - primary metric
2. **Bit subset**
3. **Equation subset**
4. **Side families**
5. **Truncation count**

## Minimum useful checkpoint thresholds
Relative to V291 baseline:
- Full: must reach **>= 824/947** to stay live.
- Equation: must reach **>= 57/155** at least once very early; otherwise targeted training is not doing its job.
- Bit: must not fall below **134/160**.
- Side families: no more than **1 total error increase** versus current 100% families aggregate.
- Truncation: must stay **<= 1**.

## Spend-kill rules
Kill a run if any of these hold after the first meaningful eval window:
- eval loss flat like V293 and subset scores unchanged for 2–3 consecutive checkpoints
- equation remains at 56 with no full-score gain
- side families regress by >1
- truncation rises >1
- full drops below 822 after initial warmup and does not recover by next checkpoint

---

## 5) Exact stop/go criteria for submission

Submit only if a checkpoint satisfies all:

1. **Local full official-like:** **>= 824/947**
   - 824 is the minimum evidence-backed improvement over V291.
2. **Equation:** **>= 57/155**
   - since V291 is 56 and postprocessor indicates 60 is plausible.
3. **Bit:** **>= 135/160**
   - no bit regression allowed for an equation-focused run.
4. **Side families:** preserve current perfect families, or at worst **net 0 overall full-score loss** from any side regression.
5. **Truncation:** **<= 1**
6. **Stability across nearby checkpoints/seeds**
   - at least one neighboring checkpoint should be within 1 example of the best full score.
   - if the gain appears only in a single noisy checkpoint, no-go.

### Strong-submit criterion
Prioritize submission if:
- **>= 826/947** locally, or
- **825/947** with equation **>= 58** and no regressions elsewhere.

### No-go criterion
Do not submit if:
- gain is only from bit while equation stays 56 and full is <=823,
- any checkpoint needs external postprocessing to realize gains,
- side-family regression offsets the target-family gain.

---

## Recommended immediate plan
1. Build **verified synthetic equation transformation** set.
2. Train **upper-layer attention+MLP LoRA**, not `lm_head`-only.
3. Mix in **small side-family preservation** and **verified bit** data.
4. Run **short

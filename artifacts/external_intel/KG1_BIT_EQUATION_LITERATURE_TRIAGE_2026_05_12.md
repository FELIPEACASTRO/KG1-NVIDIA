# KG1 Bit/Equation External Research Triage - 2026-05-12

This note records evidence gathered for improving the weak KG1 families:
`bit_manipulation` and `equation_transform`.

## Current Measured State

- Best submitted family profile remains the V291/V290 adapter-only line:
  - public Kaggle score: `0.86`
  - local official-like full: `823/947 = 0.8690601901`
  - `bit_manipulation`: `135/160 = 84.375%`
  - `equation_transform`: `56/155 = 36.13%`
  - side families: `100%`
  - truncation: `1`
- V274/V275 deterministic postprocessing found real equation signal:
  - weak: recovered `+4` equation rows, reaching `196/315`, `equation=60`, `bit=136`
  - full V291 predictions, if postprocessed: `827/947 = 0.8732840549`, `equation=60`, `bit=135`
  - status: not adapter-only packageable, so it is training signal, not direct submission signal.
- V293 attempted lm-head-only distillation of those deterministic fixes:
  - checkpoint-3: `191/315`, `bit=135`, `equation=56`
  - checkpoint-6: `191/315`, `bit=135`, `equation=56`
  - checkpoint-9: `190/315`, `bit=134`, `equation=56`, `trunc=1`
  - checkpoint-12: `192/315`, `bit=136`, `equation=56`
  - conclusion: lm-head-only concentrated distillation did not internalize the rule fixes.

## External Evidence

### Kaggle CLI

- Competition files are only `train.csv` and `test.csv`.
- Recent submissions confirm the plateau:
  - V291: `0.86`
  - V281: `0.86`
  - V199B/V194/V193/V192: `0.86`
- Leaderboard slice shows top visible score `0.87` and many teams tied at `0.86`; +1 to +4 local correct answers may matter.

### Kaggle Kernels

Pulled and summarized public kernels into:

- `artifacts/external_intel/kaggle_kernel_summary_20260512.md`
- `artifacts/external_intel/kaggle_kernel_summary_20260512.json`

Actionable observations:

- Public 0.86 notebooks mostly package or validate the Tinker adapter, not solve the remaining weak rows.
- Training notebooks tend to use broad target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj`, `lm_head`
  - some include expert `gate_up_proj`
- Several public recipes use high learning rates such as `2e-4`; our historical score regressions suggest this is too risky without strong gates.
- Ensembler/fusion notebooks are useful as audit patterns, but Kaggle scoring still expects a single adapter zip, so row-level routing is not directly deployable.

### Kaggle Datasets

Summarized into:

- `artifacts/external_intel/kaggle_external_dataset_summary_20260512.json`

Important data signals:

- `kishanvavdara/nemotron-reasoning-traj`:
  - `9500` rows.
  - target-family correctness is weak:
    - bit: `128/1602 true`, `1449/1602 false`
    - equation numeric: `176/732 true`, `543/732 false`
    - equation symbolic: `2/823 true`, `821/823 false`
  - do not use as direct positive SFT supervision for target families.
- `kienngx/nemotron-30b-competition-trainingdata-cot-labels`:
  - `9500` rows with labels.
  - includes `1489` bitwise rows and `1022` symbolic/algebraic rows.
  - use only after verifier filtering.
- `konbu17/bit-manipulation-cot-dataset` and `konbu17/bit-manipulation-synthetic-cot`:
  - contain bit-specific CoT, confidence, method, ambiguity, true-rule/solver metadata.
  - useful as bit hard-case or synthetic generator seed, not blindly mixed.

Raw downloaded Kaggle notebooks/datasets were removed after summary extraction to avoid local disk waste.

### Hugging Face / ReasoningGym

- `nvidia/Nemotron-RL-ReasoningGym-v1` exposes `train` split and rows with `question`, `answer`, `metadata`, `uuid`, `license`.
- First rows confirm tasks such as `cryptarithm`, `simple_equations`, `circuit_logic`, `manipulate_matrix`, and many verifiable algorithmic tasks.
- `open-thought/reasoning-gym` is directly relevant because it is procedural and algorithmically verifiable. It supports:
  - generation of many task types,
  - scoring functions,
  - cascade scoring for string/numeric/symbolic matching,
  - dynamic or pre-built datasets.

## Literature-To-Implementation Translation

These are the literature-backed ideas that are compatible with our constraints:

1. **Verifier-first synthetic data**
   - Generate target-family data only when an exact symbolic/programmatic checker can verify it.
   - Applies to equation equivalence, equation solving, bitwise rules, boolean circuits, binary transforms.

2. **Curriculum and anti-drift replay**
   - Train easy-to-medium verified examples first, then hard adversarial variants.
   - Preserve side-family replay because side families are already at `100%`.

3. **Representation-level adapter update**
   - V293 proved lm-head-only is insufficient.
   - Next adapter should update a small controlled subset of attention/MLP LoRA weights, not only final output bias.

4. **Preference/DPO only after verifier pipeline exists**
   - Generate multiple candidate completions and form correct-vs-incorrect pairs using exact verifiers.
   - Do not use noisy self-labeled trajectories.

5. **Distill deterministic V274 fixes into families, not exact weak rows**
   - Create many template-equivalent examples for:
     - signed minus/opposite-sign guard,
     - direct add over model add-variant,
     - colon abs-difference/unreverse same-length pattern.
   - Avoid memorizing public weak rows.

## Recommended Next Experiments

### V294: Verified Equation Patch, Small Representation LoRA

- Objective: recover V274's `+4` equation behavior in adapter-only form.
- Data:
  - verified equation transformation templates,
  - ReasoningGym `simple_equations` and `cryptarithm` filtered to KG1-style answer format,
  - V274-style synthetic variants,
  - strong side-family replay.
- Modules:
  - top-layer attention + selected MLP LoRA modules,
  - not lm-head-only.
- Gates:
  - weak must improve over V291/V293:
    - `overall >= 193/315`
    - `equation >= 57/155`, prefer `>= 60/155`
    - `bit >= 136/160`
    - `truncation <= 1`
  - full official-like only if weak passes.

### V295: Verified Bit Hard-Case Patch

- Objective: add `+1` to `+2` bit rows without harming equation.
- Data:
  - Konbu17 verified/synthetic bit rows,
  - custom exact bitwise generators matching Alice-style prompts,
  - hard negatives differing by one bit/op.
- Gates:
  - `bit >= 137/160`
  - `equation >= 56/155`
  - side-family full remains `100%`

### V296: Verifier-DPO/ORPO Probe

- Objective: use verifiers to create preference pairs after SFT plateaus.
- Cost control:
  - run only if V294/V295 do not improve weak by at least one row.
  - sample small candidate sets and discard if pair quality is low.

## OpenRouter Advisory

Two OpenRouter calls were made with a constrained, evidence-only prompt:

- `openai/gpt-5.4`: cost about `$0.039805`
- `deepseek/deepseek-v4-pro`: cost about `$0.0182712`

Both converged on the same practical conclusion:

- stop lm-head-only patching for this target;
- use verified target-family data;
- train a controlled representation-level adapter;
- protect side families with replay;
- submit only after weak and full gates show real adapter-only improvement.

## Primary Literature References

The references below are not direct KG1 evidence; they are method evidence. They are useful only after translation into adapter-only KG1 gates.

- Chain-of-thought prompting:
  - `https://arxiv.org/abs/2201.11903`
  - Use: rationale-style examples can improve reasoning, but KG1 should train/evaluate final answer correctness, not trust free-form rationales.
- Self-consistency:
  - `https://arxiv.org/abs/2203.11171`
  - Use: useful for teacher generation and filtering, but not directly deployable in adapter-only Kaggle scoring.
- Least-to-most prompting:
  - `https://arxiv.org/abs/2205.10625`
  - Use: decompose equation transformations and bit rules into smaller verified steps when generating synthetic data.
- Program of Thoughts:
  - `https://arxiv.org/abs/2211.12588`
  - Use: separate symbolic/bit computation into verifier code during data generation; do not rely on model arithmetic.
- STaR:
  - `https://arxiv.org/abs/2203.14465`
  - Use: generate rationales, keep only examples whose final answer is verifier-correct, then fine-tune.
- Training verifiers:
  - `https://arxiv.org/abs/2110.14168`
  - Use: verification is easier than generation; KG1 should rank/filter candidate traces before SFT.
- Synthetic prompting:
  - `https://arxiv.org/abs/2302.00618`
  - Use: generate additional equation/bit demonstrations from a small seed set, then select by exact verifier.
- ReasoningGym / verifiable rewards:
  - `https://arxiv.org/abs/2505.24760`
  - `https://github.com/open-thought/reasoning-gym`
  - Use: procedural verifiable tasks are the best match for KG1's weak families.
- Boolformer / Boolean symbolic regression:
  - `https://arxiv.org/abs/2309.12207`
  - Use: bit tasks should be treated as Boolean rule induction from examples, not as generic text reasoning.

# KG1 V201 Panel Context

## Confirmed leaderboard evidence

- Production baseline remains V194/ref `52275052`: public score `0.86`, user-confirmed rank `19/2613`, zip SHA `49886191bf9ce92a48106ebfcba407bf9edbe423a4ed8c476d1f6bdfdd210fd8`.
- V199B/ref `52325494`: public score `0.86`, not promoted because it did not beat V194.
- V198/ref `52301667`: public score `0.84`, known regression.
- V191 broad update-space/SVD soup: public score `0.78`.
- V174 focal candidate: public score `0.41`.
- Several stripped/packaging attempts scored `0.50` to `0.54`.

## Local audit findings

- 134 legacy notebooks were scanned.
- Common bad patterns:
  - auto-submit cells;
  - manual `files.upload`;
  - unguarded `KaggleApi` use inside notebooks;
  - private Kaggle kernel output attempts;
  - no best-baseline lineage guard;
  - no baseline-eval-before-training gate;
  - target/module or packaging drift.
- V199/V199B/V200A fixed the critical lineage path: exact V194 zip SHA must be present before training.
- V200A added a 5-step micro attention-only candidate from V194, but it is not a full strategy for `0.87`; it is a low-risk probe.

## Web and documentation evidence

- NVIDIA Megatron Bridge documentation for Nemotron 3 Nano says LoRA fine-tuning is supported and defaults to linear layers `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`, `in_proj`, `out_proj`.
- This supports attention/MLP module discipline, and warns indirectly against arbitrary target drift.
- OpenRouter API docs confirm OpenAI-compatible chat completions at `/api/v1/chat/completions`, so the panel can call heterogeneous models with one prompt.
- NVIDIA reasoning-model material emphasizes curated/blended/filtering pipelines and domain-wise/cascaded learning rather than broad one-shot retrains.

## Current technical constraints

- Kaggle submission is adapter-only; no runtime API solver or router can be used in the final submission.
- API models can only be used offline for analysis, dataset selection, label verification, and roadmap review.
- The next candidate must start from V194 exact zip/model/config SHA, not V199B, unless V199B is later proven rank-better.
- Promotion rule:
  - score `< 0.86`: discard;
  - score `= 0.86`: quarantine unless rank/selection improves;
  - score `> 0.86`: promote as new baseline.

## Candidate design space

1. V201A targeted solver-verified weak-category micro-train:
   - Init: exact V194.
   - Data: only solver/API-verified weak-category examples plus strong-category rehearsal.
   - Weak categories: bit manipulation, equation numeric/deduce/guess, cryptarithm/cipher edge cases.
   - Steps: 3 to 8.
   - LR: `3e-7 -> 1e-7` or lower.
   - Trainable modules: attention only.
   - Gate: final eval <= baseline eval, plus stratified no-regression.

2. V201B no-train delta interpolation:
   - Init/base: exact V194.
   - Delta source: V199B or another 0.86-compatible candidate.
   - Alpha: 1%, 2.5%, 5%.
   - Submit only if tensor-diff gate shows intended minimal module changes and local eval is non-regressive.

3. V201C package/control:
   - No training.
   - Revalidate V194 and V199B exact zips to isolate packaging vs training effects.

## What must be rejected

- Any candidate starting from V198, V195 checkpoints, or unknown Drive “best adapter”.
- Any broad soup with high alpha.
- Any notebook that auto-submits.
- Any zip with extra files, nested dirs, missing lm_head/up/down tensors, namespace drift, or known-regression SHA.
- Any candidate where local gate improves only by noise or lacks per-family no-regression checks.

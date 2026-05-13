# KG1 V333 External URL and EvoTD Audit - 2026-05-13

## Scope

URLs audited:

- `https://gist.github.com/igorrivin/f16d4f0aff1e4c2ebe7f70865e2264b9`
- `https://github.com/tonghuikang/nemotron`
- `https://github.com/NVIDIA-NeMo/Nemotron`
- `https://hackernoon.com/vision-model-showdown-we-ran-8-models-through-the-same-browser-agent-probe`
- `https://github.com/nousresearch/hermes-agent`
- `https://unsloth.ai/docs/models/nemotron-3/nemotron-3-super`
- `https://arxiv.org/html/2605.11666v1`

## Actionable Findings

### 1. `tonghuikang/nemotron` is the strongest actionable source

Verified local clone:

- Repo: `https://github.com/tonghuikang/nemotron`
- Commit: `82bd1880aa8a8986ad572ccd17ae35b2b5c7da85`
- Important files:
  - `reasoners/bit_manipulation.py`
  - `reasoners/equation_numeric.py`
  - `corpus.jsonl`
  - `problems.jsonl`
  - `train.csv`

CPU verification on official `train.csv` bit rows:

- Rows: `1602`
- Tong bit reasoner: `1364/1602 = 85.1436%`
- Current KG1 local bit solver: `1265/1602 = 78.9638%`
- Overlap both correct: `1207`
- Tong gains vs current: `157`
- Tong losses vs current: `58`
- Net gain on official train bit rows: `+99`

Decision:

- Implement a guarded CPU bit gate using Tong-style bit-pair, bitsum, stride, left/right run matching, and middle-fill logic.
- Do not submit it directly without package/rule compliance review.
- Use it first as a verifier/teacher source:
  - compare against current weak/full predictions;
  - allow only no-loss or guarded router rules;
  - generate short deterministic traces only when coverage beats V304/V303.

### 2. Tong `equation_numeric.py` exposes DSL gaps in current V274

Additional candidate families observed in Tong code that are not fully covered or not equivalently guarded in current V274/V324:

- integer division and reverse division;
- modulo and reverse modulo;
- max/min modulo variants;
- digit multiply and reverse digit multiply without only mod10;
- digit sum diff/sum;
- digit product diff/sum;
- cross multiply and reverse cross multiply;
- determinant and absolute determinant;
- reversed-result rendering and negative-result formatting.

Decision:

- V333/V334 equation CPU gate should expand the DSL with these candidates but keep V324/V329-style conflict gates:
  - rule class;
  - candidate count;
  - prompt/example uniqueness;
  - no-loss requirement;
  - block classes with mixed correct/incorrect behavior.

### 3. EvoTD paper is useful as method, not as direct data

Source:

- `https://arxiv.org/html/2605.11666v1`
- Title: `Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling`
- arXiv: `2605.11666v1`

Useful ideas:

- treat data generation as a directed search over:
  - algorithmic skills;
  - complexity attributes;
- generate executable tasks, not just text variants;
- mutate complexity while preserving skill identity;
- cross over compatible skills to create composition;
- filter tasks by:
  - executability;
  - skill alignment;
  - learnability / Zone of Proximal Development;
- use verifier-based RL/training only after the task is valid and non-trivial.

Decision for KG1:

- Do not run broad EvoTD RL training now.
- Adapt the method as a CPU-only synthetic fixture builder:
  - seed skills from accepted V324/V329 rules and Tong bit rules;
  - mutate numeric ranges, symbols, operator positions, and bit patterns;
  - verify every generated example with the same deterministic solver;
  - reject tasks that are trivial, ambiguous, or outside the current solver's supported class;
  - use the generated examples only if they create new coverage beyond V304/V331.

### 4. Infrastructure sources are lower priority

`NVIDIA-NeMo/Nemotron`:

- Useful as Nemotron infrastructure, recipes, and dataset catalogue.
- No direct rule or family-specific improvement found for `bit_manipulation` or `equation_transform`.
- Keep as P2 reference for HF image/runtime hygiene and dataset provenance.

`unsloth.ai/docs/models/nemotron-3/nemotron-3-super`:

- Useful for Nemotron 3 Super/Nano serving and fine-tuning notes.
- Router-layer fine-tuning disabled by default for stability is relevant as a caution.
- Nemotron 3 Super is not a direct Kaggle adapter-only route for the current package.
- Keep as P2 infra/teacher reference, not as immediate training path.

`nousresearch/hermes-agent`:

- Agent/RL automation framework with batch and optional RL tooling.
- No direct puzzle-family solver signal.
- Keep as P3 automation inspiration only.

`igorrivin` gist:

- Useful only as a broad model/news digest mentioning Nemotron 3 Super and the Kaggle competition context.
- No concrete solver, dataset, adapter, or validated family gain.
- Do not promote to P0/P1.

`HackerNoon vision browser-agent article`:

- Vision/browser-agent comparison; not a text puzzle solver source.
- No direct relevance to KG1 bit/equation families.
- Exclude from roadmap actions.

## Next Implementation Step

V333 CPU gate:

- Port/compare Tong `bit_manipulation.py` as a guarded local solver.
- Run on weak/full local contracts.
- Promote only if:
  - `bit_manipulation >= 136/160`;
  - no regression in total weak score;
  - conflicts are explicitly reported.

V334 equation DSL:

- Extend V324 with Tong numeric candidates.
- Evaluate only on current `equation_transform` misses.
- Promote only if:
  - at least `+1` new equation gain;
  - `0` losses;
  - no ambiguous/conflicting rule class.

V335 EvoTD-style fixture builder:

- Generate synthetic examples only from rules proven by V333/V334.
- Apply anti-leakage by `id`, normalized prompt hash, family counts, and duplicate assistant conflicts.
- Run tokenization/offset-mask gate before any HF job.
- HF GPU remains blocked unless CPU gate shows new measurable signal.

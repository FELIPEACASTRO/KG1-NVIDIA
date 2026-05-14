# KG1 V349 - Kaggle Discussion Double Check

Date: 2026-05-14

Inputs:

- `C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topic IDs.md`
- `C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topics URLs.md`

## Scope

- The two input files contain `140` unique Kaggle discussion topic IDs.
- The installed Kaggle CLI does not expose a direct forum/discussion command.
- The Kaggle web JSON endpoint is usable, but the live endpoint rate-limited the current pass.
- V349 therefore uses the verified local V328/V332 raw cache before any network request.
- Coverage after cache merge: `140/140` topics.

Latest reproducible triage manifest:

- `artifacts/v349_kaggle_discussion_double_check/20260514T003649Z/v349_kaggle_discussion_double_check_manifest.json`
- `cache_hit_count=140`
- `fetched_count=0`
- `missing=0`

## High-Impact Topics

| Topic | Evidence | Impact |
|---|---|---|
| `689915` - Open Progress Prize SFT to maximize minimum logprob | Huikang describes token-simple traces, rare-operation coverage, and min-logprob-oriented training. | Generic loss decrease is not enough; traces must be token-simple and rule-complete. |
| `688461` - Answers To Everything Data | Full reverse-engineering taxonomy for bit/equation style puzzles, including bit as per-output-bit logic and dynamic boolean grammar. | Strong source for CPU solvers and trace generator design; not directly submit-ready. |
| `690307` - Strategy to solve 85% of bit manipulation | Bit-pair / bitsum / stride method and its limits. | Already in roadmap; still the primary bit implementation direction. |
| `685886` - Synthetic bit trace prompt | Explicit bit delta, plausibility filters, and per-hypothesis verification. | Useful as trace-shape guidance for bit teacher data, not standalone evidence of gain. |
| `690756` - Two interpretations of bit manipulation | Separates full-byte unary-transform interpretation from per-bit/pair interpretation and notes unsolved 3-input dependencies. | New refinement: V349 bit gate should test both interpretations and add bounded 3-input fallback only under no-loss acceptance. |
| `684192` - Alice/operator ambiguity | Shows equation cases where target operator may be absent in examples or symbols may need cross-prompt priors. | New refinement: equation DSL must count ambiguity and abstain when the operator is not constrained by examples. |
| `694556` - Multiple valid symbolic candidates | Symbolic transformations can have multiple compatible latent rules from finite examples. | Strengthens `candidate_count`, `conflict_count`, and no-loss abstain requirements. |
| `698293` - Gold-conditioned symbolic solver | Research oracle can expose latent symbolic programs but uses the answer as a constraint. | Useful only for taxonomy/fixtures; not valid inference logic or direct submit route. |
| `693260` - Synthetic CoT high train accuracy can drop LB | High synthetic bit/COT accuracy did not guarantee leaderboard transfer. | Confirms FinOps rule: no HF GPU without CPU weak-gate signal and early ACC kill-switch. |
| `687798` - Metric update | Binary answers are exact strings. | Already enforced; keep exact-string scorer and brace/boxed tests. |

## Net-New Actionable Updates

1. `bit_manipulation`: V349 should evaluate full-byte unary transform, per-output-bit pair/bitsum/stride, and bounded 3-input fallback for residual hard cases, with no-loss acceptance before any LoRA transfer.

2. `equation_transform`: V349/V350 equation gate should add ambiguity accounting: unknown/unseen target operator, multiple candidate symbolic programs, cross-symbol prior only if supported by train taxonomy, and abstain unless the candidate is unique or passes no-loss validation.

3. Training policy stays unchanged but now has stronger evidence: do not run broad SFT because `eval_loss` can fall without ACC gain; train only from CPU-verified hard positives plus hard negatives; kill on the first checkpoint if weak ACC does not improve and bit is not preserved.

## Not Submit-Ready

None of these discussions produced a new adapter-only candidate.

Solver/verifier gains remain real but external to a legal adapter-only package:

- adapter-only best: `192/315`, `equation=56/155`, `bit=136/160`;
- CPU solver/verifier best: `199/315`, `equation=63/155`, `bit=136/160`;
- full verifier potential: `838/947`, but not a legal direct adapter-only submission.

## Decision

V349 discussion double check does not justify a new HF job by itself.

It justifies a stricter CPU gate:

- `V349/V350 bit solver extension`: full-byte + bit-pair + bounded 3-input fallback.
- `V349/V350 equation ambiguity gate`: unique-candidate requirement and abstain logic.
- HF remains blocked until CPU weak gate shows new no-loss gain over V343/V348.

# V512 Discussion Double Check After V514

- Generated UTC: `2026-05-16T21:59Z`
- Scope: local V512 cache with `140` Kaggle discussion topics and `586` scanned posts.
- Purpose: look for items that were not yet reflected in the active roadmap after V514.

## Result

No new public source changes the main direction: CPU-first, short verified traces, no broad SFT, no runtime solver/verifier in submit.

Two refinements were underrepresented and were added to the roadmap:

1. `bit_manipulation` residual gate:
   - Discussion `690756` distinguishes full-byte/global search from per-bit/pair search.
   - Full-byte/global search has lower divergence when constrained by hierarchy, while unconstrained per-bit matching can overfit examples.
   - Impact: before any GPU, run a V514b CPU residual gate on the `161` bit rows dropped by V514. Try full-byte/ternary/3-input solvers, but accept only unique, conflict-free predictions with short traces.

2. `equation_transform` semantic ambiguity gate:
   - Discussions `684192`, `684432`, and `691641` repeatedly show query operators absent from examples or underconstrained by too few same-operator examples.
   - Impact: mark each equation row with `query_operator_seen`, `same_operator_examples`, `candidate_count`, `conflict_count`, and `derivable_vs_guess`. Rows marked ambiguous/guess cannot become promotional teacher rows.

## Confirmed Existing Rules

- `690307`: bit-pair/bitsum/stride remains the strongest public bit blueprint.
- `689915`: winning approach used deterministic CoT and min-logprob style thinking, not generic SFT.
- `697491`: better solver coverage can still reduce LB through long traces, duplicate CoT, gradient saturation, and format clash.
- `694710`: tokenizer/response mask must be exact; loss without mask/ACC is not trustworthy.
- `698293`: gold-conditioned symbolic solvers are research oracles, not submit-safe runtime logic.

## Discarded Noise

- Public bit datasets/prompts from `685971`/`685886` are not ingested directly. They can inspire trace shape only after leakage/domain triage.
- Generic Kaggle workflow and notebook visibility posts do not alter ACC or roadmap.
- Posts reporting solver-only gains without adapter transfer do not authorize GPU.

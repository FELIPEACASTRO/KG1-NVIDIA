# V316 OpenRouter Distillation Quadruple Check

Source: `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026 (2).json`

Source SHA256: `015FC4832C0FCA26789392C99714AB038B05A67A59F58279C828AB6F68E9FC12`

Scope: fourth-pass audit of the updated OpenRouter/distillation export, the local V304/V312/V316 context, and the old V51 exact-target datasets. This pass is intentionally stricter than the previous triple-check: a finding is accepted only if it can become a concrete gate, dataset rule, trainer change, or stop condition.

## Source Coverage

- Raw JSON size: `618942` bytes.
- Extracted assistant responses: `22`.
- Substantive assistant responses: `20`.
- Candidate KG1-relevant text leaves in the raw JSON: `41`.
- KG1-relevant text characters inspected by parser: `351052`.

## Fourth-Pass Local Evidence

The most important new evidence is from the repository, not from an AI recommendation:

| Local artifact | Exact target ID hits |
|---|---:|
| `artifacts/v304_solver_trace_distill_dataset/20260512T1430Z` | `0` |
| `artifacts/v312_verifier_synthetic_distill_dataset/20260512T1545Z` | `0` |
| `data/sft_v51_perfect.jsonl` | `15` |
| `data/sft_v51_complete.jsonl` | `15` |

Interpretation:

- V304 was a pattern/solver-trace dataset, not an exact target-ID absorption dataset.
- V312 was a verifier/synthetic preference dataset, not an exact target-ID absorption dataset.
- The exact 15 target IDs existed in older V51 data, but that was not enough to produce reliable transfer into the current best adapter lineage.
- Therefore, if V316 does not promote, V317 must not be another generic "add correct rows" run. It must combine exact probes, frozen-adapter wrong outputs, answer-span weighting, deterministic variants, and bit keeper replay.

## Evidence From The Attachment

The attachment repeatedly supports these themes:

| Theme | Evidence strength | Quadruple-check decision |
|---|---|---|
| Frozen-adapter wrong outputs as real `rejected` examples | Very high | Implement if V316 fails. |
| Probe-first checkpoint gate | Very high | Implement before another expensive weak/full eval. |
| Answer-span or final-answer token weighting | High | Implement if V316 fails; target answers are very short. |
| Compact rigid traces | High | Implement; long traces dilute final-answer gradients. |
| Bit keeper replay / anti-regression | High | Required; weak bit must stay at or above current baseline. |
| Deterministic verifier variants | Medium-high | Implement only when rule-generated and verified. |
| More steps on the same objective | Low trust | Reject as first fix after V316. |
| High learning rate recipes | Low trust for this lineage | Reject until a cheap smoke proves value. |
| Dual LoRA/router | Research-only for now | Defer; too much artifact and submission complexity. |
| KL/EWC regularization | Plausible but costly | Defer behind simpler replay and probes. |

## Target IDs That Must Become Probes

Equation target IDs:

| ID | Expected answer |
|---|---|
| `7688e06e` | `-55` |
| `274def88` | `92` |
| `d1bd7478` | `30` |
| `c5b058d6` | `134` |

Bit target IDs:

| ID | Expected answer |
|---|---|
| `1abaffca` | `01000000` |
| `0e70c867` | `01000000` |
| `b8722d19` | `00100100` |
| `7192535b` | `00000010` |
| `8740ed31` | `01101000` |
| `1a7c8520` | `01100000` |
| `a6192d29` | `00001000` |
| `048cc279` | `01010000` |
| `4c327b55` | `11011100` |
| `b8aa3072` | `00000011` |
| `5ba26f21` | `01011100` |

## V316 Decision Gate

Evaluate V316 first because it is already trained and tests a different trainable-module axis (`up_proj/down_proj`).

Promotion logic:

- Promote directly only if weak eval reaches `total>=193` with `bit>=136`, no truncation regression, and equation not worse than baseline.
- If V316 reaches `bit>=136` and `equation>56` but total is still below `193`, inspect row diffs before full/package because it may contain useful merge signal.
- If V316 is flat around `191/135/56` or worse, do not spend on more steps of the same V304 objective.
- If V316 regresses to `bit<135`, stop this axis and build V317 with stronger bit keeper replay.

## Required V317 If V316 Fails

V317 should be a new objective, not a longer V304/V312 rerun:

1. Build `target_probe_manifest.json`.
   - Include the 4 equation target IDs, 11 bit target IDs, bit keepers, equation keepers, and truncation guards.
2. Run frozen baseline/adaptor inference on those IDs.
   - Store the observed wrong outputs and exact wrong final answers.
3. Build hard-negative examples.
   - `chosen` is the verified correct compact final-answer trace.
   - `rejected` is the frozen adapter's real wrong output, not a synthetic random wrong answer.
4. Use answer-span weighting.
   - Start with `3x` to `5x` multiplier on final answer tokens.
   - Keep traces compact and deterministic.
5. Add deterministic verified variants.
   - Use verifier-generated bit/equation variants only.
   - Do not use model-generated variants without deterministic checking.
6. Preserve bit behavior.
   - Full bit keeper replay is mandatory.
   - Reject any checkpoint with `bit<135`.
7. Gate before weak eval.
   - Exact target probes must move.
   - Keepers must not regress.
   - Truncation must not worsen.

## Rejected For Immediate Implementation

- High-LR suggestions such as `1e-4`, `1.5e-4`, `8e-5`, `5e-5`, or `3e-5`.
- More epochs/steps on the same V304 objective if V316 fails.
- Equation-only training without bit replay.
- Loss-only or eval-loss-only promotion.
- OpenRouter text as a measured ACC source.
- Unverified model-generated variants.
- Dual LoRA/router before exhausting a single-adapter answer-span objective.
- Full KL/EWC before a cheaper probe/replay/answer-span experiment.

## Operational Conclusion

The attachment helps decide the next experiment, but it does not prove a new score. The actionable fourth-pass conclusion is:

- Finish and weak-evaluate V316 first.
- If V316 does not promote, build V317 around exact probes, frozen wrong outputs, answer-span weighting, deterministic variants, and bit keeper replay.
- Do not spend HF budget on longer runs of the same objective unless the probe gate shows actual movement.

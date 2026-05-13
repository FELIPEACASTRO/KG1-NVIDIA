# V316/V317 Quintuple Check Across OpenRouter And IAS Files

Sources:

- `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026.json`
  - SHA256: `8E4BC64D30567AE496CBC2A8DCE4C3ADE0EBA1B00375BC5CB04B486AAFE1A03B`
  - Size: `15628566` bytes.
- `C:\Users\davis\Downloads\OpenRouter Chat Tue May 12 2026 (2).json`
  - SHA256: `015FC4832C0FCA26789392C99714AB038B05A67A59F58279C828AB6F68E9FC12`
  - Size: `618942` bytes.
- `C:\Users\davis\Downloads\ANALISE_DESAFIO_IAS_15.txt`
  - SHA256: `26C803E4A5F2BB9A2461CE4C20AB55DC6FE2D05460E71DF50F44186AFA9362D8`
  - Size: `207055` bytes; `5160` lines.

This pass combines local parsing, repository evidence, and three independent sub-agent reviews. It does not treat AI recommendations as measured KG1 score.

## Coverage

`OpenRouter Chat Tue May 12 2026.json`:

- JSON version: `orpg.3.0`.
- `36` messages, `156` items, `24` characters/personas.
- Item types: `37` message, `24` reasoning, `90` web search, `1` web fetch, `2` image generation calls, `2` model search calls.
- Structured visible message content is mostly generic `postprocessor/verifier -> LoRA` analysis. It does not contain direct measured ACC for `bit_manipulation` or `equation_transform`.
- It includes many URLs and generic LoRA claims, but those are not accepted as KG1 evidence unless corroborated by local scripts/gates.

`OpenRouter Chat Tue May 12 2026 (2).json`:

- `24` messages: `2` user prompts, `22` model responses, `20` substantive.
- KG1-specific content is present and matches prior triage: baseline `823/947`, `bit=135/160`, `equation=56/155`; postprocessor/verifier oracle `838/947`, `bit=146/160`, `equation=60/155`.
- No new measured ACC is introduced by the file.

`ANALISE_DESAFIO_IAS_15.txt`:

- Repeats the same central conclusion: no literal compilation of Python verifier/postprocessor into LoRA.
- Useful content is about offline teacher distillation, chosen/rejected pairs, loss masking, hard negatives, packaging gates, and adapter-only validation.

## New Quintuple-Check Conclusions

### 1. The direct submit gate must stay strict.

Several model responses relax promotion language to `equation>56` or use inconsistent denominators. That is not acceptable for submit promotion.

Use this distinction:

- `equation>56` is a diagnostic signal only. It can justify row-diff inspection or a follow-up experiment.
- Submit/full promotion requires `eq>=60`, `bit>=136`, `total>=193`, no truncation regression, and no full-family regression.

### 2. `bit>=136` alone is not enough.

The oracle has `11` bit gains, but a checkpoint could reach `bit=136` by moving a different row while missing the actual target IDs. Therefore V317+ must gate by exact target IDs:

- the `11` bit gain IDs must be tracked separately;
- the `4` equation gain IDs must be tracked separately;
- keepers must be tracked separately from target gains.

### 3. We already have completion masking, but not answer-span weighting.

Local scripts support completion loss masks and preference training:

- `scripts/hf_job_train_v90.py` provides loss-mask based training through tokenized assistant completions.
- `scripts/hf_job_train_v315_preference.py` implements single-policy chosen/rejected preference loss.
- V311/V312 already created verifier-distillation seed/preference data.

The missing mechanism is more specific:

- token-level extra weight inside the final answer span, especially inside `\boxed{...}` or after `ANSWER:`;
- lower or zero weight for boilerplate;
- moderate rule/check weight so the adapter learns guard conditions instead of memorizing only final strings.

This is the strongest implementable gap for V317 if V316 fails.

### 4. V51 exact rows prove that exact-row SFT is insufficient.

Local evidence:

| Artifact | Exact target ID hits |
|---|---:|
| `artifacts/v304_solver_trace_distill_dataset/20260512T1430Z` | `0` |
| `artifacts/v312_verifier_synthetic_distill_dataset/20260512T1545Z` | `0` |
| `data/sft_v51_perfect.jsonl` | `15` |
| `data/sft_v51_complete.jsonl` | `15` |

Implication:

- V304/V312 trained patterns and variants, not exact target-ID absorption.
- V51 had exact target rows, but that did not reliably transfer into the current best adapter behavior.
- V317 must use exact rows as probes and hard-negative anchors, not as plain SFT rows alone.

### 5. Verifier-to-LoRA means behavioral distillation only.

Accepted:

- Use postprocessor/verifier offline as teacher.
- Generate `prompt`, `gold`, `raw_output`, `corrected_output`, `chosen`, `rejected`, `family`, `error_type`, `source_adapter`, and hashes.
- Train the LoRA to emit correct final answers without any verifier/postprocessor at inference.

Rejected:

- Claiming regex/parser/solver/`verify_kaggle` can be compiled into LoRA weights.
- Using a verifier/postprocessor at submission inference while calling it "LoRA pure".
- Using SVD conversion unless the source is a compatible full fine-tuned or merged checkpoint of the same backbone.

## V317 Shape If V316 Fails

1. Build `target_probe_manifest`.
   - Include 4 equation gain IDs, 11 bit gain IDs, 135 bit keepers, equation keepers, truncation guards.
2. Run frozen baseline/adapter inference.
   - Store raw wrong outputs and exact wrong final answers.
3. Generate training rows with a rigid format:
   - `RULE: ...`
   - `CHECK: ...`
   - `ANSWER: \boxed{...}`
4. Use token-level answer-span weights:
   - `ANSWER` span: start `3x` to `5x`.
   - `RULE/CHECK`: moderate weight.
   - boilerplate: low or zero weight.
5. Use hard negatives from actual frozen outputs:
   - equation: `55 -> -55`, `-92 -> 92`, `03 -> 30`, `35 -> 134`;
   - bit: use the observed wrong binary output for each of the 11 target IDs and verified variants.
6. Balance equation by rule, not only row count:
   - `minus_signed_opposite_sign_guarded`: 2 target rows;
   - `colon_absdiff_unreverse_same_len`: 1 target row;
   - `add_direct_over_model_add_variant`: 1 target row.
7. Interleave data from the start:
   - equation target variants;
   - bit fullbyte target variants;
   - bit keeper replay;
   - counterexamples/guard rows.
8. Gate before weak eval:
   - reject if exact probes do not move;
   - reject if `bit<135`;
   - reject if truncation worsens;
   - full/submit promotion only with `eq>=60`, `bit>=136`, `total>=193`.

## Deferred Or Rejected

- High LR recipes from generic OpenRouter text.
- Rank `64/128` or `all-linear` sweeps as immediate HF spend. They can be smoke-tested later, but not before V316 weak eval and not before answer-span V317.
- GRPO/RAFT/RL-style training. Useful research path, but too expensive/risky before a deterministic target-probe trainer exists.
- Dual LoRA/router/merge as the next action. Defer until single-adapter answer-span training is exhausted.
- More epochs/steps of V304/V312/V313/V315 without changing the loss/data objective.
- Any promotion based on `train_loss` or `eval_loss`.

## Immediate Roadmap Decision

1. Finish V316 weak evaluation first.
2. If V316 reaches strict gate (`eq>=60`, `bit>=136`, `total>=193`), proceed to full/package checks.
3. If V316 shows partial signal (`equation>56` or target-ID movement) but misses strict gate, inspect row diffs before spending on another run.
4. If V316 is flat or regresses, implement V317 answer-span weighted distillation with target probes and frozen-output hard negatives.

# KG1 V328 - Kaggle Discussion 140 Topic Audit

Date: 2026-05-13

Inputs:

- `C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topics URLs.md`
- `C:\Users\davis\Downloads\NVIDIA Nemotron Model Reasoning Challenge - Discussion Topic IDs.md`

## Scope And Fetch Status

- Input files contain `140` topic URLs and `140` topic IDs.
- IDs and URLs match after deduplication: `140` unique topics.
- Kaggle CLI has no direct discussion/topic command in the installed CLI.
- Kaggle internal discussion endpoints were discovered from the Kaggle web bundle:
  - `discussions.DiscussionsService/GetForumTopicById`
  - `discussions.DiscussionsService/GetForumMessagesInTopic`
  - `discussions.DiscussionsService/BatchGetForumMessages`
- Cache completed for `34/140` topics before Kaggle started returning `429 RESOURCE_EXHAUSTED`.
- The rate limit prevented a complete 140-topic line-by-line audit in this pass.
- Raw cached topic JSON files are under `raw_topics/`.

This audit does not claim that all 140 topics were fully ingested. It records the concrete high-impact findings from the cached topics plus the already versioned V305 discussion audit for topics previously analyzed.

## High-Impact Findings

### 1. Submit format remains adapter-only

Sources:

- `681714` - official getting started / Discord
- `683545` - external training and upload question
- `687798` / `698106` - metric update threads
- Official rules/evaluation in V327

Finding:

- The valid Kaggle submission is still `submission.zip` with LoRA adapter files.
- Solver, verifier, postprocessor, CSV predictions, GGUF, Spaces, or API endpoints are not directly submit-ready.

Action:

- Keep V274/V275/V324 as verified solver gains only.
- Promote to Kaggle only after adapter-only weak/full gates show gains.

### 2. Bit improvement requires deterministic bit-serial traces, not generic SFT

Sources:

- `688461` - Answers To Everything Data
- `689915` - Huikang Open Progress Prize SFT/min-logprob post
- `690307` - Huikang bit manipulation strategy, already captured in prior user-provided content and roadmap
- V305 prior audit

Finding:

- `688461` states binary should be treated as eight independent one-bit problems.
- Candidate operations are scanned in a deterministic order: constants, identity, NOT, 2-input gates, then broader 3/4-input boolean gates.
- Huikang emphasizes deterministic chain-of-thought, token-level simplicity, rare-operation coverage, and min-logprob inspection.
- Generic answer-only/full-byte SFT failed in our experiments for exactly this reason: V303/V326 did not teach the stepwise policy.

Action:

- V329 should implement the full CPU bit solver/trace generator before any more bit LoRA training.
- Trace format must be short, deterministic, and token-easy, with per-output-bit verification.
- No new HF bit training unless CPU coverage is better than V304 and does not reduce already-correct bit rows.

### 3. Equation transform needs scan/verifier/fallback logic, not more epochs

Sources:

- `689915` - Huikang equation section
- `688461` - numeric/symbolic discussion and examples
- `691641` - numeric equation ambiguity
- `693260` - synthetic CoT can lower leaderboard
- V324/V325/V326 local/HF results

Finding:

- Huikang describes equation as rule discovery over two-number equations with operand/result reversal transforms and a finite operator set.
- Some equation cases are not pure direct deduction from examples; fallback choice can depend on length, reversal/sign, and learned priors.
- Correct synthetic solver data can still hurt if traces are hard to learn, oversampled, conflicting, or format-shifted.
- Our V326 smoke confirms this: the verified equation `+4` signal did not become adapter-only and equation stayed at `56/155`.

Action:

- V328/V329 should expand CPU equation DSL before any GPU:
  - operand/result reversal;
  - concat/reverse concat;
  - signed arithmetic;
  - absolute difference;
  - multiplication;
  - division/mod;
  - output-length guard;
  - second-example verification;
  - hard-stop on ambiguous candidates.
- Promote only if CPU gate shows `equation>56` with `0` losses and `bit>=136`.

### 4. Metric and formatting are score-critical

Sources:

- `687798` - binary metric update
- `698106` - brace/boxed extraction fix

Finding:

- Binary answers are exact strings, not numeric floats.
- Answers containing `}` changed metric behavior; extra trailing braces can turn a correct answer into wrong.
- Public score variance can appear after metric/runtime changes.

Action:

- Keep exact-string bit scorer.
- Keep boxed-output tests with brace edge cases.
- Do not let LoRA generate noisy completions with extra `}`.

### 5. GRPO is not the next budget-efficient move

Sources:

- `694975` - GRPO debate
- `690161` - GRPO slowness/debug thread

Finding:

- GRPO can be useful after a strong cold-start policy exists, but it is generation-heavy and expensive on Nemotron.
- Under the current HF budget, GRPO is not justified while equation remains bottlenecked by missing CPU-verifiable rules.

Action:

- Keep GRPO as P2/P3.
- Spend only after CPU solver/verifier creates a strong adapter-only training target.

## Net-New Roadmap Impact

This pass does not justify another HF GPU job.

It strengthens the existing roadmap decision:

1. Implement CPU equation DSL/synthesizer first.
2. Implement CPU bit bit-serial/pair/stride/bitsum solver next.
3. Generate small deterministic traces only from verified no-loss gains.
4. Run HF only as a short smoke after CPU gates show a new signal.

## Follow-up Required

Kaggle rate-limited the 140-topic scrape. To finish the remaining topics:

- resume later with a slower rate limiter;
- persist after each topic, as V328 already does;
- prioritize missing high-impact IDs first: `690307`, `694556`, `698293`, `688277`, `684289`, `690891`, `689840`, `685710`, `687961`;
- avoid running a high-concurrency scrape against Kaggle.

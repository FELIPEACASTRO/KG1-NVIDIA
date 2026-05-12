# V305 Requested Kaggle Discussion Audit

Discussions analyzed: 29

This audit deduplicates the requested Kaggle discussion URLs and keeps only evidence with concrete implementation impact for KG1.

## 681745 - How to Get Started + resources
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745
Priority: P3
Finding: Resource hub. The actionable signal is not algorithmic; comments and linked resources reinforce using official NeMo/Nemotron docs and not treating public notebooks/traces as authoritative without verification.
Impact: No direct score gain; useful only as source index.
Action: Keep as background; do not drive training changes from this thread alone.

## 698106 - Metric Update
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698106
Priority: P0
Finding: Metric extraction changed around boxed answer handling and brace parsing.
Impact: Submissions can differ only by formatting; local gates must preserve final answer extraction exactly.
Action: Keep final-answer suffix/boxed gates and avoid extra braces/noisy completions.

## 687798 - Rescore After Metric Update
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798
Priority: P0
Finding: Binary answers were corrected to exact-string comparison rather than float-like matching.
Impact: Bit outputs must be exact 8-character binary strings; numeric coercion is invalid.
Action: Maintain exact binary string scorer and regression tests for bit outputs.

## 681714 - Official getting started / Discord
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681714
Priority: P0
Finding: Official submission path is a LoRA adapter package, rank constrained.
Impact: Postprocessor/verifier gains are not directly submitable unless competition packaging allows them.
Action: Distill V302/V300 gains into adapter-only behavior; package gate remains adapter_config + safetensors only.

## 688360 - Distillation legality discussion
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688360
Priority: P1
Finding: Thread discusses which external/model-generated data can be distilled; no universally safe shortcut is provided.
Impact: Legal/rules risk if using closed-model outputs blindly.
Action: Use generated, deterministic, locally verifiable traces; record provenance in manifests.

## 698293 - Gold-conditioned symbolic solver
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698293
Priority: P2
Finding: Symbolic solver uses the known answer as a constraint; it is a research oracle, not inference-time solver.
Impact: Useful for understanding equation/symbolic latent rules and generating checked traces, but unsafe as direct predictor.
Action: Use only for trace design and held-out verifier research; no submit/full eval directly from gold-conditioned outputs.

## 694556 - Symbol transformation ambiguity
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556
Priority: P2
Finding: Finite examples can support multiple valid transformations; public DSL/LLM traces can regress held-out score.
Impact: Broad symbolic trace ingestion is high risk.
Action: Keep symbolic work CPU-only until no-loss gates prove a deployable rule.

## 688461 - Answers To Everything Data
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461
Priority: P0
Finding: Most actionable playbook: bit is solved as eight independent boolean functions with ordered gates, bit-serial computation and target verification; equation numeric/symbolic uses scan orders, second-example verification and hard-stop behavior.
Impact: Explains why V303 answer/full-byte traces failed and why V304 moved to bit_serial_target_verification_trace_v2.
Action: Use as blueprint for V304/V305: bit-serial traces now implemented; next add full THK-style pair/stride/bitsum scan and numeric equation scan-stop audit.

## 694975 - GRPO is must debate
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694975
Priority: P2
Finding: GRPO may help after cold-start SFT, but evidence is debated and compute-heavy.
Impact: Not the next move under small HF budget.
Action: Keep GRPO as P2 only after SFT/verifier traces show weak gains and loss/logprob bottlenecks.

## 689915 - SFT to maximize minimum logprob
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
Priority: P0
Finding: Tong emphasizes deterministic CoT, token budget, min-logprob inspection, loss masking, rare-operation coverage and training-serving alignment; winning target was around 0.877 with bit as key driver.
Impact: Loss alone is insufficient; inspect token-level low-confidence points and keep traces short/learnable.
Action: Add min-logprob audit before long HF runs; keep V304 traces under 1300 tokens and use weak gates as primary success metric.

## 685920 - Score drop / metric nondeterminism
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685920
Priority: P1
Finding: Participants saw score drift from metric update and vLLM behavior.
Impact: Public 0.86 plateau can hide small gains/losses.
Action: Compare family-level artifacts and repeat weak/full where affordable before submit.

## 684212 - Base model visualization
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212
Priority: P2
Finding: Base-model generations over train are available from Tong; useful as diagnostic/logprob reference, not a direct new solver.
Impact: Can help find prompts where the model already has priors vs where training must teach exact algorithms.
Action: Optional P2: mine base-model completions only for prompt/logprob diagnostics, not labels.

## 693260 - Synthetic CoT accuracy -> LB drop
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/693260
Priority: P0
Finding: Correct synthetic solver data can lower leaderboard if traces are hard to learn, duplicate/conflicting, oversampled, or format-shift the baseline.
Impact: Directly explains V303/V292 risks and imposes gates on V304+.
Action: Cap oversampling, preserve replay, check duplicate assistant conflicts, use low LR and stop on weak regression.

## 685031 - Open Progress Prize question
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685031
Priority: P4
Finding: Prize/logistics thread only.
Impact: No score/ACC signal.
Action: Document as no-op.

## 685710 - Deterministic solvers compute partner
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685710
Priority: P2
Finding: Donald claims factories and structured traces for all categories but provides no public artifact.
Impact: Supports solver-trace direction, but cannot be imported as evidence/data.
Action: Use as design confirmation only.

## 691318 - 0.84-0.86 variance with winning zip
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691318
Priority: P1
Finding: vLLM nondeterminism can change scores even at temperature 0.
Impact: Small deltas require controlled local evaluation, not one public score.
Action: Repeat weak eval or require bigger-than-noise local family gains before submit.

## 688482 - Midpoint cutoff logistics
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688482
Priority: P4
Finding: Leaderboard/prize timing; no algorithmic content.
Impact: No score/ACC action.
Action: No implementation change.

## 691641 - Numeric equations ambiguity
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691641
Priority: P0
Finding: Some numeric equation cases have query operators absent from examples; fallback can depend on output length, reversal, sign and learned priors rather than pure deduction.
Impact: Equation_transform +4 local postprocessor gains are plausible but must be no-loss gated.
Action: Build CPU-only V305 numeric guess fallback audit before any equation-heavy training.

## 689257 - Public leaderboard size
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689257
Priority: P1
Finding: Participants infer public LB may be only a few hundred examples, so one or two correct answers can move rank.
Impact: Micro-gains matter, but they must avoid hidden regressions.
Action: Keep +1/+2 no-loss gates; do not submit unvalidated broad changes.

## 687961 - Rank32 LoRA memory at 8192
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961
Priority: P1
Finding: Rank-32 LoRA at 8192 context is memory heavy; H200/microbatch planning matters.
Impact: Long traces and broad datasets can burn HF budget quickly.
Action: Use H200 short smoke with strict GPU/memory and checkpoint weak gates; no long run without signal.

## 684289 - Unit testing simple bit transforms
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684289
Priority: P2
Finding: Base model is only about half-correct on simple one-operation bit transforms.
Impact: Bit requires explicit SFT/verifier traces; relying on base reasoning is not enough.
Action: Use simple bit unit tests as a cheap pretrain/gate source only if provenance/no-overlap is clean.

## 688277 - Equation Transformation reasoning
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688277
Priority: P1
Finding: Community confirms bit and equation are the hardest; one participant decoded many bit samples but still lacked reproducible equation reasoning.
Impact: Supports prioritizing deterministic equation audit rather than generic prompting.
Action: Route equation work through scan/verifier/fallback rules, not LLM prompting.

## 683866 - Bit transformations uniquely determined?
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683866
Priority: P0
Finding: Some bit examples are underdetermined; multiple rules fit examples but predict different queries.
Impact: Example-fit alone is unsafe; target verification and conservative rule ordering are necessary.
Action: Add ambiguity/no-loss checks to bit solvers and do not train on unresolved ambiguous labels.

## 698649 - Stuck at 0.56 SFT approach
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698649
Priority: P1
Finding: Short CoT beat longer CoT for one participant; rank32/lower LR did not help without better data/masking.
Impact: More tokens/rank is not a solution by itself.
Action: Keep V304 traces compact; validate prompt masking and family replay before GPU spend.

## 697491 - Better dataset scored worse
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697491
Priority: P0
Finding: Algorithmic accuracy 95.8% still failed to beat 0.85 due learnability, oversampling, LR/gradient saturation and masking; traces under about 1300 tokens were reported as more practical.
Impact: This is the main training-risk warning for V304/V305.
Action: Use low-risk smoke, min-logprob/token audit, replay anchoring, oversampling cap and early kill-switch.

## 696735 - CoT length on bit
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/696735
Priority: P3
Finding: Raises token-budget tradeoff; no concrete answer in thread.
Impact: No direct new data, but consistent with compact trace policy.
Action: Keep as supporting note for trace compression.

## 690307 - Strategy to solve 85% of bit manipulation
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
Priority: P0
Finding: Tong describes a bit algorithm solving 1364/1602 train bit puzzles using pair-of-input-bit iteration, bitsum hash, stride matching and compact token budget. Comment reports even stronger bit/numeric equation solver numbers, unverified by us.
Impact: Best public blueprint for turning bit 135/160 into higher adapter behavior. V304 only implements target verification, not full bitsum/stride scan yet.
Action: Implement V305/V306 THK-style pair/stride/bitsum trace generator and evaluate before long SFT.

## 690891 - Missing pieces: equation and cryptarithm
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690891
Priority: P1
Finding: Equation/cryptarithm guess cases may be information-theoretic unless priors or output-format clues are learned.
Impact: Equation improvements need fallback priors and confidence routing, not only exact-rule deduction.
Action: Add numeric/equation fallback audit with sign, length, reversal and same-op absence features.

## 689840 - RLVR worth it?
URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689840
Priority: P1
Finding: Local SFT/CV can be misleading when category distribution is nonuniform; high local bit/equation subsets may not translate to LB.
Impact: Need stratified official-like gates.
Action: Keep V221 weak/full official-like contracts; do not trust ad hoc local CV.

## Implementation changes
- V304 bit traces were changed from full-byte final-answer style to bit_serial_target_verification_trace_v2.
- V304 dataset regenerated at artifacts/v304_solver_trace_distill_dataset/20260512T1430Z with train SHA 7935ff999cdd8318de67538922de3651170c59baa2664a10beac3334dfcf9082 and val SHA 2b06224afe035c5085798f4a4be27e764ffaebde3ff7eee11c558c0cd5bdd29d.
- V286 tokenizer gate passed for V304 suffix-mode traces with real Nemotron tokenizer, zero prompt truncation, zero fallback masks and max token length 745.

## Next actions
- Commit and push discussion audit, V304 dataset/gate, and roadmap updates.
- Before a paid HF run, implement/verify a HF preflight gate: dataset hash, tokenizer suffix mode, adapter path/config, GPU/memory, replay/duplicate conflict, weak kill-switch.
- Run one short H200 V304 smoke only; continue only if weak bit improves above 135/160 or equation above 56/155 without side-family regression.
- Build CPU-only V305 THK bitsum/stride and numeric equation fallback audits before any longer train.

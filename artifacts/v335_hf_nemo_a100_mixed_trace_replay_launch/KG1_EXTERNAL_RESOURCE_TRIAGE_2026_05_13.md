# KG1 External Resource Triage - 2026-05-13

## Scope

Inputs reviewed:

- `C:\Users\davis\Downloads\Compilacao Adicional de Recursos (Double-Check)_ Destilacao, Manipulacao de Bits e Transformacao de Equacoes.md`
- `C:\Users\davis\Downloads\Compilacao de Recursos_ Destilacao, Manipulacao de Bits e Transformacao de Equacoes (1).md`
- `C:\Users\davis\Downloads\Compilacao de Recursos_ Destilacao, Manipulacao de Bits e Transformacao de Equacoes.md`

Extracted URL inventory:

- `artifacts/v335_hf_nemo_a100_mixed_trace_replay_launch/external_resource_urls_20260513.txt`
- `artifacts/v335_hf_nemo_a100_mixed_trace_replay_launch/discussion_urls_20260513.txt`

Kaggle discussion API audit:

- `artifacts/v335_hf_nemo_a100_mixed_trace_replay_launch/discussion_audit_20260513/v332_kaggle_discussion_resume_manifest.json`
- Expected topics: `11`.
- Newly fetched: `7`.
- Missing after run: `0`.
- Errors: `0`.

## High-Signal Findings

### Bit manipulation

- Kaggle discussion `690307` remains the clearest public bit path: Tong Hui Kang describes bit-pair / bitsum / stride matching and reports `1364/1602 = 85.1%` train coverage. This aligns with our V333C result, where Tong's public reasoner reached the same `1364/1602` on train but produced `+1/-1` on the V221 weak contract, so it is not a direct override.
- Kaggle discussion `688461` expands the bit search space beyond unary/two-input gates: constants, identity, NOT, 2-input gates, asymmetric negation variants, 3-input majority/choice/parity/compositions, and 4-input compositions. This is actionable for the next CPU solver gate, but only with conflict counting and no-loss promotion.
- Kaggle discussion `693260` warns that a very accurate synthetic bit dataset can reduce LB if the trace distribution is too long or not aligned with the model's priors. This supports our current short-trace, first-checkpoint kill-switch approach.
- `konbu17/bit-manipulation-cot-dataset` contains `1134` success and `374` failed bit CoT rows, but overlaps the V221/V291 references by prompt/id (`151` total overlap across success+failed files). It must not be used directly in training/eval mixes. It can only be used as wording/taxonomy after overlap filtering.
- `konbu17/bit-manipulation-synthetic-cot` has `3000` rows and `0` overlap against V221/V291 references, but many rows expose `solver_correct=False`. It is not safe as direct SFT data. It is useful only as a fixture generator candidate if every row is rechecked by our verifier.

### Equation transform

- Kaggle discussion `698293` confirms that `equation_symbolic` has recoverable latent structure under a gold-conditioned solver: digit mappings, operator choices, interpretation mode, and query value. It is not directly usable at inference because it uses the target answer, but it validates the V329/V334 direction: mine rule classes, then promote only no-loss verifier-backed classes.
- Kaggle discussion `689877` confirms a hard failure mode: some equation prompts ask for an operator that does not appear in the examples. The practical response is not generic SFT; it is rule-prior inference with confidence gates and explicit abstain on underdetermined operator classes.
- `furkankesen/equation-solver-swap-v1` is directly relevant. It has `80` deterministic external equation rows with solver families: `rev_both_add_rev=20`, `concat=17`, `rev_both_mul_rev=11`, `mul=8`, `add=7`, `abs_sub=6`, `swap_concat=6`, `rev_both_abs_sub_rev=4`, `sub=1`. However, the deterministic subset has `7` id/prompt overlaps with V221/V291 references. It is not safe as raw training data. It is useful as a DSL taxonomy and synthetic fixture template after filtering.
- `furkankesen/hard-family-source-swap-v1` has `100` gold rows (`50` bit, `50` cipher), but also overlaps references (`6` weak, `7` full). It is lower priority than equation-solver-swap for our current target.
- `manderson240/nemotron-pure-symbolic-solver-v29` includes a simple bit solver with per-bit mapping, GF(2) XOR-linear search, constants, shifts, rotations, and basic equation heuristics. Its equation solver is too shallow for direct use, but its bit fallback classes are useful as a checklist for V333/V336 CPU bit gates.
- `manderson240/nemotron-symbolic-solver-v36-fixed` is mostly a safe-submission heuristic wrapper, not a high-accuracy solver. It fixes empty-output bugs but does not add a reliable equation gain path.

### Distillation / training methodology

- `689915` confirms the strongest public LoRA recipe is not "more epochs"; it is deterministic CoT plus minimum-logprob targeting, with rebalance of low-minlogprob traces.
- `697491` reinforces a known trap: higher synthetic solver accuracy alone can still lower leaderboard score because of trace length, gradient saturation, and distribution mismatch. This is why V335 must stay a short smoke with first-checkpoint weak gate, not a long blind training run.
- HF TRL GKD / distillation resources are method references only. They do not provide KG1-specific data or measured gains, so they are P2 unless we switch to a teacher-logit training framework.
- `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` is large, general reasoning data (`809k` train, `42.6k` validation, categories math/code/stem/chat). It is not KG1-specific enough for direct mixing. It may only support future methodology, not the current bit/equation gap.

## Accepted Roadmap Actions

1. Build the next CPU-only bit solver gate around Tong-style stride plus Donald-style boolean gate classes:
   - constants;
   - identity / NOT;
   - two-input gates and asymmetric negations;
   - majority / choice / parity;
   - conflict count per output bit;
   - no-loss promotion only.

2. Expand equation DSL using `equation-solver-swap-v1` taxonomy:
   - concat;
   - swap concat;
   - add/sub/abs-sub/mul;
   - reverse both operands, then apply op, then reverse result;
   - explicit underdetermined-operator abstain.

3. Do not ingest raw public datasets unless a CPU gate proves:
   - `id_overlap=0`;
   - `prompt_sha256_overlap=0`;
   - every row is verifier-correct;
   - family/source counts are logged;
   - weak gate projects gain with `0` losses.

4. Continue V335 only while it remains healthy and cheap:
   - inspect logs every ~40 seconds while running;
   - cancel on OOM, runtime error, upload failure, or clear no-gain signal;
   - evaluate checkpoints against weak gate before any full eval or submit.

## Rejected / Low Priority

- Generic DistilBERT, image-classification distillation, and unrelated NLP notebooks: no KG1-specific signal.
- Broad HF reasoning datasets without KG1-style prompts: no direct action.
- Raw Kienngx `final_Nemotron_training_data.csv`: `9500` rows, but it overlaps `315/315` V221 weak and `947/947` V291 full references; it is blocked for direct training in our gated pipeline.
- Direct Tong bit/equation overrides: blocked by V333C/V334 local evidence.

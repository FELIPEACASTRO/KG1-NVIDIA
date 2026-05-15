# ~anthropic/claude-opus-latest

variant: `anthropic/claude-4.7-opus-20260416`

cost: `0.250795` duration_ms: `142118` tokens: `8795`

```json
{
  "verdict": "Do NOT spend any GPU yet. Every failure mode you listed (V398/V416/V444 SFT, V436B/V440/V441 preference, V443 certified pair builder, V445 parser audit) points to the same root cause: you do not yet have a CPU-verified, leakage-free, adapter-shaped training signal that demonstrably encodes the equation_transform rule in a form the base model can imitate token-for-token. Until a CPU artifact exists that (a) reconstructs >=4-8 additional equation_transform answers that the baseline currently misses, (b) does so on rows that are provably disjoint from weak/full eval, and (c) is expressed as exact target token sequences the base model can be taught to emit, another H200 hour will reproduce the V398/V416/V444 pattern of -1 to -2 bit and flat equation. The bottleneck is signal quality and target-format alignment, not optimizer, not LR, not epochs, not data volume.",

  "key_diagnosis": {
    "primary": "Equation_transform is a rule-identification task, but every training signal you have built so far is either (i) unverified (broad/reconstructed SFT), (ii) verified but trivial and already-learned (V443 found 0 certified pairs under strict gates), or (iii) verified at the answer level but not at the trajectory level (preference on final answer / boxed payload). LoRA on a 30B MoE base will not invent a rule from final-answer supervision; it needs the intermediate reasoning tokens to match a distribution the base already produces, only nudged toward the correct rule selection.",
    "secondary": "Bit regression to 134/135 in V398/V416/V444 is a capacity/interference signature: the adapter is overwriting bit-competent attention/MLP directions because the SFT corpus is dominated by equation-style targets in a format that conflicts with how the base produces bit answers. Any new run MUST include a bit-preservation subset drawn from rows the baseline already gets right, mixed at a ratio that is itself CPU-validated by a dry-run token-overlap check, not assumed.",
    "tertiary": "V445 proves the parser is not the bottleneck and V443 proves naive string/slot DSLs do not cover the 120 equation misses. That means the missing rule family is compositional (multi-step substitution, variable renaming under constraints, or operator rewriting), and a stronger proof object is required before any token-level target can be written down honestly.",
    "what_loss_values_told_you": "Nothing. You already documented that train/eval loss and preference accuracy moved independently of family ACC. Treat all loss curves as diagnostic-only; do not let them gate promotion. This is correct and should stay locked."
  },

  "ranked_next_experiments": [
    {
      "rank": 1,
      "name": "CPU equation-DSL v2 with compositional rewrites + trajectory reconstruction",
      "hypothesis": "The 79 'boxed_or_generation_wrong' equation misses are dominated by 2-3 step rewrites (e.g. rename-then-substitute, substitute-then-simplify, reverse-concat-with-renaming) that V443's single-step DSL could not certify. A DSL that composes up to depth 3 over {rename, substitute, reverse, concat, slot-permute, constant-fold} will certify >=20 of the 120 equation misses, of which a subset will be in adapter-allowed (non-weak, non-full) rows.",
      "exact_data_source_allowed": "Public train-like rows already authorized for V435C collection. Explicitly EXCLUDE every row hash present in the weak set (315) and the official-like full set (947). Maintain the same hash-exclusion file used in V435C/D. No new data ingestion.",
      "cpu_gate_before_gpu": [
        "G1: DSL v2 must certify >=20 equation rewrites on the 120-miss audit set with leave-one-out stability (rule must hold when any one supporting example is removed).",
        "G2: Of those >=20 certified rules, >=8 must reproduce the correct boxed answer on adapter-allowed training rows that are hash-disjoint from weak/full.",
        "G3: For each certified rule, build a target trajectory by sampling 4-8 base-model continuations on the prompt and selecting the one whose token distribution is closest (by per-token logprob under the base) to a templated chain-of-rewrite that ends in the correct boxed answer. If no base sample is within a logprob threshold of the template, DROP the row. This is the 'distribution-aligned target' gate that V398/V416/V444 never applied.",
        "G4: Bit-preservation subset: select >=200 rows where baseline V290-ckpt6 already emits the correct answer; freeze those exact base outputs as targets. Verify by re-decoding under the base that the frozen targets are high-probability continuations.",
        "G5: Tokenization/offset-mask parity check vs V444 pipeline; assistant-only loss mask must cover only the reasoning+boxed span, never the prompt."
      ],
      "artifact_to_build": "A JSONL with two strata: (a) equation_rewrite_certified (>=8 rows, ideally 15-30), (b) bit_preserve_frozen (>=200 rows). Each row carries: prompt, target_tokens, base_logprob_of_target, certifying_rule_id, source_hash, leakage_check=PASS.",
      "leakage_avoidance": "Hard hash-set subtraction against weak(315) and full(947) before any row enters the JSONL. Second-pass n-gram overlap check (>=13-gram) against weak/full prompts; any hit drops the row. Log the exclusion counts; if <8 certified equation rows survive exclusion, ABORT before GPU.",
      "success_block_numbers": {
        "go_to_gpu_if": "G1>=20 AND G2>=8 AND G3 yields >=8 distribution-aligned equation targets AND G4>=200 bit-preserve rows AND G5 PASS AND leakage exclusions logged.",
        "block_gpu_if": "Any of: G2<8, G3<8, base_logprob_of_target median < a pre-registered threshold (suggest: -1.5 nats/token; calibrate on 20 baseline-correct rows first), or any weak/full hash leaks through."
      },
      "how_it_could_lift_equation_without_hurting_bit": "Distribution-aligned targets mean the adapter only has to shift probability mass toward trajectories the base already considers plausible, which is the regime LoRA actually works in. Bit-preserve frozen targets occupy a defined fraction of every batch (suggest 60-70% bit-preserve, 30-40% equation-rewrite) so the adapter cannot collapse bit-competent directions. This is the explicit mixture V398/V416/V444 did not enforce with verified base-aligned targets.",
      "why_different_from_prior_failures": "V398/V416/V444 used reconstructed/high-confidence SFT but never gated on base-model logprob of the target; targets were 'plausibly correct strings' not 'strings the base would actually produce'. V443 used a depth-1 DSL; this is depth-3 compositional with leave-one-out. V436B/V440/V441 supervised final-answer/boxed only; this supervises the trajectory aligned to base distribution. V435E built hard negatives; this builds verified positives.",
      "expected_risk_cost": "CPU cost: moderate, days of CPU for DSL v2 search but no GPU. GPU cost if gates pass: one H200 run <=45 min with first-checkpoint kill-switch. Risk: DSL v2 may also return 0 certified depth-3 rules, in which case experiment is aborted at CPU stage with zero GPU spend. Probability of any equation lift if all gates pass: realistically 25-40%; probability of bit regression if mixture and frozen targets are enforced: should be <15%."
    },
    {
      "rank": 2,
      "name": "Self-distillation from base model on rows the base already solves, with rule-tag conditioning",
      "hypothesis": "The base model already solves some equation_transform rows that the current adapter loses (because V290-ckpt6 shifted some equation behavior). Distilling the base's own correct trajectories on adapter-allowed rows, tagged by inferred rule family, can recover equation lift without inventing any new rule, and by construction cannot reduce bit if bit rows are included in the same distillation.",
      "exact_data_source_allowed": "Same V435C-authorized public train-like pool, hash-disjoint from weak/full. Use the BASE model (not the current adapter) to generate candidate trajectories.",
      "cpu_gate_before_gpu": [
        "G1: Sample N base completions (suggest N=8, temperature 0.3-0.7) for ~2000 adapter-allowed rows; keep only rows where >=2 of 8 samples produce the publicly known correct answer (these rows have labels because they are train-like, not weak/full).",
        "G2: For each kept row, select the single completion with highest base logprob among the correct ones as the target.",
        "G3: Family balance: require >=15 equation rows and >=200 bit rows survive G1-G2; if equation<15 ABORT (no signal worth GPU).",
        "G4: Re-check leakage hashes and 13-gram overlap vs weak/full.",
        "G5: Mixture pre-registration: 65% bit, 35% equation, shuffle seed fixed."
      ],
      "artifact_to_build": "Self-distill JSONL with (prompt, base_target, base_logprob, family_tag, source_hash).",
      "leakage_avoidance": "Identical to experiment 1; plus, never use weak/full labels for selection. Selection uses train-like public labels only. Document which label source is used per row; reject any row whose label provenance is weak/full.",
      "success_block_numbers": {
        "go_to_gpu_if": "G3 yields >=15 equation + >=200 bit AND median base_logprob of selected targets > -1.2 nats/token AND leakage PASS.",
        "block_gpu_if": "equation<15 after G1-G2, or median logprob worse than the pre-registered threshold, or any leakage hit."
      },
      "how_it_could_lift_equation_without_hurting_bit": "Targets are by construction in-distribution for the base, so LoRA need only sharpen, not relocate, probability mass. Bit rows in the same mixture preserve bit subspace. This is the cleanest 'do no harm' formulation available adapter-only.",
      "why_different_from_prior_failures": "Prior SFT (V398/V416/V444) used reconstructed targets whose token-level provenance was not the base's own output. This forces every target to be a literal base sample, which is the only thing LoRA can reliably amplify.",
      "expected_risk_cost": "CPU cost: significant base-model inference for sampling (~2000 rows x 8 samples). If your CPU pool cannot run the 30B base for sampling, this experiment requires a separate, pre-approved low-cost GPU sampling pass that must itself pass FinOps gates BEFORE the training H200 run. Flag this explicitly: if base sampling cost > training cost, deprioritize. Probability of equation lift: 20-35%; bit risk: low if mixture enforced."
    },
    {
      "rank": 3,
      "name": "Targeted LoRA-on-LoRA: freeze V290-ckpt6 behavior, train a delta only on certified equation rewrites",
      "hypothesis": "Bit regression in V398/V416/V444 is interference. If the new adapter is trained as a DELTA that is regularized toward V290-ckpt6 outputs on bit rows (KL or token-match penalty against the current adapter, not the base), equation gains can be added without bit loss. NOTE: this stays adapter-only at submit time because the final artifact is a single merged or re-exported LoRA; the regularizer is a training-time loss term, not a runtime component.",
      "hypothesis_caveat": "I am not certain this can be cleanly exported as a single adapter_config.json + adapter_model.safetensors if you train a second LoRA on top of a merged V290-ckpt6. Two viable export paths exist (merge V290 into base then train fresh LoRA, OR train a new LoRA from scratch with a KL-to-V290 loss term and ship only the new LoRA), but I do not know with certainty which one the official Kaggle vLLM/LoRA loader will accept without modification. This must be verified against the competition's loader spec before any GPU spend.",
      "exact_data_source_allowed": "Output of experiment 1 (certified equation rewrites + bit-preserve frozen). No new data.",
      "cpu_gate_before_gpu": [
        "G0 (NEW, blocking): Confirm in writing from competition rules or a successful dry-run that a LoRA trained against a base-with-V290-merged checkpoint is acceptable as adapter-only submission, OR confirm that KL-to-V290 as a training-time loss with from-scratch LoRA target produces a standalone adapter the official loader accepts. If neither path is confirmed, ABORT this experiment.",
        "G1-G5: inherit from experiment 1."
      ],
      "artifact_to_build": "Same JSONL as experiment 1, plus a training config that adds KL(new_adapter || V290-ckpt6) on bit-preserve rows with weight tuned on a 50-row CPU forward-pass sanity check (no training) to ensure KL magnitude is non-trivial but not dominant.",
      "leakage_avoidance": "Same as experiment 1.",
      "success_block_numbers": {
        "go_to_gpu_if": "G0 confirmed AND experiment 1 gates all PASS.",
        "block_gpu_if": "G0 unresolved, or experiment 1 gates fail."
      },
      "how_it_could_lift_equation_without_hurting_bit": "Explicit anti-interference regularizer is the most direct mechanism for the observed bit regression. Equation lift comes from the same certified-rewrite signal as experiment 1.",
      "why_different_from_prior_failures": "No prior run explicitly regularized against the current best adapter's bit behavior. All prior runs assumed mixture alone would protect bit; the V398/V416/V444 results falsify that assumption.",
      "expected_risk_cost": "Higher engineering risk because of G0 uncertainty. Do NOT pursue until experiment 1 has run and either succeeded (in which case experiment 3 may be unnecessary) or failed in a way diagnostic of interference specifically (bit drops, equation flat). Probability of equation lift conditional on experiment 1 having shown any signal: 30-45%; conditional on experiment 1 showing nothing: ~5%."
    }
  ],

  "required_cpu_gates": {
    "data_provenance": "Every training row must carry source_hash, label_source (train-like public only, never weak/full), and an explicit leakage_check field set to PASS only after hash-set subtraction AND >=13-gram overlap check against weak(315) and full(947).",
    "target_distribution_alignment": "Pre-register a base-model logprob threshold for targets (suggest median > -1.5 nats/token, p90 > -3.0). Calibrate the threshold on 20 baseline-correct rows from the bit family before applying. Drop any row whose target falls below threshold. This gate is the single most important addition vs V398/V416/V444.",
    "rule_certification": "Equation rewrites must pass leave-one-out stability on certifying examples and produce the correct boxed answer on an independent held-out adapter-allowed row before the rule's rows enter the training set.",
    "family_mixture": "Pre-register exact mixture (suggest 60-70% bit-preserve, 30-40% equation-rewrite). No mid-run rebalancing.",
    "bit_preserve_floor": ">=200 frozen bit-preserve rows whose targets are literal V290-ckpt6 or base outputs on baseline-correct rows.",
    "tokenization_offset_mask": "Same gate as V444; assistant-only loss, verified on 10 sample rows by visual diff before launch.",
    "adapter_artifact_completeness": "Dry-run load of an untrained LoRA scaffold through the official vLLM/LoRA path on a single prompt before training, to confirm the export shape is accepted. If you do not have a way to dry-run the official loader, flag this as an unknown and treat as a blocking risk.",
    "cost_preflight": "Total CPU cost of gates 1-3 above must be estimated and logged; if estimated CPU cost > 2x the planned H200 cost, reconsider whether the experiment is worth running at all."
  },

  "no_go_conditions": [
    "Any weak(315) or full(947) hash or 13-gram match leaks into training data.",
    "Fewer than 8 certified, leakage-clean equation rewrites with distribution-aligned targets.",
    "Fewer than 200 bit-preserve frozen rows.",
    "Median base logprob of equation targets below pre-registered threshold.",
    "Family mixture not pre-registered and locked before launch.",
    "First-checkpoint kill-switch shows weak total < 192 OR bit < 136 OR truncated > 0 OR equation <= 56: cancel run, do not let it cook.",
    "Any proposal involving runtime verifier/parser/postprocessor/logit processor/constrained decoding/prompt change outside the adapter: hard reject.",
    "Any proposal that uses weak or full labels to select rows: hard reject.",
    "Loss-only improvement with no family ACC gain on internal CPU eval: do not promote.",
    "Cost > 0.09 USD/min or H200 run projected > 1h without human approval: do not launch."
  ],

  "changes_to_current_gates": {
    "keep_as_is": [
      "Promotion thresholds (weak>192, equation>56 ideal>=60, bit>=136, trunc=0, then full>823).",
      "Loss values do not promote.",
      "Unit cost <=0.09 USD/min and H200 <=1h without approval.",
      "First-checkpoint kill-switch.",
      "Exact hashes/row counts/family counts logged.",
      "Tokenization/offset-mask gates.",
      "Adapter config/tensor completeness check.",
      "No weak/full leakage."
    ],
    "additions_recommended": [
      "ADD: base-model logprob threshold on every training target, pre-registered and calibrated.",
      "ADD: 13-gram overlap check vs weak/full prompts in addition to hash subtraction (hash alone misses paraphrase-level leakage).",
      "ADD: pre-registered, locked family mixture ratio per run; deviation = abort.",
      "ADD: bit-preserve floor of >=200 frozen rows in every run that touches equation.",
      "ADD: dry-run of the official vLLM/LoRA loader on an untrained scaffold adapter before any training launch, to catch export-shape issues without burning a training run.",
      "ADD: explicit label_provenance field per row (train-like vs weak vs full) and a CI-style assertion that rejects weak/full at load time.",
      "ADD: CPU-cost preflight estimate logged alongside GPU-cost estimate; abort if CPU cost dominates and the experiment is exploratory.",
      "CLARIFY: 'no prompt changes that are not part of the adapter package' should be operationalized as a byte-level diff of the inference prompt template vs baseline; any non-zero diff = reject."
    ],
    "do_not_change": "The promotion numbers themselves. They are calibrated to the locked baseline and changing them mid-campaign would invalidate comparison."
  },

  "one_hour_h200_precondition": {
    "summary": "Spend the final H200 hour only if experiment 1's CPU gates all PASS and produce a training set that, on paper, can plausibly move equation by >=4 without touching bit.",
    "exact_preconditions_all_required": [
      "P1: >=8 leakage-clean, leave-one-out-stable, depth<=3 certified equation rewrites, each producing the correct boxed answer on an adapter-allowed held-out row.",
      "P2: >=200 bit-preserve frozen rows whose targets are literal base or V290-ckpt6 outputs on baseline-correct rows, with base logprob verified high.",
      "P3: Median base logprob of all training targets above the pre-registered threshold (suggest > -1.5 nats/token); p90 above -3.0.",
      "P4: Hash-set and 13-gram leakage checks vs weak(315) and full(947) both PASS with zero hits.",
      "P5: Pre-registered family mixture (suggest 65/35 bit/equation) and pre-registered first-checkpoint kill-switch thresholds (weak>=192, bit>=136, trunc=0 at ckpt-2; abort otherwise).",
      "P6: Dry-run load of an untrained LoRA scaffold through the official vLLM/LoRA path succeeds end-to-end on a single prompt.",
      "P7: Internal CPU eval harness reproduces the V290-ckpt6 baseline (192/315, 56, 136, 0) exactly before the run, to prove the eval is not silently broken.",
      "P8: Written go/no-go decision logged with all numbers above, signed off under the existing FinOps process, with a hard <=45-minute budget (leaving 15 minutes for export, eval, and rollback)."
    ],
    "if_any_precondition_fails": "Do not launch. Spend the saved hour on either (a) deepening the DSL to depth 4 with operator-rewriting primitives, or (b) base-sampling for experiment 2. Both are CPU-only and reversible.",
    "honesty_note": "I cannot guarantee that even all 8 preconditions met will produce equation>56. The historical evidence (V398/V416/V444/V436B/V440/V441) is that LoRA transfer on this task has been very fragile. The preconditions are necessary, not sufficient. If you have any doubt about P3 or P6 in particular, the rational move is to not spend the hour and instead keep V290-ckpt6 as the final submission."
  },

  "explicit_answers_to_your_four_questions": {
    "1_are_finops_and_promotion_gates_correct": "Yes, with the additions in changes_to_current_gates. The promotion numbers, the loss-doesn't-promote rule, the unit cost cap, the first-checkpoint kill-switch, and the no-leakage rule are all correct and should not be loosened. The gaps are operational: you are not yet enforcing a target-distribution-alignment gate, a 13-gram leakage check, a pre-registered mixture, a bit-preserve floor, or an official-loader dry-run. Add those.",
    "2_most_likely_missing_ingredient_for_equation_transfer": "Base-model-distribution-aligned trajectory targets for a small set of CPU-certified compositional rewrites. Every prior failure either lacked certification (V398/V416/V444), lacked compositionality (V443 was depth-1), or lacked trajectory supervision (V440/V441 supervised only final answer or boxed payload). The combination of certified-rule + base-aligned-trajectory + bit-preserve-mixture has not been tried.",
    "3_stronger_dsl_or_different_loss_or_different_data_format_or_abandon": "In priority order: (a) different data format (base-distribution-aligned trajectory targets with bit-preserve frozen rows) is the highest-leverage change and the cheapest to test; (b) stronger symbolic DSL (depth-3 compositional with leave-one-out) is required to produce enough certified rows to feed (a); (c) a different loss (KL-to-V290 anti-interference regularizer) is a useful addition but contingent on loader-acceptance verification. Do not abandon LoRA transfer yet: you have not actually tested the configuration most likely to work. Abandon only if experiment 1 with all gates passing still fails to move equation, in which case the honest conclusion is that adapter-only on this base for this task has hit its ceiling at 192 and the right move is to submit V290-ckpt6.",
    "4_one_h200_hour_exact_precondition": "See one_hour_h200_precondition above. The single most important precondition, if forced to name one, is P3: median base logprob of training targets above the pre-registered threshold. Without that, every other gate is decoration and you will reproduce V444."
  },

  "things_i_do_not_know": [
    "I do not know the exact official vLLM/LoRA loader version and accepted adapter_config schema used by this Kaggle competition, so dry-run loader verification (P6, G0 of experiment 3) must be done by you against the actual scoring environment, not assumed.",
    "I do not know whether your CPU pool can run the 30B-A3B base for sampling at acceptable cost; experiment 2 is contingent on that and I cannot estimate the cost without that information.",
    "I do not know the exact token format (chat template, special tokens, boxed convention) of V290-ckpt6's correct outputs; the trajectory-alignment gate must use the exact same template or it will silently mis-score logprobs.",
    "I cannot verify the Tong Hui Kang bit approach reference or numbers; treat the 85%+ claim as community signal, not as a target. Your bit=136/160 is the only number that matters operationally.",
    "I do not have evidence that any specific equation rewrite rule exists in the 120-miss set; the depth-3 DSL hypothesis is a prior, not a proof. If DSL v2 finds 0 certified rewrites, that is itself a strong signal to stop and submit V290-ckpt6."
  ]
}
```

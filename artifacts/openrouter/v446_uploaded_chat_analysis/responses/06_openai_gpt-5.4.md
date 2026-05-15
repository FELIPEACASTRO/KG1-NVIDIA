# openai/gpt-5.4

variant: `openai/gpt-5.4-20260305`

cost: `0.061425` duration_ms: `61655` tokens: `3767`

```json
{
  "verdict": "Proceed only with CPU-first experiments that create adapter-trainable supervision from solver knowledge in a format the model can imitate under official decoding. Do not run another broad SFT or preference job. The highest-probability path is not a new optimizer trick; it is building a stronger equation transformation representation and only spending GPU after row-level teacher signal proves that representation can separate equation gains from bit regressions.",
  "key_diagnosis": [
    "Your current evidence says the bottleneck is not extraction, not generic SFT scale, and not current preference losses. The bottleneck is transfer: CPU-solvable rule knowledge is not being converted into a stable next-token policy that survives official decoding.",
    "Equation_transform likely requires latent algorithmic state or canonical intermediate representation that your current data does not expose. Raw input->final-answer SFT is too underdetermined; preference on failed outputs does not tell the model what internal structure to produce.",
    "Bit_manipulation is already near the practical floor/ceiling tradeoff for current LoRA capacity on this base. Any training signal that is not explicitly family-isolated will likely damage bit before improving equation.",
    "Your promotion gates are directionally correct. They are stricter than loss-based model selection and that is justified by the observed disconnect between train/eval loss and actual family ACC.",
    "The most likely missing ingredient is a canonical symbolic trace or DSL for equation_transform that is small, deterministic, and directly imitable by the model, plus family-balanced preservation constraints so bit does not drift."
  ],
  "ranked_next_experiments": [
    {
      "rank": 1,
      "name": "Equation canonical trace distillation with bit-preservation mix",
      "hypothesis": "Equation accuracy will move above 56 only if the teacher supervision includes a canonical intermediate representation of the transformation process, not just the final answer. A short deterministic trace/DSL can make the mapping learnable by LoRA while preserving bit performance by mixing in frozen-baseline-correct bit rows.",
      "exact_data_source_allowed": [
        "Public train-like allowed rows only.",
        "Inputs from the allowed training corpus only; no weak/full labels used for row selection.",
        "CPU-generated traces and final answers produced from a symbolic transformer built only from training-side inputs and permitted labels where training labels are already allowed.",
        "Bit preservation rows selected only from allowed training rows where baseline predictions already match permitted training labels; not from weak/full."
      ],
      "cpu_gate_before_gpu": [
        "Build a deterministic equation DSL that normalizes examples into a compact sequence of operations, e.g. variable renaming, side swapping, term movement, operator substitution, sign flip, simplification markers. If a row cannot be expressed, mark as UNSUPPORTED and exclude from GPU candidate set.",
        "On allowed train-like equation rows, require at least 70% trace coverage with exact replay to the known final answer. If coverage is below 70%, block GPU.",
        "For a held-out allowed train-like split, require the teacher trace to be unique or near-unique under canonicalization: less than 5% of supported rows should admit multiple trace serializations after canonicalization. If not, block GPU.",
        "Run a no-training probe using the base or current best adapter to score teacher trace tokens by forced decoding NLL. Require trace perplexity/NLL to be materially lower than free-form rationale text from prior SFT corpora on the same rows. If the model cannot even model the syntax, block GPU.",
        "Construct a family-balanced training manifest with strict cap: bit preservation rows must be at least equal in token count to equation trace rows. If this cannot be done from allowed training data, block GPU."
      ],
      "what_artifact_to_build": [
        "A deterministic `equation_trace_builder` producing JSONL training examples with fields: `prompt`, `target_trace_and_answer`, `family`, `trace_supported`, `teacher_confidence`.",
        "A canonical grammar definition and replay checker proving trace->answer correctness.",
        "A mixed SFT dataset: equation rows use canonical trace plus final answer; bit rows use answer-only preservation format sampled from allowed training rows.",
        "A small offline audit notebook/report with row counts, coverage, uniqueness, and replay success."
      ],
      "how_to_avoid_weak_full_leakage": [
        "Do not use weak/full rows or labels anywhere in builder design, row filtering, or curriculum.",
        "Design the DSL from training-side examples and generic algebraic operations, not from weak/full miss patterns.",
        "Select training rows by teacher support/confidence using only allowed train-like rows.",
        "Do not tune trace grammar against weak/full outcomes."
      ],
      "success_fail_numbers_would_allow_or_block_gpu": {
        "allow_gpu_if": [
          "Equation trace coverage on allowed train-like rows >= 70%.",
          "Exact replay correctness on supported rows >= 98%.",
          "Canonical uniqueness conflict rate <= 5%.",
          "Bit preservation pool >= 2x the number of equation rows by example count or >= 1x by token count.",
          "A tiny CPU-only behavioral probe on a local small model or forced-NLL check shows canonical traces are easier than prior free-form targets."
        ],
        "block_gpu_if": [
          "Coverage < 70%.",
          "Replay correctness < 98%.",
          "Canonical conflict > 5%.",
          "Insufficient bit preservation pool.",
          "Any evidence the trace format bloats outputs enough to increase truncation risk."
        ]
      },
      "how_it_could_realistically_improve_equation_without_reducing_bit": [
        "Equation gets a learnable, low-entropy target instead of raw final-answer mapping. This addresses the transfer gap directly.",
        "Bit is protected by explicit preservation rows and by not altering its target format.",
        "The trace is short and canonical, reducing mode spread compared with broad SFT or free-form rationale.",
        "You can optionally terminate targets with the exact official answer format used by the baseline to minimize extraction drift."
      ],
      "why_different_from_failed_attempts": [
        "Different supervision object: canonical symbolic trace, not reconstructed raw text or generic answer SFT.",
        "Different failure control: replay checker and uniqueness gate ensure the target is semantically valid and low-entropy before training.",
        "Different family strategy: explicit bit-preservation balancing rather than hoping mixed SFT will not damage bit."
      ],
      "expected_risk_cost": {
        "risk": "Medium. Main risk is that a usable canonical DSL cannot reach enough coverage or remains too unnatural for the base model.",
        "cost": "Low CPU engineering, one small GPU run only if gates pass.",
        "notes": "This is the highest-value next step because it directly tests the missing representation hypothesis."
      }
    },
    {
      "rank": 2,
      "name": "Delta-only distillation on solver-wins with baseline-preserving counterweight",
      "hypothesis": "The adapter should be trained only on rows where a trusted CPU teacher beats the current adapter, while simultaneously anchoring on rows where the current adapter is already correct, especially bit rows. This narrows the optimization target to genuine behavior changes instead of broad distribution shift.",
      "exact_data_source_allowed": [
        "Allowed public train-like rows only.",
        "Teacher outputs from permitted CPU solvers/verifiers on those rows only.",
        "Current best adapter outputs on the same allowed rows for delta detection.",
        "Allowed training labels only for auditing correctness on the train-like set."
      ],
      "cpu_gate_before_gpu": [
        "Generate a delta table on allowed train-like rows: baseline correct vs teacher correct vs both wrong.",
        "Require at least 40 high-confidence equation delta rows where teacher is correct and baseline is wrong, with teacher confidence backed by independent replay/verification. If fewer than 40, block GPU.",
        "Require bit negative deltas to be rare: among rows where baseline is correct, teacher must disagree on less than 3% of bit rows after verification. If higher, block GPU.",
        "Require answer-format conformity of teacher outputs to the official extraction style on >= 99% of candidate rows.",
        "Construct a training set with 1:1 or greater token ratio of preservation rows to delta rows. If not feasible, block GPU."
      ],
      "what_artifact_to_build": [
        "A `delta_manifest.jsonl` with fields: `input`, `teacher_target`, `baseline_target`, `teacher_verified`, `baseline_correct`, `family`, `delta_type`.",
        "A narrow SFT dataset containing only: verified positive equation deltas + baseline-preserving bit/examples + optional baseline-preserving equation examples.",
        "A row-level audit report listing exact counts by family and delta type."
      ],
      "how_to_avoid_weak_full_leakage": [
        "Delta discovery uses only allowed train-like rows.",
        "No weak/full performance used to choose rows, loss weights, or checkpoint selection.",
        "Teacher confidence and verification must be computed from train-like rows only."
      ],
      "success_fail_numbers_would_allow_or_block_gpu": {
        "allow_gpu_if": [
          "Verified positive equation deltas >= 40 rows.",
          "Teacher answer-format conformity >= 99%.",
          "Bit disagreement on baseline-correct bit rows < 3%.",
          "Preservation token ratio >= 1.0 relative to delta rows."
        ],
        "block_gpu_if": [
          "Equation delta count < 40.",
          "Any ambiguous teacher outputs that cannot be rendered in final official answer form.",
          "Bit disagreement too high.",
          "Delta rows cluster into one narrow subpattern, suggesting no generalizable signal."
        ]
      },
      "how_it_could_realistically_improve_equation_without_reducing_bit": [
        "It targets only the mistakes you need to change, reducing collateral drift.",
        "Bit is explicitly anchored using preservation examples where the current adapter is already correct.",
        "Because the teacher rows are verified and formatted exactly, the model learns the desired terminal behavior rather than broad style."
      ],
      "why_different_from_failed_attempts": [
        "This is not preference training and not broad SFT. It is supervised delta distillation with strict inclusion criteria.",
        "It avoids low-signal rows where teacher and baseline agree or both fail.",
        "It adds preservation anchors by construction, which prior hard-negative preference runs did not guarantee."
      ],
      "expected_risk_cost": {
        "risk": "Medium-high. The main risk is insufficient count/diversity of verified equation deltas to generalize.",
        "cost": "Low-to-moderate CPU, one short GPU run only if delta inventory is strong."
      }
    },
    {
      "rank": 3,
      "name": "Family-isolated low-rank adapter merge test",
      "hypothesis": "A single mixed adapter may be causing destructive interference: equation updates hurt bit. Training a tiny equation-only adapter with very conservative rank/alpha and then merging or composing against the locked baseline adapter may recover equation gains without bit loss.",
      "exact_data_source_allowed": [
        "Allowed train-like equation rows only for equation adapter training.",
        "Allowed train-like bit rows only for post-train preservation audit.",
        "No weak/full labels or row selection."
      ],
      "cpu_gate_before_gpu": [
        "Before any GPU, estimate adapter interference risk by measuring token overlap and answer-format overlap between equation and bit targets in the proposed dataset. If equation target format diverges strongly from bit target format, prefer this isolated approach; otherwise do not spend GPU here.",
        "Require experiment 1 or 2 to produce a vetted equation-only dataset first. This experiment is downstream, not standalone.",
        "Require at least one CPU evidence point that mixed data is likely the source of bit drift, e.g. prior regression concentrated after adding equation-heavy rows."
      ],
      "what_artifact_to_build": [
        "A tiny equation-only LoRA candidate trained from the locked baseline or base with minimal rank.",
        "If the infrastructure allows official-compatible LoRA composition or offline merge into a single adapter-only package, build that package; if not, do not do this experiment."
      ],
      "how_to_avoid_weak_full_leakage": [
        "All training and selection on allowed train-like only.",
        "No hyperparameter choice by weak/full."
      ],
      "success_fail_numbers_would_allow_or_block_gpu": {
        "allow_gpu_if": [
          "You have a vetted equation dataset from experiment 1 or 2.",
          "You confirm the official inference path accepts the resulting adapter package exactly as required."
        ],
        "block_gpu_if": [
          "No guaranteed official-compatible single adapter artifact.",
          "Any uncertainty about whether merge/composition preserves adapter-only submission validity."
        ]
      },
      "how_it_could_realistically_improve_equation_without_reducing_bit": [
        "Isolation reduces interference; a tiny-rank update may nudge equation behavior while leaving bit nearly untouched.",
        "This is useful only if packaging remains fully compliant."
      ],
      "why_different_from_failed_attempts": [
        "It changes the systems setup, not just the data or loss.",
        "It explicitly addresses family interference, which previous runs observed but did not isolate."
      ],
      "expected_risk_cost": {
        "risk": "High. I do not know whether your official vLLM/LoRA path and package rules permit the exact merge/composition you need without violating adapter-only requirements.",
        "cost": "Low GPU if feasible, but only after compliance is confirmed."
      }
    }
  ],
  "required_cpu_gates": [
    "No GPU unless the new data artifact encodes information not present in failed broad SFT/preference runs.",
    "For any equation-focused run, require a row-level verified teacher set on allowed train-like rows with exact final-answer renderability.",
    "Require family-separated audits: equation candidate gain potential and bit preservation risk must be quantified independently before training.",
    "Require truncation-risk audit on target lengths and format before GPU.",
    "Require exact artifact integrity checks: row counts, hashes, family counts, tokenization, offset masks, adapter package completeness."
  ],
  "no_go_conditions": [
    "Any proposal that depends on runtime verifier, parser override, postprocessor, constrained decoding, prompt modification outside adapter behavior, or test-time scripting.",
    "Any GPU run triggered only by lower loss, better preference accuracy, or more synthetic volume without new CPU evidence.",
    "Any training data selection using weak/full labels or weak/full miss rows.",
    "Any candidate that raises truncation risk materially.",
    "Any experiment whose only novelty is more epochs, LR sweep, or larger broad SFT corpus."
  ],
  "changes_to_current_gates": [
    {
      "item": "Keep current promotion gate",
      "assessment": "Correct",
      "reason": "Your observed metric-target mismatch justifies using only weak/full ACC plus truncation as promotion criteria."
    },
    {
      "item": "Add pre-GPU representation gate",
      "assessment": "Required",
      "reason": "Before GPU, require proof that the supervision format is canonical, replayable, and lower-entropy than previous targets."
    },
    {
      "item": "Add family-interference gate",
      "assessment": "Required",
      "reason": "Any equation-focused run must include explicit bit-preservation token budget and a measured negative-delta audit."
    },
    {
      "item": "Add minimum verified equation-delta count",
      "assessment": "Required",
      "reason": "Do not train on vague synthetic hope. Require a minimum inventory of verified train-like rows where the teacher beats the baseline."
    },
    {
      "item": "Keep H200 <= 1h and first-checkpoint kill-switch",
      "assessment": "Correct",
      "reason": "Given repeated non-transfer, short-run gating is appropriate."
    }
  ],
  "one_hour_h200_precondition": "I would spend the last 1-hour H200 run only if a CPU-built equation artifact satisfies all of the following simultaneously: (1) at least 40 verified equation delta rows on allowed train-like data where teacher is correct and current adapter is wrong; preferably >= 60; (2) those rows are rendered in a canonical trace or exact final-answer format with >= 98% replay/verification correctness; (3) bit-preservation examples from allowed training data provide at least a 1:1 token ratio against equation-update examples; (4) target length audit shows no increased truncation risk relative to baseline; and (5) the resulting dataset is genuinely novel versus prior failed SFT/preference corpora. Without all five, I would not spend the run.",
  "explicit_answers": {
    "1_are_finops_and_promotion_gates_correct": "Yes, mostly. They are appropriately strict given the evidence. I would keep them and add representation-quality and family-interference CPU gates before GPU.",
    "2_most_likely_missing_ingredient_for_equation_transfer": "A compact canonical symbolic representation of equation transformations that exposes intermediate state the model can imitate, plus preservation constraints to prevent bit drift.",
    "3_next_move_stronger_symbolic_dsl_different_loss_different_data_format_or_abandon_lora_transfer": "Strongest next move: a stronger symbolic DSL or canonical trace, which is fundamentally a different data format. I would not prioritize a new loss/objective before fixing representation. I would not abandon LoRA transfer yet, but I would do so after one or two gated attempts if no verified equation delta set can be turned into a canonical train target.",
    "4_if_only_one_h200_run_left_under_1_hour_what_exact_precondition": "Do not run until you have a train-like verified equation delta set of at least 40 rows, preferably 60+, expressed in a canonical replayable target format with >= 98% correctness, and enough preservation data to keep bit token budget at least equal."
  }
}
```

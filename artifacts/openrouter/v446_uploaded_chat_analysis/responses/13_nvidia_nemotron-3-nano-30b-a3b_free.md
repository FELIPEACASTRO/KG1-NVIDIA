# nvidia/nemotron-3-nano-30b-a3b:free

variant: `nvidia/nemotron-3-nano-30b-a3b:free`

cost: `0` duration_ms: `16769` tokens: `2955`

```json
{
  "verdict": "PROPOSED: concrete, adapter‑only path forward with clear CPU‑gate filters and measurable promotion thresholds.",
  "key_diagnosis": "The main missing ingredient is **program‐level symbolic structure** for the equation_transform domain.  Current LoRA training only sees raw token sequences; without an explicit DSL that encodes the syntactic and arithmetic constraints discovered by the CPU verifiers, the model cannot reliably lift equation accuracy without collapsing bit performance.",
  "ranked_next_experiments": [
    {
      "rank": 1,
      "hypothesis": "Training a LoRA that conditions on **deterministically synthesized program templates** (e.g., `x = a <op> b` where `<op>` is a bitwise or arithmetic operator) will expose the model to the correct equation reasoning scaffolding, allowing it to learns the mapping from puzzle to answer without external solvers.",
      "data_source": "`synthetic_equation_templates.json` generated from the 133 hard‑negative rows + all certified rule pairs from V443 (≈1 200 unique templates).",
      "cpu_gate": "`cpu_signal_material=true` && `has_deterministic_verifier_output=true` && `bit_accuracy_stable>=136`",
      "artifact": "`adapter_eq_template_lora.safetensors` + `adapter_eq_template_config.json`",
      "avoid_leakage": "Only rows that pass the CPU verifier **and** have a unique, non‑ambiguous program template are kept.  The verifier runs **before** any GPU work and its output hash is stored; no weak/full labels are ever fed back into the training loop.",
      "success_metric": "equation_transform ACC ≥ 60/155 **and** bit manipulation ACC ≥ 136/160, truncation = 0.",
      "fail_metric": "equation_transform ACC ≤ 56 OR bit manipulation ACC < 136 OR truncation > 0.",
      "improvement_mechanism": "Program‑template conditioning provides an explicit syntactic bias that the model can learn from; contrastive loss is applied only on the filtered template pairs to push correct templates higher in log‑probability.",
      "differentiation": "Unlike broad SFT, this approach injects **structured symbolic knowledge** that is directly tied to the missing DSL; it does not rely on raw text augmentation or additional epochs.",
      "risk_cost": "CPU verification adds ~2 min per batch; GPU cost ≤ $0.09/min for ≤ 1 h; estimated $5–$7 per run."
    },
    {
      "rank": 2,
      "hypothesis": "Using a **contrastive adapter objective** that explicitly separates *hard‑negative* equation rows from *known‑correct* rows (derived from the 201/315 CPU‑verified solutions) will shift the decision boundary toward correct program outputs without altering the decoder architecture.",
      "data_source": "`cpu_verified_eq_hard_negatives.safetensors` (≈400 rows) + `cpu_verified_eq_correct.safetensors` (≈120 rows) from V401/V428 re‑extraction.",
      "cpu_gate": "`cpu_verifier_signal=True` && `bit_accuracy_stable>=136` && `row_count_eq_correct>=120`",
      "artifact": "`adapter_eq_contrastive_lora.safetensors` + `adapter_eq_contrastive_config.json`",
      "avoid_leakage": "Hard‑negative vs. correct rows are pre‑hashed; the contrastive loss is computed only inside the LoRA training script; no runtime post‑processing.",
      "success_metric": "equation_transform ACC ≥ 60/155 **and** bit manipulation ACC ≥ 136/160, truncation = 0.",
      "fail_metric": "equation_transform ACC ≤ 56 OR bit manipulation ACC < 136 OR truncation > 0.",
      "improvement_mechanism": "Contrastive loss directly penalizes mixing of correct and incorrect equation patterns, encouraging the adapter to specialize on the correct program patterns discovered by the CPU verifier.",
      "differentiation": "Targets the *distribution shift* between correct and incorrect equation behaviours rather than adding more generic data; preserves the adapter‑only constraint.",
      "risk_cost": "CPU verification already done; LoRA training adds ~0.5 h GPU time; cost ≈ $3."
    },
    {
      "rank": 3,
      "hypothesis": "Adopt a **multi‑task LoRA** that jointly optimizes (a) standard language modelling and (b) a **rule‑preserving auxiliary loss** that rewards generation of syntactically valid bit/equation expressions (checked by a deterministic parser built on‑CPU).",
      "data_source": "`synthetic_bit_equation_rules.txt` (≈500 rule templates validated by the V443 auditor).",
      "cpu_gate": "`bit_rule_parser_stable=True` && `bit_acc_stable>=136` && `cpu_gpu_allowed=true`",
      "artifact": "`adapter_multi_task_lora.safetensors` + `adapter_multi_task_config.json`",
      "avoid_leakage": "Parser is executed solely on CPU before training; only rows that the parser deems syntactically valid are fed to the LoRA; no post‑hoc changes after generation.",
      "success_metric": "equation_transform ACC ≥ 60/155 **and** bit manipulation ACC ≥ 136/160, truncation = 0.",
      "fail_metric": "equation_transform ACC ≤ 56 OR bit manipulation ACC < 136 OR truncation > 0.",
      "improvement_mechanism": "Multi‑task loss forces the adapter to keep the bit manipulation pathway intact while nudging the equation pathway toward structurally valid outputs.",
      "differentiation": "Provides an explicit structural bias without external decoding; the parser is part of the adapter package, not a runtime component.",
      "risk_cost": "Parser generation cost negligible; LoRA training ≈ 0.8 h GPU; cost ≈ $6."
    }
  ],
  "required_cpu_gates": [
    "`bit_accuracy_stable>=136` – bit manipulation ACC must stay at least 136/160 before any GPU attempt;",
    "`equation_transform_signal=True` – a deterministic CPU verifier must flag ≥ 4 new correct equation rows beyond baseline;",
    "`cpu_verifier_output_hash` – hash stored to guarantee reproducibility and prevent leakage into later training;",
    "`hf_gpu_allowed=true` && `unit_cost_usd_min<=0.09` && `budget_check=true`"
  ],
  "no_go_conditions": [
    "equation_transform ACC ≤ 56",
    "bit manipulation ACC < 136",
    "truncated > 0",
    "total weak ACC ≤ 192/315",
    "any GPU run exceeds 1 h without meeting the above thresholds"
  ],
  "changes_to_current_gates": [
    "Tighten the equation_transform promotion gate from `>56/155` to `≥60/155` to create a clearer break‑point;",
    "Add a `program_template_uniqueness` check: each synthetic template must map to a single answer row to avoid ambiguous supervision;",
    "Make the `bit_accuracy_stable` gate stricter: reject any checkpoint where bit drops below 136 even if truncation remains 0."
  ],
  "one_hour_h200_precondition": "Before spending the hour, the CPU verifier must output **at least 4 distinct, certified program templates** for the equation_transform family that (a) were not present in the baseline training set, (b) map to **unique** answer rows, and (c) preserve bit manipulation correctness (≥ 136/160) in a test run on a dummy LoRA; only then may the H200 be allocated to train the ranked experiment that consumes those templates."
}
```

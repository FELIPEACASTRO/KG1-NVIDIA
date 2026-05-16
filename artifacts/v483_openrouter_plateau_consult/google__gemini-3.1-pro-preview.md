## Verdict
The missing `target_parameters` in the V480 continuation job is a critical, lineage-breaking bug and the primary suspect for your plateau and regressions. In Mixture-of-Experts (MoE) models like Nemotron, `target_parameters` restricts LoRA updates to specific expert layers (e.g., `mlp.experts.gate_up_proj`). Dropping this config during continuation means the framework either failed to load the seed weights correctly, applied them to the wrong tensors, or initialized new untrained LoRA adapters for all experts, instantly destroying the 0.86 plateau baseline. You must fix the PEFT loading mechanism to strictly inherit the seed config and validate tensor-key equality on CPU before launching any further paid GPU jobs.

## Evidence Assessment
*   **PROVEN**: Seed adapter V290 checkpoint-6 relies on `target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`.
*   **PROVEN**: V480 training dropped this configuration (`target_parameters: disabled`, saved as `null`).
*   **PROVEN**: V480 failed to improve weak accuracy and regressed on the bit manipulation task (136 -> 134).
*   **PLAUSIBLE**: The config mismatch caused the LoRA weights to be misapplied or reinitialized, leading to the observed eval_loss stagnation and task regression.
*   **PLAUSIBLE**: Equation +1 paired with Bit -1/-2 is a symptom of catastrophic forgetting, exacerbated by either the broken LoRA routing or an overly aggressive learning rate on a fragile, misconfigured adapter.
*   **UNKNOWN**: Whether the underlying dataset mix in V479/V480 is actually capable of reaching the promotion gate, because the PEFT bug invalidates the V480 training signal.

## Root Cause Ranking
1.  **PEFT Config/Lineage Mismatch (95% confidence)**: Dropping `target_parameters` in an MoE LoRA continuation alters which expert weights are updated/loaded. This effectively resets or corrupts the adapter lineage.
2.  **Catastrophic Forgetting of Bit Manipulation (80% confidence)**: Even with a fixed config, fine-tuning on a heavily skewed or poorly batched dataset often causes the model to forget the bit manipulation logic while slightly improving equation transforms.
3.  **Eval Loss vs. Reasoning Disconnect (75% confidence)**: `eval_loss` measures next-token prediction across all tokens (mostly formatting/copying). It is mathematically disconnected from the exact-match reasoning required to pass the weak gate.
4.  **Generation/Parsing Fragility (60% confidence)**: Truncated=1 and accuracy=0.0000 bugs strongly suggest the model's output formatting shifted slightly, breaking the regex/extraction logic during evaluation.

## Required Code/Gate Patches

**1. Correct PEFT Continuation Loading**
Do not create a new `LoraConfig` and attempt to inject weights. Use the native `PeftModel` continuation method to guarantee config inheritance:
```python
from peft import PeftModel
# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, ...)
# Load seed adapter and make it trainable
model = PeftModel.from_pretrained(model, seed_adapter_path, is_trainable=True)
```

**2. Preflight Tensor & Config Equality Gate (CPU-safe)**
Add this to `hf_job_preflight_gate.py`. It must run *before* the training loop starts:
```python
import json
from safetensors import safe_open

def verify_adapter_lineage(seed_path, output_dir):
    # 1. Check Config
    with open(f"{seed_path}/adapter_config.json") as f:
        seed_config = json.load(f)
    with open(f"{output_dir}/adapter_config.json") as f:
        new_config = json.load(f)
    
    assert seed_config.get("target_parameters") == new_config.get("target_parameters"), \
        f"Mismatch: Seed had {seed_config.get('target_parameters')}, Job has {new_config.get('target_parameters')}"

    # 2. Check Safetensors Keys
    seed_keys = set()
    with safe_open(f"{seed_path}/adapter_model.safetensors", framework="pt", device="cpu") as
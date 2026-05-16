## Verdict

The missing `target_parameters` in V480 is the single most likely root cause of the plateau and the bit-regression pattern, and it is sufficiently supported by the adapter_config sha256 mismatch and the explicit "LoRA target_parameters: disabled" log line to justify one minimal, gated relaunch. However, it is not yet *proven* — you have shown that V480 *did not train the MoE expert projections*, but you have not yet shown that *re-enabling them on the same data* recovers the 0.86-plateau seed behavior. Before any paid GPU run, add a preflight that fails closed on adapter_config drift and on zero trainable MoE-expert parameters, then run a 2-step smoke whose only job is to confirm the expert tensors actually receive gradient and the saved adapter_config matches the seed's `target_parameters`. Do not resume broad SFT until that smoke is green.

## Evidence Assessment

- PROVEN: V290 seed adapter_config has `target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` (sha a3d74c5a…).
- PROVEN: V480 saved adapter_config has `target_parameters = null` (sha ca52d6d8…), and logs show `LORA_TARGET_PARAMETERS empty`, `target_parameter_lora_params: {}`.
- PROVEN: V480 eval_loss moved only in the 4th decimal (0.9725 → 0.9739) while weak total stayed at 190–191 — consistent with training only non-expert LoRA modules on an MoE base.
- PLAUSIBLE: The 0.86 plateau lineage depended on LoRA deltas inside MoE expert projections, and V480 lost that capacity entirely. Supported by the config diff and the flat eval_loss, but not yet demonstrated by a controlled re-enable run.
- PLAUSIBLE: equation +1 / bit −1 pattern in V477/V481 is the signature of catastrophic interference when the only trainable subspace is attention+MLP+lm_head (non-expert), which biases toward equation surface form at bit's cost.
- PLAUSIBLE: V476 objective weighting bug (99/1 split) was real and is now corrected; its residual effect on cached datasets is UNKNOWN without a dataset hash log.
- UNKNOWN: Whether the V479 dataset is byte-identical to what V480 actually loaded (no dataset content hash shown in the V480 log excerpt provided).
- UNKNOWN: Whether `accuracy=0.0000` events were a logging artifact or a real eval path failure — no stack trace or row sample shown.
- UNKNOWN: Whether `truncated=1` cases on ckpt 2/6 were equation or bit rows, and whether they involved generation cutoff vs. template mismatch.

## Root Cause Ranking

1. **Lost MoE `target_parameters` in V480 LoRA config (confidence ~75%).** Direct config sha mismatch + empty target_parameter tensor dict + flat eval_loss + flat weak accuracy.
2. **Objective/loss-mask weighting residue from V476 lineage (confidence ~35%).** V477 only partially corrected and produced eq+1/bit−1; V479 fixed dataset checks but the *loss reduction* (per-token vs per-sample, family reweighting) may still favor equation.
3. **Train/eval prompt-template or offset-mask drift between seed-era inference and current training (confidence ~25%).** Would explain eval_loss moving without weak accuracy moving, since loss is computed on supervised spans that may not match the generation contract.
4. **Catastrophic interference from training non-expert modules only (confidence ~20%, partially overlaps #1).** Even with correct data, restricting LoRA to attention/MLP/lm_head on an MoE base tends to trade families.
5. **Dataset content drift or dedupe collapsing one family (confidence ~10%).** V479 added strict checks, so this is lower, but not zero without a logged content hash.

## Required Code/Gate Patches

Preflight (must run, must fail closed, CPU-only):

```python
# preflight_adapter_lineage.py
import json, hashlib, os
from safetensors import safe_open

SEED = os.environ["KG1_SEED_ADAPTER_DIR"]
OUT  = os.environ["KG1_OUTPUT_ADAPTER_DIR"]  # post-train, for output gate
EXPECTED_TPARAMS = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}

def load_cfg(p):
    with open(os.path.join(p, "adapter_config.json")) as f:
        return json.load(f)

def assert_seed_cfg(cfg):
    tp = set(cfg.get("target_parameters") or [])
    assert tp == EXPECTED_TPARAMS, f"seed target_parameters drift: {tp}"
    assert cfg.get("target_modules"), "seed target_modules empty"

def assert_env_matches(cfg):
    env_tp = set(filter(None, os.environ.get("KG1_LORA_TARGET_PARAMETERS","").split(",")))
    assert env_tp == set(cfg["target_parameters"]), \
        f"env target_parameters {env_tp} != seed {cfg['target_parameters']}"
    assert os.environ.get("REQUIRE_LORA_TARGET_PARAMETER_MATCH") == "1"

def assert_trainable_covers_experts(model):
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    has_gu = any("mlp.experts.gate_up_proj" in n and "lora_" in n for n in names)
    has_dn = any("mlp.experts.down_proj"    in n and "lora_" in n for n in names)
    assert has_gu and has_dn, f"no LoRA params on expert projections; trainable={names[:20]}"

def assert_output_cfg_equals_seed():
    s, o = load_cfg(SEED), load_cfg(OUT)
    for k in ("target_modules","target_parameters","r","lora_alpha","lora_dropout","bias","task_type"):
        assert s.get(k) == o.get(k), f"adapter_config drift on {k}: seed={s.get(k)} out={o.get(k)}"

def assert_safetensors_has_expert_keys():
    path = os.path.join(OUT, "adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    assert any("experts.gate_up_proj" in k for k in keys), "no expert gate_up keys in saved adapter"
    assert any("experts.down_proj"    in k for k in keys), "no expert down_proj keys in saved adapter"
```

Loading the seed (preferred order):

1. **Preferred: `PeftModel.from_pretrained(base, SEED, is_trainable=True)`**. This rehydrates `target_modules` *and* `target_parameters` exactly from the seed's `adapter_config.json`. Do not build a fresh `LoraConfig`.
2. Acceptable fallback: load seed `adapter_config.json` verbatim, construct `LoraConfig(**cfg)` (after stripping non-init fields), then `get_peft_model`, then `set_peft_model_state_dict(model, load_file(adapter_model.safetensors))`. Verify with the preflight above.
3. **Forbidden**: hand-authored `LoraConfig` that omits `target_parameters`, or any path where `LORA_TARGET_PAR
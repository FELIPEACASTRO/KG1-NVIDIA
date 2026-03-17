"""
Baseline LoRA Submission — Zero-weight adapter with CORRECT PEFT format
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "safetensors"])

# ============================================================
# CELL 2: Create zero-weight adapter with correct format
# ============================================================

import os
import json
import zipfile
import torch
from safetensors.torch import save_file

OUTPUT_DIR = "/kaggle/working"

# Read model config for architecture
import kagglehub
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
    model_config = json.load(f)

num_layers = model_config.get("num_hidden_layers", 52)
print(f"Model: {num_layers} layers")

# Read weight index to find ONLY non-expert target modules
with open(os.path.join(MODEL_PATH, "model.safetensors.index.json"), "r") as f:
    weight_index = json.load(f)

weight_map = weight_index.get("weight_map", {})

# The regex r".*\.(in_proj|out_proj|up_proj|down_proj)$" matches modules
# that END with these names but are NOT inside "experts" submodules.
# We need to filter: include backbone.layers.X.mixer.{in_proj|out_proj}
# and backbone.layers.X.mlp.{up_proj|down_proj} (shared expert)
# but EXCLUDE backbone.layers.X.mixer.experts.N.{up_proj|down_proj}

target_suffixes = [".in_proj.weight", ".out_proj.weight", ".up_proj.weight", ".down_proj.weight"]

# Get dimensions from safetensors metadata
from safetensors import safe_open

# Find target weights (excluding experts)
target_weights = {}
for weight_name in weight_map:
    # Check if it matches any target suffix
    matches = any(weight_name.endswith(s) for s in target_suffixes)
    if not matches:
        continue
    # EXCLUDE expert weights (they have .experts. in the path)
    if ".experts." in weight_name:
        continue
    target_weights[weight_name] = weight_map[weight_name]

print(f"Target modules (non-expert): {len(target_weights)}")
for name in sorted(target_weights.keys())[:10]:
    print(f"  {name}")

# Get dimensions from safetensors files
dim_map = {}
weight_files = set(target_weights.values())
for wf in weight_files:
    wf_path = os.path.join(MODEL_PATH, wf)
    if not os.path.exists(wf_path):
        continue
    with safe_open(wf_path, framework="pt") as f:
        for key in f.keys():
            if key in target_weights:
                shape = f.get_slice(key).get_shape()
                dim_map[key] = shape

print(f"Got dimensions for {len(dim_map)} weights")

# ============================================================
# CELL 3: Build adapter files
# ============================================================

RANK = 32

# adapter_config.json — matching official demo exactly
adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": MODEL_PATH,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": RANK,
    "revision": None,
    "target_modules": ["in_proj", "out_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}

with open(os.path.join(OUTPUT_DIR, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2)

# Build LoRA tensors with CORRECT PEFT naming
# PEFT format: base_model.model.{layer_path_without_.weight}.lora_A.weight
tensors = {}
for weight_name, shape in dim_map.items():
    out_feat, in_feat = shape[0], shape[1]
    # Convert: "backbone.layers.0.mixer.in_proj.weight" ->
    # "base_model.model.backbone.layers.0.mixer.in_proj"
    base_name = weight_name.replace(".weight", "")
    peft_prefix = f"base_model.model.{base_name}"

    # NOTE: No ".default" — standard PEFT format
    tensors[f"{peft_prefix}.lora_A.weight"] = torch.zeros(RANK, in_feat)
    tensors[f"{peft_prefix}.lora_B.weight"] = torch.zeros(out_feat, RANK)

print(f"Created {len(tensors)} LoRA tensors")
for name in sorted(tensors.keys())[:6]:
    print(f"  {name}: {tensors[name].shape}")

save_file(tensors, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, "adapter_model.safetensors")) / 1024 / 1024
print(f"adapter_model.safetensors: {size_mb:.1f} MB")

# ============================================================
# CELL 4: Package and verify
# ============================================================

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(OUTPUT_DIR, "adapter_config.json"), "adapter_config.json")
    zf.write(os.path.join(OUTPUT_DIR, "adapter_model.safetensors"), "adapter_model.safetensors")

zip_size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\nsubmission.zip: {zip_size:.1f} MB")

# Verify
with zipfile.ZipFile(zip_path, 'r') as zf:
    print(f"Contents: {zf.namelist()}")
    for info in zf.infolist():
        print(f"  {info.filename}: {info.file_size/1024:.1f} KB")

print("\nDone!")

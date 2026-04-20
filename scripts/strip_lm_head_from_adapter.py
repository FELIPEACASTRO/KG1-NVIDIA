#!/usr/bin/env python3
"""Strip lm_head weights from PEFT adapter (for Kaggle ZIP size limit).

When lm_head is in LoRA target_modules, PEFT saves save_embedding_layers=True,
embedding the FULL lm_head weight (~600 MB) alongside LoRA deltas.

Kaggle Nemotron competition rejects submissions > ~500 MB with 400 Bad Request.

This script strips embeddings to produce a pure LoRA adapter (~60 MB).

Usage:
    python scripts/strip_lm_head_from_adapter.py \
        --adapter-in /path/to/checkpoint-final/ \
        --adapter-out /path/to/checkpoint-final-stripped/
"""
import argparse, json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-in", required=True)
    p.add_argument("--adapter-out", required=True)
    args = p.parse_args()

    os.makedirs(args.adapter_out, exist_ok=True)

    # Load + filter tensors
    src_file = os.path.join(args.adapter_in, "adapter_model.safetensors")
    tensors = {}
    with safe_open(src_file, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    stripped = {k: v for k, v in tensors.items()
                if ".lora_A." in k or ".lora_B." in k
                or ".lora_embedding_A" in k or ".lora_embedding_B" in k}

    dropped = [k for k in tensors if k not in stripped]
    print(f"Original: {len(tensors)} tensors")
    print(f"Kept (LoRA only): {len(stripped)}")
    print(f"Dropped (full weights): {len(dropped)}")
    for k in dropped:
        size_mb = tensors[k].numel() * tensors[k].element_size() / 1e6
        print(f"  - {k}: {size_mb:.1f} MB")

    save_file(stripped, os.path.join(args.adapter_out, "adapter_model.safetensors"))

    # Patch config to remove lm_head from targets
    cfg = json.load(open(os.path.join(args.adapter_in, "adapter_config.json")))
    if "lm_head" in cfg.get("target_modules", []):
        cfg["target_modules"] = [t for t in cfg["target_modules"] if t != "lm_head"]
        print(f"Patched target_modules: removed 'lm_head'")

    with open(os.path.join(args.adapter_out, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy tokenizer files
    for fn in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
        src = os.path.join(args.adapter_in, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.adapter_out, fn))

    # Size report
    new_sz = os.path.getsize(os.path.join(args.adapter_out, "adapter_model.safetensors")) / 1e6
    print(f"\nStripped adapter size: {new_sz:.1f} MB (vs ~675 MB original)")


if __name__ == "__main__":
    main()

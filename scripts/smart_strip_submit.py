#!/usr/bin/env python3
r"""
Smart strip for v50: THREE submission modes.

Mode A: "v30-replica" — strip ALL expert keys (including shared_experts)
        Result: only attention + mamba LoRA. Replicates v30 exactly.

Mode B: "smart-strip" — strip ONLY routed experts (experts.\d+), KEEP shared_experts
        Result: attention + mamba + shared_expert up/down_proj LoRA.
        This is NEW and potentially better than v30.

Mode C: "no-strip" — keep EVERYTHING including routed experts
        Result: all ~6000+ LoRA modules. Requires vLLM expert LoRA support.
"""

import json
import os
import re
import zipfile
import argparse
import subprocess
from safetensors.torch import load_file, save_file


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
ALLOW_KAGGLE_SUBMIT = os.environ.get("KG1_ALLOW_KAGGLE_SUBMIT", "0").strip().lower() in {"1", "true", "yes", "on"}


def strip_adapter(adapter_dir, mode="smart-strip"):
    """Strip adapter based on mode."""
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    print(f"=== STRIP MODE: {mode} ===")
    tensors = load_file(safetensors_path)
    print(f"  Total keys: {len(tensors)}")

    routed_re = re.compile(r"\.experts\.\d+\.")

    keep = {}
    removed_routed = 0
    removed_shared = 0
    for key, val in tensors.items():
        key_lower = key.lower()

        if mode == "v30-replica":
            # Strip EVERYTHING with "expert" (same as old strip_and_submit.py)
            if "expert" in key_lower:
                if routed_re.search(key):
                    removed_routed += 1
                else:
                    removed_shared += 1
            else:
                keep[key] = val

        elif mode == "smart-strip":
            # Strip ONLY routed experts (experts.\d+), KEEP shared_experts
            if routed_re.search(key):
                removed_routed += 1
            else:
                keep[key] = val

        elif mode == "no-strip":
            # Keep EVERYTHING
            keep[key] = val

        else:
            raise ValueError(f"Unknown mode: {mode}")

    print(f"  Kept: {len(keep)} keys")
    print(f"  Removed routed: {removed_routed}")
    print(f"  Removed shared: {removed_shared}")

    # Identify kept module types
    kept_modules = set()
    for key in keep:
        for mod in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
                     "up_proj", "down_proj", "gate"]:
            if mod in key:
                kept_modules.add(mod)
    print(f"  Module types: {sorted(kept_modules)}")

    # Size info
    keep_bytes = sum(v.numel() * v.element_size() for v in keep.values())
    print(f"  Kept size: {keep_bytes / 1e6:.1f} MB")

    # Save
    out_dir = os.path.join(os.path.dirname(adapter_dir), f"stripped_{mode}")
    os.makedirs(out_dir, exist_ok=True)

    out_safetensors = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(keep, out_safetensors)

    # Update config
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["target_modules"] = sorted(kept_modules)
    out_config = os.path.join(out_dir, "adapter_config.json")
    with open(out_config, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"  Saved to: {out_dir}")
    print(f"  target_modules: {sorted(kept_modules)}")
    return out_dir


def create_zip(adapter_dir, zip_path):
    """Create submission ZIP with exactly 2 files at root."""
    print(f"\n=== CREATE ZIP ===")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["adapter_config.json", "adapter_model.safetensors"]:
            fpath = os.path.join(adapter_dir, fname)
            zf.write(fpath, fname)  # arcname=fname → root level

    zip_size = os.path.getsize(zip_path) / 1e6
    print(f"  ZIP: {zip_path} ({zip_size:.1f} MB)")

    # Verify
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 2, f"Expected 2 files, got {len(names)}"
        assert all("/" not in n for n in names), "Subdirectory found!"
    print(f"  Verified: 2 files at root, no subdirs")
    return zip_path


def submit_kaggle(zip_path, description):
    """Submit to Kaggle."""
    print(f"\n=== KAGGLE SUBMIT ===")
    print(f"  Desc: {description}")
    result = subprocess.run(
        ["kaggle", "competitions", "submit",
         "-c", COMPETITION,
         "-f", zip_path,
         "-m", description],
        capture_output=True, text=True, timeout=600
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart strip + submit")
    parser.add_argument("--adapter-dir", required=True, help="Path to adapter checkpoint")
    parser.add_argument("--mode", choices=["v30-replica", "smart-strip", "no-strip"],
                        default="smart-strip", help="Strip mode")
    parser.add_argument("--submit", action="store_true", help="Actually submit to Kaggle")
    parser.add_argument("--desc", default="", help="Submission description")
    args = parser.parse_args()

    stripped_dir = strip_adapter(args.adapter_dir, args.mode)
    zip_path = os.path.join(os.path.dirname(args.adapter_dir), f"submission_{args.mode}.zip")
    create_zip(stripped_dir, zip_path)

    if args.submit:
        if not ALLOW_KAGGLE_SUBMIT:
            raise RuntimeError("Kaggle submit is locked. Set KG1_ALLOW_KAGGLE_SUBMIT=1 only for a manually approved submission.")
        submit_kaggle(zip_path, args.desc or f"v50 {args.mode}")
    else:
        print(f"\nReady to submit: {zip_path}")
        print(f"Run with --submit to actually submit to Kaggle")

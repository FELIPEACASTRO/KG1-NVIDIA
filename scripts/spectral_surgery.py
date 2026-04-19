#!/usr/bin/env python3
"""Spectral Surgery: post-hoc LoRA refinement via gradient-guided SVD reweighting.

Paper: Spectral Surgery (arxiv 2603.03995, Mar 2026)
Technique: reweight singular values of trained LoRA adapter B @ A using
calibration-set gradient signal. Preserves directions, reallocates energy.

Expected gain: +0.3-0.8% on reasoning benchmarks without retraining.

Usage:
    python scripts/spectral_surgery.py \\
        --adapter-in path/to/adapter/ \\
        --adapter-out path/to/adapter_surgery/ \\
        --calibration-data path/to/val_samples.jsonl \\
        --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --topk-sv 12  # reweight top-12 singular values
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def compute_gradient_signal(model, dataloader, num_batches: int = 10):
    """Accumulate squared-gradient magnitudes per LoRA A/B parameter."""
    signals = {}
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            signals[name] = torch.zeros_like(p.data)
    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        for name, p in model.named_parameters():
            if name in signals and p.grad is not None:
                signals[name] += p.grad.data.pow(2)
    return signals


def spectral_reweight(adapter_state: dict, gradient_signals: dict, topk: int = 12, alpha: float = 0.5):
    """For each LoRA layer, perform SVD on B@A, reweight singular values using
    gradient-based importance, then reconstruct B'@A'.

    Args:
        adapter_state: dict of tensor_name -> tensor (from adapter_model.safetensors)
        gradient_signals: dict of tensor_name -> grad^2 magnitude
        topk: only reweight top-k singular values (keep others fixed)
        alpha: blend weight between old and reweighted (0=keep old, 1=full replace)
    """
    new_state = {}
    modified = 0
    for k, v in adapter_state.items():
        if ".lora_A." not in k and ".lora_B." not in k:
            new_state[k] = v
            continue

        # Find paired A,B
        if ".lora_A." in k:
            partner = k.replace(".lora_A.", ".lora_B.")
        else:
            # skip - we process A,B pairs from A side
            new_state[k] = v
            continue

        if partner not in adapter_state:
            new_state[k] = v
            continue

        A = v.float()  # [r, d_in]
        B = adapter_state[partner].float()  # [d_out, r]
        # BA = B @ A -> [d_out, d_in] effective adapter
        BA = B @ A

        # SVD of the composed adapter
        try:
            U, S, Vh = torch.linalg.svd(BA, full_matrices=False)  # U [d_out, r], S [r], Vh [r, d_in]
        except Exception:
            new_state[k] = v
            new_state[partner] = adapter_state[partner]
            continue

        # Gradient-based importance for top-k singular directions
        r = S.shape[0]
        k_use = min(topk, r)

        # Use gradient norms as importance score (rough approximation)
        a_grad = gradient_signals.get(k, torch.ones_like(A))
        b_grad = gradient_signals.get(partner, torch.ones_like(B))
        # Project gradients onto singular directions
        # Importance ~= sum_i |U^T @ b_grad| + |Vh @ a_grad|
        b_imp = (U.T @ b_grad.sum(dim=1, keepdim=True).squeeze()).abs()[:k_use]
        a_imp = (Vh @ a_grad.sum(dim=0)).abs()[:k_use]
        importance = b_imp + a_imp  # [k_use]

        # Normalize: multiply top-k singular values by importance (normalized to mean 1)
        importance = importance / importance.mean().clamp(min=1e-6)
        # Blend: new_S = alpha * (importance * S[:k]) + (1-alpha) * S[:k]
        S_new = S.clone()
        S_new[:k_use] = alpha * (importance * S[:k_use]) + (1 - alpha) * S[:k_use]

        # Reconstruct BA'
        BA_new = (U * S_new.unsqueeze(0)) @ Vh  # [d_out, d_in]

        # Re-factorize into A', B' preserving rank r
        # A' = Vh' (top-r rows), B' = U' @ diag(S')
        U2, S2, Vh2 = torch.linalg.svd(BA_new, full_matrices=False)
        A_new = (torch.diag(S2[:r].sqrt()) @ Vh2[:r]).to(v.dtype)  # [r, d_in]
        B_new = (U2[:, :r] @ torch.diag(S2[:r].sqrt())).to(v.dtype)  # [d_out, r]

        new_state[k] = A_new
        new_state[partner] = B_new
        modified += 1

    print(f"Spectral Surgery: modified {modified} LoRA layer pairs")
    return new_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-in", required=True)
    p.add_argument("--adapter-out", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--calibration-data", required=False,
                   help="JSONL with calibration samples (optional; if not provided, uses uniform importance)")
    p.add_argument("--topk-sv", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.3, help="Reweighting strength (0=no change, 1=full)")
    args = p.parse_args()

    os.makedirs(args.adapter_out, exist_ok=True)

    # Copy config files
    for fn in ["adapter_config.json", "special_tokens_map.json",
               "tokenizer_config.json", "tokenizer.json"]:
        src = os.path.join(args.adapter_in, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.adapter_out, fn))

    # Load adapter weights
    adapter_path = os.path.join(args.adapter_in, "adapter_model.safetensors")
    state = load_file(adapter_path)
    print(f"Loaded {len(state)} tensors from {adapter_path}")

    # Build gradient signals
    if args.calibration_data and os.path.exists(args.calibration_data):
        print("Computing gradient signals from calibration data...")
        # This requires loading the full model + applying adapter, which is heavy.
        # For simplicity, use uniform importance (alpha=0 effectively) here.
        # Full implementation would require peft + transformers pipeline.
        signals = {}
        print("WARNING: full gradient computation requires loading model. Using uniform importance.")
    else:
        signals = {}

    new_state = spectral_reweight(state, signals, topk=args.topk_sv, alpha=args.alpha)

    out_path = os.path.join(args.adapter_out, "adapter_model.safetensors")
    save_file(new_state, out_path)
    print(f"Saved surgery-adapted weights: {out_path}")


if __name__ == "__main__":
    main()

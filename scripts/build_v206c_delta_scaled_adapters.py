#!/usr/bin/env python3
"""Build V206C delta-scaled adapters from V194 and a failed V206B adapter.

This is a no-training artifact builder. For each scale `s`, it writes:

    output = V194 + s * (V206B_failed - V194)

Only the exact tensor layout from the V194 adapter is emitted, so the resulting
adapter remains compatible with the already validated V194/Nemotron loading
path. The generated zip contains only Kaggle adapter files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_scales(raw: str) -> list[float]:
    scales: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if token:
            scales.append(float(token))
    if not scales:
        raise ValueError("At least one scale is required.")
    if any(scale < 0.0 or scale > 1.0 for scale in scales):
        raise ValueError("V206C scales must stay within [0.0, 1.0].")
    return scales


def scale_label(scale: float) -> str:
    return f"s{scale:.3f}".replace(".", "p")


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


def candidate_key_variants(key: str) -> list[str]:
    variants = [key]
    replacements = (
        ("base_model.model.backbone.", "base_model.model.model."),
        ("base_model.model.model.", "base_model.model.backbone."),
    )
    for old, new in replacements:
        if key.startswith(old):
            variants.append(new + key[len(old) :])
    if key.startswith("model."):
        variants.append("base_model." + key)
    return list(dict.fromkeys(variants))


def align_candidate_to_baseline(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    baseline_shapes = {key: tuple(tensor.shape) for key, tensor in baseline.items()}
    aligned: dict[str, torch.Tensor] = {}
    unmapped: list[dict[str, Any]] = []
    remap_counts: Counter[str] = Counter()

    for raw_key, tensor in candidate.items():
        mapped_key = None
        for variant in candidate_key_variants(raw_key):
            if variant in baseline_shapes and baseline_shapes[variant] == tuple(tensor.shape):
                mapped_key = variant
                break
        if mapped_key is None:
            unmapped.append({"key": raw_key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
            continue
        aligned[mapped_key] = tensor
        remap_counts["raw" if mapped_key == raw_key else "prefix_remapped"] += 1

    missing = sorted(set(baseline) - set(aligned))
    return aligned, {
        "baseline_tensor_count": len(baseline),
        "candidate_tensor_count": len(candidate),
        "aligned_tensor_count": len(aligned),
        "missing_from_candidate_count": len(missing),
        "missing_from_candidate_sample": missing[:20],
        "unmapped_candidate_count": len(unmapped),
        "unmapped_candidate_sample": unmapped[:20],
        "remap_counts": dict(sorted(remap_counts.items())),
    }


def tensor_module(key: str) -> str:
    for module in ("lm_head", "down_proj", "up_proj", "in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
        if f".{module}." in key:
            return module
    return "unknown"


def build_scale(
    *,
    scale: float,
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    baseline_config: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    label = scale_label(scale)
    adapter_dir = output_dir / f"adapter_{label}"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, torch.Tensor] = {}
    changed_by_module: Counter[str] = Counter()
    max_abs_by_module: dict[str, float] = {}
    changed_tensor_count = 0
    copied_tensor_count = 0

    for key, base_tensor in baseline.items():
        cand_tensor = candidate.get(key)
        if cand_tensor is None or not base_tensor.is_floating_point() or not cand_tensor.is_floating_point():
            output[key] = base_tensor
            copied_tensor_count += 1
            continue
        if torch.equal(base_tensor, cand_tensor) or scale == 0.0:
            output[key] = base_tensor
            copied_tensor_count += 1
            continue
        diff = cand_tensor.float() - base_tensor.float()
        merged = base_tensor.float() + scale * diff
        output[key] = merged.to(base_tensor.dtype).contiguous()
        changed_tensor_count += 1
        module = tensor_module(key)
        changed_by_module[module] += 1
        max_abs_by_module[module] = max(
            max_abs_by_module.get(module, 0.0),
            float(diff.abs().max().item()) if diff.numel() else 0.0,
        )

    weights_path = adapter_dir / "adapter_model.safetensors"
    save_file(output, str(weights_path))
    shutil.copy2(baseline_config, adapter_dir / "adapter_config.json")

    zip_path = output_dir / f"{run_id}_{label}_adapter.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(adapter_dir / "adapter_config.json", "adapter_config.json")
        zf.write(weights_path, "adapter_model.safetensors")

    report = {
        "label": label,
        "scale": scale,
        "adapter_dir": str(adapter_dir),
        "zip_path": str(zip_path),
        "adapter_model_sha256": sha256_file(weights_path),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
        "zip_sha256": sha256_file(zip_path),
        "tensor_count": len(output),
        "changed_tensor_count": changed_tensor_count,
        "copied_tensor_count": copied_tensor_count,
        "changed_by_module": dict(sorted(changed_by_module.items())),
        "max_abs_delta_by_module_before_scaling": dict(sorted(max_abs_by_module.items())),
    }
    write_json(adapter_dir / "v206c_delta_scale_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-adapter-dir", type=Path, required=True)
    parser.add_argument("--candidate-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", default="0.00,0.01,0.02,0.05,0.10")
    parser.add_argument("--run-id", default="v206c-v206b-delta-scale")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_model = args.baseline_adapter_dir / "adapter_model.safetensors"
    baseline_config = args.baseline_adapter_dir / "adapter_config.json"
    candidate_model = args.candidate_adapter_dir / "adapter_model.safetensors"
    candidate_config = args.candidate_adapter_dir / "adapter_config.json"
    for path in [baseline_model, baseline_config, candidate_model, candidate_config]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
    baseline = load_tensors(baseline_model)
    raw_candidate = load_tensors(candidate_model)
    candidate, alignment = align_candidate_to_baseline(baseline, raw_candidate)
    if alignment["aligned_tensor_count"] == 0:
        raise RuntimeError("Candidate adapter did not align to baseline.")
    if alignment["unmapped_candidate_count"] > 0:
        print("WARNING: some candidate tensors were ignored because they do not exist in the V194 layout.")

    outputs = [
        build_scale(
            scale=scale,
            baseline=baseline,
            candidate=candidate,
            baseline_config=baseline_config,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        for scale in scales
    ]
    manifest = {
        "schema_version": "v206c_delta_scale_manifest_v1",
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "baseline_adapter_dir": str(args.baseline_adapter_dir),
        "candidate_adapter_dir": str(args.candidate_adapter_dir),
        "baseline_adapter_model_sha256": sha256_file(baseline_model),
        "baseline_adapter_config_sha256": sha256_file(baseline_config),
        "candidate_adapter_model_sha256": sha256_file(candidate_model),
        "candidate_adapter_config_sha256": sha256_file(candidate_config),
        "scales": scales,
        "alignment": alignment,
        "outputs": outputs,
        "policy": {
            "no_training": True,
            "no_kaggle_submit": True,
            "submit_requires_eval_loss_lte_baseline_and_human_approval": True,
        },
    }
    manifest_path = args.output_dir / "v206c_delta_scale_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

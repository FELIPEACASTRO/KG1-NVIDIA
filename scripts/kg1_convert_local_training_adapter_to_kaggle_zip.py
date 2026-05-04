#!/usr/bin/env python3
"""Convert a local HF-layout Nemotron PEFT adapter into a Kaggle ZIP.

Use this after a Colab/local training run.  It performs no Hub upload and no
Kaggle submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


TRAINING_PREFIX = "base_model.model.backbone."
TRAINING_LM_HEAD_PREFIX = "base_model.model.backbone.lm_head."
KAGGLE_PREFIX = "base_model.model.model."
KAGGLE_LM_HEAD_PREFIX = "base_model.model.lm_head."


CANONICAL_ADAPTER_CONFIG = {
    "peft_type": "LORA",
    "auto_mapping": None,
    "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "r": 32,
    "rank_pattern": {"in_proj": 32},
    "alpha_pattern": {"in_proj": 32},
    "target_modules": [
        "down_proj",
        "in_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "out_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ],
    "task_type": "CAUSAL_LM",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert(source_adapter_dir: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    source_adapter_dir = source_adapter_dir.resolve()
    output_dir = output_dir.resolve()
    weights_path = source_adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {weights_path}")

    out_adapter_dir = output_dir / "adapter"
    zip_dir = output_dir / "zip"
    shutil.rmtree(out_adapter_dir, ignore_errors=True)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    converted: dict[str, Any] = {}
    renamed_count = 0
    already_kaggle_count = 0
    unchanged_count = 0
    unexpected_prefix_sample: list[str] = []
    tensor_count = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            tensor = handle.get_tensor(key)
            if key.startswith(TRAINING_LM_HEAD_PREFIX):
                new_key = KAGGLE_LM_HEAD_PREFIX + key[len(TRAINING_LM_HEAD_PREFIX) :]
                renamed_count += 1
            elif key.startswith(TRAINING_PREFIX):
                new_key = KAGGLE_PREFIX + key[len(TRAINING_PREFIX) :]
                renamed_count += 1
            elif key.startswith(KAGGLE_PREFIX) or key.startswith(KAGGLE_LM_HEAD_PREFIX):
                new_key = key
                already_kaggle_count += 1
            else:
                new_key = key
                unchanged_count += 1
                unexpected_prefix_sample.append(key)
            converted[new_key] = tensor

    out_weights = out_adapter_dir / "adapter_model.safetensors"
    save_file(converted, str(out_weights))
    write_json(out_adapter_dir / "adapter_config.json", CANONICAL_ADAPTER_CONFIG)

    zip_path = zip_dir / f"{run_id}_adapter_only.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_adapter_dir / "adapter_config.json", "adapter_config.json")
        archive.write(out_weights, "adapter_model.safetensors")

    ready = bool(converted) and not unexpected_prefix_sample and (renamed_count > 0 or already_kaggle_count > 0)
    report = {
        "generated_at": utc_now(),
        "run_id": run_id,
        "source_adapter_dir": str(source_adapter_dir),
        "output_adapter_dir": str(out_adapter_dir),
        "zip_path": str(zip_path),
        "input": {
            "adapter_model_bytes": weights_path.stat().st_size,
            "adapter_model_sha256": sha256_file(weights_path),
        },
        "output": {
            "adapter_model_bytes": out_weights.stat().st_size,
            "adapter_model_sha256": sha256_file(out_weights),
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256_file(zip_path),
            "tensor_count": tensor_count,
            "converted_tensor_count": len(converted),
            "renamed_backbone_to_model_count": renamed_count,
            "already_kaggle_count": already_kaggle_count,
            "unchanged_count": unchanged_count,
            "unexpected_unchanged_prefix_sample": unexpected_prefix_sample[:20],
        },
        "decision": {
            "kaggle_layout_ready": ready,
            "note": "Submission still requires local validation and explicit authorization.",
        },
    }
    write_json(output_dir / "local_kaggle_layout_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="v195-local-training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = convert(args.source_adapter_dir, args.output_dir, args.run_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"]["kaggle_layout_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

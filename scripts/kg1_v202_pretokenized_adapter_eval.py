#!/usr/bin/env python3
"""Evaluate a local Nemotron PEFT adapter on Tong pretokenized validation splits.

This script is intentionally eval-only. It imports the same ``hf_job_train_v90``
module used for V202B/V202C so adapter loading, pretokenized masks, split
selection, and masked cross entropy stay aligned with the training gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_SPLITS = [
    {
        "name": "all720",
        "exclude_categories": "",
        "val_examples": 720,
        "eval_max_examples": 720,
    }
]


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


def parse_split_specs(value: str) -> list[dict[str, Any]]:
    if not value.strip():
        return list(DEFAULT_SPLITS)
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--split-specs-json must be a non-empty JSON list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Split spec {index} is not an object")
        name = str(item.get("name") or f"split{index}")
        normalized.append(
            {
                "name": name,
                "exclude_categories": str(item.get("exclude_categories") or ""),
                "val_examples": int(item.get("val_examples", 720)),
                "eval_max_examples": int(item.get("eval_max_examples", item.get("val_examples", 720))),
            }
        )
    return normalized


def set_eval_env(args: argparse.Namespace) -> None:
    env = {
        "MODEL_NAME": args.model_name,
        "MODEL_REVISION": args.model_revision,
        "MODEL_DEVICE_MAP": args.model_device_map,
        "PRETOKENIZED_ARCHIVE_ZIP": str(args.archive_zip),
        "EXPECTED_ARCHIVE_SHA256": args.expected_archive_sha256,
        "PRETOKENIZED_VAL_COPY_ONLY": "0",
        "INIT_ADAPTER_DIR": str(args.adapter_dir),
        "INIT_ADAPTER_LOAD_MODE": "manual",
        "PEFT_MANUAL_LOAD_METHOD": "direct",
        "ADAPTER_LOAD_LOW_CPU_MEM_USAGE": "0",
        "UPLOAD_TO_HF": "0",
        "UPLOAD_CHECKPOINTS_DURING_TRAINING": "0",
        "FAIL_ON_MISSING_ADAPTER_KEYS": "1",
        "REQUIRE_OFFSET_MASK": "1",
        "LORA_R": "32",
        "LORA_ALPHA": "32",
        "LORA_DROPOUT": "0.0",
        "LORA_TARGET_MODULES": "down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj",
        "TRAINABLE_LORA_MODULES": "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj",
        "MAX_TRAINABLE_PARAM_RATIO": "0.035",
        "MAX_LENGTH": str(args.max_length),
        "BATCH_SIZE": "1",
        "MICRO_BATCH_SIZE": "1",
        "NUM_EPOCHS": "1",
        "MAX_STEPS": "1",
        "SEED": str(args.seed),
        "DRY_RUN_VALIDATE_ONLY": "0",
        "BASELINE_EVAL_BEFORE_TRAIN": "0",
    }
    for key, value in env.items():
        os.environ[key] = value


def import_training_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("kg1_v202_eval_hf_job_train_v90", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def count_categories(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("category", "unknown"))] += 1
    return dict(sorted(counts.items()))


def eval_split(
    hf: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    archive_zip: Path,
    split: dict[str, Any],
) -> dict[str, Any]:
    hf.PRETOKENIZED_EXCLUDE_CATEGORIES = split["exclude_categories"]
    hf.PRETOKENIZED_VAL_EXAMPLES = int(split["val_examples"])
    hf.PRETOKENIZED_VAL_FRACTION = 0.0
    hf.PRETOKENIZED_VAL_COPY_ONLY = False

    rows = hf.load_tong_pretokenized_archive(archive_zip)
    train_data, val_data = hf.split_pretokenized_train_val(rows)
    sample = hf.select_eval_sample(val_data, min(int(split["eval_max_examples"]), len(val_data)))
    overall_loss = hf.evaluate_loss(model, sample, tokenizer, len(sample))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sample:
        by_category[str(item.get("category", "unknown"))].append(item)

    per_category = {}
    for category, items in sorted(by_category.items()):
        per_category[category] = {
            "loss": hf.evaluate_loss(model, items, tokenizer, len(items)),
            "examples": len(items),
            "tokens": int(sum(len(item["input_ids"]) for item in items)),
            "unmasked_tokens": int(sum(sum(item["loss_mask"]) for item in items)),
        }

    result = {
        "name": split["name"],
        "exclude_categories": split["exclude_categories"],
        "val_examples_requested": int(split["val_examples"]),
        "eval_max_examples_requested": int(split["eval_max_examples"]),
        "loaded_rows": len(rows),
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "sample_examples": len(sample),
        "sample_category_counts": count_categories(sample),
        "overall_loss": overall_loss,
        "per_category": per_category,
    }
    del rows, train_data, val_data, sample, by_category
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split-specs-json", default="")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--model-revision", default="cbd3fa9f933d55ef16a84236559f4ee2a0526848")
    parser.add_argument("--model-device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=202)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.training_script.exists():
        raise FileNotFoundError(args.training_script)
    if not args.adapter_dir.exists():
        raise FileNotFoundError(args.adapter_dir)
    if not args.archive_zip.exists():
        raise FileNotFoundError(args.archive_zip)
    if args.expected_archive_sha256:
        observed = sha256_file(args.archive_zip)
        if observed != args.expected_archive_sha256:
            raise RuntimeError(
                f"archive sha256 mismatch: observed={observed} expected={args.expected_archive_sha256}"
            )

    split_specs = parse_split_specs(args.split_specs_json)
    set_eval_env(args)
    hf = import_training_module(args.training_script)

    print(f"Loading tokenizer for {hf.MODEL_NAME}...")
    tokenizer = hf.AutoTokenizer.from_pretrained(
        hf.MODEL_NAME,
        revision=hf.MODEL_REVISION or None,
        trust_remote_code=True,
        token=hf.HF_TOKEN or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_device_map = hf.parse_model_device_map(hf.MODEL_DEVICE_MAP)
    print(f"Loading model {hf.MODEL_NAME} in BF16 with device_map={model_device_map}...")
    model = hf.AutoModelForCausalLM.from_pretrained(
        hf.MODEL_NAME,
        revision=hf.MODEL_REVISION or None,
        dtype=torch.bfloat16,
        device_map=model_device_map,
        trust_remote_code=True,
        token=hf.HF_TOKEN or None,
        attn_implementation="eager",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    post_load_device = hf.model_post_load_device(hf.MODEL_DEVICE_MAP)
    if post_load_device:
        model.to(post_load_device)

    print(f"Loading adapter for eval: {args.adapter_dir}")
    model = hf.load_trainable_adapter_or_create(model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()

    split_results = []
    for split in split_specs:
        print("\n" + "=" * 72)
        print("Evaluating split:", split["name"])
        split_results.append(eval_split(hf, model, tokenizer, args.archive_zip, split))

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "label": args.label,
        "adapter_dir": str(args.adapter_dir),
        "adapter_model_sha256": sha256_file(args.adapter_dir / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(args.adapter_dir / "adapter_config.json"),
        "archive_zip": str(args.archive_zip),
        "archive_zip_sha256": sha256_file(args.archive_zip),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "max_length": args.max_length,
        "seed": args.seed,
        "splits": split_results,
        "cuda_peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
    }
    write_json(args.output_json, report)
    print("Eval report:", args.output_json)
    print(json.dumps({
        "label": report["label"],
        "splits": [
            {
                "name": item["name"],
                "overall_loss": item["overall_loss"],
                "sample_examples": item["sample_examples"],
            }
            for item in split_results
        ],
        "cuda_peak_reserved_gib": report["cuda_peak_reserved_gib"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

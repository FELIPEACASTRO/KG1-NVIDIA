#!/usr/bin/env python3
"""Evaluate one or more local LoRA adapters on the V198 strict loss proxy.

This is an H100/A100 Colab utility for V206C. It performs no training and no
submission. It reuses the validated tokenization, adapter loading, and loss
functions from `hf_job_train_v90.py`, then swaps adapter weights in one loaded
Nemotron process.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft.utils.save_and_load import load_peft_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hf_job_train_v90 as trainlib  # noqa: E402


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_adapter_specs(raw: str) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for item in raw.split(";"):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            label, path = token.split("=", 1)
        else:
            path = token
            label = Path(path).name
        specs.append((label.strip(), Path(path.strip())))
    if not specs:
        raise ValueError("ADAPTER_EVAL_DIRS must contain at least one adapter path.")
    return specs


def main() -> int:
    print("=" * 72)
    print("KG1 V206C adapter loss-only evaluation")
    print("=" * 72)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    adapter_specs = parse_adapter_specs(env_str("ADAPTER_EVAL_DIRS"))
    output_json = Path(env_str("EVAL_OUTPUT_JSON", "/tmp/kg1_v206c_eval_loss_report.json"))
    eval_max_examples = env_int("EVAL_MAX_EXAMPLES", trainlib.EVAL_MAX_EXAMPLES)
    baseline_label = env_str("BASELINE_LABEL", adapter_specs[0][0])

    for label, adapter_dir in adapter_specs:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = adapter_dir / name
            if not path.exists():
                raise FileNotFoundError(f"{label}: missing {path}")

    trainlib.setup_causal_conv1d_stub()
    print(f"Model: {trainlib.MODEL_NAME}")
    print(f"Revision: {trainlib.MODEL_REVISION or 'default'}")
    print(f"Validation: {trainlib.DATA_REPO}/{trainlib.VAL_FILE}")
    print(f"Eval max examples: {eval_max_examples}")
    print("Adapters:")
    for label, adapter_dir in adapter_specs:
        print(f"  {label}: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        trainlib.MODEL_NAME,
        revision=trainlib.MODEL_REVISION or None,
        trust_remote_code=True,
        token=trainlib.HF_TOKEN or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_path = trainlib.resolve_data_file(trainlib.VAL_FILE)
    trainlib.assert_file_sha256(val_path, trainlib.EXPECTED_VAL_SHA256, "validation dataset")
    val_examples = trainlib.load_jsonl(val_path)
    if len(val_examples) < trainlib.MIN_VAL_EXAMPLES:
        raise RuntimeError(f"Validation dataset too small: {len(val_examples)} < {trainlib.MIN_VAL_EXAMPLES}")
    val_data = trainlib.tokenize_examples(val_examples, tokenizer, "Validation")
    if len(val_data) < trainlib.MIN_TOKENIZED_VAL_EXAMPLES:
        raise RuntimeError(
            f"Too few tokenized validation examples: {len(val_data)} < {trainlib.MIN_TOKENIZED_VAL_EXAMPLES}"
        )

    model_device_map = trainlib.parse_model_device_map(trainlib.MODEL_DEVICE_MAP)
    print(f"\nLoading model in BF16 with device_map={model_device_map}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        trainlib.MODEL_NAME,
        revision=trainlib.MODEL_REVISION or None,
        dtype=torch.bfloat16,
        device_map=model_device_map,
        trust_remote_code=True,
        token=trainlib.HF_TOKEN or None,
        attn_implementation="eager",
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    post_load_device = trainlib.model_post_load_device(trainlib.MODEL_DEVICE_MAP)
    if post_load_device:
        base_model.to(post_load_device)

    model = trainlib.create_lora_model(base_model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()

    results: list[dict[str, Any]] = []
    for label, adapter_dir in adapter_specs:
        print("\n" + "-" * 72)
        print(f"Evaluating adapter: {label}")
        print("-" * 72)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        weights = load_peft_weights(str(adapter_dir), device="cpu")
        print(f"Loaded adapter tensors: {len(weights)}")
        trainlib.load_peft_weights_with_direct_fallback(model, weights, adapter_name="default")
        start = time.time()
        eval_loss = trainlib.evaluate_loss(model, val_data, tokenizer, eval_max_examples)
        elapsed = time.time() - start
        result = {
            "label": label,
            "adapter_dir": str(adapter_dir),
            "eval_loss": eval_loss,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
            "cuda_memory": trainlib.cuda_memory_line(),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        results.append(result)
        write_json(output_json, {
            "generated_at_utc": utc_now(),
            "status": "running",
            "baseline_label": baseline_label,
            "results": results,
        })

    baseline = next((item for item in results if item["label"] == baseline_label), results[0])
    baseline_loss = float(baseline["eval_loss"])
    for item in results:
        item["delta_vs_baseline"] = float(item["eval_loss"]) - baseline_loss
        item["passes_loss_gate"] = float(item["eval_loss"]) <= baseline_loss

    non_baseline = [item for item in results if item["label"] != baseline_label]
    best = min(non_baseline or results, key=lambda item: float(item["eval_loss"]))
    approved = bool(best["label"] != baseline_label and best["passes_loss_gate"])
    report = {
        "generated_at_utc": utc_now(),
        "status": "approve_loss_prefilter" if approved else "reject_loss_prefilter",
        "approved_loss_prefilter": approved,
        "baseline_label": baseline_label,
        "baseline_eval_loss": baseline_loss,
        "best_non_baseline": best,
        "results": results,
        "policy": {
            "no_training": True,
            "no_kaggle_submit": True,
            "loss_prefilter_only": True,
            "submission_requires_preflight_solve_rate_and_human_approval": True,
        },
    }
    write_json(output_json, report)
    print("\nFINAL_V206C_EVAL_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FINAL_V206C_EVAL_REPORT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

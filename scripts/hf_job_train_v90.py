#!/usr/bin/env python3
"""
HF Jobs training script V90 - category-solver SFT over the gold-safe corpus.

This script is intended for a single remote A100/H100-class GPU job. The local
workstation used to build the v90 dataset does not have enough GPU/disk headroom
to train NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 directly.

Defaults:
  - Dataset: data/v90/v90_train_gold_safe.jsonl
  - Validation: data/v90/v90_val_gold_safe_stratified.jsonl
  - LoRA: r=32 alpha=32 dropout=0.0, guarded target modules
  - Length: 8192 tokens by default to match the Huikang-style recipe
  - LR: 2e-4 -> 1e-4 linear decay
  - Steps: 300 by default, roughly one epoch at batch size 32

Required env on the remote job:
  HF_TOKEN=<token with read access to model/data and write access to output repo>

Optional env:
  MODEL_NAME, MODEL_REVISION, DATA_REPO, DATA_FILE, VAL_FILE, OUTPUT_REPO, OUTPUT_DIR,
  MAX_LENGTH, BATCH_SIZE, MICRO_BATCH_SIZE, LEARNING_RATE, FINAL_LEARNING_RATE,
  NUM_EPOCHS, MAX_STEPS, SAVE_EVERY_STEPS, EVAL_EVERY_STEPS, EVAL_MAX_EXAMPLES,
  LOG_EVERY_STEPS, MICRO_LOG_EVERY, SEED, EXPECTED_TRAIN_SHA256,
  EXPECTED_VAL_SHA256, MIN_TRAIN_EXAMPLES, MIN_VAL_EXAMPLES,
  MIN_TOKENIZED_TRAIN_EXAMPLES, MIN_TOKENIZED_VAL_EXAMPLES, REQUIRE_OFFSET_MASK,
  LORA_TARGET_MODULES, LORA_TARGET_PARAMETERS, MAX_TRAINABLE_PARAM_RATIO, DRY_RUN_VALIDATE_ONLY,
  TOKENIZE_ONLY_DRY_RUN, UPLOAD_TO_HF,
  UPLOAD_CHECKPOINTS_DURING_TRAINING, SAMPLING_MODE, SUBCATEGORY_WEIGHTS, SOURCE_WEIGHTS,
  TRAINABLE_LORA_MODULES, TRAINABLE_LORA_NAME_SUBSTRINGS,
  REQUIRE_LORA_TARGET_PARAMETER_MATCH, REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS,
  ANSWER_SPAN_LOSS_WEIGHT, ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
  USE_ROW_LOSS_WEIGHT, REQUIRE_ROW_LOSS_WEIGHT
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import importlib.metadata as importlib_metadata
import json
import math
import os
import random
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, get_token, hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def disable_peft_torchao_dispatch_if_incompatible() -> None:
    """Avoid PEFT aborting on stale Colab/Kaggle torchao installs.

    This training path loads the Nemotron base in BF16 and injects ordinary
    LoRA modules. It does not use torchao quantization. Some hosted runtimes
    preinstall old torchao builds; recent PEFT versions probe torchao during
    adapter injection and raise before falling through to the normal dispatch.
    """

    try:
        torchao_version = importlib_metadata.version("torchao")
    except importlib_metadata.PackageNotFoundError:
        return
    except Exception as exc:
        print(f"torchao version probe failed; leaving PEFT dispatch unchanged: {exc}")
        return

    def version_tuple(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for chunk in value.replace("-", ".").split("."):
            if not chunk.isdigit():
                break
            parts.append(int(chunk))
        return tuple(parts or [0])

    if version_tuple(torchao_version) >= (0, 16, 0):
        print(f"torchao present and compatible enough for PEFT probe: {torchao_version}")
        return

    print(
        "Disabling PEFT torchao dispatcher: "
        f"found torchao=={torchao_version}, but this BF16 LoRA run does not need torchao."
    )

    def _false() -> bool:
        return False

    try:
        import peft.import_utils as peft_import_utils

        if hasattr(peft_import_utils.is_torchao_available, "cache_clear"):
            peft_import_utils.is_torchao_available.cache_clear()
        peft_import_utils.is_torchao_available = _false
    except Exception as exc:
        print(f"Warning: could not patch peft.import_utils torchao probe: {exc}")

    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = _false
    except Exception as exc:
        print(f"Warning: could not patch peft lora torchao dispatcher: {exc}")


disable_peft_torchao_dispatch_if_incompatible()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


MODEL_NAME = env_str("MODEL_NAME", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
MODEL_REVISION = env_str("MODEL_REVISION", "cbd3fa9f933d55ef16a84236559f4ee2a0526848")
MODEL_DEVICE_MAP = env_str("MODEL_DEVICE_MAP", "auto")
ATTN_IMPLEMENTATION = env_str("ATTN_IMPLEMENTATION", "eager")
TORCH_ALLOW_TF32 = env_bool("TORCH_ALLOW_TF32", True)
TORCH_FLOAT32_MATMUL_PRECISION = env_str("TORCH_FLOAT32_MATMUL_PRECISION", "high")
TORCH_DISABLE_CUDNN_SDP = env_bool("TORCH_DISABLE_CUDNN_SDP", False)
TORCH_FORCE_MATH_SDP = env_bool("TORCH_FORCE_MATH_SDP", False)
GRADIENT_CHECKPOINTING = env_bool("GRADIENT_CHECKPOINTING", True)
DATA_REPO = env_str("DATA_REPO", "felipesp1983/kg1-nemotron-training")
DATA_FILE = env_str("DATA_FILE", "data/v90/v90_train_gold_safe.jsonl")
VAL_FILE = env_str("VAL_FILE", "data/v90/v90_val_gold_safe_stratified.jsonl")
PRETOKENIZED_ARCHIVE_ZIP = env_str("PRETOKENIZED_ARCHIVE_ZIP", "")
PRETOKENIZED_EXCLUDE_CATEGORIES = env_str("PRETOKENIZED_EXCLUDE_CATEGORIES", "")
PRETOKENIZED_VAL_EXAMPLES = env_int("PRETOKENIZED_VAL_EXAMPLES", 0)
PRETOKENIZED_VAL_FRACTION = env_float("PRETOKENIZED_VAL_FRACTION", 0.0)
PRETOKENIZED_VAL_COPY_ONLY = env_bool("PRETOKENIZED_VAL_COPY_ONLY", False)
EXPECTED_ARCHIVE_SHA256 = env_str("EXPECTED_ARCHIVE_SHA256", "")
EXPECTED_TRAIN_SHA256 = env_str(
    "EXPECTED_TRAIN_SHA256",
    "ad1c4a1886e92d82d03c0d75c0615ba8c4b96d29e2b07948ff72d541f03c15e4",
)
EXPECTED_VAL_SHA256 = env_str(
    "EXPECTED_VAL_SHA256",
    "749ad2babbfb96c6191b514572e0b1f4aa976681f9a084ee774b3c8f4a44cc04",
)
MIN_TRAIN_EXAMPLES = env_int("MIN_TRAIN_EXAMPLES", 8777)
MIN_VAL_EXAMPLES = env_int("MIN_VAL_EXAMPLES", 720)
MIN_TOKENIZED_TRAIN_EXAMPLES = env_int("MIN_TOKENIZED_TRAIN_EXAMPLES", MIN_TRAIN_EXAMPLES)
MIN_TOKENIZED_VAL_EXAMPLES = env_int("MIN_TOKENIZED_VAL_EXAMPLES", MIN_VAL_EXAMPLES)

MAX_COMPETITION_LORA_R = 32
LORA_R = env_int("LORA_R", 32)
if LORA_R > MAX_COMPETITION_LORA_R:
    raise ValueError(
        f"LORA_R={LORA_R} exceeds competition serving limit "
        f"of {MAX_COMPETITION_LORA_R}."
    )
LORA_ALPHA = env_int("LORA_ALPHA", 32)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.0)
DEFAULT_LORA_TARGET_MODULES = (
    "down_proj,in_proj,k_proj,lm_head,o_proj,out_proj,q_proj,up_proj,v_proj"
)
LORA_TARGET_MODULES = env_str("LORA_TARGET_MODULES", DEFAULT_LORA_TARGET_MODULES)
LORA_TARGET_PARAMETERS = env_str("LORA_TARGET_PARAMETERS", "")
MAX_TRAINABLE_PARAM_RATIO = env_float("MAX_TRAINABLE_PARAM_RATIO", 0.08)

MAX_LENGTH = env_int("MAX_LENGTH", 8192)
BATCH_SIZE = env_int("BATCH_SIZE", 32)
MICRO_BATCH_SIZE = env_int("MICRO_BATCH_SIZE", 1)
if BATCH_SIZE < MICRO_BATCH_SIZE:
    raise ValueError("BATCH_SIZE must be >= MICRO_BATCH_SIZE")
if BATCH_SIZE % MICRO_BATCH_SIZE != 0:
    raise ValueError("BATCH_SIZE must be divisible by MICRO_BATCH_SIZE")
GRADIENT_ACCUMULATION = BATCH_SIZE // MICRO_BATCH_SIZE

LEARNING_RATE = env_float("LEARNING_RATE", 2e-4)
FINAL_LEARNING_RATE = env_float("FINAL_LEARNING_RATE", 1e-4)
ADAM_BETA1 = env_float("ADAM_BETA1", 0.9)
ADAM_BETA2 = env_float("ADAM_BETA2", 0.95)
ADAM_EPS = env_float("ADAM_EPS", 1e-8)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.0)
GRAD_CLIP_NORM = env_float("GRAD_CLIP_NORM", 1e9)

NUM_EPOCHS = env_int("NUM_EPOCHS", 1)
MAX_STEPS = env_int("MAX_STEPS", 300)
SAVE_EVERY_STEPS = env_int("SAVE_EVERY_STEPS", 50)
EVAL_EVERY_STEPS = env_int("EVAL_EVERY_STEPS", 50)
EVAL_MAX_EXAMPLES = env_int("EVAL_MAX_EXAMPLES", 720)
LOG_EVERY_STEPS = env_int("LOG_EVERY_STEPS", 5)
MICRO_LOG_EVERY = env_int("MICRO_LOG_EVERY", 0)
SEED = env_int("SEED", 90)
MAX_PROMPT_TRUNCATION_RATE = env_float("MAX_PROMPT_TRUNCATION_RATE", 0.10)
SAMPLING_MODE = env_str("SAMPLING_MODE", "shuffle")
VALID_SAMPLING_MODES = {"shuffle", "weighted_replacement"}
if SAMPLING_MODE not in VALID_SAMPLING_MODES:
    raise ValueError(
        "SAMPLING_MODE must be one of "
        f"{sorted(VALID_SAMPLING_MODES)}, got {SAMPLING_MODE!r}."
    )
SUBCATEGORY_WEIGHTS = env_str("SUBCATEGORY_WEIGHTS", "")
SOURCE_WEIGHTS = env_str("SOURCE_WEIGHTS", "")
ANSWER_SPAN_LOSS_WEIGHT = env_float("ANSWER_SPAN_LOSS_WEIGHT", 1.0)
ANSWER_SPAN_MIN_WEIGHTED_TOKENS = env_int("ANSWER_SPAN_MIN_WEIGHTED_TOKENS", 0)
USE_ROW_LOSS_WEIGHT = env_bool("USE_ROW_LOSS_WEIGHT", False)
REQUIRE_ROW_LOSS_WEIGHT = env_bool("REQUIRE_ROW_LOSS_WEIGHT", False)
LOSS_NORMALIZATION_MODE = env_str("LOSS_NORMALIZATION_MODE", "token_mean")
VALID_LOSS_NORMALIZATION_MODES = {"token_mean", "example_mean"}
if LOSS_NORMALIZATION_MODE not in VALID_LOSS_NORMALIZATION_MODES:
    raise ValueError(
        "LOSS_NORMALIZATION_MODE must be one of "
        f"{sorted(VALID_LOSS_NORMALIZATION_MODES)}, got {LOSS_NORMALIZATION_MODE!r}."
    )
ABORT_EVAL_LOSS_GT = env_float("ABORT_EVAL_LOSS_GT", 0.0)
BASELINE_EVAL_BEFORE_TRAIN = env_bool("BASELINE_EVAL_BEFORE_TRAIN", False)
ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA = env_float(
    "ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA", -1.0
)
REQUIRE_FINAL_EVAL_LTE_BASELINE = env_bool("REQUIRE_FINAL_EVAL_LTE_BASELINE", False)
MAX_FINAL_EVAL_REGRESSION = env_float("MAX_FINAL_EVAL_REGRESSION", 0.0)
ABORT_TRAIN_RISE_POINTS = env_int("ABORT_TRAIN_RISE_POINTS", 0)
ABORT_MAX_RESERVED_GIB = env_float("ABORT_MAX_RESERVED_GIB", 0.0)
COMPUTE_PROVIDER = env_str("COMPUTE_PROVIDER", "hf_jobs")

OUTPUT_DIR = env_str("OUTPUT_DIR", "/tmp/kg1_v90_output")
OUTPUT_REPO = env_str("OUTPUT_REPO", "felipesp1983/kg1-nemotron-lora-v90-category-solver")
RUN_ID = env_str("RUN_ID", f"v90-r{LORA_R}-a{LORA_ALPHA}-mlen{MAX_LENGTH}-s{MAX_STEPS}")
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token() or ""
INIT_ADAPTER_DIR = env_str("INIT_ADAPTER_DIR", "")
INIT_ADAPTER_REPO = env_str("INIT_ADAPTER_REPO", "")
INIT_ADAPTER_REVISION = env_str("INIT_ADAPTER_REVISION", "")
INIT_ADAPTER_SUBFOLDER = env_str("INIT_ADAPTER_SUBFOLDER", "")
INIT_ADAPTER_LOAD_MODE = env_str("INIT_ADAPTER_LOAD_MODE", "peft")
PEFT_MANUAL_LOAD_METHOD = env_str("PEFT_MANUAL_LOAD_METHOD", "auto")
REQUIRE_OFFSET_MASK = env_bool("REQUIRE_OFFSET_MASK", True)
DRY_RUN_VALIDATE_ONLY = env_bool("DRY_RUN_VALIDATE_ONLY", False)
TOKENIZE_ONLY_DRY_RUN = env_bool("TOKENIZE_ONLY_DRY_RUN", False)
UPLOAD_TO_HF = env_bool("UPLOAD_TO_HF", True)
UPLOAD_CHECKPOINTS_DURING_TRAINING = env_bool("UPLOAD_CHECKPOINTS_DURING_TRAINING", False)
FAIL_ON_MISSING_ADAPTER_KEYS = env_bool("FAIL_ON_MISSING_ADAPTER_KEYS", True)
TRAINABLE_LORA_MODULES = env_str("TRAINABLE_LORA_MODULES", "")
TRAINABLE_LORA_NAME_SUBSTRINGS = env_str("TRAINABLE_LORA_NAME_SUBSTRINGS", "")
REQUIRE_LORA_TARGET_PARAMETER_MATCH = env_bool("REQUIRE_LORA_TARGET_PARAMETER_MATCH", False)
REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE = env_bool("REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE", False)
REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS = env_str("REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS", "")
ADAPTER_LOAD_TORCH_DEVICE = env_str("ADAPTER_LOAD_TORCH_DEVICE", "")
ADAPTER_LOAD_LOW_CPU_MEM_USAGE = env_bool("ADAPTER_LOAD_LOW_CPU_MEM_USAGE", False)
USE_BITSANDBYTES = env_bool("USE_BITSANDBYTES", True)


def optional_torch_device(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "auto"}:
        return None
    return value


def parse_model_device_map(value: str) -> str | dict[str, int] | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "cpu_then_cuda", "cpu-then-cuda"}:
        return None
    if normalized in {"cuda", "cuda:0", "gpu", "single-gpu"}:
        return {"": 0}
    return value


def model_post_load_device(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"cpu_then_cuda", "cpu-then-cuda"}:
        return "cuda"
    return None


def parse_target_modules(value: str) -> str | list[str]:
    """Parse PEFT target_modules from env.

    The default is an explicit serving-compatible module list. ``all-linear`` is
    still accepted for deliberate diagnostics, but the trainable-parameter guard
    below keeps it from silently expanding into a much larger recipe than
    intended.
    """

    normalized = value.strip()
    if normalized == "all-linear":
        return normalized
    modules = [item.strip() for item in normalized.split(",") if item.strip()]
    if not modules:
        raise ValueError("LORA_TARGET_MODULES must be 'all-linear' or a comma-separated module list")
    return modules


def parse_target_parameters(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def target_parameter_name_matches(target_parameter: str, tensor_name: str) -> bool:
    """Match configured PEFT target_parameters against live LoRA tensor names.

    Nemotron/PEFT stores ``mlp.experts.gate_up_proj`` target-parameter adapters
    under live ``mixer.experts.<id>.up_proj`` LoRA names.  The round-trip gate
    already accepts that structural alias; the train-time filter must use the
    same rule or it will fail after a valid PEFT load.
    """

    if target_parameter in tensor_name:
        return True
    lowered_target = target_parameter.lower()
    lowered_name = tensor_name.lower()
    if lowered_target.endswith("experts.gate_up_proj"):
        return "experts." in lowered_name and ".up_proj." in lowered_name
    if lowered_target.endswith("experts.down_proj"):
        return "experts." in lowered_name and ".down_proj." in lowered_name
    return False


def parse_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def create_lora_model(model: torch.nn.Module) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": parse_target_modules(LORA_TARGET_MODULES),
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    target_parameters = parse_target_parameters(LORA_TARGET_PARAMETERS)
    if target_parameters:
        lora_config_params = inspect.signature(LoraConfig.__init__).parameters
        if "target_parameters" not in lora_config_params:
            raise RuntimeError(
                "Current PEFT LoraConfig does not support target_parameters, "
                f"but LORA_TARGET_PARAMETERS={target_parameters!r} is required "
                "to recreate the initial adapter namespace."
            )
        kwargs["target_parameters"] = target_parameters
    lora_config = LoraConfig(**kwargs)
    return get_peft_model(model, lora_config)


def remap_peft_state_dict_for_direct_load(
    weights: dict[str, torch.Tensor],
    model_state_keys: set[str],
    adapter_name: str = "default",
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map PEFT adapter keys to the live PeftModel state dict namespace.

    PEFT adapter files commonly store keys as ``lora_A.weight`` while a live
    multi-adapter PeftModel exposes ``lora_A.default.weight``.  Recent PEFT /
    Transformers combinations can fail in ``set_peft_model_state_dict`` before
    applying this adapter-name conversion, so this explicit remap gives us a
    stable direct-load path.
    """

    remapped: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    replacements = (
        (".lora_A.weight", f".lora_A.{adapter_name}.weight"),
        (".lora_B.weight", f".lora_B.{adapter_name}.weight"),
        (".lora_embedding_A", f".lora_embedding_A.{adapter_name}"),
        (".lora_embedding_B", f".lora_embedding_B.{adapter_name}"),
    )
    for key, tensor in weights.items():
        candidates = [key]
        for old, new in replacements:
            if old in key:
                candidates.append(key.replace(old, new))
        # Some saved adapters include a redundant PEFT prefix depending on
        # save/load version. Keep this as a fallback rather than assuming it.
        if key.startswith("model."):
            candidates.append("base_model." + key)
        mapped_key = next((candidate for candidate in candidates if candidate in model_state_keys), None)
        if mapped_key:
            remapped[mapped_key] = tensor
        else:
            unmapped.append(key)
    return remapped, unmapped


def load_peft_weights_with_direct_fallback(
    loaded_model: torch.nn.Module,
    weights: dict[str, torch.Tensor],
    *,
    adapter_name: str = "default",
) -> None:
    """Load adapter weights robustly across PEFT/Transformers versions."""

    method = PEFT_MANUAL_LOAD_METHOD.strip().lower()
    if method not in {"auto", "set_peft", "direct"}:
        raise ValueError("PEFT_MANUAL_LOAD_METHOD must be one of: auto,set_peft,direct")

    if method in {"auto", "set_peft"}:
        try:
            set_peft_model_state_dict(
                loaded_model,
                weights,
                adapter_name=adapter_name,
                low_cpu_mem_usage=False,
            )
            print("Manual adapter load used set_peft_model_state_dict successfully.")
            return
        except TypeError as exc:
            if method == "set_peft":
                raise
            print(
                "set_peft_model_state_dict failed; falling back to direct PEFT state_dict load: "
                f"{type(exc).__name__}: {exc}"
            )

    model_state = loaded_model.state_dict()
    remapped, unmapped = remap_peft_state_dict_for_direct_load(weights, set(model_state), adapter_name=adapter_name)
    if not remapped:
        raise RuntimeError("Direct PEFT adapter load mapped zero tensors.")
    coverage = len(remapped) / max(1, len(weights))
    print(
        "Direct PEFT adapter load mapping: "
        f"mapped={len(remapped)} unmapped={len(unmapped)} total={len(weights)} coverage={coverage:.4f}"
    )
    if coverage < 0.98:
        raise RuntimeError(
            "Direct PEFT adapter load coverage too low: "
            f"mapped={len(remapped)} total={len(weights)} sample_unmapped={unmapped[:20]}"
        )
    incompatible = loaded_model.load_state_dict(remapped, strict=False)
    missing_lora = [key for key in incompatible.missing_keys if ".lora_" in key]
    unexpected_lora = [key for key in incompatible.unexpected_keys if ".lora_" in key]
    print(
        "Direct PEFT adapter load result: "
        f"missing_lora={len(missing_lora)} unexpected_lora={len(unexpected_lora)}"
    )
    if missing_lora or unexpected_lora:
        raise RuntimeError(
            "Direct PEFT adapter load left LoRA key mismatches: "
            f"missing_sample={missing_lora[:20]} unexpected_sample={unexpected_lora[:20]}"
        )


def load_trainable_adapter_or_create(model: torch.nn.Module) -> torch.nn.Module:
    """Load an existing LoRA adapter for incremental SFT, or create a new one."""

    if INIT_ADAPTER_DIR:
        adapter_dir = Path(INIT_ADAPTER_DIR)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"INIT_ADAPTER_DIR not found: {adapter_dir}")
        print(f"Loading trainable initial adapter from local path: {adapter_dir}")
        print(
            "Adapter load settings: "
            f"torch_device={ADAPTER_LOAD_TORCH_DEVICE or 'peft-default'} "
            f"low_cpu_mem_usage={ADAPTER_LOAD_LOW_CPU_MEM_USAGE} "
            f"mode={INIT_ADAPTER_LOAD_MODE}"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if INIT_ADAPTER_LOAD_MODE.strip().lower() == "manual":
            loaded_model = create_lora_model(model)
            weights = load_peft_weights(str(adapter_dir), device="cpu")
            print(f"Manual local adapter load: tensors={len(weights)}")
            load_peft_weights_with_direct_fallback(loaded_model, weights, adapter_name="default")
            return loaded_model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded_model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                is_trainable=True,
                torch_device=optional_torch_device(ADAPTER_LOAD_TORCH_DEVICE),
                low_cpu_mem_usage=ADAPTER_LOAD_LOW_CPU_MEM_USAGE,
            )
        missing_adapter_warnings = [
            str(item.message) for item in caught if "missing adapter keys" in str(item.message).lower()
        ]
        if missing_adapter_warnings:
            print("MISSING_ADAPTER_KEYS_WARNING_BEGIN")
            for message in missing_adapter_warnings:
                print(message[:8000])
            print("MISSING_ADAPTER_KEYS_WARNING_END")
            if FAIL_ON_MISSING_ADAPTER_KEYS:
                raise RuntimeError(
                    "Initial adapter did not fully load into the HF model namespace. "
                    "Set FAIL_ON_MISSING_ADAPTER_KEYS=0 only for diagnostics."
                )
        return loaded_model

    if INIT_ADAPTER_REPO:
        print(
            "Loading trainable initial adapter from HF: "
            f"{INIT_ADAPTER_REPO}"
            + (f"/{INIT_ADAPTER_SUBFOLDER}" if INIT_ADAPTER_SUBFOLDER else "")
            + (f"@{INIT_ADAPTER_REVISION}" if INIT_ADAPTER_REVISION else "")
        )
        print(
            "Adapter load settings: "
            f"torch_device={ADAPTER_LOAD_TORCH_DEVICE or 'peft-default'} "
            f"low_cpu_mem_usage={ADAPTER_LOAD_LOW_CPU_MEM_USAGE} "
            f"mode={INIT_ADAPTER_LOAD_MODE}"
        )
        if INIT_ADAPTER_LOAD_MODE.strip().lower() == "manual":
            loaded_model = create_lora_model(model)
            weights = load_peft_weights(
                INIT_ADAPTER_REPO,
                subfolder=INIT_ADAPTER_SUBFOLDER or None,
                revision=INIT_ADAPTER_REVISION or None,
                token=HF_TOKEN or None,
                device="cpu",
            )
            print(f"Manual HF adapter load: tensors={len(weights)}")
            load_peft_weights_with_direct_fallback(loaded_model, weights, adapter_name="default")
            return loaded_model
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded_model = PeftModel.from_pretrained(
                model,
                INIT_ADAPTER_REPO,
                subfolder=INIT_ADAPTER_SUBFOLDER or None,
                revision=INIT_ADAPTER_REVISION or None,
                token=HF_TOKEN or None,
                is_trainable=True,
                torch_device=optional_torch_device(ADAPTER_LOAD_TORCH_DEVICE),
                low_cpu_mem_usage=ADAPTER_LOAD_LOW_CPU_MEM_USAGE,
            )
        missing_adapter_warnings = [
            str(item.message) for item in caught if "missing adapter keys" in str(item.message).lower()
        ]
        if missing_adapter_warnings:
            print("MISSING_ADAPTER_KEYS_WARNING_BEGIN")
            for message in missing_adapter_warnings:
                print(message[:8000])
            print("MISSING_ADAPTER_KEYS_WARNING_END")
            if FAIL_ON_MISSING_ADAPTER_KEYS:
                raise RuntimeError(
                    "Initial adapter did not fully load into the HF model namespace. "
                    "Set FAIL_ON_MISSING_ADAPTER_KEYS=0 only for diagnostics."
                )
        return loaded_model

    print("Creating a new trainable LoRA adapter.")
    return create_lora_model(model)


def parse_weight_map(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, raw_weight = item.split("=", 1)
        elif ":" in item:
            key, raw_weight = item.split(":", 1)
            print(f"Warning: weight entry uses ':'; normalized to key=value form: {item}")
        else:
            raise ValueError(f"weight entry must be key=value, got: {item}")
        key = key.strip()
        raw_weight = raw_weight.strip()
        if not key:
            raise ValueError(f"weight entry has empty key: {item}")
        if key in weights:
            raise ValueError(f"duplicate weight entry for key {key!r}: {item}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"weight entry has non-numeric value: {item}") from exc
        if not math.isfinite(weight):
            raise ValueError(f"weight must be finite: {item}")
        if weight <= 0:
            raise ValueError(f"weight must be positive: {item}")
        weights[key] = weight
    return weights


SUBCATEGORY_WEIGHT_MAP = parse_weight_map(SUBCATEGORY_WEIGHTS)
SOURCE_WEIGHT_MAP = parse_weight_map(SOURCE_WEIGHTS)


def parse_example_loss_weight(example: dict[str, Any], *, require: bool) -> tuple[float, bool]:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    raw_weight = metadata.get("loss_weight", example.get("loss_weight"))
    if raw_weight in (None, ""):
        if require:
            raise ValueError(f"example {example.get('id', '<missing>')} is missing metadata.loss_weight")
        return 1.0, False
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"example {example.get('id', '<missing>')} has non-numeric metadata.loss_weight={raw_weight!r}"
        ) from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"example {example.get('id', '<missing>')} has invalid metadata.loss_weight={raw_weight!r}"
        )
    return weight, True


def trainable_parameter_report(model: torch.nn.Module) -> dict[str, float | int]:
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    ratio = trainable / total if total else 0.0
    return {"trainable": trainable, "total": total, "ratio": ratio}


def apply_trainable_lora_module_filter(model: torch.nn.Module) -> dict[str, Any]:
    """Freeze loaded LoRA params except a deliberate module allowlist.

    The 0.86 adapter contains a full 9-module Huikang-style adapter.  Loading it
    as fully trainable uses roughly 888M trainable params and can OOM on long
    examples.  For a conservative delta run, keep the full adapter active in the
    forward pass but update only the lighter routing/attention/mamba projection
    modules named by TRAINABLE_LORA_MODULES.
    """

    modules = parse_csv_items(TRAINABLE_LORA_MODULES)
    name_substrings = parse_csv_items(TRAINABLE_LORA_NAME_SUBSTRINGS)
    required_trainable_substrings = parse_csv_items(REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS)
    target_parameters = parse_target_parameters(LORA_TARGET_PARAMETERS)
    target_parameter_lora_tensors: dict[str, int] = {item: 0 for item in target_parameters}
    target_parameter_lora_params: dict[str, int] = {item: 0 for item in target_parameters}
    target_parameter_trainable_lora_tensors: dict[str, int] = {item: 0 for item in target_parameters}
    target_parameter_trainable_lora_params: dict[str, int] = {item: 0 for item in target_parameters}
    required_trainable_lora_tensors: dict[str, int] = {item: 0 for item in required_trainable_substrings}
    required_trainable_lora_params: dict[str, int] = {item: 0 for item in required_trainable_substrings}

    if not modules and not name_substrings:
        trainable_lora_params = 0
        frozen_lora_params = 0
        trainable_lora_tensors = 0
        frozen_lora_tensors = 0
        for name, param in model.named_parameters():
            if ".lora_" not in name:
                continue
            count = int(param.numel())
            matched_target_parameters = [
                item for item in target_parameters if target_parameter_name_matches(item, name)
            ]
            for target_parameter in matched_target_parameters:
                target_parameter_lora_tensors[target_parameter] += 1
                target_parameter_lora_params[target_parameter] += count
            if param.requires_grad:
                trainable_lora_params += count
                trainable_lora_tensors += 1
                for target_parameter in matched_target_parameters:
                    target_parameter_trainable_lora_tensors[target_parameter] += 1
                    target_parameter_trainable_lora_params[target_parameter] += count
            else:
                frozen_lora_params += count
                frozen_lora_tensors += 1
        missing_target_parameters = [
            item for item, tensors in target_parameter_lora_tensors.items() if tensors <= 0
        ]
        if REQUIRE_LORA_TARGET_PARAMETER_MATCH and missing_target_parameters:
            raise RuntimeError(
                "LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found: "
                + ", ".join(missing_target_parameters)
            )
        missing_trainable_target_parameters = [
            item for item, tensors in target_parameter_trainable_lora_tensors.items() if tensors <= 0
        ]
        if target_parameters and len(missing_trainable_target_parameters) == len(target_parameters):
            target_parameters_trainability_mode = "frozen_active"
        elif missing_trainable_target_parameters:
            target_parameters_trainability_mode = "partially_trainable"
        elif target_parameters:
            target_parameters_trainability_mode = "trainable"
        else:
            target_parameters_trainability_mode = "not_configured"
        if REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE and missing_trainable_target_parameters:
            raise RuntimeError(
                "LORA_TARGET_PARAMETERS were configured but are not all trainable under the current "
                "LoRA selector. Missing trainable target_parameters: "
                + ", ".join(missing_trainable_target_parameters)
            )
        return {
            "enabled": False,
            "modules": [],
            "name_substrings": [],
            "trainable_lora_params": trainable_lora_params,
            "frozen_lora_params": frozen_lora_params,
            "trainable_lora_tensors": trainable_lora_tensors,
            "frozen_lora_tensors": frozen_lora_tensors,
            "trainable_by_module": {},
            "trainable_by_name_substring": {},
            "frozen_by_module": {},
            "target_parameter_lora_tensors": target_parameter_lora_tensors,
            "target_parameter_lora_params": target_parameter_lora_params,
            "target_parameter_trainable_lora_tensors": target_parameter_trainable_lora_tensors,
            "target_parameter_trainable_lora_params": target_parameter_trainable_lora_params,
            "target_parameters_trainability_mode": target_parameters_trainability_mode,
            "require_lora_target_parameters_trainable": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
            "required_trainable_lora_tensors": required_trainable_lora_tensors,
            "required_trainable_lora_params": required_trainable_lora_params,
        }

    trainable_by_module: dict[str, int] = {module: 0 for module in modules}
    trainable_by_name_substring: dict[str, int] = {substring: 0 for substring in name_substrings}
    frozen_by_module: dict[str, int] = {}
    trainable_lora_params = 0
    frozen_lora_params = 0
    trainable_lora_tensors = 0
    frozen_lora_tensors = 0

    for name, param in model.named_parameters():
        if ".lora_" not in name:
            continue
        matched_module = next((module for module in modules if f".{module}." in name), None)
        matched_name_substrings = [substring for substring in name_substrings if substring in name]
        matched_target_parameters: list[str] = []
        for target_parameter in target_parameters:
            if target_parameter_name_matches(target_parameter, name):
                matched_target_parameters.append(target_parameter)
                count = int(param.numel())
                target_parameter_lora_tensors[target_parameter] += 1
                target_parameter_lora_params[target_parameter] += count
        is_trainable_by_name = bool(matched_name_substrings)
        if matched_module or is_trainable_by_name:
            param.requires_grad_(True)
            count = int(param.numel())
            trainable_lora_params += count
            trainable_lora_tensors += 1
            if matched_module:
                trainable_by_module[matched_module] = trainable_by_module.get(matched_module, 0) + count
            for substring in matched_name_substrings:
                trainable_by_name_substring[substring] = (
                    trainable_by_name_substring.get(substring, 0) + count
                )
            for target_parameter in matched_target_parameters:
                target_parameter_trainable_lora_tensors[target_parameter] += 1
                target_parameter_trainable_lora_params[target_parameter] += count
            for required_substring in required_trainable_substrings:
                if required_substring in name:
                    required_trainable_lora_tensors[required_substring] += 1
                    required_trainable_lora_params[required_substring] += count
        else:
            param.requires_grad_(False)
            count = int(param.numel())
            frozen_lora_params += count
            frozen_lora_tensors += 1
            module_name = "unknown"
            for candidate in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj",
                "out_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]:
                if f".{candidate}." in name:
                    module_name = candidate
                    break
            frozen_by_module[module_name] = frozen_by_module.get(module_name, 0) + count

    if trainable_lora_params <= 0:
        raise RuntimeError(
            "Trainable LoRA selector matched no LoRA parameters: "
            f"TRAINABLE_LORA_MODULES={TRAINABLE_LORA_MODULES!r}, "
            f"TRAINABLE_LORA_NAME_SUBSTRINGS={TRAINABLE_LORA_NAME_SUBSTRINGS!r}"
        )
    if REQUIRE_LORA_TARGET_PARAMETER_MATCH:
        missing_target_parameters = [
            item for item, tensors in target_parameter_lora_tensors.items() if tensors <= 0
        ]
        if missing_target_parameters:
            raise RuntimeError(
                "LORA_TARGET_PARAMETERS were configured but no matching LoRA tensors were found: "
                + ", ".join(missing_target_parameters)
            )
    missing_trainable_target_parameters = [
        item for item, tensors in target_parameter_trainable_lora_tensors.items() if tensors <= 0
    ]
    if target_parameters and len(missing_trainable_target_parameters) == len(target_parameters):
        target_parameters_trainability_mode = "frozen_active"
    elif missing_trainable_target_parameters:
        target_parameters_trainability_mode = "partially_trainable"
    elif target_parameters:
        target_parameters_trainability_mode = "trainable"
    else:
        target_parameters_trainability_mode = "not_configured"
    if REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE and missing_trainable_target_parameters:
        raise RuntimeError(
            "LORA_TARGET_PARAMETERS were configured but are not all trainable under the current "
            "LoRA selector. Missing trainable target_parameters: "
            + ", ".join(missing_trainable_target_parameters)
        )
    missing_required_trainable = [
        item for item, tensors in required_trainable_lora_tensors.items() if tensors <= 0
    ]
    if missing_required_trainable:
        raise RuntimeError(
            "Required trainable LoRA name substrings were not matched: "
            + ", ".join(missing_required_trainable)
        )

    return {
        "enabled": True,
        "modules": modules,
        "name_substrings": name_substrings,
        "trainable_lora_params": trainable_lora_params,
        "frozen_lora_params": frozen_lora_params,
        "trainable_lora_tensors": trainable_lora_tensors,
        "frozen_lora_tensors": frozen_lora_tensors,
        "trainable_by_module": trainable_by_module,
        "trainable_by_name_substring": trainable_by_name_substring,
        "frozen_by_module": frozen_by_module,
        "target_parameter_lora_tensors": target_parameter_lora_tensors,
        "target_parameter_lora_params": target_parameter_lora_params,
        "target_parameter_trainable_lora_tensors": target_parameter_trainable_lora_tensors,
        "target_parameter_trainable_lora_params": target_parameter_trainable_lora_params,
        "target_parameters_trainability_mode": target_parameters_trainability_mode,
        "require_lora_target_parameters_trainable": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
        "required_trainable_lora_tensors": required_trainable_lora_tensors,
        "required_trainable_lora_params": required_trainable_lora_params,
    }


def cuda_memory_line() -> str:
    if not torch.cuda.is_available():
        return "cuda_mem=unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"mem_alloc={allocated:.1f}GiB mem_reserved={reserved:.1f}GiB"


def cuda_reserved_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024**3)


def cuda_peak_reserved_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / (1024**3)


def cuda_runtime_report() -> dict[str, Any]:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_names: list[str] = []
    for idx in range(device_count):
        try:
            device_names.append(torch.cuda.get_device_name(idx))
        except RuntimeError as exc:
            device_names.append(f"unavailable:{type(exc).__name__}:{exc}")
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(device_count),
        "cuda_device_names": device_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        "torch_allow_tf32": bool(TORCH_ALLOW_TF32),
        "torch_float32_matmul_precision": TORCH_FLOAT32_MATMUL_PRECISION,
        "torch_disable_cudnn_sdp": bool(TORCH_DISABLE_CUDNN_SDP),
        "torch_force_math_sdp": bool(TORCH_FORCE_MATH_SDP),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "gradient_checkpointing": bool(GRADIENT_CHECKPOINTING),
        "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
    }


def apply_runtime_performance_settings() -> None:
    """Apply conservative H100-friendly runtime knobs and log the result."""

    print("Applying runtime performance settings:")
    print(f"  TORCH_ALLOW_TF32={TORCH_ALLOW_TF32}")
    print(f"  TORCH_FLOAT32_MATMUL_PRECISION={TORCH_FLOAT32_MATMUL_PRECISION}")
    print(f"  TORCH_DISABLE_CUDNN_SDP={TORCH_DISABLE_CUDNN_SDP}")
    print(f"  TORCH_FORCE_MATH_SDP={TORCH_FORCE_MATH_SDP}")
    print(f"  ATTN_IMPLEMENTATION={ATTN_IMPLEMENTATION or 'transformers-default'}")
    print(f"  GRADIENT_CHECKPOINTING={GRADIENT_CHECKPOINTING}")
    if torch.cuda.is_available() and TORCH_ALLOW_TF32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  enabled torch CUDA/CUDNN TF32 flags")
        except Exception as exc:
            print(f"  warning: could not set TF32 flags: {type(exc).__name__}: {exc}")
    if TORCH_FLOAT32_MATMUL_PRECISION:
        try:
            torch.set_float32_matmul_precision(TORCH_FLOAT32_MATMUL_PRECISION)
            print(f"  set_float32_matmul_precision={TORCH_FLOAT32_MATMUL_PRECISION}")
        except Exception as exc:
            print(
                "  warning: could not set float32 matmul precision: "
                f"{type(exc).__name__}: {exc}"
            )
    if torch.cuda.is_available():
        try:
            if TORCH_DISABLE_CUDNN_SDP and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
                print("  disabled torch CUDA cuDNN SDPA backend")
            if TORCH_FORCE_MATH_SDP:
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(False)
                    print("  disabled torch CUDA flash SDPA backend")
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    print("  disabled torch CUDA mem-efficient SDPA backend")
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
                    print("  enabled torch CUDA math SDPA backend")
            sdpa_status = {}
            for name in [
                "flash_sdp_enabled",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "cudnn_sdp_enabled",
            ]:
                fn = getattr(torch.backends.cuda, name, None)
                if callable(fn):
                    try:
                        sdpa_status[name] = bool(fn())
                    except Exception as exc:
                        sdpa_status[name] = f"{type(exc).__name__}: {exc}"
            if sdpa_status:
                print(f"  sdpa_backend_status={json.dumps(sdpa_status, sort_keys=True)}")
        except Exception as exc:
            print(f"  warning: could not configure SDPA backends: {type(exc).__name__}: {exc}")


def setup_causal_conv1d_stub() -> None:
    """Inject a minimal causal_conv1d stub when the optional package is absent."""
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        import importlib.machinery
        import types

        stub = types.ModuleType("causal_conv1d")
        stub.causal_conv1d_fn = None
        stub.causal_conv1d_update = None
        stub.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
        sys.modules["causal_conv1d"] = stub
        print("Injected causal_conv1d stub")


def verify_model_runtime_dependencies() -> None:
    """Fail before model load if Nemotron-H runtime kernels are unavailable.

    Tokenize-only dry runs intentionally skip model import/load. Real training
    must not rely on the causal_conv1d stub because a broken CUDA extension can
    otherwise be hidden until the first expensive forward pass.
    """

    if TOKENIZE_ONLY_DRY_RUN:
        print("TOKENIZE_ONLY_DRY_RUN=1; skipping Nemotron-H runtime kernel import checks.")
        return

    import importlib

    checks = [
        ("causal_conv1d", "causal_conv1d"),
        ("mamba_ssm", "mamba_ssm"),
        ("mamba_ssm.ops.triton.layernorm_gated", "mamba_ssm layernorm_gated"),
        ("mamba_ssm.ops.selective_scan_interface", "mamba_ssm selective_scan"),
    ]
    for module_name, label in checks:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Required runtime dependency failed before model load: {label} "
                f"({module_name}) -> {type(exc).__name__}: {exc}. "
                "Install/build mamba-ssm and causal-conv1d against the exact "
                "active torch/CUDA runtime before running training."
            ) from exc
        print(
            "runtime_dependency_import_ok = "
            + json.dumps(
                {
                    "module": module_name,
                    "label": label,
                    "version": str(getattr(module, "__version__", "unknown")),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file_sha256(path: Path, expected_sha256: str, label: str) -> None:
    if not expected_sha256:
        return
    observed = file_sha256(path)
    if observed.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{label} sha256 mismatch for {path}: observed={observed} expected={expected_sha256}"
        )
    print(f"{label} sha256 OK: {observed}")


def resolve_data_file(filename: str) -> Path:
    """Use local file if present, otherwise download it from DATA_REPO."""
    local_path = Path(filename)
    if local_path.exists():
        print(f"Using local data file: {local_path}")
        return local_path

    print(f"Downloading dataset file from {DATA_REPO}/{filename}...")
    return Path(
        hf_hub_download(
            repo_id=DATA_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def load_tong_pretokenized_archive(archive_path: Path) -> list[dict[str, Any]]:
    """Load Tong Hui Kang corpus token/mask segments directly from archive.zip.

    The public Tong recipe is token-first: ``corpus.jsonl`` points to
    ``corpus/<problem_id>/synthetic.jsonl`` segment files with masked/unmasked
    token spans. This path avoids chat-template retokenization drift.
    """

    exclude_categories = parse_csv_set(PRETOKENIZED_EXCLUDE_CATEGORIES)
    rows: list[dict[str, Any]] = []
    skipped_not_included = 0
    skipped_excluded_category = 0
    skipped_missing_segment = 0
    skipped_empty_loss = 0
    mismatches: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path) as zf:
        index_lines = zf.read("corpus.jsonl").decode("utf-8", errors="replace").splitlines()
        for line_no, raw in enumerate(index_lines, start=1):
            if not raw.strip():
                continue
            entry = json.loads(raw)
            problem_id = str(entry.get("problem_id", ""))
            category = str(entry.get("category", "unknown"))
            if entry.get("included") is False:
                skipped_not_included += 1
                continue
            if category in exclude_categories:
                skipped_excluded_category += 1
                continue

            segment_name = str(entry.get("segment") or "synthetic.jsonl")
            member = f"corpus/{problem_id}/{segment_name}"
            try:
                segment_lines = zf.read(member).decode("utf-8", errors="replace").splitlines()
            except KeyError:
                skipped_missing_segment += 1
                continue

            input_ids: list[int] = []
            loss_mask: list[int] = []
            for segment_raw in segment_lines:
                if not segment_raw.strip():
                    continue
                segment = json.loads(segment_raw)
                tokens = [int(token) for token in segment.get("tokens", [])]
                is_unmasked = segment.get("type") == "unmasked"
                input_ids.extend(tokens)
                loss_mask.extend([1 if is_unmasked else 0] * len(tokens))

            if not input_ids or sum(loss_mask) == 0:
                skipped_empty_loss += 1
                continue
            if len(input_ids) > MAX_LENGTH:
                raise RuntimeError(
                    f"Pretokenized example {problem_id} has {len(input_ids)} tokens > MAX_LENGTH={MAX_LENGTH}. "
                    "Use MAX_LENGTH=8192 for Tong reproduction."
                )
            if entry.get("token_count") is not None and int(entry["token_count"]) != len(input_ids):
                mismatches.append({
                    "problem_id": problem_id,
                    "field": "token_count",
                    "expected": int(entry["token_count"]),
                    "observed": len(input_ids),
                })
            if entry.get("unmasked_token_count") is not None and int(entry["unmasked_token_count"]) != sum(loss_mask):
                mismatches.append({
                    "problem_id": problem_id,
                    "field": "unmasked_token_count",
                    "expected": int(entry["unmasked_token_count"]),
                    "observed": int(sum(loss_mask)),
                })
            rows.append(
                {
                    "id": problem_id,
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "category": category,
                    "subcategory": category,
                    "source": "tong_pretokenized_archive",
                    "answer": entry.get("answer"),
                    "token_count": len(input_ids),
                    "unmasked_token_count": int(sum(loss_mask)),
                    "index_line": line_no,
                }
            )

    if mismatches:
        raise RuntimeError(f"Pretokenized token/mask count mismatch sample: {mismatches[:5]}")
    print(
        "Pretokenized Tong archive load summary: "
        f"rows={len(rows)} skipped_not_included={skipped_not_included} "
        f"skipped_excluded_category={skipped_excluded_category} "
        f"skipped_missing_segment={skipped_missing_segment} "
        f"skipped_empty_loss={skipped_empty_loss} "
        f"excluded_categories={sorted(exclude_categories)}"
    )
    token_length_stats(rows, "Pretokenized corpus")
    return rows


def split_pretokenized_train_val(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic stratified validation split from pretokenized rows."""

    if not rows:
        raise RuntimeError("No pretokenized rows loaded")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category", "unknown")), []).append(row)

    rng = random.Random(SEED)
    val_target = PRETOKENIZED_VAL_EXAMPLES
    if val_target <= 0 and PRETOKENIZED_VAL_FRACTION > 0:
        val_target = max(1, int(round(len(rows) * PRETOKENIZED_VAL_FRACTION)))
    if val_target <= 0:
        val_target = min(720, max(1, len(rows) // 20))

    val_ids: set[str] = set()
    categories = sorted(by_category)
    base_per_category = max(1, val_target // max(1, len(categories)))
    for category in categories:
        items = list(by_category[category])
        rng.shuffle(items)
        take = min(base_per_category, len(items))
        val_ids.update(str(item["id"]) for item in items[:take])

    if len(val_ids) < val_target:
        remaining = [row for row in rows if str(row["id"]) not in val_ids]
        rng.shuffle(remaining)
        val_ids.update(str(item["id"]) for item in remaining[: max(0, val_target - len(val_ids))])

    train_data = list(rows) if PRETOKENIZED_VAL_COPY_ONLY else [row for row in rows if str(row["id"]) not in val_ids]
    val_data = [row for row in rows if str(row["id"]) in val_ids]
    print(
        f"Pretokenized split: train={len(train_data)} validation={len(val_data)} "
        f"target_validation={val_target} copy_only={PRETOKENIZED_VAL_COPY_ONLY}"
    )
    token_length_stats(train_data, "Train")
    token_length_stats(val_data, "Validation")
    return train_data, val_data


def token_length_stats(tokenized: list[dict[str, Any]], label: str) -> None:
    total_tokens = sum(len(t["input_ids"]) for t in tokenized)
    unmasked_tokens = sum(sum(t["loss_mask"]) for t in tokenized)
    print(f"\n{label}: {len(tokenized)} examples")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unmasked completion tokens: {unmasked_tokens:,}")

    cat_lens: dict[str, list[int]] = {}
    for item in tokenized:
        cat_lens.setdefault(item["category"], []).append(len(item["input_ids"]))

    for category, lens in sorted(cat_lens.items()):
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        p50 = lens_sorted[n // 2]
        p90 = lens_sorted[int(n * 0.9)]
        p99 = lens_sorted[int(n * 0.99)] if n >= 100 else lens_sorted[-1]
        truncated = sum(1 for length in lens if length == MAX_LENGTH)
        print(
            f"  {category}: n={n} p50={p50} p90={p90} "
            f"p99={p99} truncated={truncated}/{n}"
        )


def build_completion_mask(
    full_text: str,
    messages: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[int], list[float], bool, bool, int]:
    """Tokenize a chat transcript and mask loss to assistant completion tokens.

    Offset mappings are preferred because separate tokenization of the prompt
    and full transcript can disagree at the prompt/completion boundary.
    """

    assistant_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            assistant_text = str(message.get("content", ""))
            break
    if not assistant_text:
        return [], [], False, False, 0

    def answer_char_span() -> tuple[int, int] | None:
        """Return the assistant-relative final-answer span, when explicit."""

        boxed_matches = list(re.finditer(r"\\boxed\{", assistant_text))
        if boxed_matches:
            start = boxed_matches[-1].start()
            depth = 0
            for idx in range(start, len(assistant_text)):
                ch = assistant_text[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return start, idx + 1
            return start, len(assistant_text)

        marker_positions: list[int] = []
        for marker in ("Final answer:", "ANSWER:"):
            pos = assistant_text.rfind(marker)
            if pos >= 0:
                marker_positions.append(pos)
        if marker_positions:
            return max(marker_positions), len(assistant_text)
        return None

    assistant_start = full_text.rfind(assistant_text)
    weighted_tokens = 0
    if assistant_start >= 0:
        try:
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            input_ids = list(encoded["input_ids"])
            offsets = encoded.get("offset_mapping")
            if offsets and len(offsets) == len(input_ids):
                loss_mask: list[float] = []
                answer_span = answer_char_span()
                answer_start = answer_end = -1
                if answer_span is not None and ANSWER_SPAN_LOSS_WEIGHT > 1.0:
                    answer_start = assistant_start + answer_span[0]
                    answer_end = assistant_start + answer_span[1]
                for start, end in offsets:
                    token_start = int(start)
                    token_end = int(end)
                    if token_end <= assistant_start:
                        loss_mask.append(0.0)
                    elif (
                        answer_end > answer_start
                        and token_end > answer_start
                        and token_start < answer_end
                    ):
                        loss_mask.append(float(ANSWER_SPAN_LOSS_WEIGHT))
                        weighted_tokens += 1
                    else:
                        loss_mask.append(1.0)
                return input_ids, loss_mask, True, False, weighted_tokens
        except (NotImplementedError, TypeError, ValueError):
            pass

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    # enable_thinking=True aligns with Tong recipe (Progress Prize winner)
    # so the `<think>` scaffold is preserved in the prompt and the completion
    # can emit `</think>\n\boxed{answer}` naturally.
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prefix_mismatch = full_ids[: len(prompt_ids)] != prompt_ids
    prompt_len = min(len(prompt_ids), len(full_ids))
    loss_mask = [0.0] * prompt_len + [1.0] * (len(full_ids) - prompt_len)
    return full_ids, loss_mask, False, prefix_mismatch, 0


def tokenize_examples(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    label: str,
    *,
    apply_row_loss_weight: bool = False,
) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    skipped_missing_messages = 0
    skipped_no_loss = 0
    truncated_count = 0
    prompt_truncated_count = 0
    prompt_tokens_dropped_est = 0
    offset_mask_count = 0
    fallback_mask_count = 0
    fallback_prefix_mismatch_count = 0
    answer_span_weighted_examples = 0
    answer_span_weighted_tokens = 0
    row_loss_weight_examples = 0
    row_loss_weight_sum = 0.0
    row_loss_weight_min = float("inf")
    row_loss_weight_max = 0.0
    for ex in examples:
        msgs = ex.get("messages", [])
        if not msgs:
            skipped_missing_messages += 1
            continue

        try:
            full_text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
        except TypeError:
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        full_ids, loss_mask, used_offsets, prefix_mismatch, weighted_tokens = build_completion_mask(
            full_text, msgs, tokenizer
        )
        if used_offsets:
            offset_mask_count += 1
        else:
            fallback_mask_count += 1
        if prefix_mismatch:
            fallback_prefix_mismatch_count += 1
        if weighted_tokens:
            answer_span_weighted_examples += 1
            answer_span_weighted_tokens += weighted_tokens
        row_loss_weight = 1.0
        if apply_row_loss_weight and USE_ROW_LOSS_WEIGHT:
            row_loss_weight, has_explicit_row_weight = parse_example_loss_weight(
                ex,
                require=REQUIRE_ROW_LOSS_WEIGHT,
            )
            if has_explicit_row_weight:
                row_loss_weight_examples += 1
                row_loss_weight_sum += row_loss_weight
                row_loss_weight_min = min(row_loss_weight_min, row_loss_weight)
                row_loss_weight_max = max(row_loss_weight_max, row_loss_weight)

        if len(full_ids) > MAX_LENGTH:
            first_loss_idx = next((idx for idx, value in enumerate(loss_mask) if value), len(loss_mask))
            overflow = len(full_ids) - MAX_LENGTH
            dropped_loss = float(sum(loss_mask[:overflow]))
            if dropped_loss > 0:
                raise RuntimeError(
                    f"{label} truncation would drop supervised completion tokens for {ex.get('id', '')}: "
                    f"overflow={overflow} dropped_loss_weight={dropped_loss:.4f} max_length={MAX_LENGTH}"
                )
            dropped_prompt_tokens = min(overflow, first_loss_idx)
            if dropped_prompt_tokens > 0:
                prompt_truncated_count += 1
                prompt_tokens_dropped_est += dropped_prompt_tokens
            full_ids = full_ids[overflow:]
            loss_mask = loss_mask[overflow:]
            truncated_count += 1

        if sum(loss_mask) == 0:
            skipped_no_loss += 1
            continue

        tokenized.append(
            {
                "id": ex.get("id", ""),
                "input_ids": full_ids,
                "loss_mask": loss_mask,
                "category": ex.get("family", ex.get("category", "unknown")),
                "subcategory": (
                    (ex.get("metadata") or {}).get("subcategory")
                    or (ex.get("metadata") or {}).get("subtype")
                    or ex.get("subcategory")
                    or ex.get("subtype")
                    or "unknown"
                ),
                "source": ex.get("source") or (ex.get("metadata") or {}).get("source") or "unknown",
                "row_loss_weight": row_loss_weight,
            }
        )

    print(
        f"{label} tokenization summary: raw={len(examples)} tokenized={len(tokenized)} "
        f"truncated={truncated_count} prompt_truncated={prompt_truncated_count} "
        f"prompt_tokens_dropped_est={prompt_tokens_dropped_est} "
        f"skipped_missing_messages={skipped_missing_messages} "
        f"skipped_no_loss={skipped_no_loss} offset_masks={offset_mask_count} "
        f"fallback_masks={fallback_mask_count} "
        f"fallback_prefix_mismatches={fallback_prefix_mismatch_count} "
        f"answer_span_loss_weight={ANSWER_SPAN_LOSS_WEIGHT} "
        f"answer_span_weighted_examples={answer_span_weighted_examples} "
        f"answer_span_weighted_tokens={answer_span_weighted_tokens} "
        f"use_row_loss_weight={USE_ROW_LOSS_WEIGHT if apply_row_loss_weight else False} "
        f"row_loss_weight_examples={row_loss_weight_examples} "
        f"row_loss_weight_min={(row_loss_weight_min if row_loss_weight_examples else 1.0):.4f} "
        f"row_loss_weight_max={(row_loss_weight_max if row_loss_weight_examples else 1.0):.4f} "
        f"row_loss_weight_mean={(row_loss_weight_sum / row_loss_weight_examples if row_loss_weight_examples else 1.0):.4f}"
    )
    if apply_row_loss_weight and USE_ROW_LOSS_WEIGHT and REQUIRE_ROW_LOSS_WEIGHT:
        if row_loss_weight_examples != len(tokenized):
            raise RuntimeError(
                f"{label} requires explicit metadata.loss_weight for every tokenized row: "
                f"{row_loss_weight_examples}/{len(tokenized)}"
            )
    if ANSWER_SPAN_LOSS_WEIGHT > 1.0 and answer_span_weighted_examples <= 0:
        raise RuntimeError(
            f"{label} ANSWER_SPAN_LOSS_WEIGHT={ANSWER_SPAN_LOSS_WEIGHT} "
            "but no explicit Final answer/ANSWER/boxed spans were weighted."
        )
    if (
        ANSWER_SPAN_LOSS_WEIGHT > 1.0
        and ANSWER_SPAN_MIN_WEIGHTED_TOKENS > 0
        and answer_span_weighted_tokens < ANSWER_SPAN_MIN_WEIGHTED_TOKENS
    ):
        raise RuntimeError(
            f"{label} weighted answer-span tokens below floor: "
            f"{answer_span_weighted_tokens} < {ANSWER_SPAN_MIN_WEIGHTED_TOKENS}"
        )
    if REQUIRE_OFFSET_MASK and fallback_mask_count:
        raise RuntimeError(
            f"{label} tokenization used {fallback_mask_count} fallback completion masks. "
            "Set REQUIRE_OFFSET_MASK=0 only for a deliberate diagnostic run."
        )
    prompt_truncation_rate = prompt_truncated_count / max(1, len(examples))
    if prompt_truncation_rate > MAX_PROMPT_TRUNCATION_RATE:
        raise RuntimeError(
            f"{label} prompt truncation rate is too high: "
            f"{prompt_truncation_rate:.4%} > {MAX_PROMPT_TRUNCATION_RATE:.4%}. "
            "Increase MAX_LENGTH or reduce long/low-value examples before training."
        )
    token_length_stats(tokenized, label)
    return tokenized


def example_sampling_weight(item: dict[str, Any]) -> float:
    weight = 1.0
    subcategory = str(item.get("subcategory", "unknown"))
    source = str(item.get("source", "unknown"))
    weight *= SUBCATEGORY_WEIGHT_MAP.get(subcategory, 1.0)
    weight *= SOURCE_WEIGHT_MAP.get(source, 1.0)
    return weight


def weighted_sample_report(data: list[dict[str, Any]]) -> dict[str, Any]:
    total_weight = sum(example_sampling_weight(item) for item in data)
    by_subcategory: dict[str, float] = {}
    by_source: dict[str, float] = {}
    for item in data:
        weight = example_sampling_weight(item)
        by_subcategory[str(item.get("subcategory", "unknown"))] = by_subcategory.get(str(item.get("subcategory", "unknown")), 0.0) + weight
        by_source[str(item.get("source", "unknown"))] = by_source.get(str(item.get("source", "unknown")), 0.0) + weight

    def normalize(values: dict[str, float]) -> dict[str, float]:
        if total_weight <= 0:
            return values
        return {
            key: round(value / total_weight, 6)
            for key, value in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        }

    return {
        "mode": SAMPLING_MODE,
        "subcategory_weights": SUBCATEGORY_WEIGHT_MAP,
        "source_weights": SOURCE_WEIGHT_MAP,
        "weighted_share_by_subcategory": normalize(by_subcategory),
        "weighted_share_by_source": normalize(by_source),
    }


def build_epoch_train_data(train_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if SAMPLING_MODE == "shuffle":
        epoch_data = list(train_data)
        random.shuffle(epoch_data)
        return epoch_data
    if SAMPLING_MODE != "weighted_replacement":
        raise ValueError("SAMPLING_MODE must be 'shuffle' or 'weighted_replacement'")
    weights = [example_sampling_weight(item) for item in train_data]
    if not train_data or not any(weight > 0 for weight in weights):
        raise ValueError("weighted_replacement sampling has no positive weights")
    return random.choices(train_data, weights=weights, k=len(train_data))


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous().float()

    batch, seq_len, vocab = shift_logits.shape
    flat_logits = shift_logits.view(batch * seq_len, vocab)
    flat_labels = shift_labels.view(batch * seq_len)
    flat_mask = shift_mask.view(batch * seq_len)

    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if example_weights is not None:
        expanded_example_weights = example_weights.float().view(batch, 1).expand(batch, seq_len).reshape(batch * seq_len)
        effective_mask = flat_mask * expanded_example_weights
    else:
        effective_mask = flat_mask
    masked_loss = per_token_loss * effective_mask
    if LOSS_NORMALIZATION_MODE == "example_mean":
        token_loss = (per_token_loss * flat_mask).view(batch, seq_len)
        token_mask = flat_mask.view(batch, seq_len)
        per_example_counts = token_mask.sum(dim=1).clamp_min(1.0)
        active_examples = (token_mask.sum(dim=1) > 0).float()
        if active_examples.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        per_example_loss = token_loss.sum(dim=1) / per_example_counts
        if example_weights is not None:
            active_weights = example_weights.float() * active_examples
            total_active_weight = active_weights.sum()
            if total_active_weight == 0:
                return torch.tensor(0.0, device=logits.device)
            return (per_example_loss * active_weights).sum() / total_active_weight
        return (per_example_loss * active_examples).sum() / active_examples.sum()

    num_unmasked = effective_mask.sum()
    if num_unmasked == 0:
        return torch.tensor(0.0, device=logits.device)
    return masked_loss.sum() / num_unmasked


def get_lr(global_step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return LEARNING_RATE
    progress = min(1.0, max(0.0, global_step / max(1, total_steps - 1)))
    return FINAL_LEARNING_RATE + (LEARNING_RATE - FINAL_LEARNING_RATE) * (1.0 - progress)


def tensorize_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids_batch: list[list[int]] = []
    attention_mask_batch: list[list[int]] = []
    loss_mask_batch: list[list[float]] = []
    example_weights_batch: list[float] = []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids_batch.append(ex["input_ids"] + [pad_token_id] * pad_len)
        attention_mask_batch.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        loss_mask_batch.append(ex["loss_mask"] + [0] * pad_len)
        example_weights_batch.append(float(ex.get("row_loss_weight", 1.0)))

    input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(attention_mask_batch, dtype=torch.long, device="cuda")
    loss_mask = torch.tensor(loss_mask_batch, dtype=torch.float32, device="cuda")
    example_weights = torch.tensor(example_weights_batch, dtype=torch.float32, device="cuda")
    return input_ids, attention_mask, loss_mask, example_weights


def select_eval_sample(val_data: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or not val_data:
        return []
    if max_examples >= len(val_data):
        return list(val_data)

    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in val_data:
        by_category.setdefault(str(item.get("category", "unknown")), []).append(item)

    categories = sorted(by_category)
    sample: list[dict[str, Any]] = []
    per_category = max(1, max_examples // max(1, len(categories)))
    for category in categories:
        sample.extend(by_category[category][:per_category])

    cursor = per_category
    while len(sample) < max_examples:
        added = False
        for category in categories:
            values = by_category[category]
            if cursor < len(values):
                sample.append(values[cursor])
                added = True
                if len(sample) >= max_examples:
                    break
        if not added:
            break
        cursor += 1
    return sample[:max_examples]


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    val_data: list[dict[str, Any]],
    tokenizer: Any,
    max_examples: int,
    label: str = "eval",
) -> float:
    if not val_data or max_examples <= 0:
        return float("nan")

    was_training = model.training
    model.eval()
    sample = select_eval_sample(val_data, min(max_examples, len(val_data)))
    losses: list[float] = []
    total = len(sample)
    print(f"{label}_loss_eval_start examples={total} {cuda_memory_line()}", flush=True)

    with torch.no_grad():
        for index, item in enumerate(sample, start=1):
            input_ids, attention_mask, loss_mask, example_weights = tensorize_batch([item], tokenizer.pad_token_id)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss = masked_cross_entropy_loss(outputs.logits, input_ids, loss_mask, example_weights)
            losses.append(float(loss.item()))
            del input_ids, attention_mask, loss_mask, example_weights, outputs, loss
            if index == 1 or index == total or index % 8 == 0:
                partial = sum(losses) / max(1, len(losses))
                print(
                    f"{label}_loss_eval_progress {index}/{total} "
                    f"partial_loss={partial:.4f} {cuda_memory_line()}",
                    flush=True,
                )

    if was_training:
        model.train()
    result = sum(losses) / max(1, len(losses))
    print(f"{label}_loss_eval_end loss={result:.4f} {cuda_memory_line()}", flush=True)
    return result


def make_manifest(
    train_path: Path | None,
    val_path: Path | None,
    pretokenized_archive_path: Path | None,
    train_count: int,
    val_count: int,
    total_steps: int,
    final_step: int,
    best_eval_loss: float,
    elapsed_seconds: float,
    lora_filter_report: dict[str, Any] | None = None,
    trainable_report: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "data_repo": DATA_REPO,
        "train_file": DATA_FILE,
        "train_file_sha256": file_sha256(train_path) if train_path is not None else None,
        "train_records": train_count,
        "val_file": VAL_FILE,
        "val_records": val_count,
        "pretokenized_archive_zip": PRETOKENIZED_ARCHIVE_ZIP or None,
        "pretokenized_archive_sha256": (
            file_sha256(pretokenized_archive_path)
            if pretokenized_archive_path is not None and pretokenized_archive_path.exists()
            else None
        ),
        "pretokenized_exclude_categories": sorted(parse_csv_set(PRETOKENIZED_EXCLUDE_CATEGORIES)),
        "pretokenized_validation_examples": PRETOKENIZED_VAL_EXAMPLES,
        "pretokenized_validation_fraction": PRETOKENIZED_VAL_FRACTION,
        "pretokenized_validation_copy_only": PRETOKENIZED_VAL_COPY_ONLY,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "target_parameters": LORA_TARGET_PARAMETERS,
            "trainable_lora_modules": TRAINABLE_LORA_MODULES,
            "trainable_lora_name_substrings": TRAINABLE_LORA_NAME_SUBSTRINGS,
            "require_lora_target_parameter_match": REQUIRE_LORA_TARGET_PARAMETER_MATCH,
            "require_lora_target_parameters_trainable": REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE,
            "required_trainable_lora_name_substrings": REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS,
            "max_trainable_param_ratio": MAX_TRAINABLE_PARAM_RATIO,
            "trainable_lora_module_filter": lora_filter_report or {},
            "trainable_parameter_report_after_filter": trainable_report or {},
        },
        "training": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_steps": MAX_STEPS,
            "planned_total_steps": total_steps,
            "final_step": final_step,
            "best_eval_loss": best_eval_loss,
            "seed": SEED,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_seconds / 3600,
            "compute_provider": COMPUTE_PROVIDER,
            "vram_peak_gib": cuda_peak_reserved_gib(),
            "performance": {
                "model_device_map": MODEL_DEVICE_MAP,
                "attn_implementation": ATTN_IMPLEMENTATION,
                "torch_allow_tf32": TORCH_ALLOW_TF32,
                "torch_float32_matmul_precision": TORCH_FLOAT32_MATMUL_PRECISION,
                "torch_disable_cudnn_sdp": TORCH_DISABLE_CUDNN_SDP,
                "torch_force_math_sdp": TORCH_FORCE_MATH_SDP,
                "gradient_checkpointing": GRADIENT_CHECKPOINTING,
                "hf_hub_enable_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
                "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", ""),
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            },
            "abort_policy": {
                "eval_loss_gt": ABORT_EVAL_LOSS_GT,
                "baseline_eval_before_train": BASELINE_EVAL_BEFORE_TRAIN,
                "eval_relative_to_baseline_delta": ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA,
                "require_final_eval_lte_baseline": REQUIRE_FINAL_EVAL_LTE_BASELINE,
                "max_final_eval_regression": MAX_FINAL_EVAL_REGRESSION,
                "train_rise_points": ABORT_TRAIN_RISE_POINTS,
                "max_reserved_gib": ABORT_MAX_RESERVED_GIB,
            },
            "sampling": {
                "mode": SAMPLING_MODE,
                "subcategory_weights": SUBCATEGORY_WEIGHT_MAP,
                "source_weights": SOURCE_WEIGHT_MAP,
            },
            "answer_span_loss": {
                "weight": ANSWER_SPAN_LOSS_WEIGHT,
                "min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
                "markers": ["Final answer:", "ANSWER:", "\\boxed{...}"],
            },
            "row_loss_weight": {
                "enabled": USE_ROW_LOSS_WEIGHT,
                "required_for_train_rows": REQUIRE_ROW_LOSS_WEIGHT,
                "source_fields": ["metadata.loss_weight", "loss_weight"],
                "validation_ignores_row_weight": True,
            },
            "loss_normalization": {
                "mode": LOSS_NORMALIZATION_MODE,
                "meaning": "token_mean divides by all unmasked tokens; example_mean averages per-example CE first.",
            },
        },
        "output_repo": OUTPUT_REPO,
    }
    if val_path is not None and val_path.exists():
        manifest["val_file_sha256"] = file_sha256(val_path)
    return manifest


def upload_outputs(final_dir: Path) -> None:
    if not UPLOAD_TO_HF:
        print("UPLOAD_TO_HF=0; skipping adapter upload to Hugging Face.")
        return
    if not OUTPUT_REPO:
        print("OUTPUT_REPO is empty; skipping adapter upload.")
        return
    if not HF_TOKEN:
        print("HF_TOKEN not set; skipping adapter upload.")
        return

    print(f"\nUploading final adapter and checkpoints to {OUTPUT_REPO}...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(final_dir),
        path_in_repo="final",
        repo_id=OUTPUT_REPO,
        commit_message=f"{RUN_ID} final adapter",
        token=HF_TOKEN,
    )

    for checkpoint_dir in sorted(Path(OUTPUT_DIR).glob("checkpoint-*")):
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            path_in_repo=checkpoint_dir.name,
            repo_id=OUTPUT_REPO,
            commit_message=f"{RUN_ID} {checkpoint_dir.name}",
            token=HF_TOKEN,
        )
    print(f"Upload complete: {OUTPUT_REPO}")


def upload_checkpoint_during_training(checkpoint_dir: Path) -> None:
    if not UPLOAD_CHECKPOINTS_DURING_TRAINING:
        return
    if not UPLOAD_TO_HF or not OUTPUT_REPO or not HF_TOKEN:
        print("Skipping in-training checkpoint upload; HF upload settings are incomplete.")
        return

    print(f"Uploading checkpoint during training: {OUTPUT_REPO}/{checkpoint_dir.name}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        path_in_repo=checkpoint_dir.name,
        repo_id=OUTPUT_REPO,
        commit_message=f"{RUN_ID} {checkpoint_dir.name} interim",
        token=HF_TOKEN,
    )
    print(f"Checkpoint uploaded: {OUTPUT_REPO}/{checkpoint_dir.name}")


def upload_dry_run_report(dry_run_path: Path) -> None:
    if not UPLOAD_TO_HF:
        print("UPLOAD_TO_HF=0; skipping dry-run report upload.")
        return
    if not OUTPUT_REPO:
        print("OUTPUT_REPO is empty; skipping dry-run report upload.")
        return
    if not HF_TOKEN:
        print("HF_TOKEN not set; skipping dry-run report upload.")
        return

    path_in_repo = f"dry_runs/{RUN_ID}/dry_run_model_recipe_report.json"
    print(f"Uploading dry-run recipe report to {OUTPUT_REPO}/{path_in_repo}...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(dry_run_path),
        path_in_repo=path_in_repo,
        repo_id=OUTPUT_REPO,
        commit_message=f"{RUN_ID} dry-run recipe report",
        token=HF_TOKEN,
    )
    print(f"Dry-run report uploaded: {OUTPUT_REPO}/{path_in_repo}")


def train() -> None:
    print("=" * 72)
    print("KG1 v90 category-solver remote training")
    print("=" * 72)
    print(f"Run ID: {RUN_ID}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model revision: {MODEL_REVISION or 'default'}")
    if PRETOKENIZED_ARCHIVE_ZIP:
        print(f"Pretokenized Tong archive: {PRETOKENIZED_ARCHIVE_ZIP}")
        print(f"Pretokenized excluded categories: {PRETOKENIZED_EXCLUDE_CATEGORIES or 'none'}")
    else:
        print(f"Train: {DATA_REPO}/{DATA_FILE}")
        print(f"Validation: {DATA_REPO}/{VAL_FILE}")
    print(f"Output repo: {OUTPUT_REPO}")
    print(f"Upload to HF: {UPLOAD_TO_HF}")
    print(f"Require offset masks: {REQUIRE_OFFSET_MASK}")
    print(f"Dry-run validate only: {DRY_RUN_VALIDATE_ONLY}")
    if INIT_ADAPTER_DIR or INIT_ADAPTER_REPO:
        print(
            "Initial adapter: "
            f"{INIT_ADAPTER_DIR or INIT_ADAPTER_REPO}"
            + (f"/{INIT_ADAPTER_SUBFOLDER}" if INIT_ADAPTER_SUBFOLDER else "")
            + (f"@{INIT_ADAPTER_REVISION}" if INIT_ADAPTER_REVISION else "")
        )
    else:
        print("Initial adapter: new LoRA")
    print(
        f"LoRA: r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
        f"target_modules={LORA_TARGET_MODULES}"
    )
    print(f"LoRA target_parameters: {LORA_TARGET_PARAMETERS or 'disabled'}")
    print(f"Trainable LoRA module filter: {TRAINABLE_LORA_MODULES or 'disabled'}")
    print(f"Trainable LoRA name substring filter: {TRAINABLE_LORA_NAME_SUBSTRINGS or 'disabled'}")
    print(f"Require LoRA target-parameter match: {REQUIRE_LORA_TARGET_PARAMETER_MATCH}")
    print(f"Require LoRA target-parameters trainable: {REQUIRE_LORA_TARGET_PARAMETERS_TRAINABLE}")
    print(
        "Required trainable LoRA name substrings: "
        f"{REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS or 'disabled'}"
    )
    print(f"Max trainable parameter ratio: {MAX_TRAINABLE_PARAM_RATIO:.4%}")
    print(f"Length={MAX_LENGTH} batch={BATCH_SIZE} micro_batch={MICRO_BATCH_SIZE}")
    print(f"LR: {LEARNING_RATE:.2e} -> {FINAL_LEARNING_RATE:.2e}")
    print(f"Epochs={NUM_EPOCHS} max_steps={MAX_STEPS}")
    print(f"Model device map: {MODEL_DEVICE_MAP}")
    print(f"Attention implementation: {ATTN_IMPLEMENTATION or 'transformers-default'}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"HF transfer enabled: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '')}")
    print()

    if not torch.cuda.is_available() and not (DRY_RUN_VALIDATE_ONLY and TOKENIZE_ONLY_DRY_RUN):
        raise RuntimeError("CUDA GPU is required for this Nemotron v90 training job.")

    apply_runtime_performance_settings()
    if TOKENIZE_ONLY_DRY_RUN:
        setup_causal_conv1d_stub()
    else:
        verify_model_runtime_dependencies()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION or None,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path: Path | None = None
    val_path: Path | None = None
    pretokenized_archive_resolved_path: Path | None = None
    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []
    if PRETOKENIZED_ARCHIVE_ZIP:
        archive_path = Path(PRETOKENIZED_ARCHIVE_ZIP)
        if not archive_path.exists():
            print(
                f"PRETOKENIZED_ARCHIVE_ZIP is not local; downloading from "
                f"{DATA_REPO}/{PRETOKENIZED_ARCHIVE_ZIP}..."
            )
            archive_path = Path(
                hf_hub_download(
                    repo_id=DATA_REPO,
                    filename=PRETOKENIZED_ARCHIVE_ZIP,
                    repo_type="dataset",
                    token=HF_TOKEN or None,
                )
            )
        pretokenized_archive_resolved_path = archive_path
        if EXPECTED_ARCHIVE_SHA256:
            assert_file_sha256(archive_path, EXPECTED_ARCHIVE_SHA256, "pretokenized archive")
        all_tokenized = load_tong_pretokenized_archive(archive_path)
        train_data, val_data = split_pretokenized_train_val(all_tokenized)
        train_examples = train_data
        val_examples = val_data
    else:
        train_path = resolve_data_file(DATA_FILE)
        val_path = resolve_data_file(VAL_FILE)
        assert_file_sha256(train_path, EXPECTED_TRAIN_SHA256, "train dataset")
        assert_file_sha256(val_path, EXPECTED_VAL_SHA256, "validation dataset")
        train_examples = load_jsonl(train_path)
        val_examples = load_jsonl(val_path)
        print(f"Loaded train={len(train_examples)} validation={len(val_examples)}")
        if len(train_examples) < MIN_TRAIN_EXAMPLES:
            raise RuntimeError(
                f"Train dataset too small: {len(train_examples)} < MIN_TRAIN_EXAMPLES={MIN_TRAIN_EXAMPLES}"
            )
        if len(val_examples) < MIN_VAL_EXAMPLES:
            raise RuntimeError(
                f"Validation dataset too small: {len(val_examples)} < MIN_VAL_EXAMPLES={MIN_VAL_EXAMPLES}"
            )
        train_data = tokenize_examples(train_examples, tokenizer, "Train", apply_row_loss_weight=True)
        val_data = tokenize_examples(val_examples, tokenizer, "Validation", apply_row_loss_weight=False)
    if len(train_data) < MIN_TOKENIZED_TRAIN_EXAMPLES:
        raise RuntimeError(
            "Too few train examples survived tokenization: "
            f"{len(train_data)} < MIN_TOKENIZED_TRAIN_EXAMPLES={MIN_TOKENIZED_TRAIN_EXAMPLES}"
        )
    if len(val_data) < MIN_TOKENIZED_VAL_EXAMPLES:
        raise RuntimeError(
            "Too few validation examples survived tokenization: "
            f"{len(val_data)} < MIN_TOKENIZED_VAL_EXAMPLES={MIN_TOKENIZED_VAL_EXAMPLES}"
        )
    if DRY_RUN_VALIDATE_ONLY and TOKENIZE_ONLY_DRY_RUN:
        dry_run_report = {
            "run_id": RUN_ID,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "data": {
                "data_repo": DATA_REPO,
                "train_file": DATA_FILE,
                "train_file_sha256": file_sha256(train_path) if train_path else None,
                "train_records": len(train_examples),
                "tokenized_train_records": len(train_data),
                "validation_file": VAL_FILE,
                "validation_file_sha256": file_sha256(val_path) if val_path else None,
                "validation_records": len(val_examples),
                "tokenized_validation_records": len(val_data),
                "pretokenized_archive_zip": PRETOKENIZED_ARCHIVE_ZIP,
                "pretokenized_archive_resolved_path": str(pretokenized_archive_resolved_path) if pretokenized_archive_resolved_path else None,
                "pretokenized_archive_sha256": file_sha256(pretokenized_archive_resolved_path) if pretokenized_archive_resolved_path else None,
                "pretokenized_exclude_categories": sorted(parse_csv_set(PRETOKENIZED_EXCLUDE_CATEGORIES)),
                "pretokenized_validation_examples": PRETOKENIZED_VAL_EXAMPLES,
                "pretokenized_validation_fraction": PRETOKENIZED_VAL_FRACTION,
                "pretokenized_validation_copy_only": PRETOKENIZED_VAL_COPY_ONLY,
            },
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "target_parameters": LORA_TARGET_PARAMETERS,
                "parsed_target_modules": parse_target_modules(LORA_TARGET_MODULES),
                "init_adapter_dir": INIT_ADAPTER_DIR,
                "init_adapter_repo": INIT_ADAPTER_REPO,
                "init_adapter_revision": INIT_ADAPTER_REVISION,
                "init_adapter_subfolder": INIT_ADAPTER_SUBFOLDER,
                "trainable_lora_module_filter": {
                    "enabled": bool(TRAINABLE_LORA_MODULES or TRAINABLE_LORA_NAME_SUBSTRINGS),
                    "requested": sorted(parse_csv_set(TRAINABLE_LORA_MODULES)),
                    "requested_name_substrings": parse_csv_items(TRAINABLE_LORA_NAME_SUBSTRINGS),
                    "require_lora_target_parameter_match": REQUIRE_LORA_TARGET_PARAMETER_MATCH,
                    "required_trainable_lora_name_substrings": parse_csv_items(
                        REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS
                    ),
                    "checked_after_model_load": False,
                },
            },
            "training": {
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "max_trainable_param_ratio": MAX_TRAINABLE_PARAM_RATIO,
                "max_prompt_truncation_rate": MAX_PROMPT_TRUNCATION_RATE,
                "sampling": weighted_sample_report(train_data),
                "answer_span_loss": {
                    "weight": ANSWER_SPAN_LOSS_WEIGHT,
                    "min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
                },
            },
            "runtime": {
                "cuda_available": bool(torch.cuda.is_available()),
                "tokenize_only_dry_run": True,
                "torch": getattr(torch, "__version__", "unknown"),
            },
            "trainable_parameters": {
                "checked_after_model_load": False,
                "reason": "TOKENIZE_ONLY_DRY_RUN=1 skips the expensive model load.",
            },
            "decision": {
                "full_training_allowed": True,
                "note": (
                    "Tokenize-only dry run validated dataset hashes, row counts, "
                    "chat-template tokenization, offset masks, prompt truncation guard, "
                    "and weighted sampling. Model/adapter trainability is checked by "
                    "the subsequent real train cell."
                ),
            },
        }
        dry_run_path = Path(OUTPUT_DIR) / "dry_run_model_recipe_report.json"
        dry_run_path.parent.mkdir(parents=True, exist_ok=True)
        dry_run_path.write_text(json.dumps(dry_run_report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"TOKENIZE_ONLY_DRY_RUN=1; wrote {dry_run_path} and skipped expensive model load.")
        print("DRY_RUN_MODEL_RECIPE_REPORT_JSON_BEGIN")
        print(json.dumps(dry_run_report, sort_keys=True))
        print("DRY_RUN_MODEL_RECIPE_REPORT_JSON_END")
        upload_dry_run_report(dry_run_path)
        return

    model_device_map = parse_model_device_map(MODEL_DEVICE_MAP)
    print(
        f"\nLoading model {MODEL_NAME} in BF16 with "
        f"device_map={model_device_map} attn_implementation="
        f"{ATTN_IMPLEMENTATION or 'transformers-default'}..."
    )
    model_kwargs = {
        "pretrained_model_name_or_path": MODEL_NAME,
        "revision": MODEL_REVISION or None,
        "dtype": torch.bfloat16,
        "device_map": model_device_map,
        "trust_remote_code": True,
        "token": HF_TOKEN or None,
    }
    if ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    post_load_device = model_post_load_device(MODEL_DEVICE_MAP)
    if post_load_device:
        print(f"Moving fully loaded base model to {post_load_device}...")
        model.to(post_load_device)
        if torch.cuda.is_available():
            print(
                "Model moved to CUDA: "
                f"mem_alloc={torch.cuda.memory_allocated() / 1024**3:.1f}GiB "
                f"mem_reserved={torch.cuda.memory_reserved() / 1024**3:.1f}GiB"
            )

    target_modules = parse_target_modules(LORA_TARGET_MODULES)
    print("Applying/loading trainable LoRA adapter...")
    model = load_trainable_adapter_or_create(model)
    lora_filter_report = apply_trainable_lora_module_filter(model)
    if lora_filter_report["enabled"]:
        print("Applied trainable LoRA module filter:")
        print(json.dumps(lora_filter_report, indent=2, sort_keys=True))
    model.enable_input_require_grads()
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    else:
        print("Gradient checkpointing disabled by env.")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    trainable_report = trainable_parameter_report(model)
    print(
        "Trainable parameter guard: "
        f"{trainable_report['trainable']:,} / {trainable_report['total']:,} "
        f"({float(trainable_report['ratio']) * 100:.4f}%)"
    )
    if float(trainable_report["ratio"]) > MAX_TRAINABLE_PARAM_RATIO:
        raise RuntimeError(
            "LoRA recipe trains too many parameters: "
            f"{float(trainable_report['ratio']) * 100:.4f}% > "
            f"MAX_TRAINABLE_PARAM_RATIO={MAX_TRAINABLE_PARAM_RATIO * 100:.4f}%. "
            "This blocks silent all-linear expansion; set LORA_TARGET_MODULES "
            "explicitly or raise the guard only for a deliberate diagnostic run."
        )
    if DRY_RUN_VALIDATE_ONLY:
        dry_run_report = {
            "run_id": RUN_ID,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "data": {
                "data_repo": DATA_REPO,
                "train_file": DATA_FILE,
                "train_file_sha256": file_sha256(train_path) if train_path else None,
                "train_records": len(train_examples),
                "tokenized_train_records": len(train_data),
                "validation_file": VAL_FILE,
                "validation_file_sha256": file_sha256(val_path) if val_path else None,
                "validation_records": len(val_examples),
                "tokenized_validation_records": len(val_data),
                "pretokenized_archive_zip": PRETOKENIZED_ARCHIVE_ZIP,
                "pretokenized_archive_resolved_path": str(pretokenized_archive_resolved_path) if pretokenized_archive_resolved_path else None,
                "pretokenized_archive_sha256": file_sha256(pretokenized_archive_resolved_path) if pretokenized_archive_resolved_path else None,
                "pretokenized_exclude_categories": sorted(parse_csv_set(PRETOKENIZED_EXCLUDE_CATEGORIES)),
                "pretokenized_validation_examples": PRETOKENIZED_VAL_EXAMPLES,
                "pretokenized_validation_fraction": PRETOKENIZED_VAL_FRACTION,
                "pretokenized_validation_copy_only": PRETOKENIZED_VAL_COPY_ONLY,
            },
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "target_parameters": LORA_TARGET_PARAMETERS,
                "parsed_target_modules": target_modules,
                "init_adapter_dir": INIT_ADAPTER_DIR,
                "init_adapter_repo": INIT_ADAPTER_REPO,
                "init_adapter_revision": INIT_ADAPTER_REVISION,
                "init_adapter_subfolder": INIT_ADAPTER_SUBFOLDER,
                "trainable_lora_module_filter": lora_filter_report,
            },
            "training": {
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "max_trainable_param_ratio": MAX_TRAINABLE_PARAM_RATIO,
                "max_prompt_truncation_rate": MAX_PROMPT_TRUNCATION_RATE,
                "sampling": weighted_sample_report(train_data),
                "answer_span_loss": {
                    "weight": ANSWER_SPAN_LOSS_WEIGHT,
                    "min_weighted_tokens": ANSWER_SPAN_MIN_WEIGHTED_TOKENS,
                },
            },
            "runtime": cuda_runtime_report(),
            "trainable_parameters": trainable_report,
            "decision": {
                "full_training_allowed": True,
                "note": "Dry run loaded model, applied LoRA, and passed the trainable-parameter guard.",
            },
        }
        dry_run_path = Path(OUTPUT_DIR) / "dry_run_model_recipe_report.json"
        dry_run_path.parent.mkdir(parents=True, exist_ok=True)
        dry_run_path.write_text(json.dumps(dry_run_report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"DRY_RUN_VALIDATE_ONLY=1; wrote {dry_run_path} and skipped training.")
        print("DRY_RUN_MODEL_RECIPE_REPORT_JSON_BEGIN")
        print(json.dumps(dry_run_report, sort_keys=True))
        print("DRY_RUN_MODEL_RECIPE_REPORT_JSON_END")
        upload_dry_run_report(dry_run_path)
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if USE_BITSANDBYTES:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdam8bit(
                trainable_params,
                lr=LEARNING_RATE,
                betas=(ADAM_BETA1, ADAM_BETA2),
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            )
            print("Optimizer: bitsandbytes PagedAdam8bit")
        except Exception as exc:
            print(f"bitsandbytes optimizer unavailable ({exc}); using torch Adam")
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=LEARNING_RATE,
                betas=(ADAM_BETA1, ADAM_BETA2),
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            )
    else:
        print("Optimizer: torch Adam (USE_BITSANDBYTES=0)")
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    epoch_steps = math.ceil(len(train_data) / BATCH_SIZE)
    planned_steps = epoch_steps * NUM_EPOCHS
    total_steps = min(planned_steps, MAX_STEPS) if MAX_STEPS > 0 else planned_steps
    print(f"\nTraining: {len(train_data)} examples, planned_steps={total_steps}")
    print("Sampling:")
    print(json.dumps(weighted_sample_report(train_data), indent=2, sort_keys=True))

    baseline_eval_loss: float | None = None
    if BASELINE_EVAL_BEFORE_TRAIN:
        if not val_data:
            raise ValueError("BASELINE_EVAL_BEFORE_TRAIN=1 requires validation data")
        print(
            f"Baseline eval before training: max_examples={EVAL_MAX_EXAMPLES}",
            flush=True,
        )
        baseline_eval_loss = evaluate_loss(model, val_data, tokenizer, EVAL_MAX_EXAMPLES)
        if not math.isfinite(baseline_eval_loss):
            raise FloatingPointError(
                f"Non-finite baseline eval loss before training: {baseline_eval_loss}"
            )
        print(f"baseline_eval_loss={baseline_eval_loss:.4f}", flush=True)

    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    start_time = time.time()
    best_eval_loss = float("inf")
    train_eval_point_losses: list[float] = []
    step_loss_history: list[dict[str, float | int]] = []
    eval_history: list[dict[str, float | int | str]] = []
    checkpoint_history: list[dict[str, str | int]] = []

    for epoch in range(NUM_EPOCHS):
        if global_step >= total_steps:
            break

        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        epoch_data = build_epoch_train_data(train_data)

        for i in range(0, len(epoch_data), MICRO_BATCH_SIZE):
            if global_step >= total_steps:
                break

            batch = epoch_data[i : i + MICRO_BATCH_SIZE]
            if not batch:
                continue

            input_ids, attention_mask, loss_mask, example_weights = tensorize_batch(batch, tokenizer.pad_token_id)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss = masked_cross_entropy_loss(outputs.logits, input_ids, loss_mask, example_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss}")

            scaled_loss = loss / GRADIENT_ACCUMULATION
            scaled_loss.backward()

            loss_value = float(loss.item())
            accum_loss += loss_value
            accum_count += 1
            micro_in_step = accum_count % GRADIENT_ACCUMULATION
            should_step_optimizer = micro_in_step == 0

            del input_ids, attention_mask, loss_mask, example_weights, outputs, loss, scaled_loss

            if (
                MICRO_LOG_EVERY > 0
                and not should_step_optimizer
                and accum_count % MICRO_LOG_EVERY == 0
            ):
                print(
                    f"micro={micro_in_step}/{GRADIENT_ACCUMULATION} "
                    f"step={global_step}/{total_steps} "
                    f"micro_loss={loss_value:.4f} {cuda_memory_line()}",
                    flush=True,
                )

            if not should_step_optimizer:
                continue

            lr = get_lr(global_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if GRAD_CLIP_NORM < 1e8:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step = global_step + 1
            if torch.cuda.is_available() and completed_step % 5 == 0:
                torch.cuda.empty_cache()

            avg_loss = accum_loss / GRADIENT_ACCUMULATION
            step_loss_history.append(
                {
                    "step": completed_step,
                    "lr": lr,
                    "train_loss": avg_loss,
                }
            )
            elapsed = time.time() - start_time
            if LOG_EVERY_STEPS > 0 and completed_step % LOG_EVERY_STEPS == 0:
                print(
                    f"step={completed_step}/{total_steps} lr={lr:.2e} "
                    f"train_loss={avg_loss:.4f} elapsed={elapsed / 60:.1f}m "
                    f"{cuda_memory_line()}",
                    flush=True,
                )

            if ABORT_MAX_RESERVED_GIB > 0 and cuda_reserved_gib() > ABORT_MAX_RESERVED_GIB:
                checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-abort-step{completed_step}"
                model.save_pretrained(str(checkpoint_dir))
                tokenizer.save_pretrained(str(checkpoint_dir))
                print(
                    f"ABORT: mem_reserved {cuda_reserved_gib():.2f}GiB exceeded "
                    f"ABORT_MAX_RESERVED_GIB={ABORT_MAX_RESERVED_GIB:.2f}; "
                    f"emergency checkpoint saved: {checkpoint_dir}",
                    flush=True,
                )
                upload_checkpoint_during_training(checkpoint_dir)
                raise RuntimeError("abort_max_reserved_gib_exceeded")

            if (
                EVAL_EVERY_STEPS > 0
                and val_data
                and completed_step > 0
                and completed_step % EVAL_EVERY_STEPS == 0
            ):
                train_eval_point_losses.append(avg_loss)
                if (
                    ABORT_TRAIN_RISE_POINTS > 1
                    and len(train_eval_point_losses) >= ABORT_TRAIN_RISE_POINTS
                ):
                    window = train_eval_point_losses[-ABORT_TRAIN_RISE_POINTS:]
                    if all(left < right for left, right in zip(window, window[1:])):
                        checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-abort-step{completed_step}"
                        model.save_pretrained(str(checkpoint_dir))
                        tokenizer.save_pretrained(str(checkpoint_dir))
                        print(
                            f"ABORT: train_loss rose across {ABORT_TRAIN_RISE_POINTS} "
                            f"eval points: {window}; emergency checkpoint saved: {checkpoint_dir}",
                            flush=True,
                        )
                        upload_checkpoint_during_training(checkpoint_dir)
                        raise RuntimeError("abort_train_loss_rising")
                eval_loss = evaluate_loss(model, val_data, tokenizer, EVAL_MAX_EXAMPLES)
                best_eval_loss = min(best_eval_loss, eval_loss)
                eval_history.append(
                    {
                        "step": completed_step,
                        "eval_loss": eval_loss,
                        "best_eval_loss": best_eval_loss,
                        "kind": "scheduled",
                    }
                )
                print(
                    f"eval step={completed_step} "
                    f"loss={eval_loss:.4f} best={best_eval_loss:.4f}"
                )
                if not math.isfinite(eval_loss):
                    raise FloatingPointError(f"Non-finite eval loss at step {completed_step}")
                if ABORT_EVAL_LOSS_GT > 0 and eval_loss > ABORT_EVAL_LOSS_GT:
                    checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-abort-step{completed_step}"
                    model.save_pretrained(str(checkpoint_dir))
                    tokenizer.save_pretrained(str(checkpoint_dir))
                    print(
                        f"ABORT: eval_loss {eval_loss:.4f} exceeded "
                        f"ABORT_EVAL_LOSS_GT={ABORT_EVAL_LOSS_GT:.4f}; "
                        f"emergency checkpoint saved: {checkpoint_dir}",
                        flush=True,
                    )
                    upload_checkpoint_during_training(checkpoint_dir)
                    raise RuntimeError("abort_eval_loss_exceeded")
                if (
                    baseline_eval_loss is not None
                    and ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA >= 0
                    and eval_loss
                    > baseline_eval_loss + ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA
                ):
                    checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-abort-step{completed_step}"
                    model.save_pretrained(str(checkpoint_dir))
                    tokenizer.save_pretrained(str(checkpoint_dir))
                    allowed = baseline_eval_loss + ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA
                    print(
                        f"ABORT: eval_loss {eval_loss:.4f} exceeded baseline "
                        f"{baseline_eval_loss:.4f} + "
                        f"ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA="
                        f"{ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA:.4f} "
                        f"(allowed={allowed:.4f}); emergency checkpoint saved: "
                        f"{checkpoint_dir}",
                        flush=True,
                    )
                    upload_checkpoint_during_training(checkpoint_dir)
                    raise RuntimeError("abort_eval_loss_regressed_vs_baseline")

            if SAVE_EVERY_STEPS > 0 and completed_step > 0 and completed_step % SAVE_EVERY_STEPS == 0:
                checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-{completed_step}"
                model.save_pretrained(str(checkpoint_dir))
                tokenizer.save_pretrained(str(checkpoint_dir))
                print(f"Checkpoint saved: {checkpoint_dir}")
                checkpoint_history.append(
                    {
                        "step": completed_step,
                        "path": str(checkpoint_dir),
                        "kind": "scheduled",
                    }
                )
                upload_checkpoint_during_training(checkpoint_dir)

            accum_loss = 0.0
            global_step = completed_step

            if global_step % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print("\n=== TRAINING COMPLETE ===")
    print(f"Time: {elapsed / 3600:.2f}h")
    print(f"Final step: {global_step}")

    final_dir = Path(OUTPUT_DIR) / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    final_eval_loss = float("nan")
    if val_data:
        final_eval_loss = evaluate_loss(model, val_data, tokenizer, EVAL_MAX_EXAMPLES)
        best_eval_loss = min(best_eval_loss, final_eval_loss)
        eval_history.append(
            {
                "step": global_step,
                "eval_loss": final_eval_loss,
                "best_eval_loss": best_eval_loss,
                "kind": "final",
            }
        )
        print(f"Final eval loss: {final_eval_loss:.4f}; best eval loss: {best_eval_loss:.4f}")
        if (
            baseline_eval_loss is not None
            and REQUIRE_FINAL_EVAL_LTE_BASELINE
            and final_eval_loss > baseline_eval_loss + MAX_FINAL_EVAL_REGRESSION
        ):
            allowed = baseline_eval_loss + MAX_FINAL_EVAL_REGRESSION
            print(
                f"ABORT: final_eval_loss {final_eval_loss:.4f} exceeded baseline "
                f"{baseline_eval_loss:.4f} + MAX_FINAL_EVAL_REGRESSION="
                f"{MAX_FINAL_EVAL_REGRESSION:.4f} (allowed={allowed:.4f}). "
                "Final adapter was saved for forensics only and must not be submitted.",
                flush=True,
            )
            raise RuntimeError("final_eval_regressed_vs_baseline")

    manifest = make_manifest(
        train_path=train_path,
        val_path=val_path,
        pretokenized_archive_path=pretokenized_archive_resolved_path,
        train_count=len(train_examples),
        val_count=len(val_examples),
        total_steps=total_steps,
        final_step=global_step,
        best_eval_loss=best_eval_loss,
        elapsed_seconds=elapsed,
        lora_filter_report=lora_filter_report,
        trainable_report=trainable_report,
    )
    manifest["training"]["baseline_eval_loss"] = baseline_eval_loss
    manifest["training"]["final_eval_loss"] = final_eval_loss
    manifest["training"]["step_loss_history"] = step_loss_history
    manifest["training"]["eval_history"] = eval_history
    manifest["training"]["checkpoint_history"] = checkpoint_history
    manifest["training"]["sampling_report"] = weighted_sample_report(train_data)
    manifest["training"]["baseline_gate"] = {
        "baseline_eval_before_train": BASELINE_EVAL_BEFORE_TRAIN,
        "baseline_eval_loss": baseline_eval_loss,
        "final_eval_loss": final_eval_loss,
        "require_final_eval_lte_baseline": REQUIRE_FINAL_EVAL_LTE_BASELINE,
        "max_final_eval_regression": MAX_FINAL_EVAL_REGRESSION,
        "abort_eval_relative_to_baseline_delta": ABORT_EVAL_RELATIVE_TO_BASELINE_DELTA,
    }
    manifest_path = final_dir / "v90_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Final adapter saved: {final_dir}")
    print(f"Manifest saved: {manifest_path}")

    upload_outputs(final_dir)


def self_test() -> None:
    print("=== HF JOB TRAIN V90 SELF TEST START ===", flush=True)
    parsed = parse_weight_map("a=2.5,b:0.75")
    assert parsed == {"a": 2.5, "b": 0.75}
    for bad in ["a=0", "a=-1", "a=nan", "a=1,a=2", "missing_value"]:
        try:
            parse_weight_map(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_weight_map should reject {bad!r}")
    assert parse_example_loss_weight({"metadata": {"loss_weight": 2.0}}, require=True) == (2.0, True)
    assert parse_example_loss_weight({"loss_weight": "1.25"}, require=True) == (1.25, True)
    assert parse_example_loss_weight({}, require=False) == (1.0, False)
    for bad_example in [
        {"metadata": {"loss_weight": 0}},
        {"metadata": {"loss_weight": -1}},
        {"metadata": {"loss_weight": "nan"}},
        {"metadata": {"loss_weight": "bad"}},
    ]:
        try:
            parse_example_loss_weight(bad_example, require=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_example_loss_weight should reject {bad_example!r}")
    try:
        parse_example_loss_weight({}, require=True)
    except ValueError:
        pass
    else:
        raise AssertionError("parse_example_loss_weight should reject missing weight when require=True")
    assert target_parameter_name_matches(
        "mlp.experts.gate_up_proj",
        "base_model.model.backbone.layers.0.mixer.experts.7.up_proj.lora_A.default.weight",
    )
    assert target_parameter_name_matches(
        "mlp.experts.down_proj",
        "base_model.model.backbone.layers.0.mixer.experts.7.down_proj.lora_B.default.weight",
    )
    assert not target_parameter_name_matches(
        "mlp.experts.gate_up_proj",
        "base_model.model.backbone.layers.0.mixer.experts.7.down_proj.lora_B.default.weight",
    )
    toy_rows = [
        {"family": "equation_transform", "subcategory": "rule_a", "source": "source_a"},
        {"family": "bit_manipulation", "subcategory": "bit_guardrail_replay", "source": "source_b"},
    ]
    report = weighted_sample_report(toy_rows)
    assert report["mode"] == SAMPLING_MODE
    assert "weighted_share_by_source" in report
    assert "weighted_share_by_subcategory" in report
    print("hf_job_train_v90_self_test=ok", flush=True)
    print("=== HF JOB TRAIN V90 SELF TEST END ===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1:] == ["--self-test"]:
            self_test()
        else:
            raise SystemExit("unknown arguments: " + " ".join(sys.argv[1:]))
    else:
        train()

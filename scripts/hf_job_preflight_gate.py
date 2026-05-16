#!/usr/bin/env python3
"""Cheap preflight gate for Hugging Face GPU jobs.

This script is intentionally separate from notebook_release_gate.py. The
notebook gate validates the launcher before it is pushed; this gate runs inside
the paid HF Job container before expensive model download/load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_PHASES = {"preinstall", "artifacts", "postinstall", "eval-preinstall", "eval-postinstall", "all"}
VALID_SAMPLING_MODES = {"shuffle", "weighted_replacement"}
BLOCKED_DATASET_MARKERS = {
    "v461_synthetic_numeric_probe_pack": (
        "V461 is quarantined: crisis audit found a synthetic prompt/answer seed "
        "matching the full reference set. Rebuild without full-reference seeds."
    ),
    "v463_v462_synthetic_numeric_hard_negative_audit": (
        "V463 is quarantined because it depends on the V461/V462 numeric route."
    ),
    "v464_v463_numeric_multirule_dataset": (
        "V464 is quarantined: crisis audit found rejected_candidate == answer "
        "in 24 train rows and 6 validation rows. Use a later corrected dataset."
    ),
    "v468_v464_symbol_fix_dataset": (
        "V468 is quarantined: crisis audit found it still contains a full-reference "
        "exact prompt/answer seed."
    ),
    "v447_v446_trace_dataset": (
        "Current V447 is quarantined: crisis audit found hypothesis_formed traces "
        "with contradictory boxed answers. Rebuild with rule_found-only traces."
    ),
}


def log_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label} = {json.dumps(payload, sort_keys=True)}", flush=True)


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int = 0) -> int:
    value = env_str(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float = 0.0) -> float:
    value = env_str(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {value!r}") from exc


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, "1" if default else "0").lower()
    return value in {"1", "true", "yes", "y", "on"}


def blocked_dataset_matches(data_identity: str) -> list[dict[str, str]]:
    return [
        {"marker": marker, "reason": reason}
        for marker, reason in BLOCKED_DATASET_MARKERS.items()
        if marker in data_identity
    ]


def require_env(names: list[str]) -> None:
    missing = [name for name in names if not env_str(name)]
    if missing:
        raise RuntimeError("Missing required HF job environment variables: " + ", ".join(missing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not read git HEAD: {type(exc).__name__}: {exc}") from exc


def compile_repo_scripts() -> None:
    targets = [
        Path("scripts/hf_job_train_v90.py"),
        Path("scripts/hf_job_train_v315_preference.py"),
        Path("scripts/hf_job_preflight_gate.py"),
        Path("scripts/kg1_static_safety_gate.py"),
        Path("scripts/kg1_pre_paid_job_integration_gate.py"),
        Path("scripts/build_v243_training_mix.py"),
        Path("scripts/audit_jsonl_overlap.py"),
    ]
    missing = [str(path) for path in targets if not path.exists()]
    if missing:
        raise RuntimeError("Missing required repo scripts: " + ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "py_compile"] + [str(path) for path in targets])
    log_json("py_compile_ok", {"targets": [str(path) for path in targets]})


def torch_status() -> dict[str, Any]:
    import torch

    cuda_runtime = str(getattr(torch.version, "cuda", ""))
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": cuda_runtime,
        "cuda_runtime_major": int(cuda_runtime.split(".", 1)[0]) if re.match(r"^\d+", cuda_runtime) else 0,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(props.name if props else ""),
        "gpu_total_gib": float(props.total_memory / 1024**3 if props else 0.0),
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
    }


def check_torch_and_gpu(stage: str) -> None:
    status = torch_status()
    log_json(f"torch_{stage}", status)

    flavor = env_str("KG1_HF_FLAVOR")
    cuda_major = int(status.get("cuda_runtime_major") or 0)
    max_cuda_major = env_int("KG1_MAX_TORCH_CUDA_MAJOR", 0)
    if max_cuda_major and cuda_major > max_cuda_major:
        raise RuntimeError(
            "torch CUDA runtime is newer than this job allows: "
            f"runtime={status['cuda']} max_major={max_cuda_major} flavor={flavor}"
        )
    if (
        "a100" in flavor.lower()
        and cuda_major >= 13
        and not env_bool("KG1_ALLOW_CUDA13_ON_A100", False)
    ):
        raise RuntimeError(
            "Blocked CUDA 13 runtime on HF A100. Recent HF A100 jobs exposed a CUDA 12.x "
            "driver API, which made torch/vLLM fail after startup. Use H200 for this "
            "vLLM image, or use a CUDA 12-compatible image, or set KG1_ALLOW_CUDA13_ON_A100=1 "
            "only after a fresh driver gate proves compatibility."
        )

    expected_torch = env_str("KG1_EXPECTED_TORCH_VERSION")
    if expected_torch and status["torch"] != expected_torch:
        raise RuntimeError(
            f"torch changed unexpectedly at {stage}: expected {expected_torch}, got {status['torch']}"
        )

    if env_bool("KG1_REQUIRE_CUDA", True) and not status["cuda_available"]:
        raise RuntimeError("CUDA is required for this HF training job.")

    min_gpu_total_gib = env_float("KG1_MIN_GPU_TOTAL_GIB", 0.0)
    if min_gpu_total_gib and status["gpu_total_gib"] < min_gpu_total_gib:
        raise RuntimeError(
            f"GPU memory below floor: {status['gpu_total_gib']:.2f}GiB < {min_gpu_total_gib:.2f}GiB"
        )

    required_gpu_regex = env_str("KG1_REQUIRED_GPU_NAME_REGEX")
    if required_gpu_regex and not re.search(required_gpu_regex, status["gpu_name"], re.IGNORECASE):
        raise RuntimeError(
            f"GPU name mismatch: regex={required_gpu_regex!r}, observed={status['gpu_name']!r}"
        )


def check_hf_flavor_cost() -> None:
    flavor = env_str("KG1_HF_FLAVOR")
    unit_cost = env_float("KG1_HF_UNIT_COST_USD", 0.0)
    max_unit_cost = env_float("KG1_HF_MAX_UNIT_COST_USD", 0.0)
    allowed_flavors = {
        item.strip()
        for item in env_str("KG1_ALLOWED_HF_FLAVORS", "h200,h100,a100-large").split(",")
        if item.strip()
    }
    if flavor and allowed_flavors and flavor not in allowed_flavors:
        raise RuntimeError(f"HF flavor {flavor!r} not allowed; allowed={sorted(allowed_flavors)}")
    if max_unit_cost and unit_cost > max_unit_cost:
        raise RuntimeError(f"HF unit cost too high: {unit_cost} > {max_unit_cost}")
    log_json(
        "hf_flavor_cost_gate",
        {
            "flavor": flavor,
            "unit_cost_usd": unit_cost,
            "max_unit_cost_usd": max_unit_cost,
            "allowed_flavors": sorted(allowed_flavors),
        },
    )


def check_training_env() -> None:
    require_env(
        [
            "KG1_EXPECTED_COMMIT",
            "MODEL_NAME",
            "MODEL_REVISION",
            "DATA_REPO",
            "DATA_FILE",
            "VAL_FILE",
            "EXPECTED_TRAIN_SHA256",
            "EXPECTED_VAL_SHA256",
            "MIN_TRAIN_EXAMPLES",
            "MIN_VAL_EXAMPLES",
            "OUTPUT_REPO",
            "RUN_ID",
            "INIT_ADAPTER_REPO",
            "INIT_ADAPTER_SUBFOLDER",
            "LORA_R",
            "LORA_ALPHA",
            "LORA_TARGET_MODULES",
            "MAX_TRAINABLE_PARAM_RATIO",
            "MAX_LENGTH",
            "BATCH_SIZE",
            "MICRO_BATCH_SIZE",
            "LEARNING_RATE",
            "FINAL_LEARNING_RATE",
            "NUM_EPOCHS",
            "MAX_STEPS",
            "SAVE_EVERY_STEPS",
            "EVAL_EVERY_STEPS",
            "EVAL_MAX_EXAMPLES",
            "SAMPLING_MODE",
            "MAX_PROMPT_TRUNCATION_RATE",
            "REQUIRE_OFFSET_MASK",
        ]
    )

    sampling_mode = env_str("SAMPLING_MODE")
    if sampling_mode not in VALID_SAMPLING_MODES:
        raise RuntimeError(
            f"SAMPLING_MODE must be one of {sorted(VALID_SAMPLING_MODES)}, got {sampling_mode!r}"
        )

    max_steps = env_int("MAX_STEPS")
    expected_max_steps = env_int("KG1_EXPECTED_MAX_STEPS", 0)
    if expected_max_steps and max_steps != expected_max_steps:
        raise RuntimeError(f"MAX_STEPS mismatch: expected {expected_max_steps}, got {max_steps}")
    max_length = env_int("MAX_LENGTH")
    expected_max_length = env_int("KG1_EXPECTED_MAX_LENGTH", 0)
    if expected_max_length and max_length != expected_max_length:
        raise RuntimeError(f"MAX_LENGTH mismatch: expected {expected_max_length}, got {max_length}")

    if env_bool("KG1_REQUIRE_OFFSET_MASK", True) and not env_bool("REQUIRE_OFFSET_MASK", False):
        raise RuntimeError("REQUIRE_OFFSET_MASK must remain enabled for real training.")

    max_prompt_truncation_rate = env_float("MAX_PROMPT_TRUNCATION_RATE", 1.0)
    if max_prompt_truncation_rate > env_float("KG1_MAX_PROMPT_TRUNCATION_RATE", 0.0):
        raise RuntimeError(
            "MAX_PROMPT_TRUNCATION_RATE exceeds hard gate: "
            f"{max_prompt_truncation_rate} > {env_float('KG1_MAX_PROMPT_TRUNCATION_RATE', 0.0)}"
        )

    if env_bool("TOKENIZE_ONLY_DRY_RUN", False):
        raise RuntimeError("TOKENIZE_ONLY_DRY_RUN=1 is not a real HF training job.")
    if env_bool("DRY_RUN_VALIDATE_ONLY", False):
        raise RuntimeError("DRY_RUN_VALIDATE_ONLY=1 is not a real HF training job.")

    if env_bool("UPLOAD_TO_HF", True) and not env_str("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required when UPLOAD_TO_HF=1.")

    summary = {
        "model": env_str("MODEL_NAME"),
        "model_revision": env_str("MODEL_REVISION"),
        "data_repo": env_str("DATA_REPO"),
        "data_file": env_str("DATA_FILE"),
        "val_file": env_str("VAL_FILE"),
        "output_repo": env_str("OUTPUT_REPO"),
        "run_id": env_str("RUN_ID"),
        "max_steps": max_steps,
        "sampling_mode": sampling_mode,
        "max_length": max_length,
        "lora_r": env_int("LORA_R"),
        "lora_alpha": env_int("LORA_ALPHA"),
        "target_modules": env_str("LORA_TARGET_MODULES"),
        "target_parameters": env_str("LORA_TARGET_PARAMETERS"),
        "trainable_lora_modules": env_str("TRAINABLE_LORA_MODULES"),
        "trainable_lora_name_substrings": env_str("TRAINABLE_LORA_NAME_SUBSTRINGS"),
        "require_lora_target_parameter_match": env_bool("REQUIRE_LORA_TARGET_PARAMETER_MATCH", False),
        "required_trainable_lora_name_substrings": env_str(
            "REQUIRED_TRAINABLE_LORA_NAME_SUBSTRINGS"
        ),
    }
    log_json("training_env_gate", summary)

    data_identity = " ".join(
        [
            env_str("DATA_REPO"),
            env_str("DATA_FILE"),
            env_str("VAL_FILE"),
            env_str("EXPECTED_TRAIN_SHA256"),
            env_str("EXPECTED_VAL_SHA256"),
        ]
    )
    blocked = blocked_dataset_matches(data_identity)
    if blocked:
        raise RuntimeError("Blocked quarantined training dataset: " + json.dumps(blocked, sort_keys=True))


def parse_csv_env(name: str) -> list[str]:
    return [item.strip() for item in env_str(name).split(",") if item.strip()]


def check_repo_gate() -> None:
    observed = run_git_head()
    expected = env_str("KG1_EXPECTED_COMMIT")
    log_json("repo_commit_gate", {"observed": observed, "expected": expected})
    if expected and observed != expected:
        raise RuntimeError(f"repo commit mismatch: expected {expected}, got {observed}")
    compile_repo_scripts()


def count_and_audit_jsonl(path: Path, label: str) -> dict[str, Any]:
    rows = 0
    bad_rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    subcategories: dict[str, int] = {}
    gate_row_flags = [
        "gate_rows_used_for_training",
        "weak_gate_rows_used_for_training",
        "full_gate_rows_used_for_training",
    ]
    gate_row_flag_counts: dict[str, int] = {flag: 0 for flag in gate_row_flags}
    gate_row_flag_missing_counts: dict[str, int] = {flag: 0 for flag in gate_row_flags}
    gate_row_flag_bad_rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    duplicate_ids = 0
    assistant_missing = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
            except Exception as exc:
                bad_rows.append({"line": line_no, "error": repr(exc)})
                continue
            row_id = str(row.get("id", ""))
            if row_id:
                if row_id in ids:
                    duplicate_ids += 1
                ids.add(row_id)
            family = str(row.get("family") or row.get("category") or "unknown")
            families[family] = families.get(family, 0) + 1
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            for flag in gate_row_flags:
                if flag not in metadata:
                    gate_row_flag_missing_counts[flag] += 1
                    continue
                if metadata.get(flag) is False:
                    gate_row_flag_counts[flag] += 1
                else:
                    gate_row_flag_bad_rows.append(
                        {
                            "line": line_no,
                            "id": row_id,
                            "flag": flag,
                            "value": metadata.get(flag),
                        }
                    )
            subcategory = str(
                metadata.get("subcategory")
                or metadata.get("subtype")
                or row.get("subcategory")
                or row.get("subtype")
                or "unknown"
            )
            subcategories[subcategory] = subcategories.get(subcategory, 0) + 1
            messages = row.get("messages")
            if not isinstance(messages, list) or not any(
                isinstance(item, dict) and item.get("role") == "assistant" for item in messages
            ):
                assistant_missing += 1
    summary = {
        "label": label,
        "path": str(path),
        "rows": rows,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "assistant_missing": assistant_missing,
        "bad_rows_first10": bad_rows[:10],
        "gate_row_flag_counts": gate_row_flag_counts,
        "gate_row_flag_missing_counts": gate_row_flag_missing_counts,
        "gate_row_flag_bad_rows_first10": gate_row_flag_bad_rows[:10],
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "subcategory_counts_top20": dict(sorted(subcategories.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }
    log_json(f"{label}_jsonl_audit", summary)
    if bad_rows:
        raise RuntimeError(f"{label} JSONL has invalid rows: {bad_rows[:3]}")
    if assistant_missing:
        raise RuntimeError(f"{label} has rows without assistant messages: {assistant_missing}")
    if gate_row_flag_bad_rows:
        raise RuntimeError(
            f"{label} has rows marked as gate/full/weak rows used for training: "
            + json.dumps(gate_row_flag_bad_rows[:10], sort_keys=True)
        )
    return summary


def download_and_check_dataset(
    repo_id: str,
    filename: str,
    expected_sha: str,
    min_rows: int,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=env_str("HF_TOKEN") or None))
    observed_sha = sha256_file(path)
    log_json(
        f"{label}_dataset_file",
        {"repo_id": repo_id, "filename": filename, "path": str(path), "sha256": observed_sha, "expected_sha256": expected_sha},
    )
    if expected_sha and observed_sha.lower() != expected_sha.lower():
        raise RuntimeError(f"{label} dataset sha mismatch: observed={observed_sha} expected={expected_sha}")
    summary = count_and_audit_jsonl(path, label)
    if summary["rows"] < min_rows:
        raise RuntimeError(f"{label} row count below floor: {summary['rows']} < {min_rows}")
    return path, summary


def check_required_counts(summary: dict[str, Any], env_name: str, count_key: str) -> None:
    raw = env_str(env_name)
    if not raw:
        return
    observed = summary[count_key]
    missing: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in observed:
            missing.append(item)
    if missing:
        raise RuntimeError(f"{env_name} missing from {summary['label']}: {missing}")


def check_hub_artifacts() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = env_str("HF_TOKEN") or None
    api = HfApi(token=token)
    model_info = api.model_info(env_str("MODEL_NAME"), revision=env_str("MODEL_REVISION"))
    log_json(
        "model_revision_gate",
        {"model": env_str("MODEL_NAME"), "revision": env_str("MODEL_REVISION"), "sha": getattr(model_info, "sha", "")},
    )

    _train_path, train_summary = download_and_check_dataset(
        env_str("DATA_REPO"),
        env_str("DATA_FILE"),
        env_str("EXPECTED_TRAIN_SHA256"),
        env_int("MIN_TRAIN_EXAMPLES"),
        "train",
    )
    _val_path, val_summary = download_and_check_dataset(
        env_str("DATA_REPO"),
        env_str("VAL_FILE"),
        env_str("EXPECTED_VAL_SHA256"),
        env_int("MIN_VAL_EXAMPLES"),
        "validation",
    )
    check_required_counts(train_summary, "KG1_REQUIRED_TRAIN_FAMILIES", "family_counts")
    check_required_counts(train_summary, "KG1_REQUIRED_TRAIN_SUBCATEGORIES", "subcategory_counts")
    check_required_counts(val_summary, "KG1_REQUIRED_VAL_FAMILIES", "family_counts")
    check_required_counts(val_summary, "KG1_REQUIRED_VAL_SUBCATEGORIES", "subcategory_counts")

    adapter_repo = env_str("INIT_ADAPTER_REPO")
    adapter_subfolder = env_str("INIT_ADAPTER_SUBFOLDER").strip("/")
    files = set(api.list_repo_files(adapter_repo, repo_type="model"))
    adapter_config_name = f"{adapter_subfolder}/adapter_config.json" if adapter_subfolder else "adapter_config.json"
    adapter_weights_name = f"{adapter_subfolder}/adapter_model.safetensors" if adapter_subfolder else "adapter_model.safetensors"
    missing = [name for name in [adapter_config_name, adapter_weights_name] if name not in files]
    if missing:
        raise RuntimeError(f"Missing init adapter files in {adapter_repo}: {missing}")
    config_path = Path(
        hf_hub_download(
            repo_id=adapter_repo,
            filename="adapter_config.json",
            subfolder=adapter_subfolder or None,
            repo_type="model",
            token=token,
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = sorted(str(item) for item in (config.get("target_modules") or []))
    target_parameters = sorted(str(item) for item in (config.get("target_parameters") or []))
    modules_to_save = sorted(str(item) for item in (config.get("modules_to_save") or []))
    log_json(
        "init_adapter_gate",
        {
            "repo": adapter_repo,
            "subfolder": adapter_subfolder,
            "adapter_config": adapter_config_name,
            "adapter_weights": adapter_weights_name,
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": target_modules,
            "target_parameters": target_parameters,
            "modules_to_save": modules_to_save,
        },
    )
    if env_bool("KG1_STRICT_INIT_ADAPTER_CONFIG", False):
        if modules_to_save:
            raise RuntimeError(
                "Init adapter modules_to_save must be empty for KG1 adapter-only submit path: "
                + json.dumps(modules_to_save, sort_keys=True)
            )
        if int(config.get("r", -1)) != env_int("LORA_R"):
            raise RuntimeError(f"Init adapter r mismatch: {config.get('r')} != LORA_R={env_int('LORA_R')}")
        if int(config.get("lora_alpha", -1)) != env_int("LORA_ALPHA"):
            raise RuntimeError(
                f"Init adapter alpha mismatch: {config.get('lora_alpha')} != LORA_ALPHA={env_int('LORA_ALPHA')}"
            )
        configured_target_modules = sorted(parse_csv_env("LORA_TARGET_MODULES"))
        if target_modules and configured_target_modules != target_modules:
            raise RuntimeError(
                "Init adapter target_modules mismatch: "
                + json.dumps(
                    {"adapter": target_modules, "env": configured_target_modules},
                    sort_keys=True,
                )
            )
        configured_target_parameters = sorted(parse_csv_env("LORA_TARGET_PARAMETERS"))
        if configured_target_parameters != target_parameters:
            raise RuntimeError(
                "Init adapter target_parameters mismatch: "
                + json.dumps(
                    {"adapter": target_parameters, "env": configured_target_parameters},
                    sort_keys=True,
                )
            )
        if target_parameters and not env_bool("REQUIRE_LORA_TARGET_PARAMETER_MATCH", False):
            raise RuntimeError(
                "Init adapter has target_parameters but REQUIRE_LORA_TARGET_PARAMETER_MATCH is disabled."
            )
        if target_parameters and env_str("INIT_ADAPTER_LOAD_MODE", "peft").strip().lower() == "manual":
            raise RuntimeError(
                "INIT_ADAPTER_LOAD_MODE=manual is blocked for adapters with target_parameters. "
                "Use the PEFT-native PeftModel.from_pretrained path or run a dedicated CPU "
                "round-trip equivalence gate before allowing manual state_dict injection."
            )


def check_postinstall_imports() -> None:
    import importlib

    required = [
        "huggingface_hub",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
    ]
    if env_bool("KG1_REQUIRE_MAMBA_IMPORTS", True):
        required.extend(
            [
                "causal_conv1d",
                "mamba_ssm",
                "mamba_ssm.ops.triton.layernorm_gated",
                "mamba_ssm.ops.selective_scan_interface",
            ]
        )
    for module_name in required:
        module = importlib.import_module(module_name)
        log_json(
            "postinstall_import_ok",
            {
                "module": module_name,
                "version": str(getattr(module, "__version__", "unknown")),
            },
        )


def check_eval_postinstall_imports() -> None:
    import importlib

    for module_name in ["huggingface_hub", "pandas", "packaging", "safetensors", "vllm"]:
        module = importlib.import_module(module_name)
        log_json(
            "eval_postinstall_import_ok",
            {
                "module": module_name,
                "version": str(getattr(module, "__version__", "unknown")),
            },
        )


def run_phase(phase: str) -> None:
    if phase == "preinstall":
        check_repo_gate()
        check_torch_and_gpu("preinstall")
        check_hf_flavor_cost()
        check_training_env()
        return
    if phase == "artifacts":
        check_training_env()
        check_hub_artifacts()
        return
    if phase == "postinstall":
        check_torch_and_gpu("postinstall")
        check_postinstall_imports()
        check_training_env()
        return
    if phase == "eval-preinstall":
        check_repo_gate()
        check_torch_and_gpu("eval_preinstall")
        check_hf_flavor_cost()
        return
    if phase == "eval-postinstall":
        check_torch_and_gpu("eval_postinstall")
        check_eval_postinstall_imports()
        check_hf_flavor_cost()
        return
    if phase == "all":
        run_phase("preinstall")
        run_phase("artifacts")
        run_phase("postinstall")
        return
    raise RuntimeError(f"Unknown phase: {phase}")


def self_test() -> None:
    assert VALID_SAMPLING_MODES == {"shuffle", "weighted_replacement"}
    os.environ.setdefault("SAMPLING_MODE", "weighted_replacement")
    if env_str("SAMPLING_MODE") not in VALID_SAMPLING_MODES:
        raise RuntimeError("self-test failed")
    old_max_length = os.environ.get("MAX_LENGTH")
    old_expected_max_length = os.environ.get("KG1_EXPECTED_MAX_LENGTH")
    os.environ["MAX_LENGTH"] = "8192"
    os.environ["KG1_EXPECTED_MAX_LENGTH"] = "8192"
    if env_int("MAX_LENGTH") != env_int("KG1_EXPECTED_MAX_LENGTH"):
        raise RuntimeError("self-test max length equality failed")
    os.environ["KG1_EXPECTED_MAX_LENGTH"] = "4096"
    try:
        if env_int("MAX_LENGTH") == env_int("KG1_EXPECTED_MAX_LENGTH"):
            raise RuntimeError("self-test expected max length mismatch")
    finally:
        if old_max_length is None:
            os.environ.pop("MAX_LENGTH", None)
        else:
            os.environ["MAX_LENGTH"] = old_max_length
        if old_expected_max_length is None:
            os.environ.pop("KG1_EXPECTED_MAX_LENGTH", None)
        else:
            os.environ["KG1_EXPECTED_MAX_LENGTH"] = old_expected_max_length
    assert blocked_dataset_matches("repo data/v464_v463_numeric_multirule_dataset/train.jsonl")
    assert blocked_dataset_matches("repo data/v468_v464_symbol_fix_dataset/train.jsonl")
    assert blocked_dataset_matches("repo data/v447_v446_trace_dataset/train.jsonl")
    assert not blocked_dataset_matches("repo data/v469_symbol_fix_rebuilt_clean/train.jsonl")
    print("hf_job_preflight_gate_self_test=ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(VALID_PHASES), default="preinstall")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    print("=== HF JOB PREFLIGHT GATE START ===", flush=True)
    print("generated_at_utc =", datetime.now(timezone.utc).isoformat(), flush=True)
    print("phase =", args.phase, flush=True)
    if args.self_test:
        self_test()
    else:
        run_phase(args.phase)
    print("=== HF JOB PREFLIGHT GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

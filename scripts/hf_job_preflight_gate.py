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


VALID_PHASES = {"preinstall", "artifacts", "postinstall", "all"}
VALID_SAMPLING_MODES = {"shuffle", "weighted_replacement"}


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
        Path("scripts/hf_job_preflight_gate.py"),
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

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(torch.version, "cuda", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(props.name if props else ""),
        "gpu_total_gib": float(props.total_memory / 1024**3 if props else 0.0),
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
    }


def check_torch_and_gpu(stage: str) -> None:
    status = torch_status()
    log_json(f"torch_{stage}", status)

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
        "max_length": env_int("MAX_LENGTH"),
        "lora_r": env_int("LORA_R"),
        "lora_alpha": env_int("LORA_ALPHA"),
        "target_modules": env_str("LORA_TARGET_MODULES"),
        "trainable_lora_modules": env_str("TRAINABLE_LORA_MODULES"),
    }
    log_json("training_env_gate", summary)


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
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "subcategory_counts_top20": dict(sorted(subcategories.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }
    log_json(f"{label}_jsonl_audit", summary)
    if bad_rows:
        raise RuntimeError(f"{label} JSONL has invalid rows: {bad_rows[:3]}")
    if assistant_missing:
        raise RuntimeError(f"{label} has rows without assistant messages: {assistant_missing}")
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
            "target_parameters": config.get("target_parameters"),
        },
    )
    if env_bool("KG1_STRICT_INIT_ADAPTER_CONFIG", False):
        if int(config.get("r", -1)) != env_int("LORA_R"):
            raise RuntimeError(f"Init adapter r mismatch: {config.get('r')} != LORA_R={env_int('LORA_R')}")
        if int(config.get("lora_alpha", -1)) != env_int("LORA_ALPHA"):
            raise RuntimeError(
                f"Init adapter alpha mismatch: {config.get('lora_alpha')} != LORA_ALPHA={env_int('LORA_ALPHA')}"
            )


def check_postinstall_imports() -> None:
    import importlib

    required = [
        "huggingface_hub",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "causal_conv1d",
        "mamba_ssm",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.selective_scan_interface",
    ]
    for module_name in required:
        module = importlib.import_module(module_name)
        log_json(
            "postinstall_import_ok",
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

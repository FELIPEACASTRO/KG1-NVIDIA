#!/usr/bin/env python3
"""Colab-safe launcher for V1243 KG1 bit/equation specialist jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "v1243_solver_to_lora_graft"
ENV_PREVIEW = DEFAULT_ARTIFACT_DIR / "v1243_hf_env_preview.json"
SAFE_PHASES = {"bit_specialist", "equation_specialist"}
RUN_MODES = {"tokenize_dryrun", "model_dryrun", "real_train"}
GPU_RUN_MODES = {"model_dryrun", "real_train"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,180}$")
INT_ENV_OVERRIDES = {
    "KG1_V1243_OVERRIDE_MAX_STEPS": "MAX_STEPS",
    "KG1_V1243_OVERRIDE_SAVE_EVERY_STEPS": "SAVE_EVERY_STEPS",
    "KG1_V1243_OVERRIDE_EVAL_EVERY_STEPS": "EVAL_EVERY_STEPS",
    "KG1_V1243_OVERRIDE_LOG_EVERY_STEPS": "LOG_EVERY_STEPS",
    "KG1_V1243_OVERRIDE_EVAL_MAX_EXAMPLES": "EVAL_MAX_EXAMPLES",
    "KG1_V1243_OVERRIDE_SCORE_PROXY_EVAL_MAX_EXAMPLES": "SCORE_PROXY_EVAL_MAX_EXAMPLES",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(SAFE_PHASES), default="bit_specialist")
    parser.add_argument("--run-mode", choices=sorted(RUN_MODES), default="tokenize_dryrun")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--env-preview", type=Path, default=ENV_PREVIEW)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--target-accuracy", type=float, default=float(os.environ.get("KG1_TARGET_ACCURACY", "0.89")))
    parser.add_argument("--live-log-repo", default=os.environ.get("KG1_LIVE_LOG_HF_REPO", ""))
    parser.add_argument("--live-log-repo-type", default=os.environ.get("KG1_LIVE_LOG_HF_REPO_TYPE", "dataset"))
    parser.add_argument("--upload-every", type=float, default=float(os.environ.get("KG1_LIVE_LOG_UPLOAD_EVERY", "60")))
    parser.add_argument("--min-gpu-total-gib", type=float, default=float(os.environ.get("KG1_MIN_GPU_TOTAL_GIB", "70")))
    parser.add_argument("--min-gpu-free-gib", type=float, default=float(os.environ.get("KG1_MIN_GPU_FREE_GIB", "60")))
    parser.add_argument("--min-disk-free-gib", type=float, default=float(os.environ.get("KG1_MIN_CONTENT_FREE_GIB", "35")))
    parser.add_argument("--accept-gpu-spend", action="store_true", default=os.environ.get("KG1_ACCEPT_GPU_SPEND", "0") == "1")
    parser.add_argument(
        "--require-live-log-upload",
        action="store_true",
        default=os.environ.get("KG1_REQUIRE_LIVE_LOG_UPLOAD", "0") == "1",
    )
    parser.add_argument("--init-adapter-dir", default=os.environ.get("INIT_ADAPTER_DIR", ""))
    parser.add_argument("--init-adapter-repo", default=os.environ.get("INIT_ADAPTER_REPO", ""))
    parser.add_argument("--init-adapter-revision", default=os.environ.get("INIT_ADAPTER_REVISION", ""))
    parser.add_argument("--init-adapter-subfolder", default=os.environ.get("INIT_ADAPTER_SUBFOLDER", ""))
    parser.add_argument("--require-init-adapter", action="store_true", default=os.environ.get("REQUIRE_INIT_ADAPTER", "0") == "1")
    parser.add_argument(
        "--require-init-adapter-revision",
        action="store_true",
        default=os.environ.get("REQUIRE_INIT_ADAPTER_REVISION", "0") == "1",
    )
    parser.add_argument("--output-repo", default=os.environ.get("OUTPUT_REPO", ""))
    parser.add_argument("--allow-real-train", action="store_true")
    parser.add_argument("--print-env-only", action="store_true")
    return parser


def apply_safety_defaults(args: argparse.Namespace) -> None:
    if args.run_mode in GPU_RUN_MODES:
        if not args.require_live_log_upload:
            print(
                "KG1_V1243_SAFETY_DEFAULT forcing_require_live_log_upload=True "
                f"run_mode={args.run_mode}",
                flush=True,
            )
        args.require_live_log_upload = True


def build_default_run_id(args: argparse.Namespace) -> str:
    return (
        f"v1243_{args.phase}_{args.run_mode}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_p{os.getpid()}_{uuid.uuid4().hex[:6]}"
    )


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "RUN_ID must be a safe HF/path slug matching "
            "[A-Za-z0-9][A-Za-z0-9_.-]{2,180}; got "
            + repr(value)
        )
    return value


def load_env_preview(path: Path, phase: str) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload.get(phase)
    if not isinstance(env, dict):
        raise KeyError(f"phase not found in env preview: {phase}")
    return {str(key): str(value) for key, value in env.items()}


def colab_default_output_dir(run_id: str) -> str:
    if Path("/content").exists():
        return f"/content/kg1_outputs/{run_id}"
    return str(ROOT / "artifacts" / "v1243_colab_local_outputs" / run_id)


def localize_dataset_path(artifact_dir: Path, value: str) -> str:
    name = Path(value.replace("\\", "/")).name
    path = artifact_dir / name
    if not path.exists():
        raise FileNotFoundError(f"required V1243 artifact missing: {path}")
    return str(path)


def apply_run_mode(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.run_mode == "tokenize_dryrun":
        env["DRY_RUN_VALIDATE_ONLY"] = "1"
        env["TOKENIZE_ONLY_DRY_RUN"] = "1"
        env["UPLOAD_TO_HF"] = "0"
        env["OUTPUT_REPO"] = args.output_repo or "disabled-by-tokenize-dryrun"
    elif args.run_mode == "model_dryrun":
        env["DRY_RUN_VALIDATE_ONLY"] = "1"
        env["TOKENIZE_ONLY_DRY_RUN"] = "0"
        env["UPLOAD_TO_HF"] = "0"
        env["OUTPUT_REPO"] = args.output_repo or "disabled-by-model-dryrun"
    elif args.run_mode == "real_train":
        if not args.allow_real_train:
            raise RuntimeError("real_train requires --allow-real-train")
        if not args.output_repo:
            raise RuntimeError("real_train requires --output-repo so final adapter is not stranded in Colab")
        env["DRY_RUN_VALIDATE_ONLY"] = "0"
        env["TOKENIZE_ONLY_DRY_RUN"] = "0"
        env["UPLOAD_TO_HF"] = "1"
        env["OUTPUT_REPO"] = args.output_repo
    else:
        raise ValueError(f"unknown run mode: {args.run_mode}")


def apply_safe_int_overrides(env: dict[str, str]) -> dict[str, str]:
    """Apply allowlisted integer env overrides after env-preview load.

    The env preview carries production defaults. For a paid smoke run we need
    MAX_STEPS=2 to survive those defaults, while still rejecting arbitrary env
    injection into the trainer configuration.
    """
    applied: dict[str, str] = {}
    for source_key, target_key in INT_ENV_OVERRIDES.items():
        raw = os.environ.get(source_key)
        if raw in (None, ""):
            continue
        try:
            value = int(str(raw))
        except ValueError as exc:
            raise ValueError(f"{source_key} must be an integer; got {raw!r}") from exc
        if value < 0:
            raise ValueError(f"{source_key} must be non-negative; got {value}")
        env[target_key] = str(value)
        applied[target_key] = str(value)
    return applied


def build_env(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    artifact_dir = args.artifact_dir.resolve()
    env = load_env_preview(args.env_preview.resolve(), args.phase)
    run_id = validate_run_id(args.run_id or os.environ.get("RUN_ID") or build_default_run_id(args))

    env["RUN_ID"] = run_id
    env["DATA_FILE"] = localize_dataset_path(artifact_dir, env["DATA_FILE"])
    env["VAL_FILE"] = localize_dataset_path(artifact_dir, env["VAL_FILE"])
    env["DATA_REPO"] = "local-v1243-colab-launch"
    env["OUTPUT_DIR"] = colab_default_output_dir(run_id)
    env["COMPUTE_PROVIDER"] = "colab_pro_v1243_realtime"
    env["SCORE_CONTRACT_TARGET_ACCURACY"] = str(args.target_accuracy)
    env["FRIENDLY_REALTIME_LOGS"] = "1"
    env["FRIENDLY_LOG_SCORE_HINTS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["KG1_LIVE_LOG_UPLOAD_EVERY"] = str(args.upload_every)
    env["KG1_REQUIRE_LIVE_LOG_UPLOAD"] = "1" if args.require_live_log_upload else "0"
    env["KG1_ACCEPT_GPU_SPEND"] = "1" if args.accept_gpu_spend else "0"
    if args.run_mode in GPU_RUN_MODES:
        env["REQUIRE_REAL_CAUSAL_CONV1D"] = "1"

    if args.live_log_repo:
        env["KG1_LIVE_LOG_HF_REPO"] = args.live_log_repo
        env["KG1_LIVE_LOG_HF_REPO_TYPE"] = args.live_log_repo_type
        env["KG1_LIVE_LOG_HF_PATH"] = f"colab/{run_id}/train.log"
        env["KG1_LIVE_STATUS_HF_PATH"] = f"colab/{run_id}/status.json"

    if args.init_adapter_dir:
        env["INIT_ADAPTER_DIR"] = args.init_adapter_dir
    if args.init_adapter_repo:
        env["INIT_ADAPTER_REPO"] = args.init_adapter_repo
    if args.init_adapter_revision:
        env["INIT_ADAPTER_REVISION"] = args.init_adapter_revision
    if args.init_adapter_subfolder:
        env["INIT_ADAPTER_SUBFOLDER"] = args.init_adapter_subfolder
    if args.require_init_adapter:
        env["REQUIRE_INIT_ADAPTER"] = "1"
    if args.require_init_adapter_revision:
        env["REQUIRE_INIT_ADAPTER_REVISION"] = "1"

    apply_run_mode(env, args)
    applied_overrides = apply_safe_int_overrides(env)
    if applied_overrides:
        env["KG1_V1243_APPLIED_INT_OVERRIDES_JSON"] = json.dumps(applied_overrides, sort_keys=True)
    return env, run_id


def is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def adapter_weight_exists(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_initial_adapter_contract(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.run_mode == "tokenize_dryrun":
        return
    if not args.accept_gpu_spend:
        return
    require_init = args.require_init_adapter or is_truthy(env.get("REQUIRE_INIT_ADAPTER"))
    if not require_init:
        return
    init_dir = env.get("INIT_ADAPTER_DIR", "")
    init_repo = env.get("INIT_ADAPTER_REPO", "")
    init_revision = env.get("INIT_ADAPTER_REVISION", "")
    if not init_dir and not init_repo:
        raise RuntimeError(
            "V1243 model_dryrun/real_train requires a baseline initial adapter. "
            "Set INIT_ADAPTER_DIR or INIT_ADAPTER_REPO before spending GPU."
        )
    if init_dir and init_repo:
        raise RuntimeError("Set exactly one initial adapter source: INIT_ADAPTER_DIR or INIT_ADAPTER_REPO, not both")
    if init_dir:
        adapter_dir = Path(init_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"INIT_ADAPTER_DIR not found: {adapter_dir}")
        if not (adapter_dir / "adapter_config.json").exists() or not adapter_weight_exists(adapter_dir):
            raise RuntimeError(f"INIT_ADAPTER_DIR is not a complete PEFT adapter directory: {adapter_dir}")
        expected_config_sha = env.get("EXPECTED_INIT_ADAPTER_CONFIG_SHA256", "")
        expected_weights_sha = env.get("EXPECTED_INIT_ADAPTER_WEIGHTS_SHA256", "")
        config_path = adapter_dir / "adapter_config.json"
        weights_path = adapter_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            weights_path = adapter_dir / "adapter_model.bin"
        for label, path, expected in [
            ("adapter_config", config_path, expected_config_sha),
            ("adapter_weights", weights_path, expected_weights_sha),
        ]:
            observed = sha256_file(path)
            print(f"KG1_V1243_INIT_ADAPTER_SHA label={label} sha256={observed}", flush=True)
            if expected and observed.lower() != expected.lower():
                raise RuntimeError(f"{label} sha256 mismatch: {observed} != {expected}")
    if init_repo and (args.require_init_adapter_revision or is_truthy(env.get("REQUIRE_INIT_ADAPTER_REVISION"))) and not init_revision:
        raise RuntimeError("INIT_ADAPTER_REPO requires INIT_ADAPTER_REVISION when REQUIRE_INIT_ADAPTER_REVISION=1")


def runtime_preflight(args: argparse.Namespace) -> None:
    disk_root = Path("/content") if Path("/content").exists() else ROOT
    total, used, free = shutil.disk_usage(disk_root)
    free_gib = free / (1024 ** 3)
    report: dict[str, Any] = {
        "run_mode": args.run_mode,
        "disk_root": str(disk_root),
        "disk_free_gib": round(free_gib, 3),
        "min_disk_free_gib": args.min_disk_free_gib,
        "accept_gpu_spend": bool(args.accept_gpu_spend),
        "require_live_log_upload": bool(args.require_live_log_upload),
    }
    if free_gib < args.min_disk_free_gib:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(f"Not enough free disk: {free_gib:.2f} GiB < {args.min_disk_free_gib:.2f} GiB")

    if args.run_mode == "tokenize_dryrun":
        report["status"] = "PASS"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        return

    if not args.accept_gpu_spend:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(
            f"{args.run_mode} is GPU-spend mode. Set KG1_ACCEPT_GPU_SPEND=1 only after tokenization gates pass."
        )

    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        gpu_total_gib = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if cuda_available
            else 0.0
        )
        gpu_free_gib = 0.0
        gpu_reserved_gib = 0.0
        gpu_allocated_gib = 0.0
        if cuda_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_free_gib = free_bytes / (1024 ** 3)
            gpu_total_gib = total_bytes / (1024 ** 3)
            gpu_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception as exc:
        report["status"] = "FAIL"
        report["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(f"Could not probe CUDA for {args.run_mode}: {exc}") from exc

    report["cuda_available"] = cuda_available
    report["gpu_total_gib"] = round(gpu_total_gib, 3)
    report["gpu_free_gib"] = round(gpu_free_gib, 3)
    report["gpu_reserved_gib"] = round(gpu_reserved_gib, 3)
    report["gpu_allocated_gib"] = round(gpu_allocated_gib, 3)
    report["min_gpu_total_gib"] = args.min_gpu_total_gib
    report["min_gpu_free_gib"] = args.min_gpu_free_gib
    if not cuda_available:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(f"CUDA GPU is required for {args.run_mode}")
    if gpu_total_gib < args.min_gpu_total_gib:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(
            f"GPU memory too small for safe Nemotron {args.run_mode}: "
            f"{gpu_total_gib:.2f} GiB < {args.min_gpu_total_gib:.2f} GiB"
        )
    if gpu_free_gib < args.min_gpu_free_gib:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(
            f"GPU free memory too low for safe Nemotron {args.run_mode}: "
            f"{gpu_free_gib:.2f} GiB < {args.min_gpu_free_gib:.2f} GiB"
        )
    try:
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn  # noqa: F401
        report["mamba_ssm_rmsnorm"] = True
    except Exception as exc:
        report["status"] = "FAIL"
        report["mamba_ssm_rmsnorm"] = False
        report["mamba_ssm_error"] = f"{type(exc).__name__}: {exc}"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(
            "mamba_ssm is required before Nemotron model load. "
            "Install mamba-ssm==2.3.1 with --no-build-isolation, then retry."
        ) from exc
    report["causal_conv1d"] = importlib.util.find_spec("causal_conv1d") is not None
    if not report["causal_conv1d"]:
        report["status"] = "FAIL"
        print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)
        raise RuntimeError(
            "causal_conv1d real package is required before Nemotron model load. "
            "Install causal-conv1d before model_dryrun/real_train; do not rely on a stub."
        )
    report["status"] = "PASS"
    print("KG1_V1243_PREFLIGHT_JSON", json.dumps(report, sort_keys=True), flush=True)


def print_env_summary(env: dict[str, str], run_id: str, args: argparse.Namespace) -> None:
    safe_keys = [
        "RUN_ID",
        "COMPUTE_PROVIDER",
        "DATA_FILE",
        "VAL_FILE",
        "OUTPUT_DIR",
        "OUTPUT_REPO",
        "DRY_RUN_VALIDATE_ONLY",
        "TOKENIZE_ONLY_DRY_RUN",
        "UPLOAD_TO_HF",
        "MAX_STEPS",
        "SAVE_EVERY_STEPS",
        "EVAL_EVERY_STEPS",
        "LOG_EVERY_STEPS",
        "EVAL_MAX_EXAMPLES",
        "SCORE_CONTRACT_TARGET_ACCURACY",
        "SCORE_PROXY_EVAL_MAX_EXAMPLES",
        "SCORE_TRAJECTORY_CHECK",
        "REQUIRE_SCORE_TRAJECTORY_PASS",
        "KG1_WATCHDOG_STALE_SECONDS",
        "KG1_WATCHDOG_MAX_RUNTIME_SECONDS",
        "KG1_DISABLE_HEALTH_WATCHDOG",
        "KG1_REQUIRE_LIVE_LOG_UPLOAD",
        "FRIENDLY_REALTIME_LOGS",
        "KG1_LIVE_LOG_HF_REPO",
        "KG1_LIVE_LOG_HF_PATH",
        "REQUIRE_INIT_ADAPTER",
        "INIT_ADAPTER_DIR",
        "INIT_ADAPTER_REPO",
        "INIT_ADAPTER_REVISION",
        "REQUIRE_REAL_CAUSAL_CONV1D",
        "KG1_V1243_APPLIED_INT_OVERRIDES_JSON",
    ]
    print("KG1_V1243_COLAB_LAUNCH_ENV_BEGIN", flush=True)
    print(json.dumps({key: env.get(key, "") for key in safe_keys}, indent=2, sort_keys=True), flush=True)
    print("KG1_V1243_COLAB_LAUNCH_ENV_END", flush=True)
    print(
        "KG1_V1243_COLAB_LAUNCH_DECISION "
        f"phase={args.phase} run_mode={args.run_mode} run_id={run_id} "
        f"target_accuracy={args.target_accuracy}",
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    apply_safety_defaults(args)
    env_updates, run_id = build_env(args)
    print_env_summary(env_updates, run_id, args)
    if args.print_env_only:
        return 0

    child_env = os.environ.copy()
    child_env.update(env_updates)
    validate_initial_adapter_contract(child_env, args)
    runtime_preflight(args)
    if args.require_live_log_upload:
        token = child_env.get("HF_TOKEN") or child_env.get("HUGGINGFACE_HUB_TOKEN")
        if not args.live_log_repo:
            raise RuntimeError("live-log upload is required, but --live-log-repo is empty")
        if not token:
            raise RuntimeError("live-log upload is required, but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is missing")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "kg1_colab_realtime_runner.py"),
        "--upload-every",
        str(args.upload_every),
    ]
    if args.require_live_log_upload:
        command.append("--require-upload")
    if args.live_log_repo:
        command.extend(
            [
                "--hf-repo",
                args.live_log_repo,
                "--hf-path",
                child_env["KG1_LIVE_LOG_HF_PATH"],
                "--hf-status-path",
                child_env["KG1_LIVE_STATUS_HF_PATH"],
                "--hf-repo-type",
                args.live_log_repo_type,
            ]
        )
    else:
        command.append("--no-upload")
    command.extend(["--", sys.executable, str(ROOT / "scripts" / "hf_job_train_v90.py")])
    print("KG1_V1243_COLAB_COMMAND", " ".join(command), flush=True)
    return subprocess.call(command, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())

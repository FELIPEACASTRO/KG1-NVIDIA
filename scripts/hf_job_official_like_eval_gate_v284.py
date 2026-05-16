#!/usr/bin/env python3
"""Run an official-like guarded LoRA eval inside Hugging Face Jobs.

This runner is intentionally narrow for V284: it validates the 947-row
validation bridge before model load, evaluates one or more adapters with the
Kaggle Overview inference settings, and uploads outputs back to the HF model
repository. It never applies an external postprocessor, packages, or submits to
Kaggle. This is the submission-readiness gate after cheap weak triage, not a
cheap diagnostic shortcut.
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import canonical_family, classify_puzzle  # noqa: E402


EXPECTED_FULL_COUNTS = {
    "bit_manipulation": 160,
    "equation_transform": 155,
    "gravity_constant": 159,
    "numeral_system": 157,
    "text_encryption": 157,
    "unit_conversion": 159,
}
EXPECTED_FULL_ROWS = 947
EXPECTED_FULL_CSV_SHA256 = "84e90b5b4d9adad6fdd9028aae3161d1b8991f2eab11e292b32d920c0ec3c935"
DEFAULT_DATA_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_FULL_CSV_FILE = (
    "runtime_artifacts/v276_full_eval_bridge/"
    "v276-full947-bridge-20260511T1245Z/official_train_seed42_stratified10_val.csv"
)
DEFAULT_FULL_MANIFEST_FILE = (
    "runtime_artifacts/v276_full_eval_bridge/"
    "v276-full947-bridge-20260511T1245Z/v276_full947_validation_manifest.json"
)
DEFAULT_ADAPTER_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_ADAPTER_SUBFOLDER = "checkpoint-4"
DEFAULT_OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
DEFAULT_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
DEFAULT_POSTPROCESSOR = "none"
OFFICIAL_MAX_TOKENS = 7680
OFFICIAL_MAX_MODEL_LEN = 8192
OFFICIAL_MAX_NUM_SEQS = 64
OFFICIAL_GPU_MEMORY_UTILIZATION = 0.85


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int) -> int:
    raw = env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    raw = env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw!r}") from exc


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def log_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label} = {json.dumps(payload, indent=2, sort_keys=True)}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_status() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(torch.version, "cuda", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(props.name if props else ""),
        "gpu_total_gib": float(props.total_memory / 1024**3 if props else 0.0),
    }


def validate_gpu() -> None:
    status = torch_status()
    log_json("torch_gpu_status", status)
    cuda_runtime = str(status.get("cuda") or "")
    cuda_major = int(cuda_runtime.split(".", 1)[0]) if cuda_runtime[:1].isdigit() else 0
    flavor = env_str("KG1_HF_FLAVOR")
    if (
        "a100" in flavor.lower()
        and cuda_major >= 13
        and not env_bool("KG1_ALLOW_CUDA13_ON_A100", False)
    ):
        raise RuntimeError(
            "Blocked CUDA 13 runtime on HF A100 for official-like vLLM eval. "
            "Use H200 for this vLLM image or a CUDA 12-compatible image."
        )
    if env_bool("KG1_REQUIRE_CUDA", True) and not status["cuda_available"]:
        raise RuntimeError("CUDA is required for full vLLM eval.")
    min_gib = env_float("KG1_MIN_GPU_TOTAL_GIB", 79.0)
    if status["gpu_total_gib"] < min_gib:
        raise RuntimeError(f"GPU memory below required floor: {status['gpu_total_gib']:.2f} < {min_gib:.2f}")
    required_regex = env_str("KG1_REQUIRED_GPU_NAME_REGEX")
    if required_regex and not re.search(required_regex, status["gpu_name"], re.IGNORECASE):
        raise RuntimeError(f"GPU name {status['gpu_name']!r} does not match {required_regex!r}")


def validate_hf_flavor_cost() -> None:
    flavor = env_str("KG1_HF_FLAVOR")
    allowed = {part.strip() for part in env_str("KG1_ALLOWED_HF_FLAVORS", "h200,h100,a100-large").split(",") if part.strip()}
    if flavor and allowed and flavor not in allowed:
        raise RuntimeError(f"HF flavor {flavor!r} not allowed; allowed={sorted(allowed)}")
    unit_cost = env_float("KG1_HF_UNIT_COST_USD", 0.0)
    max_unit_cost = env_float("KG1_HF_MAX_UNIT_COST_USD", 0.09)
    if unit_cost and max_unit_cost and unit_cost > max_unit_cost:
        raise RuntimeError(f"HF unit cost too high: {unit_cost} > {max_unit_cost}")
    log_json(
        "hf_flavor_cost_gate",
        {"flavor": flavor, "allowed_flavors": sorted(allowed), "unit_cost_usd": unit_cost, "max_unit_cost_usd": max_unit_cost},
    )


def validate_official_like_controls(
    *,
    max_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    disable_thinking: bool,
    no_prompt_suffix: bool,
    prompt_suffix: str,
    prediction_postprocessor: str,
) -> dict[str, Any]:
    strict = env_bool("KG1_OFFICIAL_LIKE_STRICT", True)
    controls = {
        "strict": strict,
        "max_tokens": max_tokens,
        "official_max_tokens": OFFICIAL_MAX_TOKENS,
        "max_model_len": max_model_len,
        "official_max_model_len": OFFICIAL_MAX_MODEL_LEN,
        "max_num_seqs": max_num_seqs,
        "official_max_num_seqs": OFFICIAL_MAX_NUM_SEQS,
        "gpu_memory_utilization": gpu_memory_utilization,
        "official_gpu_memory_utilization": OFFICIAL_GPU_MEMORY_UTILIZATION,
        "disable_thinking": disable_thinking,
        "no_prompt_suffix": no_prompt_suffix,
        "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
        "prediction_postprocessor": prediction_postprocessor,
    }
    log_json("official_like_control_gate", controls)
    if not strict:
        return controls
    if max_tokens != OFFICIAL_MAX_TOKENS:
        raise RuntimeError(f"official-like gate requires max_tokens={OFFICIAL_MAX_TOKENS}, got {max_tokens}")
    if max_model_len != OFFICIAL_MAX_MODEL_LEN:
        raise RuntimeError(f"official-like gate requires max_model_len={OFFICIAL_MAX_MODEL_LEN}, got {max_model_len}")
    if max_num_seqs != OFFICIAL_MAX_NUM_SEQS:
        raise RuntimeError(f"official-like gate requires max_num_seqs={OFFICIAL_MAX_NUM_SEQS}, got {max_num_seqs}")
    if abs(gpu_memory_utilization - OFFICIAL_GPU_MEMORY_UTILIZATION) > 1e-9:
        raise RuntimeError(
            "official-like gate requires gpu_memory_utilization="
            f"{OFFICIAL_GPU_MEMORY_UTILIZATION}, got {gpu_memory_utilization}"
        )
    if disable_thinking:
        raise RuntimeError("official-like gate requires thinking enabled; KG1_DISABLE_THINKING must be 0")
    if no_prompt_suffix:
        raise RuntimeError("official-like gate requires the official prompt suffix; KG1_NO_PROMPT_SUFFIX must be 0")
    if prompt_suffix != DEFAULT_PROMPT_SUFFIX:
        raise RuntimeError("official-like gate requires the exact Kaggle Overview prompt suffix")
    if prediction_postprocessor not in {"", "none"}:
        raise RuntimeError("official-like gate forbids external prediction postprocessors")
    return controls


def validate_repo_commit() -> str:
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = env_str("KG1_EXPECTED_COMMIT")
    print("repo_commit =", observed, flush=True)
    print("expected_repo_commit =", expected, flush=True)
    if expected and observed != expected:
        raise RuntimeError(f"repo commit mismatch: expected {expected}, got {observed}")
    return observed


def ensure_import(name: str) -> None:
    code = f"import importlib; m=importlib.import_module({name!r}); print(getattr(m, '__version__', 'unknown'))"
    subprocess.check_call([sys.executable, "-c", code])
    print(f"import_ok = {name}", flush=True)


def validate_full_csv(path: Path, expected_csv_sha: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    observed_csv_sha = sha256_file(path)
    if expected_csv_sha and observed_csv_sha != expected_csv_sha:
        raise RuntimeError(f"full CSV sha mismatch: expected {expected_csv_sha}, got {observed_csv_sha}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "prompt", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("full CSV missing required columns: " + json.dumps(missing))
    if "family" not in frame.columns:
        if "type" in frame.columns:
            frame["family"] = frame["type"]
        else:
            frame["family"] = frame["prompt"].map(classify_puzzle)
    frame["family"] = frame["family"].map(canonical_family)
    inferred = frame["prompt"].map(classify_puzzle)
    mismatch = frame["family"].ne("unknown") & inferred.ne("unknown") & frame["family"].ne(inferred)
    if mismatch.any():
        sample = frame.loc[mismatch, ["id", "family", "prompt"]].head(5).to_dict(orient="records")
        raise RuntimeError("full CSV family/prompt mismatch: " + json.dumps(sample, sort_keys=True))
    if frame["id"].fillna("").astype(str).eq("").any():
        raise RuntimeError("full CSV contains empty ids")
    if frame["answer"].fillna("").astype(str).eq("").any():
        raise RuntimeError("full CSV contains empty answers")
    counts = {str(k): int(v) for k, v in frame["family"].value_counts().sort_index().to_dict().items()}
    expected_counts = json.loads(env_str("KG1_EXPECTED_FULL_COUNTS_JSON", json.dumps(EXPECTED_FULL_COUNTS)))
    expected_rows = env_int("KG1_EXPECTED_FULL_ROWS", EXPECTED_FULL_ROWS)
    if counts != expected_counts:
        raise RuntimeError(f"full family counts mismatch: expected {expected_counts}, got {counts}")
    if len(frame) != expected_rows:
        raise RuntimeError(f"full row count mismatch: expected {expected_rows}, got {len(frame)}")
    duplicate_ids = int(frame["id"].duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(f"full CSV has duplicate ids: {duplicate_ids}")
    frame["prompt_sha256"] = frame["prompt"].map(sha256_text)
    digest_payload = "\n".join(
        f"{row.id}\t{row.family}\t{row.answer}\t{row.prompt_sha256}"
        for row in frame.sort_values("id").itertuples(index=False)
    )
    return {
        "path": str(path),
        "sha256": observed_csv_sha,
        "row_contract_sha256": sha256_text(digest_payload),
        "rows": int(len(frame)),
        "family_counts": counts,
        "duplicate_ids": duplicate_ids,
    }


def validate_full_manifest(path: Path, expected_csv_sha: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if str(payload.get("sha256", "")) != expected_csv_sha:
        raise RuntimeError(f"full manifest sha mismatch: expected {expected_csv_sha}, got {payload.get('sha256')}")
    if int(payload.get("rows", -1)) != EXPECTED_FULL_ROWS:
        raise RuntimeError(f"full manifest row mismatch: {payload.get('rows')}")
    return payload


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    config = read_json(config_path)
    expected_r = env_int("KG1_EXPECTED_LORA_R", 32)
    expected_alpha = env_int("KG1_EXPECTED_LORA_ALPHA", 32)
    if int(config.get("r", -1)) != expected_r:
        raise RuntimeError(f"adapter r mismatch: expected {expected_r}, got {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != expected_alpha:
        raise RuntimeError(f"adapter alpha mismatch: expected {expected_alpha}, got {config.get('lora_alpha')}")
    config_sha = sha256_file(config_path)
    weights_sha = sha256_file(weights_path)
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_weights_bytes": int(weights_path.stat().st_size),
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": weights_sha,
        "target_modules": config.get("target_modules"),
        "target_parameters": config.get("target_parameters"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }


def parse_adapter_specs(adapter_repo: str, adapter_subfolders_raw: str) -> list[dict[str, str]]:
    specs_raw = env_str("KG1_ADAPTER_SPECS_JSON")
    if specs_raw:
        parsed = json.loads(specs_raw)
        if not isinstance(parsed, list) or not parsed:
            raise RuntimeError("KG1_ADAPTER_SPECS_JSON must be a non-empty JSON list")
        specs: list[dict[str, str]] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise RuntimeError(f"adapter spec at index {index} must be an object")
            repo = str(item.get("repo", "")).strip()
            subfolder = str(item.get("subfolder", "")).strip().strip("/")
            name = str(item.get("name", "")).strip()
            revision = str(item.get("revision", "")).strip()
            if not repo:
                raise RuntimeError(f"adapter spec at index {index} missing repo")
            if not name:
                name = f"candidate_{index}_{subfolder.replace('/', '_') or 'root'}"
            specs.append({"repo": repo, "subfolder": subfolder, "name": name, "revision": revision})
        return specs
    subfolders = [part.strip().strip("/") for part in adapter_subfolders_raw.split(",") if part.strip()]
    if not subfolders:
        subfolders = [env_str("KG1_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER).strip("/")]
    names = [part.strip() for part in env_str("KG1_CANDIDATE_NAMES").split(",") if part.strip()]
    if names and len(names) != len(subfolders):
        raise RuntimeError("KG1_CANDIDATE_NAMES count must match adapter subfolder count")
    return [
        {
            "repo": adapter_repo,
            "subfolder": subfolder,
            "name": names[index] if names else f"v284_official_like_{subfolder.replace('/', '_') or 'root'}",
            "revision": env_str("KG1_ADAPTER_REVISION"),
        }
        for index, subfolder in enumerate(subfolders)
    ]


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, timeout_s: int) -> int:
    printable = " ".join(str(part) for part in cmd)
    print("--- COMMAND START ---", flush=True)
    print("cwd =", cwd, flush=True)
    print("+", printable, flush=True)
    print("timeout_s =", timeout_s, flush=True)
    print("log_path =", log_path, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        rc = proc.wait(timeout=timeout_s)
    print("returncode =", rc, flush=True)
    print("--- COMMAND END ---", flush=True)
    if rc:
        raise RuntimeError(f"command failed rc={rc}: {printable}")
    return rc


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V284 HF OFFICIAL-LIKE EVAL GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    validate_hf_flavor_cost()
    validate_gpu()
    repo_commit = validate_repo_commit()
    ensure_import("vllm")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = env_str("HF_TOKEN")
    data_repo = env_str("KG1_DATA_REPO", DEFAULT_DATA_REPO)
    full_csv_file = env_str("KG1_FULL_CSV_FILE", DEFAULT_FULL_CSV_FILE)
    full_manifest_file = env_str("KG1_FULL_MANIFEST_FILE", DEFAULT_FULL_MANIFEST_FILE)
    adapter_repo = env_str("KG1_ADAPTER_REPO", DEFAULT_ADAPTER_REPO)
    adapter_specs = parse_adapter_specs(adapter_repo, env_str("KG1_ADAPTER_SUBFOLDERS"))
    output_repo = env_str("KG1_OUTPUT_REPO", DEFAULT_OUTPUT_REPO)
    run_id = env_str(
        "KG1_RUN_ID",
        "v284-hf-official-like-eval-gate-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    output_dir = Path(env_str("KG1_OUTPUT_DIR", "/tmp/kg1_v284_official_like_eval_gate")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_json(
        "v284_official_like_eval_inputs",
        {
            "data_repo": data_repo,
            "full_csv_file": full_csv_file,
            "full_manifest_file": full_manifest_file,
            "adapter_specs": adapter_specs,
            "output_repo": output_repo,
            "run_id": run_id,
            "output_dir": str(output_dir),
        },
    )

    full_csv = Path(hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=full_csv_file, token=token or None))
    full_manifest = Path(
        hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=full_manifest_file, token=token or None)
    )
    expected_csv_sha = env_str("KG1_EXPECTED_FULL_CSV_SHA256", EXPECTED_FULL_CSV_SHA256)
    full_meta = validate_full_csv(full_csv, expected_csv_sha)
    manifest_meta = validate_full_manifest(full_manifest, expected_csv_sha)
    log_json("full_csv_gate", full_meta)
    log_json("full_manifest_gate", manifest_meta)

    adapter_cache_dir = Path(env_str("KG1_ADAPTER_CACHE_DIR", "/tmp/kg1_v284_adapter_snapshots")) / run_id
    hub_api = HfApi(token=token or None)
    repo_roots: dict[str, Path] = {}
    for repo in sorted({spec["repo"] for spec in adapter_specs}):
        subfolders = sorted({spec["subfolder"] for spec in adapter_specs if spec["repo"] == repo})
        revisions = sorted({spec.get("revision", "") for spec in adapter_specs if spec["repo"] == repo})
        if len(revisions) > 1:
            raise RuntimeError(f"adapter specs for repo {repo} use multiple revisions: {revisions}")
        revision = revisions[0] if revisions else ""
        allow_patterns = [f"{subfolder}/*" for subfolder in subfolders if subfolder] or ["*"]
        repo_cache_dir = adapter_cache_dir / hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
        model_info = hub_api.model_info(repo_id=repo, revision=revision or None)
        print("snapshot_adapter_repo =", repo, flush=True)
        print("snapshot_adapter_revision =", revision or "<default>", flush=True)
        print("snapshot_adapter_resolved_revision =", model_info.sha, flush=True)
        print("snapshot_allow_patterns =", json.dumps(allow_patterns), flush=True)
        repo_roots[repo] = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="model",
                revision=revision or None,
                allow_patterns=allow_patterns,
                local_dir=str(repo_cache_dir),
                token=token or None,
            )
        )
        for spec in adapter_specs:
            if spec["repo"] == repo:
                spec["resolved_revision"] = str(model_info.sha or "")

    adapter_metas: list[dict[str, Any]] = []
    candidate_payload: list[dict[str, str]] = []
    for spec in adapter_specs:
        adapter_dir = repo_roots[spec["repo"]] / spec["subfolder"] if spec["subfolder"] else repo_roots[spec["repo"]]
        adapter_meta = validate_adapter(adapter_dir)
        adapter_meta.update(
            {
                "repo": spec["repo"],
                "subfolder": spec["subfolder"],
                "revision": spec.get("revision", ""),
                "resolved_revision": spec.get("resolved_revision", ""),
                "candidate_name": spec["name"],
            }
        )
        adapter_metas.append(adapter_meta)
        candidate_payload.append({"name": spec["name"], "adapter": str(adapter_dir), "source_kind": "hf_model_repo"})
    log_json("adapter_gates", {"count": len(adapter_metas), "adapters": adapter_metas})

    candidate_json = output_dir / "v284_official_like_eval_candidates.json"
    candidate_json.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True), encoding="utf-8")
    eval_out = output_dir / "eval"
    disable_thinking = env_bool("KG1_DISABLE_THINKING", False)
    no_prompt_suffix = env_bool("KG1_NO_PROMPT_SUFFIX", False)
    prompt_suffix = os.environ.get("KG1_PROMPT_SUFFIX", DEFAULT_PROMPT_SUFFIX)
    prediction_postprocessor = env_str("KG1_PREDICTION_POSTPROCESSOR", DEFAULT_POSTPROCESSOR)
    max_tokens = env_int("KG1_MAX_TOKENS", OFFICIAL_MAX_TOKENS)
    max_model_len = env_int("KG1_MAX_MODEL_LEN", OFFICIAL_MAX_MODEL_LEN)
    max_num_seqs = env_int("KG1_MAX_NUM_SEQS", OFFICIAL_MAX_NUM_SEQS)
    gpu_memory_utilization = env_float("KG1_GPU_MEMORY_UTILIZATION", OFFICIAL_GPU_MEMORY_UTILIZATION)
    official_like_control_gate = validate_official_like_controls(
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_thinking=disable_thinking,
        no_prompt_suffix=no_prompt_suffix,
        prompt_suffix=prompt_suffix,
        prediction_postprocessor=prediction_postprocessor,
    )
    log_json(
        "eval_prompt_controls",
        {
            "disable_thinking": disable_thinking,
            "no_prompt_suffix": no_prompt_suffix,
            "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
            "prediction_postprocessor": prediction_postprocessor,
        },
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_lora_adapters_batch.py"),
        "--solution-csv",
        str(full_csv),
        "--questions-csv",
        str(full_csv),
        "--candidates-json",
        str(candidate_json),
        "--base-model-path",
        env_str("KG1_MODEL_NAME", DEFAULT_MODEL_NAME),
        "--label-prefix",
        env_str("KG1_LABEL_PREFIX", "v284_hf_official_like"),
        "--seed",
        env_str("KG1_SEED", "42"),
        "--limit",
        "0",
        "--output-dir",
        str(eval_out),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--warmup-rows",
        "0",
        "--prediction-postprocessor",
        prediction_postprocessor,
        "--continue-on-error",
    ]
    if disable_thinking:
        cmd.append("--disable-thinking")
    if no_prompt_suffix:
        cmd.append("--no-prompt-suffix")
    elif prompt_suffix:
        cmd.extend(["--prompt-suffix", prompt_suffix])
    run_cmd(
        cmd,
        cwd=ROOT,
        log_path=output_dir / "v284_hf_official_like_eval.log",
        timeout_s=env_int("KG1_EVAL_TIMEOUT_S", 7200),
    )

    summary_path = eval_out / "batch_candidate_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    log_json("candidate_summary_payload", summary)
    best_rows = [row for row in summary.get("rows", []) if str(row.get("status", "")) == "ok"]
    full_candidate_gate = False
    full_min_candidate = env_int("KG1_FULL_MIN_CANDIDATE", 831)
    full_max_trunc = env_int("KG1_FULL_MAX_TRUNC", 4)
    if best_rows:
        passed_rows = [
            row
            for row in best_rows
            if int(row.get("correct", 0)) >= full_min_candidate
            and int(row.get("truncated", 999999)) <= full_max_trunc
        ]
        ranked_rows = passed_rows or best_rows
        best = max(
            ranked_rows,
            key=lambda row: (
                int(row.get("correct", 0)),
                -int(row.get("truncated", 999999)),
            ),
        )
        full_candidate_gate = bool(passed_rows)
        log_json("best_full_candidate", best)
    else:
        best = {}
    print("full_candidate_gate =", full_candidate_gate, flush=True)

    final_manifest = {
        "schema_version": "kg1_v284_hf_official_like_eval_gate_manifest_v1",
        "generated_at_utc": utc_now(),
        "repo_commit": repo_commit,
        "run_id": run_id,
        "full_csv": full_meta,
        "adapters": adapter_metas,
        "eval_summary_json": str(summary_path),
        "candidate_summary": summary,
        "best_full_candidate": best,
        "full_candidate_gate": bool(full_candidate_gate),
        "eval_prompt_controls": {
            "disable_thinking": disable_thinking,
            "no_prompt_suffix": no_prompt_suffix,
            "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
            "prediction_postprocessor": prediction_postprocessor,
        },
        "official_like_control_gate": official_like_control_gate,
        "blocked_actions": ["package", "kaggle_submit"],
    }
    final_manifest_path = output_dir / "v284_hf_official_like_eval_gate_manifest.json"
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("final_manifest_path =", final_manifest_path, flush=True)

    if env_bool("KG1_UPLOAD_TO_HF", True):
        api = HfApi(token=token or None)
        path_in_repo = env_str("KG1_OUTPUT_PATH_IN_REPO", f"evals/{run_id}")
        final_manifest["path_in_repo"] = path_in_repo
        final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("upload_folder_repo =", output_repo, flush=True)
        print("upload_folder_path_in_repo =", path_in_repo, flush=True)
        upload_info = api.upload_folder(
            repo_id=output_repo,
            repo_type="model",
            folder_path=str(output_dir),
            path_in_repo=path_in_repo,
            commit_message=f"Add {run_id} official-like eval outputs",
            ignore_patterns=["adapter_snapshot/**"],
        )
        final_manifest["upload_info"] = str(upload_info)
        final_manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8")
        print("upload_info =", upload_info, flush=True)

    print("=== V284 HF OFFICIAL-LIKE EVAL GATE END ===", flush=True)
    return final_manifest


def self_test() -> int:
    print("=== V284 HF OFFICIAL-LIKE EVAL GATE SELF TEST START ===", flush=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "kg1_v284_official_like_eval_gate_self_test"
    tmp.mkdir(parents=True, exist_ok=True)
    csv_path = tmp / "full.csv"
    pd.DataFrame(
        [
            {"id": "b", "prompt": "Perform bit manipulation on 8-bit binary 00000000.", "answer": "00000000", "type": "bit_manipulation"},
            {"id": "e", "prompt": "Apply the transformation rule a -> b.", "answer": "b", "type": "equation_transform"},
        ]
    ).to_csv(csv_path, index=False)
    try:
        validate_full_csv(csv_path, "")
    except RuntimeError as exc:
        if "full family counts mismatch" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected small full CSV count failure")
    validate_official_like_controls(
        max_tokens=OFFICIAL_MAX_TOKENS,
        max_model_len=OFFICIAL_MAX_MODEL_LEN,
        max_num_seqs=OFFICIAL_MAX_NUM_SEQS,
        gpu_memory_utilization=OFFICIAL_GPU_MEMORY_UTILIZATION,
        disable_thinking=False,
        no_prompt_suffix=False,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        prediction_postprocessor="none",
    )
    try:
        validate_official_like_controls(
            max_tokens=96,
            max_model_len=OFFICIAL_MAX_MODEL_LEN,
            max_num_seqs=OFFICIAL_MAX_NUM_SEQS,
            gpu_memory_utilization=OFFICIAL_GPU_MEMORY_UTILIZATION,
            disable_thinking=False,
            no_prompt_suffix=False,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
            prediction_postprocessor="none",
        )
    except RuntimeError as exc:
        if "max_tokens" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected cheap max_tokens rejection")
    try:
        validate_official_like_controls(
            max_tokens=OFFICIAL_MAX_TOKENS,
            max_model_len=OFFICIAL_MAX_MODEL_LEN,
            max_num_seqs=OFFICIAL_MAX_NUM_SEQS,
            gpu_memory_utilization=OFFICIAL_GPU_MEMORY_UTILIZATION,
            disable_thinking=False,
            no_prompt_suffix=False,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
            prediction_postprocessor="v274_numeric_operator_overrides",
        )
    except RuntimeError as exc:
        if "postprocessor" not in str(exc):
            raise
    else:
        raise RuntimeError("self-test expected postprocessor rejection")
    print("v284_hf_official_like_eval_gate_self_test=ok", flush=True)
    print("=== V284 HF OFFICIAL-LIKE EVAL GATE SELF TEST END ===", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    manifest = run_eval(args)
    if manifest.get("full_candidate_gate") is not True and not env_bool("KG1_ALLOW_FAILED_GATE_EXIT_0", False):
        raise RuntimeError("official-like full eval gate failed; refusing successful exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

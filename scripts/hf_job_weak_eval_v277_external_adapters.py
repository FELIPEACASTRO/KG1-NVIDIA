#!/usr/bin/env python3
"""Run a guarded weak eval for newly triaged public external adapters.

V277 is evaluation-only. It validates the canonical V221 315-row weak bridge,
downloads only adapter config/weights for public candidates that passed static
triage, evaluates them in one vLLM load, and uploads only eval outputs. It never
trains, packages, or submits to Kaggle.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hf_job_weak_eval_v245 import (  # noqa: E402
    DEFAULT_DATA_REPO,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROMPT_SUFFIX,
    DEFAULT_WEAK_CSV_FILE,
    DEFAULT_WEAK_MANIFEST_FILE,
    EXPECTED_SHARED_ROW_CONTRACT_SHA256,
    EXPECTED_WEAK_CSV_SHA256,
    env_bool,
    env_int,
    env_str,
    ensure_import,
    log_json,
    read_json,
    run_cmd,
    utc_now,
    validate_gpu,
    validate_repo_commit,
    validate_weak_csv,
    validate_weak_manifest,
)


DEFAULT_OUTPUT_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_EXTERNAL_ADAPTER_SPECS = [
    {
        "repo": "gfinin/nemotron-reasoning-lora",
        "subfolder": "",
        "name": "gfinin_nemotron_reasoning_lora_root",
        "expected_r": 16,
        "expected_alpha": 32,
    },
    {
        "repo": "etencore/nemotron-30b-reasoning-lora",
        "subfolder": "",
        "name": "etencore_nemotron_30b_reasoning_lora_root",
        "expected_r": 32,
        "expected_alpha": 64,
    },
    {
        "repo": "etencore/nemotron-30b-reasoning-lora",
        "subfolder": "checkpoint-1000",
        "name": "etencore_nemotron_30b_reasoning_lora_checkpoint_1000",
        "expected_r": 32,
        "expected_alpha": 64,
    },
    {
        "repo": "etencore/nemotron-30b-reasoning-lora",
        "subfolder": "checkpoint-1188",
        "name": "etencore_nemotron_30b_reasoning_lora_checkpoint_1188",
        "expected_r": 32,
        "expected_alpha": 64,
    },
]
EXPECTED_BASE_MODEL_RE = r"NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def validate_hf_flavor_cost() -> None:
    flavor = env_str("KG1_HF_FLAVOR")
    allowed = {part.strip() for part in env_str("KG1_ALLOWED_HF_FLAVORS", "a100-large,h100,h200").split(",") if part.strip()}
    if flavor and allowed and flavor not in allowed:
        raise RuntimeError(f"HF flavor {flavor!r} not allowed; allowed={sorted(allowed)}")
    unit_cost = float(env_str("KG1_HF_UNIT_COST_USD", "0") or "0")
    max_unit_cost = float(env_str("KG1_HF_MAX_UNIT_COST_USD", "0.09") or "0.09")
    if unit_cost and unit_cost > max_unit_cost:
        raise RuntimeError(f"HF unit cost too high: {unit_cost} > {max_unit_cost}")
    log_json(
        "hf_flavor_cost_gate",
        {"flavor": flavor, "allowed_flavors": sorted(allowed), "unit_cost_usd": unit_cost, "max_unit_cost_usd": max_unit_cost},
    )


def parse_external_adapter_specs() -> list[dict[str, Any]]:
    specs_raw = env_str("KG1_ADAPTER_SPECS_JSON")
    parsed = json.loads(specs_raw) if specs_raw else DEFAULT_EXTERNAL_ADAPTER_SPECS
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("KG1_ADAPTER_SPECS_JSON must be a non-empty JSON list")
    specs: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(f"adapter spec at index {index} must be an object")
        repo = str(item.get("repo", "")).strip()
        subfolder = str(item.get("subfolder", "")).strip().strip("/")
        name = str(item.get("name", "")).strip()
        if not repo:
            raise RuntimeError(f"adapter spec at index {index} missing repo")
        if not name:
            name = f"v277_external_{index}_{subfolder.replace('/', '_') or 'root'}"
        spec = dict(item)
        spec.update({"repo": repo, "subfolder": subfolder, "name": name})
        specs.append(spec)
    return specs


def allow_patterns_for_specs(specs: list[dict[str, Any]]) -> list[str]:
    patterns: list[str] = []
    for spec in specs:
        subfolder = str(spec["subfolder"]).strip("/")
        if subfolder:
            patterns.extend(
                [
                    f"{subfolder}/adapter_config.json",
                    f"{subfolder}/adapter_model.safetensors",
                    f"{subfolder}/README.md",
                    f"{subfolder}/tokenizer.json",
                    f"{subfolder}/tokenizer_config.json",
                    f"{subfolder}/chat_template.jinja",
                ]
            )
        else:
            patterns.extend(
                [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "README.md",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ]
            )
    return sorted(set(patterns))


def validate_adapter_flexible(adapter_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    config = read_json(config_path)
    base_model = str(config.get("base_model_name_or_path", ""))
    expected_base_re = env_str("KG1_EXPECTED_BASE_MODEL_REGEX", EXPECTED_BASE_MODEL_RE)
    if expected_base_re and not re.search(expected_base_re, base_model, re.IGNORECASE):
        raise RuntimeError(f"adapter base model mismatch for {spec['name']}: {base_model!r}")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise RuntimeError(f"adapter peft_type must be LORA for {spec['name']}: {config.get('peft_type')!r}")
    r_value = int(config.get("r", -1))
    alpha_value = int(config.get("lora_alpha", -1))
    max_rank = env_int("KG1_MAX_EXTERNAL_LORA_R", 32)
    max_alpha = env_int("KG1_MAX_EXTERNAL_LORA_ALPHA", 128)
    if r_value <= 0 or r_value > max_rank:
        raise RuntimeError(f"adapter rank out of allowed range for {spec['name']}: {r_value}")
    if alpha_value <= 0 or alpha_value > max_alpha:
        raise RuntimeError(f"adapter alpha out of allowed range for {spec['name']}: {alpha_value}")
    expected_r = spec.get("expected_r")
    expected_alpha = spec.get("expected_alpha")
    if expected_r is not None and r_value != int(expected_r):
        raise RuntimeError(f"adapter rank mismatch for {spec['name']}: expected {expected_r}, got {r_value}")
    if expected_alpha is not None and alpha_value != int(expected_alpha):
        raise RuntimeError(f"adapter alpha mismatch for {spec['name']}: expected {expected_alpha}, got {alpha_value}")
    target_modules = config.get("target_modules")
    if not target_modules:
        raise RuntimeError(f"adapter has empty target_modules for {spec['name']}")
    min_bytes = env_int("KG1_MIN_ADAPTER_WEIGHT_BYTES", 1024 * 1024)
    weight_bytes = int(weights_path.stat().st_size)
    if weight_bytes < min_bytes:
        raise RuntimeError(f"adapter weights too small for {spec['name']}: {weight_bytes} < {min_bytes}")
    return {
        "candidate_name": spec["name"],
        "repo": spec["repo"],
        "subfolder": spec["subfolder"],
        "adapter_dir": str(adapter_dir),
        "adapter_weights_bytes": weight_bytes,
        "base_model_name_or_path": base_model,
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "r": r_value,
        "lora_alpha": alpha_value,
        "target_modules": target_modules,
        "target_parameters": config.get("target_parameters"),
    }


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows", summary.get("candidates", []))
    if not isinstance(rows, list):
        raise RuntimeError("batch summary must contain list under rows or candidates")
    return [row for row in rows if isinstance(row, dict)]


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V277 EXTERNAL ADAPTER WEAK EVAL JOB START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    validate_hf_flavor_cost()
    validate_gpu()
    repo_commit = validate_repo_commit()
    ensure_import("vllm")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = env_str("HF_TOKEN")
    data_repo = env_str("KG1_DATA_REPO", DEFAULT_DATA_REPO)
    weak_csv_file = env_str("KG1_WEAK_CSV_FILE", DEFAULT_WEAK_CSV_FILE)
    weak_manifest_file = env_str("KG1_WEAK_MANIFEST_FILE", DEFAULT_WEAK_MANIFEST_FILE)
    adapter_specs = parse_external_adapter_specs()
    output_repo = env_str("KG1_OUTPUT_REPO", DEFAULT_OUTPUT_REPO)
    run_id = env_str("KG1_RUN_ID", "v277-external-adapter-weak-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output_dir = Path(env_str("KG1_OUTPUT_DIR", "/tmp/kg1_v277_external_adapter_weak_eval")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_json(
        "v277_inputs",
        {
            "data_repo": data_repo,
            "weak_csv_file": weak_csv_file,
            "weak_manifest_file": weak_manifest_file,
            "adapter_specs": adapter_specs,
            "output_repo": output_repo,
            "run_id": run_id,
            "output_dir": str(output_dir),
        },
    )

    weak_csv = Path(hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=weak_csv_file, token=token or None))
    weak_manifest = Path(hf_hub_download(repo_id=data_repo, repo_type="dataset", filename=weak_manifest_file, token=token or None))
    expected_csv_sha = env_str("KG1_EXPECTED_WEAK_CSV_SHA256", EXPECTED_WEAK_CSV_SHA256)
    expected_contract = env_str("KG1_EXPECTED_SHARED_ROW_CONTRACT_SHA256", EXPECTED_SHARED_ROW_CONTRACT_SHA256)
    weak_meta = validate_weak_csv(weak_csv, expected_csv_sha, expected_contract)
    manifest_meta = validate_weak_manifest(weak_manifest, expected_contract)
    log_json("weak_csv_gate", weak_meta)
    log_json("weak_manifest_gate", {"schema_version": manifest_meta.get("schema_version"), "canonical_weak_csv": manifest_meta.get("canonical_weak_csv")})

    specs_by_repo: dict[str, list[dict[str, Any]]] = {}
    for spec in adapter_specs:
        specs_by_repo.setdefault(spec["repo"], []).append(spec)
    adapter_cache_dir = Path(env_str("KG1_ADAPTER_CACHE_DIR", "/tmp/kg1_v277_adapter_snapshots")) / run_id
    repo_roots: dict[str, Path] = {}
    for repo, specs in specs_by_repo.items():
        allow_patterns = allow_patterns_for_specs(specs)
        repo_cache_dir = adapter_cache_dir / hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
        print("snapshot_adapter_repo =", repo, flush=True)
        print("snapshot_allow_patterns =", json.dumps(allow_patterns, indent=2), flush=True)
        repo_roots[repo] = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="model",
                allow_patterns=allow_patterns,
                local_dir=str(repo_cache_dir),
                token=token or None,
            )
        )

    adapter_metas: list[dict[str, Any]] = []
    candidate_payload: list[dict[str, str]] = []
    for spec in adapter_specs:
        adapter_root = repo_roots[spec["repo"]]
        adapter_dir = adapter_root / spec["subfolder"] if spec["subfolder"] else adapter_root
        adapter_meta = validate_adapter_flexible(adapter_dir, spec)
        adapter_metas.append(adapter_meta)
        candidate_payload.append({"name": spec["name"], "adapter": str(adapter_dir), "source_kind": "external_hf_public_adapter"})
    log_json("adapter_gates", {"count": len(adapter_metas), "adapters": adapter_metas})

    candidate_json = output_dir / "v277_external_adapter_candidates.json"
    candidate_json.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True), encoding="utf-8")
    eval_out = output_dir / "eval"
    disable_thinking = env_bool("KG1_DISABLE_THINKING", True)
    no_prompt_suffix = env_bool("KG1_NO_PROMPT_SUFFIX", False)
    prompt_suffix = os.environ.get("KG1_PROMPT_SUFFIX", DEFAULT_PROMPT_SUFFIX)
    prediction_postprocessor = env_str("KG1_PREDICTION_POSTPROCESSOR", "none")
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
        str(weak_csv),
        "--questions-csv",
        str(weak_csv),
        "--candidates-json",
        str(candidate_json),
        "--base-model-path",
        env_str("KG1_MODEL_NAME", DEFAULT_MODEL_NAME),
        "--label-prefix",
        env_str("KG1_LABEL_PREFIX", "v277_external_weak"),
        "--seed",
        env_str("KG1_SEED", "42"),
        "--limit",
        "0",
        "--output-dir",
        str(eval_out),
        "--max-tokens",
        str(env_int("KG1_MAX_TOKENS", 96)),
        "--max-model-len",
        str(env_int("KG1_MAX_MODEL_LEN", 4096)),
        "--max-num-seqs",
        str(env_int("KG1_MAX_NUM_SEQS", 8)),
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
    run_cmd(cmd, cwd=ROOT, log_path=output_dir / "v277_external_weak_eval.log", timeout_s=env_int("KG1_EVAL_TIMEOUT_S", 2400))

    summary_path = eval_out / "batch_candidate_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    rows = summary_rows(summary)
    best = max([row for row in rows if str(row.get("status", "")) == "ok"], key=lambda row: int(row.get("correct", 0)), default={})
    weak_gate_pass = bool(
        best
        and int(best.get("correct", 0)) >= env_int("KG1_WEAK_TOTAL_MIN", 193)
        and int(best.get("equation_transform_correct", 0)) >= env_int("KG1_WEAK_EQ_MIN", 60)
        and int(best.get("bit_manipulation_correct", 0)) >= env_int("KG1_WEAK_BIT_MIN", 133)
        and int(best.get("truncated", 999999)) <= env_int("KG1_WEAK_TRUNC_MAX", 3)
    )
    log_json("candidate_summary_payload", summary)
    log_json("v277_best_candidate_gate", {"best": best, "weak_gate_pass": weak_gate_pass})

    final_manifest = {
        "schema_version": "kg1_v277_external_adapter_weak_eval_manifest_v1",
        "generated_at_utc": utc_now(),
        "repo_commit": repo_commit,
        "run_id": run_id,
        "weak_csv": weak_meta,
        "adapters": adapter_metas,
        "eval_summary_json": str(summary_path),
        "candidate_summary": summary,
        "best_candidate": best,
        "weak_gate_pass": weak_gate_pass,
        "eval_prompt_controls": {
            "disable_thinking": disable_thinking,
            "no_prompt_suffix": no_prompt_suffix,
            "prompt_suffix": "" if no_prompt_suffix else prompt_suffix,
            "prediction_postprocessor": prediction_postprocessor,
        },
        "blocked_actions": ["train", "full_eval", "package", "kaggle_submit"],
    }
    final_manifest_path = output_dir / "v277_external_adapter_weak_eval_manifest.json"
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
            commit_message=f"Add {run_id} external weak eval outputs",
            ignore_patterns=["adapter_snapshot/**"],
        )
        print("upload_info =", upload_info, flush=True)

    print("=== V277 EXTERNAL ADAPTER WEAK EVAL JOB END ===", flush=True)
    return final_manifest


def self_test() -> int:
    print("=== V277 EXTERNAL ADAPTER WEAK EVAL SELF TEST START ===", flush=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "kg1_v277_external_adapter_self_test"
    tmp.mkdir(parents=True, exist_ok=True)
    adapter_dir = tmp / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
                "target_parameters": None,
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"0" * 2048)
    old_min = os.environ.get("KG1_MIN_ADAPTER_WEIGHT_BYTES")
    os.environ["KG1_MIN_ADAPTER_WEIGHT_BYTES"] = "1024"
    try:
        meta = validate_adapter_flexible(
            adapter_dir,
            {"repo": "self/test", "subfolder": "", "name": "self_test", "expected_r": 16, "expected_alpha": 32},
        )
        assert meta["r"] == 16
    finally:
        if old_min is None:
            os.environ.pop("KG1_MIN_ADAPTER_WEIGHT_BYTES", None)
        else:
            os.environ["KG1_MIN_ADAPTER_WEIGHT_BYTES"] = old_min
    print("v277_external_adapter_weak_eval_self_test=ok", flush=True)
    print("=== V277 EXTERNAL ADAPTER WEAK EVAL SELF TEST END ===", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

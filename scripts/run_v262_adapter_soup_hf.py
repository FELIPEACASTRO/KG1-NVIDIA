#!/usr/bin/env python3
"""Build validated LoRA adapter soups from existing HF adapters.

This is a CPU-only preparation job. It does not train, evaluate, package, or
submit anything. The goal is to test a low-cost hypothesis before another GPU
run: weighted averaging of adapters from the same lineage may preserve the
strong bit behavior while nudging equation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, hf_hub_download


DEFAULT_INPUTS = [
    {
        "name": "v226_checkpoint1",
        "repo": "felipesp1983/kg1-strong-adapters-v194-v226",
        "subfolder": "v226_checkpoint1",
    },
    {
        "name": "v257_checkpoint4",
        "repo": "felipesp1983/kg1-nemotron-lora-v257-v249-v226-smoke",
        "subfolder": "checkpoint-4",
    },
    {
        "name": "v259_checkpoint4",
        "repo": "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke",
        "subfolder": "checkpoint-4",
    },
]

DEFAULT_RECIPES = [
    {
        "name": "soup_v226_050_v257_050",
        "weights": {"v226_checkpoint1": 0.50, "v257_checkpoint4": 0.50},
        "primary": "v226_checkpoint1",
    },
    {
        "name": "soup_v226_050_v259_050",
        "weights": {"v226_checkpoint1": 0.50, "v259_checkpoint4": 0.50},
        "primary": "v226_checkpoint1",
    },
    {
        "name": "soup_v226_034_v257_033_v259_033",
        "weights": {
            "v226_checkpoint1": 0.34,
            "v257_checkpoint4": 0.33,
            "v259_checkpoint4": 0.33,
        },
        "primary": "v226_checkpoint1",
    },
]

OPTIONAL_COPY_FILES = [
    "README.md",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_env(name: str, default: Any) -> Any:
    raw = env_str(name, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON") from exc


def log_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label} = {json.dumps(payload, indent=2, sort_keys=True)}", flush=True)


def require_repo_commit() -> str:
    import subprocess

    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected = env_str("KG1_EXPECTED_COMMIT", "")
    print("repo_commit =", observed, flush=True)
    print("expected_repo_commit =", expected, flush=True)
    if expected and observed != expected:
        raise RuntimeError(f"repo commit mismatch: expected {expected}, got {observed}")
    return observed


def download_adapter(spec: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    name = str(spec["name"])
    repo = str(spec["repo"])
    subfolder = str(spec["subfolder"]).strip("/")
    local_dir = work_dir / "inputs" / name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{utc_now()}] download_adapter_start name={name} repo={repo} subfolder={subfolder}", flush=True)
    config_src = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=f"{subfolder}/adapter_config.json"))
    weights_src = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=f"{subfolder}/adapter_model.safetensors"))
    config_path = local_dir / "adapter_config.json"
    weights_path = local_dir / "adapter_model.safetensors"
    shutil.copy2(config_src, config_path)
    shutil.copy2(weights_src, weights_path)

    for rel in OPTIONAL_COPY_FILES:
        try:
            optional_src = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=f"{subfolder}/{rel}"))
        except Exception:
            continue
        shutil.copy2(optional_src, local_dir / rel)

    config = read_json(config_path)
    if int(config.get("r", -1)) != 32:
        raise RuntimeError(f"{name} adapter r mismatch: {config.get('r')}")
    if int(config.get("lora_alpha", -1)) != 32:
        raise RuntimeError(f"{name} adapter alpha mismatch: {config.get('lora_alpha')}")

    row = {
        "name": name,
        "repo": repo,
        "subfolder": subfolder,
        "local_dir": str(local_dir),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "r": int(config.get("r")),
        "lora_alpha": int(config.get("lora_alpha")),
        "target_modules": config.get("target_modules"),
        "target_parameters": config.get("target_parameters"),
    }
    log_json("adapter_input", row)
    return row


def tensor_index(weights_path: Path) -> dict[str, tuple[list[int], str]]:
    from safetensors import safe_open

    out: dict[str, tuple[list[int], str]] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            out[key] = (list(tensor.shape), str(tensor.dtype))
    return out


def validate_tensor_contract(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    indexes: dict[str, dict[str, tuple[list[int], str]]] = {}
    for name, row in inputs.items():
        indexes[name] = tensor_index(Path(row["local_dir"]) / "adapter_model.safetensors")

    names = sorted(indexes)
    base_name = names[0]
    base = indexes[base_name]
    for name in names[1:]:
        current = indexes[name]
        if set(current) != set(base):
            missing = sorted(set(base) - set(current))[:20]
            extra = sorted(set(current) - set(base))[:20]
            raise RuntimeError(f"tensor key mismatch for {name}; missing={missing}; extra={extra}")
        bad = [
            key
            for key in base
            if current[key][0] != base[key][0] or current[key][1] != base[key][1]
        ]
        if bad:
            raise RuntimeError(f"tensor shape/dtype mismatch for {name}: {bad[:10]}")

    contract = {
        "base_input": base_name,
        "adapter_count": len(names),
        "tensor_count": len(base),
        "tensor_contract_sha256": hashlib.sha256(
            "\n".join(f"{key}\t{base[key][0]}\t{base[key][1]}" for key in sorted(base)).encode("utf-8")
        ).hexdigest(),
    }
    log_json("tensor_contract", contract)
    return contract


def normalize_recipe(recipe: dict[str, Any], input_names: set[str]) -> dict[str, Any]:
    name = str(recipe["name"])
    weights = {str(k): float(v) for k, v in dict(recipe["weights"]).items()}
    if not weights:
        raise RuntimeError(f"{name} has no weights")
    unknown = sorted(set(weights) - input_names)
    if unknown:
        raise RuntimeError(f"{name} references unknown inputs: {unknown}")
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError(f"{name} weight sum must be positive")
    weights = {key: value / total for key, value in weights.items()}
    primary = str(recipe.get("primary") or next(iter(weights)))
    if primary not in input_names:
        raise RuntimeError(f"{name} primary input is unknown: {primary}")
    return {"name": name, "weights": weights, "primary": primary}


def build_soup(
    recipe: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    recipe_name = str(recipe["name"])
    output_dir = output_root / recipe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_dir = Path(inputs[str(recipe["primary"])]["local_dir"])
    shutil.copy2(primary_dir / "adapter_config.json", output_dir / "adapter_config.json")
    for rel in OPTIONAL_COPY_FILES:
        src = primary_dir / rel
        if src.exists():
            shutil.copy2(src, output_dir / rel)

    merged: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(
                safe_open(str(Path(inputs[name]["local_dir"]) / "adapter_model.safetensors"), framework="pt", device="cpu")
            )
            for name in recipe["weights"]
        }
        keys = list(handles[next(iter(handles))].keys())
        print(f"[{utc_now()}] soup_start name={recipe_name} tensor_count={len(keys)}", flush=True)
        for index, key in enumerate(keys, start=1):
            acc = None
            target_dtype = None
            for input_name, weight in recipe["weights"].items():
                tensor = handles[input_name].get_tensor(key)
                target_dtype = tensor.dtype
                value = tensor.to(torch.float32).mul(float(weight))
                acc = value if acc is None else acc.add(value)
            if acc is None or target_dtype is None:
                raise RuntimeError(f"empty tensor merge for {key}")
            merged[key] = acc.to(target_dtype).contiguous()
            if index % 1000 == 0 or index == len(keys):
                print(f"[{utc_now()}] soup_progress name={recipe_name} tensors={index}/{len(keys)}", flush=True)

    weights_path = output_dir / "adapter_model.safetensors"
    save_file(merged, str(weights_path))
    del merged

    row = {
        "name": recipe_name,
        "output_dir": str(output_dir),
        "weights": recipe["weights"],
        "primary": recipe["primary"],
        "adapter_config_sha256": sha256_file(output_dir / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(weights_path),
        "adapter_weights_bytes": weights_path.stat().st_size,
    }
    log_json("soup_output", row)
    return row


def write_readme(stage_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# KG1 V262 Adapter Soups",
        "",
        "CPU-only adapter soups built from already validated KG1 adapters.",
        "",
        "This repo does not authorize full eval or Kaggle submission. Each soup must pass the canonical V221 weak gate before promotion.",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Generated at UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Soups",
        "",
        "| Soup | Weights | SHA256 | Bytes |",
        "|---|---|---|---:|",
    ]
    for row in manifest["soups"]:
        weights = ", ".join(f"{k}={v:.4f}" for k, v in sorted(row["weights"].items()))
        lines.append(
            f"| `{row['name']}` | `{weights}` | `{row['adapter_weights_sha256']}` | {row['adapter_weights_bytes']} |"
        )
    lines.append("")
    (stage_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def upload_stage(stage_dir: Path, repo_id: str, private: bool) -> None:
    api = HfApi()
    print(f"[{utc_now()}] create_or_reuse_repo repo_id={repo_id} private={private}", flush=True)
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[{utc_now()}] upload_folder_start repo_id={repo_id} folder={stage_dir}", flush=True)
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload V262 adapter soups {stage_dir.name}",
    )
    print(f"[{utc_now()}] upload_folder_done repo_id={repo_id}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = env_str("KG1_RUN_ID", f"v262-adapter-soup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    output_repo = env_str("KG1_OUTPUT_REPO", "felipesp1983/kg1-nemotron-lora-v262-adapter-soups")
    inputs_raw = parse_json_env("KG1_SOUP_INPUTS_JSON", DEFAULT_INPUTS)
    recipes_raw = parse_json_env("KG1_SOUP_RECIPES_JSON", DEFAULT_RECIPES)
    if not isinstance(inputs_raw, list) or not inputs_raw:
        raise RuntimeError("KG1_SOUP_INPUTS_JSON must be a non-empty list")
    if not isinstance(recipes_raw, list) or not recipes_raw:
        raise RuntimeError("KG1_SOUP_RECIPES_JSON must be a non-empty list")

    repo_commit = require_repo_commit() if env_bool("KG1_REQUIRE_REPO_COMMIT", True) else ""
    with tempfile.TemporaryDirectory(prefix="kg1_v262_soup_") as tmp:
        work_dir = Path(tmp)
        stage_dir = Path(args.output_dir or work_dir / "stage")
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_rows = [download_adapter(spec, work_dir) for spec in inputs_raw]
        inputs = {str(row["name"]): row for row in input_rows}
        if len(inputs) != len(input_rows):
            raise RuntimeError("duplicate input names")
        tensor_contract = validate_tensor_contract(inputs)
        recipes = [normalize_recipe(recipe, set(inputs)) for recipe in recipes_raw]
        soup_rows = [build_soup(recipe, inputs, stage_dir) for recipe in recipes]

        manifest = {
            "schema_version": "kg1_v262_adapter_soup_manifest_v1",
            "generated_at_utc": utc_now(),
            "run_id": run_id,
            "repo_commit": repo_commit,
            "output_repo": output_repo,
            "inputs": input_rows,
            "recipes": recipes,
            "soups": soup_rows,
            "tensor_contract": tensor_contract,
            "gates": {
                "cpu_only": True,
                "train_allowed": False,
                "eval_allowed": False,
                "full_scoring_allowed": False,
                "package_allowed": False,
                "kaggle_submit_allowed": False,
            },
            "decision": {
                "decision": "adapter_soups_ready_for_weak_eval_only",
                "next_action": "Run V221-contract weak eval on the soups; stop if first result regresses.",
            },
        }
        write_json(stage_dir / "v262_adapter_soup_manifest.json", manifest)
        write_readme(stage_dir, manifest)
        if env_bool("KG1_UPLOAD_TO_HF", True):
            upload_stage(stage_dir, output_repo, env_bool("KG1_OUTPUT_PRIVATE", True))
        print("manifest_path =", str(stage_dir / "v262_adapter_soup_manifest.json"), flush=True)
        print("output_repo =", output_repo, flush=True)
        print("=== V262 ADAPTER SOUP END ===", flush=True)
        return manifest


def self_test() -> None:
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory(prefix="kg1_v262_selftest_") as tmp:
        root = Path(tmp)
        stage = root / "stage"
        stage.mkdir()
        a = root / "a"
        b = root / "b"
        for directory, value in [(a, 1.0), (b, 3.0)]:
            directory.mkdir()
            write_json(directory / "adapter_config.json", {"r": 32, "lora_alpha": 32, "target_modules": ["q_proj"]})
            save_file({"x": torch.tensor([value], dtype=torch.float32)}, str(directory / "adapter_model.safetensors"))
        inputs = {
            "a": {"name": "a", "local_dir": str(a)},
            "b": {"name": "b", "local_dir": str(b)},
        }
        validate_tensor_contract(inputs)
        row = build_soup({"name": "ab", "weights": {"a": 0.25, "b": 0.75}, "primary": "a"}, inputs, stage)
        from safetensors.torch import load_file

        value = float(load_file(str(stage / "ab" / "adapter_model.safetensors"))["x"][0])
        if abs(value - 2.5) > 1e-6:
            raise AssertionError(value)
        if row["adapter_weights_bytes"] <= 0:
            raise AssertionError("empty self-test weights")
    print("v262_adapter_soup_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    print("=== V262 ADAPTER SOUP START ===", flush=True)
    args = parse_args()
    if args.self_test:
        self_test()
        print("=== V262 ADAPTER SOUP END ===", flush=True)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

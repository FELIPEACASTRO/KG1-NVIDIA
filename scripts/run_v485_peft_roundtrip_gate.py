#!/usr/bin/env python3
"""V485 PEFT adapter structural round-trip gate.

This is a cheap CPU/HF-metadata gate that must pass before opening a paid GPU
job. It validates the adapter lineage topology that matters for KG1:

- adapter_config fields, especially target_modules and target_parameters
- modules_to_save must be empty for the adapter-only submit path
- adapter_model.safetensors metadata must contain LoRA tensors for all required
  target_modules and target_parameters
- no full saved module tensors are allowed

The operational path intentionally uses safetensors metadata from the Hub instead
of downloading the multi-GB adapter weights. The self-test uses an in-memory
fixture so CI can validate the rules without HF credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "lm_head",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
DEFAULT_TARGET_PARAMETERS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: list[int]
    parameter_count: int


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_sha256(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tensor_meta(raw: Any) -> TensorMeta:
    if isinstance(raw, TensorMeta):
        return raw
    if isinstance(raw, dict):
        dtype = str(raw["dtype"])
        shape = [int(item) for item in raw["shape"]]
        parameter_count = int(raw.get("parameter_count", 1))
        return TensorMeta(dtype=dtype, shape=shape, parameter_count=parameter_count)
    dtype = str(getattr(raw, "dtype"))
    shape = [int(item) for item in getattr(raw, "shape")]
    parameter_count = int(getattr(raw, "parameter_count"))
    return TensorMeta(dtype=dtype, shape=shape, parameter_count=parameter_count)


def tensor_fingerprint(tensors: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    total_params = 0
    for name, raw_meta in sorted(tensors.items()):
        meta = normalize_tensor_meta(raw_meta)
        rows.append(
            {
                "name": name,
                "dtype": meta.dtype,
                "shape": meta.shape,
                "parameter_count": meta.parameter_count,
            }
        )
        dtype_counts[meta.dtype] = dtype_counts.get(meta.dtype, 0) + 1
        total_params += meta.parameter_count
    return {
        "tensor_count": len(rows),
        "total_parameters": total_params,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "key_shape_dtype_sha256": stable_json_sha256(rows),
    }


def target_parameter_key_matches(target_parameter: str, tensor_name: str) -> bool:
    if target_parameter in tensor_name:
        return True
    lowered = target_parameter.lower()
    tensor_lower = tensor_name.lower()
    # Nemotron/PEFT saves the configured mlp.experts.gate_up_proj adapter under
    # mixer.experts.<id>.up_proj LoRA keys. Treat that as the structural alias,
    # while still requiring the adapter_config target_parameters field to match.
    if lowered.endswith("experts.gate_up_proj"):
        return "experts." in tensor_lower and "up_proj" in tensor_lower
    if lowered.endswith("experts.down_proj"):
        return "experts." in tensor_lower and "down_proj" in tensor_lower
    return False


def is_allowed_non_lora_tensor(name: str) -> bool:
    # The current V290/V291 lineage includes lm_head as a LoRA target and PEFT
    # metadata exposes the wrapped base layer weight. This is allowed only as a
    # known, fingerprinted lineage artifact; modules_to_save remains forbidden.
    return name.endswith("lm_head.base_layer.weight")


def audit_adapter(
    *,
    config: dict[str, Any],
    tensors: dict[str, Any],
    expected_r: int,
    expected_alpha: int,
    expected_target_modules: list[str],
    expected_target_parameters: list[str],
    require_target_parameter_match: bool,
    allowed_extra_target_modules: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    config_target_modules = sorted(str(item) for item in (config.get("target_modules") or []))
    config_target_parameters = sorted(str(item) for item in (config.get("target_parameters") or []))
    modules_to_save = sorted(str(item) for item in (config.get("modules_to_save") or []))
    expected_modules_sorted = sorted(expected_target_modules)
    expected_parameters_sorted = sorted(expected_target_parameters)
    allowed_extra_modules_sorted = sorted(set(allowed_extra_target_modules or []))
    disallowed_extra_modules = [
        item for item in allowed_extra_modules_sorted if item not in {"lm_head", "embed_tokens", "word_embeddings"}
    ]
    if disallowed_extra_modules:
        errors.append(
            "allowed extra target_modules may only be output/embedding modules: "
            + json.dumps(disallowed_extra_modules, sort_keys=True)
        )
    effective_config_target_modules = sorted(
        item for item in config_target_modules if item not in set(allowed_extra_modules_sorted)
    )
    missing_allowed_extra_modules = [
        item for item in allowed_extra_modules_sorted if item not in set(config_target_modules)
    ]
    if missing_allowed_extra_modules:
        errors.append(
            "allowed extra target_modules absent from adapter config: "
            + json.dumps(missing_allowed_extra_modules, sort_keys=True)
        )

    if int(config.get("r", -1)) != expected_r:
        errors.append(f"adapter r mismatch: {config.get('r')} != {expected_r}")
    if int(config.get("lora_alpha", -1)) != expected_alpha:
        errors.append(f"adapter lora_alpha mismatch: {config.get('lora_alpha')} != {expected_alpha}")
    if expected_modules_sorted and effective_config_target_modules != expected_modules_sorted:
        errors.append(
            "adapter target_modules mismatch: "
            + json.dumps(
                {
                    "adapter": config_target_modules,
                    "allowed_extra_target_modules": allowed_extra_modules_sorted,
                    "effective_adapter": effective_config_target_modules,
                    "expected": expected_modules_sorted,
                },
                sort_keys=True,
            )
        )
    if expected_parameters_sorted and config_target_parameters != expected_parameters_sorted:
        errors.append(
            "adapter target_parameters mismatch: "
            + json.dumps(
                {"adapter": config_target_parameters, "expected": expected_parameters_sorted},
                sort_keys=True,
            )
        )
    if modules_to_save:
        errors.append("adapter modules_to_save must be empty: " + json.dumps(modules_to_save, sort_keys=True))

    module_lora_tensors = {item: 0 for item in expected_modules_sorted}
    module_lora_params = {item: 0 for item in expected_modules_sorted}
    target_parameter_lora_tensors = {item: 0 for item in expected_parameters_sorted}
    target_parameter_lora_params = {item: 0 for item in expected_parameters_sorted}
    modules_to_save_keys: list[str] = []
    non_lora_tensor_keys: list[str] = []
    allowed_non_lora_tensor_keys: list[str] = []

    for name, raw_meta in sorted(tensors.items()):
        meta = normalize_tensor_meta(raw_meta)
        if "modules_to_save" in name:
            modules_to_save_keys.append(name)
        is_lora = ".lora_" in name or "lora_" in name
        if not is_lora:
            if is_allowed_non_lora_tensor(name):
                allowed_non_lora_tensor_keys.append(name)
            else:
                non_lora_tensor_keys.append(name)
        for module in expected_modules_sorted:
            if module in name and is_lora:
                module_lora_tensors[module] += 1
                module_lora_params[module] += meta.parameter_count
        for target_parameter in expected_parameters_sorted:
            if target_parameter_key_matches(target_parameter, name) and is_lora:
                target_parameter_lora_tensors[target_parameter] += 1
                target_parameter_lora_params[target_parameter] += meta.parameter_count

    if modules_to_save_keys:
        errors.append(
            "adapter_model contains modules_to_save tensors: "
            + json.dumps(modules_to_save_keys[:20], sort_keys=True)
        )
    if non_lora_tensor_keys:
        errors.append(
            "adapter_model contains non-LoRA tensors in strict adapter-only path: "
            + json.dumps(non_lora_tensor_keys[:20], sort_keys=True)
        )
    missing_modules = [name for name, count in module_lora_tensors.items() if count <= 0]
    if missing_modules:
        errors.append("missing LoRA tensors for target_modules: " + ", ".join(missing_modules))
    missing_target_parameters = [
        name for name, count in target_parameter_lora_tensors.items() if count <= 0
    ]
    if require_target_parameter_match and missing_target_parameters:
        errors.append(
            "missing LoRA tensors for target_parameters: " + ", ".join(missing_target_parameters)
        )

    fingerprint = tensor_fingerprint(tensors)
    summary: dict[str, Any] = {
        "config": {
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config_target_modules,
            "effective_target_modules": effective_config_target_modules,
            "allowed_extra_target_modules": allowed_extra_modules_sorted,
            "target_parameters": config_target_parameters,
            "modules_to_save": modules_to_save,
            "adapter_config_sha256": stable_json_sha256(config),
        },
        "safetensors": fingerprint,
        "coverage": {
            "module_lora_tensors": module_lora_tensors,
            "module_lora_params": module_lora_params,
            "target_parameter_lora_tensors": target_parameter_lora_tensors,
            "target_parameter_lora_params": target_parameter_lora_params,
            "modules_to_save_key_count": len(modules_to_save_keys),
            "non_lora_tensor_key_count": len(non_lora_tensor_keys),
            "allowed_non_lora_tensor_keys": allowed_non_lora_tensor_keys,
        },
        "hf_gpu_allowed": not errors,
        "errors": errors,
    }
    return summary, errors


def load_hub_adapter_metadata(
    *,
    repo_id: str,
    subfolder: str,
    revision: str,
    token: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download, parse_safetensors_file_metadata

    api = HfApi(token=token)
    repo_info = api.model_info(repo_id, revision=revision or None, files_metadata=True)
    prefix = subfolder.strip("/")
    config_filename = f"{prefix}/adapter_config.json" if prefix else "adapter_config.json"
    weights_filename = f"{prefix}/adapter_model.safetensors" if prefix else "adapter_model.safetensors"
    siblings = {sibling.rfilename: sibling for sibling in repo_info.siblings}
    missing = [name for name in [config_filename, weights_filename] if name not in siblings]
    if missing:
        raise FileNotFoundError(f"missing adapter files in {repo_id}: {missing}")

    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            subfolder=prefix or None,
            revision=revision or None,
            token=token,
            repo_type="model",
        )
    )
    config = read_json(config_path)
    metadata = parse_safetensors_file_metadata(
        repo_id=repo_id,
        filename=weights_filename,
        revision=revision or None,
        token=token,
        repo_type="model",
    )
    tensors = {name: normalize_tensor_meta(meta) for name, meta in metadata.tensors.items()}
    file_info = {
        "repo_id": repo_id,
        "revision": revision or getattr(repo_info, "sha", ""),
        "resolved_revision": getattr(repo_info, "sha", ""),
        "config_filename": config_filename,
        "weights_filename": weights_filename,
        "adapter_model_size": getattr(siblings[weights_filename], "size", None),
    }
    return config, tensors, file_info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_self_test() -> int:
    print("=== V485 PEFT ROUNDTRIP GATE SELF TEST START ===", flush=True)
    config = {
        "r": 32,
        "lora_alpha": 32,
        "target_modules": DEFAULT_TARGET_MODULES,
        "target_parameters": DEFAULT_TARGET_PARAMETERS,
        "modules_to_save": None,
    }
    tensors: dict[str, TensorMeta] = {}
    for idx, module in enumerate(DEFAULT_TARGET_MODULES):
        tensors[f"base_model.model.layers.0.self_attn.{module}.lora_A.default.weight"] = TensorMeta(
            "BF16", [32, 64 + idx], 32 * (64 + idx)
        )
        tensors[f"base_model.model.layers.0.self_attn.{module}.lora_B.default.weight"] = TensorMeta(
            "BF16", [64 + idx, 32], (64 + idx) * 32
        )
    for target_parameter in DEFAULT_TARGET_PARAMETERS:
        tensors[f"base_model.model.layers.0.{target_parameter}.lora_A.default.weight"] = TensorMeta(
            "BF16", [32, 128], 4096
        )
        tensors[f"base_model.model.layers.0.{target_parameter}.lora_B.default.weight"] = TensorMeta(
            "BF16", [128, 32], 4096
        )
    summary, errors = audit_adapter(
        config=config,
        tensors=tensors,
        expected_r=32,
        expected_alpha=32,
        expected_target_modules=DEFAULT_TARGET_MODULES,
        expected_target_parameters=DEFAULT_TARGET_PARAMETERS,
        require_target_parameter_match=True,
        allowed_extra_target_modules=[],
    )
    if errors or not summary["hf_gpu_allowed"]:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 1

    bad_config = dict(config)
    bad_config["modules_to_save"] = ["lm_head"]
    bad_summary, bad_errors = audit_adapter(
        config=bad_config,
        tensors=tensors,
        expected_r=32,
        expected_alpha=32,
        expected_target_modules=DEFAULT_TARGET_MODULES,
        expected_target_parameters=DEFAULT_TARGET_PARAMETERS,
        require_target_parameter_match=True,
        allowed_extra_target_modules=[],
    )
    if not bad_errors or bad_summary["hf_gpu_allowed"]:
        print("self-test failed to block modules_to_save", flush=True)
        print(json.dumps(bad_summary, indent=2, sort_keys=True), flush=True)
        return 1
    print("v485_peft_roundtrip_gate_self_test=ok", flush=True)
    print("=== V485 PEFT ROUNDTRIP GATE SELF TEST END ===", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-repo", default=os.environ.get("INIT_ADAPTER_REPO", ""))
    parser.add_argument("--adapter-subfolder", default=os.environ.get("INIT_ADAPTER_SUBFOLDER", ""))
    parser.add_argument("--adapter-revision", default=os.environ.get("INIT_ADAPTER_REVISION", ""))
    parser.add_argument("--expected-r", type=int, default=int(os.environ.get("LORA_R", "32")))
    parser.add_argument("--expected-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    parser.add_argument(
        "--expected-target-modules",
        default=os.environ.get("LORA_TARGET_MODULES", ",".join(DEFAULT_TARGET_MODULES)),
    )
    parser.add_argument(
        "--expected-target-parameters",
        default=os.environ.get("LORA_TARGET_PARAMETERS", ",".join(DEFAULT_TARGET_PARAMETERS)),
    )
    parser.add_argument(
        "--require-target-parameter-match",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("REQUIRE_LORA_TARGET_PARAMETER_MATCH", "1").strip().lower()
        not in {"0", "false", "no", "off"},
    )
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/v485_peft_roundtrip_gate_manifest.json"))
    parser.add_argument(
        "--allowed-extra-target-modules",
        default=os.environ.get("DROP_INIT_ADAPTER_TARGET_MODULES", ""),
        help="Adapter config target_modules allowed in the initial adapter but dropped from the effective train config.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    print("=== V485 PEFT ROUNDTRIP GATE START ===", flush=True)
    print("generated_at_utc =", datetime.now(timezone.utc).isoformat(), flush=True)
    print("adapter_repo =", args.adapter_repo, flush=True)
    print("adapter_subfolder =", args.adapter_subfolder, flush=True)
    print("adapter_revision =", args.adapter_revision or "default", flush=True)
    if not args.adapter_repo:
        raise SystemExit("missing --adapter-repo or INIT_ADAPTER_REPO")

    token = os.environ.get("HF_TOKEN") or None
    config, tensors, file_info = load_hub_adapter_metadata(
        repo_id=args.adapter_repo,
        subfolder=args.adapter_subfolder,
        revision=args.adapter_revision,
        token=token,
    )
    summary, errors = audit_adapter(
        config=config,
        tensors=tensors,
        expected_r=args.expected_r,
        expected_alpha=args.expected_alpha,
        expected_target_modules=parse_csv(args.expected_target_modules),
        expected_target_parameters=parse_csv(args.expected_target_parameters),
        require_target_parameter_match=bool(args.require_target_parameter_match),
        allowed_extra_target_modules=parse_csv(args.allowed_extra_target_modules),
    )
    payload = {
        "version": "V485_PEFT_ROUNDTRIP_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_info": file_info,
        "summary": summary,
    }
    write_json(args.output_json, payload)
    print("output_json =", args.output_json, flush=True)
    print("hf_gpu_allowed =", summary["hf_gpu_allowed"], flush=True)
    print("adapter_config_sha256 =", summary["config"]["adapter_config_sha256"], flush=True)
    print("key_shape_dtype_sha256 =", summary["safetensors"]["key_shape_dtype_sha256"], flush=True)
    print("target_parameter_lora_tensors =", json.dumps(summary["coverage"]["target_parameter_lora_tensors"], sort_keys=True), flush=True)
    print("module_lora_tensors =", json.dumps(summary["coverage"]["module_lora_tensors"], sort_keys=True), flush=True)
    if errors:
        print("errors =", json.dumps(errors, indent=2, sort_keys=True), flush=True)
        print("=== V485 PEFT ROUNDTRIP GATE END ===", flush=True)
        return 1
    print("v485_peft_roundtrip_gate=ok", flush=True)
    print("=== V485 PEFT ROUNDTRIP GATE END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert and gate V198 post-training adapters.

This is intentionally offline and non-submitting. It converts the final adapter
and checkpoint-30 to Kaggle adapter ZIPs, then checks the deployable layout and
tensor prefixes before any Kaggle upload is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open


EXPECTED_ZIP_ENTRIES = ["adapter_config.json", "adapter_model.safetensors"]
FORBIDDEN_TENSOR_PREFIXES = (
    "base_model.model.backbone.",
    "base_model.model.base_model.",
)
REQUIRED_TENSOR_PREFIXES = (
    "base_model.model.model.",
    "base_model.model.lm_head.",
)
REQUIRED_TRAINABLE_MODULES = {
    "in_proj",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}
EXPECTED_TARGET_MODULES = {
    "down_proj",
    "in_proj",
    "k_proj",
    "lm_head",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
KNOWN_REGRESSION_ADAPTER_SHA256 = {
    "e831c5263951c012328df14f940936809276462ffaa23a6733885945a47c7bd4",
    "b0eea0d4c9f14a1df51371284804563078cd038d1daa1d7de5e89ff365e0dd11",
    "c8a4bc30c37227541cba946345a07547813265cb5cf4202315109396f7fd0236",
    "83c83c865e5bb9cb153430e2bcf7e0f3fa4327cf0ab857060c7923156124d9a0",
    "6946c653f7684b08a78042da6fa3a346212e7c5b4648496755399d061f98d563",
}
KNOWN_REGRESSION_ZIP_SHA256 = {
    "fe2d6c5a33445a09cca9628c36cb49d4e913d7dcf8921512cba76ef91c1a3c16",
    "ba2fb4eef9178083ef097dc692ebb7b89e6a6252a0c6e12dd2606950e4e21ba5",
    "2197e123a50e78c9c2a28f718ba5f236ffe769d5b388f3a23d7efd529d02f9c0",
    "be106a972df530caeac6e241c6814a72b380b3ab8e471a228ba8922e81e89d92",
    "09af34c657b88e5c7f37ce412dc3340978eea8919e3460cc1cc82f8d16653ada",
    "c9be81b56962131861f217045b0b8d83e1175800c9523571a5fbf6db7090fb44",
}


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_module(key: str) -> str:
    for module in sorted(EXPECTED_TARGET_MODULES, key=len, reverse=True):
        if f".{module}." in key or key.endswith(f".{module}.lora_A.weight") or key.endswith(
            f".{module}.lora_B.weight"
        ):
            return module
    return "unknown"


def inspect_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
    entries = [info.filename.replace("\\", "/") for info in infos]
    zip_sha = sha256_file(zip_path)
    reasons: list[str] = []
    if sorted(entries) != EXPECTED_ZIP_ENTRIES:
        reasons.append("zip_entries_not_exactly_adapter_config_and_safetensors")
    if any("/" in entry.strip("/") for entry in entries):
        reasons.append("nested_zip_entries_present")
    if zip_sha in KNOWN_REGRESSION_ZIP_SHA256:
        reasons.append("known_regression_zip_hash")
    return {
        "path": str(zip_path),
        "sha256": zip_sha,
        "entries": entries,
        "sizes": {info.filename.replace("\\", "/"): info.file_size for info in infos},
        "reasons": reasons,
        "ok": not reasons,
    }


def inspect_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    reasons: list[str] = []
    if not config_path.exists():
        reasons.append("missing_adapter_config")
    if not weights_path.exists():
        reasons.append("missing_adapter_model_safetensors")
    if reasons:
        return {"path": str(adapter_dir), "reasons": reasons, "ok": False}

    config = read_json(config_path)
    target_modules = set(config.get("target_modules") or [])
    missing_targets = sorted(EXPECTED_TARGET_MODULES - target_modules)
    if missing_targets:
        reasons.append("missing_expected_target_modules")

    tensor_count = 0
    forbidden_samples: list[str] = []
    modules_seen: set[str] = set()
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            if key.startswith(FORBIDDEN_TENSOR_PREFIXES) and len(forbidden_samples) < 20:
                forbidden_samples.append(key)
            if key.startswith(REQUIRED_TENSOR_PREFIXES):
                modules_seen.add(classify_module(key))
    if forbidden_samples:
        reasons.append("forbidden_training_tensor_prefix_present")

    missing_trainable_modules = sorted(REQUIRED_TRAINABLE_MODULES - modules_seen)
    if missing_trainable_modules:
        reasons.append("missing_required_trainable_lora_modules")

    adapter_sha = sha256_file(weights_path)
    if adapter_sha in KNOWN_REGRESSION_ADAPTER_SHA256:
        reasons.append("known_regression_adapter_hash")

    return {
        "path": str(adapter_dir),
        "adapter_model_sha256": adapter_sha,
        "adapter_config_sha256": sha256_file(config_path),
        "tensor_count": tensor_count,
        "target_modules": sorted(target_modules),
        "missing_target_modules": missing_targets,
        "modules_seen": sorted(modules_seen),
        "missing_trainable_modules": missing_trainable_modules,
        "forbidden_tensor_samples": forbidden_samples,
        "reasons": reasons,
        "ok": not reasons,
    }


def convert_candidate(root: Path, source_dir: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = root / "scripts" / "kg1_convert_local_training_adapter_to_kaggle_zip.py"
    if not converter.exists():
        raise FileNotFoundError(f"converter script missing: {converter}")
    if not source_dir.exists():
        return {
            "name": source_dir.name,
            "source_adapter_dir": str(source_dir),
            "available": False,
            "decision": {"ready": False, "reasons": ["source_adapter_dir_missing"]},
        }
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--source-adapter-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
        ],
        check=True,
        cwd=str(root),
    )
    conversion_report = read_json(output_dir / "local_kaggle_layout_report.json")
    zip_path = Path(conversion_report["zip_path"])
    adapter_dir = Path(conversion_report["output_adapter_dir"])
    zip_report = inspect_zip(zip_path)
    adapter_report = inspect_adapter(adapter_dir)
    reasons: list[str] = []
    if not conversion_report.get("decision", {}).get("kaggle_layout_ready"):
        reasons.append("converter_report_not_kaggle_layout_ready")
    reasons.extend(f"zip:{reason}" for reason in zip_report["reasons"])
    reasons.extend(f"adapter:{reason}" for reason in adapter_report["reasons"])
    return {
        "name": source_dir.name,
        "source_adapter_dir": str(source_dir),
        "available": True,
        "conversion_report": conversion_report,
        "zip": zip_report,
        "adapter": adapter_report,
        "decision": {"ready": not reasons, "reasons": reasons},
    }


def maybe_manifest(adapter_dir: Path) -> dict[str, Any] | None:
    manifest_path = adapter_dir / "v90_training_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    return {
        "path": str(manifest_path),
        "run_id": manifest.get("run_id"),
        "best_eval_loss": manifest.get("training", {}).get("best_eval_loss"),
        "final_step": manifest.get("training", {}).get("final_step"),
        "elapsed_hours": manifest.get("training", {}).get("elapsed_hours"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/kg1_v198"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    gate_dir = output_root / "posttrain_kaggle_gate"
    candidates = [
        ("final", output_root / "final_adapter", gate_dir / "final", "v198-micro-distill-final"),
        ("checkpoint30", output_root / "checkpoint-30", gate_dir / "checkpoint30", "v198-micro-distill-checkpoint30"),
    ]
    results = []
    for label, source_dir, out_dir, run_id in candidates:
        result = convert_candidate(root, source_dir, out_dir, run_id)
        result["label"] = label
        result["training_manifest"] = maybe_manifest(source_dir)
        results.append(result)

    ready = [result for result in results if result["decision"]["ready"]]
    primary = next((result for result in ready if result["label"] == "final"), ready[0] if ready else None)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root": str(root),
        "output_root": str(output_root),
        "candidates": results,
        "decision": {
            "ready_candidate_count": len(ready),
            "primary_label": primary["label"] if primary else None,
            "primary_zip": primary["zip"]["path"] if primary else None,
            "do_not_submit_without_explicit_authorization": True,
            "ready": bool(primary),
        },
    }
    report_path = gate_dir / "v198_posttrain_gate_report.json"
    write_json(report_path, report)

    print("\n=== V198 POSTTRAIN GATE ===")
    for result in results:
        status = "READY" if result["decision"]["ready"] else "BLOCKED"
        print(f"{result['label']}: {status}")
        if result.get("available") and result.get("zip"):
            print(f"  zip: {result['zip']['path']}")
            print(f"  zip_sha256: {result['zip']['sha256']}")
        if result["decision"]["reasons"]:
            print(f"  reasons: {result['decision']['reasons']}")
    print(f"report: {report_path}")
    if primary:
        print(f"PRIMARY_CANDIDATE_ZIP={primary['zip']['path']}")
    else:
        print("PRIMARY_CANDIDATE_ZIP=NONE")
    if args.fail_on_block and not primary:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

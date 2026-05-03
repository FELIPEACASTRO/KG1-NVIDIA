#!/usr/bin/env python3
"""Final offline gate before submitting a V198 Nemotron adapter ZIP.

The script does not submit to Kaggle. It cross-checks the candidate ZIP against
the post-training conversion report and the production preflight report, then
independently inspects adapter_config.json and adapter_model.safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open


EXPECTED_ROOT_ENTRIES = {"adapter_config.json", "adapter_model.safetensors"}
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
REQUIRED_CHANGED_MODULES = {
    "in_proj",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}
ALLOWED_TENSOR_PREFIXES = (
    "base_model.model.model.",
    "base_model.model.lm_head.",
)
FORBIDDEN_TENSOR_PREFIXES = (
    "base_model.model.backbone.",
    "base_model.model.base_model.",
    "base_model.model.backbone.lm_head.",
)
KNOWN_REGRESSION_ZIP_SHA256 = {
    "fe2d6c5a33445a09cca9628c36cb49d4e913d7dcf8921512cba76ef91c1a3c16",
    "ba2fb4eef9178083ef097dc692ebb7b89e6a6252a0c6e12dd2606950e4e21ba5",
    "2197e123a50e78c9c2a28f718ba5f236ffe769d5b388f3a23d7efd529d02f9c0",
    "be106a972df530caeac6e241c6814a72b380b3ab8e471a228ba8922e81e89d92",
    "09af34c657b88e5c7f37ce412dc3340978eea8919e3460cc1cc82f8d16653ada",
    "c9be81b56962131861f217045b0b8d83e1175800c9523571a5fbf6db7090fb44",
}
KNOWN_REGRESSION_ADAPTER_SHA256 = {
    "e831c5263951c012328df14f940936809276462ffaa23a6733885945a47c7bd4",
    "b0eea0d4c9f14a1df51371284804563078cd038d1daa1d7de5e89ff365e0dd11",
    "c8a4bc30c37227541cba946345a07547813265cb5cf4202315109396f7fd0236",
    "83c83c865e5bb9cb153430e2bcf7e0f3fa4327cf0ab857060c7923156124d9a0",
    "6946c653f7684b08a78042da6fa3a346212e7c5b4648496755399d061f98d563",
}
CANONICAL_086_ZIP_SHA256 = "a3b64b154a6690a58f2338ba1c405422eadc6e1c1357f662eecb187463dfdeee"
CANONICAL_086_ADAPTER_SHA256 = "559fd024f5ffcaff0caceddeaf25c3801009d6cabf247fc8dfccbfaf2addd916"


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
        if f".{module}." in key:
            return module
    return "unknown"


def default_posttrain_report(candidate_zip: Path) -> Path:
    cursor = candidate_zip.resolve()
    for parent in [cursor, *cursor.parents]:
        if parent.name == "posttrain_kaggle_gate":
            return parent / "v198_posttrain_gate_report.json"
    return candidate_zip.parent.parent.parent / "v198_posttrain_gate_report.json"


def default_preflight_report(candidate_zip: Path, expected_label: str) -> Path:
    posttrain = default_posttrain_report(candidate_zip).parent
    label = "checkpoint30" if expected_label.lower() in {"checkpoint30", "checkpoint-30"} else "final"
    return posttrain / f"{label}_preflight.json"


def adapter_path_near_zip(candidate_zip: Path) -> Path | None:
    candidate = candidate_zip.parent.parent / "adapter" / "adapter_model.safetensors"
    return candidate if candidate.exists() else None


def inspect_zip(candidate_zip: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not candidate_zip.exists():
        return {"ok": False, "reasons": ["candidate_zip_missing"], "path": str(candidate_zip)}
    with zipfile.ZipFile(candidate_zip) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        entries = [info.filename.replace("\\", "/") for info in infos]
        sizes = {info.filename.replace("\\", "/"): info.file_size for info in infos}
        compressed_sizes = {info.filename.replace("\\", "/"): info.compress_size for info in infos}
        config_text = archive.read("adapter_config.json").decode("utf-8") if "adapter_config.json" in entries else "{}"
    root_entries = {entry for entry in entries if "/" not in entry.strip("/")}
    nested_entries = [entry for entry in entries if "/" in entry.strip("/")]
    if set(entries) != EXPECTED_ROOT_ENTRIES:
        reasons.append("zip_entries_not_exactly_adapter_config_and_safetensors")
    if root_entries != EXPECTED_ROOT_ENTRIES:
        reasons.append("root_entries_not_exactly_expected")
    if nested_entries:
        reasons.append("nested_entries_present")
    zip_sha = sha256_file(candidate_zip)
    if zip_sha in KNOWN_REGRESSION_ZIP_SHA256:
        reasons.append("known_regression_zip_hash")
    if zip_sha == CANONICAL_086_ZIP_SHA256:
        reasons.append("candidate_is_canonical_086_baseline_zip")
    if sizes.get("adapter_model.safetensors", 0) < 4_000_000_000:
        reasons.append("adapter_model_smaller_than_expected_for_full_v198")
    return {
        "ok": not reasons,
        "path": str(candidate_zip),
        "sha256": zip_sha,
        "entries": entries,
        "sizes": sizes,
        "compressed_sizes": compressed_sizes,
        "config": json.loads(config_text),
        "reasons": reasons,
    }


def inspect_config(config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    target_modules = set(config.get("target_modules") or [])
    if config.get("peft_type") != "LORA":
        reasons.append("peft_type_not_lora")
    if config.get("task_type") != "CAUSAL_LM":
        reasons.append("task_type_not_causal_lm")
    if config.get("base_model_name_or_path") != "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
        reasons.append("unexpected_base_model_name")
    if int(config.get("r") or -1) != 32:
        reasons.append("unexpected_lora_rank")
    if int(config.get("lora_alpha") or -1) != 32:
        reasons.append("unexpected_lora_alpha")
    if float(config.get("lora_dropout") or 0.0) != 0.0:
        reasons.append("unexpected_lora_dropout")
    missing_targets = sorted(EXPECTED_TARGET_MODULES - target_modules)
    if missing_targets:
        reasons.append("missing_expected_target_modules")
    return {
        "ok": not reasons,
        "target_modules": sorted(target_modules),
        "missing_target_modules": missing_targets,
        "reasons": reasons,
    }


def inspect_tensors(adapter_model: Path) -> dict[str, Any]:
    reasons: list[str] = []
    module_counts: Counter[str] = Counter()
    forbidden_samples: list[str] = []
    disallowed_samples: list[str] = []
    rank_anomaly_samples: list[dict[str, Any]] = []
    tensor_count = 0
    with safe_open(str(adapter_model), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            shape = tuple(int(dim) for dim in handle.get_tensor(key).shape)
            module_counts[classify_module(key)] += 1
            if key.startswith(FORBIDDEN_TENSOR_PREFIXES) and len(forbidden_samples) < 20:
                forbidden_samples.append(key)
            if not key.startswith(ALLOWED_TENSOR_PREFIXES) and len(disallowed_samples) < 20:
                disallowed_samples.append(key)
            if (".lora_A." in key or ".lora_B." in key) and 32 not in shape and len(rank_anomaly_samples) < 20:
                rank_anomaly_samples.append({"key": key, "shape": list(shape)})
    if tensor_count != 12011:
        reasons.append("unexpected_tensor_count")
    if forbidden_samples:
        reasons.append("forbidden_training_tensor_prefix_present")
    if disallowed_samples:
        reasons.append("disallowed_tensor_prefix_present")
    missing_modules = sorted(REQUIRED_CHANGED_MODULES - set(module_counts))
    if missing_modules:
        reasons.append("missing_required_lora_modules")
    adapter_sha = sha256_file(adapter_model)
    if adapter_sha in KNOWN_REGRESSION_ADAPTER_SHA256:
        reasons.append("known_regression_adapter_hash")
    if adapter_sha == CANONICAL_086_ADAPTER_SHA256:
        reasons.append("candidate_is_canonical_086_baseline_adapter")
    return {
        "ok": not reasons,
        "path": str(adapter_model),
        "sha256": adapter_sha,
        "tensor_count": tensor_count,
        "module_counts": dict(sorted(module_counts.items())),
        "missing_required_modules": missing_modules,
        "forbidden_tensor_samples": forbidden_samples,
        "disallowed_tensor_samples": disallowed_samples,
        "rank_anomaly_samples": rank_anomaly_samples,
        "reasons": reasons,
    }


def get_adapter_model_path(candidate_zip: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    nearby = adapter_path_near_zip(candidate_zip)
    if nearby is not None:
        return nearby, None
    tmp = tempfile.TemporaryDirectory(prefix="kg1_v198_submit_gate_")
    tmp_path = Path(tmp.name)
    with zipfile.ZipFile(candidate_zip) as archive:
        archive.extract("adapter_model.safetensors", tmp_path)
    return tmp_path / "adapter_model.safetensors", tmp


def inspect_posttrain_report(path: Path, candidate_zip: Path, expected_label: str, zip_sha: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["posttrain_report_missing"]}
    report = read_json(path)
    candidates = report.get("candidates") or []
    matches = [
        item for item in candidates
        if item.get("label") == expected_label
        or Path(str(item.get("zip", {}).get("path", ""))).name == candidate_zip.name
    ]
    if not matches:
        reasons.append("candidate_not_found_in_posttrain_report")
        item = {}
    else:
        item = matches[0]
        if not item.get("decision", {}).get("ready"):
            reasons.append("posttrain_candidate_not_ready")
        if item.get("zip", {}).get("sha256") != zip_sha:
            reasons.append("posttrain_zip_sha_mismatch")
    decision = report.get("decision") or {}
    if expected_label == "final" and decision.get("primary_label") not in {None, "final"}:
        reasons.append("posttrain_primary_is_not_final")
    return {
        "ok": not reasons,
        "path": str(path),
        "matched_candidate": item.get("label"),
        "primary_label": decision.get("primary_label"),
        "primary_zip": decision.get("primary_zip"),
        "reasons": reasons,
    }


def inspect_preflight_report(path: Path, candidate_zip: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "path": str(path), "reasons": ["preflight_report_missing"]}
    report = read_json(path)
    decision = report.get("decision") or {}
    if not decision.get("production_ready"):
        reasons.append("preflight_not_production_ready")
    if decision.get("reasons"):
        reasons.append("preflight_has_reasons")
    report_zip = Path(str(report.get("adapter_zip") or ""))
    if report_zip.name != candidate_zip.name:
        reasons.append("preflight_adapter_zip_name_mismatch")
    return {
        "ok": not reasons,
        "path": str(path),
        "production_ready": decision.get("production_ready"),
        "decision_reasons": decision.get("reasons"),
        "report_adapter_zip": str(report_zip),
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-zip", type=Path, required=True)
    parser.add_argument("--expected-label", default="final", choices=["final", "checkpoint30", "checkpoint-30"])
    parser.add_argument("--posttrain-report", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fail-on-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_zip = args.candidate_zip.resolve()
    expected_label = "checkpoint30" if args.expected_label == "checkpoint-30" else args.expected_label
    posttrain_report = args.posttrain_report or default_posttrain_report(candidate_zip)
    preflight_report = args.preflight_report or default_preflight_report(candidate_zip, expected_label)

    zip_report = inspect_zip(candidate_zip)
    config_report = inspect_config(zip_report.get("config") or {})
    adapter_model, tmp = get_adapter_model_path(candidate_zip)
    try:
        tensor_report = inspect_tensors(adapter_model)
    finally:
        if tmp is not None:
            tmp.cleanup()
    posttrain = inspect_posttrain_report(posttrain_report, candidate_zip, expected_label, zip_report.get("sha256", ""))
    preflight = inspect_preflight_report(preflight_report, candidate_zip)

    sections = {
        "zip": zip_report,
        "config": config_report,
        "tensors": tensor_report,
        "posttrain_report": posttrain,
        "preflight_report": preflight,
    }
    reasons: list[str] = []
    for section_name, section in sections.items():
        if not section.get("ok"):
            reasons.extend(f"{section_name}:{reason}" for reason in section.get("reasons", []))

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "candidate_zip": str(candidate_zip),
        "expected_label": expected_label,
        **sections,
        "decision": {
            "submit_ready": not reasons,
            "reasons": reasons,
            "requires_explicit_kaggle_submit_authorization": True,
        },
    }
    write_json(args.output_json, report)
    print("\n=== V198 FINAL SUBMIT DOUBLECHECK ===")
    print(f"candidate_zip: {candidate_zip}")
    print(f"zip_sha256: {zip_report.get('sha256')}")
    print(f"adapter_model_sha256: {tensor_report.get('sha256')}")
    print(f"submit_ready: {report['decision']['submit_ready']}")
    if reasons:
        print("reasons:")
        for reason in reasons:
            print(f"  - {reason}")
    print(f"report: {args.output_json}")
    if args.fail_on_block and reasons:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

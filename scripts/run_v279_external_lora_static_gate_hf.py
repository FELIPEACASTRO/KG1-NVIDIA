#!/usr/bin/env python3
"""Run CPU-only static gates for external Nemotron LoRA repos.

V279 inspects candidate Hugging Face model repos without spending GPU. It uses
Hub metadata and downloads only adapter_config.json files. It does not download
large weights, train, evaluate, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPOS = [
    "passagereptile455/nemotron-reasoning-lora-v8-kaggle",
    "passagereptile455/nemotron-reasoning-lora-v9-kaggle-alpha64",
    "passagereptile455/nemotron-reasoning-lora-v10-kaggle-1epoch",
    "passagereptile455/nemotron-reasoning-lora-v11-kaggle-alpha64-1epoch",
]
EXPECTED_BASE_MODEL_TOKEN = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

CSV_COLUMNS = [
    "repo_id",
    "config_path",
    "weights_path",
    "weights_size",
    "has_partial_files",
    "base_model_name_or_path",
    "peft_type",
    "r",
    "lora_alpha",
    "target_modules",
    "target_parameters",
    "static_gate_pass",
    "failure_reasons",
    "next_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def sibling_size(sibling: Any) -> int:
    value = getattr(sibling, "size", None)
    try:
        return int(value or 0)
    except Exception:
        return 0


def sibling_name(sibling: Any) -> str:
    return str(getattr(sibling, "rfilename", "") or "")


def subfolder_of(path: str) -> str:
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0]


def join_rel(folder: str, filename: str) -> str:
    return f"{folder}/{filename}" if folder else filename


def list_adapter_config_paths(file_names: list[str]) -> list[str]:
    return sorted(name for name in file_names if name.endswith("adapter_config.json"))


def inspect_repo(repo_id: str, token: str | None) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V279") from exc

    api = HfApi(token=token)
    info = api.model_info(repo_id, files_metadata=True)
    siblings = list(getattr(info, "siblings", []) or [])
    names = [sibling_name(item) for item in siblings]
    size_by_name = {sibling_name(item): sibling_size(item) for item in siblings}
    partial_files = sorted(name for name in names if name.endswith(".partial") or ".partial" in name)
    config_paths = list_adapter_config_paths(names)
    if not config_paths:
        return [
            {
                "repo_id": repo_id,
                "config_path": "",
                "weights_path": "",
                "weights_size": 0,
                "has_partial_files": bool(partial_files),
                "base_model_name_or_path": "",
                "peft_type": "",
                "r": "",
                "lora_alpha": "",
                "target_modules": "",
                "target_parameters": "",
                "static_gate_pass": False,
                "failure_reasons": "missing_adapter_config_json",
                "next_action": "discard_without_gpu",
            }
        ]

    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        folder = subfolder_of(config_path)
        weights_path = join_rel(folder, "adapter_model.safetensors")
        alt_bin_path = join_rel(folder, "adapter_model.bin")
        config_file = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=config_path,
                token=token,
            )
        )
        config = json.loads(config_file.read_text(encoding="utf-8"))
        base_model = str(config.get("base_model_name_or_path", ""))
        target_modules = config.get("target_modules")
        target_parameters = config.get("target_parameters")
        failures: list[str] = []
        if EXPECTED_BASE_MODEL_TOKEN not in base_model:
            failures.append("base_model_mismatch")
        if weights_path not in names:
            if alt_bin_path in names:
                failures.append("missing_adapter_model_safetensors_has_bin_only")
                weights_path = alt_bin_path
            else:
                failures.append("missing_adapter_weights")
        if weights_path in names and size_by_name.get(weights_path, 0) <= 0:
            failures.append("adapter_weights_size_unavailable_or_zero")
        if partial_files:
            failures.append("partial_files_present")
        if not target_modules and not target_parameters:
            failures.append("missing_target_modules_and_target_parameters")

        passed = not failures
        rows.append(
            {
                "repo_id": repo_id,
                "config_path": config_path,
                "weights_path": weights_path if weights_path in names else "",
                "weights_size": size_by_name.get(weights_path, 0),
                "has_partial_files": bool(partial_files),
                "base_model_name_or_path": base_model,
                "peft_type": str(config.get("peft_type", "")),
                "r": str(config.get("r", "")),
                "lora_alpha": str(config.get("lora_alpha", "")),
                "target_modules": json.dumps(target_modules, sort_keys=True),
                "target_parameters": json.dumps(target_parameters, sort_keys=True),
                "static_gate_pass": passed,
                "failure_reasons": ",".join(failures),
                "next_action": "eligible_for_short_weak_eval" if passed else "discard_without_gpu",
            }
        )
    return rows


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V279 EXTERNAL LORA STATIC GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("repo_count =", len(args.repo), flush=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for repo_id in args.repo:
        print("inspect_repo =", repo_id, flush=True)
        try:
            repo_rows = inspect_repo(repo_id, token)
            rows.extend(repo_rows)
            print("inspect_repo_rows =", len(repo_rows), flush=True)
        except Exception as exc:
            message = repr(exc)
            errors.append({"repo_id": repo_id, "error": message})
            rows.append(
                {
                    "repo_id": repo_id,
                    "config_path": "",
                    "weights_path": "",
                    "weights_size": 0,
                    "has_partial_files": "",
                    "base_model_name_or_path": "",
                    "peft_type": "",
                    "r": "",
                    "lora_alpha": "",
                    "target_modules": "",
                    "target_parameters": "",
                    "static_gate_pass": False,
                    "failure_reasons": "repo_inspection_error:" + message[:300],
                    "next_action": "discard_without_gpu",
                }
            )

    passed = [row for row in rows if bool(row.get("static_gate_pass"))]
    outputs = {
        "static_gate_csv": args.output_dir / "v279_external_lora_static_gate.csv",
        "manifest_json": args.output_dir / "v279_external_lora_static_gate_manifest.json",
    }
    write_csv(outputs["static_gate_csv"], rows)

    if passed:
        decision = {
            "decision": "short_weak_eval_optional_for_static_pass_repos",
            "reason": f"static_pass={len(passed)}; inspected_rows={len(rows)}; errors={len(errors)}",
            "next_action": "Run one short weak eval only if budget allows; stop immediately if first adapter is far below baseline.",
        }
    else:
        decision = {
            "decision": "discard_passagereptile_static_gate",
            "reason": f"static_pass=0; inspected_rows={len(rows)}; errors={len(errors)}",
            "next_action": "Do not spend GPU on these repos.",
        }

    manifest = {
        "schema_version": "kg1_v279_external_lora_static_gate_v1",
        "generated_at_utc": utc_now(),
        "expected_base_model_token": EXPECTED_BASE_MODEL_TOKEN,
        "repos": args.repo,
        "counts": {
            "inspected_rows": len(rows),
            "static_gate_pass": len(passed),
            "inspection_errors": len(errors),
        },
        "errors": errors,
        "passed_rows": passed,
        "decision": decision,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V279 EXTERNAL LORA STATIC GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    names = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint-1/adapter_config.json",
        "checkpoint-1/adapter_model.safetensors",
    ]
    configs = list_adapter_config_paths(names)
    if configs != ["adapter_config.json", "checkpoint-1/adapter_config.json"]:
        raise AssertionError(configs)
    if subfolder_of("checkpoint-1/adapter_config.json") != "checkpoint-1":
        raise AssertionError("subfolder failed")
    if join_rel("checkpoint-1", "adapter_model.safetensors") != "checkpoint-1/adapter_model.safetensors":
        raise AssertionError("join failed")
    print("v279_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], help="HF model repo to inspect; may be repeated")
    parser.add_argument("--output-dir", type=Path, default=Path(f"artifacts/hf_cpu_runs/v279_external_lora_static_gate_{utc_compact()}"))
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.repo:
        args.repo = list(DEFAULT_REPOS)
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

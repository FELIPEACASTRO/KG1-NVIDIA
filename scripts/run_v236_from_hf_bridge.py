#!/usr/bin/env python3
"""Run V236 local solver DSL probes from V240 artifacts on Hugging Face Jobs.

The V240 bridge contains the V232 manifest plus the equation/bit workitems
needed by V236. The original V232 manifest points to Google Drive paths, so
this runner rewrites those paths after download and creates the two auxiliary
contract files that V236 validates for presence but does not otherwise consume.

This is CPU-only and diagnostic-only. It does not train, run model generation,
score a model, package artifacts, or submit to Kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_v236_local_solver_dsl_probes import (
    EXPECTED_ROW_CONTRACT_SHA256,
    run_analysis,
    write_json,
)


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_OUTPUT_PREFIX = "runtime_artifacts/v236_local_solver_dsl_probes"


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def download_bridge(repo_id: str, bridge_path: str, target_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download bridge artifacts") from exc

    bridge_path = bridge_path.strip("/")
    if not bridge_path:
        raise RuntimeError("--bridge-path is required unless --local-bridge-dir is used")
    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{bridge_path}/bridge_manifest.json",
            local_dir=str(target_dir),
            token=token,
        )
    )
    manifest = read_json(manifest_path)
    for item in manifest.get("files", []):
        path_in_repo = str(item.get("path_in_repo", ""))
        if not path_in_repo:
            raise RuntimeError("bridge manifest file missing path_in_repo")
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            local_dir=str(target_dir),
            token=token,
        )
    return manifest_path


def copy_local_bridge(local_bridge_dir: Path, target_dir: Path) -> Path:
    if not local_bridge_dir.exists():
        raise FileNotFoundError(local_bridge_dir)
    if not local_bridge_dir.is_dir():
        raise NotADirectoryError(local_bridge_dir)
    for source in local_bridge_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
    manifest_path = target_dir / "bridge_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return manifest_path


def file_by_label(bridge_manifest: dict[str, Any], download_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in bridge_manifest.get("files", []):
        label = str(item.get("label", ""))
        staged_name = str(item.get("staged_name", ""))
        if not label or not staged_name:
            raise RuntimeError("bridge manifest file missing label/staged_name")
        candidates = sorted(download_root.rglob(staged_name))
        if not candidates:
            raise FileNotFoundError(f"downloaded artifact not found for {label}: {staged_name}")
        mapping[label] = candidates[0]
    required = {
        "v232_manifest_json",
        "v232_equation_solver_workitems_jsonl",
        "v232_bit_guardrail_workitems_jsonl",
    }
    missing = sorted(required - set(mapping))
    if missing:
        raise RuntimeError("bridge manifest missing required labels: " + json.dumps(missing))
    return mapping


def observed_contract(manifest: dict[str, Any]) -> str:
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    return str(
        inputs.get("observed_shared_row_contract_sha256")
        or manifest.get("observed_shared_row_contract_sha256")
        or inputs.get("expected_shared_row_contract_sha256")
        or manifest.get("expected_shared_row_contract_sha256")
        or ""
    )


def build_rewritten_v232(mapping: dict[str, Path], output_dir: Path, expected_contract: str) -> Path:
    original = read_json(mapping["v232_manifest_json"])
    contract = observed_contract(original)
    if expected_contract and contract != expected_contract:
        raise RuntimeError(f"V232 shared row contract mismatch: expected {expected_contract}, got {contract}")

    acceptance = output_dir / "v236_bridge_acceptance_matrix.csv"
    contracts = output_dir / "v236_bridge_solver_contracts.json"
    acceptance.write_text("schema_version,note\nkg1_v236_bridge_auxiliary,not_used_by_v236_probe_logic\n", encoding="utf-8")
    write_json(contracts, {"schema_version": "kg1_v236_bridge_auxiliary_solver_contracts_v1"})

    rewritten = dict(original)
    rewritten["outputs"] = dict(original.get("outputs", {}))
    rewritten["outputs"]["equation_solver_workitems_jsonl"] = str(mapping["v232_equation_solver_workitems_jsonl"])
    rewritten["outputs"]["bit_guardrail_workitems_jsonl"] = str(mapping["v232_bit_guardrail_workitems_jsonl"])
    rewritten["outputs"]["acceptance_matrix_csv"] = str(acceptance)
    rewritten["outputs"]["solver_contracts_json"] = str(contracts)
    rewritten["inputs"] = dict(original.get("inputs", {}))
    rewritten["inputs"]["observed_shared_row_contract_sha256"] = contract
    rewritten["bridge_rewrite"] = {
        "source": "run_v236_from_hf_bridge",
        "auxiliary_files_created": [str(acceptance), str(contracts)],
    }
    rewritten_path = output_dir / "rewritten_v232_manifest.json"
    write_json(rewritten_path, rewritten)
    return rewritten_path


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V236 HF solver DSL probes {path_in_repo.strip('/')}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V236 HF BRIDGE RUNNER START ===", flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("bridge_path =", args.bridge_path, flush=True)
    print("local_bridge_dir =", args.local_bridge_dir or "", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        download_root = Path(tmp) / "bridge_download"
        download_root.mkdir(parents=True, exist_ok=True)
        if args.local_bridge_dir:
            bridge_manifest_path = copy_local_bridge(args.local_bridge_dir, download_root)
        else:
            bridge_manifest_path = download_bridge(
                args.hf_dataset_repo,
                args.bridge_path,
                download_root,
                args.hf_token or os.environ.get("HF_TOKEN"),
            )
        bridge_manifest = read_json(bridge_manifest_path)
        print("bridge_manifest_path =", bridge_manifest_path, flush=True)
        print("bridge_file_count =", len(bridge_manifest.get("files", [])), flush=True)
        mapping = file_by_label(bridge_manifest, download_root)
        rewritten_v232 = build_rewritten_v232(
            mapping,
            args.output_dir,
            args.expected_shared_row_contract_sha256,
        )
        analysis_args = argparse.Namespace(
            v232_analysis_manifest_json=rewritten_v232,
            output_dir=args.output_dir,
            label=args.label,
            expected_shared_row_contract_sha256=args.expected_shared_row_contract_sha256,
            equation_target_gain=args.equation_target_gain,
        )
        manifest = run_analysis(analysis_args)

    upload_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.hf_dataset_repo, args.output_dir, args.output_path_in_repo, token)
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.hf_dataset_repo,
        "path_in_repo": str(args.output_path_in_repo or ""),
        "upload_info": upload_info,
    }
    manifest_path = Path(manifest["outputs"]["manifest_json"])
    write_json(manifest_path, manifest)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("=== V236 HF BRIDGE RUNNER END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--bridge-path", default="")
    parser.add_argument("--local-bridge-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v236_local_solver_dsl_probes_from_hf_bridge")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--equation-target-gain", type=int, default=5)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bridge = root / "bridge"
        out = root / "out"
        bridge.mkdir(parents=True)
        equation = bridge / "v232_equation_workitems.jsonl"
        bit = bridge / "v232_bit_workitems.jsonl"
        v232_manifest = bridge / "v232_manifest.json"
        equation.write_text(
            json.dumps(
                {
                    "schema_version": "kg1_v232_solver_workitem_v1",
                    "id": "eq_symbolic",
                    "family": "equation_transform",
                    "solver_route": "sympy_symbolic_transform",
                    "expected_answer": "dc",
                    "baseline_prediction": "xx",
                    "prompt": "Examples: ab = ba cd = dc Now, determine the result for: cd",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bit.write_text(
            json.dumps(
                {
                    "id": "bit1",
                    "family": "bit_manipulation",
                    "expected_answer": "1010",
                    "baseline_prediction": "0000",
                    "prompt": "allowed ops: AND OR XOR NOT << >>",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            v232_manifest,
            {
                "schema_version": "kg1_v232_verified_solver_workbench_manifest_v1",
                "inputs": {"observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256},
                "outputs": {
                    "equation_solver_workitems_jsonl": "/drive/equation.jsonl",
                    "bit_guardrail_workitems_jsonl": "/drive/bit.jsonl",
                    "acceptance_matrix_csv": "/drive/acceptance.csv",
                    "solver_contracts_json": "/drive/contracts.json",
                },
            },
        )
        write_json(
            bridge / "bridge_manifest.json",
            {
                "schema_version": "kg1_v240_hf_runtime_artifact_bridge_manifest_v1",
                "files": [
                    {
                        "label": "v232_manifest_json",
                        "staged_name": v232_manifest.name,
                        "path_in_repo": f"selftest/{v232_manifest.name}",
                    },
                    {
                        "label": "v232_equation_solver_workitems_jsonl",
                        "staged_name": equation.name,
                        "path_in_repo": f"selftest/{equation.name}",
                    },
                    {
                        "label": "v232_bit_guardrail_workitems_jsonl",
                        "staged_name": bit.name,
                        "path_in_repo": f"selftest/{bit.name}",
                    },
                ],
            },
        )
        args = argparse.Namespace(
            hf_dataset_repo=DEFAULT_HF_DATASET_REPO,
            bridge_path="",
            local_bridge_dir=bridge,
            output_dir=out,
            label="v236_local_solver_dsl_probes_from_hf_bridge",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            equation_target_gain=1,
            upload_to_hf=False,
            output_path_in_repo="",
            hf_token="",
        )
        manifest = run(args)
        if manifest["probe_counts"]["equation_workitems"] != 1:
            raise AssertionError("expected one equation workitem")
        if manifest["probe_counts"]["bit_guardrail_workitems"] != 1:
            raise AssertionError("expected one bit workitem")
    print("v236_hf_bridge_runner_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        build_parser().error("--output-dir is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

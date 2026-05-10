#!/usr/bin/env python3
"""Run V239 in HF Jobs from artifacts uploaded by the V240 bridge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from analyze_v239_alice_abstain_mining import EXPECTED_ROW_CONTRACT_SHA256, run_analysis, write_json


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"


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
    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{bridge_path.rstrip('/')}/bridge_manifest.json",
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
        candidates = list(download_root.rglob(staged_name))
        if not candidates:
            raise FileNotFoundError(f"downloaded artifact not found for {label}: {staged_name}")
        mapping[label] = candidates[0]
    return mapping


def build_rewritten_manifests(mapping: dict[str, Path], output_dir: Path) -> Path:
    v232_original = read_json(mapping["v232_manifest_json"])
    v238_original = read_json(mapping["v238_manifest_json"])
    v232_rewritten = dict(v232_original)
    v232_rewritten["outputs"] = dict(v232_original.get("outputs", {}))
    v232_rewritten["outputs"]["equation_solver_workitems_jsonl"] = str(mapping["v232_equation_solver_workitems_jsonl"])
    v232_rewritten["outputs"]["bit_guardrail_workitems_jsonl"] = str(mapping["v232_bit_guardrail_workitems_jsonl"])
    v232_path = output_dir / "rewritten_v232_manifest.json"
    write_json(v232_path, v232_rewritten)

    v238_rewritten = dict(v238_original)
    v238_rewritten["inputs"] = dict(v238_original.get("inputs", {}))
    v238_rewritten["outputs"] = dict(v238_original.get("outputs", {}))
    v238_rewritten["inputs"]["v232_analysis_manifest_json"] = str(v232_path)
    v238_rewritten["outputs"]["alice_parser_probe_results_csv"] = str(mapping["v238_alice_parser_probe_results_csv"])
    v238_rewritten["outputs"]["alice_parser_probe_summary_csv"] = str(mapping["v238_alice_parser_probe_summary_csv"])
    v238_rewritten["outputs"]["manifest_json"] = str(mapping["v238_manifest_json"])
    v238_path = output_dir / "rewritten_v238_manifest.json"
    write_json(v238_path, v238_rewritten)
    return v238_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V239 HF BRIDGE RUNNER START ===", flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("bridge_path =", args.bridge_path, flush=True)
    print("local_bridge_dir =", args.local_bridge_dir or "", flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        download_root = Path(tmp) / "bridge_download"
        download_root.mkdir(parents=True, exist_ok=True)
        if args.local_bridge_dir:
            bridge_manifest_path = copy_local_bridge(args.local_bridge_dir, download_root)
        else:
            if not args.bridge_path:
                raise RuntimeError("--bridge-path is required unless --local-bridge-dir is used")
            bridge_manifest_path = download_bridge(
                args.hf_dataset_repo,
                args.bridge_path,
                download_root,
                args.hf_token or os.environ.get("HF_TOKEN"),
            )
        bridge_manifest = read_json(bridge_manifest_path)
        mapping = file_by_label(bridge_manifest, download_root)
        rewritten_v238 = build_rewritten_manifests(mapping, args.output_dir)
        analysis_args = argparse.Namespace(
            v238_analysis_manifest_json=rewritten_v238,
            output_dir=args.output_dir,
            label=args.label,
            expected_shared_row_contract_sha256=args.expected_shared_row_contract_sha256,
            target_gain=args.target_gain,
        )
        manifest = run_analysis(analysis_args)
    print("=== V239 HF BRIDGE RUNNER END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--bridge-path", default="")
    parser.add_argument("--local-bridge-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v239_alice_abstain_mining_from_hf_bridge")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-gain", type=int, default=5)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bridge = root / "bridge"
        out = root / "out"
        bridge.mkdir(parents=True)
        equation = bridge / "v232_equation_solver_workitems_jsonl.jsonl"
        bit = bridge / "v232_bit_guardrail_workitems_jsonl.jsonl"
        v238_results = bridge / "v238_alice_parser_probe_results_csv.csv"
        v238_summary = bridge / "v238_alice_parser_probe_summary_csv.csv"
        equation.write_text(
            json.dumps(
                {
                    "id": "num_abstain",
                    "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 72)27 = 99 26#48 = 22 Now, determine the result for: 11-50",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bit.write_text(json.dumps({"id": "bit1", "prompt": "noop"}) + "\n", encoding="utf-8")
        v238_results.write_text(
            "id,prompt_kind,status,query,expected_answer,baseline_prediction,proof\n"
            "num_abstain,alice_numeric_binary_operator,abstain,11-50,-39,39,no_examples_for_query_operator='-'\n",
            encoding="utf-8",
        )
        v238_summary.write_text("status,rows\nabstain,1\n", encoding="utf-8")
        v232_manifest = bridge / "v232_manifest_json.json"
        v238_manifest = bridge / "v238_manifest_json.json"
        write_json(
            v232_manifest,
            {
                "observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "outputs": {
                    "equation_solver_workitems_jsonl": "/drive/equation.jsonl",
                    "bit_guardrail_workitems_jsonl": "/drive/bit.jsonl",
                },
            },
        )
        write_json(
            v238_manifest,
            {
                "inputs": {
                    "observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                    "v232_analysis_manifest_json": "/drive/v232_manifest.json",
                },
                "outputs": {
                    "alice_parser_probe_results_csv": "/drive/results.csv",
                    "alice_parser_probe_summary_csv": "/drive/summary.csv",
                    "manifest_json": "/drive/v238_manifest.json",
                },
            },
        )
        files = [
            ("v232_manifest_json", v232_manifest),
            ("v238_manifest_json", v238_manifest),
            ("v232_equation_solver_workitems_jsonl", equation),
            ("v232_bit_guardrail_workitems_jsonl", bit),
            ("v238_alice_parser_probe_results_csv", v238_results),
            ("v238_alice_parser_probe_summary_csv", v238_summary),
        ]
        write_json(
            bridge / "bridge_manifest.json",
            {
                "schema_version": "kg1_v240_hf_runtime_artifact_bridge_manifest_v1",
                "files": [
                    {"label": label, "staged_name": path.name, "path_in_repo": f"selftest/{path.name}"}
                    for label, path in files
                ],
            },
        )
        args = argparse.Namespace(
            hf_dataset_repo=DEFAULT_HF_DATASET_REPO,
            bridge_path="",
            local_bridge_dir=bridge,
            output_dir=out,
            label="v239_alice_abstain_mining_from_hf_bridge",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            target_gain=5,
            hf_token="",
        )
        manifest = run(args)
        if manifest["counts"]["numeric_abstain"] != 1:
            raise AssertionError("expected one numeric abstain")
    print("v239_hf_bridge_runner_self_test=ok", flush=True)
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

#!/usr/bin/env python3
"""Upload KG1 runtime diagnostic artifacts from Drive to a private HF dataset.

This is the bridge that lets Hugging Face Jobs consume Colab/Drive artifacts.
It uploads manifests and small CSV/JSONL diagnostic outputs only. It does not
train, run model generation, score models, package submissions, or submit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_PREFIX = "runtime_artifacts/v240_hf_bridge"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return {
        "bytes": path.stat().st_size,
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._")
    if not text:
        raise ValueError("empty safe path component")
    return text[:180]


def observed_contract(manifest: dict[str, Any]) -> str:
    inputs = manifest.get("inputs", {}) if isinstance(manifest.get("inputs", {}), dict) else {}
    return str(
        inputs.get("observed_shared_row_contract_sha256")
        or manifest.get("observed_shared_row_contract_sha256")
        or inputs.get("expected_shared_row_contract_sha256")
        or manifest.get("expected_shared_row_contract_sha256")
        or ""
    )


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label}: {path}")
    return path


def add_artifact(artifacts: list[dict[str, Any]], label: str, path: Path) -> None:
    require_file(path, label)
    artifacts.append({"label": label, "source_path": str(path), "meta": file_meta(path)})


def collect_artifacts(v232_manifest_path: Path, v238_manifest_path: Path, expected_contract: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v232_manifest = read_json(v232_manifest_path)
    v238_manifest = read_json(v238_manifest_path)
    v232_contract = observed_contract(v232_manifest)
    v238_contract = observed_contract(v238_manifest)
    if expected_contract and v232_contract != expected_contract:
        raise RuntimeError(f"V232 shared row contract mismatch: expected {expected_contract}, got {v232_contract}")
    if expected_contract and v238_contract != expected_contract:
        raise RuntimeError(f"V238 shared row contract mismatch: expected {expected_contract}, got {v238_contract}")

    artifacts: list[dict[str, Any]] = []
    add_artifact(artifacts, "v232_manifest_json", v232_manifest_path)
    add_artifact(artifacts, "v238_manifest_json", v238_manifest_path)

    v232_outputs = v232_manifest.get("outputs", {})
    for key in ["equation_solver_workitems_jsonl", "bit_guardrail_workitems_jsonl"]:
        value = v232_outputs.get(key)
        if not value:
            raise RuntimeError(f"V232 manifest missing outputs.{key}")
        add_artifact(artifacts, f"v232_{key}", Path(str(value)))

    v238_outputs = v238_manifest.get("outputs", {})
    for key in ["alice_parser_probe_results_csv", "alice_parser_probe_summary_csv"]:
        value = v238_outputs.get(key)
        if not value:
            raise RuntimeError(f"V238 manifest missing outputs.{key}")
        add_artifact(artifacts, f"v238_{key}", Path(str(value)))

    context = {
        "v232_contract": v232_contract,
        "v238_contract": v238_contract,
        "v238_counts": v238_manifest.get("counts", {}),
        "v238_decision": v238_manifest.get("decision", {}),
    }
    return artifacts, context


def stage_artifacts(
    staging_root: Path,
    artifacts: list[dict[str, Any]],
    context: dict[str, Any],
    repo_id: str,
    path_in_repo: str,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for item in artifacts:
        label = safe_component(str(item["label"]))
        source = Path(str(item["source_path"]))
        suffix = "".join(source.suffixes) or ".artifact"
        dest = staging_root / f"{label}{suffix}"
        shutil.copy2(source, dest)
        files.append(
            {
                "label": item["label"],
                "source_path": str(source),
                "path_in_repo": f"{path_in_repo}/{dest.name}",
                "staged_name": dest.name,
                "meta": file_meta(dest),
            }
        )
    manifest = {
        "schema_version": "kg1_v240_hf_runtime_artifact_bridge_manifest_v1",
        "generated_at_utc": utc_now(),
        "repo_id": repo_id,
        "repo_type": "dataset",
        "path_in_repo": path_in_repo,
        "context": context,
        "files": files,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(staging_root / "bridge_manifest.json", manifest)
    return manifest


def upload_folder(repo_id: str, staging_root: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload; install it before running without --dry-run") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        path_in_repo=path_in_repo,
        commit_message=f"Upload KG1 runtime artifacts {path_in_repo}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V240 HF ARTIFACT BRIDGE SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v232_manifest_json =", args.v232_manifest_json, flush=True)
    print("v238_manifest_json =", args.v238_manifest_json, flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("path_prefix =", args.path_prefix, flush=True)
    print("run_id =", args.run_id, flush=True)
    print("dry_run =", args.dry_run, flush=True)

    artifacts, context = collect_artifacts(
        args.v232_manifest_json,
        args.v238_manifest_json,
        args.expected_shared_row_contract_sha256,
    )
    prefix_parts = [part for part in args.path_prefix.strip("/").split("/") if part]
    path_in_repo = "/".join(safe_component(part) for part in [*prefix_parts, args.run_id])
    with tempfile.TemporaryDirectory() as tmp:
        staging_root = Path(tmp) / "hf_bridge_payload"
        staging_root.mkdir(parents=True, exist_ok=True)
        manifest = stage_artifacts(staging_root, artifacts, context, args.hf_dataset_repo, path_in_repo)
        print("artifact_count =", len(artifacts), flush=True)
        print("path_in_repo =", path_in_repo, flush=True)
        print("staged_files =", json.dumps(manifest["files"], indent=2, sort_keys=True), flush=True)
        if args.output_manifest_json:
            write_json(args.output_manifest_json, manifest)
            print("output_manifest_json =", args.output_manifest_json, flush=True)
            print("output_manifest_sha256 =", sha256_file(args.output_manifest_json), flush=True)
        if args.dry_run:
            upload_info = "dry_run_no_upload"
            print("upload_info =", upload_info, flush=True)
        else:
            token = args.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for upload. Set env HF_TOKEN or pass --hf-token.")
            upload_info = upload_folder(args.hf_dataset_repo, staging_root, path_in_repo, token)
            print("upload_info =", upload_info, flush=True)
        manifest["upload_info"] = upload_info
        if args.output_manifest_json:
            write_json(args.output_manifest_json, manifest)
    print("=== V240 HF ARTIFACT BRIDGE SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-manifest-json", type=Path)
    parser.add_argument("--v238-manifest-json", type=Path)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--path-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--output-manifest-json", type=Path)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        equation = root / "equation.jsonl"
        bit = root / "bit.jsonl"
        v238_results = root / "v238_results.csv"
        v238_summary = root / "v238_summary.csv"
        equation.write_text(json.dumps({"id": "eq1"}) + "\n", encoding="utf-8")
        bit.write_text(json.dumps({"id": "bit1"}) + "\n", encoding="utf-8")
        v238_results.write_text("id,status\n1,abstain\n", encoding="utf-8")
        v238_summary.write_text("status,rows\nabstain,1\n", encoding="utf-8")
        v232_manifest = root / "v232_manifest.json"
        v238_manifest = root / "v238_manifest.json"
        write_json(
            v232_manifest,
            {
                "observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "outputs": {
                    "equation_solver_workitems_jsonl": str(equation),
                    "bit_guardrail_workitems_jsonl": str(bit),
                },
            },
        )
        write_json(
            v238_manifest,
            {
                "inputs": {"observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256},
                "counts": {"deployable_verified_overrides": 1, "deployable_incorrect_overrides": 0},
                "decision": {"decision": "continue_alice_parser_development"},
                "outputs": {
                    "alice_parser_probe_results_csv": str(v238_results),
                    "alice_parser_probe_summary_csv": str(v238_summary),
                },
            },
        )
        out_manifest = root / "bridge_manifest.json"
        args = argparse.Namespace(
            v232_manifest_json=v232_manifest,
            v238_manifest_json=v238_manifest,
            hf_dataset_repo=DEFAULT_HF_DATASET_REPO,
            path_prefix=DEFAULT_PREFIX,
            run_id="selftest",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            output_manifest_json=out_manifest,
            hf_token="",
            dry_run=True,
        )
        manifest = run(args)
        if len(manifest["files"]) != 6:
            raise AssertionError("expected six staged files")
        if manifest["upload_info"] != "dry_run_no_upload":
            raise AssertionError("self-test must not upload")
        if not out_manifest.exists():
            raise AssertionError("bridge manifest was not written")
    print("v240_hf_artifact_bridge_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.v232_manifest_json is None:
        parser.error("--v232-manifest-json is required unless --self-test is used")
    if args.v238_manifest_json is None:
        parser.error("--v238-manifest-json is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

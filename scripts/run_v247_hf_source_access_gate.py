#!/usr/bin/env python3
"""Run HF source access gate for KG1 external trace candidates.

This is CPU-only and access-only. It inspects Hugging Face dataset metadata and
uses small HTTP range reads to determine whether candidate external datasets
are accessible with the current HF token. It does not train, evaluate a model,
download large payloads, package artifacts, or submit to Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_OUTPUT_PREFIX = "runtime_artifacts/v247_hf_source_access_gate"

TARGETS = [
    {
        "repo_id": "andy279/nemotron-reasoning-challenge-raw-traces",
        "repo_type": "dataset",
        "priority": "P0",
        "reason": "raw teacher/solver traces for equation_transform and bit_manipulation",
        "files": [
            "solver_transformation_traces_gpt54.jsonl",
            "solver_transformation_traces_merged.jsonl",
            "solver_bit_manipulation_traces_merged.jsonl",
        ],
    },
    {
        "repo_id": "andy279/nemotron-reasoning-challenge",
        "repo_type": "dataset",
        "priority": "P0",
        "reason": "ready SFT train/validation traces with held-out transformation puzzles",
        "files": ["sft_val.jsonl", "sft_train.jsonl"],
    },
    {
        "repo_id": "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge",
        "repo_type": "dataset",
        "priority": "P1",
        "reason": "public challenge mirror/source sanity check",
        "files": ["train.csv", "test.csv"],
    },
]

ACCESS_COLUMNS = [
    "schema_version",
    "repo_id",
    "filename",
    "priority",
    "reason",
    "metadata_status",
    "gated",
    "file_size",
    "range_status_code",
    "range_accessible",
    "range_error",
    "sample_sha256",
    "sample_preview",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() and path.is_file() else "",
    }


def token_value(cli_token: str) -> str | None:
    if cli_token or os.environ.get("HF_TOKEN"):
        return cli_token or os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def dataset_info_map(api: Any, repo_id: str, token: str | None) -> tuple[dict[str, Any], str]:
    try:
        info = api.dataset_info(repo_id=repo_id, files_metadata=True, token=token)
        siblings = getattr(info, "siblings", []) or []
        by_file: dict[str, Any] = {str(getattr(item, "rfilename", "")): item for item in siblings}
        gated = str(getattr(info, "gated", ""))
        return {"gated": gated, "siblings": by_file, "sha": str(getattr(info, "sha", ""))}, "ok"
    except Exception as exc:
        return {"gated": "", "siblings": {}, "sha": ""}, repr(exc)


def sibling_size(sibling: Any) -> int:
    for attr in ("size", "lfs_size"):
        value = getattr(sibling, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return 0


def range_read(repo_id: str, filename: str, token: str | None, max_bytes: int) -> tuple[int, bool, str, bytes]:
    try:
        import requests
        from huggingface_hub import hf_hub_url
    except ImportError as exc:
        return 0, False, "missing_dependency:" + repr(exc), b""

    headers = {"Range": f"bytes=0-{max_bytes - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = hf_hub_url(repo_id=repo_id, repo_type="dataset", filename=filename)
    try:
        with requests.get(url, headers=headers, timeout=45, stream=True, allow_redirects=True) as response:
            status = int(response.status_code)
            if status not in {200, 206}:
                text = response.text[:500]
                return status, False, text.replace("\n", " "), b""
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            data = b"".join(chunks)[:max_bytes]
            return status, True, "", data
    except Exception as exc:
        return 0, False, repr(exc), b""


def normalize_preview(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    text = " ".join(text.replace("\r", "\n").split())
    return text[:500]


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V247 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V247 HF source access gate {path_in_repo.strip('/')}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V247 HF SOURCE ACCESS GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN is required for source access gate")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for source access gate") from exc
    api = HfApi(token=token)

    rows: list[dict[str, Any]] = []
    repo_status: dict[str, Any] = {}
    for target in TARGETS:
        repo_id = str(target["repo_id"])
        print("source_repo_start =", repo_id, flush=True)
        info, metadata_status = dataset_info_map(api, repo_id, token)
        repo_status[repo_id] = {
            "metadata_status": metadata_status,
            "gated": info.get("gated", ""),
            "sha": info.get("sha", ""),
        }
        siblings = info.get("siblings", {})
        if not isinstance(siblings, dict):
            siblings = {}
        print("source_repo_metadata =", json.dumps(repo_status[repo_id], sort_keys=True), flush=True)
        for filename in target["files"]:
            sibling = siblings.get(filename)
            file_size = sibling_size(sibling) if sibling is not None else 0
            status_code, accessible, error, sample = range_read(repo_id, filename, token, args.max_sample_bytes)
            row = {
                "schema_version": "kg1_v247_hf_source_access_gate_row_v1",
                "repo_id": repo_id,
                "filename": filename,
                "priority": target["priority"],
                "reason": target["reason"],
                "metadata_status": metadata_status,
                "gated": info.get("gated", ""),
                "file_size": file_size,
                "range_status_code": status_code,
                "range_accessible": accessible,
                "range_error": error,
                "sample_sha256": sha256_bytes(sample) if sample else "",
                "sample_preview": normalize_preview(sample) if sample else "",
            }
            rows.append(row)
            print("source_file_access =", json.dumps({k: row[k] for k in row if k != "sample_preview"}, sort_keys=True), flush=True)

    p0_rows = [row for row in rows if row["priority"] == "P0"]
    p0_accessible = [row for row in p0_rows if row["range_accessible"]]
    p0_denied = [row for row in p0_rows if not row["range_accessible"]]
    public_rows = [row for row in rows if row["priority"] == "P1"]
    if p0_accessible:
        decision = {
            "decision": "prepare_external_trace_ingestion",
            "reason": f"p0_accessible_files={len(p0_accessible)}; p0_denied_files={len(p0_denied)}",
            "next_action": "Create a trace ingestion job that samples/filters only equation_transform and bit_manipulation examples.",
        }
    elif public_rows and any(row["range_accessible"] for row in public_rows):
        decision = {
            "decision": "p0_gated_terms_required_public_mirror_available",
            "reason": f"p0_accessible_files=0; p0_denied_files={len(p0_denied)}; public_accessible_files={sum(bool(row['range_accessible']) for row in public_rows)}",
            "next_action": "Human must accept HF gated terms for andy279 datasets; public mirror is source sanity only.",
        }
    else:
        decision = {
            "decision": "source_access_blocked",
            "reason": f"p0_accessible_files=0; p0_denied_files={len(p0_denied)}",
            "next_action": "Human must fix HF token/dataset access before external-trace route can proceed.",
        }

    outputs = {
        "access_csv": args.output_dir / f"{args.label}_access.csv",
        "manifest_json": args.output_dir / f"{args.label}_manifest.json",
    }
    write_csv(outputs["access_csv"], rows, ACCESS_COLUMNS)
    manifest = {
        "schema_version": "kg1_v247_hf_source_access_gate_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "max_sample_bytes": args.max_sample_bytes,
            "targets": TARGETS,
        },
        "repo_status": repo_status,
        "counts": {
            "rows": len(rows),
            "p0_rows": len(p0_rows),
            "p0_accessible_files": len(p0_accessible),
            "p0_denied_files": len(p0_denied),
            "public_accessible_files": sum(bool(row["range_accessible"]) for row in public_rows),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {"access_csv": file_meta(outputs["access_csv"])},
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(outputs["manifest_json"], manifest)

    upload_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.hf_dataset_repo, args.output_dir, args.output_path_in_repo, token)
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.hf_dataset_repo,
        "path_in_repo": str(args.output_path_in_repo or ""),
        "upload_info": upload_info,
    }
    write_json(outputs["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V247 HF SOURCE ACCESS GATE END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v247_hf_source_access_gate")
    parser.add_argument("--max-sample-bytes", type=int, default=2048)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    preview = normalize_preview(b"{\"a\": 1}\n{\"b\": 2}\n")
    if "{\"a\": 1}" not in preview:
        raise AssertionError("preview normalization failed")
    if len(sha256_bytes(b"x")) != 64:
        raise AssertionError("sha256 failed")
    print("v247_hf_source_access_gate_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

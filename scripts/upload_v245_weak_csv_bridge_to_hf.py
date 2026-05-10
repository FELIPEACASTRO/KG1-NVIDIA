#!/usr/bin/env python3
"""Validate and upload the KG1 315-row weak eval CSV to a HF dataset.

This bridge is intentionally small and CPU-only. It exists because HF Jobs
cannot mount Google Drive, while V245 weak evaluation needs the exact V221
weak row contract before spending H200/vLLM time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_PREFIX = "runtime_artifacts/v245_weak_eval_bridge"
WEAK_FAMILIES = {"bit_manipulation", "equation_transform"}
EXPECTED_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_and_validate_weak_csv(path: Path, expected_contract: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    frame = pd.read_csv(path)
    required = {"id", "prompt", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"weak CSV missing required columns: {missing}")
    family_col = "family" if "family" in frame.columns else "type" if "type" in frame.columns else ""
    if not family_col:
        raise RuntimeError("weak CSV must include either family or type column")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["prompt"] = frame["prompt"].astype(str)
    frame["answer"] = frame["answer"].astype(str)
    frame["family"] = frame[family_col].astype(str)
    non_weak = sorted(set(frame["family"]) - WEAK_FAMILIES)
    if non_weak:
        raise RuntimeError(f"weak CSV contains non-weak families: {non_weak}")
    if int(frame["id"].duplicated().sum()):
        duplicates = frame.loc[frame["id"].duplicated(), "id"].head(10).tolist()
        raise RuntimeError("weak CSV contains duplicate ids: " + json.dumps(duplicates))
    counts = {str(k): int(v) for k, v in frame["family"].value_counts().sort_index().to_dict().items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"weak family counts mismatch: expected {EXPECTED_COUNTS}, got {counts}")
    if len(frame) != 315:
        raise RuntimeError(f"weak CSV row count mismatch: expected 315, got {len(frame)}")
    frame["prompt_sha256"] = frame["prompt"].map(sha256_text)
    records = {
        str(row.id): (str(row.family), str(row.answer), str(row.prompt_sha256))
        for row in frame.itertuples(index=False)
    }
    digest_payload = "\n".join(
        f"{row_id}\t{family}\t{answer}\t{prompt_hash}"
        for row_id, (family, answer, prompt_hash) in sorted(records.items())
    )
    observed_contract = sha256_text(digest_payload)
    if expected_contract and observed_contract != expected_contract:
        raise RuntimeError(f"weak row contract mismatch: expected {expected_contract}, got {observed_contract}")
    canonical_cols = [col for col in ["id", "prompt", "answer", "type", "family"] if col in frame.columns]
    canonical = frame[canonical_cols].copy()
    if "type" not in canonical.columns:
        canonical["type"] = frame["family"]
    canonical = canonical[["id", "prompt", "answer", "type"]]
    meta = {
        "source_csv": str(path),
        "source_sha256": sha256_file(path),
        "rows": int(len(canonical)),
        "family_counts": counts,
        "observed_shared_row_contract_sha256": observed_contract,
    }
    return canonical, meta


def upload_folder(repo_id: str, folder_path: Path, path_in_repo: str, token: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        commit_message=f"Upload KG1 V245 weak eval bridge {path_in_repo}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V245 WEAK CSV HF BRIDGE SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("source_csv =", args.source_csv, flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("path_prefix =", args.path_prefix, flush=True)
    print("run_id =", args.run_id, flush=True)
    print("dry_run =", args.dry_run, flush=True)
    weak_df, source_meta = load_and_validate_weak_csv(
        args.source_csv,
        args.expected_shared_row_contract_sha256,
    )
    path_in_repo = "/".join(part.strip("/") for part in [args.path_prefix, args.run_id] if part.strip("/"))
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "v245_weak_eval_bridge"
        staging.mkdir(parents=True, exist_ok=True)
        weak_csv = staging / "v221_weak_315.csv"
        manifest_path = staging / "v245_weak_eval_bridge_manifest.json"
        weak_df.to_csv(weak_csv, index=False)
        manifest = {
            "schema_version": "kg1_v245_weak_eval_bridge_manifest_v1",
            "generated_at_utc": utc_now(),
            "hf_dataset_repo": args.hf_dataset_repo,
            "path_in_repo": path_in_repo,
            "source": source_meta,
            "uploaded_files": {
                "weak_csv": f"{path_in_repo}/v221_weak_315.csv",
                "manifest_json": f"{path_in_repo}/v245_weak_eval_bridge_manifest.json",
            },
            "canonical_weak_csv": {
                "bytes": weak_csv.stat().st_size,
                "sha256": sha256_file(weak_csv),
                "rows": int(len(weak_df)),
                "family_counts": source_meta["family_counts"],
                "observed_shared_row_contract_sha256": source_meta["observed_shared_row_contract_sha256"],
            },
            "blocked_actions": ["train", "full_eval", "package", "kaggle_submit"],
        }
        write_json(manifest_path, manifest)
        print("canonical_weak_csv =", weak_csv, flush=True)
        print("canonical_weak_csv_sha256 =", manifest["canonical_weak_csv"]["sha256"], flush=True)
        print(
            "observed_shared_row_contract_sha256 =",
            source_meta["observed_shared_row_contract_sha256"],
            flush=True,
        )
        if args.output_manifest_json:
            write_json(args.output_manifest_json, manifest)
            print("output_manifest_json =", args.output_manifest_json, flush=True)
        if args.dry_run:
            upload_info = "dry_run_no_upload"
        else:
            token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required unless --dry-run is used")
            upload_info = upload_folder(args.hf_dataset_repo, staging, path_in_repo, token)
        print("upload_info =", upload_info, flush=True)
        manifest["upload_info"] = upload_info
        if args.output_manifest_json:
            write_json(args.output_manifest_json, manifest)
    print("=== V245 WEAK CSV HF BRIDGE SCRIPT END ===", flush=True)
    return manifest


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_path = root / "weak.csv"
        rows = []
        for idx in range(160):
            rows.append({"id": f"bit-{idx:03d}", "prompt": f"bit prompt {idx}", "answer": str(idx), "type": "bit_manipulation"})
        for idx in range(155):
            rows.append({"id": f"eq-{idx:03d}", "prompt": f"eq prompt {idx}", "answer": str(idx), "type": "equation_transform"})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _, meta = load_and_validate_weak_csv(csv_path, expected_contract="")
        out_manifest = root / "manifest.json"
        args = argparse.Namespace(
            source_csv=csv_path,
            hf_dataset_repo=DEFAULT_HF_DATASET_REPO,
            path_prefix=DEFAULT_PREFIX,
            run_id="selftest",
            expected_shared_row_contract_sha256=meta["observed_shared_row_contract_sha256"],
            output_manifest_json=out_manifest,
            hf_token="",
            dry_run=True,
        )
        manifest = run(args)
        assert manifest["upload_info"] == "dry_run_no_upload"
        assert out_manifest.exists()
    print("v245_weak_csv_bridge_self_test=ok", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=False)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--path-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--output-manifest-json", type=Path)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.source_csv is None:
        parser.error("--source-csv is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

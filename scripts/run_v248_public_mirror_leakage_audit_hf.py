#!/usr/bin/env python3
"""Audit public HF challenge mirror for weak leakage and usable coverage.

This is CPU-only and data-audit-only. It downloads the public mirror
`jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge` plus the canonical
V245 weak CSV bridge from the private KG1 HF dataset, then emits leakage and
coverage manifests. It does not train, generate model outputs, package, or
submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_PUBLIC_REPO = "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge"
DEFAULT_WEAK_CSV_PATH = "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
EXPECTED_WEAK_ROWS = 315
EXPECTED_WEAK_FAMILY_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}

FAMILY_SUMMARY_COLUMNS = [
    "schema_version",
    "family",
    "train_rows",
    "weak_overlap_rows",
    "nonweak_train_rows",
    "test_rows",
]
LEAKAGE_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "public_answer",
    "weak_answer",
    "answer_match",
    "prompt_match_normalized",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def token_value(cli_token: str) -> str | None:
    if cli_token or os.environ.get("HF_TOKEN"):
        return cli_token or os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def download_file(repo_id: str, filename: str, repo_type: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V248") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def classify_family(prompt: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    text = str(prompt).lower()
    if "bit manipulation" in text or "8-bit binary" in text or "input -> output" in text:
        return "bit_manipulation"
    if "transformation rules is applied to equations" in text:
        return "equation_transform"
    if "cipher" in text or "encrypted" in text or "encryption" in text:
        return "text_encryption"
    if "base-" in text or "base " in text or "numeral" in text:
        return "numeral_system"
    if "gravity" in text:
        return "gravity_constant"
    if "unit conversion" in text or "convert" in text:
        return "unit_conversion"
    return "unknown"


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V248 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V248 public mirror leakage audit {path_in_repo.strip('/')}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V248 PUBLIC MIRROR LEAKAGE AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("public_repo =", args.public_repo, flush=True)
    print("weak_csv_path =", args.weak_csv_path, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN is required for V248")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "download"
        train_path = download_file(args.public_repo, "train.csv", "dataset", download_root, token)
        test_path = download_file(args.public_repo, "test.csv", "dataset", download_root, token)
        weak_path = download_file(args.hf_dataset_repo, args.weak_csv_path, "dataset", download_root, token)

        print("train_path =", train_path, flush=True)
        print("test_path =", test_path, flush=True)
        print("weak_path =", weak_path, flush=True)
        weak_sha = sha256_file(weak_path)
        print("weak_csv_sha256 =", weak_sha, flush=True)
        if args.expected_weak_csv_sha256 and weak_sha != args.expected_weak_csv_sha256:
            raise RuntimeError(f"weak CSV SHA mismatch: expected {args.expected_weak_csv_sha256}, got {weak_sha}")

        train_rows = read_csv_rows(train_path)
        test_rows = read_csv_rows(test_path)
        weak_rows = read_csv_rows(weak_path)
        download_artifact_hashes = {
            "public_train_csv": file_meta(train_path),
            "public_test_csv": file_meta(test_path),
            "weak_csv": file_meta(weak_path),
        }

    if len(weak_rows) != args.expected_weak_rows:
        raise RuntimeError(f"weak row count mismatch: expected {args.expected_weak_rows}, got {len(weak_rows)}")
    expected_weak_family_counts = (
        json.loads(args.expected_weak_family_counts)
        if isinstance(args.expected_weak_family_counts, str)
        else dict(args.expected_weak_family_counts)
    )

    train_by_id = {row["id"]: row for row in train_rows}
    test_by_id = {row["id"]: row for row in test_rows}
    weak_by_id = {row["id"]: row for row in weak_rows}
    family_by_id: dict[str, str] = {}
    for row in train_rows:
        family_by_id[row["id"]] = classify_family(row.get("prompt", ""))
    for row in weak_rows:
        family_by_id[row["id"]] = classify_family(row.get("prompt", ""), row.get("type", ""))
    for row in test_rows:
        family_by_id[row["id"]] = classify_family(row.get("prompt", ""))

    weak_family_counts = Counter(classify_family(row.get("prompt", ""), row.get("type", "")) for row in weak_rows)
    if dict(weak_family_counts) != expected_weak_family_counts:
        raise RuntimeError(
            "weak family counts mismatch: expected "
            + json.dumps(expected_weak_family_counts, sort_keys=True)
            + ", got "
            + json.dumps(dict(weak_family_counts), sort_keys=True)
        )

    overlap_ids = sorted(set(train_by_id) & set(weak_by_id))
    leakage_rows: list[dict[str, Any]] = []
    for row_id in overlap_ids:
        public_row = train_by_id[row_id]
        weak_row = weak_by_id[row_id]
        leakage_rows.append(
            {
                "schema_version": "kg1_v248_public_mirror_leakage_row_v1",
                "id": row_id,
                "family": family_by_id.get(row_id, "unknown"),
                "public_answer": public_row.get("answer", ""),
                "weak_answer": weak_row.get("answer", ""),
                "answer_match": normalize_answer(public_row.get("answer", "")) == normalize_answer(weak_row.get("answer", "")),
                "prompt_match_normalized": normalize_text(public_row.get("prompt", "")) == normalize_text(weak_row.get("prompt", "")),
            }
        )

    family_names = sorted(set(family_by_id.values()) | set(EXPECTED_WEAK_FAMILY_COUNTS))
    family_summary: list[dict[str, Any]] = []
    train_counter = Counter(family_by_id[row["id"]] for row in train_rows)
    weak_overlap_counter = Counter(family_by_id[row_id] for row_id in overlap_ids)
    test_counter = Counter(family_by_id[row["id"]] for row in test_rows)
    for family in family_names:
        family_summary.append(
            {
                "schema_version": "kg1_v248_family_summary_v1",
                "family": family,
                "train_rows": train_counter.get(family, 0),
                "weak_overlap_rows": weak_overlap_counter.get(family, 0),
                "nonweak_train_rows": train_counter.get(family, 0) - weak_overlap_counter.get(family, 0),
                "test_rows": test_counter.get(family, 0),
            }
        )

    nonweak_target_rows = [
        row
        for row in train_rows
        if row["id"] not in weak_by_id and family_by_id.get(row["id"]) in {"equation_transform", "bit_manipulation"}
    ]
    weak_leakage_is_exact = (
        len(overlap_ids) == args.expected_weak_rows
        and all(bool(row["answer_match"]) for row in leakage_rows)
    )

    if weak_leakage_is_exact and nonweak_target_rows:
        decision = {
            "decision": "public_mirror_usable_only_after_weak_id_exclusion",
            "reason": f"weak_overlap={len(overlap_ids)}; nonweak_target_rows={len(nonweak_target_rows)}",
            "next_action": "Use public mirror only for non-weak target-family training/evaluation design; never use weak-overlap labels for tuning.",
        }
    elif weak_leakage_is_exact:
        decision = {
            "decision": "public_mirror_leakage_only_no_target_surplus",
            "reason": f"weak_overlap={len(overlap_ids)}; nonweak_target_rows=0",
            "next_action": "Do not use this mirror for score improvement.",
        }
    else:
        decision = {
            "decision": "public_mirror_contract_untrusted",
            "reason": f"weak_overlap={len(overlap_ids)}; exact_answer_match={weak_leakage_is_exact}",
            "next_action": "Do not use public mirror until discrepancy is resolved.",
        }

    outputs = {
        "family_summary_csv": args.output_dir / f"{args.label}_family_summary.csv",
        "weak_leakage_csv": args.output_dir / f"{args.label}_weak_leakage.csv",
        "manifest_json": args.output_dir / f"{args.label}_manifest.json",
    }
    write_csv(outputs["family_summary_csv"], family_summary, FAMILY_SUMMARY_COLUMNS)
    write_csv(outputs["weak_leakage_csv"], leakage_rows, LEAKAGE_COLUMNS)

    manifest = {
        "schema_version": "kg1_v248_public_mirror_leakage_audit_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "public_repo": args.public_repo,
            "weak_csv_path": args.weak_csv_path,
            "expected_weak_csv_sha256": args.expected_weak_csv_sha256,
        },
        "download_artifact_hashes": {
            **download_artifact_hashes,
        },
        "counts": {
            "public_train_rows": len(train_rows),
            "public_test_rows": len(test_rows),
            "weak_rows": len(weak_rows),
            "weak_overlap_rows": len(overlap_ids),
            "weak_overlap_answer_mismatch_rows": sum(not bool(row["answer_match"]) for row in leakage_rows),
            "weak_overlap_prompt_mismatch_rows": sum(not bool(row["prompt_match_normalized"]) for row in leakage_rows),
            "nonweak_target_rows": len(nonweak_target_rows),
        },
        "family_summary": family_summary,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in outputs.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
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
    print("family_summary =", json.dumps(family_summary, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V248 PUBLIC MIRROR LEAKAGE AUDIT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--public-repo", default=DEFAULT_PUBLIC_REPO)
    parser.add_argument("--weak-csv-path", default=DEFAULT_WEAK_CSV_PATH)
    parser.add_argument("--expected-weak-csv-sha256", default=EXPECTED_WEAK_CSV_SHA256)
    parser.add_argument("--expected-weak-rows", type=int, default=EXPECTED_WEAK_ROWS)
    parser.add_argument("--expected-weak-family-counts", type=json.loads, default=EXPECTED_WEAK_FAMILY_COUNTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v248_public_mirror_leakage_audit")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    if classify_family("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.") != "bit_manipulation":
        raise AssertionError("bit family classification failed")
    if classify_family("In Alice's Wonderland, a secret set of transformation rules is applied to equations.") != "equation_transform":
        raise AssertionError("equation family classification failed")
    if normalize_answer(" 1 2 ") != "12":
        raise AssertionError("answer normalization failed")
    print("v248_public_mirror_leakage_audit_self_test=ok", flush=True)
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

#!/usr/bin/env python3
"""Build V249 non-weak target-family dataset from the public HF mirror.

This is CPU-only data preparation. It excludes every canonical weak ID before
writing train/validation JSONL files for bit_manipulation and
equation_transform. It does not train, run model generation, evaluate, package,
or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
EXPECTED_TARGET_TOTAL = 2842
EXPECTED_TARGET_FAMILY_COUNTS = {"bit_manipulation": 1442, "equation_transform": 1400}
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)
ID_COLUMNS = ["schema_version", "id", "family", "split"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
        raise RuntimeError("huggingface_hub is required for V249") from exc
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
    return "other"


def make_example(row: dict[str, str], family: str, split: str) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip()
    if not answer:
        raise RuntimeError(f"missing answer for row {row.get('id')}")
    prompt = str(row.get("prompt", "")).strip()
    row_id = str(row.get("id", "")).strip()
    return {
        "id": f"v249_{split}_{row_id}",
        "original_id": row_id,
        "family": family,
        "subcategory": family,
        "source": "jasonkung98_public_mirror_nonweak_target",
        "prompt": prompt,
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Final answer: {answer}"},
        ],
        "metadata": {
            "source": "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge",
            "original_id": row_id,
            "weak_id_excluded": True,
            "split": split,
            "family": family,
            "v249_role": split,
            "answer_style": "final_answer_one_line_unboxed",
        },
    }


def stratified_split(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for family in sorted(by_family):
        items = list(by_family[family])
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_fraction)))
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    for row in train_rows:
        row["id"] = row["id"].replace("v249_candidate_", "v249_train_")
        row["metadata"]["split"] = "train"
        row["metadata"]["v249_role"] = "train"
    for row in val_rows:
        row["id"] = row["id"].replace("v249_candidate_", "v249_val_")
        row["metadata"]["split"] = "validation"
        row["metadata"]["v249_role"] = "validation"
    return train_rows, val_rows


def validate_jsonl_rows(rows: list[dict[str, Any]], split: str, weak_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [str(row.get("id", "")) for row in rows]
    original_ids = [str(row.get("original_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_ids")
    if set(original_ids) & weak_ids:
        errors.append("weak_id_overlap")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            errors.append(f"bad_messages:{row.get('id')}")
            continue
        if messages[-1].get("role") != "assistant":
            errors.append(f"missing_assistant:{row.get('id')}")
        expected = f"Final answer: {row.get('answer', '')}"
        if messages[-1].get("content") != expected:
            errors.append(f"assistant_mismatch:{row.get('id')}")
    if errors:
        raise RuntimeError(f"{split} validation failed: {errors[:20]}")
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "family_counts": dict(Counter(str(row.get("family", "")) for row in rows)),
        "assistant_chars": {
            "min": min(len(row["messages"][-1]["content"]) for row in rows) if rows else 0,
            "max": max(len(row["messages"][-1]["content"]) for row in rows) if rows else 0,
        },
    }


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V249 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V249 public nonweak target dataset {path_in_repo.strip('/')}",
    )
    return str(info)


def upload_manifest(repo_id: str, manifest_path: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V249 manifest") from exc
    api = HfApi(token=token)
    info = api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(manifest_path),
        path_in_repo=(path_in_repo.strip("/") + "/" + manifest_path.name).strip("/"),
        commit_message=f"Refresh KG1 V249 manifest {path_in_repo.strip('/')}",
    )
    return str(info)


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V249 PUBLIC NONWEAK TARGET DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("public_repo =", args.public_repo, flush=True)
    print("weak_csv_path =", args.weak_csv_path, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("val_fraction =", args.val_fraction, flush=True)
    print("seed =", args.seed, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN is required for V249")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "download"
        train_csv = download_file(args.public_repo, "train.csv", "dataset", download_root, token)
        weak_csv = download_file(args.hf_dataset_repo, args.weak_csv_path, "dataset", download_root, token)
        print("train_csv =", train_csv, flush=True)
        print("weak_csv =", weak_csv, flush=True)
        weak_sha = sha256_file(weak_csv)
        print("weak_csv_sha256 =", weak_sha, flush=True)
        if args.expected_weak_csv_sha256 and weak_sha != args.expected_weak_csv_sha256:
            raise RuntimeError(f"weak CSV SHA mismatch: expected {args.expected_weak_csv_sha256}, got {weak_sha}")
        public_rows = read_csv_rows(train_csv)
        weak_rows = read_csv_rows(weak_csv)
        download_artifact_hashes = {"public_train_csv": file_meta(train_csv), "weak_csv": file_meta(weak_csv)}

    weak_ids = {row["id"] for row in weak_rows}
    candidate_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for row in public_rows:
        row_id = str(row.get("id", ""))
        family = classify_family(str(row.get("prompt", "")))
        if row_id in weak_ids:
            blocked_rows.append({"schema_version": "kg1_v249_blocked_id_v1", "id": row_id, "family": family, "split": "weak_overlap"})
            continue
        if family not in {"bit_manipulation", "equation_transform"}:
            continue
        candidate_rows.append(make_example(row, family, "candidate"))

    target_counts = Counter(str(row["family"]) for row in candidate_rows)
    if len(candidate_rows) != args.expected_target_total:
        raise RuntimeError(f"target row count mismatch: expected {args.expected_target_total}, got {len(candidate_rows)}")
    expected_counts = (
        json.loads(args.expected_target_family_counts)
        if isinstance(args.expected_target_family_counts, str)
        else dict(args.expected_target_family_counts)
    )
    if dict(target_counts) != expected_counts:
        raise RuntimeError(
            "target family counts mismatch: expected "
            + json.dumps(expected_counts, sort_keys=True)
            + ", got "
            + json.dumps(dict(target_counts), sort_keys=True)
        )

    train_rows, val_rows = stratified_split(candidate_rows, args.val_fraction, args.seed)
    train_validation = validate_jsonl_rows(train_rows, "train", weak_ids)
    val_validation = validate_jsonl_rows(val_rows, "validation", weak_ids)

    outputs = {
        "train_jsonl": args.output_dir / "v249_public_nonweak_target_train.jsonl",
        "val_jsonl": args.output_dir / "v249_public_nonweak_target_val.jsonl",
        "blocked_weak_ids_csv": args.output_dir / "v249_blocked_weak_ids.csv",
        "manifest_json": args.output_dir / "v249_public_nonweak_target_manifest.json",
    }
    write_jsonl(outputs["train_jsonl"], train_rows)
    write_jsonl(outputs["val_jsonl"], val_rows)
    write_csv(outputs["blocked_weak_ids_csv"], blocked_rows, ID_COLUMNS)

    decision = {
        "decision": "dataset_ready_for_tokenization_gate_not_training_yet",
        "reason": f"train_rows={len(train_rows)}; val_rows={len(val_rows)}; weak_overlap=0",
        "next_action": "Run tokenizer/mask dry-run and compare against V217/V226 before any GPU training.",
    }
    manifest = {
        "schema_version": "kg1_v249_public_nonweak_target_dataset_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "public_repo": args.public_repo,
            "weak_csv_path": args.weak_csv_path,
            "expected_weak_csv_sha256": args.expected_weak_csv_sha256,
            "expected_target_total": args.expected_target_total,
            "expected_target_family_counts": expected_counts,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "download_artifact_hashes": download_artifact_hashes,
        "counts": {
            "public_train_rows": len(public_rows),
            "weak_rows": len(weak_rows),
            "blocked_weak_overlap_rows": len(blocked_rows),
            "candidate_target_rows": len(candidate_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_family_counts": train_validation["family_counts"],
            "val_family_counts": val_validation["family_counts"],
        },
        "train_validation": train_validation,
        "val_validation": val_validation,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {name: file_meta(path) for name, path in outputs.items() if name != "manifest_json"},
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    upload_info = "upload_disabled"
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.hf_dataset_repo,
        "path_in_repo": str(args.output_path_in_repo or ""),
        "upload_info": upload_info,
        "manifest_uploaded_after_folder_upload": False,
    }
    write_json(outputs["manifest_json"], manifest)
    manifest_refresh_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.hf_dataset_repo, args.output_dir, args.output_path_in_repo, token)
        manifest["hf_upload"]["upload_info"] = upload_info
        manifest["hf_upload"]["manifest_uploaded_after_folder_upload"] = True
        write_json(outputs["manifest_json"], manifest)
        manifest_refresh_info = upload_manifest(
            args.hf_dataset_repo, outputs["manifest_json"], args.output_path_in_repo, token
        )
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("manifest_refresh_info =", manifest_refresh_info, flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V249 PUBLIC NONWEAK TARGET DATASET END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--public-repo", default=DEFAULT_PUBLIC_REPO)
    parser.add_argument("--weak-csv-path", default=DEFAULT_WEAK_CSV_PATH)
    parser.add_argument("--expected-weak-csv-sha256", default=EXPECTED_WEAK_CSV_SHA256)
    parser.add_argument("--expected-target-total", type=int, default=EXPECTED_TARGET_TOTAL)
    parser.add_argument("--expected-target-family-counts", type=json.loads, default=EXPECTED_TARGET_FAMILY_COUNTS)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v249_public_nonweak_target_dataset")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    row = {
        "id": "abc",
        "prompt": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        "answer": "0101",
    }
    ex = make_example(row, "bit_manipulation", "candidate")
    if ex["messages"][-1]["content"] != "Final answer: 0101":
        raise AssertionError("assistant format failed")
    if classify_family(row["prompt"]) != "bit_manipulation":
        raise AssertionError("family classification failed")
    validate_jsonl_rows([ex], "self_test", set())
    print("v249_public_nonweak_target_dataset_self_test=ok", flush=True)
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

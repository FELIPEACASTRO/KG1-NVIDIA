#!/usr/bin/env python3
"""Build a leak-filtered V265 target-family mix from historical score-0.86 data.

This is CPU-only data preparation. It combines the already validated V249
public non-weak target data with the historical V189 score-improvement corpus,
but blocks every canonical weak ID and every normalized weak prompt hash before
writing train/validation JSONL files.

The script intentionally writes V249-compatible filenames so the existing V250
tokenization gate can validate the output with custom hashes and row counts.
It does not train, run generation, evaluate, package, or submit.
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
DEFAULT_WEAK_CSV_PATH = "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
DEFAULT_V249_PATH = "data/v249_public_nonweak_target/v249-hf-public-nonweak-target-20260510T210850Z"
DEFAULT_V189_PATH = "data/v189_equation_answer_short/submission_gain_train.jsonl"
EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)
BLOCKED_COLUMNS = ["schema_version", "id", "family", "split"]
LEAK_COLUMNS = ["schema_version", "source_file", "row_id", "family", "reason", "prompt_sha256"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        raise RuntimeError("huggingface_hub is required for V265") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V265 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V265 leak-filtered target mix {path_in_repo.strip('/')}",
    )
    return str(info)


def upload_manifest(repo_id: str, manifest_path: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V265 manifest") from exc
    api = HfApi(token=token)
    info = api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(manifest_path),
        path_in_repo=(path_in_repo.strip("/") + "/" + manifest_path.name).strip("/"),
        commit_message=f"Refresh KG1 V265 manifest {path_in_repo.strip('/')}",
    )
    return str(info)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def prompt_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_prompt(value).encode("utf-8")).hexdigest()


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def prompt_answer_sha256(prompt: Any, answer: Any) -> str:
    raw = normalize_prompt(prompt) + "\0" + normalize_answer(answer)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def classify_family(prompt: Any, fallback: str = "") -> str:
    if fallback:
        return str(fallback)
    text = str(prompt).lower()
    if "bit manipulation" in text or "8-bit binary" in text or "input -> output" in text:
        return "bit_manipulation"
    if "transformation rules is applied to equations" in text:
        return "equation_transform"
    return "other"


def make_example(
    *,
    row_id: str,
    prompt: str,
    answer: str,
    family: str,
    source: str,
    split: str,
    source_file: str,
) -> dict[str, Any]:
    if family not in {"bit_manipulation", "equation_transform"}:
        raise RuntimeError(f"unsupported target family: {family}")
    if not prompt.strip() or not answer.strip():
        raise RuntimeError(f"empty prompt or answer for {row_id}")
    safe_source = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower()[:48]
    return {
        "id": f"v265_{split}_{safe_source}_{row_id}",
        "original_id": row_id,
        "family": family,
        "subcategory": family,
        "source": source,
        "prompt": prompt.strip(),
        "answer": answer.strip(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": f"Final answer: {answer.strip()}"},
        ],
        "metadata": {
            "source": source,
            "source_file": source_file,
            "original_id": row_id,
            "weak_id_excluded": True,
            "weak_prompt_sha256_excluded": True,
            "split": split,
            "family": family,
            "v265_role": split,
            "answer_style": "final_answer_one_line_unboxed",
            "prompt_sha256": prompt_sha256(prompt),
            "prompt_answer_sha256": prompt_answer_sha256(prompt, answer),
        },
    }


def load_v249_rows(path: Path, source_file: str) -> list[dict[str, Any]]:
    rows = read_jsonl_rows(path)
    examples: list[dict[str, Any]] = []
    for row in rows:
        family = classify_family(row.get("prompt", ""), str(row.get("family", "")))
        if family not in {"bit_manipulation", "equation_transform"}:
            continue
        examples.append(
            make_example(
                row_id=str(row.get("original_id") or row.get("id", "")),
                prompt=str(row.get("prompt", "")),
                answer=str(row.get("answer", "")),
                family=family,
                source=str(row.get("source") or "v249_public_nonweak_target"),
                split="candidate",
                source_file=source_file,
            )
        )
    return examples


def load_v189_rows(path: Path, source_file: str) -> list[dict[str, Any]]:
    rows = read_jsonl_rows(path)
    examples: list[dict[str, Any]] = []
    for row in rows:
        family = classify_family(row.get("prompt", ""), str(row.get("family", "")))
        if family not in {"equation_transform"}:
            continue
        examples.append(
            make_example(
                row_id=str(row.get("id", "")),
                prompt=str(row.get("prompt", "")),
                answer=str(row.get("answer", "")),
                family=family,
                source="v189_score086_equation_answer_short_filtered",
                split="candidate",
                source_file=source_file,
            )
        )
    return examples


def filter_rows(
    rows: list[dict[str, Any]],
    weak_ids: set[str],
    weak_prompt_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    for row in rows:
        original_id = str(row.get("original_id", ""))
        p_hash = str(row.get("metadata", {}).get("prompt_sha256") or prompt_sha256(row.get("prompt", "")))
        reason = ""
        if original_id in weak_ids:
            reason = "weak_id_overlap"
        elif p_hash in weak_prompt_hashes:
            reason = "weak_prompt_sha256_overlap"
        elif p_hash in seen_prompt_hashes:
            reason = "duplicate_prompt_sha256"
        if reason:
            blocked.append(
                {
                    "schema_version": "kg1_v265_leak_block_row_v1",
                    "source_file": str(row.get("metadata", {}).get("source_file", "")),
                    "row_id": original_id,
                    "family": row.get("family", ""),
                    "reason": reason,
                    "prompt_sha256": p_hash,
                }
            )
            continue
        seen_prompt_hashes.add(p_hash)
        kept.append(row)
    return kept, blocked


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((str(row["family"]), str(row["source"])), []).append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for key in sorted(by_group):
        items = list(by_group[key])
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_fraction))) if len(items) >= 10 else 0
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    for row in train_rows:
        row["id"] = row["id"].replace("_candidate_", "_train_")
        row["metadata"]["split"] = "train"
        row["metadata"]["v265_role"] = "train"
    for row in val_rows:
        row["id"] = row["id"].replace("_candidate_", "_val_")
        row["metadata"]["split"] = "validation"
        row["metadata"]["v265_role"] = "validation"
    return train_rows, val_rows


def validate_rows(rows: list[dict[str, Any]], split: str, weak_ids: set[str], weak_prompt_hashes: set[str]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    original_ids = [str(row.get("original_id", "")) for row in rows]
    prompt_hashes = [str(row.get("metadata", {}).get("prompt_sha256") or prompt_sha256(row.get("prompt", ""))) for row in rows]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate_ids")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        errors.append("duplicate_prompt_sha256")
    weak_id_overlap = sorted(set(original_ids) & weak_ids)
    if weak_id_overlap:
        errors.append("weak_id_overlap=" + ",".join(weak_id_overlap[:10]))
    weak_prompt_overlap = sorted(set(prompt_hashes) & weak_prompt_hashes)
    if weak_prompt_overlap:
        errors.append("weak_prompt_sha256_overlap=" + ",".join(weak_prompt_overlap[:5]))
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            errors.append(f"bad_messages:{row.get('id')}")
            continue
        if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            errors.append(f"bad_roles:{row.get('id')}")
        if messages[1].get("content") != row.get("prompt"):
            errors.append(f"user_prompt_mismatch:{row.get('id')}")
        if messages[2].get("content") != "Final answer: " + str(row.get("answer", "")):
            errors.append(f"assistant_mismatch:{row.get('id')}")
    if errors:
        raise RuntimeError(f"{split} validation failed: {errors[:20]}")
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_prompt_sha256": len(set(prompt_hashes)),
        "family_counts": dict(Counter(str(row.get("family", "")) for row in rows)),
        "source_counts": dict(Counter(str(row.get("source", "")) for row in rows)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V265 SCORE086 FILTERED MIX START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("weak_csv_path =", args.weak_csv_path, flush=True)
    print("v249_path =", args.v249_path, flush=True)
    print("v189_path =", args.v189_path, flush=True)
    print("val_fraction =", args.val_fraction, flush=True)
    print("seed =", args.seed, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("upload_to_hf =", bool(args.upload_to_hf), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN is required for V265")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "download"
        weak_csv = download_file(args.hf_dataset_repo, args.weak_csv_path, "dataset", download_root, token)
        v249_train = download_file(
            args.hf_dataset_repo,
            args.v249_path.rstrip("/") + "/v249_public_nonweak_target_train.jsonl",
            "dataset",
            download_root,
            token,
        )
        v249_val = download_file(
            args.hf_dataset_repo,
            args.v249_path.rstrip("/") + "/v249_public_nonweak_target_val.jsonl",
            "dataset",
            download_root,
            token,
        )
        v189_path = download_file(args.hf_dataset_repo, args.v189_path, "dataset", download_root, token)
        print("weak_csv =", weak_csv, flush=True)
        print("v249_train =", v249_train, flush=True)
        print("v249_val =", v249_val, flush=True)
        print("v189_path =", v189_path, flush=True)
        weak_sha = sha256_file(weak_csv)
        print("weak_csv_sha256 =", weak_sha, flush=True)
        if args.expected_weak_csv_sha256 and weak_sha != args.expected_weak_csv_sha256:
            raise RuntimeError(f"weak CSV SHA mismatch: expected {args.expected_weak_csv_sha256}, got {weak_sha}")
        weak_rows = read_csv_rows(weak_csv)
        download_artifact_hashes = {
            "weak_csv": file_meta(weak_csv),
            "v249_train_jsonl": file_meta(v249_train),
            "v249_val_jsonl": file_meta(v249_val),
            "v189_jsonl": file_meta(v189_path),
        }
        v249_examples = load_v249_rows(v249_train, args.v249_path.rstrip("/") + "/v249_public_nonweak_target_train.jsonl")
        v249_examples.extend(load_v249_rows(v249_val, args.v249_path.rstrip("/") + "/v249_public_nonweak_target_val.jsonl"))
        v189_examples = load_v189_rows(v189_path, args.v189_path)

    weak_ids = {str(row.get("id", "")) for row in weak_rows}
    weak_prompt_hashes = {prompt_sha256(row.get("prompt", "")) for row in weak_rows}
    blocked_weak_rows = [
        {
            "schema_version": "kg1_v265_blocked_weak_id_v1",
            "id": str(row.get("id", "")),
            "family": classify_family(row.get("prompt", ""), str(row.get("type", ""))),
            "split": "weak_reference",
        }
        for row in weak_rows
    ]

    raw_examples = [*v249_examples, *v189_examples]
    kept_rows, leak_rows = filter_rows(raw_examples, weak_ids, weak_prompt_hashes)
    train_rows, val_rows = split_rows(kept_rows, args.val_fraction, args.seed)
    train_validation = validate_rows(train_rows, "train", weak_ids, weak_prompt_hashes)
    val_validation = validate_rows(val_rows, "validation", weak_ids, weak_prompt_hashes)
    if {row["metadata"]["prompt_sha256"] for row in train_rows} & {row["metadata"]["prompt_sha256"] for row in val_rows}:
        raise RuntimeError("train/validation prompt hash overlap detected")

    outputs = {
        "train_jsonl": args.output_dir / "v249_public_nonweak_target_train.jsonl",
        "val_jsonl": args.output_dir / "v249_public_nonweak_target_val.jsonl",
        "blocked_weak_ids_csv": args.output_dir / "v249_blocked_weak_ids.csv",
        "leak_block_audit_csv": args.output_dir / "v265_leak_block_audit.csv",
        "manifest_json": args.output_dir / "v249_public_nonweak_target_manifest.json",
    }
    write_jsonl(outputs["train_jsonl"], train_rows)
    write_jsonl(outputs["val_jsonl"], val_rows)
    write_csv(outputs["blocked_weak_ids_csv"], blocked_weak_rows, BLOCKED_COLUMNS)
    write_csv(outputs["leak_block_audit_csv"], leak_rows, LEAK_COLUMNS)
    leak_reason_counts = dict(Counter(str(row.get("reason", "")) for row in leak_rows))
    input_family_counts = dict(Counter(str(row.get("family", "")) for row in raw_examples))
    kept_family_counts = dict(Counter(str(row.get("family", "")) for row in kept_rows))
    if train_validation["family_counts"].get("equation_transform", 0) < args.min_train_equation_rows:
        raise RuntimeError(
            "train equation row count below minimum: "
            + str(train_validation["family_counts"].get("equation_transform", 0))
        )
    if train_validation["family_counts"].get("bit_manipulation", 0) < args.min_train_bit_rows:
        raise RuntimeError(
            "train bit row count below minimum: "
            + str(train_validation["family_counts"].get("bit_manipulation", 0))
        )
    decision = {
        "decision": "dataset_ready_for_v250_tokenization_gate_not_training_yet",
        "reason": (
            f"train_rows={len(train_rows)}; val_rows={len(val_rows)}; "
            f"blocked_leaks={len(leak_rows)}; "
            f"train_equation={train_validation['family_counts'].get('equation_transform', 0)}; "
            f"train_bit={train_validation['family_counts'].get('bit_manipulation', 0)}"
        ),
        "next_action": "Run V250 tokenization gate with the emitted hashes before any H200 smoke.",
    }
    manifest = {
        "schema_version": "kg1_v265_score086_filtered_mix_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "weak_csv_path": args.weak_csv_path,
            "v249_path": args.v249_path,
            "v189_path": args.v189_path,
            "expected_weak_csv_sha256": args.expected_weak_csv_sha256,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "min_train_equation_rows": args.min_train_equation_rows,
            "min_train_bit_rows": args.min_train_bit_rows,
        },
        "download_artifact_hashes": download_artifact_hashes,
        "counts": {
            "weak_rows": len(weak_rows),
            "weak_prompt_hashes": len(weak_prompt_hashes),
            "raw_candidate_rows": len(raw_examples),
            "raw_family_counts": input_family_counts,
            "kept_rows_after_leak_filter": len(kept_rows),
            "kept_family_counts": kept_family_counts,
            "leak_blocked_rows": len(leak_rows),
            "leak_reason_counts": leak_reason_counts,
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
        manifest_refresh_info = upload_manifest(args.hf_dataset_repo, outputs["manifest_json"], args.output_path_in_repo, token)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("train_validation =", json.dumps(train_validation, sort_keys=True), flush=True)
    print("val_validation =", json.dumps(val_validation, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("manifest_refresh_info =", manifest_refresh_info, flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V265 SCORE086 FILTERED MIX END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--weak-csv-path", default=DEFAULT_WEAK_CSV_PATH)
    parser.add_argument("--v249-path", default=DEFAULT_V249_PATH)
    parser.add_argument("--v189-path", default=DEFAULT_V189_PATH)
    parser.add_argument("--expected-weak-csv-sha256", default=EXPECTED_WEAK_CSV_SHA256)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=265)
    parser.add_argument("--min-train-equation-rows", type=int, default=1700)
    parser.add_argument("--min-train-bit-rows", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v265_score086_filtered_mix")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    weak_rows = [{"id": "weak1", "prompt": "P A", "type": "equation_transform"}]
    weak_ids = {row["id"] for row in weak_rows}
    weak_hashes = {prompt_sha256(row["prompt"]) for row in weak_rows}
    rows = [
        make_example(
            row_id="weak1",
            prompt="different prompt",
            answer="1",
            family="equation_transform",
            source="self",
            split="candidate",
            source_file="self",
        ),
        make_example(
            row_id="ok1",
            prompt="P A",
            answer="2",
            family="equation_transform",
            source="self",
            split="candidate",
            source_file="self",
        ),
        make_example(
            row_id="ok2",
            prompt="P B",
            answer="3",
            family="equation_transform",
            source="self",
            split="candidate",
            source_file="self",
        ),
    ]
    kept, blocked = filter_rows(rows, weak_ids, weak_hashes)
    if len(kept) != 1 or len(blocked) != 2:
        raise AssertionError("leak filter self-test failed")
    validate_rows(kept, "self", weak_ids, weak_hashes)
    print("v265_score086_filtered_mix_self_test=ok", flush=True)
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

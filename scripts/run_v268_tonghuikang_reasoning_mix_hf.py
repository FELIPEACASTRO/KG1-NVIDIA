#!/usr/bin/env python3
"""Build a leak-filtered V268 reasoning mix from tonghuikang/nemotron.

This is CPU-only data preparation. It downloads public GitHub artifacts from
tonghuikang/nemotron, blocks every canonical weak ID, decodes only
``corpus/<id>/synthetic.jsonl`` token files whose final ``\\boxed{}`` answer
matches train.csv, and writes V249-compatible JSONL filenames so the existing
V250 tokenization gate can validate the dataset.

It does not train, evaluate, package, or submit anything.
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
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_WEAK_CSV_PATH = "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
DEFAULT_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tonghuikang/nemotron/master"
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, show the reasoning, then finish with exactly one boxed answer."
)
TARGET_CATEGORIES = {
    "bit_manipulation": "bit_manipulation",
    "cryptarithm_deduce": "equation_transform",
    "cryptarithm_guess": "equation_transform",
    "equation_numeric_deduce": "equation_transform",
    "equation_numeric_guess": "equation_transform",
}
BLOCKED_COLUMNS = ["schema_version", "id", "family", "split"]
AUDIT_COLUMNS = [
    "schema_version",
    "id",
    "category",
    "family",
    "status",
    "reason",
    "expected_answer",
    "decoded_boxed_answer",
    "token_count",
    "completion_chars",
]


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


def download_hf_file(repo_id: str, filename: str, repo_type: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V268") from exc
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
        raise RuntimeError("huggingface_hub is required to upload V268 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V268 tonghuikang reasoning mix {path_in_repo.strip('/')}",
    )
    return str(info)


def upload_manifest(repo_id: str, manifest_path: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V268 manifest") from exc
    api = HfApi(token=token)
    info = api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(manifest_path),
        path_in_repo=(path_in_repo.strip("/") + "/" + manifest_path.name).strip("/"),
        commit_message=f"Refresh KG1 V268 manifest {path_in_repo.strip('/')}",
    )
    return str(info)


def github_bytes(base_url: str, path: str, retries: int = 3) -> bytes:
    url = base_url.rstrip("/") + "/" + path.strip("/")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "kg1-v268-builder"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_exc = exc
        except Exception as exc:  # pragma: no cover - network variability
            last_exc = exc
        time.sleep(0.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def github_text(base_url: str, path: str) -> str:
    return github_bytes(base_url, path).decode("utf-8")


def read_csv_text(text: str) -> list[dict[str, str]]:
    import io

    return list(csv.DictReader(io.StringIO(text)))


def read_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL row {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row {line_no} is not an object")
        rows.append(row)
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_answer(value: Any) -> str:
    return "".join(str(value).strip().split())


def prompt_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def prompt_answer_sha256(prompt: Any, answer: Any) -> str:
    raw = normalize_text(prompt) + "\0" + normalize_answer(answer)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def final_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1] if matches else ""


def decode_synthetic(raw: bytes, tokenizer: Any) -> tuple[str, int]:
    texts: list[str] = []
    token_count = 0
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row.get("tokens", [])
        if not isinstance(tokens, list):
            raise RuntimeError(f"synthetic row {line_no} tokens is not a list")
        token_count += len(tokens)
        if row.get("type") == "unmasked":
            texts.append(tokenizer.decode(tokens))
    return "\n".join(texts).strip(), token_count


def split_rows(rows: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["metadata"]["tonghuikang_category"]))
        grouped.setdefault(key, []).append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        val_count = max(1, round(len(bucket) * validation_fraction)) if len(bucket) >= 10 else 0
        val.extend(bucket[:val_count])
        train.extend(bucket[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_example(
    *,
    row_id: str,
    prompt: str,
    answer: str,
    family: str,
    category: str,
    completion: str,
    token_count: int,
    split: str,
) -> dict[str, Any]:
    return {
        "id": f"v268_{split}_tonghuikang_{row_id}",
        "original_id": row_id,
        "family": family,
        "subcategory": category,
        "source": "tonghuikang_nemotron_reasoning_synthetic_nonweak",
        "prompt": prompt.strip(),
        "answer": answer.strip(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": completion.strip()},
        ],
        "metadata": {
            "source": "tonghuikang/nemotron",
            "source_file": f"corpus/{row_id}/synthetic.jsonl",
            "original_id": row_id,
            "tonghuikang_category": category,
            "weak_id_excluded": True,
            "assistant_style": "reasoning_boxed",
            "split": split,
            "family": family,
            "token_count_tonghuikang": token_count,
            "completion_chars": len(completion),
            "prompt_sha256": prompt_sha256(prompt),
            "prompt_answer_sha256": prompt_answer_sha256(prompt, answer),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V268 TONGHUIKANG REASONING MIX START ===", flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("github_raw_base =", args.github_raw_base, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("max_workers =", args.max_workers, flush=True)
    print("validation_fraction =", args.validation_fraction, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args.hf_token)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        weak_csv = download_hf_file(args.hf_dataset_repo, args.weak_csv_path, "dataset", tmp, token)
        weak_sha = sha256_file(weak_csv)
        print("weak_csv_sha256 =", weak_sha, flush=True)
        if weak_sha != EXPECTED_WEAK_CSV_SHA256:
            raise RuntimeError(f"weak CSV SHA mismatch: expected {EXPECTED_WEAK_CSV_SHA256}, got {weak_sha}")
        weak_rows = list(csv.DictReader(weak_csv.open("r", encoding="utf-8", newline="")))
        weak_ids = {str(row.get("id", "")).strip() for row in weak_rows}
        print("weak_id_count =", len(weak_ids), flush=True)

    tokenizer_json = github_bytes(args.github_raw_base, "tokenizer.json")
    tokenizer_path = args.output_dir / "tonghuikang_tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_json)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required for V268") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train_rows = read_csv_text(github_text(args.github_raw_base, "train.csv"))
    problems = read_jsonl_text(github_text(args.github_raw_base, "problems.jsonl"))
    corpus_index = read_jsonl_text(github_text(args.github_raw_base, "corpus.jsonl"))
    train_by_id = {str(row["id"]): row for row in train_rows}
    category_by_id = {str(row["id"]): str(row.get("category", "")) for row in problems}

    candidate_ids: list[str] = []
    for row in corpus_index:
        row_id = str(row.get("problem_id", "")).strip()
        category = str(row.get("category", ""))
        if not row_id or row_id in weak_ids:
            continue
        if category not in TARGET_CATEGORIES:
            continue
        if row.get("included") is not True:
            continue
        if row_id not in train_by_id:
            continue
        candidate_ids.append(row_id)
    candidate_ids = sorted(set(candidate_ids))
    print("candidate_id_count =", len(candidate_ids), flush=True)
    print("candidate_category_counts =", json.dumps(dict(Counter(category_by_id.get(row_id, "") for row_id in candidate_ids)), sort_keys=True), flush=True)

    audit_rows: list[dict[str, Any]] = []

    def fetch_one(row_id: str) -> dict[str, Any] | None:
        category = category_by_id.get(row_id, "")
        family = TARGET_CATEGORIES.get(category, "")
        expected = str(train_by_id[row_id]["answer"]).strip()
        try:
            raw = github_bytes(args.github_raw_base, f"corpus/{row_id}/synthetic.jsonl", retries=2)
        except urllib.error.HTTPError as exc:
            return {
                "audit_only": True,
                "schema_version": "kg1_v268_tonghuikang_reasoning_audit_v1",
                "id": row_id,
                "category": category,
                "family": family,
                "status": "missing_synthetic",
                "reason": f"http_{exc.code}",
                "expected_answer": expected,
                "decoded_boxed_answer": "",
                "token_count": "",
                "completion_chars": "",
            }
        except Exception as exc:
            return {
                "audit_only": True,
                "schema_version": "kg1_v268_tonghuikang_reasoning_audit_v1",
                "id": row_id,
                "category": category,
                "family": family,
                "status": "download_error",
                "reason": repr(exc)[:200],
                "expected_answer": expected,
                "decoded_boxed_answer": "",
                "token_count": "",
                "completion_chars": "",
            }
        try:
            completion, token_count = decode_synthetic(raw, tokenizer)
        except Exception as exc:
            return {
                "audit_only": True,
                "schema_version": "kg1_v268_tonghuikang_reasoning_audit_v1",
                "id": row_id,
                "category": category,
                "family": family,
                "status": "decode_error",
                "reason": repr(exc)[:200],
                "expected_answer": expected,
                "decoded_boxed_answer": "",
                "token_count": "",
                "completion_chars": "",
            }
        boxed = final_boxed_answer(completion)
        if normalize_answer(boxed) != normalize_answer(expected):
            return {
                "audit_only": True,
                "schema_version": "kg1_v268_tonghuikang_reasoning_audit_v1",
                "id": row_id,
                "category": category,
                "family": family,
                "status": "boxed_answer_mismatch",
                "reason": "final_boxed_answer_differs_from_train_csv",
                "expected_answer": expected,
                "decoded_boxed_answer": boxed,
                "token_count": token_count,
                "completion_chars": len(completion),
            }
        row = make_example(
            row_id=row_id,
            prompt=str(train_by_id[row_id]["prompt"]),
            answer=expected,
            family=family,
            category=category,
            completion=completion,
            token_count=token_count,
            split="candidate",
        )
        row["audit"] = {
            "schema_version": "kg1_v268_tonghuikang_reasoning_audit_v1",
            "id": row_id,
            "category": category,
            "family": family,
            "status": "accepted",
            "reason": "",
            "expected_answer": expected,
            "decoded_boxed_answer": boxed,
            "token_count": token_count,
            "completion_chars": len(completion),
        }
        return row

    accepted: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for index, result in enumerate(executor.map(fetch_one, candidate_ids), start=1):
            if index % 200 == 0:
                print(f"fetch_progress = {index}/{len(candidate_ids)}", flush=True)
            if result is None:
                continue
            if result.get("audit_only"):
                audit_rows.append(result)
            else:
                audit_rows.append(result.pop("audit"))
                accepted.append(result)

    if len(accepted) < args.min_accepted_rows:
        raise RuntimeError(f"accepted rows below floor: {len(accepted)} < {args.min_accepted_rows}")
    accepted_counts = Counter(str(row["family"]) for row in accepted)
    accepted_category_counts = Counter(str(row["metadata"]["tonghuikang_category"]) for row in accepted)
    print("accepted_rows =", len(accepted), flush=True)
    print("accepted_family_counts =", json.dumps(dict(accepted_counts), sort_keys=True), flush=True)
    print("accepted_category_counts =", json.dumps(dict(accepted_category_counts), sort_keys=True), flush=True)
    print("audit_status_counts =", json.dumps(dict(Counter(str(row["status"]) for row in audit_rows)), sort_keys=True), flush=True)

    train, val = split_rows(accepted, args.validation_fraction, args.seed)
    for row in train:
        row["id"] = row["id"].replace("_candidate_", "_train_")
        row["metadata"]["split"] = "train"
    for row in val:
        row["id"] = row["id"].replace("_candidate_", "_val_")
        row["metadata"]["split"] = "validation"

    out_train = args.output_dir / "v249_public_nonweak_target_train.jsonl"
    out_val = args.output_dir / "v249_public_nonweak_target_val.jsonl"
    out_blocked = args.output_dir / "v249_blocked_weak_ids.csv"
    out_audit = args.output_dir / "v268_tonghuikang_reasoning_audit.csv"
    out_manifest = args.output_dir / "v249_public_nonweak_target_manifest.json"
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    write_csv(
        out_blocked,
        [
            {
                "schema_version": "kg1_v268_blocked_weak_id_v1",
                "id": row_id,
                "family": "",
                "split": "weak_block",
            }
            for row_id in sorted(weak_ids)
        ],
        BLOCKED_COLUMNS,
    )
    write_csv(out_audit, audit_rows, AUDIT_COLUMNS)

    manifest = {
        "schema_version": "kg1_v268_tonghuikang_reasoning_mix_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "weak_csv_path": args.weak_csv_path,
            "weak_csv_sha256": EXPECTED_WEAK_CSV_SHA256,
            "github_raw_base": args.github_raw_base,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "target_categories": TARGET_CATEGORIES,
        },
        "counts": {
            "candidate_ids": len(candidate_ids),
            "accepted_rows": len(accepted),
            "train_rows": len(train),
            "val_rows": len(val),
            "blocked_weak_ids": len(weak_ids),
            "accepted_family_counts": dict(Counter(str(row["family"]) for row in accepted)),
            "train_family_counts": dict(Counter(str(row["family"]) for row in train)),
            "val_family_counts": dict(Counter(str(row["family"]) for row in val)),
            "accepted_category_counts": dict(accepted_category_counts),
            "audit_status_counts": dict(Counter(str(row["status"]) for row in audit_rows)),
        },
        "files": {
            "train": file_meta(out_train),
            "val": file_meta(out_val),
            "blocked_weak_ids": file_meta(out_blocked),
            "audit": file_meta(out_audit),
        },
        "decision": {
            "decision": "run_v250_tokenization_gate_before_any_gpu_train",
            "reason": "Accepted only non-weak decoded reasoning rows whose final boxed answer matches train.csv.",
            "next_action": "Run V250 tokenization gate with assistant_style=reasoning_boxed and max_length=8192.",
        },
        "blocked_actions": ["gpu_train", "weak_eval", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_manifest, manifest)

    upload_info = "upload_disabled"
    refresh_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.hf_dataset_repo, args.output_dir, args.output_path_in_repo, token)
        refresh_info = upload_manifest(args.hf_dataset_repo, out_manifest, args.output_path_in_repo, token)
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.hf_dataset_repo,
        "path_in_repo": args.output_path_in_repo,
        "upload_info": upload_info,
        "manifest_refresh_info": refresh_info,
    }
    write_json(out_manifest, manifest)
    print("train_sha256 =", sha256_file(out_train), flush=True)
    print("val_sha256 =", sha256_file(out_val), flush=True)
    print("blocked_sha256 =", sha256_file(out_blocked), flush=True)
    print("manifest_path =", out_manifest, flush=True)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("=== V268 TONGHUIKANG REASONING MIX END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--weak-csv-path", default=DEFAULT_WEAK_CSV_PATH)
    parser.add_argument("--github-raw-base", default=DEFAULT_GITHUB_RAW_BASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v268_tonghuikang_reasoning_mix")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=268)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--min-accepted-rows", type=int, default=1500)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

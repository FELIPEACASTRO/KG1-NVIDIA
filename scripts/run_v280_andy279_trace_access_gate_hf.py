#!/usr/bin/env python3
"""CPU-only access/schema/leakage gate for Andy279 trace datasets.

V280 is a pre-GPU gate. It checks whether the high-priority external datasets
are accessible, samples their schema, and can optionally run a bounded full
download audit. It does not train, run model generation, evaluate adapters,
package artifacts, or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_WEAK_CSV_PATH = "runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
EXPECTED_WEAK_ROWS = 315
EXPECTED_WEAK_FAMILY_COUNTS = {"bit_manipulation": 160, "equation_transform": 155}

TARGETS = [
    {
        "repo_id": "andy279/nemotron-reasoning-challenge-raw-traces",
        "repo_type": "dataset",
        "priority": "P0",
        "family_focus": "equation_transform,bit_manipulation",
        "reason": "raw teacher/solver traces named for transformation and bit-manipulation routes",
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
        "family_focus": "equation_transform,bit_manipulation",
        "reason": "SFT train/validation data advertised for the same challenge",
        "files": ["sft_train.jsonl", "sft_val.jsonl"],
    },
    {
        "repo_id": "jasonkung98/NVIDIA-Nemotron-Model-Reasoning-Challenge",
        "repo_type": "dataset",
        "priority": "P1",
        "family_focus": "source_sanity_only",
        "reason": "public challenge mirror already audited; never enough by itself to justify GPU",
        "files": ["train.csv", "test.csv"],
    },
]

CSV_COLUMNS = [
    "schema_version",
    "repo_id",
    "filename",
    "priority",
    "family_focus",
    "reason",
    "metadata_status",
    "repo_gated",
    "repo_sha",
    "file_exists_in_metadata",
    "file_size",
    "range_status_code",
    "range_accessible",
    "range_error",
    "sample_sha256",
    "sample_line_count",
    "sample_parse_ok_rows",
    "sample_parse_error_rows",
    "sample_format",
    "sample_schema_keys",
    "sample_family_counts",
    "sample_id_overlap_with_weak",
    "sample_prompt_hash_overlap_with_weak",
    "full_download_status",
    "full_rows",
    "full_bytes",
    "full_sha256",
    "full_family_counts",
    "full_id_overlap_with_weak",
    "full_prompt_hash_overlap_with_weak",
    "gate_status",
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def token_value(cli_token: str) -> str | None:
    if cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def prompt_hash(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def classify_family(prompt: str, fallback: str = "") -> str:
    fallback = str(fallback or "").strip()
    if fallback:
        return fallback
    text = str(prompt).lower()
    if "bit manipulation" in text or "8-bit binary" in text or "input -> output" in text:
        return "bit_manipulation"
    if "transformation rules is applied to equations" in text or "transformation/equation" in text:
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


def nested_get(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata and metadata[key] not in (None, ""):
                return metadata[key]
    return ""


def infer_family(row: dict[str, Any]) -> str:
    family = nested_get(row, ("family", "task_type", "category", "subcategory", "type"))
    prompt = nested_get(row, ("prompt", "question", "input", "problem"))
    return classify_family(str(prompt), str(family))


def row_id(row: dict[str, Any]) -> str:
    return str(nested_get(row, ("id", "original_id", "problem_id", "row_id"))).strip()


def row_prompt(row: dict[str, Any]) -> str:
    return str(nested_get(row, ("prompt", "question", "input", "problem"))).strip()


def dataset_info_map(api: Any, repo_id: str, token: str | None) -> tuple[dict[str, Any], str]:
    try:
        info = api.dataset_info(repo_id=repo_id, files_metadata=True, token=token)
        siblings = getattr(info, "siblings", []) or []
        by_file = {str(getattr(item, "rfilename", "")): item for item in siblings}
        return {
            "gated": str(getattr(info, "gated", "")),
            "sha": str(getattr(info, "sha", "")),
            "siblings": by_file,
        }, "ok"
    except Exception as exc:
        return {"gated": "", "sha": "", "siblings": {}}, repr(exc)


def sibling_size(sibling: Any) -> int:
    for attr in ("size", "lfs_size"):
        value = getattr(sibling, attr, None)
        if value is None:
            continue
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
                return status, False, response.text[:500].replace("\n", " "), b""
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            return status, True, "", b"".join(chunks)[:max_bytes]
    except Exception as exc:
        return 0, False, repr(exc), b""


def parse_jsonl_sample(text: str, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    errors = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            errors += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            errors += 1
        if len(rows) >= max_rows:
            break
    return rows, errors


def parse_csv_sample(text: str, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(row) for _, row in zip(range(max_rows), reader)]
        return rows, 0
    except Exception:
        return [], 1


def sample_audit(filename: str, data: bytes, weak_ids: set[str], weak_prompt_hashes: set[str], max_rows: int) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    if filename.endswith(".csv"):
        parsed_rows, parse_errors = parse_csv_sample(text, max_rows)
        sample_format = "csv"
    else:
        parsed_rows, parse_errors = parse_jsonl_sample(text, max_rows)
        sample_format = "jsonl"
    schema_keys = sorted({key for row in parsed_rows for key in row.keys()})
    families = Counter(infer_family(row) for row in parsed_rows)
    ids = {row_id(row) for row in parsed_rows if row_id(row)}
    hashes = {prompt_hash(row_prompt(row)) for row in parsed_rows if row_prompt(row)}
    return {
        "sample_line_count": len([line for line in text.splitlines() if line.strip()]),
        "sample_parse_ok_rows": len(parsed_rows),
        "sample_parse_error_rows": parse_errors,
        "sample_format": sample_format,
        "sample_schema_keys": json.dumps(schema_keys, sort_keys=True),
        "sample_family_counts": json.dumps(dict(sorted(families.items())), sort_keys=True),
        "sample_id_overlap_with_weak": len(ids & weak_ids),
        "sample_prompt_hash_overlap_with_weak": len(hashes & weak_prompt_hashes),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_jsonl_rows(path: Path) -> tuple[int, Counter[str], set[str], set[str]]:
    rows = 0
    families: Counter[str] = Counter()
    ids: set[str] = set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            rows += 1
            families[infer_family(obj)] += 1
            ident = row_id(obj)
            prompt = row_prompt(obj)
            if ident:
                ids.add(ident)
            if prompt:
                hashes.add(prompt_hash(prompt))
    return rows, families, ids, hashes


def iter_csv_file(path: Path) -> tuple[int, Counter[str], set[str], set[str]]:
    rows = 0
    families: Counter[str] = Counter()
    ids: set[str] = set()
    hashes: set[str] = set()
    for row in read_csv_rows(path):
        rows += 1
        families[infer_family(row)] += 1
        ident = row_id(row)
        prompt = row_prompt(row)
        if ident:
            ids.add(ident)
        if prompt:
            hashes.add(prompt_hash(prompt))
    return rows, families, ids, hashes


def full_file_audit(path: Path, filename: str, weak_ids: set[str], weak_prompt_hashes: set[str]) -> dict[str, Any]:
    if filename.endswith(".csv"):
        rows, families, ids, hashes = iter_csv_file(path)
    else:
        rows, families, ids, hashes = iter_jsonl_rows(path)
    return {
        "full_download_status": "audited",
        "full_rows": rows,
        "full_bytes": int(path.stat().st_size),
        "full_sha256": sha256_file(path),
        "full_family_counts": json.dumps(dict(sorted(families.items())), sort_keys=True),
        "full_id_overlap_with_weak": len(ids & weak_ids),
        "full_prompt_hash_overlap_with_weak": len(hashes & weak_prompt_hashes),
    }


def download_hf_file(repo_id: str, repo_type: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V280") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
    )


def load_weak_contract(
    hf_dataset_repo: str,
    weak_csv_path: str,
    expected_sha256: str,
    expected_rows: int,
    expected_family_counts: dict[str, int],
    token: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        weak_path = download_hf_file(hf_dataset_repo, "dataset", weak_csv_path, Path(tmp), token)
        weak_sha = sha256_file(weak_path)
        weak_rows = read_csv_rows(weak_path)
    family_counts = Counter(classify_family(row.get("prompt", ""), row.get("type", "")) for row in weak_rows)
    if weak_sha != expected_sha256:
        raise RuntimeError(f"weak CSV SHA mismatch: expected {expected_sha256}, got {weak_sha}")
    if len(weak_rows) != expected_rows:
        raise RuntimeError(f"weak row count mismatch: expected {expected_rows}, got {len(weak_rows)}")
    if dict(family_counts) != expected_family_counts:
        raise RuntimeError(f"weak family counts mismatch: expected {expected_family_counts}, got {dict(family_counts)}")
    ids = {str(row.get("id", "")).strip() for row in weak_rows if str(row.get("id", "")).strip()}
    prompt_hashes = {prompt_hash(row.get("prompt", "")) for row in weak_rows if row.get("prompt", "")}
    return {
        "path": weak_csv_path,
        "sha256": weak_sha,
        "rows": len(weak_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "ids": ids,
        "prompt_hashes": prompt_hashes,
        "id_count": len(ids),
        "prompt_hash_count": len(prompt_hashes),
    }


def gate_row(row: dict[str, Any]) -> tuple[str, str]:
    if not row.get("range_accessible"):
        return "blocked_no_access", "resolve_hf_gated_terms_or_token_before_any_gpu"
    if int(row.get("sample_parse_ok_rows") or 0) == 0:
        return "blocked_unparseable_sample", "manual_schema_review_before_download_or_gpu"
    if int(row.get("sample_id_overlap_with_weak") or 0) or int(row.get("sample_prompt_hash_overlap_with_weak") or 0):
        return "blocked_sample_weak_overlap", "exclude_or_rebuild_source_before_any_training"
    if row.get("full_download_status") == "audited":
        if int(row.get("full_id_overlap_with_weak") or 0) or int(row.get("full_prompt_hash_overlap_with_weak") or 0):
            return "blocked_full_weak_overlap", "exclude_overlaps_before_any_training"
        return "passed_full_audit", "eligible_for_trace_ingestion_pretraining_gate"
    return "passed_access_sample_only", "run_full_download_audit_before_training_or_gpu"


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V280 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V280 Andy279 trace access gate {path_in_repo.strip('/')}",
    )
    return str(info)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V280 ANDY279 TRACE ACCESS GATE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("allow_full_download =", args.allow_full_download, flush=True)
    print("max_full_download_bytes =", args.max_full_download_bytes, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token = token_value(args.hf_token)
    if not token:
        raise RuntimeError("HF_TOKEN or cached Hugging Face auth is required for V280")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V280") from exc

    api = HfApi(token=token)
    weak_contract = load_weak_contract(
        args.hf_dataset_repo,
        args.weak_csv_path,
        args.expected_weak_csv_sha256,
        args.expected_weak_rows,
        json.loads(args.expected_weak_family_counts),
        token,
    )
    print(
        "weak_contract =",
        json.dumps(
            {
                "rows": weak_contract["rows"],
                "sha256": weak_contract["sha256"],
                "family_counts": weak_contract["family_counts"],
                "id_count": weak_contract["id_count"],
                "prompt_hash_count": weak_contract["prompt_hash_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    repo_status: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "downloads"
        for target in TARGETS:
            repo_id = str(target["repo_id"])
            print("target_repo_start =", repo_id, flush=True)
            info, metadata_status = dataset_info_map(api, repo_id, token)
            repo_status[repo_id] = {
                "metadata_status": metadata_status,
                "gated": info.get("gated", ""),
                "sha": info.get("sha", ""),
            }
            print("target_repo_metadata =", json.dumps(repo_status[repo_id], sort_keys=True), flush=True)
            siblings = info.get("siblings", {})
            if not isinstance(siblings, dict):
                siblings = {}
            for filename in target["files"]:
                print("target_file_start =", json.dumps({"repo_id": repo_id, "filename": filename}), flush=True)
                sibling = siblings.get(filename)
                file_size = sibling_size(sibling) if sibling is not None else 0
                status_code, accessible, error, sample = range_read(repo_id, filename, token, args.max_sample_bytes)
                sample_info = {
                    "sample_line_count": 0,
                    "sample_parse_ok_rows": 0,
                    "sample_parse_error_rows": 0,
                    "sample_format": "",
                    "sample_schema_keys": "[]",
                    "sample_family_counts": "{}",
                    "sample_id_overlap_with_weak": 0,
                    "sample_prompt_hash_overlap_with_weak": 0,
                }
                if accessible and sample:
                    sample_info = sample_audit(
                        filename,
                        sample,
                        set(weak_contract["ids"]),
                        set(weak_contract["prompt_hashes"]),
                        args.max_sample_rows,
                    )
                full_info = {
                    "full_download_status": "skipped_no_access" if not accessible else "skipped_sample_only",
                    "full_rows": 0,
                    "full_bytes": 0,
                    "full_sha256": "",
                    "full_family_counts": "{}",
                    "full_id_overlap_with_weak": 0,
                    "full_prompt_hash_overlap_with_weak": 0,
                }
                if accessible and args.allow_full_download:
                    if file_size > args.max_full_download_bytes:
                        full_info["full_download_status"] = f"skipped_size_over_limit:{file_size}"
                    else:
                        try:
                            full_path = download_hf_file(repo_id, str(target["repo_type"]), filename, download_root, token)
                            full_info = full_file_audit(
                                full_path,
                                filename,
                                set(weak_contract["ids"]),
                                set(weak_contract["prompt_hashes"]),
                            )
                        except Exception as exc:
                            full_info["full_download_status"] = "failed:" + repr(exc)[:300]
                row = {
                    "schema_version": "kg1_v280_andy279_trace_access_gate_row_v1",
                    "repo_id": repo_id,
                    "filename": filename,
                    "priority": target["priority"],
                    "family_focus": target["family_focus"],
                    "reason": target["reason"],
                    "metadata_status": metadata_status,
                    "repo_gated": info.get("gated", ""),
                    "repo_sha": info.get("sha", ""),
                    "file_exists_in_metadata": sibling is not None,
                    "file_size": file_size,
                    "range_status_code": status_code,
                    "range_accessible": accessible,
                    "range_error": error,
                    "sample_sha256": sha256_bytes(sample) if sample else "",
                    **sample_info,
                    **full_info,
                }
                gate_status, next_action = gate_row(row)
                row["gate_status"] = gate_status
                row["next_action"] = next_action
                rows.append(row)
                print(
                    "target_file_gate =",
                    json.dumps(
                        {
                            "repo_id": repo_id,
                            "filename": filename,
                            "priority": row["priority"],
                            "range_accessible": row["range_accessible"],
                            "sample_parse_ok_rows": row["sample_parse_ok_rows"],
                            "full_download_status": row["full_download_status"],
                            "gate_status": gate_status,
                            "next_action": next_action,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    p0_rows = [row for row in rows if row["priority"] == "P0"]
    p0_accessible = [row for row in p0_rows if row["range_accessible"]]
    p0_full_passed = [row for row in p0_rows if row["gate_status"] == "passed_full_audit"]
    p0_sample_passed = [row for row in p0_rows if row["gate_status"] == "passed_access_sample_only"]
    p0_blocked = [row for row in p0_rows if str(row["gate_status"]).startswith("blocked")]
    public_accessible = [row for row in rows if row["priority"] == "P1" and row["range_accessible"]]
    if p0_full_passed:
        decision = {
            "decision": "p0_trace_full_audit_ready_for_ingestion",
            "reason": f"p0_full_passed_files={len(p0_full_passed)}; p0_blocked_files={len(p0_blocked)}",
            "next_action": "Build a filtered trace dataset with anti-leakage and verifier labels before any GPU training.",
        }
    elif p0_sample_passed:
        decision = {
            "decision": "p0_trace_accessible_sample_only",
            "reason": f"p0_sample_passed_files={len(p0_sample_passed)}; full_download_audit_required_before_gpu",
            "next_action": "Run V280 again with --allow-full-download inside disk limits, then build filtered trace data.",
        }
    elif p0_accessible:
        decision = {
            "decision": "p0_trace_accessible_but_blocked_by_schema_or_leakage",
            "reason": f"p0_accessible_files={len(p0_accessible)}; p0_blocked_files={len(p0_blocked)}",
            "next_action": "Manual schema/leakage review; do not train.",
        }
    elif public_accessible:
        decision = {
            "decision": "p0_gated_terms_required_no_gpu",
            "reason": f"p0_accessible_files=0; p0_blocked_files={len(p0_blocked)}; public_accessible_files={len(public_accessible)}",
            "next_action": "Human must request/accept access to andy279 gated datasets before this route can continue.",
        }
    else:
        decision = {
            "decision": "all_external_trace_sources_blocked_no_gpu",
            "reason": f"p0_accessible_files=0; p0_blocked_files={len(p0_blocked)}; public_accessible_files=0",
            "next_action": "Fix HF token/network/access before any further external-trace work.",
        }

    outputs = {
        "access_gate_csv": args.output_dir / f"{args.label}_access_gate.csv",
        "manifest_json": args.output_dir / f"{args.label}_manifest.json",
    }
    write_csv(outputs["access_gate_csv"], rows)
    manifest = {
        "schema_version": "kg1_v280_andy279_trace_access_gate_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "weak_csv_path": args.weak_csv_path,
            "expected_weak_csv_sha256": args.expected_weak_csv_sha256,
            "expected_weak_rows": args.expected_weak_rows,
            "expected_weak_family_counts": json.loads(args.expected_weak_family_counts),
            "targets": TARGETS,
            "allow_full_download": bool(args.allow_full_download),
            "max_sample_bytes": args.max_sample_bytes,
            "max_sample_rows": args.max_sample_rows,
            "max_full_download_bytes": args.max_full_download_bytes,
        },
        "weak_contract": {
            "path": weak_contract["path"],
            "sha256": weak_contract["sha256"],
            "rows": weak_contract["rows"],
            "family_counts": weak_contract["family_counts"],
            "id_count": weak_contract["id_count"],
            "prompt_hash_count": weak_contract["prompt_hash_count"],
        },
        "repo_status": repo_status,
        "counts": {
            "rows": len(rows),
            "p0_rows": len(p0_rows),
            "p0_accessible_files": len(p0_accessible),
            "p0_full_passed_files": len(p0_full_passed),
            "p0_sample_passed_files": len(p0_sample_passed),
            "p0_blocked_files": len(p0_blocked),
            "public_accessible_files": len(public_accessible),
        },
        "decision": decision,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {"access_gate_csv": file_meta(outputs["access_gate_csv"])},
        "blocked_actions": ["gpu_train", "model_generation", "full_eval", "package", "kaggle_submit"],
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
    print("=== V280 ANDY279 TRACE ACCESS GATE END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--weak-csv-path", default=DEFAULT_WEAK_CSV_PATH)
    parser.add_argument("--expected-weak-csv-sha256", default=EXPECTED_WEAK_CSV_SHA256)
    parser.add_argument("--expected-weak-rows", type=int, default=EXPECTED_WEAK_ROWS)
    parser.add_argument("--expected-weak-family-counts", default=json.dumps(EXPECTED_WEAK_FAMILY_COUNTS, sort_keys=True))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v280_andy279_trace_access_gate")
    parser.add_argument("--max-sample-bytes", type=int, default=65536)
    parser.add_argument("--max-sample-rows", type=int, default=25)
    parser.add_argument("--allow-full-download", action="store_true")
    parser.add_argument("--max-full-download-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    weak_ids = {"abc"}
    weak_hashes = {prompt_hash("same prompt")}
    sample = b'{"id":"abc","prompt":"same prompt","family":"equation_transform"}\n{"id":"def","prompt":"bit manipulation task"}\n'
    audit = sample_audit("x.jsonl", sample, weak_ids, weak_hashes, 10)
    if audit["sample_parse_ok_rows"] != 2:
        raise AssertionError("jsonl sample parsing failed")
    if audit["sample_id_overlap_with_weak"] != 1 or audit["sample_prompt_hash_overlap_with_weak"] != 1:
        raise AssertionError("weak overlap detection failed")
    csv_audit = sample_audit("x.csv", b"id,prompt,answer\n1,unit conversion,2\n", set(), set(), 10)
    if csv_audit["sample_format"] != "csv" or csv_audit["sample_parse_ok_rows"] != 1:
        raise AssertionError("csv sample parsing failed")
    print("v280_andy279_trace_access_gate_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        args.output_dir = Path("artifacts") / "hf_cpu_runs" / f"v280_andy279_trace_access_gate_{utc_compact()}"
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

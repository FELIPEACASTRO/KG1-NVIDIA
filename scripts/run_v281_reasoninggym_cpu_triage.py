#!/usr/bin/env python3
"""CPU-only triage for nvidia/Nemotron-RL-ReasoningGym-v1.

V281 is an evidence gate, not a training step. It validates the HF Datasets
Server schema/split/license, streams a bounded number of JSONL rows, filters
only KG1-relevant source datasets, checks overlap against the canonical weak
CSV when available, and writes small manifests/fixtures. It never trains,
runs model generation, packages, or submits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.hf_job_weak_eval_v245 import (  # noqa: E402
        DEFAULT_DATA_REPO,
        DEFAULT_WEAK_CSV_FILE,
        EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        EXPECTED_WEAK_CSV_SHA256,
        validate_weak_csv,
    )
except Exception:  # pragma: no cover - self-test can run without optional deps
    DEFAULT_DATA_REPO = "felipesp1983/kg1-nemotron-training"
    DEFAULT_WEAK_CSV_FILE = (
        "runtime_artifacts/v245_weak_eval_bridge/"
        "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
    )
    EXPECTED_SHARED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
    EXPECTED_WEAK_CSV_SHA256 = "85da758e14d57ea40270de5747f98726a0ad0b6d1795bff7dd46183005e0f9b6"
    validate_weak_csv = None  # type: ignore[assignment]


DEFAULT_DATASET = "nvidia/Nemotron-RL-ReasoningGym-v1"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_JSONL_URL = "https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1/resolve/main/data/train.jsonl"
REQUIRED_FIELDS = {"responses_create_params", "question", "answer", "metadata", "agent_ref", "uuid", "license"}
RELEVANT_SOURCES = {
    "base_conversion",
    "binary_alternation",
    "bitwise_arithmetic",
    "circuit_logic",
    "count_bits",
    "cryptarithm",
    "number_format",
    "polynomial_multiplication",
    "simple_equations",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def json_url(url: str, timeout_s: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def datasets_server_url(endpoint: str, dataset: str, config: str, split: str | None = None) -> str:
    params = {"dataset": dataset, "config": config}
    if split:
        params["split"] = split
    return "https://datasets-server.huggingface.co/" + endpoint + "?" + urllib.parse.urlencode(params)


def validate_splits(payload: dict[str, Any], split: str) -> dict[str, Any]:
    if payload.get("pending"):
        raise RuntimeError("HF datasets-server splits pending: " + json.dumps(payload.get("pending")))
    if payload.get("failed"):
        raise RuntimeError("HF datasets-server splits failed: " + json.dumps(payload.get("failed")))
    splits = payload.get("splits", [])
    split_names = sorted({str(item.get("split", "")) for item in splits if isinstance(item, dict)})
    if split not in split_names:
        raise RuntimeError(f"required split {split!r} not found in {split_names}")
    return {"split_names": split_names, "split_count": len(splits)}


def validate_first_rows(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("first-rows returned no rows")
    observed_fields = set()
    source_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    relevant_rows = 0
    for wrapper in rows:
        row = wrapper.get("row", {}) if isinstance(wrapper, dict) else {}
        observed_fields.update(row)
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        source = str(metadata.get("source_dataset", ""))
        source_counts[source] += 1
        license_value = str(row.get("license", ""))
        license_counts[license_value] += 1
        if source in RELEVANT_SOURCES:
            relevant_rows += 1
    missing = sorted(REQUIRED_FIELDS - observed_fields)
    if missing:
        raise RuntimeError("first-rows missing required fields: " + json.dumps(missing))
    if any(key and key.lower() != "cc-by-4.0" for key in license_counts):
        raise RuntimeError("unexpected license in first rows: " + json.dumps(dict(license_counts), sort_keys=True))
    if relevant_rows <= 0:
        raise RuntimeError("first-rows contained no KG1-relevant source_dataset rows")
    return {
        "row_count": len(rows),
        "observed_fields": sorted(observed_fields),
        "source_counts": dict(source_counts),
        "license_counts": dict(license_counts),
        "relevant_rows": relevant_rows,
        "truncated": bool(payload.get("truncated", False)),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_weak_contract(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        if args.skip_weak_overlap:
            return {"skipped": True, "reason": "huggingface_hub_not_installed"}
        raise RuntimeError("huggingface_hub is required unless --skip-weak-overlap is set") from exc
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = Path(
        hf_hub_download(
            repo_id=args.weak_data_repo,
            repo_type="dataset",
            filename=args.weak_csv_file,
            token=token or None,
        )
    )
    if validate_weak_csv is not None:
        meta = validate_weak_csv(path, args.expected_weak_csv_sha256, args.expected_shared_row_contract_sha256)
    else:
        meta = {"path": str(path), "sha256": "", "rows": len(read_csv_rows(path))}
    rows = read_csv_rows(path)
    weak_ids = {str(row.get("id", "")).strip() for row in rows}
    weak_prompt_hashes = {sha256_text(str(row.get("prompt", ""))) for row in rows if row.get("prompt")}
    weak_normalized_prompt_hashes = {sha256_text(normalize_prompt(str(row.get("prompt", "")))) for row in rows if row.get("prompt")}
    return {
        "skipped": False,
        "path": str(path),
        "meta": meta,
        "id_count": len(weak_ids),
        "prompt_hash_count": len(weak_prompt_hashes),
        "normalized_prompt_hash_count": len(weak_normalized_prompt_hashes),
        "ids": weak_ids,
        "prompt_hashes": weak_prompt_hashes,
        "normalized_prompt_hashes": weak_normalized_prompt_hashes,
    }


def source_family(source: str) -> str:
    if source in {"bitwise_arithmetic", "circuit_logic", "count_bits", "binary_alternation"}:
        return "bit_manipulation"
    if source in {"simple_equations", "cryptarithm", "polynomial_multiplication"}:
        return "equation_transform"
    if source in {"base_conversion", "number_format"}:
        return "numeral_or_bit_support"
    return "other"


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    question = str(row.get("question", ""))
    source = str(metadata.get("source_dataset", ""))
    return {
        "uuid": str(row.get("uuid", "")),
        "source_dataset": source,
        "kg1_relevance": source_family(source),
        "license": str(row.get("license", "")),
        "question": question,
        "answer": str(row.get("answer", "")),
        "question_sha256": sha256_text(question),
        "question_normalized_sha256": sha256_text(normalize_prompt(question)),
        "metadata": metadata,
    }


def iter_jsonl_url(url: str, timeout_s: int):
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def stream_and_filter(args: argparse.Namespace, weak: dict[str, Any]) -> dict[str, Any]:
    weak_ids = set(weak.get("ids", set()))
    weak_prompt_hashes = set(weak.get("prompt_hashes", set()))
    weak_norm_hashes = set(weak.get("normalized_prompt_hashes", set()))
    source_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    seen_uuid: set[str] = set()
    seen_question_hash: set[str] = set()
    relevant_sources = {item.strip() for item in args.relevant_sources.split(",") if item.strip()}

    for index, raw in enumerate(iter_jsonl_url(args.jsonl_url, args.timeout_s), start=1):
        if index > args.max_rows:
            break
        row = compact_row(raw)
        source = row["source_dataset"]
        source_counts[source] += 1
        license_counts[row["license"]] += 1
        if row["license"].lower() != "cc-by-4.0":
            continue
        if source not in relevant_sources:
            continue
        uuid = row["uuid"]
        question_hash = row["question_sha256"]
        norm_hash = row["question_normalized_sha256"]
        if uuid in seen_uuid or question_hash in seen_question_hash:
            continue
        if uuid in weak_ids:
            overlap_counts["uuid_vs_weak_id"] += 1
            continue
        if question_hash in weak_prompt_hashes:
            overlap_counts["question_sha_vs_weak_prompt_sha"] += 1
            continue
        if norm_hash in weak_norm_hashes:
            overlap_counts["question_norm_sha_vs_weak_prompt_norm_sha"] += 1
            continue
        if selected_counts[source] >= args.max_selected_per_source:
            continue
        seen_uuid.add(uuid)
        seen_question_hash.add(question_hash)
        selected_counts[source] += 1
        selected.append(row)

    if not selected:
        raise RuntimeError("no relevant rows selected from ReasoningGym stream")
    return {
        "scanned_rows": min(args.max_rows, sum(source_counts.values())),
        "source_counts": dict(source_counts),
        "selected_counts": dict(selected_counts),
        "license_counts": dict(license_counts),
        "overlap_counts": dict(overlap_counts),
        "selected_rows": selected,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V281 REASONINGGYM CPU TRIAGE START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("dataset =", args.dataset, flush=True)
    print("jsonl_url =", args.jsonl_url, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = json_url(datasets_server_url("splits", args.dataset, args.config), args.timeout_s)
    splits_meta = validate_splits(splits, args.split)
    print("splits_meta =", json.dumps(splits_meta, sort_keys=True), flush=True)
    first_rows = json_url(datasets_server_url("first-rows", args.dataset, args.config, args.split), args.timeout_s)
    first_rows_meta = validate_first_rows(first_rows)
    print("first_rows_meta =", json.dumps(first_rows_meta, sort_keys=True), flush=True)
    weak = load_weak_contract(args) if not args.skip_weak_overlap else {"skipped": True, "reason": "skip_weak_overlap"}
    print(
        "weak_overlap_gate =",
        json.dumps({key: value for key, value in weak.items() if key not in {"ids", "prompt_hashes", "normalized_prompt_hashes"}}, sort_keys=True),
        flush=True,
    )
    filtered = stream_and_filter(args, weak)
    selected_rows = filtered.pop("selected_rows")
    source_summary_rows = [
        {
            "source_dataset": source,
            "scanned_rows": int(filtered["source_counts"].get(source, 0)),
            "selected_rows": int(filtered["selected_counts"].get(source, 0)),
            "kg1_relevance": source_family(source),
        }
        for source in sorted(filtered["source_counts"])
    ]
    selected_path = args.output_dir / "v281_reasoninggym_selected_rows.jsonl"
    source_summary_path = args.output_dir / "v281_reasoninggym_source_summary.csv"
    manifest_path = args.output_dir / "v281_reasoninggym_cpu_triage_manifest.json"
    write_jsonl(selected_path, selected_rows)
    write_csv(source_summary_path, source_summary_rows, ["source_dataset", "kg1_relevance", "scanned_rows", "selected_rows"])
    manifest = {
        "schema_version": "kg1_v281_reasoninggym_cpu_triage_v1",
        "generated_at_utc": utc_now(),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "jsonl_url": args.jsonl_url,
        "splits_meta": splits_meta,
        "first_rows_meta": first_rows_meta,
        "weak_overlap_gate": {key: value for key, value in weak.items() if key not in {"ids", "prompt_hashes", "normalized_prompt_hashes"}},
        "stream_filter": filtered,
        "selected_rows": len(selected_rows),
        "outputs": {
            "selected_rows_jsonl": str(selected_path),
            "source_summary_csv": str(source_summary_path),
            "manifest_json": str(manifest_path),
        },
        "decision": {
            "decision": "reasoninggym_triage_ready_for_verifier_probes",
            "reason": f"selected_rows={len(selected_rows)}; sources={len(filtered['selected_counts'])}; overlaps={dict(filtered['overlap_counts'])}",
            "next_action": "Build CPU verifier probes from selected source datasets; do not train LoRA directly.",
        },
        "blocked_actions": ["gpu_train", "model_generation", "full_eval", "package", "kaggle_submit"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("selected_rows =", len(selected_rows), flush=True)
    print("selected_counts =", json.dumps(filtered["selected_counts"], sort_keys=True), flush=True)
    print("overlap_counts =", json.dumps(filtered["overlap_counts"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V281 REASONINGGYM CPU TRIAGE END ===", flush=True)
    return manifest


def self_test() -> None:
    fake = [
        {"uuid": "a", "question": "q bit", "answer": "1", "license": "cc-by-4.0", "metadata": {"source_dataset": "bitwise_arithmetic"}},
        {"uuid": "b", "question": "q eq", "answer": "x", "license": "cc-by-4.0", "metadata": {"source_dataset": "simple_equations"}},
        {"uuid": "c", "question": "q other", "answer": "z", "license": "cc-by-4.0", "metadata": {"source_dataset": "aiw"}},
    ]
    rows = [compact_row(item) for item in fake]
    if rows[0]["kg1_relevance"] != "bit_manipulation" or rows[1]["kg1_relevance"] != "equation_transform":
        raise AssertionError(rows)
    first_payload = {"rows": [{"row": item} for item in fake], "truncated": False}
    try:
        validate_first_rows(first_payload)
    except RuntimeError as exc:
        if "missing required fields" not in str(exc):
            raise
    print("v281_reasoninggym_cpu_triage_self_test=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--jsonl-url", default=DEFAULT_JSONL_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v281_reasoninggym_cpu_triage") / utc_compact())
    parser.add_argument("--max-rows", type=int, default=15000)
    parser.add_argument("--max-selected-per-source", type=int, default=200)
    parser.add_argument("--relevant-sources", default=",".join(sorted(RELEVANT_SOURCES)))
    parser.add_argument("--weak-data-repo", default=DEFAULT_DATA_REPO)
    parser.add_argument("--weak-csv-file", default=DEFAULT_WEAK_CSV_FILE)
    parser.add_argument("--expected-weak-csv-sha256", default=EXPECTED_WEAK_CSV_SHA256)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_SHARED_ROW_CONTRACT_SHA256)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--skip-weak-overlap", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

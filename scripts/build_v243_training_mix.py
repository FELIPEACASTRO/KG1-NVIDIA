#!/usr/bin/env python3
"""Build a guarded V243 train/validation mix from V217 plus V242 fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jsonl_overlap import build_reference, prompt_variants, read_jsonl, sha256_text  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_family(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("task_type") or row.get("type") or "unknown")


def row_source(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("metadata", {}).get("source") or "unknown")


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicate_ids = 0
    duplicate_prompts = 0
    for row in rows:
        rid = str(row.get("id") or "")
        prompt_hashes = {
            sha256_text(variant)
            for variant in prompt_variants(row)
            if variant
        }
        if rid and rid in seen_ids:
            duplicate_ids += 1
            continue
        if prompt_hashes and seen_prompts.intersection(prompt_hashes):
            duplicate_prompts += 1
            continue
        if rid:
            seen_ids.add(rid)
        seen_prompts.update(prompt_hashes)
        kept.append(row)
    return kept, {"duplicate_ids_removed": duplicate_ids, "duplicate_prompts_removed": duplicate_prompts}


def overlap_guard(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    reference = build_reference(reference_rows)
    ref_ids: set[str] = reference["ids"]
    ref_prompt_hashes: dict[str, str] = reference["prompt_hashes"]
    id_overlap: list[str] = []
    prompt_overlap: list[str] = []
    for row in rows:
        rid = str(row.get("id") or "")
        if rid and rid in ref_ids:
            id_overlap.append(rid)
        for variant in prompt_variants(row):
            digest = sha256_text(variant)
            if digest in ref_prompt_hashes:
                prompt_overlap.append(rid or digest)
                break
    report = {
        "label": label,
        "rows": len(rows),
        "reference_rows": len(reference_rows),
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "id_overlap_sample": id_overlap[:10],
        "prompt_overlap_sample": prompt_overlap[:10],
    }
    if id_overlap or prompt_overlap:
        raise RuntimeError(f"{label} overlaps reference rows: {json.dumps(report, sort_keys=True)}")
    return report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "families": dict(sorted(Counter(row_family(row) for row in rows).items())),
        "sources": dict(sorted(Counter(row_source(row) for row in rows).items())),
    }


def build_mix(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v217_train = read_jsonl(args.v217_train_jsonl)
    v217_val = read_jsonl(args.v217_val_jsonl)
    v242_train = read_jsonl(args.v242_train_jsonl)
    v242_val = read_jsonl(args.v242_validation_jsonl)

    train_rows, train_dedupe = dedupe_rows(v217_train + v242_train)
    val_rows, val_dedupe = dedupe_rows(v217_val + v242_val)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    overlap_reports: list[dict[str, Any]] = []
    if args.reference_jsonl:
        reference_rows = read_jsonl(args.reference_jsonl)
        overlap_reports.append(overlap_guard(train_rows, reference_rows, "train_vs_reference"))
        overlap_reports.append(overlap_guard(val_rows, reference_rows, "validation_vs_reference"))

    if len(train_rows) < args.min_train_rows:
        raise RuntimeError(f"train rows below floor: {len(train_rows)} < {args.min_train_rows}")
    if len(val_rows) < args.min_validation_rows:
        raise RuntimeError(f"validation rows below floor: {len(val_rows)} < {args.min_validation_rows}")

    train_path = output_dir / f"{args.label}_train.jsonl"
    val_path = output_dir / f"{args.label}_validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v243_training_mix_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "inputs": {
            "v217_train_jsonl": str(args.v217_train_jsonl),
            "v217_val_jsonl": str(args.v217_val_jsonl),
            "v242_train_jsonl": str(args.v242_train_jsonl),
            "v242_validation_jsonl": str(args.v242_validation_jsonl),
            "reference_jsonl": str(args.reference_jsonl) if args.reference_jsonl else "",
        },
        "dedupe": {"train": train_dedupe, "validation": val_dedupe},
        "overlap_reports": overlap_reports,
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "validation_jsonl": str(val_path),
            "validation_sha256": sha256_file(val_path),
        },
        "input_summaries": {
            "v217_train": summarize(v217_train),
            "v217_validation": summarize(v217_val),
            "v242_train": summarize(v242_train),
            "v242_validation": summarize(v242_val),
        },
        "output_summaries": {
            "train": summarize(train_rows),
            "validation": summarize(val_rows),
        },
        "decision": {
            "status": "ok",
            "next_action": "Run tokenization dry-run before any GPU training.",
        },
    }
    manifest_path = output_dir / f"{args.label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("v243_training_mix_manifest =", json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v243_training_mix")
    parser.add_argument("--v217-train-jsonl", type=Path, default=ROOT / "data/v217/v217_short_answer_train.jsonl")
    parser.add_argument("--v217-val-jsonl", type=Path, default=ROOT / "data/v217/v217_short_answer_val.jsonl")
    parser.add_argument("--v242-train-jsonl", type=Path, required=True)
    parser.add_argument("--v242-validation-jsonl", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=243)
    parser.add_argument("--min-train-rows", type=int, default=12000)
    parser.add_argument("--min-validation-rows", type=int, default=900)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v217_train = root / "v217_train.jsonl"
        v217_val = root / "v217_val.jsonl"
        v242_train = root / "v242_train.jsonl"
        v242_val = root / "v242_val.jsonl"
        ref = root / "ref.jsonl"
        write_jsonl(v217_train, [{"id": "a", "prompt": "p a", "answer": "1", "family": "equation_transform", "source": "v217"}])
        write_jsonl(v217_val, [{"id": "b", "prompt": "p b", "answer": "2", "family": "equation_transform", "source": "v217"}])
        write_jsonl(v242_train, [{"id": "c", "prompt": "p c", "answer": "3", "family": "equation_transform", "source": "v242"}])
        write_jsonl(v242_val, [{"id": "d", "prompt": "p d", "answer": "4", "family": "equation_transform", "source": "v242"}])
        write_jsonl(ref, [{"id": "z", "prompt": "p z", "answer": "9"}])
        ns = argparse.Namespace(
            output_dir=root / "out",
            label="selftest",
            v217_train_jsonl=v217_train,
            v217_val_jsonl=v217_val,
            v242_train_jsonl=v242_train,
            v242_validation_jsonl=v242_val,
            reference_jsonl=ref,
            seed=1,
            min_train_rows=2,
            min_validation_rows=2,
        )
        manifest = build_mix(ns)
        assert manifest["output_summaries"]["train"]["rows"] == 2
        assert manifest["output_summaries"]["validation"]["rows"] == 2
    print("v243_training_mix_self_test=ok", flush=True)


def main() -> int:
    args = parse_args()
    print("=== V243 TRAINING MIX START ===", flush=True)
    if args.self_test:
        self_test()
    else:
        build_mix(args)
    print("=== V243 TRAINING MIX END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

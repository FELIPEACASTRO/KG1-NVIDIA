#!/usr/bin/env python3
"""Audit JSONL datasets for ID/prompt overlap against reference JSONL rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def prompt_variants(row: dict[str, Any]) -> list[str]:
    variants: list[str] = []
    if row.get("prompt") is not None:
        variants.append(str(row.get("prompt", "")))
    messages = row.get("messages")
    if isinstance(messages, list):
        user_parts: list[str] = []
        all_parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            if not content:
                continue
            all_parts.append(content)
            if message.get("role") == "user":
                user_parts.append(content)
        if user_parts:
            variants.append("\n".join(user_parts))
        if all_parts:
            variants.append("\n".join(all_parts))
    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = normalize_text(variant)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(variant)
    return unique


def build_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {str(row.get("id", "")) for row in rows if str(row.get("id", ""))}
    prompt_hashes: dict[str, str] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        for variant in prompt_variants(row):
            prompt_hashes[sha256_text(variant)] = row_id
    return {"ids": ids, "prompt_hashes": prompt_hashes}


def audit_candidate(path: Path, reference: dict[str, Any], preview_limit: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    ref_ids: set[str] = reference["ids"]
    ref_prompt_hashes: dict[str, str] = reference["prompt_hashes"]
    id_overlap: list[dict[str, str]] = []
    prompt_overlap: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id and row_id in ref_ids:
            id_overlap.append({"candidate_id": row_id, "reference_id": row_id})
        for variant in prompt_variants(row):
            digest = sha256_text(variant)
            if digest in ref_prompt_hashes:
                prompt_overlap.append(
                    {
                        "candidate_id": row_id,
                        "reference_id": ref_prompt_hashes[digest],
                        "prompt_sha256": digest,
                    }
                )
    return {
        "candidate_jsonl": str(path),
        "rows": len(rows),
        "id_overlap_count": len(id_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "id_overlap_preview": id_overlap[:preview_limit],
        "prompt_overlap_preview": prompt_overlap[:preview_limit],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== JSONL OVERLAP AUDIT START ===", flush=True)
    print("reference_jsonl =", args.reference_jsonl, flush=True)
    print("candidate_jsonl =", [str(path) for path in args.candidate_jsonl], flush=True)
    print("fail_on_id_overlap =", args.fail_on_id_overlap, flush=True)
    print("fail_on_prompt_overlap =", args.fail_on_prompt_overlap, flush=True)
    reference_rows = read_jsonl(args.reference_jsonl)
    reference = build_reference(reference_rows)
    results = [audit_candidate(path, reference, args.preview_limit) for path in args.candidate_jsonl]
    payload = {
        "schema_version": "kg1_jsonl_overlap_audit_v1",
        "reference_jsonl": str(args.reference_jsonl),
        "reference_rows": len(reference_rows),
        "reference_id_count": len(reference["ids"]),
        "reference_prompt_hash_count": len(reference["prompt_hashes"]),
        "results": results,
    }
    if args.output_json:
        write_json(args.output_json, payload)
        print("output_json =", args.output_json, flush=True)
    print("summary =", json.dumps(results, indent=2, sort_keys=True), flush=True)
    print("=== JSONL OVERLAP AUDIT END ===", flush=True)
    blocked: list[str] = []
    for result in results:
        if args.fail_on_id_overlap and int(result["id_overlap_count"]):
            blocked.append(f"{result['candidate_jsonl']}: id_overlap_count={result['id_overlap_count']}")
        if args.fail_on_prompt_overlap and int(result["prompt_overlap_count"]):
            blocked.append(f"{result['candidate_jsonl']}: prompt_overlap_count={result['prompt_overlap_count']}")
    if blocked:
        raise RuntimeError("JSONL overlap audit blocked: " + "; ".join(blocked))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--preview-limit", type=int, default=10)
    parser.add_argument("--fail-on-id-overlap", action="store_true")
    parser.add_argument("--fail-on-prompt-overlap", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference = root / "reference.jsonl"
        clean = root / "clean.jsonl"
        dirty = root / "dirty.jsonl"
        reference.write_text(
            json.dumps({"id": "weak1", "prompt": "Now solve A"}) + "\n",
            encoding="utf-8",
        )
        clean.write_text(
            json.dumps({"id": "train1", "messages": [{"role": "user", "content": "Now solve B"}]}) + "\n",
            encoding="utf-8",
        )
        dirty.write_text(
            json.dumps({"id": "weak1", "messages": [{"role": "user", "content": "Now solve A"}]}) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            reference_jsonl=reference,
            candidate_jsonl=[clean],
            output_json=None,
            preview_limit=10,
            fail_on_id_overlap=True,
            fail_on_prompt_overlap=True,
        )
        payload = run(args)
        if payload["results"][0]["id_overlap_count"] != 0:
            raise AssertionError("clean self-test unexpectedly overlapped by id")
        args.candidate_jsonl = [dirty]
        try:
            run(args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dirty self-test did not block")
    print("jsonl_overlap_audit_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.reference_jsonl is None:
        parser.error("--reference-jsonl is required unless --self-test is used")
    if not args.candidate_jsonl:
        parser.error("--candidate-jsonl is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

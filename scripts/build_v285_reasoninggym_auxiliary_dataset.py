#!/usr/bin/env python3
"""Build V285 ReasoningGym auxiliary fixtures for KG1 probe design.

This is a CPU-only data-preparation step. It consumes the public V281
ReasoningGym triage rows plus the V282 deterministic verifier audit, keeps only
rows whose answers were independently verified, and emits a small JSONL corpus
in the same chat format used by the KG1 SFT scripts.

V285 is deliberately not a training authorization. The output is meant to pass
tokenization/leakage gates first, then support a budget-conscious ablation only
if the gate and roadmap justify it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

DEFAULT_SELECTED_ROWS = Path(
    "artifacts/v281_reasoninggym_cpu_triage/20260511T1835Z/v281_reasoninggym_selected_rows.jsonl"
)
DEFAULT_V281_MANIFEST = Path(
    "artifacts/v281_reasoninggym_cpu_triage/20260511T1835Z/v281_reasoninggym_cpu_triage_manifest.json"
)
DEFAULT_VERIFIER_AUDIT = Path(
    "artifacts/v282_reasoninggym_verifier_probes/20260511T1855Z/v282_reasoninggym_verifier_audit.csv"
)
DEFAULT_CORE_SOURCES = (
    "binary_alternation,bitwise_arithmetic,count_bits,cryptarithm,simple_equations"
)
DEFAULT_SUPPORT_SOURCES = "base_conversion,number_format"
EXPECTED_SHARED_ROW_CONTRACT_SHA256 = (
    "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def normalize_answer(answer: Any) -> str:
    return re.sub(r"\s+", "", str(answer)).strip()


def parse_source_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def source_family(source: str) -> str:
    if source in {"binary_alternation", "bitwise_arithmetic", "count_bits"}:
        return "bit_manipulation"
    if source in {"cryptarithm", "simple_equations"}:
        return "equation_transform"
    if source in {"base_conversion", "number_format"}:
        return "numeral_or_bit_support"
    return "other"


def source_subcategory(source: str) -> str:
    if source == "binary_alternation":
        return "bit_alternation_min_swaps"
    if source == "bitwise_arithmetic":
        return "bitwise_arithmetic_expression"
    if source == "count_bits":
        return "bit_counting"
    if source == "cryptarithm":
        return "equation_cryptarithm_mapping"
    if source == "simple_equations":
        return "equation_linear_numeric"
    if source == "base_conversion":
        return "numeral_base_conversion"
    if source == "number_format":
        return "numeral_format_selection"
    return source


def row_id(split: str, source: str, uuid: str) -> str:
    safe_uuid = re.sub(r"[^a-zA-Z0-9]+", "", uuid)[:18]
    return f"v285_{split}_{source}_{safe_uuid}"


def make_messages(prompt: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Final answer: " + str(answer).strip()},
    ]


def validate_v281_manifest(path: Path, expected_contract: str) -> dict[str, Any]:
    manifest = read_json(path)
    print("v281_manifest_path =", path, flush=True)
    print("v281_schema_version =", manifest.get("schema_version", ""), flush=True)
    overlap_counts = manifest.get("stream_filter", {}).get("overlap_counts", {})
    print("v281_overlap_counts =", json.dumps(overlap_counts, sort_keys=True), flush=True)
    if overlap_counts:
        raise RuntimeError("V281 selected rows overlap weak rows: " + json.dumps(overlap_counts, sort_keys=True))
    observed_contract = str(
        manifest.get("weak_overlap_gate", {})
        .get("meta", {})
        .get("observed_shared_row_contract_sha256", "")
    )
    print("v281_observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if expected_contract and observed_contract != expected_contract:
        raise RuntimeError(
            "V281 weak contract mismatch: expected "
            + expected_contract
            + ", got "
            + observed_contract
        )
    return manifest


def verified_lookup(audit_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    duplicate_uuid: list[str] = []
    for row in audit_rows:
        uuid = str(row.get("uuid", "")).strip()
        if not uuid:
            raise RuntimeError("verifier audit row without uuid")
        if uuid in lookup:
            duplicate_uuid.append(uuid)
        lookup[uuid] = row
    if duplicate_uuid:
        raise RuntimeError("duplicate uuid in verifier audit: " + json.dumps(duplicate_uuid[:10]))
    return lookup


def convert_row(row: dict[str, Any], audit_row: dict[str, str], split: str, label: str) -> dict[str, Any]:
    source = str(row.get("source_dataset", ""))
    family = source_family(source)
    prompt = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    uuid = str(row.get("uuid", "")).strip()
    if not prompt or not answer or not uuid:
        raise RuntimeError("selected row missing prompt, answer, or uuid")
    expected = str(audit_row.get("expected", "")).strip()
    prediction = str(audit_row.get("prediction", "")).strip()
    if normalize_answer(expected) != normalize_answer(answer):
        raise RuntimeError(f"verifier expected answer differs from selected answer for uuid={uuid}")
    if normalize_answer(prediction) != normalize_answer(answer):
        raise RuntimeError(f"verifier prediction answer differs from selected answer for uuid={uuid}")
    return {
        "id": row_id(split, source, uuid),
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "subcategory": source_subcategory(source),
        "source": "v285_reasoninggym_verified_auxiliary",
        "messages": make_messages(prompt, answer),
        "metadata": {
            "answer_style": "final_answer_one_line_unboxed",
            "family": family,
            "kg1_relevance": row.get("kg1_relevance", ""),
            "license": row.get("license", ""),
            "question_normalized_sha256": row.get("question_normalized_sha256", ""),
            "question_sha256": row.get("question_sha256", ""),
            "source_dataset": source,
            "split": split,
            "subcategory": source_subcategory(source),
            "train_allowed": False,
            "v285_label": label,
            "v285_role": split,
            "v285_training_authorization": "blocked_until_v286_tokenization_gate_and_budget_decision",
            "verifier_proof": audit_row.get("proof", ""),
            "verifier_status": audit_row.get("status", ""),
            "weak_gate_rows_used_for_training": False,
        },
    }


def validate_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    ids: list[str] = []
    prompt_hashes: list[str] = []
    prompt_answer_hashes: list[str] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    bad_rows: list[str] = []
    for row in rows:
        rid = str(row.get("id", ""))
        ids.append(rid)
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        prompt_hashes.append(sha256_text(normalize_prompt(prompt)))
        prompt_answer_hashes.append(sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer)))
        family_counts[str(row.get("family", ""))] += 1
        source_counts[str(row.get("metadata", {}).get("source_dataset", ""))] += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            bad_rows.append(rid)
            continue
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            bad_rows.append(rid)
            continue
        if messages[1].get("content") != prompt:
            bad_rows.append(rid)
            continue
        if messages[2].get("content") != "Final answer: " + answer:
            bad_rows.append(rid)
    if bad_rows:
        raise RuntimeError(f"{split} bad chat/message rows: {bad_rows[:20]}")
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompt_hashes) - len(set(prompt_hashes))
    duplicate_prompt_answers = len(prompt_answer_hashes) - len(set(prompt_answer_hashes))
    if duplicate_ids or duplicate_prompts or duplicate_prompt_answers:
        raise RuntimeError(
            f"{split} duplicates detected: ids={duplicate_ids} prompts={duplicate_prompts} "
            f"prompt_answers={duplicate_prompt_answers}"
        )
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "prompt_hash_count": len(set(prompt_hashes)),
        "prompt_answer_hash_count": len(set(prompt_answer_hashes)),
    }


def split_stratified(rows: list[dict[str, Any]], seed: int, val_fraction: float, min_val_per_source: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("metadata", {}).get("source_dataset", ""))].append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for source, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        val_count = max(min_val_per_source, int(round(len(items) * val_fraction)))
        val_count = min(max(1, val_count), len(items) - 1)
        for item in items[:val_count]:
            item = json.loads(json.dumps(item))
            item["id"] = item["id"].replace("_train_", "_validation_")
            item["metadata"]["split"] = "validation"
            item["metadata"]["v285_role"] = "validation"
            val_rows.append(item)
        for item in items[val_count:]:
            train_rows.append(item)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V285 REASONINGGYM AUXILIARY DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("selected_rows_jsonl =", args.selected_rows_jsonl, flush=True)
    print("verifier_audit_csv =", args.verifier_audit_csv, flush=True)
    print("v281_manifest_json =", args.v281_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("include_support =", args.include_support, flush=True)
    print("val_fraction =", args.val_fraction, flush=True)
    print("seed =", args.seed, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v281_manifest = validate_v281_manifest(args.v281_manifest_json, args.expected_shared_row_contract_sha256)
    selected_rows = read_jsonl(args.selected_rows_jsonl)
    audit_rows = read_csv(args.verifier_audit_csv)
    lookup = verified_lookup(audit_rows)
    include_sources = parse_source_list(args.core_sources)
    if args.include_support:
        include_sources |= parse_source_list(args.support_sources)
    print("include_sources =", json.dumps(sorted(include_sources)), flush=True)

    skipped_status: Counter[str] = Counter()
    skipped_source: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        uuid = str(row.get("uuid", "")).strip()
        source = str(row.get("source_dataset", "")).strip()
        audit_row = lookup.get(uuid)
        if audit_row is None:
            raise RuntimeError("selected row missing from verifier audit uuid=" + uuid)
        status = str(audit_row.get("status", "")).strip()
        if status != "verified_match":
            skipped_status[status or "missing"] += 1
            continue
        if source not in include_sources:
            skipped_source[source or "missing"] += 1
            continue
        candidate_rows.append(convert_row(row, audit_row, "train", args.label))

    if len(candidate_rows) < args.min_total_rows:
        raise RuntimeError(f"not enough V285 candidate rows: {len(candidate_rows)} < {args.min_total_rows}")
    train_rows, val_rows = split_stratified(
        candidate_rows,
        args.seed,
        args.val_fraction,
        args.min_val_per_source,
    )
    train_validation = validate_rows(train_rows, "train")
    val_validation = validate_rows(val_rows, "validation")
    train_pa = {sha256_text(normalize_prompt(row["prompt"]) + "\0" + normalize_answer(row["answer"])) for row in train_rows}
    val_pa = {sha256_text(normalize_prompt(row["prompt"]) + "\0" + normalize_answer(row["answer"])) for row in val_rows}
    if train_pa & val_pa:
        raise RuntimeError("train/validation prompt+answer overlap detected")

    train_path = args.output_dir / "v285_reasoninggym_auxiliary_train.jsonl"
    val_path = args.output_dir / "v285_reasoninggym_auxiliary_val.jsonl"
    manifest_path = args.output_dir / "v285_reasoninggym_auxiliary_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v285_reasoninggym_auxiliary_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_dataset": "nvidia/Nemotron-RL-ReasoningGym-v1",
        "inputs": {
            "selected_rows_jsonl": str(args.selected_rows_jsonl),
            "selected_rows_sha256": sha256_file(args.selected_rows_jsonl),
            "verifier_audit_csv": str(args.verifier_audit_csv),
            "verifier_audit_sha256": sha256_file(args.verifier_audit_csv),
            "v281_manifest_json": str(args.v281_manifest_json),
            "v281_manifest_sha256": sha256_file(args.v281_manifest_json),
            "v281_stream_filter_overlap_counts": v281_manifest.get("stream_filter", {}).get("overlap_counts", {}),
            "v281_shared_row_contract_sha256": (
                v281_manifest.get("weak_overlap_gate", {})
                .get("meta", {})
                .get("observed_shared_row_contract_sha256", "")
            ),
        },
        "config": {
            "core_sources": sorted(parse_source_list(args.core_sources)),
            "support_sources": sorted(parse_source_list(args.support_sources)),
            "include_support": bool(args.include_support),
            "included_sources": sorted(include_sources),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "min_val_per_source": args.min_val_per_source,
        },
        "selection": {
            "selected_rows": len(selected_rows),
            "verifier_audit_rows": len(audit_rows),
            "candidate_rows": len(candidate_rows),
            "skipped_status_counts": dict(sorted(skipped_status.items())),
            "skipped_source_counts": dict(sorted(skipped_source.items())),
        },
        "validation": {
            "train": train_validation,
            "validation": val_validation,
            "train_val_prompt_answer_overlap": 0,
        },
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "blocked_actions": [
            "gpu_train_without_v286_tokenization_gate",
            "model_generation",
            "full_eval",
            "package",
            "kaggle_submit",
        ],
        "decision": {
            "status": "dataset_ready_for_tokenization_gate",
            "next_action": "Run V286 generic tokenization/leakage gate before any HF GPU ablation.",
            "reason": (
                f"candidate_rows={len(candidate_rows)}; train={len(train_rows)}; "
                f"validation={len(val_rows)}; verified_mismatch=0; weak_overlap=0"
            ),
        },
    }
    write_json(manifest_path, manifest)
    print("v285_manifest_json =", manifest_path, flush=True)
    print("v285_outputs =", json.dumps(manifest["outputs"], sort_keys=True), flush=True)
    print("v285_validation =", json.dumps(manifest["validation"], sort_keys=True), flush=True)
    print("v285_decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("=== V285 REASONINGGYM AUXILIARY DATASET END ===", flush=True)
    return manifest


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        selected = tmp / "selected.jsonl"
        audit = tmp / "audit.csv"
        v281_manifest = tmp / "v281.json"
        out = tmp / "out"
        rows = [
            {
                "uuid": "u1",
                "source_dataset": "count_bits",
                "kg1_relevance": "bit_manipulation",
                "license": "cc-by-4.0",
                "question": "Count ones in the binary representation of the number 7",
                "answer": "3",
                "question_sha256": "q1",
                "question_normalized_sha256": "nq1",
            },
            {
                "uuid": "u2",
                "source_dataset": "simple_equations",
                "kg1_relevance": "equation_transform",
                "license": "cc-by-4.0",
                "question": "Find the value of x in the equation: 2*x = 8",
                "answer": "4",
                "question_sha256": "q2",
                "question_normalized_sha256": "nq2",
            },
            {
                "uuid": "u3",
                "source_dataset": "count_bits",
                "kg1_relevance": "bit_manipulation",
                "license": "cc-by-4.0",
                "question": "Count ones in the binary representation of the number 3",
                "answer": "2",
                "question_sha256": "q3",
                "question_normalized_sha256": "nq3",
            },
            {
                "uuid": "u4",
                "source_dataset": "simple_equations",
                "kg1_relevance": "equation_transform",
                "license": "cc-by-4.0",
                "question": "Find the value of z in the equation: z + 1 = 5",
                "answer": "4",
                "question_sha256": "q4",
                "question_normalized_sha256": "nq4",
            },
        ]
        write_jsonl(selected, rows)
        with audit.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["uuid", "source_dataset", "kg1_relevance", "status", "expected", "prediction", "proof", "question_sha256"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "uuid": row["uuid"],
                        "source_dataset": row["source_dataset"],
                        "kg1_relevance": row["kg1_relevance"],
                        "status": "verified_match",
                        "expected": row["answer"],
                        "prediction": row["answer"],
                        "proof": "toy",
                        "question_sha256": row["question_sha256"],
                    }
                )
        write_json(
            v281_manifest,
            {
                "schema_version": "toy",
                "stream_filter": {"overlap_counts": {}},
                "weak_overlap_gate": {
                    "meta": {"observed_shared_row_contract_sha256": EXPECTED_SHARED_ROW_CONTRACT_SHA256}
                },
            },
        )
        args = argparse.Namespace(
            selected_rows_jsonl=selected,
            verifier_audit_csv=audit,
            v281_manifest_json=v281_manifest,
            output_dir=out,
            label="selftest",
            core_sources="count_bits,simple_equations",
            support_sources="",
            include_support=False,
            seed=1,
            val_fraction=0.25,
            min_val_per_source=1,
            min_total_rows=4,
            expected_shared_row_contract_sha256=EXPECTED_SHARED_ROW_CONTRACT_SHA256,
        )
        manifest = build(args)
        assert manifest["validation"]["train"]["rows"] == 2
        assert manifest["validation"]["validation"]["rows"] == 2
    print("v285_reasoninggym_auxiliary_dataset_self_test=ok", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-rows-jsonl", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--verifier-audit-csv", type=Path, default=DEFAULT_VERIFIER_AUDIT)
    parser.add_argument("--v281-manifest-json", type=Path, default=DEFAULT_V281_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v285_reasoninggym_auxiliary_dataset") / utc_compact())
    parser.add_argument("--label", default="v285_reasoninggym_auxiliary")
    parser.add_argument("--core-sources", default=DEFAULT_CORE_SOURCES)
    parser.add_argument("--support-sources", default=DEFAULT_SUPPORT_SOURCES)
    parser.add_argument("--include-support", action="store_true")
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--min-val-per-source", type=int, default=12)
    parser.add_argument("--min-total-rows", type=int, default=700)
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_SHARED_ROW_CONTRACT_SHA256)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

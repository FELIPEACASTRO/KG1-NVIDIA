#!/usr/bin/env python3
"""Build V322 V321+V51 filtered hybrid answer-span dataset.

V321 showed no complementarity versus V290. V322 changes the data source instead
of only changing LR/modules: it keeps the audited V321 replay/focused rows and
adds V51 solver-enhanced public-train rows after removing all known weak/full
gate rows by id and prompt hash. The added V51 rows are converted to a compact
RULE/CHECK/Final answer format so answer-span weighting is not diluted by long
free-form CoT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V321_ROOT = REPO_ROOT / "artifacts/v321_hybrid_answer_span_dataset/20260513T0400Z"
DEFAULT_V51_JSONL = REPO_ROOT / "data/sft_v51_complete.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v322_v51_filtered_hybrid_dataset/20260513T0620Z"
DEFAULT_REFERENCE_PATHS = [
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv",
]

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify the candidate briefly, "
    "then end with exactly one final answer in \\boxed{}."
)

FAMILY_MAP = {
    "bit": "bit_manipulation",
    "equation": "equation_transform",
    "gravity": "gravity_constant",
    "numeral": "numeral_system",
    "cipher": "text_encryption",
    "unit": "unit_conversion",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def prompt_hash(prompt: Any) -> str:
    return sha256_text(normalize_prompt(prompt))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_reference_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_reference_fingerprints(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_reference_rows(path)
        for row in rows:
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                hashes.add(prompt_hash(prompt))
        summaries.append({"path": str(path), "rows": len(rows), "sha256": sha256_file(path)})
    return ids, hashes, summaries


def compact_rule_trace(completion: str, answer: str, family: str) -> str:
    text = str(completion)
    text = re.sub(r"</?think>", "\n", text)
    text = re.sub(r"\\boxed\{.*?\}", "", text, flags=re.DOTALL)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.lower().startswith("final answer")]
    preferred: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(key in lower for key in ("global", "solver", "rule", "result:", "transformation", "testing", "produces")):
            preferred.append(line)
    if not preferred:
        preferred = lines[:2]
    trace = " ".join(preferred[:3])
    trace = trace[:700].strip()
    if not trace:
        trace = f"Infer the {family} transformation from the examples and apply it to the query."
    return trace


def make_v51_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_family = str(row.get("family", "")).strip()
    family = FAMILY_MAP.get(raw_family, raw_family)
    answer = str(row.get("answer", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    rule_trace = compact_rule_trace(str(row.get("completion", "")), answer, family)
    assistant = (
        "RULE: " + rule_trace + "\n"
        "CHECK: Apply the inferred rule to the query and keep the answer exact.\n"
        "Final answer: " + r"\boxed{" + answer + "}"
    )
    source = "v51_complete_filtered"
    subcategory = "v51_" + raw_family + "_solver_cot_compact"
    return {
        "id": "v322_v51_" + str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "source": source,
        "subcategory": subcategory,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source_dataset": source,
            "source_row_id": str(row.get("id", "")).strip(),
            "source_family": raw_family,
            "source_method": str(row.get("source") or row.get("method") or ""),
            "weak_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "v322_gate_filtered": True,
        },
    }


def normalize_existing_row(row: dict[str, Any], source_tag: str) -> dict[str, Any]:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source_dataset", item.get("source") or source_tag)
    metadata["weak_gate_rows_used_for_training"] = False
    metadata["gate_rows_used_for_training"] = False
    metadata["full_gate_rows_used_for_training"] = False
    metadata["v322_inherited_source"] = source_tag
    answer = str(item.get("answer", "")).strip()
    messages = item.get("messages")
    if isinstance(messages, list) and len(messages) == 3 and answer:
        normalized_messages = [dict(message) for message in messages]
        assistant = str(normalized_messages[2].get("content", ""))
        boxed_suffix = "Final answer: " + r"\boxed{" + answer + "}"
        if not assistant.rstrip().endswith(boxed_suffix):
            metadata["v322_original_assistant_sha256"] = sha256_text(assistant)
            assistant = re.sub(r"\s*Final answer:\s*.*$", "", assistant.rstrip(), flags=re.DOTALL).rstrip()
            if assistant:
                assistant += "\n"
            assistant += boxed_suffix
            normalized_messages[2]["content"] = assistant
        item["messages"] = normalized_messages
    item["metadata"] = metadata
    return item


def audit(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [prompt_hash(row.get("prompt", "")) for row in rows]
    families = Counter(str(row.get("family", "")) for row in rows)
    sources = Counter(str(row.get("source", "")) for row in rows)
    subcategories = Counter(str(row.get("subcategory", "")) for row in rows)
    bad: list[dict[str, Any]] = []
    if len(ids) != len(set(ids)):
        bad.append({"reason": "duplicate ids", "count": len(ids) - len(set(ids))})
    if len(prompts) != len(set(prompts)):
        bad.append({"reason": "duplicate prompts", "count": len(prompts) - len(set(prompts))})
    for index, row in enumerate(rows[:]):
        messages = row.get("messages")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not row.get("id") or not row.get("prompt") or not row.get("answer"):
            bad.append({"index": index, "id": row.get("id", ""), "reason": "missing id/prompt/answer"})
        if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            bad.append({"index": index, "id": row.get("id", ""), "reason": "bad messages"})
        for flag in ("weak_gate_rows_used_for_training", "gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if metadata.get(flag) is not False:
                bad.append({"index": index, "id": row.get("id", ""), "reason": f"{flag} not false"})
    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_prompt_hashes": len(set(prompts)),
        "family_counts": dict(sorted(families.items())),
        "source_counts": dict(sorted(sources.items())),
        "subcategory_counts": dict(sorted(subcategories.items())),
        "bad_rows_first10": bad[:10],
    }


def deterministic_val(row: dict[str, Any]) -> bool:
    digest = sha256_text(str(row.get("id", "")))
    return int(digest[:8], 16) % 20 == 0


def build(v321_root: Path, v51_jsonl: Path, output_root: Path, reference_paths: list[Path]) -> dict[str, Any]:
    print("=== V322 V51 FILTERED HYBRID DATASET BUILD START ===", flush=True)
    print("v321_root =", v321_root, flush=True)
    print("v51_jsonl =", v51_jsonl, flush=True)
    print("output_root =", output_root, flush=True)
    for path in [v321_root / "v321_hybrid_answer_span_train.jsonl", v321_root / "v321_hybrid_answer_span_val.jsonl", v51_jsonl]:
        print("input_exists =", path, path.exists(), flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    ref_ids, ref_hashes, ref_summaries = read_reference_fingerprints(reference_paths)
    print("reference_id_count =", len(ref_ids), flush=True)
    print("reference_prompt_hash_count =", len(ref_hashes), flush=True)

    v321_train = [
        normalize_existing_row(row, "v321_hybrid_answer_span")
        for row in read_jsonl(v321_root / "v321_hybrid_answer_span_train.jsonl")
    ]
    v321_val = [
        normalize_existing_row(row, "v321_hybrid_answer_span")
        for row in read_jsonl(v321_root / "v321_hybrid_answer_span_val.jsonl")
    ]
    existing_ids = {str(row.get("id", "")) for row in v321_train + v321_val}
    existing_prompt_hashes = {prompt_hash(row.get("prompt", "")) for row in v321_train + v321_val}

    v51_rows = read_jsonl(v51_jsonl)
    filtered: list[dict[str, Any]] = []
    filter_reasons: Counter[str] = Counter()
    for row in v51_rows:
        rid = str(row.get("id", "")).strip()
        phash = prompt_hash(row.get("prompt", ""))
        raw_family = str(row.get("family", "")).strip()
        if rid in ref_ids or phash in ref_hashes:
            filter_reasons["known_gate_reference"] += 1
            continue
        if "v322_v51_" + rid in existing_ids or phash in existing_prompt_hashes:
            filter_reasons["duplicate_with_v321"] += 1
            continue
        if raw_family not in FAMILY_MAP:
            filter_reasons["unknown_family"] += 1
            continue
        filtered.append(make_v51_row(row))

    v51_train = [row for row in filtered if not deterministic_val(row)]
    v51_val = [row for row in filtered if deterministic_val(row)]
    train_rows = v321_train + v51_train
    val_rows = v321_val + v51_val

    train_audit = audit(train_rows, "train")
    val_audit = audit(val_rows, "validation")
    print("v51_input_rows =", len(v51_rows), flush=True)
    print("v51_filtered_rows =", len(filtered), flush=True)
    print("filter_reasons =", json.dumps(dict(sorted(filter_reasons.items())), sort_keys=True), flush=True)
    print("train_audit =", json.dumps(train_audit, sort_keys=True), flush=True)
    print("val_audit =", json.dumps(val_audit, sort_keys=True), flush=True)
    if train_audit["bad_rows_first10"] or val_audit["bad_rows_first10"]:
        raise RuntimeError("V322 dataset audit failed")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "v322_v51_filtered_hybrid_train.jsonl"
    val_path = output_root / "v322_v51_filtered_hybrid_val.jsonl"
    manifest_path = output_root / "v322_v51_filtered_hybrid_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "schema_version": "kg1_v322_v51_filtered_hybrid_dataset_v1",
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
        },
        "v51_input_rows": len(v51_rows),
        "v51_filtered_rows": len(filtered),
        "v51_train_rows": len(v51_train),
        "v51_val_rows": len(v51_val),
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "reference_summaries": ref_summaries,
        "train_audit": train_audit,
        "val_audit": val_audit,
        "policy": {
            "known_weak_full_gate_rows_filtered": True,
            "known_gate_reference_ids": len(ref_ids),
            "known_gate_reference_prompt_hashes": len(ref_hashes),
            "assistant_format": "compact RULE/CHECK/Final answer boxed suffix",
            "promotion_gate": "weak total>=193, equation>=60, bit>=136, truncation no worse",
        },
    }
    write_json(manifest_path, manifest)
    print("manifest_path =", manifest_path, flush=True)
    print("manifest =", json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V322 V51 FILTERED HYBRID DATASET BUILD END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v321-root", type=Path, default=DEFAULT_V321_ROOT)
    parser.add_argument("--v51-jsonl", type=Path, default=DEFAULT_V51_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-path", type=Path, action="append", default=None)
    args = parser.parse_args()
    build(args.v321_root, args.v51_jsonl, args.output_root, args.reference_path or DEFAULT_REFERENCE_PATHS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

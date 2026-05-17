#!/usr/bin/env python3
"""Build V536 mixed pack from V534 bit traces plus V523 equation traces.

V536 is the first combined dataset after the V534 bit-only source pack. It
keeps the proven V523 equation source rows and replaces the V523 bit rows with
shorter V534 bit source-only traces at the same row quota, so the next gates can
compare one controlled change instead of another broad SFT mixture.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer_for_expected, verify_answer  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v536_v534_bit_v523_equation_pack"
DEFAULT_V523_ROOT = ROOT / "artifacts/v523_targeted_source_trace_pack"
DEFAULT_V534_ROOT = ROOT / "artifacts/v534_bit_source_only_trace_pack"
DEFAULT_WEAK_REFERENCE = ROOT / "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv"
DEFAULT_FULL_REFERENCE = ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def latest_manifest(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(f"*/{pattern}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not hits:
        raise RuntimeError(f"no manifest matching {pattern} under {root}")
    return hits[0]


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def prompt_hash(prompt: Any) -> str:
    return sha256_text(normalize_prompt(prompt))


def prompt_answer_hash(prompt: Any, answer: Any) -> str:
    return sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer))


def load_reference_csvs(paths: list[Path]) -> dict[str, set[str]]:
    ids: set[str] = set()
    prompts: set[str] = set()
    prompt_answers: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"reference CSV missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rid = str(row.get("id", "") or row.get("row_id", "")).strip()
                prompt = row.get("prompt")
                answer = row.get("answer")
                if rid:
                    ids.add(rid)
                if prompt:
                    prompts.add(prompt_hash(prompt))
                    if answer:
                        prompt_answers.add(prompt_answer_hash(prompt, answer))
    return {"ids": ids, "prompts": prompts, "prompt_answers": prompt_answers}


def family_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("family") or metadata.get("family") or "")


def subcategory_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("subcategory") or metadata.get("subcategory") or "")


def source_dataset_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("source_dataset") or metadata.get("source_dataset") or "")


def row_overlap(row: dict[str, Any], reference: dict[str, set[str]]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    reasons: list[str] = []
    ids = {str(row.get("id", "")).strip(), str(metadata.get("original_id", "")).strip()}
    if any(value and value in reference["ids"] for value in ids):
        reasons.append("id")
    if prompt_hash(row.get("prompt", "")) in reference["prompts"]:
        reasons.append("prompt")
    if prompt_answer_hash(row.get("prompt", ""), row.get("answer", "")) in reference["prompt_answers"]:
        reasons.append("prompt_answer")
    return reasons


def validate_final_answer(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return False
    assistant = str(messages[2].get("content", ""))
    answer = str(row.get("answer", ""))
    extracted = extract_final_answer_for_expected(assistant, answer)
    return verify_answer(answer, extracted)


def normalize_row(row: dict[str, Any], *, split: str, component: str, index: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    old_id = str(out.get("id", ""))
    metadata = dict(out.get("metadata", {}) if isinstance(out.get("metadata"), dict) else {})
    metadata.update(
        {
            "schema_version": "kg1_v536_v534_bit_v523_equation_pack_v1",
            "source": "v536_v534_bit_v523_equation_pack",
            "source_dataset": "v536_v534_bit_v523_equation_pack",
            "v536_component": component,
            "v536_split": split,
            "v536_original_id": old_id,
            "v536_source_only": True,
        }
    )
    for flag in ANTI_LEAK_FLAGS:
        metadata[flag] = False
    out["metadata"] = metadata
    out["source"] = "v536_v534_bit_v523_equation_pack"
    out["source_dataset"] = "v536_v534_bit_v523_equation_pack"
    out["id"] = f"v536_{split}_{component}_{index:05d}_{sha256_text(old_id)[:10]}"
    return out


def balanced_take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[subcategory_of(row)].append(row)
    if not buckets:
        return []
    base = limit // len(buckets)
    remainder = limit % len(buckets)
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for idx, key in enumerate(sorted(buckets)):
        quota = base + (1 if idx < remainder else 0)
        for row in sorted(buckets[key], key=lambda item: str(item.get("id", "")))[:quota]:
            selected.append(row)
            used_keys.add(str(row.get("id", "")))
    if len(selected) < limit:
        for row in sorted(rows, key=lambda item: str(item.get("id", ""))):
            row_id = str(row.get("id", ""))
            if row_id in used_keys:
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected[:limit]


def build_split(
    *,
    split: str,
    v523_rows: list[dict[str, Any]],
    v534_rows: list[dict[str, Any]],
    bit_limit: int,
    equation_limit: int,
) -> list[dict[str, Any]]:
    bit_rows = balanced_take([row for row in v534_rows if family_of(row) == "bit_manipulation"], bit_limit)
    equation_rows = balanced_take([row for row in v523_rows if family_of(row) == "equation_transform"], equation_limit)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(bit_rows, 1):
        out.append(normalize_row(row, split=split, component="v534_bit", index=index))
    for index, row in enumerate(equation_rows, 1):
        out.append(normalize_row(row, split=split, component="v523_equation", index=index))
    out.sort(key=lambda row: row["id"])
    return out


def summarize(rows: list[dict[str, Any]], reference: dict[str, set[str]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    prompt_answer_counts: Counter[str] = Counter()
    bad_final_answer = 0
    flag_counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        family_counts[family_of(row)] += 1
        subcategory_counts[subcategory_of(row)] += 1
        source_counts[source_dataset_of(row)] += 1
        component_counts[str(metadata.get("v536_component", ""))] += 1
        for reason in row_overlap(row, reference):
            overlap_counts[reason] += 1
        prompt_answer_counts[prompt_answer_hash(row.get("prompt", ""), row.get("answer", ""))] += 1
        if not validate_final_answer(row):
            bad_final_answer += 1
        for flag in ANTI_LEAK_FLAGS:
            if metadata.get(flag) not in (False, None):
                flag_counts[flag] += 1
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "reference_overlap_counts": dict(sorted(overlap_counts.items())),
        "duplicate_prompt_answer": sum(1 for count in prompt_answer_counts.values() if count > 1),
        "bad_final_answer": bad_final_answer,
        "training_flag_counts": dict(sorted(flag_counts.items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V536 V534 BIT + V523 EQUATION PACK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v523_manifest_json =", args.v523_manifest_json, flush=True)
    print("v534_manifest_json =", args.v534_manifest_json, flush=True)
    print("output_dir_root =", args.output_dir, flush=True)
    v523 = read_json(args.v523_manifest_json)
    v534 = read_json(args.v534_manifest_json)
    reference = load_reference_csvs([args.weak_reference_csv, args.full_reference_csv])
    output_dir = args.output_dir / utc_compact()
    output_dir.mkdir(parents=True, exist_ok=True)

    v523_train = read_jsonl(Path(v523["outputs"]["train_jsonl"]))
    v523_val = read_jsonl(Path(v523["outputs"]["val_jsonl"]))
    v534_train = read_jsonl(Path(v534["outputs"]["train_jsonl"]))
    v534_val = read_jsonl(Path(v534["outputs"]["val_jsonl"]))
    train_rows = build_split(
        split="train",
        v523_rows=v523_train,
        v534_rows=v534_train,
        bit_limit=args.train_bit_rows,
        equation_limit=args.train_equation_rows,
    )
    val_rows = build_split(
        split="validation",
        v523_rows=v523_val,
        v534_rows=v534_val,
        bit_limit=args.val_bit_rows,
        equation_limit=args.val_equation_rows,
    )

    train_summary = summarize(train_rows, reference)
    val_summary = summarize(val_rows, reference)
    blockers: list[str] = []
    for label, summary, expected_rows in (
        ("train", train_summary, args.train_bit_rows + args.train_equation_rows),
        ("validation", val_summary, args.val_bit_rows + args.val_equation_rows),
    ):
        if summary["rows"] != expected_rows:
            blockers.append(f"{label}:unexpected_row_count")
        if summary["reference_overlap_counts"]:
            blockers.append(f"{label}:reference_overlap")
        if summary["duplicate_prompt_answer"]:
            blockers.append(f"{label}:duplicate_prompt_answer")
        if summary["bad_final_answer"]:
            blockers.append(f"{label}:bad_final_answer")
        if summary["training_flag_counts"]:
            blockers.append(f"{label}:training_flags")

    train_path = output_dir / "v536_v534_bit_v523_equation_pack_train.jsonl"
    val_path = output_dir / "v536_v534_bit_v523_equation_pack_val.jsonl"
    manifest_path = output_dir / "v536_v534_bit_v523_equation_pack_manifest.json"
    comparison_path = output_dir / "V536_VS_V523.md"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    decision = {
        "gpu_allowed": False,
        "dataset_ready_for_cpu_gates": not blockers,
        "status": "dataset_ready_for_cpu_gates" if not blockers else "dataset_blocked",
        "reason": (
            "V536 is a controlled replacement of V523 bit rows with V534 shorter bit traces. "
            "It still requires V286, V513, objective audit, and FinOps gates before GPU."
        ),
        "next_action": "Run V286 boxed_suffix, V513, V524, then only a short example_mean smoke if all pass.",
    }
    manifest = {
        "version": "V536",
        "label": "v536_v534_bit_v523_equation_pack",
        "schema_version": "kg1_v536_v534_bit_v523_equation_pack_v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "blockers": blockers,
        "inputs": {
            "v523_manifest_json": str(args.v523_manifest_json),
            "v523_manifest_sha256": sha256_file(args.v523_manifest_json),
            "v534_manifest_json": str(args.v534_manifest_json),
            "v534_manifest_sha256": sha256_file(args.v534_manifest_json),
            "train_bit_rows": args.train_bit_rows,
            "train_equation_rows": args.train_equation_rows,
            "val_bit_rows": args.val_bit_rows,
            "val_equation_rows": args.val_equation_rows,
        },
        "forbidden_reference_csvs": [
            str(args.weak_reference_csv),
            str(args.full_reference_csv),
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
            "comparison_md": str(comparison_path),
        },
        "blocked_actions": ["train_gpu", "full_eval", "package", "kaggle_submit"],
    }
    write_json(manifest_path, manifest)
    write_comparison(comparison_path, manifest)
    print("v536_manifest_json =", manifest_path, flush=True)
    print("v536_decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("v536_train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("v536_validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("=== V536 V534 BIT + V523 EQUATION PACK END ===", flush=True)
    return manifest


def write_comparison(path: Path, manifest: dict[str, Any]) -> None:
    train = manifest["train_summary"]["family_counts"]
    val = manifest["validation_summary"]["family_counts"]
    lines = [
        "# V536 vs V523",
        "",
        "| Metric | V523 | V536 |",
        "|---|---:|---:|",
        "| train rows | 1026 | {rows} |".format(rows=manifest["train_summary"]["rows"]),
        "| val rows | 219 | {rows} |".format(rows=manifest["validation_summary"]["rows"]),
        "| train bit rows | 706 | {rows} |".format(rows=train.get("bit_manipulation", 0)),
        "| train equation rows | 320 | {rows} |".format(rows=train.get("equation_transform", 0)),
        "| val bit rows | 139 | {rows} |".format(rows=val.get("bit_manipulation", 0)),
        "| val equation rows | 80 | {rows} |".format(rows=val.get("equation_transform", 0)),
        "| bit source | V523/V304 old bit traces | V534 Konbu high + Huikang CHO/MAJ source-only |",
        "| equation source | V523 equation | V523 equation |",
        "| GPU allowed now | no | no |",
        "",
        "V536 controls the experiment: only the bit source changes, while row quotas match V523.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    row = {
        "id": "old",
        "prompt": "p",
        "answer": "01010101",
        "family": "bit_manipulation",
        "subcategory": "bit_a",
        "source_dataset": "old_source",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "Trace bit XOR.\nFinal answer: \\boxed{01010101}"},
        ],
        "metadata": {"source_dataset": "old_source"},
    }
    out = normalize_row(row, split="train", component="unit", index=1)
    if out["metadata"]["weak_gate_rows_used_for_training"] is not False:
        raise AssertionError("anti-leak flag missing")
    if not validate_final_answer(out):
        raise AssertionError("final answer should verify")
    selected = balanced_take([row, {**row, "id": "old2", "subcategory": "bit_b"}], 2)
    if len(selected) != 2:
        raise AssertionError("balanced_take failed")
    print("build_v536_v534_bit_v523_equation_pack_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v523-manifest-json", type=Path)
    parser.add_argument("--v534-manifest-json", type=Path)
    parser.add_argument("--weak-reference-csv", type=Path, default=DEFAULT_WEAK_REFERENCE)
    parser.add_argument("--full-reference-csv", type=Path, default=DEFAULT_FULL_REFERENCE)
    parser.add_argument("--train-bit-rows", type=int, default=706)
    parser.add_argument("--train-equation-rows", type=int, default=320)
    parser.add_argument("--val-bit-rows", type=int, default=139)
    parser.add_argument("--val-equation-rows", type=int, default=80)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.v523_manifest_json is None:
        args.v523_manifest_json = latest_manifest(DEFAULT_V523_ROOT, "v523_targeted_source_trace_pack_manifest.json")
    if args.v534_manifest_json is None:
        args.v534_manifest_json = latest_manifest(DEFAULT_V534_ROOT, "v534_bit_source_only_trace_pack_manifest.json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

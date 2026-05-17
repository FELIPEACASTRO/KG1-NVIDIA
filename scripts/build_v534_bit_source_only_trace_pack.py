#!/usr/bin/env python3
"""Build V534 bit source-only trace pack.

V534 is intentionally CPU-first. It converts two externally audited bit-trace
sources into compact KG1 chat rows, filters every row against the active weak
and full reference CSVs by id, prompt hash, and prompt+answer hash, and leaves
GPU disabled until the regular V286/V513 gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import box_answer, extract_final_answer_for_expected, verify_answer  # noqa: E402


DOWNLOADS = Path.home() / "Downloads"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v534_bit_source_only_trace_pack"
DEFAULT_KONBU_ZIPS = [
    DOWNLOADS / "archive.zip",
    DOWNLOADS / "archive (5).zip",
]
DEFAULT_HUIKANG_ZIP = DOWNLOADS / "archive (9).zip"
DEFAULT_WEAK_REFERENCE = ROOT / "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv"
DEFAULT_FULL_REFERENCE = ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

BIT_TERMS = "bit ROT SHL SHR XOR XNOR AND NAND OR NOR NOT IMPL INHIB stride bitsum CHO MAJ"
ALLOWED_BIT_OPERATORS = {
    "AND",
    "NAND",
    "OR",
    "NOR",
    "XOR",
    "XNOR",
    "NOT",
    "ROT",
    "SHL",
    "SHR",
    "CHO",
    "MAJ",
    "IMPL",
    "INHIB",
}
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def prompt_hash(prompt: Any) -> str:
    return sha256_text(normalize_prompt(prompt))


def prompt_answer_hash(prompt: Any, answer: Any) -> str:
    return sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer))


def deterministic_split(key: str) -> str:
    return "validation" if int(sha256_text(key)[:8], 16) % 100 < 15 else "train"


def find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    hits = [name for name in zf.namelist() if name.endswith(suffix)]
    if not hits:
        raise RuntimeError(f"zip member not found: {suffix}")
    hits.sort(key=lambda value: (len(value), value))
    return hits[0]


def load_reference_csvs(paths: list[Path]) -> dict[str, set[str]]:
    ids: set[str] = set()
    prompts: set[str] = set()
    prompt_answers: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"reference CSV missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            for row in reader:
                rid = str(row.get("id", "") or row.get("row_id", "")).strip()
                if rid:
                    ids.add(rid)
                prompt = row.get("prompt")
                answer = row.get("answer")
                if prompt is not None and str(prompt).strip():
                    prompts.add(prompt_hash(prompt))
                    if answer is not None and str(answer).strip():
                        prompt_answers.add(prompt_answer_hash(prompt, answer))
                elif "prompt_sha256" in fields and row.get("prompt_sha256"):
                    prompts.add(str(row["prompt_sha256"]).strip())
    return {"ids": ids, "prompts": prompts, "prompt_answers": prompt_answers}


def compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def compact_konbu_trace(raw: str, *, method: str, confidence: str, n_ambig_bits: str) -> str:
    text = compact_spaces(raw)
    windows: list[str] = []
    lowered = text.lower()
    for marker in (
        "rule",
        "best",
        "rotation",
        "shift",
        "xor",
        "and",
        "or",
        "not",
        "answer",
    ):
        pos = lowered.find(marker)
        if pos >= 0:
            windows.append(text[max(0, pos - 120) : pos + 360])
    if not windows:
        windows.append(text[:700])
    windows.append(text[-360:])
    merged = compact_spaces(" ".join(windows))
    if len(merged) > 1400:
        merged = merged[:1400].rsplit(" ", 1)[0]
    return (
        "Trace summary: infer the 8-bit transformation from examples, then test "
        "candidate bit relations. "
        f"Source solver method={method or 'unknown'}, confidence={confidence or 'unknown'}, "
        f"ambiguous_bits={n_ambig_bits or 'unknown'}. "
        f"Rule vocabulary: {BIT_TERMS}. Evidence: {merged}"
    )


def rule_group(rule: str) -> str:
    value = str(rule or "").strip().upper()
    if value.startswith("CHO("):
        return "CHO"
    if value.startswith("MAJ("):
        return "MAJ"
    if value.startswith("XOR(") or " XOR" in value:
        return "XOR"
    return "OTHER"


def compact_huikang_trace(reasoning_text: str, rule: str) -> str:
    text = str(reasoning_text or "")
    lines = [compact_spaces(line) for line in text.splitlines() if line.strip()]
    evidence: list[str] = []
    for line in lines:
        lowered = line.lower()
        if (
            lowered.startswith("example ")
            or "applying" in lowered
            or "output" in lowered
            or "rule" in lowered
            or "input" in lowered
        ):
            evidence.append(line)
        if len(evidence) >= 6:
            break
    evidence_text = compact_spaces(" ".join(evidence))[:900]
    return (
        "Trace summary: infer a per-bit 3-input transformation and verify it on all examples. "
        f"Rule: {rule}. Operator family: {rule_group(rule)}. "
        f"Rule vocabulary: {BIT_TERMS}. "
        f"Evidence: {evidence_text or 'the examples are checked against the same bit rule.'}"
    )


def bit_operator_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in re.findall(r"\b[A-Z][A-Z0-9_-]*\b", str(text or "").upper()):
        if token in ALLOWED_BIT_OPERATORS:
            counts[token] += 1
    return counts


def make_chat_row(
    *,
    source_key: str,
    source_dataset: str,
    source_zip: Path,
    source_member: str,
    original_id: str,
    prompt: str,
    answer: str,
    assistant_trace: str,
    subcategory: str,
    extra_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if not re.fullmatch(r"[01]{8}", str(answer or "").strip()):
        return None
    assistant = assistant_trace.strip() + "\nFinal answer: " + box_answer(answer)
    extracted = extract_final_answer_for_expected(assistant, answer)
    if not verify_answer(answer, extracted):
        return None
    row_id = f"v534_{source_key}_{sha256_text(original_id + prompt + answer)[:16]}"
    metadata = {
        "schema_version": "kg1_v534_bit_source_only_trace_pack_v1",
        "source": "v534_bit_source_only_trace_pack",
        "source_dataset": source_dataset,
        "source_zip": str(source_zip),
        "source_member": source_member,
        "original_id": original_id,
        "source_only": True,
        "family": "bit_manipulation",
        "subcategory": subcategory,
        "prompt_sha256": prompt_hash(prompt),
        "prompt_answer_sha256": prompt_answer_hash(prompt, answer),
        "answer_sha256": sha256_text(normalize_answer(answer)),
    }
    for flag in ANTI_LEAK_FLAGS:
        metadata[flag] = False
    metadata.update(extra_metadata)
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "bit_manipulation",
        "subcategory": subcategory,
        "source": "v534_bit_source_only_trace_pack",
        "source_dataset": source_dataset,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def iter_konbu_rows(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        member = find_zip_member(zf, "bit_manipulation_cot_success.csv")
        with zf.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
            for source_row in reader:
                prompt = str(source_row.get("prompt", ""))
                answer = normalize_answer(source_row.get("answer", ""))
                confidence = str(source_row.get("confidence", "")).strip().lower()
                if confidence != "high":
                    continue
                row = make_chat_row(
                    source_key="konbu_high",
                    source_dataset="konbu_bit_manipulation_cot_success_high",
                    source_zip=zip_path,
                    source_member=member,
                    original_id=str(source_row.get("id", "")),
                    prompt=prompt,
                    answer=answer,
                    assistant_trace=compact_konbu_trace(
                        str(source_row.get("generated_cot", "")),
                        method=str(source_row.get("method", "")),
                        confidence=confidence,
                        n_ambig_bits=str(source_row.get("n_ambig_bits", "")),
                    ),
                    subcategory="bit_konbu_high_confidence_trace",
                    extra_metadata={
                        "konbu_method": str(source_row.get("method", "")),
                        "konbu_confidence": confidence,
                        "konbu_n_ambig_bits": str(source_row.get("n_ambig_bits", "")),
                    },
                )
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda row: (row["metadata"].get("original_id", ""), row["id"]))
    return rows


def iter_huikang_synthetic_rows(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        member = find_zip_member(zf, "bit_manip_3input_synthesized_traces.jsonl")
        with zf.open(member) as raw:
            for line_no, raw_line in enumerate(raw, 1):
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                source_row = json.loads(text)
                prompt = str(source_row.get("prompt", ""))
                answer = normalize_answer(source_row.get("answer", ""))
                rule = str(source_row.get("rule", ""))
                group = rule_group(rule)
                row = make_chat_row(
                    source_key=f"huikang_synth_{group.lower()}",
                    source_dataset="huikang_synthetic_3input_bit_traces",
                    source_zip=zip_path,
                    source_member=member,
                    original_id=str(source_row.get("problem_id", f"line_{line_no}")),
                    prompt=prompt,
                    answer=answer,
                    assistant_trace=compact_huikang_trace(str(source_row.get("reasoning_text", "")), rule),
                    subcategory=f"bit_huikang_synthetic_{group.lower()}",
                    extra_metadata={
                        "huikang_rule": rule,
                        "huikang_rule_group": group,
                        "huikang_category": str(source_row.get("category", "")),
                    },
                )
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda row: (row["metadata"].get("huikang_rule_group", ""), row["metadata"].get("original_id", "")))
    return rows


def row_overlap(row: dict[str, Any], reference: dict[str, set[str]]) -> list[str]:
    reasons: list[str] = []
    original_id = str(row.get("metadata", {}).get("original_id", "")).strip()
    if str(row.get("id", "")).strip() in reference["ids"] or original_id in reference["ids"]:
        reasons.append("id")
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    if prompt_hash(prompt) in reference["prompts"]:
        reasons.append("prompt")
    if prompt_answer_hash(prompt, answer) in reference["prompt_answers"]:
        reasons.append("prompt_answer")
    return reasons


def select_source_rows(
    rows: list[dict[str, Any]],
    *,
    reference: dict[str, set[str]],
    limit: int,
    used_prompt_answer: set[str],
    balance_by_rule_group: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    if not balance_by_rule_group:
        iterable = rows
    else:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group = str(row.get("metadata", {}).get("huikang_rule_group", "OTHER"))
            buckets.setdefault(group, []).append(row)
        per_group = max(1, limit // max(1, len(buckets)))
        balanced: list[dict[str, Any]] = []
        for group in sorted(buckets):
            balanced.extend(buckets[group][:per_group])
        if len(balanced) < limit:
            seen_ids = {id(row) for row in balanced}
            for row in rows:
                if id(row) not in seen_ids:
                    balanced.append(row)
                if len(balanced) >= limit:
                    break
        iterable = balanced
    for row in iterable:
        if len(selected) >= limit:
            break
        overlap = row_overlap(row, reference)
        if overlap:
            for reason in overlap:
                skipped[f"reference_overlap_{reason}"] += 1
            continue
        key = prompt_answer_hash(row.get("prompt", ""), row.get("answer", ""))
        if key in used_prompt_answer:
            skipped["duplicate_prompt_answer"] += 1
            continue
        used_prompt_answer.add(key)
        selected.append(row)
    return selected, skipped


def summarize(rows: list[dict[str, Any]], reference: dict[str, set[str]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    prompt_answer_counts: Counter[str] = Counter()
    bad_final_answer = 0
    flag_counts: Counter[str] = Counter()
    assistant_word_counts: list[int] = []
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        family_counts[str(row.get("family", ""))] += 1
        subcategory_counts[str(row.get("subcategory", ""))] += 1
        source_counts[str(row.get("source_dataset", metadata.get("source_dataset", "")))] += 1
        rule_counts[str(metadata.get("huikang_rule_group") or metadata.get("konbu_method") or "unknown")] += 1
        for reason in row_overlap(row, reference):
            overlap_counts[reason] += 1
        key = prompt_answer_hash(row.get("prompt", ""), row.get("answer", ""))
        prompt_answer_counts[key] += 1
        messages = row.get("messages", [])
        assistant = str(messages[2].get("content", "")) if isinstance(messages, list) and len(messages) == 3 else ""
        assistant_word_counts.append(len(assistant.split()))
        extracted = extract_final_answer_for_expected(assistant, row.get("answer", ""))
        if not verify_answer(row.get("answer", ""), extracted):
            bad_final_answer += 1
        for flag in ANTI_LEAK_FLAGS:
            if metadata.get(flag) not in (False, None):
                flag_counts[flag] += 1
    assistant_word_counts.sort()
    p50 = assistant_word_counts[len(assistant_word_counts) // 2] if assistant_word_counts else 0
    p95 = assistant_word_counts[int((len(assistant_word_counts) - 1) * 0.95)] if assistant_word_counts else 0
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "rule_or_method_counts": dict(sorted(rule_counts.items())),
        "reference_overlap_counts": dict(sorted(overlap_counts.items())),
        "duplicate_prompt_answer": sum(1 for count in prompt_answer_counts.values() if count > 1),
        "bad_final_answer": bad_final_answer,
        "training_flag_counts": dict(sorted(flag_counts.items())),
        "assistant_words_p50": p50,
        "assistant_words_p95": p95,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V534 BIT SOURCE-ONLY TRACE PACK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir_root =", args.output_dir, flush=True)
    print("konbu_zips =", [str(path) for path in args.konbu_zip], flush=True)
    print("huikang_zip =", args.huikang_zip, flush=True)
    print("weak_reference_csv =", args.weak_reference_csv, flush=True)
    print("full_reference_csv =", args.full_reference_csv, flush=True)
    output_dir = args.output_dir / utc_compact()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs = [str(path) for path in [*args.konbu_zip, args.huikang_zip] if not path.is_file()]
    if missing_inputs:
        raise RuntimeError("missing input files: " + json.dumps(missing_inputs))

    reference = load_reference_csvs([args.weak_reference_csv, args.full_reference_csv])
    print("reference_counts =", {key: len(value) for key, value in reference.items()}, flush=True)

    konbu_candidates: list[dict[str, Any]] = []
    for zip_path in args.konbu_zip:
        print("reading_konbu_zip =", zip_path, flush=True)
        rows = iter_konbu_rows(zip_path)
        print("konbu_high_candidate_count =", len(rows), "zip =", zip_path, flush=True)
        konbu_candidates.extend(rows)
    huikang_candidates = iter_huikang_synthetic_rows(args.huikang_zip)
    print("huikang_synthetic_candidate_count =", len(huikang_candidates), flush=True)

    used_prompt_answer: set[str] = set()
    selected_konbu, skipped_konbu = select_source_rows(
        konbu_candidates,
        reference=reference,
        limit=args.max_konbu_high,
        used_prompt_answer=used_prompt_answer,
    )
    selected_huikang, skipped_huikang = select_source_rows(
        huikang_candidates,
        reference=reference,
        limit=args.max_huikang_synth,
        used_prompt_answer=used_prompt_answer,
        balance_by_rule_group=True,
    )
    all_rows = selected_konbu + selected_huikang

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in all_rows:
        split = deterministic_split(prompt_answer_hash(row["prompt"], row["answer"]))
        row["metadata"]["v534_split"] = split
        if split == "validation":
            val_rows.append(row)
        else:
            train_rows.append(row)
    train_rows.sort(key=lambda row: row["id"])
    val_rows.sort(key=lambda row: row["id"])

    train_summary = summarize(train_rows, reference)
    val_summary = summarize(val_rows, reference)
    all_summary = summarize(train_rows + val_rows, reference)
    blockers: list[str] = []
    for label, summary, min_rows in (
        ("train", train_summary, args.min_train_rows),
        ("validation", val_summary, args.min_val_rows),
    ):
        if summary["rows"] < min_rows:
            blockers.append(f"{label}:rows_lt_{min_rows}")
        if summary["reference_overlap_counts"]:
            blockers.append(f"{label}:reference_overlap")
        if summary["duplicate_prompt_answer"]:
            blockers.append(f"{label}:duplicate_prompt_answer")
        if summary["bad_final_answer"]:
            blockers.append(f"{label}:bad_final_answer")
        if summary["training_flag_counts"]:
            blockers.append(f"{label}:training_flags")
        if summary["family_counts"].get("bit_manipulation", 0) != summary["rows"]:
            blockers.append(f"{label}:non_bit_family")
    if not selected_konbu:
        blockers.append("konbu:no_selected_rows")
    if not selected_huikang:
        blockers.append("huikang:no_selected_rows")

    train_path = output_dir / "v534_bit_source_only_trace_pack_train.jsonl"
    val_path = output_dir / "v534_bit_source_only_trace_pack_val.jsonl"
    manifest_path = output_dir / "v534_bit_source_only_trace_pack_manifest.json"
    comparison_path = output_dir / "V534_VS_V523.md"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    input_sha256 = {str(path): sha256_file(path) for path in [*args.konbu_zip, args.huikang_zip]}
    decision = {
        "gpu_allowed": False,
        "dataset_ready_for_cpu_gates": not blockers,
        "status": "dataset_ready_for_cpu_gates" if not blockers else "dataset_blocked",
        "reason": (
            "V534 is bit-only source material. It must pass V286 tokenization, "
            "V513 learnability, objective contribution, and FinOps gates before any GPU."
        ),
        "next_action": "Run V286 boxed_suffix tokenization gate and V513 learnability gate. Do not train or submit directly from this manifest.",
    }
    manifest = {
        "version": "V534",
        "label": "v534_bit_source_only_trace_pack",
        "schema_version": "kg1_v534_bit_source_only_trace_pack_v1",
        "generated_at_utc": utc_now(),
        "decision": decision,
        "blockers": blockers,
        "inputs": {
            "konbu_zips": [str(path) for path in args.konbu_zip],
            "huikang_zip": str(args.huikang_zip),
            "input_sha256": input_sha256,
            "max_konbu_high": args.max_konbu_high,
            "max_huikang_synth": args.max_huikang_synth,
        },
        "forbidden_reference_csvs": [
            str(args.weak_reference_csv),
            str(args.full_reference_csv),
        ],
        "selection": {
            "konbu_selected": len(selected_konbu),
            "huikang_selected": len(selected_huikang),
            "skipped_konbu": dict(sorted(skipped_konbu.items())),
            "skipped_huikang": dict(sorted(skipped_huikang.items())),
        },
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "all_summary": all_summary,
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

    print("v534_manifest_json =", manifest_path, flush=True)
    print("v534_decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("v534_selection =", json.dumps(manifest["selection"], sort_keys=True), flush=True)
    print("v534_train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("v534_validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("=== V534 BIT SOURCE-ONLY TRACE PACK END ===", flush=True)
    return manifest


def write_comparison(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V534 vs V523",
        "",
        "| Metric | V523 | V534 |",
        "|---|---:|---:|",
        f"| train rows | 1026 | {manifest['train_summary']['rows']} |",
        f"| val rows | 219 | {manifest['validation_summary']['rows']} |",
        f"| train bit rows | 706 | {manifest['train_summary']['family_counts'].get('bit_manipulation', 0)} |",
        "| train equation rows | 320 | 0 |",
        "| source focus | mixed bit/equation | bit source-only CHO/MAJ/Konbu high-confidence |",
        "| weak/full overlap | 0 required | 0 required |",
        "| GPU allowed now | no | no |",
        "",
        "V534 is not a submit candidate. It is the source-only bit trace pack requested by the V535 roadmap,",
        "built to test whether richer bit traces can be learned without reusing weak/full rows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    prompt = "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\nNow solve: 10101010"
    row = make_chat_row(
        source_key="test",
        source_dataset="unit_test",
        source_zip=Path("unit.zip"),
        source_member="unit.jsonl",
        original_id="unit1",
        prompt=prompt,
        answer="01010101",
        assistant_trace="Trace summary: bit ROT SHL SHR XOR AND OR NOT stride bitsum CHO MAJ.",
        subcategory="bit_unit",
        extra_metadata={"unit": True},
    )
    if row is None:
        raise AssertionError("expected a valid row")
    assistant = row["messages"][2]["content"]
    extracted = extract_final_answer_for_expected(assistant, "01010101")
    if not verify_answer("01010101", extracted):
        raise AssertionError("boxed final answer did not verify")
    if row["metadata"]["weak_gate_rows_used_for_training"] is not False:
        raise AssertionError("anti-leak flag missing")
    ref = {"ids": {"blocked"}, "prompts": {prompt_hash(prompt)}, "prompt_answers": set()}
    if "prompt" not in row_overlap(row, ref):
        raise AssertionError("reference prompt overlap not detected")
    print("build_v534_bit_source_only_trace_pack_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--konbu-zip", action="append", type=Path, default=[])
    parser.add_argument("--huikang-zip", type=Path, default=DEFAULT_HUIKANG_ZIP)
    parser.add_argument("--weak-reference-csv", type=Path, default=DEFAULT_WEAK_REFERENCE)
    parser.add_argument("--full-reference-csv", type=Path, default=DEFAULT_FULL_REFERENCE)
    parser.add_argument("--max-konbu-high", type=int, default=700)
    parser.add_argument("--max-huikang-synth", type=int, default=1200)
    parser.add_argument("--min-train-rows", type=int, default=900)
    parser.add_argument("--min-val-rows", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.konbu_zip:
        args.konbu_zip = [path for path in DEFAULT_KONBU_ZIPS if path.is_file()]
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

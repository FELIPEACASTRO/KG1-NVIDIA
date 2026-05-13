#!/usr/bin/env python3
"""Build V312 synthetic verifier-distillation data.

V311 intentionally produced only a seed pack from V306 gate gains. This builder
uses the rule classes from that seed pack to generate new out-of-gate synthetic
rows, plus preference pairs. It must remain CPU-only and must not train on weak
or full gate rows directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for item in (SCRIPT_DIR, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import build_v282_rank19_micro_patch_dataset as v282  # noqa: E402
import build_v303_bit_fullbyte_distill_dataset as v303  # noqa: E402
import build_v304_solver_trace_distill_dataset as v304  # noqa: E402


DEFAULT_V311_MANIFEST = (
    REPO_ROOT
    / "artifacts/v311_verifier_distillation_preference_pack/20260512T1535Z/v311_v306_seed_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v312_verifier_synthetic_distill_dataset"
DEFAULT_REFERENCE_PATHS = [
    REPO_ROOT / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv",
]

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify the candidate briefly, "
    "then end with exactly one final answer in \\boxed{}."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"bad JSONL row {path}:{line_no}: {exc}") from exc
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def box(answer: str) -> str:
    text = str(answer).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "\\boxed{" + text + "}"


def final_answer_line(answer: str) -> str:
    return "Final answer: " + box(answer)


def normalize_trace_final(trace: str, answer: str) -> str:
    lines = [line.rstrip() for line in str(trace).strip().splitlines() if line.rstrip()]
    if lines and re.fullmatch(r"Final answer:\s*.+", lines[-1]):
        lines[-1] = final_answer_line(answer)
    else:
        lines.append(final_answer_line(answer))
    return "\n".join(lines)


def make_messages(prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


def read_reference_fingerprints(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, str]]]:
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    summaries: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_rows(path)
        for row in rows:
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompt_hashes.add(sha256_text(prompt))
        summaries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": str(len(rows)),
            }
        )
    return ids, prompt_hashes, summaries


def validate_v311_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v311_verifier_distillation_preference_pack_v1":
        raise RuntimeError("unexpected V311 manifest schema: " + str(payload.get("schema_version")))
    if int(payload.get("seed_gain_rows", -1)) != 15:
        raise RuntimeError("V311 seed_gain_rows must be 15")
    if payload.get("training_authorization") != "blocked_seed_only_until_synthetic_out_of_gate_variants":
        raise RuntimeError("V311 manifest is not in the expected blocked seed-only state")
    expected_rules = {
        "fullbyte_safe_ternary",
        "fullbyte_binary",
        "minus_signed_opposite_sign_guarded",
        "colon_absdiff_unreverse_same_len",
        "add_direct_over_model_add_variant",
    }
    observed_rules = set((payload.get("rule_counts") or {}).keys())
    missing = sorted(expected_rules - observed_rules)
    if missing:
        raise RuntimeError("V311 manifest missing expected rule classes: " + json.dumps(missing))
    return payload


def build_bit_row(rng: random.Random, *, split: str, pattern_index: int, row_index: int) -> dict[str, Any]:
    op, transforms = v303.EXACT_GAIN_PATTERNS[pattern_index]
    raw = v303.build_single_synthetic_row(
        rng,
        op=op,
        transforms=transforms,
        split=split,
        index=pattern_index * 100000 + row_index,
        source="v312_verifier_synthetic_bit_exact",
        subcategory="bit_fullbyte_v311_rule_variant",
        example_count=7 + (row_index % 3),
    )
    rule_name = v303.expr_name(op, transforms)
    prompt = str(raw["prompt"])
    answer = str(raw["answer"])
    trace = normalize_trace_final(v304.bit_trace(prompt, answer, rule_name), answer)
    row_id = f"v312_{split}_bit_{pattern_index:02d}_{row_index:05d}"
    metadata = {
        "schema_version": "kg1_v312_verifier_synthetic_distill_v1",
        "source": "v312_synthetic_from_v311_rule_class",
        "split": split,
        "family": "bit_manipulation",
        "subcategory": "bit_fullbyte_v311_rule_variant",
        "rule_name": rule_name,
        "rule_class": "fullbyte_binary" if len(transforms) == 2 else "fullbyte_safe_ternary",
        "teacher": "v303_generator_plus_v304_bit_trace",
        "gate_rows_used_for_training": False,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "loss_regime": "check_plus_final_candidate",
    }
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "bit_manipulation",
        "subcategory": "bit_fullbyte_v311_rule_variant",
        "source": "v312_verifier_synthetic",
        "messages": make_messages(prompt, trace),
        "metadata": metadata,
    }


def build_equation_row(rng: random.Random, *, split: str, rule_index: int, row_index: int) -> dict[str, Any]:
    builders = [
        v282.build_minus_signed,
        v282.build_minus_direct_negative_restore_sign,
        v282.build_colon_absdiff,
        v282.build_colon_absdiff_restore_trailing_zero,
        v282.build_add_direct,
    ]
    raw = builders[rule_index](rng, row_index, split)
    prompt = str(raw["prompt"])
    answer = str(raw["answer"])
    rule_name = str((raw.get("metadata") or {}).get("rule_name", ""))
    trace = normalize_trace_final(v304.equation_trace(prompt, answer, rule_name), answer)
    row_id = f"v312_{split}_equation_{rule_index:02d}_{row_index:05d}"
    metadata = {
        "schema_version": "kg1_v312_verifier_synthetic_distill_v1",
        "source": "v312_synthetic_from_v311_rule_class",
        "split": split,
        "family": "equation_transform",
        "subcategory": str(raw.get("subcategory", "")),
        "rule_name": rule_name,
        "rule_class": rule_name,
        "teacher": "v282_generator_plus_v304_equation_trace",
        "gate_rows_used_for_training": False,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "loss_regime": "check_plus_final_candidate",
    }
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "equation_transform",
        "subcategory": str(raw.get("subcategory", "")),
        "source": "v312_verifier_synthetic",
        "messages": make_messages(prompt, trace),
        "metadata": metadata,
    }


def flip_one_bit(answer: str) -> str:
    bits = list(str(answer))
    idx = max(0, len(bits) // 2)
    bits[idx] = "1" if bits[idx] == "0" else "0"
    return "".join(bits)


def equation_near_miss(answer: str) -> str:
    text = str(answer)
    candidates: list[str] = []
    if text.startswith("-"):
        candidates.append(text[1:] or "0")
    if text.isdigit() and len(text) > 1:
        candidates.append(text[::-1])
    if text.isdigit() and len(text) > 1:
        candidates.append(text[:-1] + str((int(text[-1]) + 1) % 10))
    if text.isdigit():
        candidates.append(str(int(text) + 1))
    candidates.append(text + "0")
    for candidate in candidates:
        if candidate != text:
            return candidate
    return text + "x"


def format_negative(answer: str, mode: str) -> str:
    if mode == "no_box":
        return "Final answer: " + str(answer)
    if mode == "multiple_boxes":
        return "First guess: " + box(answer) + "\nFinal answer: " + box(str(answer)[::-1])
    if mode == "trailing_text":
        return final_answer_line(answer) + "\nThis is the answer."
    raise ValueError("unknown format negative: " + mode)


def build_preferences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        answer = str(row["answer"])
        family = str(row["family"])
        chosen = str(row["messages"][-1]["content"])
        if family == "bit_manipulation":
            hard_negative = final_answer_line(flip_one_bit(answer))
            hard_type = "hard_negative_bit_flip_one"
        else:
            hard_negative = final_answer_line(equation_near_miss(answer))
            hard_type = "hard_negative_equation_near_miss"
        base_meta = {
            **dict(row.get("metadata") or {}),
            "preference_source_row_id": row["id"],
            "schema_version": "kg1_v312_verifier_synthetic_preference_v1",
        }
        candidates = [
            (hard_type, hard_negative),
            ("format_negative_no_box", format_negative(answer, "no_box")),
            ("format_negative_multiple_boxes", format_negative(answer, "multiple_boxes")),
            ("format_negative_trailing_text", format_negative(answer, "trailing_text")),
        ]
        for negative_type, rejected in candidates:
            out.append(
                {
                    "id": f"{row['id']}_{negative_type}",
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {**base_meta, "negative_type": negative_type},
                }
            )
    return out


def validate_boxed_answer(text: str, answer: str) -> bool:
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return len(boxes) == 1 and boxes[0] == str(answer) and text.rstrip().endswith("\\boxed{" + str(answer) + "}")


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    ref_ids: set[str],
    ref_prompt_hashes: set[str],
    label: str,
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    bad: list[str] = []
    family_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        family = str(row.get("family", "")).strip()
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        prompt_hash = sha256_text(prompt)
        if not row_id:
            bad.append("missing_id")
        if row_id in seen_ids:
            bad.append(f"{row_id}:duplicate_id")
        if row_id in ref_ids:
            bad.append(f"{row_id}:reference_id_overlap")
        if not prompt:
            bad.append(f"{row_id}:missing_prompt")
        if prompt_hash in seen_prompts:
            bad.append(f"{row_id}:duplicate_prompt")
        if prompt_hash in ref_prompt_hashes:
            bad.append(f"{row_id}:reference_prompt_overlap")
        if not answer:
            bad.append(f"{row_id}:missing_answer")
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{row_id}:bad_messages")
        else:
            assistant = str(messages[-1].get("content", ""))
            if not validate_boxed_answer(assistant, answer):
                bad.append(f"{row_id}:assistant_not_exactly_one_final_box")
        for flag in ("gate_rows_used_for_training", "weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if bool(metadata.get(flag)):
                bad.append(f"{row_id}:{flag}_true")
        seen_ids.add(row_id)
        seen_prompts.add(prompt_hash)
        family_counts[family] += 1
        subcategory_counts[str(row.get("subcategory", ""))] += 1
    if bad:
        raise RuntimeError(f"{label} validation failed: " + json.dumps(bad[:20], ensure_ascii=False))
    return {
        "rows": len(rows),
        "unique_ids": len(seen_ids),
        "unique_prompt_hashes": len(seen_prompts),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "reference_id_overlap": 0,
        "reference_prompt_overlap": 0,
    }


def validate_preferences(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    bad: list[str] = []
    negative_counts: Counter[str] = Counter()
    for row in rows:
        row_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        negative_type = str((row.get("metadata") or {}).get("negative_type", ""))
        if not row_id or not prompt or not chosen or not rejected:
            bad.append(f"{row_id}:missing_field")
        if chosen == rejected:
            bad.append(f"{row_id}:chosen_equals_rejected")
        chosen_boxes = re.findall(r"\\boxed\{([^{}]*)\}", chosen)
        rejected_boxes = re.findall(r"\\boxed\{([^{}]*)\}", rejected)
        if len(chosen_boxes) != 1:
            bad.append(f"{row_id}:chosen_box_count")
        if negative_type.startswith("hard_negative_") and len(rejected_boxes) == 1 and chosen_boxes == rejected_boxes:
            bad.append(f"{row_id}:hard_negative_same_box")
        negative_counts[negative_type] += 1
    if bad:
        raise RuntimeError(f"{label} preference validation failed: " + json.dumps(bad[:20], ensure_ascii=False))
    return {"rows": len(rows), "negative_type_counts": dict(sorted(negative_counts.items()))}


def build_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    if split == "train":
        bit_per_pattern = args.train_bit_per_pattern
        equation_per_rule = args.train_equation_per_rule
        seed = args.seed
    else:
        bit_per_pattern = args.val_bit_per_pattern
        equation_per_rule = args.val_equation_per_rule
        seed = args.seed + 10000

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for pattern_index in range(len(v303.EXACT_GAIN_PATTERNS)):
        for row_index in range(bit_per_pattern):
            rows.append(build_bit_row(rng, split=split, pattern_index=pattern_index, row_index=row_index))
    for rule_index in range(3):
        for row_index in range(equation_per_rule):
            rows.append(build_equation_row(rng, split=split, rule_index=rule_index, row_index=row_index))
    rng.shuffle(rows)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V312 VERIFIER SYNTHETIC DISTILL DATASET START ===", flush=True)
    print("v311_manifest_json =", args.v311_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("seed =", args.seed, flush=True)
    print("train_bit_per_pattern =", args.train_bit_per_pattern, flush=True)
    print("train_equation_per_rule =", args.train_equation_per_rule, flush=True)
    print("val_bit_per_pattern =", args.val_bit_per_pattern, flush=True)
    print("val_equation_per_rule =", args.val_equation_per_rule, flush=True)

    v311_manifest = validate_v311_manifest(args.v311_manifest_json)
    ref_ids, ref_prompt_hashes, reference_summary = read_reference_fingerprints(args.reference_path)
    print("reference_id_count =", len(ref_ids), flush=True)
    print("reference_prompt_hash_count =", len(ref_prompt_hashes), flush=True)

    train_rows = build_split(args, "train")
    val_rows = build_split(args, "validation")
    train_preferences = build_preferences(train_rows)
    val_preferences = build_preferences(val_rows)

    train_summary = validate_rows(train_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="train")
    val_summary = validate_rows(val_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="validation")
    train_pref_summary = validate_preferences(train_preferences, "train")
    val_pref_summary = validate_preferences(val_preferences, "validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_sft_train.jsonl"
    val_path = args.output_dir / f"{args.label}_sft_val.jsonl"
    train_pref_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    val_pref_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(train_pref_path, train_preferences)
    write_jsonl(val_pref_path, val_preferences)

    manifest = {
        "schema_version": "kg1_v312_verifier_synthetic_distill_dataset_v1",
        "generated_at_utc": utc_now(),
        "v311_manifest_json": str(args.v311_manifest_json),
        "v311_manifest_sha256": sha256_file(args.v311_manifest_json),
        "v311_seed_gain_rows": int(v311_manifest.get("seed_gain_rows", -1)),
        "v311_rule_counts": v311_manifest.get("rule_counts", {}),
        "reference_summary": reference_summary,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "train_preference_summary": train_pref_summary,
        "validation_preference_summary": val_pref_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
            "sft_train_jsonl": str(train_path),
            "sft_val_jsonl": str(val_path),
            "preferences_train_jsonl": str(train_pref_path),
            "preferences_val_jsonl": str(val_pref_path),
            "manifest_json": str(manifest_path),
        },
        "hashes": {
            "sft_train_sha256": sha256_file(train_path),
            "sft_val_sha256": sha256_file(val_path),
            "preferences_train_sha256": sha256_file(train_pref_path),
            "preferences_val_sha256": sha256_file(val_pref_path),
        },
        "training_authorization": "blocked_until_real_tokenization_and_no_regression_gate",
        "required_next_gate": [
            "real_tokenization_gate_with_offset_masks",
            "weak_eval_before_full_eval",
            "absorption_ratio_measurement_without_postprocessor",
            "adapter_only_package_gate",
        ],
    }
    write_json(manifest_path, manifest)

    print("sft_train_jsonl =", train_path, flush=True)
    print("sft_val_jsonl =", val_path, flush=True)
    print("preferences_train_jsonl =", train_pref_path, flush=True)
    print("preferences_val_jsonl =", val_pref_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("train_preference_summary =", json.dumps(train_pref_summary, sort_keys=True), flush=True)
    print("validation_preference_summary =", json.dumps(val_pref_summary, sort_keys=True), flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V312 VERIFIER SYNTHETIC DISTILL DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    tmp = REPO_ROOT / "artifacts/_tmp_v312_self_test"
    if tmp.exists():
        for path in tmp.glob("*"):
            path.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        v311_manifest_json=DEFAULT_V311_MANIFEST,
        output_dir=tmp,
        label="v312_self_test",
        seed=123,
        train_bit_per_pattern=1,
        train_equation_per_rule=1,
        val_bit_per_pattern=1,
        val_equation_per_rule=1,
        reference_path=DEFAULT_REFERENCE_PATHS,
    )
    manifest = run(args)
    if manifest["train_summary"]["rows"] != 14:
        raise AssertionError(manifest["train_summary"])
    if manifest["validation_summary"]["rows"] != 14:
        raise AssertionError(manifest["validation_summary"])
    if manifest["train_preference_summary"]["rows"] != 56:
        raise AssertionError(manifest["train_preference_summary"])
    for path in tmp.glob("*"):
        path.unlink()
    tmp.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v311-manifest-json", type=Path, default=DEFAULT_V311_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v312_verifier_synthetic_distill")
    parser.add_argument("--seed", type=int, default=312)
    parser.add_argument("--train-bit-per-pattern", type=int, default=12)
    parser.add_argument("--train-equation-per-rule", type=int, default=24)
    parser.add_argument("--val-bit-per-pattern", type=int, default=3)
    parser.add_argument("--val-equation-per-rule", type=int, default=6)
    parser.add_argument("--reference-path", type=Path, action="append", default=list(DEFAULT_REFERENCE_PATHS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("v312_verifier_synthetic_distill_self_test=ok", flush=True)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

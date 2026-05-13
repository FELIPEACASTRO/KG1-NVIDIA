#!/usr/bin/env python3
"""Build V337D minimal transfer dataset after V336 package gate.

V337D is the adapter-only fallback after V336A found no-loss solver gains and
V336B proved direct solver packaging is blocked. It intentionally differs from
the failed V335 mix: only the no-loss equation traces are used, plus a balanced
bit replay slice for non-regression. Broad equation replay is excluded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V336A_MANIFEST = (
    REPO_ROOT
    / "artifacts/v336_integrated_no_loss_solver_gate/20260513T_cpu_gate/"
    / "v336a_integrated_no_loss_solver_gate_manifest.json"
)
DEFAULT_V336B_MANIFEST = (
    REPO_ROOT
    / "artifacts/v336b_package_permission_gate/20260513T_cpu_gate/"
    / "v336b_package_permission_gate_manifest.json"
)
DEFAULT_V325_MANIFEST = (
    REPO_ROOT
    / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/"
    / "v325_equation_no_loss_distill_manifest.json"
)
DEFAULT_V330_MANIFEST = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_manifest.json"
)
DEFAULT_V325_TRAIN = (
    REPO_ROOT
    / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/"
    / "v325_equation_no_loss_distill_sft_train.jsonl"
)
DEFAULT_V325_VAL = (
    REPO_ROOT
    / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/"
    / "v325_equation_no_loss_distill_sft_val.jsonl"
)
DEFAULT_V330_TRAIN = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_sft_train.jsonl"
)
DEFAULT_V330_VAL = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_sft_val.jsonl"
)
DEFAULT_V325_PREF_TRAIN = (
    REPO_ROOT
    / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/"
    / "v325_equation_no_loss_distill_preferences_train.jsonl"
)
DEFAULT_V325_PREF_VAL = (
    REPO_ROOT
    / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/"
    / "v325_equation_no_loss_distill_preferences_val.jsonl"
)
DEFAULT_V330_PREF_TRAIN = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_preferences_train.jsonl"
)
DEFAULT_V330_PREF_VAL = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_preferences_val.jsonl"
)
DEFAULT_V217_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
DEFAULT_V217_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
DEFAULT_REFERENCE_CSVS = [
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv",
]
EXPECTED_V217_TRAIN_SHA256 = "a56938b1ae9eb471b779ebfc415ee88c05322941732128752680317495157984"
EXPECTED_V217_VAL_SHA256 = "65c4cb88b8ff2fc96940ccea33b8ca493769790c7ae80d27f2b69ac818fc6451"


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def prompt_sha(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


def boxed_final_line(answer: Any) -> str:
    return "Final answer: " + r"\boxed{" + str(answer) + "}"


def normalize_assistant_content(content: str, answer: str) -> tuple[str, str]:
    boxed = boxed_final_line(answer)
    text = str(content or "").rstrip()
    pattern = r"Final answer:\s*(?:\\boxed\{.*\}|[^\n]+)\s*$"
    if re.search(pattern, text, flags=re.S):
        updated = re.sub(pattern, lambda _match: boxed, text, count=1, flags=re.S)
        return updated, "replaced_final_answer_suffix"
    return (text + "\n" + boxed if text else boxed), "appended_final_answer_suffix"


def read_reference_csv(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompts: set[str] = set()
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "ids": ids, "prompt_hashes": prompts}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompts.add(sha256_text(normalize_prompt(prompt)))
    return {"path": str(path), "exists": True, "rows": len(ids), "ids": ids, "prompt_hashes": prompts}


def validate_v336_inputs(v336a_path: Path, v336b_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    v336a = read_json(v336a_path)
    v336b = read_json(v336b_path)
    if v336a.get("schema_version") != "kg1_v336a_integrated_no_loss_solver_gate_v1":
        raise RuntimeError("unexpected V336A schema")
    if v336a.get("decision", {}).get("decision") != "v336a_cpu_integrated_no_loss_gate_passed":
        raise RuntimeError("V336A did not pass")
    integrated = v336a.get("integrated", {})
    if int(integrated.get("correct", -1)) != 197 or int(integrated.get("loss_count", -1)) != 0:
        raise RuntimeError("V336A integrated gate drift")
    if v336b.get("schema_version") != "kg1_v336b_package_permission_gate_v1":
        raise RuntimeError("unexpected V336B schema")
    if v336b.get("decision", {}).get("decision") != "solver_verifier_direct_package_blocked_adapter_only_required":
        raise RuntimeError("V336B did not block direct solver package")
    return v336a, v336b


def validate_component_manifest(path: Path, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"unexpected schema for {path}: {payload.get('schema_version')}")
    return payload


def normalize_row(row: dict[str, Any], component: str, split: str) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(row)
    answer = str(out.get("answer", ""))
    prompt = str(out.get("prompt", ""))
    messages = out.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise RuntimeError(f"bad messages for row {out.get('id')}")
    if messages[1].get("content") != prompt:
        raise RuntimeError(f"user message/prompt mismatch for row {out.get('id')}")
    updated, status = normalize_assistant_content(str(messages[2].get("content", "")), answer)
    messages[2]["content"] = updated
    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["weak_gate_rows_used_for_training"] = False
    metadata["full_gate_rows_used_for_training"] = False
    metadata["gate_rows_used_for_training"] = False
    metadata["v337d_component"] = component
    metadata["v337d_normalized_final_answer"] = "boxed_suffix"
    metadata["v337d_normalization_status"] = status
    metadata["split"] = split
    out["metadata"] = metadata
    out["source"] = str(out.get("source") or metadata.get("source_dataset") or component)
    return out, status


def sample_bit_rows(
    path: Path,
    *,
    count: int,
    split: str,
    seed: int,
    references: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if str(row.get("family", "")) == "bit_manipulation"]
    ref_ids = set().union(*(ref["ids"] for ref in references))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in references))
    filtered: list[dict[str, Any]] = []
    skipped_id = 0
    skipped_prompt = 0
    for row in rows:
        if str(row.get("id", "")) in ref_ids:
            skipped_id += 1
            continue
        if prompt_sha(row) in ref_prompts:
            skipped_prompt += 1
            continue
        filtered.append(row)
    if len(filtered) < count:
        raise RuntimeError(f"not enough bit rows in {path}: {len(filtered)} < {count}")
    rng = random.Random(seed)
    keyed = [(sha256_text(str(seed) + "\0" + str(row.get("id", ""))), row) for row in filtered]
    keyed.sort(key=lambda item: item[0])
    selected = [row for _, row in keyed[: count * 2]]
    rng.shuffle(selected)
    selected = selected[:count]
    normalized: list[dict[str, Any]] = []
    norm_counts: Counter[str] = Counter()
    for index, row in enumerate(selected):
        out = copy.deepcopy(row)
        out["id"] = f"v337d_{split}_bit_replay_{index:05d}_{sha256_text(str(row.get('id', '')))[:8]}"
        metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
        metadata = dict(metadata)
        metadata["source_dataset"] = "v337d_v217_bit_replay"
        metadata["source_original_id"] = str(row.get("id", ""))
        metadata["source_original_prompt_sha256"] = prompt_sha(row)
        metadata["weak_gate_rows_used_for_training"] = False
        metadata["full_gate_rows_used_for_training"] = False
        metadata["gate_rows_used_for_training"] = False
        out["metadata"] = metadata
        out["source"] = "v337d_v217_bit_replay"
        out["subcategory"] = str(out.get("subcategory") or "bit_manipulation")
        normalized_row, status = normalize_row(out, "v337d_v217_bit_replay", split)
        norm_counts[status] += 1
        normalized.append(normalized_row)
    return normalized, {
        "path": str(path),
        "available_bit_rows": len(rows),
        "eligible_bit_rows": len(filtered),
        "selected_bit_rows": len(normalized),
        "skipped_id_overlap": skipped_id,
        "skipped_prompt_overlap": skipped_prompt,
        "normalization": dict(norm_counts),
    }


def load_component_rows(path: Path, component: str, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        out, status = normalize_row(row, component, split)
        counts[status] += 1
        normalized.append(out)
    return normalized, dict(counts)


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    kept: list[dict[str, Any]] = []
    stats = {"duplicate_ids": 0, "duplicate_prompts": 0}
    for row in rows:
        rid = str(row.get("id", ""))
        psha = prompt_sha(row)
        if rid in seen_ids:
            stats["duplicate_ids"] += 1
            continue
        if psha in seen_prompts:
            stats["duplicate_prompts"] += 1
            continue
        seen_ids.add(rid)
        seen_prompts.add(psha)
        kept.append(row)
    return kept, stats


def validate_rows(rows: list[dict[str, Any]], split: str, references: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[str] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    ref_ids = set().union(*(ref["ids"] for ref in references))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in references))
    for row in rows:
        rid = str(row.get("id", ""))
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        messages = row.get("messages")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        psha = prompt_sha(row)
        if not rid or not prompt or not answer:
            bad.append(f"{rid}:missing_required_field")
        if rid in ids:
            bad.append(f"{rid}:duplicate_id")
        if psha in prompts:
            bad.append(f"{rid}:duplicate_prompt")
        if rid in ref_ids:
            bad.append(f"{rid}:reference_id_overlap")
        if psha in ref_prompts:
            bad.append(f"{rid}:reference_prompt_overlap")
        if not isinstance(messages, list) or len(messages) != 3:
            bad.append(f"{rid}:bad_messages")
        else:
            if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
                bad.append(f"{rid}:bad_roles")
            if str(messages[1].get("content", "")) != prompt:
                bad.append(f"{rid}:prompt_mismatch")
            assistant = str(messages[2].get("content", ""))
            expected_suffix = boxed_final_line(answer)
            if not assistant.rstrip().endswith(expected_suffix):
                bad.append(f"{rid}:assistant_not_boxed_suffix")
            if len(re.findall(r"\\boxed\{([^{}]*)\}", assistant)) < 1:
                bad.append(f"{rid}:missing_boxed_answer")
        for flag in ("weak_gate_rows_used_for_training", "full_gate_rows_used_for_training", "gate_rows_used_for_training"):
            if metadata.get(flag) is not False:
                bad.append(f"{rid}:{flag}_not_false")
        ids.add(rid)
        prompts.add(psha)
        family_counts[str(row.get("family", ""))] += 1
        source_counts[str(metadata.get("source_dataset", row.get("source", "")))] += 1
        component_counts[str(metadata.get("v337d_component", ""))] += 1
        subcategory_counts[str(row.get("subcategory", metadata.get("subcategory", "")))] += 1
    if bad:
        raise RuntimeError(f"{split} validation failed: " + json.dumps(bad[:30], ensure_ascii=False))
    return {
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_prompt_hashes": len(prompts),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(source_counts.most_common(40)),
        "component_counts": dict(sorted(component_counts.items())),
        "subcategory_counts": dict(subcategory_counts.most_common(40)),
        "reference_id_overlap": 0,
        "reference_prompt_overlap": 0,
    }


def merge_preferences(paths: list[Path], output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    write_jsonl(output_path, rows)
    negative_counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        negative_counts[str(metadata.get("negative_type", "unknown"))] += 1
    return {
        "rows": len(rows),
        "sha256": sha256_file(output_path),
        "negative_type_counts": dict(sorted(negative_counts.items())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V337D MINIMAL TRANSFER DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("seed =", args.seed, flush=True)
    print("bit_train_rows =", args.bit_train_rows, flush=True)
    print("bit_val_rows =", args.bit_val_rows, flush=True)
    v336a, v336b = validate_v336_inputs(args.v336a_manifest_json, args.v336b_manifest_json)
    v325 = validate_component_manifest(args.v325_manifest_json, "kg1_v325_equation_no_loss_distill_dataset_v1")
    v330 = validate_component_manifest(args.v330_manifest_json, "kg1_v330_symbolic_cryptarithm_distill_dataset_v1")
    if sha256_file(args.v217_train_jsonl) != EXPECTED_V217_TRAIN_SHA256:
        raise RuntimeError("V217 train SHA mismatch")
    if sha256_file(args.v217_val_jsonl) != EXPECTED_V217_VAL_SHA256:
        raise RuntimeError("V217 val SHA mismatch")

    references = [read_reference_csv(path) for path in args.reference_csv]
    train_v325, norm_v325_train = load_component_rows(args.v325_train_jsonl, "v325_equation_numeric_no_loss", "train")
    val_v325, norm_v325_val = load_component_rows(args.v325_val_jsonl, "v325_equation_numeric_no_loss", "validation")
    train_v330, norm_v330_train = load_component_rows(args.v330_train_jsonl, "v330_symbolic_cryptarithm_no_loss", "train")
    val_v330, norm_v330_val = load_component_rows(args.v330_val_jsonl, "v330_symbolic_cryptarithm_no_loss", "validation")
    train_bit, bit_train_summary = sample_bit_rows(
        args.v217_train_jsonl,
        count=args.bit_train_rows,
        split="train",
        seed=args.seed,
        references=references,
    )
    val_bit, bit_val_summary = sample_bit_rows(
        args.v217_val_jsonl,
        count=args.bit_val_rows,
        split="validation",
        seed=args.seed + 10000,
        references=references,
    )

    train_rows, train_dedupe = dedupe_rows(train_v325 + train_v330 + train_bit)
    val_rows, val_dedupe = dedupe_rows(val_v325 + val_v330 + val_bit)
    train_summary = validate_rows(train_rows, "train", references)
    val_summary = validate_rows(val_rows, "validation", references)

    train_prompt_overlap = {prompt_sha(row) for row in train_rows} & {prompt_sha(row) for row in val_rows}
    train_id_overlap = {str(row.get("id", "")) for row in train_rows} & {str(row.get("id", "")) for row in val_rows}
    if train_prompt_overlap or train_id_overlap:
        raise RuntimeError("train/validation overlap detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_train.jsonl"
    val_path = args.output_dir / f"{args.label}_val.jsonl"
    pref_train_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    pref_val_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    pref_train_summary = merge_preferences([args.v325_pref_train_jsonl, args.v330_pref_train_jsonl], pref_train_path)
    pref_val_summary = merge_preferences([args.v325_pref_val_jsonl, args.v330_pref_val_jsonl], pref_val_path)

    manifest = {
        "schema_version": "kg1_v337d_minimal_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "inputs": {
            "v336a_manifest_json": str(args.v336a_manifest_json),
            "v336a_manifest_sha256": sha256_file(args.v336a_manifest_json),
            "v336b_manifest_json": str(args.v336b_manifest_json),
            "v336b_manifest_sha256": sha256_file(args.v336b_manifest_json),
            "v325_manifest_json": str(args.v325_manifest_json),
            "v325_manifest_sha256": sha256_file(args.v325_manifest_json),
            "v330_manifest_json": str(args.v330_manifest_json),
            "v330_manifest_sha256": sha256_file(args.v330_manifest_json),
            "v217_train_jsonl": str(args.v217_train_jsonl),
            "v217_train_sha256": sha256_file(args.v217_train_jsonl),
            "v217_val_jsonl": str(args.v217_val_jsonl),
            "v217_val_sha256": sha256_file(args.v217_val_jsonl),
        },
        "source_gate": {
            "v336a_decision": v336a.get("decision"),
            "v336b_decision": v336b.get("decision"),
            "v336a_integrated": v336a.get("integrated"),
            "v325_source_gate": v325.get("source_gate", {"accepted_candidate_ids": v325.get("v324_accepted_candidate_ids")}),
            "v330_source_gate": v330.get("source_gate"),
        },
        "source_policy": {
            "purpose": "minimal adapter-only transfer after direct solver package was blocked",
            "equation_rows": "V325 numeric no-loss and V330 symbolic cryptarithm only",
            "bit_rows": "balanced V217 bit replay slice with reference-overlap filtering",
            "excluded": ["V304 broad equation replay", "V335 broad mixed replay", "weak/full gate rows as train rows"],
            "final_answer_format": "boxed_suffix",
            "weak_or_full_gate_rows_used_for_training": False,
        },
        "component_normalization": {
            "v325_train": norm_v325_train,
            "v325_validation": norm_v325_val,
            "v330_train": norm_v330_train,
            "v330_validation": norm_v330_val,
            "bit_train": bit_train_summary,
            "bit_validation": bit_val_summary,
        },
        "dedupe": {"train": train_dedupe, "validation": val_dedupe},
        "validation": {"train": train_summary, "validation": val_summary},
        "train_val_overlap": {"id_overlap": 0, "prompt_overlap": 0},
        "reference_summary": [
            {
                "path": ref["path"],
                "exists": ref["exists"],
                "rows": ref["rows"],
                "sha256": sha256_file(Path(ref["path"])) if ref["exists"] else "",
            }
            for ref in references
        ],
        "preference_summary": {"train": pref_train_summary, "validation": pref_val_summary},
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "preferences_train_jsonl": str(pref_train_path),
            "preferences_train_sha256": sha256_file(pref_train_path),
            "preferences_val_jsonl": str(pref_val_path),
            "preferences_val_sha256": sha256_file(pref_val_path),
            "manifest_json": str(manifest_path),
        },
        "training_authorization": "blocked_until_v286_boxed_suffix_tokenization_gate",
        "required_next_gate": [
            "scripts/run_v286_generic_tokenization_gate.py --assistant-final-answer-mode boxed_suffix",
            "HF smoke only; first checkpoint kill-switch: total>192, equation>56, bit>=136",
            "no full eval/package/submit until adapter-only weak gate improves without family regression",
        ],
    }
    write_json(manifest_path, manifest)
    print("train_rows =", len(train_rows), flush=True)
    print("validation_rows =", len(val_rows), flush=True)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V337D MINIMAL TRANSFER DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    assert boxed_final_line("1010") == r"Final answer: \boxed{1010}"
    updated, status = normalize_assistant_content("reason\nFinal answer: 1010", "1010")
    if status != "replaced_final_answer_suffix" or not updated.endswith(r"\boxed{1010}"):
        raise AssertionError((updated, status))
    print("v337d_minimal_transfer_dataset_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v336a-manifest-json", type=Path, default=DEFAULT_V336A_MANIFEST)
    parser.add_argument("--v336b-manifest-json", type=Path, default=DEFAULT_V336B_MANIFEST)
    parser.add_argument("--v325-manifest-json", type=Path, default=DEFAULT_V325_MANIFEST)
    parser.add_argument("--v330-manifest-json", type=Path, default=DEFAULT_V330_MANIFEST)
    parser.add_argument("--v325-train-jsonl", type=Path, default=DEFAULT_V325_TRAIN)
    parser.add_argument("--v325-val-jsonl", type=Path, default=DEFAULT_V325_VAL)
    parser.add_argument("--v330-train-jsonl", type=Path, default=DEFAULT_V330_TRAIN)
    parser.add_argument("--v330-val-jsonl", type=Path, default=DEFAULT_V330_VAL)
    parser.add_argument("--v325-pref-train-jsonl", type=Path, default=DEFAULT_V325_PREF_TRAIN)
    parser.add_argument("--v325-pref-val-jsonl", type=Path, default=DEFAULT_V325_PREF_VAL)
    parser.add_argument("--v330-pref-train-jsonl", type=Path, default=DEFAULT_V330_PREF_TRAIN)
    parser.add_argument("--v330-pref-val-jsonl", type=Path, default=DEFAULT_V330_PREF_VAL)
    parser.add_argument("--v217-train-jsonl", type=Path, default=DEFAULT_V217_TRAIN)
    parser.add_argument("--v217-val-jsonl", type=Path, default=DEFAULT_V217_VAL)
    parser.add_argument("--reference-csv", type=Path, action="append", default=list(DEFAULT_REFERENCE_CSVS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/v337d_minimal_transfer_dataset" / utc_compact(),
    )
    parser.add_argument("--label", default="v337d_minimal_transfer")
    parser.add_argument("--seed", type=int, default=337)
    parser.add_argument("--bit-train-rows", type=int, default=720)
    parser.add_argument("--bit-val-rows", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

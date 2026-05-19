"""Build V673 guarded transfer data from the V672 residual-miss ledger.

V673 is CPU-only and does not train on the weak 315 gate rows.  The V672 ledger
authorizes rule classes; this builder creates fresh synthetic rows for the
equation classes and reuses the already-synthetic V367 bit replay rows.
"""

from __future__ import annotations

import argparse
import copy
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

REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import build_v312_verifier_synthetic_distill_dataset as v312  # noqa: E402
from src.competition_utils import (  # noqa: E402
    PROMPT_SUFFIX,
    extract_boxed_answers,
    extract_final_answer,
    verify_answer,
)


SCHEMA_VERSION = "kg1_v673_guarded_equation_bit_transfer_dataset_v1"
DEFAULT_LEDGER_MANIFEST = (
    REPO_ROOT
    / "artifacts/v672_residual_miss_ledger/20260519T173138Z/"
    / "v672_residual_miss_ledger_manifest.json"
)
DEFAULT_V367_TRAIN = (
    REPO_ROOT
    / "artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/"
    / "v367_v366_bit_ternary_transfer_train.jsonl"
)
DEFAULT_V367_VAL = (
    REPO_ROOT
    / "artifacts/v367_v366_bit_ternary_transfer_dataset/20260514T_cpu_gate/"
    / "v367_v366_bit_ternary_transfer_val.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v673_guarded_equation_bit_transfer_dataset"
DEFAULT_REFERENCE_CSVS = [
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v516_label_free_weak_baseline/v516_label_free_v290_checkpoint6_baseline.csv",
]

RULE_CLASS_TO_V312_INDEX = {
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded": 0,
    "v274_guarded_numeric_colon_absdiff_restore_trailing_zero": 3,
    "v274_guarded_numeric_add_direct_over_model_add_variant": 4,
}

BIN8_RE = re.compile(r"\b[01]{8}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: bad JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).split())


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content") or "")
            return content.removesuffix(PROMPT_SUFFIX)
    return ""


def official_messages(prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(prompt) + PROMPT_SUFFIX},
        {"role": "assistant", "content": str(assistant).strip()},
    ]


def bit_trace_assistant(row: dict[str, Any], answer: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt = prompt_text(row)
    bit_values = BIN8_RE.findall(prompt)
    query = bit_values[-1] if bit_values else ""
    if query:
        query_hex = f"{int(query, 2):02X}"
        # V513 intentionally normalizes decimal-like spans.  Encode the query
        # byte as letters a-p so the trace remains row-specific without adding
        # a numeric shortcut or the target answer.
        query_code = "q" + "".join(chr(ord("a") + int(ch, 16)) for ch in query_hex)
    else:
        query_code = "unknown"
    expr = str(metadata.get("expr") or metadata.get("rule_slug") or metadata.get("rule_class") or "unknown_bit_rule")
    rule_class = str(metadata.get("rule_class") or "bit_rule")
    return "\n".join(
        [
            f"Rule: apply the 8-bit operation {expr}.",
            f"Rule class: {rule_class}.",
            f"Query byte code: {query_code}.",
            "Use the same bit transformation shown by the prompt examples.",
            f"Final answer: \\boxed{{{answer}}}",
        ]
    )


def with_v673_contract(
    row: dict[str, Any],
    *,
    new_id: str,
    split: str,
    component: str,
    source_rule: str,
    loss_weight: float,
    source_ledger_ids: list[str],
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    prompt = prompt_text(out)
    assistant = assistant_text(out)
    answer = str(out.get("answer") or "").strip()
    raw_box = "\\boxed{" + answer + "}"
    final_box_line = "Final answer: " + raw_box
    stripped_assistant = assistant.rstrip()
    if str(out.get("family") or "") == "bit_manipulation":
        assistant = bit_trace_assistant(out, answer)
    elif stripped_assistant == raw_box:
        assistant = final_box_line
    elif stripped_assistant.endswith(raw_box) and not stripped_assistant.endswith(final_box_line):
        assistant = stripped_assistant[: -len(raw_box)].rstrip() + "\n" + final_box_line
    completion_format = "trace_plus_final_boxed" if "\n" in assistant.strip() else "final_boxed"
    out["id"] = new_id
    out["prompt"] = prompt
    out["messages"] = official_messages(prompt, assistant)
    out["source"] = "v673_guarded_equation_bit_transfer_dataset"
    out["source_dataset"] = "v673_guarded_equation_bit_transfer_dataset"
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "v673_component": component,
            "v673_source_rule": source_rule,
            "v673_source_ledger_ids": source_ledger_ids,
            "prompt_contract": "official_like",
            "prompt_suffix": PROMPT_SUFFIX,
            "completion_format": completion_format,
            "loss_weight": loss_weight,
            "weak_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "submit_direct": False,
        }
    )
    out["metadata"] = metadata
    out["loss_weight"] = loss_weight
    if not verify_answer(answer, extract_final_answer(assistant)):
        raise RuntimeError(f"{new_id}: assistant does not verify against answer")
    boxes = [item.strip() for item in extract_boxed_answers(assistant)]
    if len(boxes) != 1:
        raise RuntimeError(f"{new_id}: expected exactly one boxed answer, got {len(boxes)}")
    return out


def load_reference_fingerprints(paths: list[Path]) -> tuple[set[str], set[str]]:
    ref_ids: set[str] = set()
    ref_prompt_hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_csv(path)
        for row in rows:
            rid = str(row.get("id") or "").strip()
            if rid:
                ref_ids.add(rid)
            prompt = str(row.get("prompt") or "").strip()
            if prompt:
                ref_prompt_hashes.add(sha256_text(normalize_prompt(prompt)))
    return ref_ids, ref_prompt_hashes


def select_v367_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("subcategory") or (row.get("metadata") or {}).get("rule_class") or "unknown")
        grouped.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(grouped)
    for key in keys:
        rng.shuffle(grouped[key])
    cursor = 0
    while len(selected) < count and any(grouped.values()):
        key = keys[cursor % len(keys)]
        if grouped[key]:
            selected.append(grouped[key].pop())
        cursor += 1
    if len(selected) < count:
        raise RuntimeError(f"not enough V367 rows: requested {count}, got {len(selected)}")
    rng.shuffle(selected)
    return selected


def ledger_rows(ledger_manifest: dict[str, Any]) -> list[dict[str, str]]:
    outputs = ledger_manifest.get("outputs") or {}
    ledger_csv = outputs.get("ledger_csv")
    if not ledger_csv:
        raise RuntimeError("ledger manifest missing outputs.ledger_csv")
    path = Path(str(ledger_csv))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return read_csv(path)


def validate_ledger(manifest: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    gate = manifest.get("gpu_gate")
    if gate != "allow_a100_large_equation_transfer_probe_guarded":
        raise RuntimeError(f"V672 ledger does not authorize V673 guarded probe: {gate}")
    eq_rows = [
        row
        for row in rows
        if row.get("miss_class") == "equation_numeric_miss"
        and row.get("trainability_decision") in {"trainable", "trainable_guarded"}
        and str(row.get("candidate_correct")).lower() == "true"
    ]
    bit_rows = [
        row
        for row in rows
        if row.get("miss_class") == "bit_residual_miss"
        and row.get("trainability_decision") == "trainable"
        and str(row.get("candidate_correct")).lower() == "true"
    ]
    if len(eq_rows) < 4:
        raise RuntimeError(f"need at least 4 usable equation rows, got {len(eq_rows)}")
    if len(bit_rows) < 4:
        raise RuntimeError(f"need at least 4 usable bit replay rows, got {len(bit_rows)}")
    rule_counts: Counter[str] = Counter()
    rule_ids: dict[str, list[str]] = {}
    for row in eq_rows:
        rule = str(row.get("candidate_rule") or "")
        if rule not in RULE_CLASS_TO_V312_INDEX:
            raise RuntimeError(f"unsupported V673 equation rule: {rule}")
        rule_counts[rule] += 1
        rule_ids.setdefault(rule, []).append(str(row.get("id")))
    return {
        "equation_rows": eq_rows,
        "bit_rows": bit_rows,
        "equation_rule_counts": dict(sorted(rule_counts.items())),
        "equation_rule_ids": rule_ids,
    }


def build_equation_rows(
    *,
    split: str,
    rows_per_evidence: int,
    seed: int,
    ledger_info: dict[str, Any],
    loss_weight: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    row_index = 0
    for rule, count in sorted(ledger_info["equation_rule_counts"].items()):
        rule_index = RULE_CLASS_TO_V312_INDEX[rule]
        source_ids = ledger_info["equation_rule_ids"][rule]
        for _ in range(count * rows_per_evidence):
            raw = v312.build_equation_row(
                rng,
                split=split,
                rule_index=rule_index,
                row_index=row_index,
            )
            rows.append(
                with_v673_contract(
                    raw,
                    new_id=f"v673_{split}_eq_{row_index:05d}_{sha256_text(raw['prompt'])[:10]}",
                    split=split,
                    component="equation_primary_synthetic",
                    source_rule=rule,
                    loss_weight=loss_weight,
                    source_ledger_ids=source_ids,
                )
            )
            row_index += 1
    rng.shuffle(rows)
    return rows


def build_bit_rows(
    *,
    split: str,
    source_rows: list[dict[str, Any]],
    count: int,
    seed: int,
    loss_weight: float,
    source_ledger_ids: list[str],
) -> list[dict[str, Any]]:
    selected = select_v367_rows(source_rows, count, seed)
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(selected):
        metadata = raw.get("metadata") or {}
        source_rule = str(metadata.get("rule_class") or metadata.get("expr") or raw.get("subcategory") or "v367_bit_replay")
        out.append(
            with_v673_contract(
                raw,
                new_id=f"v673_{split}_bit_{idx:05d}_{sha256_text(prompt_text(raw))[:10]}",
                split=split,
                component="bit_replay_synthetic",
                source_rule=source_rule,
                loss_weight=loss_weight,
                source_ledger_ids=source_ledger_ids,
            )
        )
    return out


def validate_rows(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    ref_ids: set[str],
    ref_prompt_hashes: set[str],
) -> dict[str, Any]:
    seen_ids: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    split_summaries: dict[str, Any] = {}
    for split, rows in [("train", train_rows), ("validation", val_rows)]:
        family = Counter()
        subcategory = Counter()
        component = Counter()
        weights = Counter()
        boxed_count = 0
        for row in rows:
            row_id = str(row.get("id") or "")
            if not row_id:
                raise RuntimeError(f"{split}: missing id")
            if row_id in seen_ids:
                raise RuntimeError(f"{split}: duplicate id {row_id}")
            if row_id in ref_ids:
                raise RuntimeError(f"{split}: id overlaps reference/gate row {row_id}")
            seen_ids.add(row_id)
            prompt_hash = sha256_text(normalize_prompt(prompt_text(row)))
            if prompt_hash in seen_prompt_hashes:
                raise RuntimeError(f"{split}: duplicate prompt hash {prompt_hash}")
            if prompt_hash in ref_prompt_hashes:
                raise RuntimeError(f"{split}: prompt overlaps reference/gate row {row_id}")
            seen_prompt_hashes.add(prompt_hash)
            answer = str(row.get("answer") or "").strip()
            assistant = assistant_text(row)
            extracted = extract_final_answer(assistant)
            if not verify_answer(answer, extracted):
                raise RuntimeError(f"{split}:{row_id}: answer mismatch {answer!r} vs {extracted!r}")
            boxes = extract_boxed_answers(assistant)
            if len(boxes) != 1:
                raise RuntimeError(f"{split}:{row_id}: expected one boxed answer, got {len(boxes)}")
            boxed_count += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            for flag in ("weak_gate_rows_used_for_training", "gate_rows_used_for_training", "full_gate_rows_used_for_training"):
                if metadata.get(flag) is True:
                    raise RuntimeError(f"{split}:{row_id}: anti-leak flag {flag}=true")
            family[str(row.get("family") or metadata.get("family") or "")] += 1
            subcategory[str(row.get("subcategory") or metadata.get("subcategory") or "")] += 1
            component[str(metadata.get("v673_component") or "")] += 1
            weights[str(metadata.get("loss_weight") or row.get("loss_weight") or "")] += 1
        split_summaries[split] = {
            "rows": len(rows),
            "family_counts": dict(sorted(family.items())),
            "subcategory_counts": dict(sorted(subcategory.items())),
            "component_counts": dict(sorted(component.items())),
            "loss_weight_counts": dict(sorted(weights.items())),
            "boxed_rows": boxed_count,
            "boxed_rate": boxed_count / len(rows) if rows else 0.0,
        }
    overlap = set(sha256_text(normalize_prompt(prompt_text(row))) for row in train_rows) & set(
        sha256_text(normalize_prompt(prompt_text(row))) for row in val_rows
    )
    if overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(overlap)}")
    return split_summaries


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V673 GUARDED EQUATION BIT TRANSFER DATASET START ===", flush=True)
    print("ledger_manifest_json =", args.ledger_manifest_json, flush=True)
    print("v367_train_jsonl =", args.v367_train_jsonl, flush=True)
    print("v367_val_jsonl =", args.v367_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ledger_manifest = read_json(args.ledger_manifest_json)
    ledger_info = validate_ledger(ledger_manifest, ledger_rows(ledger_manifest))
    ref_ids, ref_prompt_hashes = load_reference_fingerprints(args.reference_csv)

    eq_train = build_equation_rows(
        split="train",
        rows_per_evidence=args.equation_train_rows_per_evidence,
        seed=args.seed,
        ledger_info=ledger_info,
        loss_weight=args.equation_loss_weight,
    )
    eq_val = build_equation_rows(
        split="validation",
        rows_per_evidence=args.equation_val_rows_per_evidence,
        seed=args.seed + 10_000,
        ledger_info=ledger_info,
        loss_weight=args.equation_loss_weight,
    )
    bit_train = build_bit_rows(
        split="train",
        source_rows=read_jsonl(args.v367_train_jsonl),
        count=args.bit_train_rows,
        seed=args.seed + 20_000,
        loss_weight=args.bit_loss_weight,
        source_ledger_ids=[str(row["id"]) for row in ledger_info["bit_rows"]],
    )
    bit_val = build_bit_rows(
        split="validation",
        source_rows=read_jsonl(args.v367_val_jsonl),
        count=args.bit_val_rows,
        seed=args.seed + 30_000,
        loss_weight=args.bit_loss_weight,
        source_ledger_ids=[str(row["id"]) for row in ledger_info["bit_rows"]],
    )

    train_rows = eq_train + bit_train
    val_rows = eq_val + bit_val
    random.Random(args.seed + 40_000).shuffle(train_rows)
    random.Random(args.seed + 50_000).shuffle(val_rows)

    summary = validate_rows(train_rows, val_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes)

    train_path = args.output_dir / "v673_guarded_equation_bit_transfer_train.jsonl"
    val_path = args.output_dir / "v673_guarded_equation_bit_transfer_val.jsonl"
    manifest_path = args.output_dir / "v673_guarded_equation_bit_transfer_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "ledger_manifest_json": rel(args.ledger_manifest_json),
        "ledger_manifest_sha256": sha256_file(args.ledger_manifest_json),
        "v367_train_jsonl": rel(args.v367_train_jsonl),
        "v367_train_sha256": sha256_file(args.v367_train_jsonl),
        "v367_val_jsonl": rel(args.v367_val_jsonl),
        "v367_val_sha256": sha256_file(args.v367_val_jsonl),
        "equation_rule_counts": ledger_info["equation_rule_counts"],
        "equation_rule_ids": ledger_info["equation_rule_ids"],
        "bit_source_ledger_ids": [str(row["id"]) for row in ledger_info["bit_rows"]],
        "parameters": {
            "seed": args.seed,
            "equation_train_rows_per_evidence": args.equation_train_rows_per_evidence,
            "equation_val_rows_per_evidence": args.equation_val_rows_per_evidence,
            "bit_train_rows": args.bit_train_rows,
            "bit_val_rows": args.bit_val_rows,
            "equation_loss_weight": args.equation_loss_weight,
            "bit_loss_weight": args.bit_loss_weight,
        },
        "summary": summary,
        "reference_csvs": [rel(path) for path in args.reference_csv],
        "reference_sha256": {rel(path): sha256_file(path) for path in args.reference_csv},
        "outputs": {
            "sft_train_jsonl": rel(train_path),
            "sft_val_jsonl": rel(val_path),
            "train_jsonl": rel(train_path),
            "val_jsonl": rel(val_path),
            "manifest_json": rel(manifest_path),
        },
        "hashes": {
            "sft_train_sha256": sha256_file(train_path),
            "sft_val_sha256": sha256_file(val_path),
        },
        "training_authorization": "blocked_until_v286_tokenization_gate_then_a100_large_probe",
        "next_gate": [
            "python scripts/run_v286_generic_tokenization_gate.py --assistant-final-answer-mode boxed_suffix",
            "cheap A100-large probe only; H200 blocked",
            "weak eval must reach bit>=136 equation>=60 total>=196 with no protected backfire",
        ],
    }
    write_json(manifest_path, manifest)

    print("train_jsonl =", train_path, flush=True)
    print("val_jsonl =", val_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("summary =", json.dumps(summary, sort_keys=True), flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V673 GUARDED EQUATION BIT TRANSFER DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-manifest-json", type=Path, default=DEFAULT_LEDGER_MANIFEST)
    parser.add_argument("--v367-train-jsonl", type=Path, default=DEFAULT_V367_TRAIN)
    parser.add_argument("--v367-val-jsonl", type=Path, default=DEFAULT_V367_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / utc_tag())
    parser.add_argument("--seed", type=int, default=67319)
    parser.add_argument("--equation-train-rows-per-evidence", type=int, default=120)
    parser.add_argument("--equation-val-rows-per-evidence", type=int, default=30)
    parser.add_argument("--bit-train-rows", type=int, default=240)
    parser.add_argument("--bit-val-rows", type=int, default=60)
    parser.add_argument("--equation-loss-weight", type=float, default=1.0)
    parser.add_argument("--bit-loss-weight", type=float, default=0.35)
    parser.add_argument("--reference-csv", type=Path, action="append", default=list(DEFAULT_REFERENCE_CSVS))
    return parser.parse_args()


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

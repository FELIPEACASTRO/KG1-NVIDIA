#!/usr/bin/env python3
"""Build the V217 short-answer rescue dataset.

V216 executed successfully but failed the weak gate with many truncated
generations. This dataset keeps the same audited V216 prompt/answer pool, drops
only the four train prompts proven to prompt-truncate at max_length=4096, and
rewrites assistant completions to a one-line final answer. The goal is to teach
the adapter to terminate quickly without changing the validation split.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition_utils import classify_puzzle, extract_final_answer, verify_answer  # noqa: E402

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

SRC_TRAIN = PROJECT_ROOT / "data/v216/v216_score_push_train.jsonl"
SRC_VAL = PROJECT_ROOT / "data/v216/v216_score_push_val.jsonl"
SRC_MANIFEST = PROJECT_ROOT / "data/v216/v216_score_push_manifest.json"
OUT_DIR = PROJECT_ROOT / "data/v217"
TRAIN_OUT = OUT_DIR / "v217_short_answer_train.jsonl"
VAL_OUT = OUT_DIR / "v217_short_answer_val.jsonl"
MANIFEST_OUT = OUT_DIR / "v217_short_answer_manifest.json"

EXPECTED_SRC_TRAIN_SHA = "8cfd065c102187b12c131aae7475c35e28073721175b4e6108004b0afc4d5d03"
EXPECTED_SRC_VAL_SHA = "80efe71260c8589b998699543c85aff3ff140bc90e431dfa0ec33bce3e0921c0"
EXPECTED_SRC_TRAIN_ROWS = 10210
EXPECTED_SRC_VAL_ROWS = 681

PROMPT_TRUNCATED_TRAIN_IDS = {
    "clean_safe_strict_de25c8b4c874f62d",
    "clean_safe_strict_5719011b2b3da39f",
    "clean_safe_strict_384d775636c751d4",
    "clean_safe_strict_33a9e59cca3d55f8",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def signature(prompt: object, answer: object) -> str:
    payload = json.dumps(
        {"prompt": str(prompt or "").strip(), "answer": str(answer or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def short_answer_text(answer: object) -> str:
    value = str(answer or "").strip()
    if "\n" in value or "\r" in value:
        value = " ".join(part.strip() for part in value.splitlines() if part.strip())
    return f"Final answer: {value}"


def normalize_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not prompt or not answer:
        raise ValueError(f"Missing prompt/answer in {split}: {row.get('id')}")

    family = str(row.get("family") or classify_puzzle(prompt))
    subtype = str(row.get("subcategory") or (row.get("metadata") or {}).get("subtype") or "unknown")
    source = str(row.get("source") or (row.get("metadata") or {}).get("source") or "unknown")
    original_id = str(row.get("id") or "")
    assistant = short_answer_text(answer)
    extracted = extract_final_answer(assistant)
    if not verify_answer(answer, extracted):
        raise ValueError(f"Short answer extraction mismatch for {original_id}: {answer!r} vs {extracted!r}")
    if family != classify_puzzle(prompt):
        raise ValueError(f"Family classifier mismatch for {original_id}: {family} vs {classify_puzzle(prompt)}")

    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "original_id": original_id,
            "original_source": source,
            "source": source,
            "subtype": subtype,
            "subcategory": subtype,
            "train_allowed": True,
            "v217_short_answer_role": f"v217_{split}",
            "v217_short_answer_source": source,
            "v217_answer_style": "final_answer_one_line_unboxed",
        }
    )

    row_id_payload = f"v217|{split}|{original_id}|{prompt}|{answer}"
    return {
        "id": f"v217_{split}_{hashlib.sha1(row_id_payload.encode('utf-8')).hexdigest()[:16]}",
        "family": family,
        "subcategory": subtype,
        "prompt": prompt,
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
        "source": source,
    }


def summarize(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source") or "unknown") for row in rows)
    subtype_counts = Counter(str(row.get("subcategory") or "unknown") for row in rows)
    completion_lengths = [len(row["messages"][-1]["content"]) for row in rows]
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "assistant_chars": {
            "min": min(completion_lengths) if completion_lengths else 0,
            "p50": sorted(completion_lengths)[len(completion_lengths) // 2] if completion_lengths else 0,
            "max": max(completion_lengths) if completion_lengths else 0,
        },
    }


def validate_rows(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        sig = signature(row.get("prompt"), row.get("answer"))
        if sig in seen:
            errors["duplicate_prompt_answer"] += 1
        seen.add(sig)
        messages = row.get("messages")
        if not isinstance(messages, list) or [msg.get("role") for msg in messages] != ["system", "user", "assistant"]:
            errors["bad_messages"] += 1
            continue
        extracted = extract_final_answer(str(messages[-1].get("content") or ""))
        if not verify_answer(row.get("answer"), extracted):
            errors["extract_mismatch"] += 1
        if str(row.get("family") or "") != classify_puzzle(str(row.get("prompt") or "")):
            errors["family_classifier_mismatch"] += 1
    return {"split": split, "errors": dict(sorted(errors.items())), "unique_prompt_answers": len(seen)}


def main() -> None:
    for required in [SRC_TRAIN, SRC_VAL, SRC_MANIFEST]:
        if not required.exists():
            raise FileNotFoundError(required)
    if sha256_file(SRC_TRAIN) != EXPECTED_SRC_TRAIN_SHA:
        raise RuntimeError("Source V216 train SHA mismatch")
    if sha256_file(SRC_VAL) != EXPECTED_SRC_VAL_SHA:
        raise RuntimeError("Source V216 val SHA mismatch")
    src_manifest = load_json(SRC_MANIFEST)
    if src_manifest.get("status") != "PASS":
        raise RuntimeError("Source V216 manifest is not PASS")

    raw_train = load_jsonl(SRC_TRAIN)
    raw_val = load_jsonl(SRC_VAL)
    if len(raw_train) != EXPECTED_SRC_TRAIN_ROWS:
        raise RuntimeError(f"Source train row count mismatch: {len(raw_train)}")
    if len(raw_val) != EXPECTED_SRC_VAL_ROWS:
        raise RuntimeError(f"Source val row count mismatch: {len(raw_val)}")

    removed: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(raw_train, start=1):
        original_id = str(row.get("id") or "")
        if original_id in PROMPT_TRUNCATED_TRAIN_IDS:
            removed.append(
                {
                    "line_no": line_no,
                    "id": original_id,
                    "family": row.get("family"),
                    "source": row.get("source"),
                    "subcategory": row.get("subcategory"),
                }
            )
            continue
        train_rows.append(normalize_row(row, split="train"))
    if {row["id"] for row in removed} != PROMPT_TRUNCATED_TRAIN_IDS:
        raise RuntimeError("Prompt-truncated row filter mismatch")

    val_rows = [normalize_row(row, split="val") for row in raw_val]
    train_validation = validate_rows(train_rows, split="train")
    val_validation = validate_rows(val_rows, split="validation")
    if train_validation["errors"] or val_validation["errors"]:
        raise RuntimeError(f"Validation errors: train={train_validation} val={val_validation}")

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(VAL_OUT, val_rows)
    manifest = {
        "version": "v217_short_answer_rescue",
        "created_at_utc": utc_now(),
        "objective": "Reduce weak-eval truncation by training short final-answer completions from the audited V216 score-push pool.",
        "status": "PASS",
        "inputs": {
            "src_train": str(SRC_TRAIN),
            "src_val": str(SRC_VAL),
            "src_manifest": str(SRC_MANIFEST),
            "src_train_sha256": EXPECTED_SRC_TRAIN_SHA,
            "src_val_sha256": EXPECTED_SRC_VAL_SHA,
        },
        "removed_prompt_truncated_train_rows": removed,
        "train_validation": train_validation,
        "validation_validation": val_validation,
        "recommended_training_env": {
            "MIN_TRAIN_EXAMPLES": str(len(train_rows)),
            "MIN_VAL_EXAMPLES": str(len(val_rows)),
            "MIN_TOKENIZED_TRAIN_EXAMPLES": str(len(train_rows)),
            "MIN_TOKENIZED_VAL_EXAMPLES": str(len(val_rows)),
            "SAMPLING_MODE": "weighted_replacement",
            "SOURCE_WEIGHTS": (
                "v216_synthetic_kg1_symbolic_rules=1.0,"
                "v216_synthetic_kg1_numeric_rules=0.9,"
                "v216_synthetic_kg1_bit_rules=1.05,"
                "v216_base_clean_safe_strict_equation=1.0,"
                "v216_base_clean_safe_strict_bit=1.05,"
                "v215_replay_anchor=1.0"
            ),
            "SUBCATEGORY_WEIGHTS": (
                "equation_symbolic_binary=1.05,"
                "equation_symbolic_unary=1.05,"
                "equation_numeric=0.9,"
                "equation numeric=0.9,"
                "equation symbolic/mixed=1.0,"
                "bit_manipulation=1.05,"
                "unknown=1.0"
            ),
        },
    }
    write_json(MANIFEST_OUT, manifest)
    manifest["train"] = summarize(train_rows, TRAIN_OUT)
    manifest["validation"] = summarize(val_rows, VAL_OUT)
    write_json(MANIFEST_OUT, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

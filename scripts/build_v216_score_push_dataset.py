#!/usr/bin/env python3
"""Build the V216 score-push train/validation package.

This step is CPU-only. It combines:

- the previously audited strict clean-safe public weak-family pool;
- the new KG1-shaped V216 equation/bit synthetic augmentation;
- the small V215 strong/bit replay anchor.

The output is intended for a gated Colab continuation run. It does not train,
evaluate, package, or submit anything.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition_utils import classify_puzzle, extract_final_answer, verify_answer  # noqa: E402

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Use concise reasoning, infer the hidden rule from examples, and give exactly one final answer."
)

STRICT_POOL = (
    PROJECT_ROOT
    / "artifacts/weak_family_mega_search_20260507/"
    "candidate_weak_family_teacher_pool_historic_mega_strict_kg1_clean_safe.jsonl"
)
V216_TRAIN = PROJECT_ROOT / "data/v216/v216_equation_symbolic_focus_train.jsonl"
V216_VAL = PROJECT_ROOT / "data/v216/v216_equation_symbolic_focus_val.jsonl"
V215_TRAIN = PROJECT_ROOT / "data/v215/v215_bit_focused_train.jsonl"
V215_VAL = PROJECT_ROOT / "data/v215/v215_bit_focused_val.jsonl"
WEAK_VALIDATION = PROJECT_ROOT / "artifacts/weak_family_mega_search_20260507/weak_validation_recomputed_rows.csv"

OUT_DIR = PROJECT_ROOT / "data/v216"
TRAIN_OUT = OUT_DIR / "v216_score_push_train.jsonl"
VAL_OUT = OUT_DIR / "v216_score_push_val.jsonl"
MANIFEST_OUT = OUT_DIR / "v216_score_push_manifest.json"
REPORT_OUT = PROJECT_ROOT / "artifacts/analysis_ias_11_12_quadruple_20260507/V216_SCORE_PUSH_DATASET_REPORT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(prompt: object, answer: object) -> str:
    payload = json.dumps(
        {"prompt": str(prompt or "").strip(), "answer": str(answer or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    for key in ("assistant_or_cot", "completion", "assistant", "cot"):
        if row.get(key):
            return str(row[key])
    return ""


def text_field(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def has_prompt_answer(row: dict[str, Any]) -> bool:
    return bool(text_field(row, "prompt").strip() and text_field(row, "answer").strip())


def normalize_existing_row(row: dict[str, Any], *, source: str, role: str) -> dict[str, Any]:
    prompt = text_field(row, "prompt").strip()
    answer = text_field(row, "answer").strip()
    if not prompt or not answer:
        raise ValueError("Cannot normalize row without prompt and answer")

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text(row)},
        ]

    family = str(
        row.get("family")
        or row.get("computed_family")
        or row.get("family_target")
        or classify_puzzle(prompt)
    )
    metadata = dict(row.get("metadata") or {})
    subtype = row.get("subtype") or metadata.get("subtype") or metadata.get("subcategory") or ""
    metadata.update(
        {
            "train_allowed": True,
            "v216_score_push_role": role,
            "v216_score_push_source": source,
            "original_id": row.get("id", ""),
            "original_source": row.get("source") or row.get("source_ref") or "",
            "quality_tag": row.get("quality_tag", metadata.get("quality_tag", "")),
            "subtype": subtype,
            "subcategory": subtype,
        }
    )
    return {
        "id": f"{role}_{hashlib.sha1((prompt + answer + source).encode('utf-8')).hexdigest()[:16]}",
        "family": family,
        "subcategory": subtype or "unknown",
        "prompt": prompt,
        "answer": answer,
        "messages": messages,
        "metadata": metadata,
        "source": source,
    }


def normalize_v216_row(row: dict[str, Any], *, role: str) -> dict[str, Any]:
    copied = normalize_existing_row(
        row,
        source=str(row.get("source") or (row.get("metadata") or {}).get("source") or "v216_synthetic"),
        role=role,
    )
    metadata = dict(copied["metadata"])
    metadata["v216_score_push_source"] = copied["source"]
    copied["metadata"] = metadata
    return copied


def row_ok(row: dict[str, Any]) -> tuple[bool, str]:
    prompt = text_field(row, "prompt")
    answer = text_field(row, "answer").strip()
    messages = row.get("messages")
    if not prompt or not answer:
        return False, "missing_prompt_or_answer"
    if not isinstance(messages, list) or len(messages) < 3:
        return False, "bad_messages"
    roles = [str(msg.get("role") or "") for msg in messages if isinstance(msg, dict)]
    if roles[0] != "system" or roles[-2] != "user" or roles[-1] != "assistant":
        return False, "bad_message_roles"
    extracted = extract_final_answer(assistant_text(row))
    if not verify_answer(answer, extracted):
        return False, "extract_mismatch"
    family = str(row.get("family") or classify_puzzle(prompt))
    if family == "bit_manipulation" and not (len(answer) == 8 and set(answer) <= {"0", "1"}):
        return False, "bad_bit_answer_shape"
    if family not in {"bit_manipulation", "equation_transform", "gravity_constant", "numeral_system", "text_encryption", "unit_conversion"}:
        return False, "unexpected_family"
    if family != classify_puzzle(prompt):
        return False, "family_classifier_mismatch"
    return True, "ok"


def load_validation_signatures(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    sigs: set[str] = set()
    for _, row in df.iterrows():
        prompt = row.get("prompt") or row.get("prompt_x") or row.get("prompt_y") or ""
        answer = row.get("answer") or row.get("solution") or ""
        if prompt and answer:
            sigs.add(signature(prompt, answer))
    return sigs


def take_rows(
    rows: list[dict[str, Any]],
    *,
    family: str | None,
    limit: int,
    seed: int,
    skip_sigs: set[str],
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (family is None or str(row.get("family") or classify_puzzle(str(row.get("prompt") or ""))) == family)
        and signature(row.get("prompt"), row.get("answer")) not in skip_sigs
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit]


def add_unique(
    target: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    used_sigs: set[str],
    rejected: Counter[str],
) -> None:
    for row in rows:
        sig = signature(row.get("prompt"), row.get("answer"))
        if sig in used_sigs:
            rejected["duplicate_prompt_answer"] += 1
            continue
        ok, reason = row_ok(row)
        if not ok:
            rejected[reason] += 1
            continue
        used_sigs.add(sig)
        target.append(row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source") or "unknown") for row in rows)
    subtype_counts = Counter(
        str((row.get("metadata") or {}).get("subtype") or "unknown")
        for row in rows
    )
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    required = [STRICT_POOL, V216_TRAIN, V216_VAL, V215_TRAIN, V215_VAL]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    validation_sigs = load_validation_signatures(WEAK_VALIDATION)
    prefilter_rejected: Counter[str] = Counter()
    strict_raw = load_jsonl(STRICT_POOL)
    v216_train_raw = load_jsonl(V216_TRAIN)
    v216_val_raw = load_jsonl(V216_VAL)
    prefilter_rejected["v216_train_empty_prompt_or_answer"] = sum(
        1 for row in v216_train_raw if not has_prompt_answer(row)
    )
    prefilter_rejected["v216_val_empty_prompt_or_answer"] = sum(
        1 for row in v216_val_raw if not has_prompt_answer(row)
    )
    v216_train = [
        normalize_v216_row(row, role="v216_target_train")
        for row in v216_train_raw
        if has_prompt_answer(row)
    ]
    v216_val = [
        normalize_v216_row(row, role="v216_target_val")
        for row in v216_val_raw
        if has_prompt_answer(row)
    ]
    v215_train = [
        normalize_existing_row(row, source="v215_replay_anchor", role="v215_anchor_train")
        for row in load_jsonl(V215_TRAIN)
    ]
    v215_val = [
        normalize_existing_row(row, source="v215_replay_anchor", role="v215_anchor_val")
        for row in load_jsonl(V215_VAL)
    ]

    strict_norm: list[dict[str, Any]] = []
    for row in strict_raw:
        family = str(row.get("computed_family") or row.get("family_target") or "")
        if family == "equation_transform":
            source = "v216_base_clean_safe_strict_equation"
        elif family == "bit_manipulation":
            source = "v216_base_clean_safe_strict_bit"
        else:
            continue
        strict_norm.append(normalize_existing_row(row, source=source, role="clean_safe_strict"))

    used_train = set(validation_sigs)
    used_val = set(validation_sigs)
    rejected_train: Counter[str] = Counter()
    rejected_val: Counter[str] = Counter()
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    add_unique(train_rows, v216_train, used_train, rejected_train)
    add_unique(train_rows, take_rows(strict_norm, family="equation_transform", limit=args.strict_equation_train, seed=args.seed + 1, skip_sigs=used_train), used_train, rejected_train)
    add_unique(train_rows, take_rows(strict_norm, family="bit_manipulation", limit=args.strict_bit_train, seed=args.seed + 2, skip_sigs=used_train), used_train, rejected_train)
    add_unique(train_rows, v215_train, used_train, rejected_train)

    # Validation is diagnostic only; keep it disjoint from train and external weak validation.
    used_val.update(signature(row.get("prompt"), row.get("answer")) for row in train_rows)
    add_unique(val_rows, v216_val, used_val, rejected_val)
    add_unique(val_rows, v215_val, used_val, rejected_val)
    add_unique(val_rows, take_rows(strict_norm, family="equation_transform", limit=args.strict_equation_val, seed=args.seed + 3, skip_sigs=used_val), used_val, rejected_val)
    add_unique(val_rows, take_rows(strict_norm, family="bit_manipulation", limit=args.strict_bit_val, seed=args.seed + 4, skip_sigs=used_val), used_val, rejected_val)

    train_sigs = {signature(row.get("prompt"), row.get("answer")) for row in train_rows}
    val_sigs = {signature(row.get("prompt"), row.get("answer")) for row in val_rows}
    overlap = train_sigs & val_sigs
    validation_leaks_train = train_sigs & validation_sigs
    validation_leaks_val = val_sigs & validation_sigs

    issues: list[dict[str, Any]] = []
    if overlap:
        issues.append({"severity": "ERROR", "code": "train_val_overlap", "count": len(overlap)})
    if validation_leaks_train or validation_leaks_val:
        issues.append(
            {
                "severity": "ERROR",
                "code": "validation_leak",
                "train_count": len(validation_leaks_train),
                "val_count": len(validation_leaks_val),
            }
        )
    if len(train_rows) < args.min_train:
        issues.append({"severity": "ERROR", "code": "train_too_small", "count": len(train_rows)})
    if len(val_rows) < args.min_val:
        issues.append({"severity": "ERROR", "code": "val_too_small", "count": len(val_rows)})

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(VAL_OUT, val_rows)

    source_weights = {
        "v216_synthetic_kg1_symbolic_rules": 0.55,
        "v216_synthetic_kg1_numeric_rules": 0.45,
        "v216_synthetic_kg1_bit_rules": 0.25,
        "v216_base_clean_safe_strict_equation": 0.85,
        "v216_base_clean_safe_strict_bit": 0.75,
        "v215_replay_anchor": 0.90,
    }
    subcategory_weights = {
        "equation_symbolic_binary": 1.20,
        "equation_symbolic_unary": 1.20,
        "equation_numeric": 0.85,
        "bit_manipulation": 0.75,
    }

    manifest = {
        "version": "v216_score_push",
        "created_at_utc": utc_now(),
        "status": "PASS" if not any(item["severity"] == "ERROR" for item in issues) else "FAIL",
        "objective": "Score-push continuation package focused on equation_transform with bit preservation and strong replay anchor.",
        "inputs": {
            "strict_pool": str(STRICT_POOL),
            "v216_train": str(V216_TRAIN),
            "v216_val": str(V216_VAL),
            "v215_train": str(V215_TRAIN),
            "v215_val": str(V215_VAL),
            "weak_validation": str(WEAK_VALIDATION),
        },
        "selection": {
            "seed": args.seed,
            "strict_equation_train": args.strict_equation_train,
            "strict_bit_train": args.strict_bit_train,
            "strict_equation_val": args.strict_equation_val,
            "strict_bit_val": args.strict_bit_val,
            "v216_train_all": True,
            "v215_train_all": True,
        },
        "train": {
            **summarize(train_rows),
            "path": str(TRAIN_OUT),
            "sha256": sha256_file(TRAIN_OUT),
        },
        "validation": {
            **summarize(val_rows),
            "path": str(VAL_OUT),
            "sha256": sha256_file(VAL_OUT),
        },
        "recommended_training_env": {
            "SAMPLING_MODE": "weighted_replacement",
            "SOURCE_WEIGHTS": ",".join(f"{key}={value}" for key, value in source_weights.items()),
            "SUBCATEGORY_WEIGHTS": ",".join(f"{key}={value}" for key, value in subcategory_weights.items()),
            "MIN_TRAIN_EXAMPLES": str(len(train_rows)),
            "MIN_VAL_EXAMPLES": str(len(val_rows)),
            "MIN_TOKENIZED_TRAIN_EXAMPLES": str(len(train_rows)),
            "MIN_TOKENIZED_VAL_EXAMPLES": str(len(val_rows)),
        },
        "gates": {
            "baseline_weak_total_expected": 190,
            "baseline_equation_transform_expected": 55,
            "baseline_bit_manipulation_expected": 135,
            "promote_weak_total_min": 193,
            "promote_equation_transform_min": 60,
            "promote_bit_manipulation_min": 133,
            "full_candidate_min": 831,
            "full_truncation_max": 4,
        },
        "rejected_train": dict(sorted(rejected_train.items())),
        "rejected_val": dict(sorted(rejected_val.items())),
        "prefilter_rejected": dict(sorted(prefilter_rejected.items())),
        "issues": issues,
    }
    write_json(MANIFEST_OUT, manifest)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "# V216 Score Push Dataset",
                "",
                f"Status: **{manifest['status']}**",
                "",
                "## Train",
                f"- rows: {manifest['train']['rows']}",
                f"- path: `{TRAIN_OUT}`",
                f"- sha256: `{manifest['train']['sha256']}`",
                f"- family_counts: `{manifest['train']['family_counts']}`",
                "",
                "## Validation",
                f"- rows: {manifest['validation']['rows']}",
                f"- path: `{VAL_OUT}`",
                f"- sha256: `{manifest['validation']['sha256']}`",
                f"- family_counts: `{manifest['validation']['family_counts']}`",
                "",
                "## Recommended Gates",
                "- Promote only if weak total >= 193, equation_transform >= 60, bit_manipulation >= 133.",
                "- Run full validation only after weak gate passes.",
                "- Package only if full validation >= 831 and truncation <= 4.",
                "",
                "## Recommended Training Env",
                f"- SOURCE_WEIGHTS: `{manifest['recommended_training_env']['SOURCE_WEIGHTS']}`",
                f"- SUBCATEGORY_WEIGHTS: `{manifest['recommended_training_env']['SUBCATEGORY_WEIGHTS']}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=21650)
    parser.add_argument("--strict-equation-train", type=int, default=3200)
    parser.add_argument("--strict-bit-train", type=int, default=1800)
    parser.add_argument("--strict-equation-val", type=int, default=120)
    parser.add_argument("--strict-bit-val", type=int, default=80)
    parser.add_argument("--min-train", type=int, default=10000)
    parser.add_argument("--min-val", type=int, default=650)
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

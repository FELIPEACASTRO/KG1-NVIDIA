#!/usr/bin/env python3
"""Build V206B answer-only micro training data.

V206A regressed on validation loss after three very small updates. The likely
cause is objective mismatch: V206A trained verbose reasoning completions while
the validation proxy and Kaggle parser reward concise final answers. V206B keeps
the verified prompt/answer knowledge but rewrites every assistant turn to a
single boxed answer and balances the hard families against rehearsal families.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


SOURCE = PROJECT_ROOT / "data" / "v206" / "v206_curated_train.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "v206b"
TRAIN_OUT = OUT_DIR / "v206b_answer_only_micro_train.jsonl"
MANIFEST_OUT = OUT_DIR / "v206b_answer_only_manifest.json"

SEED = 2062
TARGET_PER_FAMILY = {
    "bit_manipulation": 600,
    "equation_transform": 600,
    "gravity_constant": 120,
    "numeral_system": 120,
    "text_encryption": 120,
    "unit_conversion": 120,
}
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Return exactly one final answer in \\boxed{...} and no extra text."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_key(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {"prompt": str(row.get("prompt") or "").strip(), "answer": str(row.get("answer") or "").strip()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def source_priority(row: dict[str, Any]) -> tuple[int, str]:
    source = str(row.get("source") or "")
    family = str(row.get("family") or "")
    if family == "bit_manipulation":
        order = [
            "v206::v95_bit_rehearsal_train",
            "v206::v92_delta_train",
            "v206::v100_programmatic_repair_train",
            "v206::v90_gold_safe_train",
        ]
    elif family == "equation_transform":
        order = [
            "v206::v100_programmatic_repair_train",
            "v206::v94_equation_crypt_train",
            "v206::v92_delta_train",
            "v206::v95_bit_rehearsal_train",
            "v206::v90_gold_safe_train",
        ]
    else:
        order = [
            "v206::v90_gold_safe_train",
            "v206::v95_bit_rehearsal_train",
            "v206::v92_delta_train",
            "v206::v100_programmatic_repair_train",
            "v206::v94_equation_crypt_train",
        ]
    return (order.index(source) if source in order else len(order), source)


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row["answer"]).strip()
    family = str(row["family"]).strip()
    prompt = str(row.get("prompt") or "")
    out = {
        "id": f"v206b_{row['id']}",
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "source": f"v206b_answer_only::{row.get('source', 'unknown')}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ],
        "metadata": {
            "subcategory": family,
            "v206b_role": "answer_only_micro",
            "v206b_origin_id": row.get("id"),
            "v206b_origin_source": row.get("source"),
        },
    }
    extracted = extract_final_answer(out["messages"][-1]["content"])
    if not verify_answer(answer, extracted):
        raise ValueError(f"Answer-only conversion failed for {row.get('id')}: {answer!r} -> {extracted!r}")
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = load_jsonl(SOURCE)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    rejected = Counter()

    for row in rows:
        family = str(row.get("family") or "")
        answer = str(row.get("answer") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        if family not in TARGET_PER_FAMILY:
            rejected["family_not_targeted"] += 1
            continue
        if not answer or not prompt:
            rejected["missing_prompt_or_answer"] += 1
            continue
        key = row_key(row)
        if key in seen:
            rejected["duplicate_prompt_answer"] += 1
            continue
        seen.add(key)
        by_family[family].append(row)

    selected: list[dict[str, Any]] = []
    selected_source_counts: Counter[str] = Counter()
    for family, target in TARGET_PER_FAMILY.items():
        family_rows = by_family[family]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            grouped[str(row.get("source") or "")].append(row)
        for values in grouped.values():
            rng.shuffle(values)
        ordered_sources = sorted(grouped, key=lambda source: source_priority({"family": family, "source": source}))

        family_selected: list[dict[str, Any]] = []
        while len(family_selected) < target and any(grouped.values()):
            for source in ordered_sources:
                values = grouped[source]
                if values and len(family_selected) < target:
                    item = values.pop()
                    family_selected.append(item)
                    selected_source_counts[str(item.get("source") or "")] += 1
        if len(family_selected) < target:
            raise RuntimeError(f"Not enough rows for {family}: {len(family_selected)} < {target}")
        selected.extend(family_selected)

    converted = [convert_row(row) for row in selected]
    rng.shuffle(converted)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    family_counts = Counter(row["family"] for row in converted)
    source_counts = Counter(row["source"] for row in converted)
    manifest = {
        "schema_version": "v206b_answer_only_micro_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_path": str(SOURCE.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "train_path": str(TRAIN_OUT.relative_to(PROJECT_ROOT)),
        "train_rows": len(converted),
        "train_sha256": sha256_file(TRAIN_OUT),
        "seed": SEED,
        "target_per_family": TARGET_PER_FAMILY,
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "origin_source_counts": dict(sorted(selected_source_counts.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "decision": {
            "use": "V206B answer-only micro candidate, not for direct Kaggle submit without gates",
            "rationale": [
                "V206A regressed after verbose-reasoning SFT",
                "answer-only completions match the validation proxy and Kaggle final-answer objective",
                "hard families are emphasized while strong families are retained as rehearsal",
            ],
        },
    }
    write_json(MANIFEST_OUT, manifest)
    print(json.dumps({"train": str(TRAIN_OUT), "manifest": str(MANIFEST_OUT), "sha256": manifest["train_sha256"], "rows": len(converted)}, indent=2))


if __name__ == "__main__":
    main()

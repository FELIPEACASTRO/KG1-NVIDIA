#!/usr/bin/env python3
"""Build V479 by filtering V475 to V324-evidenced equation rules plus bit replay.

V476 physically included bit replay, but weighted sampling reduced bit to less
than 1% of the effective objective.  V479 is CPU-only: it creates a smaller
dataset that keeps equation rules directly supported by V324 accepted candidates
and keeps the V217 bit replay guardrail with comparable physical presence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V475_ROOT = ROOT / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix"
V475_MANIFEST = V475_ROOT / "v475_equation_bit_replay_mix_manifest.json"
V475_TRAIN = V475_ROOT / "v475_equation_bit_replay_mix_train.jsonl"
V475_VAL = V475_ROOT / "v475_equation_bit_replay_mix_val.jsonl"
OUT_DIR = ROOT / "artifacts/v479_objective_aligned_filter/20260516T_v479_objective_aligned_filter"
TRAIN_OUT = OUT_DIR / "v479_objective_aligned_filter_train.jsonl"
VAL_OUT = OUT_DIR / "v479_objective_aligned_filter_val.jsonl"
MANIFEST_OUT = OUT_DIR / "v479_objective_aligned_filter_manifest.json"
COMPARISON_OUT = OUT_DIR / "V479_VS_PREVIOUS.md"

ALLOWED_SUBCATEGORIES = {
    "bit_guardrail_replay",
    "equation_numeric_add_direct",
    "equation_numeric_colon_trailing_zero",
    "equation_numeric_minus_signed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def audit_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids: Counter[str] = Counter(str(row.get("id", "")) for row in rows)
    prompts: Counter[str] = Counter(str(row.get("prompt", "")) for row in rows)
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    subcategory_counts = Counter(str(row.get("subcategory", "")) for row in rows)
    bad_rows: list[str] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        messages = row.get("messages")
        answer = str(row.get("answer", ""))
        if str(row.get("subcategory", "")) not in ALLOWED_SUBCATEGORIES:
            bad_rows.append(f"{row_id}:disallowed_subcategory:{row.get('subcategory')}")
        if not isinstance(messages, list) or len(messages) != 3:
            bad_rows.append(f"{row_id}:bad_messages")
        elif [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            bad_rows.append(f"{row_id}:bad_roles")
        elif messages[1].get("content") != row.get("prompt"):
            bad_rows.append(f"{row_id}:prompt_mismatch")
        elif not str(messages[2].get("content", "")).rstrip().endswith(f"Final answer: \\boxed{{{answer}}}"):
            bad_rows.append(f"{row_id}:assistant_suffix_mismatch")
        if len(bad_rows) >= 20:
            break
    if bad_rows:
        raise RuntimeError(f"{label} audit failed: {bad_rows[:20]}")
    return {
        "label": label,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "duplicate_ids": sum(1 for value in ids.values() if value > 1),
        "duplicate_prompts": sum(1 for value in prompts.values() if value > 1),
        "unique_ids": len(ids),
        "unique_prompts": len(prompts),
        "bad_rows_first20": bad_rows[:20],
    }


def build_split(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected = [row for row in rows if str(row.get("subcategory", "")) in ALLOWED_SUBCATEGORIES]
    selected.sort(key=lambda row: (str(row.get("family", "")), str(row.get("subcategory", "")), str(row.get("id", ""))))
    return selected


def write_comparison(manifest: dict[str, Any]) -> None:
    train = manifest["train_summary"]
    val = manifest["validation_summary"]
    text = "\n".join(
        [
            "# V479 vs V475/V476",
            "",
            "| Item | V475/V476 | V479 |",
            "|---|---:|---:|",
            "| Train rows | 1312 | {rows} |".format(rows=train["rows"]),
            "| Train equation rows | 800 | {rows} |".format(rows=train["family_counts"].get("equation_transform", 0)),
            "| Train bit rows | 512 | {rows} |".format(rows=train["family_counts"].get("bit_manipulation", 0)),
            "| Validation rows | 328 | {rows} |".format(rows=val["rows"]),
            "| Validation equation rows | 200 | {rows} |".format(rows=val["family_counts"].get("equation_transform", 0)),
            "| Validation bit rows | 128 | {rows} |".format(rows=val["family_counts"].get("bit_manipulation", 0)),
            "| Equation classes | 5 including exploratory | 3 V324-evidenced classes |",
            "| Effective objective policy | V476 weighted equation 99.0508%, bit 0.9492% | require V478 objective gate before GPU |",
            "",
            "V479 does not authorize GPU by itself. It is a cleaner CPU candidate for",
            "tokenization and objective-alignment gates before any paid job.",
            "",
        ]
    )
    COMPARISON_OUT.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    print("=== V479 OBJECTIVE ALIGNED FILTER START ===", flush=True)
    print("v475_manifest =", V475_MANIFEST, flush=True)
    print("output_dir =", OUT_DIR, flush=True)
    v475_manifest = read_json(V475_MANIFEST)
    train_rows = build_split(V475_TRAIN)
    val_rows = build_split(V475_VAL)
    train_audit = audit_rows(train_rows, "train")
    val_audit = audit_rows(val_rows, "validation")
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(VAL_OUT, val_rows)
    train_sha = sha256_file(TRAIN_OUT)
    val_sha = sha256_file(VAL_OUT)
    manifest = {
        "schema_version": "kg1_v479_objective_aligned_filter_v1",
        "generated_at_utc": utc_now(),
        "source_manifest": str(V475_MANIFEST),
        "source_manifest_sha256": sha256_file(V475_MANIFEST),
        "allowed_subcategories": sorted(ALLOWED_SUBCATEGORIES),
        "baseline": v475_manifest.get("baseline"),
        "v476_v477_failure_mode": {
            "checkpoint_2": {"weak_total": "192/315", "equation_transform": "57/155", "bit_manipulation": "135/160"},
            "checkpoint_4": {"weak_total": "191/315", "equation_transform": "57/155", "bit_manipulation": "134/160"},
            "root_gap": "V476 effective sampling made bit only 0.9492% of weighted objective.",
        },
        "outputs": {
            "manifest_json": str(MANIFEST_OUT),
            "comparison_md": str(COMPARISON_OUT),
            "train_jsonl": str(TRAIN_OUT),
            "val_jsonl": str(VAL_OUT),
            "train_sha256": train_sha,
            "val_sha256": val_sha,
        },
        "train_summary": train_audit,
        "validation_summary": val_audit,
        "required_next_gate": [
            "V286 real tokenization gate with boxed_suffix",
            "V478 objective-alignment gate with bit effective share >= 20%",
            "static safety gate",
            "no HF GPU unless weak kill-switch remains total>192 equation>56 bit>=136 truncated=0",
        ],
        "training_authorization": "blocked_until_v286_and_v478_gates_pass",
    }
    write_comparison(manifest)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("train_summary =", json.dumps(train_audit, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_audit, sort_keys=True), flush=True)
    print("train_sha256 =", train_sha, flush=True)
    print("val_sha256 =", val_sha, flush=True)
    print("manifest_json =", MANIFEST_OUT, flush=True)
    print("comparison_md =", COMPARISON_OUT, flush=True)
    print("=== V479 OBJECTIVE ALIGNED FILTER END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

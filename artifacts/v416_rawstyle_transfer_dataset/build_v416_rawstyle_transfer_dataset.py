#!/usr/bin/env python3
"""Build V416 raw-output-style transfer dataset.

V416 keeps the leak-safe synthetic prompts from V410, but changes the assistant
completion format to better match the long raw outputs the current adapter emits
at inference time. It does not train on weak/full gate rows.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V410_DIR = REPO_ROOT / "artifacts/v410_solver_first_transfer_dataset/20260514T_v410_solver_first_transfer"
V410_TRAIN = V410_DIR / "v410_solver_first_transfer_train.jsonl"
V410_VAL = V410_DIR / "v410_solver_first_transfer_val.jsonl"
V410_MANIFEST = V410_DIR / "v410_solver_first_transfer_manifest.json"
OUT_DIR = REPO_ROOT / "artifacts/v416_rawstyle_transfer_dataset/20260515T_v416_rawstyle_transfer"


BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assistant_content(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def set_assistant_content(row: dict[str, Any], content: str) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row))
    for message in new_row.get("messages", []):
        if message.get("role") == "assistant":
            message["content"] = content
            return new_row
    raise ValueError(f"row missing assistant message: {row.get('id')}")


def strip_final_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"Final answer:\s*\\boxed\{[^{}]*\}\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\\boxed\{[^{}]*\}\s*$", "", text).strip()
    return text


def rawstyle_completion(row: dict[str, Any]) -> str:
    answer = str(row["answer"]).strip()
    family = str(row.get("family", ""))
    source = str(row.get("source", ""))
    subcategory = str(row.get("subcategory", ""))
    previous = strip_final_answer(assistant_content(row))

    if family == "bit_manipulation":
        header = "We need to infer the bit manipulation rule from the examples."
        if previous:
            body = previous
        else:
            body = "The examples define a consistent 8-bit transformation. Apply the same transformation to the query."
        check = "The query output produced by this rule is " + answer + "."
    elif family == "equation_transform":
        header = "We need to infer the equation transformation rule from the examples."
        if previous:
            body = previous
        else:
            body = "The examples define a consistent operator transformation. Apply the same operator rule to the query."
        check = "Applying the same rule to the query gives " + answer + "."
    else:
        header = "We need to infer the hidden transformation rule from the examples."
        body = previous or "Apply the same transformation to the query."
        check = "The query result is " + answer + "."

    provenance = f"Source tag: {source}; subtype: {subcategory}."
    return "\n".join(
        [
            header,
            "",
            body,
            "",
            provenance,
            check,
            "",
            "I will now return the answer in \\boxed{}.",
            f"Final answer: \\boxed{{{answer}}}",
        ]
    )


def convert_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        content = rawstyle_completion(row)
        boxed = BOXED_RE.findall(content)
        if not boxed or boxed[-1] != str(row["answer"]).strip():
            raise RuntimeError(f"bad final boxed answer for {row.get('id')}")
        new_row = set_assistant_content(row, content)
        new_row["id"] = str(row["id"]).replace("v410_", "v416_", 1)
        new_row["source"] = str(row.get("source", "")) + "_rawstyle_v416"
        metadata = dict(new_row.get("metadata", {}))
        metadata.update(
            {
                "v416_origin_id": row.get("id"),
                "v416_transfer_format": "raw_output_style_boxed_suffix",
                "v416_uses_weak_or_full_rows_for_training": False,
                "weak_gate_rows_used_for_training": False,
                "full_gate_rows_used_for_training": False,
                "split": split,
            }
        )
        new_row["metadata"] = metadata
        converted.append(new_row)
    return converted


def audit(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [str(row.get("prompt", "")) for row in rows]
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    source_counts = Counter(str(row.get("source", "")) for row in rows)
    bad_rows = []
    for row in rows[:]:
        content = assistant_content(row)
        boxed = BOXED_RE.findall(content)
        if not boxed or boxed[-1] != str(row.get("answer", "")).strip():
            bad_rows.append(row.get("id"))
            if len(bad_rows) >= 10:
                break
        if row.get("metadata", {}).get("weak_gate_rows_used_for_training") is not False:
            bad_rows.append(row.get("id"))
            if len(bad_rows) >= 10:
                break
    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "unique_prompts": len(set(prompts)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "bad_rows_first10": bad_rows[:10],
    }


def main() -> int:
    print("=== V416 RAWSTYLE TRANSFER DATASET START ===", flush=True)
    for path in [V410_TRAIN, V410_VAL, V410_MANIFEST]:
        print("input_path =", path, "exists =", path.is_file(), flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    train = convert_rows(read_jsonl(V410_TRAIN), "train")
    val = convert_rows(read_jsonl(V410_VAL), "validation")
    train_ids = {row["id"] for row in train}
    val_ids = {row["id"] for row in val}
    if train_ids & val_ids:
        raise RuntimeError("train/val id overlap")
    if {row["prompt"] for row in train} & {row["prompt"] for row in val}:
        raise RuntimeError("train/val prompt overlap")

    train_summary = audit(train, "train")
    val_summary = audit(val, "validation")
    if train_summary["duplicate_ids"] or train_summary["duplicate_prompts"] or val_summary["duplicate_ids"] or val_summary["duplicate_prompts"]:
        raise RuntimeError("duplicate ids/prompts in V416 dataset")
    if train_summary["bad_rows_first10"] or val_summary["bad_rows_first10"]:
        raise RuntimeError("bad rows in V416 dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "v416_rawstyle_transfer_train.jsonl"
    val_path = OUT_DIR / "v416_rawstyle_transfer_val.jsonl"
    manifest_path = OUT_DIR / "v416_rawstyle_transfer_manifest.json"
    report_path = OUT_DIR / "V416_VS_PREVIOUS.md"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    v410_manifest = json.loads(V410_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "kg1_v416_rawstyle_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "source_policy": {
            "weak_or_full_gate_rows_used_for_training": False,
            "v414_rows_used_as_training": False,
            "v414_rules_used_as_motivation_only": True,
            "adapter_training_authorization": "blocked_until_tokenization_gate_and_debug_launch",
        },
        "previous_dataset": {
            "version": "V410",
            "manifest_json": str(V410_MANIFEST.relative_to(REPO_ROOT)),
            "train_sha256": v410_manifest["outputs"]["train_sha256"],
            "val_sha256": v410_manifest["outputs"]["val_sha256"],
            "failure_evidence": "V413 weak eval checkpoint-2=190/315, equation=56/155, bit=134/160, truncated=1.",
        },
        "material_difference_vs_v410": [
            "same leak-safe synthetic prompts and labels",
            "assistant completion rewritten to raw-output-style boxed suffix",
            "no weak/full gate rows used directly",
            "intended to test V370 format-mismatch finding without changing source labels",
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "outputs": {
            "train_jsonl": str(train_path.relative_to(REPO_ROOT)),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path.relative_to(REPO_ROOT)),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path.relative_to(REPO_ROOT)),
            "report_md": str(report_path.relative_to(REPO_ROOT)),
        },
        "promotion_gate": {
            "weak_total_min_exclusive": 192,
            "equation_transform_min_exclusive": 56,
            "bit_manipulation_min": 136,
            "truncated_max": 0,
            "first_checkpoint_kill_switch": True,
        },
    }
    write_json(manifest_path, manifest)

    report = [
        "# V416 Rawstyle Transfer Dataset",
        "",
        "| Item | V410 | V416 |",
        "|---|---:|---:|",
        f"| Train rows | `{v410_manifest['train_summary']['rows']}` | `{train_summary['rows']}` |",
        f"| Val rows | `{v410_manifest['validation_summary']['rows']}` | `{val_summary['rows']}` |",
        "| Weak/full rows used for train | `0` | `0` |",
        "| Completion style | `Rule / Final answer` | `raw-output-style boxed suffix` |",
        "| Prior transfer result | V413 failed: `190/315`, eq `56`, bit `134` | not launched yet |",
        "",
        "V416 exists only to test the V370/V415 finding that solver teachers are not transferring into the adapter. It is not a submit artifact.",
        "",
        "HF/Kaggle GPU remains blocked until tokenization and debug gates pass.",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("train_path =", train_path, "sha256 =", sha256_file(train_path), flush=True)
    print("val_path =", val_path, "sha256 =", sha256_file(val_path), flush=True)
    print("manifest_path =", manifest_path, flush=True)
    print("=== V416 RAWSTYLE TRANSFER DATASET END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

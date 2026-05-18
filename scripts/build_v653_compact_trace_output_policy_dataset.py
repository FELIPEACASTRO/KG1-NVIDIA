#!/usr/bin/env python3
"""Build V653 compact trace/output-policy dataset from V643.

V652 is blocked because V613 is answer-only and fails V513 learnability.
V653 keeps the V643 traceable source mix, but compresses long bit traces so the
adapter still sees rule terms and a boxed suffix without learning runaway
outputs.  No weak/full gate rows are introduced.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_V643_MANIFEST = (
    ROOT
    / "artifacts/v643_v641_plus_v367_bit_signal_mix/20260518T_v643_cpu_gate/"
    / "v643_v641_plus_v367_bit_signal_mix_manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v653_compact_trace_output_policy_dataset/20260518T_v653_cpu_gate"

ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)
SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify it briefly, then end "
    "with exactly one final answer in \\boxed{}."
)
BIN8_RE = re.compile(r"\b[01]{8}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if (Path.cwd() / path).exists():
        return Path.cwd() / path
    return ROOT / path


def manifest_output_path(manifest: dict[str, Any], key: str) -> Path:
    value = (manifest.get("outputs") or {}).get(key)
    if not value:
        raise RuntimeError(f"manifest missing outputs.{key}")
    return resolve_path(value)


def assistant_text(row: dict[str, Any]) -> str:
    for item in reversed(row.get("messages") or []):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return str(item.get("content") or "")
    return ""


def family_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("family") or metadata.get("family") or "")


def subcategory_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("subcategory") or metadata.get("subcategory") or "unknown")


def answer_of(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None:
        raise RuntimeError(f"row {row.get('id')} missing answer")
    return str(answer)


def box_answer(answer: str) -> str:
    return r"\boxed{" + answer + "}"


def compact_rules(metadata: dict[str, Any]) -> str:
    expr = str(metadata.get("expr") or "").strip()
    if expr:
        return expr
    rule_slug = str(metadata.get("rule_slug") or "").strip()
    if rule_slug:
        return rule_slug
    selected = metadata.get("v571_selected_rules")
    if isinstance(selected, list) and selected:
        return ", ".join(str(item) for item in selected[:8])
    rule = str(metadata.get("huikang_rule") or metadata.get("rule") or "").strip()
    if rule:
        return rule
    group = str(metadata.get("huikang_rule_group") or "").strip()
    if group:
        return group
    return "bit relation terms from the examples"


def query_bits_words(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    matches = BIN8_RE.findall(prompt)
    query = matches[-1] if matches else ""
    if not query:
        return "unknown-query"
    return "-".join("one" if char == "1" else "zero" for char in query)


def compact_assistant(row: dict[str, Any]) -> str:
    answer = answer_of(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    family = family_of(row)
    original = assistant_text(row).strip()
    if family != "bit_manipulation":
        return original
    rules = compact_rules(metadata)
    query_pattern = query_bits_words(row)
    return (
        f"Rule: bit trace terms {rules}.\n"
        f"Query pattern: {query_pattern}.\n"
        "Check: apply the same bit terms to the query and keep one 8-bit output.\n"
        f"Result: {answer}.\n"
        f"Final answer: {box_answer(answer)}"
    )


def project_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    compressed = 0
    for idx, row in enumerate(rows):
        new_row = copy.deepcopy(row)
        original_id = str(row.get("id") or f"row_{idx:06d}")
        family = family_of(row)
        assistant = compact_assistant(row)
        if assistant != assistant_text(row).strip():
            compressed += 1
        answer = answer_of(row)
        extracted = extract_final_answer(assistant)
        if not verify_answer(answer, extracted):
            raise RuntimeError(
                f"{split}:{original_id}: compact assistant mismatch answer={answer!r} extracted={extracted!r}"
            )
        metadata = new_row.get("metadata") if isinstance(new_row.get("metadata"), dict) else {}
        messages = new_row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"{split}:{original_id}: missing messages")
        prompt_contract = str(metadata.get("prompt_contract", "legacy_system"))
        if prompt_contract != "official_like":
            system_seen = False
            for item in messages:
                if isinstance(item, dict) and item.get("role") == "system":
                    item["content"] = SYSTEM_PROMPT
                    system_seen = True
                    break
            if not system_seen:
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "assistant":
                item["content"] = assistant
                break
        else:
            raise RuntimeError(f"{split}:{original_id}: missing assistant message")
        metadata = dict(metadata)
        for flag in ANTI_LEAK_FLAGS:
            if metadata.get(flag) is True:
                raise RuntimeError(f"{split}:{original_id}: anti-leak flag {flag}=true")
            metadata[flag] = False
        metadata.update(
            {
                "schema_version": "kg1_v653_compact_trace_output_policy_dataset_v1",
                "source": "v653_compact_trace_output_policy_dataset",
                "source_dataset": "v653_compact_trace_output_policy_dataset",
                "v653_original_id": original_id,
                "v653_original_source": row.get("source") or metadata.get("source"),
                "v653_original_subcategory": subcategory_of(row),
                "v653_split": split,
                "v653_compacted_bit_trace": family == "bit_manipulation",
            }
        )
        new_row["metadata"] = metadata
        new_row["id"] = f"v653_{split}_{idx:05d}_{sha256_text(original_id)[:10]}"
        new_row["family"] = family
        new_row["subcategory"] = subcategory_of(row)
        new_row["source"] = "v653_compact_trace_output_policy_dataset"
        new_row["source_dataset"] = "v653_compact_trace_output_policy_dataset"
        projected.append(new_row)
    return projected, {"compressed_bit_rows": compressed}


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    family_counts = Counter(family_of(row) for row in rows)
    subcategory_counts = Counter(subcategory_of(row) for row in rows)
    words_by_family: dict[str, list[int]] = {}
    chars_by_family: dict[str, list[int]] = {}
    compacted = 0
    for row in rows:
        family = family_of(row)
        text = assistant_text(row)
        words_by_family.setdefault(family, []).append(len(text.split()))
        chars_by_family.setdefault(family, []).append(len(text))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("v653_compacted_bit_trace"):
            compacted += 1

    def stats(values: list[int]) -> dict[str, int]:
        if not values:
            return {"min": 0, "p50": 0, "p95": 0, "max": 0}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "max": ordered[-1],
        }

    return {
        "split": split,
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "compacted_bit_rows": compacted,
        "assistant_word_stats": {key: stats(values) for key, values in sorted(words_by_family.items())},
        "assistant_char_stats": {key: stats(values) for key, values in sorted(chars_by_family.items())},
    }


def assert_no_train_val_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_prompts = {sha256_text(str(row.get("prompt") or "")) for row in train_rows}
    val_prompts = {sha256_text(str(row.get("prompt") or "")) for row in val_rows}
    overlap = train_prompts & val_prompts
    if overlap:
        raise RuntimeError(f"train/val prompt overlap: {len(overlap)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v643-manifest", type=Path, default=DEFAULT_V643_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = read_json(args.v643_manifest)
    train_in = manifest_output_path(manifest, "train_jsonl")
    val_in = manifest_output_path(manifest, "val_jsonl")
    train_rows, train_projection = project_rows(read_jsonl(train_in), "train")
    val_rows, val_projection = project_rows(read_jsonl(val_in), "val")
    assert_no_train_val_overlap(train_rows, val_rows)

    output_dir = args.output_dir
    train_out = output_dir / "v653_compact_trace_output_policy_train.jsonl"
    val_out = output_dir / "v653_compact_trace_output_policy_val.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    train_sha = sha256_file(train_out)
    val_sha = sha256_file(val_out)
    report = {
        "schema_version": "kg1_v653_compact_trace_output_policy_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v643_manifest": str(args.v643_manifest),
            "v643_manifest_sha256": sha256_file(args.v643_manifest),
            "v643_train_jsonl": str(train_in),
            "v643_train_sha256": sha256_file(train_in),
            "v643_val_jsonl": str(val_in),
            "v643_val_sha256": sha256_file(val_in),
        },
        "outputs": {
            "train_jsonl": str(train_out),
            "train_sha256": train_sha,
            "val_jsonl": str(val_out),
            "val_sha256": val_sha,
            "manifest_json": str(output_dir / "v653_compact_trace_output_policy_manifest.json"),
            "summary_md": str(output_dir / "KG1_V653_COMPACT_TRACE_OUTPUT_POLICY_DATASET.md"),
        },
        "decision": {
            "status": "dataset_ready_for_cpu_gates",
            "gpu_allowed": False,
            "submit_allowed": False,
            "next_action": "Run V286, V513, objective/pre-paid gates before any paid job.",
        },
        "train_summary": summarize(train_rows, "train"),
        "validation_summary": summarize(val_rows, "validation"),
        "projection": {"train": train_projection, "validation": val_projection},
    }
    manifest_out = output_dir / "v653_compact_trace_output_policy_manifest.json"
    summary_out = output_dir / "KG1_V653_COMPACT_TRACE_OUTPUT_POLICY_DATASET.md"
    write_json(manifest_out, report)
    summary = [
        "# KG1 V653 Compact Trace Output Policy Dataset",
        "",
        "## Decision",
        "",
        "- Status: `dataset_ready_for_cpu_gates`",
        "- GPU allowed: `False`",
        "- Submit allowed: `False`",
        "",
        "## Counts",
        "",
        f"- Train rows: `{len(train_rows)}`",
        f"- Validation rows: `{len(val_rows)}`",
        f"- Train SHA: `{train_sha}`",
        f"- Validation SHA: `{val_sha}`",
        f"- Train summary: `{json.dumps(report['train_summary'], sort_keys=True)}`",
        f"- Validation summary: `{json.dumps(report['validation_summary'], sort_keys=True)}`",
        "",
        "## Required Gates",
        "",
        "- V286 tokenization with `boxed_suffix` or `submit_safe_suffix`.",
        "- V513 learnability must be `passed_cpu_structure_only` with 0 blockers/warnings.",
        "- Objective/pre-paid gate must include V513 manifest.",
        "- No package or submit before adapter-only weak gain.",
    ]
    summary_out.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

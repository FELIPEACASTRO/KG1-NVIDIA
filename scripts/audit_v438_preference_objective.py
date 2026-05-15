#!/usr/bin/env python3
"""Audit V435E preference pairs before another GPU preference job.

This is a CPU-only structural audit. It does not score the model. The goal is to
catch objective/data issues that can make a chosen/rejected mean-NLL objective
move in the wrong direction even when the labels are semantically correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import answers_equivalent, extract_boxed_answers  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            row["_path"] = str(path)
            rows.append(row)
    return rows


def boxed_values(text: str) -> list[str]:
    return [value.strip() for value in extract_boxed_answers(text or "")]


def last_boxed(text: str) -> str:
    values = boxed_values(text)
    return values[-1] if values else ""


def text_without_boxed_payloads(text: str) -> str:
    value = text or ""
    starts = list(re.finditer(r"\\boxed\{", value))
    if not starts:
        return value
    chunks: list[str] = []
    cursor = 0
    for index, match in enumerate(starts):
        chunks.append(value[cursor : match.start()])
        segment_end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
        segment = value[match.end() : segment_end]
        last_brace = segment.rfind("}")
        if last_brace == -1:
            cursor = segment_end
        else:
            cursor = match.end() + last_brace + 1
        chunks.append("\\boxed{}")
    chunks.append(value[cursor:])
    return "".join(chunks)


def approx_tokens(text: str) -> int:
    # Cheap stable proxy: words, boxed answers, punctuation runs.
    return len(re.findall(r"\\boxed\{[^{}]*\}|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text or ""))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "rows": len(rows),
        "chosen_tokens_mean": round(mean(r["chosen_tokens"] for r in rows), 4),
        "rejected_tokens_mean": round(mean(r["rejected_tokens"] for r in rows), 4),
        "chosen_rejected_token_ratio_mean": round(mean(r["chosen_rejected_token_ratio"] for r in rows), 4),
        "rejected_much_shorter_rows": sum(r["rejected_much_shorter"] for r in rows),
        "chosen_mentions_adapter_prediction_rows": sum(r["chosen_mentions_adapter_prediction"] for r in rows),
        "chosen_mentions_public_train_label_audit_rows": sum(r["chosen_mentions_public_train_label_audit"] for r in rows),
        "answer_box_mismatch_rows": sum(not r["chosen_box_matches_answer"] for r in rows),
        "rejected_box_mismatch_rows": sum(not r["rejected_box_matches_adapter_prediction"] for r in rows),
    }


def audit_rows(split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        answer = str(metadata.get("answer") or row.get("answer") or "")
        adapter_prediction = str(metadata.get("adapter_prediction") or "")
        chosen_box = last_boxed(chosen)
        rejected_box = last_boxed(rejected)
        chosen_without_boxed = text_without_boxed_payloads(chosen)
        chosen_tokens = approx_tokens(chosen)
        rejected_tokens = approx_tokens(rejected)
        ratio = chosen_tokens / max(1, rejected_tokens)
        audited.append(
            {
                "split": split,
                "id": row.get("id", ""),
                "line_no": row.get("_line_no", ""),
                "family": row.get("family") or metadata.get("family", ""),
                "subcategory": row.get("subcategory") or metadata.get("rule_class", ""),
                "rule_class": metadata.get("rule_class", ""),
                "negative_type": metadata.get("negative_type", ""),
                "chosen_box": chosen_box,
                "rejected_box": rejected_box,
                "answer": answer,
                "adapter_prediction": adapter_prediction,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "chosen_rejected_token_ratio": round(ratio, 6),
                "rejected_much_shorter": rejected_tokens < 0.60 * chosen_tokens,
                "chosen_box_matches_answer": bool(answer)
                and answers_equivalent(answer, chosen_box, observed_is_boxed_payload=True),
                "rejected_box_matches_adapter_prediction": bool(adapter_prediction)
                and answers_equivalent(
                    adapter_prediction,
                    rejected_box,
                    observed_is_boxed_payload=True,
                ),
                "chosen_box_equals_adapter_prediction": bool(adapter_prediction)
                and answers_equivalent(
                    adapter_prediction,
                    chosen_box,
                    observed_is_boxed_payload=True,
                ),
                "chosen_mentions_adapter_prediction": bool(adapter_prediction)
                and adapter_prediction in chosen_without_boxed,
                "chosen_mentions_public_train_label_audit": "public-train label audit" in chosen,
                "chosen_box_count": len(boxed_values(chosen)),
                "rejected_box_count": len(boxed_values(rejected)),
                "prompt_sha256": metadata.get("prompt_sha256", ""),
                "source_id": metadata.get("source_id", ""),
            }
        )
    return audited


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v438_preference_objective_audit")
    args = parser.parse_args()

    print("=== V438 PREFERENCE OBJECTIVE AUDIT START ===", flush=True)
    print(f"train_jsonl = {args.train_jsonl}", flush=True)
    print(f"val_jsonl = {args.val_jsonl}", flush=True)
    print(f"output_dir = {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    audited = audit_rows("train", train_rows) + audit_rows("validation", val_rows)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audited:
        by_family[str(row["family"])].append(row)
        by_subcategory[str(row["subcategory"])].append(row)
        by_split[str(row["split"])].append(row)

    total = summarize_group(audited)
    family_summary = {k: summarize_group(v) for k, v in sorted(by_family.items())}
    subcategory_summary = {k: summarize_group(v) for k, v in sorted(by_subcategory.items())}
    split_summary = {k: summarize_group(v) for k, v in sorted(by_split.items())}

    negative_type_counts = Counter(str(r["negative_type"]) for r in audited)
    decision_flags = {
        "answer_boxes_all_match": total.get("answer_box_mismatch_rows", 0) == 0,
        "rejected_boxes_all_match_adapter_prediction": total.get("rejected_box_mismatch_rows", 0) == 0,
        "format_negatives_absent": set(negative_type_counts) == {"hard_negative_adapter_exact_wrong"},
        "length_style_confound_majority": total.get("rejected_much_shorter_rows", 0)
        > (len(audited) / 2),
        "chosen_leaks_adapter_wrong_answer_text_majority": total.get(
            "chosen_mentions_adapter_prediction_rows", 0
        )
        > (len(audited) / 2),
        "chosen_template_mentions_label_audit_majority": total.get(
            "chosen_mentions_public_train_label_audit_rows", 0
        )
        > (len(audited) / 2),
    }
    hf_gpu_allowed_for_same_objective = (
        decision_flags["answer_boxes_all_match"]
        and decision_flags["rejected_boxes_all_match_adapter_prediction"]
        and decision_flags["format_negatives_absent"]
        and not decision_flags["length_style_confound_majority"]
        and not decision_flags["chosen_leaks_adapter_wrong_answer_text_majority"]
    )
    recommendation = (
        "block_mean_nll_preference_gpu; build equalized answer-only/contrastive-final-answer objective"
        if not hf_gpu_allowed_for_same_objective
        else "mean_nll_preference_structure_ok; require model-side CPU or tiny GPU orientation check"
    )

    detail_csv = args.output_dir / f"{args.label}_detail.csv"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    write_csv(detail_csv, audited)
    manifest = {
        "schema_version": "kg1_v438_preference_objective_audit_v1",
        "label": args.label,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "rows": len(audited),
        "negative_type_counts": dict(negative_type_counts),
        "total_summary": total,
        "split_summary": split_summary,
        "family_summary": family_summary,
        "subcategory_summary": subcategory_summary,
        "decision_flags": decision_flags,
        "hf_gpu_allowed_for_same_objective": hf_gpu_allowed_for_same_objective,
        "recommendation": recommendation,
        "outputs": {
            "detail_csv": str(detail_csv),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("total_summary =", json.dumps(total, sort_keys=True), flush=True)
    print("family_summary =", json.dumps(family_summary, sort_keys=True), flush=True)
    print("decision_flags =", json.dumps(decision_flags, sort_keys=True), flush=True)
    print("hf_gpu_allowed_for_same_objective =", hf_gpu_allowed_for_same_objective, flush=True)
    print("recommendation =", recommendation, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V438 PREFERENCE OBJECTIVE AUDIT END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the V335 mixed trace/replay dataset.

V335 merges the best CPU-gated training material before any new HF spend:

* V304 broad solver-trace replay with bit protection.
* V325 numeric equation no-loss traces from V324.
* V330 symbolic cryptarithm traces from V329.

The builder normalizes every assistant completion to the same boxed final-answer
suffix so the generic tokenization gate can validate a single contract.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPONENTS = [
    (
        "v304_solver_trace",
        REPO_ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_train.jsonl",
        REPO_ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z/v304_solver_trace_distill_val.jsonl",
    ),
    (
        "v325_equation_numeric_no_loss",
        REPO_ROOT / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/v325_equation_no_loss_distill_sft_train.jsonl",
        REPO_ROOT / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate/v325_equation_no_loss_distill_sft_val.jsonl",
    ),
    (
        "v330_symbolic_cryptarithm_no_loss",
        REPO_ROOT / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/v330_symbolic_cryptarithm_distill_sft_train.jsonl",
        REPO_ROOT / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/v330_symbolic_cryptarithm_distill_sft_val.jsonl",
    ),
]

DEFAULT_REFERENCE_CSVS = [
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv",
]


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


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def prompt_sha(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_reference_csv(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompts: set[str] = set()
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "ids": ids, "prompt_hashes": prompts}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ids.add(str(row.get("id", "")).strip())
            prompts.add(sha256_text(normalize_prompt(row.get("prompt", ""))))
    return {"path": str(path), "exists": True, "rows": len(ids), "ids": ids, "prompt_hashes": prompts}


def boxed_final_line(answer: Any) -> str:
    return "Final answer: " + r"\boxed{" + str(answer) + "}"


def normalize_assistant_content(content: str, answer: str) -> tuple[str, str]:
    boxed = boxed_final_line(answer)
    text = str(content or "").rstrip()
    pattern = r"Final answer:\s*(?:\\boxed\{.*\}|[^\n]+)\s*$"
    if re.search(pattern, text, flags=re.S):
        updated = re.sub(pattern, lambda _match: boxed, text, count=1, flags=re.S)
        status = "replaced_final_answer_suffix"
    else:
        updated = text + "\n" + boxed if text else boxed
        status = "appended_final_answer_suffix"
    return updated, status


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
    metadata["v335_component"] = component
    metadata["v335_normalized_final_answer"] = "boxed_suffix"
    metadata["v335_normalization_status"] = status
    metadata["split"] = split
    out["metadata"] = metadata
    out["source"] = str(out.get("source") or metadata.get("source_dataset") or component)
    return out, status


def merge_split(components: list[tuple[str, Path, Path]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = 1 if split == "train" else 2
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    merged: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "input_rows": Counter(),
        "kept_rows": Counter(),
        "duplicate_ids": Counter(),
        "duplicate_prompts": Counter(),
        "normalization": Counter(),
    }
    for component in components:
        name = component[0]
        path = component[index]
        rows = read_jsonl(path)
        stats["input_rows"][name] += len(rows)
        for row in rows:
            normalized, status = normalize_row(row, name, split)
            rid = str(normalized.get("id", ""))
            psha = prompt_sha(normalized)
            if rid in seen_ids:
                stats["duplicate_ids"][name] += 1
                continue
            if psha in seen_prompts:
                stats["duplicate_prompts"][name] += 1
                continue
            seen_ids.add(rid)
            seen_prompts.add(psha)
            stats["normalization"][status] += 1
            stats["kept_rows"][name] += 1
            merged.append(normalized)
    return merged, {
        "input_rows": dict(stats["input_rows"]),
        "kept_rows": dict(stats["kept_rows"]),
        "duplicate_ids": dict(stats["duplicate_ids"]),
        "duplicate_prompts": dict(stats["duplicate_prompts"]),
        "normalization": dict(stats["normalization"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    components: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        families[str(row.get("family", ""))] += 1
        subcategories[str(row.get("subcategory", metadata.get("subcategory", "")))] += 1
        sources[str(metadata.get("source_dataset", row.get("source", "")))] += 1
        components[str(metadata.get("v335_component", ""))] += 1
    return {
        "rows": len(rows),
        "unique_ids": len({str(row.get("id", "")) for row in rows}),
        "unique_prompt_hashes": len({prompt_sha(row) for row in rows}),
        "family_counts": dict(sorted(families.items())),
        "subcategory_counts_top40": dict(subcategories.most_common(40)),
        "source_counts_top40": dict(sources.most_common(40)),
        "component_counts": dict(sorted(components.items())),
    }


def overlap_report(rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_ids = {str(row.get("id", "")) for row in rows}
    row_prompts = {prompt_sha(row) for row in rows}
    reports: list[dict[str, Any]] = []
    for ref in references:
        ids = ref["ids"]
        prompts = ref["prompt_hashes"]
        reports.append(
            {
                "path": ref["path"],
                "exists": ref["exists"],
                "reference_rows": ref["rows"],
                "id_overlap_count": len(row_ids & ids),
                "prompt_overlap_count": len(row_prompts & prompts),
                "id_overlap_sample": sorted(row_ids & ids)[:10],
                "prompt_overlap_sample": sorted(row_prompts & prompts)[:10],
            }
        )
    return reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V335 MIXED TRACE REPLAY DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    components = DEFAULT_COMPONENTS
    for name, train, val in components:
        print(f"component = {name} train={train} val={val}", flush=True)
        if not train.exists() or not val.exists():
            raise FileNotFoundError(f"missing component files for {name}")

    train_rows, train_merge = merge_split(components, "train")
    val_rows, val_merge = merge_split(components, "validation")
    references = [read_reference_csv(path) for path in DEFAULT_REFERENCE_CSVS]
    train_overlap = overlap_report(train_rows, references)
    val_overlap = overlap_report(val_rows, references)
    bad_overlap = [
        item
        for item in train_overlap + val_overlap
        if int(item["id_overlap_count"]) > 0 or int(item["prompt_overlap_count"]) > 0
    ]
    if bad_overlap:
        raise RuntimeError("reference overlap detected: " + json.dumps(bad_overlap, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_train.jsonl"
    val_path = args.output_dir / f"{args.label}_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "schema_version": "kg1_v335_mixed_trace_replay_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "components": [
            {"name": name, "train_jsonl": str(train), "val_jsonl": str(val)}
            for name, train, val in components
        ],
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "merge": {"train": train_merge, "validation": val_merge},
        "validation": {"train": summarize(train_rows), "validation": summarize(val_rows)},
        "reference_overlap": {"train": train_overlap, "validation": val_overlap},
        "training_authorization": "blocked_until_v286_boxed_suffix_tokenization_gate_and_weak_no_regression_smoke",
        "required_next_gate": [
            "scripts/run_v286_generic_tokenization_gate.py --assistant-final-answer-mode boxed_suffix",
            "HF smoke only with first-checkpoint weak kill-switch: total>192, equation>56, bit>=136",
            "no full eval or submit until weak gate improves over V259/V290 adapter-only baseline",
        ],
    }
    write_json(manifest_path, manifest)
    print("train_rows =", len(train_rows), flush=True)
    print("validation_rows =", len(val_rows), flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V335 MIXED TRACE REPLAY DATASET END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v335_mixed_trace_replay_dataset" / utc_compact(),
    )
    parser.add_argument("--label", default="v335_mixed_trace_replay")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

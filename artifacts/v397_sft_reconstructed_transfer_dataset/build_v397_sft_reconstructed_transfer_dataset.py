#!/usr/bin/env python3
"""Build V397 adapter-transfer dataset from local reconstructed SFT traces.

This is a CPU-only gate artifact. It reads the local public-train reconstructed
SFT JSONL, verifies every final boxed answer against the local competition
train labels, excludes all weak315 row ids from training/validation, and emits
a focused bit/equation chat dataset candidate. It does not launch HF, train,
package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for item in (REPO_ROOT, REPO_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import extract_final_answer, verify_answer  # noqa: E402


SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)

CATEGORY_TO_FAMILY = {
    "bit_manipulation": "bit_manipulation",
    "cryptarithm_deduce": "equation_transform",
    "cryptarithm_guess": "equation_transform",
    "equation_numeric_deduce": "equation_transform",
    "equation_numeric_guess": "equation_transform",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_for_validation(problem_id: str, val_mod: int) -> str:
    bucket = int(hashlib.sha256(problem_id.encode("utf-8")).hexdigest()[:8], 16) % val_mod
    return "validation" if bucket == 0 else "train"


def load_train_labels(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for row in read_csv(path):
        rows[str(row["id"])] = {
            "prompt": str(row["prompt"]),
            "answer": str(row["answer"]).strip(),
        }
    return rows


def assistant_with_boxed_suffix(original: str, answer: str) -> str:
    suffix = r"Final answer: \boxed{" + answer + "}"
    text = str(original or "").rstrip()
    if text.endswith(suffix):
        return text
    return text + "\n" + suffix


def build_row(
    source_obj: dict[str, Any],
    train_row: dict[str, str],
    *,
    split: str,
    max_assistant_chars: int,
) -> dict[str, Any]:
    metadata = dict(source_obj.get("_metadata", {}))
    problem_id = str(metadata.get("problem_id", ""))
    category = str(metadata.get("category", ""))
    family = CATEGORY_TO_FAMILY[category]
    answer = str(train_row["answer"]).strip()
    original_assistant = str(source_obj["messages"][-1]["content"])
    if len(original_assistant) > max_assistant_chars:
        # Keep the final answer target exact while preventing extreme long-tail
        # rows from dominating the first transfer smoke.
        original_assistant = original_assistant[-max_assistant_chars:]
    assistant = assistant_with_boxed_suffix(original_assistant, answer)
    prompt = str(train_row["prompt"])
    row_id = "v397_" + split + "_" + problem_id
    metadata.update(
        {
            "schema_version": "kg1_v397_sft_reconstructed_transfer_dataset_v1",
            "source_dataset": "local_sft_reconstructed_jsonl",
            "source_problem_id": problem_id,
            "source_category": category,
            "source_status": str(metadata.get("status", "")),
            "weak_gate_rows_used_for_training": False,
            "split": split,
            "prompt_sha256": sha256_text(normalize_prompt(prompt)),
            "assistant_char_count": len(assistant),
            "max_assistant_chars_applied": max_assistant_chars,
        }
    )
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "subcategory": category,
        "source": "v397_sft_reconstructed_transfer",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row["family"]) for row in rows)
    subcategory_counts = Counter(str(row["subcategory"]) for row in rows)
    status_counts = Counter(str(row["metadata"].get("source_status", "")) for row in rows)
    assistant_chars = [int(row["metadata"]["assistant_char_count"]) for row in rows]
    return {
        "rows": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "unique_source_problem_ids": len({row["metadata"]["source_problem_id"] for row in rows}),
        "prompt_hash_count": len({row["metadata"]["prompt_sha256"] for row in rows}),
        "family_counts": dict(sorted(family_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "source_status_counts": dict(sorted(status_counts.items())),
        "assistant_char_min": min(assistant_chars) if assistant_chars else 0,
        "assistant_char_max": max(assistant_chars) if assistant_chars else 0,
        "assistant_char_mean": sum(assistant_chars) / len(assistant_chars) if assistant_chars else 0.0,
        "boxed_suffix_rows": sum(
            str(row["messages"][-1]["content"]).rstrip().endswith(
                r"Final answer: \boxed{" + str(row["answer"]) + "}"
            )
            for row in rows
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V397 SFT RECONSTRUCTED TRANSFER DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("sft_reconstructed_jsonl =", args.sft_reconstructed_jsonl, flush=True)
    print("competition_train_csv =", args.competition_train_csv, flush=True)
    print("weak_csv =", args.weak_csv, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("include_statuses =", ",".join(args.include_statuses), flush=True)
    print("val_mod =", args.val_mod, flush=True)
    print("max_assistant_chars =", args.max_assistant_chars, flush=True)

    for path in [args.sft_reconstructed_jsonl, args.competition_train_csv, args.weak_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_labels = load_train_labels(args.competition_train_csv)
    weak_ids = {str(row["id"]) for row in read_csv(args.weak_csv)}
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    skipped = Counter()
    source_counts = Counter()
    verified_final_counts = Counter()

    with args.sft_reconstructed_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            metadata = obj.get("_metadata", {})
            problem_id = str(metadata.get("problem_id", ""))
            category = str(metadata.get("category", ""))
            status = str(metadata.get("status", ""))
            source_counts[(category, status)] += 1
            if category not in CATEGORY_TO_FAMILY:
                skipped["non_target_category"] += 1
                continue
            if problem_id in weak_ids:
                skipped["weak_id_excluded"] += 1
                continue
            if problem_id not in train_labels:
                skipped["missing_train_label"] += 1
                continue
            if status not in args.include_statuses:
                skipped["status_excluded"] += 1
                continue
            assistant = str(obj.get("messages", [{}])[-1].get("content", ""))
            final = extract_final_answer(assistant)
            answer = train_labels[problem_id]["answer"]
            if not verify_answer(answer, final):
                skipped["final_answer_mismatch"] += 1
                verified_final_counts[(category, status, "bad")] += 1
                continue
            verified_final_counts[(category, status, "ok")] += 1
            split = split_for_validation(problem_id, args.val_mod)
            row = build_row(obj, train_labels[problem_id], split=split, max_assistant_chars=args.max_assistant_chars)
            if split == "validation":
                val_rows.append(row)
            else:
                train_rows.append(row)
            if (line_no % 1000) == 0:
                print(f"sft_scan_progress = {line_no}", flush=True)

    train_prompt_hashes = {row["metadata"]["prompt_sha256"] for row in train_rows}
    val_prompt_hashes = {row["metadata"]["prompt_sha256"] for row in val_rows}
    prompt_overlap = train_prompt_hashes & val_prompt_hashes
    if prompt_overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(prompt_overlap)}")
    source_ids = {row["metadata"]["source_problem_id"] for row in train_rows + val_rows}
    weak_overlap = source_ids & weak_ids
    if weak_overlap:
        raise RuntimeError(f"weak ids leaked into V397 dataset: {sorted(weak_overlap)[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "v397_sft_reconstructed_transfer_train.jsonl"
    val_out = args.output_dir / "v397_sft_reconstructed_transfer_val.jsonl"
    manifest_out = args.output_dir / "v397_sft_reconstructed_transfer_manifest.json"
    comparison_out = args.output_dir / "V397_VS_PREVIOUS.md"

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    train_summary = summarize(train_rows)
    val_summary = summarize(val_rows)
    manifest = {
        "schema_version": "kg1_v397_sft_reconstructed_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "sft_reconstructed_jsonl": str(args.sft_reconstructed_jsonl),
            "sft_reconstructed_sha256": sha256_file(args.sft_reconstructed_jsonl),
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
            "weak_csv": str(args.weak_csv),
            "weak_csv_sha256": sha256_file(args.weak_csv),
        },
        "policy": {
            "cpu_only": True,
            "hf_gpu_allowed": False,
            "kaggle_submit_allowed": False,
            "weak_gate_rows_used_for_training": False,
            "assistant_final_answer_mode": "boxed_suffix",
            "next_required_gate": "scripts/run_v286_generic_tokenization_gate.py",
        },
        "filters": {
            "target_categories": sorted(CATEGORY_TO_FAMILY),
            "include_statuses": list(args.include_statuses),
            "val_mod": args.val_mod,
            "max_assistant_chars": args.max_assistant_chars,
            "skipped_counts": dict(sorted(skipped.items())),
            "source_counts": {"/".join(key): value for key, value in sorted(source_counts.items())},
            "verified_final_counts": {"/".join(key): value for key, value in sorted(verified_final_counts.items())},
        },
        "validation": {
            "train": train_summary,
            "validation": val_summary,
            "train_val_prompt_overlap": len(prompt_overlap),
            "weak_id_overlap": len(weak_overlap),
        },
        "outputs": {
            "train_jsonl": str(train_out),
            "train_sha256": sha256_file(train_out),
            "val_jsonl": str(val_out),
            "val_sha256": sha256_file(val_out),
            "manifest_json": str(manifest_out),
            "comparison_md": str(comparison_out),
        },
        "decision": {
            "decision": "v397_dataset_ready_for_tokenization_gate",
            "reason": (
                f"train={train_summary['rows']}; val={val_summary['rows']}; "
                f"weak_overlap=0; train_val_prompt_overlap=0"
            ),
            "next_action": "Run V286 tokenization gate with boxed_suffix. Launch HF only if tokenization has zero completion truncation.",
        },
    }
    write_json(manifest_out, manifest)
    comparison_out.write_text(
        "\n".join(
            [
                "# V397 vs Previous",
                "",
                "| Metric | Previous active state | V397 candidate | Decision |",
                "|---|---:|---:|---|",
                "| Best adapter-only weak | `192/315` | not trained | preserve as baseline |",
                "| equation_transform weak | `56/155` | not trained | tokenization gate first |",
                "| bit_manipulation weak | `136/160` | not trained | tokenization gate first |",
                f"| Train rows | n/a | `{train_summary['rows']}` | new transfer corpus |",
                f"| Validation rows | n/a | `{val_summary['rows']}` | new transfer corpus |",
                f"| Weak row overlap | must be `0` | `{len(weak_overlap)}` | pass |",
                f"| Train/val prompt overlap | must be `0` | `{len(prompt_overlap)}` | pass |",
                "",
                "V397 is not a submit candidate. It is a CPU-gated dataset candidate built from local reconstructed public-train traces.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("val_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("skipped_counts =", json.dumps(dict(sorted(skipped.items())), sort_keys=True), flush=True)
    print("manifest_json =", manifest_out, flush=True)
    print("comparison_md =", comparison_out, flush=True)
    print("=== V397 SFT RECONSTRUCTED TRANSFER DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-reconstructed-jsonl", type=Path, default=Path(r"C:\Users\davis\Downloads\sft_reconstructed.jsonl"))
    parser.add_argument("--competition-train-csv", type=Path, default=Path(r"C:\Users\davis\Downloads\competition_train.csv"))
    parser.add_argument(
        "--weak-csv",
        type=Path,
        default=REPO_ROOT
        / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
        / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/v397_sft_reconstructed_transfer_dataset/20260514T_cpu_gate")
    parser.add_argument("--include-statuses", nargs="+", default=["rule_found", "hypothesis_formed", "rule_unknown"])
    parser.add_argument("--val-mod", type=int, default=10)
    parser.add_argument("--max-assistant-chars", type=int, default=14000)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

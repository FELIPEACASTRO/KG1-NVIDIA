#!/usr/bin/env python3
"""Build V435B prompt-only adapter probe pack.

The pack contains permitted public-train prompts for collecting V291/V290 raw
outputs later. It deliberately excludes answers and excludes weak/full rows by
id and normalized prompt hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from competition_utils import classify_puzzle  # noqa: E402


DEFAULT_TRAIN_CSV = Path(os.environ.get("KG1_COMPETITION_TRAIN_CSV", r"C:\Users\davis\Downloads\competition_train.csv"))
DEFAULT_REFERENCE_WEAK_CSV = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_REFERENCE_FULL_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v435b_adapter_probe_prompt_pack"

OUTPUT_COLUMNS = ["id", "family", "prompt_sha256", "prompt_normalized_sha256", "prompt"]


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


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def reference_set(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            ids.add(rid)
        prompt = str(row.get("prompt") or row.get("generated_prompt") or "")
        if prompt:
            prompt_hashes.add(sha256_text(normalize_prompt(prompt)))
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows), "ids": ids, "prompt_hashes": prompt_hashes}


def select_rows(
    train_rows: list[dict[str, str]],
    *,
    reference_ids: set[str],
    reference_prompt_hashes: set[str],
    max_equation: int,
    max_bit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    family_seen = Counter()
    family_selected = Counter()
    caps = {"equation_transform": max_equation, "bit_manipulation": max_bit}

    for row in sorted(train_rows, key=lambda item: str(item.get("id", ""))):
        rid = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", ""))
        family = classify_puzzle(prompt)
        if family not in caps:
            continue
        family_seen[family] += 1
        prompt_sha = sha256_text(prompt.replace("\r\n", "\n"))
        prompt_norm_sha = sha256_text(normalize_prompt(prompt))
        if rid in reference_ids:
            skipped["reference_id_overlap"] += 1
            continue
        if prompt_norm_sha in reference_prompt_hashes:
            skipped["reference_prompt_overlap"] += 1
            continue
        if family_selected[family] >= caps[family]:
            skipped[f"{family}_cap"] += 1
            continue
        selected.append(
            {
                "id": rid,
                "family": family,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "prompt_normalized_sha256": prompt_norm_sha,
            }
        )
        family_selected[family] += 1

    summary = {
        "family_seen": dict(sorted(family_seen.items())),
        "family_selected": dict(sorted(family_selected.items())),
        "skipped": dict(sorted(skipped.items())),
        "selected_rows": len(selected),
    }
    return selected, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V435B ADAPTER PROBE PROMPT PACK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("competition_train_csv =", args.competition_train_csv, flush=True)
    print("reference_weak_csv =", args.reference_weak_csv, flush=True)
    print("reference_full_csv =", args.reference_full_csv, flush=True)
    print("max_equation =", args.max_equation, flush=True)
    print("max_bit =", args.max_bit, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    if not args.competition_train_csv.is_file():
        raise FileNotFoundError(args.competition_train_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weak_ref = reference_set(args.reference_weak_csv)
    full_ref = reference_set(args.reference_full_csv)
    train_rows = read_csv(args.competition_train_csv)
    selected, summary = select_rows(
        train_rows,
        reference_ids=set(weak_ref["ids"]) | set(full_ref["ids"]),
        reference_prompt_hashes=set(weak_ref["prompt_hashes"]) | set(full_ref["prompt_hashes"]),
        max_equation=args.max_equation,
        max_bit=args.max_bit,
    )

    csv_path = args.output_dir / f"{args.label}_prompts.csv"
    jsonl_path = args.output_dir / f"{args.label}_prompts.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_csv(csv_path, selected)
    write_jsonl(jsonl_path, selected)
    manifest = {
        "schema_version": "kg1_v435b_adapter_probe_prompt_pack_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "answers_exported": False,
            "weak_or_full_rows_exported": False,
            "purpose": "Collect V291/V290 raw outputs on permitted prompts before V436.",
        },
        "inputs": {
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
            "reference_weak_csv": str(args.reference_weak_csv),
            "reference_weak_sha256": weak_ref["sha256"],
            "reference_full_csv": str(args.reference_full_csv),
            "reference_full_sha256": full_ref["sha256"],
        },
        "selection": {
            "max_equation": args.max_equation,
            "max_bit": args.max_bit,
            **summary,
        },
        "outputs": {
            "prompts_csv": str(csv_path),
            "prompts_csv_sha256": sha256_file(csv_path),
            "prompts_jsonl": str(jsonl_path),
            "prompts_jsonl_sha256": sha256_file(jsonl_path),
            "manifest_json": str(manifest_path),
        },
        "next_action": "Run V291/V290 adapter inference on this prompt-only pack; then rerun V435 with real raw-output hard negatives.",
    }
    write_json(manifest_path, manifest)
    print("selection_summary =", json.dumps(summary, sort_keys=True), flush=True)
    print("prompts_jsonl =", jsonl_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("=== V435B ADAPTER PROBE PROMPT PACK END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--reference-weak-csv", type=Path, default=DEFAULT_REFERENCE_WEAK_CSV)
    parser.add_argument("--reference-full-csv", type=Path, default=DEFAULT_REFERENCE_FULL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v435b_adapter_probe_prompt_pack")
    parser.add_argument("--max-equation", type=int, default=600)
    parser.add_argument("--max-bit", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

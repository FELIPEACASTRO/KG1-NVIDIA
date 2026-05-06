#!/usr/bin/env python
"""Create a deterministic train/val split for the V214 replay candidate.

The split is for training loss diagnostics only. Promotion still depends on
the protected V194 solve-rate gates, not this internal validation slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def prompt_answer_signature(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "prompt": str(row.get("prompt") or "").strip(),
            "answer": str(row.get("answer") or "").strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def split_rows(rows: list[dict[str, Any]], seed: int, val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family") or "unknown")
        by_family[family].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for family in sorted(by_family):
        family_rows = sorted(by_family[family], key=lambda item: str(item.get("id") or ""))
        rng.shuffle(family_rows)
        val_count = max(1, round(len(family_rows) * val_fraction))
        val_rows.extend(family_rows[:val_count])
        train_rows.extend(family_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def add_split_metadata(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        metadata = dict(copied.get("metadata") or {})
        metadata["v214_split"] = split
        metadata["v214_split_note"] = "internal_loss_diagnostic_only"
        copied["metadata"] = metadata
        out.append(copied)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/v214/v214_micro_replay_candidate.jsonl")
    parser.add_argument("--output-dir", default="data/v214")
    parser.add_argument("--seed", type=int, default=214)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    train_path = output_dir / "v214_micro_train.jsonl"
    val_path = output_dir / "v214_micro_val.jsonl"
    manifest_path = output_dir / "v214_micro_split_manifest.json"

    print("[split_v214_micro_replay_candidate] START")
    print(f"[split_v214_micro_replay_candidate] input={input_path}")
    print(f"[split_v214_micro_replay_candidate] output_dir={output_dir}")
    print(f"[split_v214_micro_replay_candidate] seed={args.seed}")
    print(f"[split_v214_micro_replay_candidate] val_fraction={args.val_fraction}")

    rows = load_jsonl(input_path)
    signatures = [prompt_answer_signature(row) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("Input contains duplicate prompt+answer signatures")

    train_rows, val_rows = split_rows(rows, args.seed, args.val_fraction)
    train_rows = add_split_metadata(train_rows, "train")
    val_rows = add_split_metadata(val_rows, "val")
    train_sigs = {prompt_answer_signature(row) for row in train_rows}
    val_sigs = {prompt_answer_signature(row) for row in val_rows}
    overlap = train_sigs & val_sigs
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)}")

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "v214_micro_split_v1",
        "generated_at_utc": utc_now(),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_family_counts": dict(sorted(Counter(str(row.get("family") or "unknown") for row in train_rows).items())),
        "val_family_counts": dict(sorted(Counter(str(row.get("family") or "unknown") for row in val_rows).items())),
        "train_val_prompt_answer_overlap": len(overlap),
        "decision": "internal_loss_split_only_not_submission_gate",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[split_v214_micro_replay_candidate] train_rows={len(train_rows)} train_path={train_path}")
    print(f"[split_v214_micro_replay_candidate] val_rows={len(val_rows)} val_path={val_path}")
    print(f"[split_v214_micro_replay_candidate] manifest={manifest_path}")
    print("[split_v214_micro_replay_candidate] SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

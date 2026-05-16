#!/usr/bin/env python3
"""Audit KG1 training objective alignment before paid GPU jobs.

This CPU-only gate checks a gap that tokenization gates do not catch: a dataset
can contain enough guardrail rows physically while weighted replacement makes
those rows nearly absent from the effective training objective.

The script intentionally does not train, evaluate a model, package, or submit.
It reads local JSONL data plus the exact source/subcategory weights that a HF
launcher will export, then reports physical and effective family shares.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_weight_map(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise RuntimeError(f"weight entry must be key=value: {item}")
        key = key.strip()
        if not key:
            raise RuntimeError(f"empty weight key in entry: {item}")
        if key in weights:
            raise RuntimeError(f"duplicate weight key: {key}")
        try:
            weight = float(value.strip())
        except ValueError as exc:
            raise RuntimeError(f"non-numeric weight value in entry: {item}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise RuntimeError(f"weight must be finite and positive: {item}")
        weights[key] = weight
    return weights


def resolve_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    value = outputs.get(key)
    if not value:
        return None
    return Path(str(value))


def item_weight(row: dict[str, Any], source_weights: dict[str, float], subcategory_weights: dict[str, float]) -> float:
    source = str(row.get("source", ""))
    subcategory = str(row.get("subcategory", ""))
    return source_weights.get(source, 1.0) * subcategory_weights.get(subcategory, 1.0)


def summarize_split(
    rows: list[dict[str, Any]],
    source_weights: dict[str, float],
    subcategory_weights: dict[str, float],
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    family_weight: dict[str, float] = defaultdict(float)
    source_weight: dict[str, float] = defaultdict(float)
    subcategory_weight: dict[str, float] = defaultdict(float)
    unknown_source_rows = 0
    unknown_subcategory_rows = 0
    total_weight = 0.0

    for row in rows:
        family = str(row.get("family", ""))
        source = str(row.get("source", ""))
        subcategory = str(row.get("subcategory", ""))
        family_counts[family] += 1
        source_counts[source] += 1
        subcategory_counts[subcategory] += 1
        if source_weights and source not in source_weights:
            unknown_source_rows += 1
        if subcategory_weights and subcategory not in subcategory_weights:
            unknown_subcategory_rows += 1
        weight = item_weight(row, source_weights, subcategory_weights)
        total_weight += weight
        family_weight[family] += weight
        source_weight[source] += weight
        subcategory_weight[subcategory] += weight

    def share_from_counts(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
        total = sum(counter.values())
        return {
            key: {"rows": int(value), "share": round(value / total, 6) if total else 0.0}
            for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        }

    def share_from_weights(values: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            key: {"weight": round(value, 6), "share": round(value / total_weight, 6) if total_weight else 0.0}
            for key, value in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        }

    return {
        "rows": len(rows),
        "physical_share_by_family": share_from_counts(family_counts),
        "physical_share_by_source": share_from_counts(source_counts),
        "physical_share_by_subcategory": share_from_counts(subcategory_counts),
        "effective_total_weight": round(total_weight, 6),
        "effective_share_by_family": share_from_weights(family_weight),
        "effective_share_by_source": share_from_weights(source_weight),
        "effective_share_by_subcategory": share_from_weights(subcategory_weight),
        "unknown_source_rows": int(unknown_source_rows),
        "unknown_subcategory_rows": int(unknown_subcategory_rows),
    }


def gate_findings(args: argparse.Namespace, report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    train = report["train"]
    val = report["validation"]
    train_family = train["effective_share_by_family"]
    bit_share = float(train_family.get(args.bit_family, {}).get("share", 0.0))
    equation_share = float(train_family.get(args.equation_family, {}).get("share", 0.0))
    max_family = max((float(item.get("share", 0.0)) for item in train_family.values()), default=0.0)
    if bit_share < args.min_bit_effective_share:
        findings.append(
            {
                "level": "error",
                "code": "bit_effective_share_below_floor",
                "detail": f"{bit_share:.6f} < {args.min_bit_effective_share:.6f}",
            }
        )
    if equation_share > args.max_equation_effective_share:
        findings.append(
            {
                "level": "error",
                "code": "equation_effective_share_above_ceiling",
                "detail": f"{equation_share:.6f} > {args.max_equation_effective_share:.6f}",
            }
        )
    if max_family > args.max_any_family_effective_share:
        findings.append(
            {
                "level": "error",
                "code": "one_family_dominates_effective_objective",
                "detail": f"{max_family:.6f} > {args.max_any_family_effective_share:.6f}",
            }
        )
    for split_name, split in [("train", train), ("validation", val)]:
        if int(split["unknown_source_rows"]) > 0:
            findings.append(
                {
                    "level": "error",
                    "code": f"{split_name}_rows_with_unweighted_source",
                    "detail": str(split["unknown_source_rows"]),
                }
            )
        if int(split["unknown_subcategory_rows"]) > 0:
            findings.append(
                {
                    "level": "error",
                    "code": f"{split_name}_rows_with_unweighted_subcategory",
                    "detail": str(split["unknown_subcategory_rows"]),
                }
            )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest-json", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--source-weights", required=True)
    parser.add_argument("--subcategory-weights", required=True)
    parser.add_argument("--bit-family", default="bit_manipulation")
    parser.add_argument("--equation-family", default="equation_transform")
    parser.add_argument("--min-bit-effective-share", type=float, default=0.20)
    parser.add_argument("--max-equation-effective-share", type=float, default=0.80)
    parser.add_argument("--max-any-family-effective-share", type=float, default=0.80)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] = {}
    train_path = args.train_jsonl
    val_path = args.val_jsonl
    if args.dataset_manifest_json:
        manifest = read_json(args.dataset_manifest_json)
        train_path = train_path or resolve_manifest_path(manifest, "train_jsonl")
        val_path = val_path or resolve_manifest_path(manifest, "val_jsonl")
    if train_path is None or val_path is None:
        raise RuntimeError("train/validation JSONL paths are required directly or via dataset manifest outputs")
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    source_weights = parse_weight_map(args.source_weights)
    subcategory_weights = parse_weight_map(args.subcategory_weights)
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    report = {
        "schema_version": "kg1_v478_training_objective_alignment_gate_v1",
        "generated_at_utc": utc_now(),
        "dataset_manifest_json": str(args.dataset_manifest_json) if args.dataset_manifest_json else "",
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "source_weights": source_weights,
        "subcategory_weights": subcategory_weights,
        "thresholds": {
            "min_bit_effective_share": args.min_bit_effective_share,
            "max_equation_effective_share": args.max_equation_effective_share,
            "max_any_family_effective_share": args.max_any_family_effective_share,
        },
        "train": summarize_split(train_rows, source_weights, subcategory_weights),
        "validation": summarize_split(val_rows, source_weights, subcategory_weights),
    }
    findings = gate_findings(args, report)
    report["findings"] = findings
    report["hf_gpu_allowed"] = not any(item["level"] == "error" for item in findings)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if args.enforce and not report["hf_gpu_allowed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

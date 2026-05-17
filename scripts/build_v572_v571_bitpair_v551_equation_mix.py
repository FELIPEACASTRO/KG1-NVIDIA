#!/usr/bin/env python3
"""Build V572 mixed source-only pack: V571 bit-pair traces + V551 equation traces.

V571 is structurally clean but bit-only, which makes the objective unsafe for a
paid smoke. V572 adds the already-gated V551 source-only equation rows and sets
explicit row loss weights so the trainer can run with example_mean +
USE_ROW_LOSS_WEIGHT without letting one family dominate the objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import extract_final_answer_for_expected, verify_answer  # noqa: E402


DEFAULT_V571_DIR = ROOT / "artifacts/v571_bitpair_source_only_trace_pack/20260517T_v571_cpu_gate"
DEFAULT_V551_DIR = ROOT / "artifacts/v551_short_bit_trace_pack/20260517T_v551_cpu_gate"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v572_v571_bitpair_v551_equation_mix/20260517T_v572_cpu_gate"

BIT_WEIGHT = 0.50
EQUATION_WEIGHT = 1.50
ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def assistant_text(row: dict[str, Any]) -> str:
    for item in reversed(row.get("messages", [])):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def validate_row(row: dict[str, Any], expected_family: str) -> None:
    row_id = str(row.get("id", ""))
    if str(row.get("family", "")) != expected_family:
        raise RuntimeError(f"{row_id}: expected family {expected_family}, got {row.get('family')}")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_only = bool(
        metadata.get("source_only", False)
        or metadata.get("v523_source_only", False)
        or metadata.get("v536_source_only", False)
    )
    if not source_only:
        raise RuntimeError(f"{row_id}: source_only flag missing/false")
    for flag in ANTI_LEAK_FLAGS:
        if metadata.get(flag) is not False:
            raise RuntimeError(f"{row_id}: {flag} must be false")
    answer = str(row.get("answer", "")).strip()
    extracted = extract_final_answer_for_expected(assistant_text(row), answer)
    if not verify_answer(answer, extracted):
        raise RuntimeError(f"{row_id}: assistant answer mismatch expected={answer} extracted={extracted}")


def rewrite_row(row: dict[str, Any], split: str, component: str, family: str, weight: float, ordinal: int) -> dict[str, Any]:
    validate_row(row, family)
    metadata = dict(row.get("metadata") if isinstance(row.get("metadata"), dict) else {})
    metadata.update(
        {
            "schema_version": "kg1_v572_v571_bitpair_v551_equation_mix_v1",
            "source": "v572_v571_bitpair_v551_equation_mix",
            "source_dataset": "v572_v571_bitpair_v551_equation_mix",
            "source_only": True,
            "loss_weight": weight,
            "v572_component": component,
            "v572_original_id": row.get("id", ""),
            "v572_split": split,
        }
    )
    for flag in ANTI_LEAK_FLAGS:
        metadata[flag] = False
    out = dict(row)
    out["id"] = f"v572_{split}_{component}_{ordinal:05d}"
    out["source"] = "v572_v571_bitpair_v551_equation_mix"
    out["source_dataset"] = "v572_v571_bitpair_v551_equation_mix"
    out["metadata"] = metadata
    return out


def source_paths(args: argparse.Namespace, split: str) -> tuple[Path, Path]:
    suffix = "train" if split == "train" else "val"
    v571 = Path(args.v571_dir) / f"v571_bitpair_source_only_trace_pack_{suffix}.jsonl"
    v551 = Path(args.v551_dir) / f"v551_short_bit_trace_pack_{suffix}.jsonl"
    return v571, v551


def build_split(args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    v571_path, v551_path = source_paths(args, split)
    bit_rows = read_jsonl(v571_path)
    equation_rows = [row for row in read_jsonl(v551_path) if str(row.get("family", "")) == "equation_transform"]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(bit_rows, 1):
        out.append(rewrite_row(row, split, "v571_bitpair", "bit_manipulation", args.bit_loss_weight, idx))
    for idx, row in enumerate(equation_rows, 1):
        out.append(rewrite_row(row, split, "v551_equation", "equation_transform", args.equation_loss_weight, idx))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family", "")) for row in rows)
    component_counts = Counter(str(row.get("metadata", {}).get("v572_component", "")) for row in rows)
    weight_sum: dict[str, float] = {}
    for row in rows:
        family = str(row.get("family", ""))
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        weight_sum[family] = weight_sum.get(family, 0.0) + float(metadata.get("loss_weight", 1.0))
    total_weight = sum(weight_sum.values())
    return {
        "rows": len(rows),
        "family_counts": dict(family_counts.most_common()),
        "component_counts": dict(component_counts.most_common()),
        "loss_weight_sum_by_family": {key: round(value, 6) for key, value in sorted(weight_sum.items())},
        "loss_weight_share_by_family": {
            key: round(value / total_weight, 6) if total_weight else 0.0 for key, value in sorted(weight_sum.items())
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V572 V571 BITPAIR + V551 EQUATION MIX START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v571_dir =", args.v571_dir, flush=True)
    print("v551_dir =", args.v551_dir, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_split(args, "train")
    val_rows = build_split(args, "validation")
    train_jsonl = args.output_dir / "v572_v571_bitpair_v551_equation_mix_train.jsonl"
    val_jsonl = args.output_dir / "v572_v571_bitpair_v551_equation_mix_val.jsonl"
    manifest_json = args.output_dir / "v572_v571_bitpair_v551_equation_mix_manifest.json"
    summary_md = args.output_dir / "KG1_V572_V571_BITPAIR_V551_EQUATION_MIX.md"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    train_summary = summarize(train_rows)
    val_summary = summarize(val_rows)
    blockers: list[str] = []
    if train_summary["family_counts"].get("bit_manipulation", 0) < args.min_bit_train_rows:
        blockers.append("bit_train_rows_below_floor")
    if train_summary["family_counts"].get("equation_transform", 0) < args.min_equation_train_rows:
        blockers.append("equation_train_rows_below_floor")
    if val_summary["family_counts"].get("bit_manipulation", 0) < args.min_bit_val_rows:
        blockers.append("bit_val_rows_below_floor")
    if val_summary["family_counts"].get("equation_transform", 0) < args.min_equation_val_rows:
        blockers.append("equation_val_rows_below_floor")

    manifest = {
        "version": "V572",
        "schema_version": "kg1_v572_v571_bitpair_v551_equation_mix_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v571_dir": str(args.v571_dir),
            "v551_dir": str(args.v551_dir),
            "v571_train_sha256": sha256_file(source_paths(args, "train")[0]),
            "v571_val_sha256": sha256_file(source_paths(args, "validation")[0]),
            "v551_train_sha256": sha256_file(source_paths(args, "train")[1]),
            "v551_val_sha256": sha256_file(source_paths(args, "validation")[1]),
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "val_sha256": sha256_file(val_jsonl),
            "manifest_json": str(manifest_json),
            "summary_md": str(summary_md),
        },
        "weights": {
            "bit_loss_weight": args.bit_loss_weight,
            "equation_loss_weight": args.equation_loss_weight,
            "required_train_env": {
                "LOSS_NORMALIZATION_MODE": "example_mean",
                "USE_ROW_LOSS_WEIGHT": "1",
                "REQUIRE_ROW_LOSS_WEIGHT": "1",
            },
        },
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "blocked_actions": ["train_gpu", "full_eval", "package", "kaggle_submit"],
        "blockers": blockers,
        "decision": {
            "status": "dataset_ready_for_cpu_gates" if not blockers else "blocked_by_row_floor",
            "gpu_allowed": False,
            "submit_allowed": False,
            "reason": (
                "Mixed source-only candidate built with explicit row weights."
                if not blockers
                else "Insufficient bit/equation rows for a mixed candidate."
            ),
            "next_action": (
                "Run V509, V286, V513 and V478 objective alignment. A paid smoke must use example_mean and required row weights."
                if not blockers
                else "Do not train; fix source row coverage first."
            ),
        },
    }
    write_json(manifest_json, manifest)
    lines = [
        "# KG1 V572 V571 Bit-Pair + V551 Equation Mix",
        "",
        f"Generated at UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{manifest['decision']['status']}`",
        f"- Train summary: `{json.dumps(train_summary, sort_keys=True)}`",
        f"- Validation summary: `{json.dumps(val_summary, sort_keys=True)}`",
        f"- Required training env: `{json.dumps(manifest['weights']['required_train_env'], sort_keys=True)}`",
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V572 V571 BITPAIR + V551 EQUATION MIX END ===", flush=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v571-dir", type=Path, default=DEFAULT_V571_DIR)
    parser.add_argument("--v551-dir", type=Path, default=DEFAULT_V551_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bit-loss-weight", type=float, default=BIT_WEIGHT)
    parser.add_argument("--equation-loss-weight", type=float, default=EQUATION_WEIGHT)
    parser.add_argument("--min-bit-train-rows", type=int, default=300)
    parser.add_argument("--min-equation-train-rows", type=int, default=200)
    parser.add_argument("--min-bit-val-rows", type=int, default=50)
    parser.add_argument("--min-equation-val-rows", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

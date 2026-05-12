#!/usr/bin/env python3
"""Build V294 verified equation patch data for representation-level LoRA tuning.

V293 proved that the verified V274 equation rules are not enough when only
``lm_head`` is trainable.  V294 keeps those verified rule rows, reduces their
concentration, and adds more ordinary equation/side-family replay so a short
HF run can update attention representation modules without drifting away from
the 0.86 lineage.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_v282_rank19_micro_patch_dataset import (
    assert_no_reference_overlap,
    dedupe,
    generate_patch_rows,
    normalize_replay_rows,
    read_jsonl,
    reference_from_csv,
    sha256_file,
    summarize,
    validate_rows,
    write_json,
    write_jsonl,
)


SIDE_FAMILIES = ("gravity_constant", "numeral_system", "text_encryption", "unit_conversion")
PATCH_SOURCE = "v294_v274_rule_representation_patch_synthetic"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def retag_patch_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    retagged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["id"] = str(item.get("id", "")).replace("v282_", "v294_", 1)
        item["source"] = PATCH_SOURCE
        metadata = dict(item.get("metadata") or {})
        metadata["source"] = PATCH_SOURCE
        metadata["split"] = split
        metadata["v294_role"] = split
        metadata["v294_base_adapter"] = "v290_checkpoint6_rank19_micro_patch"
        metadata["v294_training_intent"] = "teach_verified_v274_equation_rules_with_attention_representation_modules"
        metadata["weak_gate_rows_used_for_training"] = False
        item["metadata"] = metadata
        retagged.append(item)
    return retagged


def family_buckets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("family", "")), []).append(row)
    return buckets


def sample_rows(
    buckets: dict[str, list[dict[str, Any]]],
    family: str,
    count: int,
    rng: random.Random,
    *,
    required: bool = True,
) -> list[dict[str, Any]]:
    rows = list(buckets.get(family, []))
    if required and not rows:
        raise RuntimeError(f"no replay rows available for family {family!r}")
    rng.shuffle(rows)
    return rows[: min(count, len(rows))]


def select_replay_rows(rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed + (29 if split == "train" else 1029))
    buckets = family_buckets(rows)
    selected: list[dict[str, Any]] = []
    if split == "train":
        selected.extend(sample_rows(buckets, "bit_manipulation", args.bit_train_rows, rng))
        selected.extend(sample_rows(buckets, "equation_transform", args.equation_train_rows, rng))
        for family in SIDE_FAMILIES:
            selected.extend(sample_rows(buckets, family, args.side_train_rows_per_family, rng))
    else:
        selected.extend(sample_rows(buckets, "bit_manipulation", args.bit_val_rows, rng))
        selected.extend(sample_rows(buckets, "equation_transform", args.equation_val_rows, rng))
        for family in SIDE_FAMILIES:
            selected.extend(sample_rows(buckets, family, args.side_val_rows_per_family, rng))
    rng.shuffle(selected)
    return selected, {
        "available_family_counts": dict(sorted((family, len(items)) for family, items in buckets.items())),
        "selected_summary": summarize(selected),
    }


def count_metadata(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str((row.get("metadata") or {}).get(key, "")) for row in rows).items()))


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source", "")) for row in rows).items()))


def enforce_mix_gates(rows: list[dict[str, Any]], args: argparse.Namespace, label: str) -> dict[str, Any]:
    summary = summarize(rows)
    sources = source_counts(rows)
    patch_rows = int(sources.get(PATCH_SOURCE, 0))
    rows_count = int(summary["rows"])
    patch_fraction = patch_rows / rows_count if rows_count else 0.0
    family_counts = dict(summary["family_counts"])
    equation_total = int(family_counts.get("equation_transform", 0))
    bit_total = int(family_counts.get("bit_manipulation", 0))
    side_total = sum(int(family_counts.get(family, 0)) for family in SIDE_FAMILIES)
    gates = {
        "label": label,
        "rows": rows_count,
        "patch_rows": patch_rows,
        "patch_fraction": patch_fraction,
        "equation_total": equation_total,
        "bit_total": bit_total,
        "side_total": side_total,
        "max_patch_fraction": args.max_patch_train_fraction if label == "train" else args.max_patch_val_fraction,
        "min_side_rows": args.min_train_side_rows if label == "train" else args.min_val_side_rows,
    }
    if patch_fraction > gates["max_patch_fraction"]:
        raise RuntimeError(f"{label} patch fraction above gate: {patch_fraction:.4f} > {gates['max_patch_fraction']:.4f}")
    if side_total < gates["min_side_rows"]:
        raise RuntimeError(f"{label} side-family rows below gate: {side_total} < {gates['min_side_rows']}")
    if bit_total <= 0 or equation_total <= 0:
        raise RuntimeError(f"{label} missing required bit/equation rows: {family_counts}")
    return gates


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V294 VERIFIED EQUATION PATCH DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("seed =", args.seed, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay_train_all = normalize_replay_rows(read_jsonl(args.replay_train_jsonl), "train")
    replay_val_all = normalize_replay_rows(read_jsonl(args.replay_val_jsonl), "validation")
    replay_train, replay_train_selection = select_replay_rows(replay_train_all, args, "train")
    replay_val, replay_val_selection = select_replay_rows(replay_val_all, args, "validation")

    patch_train = retag_patch_rows(generate_patch_rows(args.patch_train_per_rule, "train", args.seed + 294000), "train")
    patch_val = retag_patch_rows(generate_patch_rows(args.patch_val_per_rule, "validation", args.seed + 394000), "validation")

    ref_ids, ref_prompt_hashes = reference_from_csv([path for path in args.reference_csv if path])
    overlap_reports = [
        assert_no_reference_overlap(patch_train, ref_ids, ref_prompt_hashes, "patch_train_vs_reference"),
        assert_no_reference_overlap(patch_val, ref_ids, ref_prompt_hashes, "patch_val_vs_reference"),
    ]

    rng = random.Random(args.seed)
    train_rows, train_dedupe = dedupe(replay_train + patch_train)
    val_rows, val_dedupe = dedupe(replay_val + patch_val)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    if len(train_rows) < args.min_train_rows:
        raise RuntimeError(f"train rows below floor: {len(train_rows)} < {args.min_train_rows}")
    if len(val_rows) < args.min_val_rows:
        raise RuntimeError(f"validation rows below floor: {len(val_rows)} < {args.min_val_rows}")

    train_validation = validate_rows(train_rows, "train")
    val_validation = validate_rows(val_rows, "validation")
    train_mix_gates = enforce_mix_gates(train_rows, args, "train")
    val_mix_gates = enforce_mix_gates(val_rows, args, "validation")

    train_path = args.output_dir / "v294_verified_equation_patch_train.jsonl"
    val_path = args.output_dir / "v294_verified_equation_patch_val.jsonl"
    manifest_path = args.output_dir / "v294_verified_equation_patch_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v294_verified_equation_patch_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "seed": args.seed,
        "hypothesis": (
            "V293 failed because lm_head-only updates could not move the representation. "
            "A small attention representation patch should preserve the V290/V291 0.86 lineage "
            "while attempting to convert the verified V274 equation rules into model behavior."
        ),
        "inputs": {
            "replay_train_jsonl": str(args.replay_train_jsonl),
            "replay_val_jsonl": str(args.replay_val_jsonl),
            "reference_csv": [str(path) for path in args.reference_csv],
            "patch_train_per_rule": args.patch_train_per_rule,
            "patch_val_per_rule": args.patch_val_per_rule,
            "bit_train_rows": args.bit_train_rows,
            "equation_train_rows": args.equation_train_rows,
            "side_train_rows_per_family": args.side_train_rows_per_family,
            "bit_val_rows": args.bit_val_rows,
            "equation_val_rows": args.equation_val_rows,
            "side_val_rows_per_family": args.side_val_rows_per_family,
        },
        "selection": {"train_replay": replay_train_selection, "validation_replay": replay_val_selection},
        "dedupe": {"train": train_dedupe, "validation": val_dedupe},
        "overlap_reports": overlap_reports,
        "validation": {"train": train_validation, "validation": val_validation},
        "mix_gates": {"train": train_mix_gates, "validation": val_mix_gates},
        "metadata_counts": {
            "train_rule_counts": count_metadata(train_rows, "rule_name"),
            "validation_rule_counts": count_metadata(val_rows, "rule_name"),
        },
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
        },
        "decision": {
            "status": "dataset_ready_for_hf_upload_and_tokenization_gate",
            "next_action": "Upload to HF and launch a short H200 run with trainable lm_head,o_proj,q_proj,k_proj modules only.",
        },
    }
    write_json(manifest_path, manifest)
    print("v294_dataset_manifest =", json.dumps(manifest, sort_keys=True), flush=True)
    print("=== V294 VERIFIED EQUATION PATCH DATASET END ===", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v294_verified_equation_patch_dataset"))
    parser.add_argument("--label", default="v294_verified_equation_patch")
    parser.add_argument("--replay-train-jsonl", type=Path, default=Path("data/v217/v217_short_answer_train.jsonl"))
    parser.add_argument("--replay-val-jsonl", type=Path, default=Path("data/v217/v217_short_answer_val.jsonl"))
    parser.add_argument("--reference-csv", type=Path, action="append", default=[])
    parser.add_argument("--patch-train-per-rule", type=int, default=1200)
    parser.add_argument("--patch-val-per-rule", type=int, default=90)
    parser.add_argument("--bit-train-rows", type=int, default=1800)
    parser.add_argument("--equation-train-rows", type=int, default=1800)
    parser.add_argument("--side-train-rows-per-family", type=int, default=144)
    parser.add_argument("--bit-val-rows", type=int, default=164)
    parser.add_argument("--equation-val-rows", type=int, default=180)
    parser.add_argument("--side-val-rows-per-family", type=int, default=16)
    parser.add_argument("--seed", type=int, default=294)
    parser.add_argument("--min-train-rows", type=int, default=7600)
    parser.add_argument("--min-val-rows", type=int, default=650)
    parser.add_argument("--max-patch-train-fraction", type=float, default=0.50)
    parser.add_argument("--max-patch-val-fraction", type=float, default=0.50)
    parser.add_argument("--min-train-side-rows", type=int, default=576)
    parser.add_argument("--min-val-side-rows", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    import tempfile

    from build_v282_rank19_micro_patch_dataset import make_row

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        replay_train = tmp / "train.jsonl"
        replay_val = tmp / "val.jsonl"
        ref_csv = tmp / "ref.csv"
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        for family in ("bit_manipulation", "equation_transform", *SIDE_FAMILIES):
            for idx in range(4):
                row = make_row(
                    row_id=f"train_{family}_{idx}",
                    prompt=f"train {family} {idx}",
                    answer=str(idx),
                    split="train",
                    source="selftest_replay",
                    subcategory=family,
                    rule_name="replay",
                )
                row["family"] = family
                train_rows.append(row)
            row = make_row(
                row_id=f"val_{family}",
                prompt=f"val {family}",
                answer="1",
                split="validation",
                source="selftest_replay",
                subcategory=family,
                rule_name="replay",
            )
            row["family"] = family
            val_rows.append(row)
        write_jsonl(replay_train, train_rows)
        write_jsonl(replay_val, val_rows)
        ref_csv.write_text("id,prompt\nweak1,not this prompt\n", encoding="utf-8")
        ns = argparse.Namespace(
            output_dir=tmp / "out",
            label="selftest",
            replay_train_jsonl=replay_train,
            replay_val_jsonl=replay_val,
            reference_csv=[ref_csv],
            patch_train_per_rule=2,
            patch_val_per_rule=1,
            bit_train_rows=2,
            equation_train_rows=2,
            side_train_rows_per_family=2,
            bit_val_rows=1,
            equation_val_rows=1,
            side_val_rows_per_family=1,
            seed=1,
            min_train_rows=16,
            min_val_rows=9,
            max_patch_train_fraction=0.50,
            max_patch_val_fraction=0.50,
            min_train_side_rows=8,
            min_val_side_rows=4,
        )
        manifest = build(ns)
        assert manifest["validation"]["train"]["rows"] >= 16
        assert manifest["validation"]["validation"]["rows"] >= 9
    print("v294_verified_equation_patch_dataset_self_test=ok", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

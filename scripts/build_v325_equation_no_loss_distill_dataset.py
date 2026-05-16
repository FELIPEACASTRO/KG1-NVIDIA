#!/usr/bin/env python3
"""Build V325 equation-only no-loss distillation data from the V324 gate.

V325 is CPU-only. It verifies that V324 found the expected no-loss equation
signal, then generates synthetic out-of-gate variants for the three guarded
numeric rules that produced the weak gain from equation 56/155 to 60/155.

The V324 weak rows are not used as train rows. They are used only as rule
evidence, exactly like the previous V311/V312 seed workflow.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for item in (SCRIPT_DIR, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import build_v312_verifier_synthetic_distill_dataset as v312  # noqa: E402


DEFAULT_V324_MANIFEST = (
    REPO_ROOT
    / "artifacts/v324_equation_expanded_solver_gate/20260513T_cpu_gate/v324_equation_expanded_solver_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v325_equation_no_loss_distill_dataset"
ALLOWED_V324_ACCEPTED_IDS = {
    "274def88",
    "528ec0d8",
    "7688e06e",
    "c5b058d6",
    "d1bd7478",
    "fb623471",
}
MIN_V324_ACCEPTED_COUNT = 4
MIN_V324_PROJECTED_EQUATION_CORRECT = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v324_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("schema_version") != "kg1_v324_equation_expanded_solver_gate_v1":
        raise RuntimeError("unexpected V324 schema: " + str(payload.get("schema_version")))
    ids = set(str(item) for item in payload.get("accepted_candidate_ids", []))
    unexpected_ids = sorted(ids - ALLOWED_V324_ACCEPTED_IDS)
    if unexpected_ids:
        raise RuntimeError("unexpected V324 accepted ids: " + json.dumps(unexpected_ids))
    accepted_count = int(payload.get("accepted_candidate_count", -1))
    if accepted_count != len(ids) or accepted_count < MIN_V324_ACCEPTED_COUNT:
        raise RuntimeError(f"V324 accepted candidate count must be >= {MIN_V324_ACCEPTED_COUNT}: {accepted_count}")
    projected_equation = int(payload.get("projected_equation_correct", -1))
    if projected_equation < MIN_V324_PROJECTED_EQUATION_CORRECT:
        raise RuntimeError(
            "V324 projected equation correct must be >= "
            f"{MIN_V324_PROJECTED_EQUATION_CORRECT}: {projected_equation}"
        )
    decision = payload.get("decision") or {}
    if decision.get("decision") != "equation_cpu_gate_found_distillation_signal":
        raise RuntimeError("V324 decision does not authorize distillation seed: " + str(decision))
    baseline = payload.get("baseline_family_counts") or {}
    bit = int((baseline.get("bit_manipulation") or {}).get("correct", -1))
    equation = int((baseline.get("equation_transform") or {}).get("correct", -1))
    if bit < 136 or equation != 56:
        raise RuntimeError(f"unexpected V324 baseline family counts: bit={bit} equation={equation}")
    if equation + accepted_count != projected_equation:
        raise RuntimeError(
            f"V324 projected equation mismatch: baseline={equation} "
            f"accepted={accepted_count} projected={projected_equation}"
        )
    return payload


def patch_row(row: dict[str, Any], *, split: str, rule_index: int, row_index: int, v324_manifest: Path) -> dict[str, Any]:
    out = dict(row)
    out["id"] = f"v325_{split}_equation_{rule_index:02d}_{row_index:05d}"
    out["source"] = "v325_equation_no_loss_distill"
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "schema_version": "kg1_v325_equation_no_loss_distill_v1",
            "source": "v325_synthetic_from_v324_no_loss_rule_class",
            "source_dataset": "v325_equation_no_loss_distill",
            "split": split,
            "family": "equation_transform",
            "teacher": "v282_generator_plus_v304_equation_trace",
            "v324_manifest_json": str(v324_manifest),
            "v324_seed_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "loss_regime": "equation_narrow_check_plus_final_candidate",
        }
    )
    out["metadata"] = metadata
    return out


def build_split(
    *,
    split: str,
    rows_per_rule: int,
    seed: int,
    v324_manifest: Path,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for rule_index in range(5):
        for row_index in range(rows_per_rule):
            row = v312.build_equation_row(
                rng,
                split=split,
                rule_index=rule_index,
                row_index=row_index,
            )
            rows.append(patch_row(row, split=split, rule_index=rule_index, row_index=row_index, v324_manifest=v324_manifest))
    rng.shuffle(rows)
    return rows


def rule_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str((row.get("metadata") or {}).get("rule_name", ""))] += 1
    return dict(sorted(counts.items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V325 EQUATION NO-LOSS DISTILL DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v324_manifest_json =", args.v324_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("seed =", args.seed, flush=True)
    print("train_rows_per_rule =", args.train_rows_per_rule, flush=True)
    print("val_rows_per_rule =", args.val_rows_per_rule, flush=True)

    v324_manifest = validate_v324_manifest(args.v324_manifest_json)
    ref_ids, ref_prompt_hashes, reference_summary = v312.read_reference_fingerprints(args.reference_path)
    print("reference_id_count =", len(ref_ids), flush=True)
    print("reference_prompt_hash_count =", len(ref_prompt_hashes), flush=True)

    train_rows = build_split(
        split="train",
        rows_per_rule=args.train_rows_per_rule,
        seed=args.seed,
        v324_manifest=args.v324_manifest_json,
    )
    val_rows = build_split(
        split="validation",
        rows_per_rule=args.val_rows_per_rule,
        seed=args.seed + 10000,
        v324_manifest=args.v324_manifest_json,
    )
    train_preferences = v312.build_preferences(train_rows)
    val_preferences = v312.build_preferences(val_rows)

    train_summary = v312.validate_rows(train_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="train")
    val_summary = v312.validate_rows(val_rows, ref_ids=ref_ids, ref_prompt_hashes=ref_prompt_hashes, label="validation")
    train_pref_summary = v312.validate_preferences(train_preferences, "train")
    val_pref_summary = v312.validate_preferences(val_preferences, "validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_sft_train.jsonl"
    val_path = args.output_dir / f"{args.label}_sft_val.jsonl"
    train_pref_path = args.output_dir / f"{args.label}_preferences_train.jsonl"
    val_pref_path = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"

    v312.write_jsonl(train_path, train_rows)
    v312.write_jsonl(val_path, val_rows)
    v312.write_jsonl(train_pref_path, train_preferences)
    v312.write_jsonl(val_pref_path, val_preferences)

    manifest = {
        "schema_version": "kg1_v325_equation_no_loss_distill_dataset_v1",
        "generated_at_utc": utc_now(),
        "v324_manifest_json": str(args.v324_manifest_json),
        "v324_manifest_sha256": v312.sha256_file(args.v324_manifest_json),
        "v324_accepted_candidate_ids": v324_manifest.get("accepted_candidate_ids", []),
        "v324_projected_equation_correct": v324_manifest.get("projected_equation_correct"),
        "reference_summary": reference_summary,
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "train_preference_summary": train_pref_summary,
        "validation_preference_summary": val_pref_summary,
        "train_rule_counts": rule_counts(train_rows),
        "validation_rule_counts": rule_counts(val_rows),
        "outputs": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "train_sha256": v312.sha256_file(train_path),
            "val_sha256": v312.sha256_file(val_path),
            "sft_train_jsonl": str(train_path),
            "sft_val_jsonl": str(val_path),
            "preferences_train_jsonl": str(train_pref_path),
            "preferences_val_jsonl": str(val_pref_path),
            "manifest_json": str(manifest_path),
        },
        "hashes": {
            "sft_train_sha256": v312.sha256_file(train_path),
            "sft_val_sha256": v312.sha256_file(val_path),
            "preferences_train_sha256": v312.sha256_file(train_pref_path),
            "preferences_val_sha256": v312.sha256_file(val_pref_path),
        },
        "training_authorization": "blocked_until_real_tokenization_and_no_regression_gate",
        "required_next_gate": [
            "real_tokenization_gate_with_offset_masks",
            "combine_with_strong_bit_replay_before_any_hf_train",
            "first_checkpoint_kill_switch_bit_ge_136_equation_gt_56",
            "weak_eval_before_full_eval_or_submit",
        ],
    }
    v312.write_json(manifest_path, manifest)

    print("sft_train_jsonl =", train_path, flush=True)
    print("sft_val_jsonl =", val_path, flush=True)
    print("preferences_train_jsonl =", train_pref_path, flush=True)
    print("preferences_val_jsonl =", val_pref_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("train_rule_counts =", json.dumps(manifest["train_rule_counts"], sort_keys=True), flush=True)
    print("validation_rule_counts =", json.dumps(manifest["validation_rule_counts"], sort_keys=True), flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V325 EQUATION NO-LOSS DISTILL DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v325_selftest_") as temp_name:
        tmp = Path(temp_name)
        fake_v324 = tmp / "v324.json"
        v312.write_json(
            fake_v324,
            {
                "schema_version": "kg1_v324_equation_expanded_solver_gate_v1",
                "accepted_candidate_count": 6,
                "accepted_candidate_ids": sorted(ALLOWED_V324_ACCEPTED_IDS),
                "projected_equation_correct": 62,
                "baseline_family_counts": {
                    "bit_manipulation": {"correct": 136},
                    "equation_transform": {"correct": 56},
                },
                "decision": {"decision": "equation_cpu_gate_found_distillation_signal"},
            },
        )
        payload = validate_v324_manifest(fake_v324)
        if int(payload["projected_equation_correct"]) != 62:
            raise AssertionError(payload)
        rows = build_split(split="train", rows_per_rule=1, seed=123, v324_manifest=fake_v324)
        summary = v312.validate_rows(rows, ref_ids=set(), ref_prompt_hashes=set(), label="self_test")
        if summary["rows"] != 5:
            raise AssertionError(summary)
    print("v325_equation_no_loss_distill_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v324-manifest-json", type=Path, default=DEFAULT_V324_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / utc_compact())
    parser.add_argument("--label", default="v325_equation_no_loss_distill")
    parser.add_argument("--seed", type=int, default=32513)
    parser.add_argument("--train-rows-per-rule", type=int, default=160)
    parser.add_argument("--val-rows-per-rule", type=int, default=40)
    parser.add_argument("--reference-path", type=Path, action="append", default=list(v312.DEFAULT_REFERENCE_PATHS))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

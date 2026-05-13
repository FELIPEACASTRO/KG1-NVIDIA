#!/usr/bin/env python3
"""Build V331 equation + bit replay + symbolic cryptarithm mix.

V331 extends V326 with the V330 symbolic cryptarithm signal while preserving
the V304 bit replay guardrail. It is still a dataset/gate artifact, not a
training authorization by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for item in (SCRIPT_DIR, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import build_v326_equation_bit_replay_mix_dataset as v326  # noqa: E402


DEFAULT_V304_ROOT = REPO_ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z"
DEFAULT_V325_ROOT = REPO_ROOT / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate"
DEFAULT_V330_ROOT = REPO_ROOT / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v331_equation_bit_symbolic_mix_dataset/20260513T_cpu_gate"
SCHEMA_VERSION = "kg1_v331_equation_bit_symbolic_mix_dataset_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def validate_input_manifests(v304_root: Path, v325_root: Path, v330_root: Path) -> dict[str, Any]:
    base_inputs = v326.validate_input_manifest(v304_root, v325_root)
    v330_manifest_path = v330_root / "v330_symbolic_cryptarithm_distill_manifest.json"
    v330_manifest = read_json(v330_manifest_path)
    if v330_manifest.get("schema_version") != "kg1_v330_symbolic_cryptarithm_distill_dataset_v1":
        raise RuntimeError("unexpected V330 schema: " + str(v330_manifest.get("schema_version")))
    if int((v330_manifest.get("train_summary") or {}).get("rows", -1)) != 240:
        raise RuntimeError("V330 train rows drift")
    if int((v330_manifest.get("validation_summary") or {}).get("rows", -1)) != 60:
        raise RuntimeError("V330 validation rows drift")
    source_gate = v330_manifest.get("source_gate") if isinstance(v330_manifest.get("source_gate"), dict) else {}
    if int(source_gate.get("projected_equation_correct", -1)) != 61:
        raise RuntimeError("V330 is not based on equation=61 CPU projection")
    return {
        **base_inputs,
        "v330_manifest_json": str(v330_manifest_path),
        "v330_manifest_sha256": sha256_file(v330_manifest_path),
    }


def select_v330_symbolic_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if v326.family_of(row) != "equation_transform":
            raise RuntimeError("V330 row is not equation_transform: " + str(row.get("id", "")))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("v329_seed_rows_used_for_training") is not False:
            raise RuntimeError("V330 seed leakage flag drift: " + str(row.get("id", "")))
        out.append(v326.normalize_boxed_suffix(row, split=split, origin="v330_symbolic_cryptarithm_distill"))
    return out


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V331 EQUATION BIT SYMBOLIC MIX DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v304_root =", args.v304_root, flush=True)
    print("v325_root =", args.v325_root, flush=True)
    print("v330_root =", args.v330_root, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    manifest_inputs = validate_input_manifests(args.v304_root, args.v325_root, args.v330_root)
    paths = {
        "v304_train": args.v304_root / "v304_solver_trace_distill_train.jsonl",
        "v304_val": args.v304_root / "v304_solver_trace_distill_val.jsonl",
        "v325_train": args.v325_root / "v325_equation_no_loss_distill_sft_train.jsonl",
        "v325_val": args.v325_root / "v325_equation_no_loss_distill_sft_val.jsonl",
        "v330_train": args.v330_root / "v330_symbolic_cryptarithm_distill_sft_train.jsonl",
        "v330_val": args.v330_root / "v330_symbolic_cryptarithm_distill_sft_val.jsonl",
    }
    for label, path in paths.items():
        print(f"input_{label} = {path} exists={path.is_file()}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    train_rows = (
        v326.select_v304_bit_rows(v326.read_jsonl(paths["v304_train"]), split="train")
        + v326.select_v325_equation_rows(v326.read_jsonl(paths["v325_train"]), split="train")
        + select_v330_symbolic_rows(v326.read_jsonl(paths["v330_train"]), split="train")
    )
    val_rows = (
        v326.select_v304_bit_rows(v326.read_jsonl(paths["v304_val"]), split="validation")
        + v326.select_v325_equation_rows(v326.read_jsonl(paths["v325_val"]), split="validation")
        + select_v330_symbolic_rows(v326.read_jsonl(paths["v330_val"]), split="validation")
    )

    train_audit = v326.audit_rows(train_rows, label="train")
    val_audit = v326.audit_rows(val_rows, label="validation")
    overlaps = v326.overlap_summary(train_rows, val_rows)
    print("train_audit =", json.dumps(train_audit, sort_keys=True), flush=True)
    print("validation_audit =", json.dumps(val_audit, sort_keys=True), flush=True)
    print("overlap_summary =", json.dumps(overlaps, sort_keys=True), flush=True)

    if train_audit["bad_rows_first10"] or val_audit["bad_rows_first10"]:
        raise RuntimeError("V331 row audit failed")
    if overlaps["train_val_id_overlap"] or overlaps["train_val_prompt_overlap"]:
        raise RuntimeError("V331 train/validation overlap detected")
    if int(train_audit["family_counts"].get("bit_manipulation", 0)) < args.min_train_bit_rows:
        raise RuntimeError("V331 train bit rows below required floor")
    if int(train_audit["family_counts"].get("equation_transform", 0)) < args.min_train_equation_rows:
        raise RuntimeError("V331 train equation rows below required floor")
    if int(val_audit["family_counts"].get("bit_manipulation", 0)) < args.min_val_bit_rows:
        raise RuntimeError("V331 validation bit rows below required floor")
    if int(val_audit["family_counts"].get("equation_transform", 0)) < args.min_val_equation_rows:
        raise RuntimeError("V331 validation equation rows below required floor")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_train.jsonl"
    val_path = args.output_dir / f"{args.label}_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "inputs": {
            **manifest_inputs,
            **{label: {"path": str(path), "sha256": sha256_file(path)} for label, path in paths.items()},
        },
        "train_summary": train_audit,
        "validation_summary": val_audit,
        "overlap_summary": overlaps,
        "outputs": {
            "manifest_json": str(manifest_path),
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
        },
        "training_authorization": "blocked_until_v286_real_tokenization_gate_and_hf_smoke_kill_switch",
        "source_policy": {
            "bit_rows": "all V304 bit_manipulation rows retained for non-regression replay",
            "equation_rows": "V325 numeric no-loss rows plus V330 symbolic cryptarithm rows only",
            "final_answer_format": "boxed_suffix",
            "physical_duplicates": False,
            "weak_or_full_gate_rows_used_for_training": False,
            "v329_seed_rows_used_for_training": False,
        },
        "recommended_hf_controls": {
            "first_checkpoint_kill_switch": "bit>=136 and equation>56 on weak gate",
            "no_full_eval_or_submit_until": "adapter-only weak gate shows measured gain with no bit regression",
            "suggested_source_weights": {
                "v304_bit_replay_only": 1.0,
                "v325_equation_no_loss_distill": 4.0,
                "v330_symbolic_cryptarithm_distill": 3.0,
            },
        },
    }
    write_json(manifest_path, manifest)

    print("train_jsonl =", train_path, flush=True)
    print("val_jsonl =", val_path, flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("train_sha256 =", manifest["outputs"]["train_sha256"], flush=True)
    print("val_sha256 =", manifest["outputs"]["val_sha256"], flush=True)
    print("training_authorization =", manifest["training_authorization"], flush=True)
    print("=== V331 EQUATION BIT SYMBOLIC MIX DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v331_selftest_") as temp_name:
        tmp = Path(temp_name)
        v304_root = tmp / "v304"
        v325_root = tmp / "v325"
        v330_root = tmp / "v330"
        out = tmp / "out"
        v304_root.mkdir()
        v325_root.mkdir()
        v330_root.mkdir()

        bit_row = {
            "id": "bit_train_1",
            "prompt": "bit prompt",
            "answer": "00000001",
            "family": "bit_manipulation",
            "source": "v304_solver_trace_bit_fullbyte_distill_exact",
            "subcategory": "bit_fullbyte_v300_gain_pattern",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "bit prompt"},
                {"role": "assistant", "content": "Trace\nFinal answer: 00000001"},
            ],
            "metadata": {"family": "bit_manipulation", "source": "v304_solver_trace_bit_fullbyte_distill_exact"},
        }
        eq_row = {
            "id": "eq_train_1",
            "prompt": "equation prompt",
            "answer": "42",
            "family": "equation_transform",
            "source": "v325_equation_no_loss_distill",
            "subcategory": "equation_numeric_minus_signed",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "equation prompt"},
                {"role": "assistant", "content": r"Trace" + "\n" + r"Final answer: \boxed{42}"},
            ],
            "metadata": {"family": "equation_transform", "source": "v325_equation_no_loss_distill"},
        }
        sym_row = {
            "id": "sym_train_1",
            "prompt": "symbolic prompt",
            "answer": "?()<",
            "family": "equation_transform",
            "source": "v330_symbolic_cryptarithm_distill",
            "subcategory": "equation_symbolic_cryptarithm_single_operator_mul",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "symbolic prompt"},
                {"role": "assistant", "content": r"Trace" + "\n" + r"Final answer: \boxed{?()<}"},
            ],
            "metadata": {
                "family": "equation_transform",
                "source": "v330_symbolic_cryptarithm_distill",
                "v329_seed_rows_used_for_training": False,
            },
        }
        v326.write_json(v304_root / "v304_solver_trace_distill_manifest.json", {"validation": {"train": {"family_counts": {"bit_manipulation": 4000}}}})
        v326.write_json(
            v325_root / "v325_equation_no_loss_distill_manifest.json",
            {"schema_version": "kg1_v325_equation_no_loss_distill_dataset_v1", "v324_projected_equation_correct": 60},
        )
        write_json(
            v330_root / "v330_symbolic_cryptarithm_distill_manifest.json",
            {
                "schema_version": "kg1_v330_symbolic_cryptarithm_distill_dataset_v1",
                "train_summary": {"rows": 240},
                "validation_summary": {"rows": 60},
                "source_gate": {"projected_equation_correct": 61},
            },
        )
        v326.write_jsonl(v304_root / "v304_solver_trace_distill_train.jsonl", [bit_row])
        v326.write_jsonl(v304_root / "v304_solver_trace_distill_val.jsonl", [{**bit_row, "id": "bit_val_1", "prompt": "bit val prompt"}])
        v326.write_jsonl(v325_root / "v325_equation_no_loss_distill_sft_train.jsonl", [eq_row])
        v326.write_jsonl(v325_root / "v325_equation_no_loss_distill_sft_val.jsonl", [{**eq_row, "id": "eq_val_1", "prompt": "equation val prompt"}])
        v326.write_jsonl(v330_root / "v330_symbolic_cryptarithm_distill_sft_train.jsonl", [sym_row])
        v326.write_jsonl(v330_root / "v330_symbolic_cryptarithm_distill_sft_val.jsonl", [{**sym_row, "id": "sym_val_1", "prompt": "symbolic val prompt"}])
        args = argparse.Namespace(
            v304_root=v304_root,
            v325_root=v325_root,
            v330_root=v330_root,
            output_dir=out,
            label="v331_selftest",
            min_train_bit_rows=1,
            min_train_equation_rows=2,
            min_val_bit_rows=1,
            min_val_equation_rows=2,
        )
        manifest = build_dataset(args)
        if manifest["train_summary"]["family_counts"] != {"bit_manipulation": 1, "equation_transform": 2}:
            raise AssertionError(manifest["train_summary"])
    print("v331_equation_bit_symbolic_mix_self_test=ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v304-root", type=Path, default=DEFAULT_V304_ROOT)
    parser.add_argument("--v325-root", type=Path, default=DEFAULT_V325_ROOT)
    parser.add_argument("--v330-root", type=Path, default=DEFAULT_V330_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v331_equation_bit_symbolic_mix")
    parser.add_argument("--min-train-bit-rows", type=int, default=4000)
    parser.add_argument("--min-train-equation-rows", type=int, default=720)
    parser.add_argument("--min-val-bit-rows", type=int, default=300)
    parser.add_argument("--min-val-equation-rows", type=int, default=180)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

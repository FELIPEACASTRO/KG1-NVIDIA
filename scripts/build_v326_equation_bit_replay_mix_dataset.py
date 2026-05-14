#!/usr/bin/env python3
"""Build V326 equation patch plus bit replay mix.

V326 is the first trainable dataset after the V324/V325 CPU gate. It keeps the
new narrow equation signal from V325 and combines it with only the
bit_manipulation replay from V304. Broad historical equation rows are excluded
because repeated SFT on those rows has not moved the equation ceiling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
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


DEFAULT_V304_ROOT = REPO_ROOT / "artifacts/v304_solver_trace_distill_dataset/20260512T1430Z"
DEFAULT_V325_ROOT = REPO_ROOT / "artifacts/v325_equation_no_loss_distill_dataset/20260513T_cpu_gate"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v326_equation_bit_replay_mix_dataset/20260513T_cpu_gate"

FINAL_ANSWER_RE = re.compile(r"(?m)^Final answer:\s*.*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not a JSON object")
            rows.append(row)
    return rows


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return repo_relative


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def family_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("family") or metadata.get("family") or "")


def source_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("source") or metadata.get("source") or "")


def subcategory_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("subcategory") or metadata.get("subcategory") or "")


def assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise RuntimeError("row has no assistant message")


def normalize_boxed_suffix(row: dict[str, Any], *, split: str, origin: str) -> dict[str, Any]:
    item = copy.deepcopy(row)
    answer = str(item.get("answer", ""))
    if not answer:
        raise RuntimeError("row missing answer: " + str(item.get("id", "")))
    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise RuntimeError("row missing messages: " + str(item.get("id", "")))
    idx = assistant_index(messages)
    content = str(messages[idx].get("content", "")).rstrip()
    final_line = r"Final answer: \boxed{" + answer + "}"
    if FINAL_ANSWER_RE.search(content):
        content = FINAL_ANSWER_RE.sub(lambda _: final_line, content)
    elif content:
        content = content + "\n" + final_line
    else:
        content = final_line
    messages[idx]["content"] = content
    item["messages"] = messages

    metadata = dict(item.get("metadata") or {})
    metadata.update(
        {
            "v326_mix_origin": origin,
            "v326_split": split,
            "v326_final_answer_normalization": "boxed_suffix",
            "v326_physical_duplicate": False,
            "gate_rows_used_for_training": False,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    item["metadata"] = metadata
    item["family"] = family_of(item)
    item["source"] = source_of(item)
    item["subcategory"] = subcategory_of(item)
    return item


def prompt_hash(row: dict[str, Any]) -> str:
    return sha256_text(str(row.get("prompt", "")).strip().replace("\r\n", "\n"))


def select_v304_bit_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if family_of(row) != "bit_manipulation":
            continue
        out.append(normalize_boxed_suffix(row, split=split, origin="v304_bit_replay_only"))
    return out


def select_v325_equation_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if family_of(row) != "equation_transform":
            raise RuntimeError("V325 row is not equation_transform: " + str(row.get("id", "")))
        out.append(normalize_boxed_suffix(row, split=split, origin="v325_equation_no_loss_distill"))
    return out


def audit_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    ids: list[str] = []
    prompts: list[str] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    bad_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        row_id = str(row.get("id", ""))
        answer = str(row.get("answer", ""))
        messages = row.get("messages")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        ids.append(row_id)
        prompts.append(prompt_hash(row))
        family_counts[family_of(row)] += 1
        source_counts[source_of(row)] += 1
        subcategory_counts[subcategory_of(row)] += 1
        origin_counts[str(metadata.get("v326_mix_origin", ""))] += 1

        if not row_id or not row.get("prompt") or not answer or not isinstance(messages, list):
            bad_rows.append({"index": index, "id": row_id, "reason": "missing required field"})
            continue
        try:
            idx = assistant_index(messages)
        except RuntimeError:
            bad_rows.append({"index": index, "id": row_id, "reason": "missing assistant"})
            continue
        expected = r"Final answer: \boxed{" + answer + "}"
        assistant = str(messages[idx].get("content", "")).rstrip()
        if not assistant.endswith(expected):
            bad_rows.append({"index": index, "id": row_id, "reason": "assistant does not end with boxed answer"})
        for flag in ("gate_rows_used_for_training", "weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if bool(metadata.get(flag) or row.get(flag)):
                bad_rows.append({"index": index, "id": row_id, "reason": "gate row training flag present", "flag": flag})

    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompts) - len(set(prompts))
    if duplicate_ids:
        bad_rows.append({"reason": "duplicate ids", "count": duplicate_ids})
    if duplicate_prompts:
        bad_rows.append({"reason": "duplicate prompts", "count": duplicate_prompts})

    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_prompt_hashes": len(set(prompts)),
        "duplicate_ids": duplicate_ids,
        "duplicate_prompts": duplicate_prompts,
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "bad_rows_first10": bad_rows[:10],
    }


def overlap_summary(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row.get("id", "")) for row in train_rows}
    val_ids = {str(row.get("id", "")) for row in val_rows}
    train_prompts = {prompt_hash(row) for row in train_rows}
    val_prompts = {prompt_hash(row) for row in val_rows}
    return {
        "train_val_id_overlap": len(train_ids & val_ids),
        "train_val_prompt_overlap": len(train_prompts & val_prompts),
        "train_val_id_overlap_sample": sorted(train_ids & val_ids)[:10],
        "train_val_prompt_overlap_sample": sorted(train_prompts & val_prompts)[:10],
    }


def validate_input_manifest(v304_root: Path, v325_root: Path) -> dict[str, Any]:
    v304_manifest_path = v304_root / "v304_solver_trace_distill_manifest.json"
    v325_manifest_path = v325_root / "v325_equation_no_loss_distill_manifest.json"
    if not v304_manifest_path.is_file():
        raise FileNotFoundError(v304_manifest_path)
    if not v325_manifest_path.is_file():
        candidates = sorted(v325_root.glob("*_manifest.json"))
        if len(candidates) != 1:
            raise FileNotFoundError(v325_manifest_path)
        v325_manifest_path = candidates[0]
    v304_manifest = read_json(v304_manifest_path)
    v325_manifest = read_json(v325_manifest_path)
    if v325_manifest.get("schema_version") != "kg1_v325_equation_no_loss_distill_dataset_v1":
        raise RuntimeError("unexpected V325 schema: " + str(v325_manifest.get("schema_version")))
    if int(v325_manifest.get("v324_projected_equation_correct", -1)) < 60:
        raise RuntimeError("V325 is not based on equation>=60 CPU projection")
    v304_train_bit = int((v304_manifest.get("validation") or {}).get("train", {}).get("family_counts", {}).get("bit_manipulation", 0))
    if v304_train_bit < 4000:
        raise RuntimeError("V304 bit replay is unexpectedly small: " + str(v304_train_bit))
    return {
        "v304_manifest_json": str(v304_manifest_path),
        "v304_manifest_sha256": sha256_file(v304_manifest_path),
        "v325_manifest_json": str(v325_manifest_path),
        "v325_manifest_sha256": sha256_file(v325_manifest_path),
        "v325_manifest": v325_manifest,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V326 EQUATION BIT REPLAY MIX DATASET START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v304_root =", args.v304_root, flush=True)
    print("v325_root =", args.v325_root, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    manifest_inputs = validate_input_manifest(args.v304_root, args.v325_root)
    v325_manifest = manifest_inputs.pop("v325_manifest")
    v325_outputs = v325_manifest.get("outputs") or {}
    v325_train = resolve_manifest_path(
        args.v325_root / "v325_equation_no_loss_distill_manifest.json",
        str(v325_outputs.get("sft_train_jsonl") or v325_outputs.get("train_jsonl") or ""),
    )
    v325_val = resolve_manifest_path(
        args.v325_root / "v325_equation_no_loss_distill_manifest.json",
        str(v325_outputs.get("sft_val_jsonl") or v325_outputs.get("val_jsonl") or ""),
    )
    paths = {
        "v304_train": args.v304_root / "v304_solver_trace_distill_train.jsonl",
        "v304_val": args.v304_root / "v304_solver_trace_distill_val.jsonl",
        "v325_train": v325_train,
        "v325_val": v325_val,
    }
    for label, path in paths.items():
        print(f"input_{label} = {path} exists={path.is_file()}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    train_rows = (
        select_v304_bit_rows(read_jsonl(paths["v304_train"]), split="train")
        + select_v325_equation_rows(read_jsonl(paths["v325_train"]), split="train")
    )
    val_rows = (
        select_v304_bit_rows(read_jsonl(paths["v304_val"]), split="validation")
        + select_v325_equation_rows(read_jsonl(paths["v325_val"]), split="validation")
    )

    train_audit = audit_rows(train_rows, label="train")
    val_audit = audit_rows(val_rows, label="validation")
    overlaps = overlap_summary(train_rows, val_rows)
    print("train_audit =", json.dumps(train_audit, sort_keys=True), flush=True)
    print("validation_audit =", json.dumps(val_audit, sort_keys=True), flush=True)
    print("overlap_summary =", json.dumps(overlaps, sort_keys=True), flush=True)

    if train_audit["bad_rows_first10"] or val_audit["bad_rows_first10"]:
        raise RuntimeError("V326 row audit failed")
    if overlaps["train_val_id_overlap"] or overlaps["train_val_prompt_overlap"]:
        raise RuntimeError("V326 train/validation overlap detected")
    if int(train_audit["family_counts"].get("bit_manipulation", 0)) < args.min_train_bit_rows:
        raise RuntimeError("V326 train bit rows below required floor")
    if int(train_audit["family_counts"].get("equation_transform", 0)) < args.min_train_equation_rows:
        raise RuntimeError("V326 train equation rows below required floor")
    if int(val_audit["family_counts"].get("bit_manipulation", 0)) < args.min_val_bit_rows:
        raise RuntimeError("V326 validation bit rows below required floor")
    if int(val_audit["family_counts"].get("equation_transform", 0)) < args.min_val_equation_rows:
        raise RuntimeError("V326 validation equation rows below required floor")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.label}_train.jsonl"
    val_path = args.output_dir / f"{args.label}_val.jsonl"
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "schema_version": "kg1_v326_equation_bit_replay_mix_dataset_v1",
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
        "training_authorization": "blocked_until_v286_tokenization_gate_and_hf_smoke_kill_switch",
        "source_policy": {
            "equation_rows": "V325 only; V304 broad equation replay intentionally excluded",
            "bit_rows": "all V304 bit_manipulation rows retained for non-regression replay",
            "final_answer_format": "boxed_suffix",
            "physical_duplicates": False,
            "weak_or_full_gate_rows_used_for_training": False,
        },
        "recommended_hf_controls": {
            "first_checkpoint_kill_switch": "bit>=136 and equation>56 on weak gate",
            "no_full_eval_or_submit_until": "adapter-only weak gate shows measured gain with no bit regression",
            "suggested_source_weights": {
                "v304_bit_replay_only": 1.0,
                "v325_equation_no_loss_distill": 4.0,
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
    print("=== V326 EQUATION BIT REPLAY MIX DATASET END ===", flush=True)
    return manifest


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="kg1_v326_selftest_") as temp_name:
        tmp = Path(temp_name)
        v304_root = tmp / "v304"
        v325_root = tmp / "v325"
        out = tmp / "out"
        v304_root.mkdir()
        v325_root.mkdir()

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
        write_json(v304_root / "v304_solver_trace_distill_manifest.json", {"validation": {"train": {"family_counts": {"bit_manipulation": 4000}}}})
        write_json(
            v325_root / "v325_equation_no_loss_distill_manifest.json",
            {
                "schema_version": "kg1_v325_equation_no_loss_distill_dataset_v1",
                "v324_projected_equation_correct": 62,
                "outputs": {
                    "sft_train_jsonl": str(v325_root / "v325_equation_no_loss_distill_sft_train.jsonl"),
                    "sft_val_jsonl": str(v325_root / "v325_equation_no_loss_distill_sft_val.jsonl"),
                },
            },
        )
        write_jsonl(v304_root / "v304_solver_trace_distill_train.jsonl", [bit_row])
        write_jsonl(v304_root / "v304_solver_trace_distill_val.jsonl", [{**bit_row, "id": "bit_val_1", "prompt": "bit val prompt"}])
        write_jsonl(v325_root / "v325_equation_no_loss_distill_sft_train.jsonl", [eq_row])
        write_jsonl(v325_root / "v325_equation_no_loss_distill_sft_val.jsonl", [{**eq_row, "id": "eq_val_1", "prompt": "equation val prompt"}])
        args = argparse.Namespace(
            v304_root=v304_root,
            v325_root=v325_root,
            output_dir=out,
            label="v326_selftest",
            min_train_bit_rows=1,
            min_train_equation_rows=1,
            min_val_bit_rows=1,
            min_val_equation_rows=1,
        )
        manifest = build_dataset(args)
        if manifest["train_summary"]["family_counts"] != {"bit_manipulation": 1, "equation_transform": 1}:
            raise AssertionError(manifest["train_summary"])
    print("v326_equation_bit_replay_mix_self_test=ok", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v304-root", type=Path, default=DEFAULT_V304_ROOT)
    parser.add_argument("--v325-root", type=Path, default=DEFAULT_V325_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="v326_equation_bit_replay_mix")
    parser.add_argument("--min-train-bit-rows", type=int, default=4000)
    parser.add_argument("--min-train-equation-rows", type=int, default=480)
    parser.add_argument("--min-val-bit-rows", type=int, default=300)
    parser.add_argument("--min-val-equation-rows", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

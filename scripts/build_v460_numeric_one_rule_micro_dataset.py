#!/usr/bin/env python3
"""Build a V460 one-rule numeric micro dataset proposal.

This dataset is deliberately small and CPU-only. It converts the V459 audited
numeric equation probes into short SFT rows and adds bit replay rows from V217
to protect the bit guardrail. It does not launch GPU, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_V459_DIR = REPO_ROOT / "artifacts/v459_v458_numeric_hard_negative_audit/20260515T_v459_cpu_audit"
DEFAULT_BIT_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
DEFAULT_BIT_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v460_numeric_one_rule_micro_dataset"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify compactly, and end with exactly one final answer in \\boxed{}."
)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def escape_boxed(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def boxed(value: object) -> str:
    return "\\boxed{" + escape_boxed(value) + "}"


def hard_flag(row: dict[str, str]) -> bool:
    return str(row.get("real_hard_negative_candidate", "")).lower() == "true"


def make_equation_row(row: dict[str, str], split: str) -> dict[str, Any]:
    answer = str(row["answer"])
    wrong = str(row.get("adapter_prediction") or row.get("simulated_wrong_prediction") or "")
    source_role = "hard_negative" if hard_flag(row) else "positive_rule_replay"
    proof = (
        "Numeric sign guard. Compare the minus examples, reject the opposite-sign candidate "
        f"{wrong!r}, and preserve the signed target value."
    )
    assistant = f"Verification: {proof}\nFinal answer: {boxed(answer)}"
    metadata = {
        "source": "v460_v459_numeric_one_rule_micro_dataset",
        "source_dataset": "v460_v459_numeric_one_rule_micro_dataset",
        "source_role": source_role,
        "family": "equation_transform",
        "rule_class": row["target_rule_class"],
        "target_rule_class": row["target_rule_class"],
        "v459_source_id": row["id"],
        "adapter_prediction": row.get("adapter_prediction", ""),
        "simulated_wrong_prediction": row.get("simulated_wrong_prediction", ""),
        "postprocessor_prediction": row.get("postprocessor_prediction", ""),
        "real_hard_negative_candidate": hard_flag(row),
        "raw_output_collected_without_labels": True,
        "labels_joined_after_collection_from_public_train": True,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "gate_rows_used_for_training": False,
        "split": split,
    }
    return {
        "id": f"v460_{split}_equation_{row['id']}",
        "family": "equation_transform",
        "subcategory": row["target_rule_class"],
        "source": "v460_v459_numeric_one_rule_micro_dataset",
        "prompt": row["prompt"],
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def make_bit_replay_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    prompt = str(row["prompt"])
    answer = str(row["answer"])
    assistant = f"Final answer: {boxed(answer)}"
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source": "v217_bit_replay_guardrail",
            "source_dataset": "v217_bit_replay_guardrail",
            "source_role": "bit_guardrail_replay",
            "v460_split": split,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
        }
    )
    return {
        "id": f"v460_{split}_bit_replay_{index:04d}_{row.get('id', '')}",
        "family": "bit_manipulation",
        "subcategory": "bit_guardrail_replay",
        "source": "v217_bit_replay_guardrail",
        "prompt": prompt,
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": metadata,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row.get("family", "")) for row in rows)
    subcat = Counter(str(row.get("subcategory", "")) for row in rows)
    source_role = Counter(str((row.get("metadata") or {}).get("source_role", "")) for row in rows)
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family.items())),
        "subcategory_counts": dict(subcat.most_common(20)),
        "source_role_counts": dict(sorted(source_role.items())),
    }


def render_report(manifest: dict[str, Any]) -> str:
    train = manifest["summary"]["train"]
    val = manifest["summary"]["validation"]
    decision = manifest["decision"]
    return "\n".join(
        [
            "# V460 Numeric One-Rule Micro Dataset",
            "",
            "## Purpose",
            "",
            "CPU proposal for a one-rule numeric equation smoke. It uses the V459 real adapter mistakes and bit replay guardrail.",
            "",
            "## Counts",
            "",
            f"- Train rows: `{train['rows']}`; families: `{train['family_counts']}`.",
            f"- Validation rows: `{val['rows']}`; families: `{val['family_counts']}`.",
            f"- Real hard negatives in train: `{manifest['selection']['train_real_hard_negatives']}`.",
            f"- Equation validation rows are positive replay only: `{manifest['selection']['validation_equation_positive_only']}`.",
            "",
            "## Decision",
            "",
            f"- HF GPU allowed: `{str(decision['hf_gpu_allowed']).lower()}`.",
            f"- Decision: `{decision['decision']}`.",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V460 NUMERIC ONE RULE MICRO DATASET START ===", flush=True)
    print("v459_manifest_json =", args.v459_manifest_json, flush=True)
    print("v459_detail_csv =", args.v459_detail_csv, flush=True)
    print("bit_train_jsonl =", args.bit_train_jsonl, flush=True)
    print("bit_val_jsonl =", args.bit_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    for path in [args.v459_manifest_json, args.v459_detail_csv, args.bit_train_jsonl, args.bit_val_jsonl]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v459 = read_json(args.v459_manifest_json)
    expected_detail_sha = str(v459.get("outputs", {}).get("detail_sha256", ""))
    if expected_detail_sha and sha256_file(args.v459_detail_csv) != expected_detail_sha:
        raise RuntimeError("V459 detail CSV hash mismatch")

    detail_rows = read_csv(args.v459_detail_csv)
    hard_rows = [row for row in detail_rows if hard_flag(row)]
    positive_rows = [row for row in detail_rows if not hard_flag(row)]
    if len(hard_rows) < args.min_hard_negatives:
        raise RuntimeError(f"hard negatives below floor: {len(hard_rows)} < {args.min_hard_negatives}")
    val_equation = sorted(positive_rows, key=lambda row: sha256_text(row["id"]))[: args.equation_val_rows]
    val_ids = {row["id"] for row in val_equation}
    train_equation = [row for row in detail_rows if row["id"] not in val_ids]
    bit_train = [row for row in read_jsonl(args.bit_train_jsonl) if row.get("family") == "bit_manipulation"][
        : args.bit_train_rows
    ]
    bit_val = [row for row in read_jsonl(args.bit_val_jsonl) if row.get("family") == "bit_manipulation"][
        : args.bit_val_rows
    ]
    if len(bit_train) < args.bit_train_rows or len(bit_val) < args.bit_val_rows:
        raise RuntimeError("not enough V217 bit replay rows")

    train_rows = [make_equation_row(row, "train") for row in train_equation]
    train_rows.extend(make_bit_replay_row(row, "train", idx) for idx, row in enumerate(bit_train))
    val_rows = [make_equation_row(row, "validation") for row in val_equation]
    val_rows.extend(make_bit_replay_row(row, "validation", idx) for idx, row in enumerate(bit_val))

    train_jsonl = args.output_dir / f"{args.label}_train.jsonl"
    val_jsonl = args.output_dir / f"{args.label}_val.jsonl"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    report_md = args.output_dir / f"{args.label}.md"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    train_sha = sha256_file(train_jsonl)
    val_sha = sha256_file(val_jsonl)
    selection = {
        "equation_rows_total": len(detail_rows),
        "equation_train_rows": len(train_equation),
        "equation_val_rows": len(val_equation),
        "train_real_hard_negatives": sum(1 for row in train_equation if hard_flag(row)),
        "train_positive_rule_replay": sum(1 for row in train_equation if not hard_flag(row)),
        "validation_equation_positive_only": all(not hard_flag(row) for row in val_equation),
        "bit_train_rows": len(bit_train),
        "bit_val_rows": len(bit_val),
    }
    conditions = {
        "hard_negatives_ge_min": selection["train_real_hard_negatives"] >= args.min_hard_negatives,
        "bit_replay_train_ge_min": selection["bit_train_rows"] >= args.bit_train_rows,
        "bit_replay_val_ge_min": selection["bit_val_rows"] >= args.bit_val_rows,
        "one_rule_risk_acknowledged": args.allow_one_rule_micro_smoke,
    }
    hf_gpu_allowed = all(conditions.values())
    manifest = {
        "schema_version": "kg1_v460_numeric_one_rule_micro_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_public_train": True,
            "weak_full_rows_used_for_training": False,
            "bit_replay_guardrail": True,
            "one_rule_risk": True,
        },
        "inputs": {
            "v459_manifest_json": str(args.v459_manifest_json),
            "v459_manifest_sha256": sha256_file(args.v459_manifest_json),
            "v459_detail_csv": str(args.v459_detail_csv),
            "v459_detail_sha256": sha256_file(args.v459_detail_csv),
            "bit_train_jsonl": str(args.bit_train_jsonl),
            "bit_train_sha256": sha256_file(args.bit_train_jsonl),
            "bit_val_jsonl": str(args.bit_val_jsonl),
            "bit_val_sha256": sha256_file(args.bit_val_jsonl),
        },
        "selection": selection,
        "summary": {"train": summarize(train_rows), "validation": summarize(val_rows)},
        "conditions": conditions,
        "decision": {
            "hf_gpu_allowed": hf_gpu_allowed,
            "decision": "v460_allows_one_rule_micro_smoke" if hf_gpu_allowed else "v460_blocks_gpu_one_rule_risk_not_acknowledged",
            "blocking_conditions": [key for key, value in conditions.items() if not value],
            "next_action": (
                "Upload dataset and run one H200 micro-smoke with first-checkpoint weak gate."
                if hf_gpu_allowed
                else "Run tokenization gate, then require explicit one-rule micro-smoke risk acceptance before paid GPU."
            ),
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": train_sha,
            "val_jsonl": str(val_jsonl),
            "val_sha256": val_sha,
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_json, manifest)
    report_md.write_text(render_report(manifest), encoding="utf-8")
    manifest["outputs"]["manifest_sha256"] = sha256_file(manifest_json)
    manifest["outputs"]["report_sha256"] = sha256_file(report_md)
    write_json(manifest_json, manifest)
    print("selection =", json.dumps(selection, sort_keys=True), flush=True)
    print("summary =", json.dumps(manifest["summary"], sort_keys=True), flush=True)
    print("conditions =", json.dumps(conditions, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V460 NUMERIC ONE RULE MICRO DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    assert boxed("a{b}") == "\\boxed{a\\{b\\}}"
    assert hard_flag({"real_hard_negative_candidate": "true"})
    assert not hard_flag({"real_hard_negative_candidate": "false"})
    print("v460_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v459-manifest-json", type=Path, default=DEFAULT_V459_DIR / "v459_v458_numeric_hard_negative_audit_manifest.json")
    parser.add_argument("--v459-detail-csv", type=Path, default=DEFAULT_V459_DIR / "v459_v458_numeric_hard_negative_audit_detail.csv")
    parser.add_argument("--bit-train-jsonl", type=Path, default=DEFAULT_BIT_TRAIN)
    parser.add_argument("--bit-val-jsonl", type=Path, default=DEFAULT_BIT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / utc_compact())
    parser.add_argument("--label", default="v460_numeric_one_rule_micro_dataset")
    parser.add_argument("--min-hard-negatives", type=int, default=7)
    parser.add_argument("--equation-val-rows", type=int, default=4)
    parser.add_argument("--bit-train-rows", type=int, default=128)
    parser.add_argument("--bit-val-rows", type=int, default=32)
    parser.add_argument("--allow-one-rule-micro-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

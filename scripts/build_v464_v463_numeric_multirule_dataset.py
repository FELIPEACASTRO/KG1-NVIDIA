#!/usr/bin/env python3
"""Build V464 multi-rule numeric dataset proposal from V463 signal.

V464 consumes V463 audited synthetic numeric hard negatives and V217 bit replay
rows. It creates a compact SFT dataset proposal only after adapter raw outputs
proved real multi-rule mistakes. It does not launch HF, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import verify_answer  # noqa: E402


VERSION = "v464_v463_numeric_multirule_dataset"
DEFAULT_V463_DIR = REPO_ROOT / "artifacts/v463_v462_synthetic_numeric_hard_negative_audit/20260515T_cpu_gate"
DEFAULT_BIT_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
DEFAULT_BIT_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v464_v463_numeric_multirule_dataset/20260515T_cpu_gate"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify compactly, and end with exactly one final answer in \\boxed{}."
)

ADD_RULE = "v274_guarded_numeric_add_direct_over_model_add_variant"
COLON_RULE = "v274_guarded_numeric_colon_absdiff_restore_trailing_zero"
MINUS_DIRECT_RULE = "v274_guarded_numeric_minus_direct_negative_restore_sign"
MINUS_SIGNED_RULE = "v274_guarded_numeric_minus_signed_opposite_sign_guarded"


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


def normalize_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


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


def flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def hard_flag(row: dict[str, str]) -> bool:
    return flag(row, "real_hard_negative_candidate")


def rule_explanation(row: dict[str, str]) -> str:
    rule = str(row["target_rule_class"])
    answer = str(row["answer"])
    wrong = str(row.get("adapter_prediction") or row.get("simulated_wrong_prediction") or "")
    query = str(row.get("query", ""))
    if rule == ADD_RULE:
        return (
            f"For {query}, use the direct operands in the query. "
            f"The reversed-operand/reversed-result candidate {wrong!r} fits the symmetric examples but not the target. "
            f"The direct addition result is {answer}."
        )
    if rule == COLON_RULE:
        return (
            f"For {query}, ':' maps to absolute difference. Preserve a trailing zero when the computed result has one. "
            f"The shortened candidate {wrong!r} is rejected; the verified result is {answer}."
        )
    if rule == MINUS_DIRECT_RULE:
        return (
            f"For {query}, '-' is direct subtraction. Preserve the negative sign when the result is below zero. "
            f"The unsigned candidate {wrong!r} is rejected; the verified result is {answer}."
        )
    if rule == MINUS_SIGNED_RULE:
        return (
            f"For {query}, compare the minus examples and keep the signed rule output. "
            f"The opposite-sign candidate {wrong!r} is rejected; the verified result is {answer}."
        )
    return f"Reject candidate {wrong!r}; verified result is {answer}."


def make_equation_row(row: dict[str, str], split: str, role: str) -> dict[str, Any]:
    answer = str(row["answer"])
    if not verify_answer(answer, row.get("postprocessor_prediction", "")):
        raise RuntimeError(f"postprocessor answer mismatch for {row.get('id')}")
    if hard_flag(row) and verify_answer(answer, row.get("adapter_prediction", "")):
        raise RuntimeError(f"hard row adapter is already correct for {row.get('id')}")
    assistant = f"Verification: {rule_explanation(row)}\nFinal answer: {boxed(answer)}"
    metadata = {
        "source": VERSION,
        "source_dataset": VERSION,
        "source_role": role,
        "family": "equation_transform",
        "rule_class": row["target_rule_class"],
        "target_rule_class": row["target_rule_class"],
        "v463_source_id": row["id"],
        "adapter_prediction": row.get("adapter_prediction", ""),
        "simulated_wrong_prediction": row.get("simulated_wrong_prediction", ""),
        "postprocessor_prediction": row.get("postprocessor_prediction", ""),
        "real_hard_negative_candidate": hard_flag(row),
        "raw_output_collected_without_labels": True,
        "labels_joined_after_collection_from_local_synthetic_audit": True,
        "weak_gate_rows_used_for_training": False,
        "full_gate_rows_used_for_training": False,
        "gate_rows_used_for_training": False,
        "split": split,
    }
    return {
        "id": f"v464_{split}_equation_{row['id']}",
        "family": "equation_transform",
        "subcategory": row["target_rule_class"],
        "source": VERSION,
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
    assistant = f"Verification: preserve the learned bit rule and return only the final 8-bit output.\nFinal answer: {boxed(answer)}"
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source": "v217_bit_replay_guardrail",
            "source_dataset": "v217_bit_replay_guardrail",
            "source_role": "bit_guardrail_replay",
            "v464_split": split,
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
        }
    )
    return {
        "id": f"v464_{split}_bit_replay_{index:04d}_{row.get('id', '')}",
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


def stable_take(rows: list[dict[str, str]], count: int, salt: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ordered = sorted(rows, key=lambda row: sha256_text(salt + "\0" + row["id"]))
    selected = ordered[:count]
    selected_ids = {row["id"] for row in selected}
    remaining = [row for row in rows if row["id"] not in selected_ids]
    return selected, remaining


def split_equation_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_rule: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_rule[row["target_rule_class"]].append(row)

    validation: list[dict[str, str]] = []
    training: list[dict[str, str]] = []
    for rule, rule_rows in sorted(by_rule.items()):
        hard_rows = [row for row in rule_rows if hard_flag(row)]
        positive_rows = [row for row in rule_rows if not hard_flag(row)]
        hard_val_count = 0
        if len(hard_rows) >= 8:
            hard_val_count = 2
        elif len(hard_rows) >= 4:
            hard_val_count = 1
        positive_val_count = min(3, max(0, len(positive_rows) // 4))
        hard_val, hard_train = stable_take(hard_rows, hard_val_count, "hard")
        positive_val, positive_train = stable_take(positive_rows, positive_val_count, "positive")
        validation.extend(hard_val)
        validation.extend(positive_val)
        training.extend(hard_train)
        training.extend(positive_train)

    train_prompt_keys = {sha256_text(normalize_prompt(row["prompt"])) for row in training}
    val_prompt_keys = {sha256_text(normalize_prompt(row["prompt"])) for row in validation}
    overlap = train_prompt_keys & val_prompt_keys
    if overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(overlap)}")
    return training, validation


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row.get("family", "")) for row in rows)
    subcat = Counter(str(row.get("subcategory", "")) for row in rows)
    source_role = Counter(str((row.get("metadata") or {}).get("source_role", "")) for row in rows)
    hard = sum(1 for row in rows if (row.get("metadata") or {}).get("real_hard_negative_candidate") is True)
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(family.items())),
        "subcategory_counts": dict(sorted(subcat.items())),
        "source_role_counts": dict(sorted(source_role.items())),
        "real_hard_negative_rows": hard,
    }


def validate_dataset_rows(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    all_rows = train_rows + val_rows
    ids = [str(row.get("id", "")) for row in all_rows]
    if len(ids) != len(set(ids)):
        issues.append("duplicate_ids")
    train_prompt_keys = {sha256_text(normalize_prompt(row.get("prompt", ""))) for row in train_rows}
    val_prompt_keys = {sha256_text(normalize_prompt(row.get("prompt", ""))) for row in val_rows}
    if train_prompt_keys & val_prompt_keys:
        issues.append("train_val_prompt_overlap")
    for row in all_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or [msg.get("role") for msg in messages] != ["system", "user", "assistant"]:
            issues.append(f"bad_messages:{row.get('id')}")
            continue
        if messages[1].get("content") != row.get("prompt"):
            issues.append(f"user_prompt_mismatch:{row.get('id')}")
        answer = str(row.get("answer", ""))
        assistant = str(messages[2].get("content", ""))
        if not assistant.rstrip().endswith(boxed(answer)):
            issues.append(f"assistant_not_boxed_suffix:{row.get('id')}")
    return issues


def render_report(manifest: dict[str, Any]) -> str:
    train = manifest["summary"]["train"]
    val = manifest["summary"]["validation"]
    decision = manifest["decision"]
    return "\n".join(
        [
            "# V464 V463 Numeric Multi-Rule Dataset",
            "",
            "## Purpose",
            "",
            "CPU dataset proposal from V463 multi-rule real adapter mistakes, plus V217 bit replay guardrail.",
            "",
            "## Counts",
            "",
            f"- Train rows: `{train['rows']}`; families: `{train['family_counts']}`.",
            f"- Validation rows: `{val['rows']}`; families: `{val['family_counts']}`.",
            f"- Train hard negatives: `{train['real_hard_negative_rows']}`.",
            f"- Validation hard negatives: `{val['real_hard_negative_rows']}`.",
            f"- Rule classes in train hard negatives: `{manifest['selection']['train_hard_negative_rule_classes']}`.",
            "",
            "## Decision",
            "",
            f"- Tokenization gate required: `{str(decision['tokenization_gate_required']).lower()}`.",
            f"- HF GPU allowed: `{str(decision['hf_gpu_allowed']).lower()}`.",
            f"- Decision: `{decision['decision']}`.",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V464 V463 NUMERIC MULTIRULE DATASET START ===", flush=True)
    print("v463_manifest_json =", args.v463_manifest_json, flush=True)
    print("v463_detail_csv =", args.v463_detail_csv, flush=True)
    print("bit_train_jsonl =", args.bit_train_jsonl, flush=True)
    print("bit_val_jsonl =", args.bit_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    for path in [args.v463_manifest_json, args.v463_detail_csv, args.bit_train_jsonl, args.bit_val_jsonl]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v463 = read_json(args.v463_manifest_json)
    if v463.get("schema_version") != "kg1_v463_v462_synthetic_numeric_hard_negative_audit_v1":
        raise RuntimeError("unexpected V463 manifest schema")
    expected_detail_sha = str(v463.get("outputs", {}).get("detail_sha256", ""))
    if expected_detail_sha and sha256_file(args.v463_detail_csv) != expected_detail_sha:
        raise RuntimeError("V463 detail CSV hash mismatch")
    if not v463.get("decision", {}).get("v464_dataset_build_allowed"):
        raise RuntimeError("V463 did not allow V464 dataset build")

    detail_rows = read_csv(args.v463_detail_csv)
    hard_rows = [row for row in detail_rows if hard_flag(row)]
    hard_rule_classes = sorted({row["target_rule_class"] for row in hard_rows})
    if len(hard_rows) < args.min_hard_negatives:
        raise RuntimeError(f"hard negatives below floor: {len(hard_rows)} < {args.min_hard_negatives}")
    if len(hard_rule_classes) < args.min_rule_classes:
        raise RuntimeError(f"hard-negative rule classes below floor: {len(hard_rule_classes)} < {args.min_rule_classes}")

    train_equation, val_equation = split_equation_rows(detail_rows)
    bit_train = [row for row in read_jsonl(args.bit_train_jsonl) if row.get("family") == "bit_manipulation"][
        : args.bit_train_rows
    ]
    bit_val = [row for row in read_jsonl(args.bit_val_jsonl) if row.get("family") == "bit_manipulation"][
        : args.bit_val_rows
    ]
    if len(bit_train) < args.bit_train_rows or len(bit_val) < args.bit_val_rows:
        raise RuntimeError("not enough V217 bit replay rows")

    train_rows = [
        make_equation_row(row, "train", "equation_real_hard_negative" if hard_flag(row) else "equation_rule_replay")
        for row in train_equation
    ]
    train_rows.extend(make_bit_replay_row(row, "train", idx) for idx, row in enumerate(bit_train))
    val_rows = [
        make_equation_row(row, "validation", "equation_real_hard_negative" if hard_flag(row) else "equation_rule_replay")
        for row in val_equation
    ]
    val_rows.extend(make_bit_replay_row(row, "validation", idx) for idx, row in enumerate(bit_val))

    validation_issues = validate_dataset_rows(train_rows, val_rows)
    train_hard_rules = sorted(
        {
            str((row.get("metadata") or {}).get("target_rule_class", ""))
            for row in train_rows
            if (row.get("metadata") or {}).get("real_hard_negative_candidate") is True
        }
    )
    conditions = {
        "validation_issues_absent": not validation_issues,
        "train_hard_negatives_ge_min": sum(1 for row in train_rows if (row.get("metadata") or {}).get("real_hard_negative_candidate") is True)
        >= args.min_train_hard_negatives,
        "train_hard_negative_rule_classes_ge_min": len(train_hard_rules) >= args.min_rule_classes,
        "bit_replay_train_ge_min": len(bit_train) >= args.bit_train_rows,
        "bit_replay_val_ge_min": len(bit_val) >= args.bit_val_rows,
        "hf_gpu_blocked_until_tokenization_gate": True,
    }

    train_jsonl = args.output_dir / f"{args.label}_train.jsonl"
    val_jsonl = args.output_dir / f"{args.label}_val.jsonl"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"
    report_md = args.output_dir / f"{args.label}.md"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    hf_gpu_allowed = False
    dataset_ready_for_tokenization = all(conditions.values())
    manifest = {
        "schema_version": "kg1_v464_v463_numeric_multirule_dataset_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "source_policy": {
            "raw_outputs_collected_without_labels": True,
            "labels_joined_after_collection_from_local_synthetic_audit": True,
            "weak_full_rows_used_for_training": False,
            "bit_replay_guardrail": True,
            "training": False,
            "submission": False,
        },
        "inputs": {
            "v463_manifest_json": str(args.v463_manifest_json),
            "v463_manifest_sha256": sha256_file(args.v463_manifest_json),
            "v463_detail_csv": str(args.v463_detail_csv),
            "v463_detail_sha256": sha256_file(args.v463_detail_csv),
            "bit_train_jsonl": str(args.bit_train_jsonl),
            "bit_train_sha256": sha256_file(args.bit_train_jsonl),
            "bit_val_jsonl": str(args.bit_val_jsonl),
            "bit_val_sha256": sha256_file(args.bit_val_jsonl),
        },
        "thresholds": {
            "min_hard_negatives": args.min_hard_negatives,
            "min_train_hard_negatives": args.min_train_hard_negatives,
            "min_rule_classes": args.min_rule_classes,
            "bit_train_rows": args.bit_train_rows,
            "bit_val_rows": args.bit_val_rows,
        },
        "selection": {
            "equation_rows_total": len(detail_rows),
            "equation_train_rows": len(train_equation),
            "equation_val_rows": len(val_equation),
            "hard_negatives_total": len(hard_rows),
            "hard_negative_rule_classes_total": hard_rule_classes,
            "train_hard_negatives": sum(
                1 for row in train_equation if hard_flag(row)
            ),
            "validation_hard_negatives": sum(1 for row in val_equation if hard_flag(row)),
            "train_hard_negative_rule_classes": train_hard_rules,
            "bit_train_rows": len(bit_train),
            "bit_val_rows": len(bit_val),
        },
        "summary": {"train": summarize(train_rows), "validation": summarize(val_rows)},
        "validation_issues": validation_issues,
        "conditions": conditions,
        "decision": {
            "dataset_ready_for_tokenization_gate": dataset_ready_for_tokenization,
            "tokenization_gate_required": True,
            "hf_gpu_allowed": hf_gpu_allowed,
            "decision": (
                "v464_dataset_ready_for_tokenization_gate"
                if dataset_ready_for_tokenization
                else "v464_dataset_blocked"
            ),
            "blocking_conditions": [key for key, value in conditions.items() if not value],
            "next_action": (
                "Run V286 generic tokenization gate with boxed_suffix mode; only then consider V465 one-checkpoint HF smoke."
                if dataset_ready_for_tokenization
                else "Fix dataset validation issues before tokenization or training."
            ),
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "val_sha256": sha256_file(val_jsonl),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
    }
    write_json(manifest_json, manifest)
    report_md.write_text(render_report(manifest), encoding="utf-8")
    manifest["outputs"]["report_sha256"] = sha256_file(report_md)
    write_json(manifest_json, manifest)
    print("selection =", json.dumps(manifest["selection"], sort_keys=True), flush=True)
    print("summary =", json.dumps(manifest["summary"], sort_keys=True), flush=True)
    print("conditions =", json.dumps(conditions, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V464 V463 NUMERIC MULTIRULE DATASET END ===", flush=True)
    return manifest


def self_test() -> None:
    assert boxed("a{b}") == "\\boxed{a\\{b\\}}"
    assert hard_flag({"real_hard_negative_candidate": "true"})
    assert not hard_flag({"real_hard_negative_candidate": "false"})
    row = {
        "id": "x",
        "target_rule_class": ADD_RULE,
        "query": "94)40",
        "answer": "134",
        "adapter_prediction": "35",
        "simulated_wrong_prediction": "35",
        "postprocessor_prediction": "134",
        "prompt": "p",
        "real_hard_negative_candidate": "true",
    }
    item = make_equation_row(row, "train", "equation_real_hard_negative")
    assert item["messages"][2]["content"].endswith("\\boxed{134}")
    assert "35" in item["messages"][2]["content"]
    print("v464_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--v463-manifest-json", type=Path, default=DEFAULT_V463_DIR / "v463_v462_synthetic_numeric_hard_negative_audit_manifest.json")
    parser.add_argument("--v463-detail-csv", type=Path, default=DEFAULT_V463_DIR / "v463_v462_synthetic_numeric_hard_negative_audit_detail.csv")
    parser.add_argument("--bit-train-jsonl", type=Path, default=DEFAULT_BIT_TRAIN)
    parser.add_argument("--bit-val-jsonl", type=Path, default=DEFAULT_BIT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=VERSION)
    parser.add_argument("--min-hard-negatives", type=int, default=20)
    parser.add_argument("--min-train-hard-negatives", type=int, default=20)
    parser.add_argument("--min-rule-classes", type=int, default=3)
    parser.add_argument("--bit-train-rows", type=int, default=512)
    parser.add_argument("--bit-val-rows", type=int, default=128)
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

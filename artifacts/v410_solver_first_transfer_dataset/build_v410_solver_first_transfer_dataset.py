#!/usr/bin/env python3
"""Build V410 solver-first transfer dataset.

V410 extends V406 by adding synthetic traces for the V408 asymmetric per-bit
bit rule that produced the new CPU gain on row 4ef88f92. It still does not train
on weak/full gate rows.
"""

from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V406_SCRIPT = REPO_ROOT / "artifacts/v406_solver_first_transfer_dataset/build_v406_solver_first_transfer_dataset.py"
OUT_DIR = REPO_ROOT / "artifacts/v410_solver_first_transfer_dataset/20260514T_v410_solver_first_transfer"


def load_v406_module():
    spec = importlib.util.spec_from_file_location("kg1_v406_builder", V406_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {V406_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v406 = load_v406_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bit_list(value: str) -> list[int]:
    return [int(ch) for ch in value]


def xnor(a: int, b: int) -> int:
    return 1 - (a ^ b)


def impl(a: int, b: int) -> int:
    return (1 - a) | b


def v408_perbit_4ef88f92_rule(value: str) -> str:
    b = bit_list(value)
    out = [
        xnor(b[1], b[2]),
        xnor(b[2], b[3]),
        xnor(b[3], b[4]),
        xnor(b[4], b[5]),
        1 - b[2],
        impl(b[0], b[1]),
        b[2] | b[4],
        impl(b[0], b[3]),
    ]
    return "".join(str(bit) for bit in out)


BIT_RULES: dict[str, tuple[str, Any, str, list[str]]] = {
    "v403_or_rol2_shl4": (
        "OR(ROL2(input), SHL4(input))",
        lambda x: v406.bit_or(v406.rol(x, 2), v406.shl(x, 4)),
        "v410_v403_exact_global_bit_rule",
        ["4ada9150"],
    ),
    "v403_xor_shl1_shr4": (
        "XOR(SHL1(input), SHR4(input))",
        lambda x: v406.bit_xor(v406.shl(x, 1), v406.shr(x, 4)),
        "v410_v403_exact_global_bit_rule",
        ["4c327b55"],
    ),
    "v408_perbit_asym_4ef88f92": (
        "per-bit: [XNOR(1,2), XNOR(2,3), XNOR(3,4), XNOR(4,5), NOT(2), IMPL(0,1), OR(2,4), IMPL(0,3)]",
        v408_perbit_4ef88f92_rule,
        "v410_v408_perbit_asym_rule",
        ["4ef88f92"],
    ),
}


def build_bit_rule_rows(*, split: str, rows_per_rule: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for rule_name, (rule_text, func, source, gain_ids) in BIT_RULES.items():
        seen_prompts: set[str] = set()
        row_index = 0
        while row_index < rows_per_rule:
            inputs: list[str] = []
            while len(inputs) < 10:
                value = v406.rand_byte(rng)
                if value not in inputs:
                    inputs.append(value)
            examples = [(value, func(value)) for value in inputs[:9]]
            query = inputs[9]
            answer = func(query)
            prompt = v406.build_bit_prompt(examples, query)
            p_hash = v406.sha256_text(v406.normalize_prompt(prompt))
            if p_hash in seen_prompts:
                continue
            seen_prompts.add(p_hash)
            assistant = (
                f"Rule: output = {rule_text}.\n"
                "Check examples: every listed input-output pair matches the rule.\n"
                f"{v406.boxed_answer(answer)}"
            )
            rows.append(
                {
                    "answer": answer,
                    "family": "bit_manipulation",
                    "id": f"v410_{split}_{rule_name}_{row_index:04d}",
                    "messages": [
                        {"role": "system", "content": v406.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "metadata": {
                        "source": source,
                        "rule_name": rule_name,
                        "rule_text": rule_text,
                        "weak_gain_ids": gain_ids,
                        "weak_gate_rows_used_for_training": False,
                        "full_gate_rows_used_for_training": False,
                    },
                    "prompt": prompt,
                    "source": source,
                    "subcategory": "bit_solver_first_v410",
                }
            )
            row_index += 1
    return rows


def main() -> int:
    print("=== V410 SOLVER FIRST TRANSFER DATASET START ===", flush=True)
    refs = [v406.reference_fingerprints(v406.WEAK_REF), v406.reference_fingerprints(v406.FULL_REF)]
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []

    train.extend(v406.normalize_row(row, origin="v410_v390_v325_equation_numeric", split="train") for row in v406.read_jsonl(v406.V325_TRAIN))
    val.extend(v406.normalize_row(row, origin="v410_v390_v325_equation_numeric", split="validation") for row in v406.read_jsonl(v406.V325_VAL))
    train.extend(v406.normalize_row(row, origin="v410_v330_symbolic_cryptarithm", split="train") for row in v406.read_jsonl(v406.V330_TRAIN))
    val.extend(v406.normalize_row(row, origin="v410_v330_symbolic_cryptarithm", split="validation") for row in v406.read_jsonl(v406.V330_VAL))
    train.extend(build_bit_rule_rows(split="train", rows_per_rule=256, seed=410))
    val.extend(build_bit_rule_rows(split="validation", rows_per_rule=64, seed=1410))
    train.extend(v406.select_v217_bit_replay(v406.V217_TRAIN, split="train", count=512, seed=2410, refs=refs))
    val.extend(v406.select_v217_bit_replay(v406.V217_VAL, split="validation", count=128, seed=3410, refs=refs))

    random.Random(4410).shuffle(train)
    random.Random(5410).shuffle(val)

    train_summary = v406.audit_rows(train, label="train", refs=refs)
    val_summary = v406.audit_rows(val, label="validation", refs=refs)
    if train_summary["duplicate_ids"] or train_summary["duplicate_prompts"] or val_summary["duplicate_ids"] or val_summary["duplicate_prompts"]:
        raise RuntimeError("duplicate rows in V410 dataset")
    if train_summary["reference_id_overlap"] or train_summary["reference_prompt_overlap"] or val_summary["reference_id_overlap"] or val_summary["reference_prompt_overlap"]:
        raise RuntimeError("reference overlap in V410 dataset")
    if train_summary["bad_rows_first10"] or val_summary["bad_rows_first10"]:
        raise RuntimeError("bad rows in V410 dataset")
    if {str(row.get("id", "")) for row in train} & {str(row.get("id", "")) for row in val}:
        raise RuntimeError("train/validation id overlap in V410 dataset")
    if {v406.prompt_hash(row) for row in train} & {v406.prompt_hash(row) for row in val}:
        raise RuntimeError("train/validation prompt overlap in V410 dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "v410_solver_first_transfer_train.jsonl"
    val_path = OUT_DIR / "v410_solver_first_transfer_val.jsonl"
    manifest_path = OUT_DIR / "v410_solver_first_transfer_manifest.json"
    comparison_path = OUT_DIR / "V410_VS_PREVIOUS.md"
    v406.write_jsonl(train_path, train)
    v406.write_jsonl(val_path, val)

    manifest = {
        "schema_version": "kg1_v410_solver_first_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "source_policy": {
            "weak_or_full_gate_rows_used_for_training": False,
            "v409_accepted_rows_used_as_training": False,
            "v409_accepted_rows_used_as_rule_evidence_only": True,
            "adapter_training_authorization": "blocked_until_real_tokenization_gate",
        },
        "v409_projection": {
            "baseline": {"total": "192/315", "equation_transform": "56/155", "bit_manipulation": "136/160"},
            "cpu_solver_projection": {"total": "202/315", "equation_transform": "63/155", "bit_manipulation": "139/160"},
            "accepted_csv": str(REPO_ROOT / "artifacts/v409_integrated_solver_projection_v2/20260514T_v409_integrated_projection_v2/v409_integrated_solver_accepted.csv"),
        },
        "bit_rules": {
            name: {"rule_text": rule_text, "source": source, "weak_gain_ids": gain_ids}
            for name, (rule_text, _func, source, gain_ids) in BIT_RULES.items()
        },
        "inputs": {
            "v325_train": str(v406.V325_TRAIN),
            "v325_train_sha256": v406.sha256_file(v406.V325_TRAIN),
            "v325_val": str(v406.V325_VAL),
            "v325_val_sha256": v406.sha256_file(v406.V325_VAL),
            "v330_train": str(v406.V330_TRAIN),
            "v330_train_sha256": v406.sha256_file(v406.V330_TRAIN),
            "v330_val": str(v406.V330_VAL),
            "v330_val_sha256": v406.sha256_file(v406.V330_VAL),
            "v217_train": str(v406.V217_TRAIN),
            "v217_train_sha256": v406.sha256_file(v406.V217_TRAIN),
            "v217_val": str(v406.V217_VAL),
            "v217_val_sha256": v406.sha256_file(v406.V217_VAL),
        },
        "reference_summary": [
            {key: value for key, value in ref.items() if key not in {"ids", "prompt_hashes"}} for ref in refs
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": v406.sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": v406.sha256_file(val_path),
            "manifest_json": str(manifest_path),
            "comparison_md": str(comparison_path),
        },
        "required_next_gate": [
            "python scripts/run_v286_generic_tokenization_gate.py --dataset-manifest-json <manifest> --assistant-final-answer-mode boxed_suffix --min-train-rows 2000 --min-val-rows 550",
            "Only then launch a short HF/Kaggle smoke; first checkpoint kill-switch total>192 equation>56 bit>=136 truncated=0.",
        ],
    }
    v406.write_json(manifest_path, manifest)

    comparison = [
        "# V410 Vs Previous",
        "",
        "| Item | V406 | V410 |",
        "|---|---:|---:|",
        "| Train rows | `2064` | `{}` |".format(train_summary["rows"]),
        "| Val rows | `516` | `{}` |".format(val_summary["rows"]),
        "| CPU projection basis | `201/315`, bit `138` | `202/315`, bit `139` |",
        "| Bit exact-global rules | `2` | `2` |",
        "| Bit asymmetric per-bit rule | `0` | `1` (`4ef88f92`) |",
        "| Weak/full rows used for train | `0` | `0` |",
        "",
        "V410 is not a submit artifact. It is the transfer dataset for the V409 CPU projection.",
    ]
    comparison_path.write_text("\n".join(comparison) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V410 SOLVER FIRST TRANSFER DATASET END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


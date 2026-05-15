#!/usr/bin/env python3
"""Build V406 solver-first transfer dataset.

V406 is intentionally conservative:
- it does not train on weak/full gate rows;
- it reuses already-gated synthetic equation rows from V390/V325 and V330;
- it adds synthetic rows for the two V403 exact global bit rules;
- it includes a small V217 bit replay guardrail.

This is a dataset/gate artifact only. It does not launch HF, package, or submit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts/v406_solver_first_transfer_dataset/20260514T_v406_solver_first_transfer"

V325_TRAIN = (
    REPO_ROOT
    / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/"
    / "v390_v325_equation_no_loss_distill_sft_train.jsonl"
)
V325_VAL = (
    REPO_ROOT
    / "artifacts/v390_v325_equation_no_loss_distill_dataset/20260514T193847Z/"
    / "v390_v325_equation_no_loss_distill_sft_val.jsonl"
)
V330_TRAIN = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_sft_train.jsonl"
)
V330_VAL = (
    REPO_ROOT
    / "artifacts/v330_symbolic_cryptarithm_distill_dataset/20260513T_cpu_gate/"
    / "v330_symbolic_cryptarithm_distill_sft_val.jsonl"
)
V217_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
V217_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
V405_ACCEPTED = (
    REPO_ROOT
    / "artifacts/v405_integrated_solver_projection/20260514T_v405_integrated_projection/"
    / "v405_integrated_solver_accepted.csv"
)
WEAK_REF = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
FULL_REF = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)
FINAL_RE = re.compile(r"(?m)^Final answer:\s*.*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def prompt_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reference_fingerprints(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = str(row.get("id", "")).strip()
            if rid:
                ids.add(rid)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                prompts.add(sha256_text(normalize_prompt(prompt)))
    return {
        "path": str(path),
        "rows": len(ids),
        "sha256": sha256_file(path),
        "ids": ids,
        "prompt_hashes": prompts,
    }


def assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise RuntimeError("row has no assistant message")


def boxed_answer(answer: Any) -> str:
    return r"Final answer: \boxed{" + str(answer) + "}"


def normalize_row(row: dict[str, Any], *, origin: str, split: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    answer = str(out.get("answer", ""))
    messages = out.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("missing messages for " + str(out.get("id", "")))
    idx = assistant_index(messages)
    content = str(messages[idx].get("content", "")).rstrip()
    final = boxed_answer(answer)
    if FINAL_RE.search(content):
        content = FINAL_RE.sub(lambda _match: final, content)
    else:
        content = (content + "\n" if content else "") + final
    messages[idx]["content"] = content
    out["messages"] = messages
    out["source"] = origin
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "v406_origin": origin,
            "v406_split": split,
            "v406_final_answer_format": "boxed_suffix",
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
        }
    )
    out["metadata"] = metadata
    return out


def bit_to_list(value: str) -> list[int]:
    return [int(ch) for ch in value]


def list_to_bit(bits: list[int]) -> str:
    return "".join(str(bit) for bit in bits)


def rol(value: str, amount: int) -> str:
    bits = bit_to_list(value)
    return list_to_bit(bits[amount:] + bits[:amount])


def shl(value: str, amount: int) -> str:
    bits = bit_to_list(value)
    return list_to_bit(bits[amount:] + [0] * amount)


def shr(value: str, amount: int) -> str:
    bits = bit_to_list(value)
    return list_to_bit([0] * amount + bits[:-amount])


def bit_or(a: str, b: str) -> str:
    return list_to_bit([x | y for x, y in zip(bit_to_list(a), bit_to_list(b))])


def bit_xor(a: str, b: str) -> str:
    return list_to_bit([x ^ y for x, y in zip(bit_to_list(a), bit_to_list(b))])


BIT_RULES: dict[str, tuple[str, Callable[[str], str]]] = {
    "v403_or_rol2_shl4": ("OR(ROL2(input), SHL4(input))", lambda x: bit_or(rol(x, 2), shl(x, 4))),
    "v403_xor_shl1_shr4": ("XOR(SHL1(input), SHR4(input))", lambda x: bit_xor(shl(x, 1), shr(x, 4))),
}


def rand_byte(rng: random.Random) -> str:
    return format(rng.randrange(0, 256), "08b")


def build_bit_prompt(examples: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{src} -> {dst}" for src, dst in examples)
    return (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n\n"
        "Here are some examples of input -> output:\n"
        f"{body}\n\n"
        f"Now, determine the output for: {query}"
    )


def build_bit_rule_rows(*, split: str, rows_per_rule: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for rule_name, (rule_text, func) in BIT_RULES.items():
        seen_prompts: set[str] = set()
        row_index = 0
        while row_index < rows_per_rule:
            inputs: list[str] = []
            while len(inputs) < 10:
                value = rand_byte(rng)
                if value not in inputs:
                    inputs.append(value)
            examples = [(value, func(value)) for value in inputs[:9]]
            query = inputs[9]
            answer = func(query)
            prompt = build_bit_prompt(examples, query)
            p_hash = sha256_text(normalize_prompt(prompt))
            if p_hash in seen_prompts:
                continue
            seen_prompts.add(p_hash)
            assistant = (
                f"Rule: output = {rule_text}.\n"
                "Check examples: every listed input-output pair matches the rule.\n"
                f"{boxed_answer(answer)}"
            )
            rows.append(
                {
                    "answer": answer,
                    "family": "bit_manipulation",
                    "id": f"v406_{split}_{rule_name}_{row_index:04d}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "metadata": {
                        "source": "v406_v403_exact_global_bit_rule",
                        "rule_name": rule_name,
                        "rule_text": rule_text,
                        "v403_weak_gain_ids": ["4ada9150", "4c327b55"],
                        "weak_gate_rows_used_for_training": False,
                        "full_gate_rows_used_for_training": False,
                    },
                    "prompt": prompt,
                    "source": "v406_v403_exact_global_bit_rule",
                    "subcategory": "bit_exact_global_v403",
                }
            )
            row_index += 1
    return rows


def select_v217_bit_replay(path: Path, *, split: str, count: int, seed: int, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = [row for row in read_jsonl(path) if row.get("family") == "bit_manipulation"]
    rng.shuffle(rows)
    ref_ids = set().union(*(ref["ids"] for ref in refs))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in refs))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("id") in ref_ids or prompt_hash(row) in ref_prompts:
            continue
        selected.append(normalize_row(row, origin="v406_v217_bit_replay_guardrail", split=split))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could only select {len(selected)} V217 bit replay rows for {split}")
    return selected


def audit_rows(rows: list[dict[str, Any]], *, label: str, refs: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [prompt_hash(row) for row in rows]
    ref_ids = set().union(*(ref["ids"] for ref in refs))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in refs))
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    bad_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        family_counts[str(row.get("family", ""))] += 1
        source_counts[str(row.get("source", ""))] += 1
        subcategory_counts[str(row.get("subcategory", ""))] += 1
        messages = row.get("messages")
        answer = str(row.get("answer", ""))
        if not row.get("id") or not row.get("prompt") or not answer or not isinstance(messages, list):
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "missing_required_field"})
            continue
        try:
            assistant = str(messages[assistant_index(messages)].get("content", "")).rstrip()
        except RuntimeError:
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "missing_assistant"})
            continue
        if not assistant.endswith(boxed_answer(answer)):
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "assistant_not_boxed_suffix"})
    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "unique_prompt_hashes": len(set(prompts)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "reference_id_overlap": len(set(ids) & ref_ids),
        "reference_prompt_overlap": len(set(prompts) & ref_prompts),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "bad_rows_first10": bad_rows[:10],
    }


def main() -> int:
    print("=== V406 SOLVER FIRST TRANSFER DATASET START ===", flush=True)
    refs = [reference_fingerprints(WEAK_REF), reference_fingerprints(FULL_REF)]
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []

    train.extend(normalize_row(row, origin="v406_v390_v325_equation_numeric", split="train") for row in read_jsonl(V325_TRAIN))
    val.extend(normalize_row(row, origin="v406_v390_v325_equation_numeric", split="validation") for row in read_jsonl(V325_VAL))
    train.extend(normalize_row(row, origin="v406_v330_symbolic_cryptarithm", split="train") for row in read_jsonl(V330_TRAIN))
    val.extend(normalize_row(row, origin="v406_v330_symbolic_cryptarithm", split="validation") for row in read_jsonl(V330_VAL))
    train.extend(build_bit_rule_rows(split="train", rows_per_rule=256, seed=406))
    val.extend(build_bit_rule_rows(split="validation", rows_per_rule=64, seed=1406))
    train.extend(select_v217_bit_replay(V217_TRAIN, split="train", count=512, seed=2406, refs=refs))
    val.extend(select_v217_bit_replay(V217_VAL, split="validation", count=128, seed=3406, refs=refs))

    random.Random(4406).shuffle(train)
    random.Random(5406).shuffle(val)

    train_summary = audit_rows(train, label="train", refs=refs)
    val_summary = audit_rows(val, label="validation", refs=refs)
    if train_summary["duplicate_ids"] or train_summary["duplicate_prompts"] or val_summary["duplicate_ids"] or val_summary["duplicate_prompts"]:
        raise RuntimeError("duplicate rows in V406 dataset")
    if train_summary["reference_id_overlap"] or train_summary["reference_prompt_overlap"] or val_summary["reference_id_overlap"] or val_summary["reference_prompt_overlap"]:
        raise RuntimeError("reference overlap in V406 dataset")
    if train_summary["bad_rows_first10"] or val_summary["bad_rows_first10"]:
        raise RuntimeError("bad rows in V406 dataset")
    train_prompt_hashes = {prompt_hash(row) for row in train}
    val_prompt_hashes = {prompt_hash(row) for row in val}
    train_ids = {str(row.get("id", "")) for row in train}
    val_ids = {str(row.get("id", "")) for row in val}
    if train_ids & val_ids or train_prompt_hashes & val_prompt_hashes:
        raise RuntimeError("train/validation overlap in V406 dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "v406_solver_first_transfer_train.jsonl"
    val_path = OUT_DIR / "v406_solver_first_transfer_val.jsonl"
    manifest_path = OUT_DIR / "v406_solver_first_transfer_manifest.json"
    comparison_path = OUT_DIR / "V406_VS_PREVIOUS.md"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    manifest = {
        "schema_version": "kg1_v406_solver_first_transfer_dataset_v1",
        "generated_at_utc": utc_now(),
        "source_policy": {
            "weak_or_full_gate_rows_used_for_training": False,
            "v405_accepted_rows_used_as_training": False,
            "v405_accepted_rows_used_as_rule_evidence_only": True,
            "adapter_training_authorization": "blocked_until_real_tokenization_gate",
        },
        "v405_projection": {
            "baseline": {"total": "192/315", "equation_transform": "56/155", "bit_manipulation": "136/160"},
            "cpu_solver_projection": {"total": "201/315", "equation_transform": "63/155", "bit_manipulation": "138/160"},
            "accepted_csv": str(V405_ACCEPTED),
            "accepted_sha256": sha256_file(V405_ACCEPTED),
        },
        "inputs": {
            "v325_train": str(V325_TRAIN),
            "v325_train_sha256": sha256_file(V325_TRAIN),
            "v325_val": str(V325_VAL),
            "v325_val_sha256": sha256_file(V325_VAL),
            "v330_train": str(V330_TRAIN),
            "v330_train_sha256": sha256_file(V330_TRAIN),
            "v330_val": str(V330_VAL),
            "v330_val_sha256": sha256_file(V330_VAL),
            "v217_train": str(V217_TRAIN),
            "v217_train_sha256": sha256_file(V217_TRAIN),
            "v217_val": str(V217_VAL),
            "v217_val_sha256": sha256_file(V217_VAL),
        },
        "reference_summary": [
            {key: value for key, value in ref.items() if key not in {"ids", "prompt_hashes"}} for ref in refs
        ],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "outputs": {
            "train_jsonl": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_jsonl": str(val_path),
            "val_sha256": sha256_file(val_path),
            "manifest_json": str(manifest_path),
            "comparison_md": str(comparison_path),
        },
        "required_next_gate": [
            "python scripts/run_v286_generic_tokenization_gate.py --dataset-manifest-json <manifest> --assistant-final-answer-mode boxed_suffix --min-train-rows 1800 --min-val-rows 500",
            "Only then launch a short HF smoke; first checkpoint kill-switch total>192 equation>56 bit>=136 truncated=0.",
        ],
    }
    write_json(manifest_path, manifest)

    comparison = [
        "# V406 Vs Previous",
        "",
        "| Item | Previous V326/V337 style | V406 solver-first transfer |",
        "|---|---:|---:|",
        "| Train rows | `5031` V326 or `1440` V337D | `{}` |".format(train_summary["rows"]),
        "| Val rows | `532` V326 or `340` V337D | `{}` |".format(val_summary["rows"]),
        "| Equation numeric synthetic | V325/V390 only | kept V390/V325 `800 train / 200 val` |",
        "| Symbolic cryptarithm | absent in V326 | added V330 `240 train / 60 val` |",
        "| New bit exact-global rules | absent | added V403 OR/XOR rules `512 train / 128 val` |",
        "| Bit replay guardrail | broad V304 or V217 replay | compact V217 `512 train / 128 val` |",
        "| Weak/full rows used for train | `0` | `0` |",
        "",
        "V406 is not a submit artifact. It is the smallest responsible adapter-transfer candidate after V405 showed `201/315` CPU solver projection.",
    ]
    comparison_path.write_text("\n".join(comparison) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("=== V406 SOLVER FIRST TRANSFER DATASET END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

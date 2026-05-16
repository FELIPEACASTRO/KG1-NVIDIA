#!/usr/bin/env python3
"""Build the V475 equation no-loss + bit replay transfer dataset.

This is a CPU-only dataset builder. It combines:

- the current V475/V325 equation no-loss distillation rows, whose source CPU
  gate projects equation 56 -> 60 without bit loss; and
- V217 bit_manipulation replay rows, filtered against weak/full references.

The weak/full rows are never used as training data. They are used only as
forbidden reference fingerprints for id/prompt/prompt+answer overlap checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from competition_utils import box_answer  # noqa: E402


OUT_DIR = REPO_ROOT / "artifacts/v475_equation_bit_replay_mix/20260516T_v475_equation_bit_replay_mix"
V325_MANIFEST = (
    REPO_ROOT
    / "artifacts/v475_equation_no_loss_distill_dataset/20260516T_v325_current_baseline/"
    / "v475_v325_equation_no_loss_distill_manifest.json"
)
V325_TOKEN_GATE = (
    REPO_ROOT
    / "artifacts/v475_equation_no_loss_distill_dataset/20260516T_v325_current_baseline/"
    / "tokenization_gate_real/v286_generic_tokenization_gate_manifest.json"
)
V217_TRAIN = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
V217_VAL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
WEAK_REF = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
FULL_REF = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"

BIT_REPLAY_TRAIN_ROWS = 512
BIT_REPLAY_VAL_ROWS = 128
EXPECTED_V325_PROJECTED_EQUATION = 60
EXPECTED_V325_TRAIN_ROWS = 800
EXPECTED_V325_VAL_ROWS = 200


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def normalize_prompt(prompt: Any) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


def normalize_answer(answer: Any) -> str:
    return re.sub(r"\s+", "", str(answer or "")).strip()


def prompt_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")))


def prompt_answer_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_prompt(row.get("prompt", "")) + "\0" + normalize_answer(row.get("answer", "")))


def reference_fingerprints(path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    prompts: set[str] = set()
    prompt_answers: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            rid = str(row.get("id", "") or row.get("row_id", "")).strip()
            prompt = str(row.get("prompt", "") or "")
            answer = str(row.get("answer", "") or "")
            if rid:
                ids.add(rid)
            if prompt:
                prompts.add(sha256_text(normalize_prompt(prompt)))
            if prompt and answer:
                prompt_answers.add(sha256_text(normalize_prompt(prompt) + "\0" + normalize_answer(answer)))
    return {
        "path": str(path),
        "rows": rows,
        "sha256": sha256_file(path),
        "ids": ids,
        "prompt_hashes": prompts,
        "prompt_answer_hashes": prompt_answers,
    }


def assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise RuntimeError("row has no assistant message")


FINAL_RE = re.compile(r"(?m)^Final answer:\s*.*$")


def boxed_final_answer(answer: Any) -> str:
    return "Final answer: " + box_answer(str(answer))


def normalize_row(row: dict[str, Any], *, origin: str, split: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    answer = str(out.get("answer", ""))
    messages = out.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("missing messages for " + str(out.get("id", "")))
    idx = assistant_index(messages)
    content = str(messages[idx].get("content", "")).rstrip()
    final = boxed_final_answer(answer)
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
            "v475_origin": origin,
            "v475_split": split,
            "v475_final_answer_format": "boxed_suffix",
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "gate_rows_used_for_training": False,
        }
    )
    out["metadata"] = metadata
    if out.get("family") == "bit_manipulation" and origin == "v475_v217_bit_replay_guardrail":
        out["subcategory"] = "bit_guardrail_replay"
        metadata["subcategory"] = "bit_guardrail_replay"
    return out


def validate_v325_inputs(manifest_path: Path, token_gate_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "kg1_v325_equation_no_loss_distill_dataset_v1":
        raise RuntimeError("unexpected V325 schema: " + str(manifest.get("schema_version")))
    if int(manifest.get("v324_projected_equation_correct", -1)) != EXPECTED_V325_PROJECTED_EQUATION:
        raise RuntimeError("V325 projected equation is not the current +4 signal")
    train_summary = manifest.get("train_summary") or {}
    val_summary = manifest.get("validation_summary") or {}
    if int(train_summary.get("rows", -1)) != EXPECTED_V325_TRAIN_ROWS:
        raise RuntimeError("unexpected V325 train rows: " + str(train_summary.get("rows")))
    if int(val_summary.get("rows", -1)) != EXPECTED_V325_VAL_ROWS:
        raise RuntimeError("unexpected V325 val rows: " + str(val_summary.get("rows")))
    if train_summary.get("reference_id_overlap") or train_summary.get("reference_prompt_overlap"):
        raise RuntimeError("V325 train reference overlap is non-zero")
    if val_summary.get("reference_id_overlap") or val_summary.get("reference_prompt_overlap"):
        raise RuntimeError("V325 val reference overlap is non-zero")

    token_gate = read_json(token_gate_path)
    decision = token_gate.get("decision") or {}
    if decision.get("status") != "tokenization_gate_passed":
        raise RuntimeError("V325 tokenization gate did not pass: " + str(decision))
    tokenization = token_gate.get("tokenization") or {}
    train_tokenization = tokenization.get("train") or {}
    validation_tokenization = tokenization.get("validation") or {}
    if train_tokenization.get("prompt_truncated") != 0:
        raise RuntimeError("V325 train prompt truncation is non-zero")
    if validation_tokenization.get("prompt_truncated") != 0:
        raise RuntimeError("V325 validation prompt truncation is non-zero")
    if train_tokenization.get("offset_masks") != EXPECTED_V325_TRAIN_ROWS:
        raise RuntimeError("V325 train offset-mask count is not complete")
    if validation_tokenization.get("offset_masks") != EXPECTED_V325_VAL_ROWS:
        raise RuntimeError("V325 validation offset-mask count is not complete")
    return manifest


def select_v217_bit_replay(path: Path, *, split: str, count: int, seed: int, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = [row for row in read_jsonl(path) if row.get("family") == "bit_manipulation"]
    rng.shuffle(rows)
    ref_ids = set().union(*(ref["ids"] for ref in refs))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in refs))
    ref_prompt_answers = set().union(*(ref["prompt_answer_hashes"] for ref in refs))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("id", "")) in ref_ids:
            continue
        if prompt_hash(row) in ref_prompts:
            continue
        if prompt_answer_hash(row) in ref_prompt_answers:
            continue
        selected.append(normalize_row(row, origin="v475_v217_bit_replay_guardrail", split=split))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could only select {len(selected)} V217 bit replay rows for {split}")
    return selected


def audit_rows(rows: list[dict[str, Any]], *, label: str, refs: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    prompts = [prompt_hash(row) for row in rows]
    prompt_answers = [prompt_answer_hash(row) for row in rows]
    ref_ids = set().union(*(ref["ids"] for ref in refs))
    ref_prompts = set().union(*(ref["prompt_hashes"] for ref in refs))
    ref_prompt_answers = set().union(*(ref["prompt_answer_hashes"] for ref in refs))
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
        if row.get("family") not in {"equation_transform", "bit_manipulation"}:
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "unexpected_family"})
        try:
            assistant = str(messages[assistant_index(messages)].get("content", "")).rstrip()
        except RuntimeError:
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "missing_assistant"})
            continue
        if not assistant.endswith(boxed_final_answer(answer)):
            bad_rows.append({"index": index, "id": row.get("id", ""), "reason": "assistant_not_boxed_suffix"})
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for flag in ("gate_rows_used_for_training", "weak_gate_rows_used_for_training", "full_gate_rows_used_for_training"):
            if metadata.get(flag) is not False:
                bad_rows.append({"index": index, "id": row.get("id", ""), "reason": flag + "_not_false"})
    return {
        "label": label,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "unique_prompt_hashes": len(set(prompts)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "unique_prompt_answer_hashes": len(set(prompt_answers)),
        "duplicate_prompt_answers": len(prompt_answers) - len(set(prompt_answers)),
        "reference_id_overlap": len(set(ids) & ref_ids),
        "reference_prompt_overlap": len(set(prompts) & ref_prompts),
        "reference_prompt_answer_overlap": len(set(prompt_answers) & ref_prompt_answers),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "bad_rows_first20": bad_rows[:20],
    }


def main() -> int:
    print("=== V475 EQUATION BIT REPLAY MIX START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v325_manifest =", V325_MANIFEST, flush=True)
    print("v325_token_gate =", V325_TOKEN_GATE, flush=True)
    print("output_dir =", OUT_DIR, flush=True)

    v325_manifest = validate_v325_inputs(V325_MANIFEST, V325_TOKEN_GATE)
    refs = [reference_fingerprints(WEAK_REF), reference_fingerprints(FULL_REF)]

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    train.extend(
        normalize_row(row, origin="v475_v325_equation_no_loss_distill", split="train")
        for row in read_jsonl(Path(v325_manifest["outputs"]["sft_train_jsonl"]))
    )
    val.extend(
        normalize_row(row, origin="v475_v325_equation_no_loss_distill", split="validation")
        for row in read_jsonl(Path(v325_manifest["outputs"]["sft_val_jsonl"]))
    )
    train.extend(select_v217_bit_replay(V217_TRAIN, split="train", count=BIT_REPLAY_TRAIN_ROWS, seed=2475, refs=refs))
    val.extend(select_v217_bit_replay(V217_VAL, split="validation", count=BIT_REPLAY_VAL_ROWS, seed=3475, refs=refs))

    random.Random(4475).shuffle(train)
    random.Random(5475).shuffle(val)

    train_summary = audit_rows(train, label="train", refs=refs)
    val_summary = audit_rows(val, label="validation", refs=refs)
    train_prompt_hashes = {prompt_hash(row) for row in train}
    val_prompt_hashes = {prompt_hash(row) for row in val}
    train_ids = {str(row.get("id", "")) for row in train}
    val_ids = {str(row.get("id", "")) for row in val}
    hard_fail = [
        train_summary["duplicate_ids"],
        train_summary["duplicate_prompts"],
        train_summary["duplicate_prompt_answers"],
        train_summary["reference_id_overlap"],
        train_summary["reference_prompt_overlap"],
        train_summary["reference_prompt_answer_overlap"],
        val_summary["duplicate_ids"],
        val_summary["duplicate_prompts"],
        val_summary["duplicate_prompt_answers"],
        val_summary["reference_id_overlap"],
        val_summary["reference_prompt_overlap"],
        val_summary["reference_prompt_answer_overlap"],
        len(train_ids & val_ids),
        len(train_prompt_hashes & val_prompt_hashes),
    ]
    if any(hard_fail):
        raise RuntimeError("V475 mix overlap/duplicate gate failed: " + json.dumps(hard_fail))
    if train_summary["bad_rows_first20"] or val_summary["bad_rows_first20"]:
        raise RuntimeError("V475 mix bad rows: " + json.dumps([train_summary["bad_rows_first20"], val_summary["bad_rows_first20"]]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "v475_equation_bit_replay_mix_train.jsonl"
    val_path = OUT_DIR / "v475_equation_bit_replay_mix_val.jsonl"
    manifest_path = OUT_DIR / "v475_equation_bit_replay_mix_manifest.json"
    comparison_path = OUT_DIR / "V475_VS_PREVIOUS.md"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    manifest = {
        "schema_version": "kg1_v475_equation_bit_replay_mix_v1",
        "generated_at_utc": utc_now(),
        "source_policy": {
            "weak_or_full_gate_rows_used_for_training": False,
            "v324_accepted_rows_used_as_training": False,
            "v324_accepted_rows_used_as_rule_evidence_only": True,
            "adapter_training_authorization": "blocked_until_combined_real_tokenization_gate_and_hf_preflight",
        },
        "baseline": {
            "weak_total": "192/315",
            "equation_transform": "56/155",
            "bit_manipulation": "136/160",
            "truncated": 0,
        },
        "cpu_projection": {
            "source_manifest": str(V325_MANIFEST),
            "equation_transform": "60/155",
            "weak_total_if_equation_only": "196/315",
            "bit_guardrail_required": ">=136/160",
            "submit_ready": False,
        },
        "inputs": {
            "v325_manifest": str(V325_MANIFEST),
            "v325_manifest_sha256": sha256_file(V325_MANIFEST),
            "v325_token_gate": str(V325_TOKEN_GATE),
            "v325_token_gate_sha256": sha256_file(V325_TOKEN_GATE),
            "v217_train": str(V217_TRAIN),
            "v217_train_sha256": sha256_file(V217_TRAIN),
            "v217_val": str(V217_VAL),
            "v217_val_sha256": sha256_file(V217_VAL),
        },
        "reference_summary": [
            {key: value for key, value in ref.items() if key not in {"ids", "prompt_hashes", "prompt_answer_hashes"}}
            for ref in refs
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
            "python scripts/run_v286_generic_tokenization_gate.py --dataset-manifest-json <manifest> --assistant-final-answer-mode boxed_suffix --min-train-rows 1312 --min-val-rows 328",
            "run scripts/kg1_static_safety_gate.py against changed scripts/artifacts",
            "do not launch HF unless combined tokenization passes with zero truncation and no reference overlap",
            "first HF checkpoint kill-switch: total>192 equation>56 bit>=136 truncated=0",
        ],
    }
    write_json(manifest_path, manifest)

    comparison = [
        "# V475 Vs Previous",
        "",
        "| Item | Previous active state | V475 CPU gated state |",
        "|---|---:|---:|",
        "| Baseline weak | `192/315` | `192/315` |",
        "| CPU equation projection | V324 historic varied; current roadmap had no clean V475 entry | `56 -> 60` from 4 accepted candidates |",
        "| Train rows | no current mixed dataset | `{}` |".format(train_summary["rows"]),
        "| Validation rows | no current mixed dataset | `{}` |".format(val_summary["rows"]),
        "| Equation train/val | V325 current only | `800 / 200` |",
        "| Bit replay train/val | required but not yet combined | `512 / 128` |",
        "| Weak/full rows used for train | `0` | `0` |",
        "| Token gate status | V325-only passed | combined gate still required |",
        "",
        "V475 is not a submit artifact. It is the smallest responsible adapter-transfer candidate after the current CPU gate found `+4` equation signal.",
    ]
    comparison_path.write_text("\n".join(comparison) + "\n", encoding="utf-8")

    print("train_summary =", json.dumps(train_summary, sort_keys=True), flush=True)
    print("validation_summary =", json.dumps(val_summary, sort_keys=True), flush=True)
    print("manifest_json =", manifest_path, flush=True)
    print("comparison_md =", comparison_path, flush=True)
    print("=== V475 EQUATION BIT REPLAY MIX END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

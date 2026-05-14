#!/usr/bin/env python3
"""V381 filtered teacher dataset builder.

This CPU-only gate converts the V380 reexecuted teacher signal into a small
chat JSONL dataset candidate. It intentionally does not launch HF, train,
package, submit, or treat V380 oracle rows as deployable rules.
"""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analyze_v238_alice_parser_probes import parse_alice_prompt  # noqa: E402
from analyze_v380_solver_results_patch_gate import expression_candidates  # noqa: E402


DEFAULT_V380_CANDIDATE_CSV = (
    REPO_ROOT
    / "artifacts/v380_solver_results_patch_gate/20260514T_cpu_gate/v380_solver_results_candidate_patch.csv"
)
DEFAULT_V366_CSV = (
    REPO_ROOT / "artifacts/v366_bit_fullbyte_ternary_op_gate/20260514T_cpu_gate/v366_integrated_predictions.csv"
)
DEFAULT_V217_TRAIN_JSONL = REPO_ROOT / "data/v217/v217_short_answer_train.jsonl"
DEFAULT_V217_VAL_JSONL = REPO_ROOT / "data/v217/v217_short_answer_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/v381_filtered_teacher_dataset/20260514T_cpu_gate"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, then answer with exactly one short final answer."
)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt)).strip()


def prompt_hash(prompt: str) -> str:
    return sha256_text(normalize_prompt(prompt))


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_json_loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return default


def prompt_prefix(prompt: str) -> str:
    marker = "Now, determine the result for:"
    if marker not in prompt:
        raise RuntimeError("prompt missing Alice query marker")
    return prompt.split(marker, 1)[0] + marker + " "


def build_prompt_with_query(prompt: str, query: str) -> str:
    return prompt_prefix(prompt) + query


def deterministic_split(row_id: str, val_mod: int = 5) -> str:
    value = int(sha256_text(row_id)[:8], 16)
    return "val" if value % val_mod == 0 else "train"


def build_messages(prompt: str, answer: str, trace: str = "") -> list[dict[str, str]]:
    assistant = (trace.rstrip() + "\n" if trace else "") + "Final answer: " + answer
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


def make_chat_row(
    *,
    row_id: str,
    prompt: str,
    answer: str,
    subcategory: str,
    source: str,
    metadata: dict[str, Any],
    trace: str = "",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": "equation_transform" if subcategory.startswith("equation") else "bit_manipulation",
        "subcategory": subcategory,
        "source": source,
        "messages": build_messages(prompt, answer, trace),
        "metadata": {
            **metadata,
            "source_dataset": source,
            "weak_gate_rows_used_for_training": False,
        },
    }


def candidate_queries(
    *,
    original_query: str,
    ops: dict[str, str],
    mapping: dict[str, int],
    seed: str,
    per_rule: int,
) -> list[tuple[str, str]]:
    rng = random.Random(int(sha256_text(seed)[:16], 16))
    symbols = sorted(mapping)
    operators = sorted(ops)
    if len(symbols) < 2 or not operators:
        return []
    out: list[tuple[str, str]] = []
    seen = {original_query}
    attempts = 0
    while len(out) < per_rule and attempts < per_rule * 200:
        attempts += 1
        token_len_left = 1 if rng.random() < 0.82 else 2
        token_len_right = 1 if rng.random() < 0.82 else 2
        left = "".join(rng.choice(symbols) for _ in range(token_len_left))
        right = "".join(rng.choice(symbols) for _ in range(token_len_right))
        query = left + rng.choice(operators) + right
        if query in seen:
            continue
        try:
            candidates = expression_candidates(query, ops, mapping, little_endian=False)
        except Exception:
            continue
        if not candidates:
            continue
        answer = candidates[0]
        if not answer or len(answer) > 18:
            continue
        seen.add(query)
        out.append((query, answer))
    return out


def load_v217_replay(path: Path, family: str, limit: int) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("family") == family]
    rows.sort(key=lambda row: str(row.get("id", "")))
    replay: list[dict[str, Any]] = []
    for row in rows[:limit]:
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        replay.append(
            {
                "id": "v381_replay_" + str(row.get("id", "")),
                "prompt": prompt,
                "answer": answer,
                "family": str(row.get("family", family)),
                "subcategory": str(row.get("subcategory", row.get("metadata", {}).get("subcategory", family))),
                "source": "v381_v217_bit_replay",
                "messages": build_messages(prompt, answer),
                "metadata": {
                    "source_dataset": "v381_v217_bit_replay",
                    "original_id": row.get("id"),
                    "weak_gate_rows_used_for_training": False,
                },
            }
        )
    return replay


def audit_overlaps(rows: list[dict[str, Any]], reference: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ref_hashes = {prompt_hash(str(row.get("prompt", ""))) for row in reference}
    overlaps = [str(row.get("id")) for row in rows if prompt_hash(str(row.get("prompt", ""))) in ref_hashes]
    return {
        "reference": name,
        "reference_rows": len(reference),
        "overlap_rows": len(overlaps),
        "overlap_ids_first20": overlaps[:20],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "family_counts": dict(sorted(Counter(str(row.get("family", "")) for row in rows).items())),
        "source_counts": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "subcategory_counts": dict(sorted(Counter(str(row.get("subcategory", "")) for row in rows).items())),
        "unique_ids": len({str(row.get("id", "")) for row in rows}),
        "prompt_hash_count": len({prompt_hash(str(row.get("prompt", ""))) for row in rows}),
        "max_prompt_chars": max((len(str(row.get("prompt", ""))) for row in rows), default=0),
        "max_answer_chars": max((len(str(row.get("answer", ""))) for row in rows), default=0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V381 FILTERED TEACHER DATASET START ===", flush=True)
    print("v380_candidate_csv =", args.v380_candidate_csv, flush=True)
    print("v366_csv =", args.v366_csv, flush=True)
    print("v217_train_jsonl =", args.v217_train_jsonl, flush=True)
    print("v217_val_jsonl =", args.v217_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(args.v380_candidate_csv)
    v366_rows = read_csv(args.v366_csv)
    v366_by_id = {row["id"]: row for row in v366_rows}
    v217_train = read_jsonl(args.v217_train_jsonl)
    v217_val = read_jsonl(args.v217_val_jsonl)
    accepted = [row for row in candidates if bool_text(row.get("accepted_for_v381_teacher"))]

    prototypes: list[dict[str, Any]] = []
    synthetic: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for row in accepted:
        source = v366_by_id.get(row["id"])
        if source is None:
            raise RuntimeError("V380 candidate missing from V366: " + row["id"])
        _examples, original_query, parse_status = parse_alice_prompt(source["prompt"])
        if parse_status != "ok":
            raise RuntimeError(f"failed to parse V366 Alice prompt for {row['id']}: {parse_status}")
        ops = safe_json_loads(row.get("solver_ops", "{}"), {})
        mapping = safe_json_loads(row.get("solver_mapping", "{}"), {})
        if not isinstance(ops, dict) or not isinstance(mapping, dict):
            continue
        mapping = {str(key): int(value) for key, value in mapping.items()}
        original_query = str(original_query)
        trace = (
            "Use the inferred symbolic numeric mapping and apply the operators left to right. "
            f"solver_category={row.get('solver_category')}; operations={json.dumps(ops, sort_keys=True)}."
        )
        prototypes.append(
            make_chat_row(
                row_id="v381_teacher_prototype_" + row["id"],
                prompt=source["prompt"],
                answer=str(row["reexecuted_answer"]),
                subcategory="equation_teacher_prototype",
                source="v381_v380_reexecuted_teacher_prototype",
                metadata={
                    "original_id": row["id"],
                    "solver_category": row.get("solver_category"),
                    "conditioned_on_answer": bool_text(row.get("conditioned_on_answer")),
                    "train_allowed": False,
                    "reason": "original weak prompt retained only for audit/prototype",
                },
                trace=trace,
            )
        )
        generated = candidate_queries(
            original_query=original_query,
            ops=ops,
            mapping=mapping,
            seed=row["id"],
            per_rule=args.synthetic_per_rule,
        )
        for idx, (query, answer) in enumerate(generated):
            prompt = build_prompt_with_query(source["prompt"], query)
            split = deterministic_split(f"{row['id']}:{idx}")
            synthetic_id = f"v381_synth_{row['id']}_{idx:02d}"
            synthetic.append(
                make_chat_row(
                    row_id=synthetic_id,
                    prompt=prompt,
                    answer=answer,
                    subcategory="equation_symbolic_teacher_synthetic",
                    source="v381_v380_reexecuted_teacher_synthetic",
                    metadata={
                        "original_id": row["id"],
                        "split": split,
                        "solver_category": row.get("solver_category"),
                        "solver_ops": ops,
                        "conditioned_on_answer": bool_text(row.get("conditioned_on_answer")),
                        "train_allowed": True,
                        "weak_original_prompt_hash": prompt_hash(source["prompt"]),
                    },
                    trace=trace,
                )
            )
            generation_rows.append(
                {
                    "id": synthetic_id,
                    "original_id": row["id"],
                    "split": split,
                    "solver_category": row.get("solver_category"),
                    "query": query,
                    "answer": answer,
                    "prompt_sha256": prompt_hash(prompt),
                }
            )

    train_rows = [row for row in synthetic if row["metadata"].get("split") == "train"]
    val_rows = [row for row in synthetic if row["metadata"].get("split") == "val"]
    train_rows.extend(load_v217_replay(args.v217_train_jsonl, "bit_manipulation", args.bit_replay_train_rows))
    val_rows.extend(load_v217_replay(args.v217_val_jsonl, "bit_manipulation", args.bit_replay_val_rows))

    train_jsonl = args.output_dir / "v381_train.jsonl"
    val_jsonl = args.output_dir / "v381_val.jsonl"
    prototypes_jsonl = args.output_dir / "v381_teacher_prototypes_not_for_training.jsonl"
    generation_csv = args.output_dir / "v381_synthetic_generation_detail.csv"
    dataset_manifest_json = args.output_dir / "v381_dataset_manifest.json"
    manifest_json = args.output_dir / "v381_filtered_teacher_dataset_manifest.json"

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    write_jsonl(prototypes_jsonl, prototypes)
    write_csv(generation_csv, generation_rows, ["id", "original_id", "split", "solver_category", "query", "answer", "prompt_sha256"])
    dataset_manifest = {
        "schema_version": "kg1_v381_dataset_manifest_v1",
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "val_sha256": sha256_file(val_jsonl),
        },
    }
    write_json(dataset_manifest_json, dataset_manifest)

    overlap_v366_train = audit_overlaps(train_rows, v366_rows, "v366_weak")
    overlap_v366_val = audit_overlaps(val_rows, v366_rows, "v366_weak")
    overlap_v217_train = audit_overlaps(train_rows, v217_train, "v217_train")
    overlap_v217_val = audit_overlaps(train_rows, v217_val, "v217_val")
    train_summary = summarize_rows(train_rows)
    val_summary = summarize_rows(val_rows)
    prototype_summary = summarize_rows(prototypes)

    gate_pass = (
        len(accepted) == 70
        and train_summary["rows"] >= args.min_train_rows
        and val_summary["rows"] >= args.min_val_rows
        and overlap_v366_train["overlap_rows"] == 0
        and overlap_v366_val["overlap_rows"] == 0
        and overlap_v217_val["overlap_rows"] == 0
    )
    decision = {
        "status": "dataset_gate_passed_tokenization_required" if gate_pass else "dataset_gate_blocked",
        "hf_gpu_allowed": False,
        "tokenization_gate_required": True,
        "reason": (
            "synthetic dataset built without exact weak prompt overlap; run V286 real tokenization before HF"
            if gate_pass
            else "dataset failed row-count or overlap gate"
        ),
        "next_action": "Run V286 generic tokenization gate on v381_dataset_manifest.json; only then consider a short HF micro-train.",
    }
    manifest = {
        "schema_version": "kg1_v381_filtered_teacher_dataset_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v380_candidate_csv": str(args.v380_candidate_csv),
            "v366_csv": str(args.v366_csv),
            "v217_train_jsonl": str(args.v217_train_jsonl),
            "v217_val_jsonl": str(args.v217_val_jsonl),
        },
        "parameters": {
            "synthetic_per_rule": args.synthetic_per_rule,
            "bit_replay_train_rows": args.bit_replay_train_rows,
            "bit_replay_val_rows": args.bit_replay_val_rows,
        },
        "counts": {
            "accepted_v380_teacher_rows": len(accepted),
            "prototypes_not_for_training": len(prototypes),
            "synthetic_equation_rows": len(synthetic),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "summaries": {
            "train": train_summary,
            "validation": val_summary,
            "prototypes": prototype_summary,
        },
        "overlap_audit": {
            "train_vs_v366_weak": overlap_v366_train,
            "val_vs_v366_weak": overlap_v366_val,
            "train_vs_v217_train": overlap_v217_train,
            "train_vs_v217_val": overlap_v217_val,
        },
        "decision": decision,
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "prototypes_jsonl": str(prototypes_jsonl),
            "generation_csv": str(generation_csv),
            "dataset_manifest_json": str(dataset_manifest_json),
            "manifest_json": str(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    print("counts =", json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    print("train_summary =", json.dumps(train_summary, indent=2, sort_keys=True), flush=True)
    print("val_summary =", json.dumps(val_summary, indent=2, sort_keys=True), flush=True)
    print("overlap_audit =", json.dumps(manifest["overlap_audit"], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    print("=== V381 FILTERED TEACHER DATASET END ===", flush=True)
    return manifest


def self_test() -> int:
    ops = {"+": "add", "*": "mul"}
    mapping = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    generated = candidate_queries(original_query="a+b", ops=ops, mapping=mapping, seed="self", per_rule=5)
    assert len(generated) == 5
    for query, answer in generated:
        assert answer == expression_candidates(query, ops, mapping, little_endian=False)[0]
    prompt = "In Alice's Wonderland, a secret set of transformation rules is applied to equations.\nNow, determine the result for: a+b"
    assert build_prompt_with_query(prompt, "b+c").endswith("b+c")
    print("v381_filtered_teacher_dataset_self_test=ok", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v380-candidate-csv", type=Path, default=DEFAULT_V380_CANDIDATE_CSV)
    parser.add_argument("--v366-csv", type=Path, default=DEFAULT_V366_CSV)
    parser.add_argument("--v217-train-jsonl", type=Path, default=DEFAULT_V217_TRAIN_JSONL)
    parser.add_argument("--v217-val-jsonl", type=Path, default=DEFAULT_V217_VAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--synthetic-per-rule", type=int, default=12)
    parser.add_argument("--bit-replay-train-rows", type=int, default=240)
    parser.add_argument("--bit-replay-val-rows", type=int, default=40)
    parser.add_argument("--min-train-rows", type=int, default=600)
    parser.add_argument("--min-val-rows", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

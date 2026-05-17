#!/usr/bin/env python3
"""Build V571 certified bit-pair source-only trace pack.

V570 showed that the protected weak rows are solved by a long bit-pair /
bitsum / stride-style trajectory, but those weak rows cannot be used as
training positives.  V571 therefore mines only source-only V534 rows and keeps
only rows where a deterministic per-output-bit rule, inferred from the examples
without weak/full labels, reproduces the source answer exactly.

This script never trains, evaluates a model, packages, or submits.  Its output
is a CPU-gate dataset candidate plus a manifest explaining whether the candidate
is worth passing through V509/V286/V513/V524.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competition_utils import box_answer, extract_final_answer_for_expected, verify_answer  # noqa: E402


DEFAULT_V534_DIR = ROOT / "artifacts/v534_bit_source_only_trace_pack/20260517T024405Z"
DEFAULT_V534_TRAIN = DEFAULT_V534_DIR / "v534_bit_source_only_trace_pack_train.jsonl"
DEFAULT_V534_VAL = DEFAULT_V534_DIR / "v534_bit_source_only_trace_pack_val.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/v571_bitpair_source_only_trace_pack" / "20260517T_v571_cpu_gate"

SYSTEM_PROMPT = (
    "You are solving Kaggle Game Arena / NVIDIA Nemotron KG1 puzzles. "
    "Infer the hidden rule from the examples, verify it briefly, then end with exactly one final answer in \\boxed{}."
)

ANTI_LEAK_FLAGS = (
    "weak_gate_rows_used_for_training",
    "gate_rows_used_for_training",
    "full_gate_rows_used_for_training",
)

BIT_PROMPT_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
QUERY_RE = re.compile(r"Now,\s*determine\s+the\s+output\s+for:\s*([01]{8})", re.I)


@dataclass(frozen=True)
class Candidate:
    op: str
    a: int | None = None
    b: int | None = None
    const: int | None = None

    @property
    def label(self) -> str:
        if self.op == "I":
            return f"I{self.a}"
        if self.op == "NOT":
            return f"NOT{self.a}"
        if self.op == "C":
            return f"C{self.const}"
        return f"{self.op}{self.a}{self.b}"

    @property
    def description(self) -> str:
        if self.op == "I":
            return f"copy input bit {self.a}"
        if self.op == "NOT":
            return f"invert input bit {self.a}"
        if self.op == "C":
            return f"constant {self.const}"
        return f"{self.op} of input bits {self.a} and {self.b}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            row["_source_path"] = str(path)
            row["_source_line_no"] = line_no
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_problem(prompt: str) -> tuple[list[tuple[str, str]], str] | None:
    examples = BIT_PROMPT_RE.findall(prompt)
    query_match = QUERY_RE.search(prompt)
    if not examples or query_match is None:
        return None
    return examples, query_match.group(1)


def bit(value: str, index: int) -> int:
    return 1 if value[index] == "1" else 0


def eval_candidate(candidate: Candidate, value: str) -> int:
    if candidate.op == "I":
        return bit(value, int(candidate.a))
    if candidate.op == "NOT":
        return 1 - bit(value, int(candidate.a))
    if candidate.op == "C":
        return int(candidate.const)
    a = bit(value, int(candidate.a))
    b = bit(value, int(candidate.b))
    if candidate.op == "AND":
        return a & b
    if candidate.op == "OR":
        return a | b
    if candidate.op == "XOR":
        return a ^ b
    if candidate.op == "AND-NOT":
        return a & (1 - b)
    if candidate.op == "OR-NOT":
        return a | (1 - b)
    if candidate.op == "XOR-NOT":
        return a ^ (1 - b)
    raise ValueError(f"unknown candidate op: {candidate.op}")


def candidate_list() -> list[Candidate]:
    candidates: list[Candidate] = []
    for index in range(8):
        candidates.append(Candidate("I", index))
    for index in range(8):
        candidates.append(Candidate("NOT", index))
    candidates.extend([Candidate("C", const=0), Candidate("C", const=1)])
    for op in ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"):
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                candidates.append(Candidate(op, a, b))
    return candidates


CANDIDATES = candidate_list()


def solve_by_bitpair(prompt: str) -> dict[str, Any] | None:
    parsed = parse_problem(prompt)
    if parsed is None:
        return None
    examples, query = parsed
    selected: list[Candidate] = []
    match_counts: list[int] = []
    columns: list[dict[str, Any]] = []
    for output_index in range(8):
        target_vector = "".join(output[output_index] for _input, output in examples)
        matches: list[Candidate] = []
        for candidate in CANDIDATES:
            vector = "".join(str(eval_candidate(candidate, input_bits)) for input_bits, _output in examples)
            if vector == target_vector:
                matches.append(candidate)
        if not matches:
            return None
        chosen = matches[0]
        selected.append(chosen)
        match_counts.append(len(matches))
        columns.append(
            {
                "output_bit": output_index,
                "target_vector": target_vector,
                "bitsum": target_vector.count("1"),
                "chosen": chosen.label,
                "match_count": len(matches),
                "preview_matches": [item.label for item in matches[:8]],
            }
        )
    predicted = "".join(str(eval_candidate(candidate, query)) for candidate in selected)
    return {
        "examples": examples,
        "query": query,
        "prediction": predicted,
        "selected": selected,
        "match_counts": match_counts,
        "columns": columns,
    }


def normalize_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return dict(metadata)


def make_trace(row: dict[str, Any], solved: dict[str, Any]) -> str:
    examples: list[tuple[str, str]] = solved["examples"]
    query = str(solved["query"])
    selected: list[Candidate] = solved["selected"]
    lines: list[str] = [
        "Bit-pair certified trace.",
        "The rule is inferred per output bit from the example output columns.",
        "For each bit, I keep the first simple unary/binary relation that matches every example column.",
        "",
        "Example output columns with bitsum:",
    ]
    for output_index in range(8):
        vector = "".join(output[output_index] for _input, output in examples)
        lines.append(f"- out{output_index}: {vector} bitsum={vector.count('1')}")
    lines.extend(["", "Selected column rules:"])
    for output_index, candidate in enumerate(selected):
        col = solved["columns"][output_index]
        previews = " ".join(col["preview_matches"][:5])
        lines.append(
            f"- out{output_index}: {candidate.label} ({candidate.description}); "
            f"matches={col['match_count']} preview={previews}"
        )
    lines.extend(["", f"Apply to query {query}:"])
    query_bits = {str(index): query[index] for index in range(8)}
    lines.append("Input bits: " + " ".join(f"{idx}={value}" for idx, value in query_bits.items()))
    output_bits: list[str] = []
    for output_index, candidate in enumerate(selected):
        value = str(eval_candidate(candidate, query))
        output_bits.append(value)
        lines.append(f"- out{output_index}: {candidate.label} -> {value}")
    predicted = "".join(output_bits)
    lines.extend(
        [
            "",
            f"Certified answer from selected rules: {predicted}",
            f"Final answer: {box_answer(predicted)}",
        ]
    )
    return "\n".join(lines)


def convert_row(row: dict[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    if str(row.get("family", "")).strip() != "bit_manipulation":
        return None, "not_bit"
    metadata = normalize_metadata(row)
    if not metadata.get("source_only", False):
        return None, "not_source_only"
    if any(bool(metadata.get(flag)) for flag in ANTI_LEAK_FLAGS):
        return None, "anti_leak_flag_true"
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    if not re.fullmatch(r"[01]{8}", answer):
        return None, "bad_answer"
    solved = solve_by_bitpair(prompt)
    if solved is None:
        return None, "no_certified_rule"
    if solved["prediction"] != answer:
        return None, "certified_prediction_mismatch"
    assistant = make_trace(row, solved)
    extracted = extract_final_answer_for_expected(assistant, answer)
    if not verify_answer(answer, extracted):
        return None, "assistant_extraction_mismatch"
    source_id = str(row.get("id", "")).strip()
    out_id = f"v571_{split}_{sha256_text(source_id + prompt + answer)[:16]}"
    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "schema_version": "kg1_v571_bitpair_source_only_trace_pack_v1",
            "source": "v571_bitpair_source_only_trace_pack",
            "source_dataset": "v571_from_v534_source_only",
            "source_only": True,
            "v571_original_id": source_id,
            "v571_original_source": metadata.get("source_dataset", row.get("source", "")),
            "v571_split": split,
            "v571_certified_rule_count": len(solved["selected"]),
            "v571_certified_answer": solved["prediction"],
            "v571_selected_rules": [candidate.label for candidate in solved["selected"]],
            "v571_match_count_min": min(solved["match_counts"]),
            "v571_match_count_max": max(solved["match_counts"]),
        }
    )
    for flag in ANTI_LEAK_FLAGS:
        out_metadata[flag] = False
    out_row = {
        "id": out_id,
        "family": "bit_manipulation",
        "answer": answer,
        "prompt": prompt,
        "source": "v571_bitpair_source_only_trace_pack",
        "source_dataset": "v571_from_v534_source_only",
        "subcategory": "bit_bitpair_certified_source_only",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": out_metadata,
    }
    return out_row, "accepted"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(str(row["messages"][-1]["content"]).split()) for row in rows]
    chars = [len(str(row["messages"][-1]["content"])) for row in rows]
    sources = Counter(str(row["metadata"].get("v571_original_source", "")) for row in rows)
    rule_labels = Counter()
    for row in rows:
        for label in row["metadata"].get("v571_selected_rules", []):
            rule_labels[str(label).split("0", 1)[0]] += 1
    return {
        "rows": len(rows),
        "family_counts": {"bit_manipulation": len(rows)} if rows else {},
        "source_counts": dict(sources.most_common()),
        "assistant_words_min": min(lengths) if lengths else 0,
        "assistant_words_p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "assistant_words_max": max(lengths) if lengths else 0,
        "assistant_chars_min": min(chars) if chars else 0,
        "assistant_chars_p50": sorted(chars)[len(chars) // 2] if chars else 0,
        "assistant_chars_max": max(chars) if chars else 0,
        "selected_rule_prefix_counts": dict(rule_labels.most_common(20)),
    }


def process_split(path: Path, split: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in read_jsonl(path):
        converted, reason = convert_row(row, split)
        reasons[reason] += 1
        if converted is not None:
            accepted.append(converted)
    return accepted, reasons


def build(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V571 BITPAIR SOURCE-ONLY TRACE PACK START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v534_train_jsonl =", args.v534_train_jsonl, flush=True)
    print("v534_val_jsonl =", args.v534_val_jsonl, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_reasons = process_split(args.v534_train_jsonl, "train")
    val_rows, val_reasons = process_split(args.v534_val_jsonl, "validation")

    train_jsonl = args.output_dir / "v571_bitpair_source_only_trace_pack_train.jsonl"
    val_jsonl = args.output_dir / "v571_bitpair_source_only_trace_pack_val.jsonl"
    manifest_json = args.output_dir / "v571_bitpair_source_only_trace_pack_manifest.json"
    summary_md = args.output_dir / "KG1_V571_BITPAIR_SOURCE_ONLY_TRACE_PACK.md"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    blockers: list[str] = []
    if len(train_rows) < int(args.min_train_rows):
        blockers.append(f"train_rows_lt_min:{len(train_rows)}<{args.min_train_rows}")
    if len(val_rows) < int(args.min_val_rows):
        blockers.append(f"val_rows_lt_min:{len(val_rows)}<{args.min_val_rows}")
    if not train_rows or not val_rows:
        blockers.append("missing_train_or_validation")
    decision_status = "dataset_ready_for_cpu_gates" if not blockers else "blocked_insufficient_certified_rows"

    manifest = {
        "version": "V571",
        "schema_version": "kg1_v571_bitpair_source_only_trace_pack_v1",
        "generated_at_utc": utc_now(),
        "inputs": {
            "v534_train_jsonl": str(args.v534_train_jsonl),
            "v534_train_sha256": sha256_file(args.v534_train_jsonl),
            "v534_val_jsonl": str(args.v534_val_jsonl),
            "v534_val_sha256": sha256_file(args.v534_val_jsonl),
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "val_sha256": sha256_file(val_jsonl),
            "manifest_json": str(manifest_json),
            "summary_md": str(summary_md),
        },
        "selection": {
            "train_reason_counts": dict(train_reasons.most_common()),
            "validation_reason_counts": dict(val_reasons.most_common()),
            "accepted_train": len(train_rows),
            "accepted_validation": len(val_rows),
        },
        "train_summary": summarize(train_rows),
        "validation_summary": summarize(val_rows),
        "blocked_actions": ["train_gpu", "full_eval", "package", "kaggle_submit"],
        "blockers": blockers,
        "decision": {
            "status": decision_status,
            "gpu_allowed": False,
            "submit_allowed": False,
            "reason": (
                "Certified source-only bit-pair traces are available for CPU gates."
                if not blockers
                else "Too few source-only rows could be certified without using weak/full labels."
            ),
            "next_action": (
                "Run V509 integrity, V286 boxed_suffix tokenization, V513 learnability, then objective/pre-paid gates. "
                "Do not train or submit directly from this manifest."
            )
            if not blockers
            else "Discard this route or expand the deterministic grammar before any GPU.",
        },
    }
    write_json(manifest_json, manifest)
    lines = [
        "# KG1 V571 Bit-Pair Source-Only Trace Pack",
        "",
        f"Generated at UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{decision_status}`",
        f"- Train accepted: `{len(train_rows)}`",
        f"- Validation accepted: `{len(val_rows)}`",
        f"- Blockers: `{blockers}`",
        "",
        "## Selection",
        "",
        f"- Train reasons: `{json.dumps(dict(train_reasons.most_common()), sort_keys=True)}`",
        f"- Validation reasons: `{json.dumps(dict(val_reasons.most_common()), sort_keys=True)}`",
        "",
        "## Meaning",
        "",
        "V571 is not a weak-label replay dataset. It keeps only source-only rows where simple bit relations inferred from examples reproduce the source answer.",
        "It is meant to test whether bit-pair/bitsum style can be transferred without using protected weak rows as training positives.",
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print("accepted_train =", len(train_rows), flush=True)
    print("accepted_validation =", len(val_rows), flush=True)
    print("train_reasons =", json.dumps(dict(train_reasons.most_common()), sort_keys=True), flush=True)
    print("validation_reasons =", json.dumps(dict(val_reasons.most_common()), sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("=== V571 BITPAIR SOURCE-ONLY TRACE PACK END ===", flush=True)
    return manifest


def self_test() -> None:
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n"
        "Here are some examples of input -> output:\n"
        "00000000 -> 00000000\n"
        "11111111 -> 11111111\n"
        "10101010 -> 10101010\n"
        "Now, determine the output for: 01010101"
    )
    solved = solve_by_bitpair(prompt)
    if solved is None or solved["prediction"] != "01010101":
        raise AssertionError(solved)
    assistant = make_trace({}, solved)
    if extract_final_answer_for_expected(assistant, "01010101") != "01010101":
        raise AssertionError(assistant)
    print("build_v571_bitpair_source_only_trace_pack_self_test=ok", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v534-train-jsonl", type=Path, default=DEFAULT_V534_TRAIN)
    parser.add_argument("--v534-val-jsonl", type=Path, default=DEFAULT_V534_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--min-val-rows", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

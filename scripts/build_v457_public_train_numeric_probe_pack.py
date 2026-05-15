#!/usr/bin/env python3
"""Build V457 public-train numeric probe pack.

V456 showed that two missing equation classes already had heavy synthetic
coverage, while the trailing-zero colon class has a builder but no public-train
adapter hard-negative evidence. V457 creates a prompt-only pack from public
train rows whose labels indicate they are candidate probes for the missing
numeric classes. The exported prompt pack intentionally omits answers.

This script is CPU-only. It does not train, infer, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for item in (ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import classify_puzzle, verify_answer  # noqa: E402
from kg1_v274_numeric_postprocessor import (  # noqa: E402
    choose_guarded_numeric_override,
    group_examples_by_operator,
    normalize_payload,
    numeric_candidates,
    parse_alice_prompt,
    parse_numeric_token,
    reverse_normalized_keep_sign,
)


DEFAULT_TRAIN_CSV = Path(os.environ.get("KG1_COMPETITION_TRAIN_CSV", r"C:\Users\davis\Downloads\competition_train.csv"))
DEFAULT_REFERENCE_WEAK_CSV = (
    ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_REFERENCE_FULL_CSV = ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "v457_public_train_numeric_probe_pack" / "20260515T_cpu_gate"

TARGET_RULES = {
    "v274_guarded_numeric_add_direct_over_model_add_variant",
    "v274_guarded_numeric_colon_absdiff_restore_trailing_zero",
    "v274_guarded_numeric_minus_signed_opposite_sign_guarded",
}

PROMPT_COLUMNS = ["id", "family", "target_rule_class", "prompt_sha256", "prompt_normalized_sha256", "prompt"]
AUDIT_COLUMNS = [
    "id",
    "family",
    "target_rule_class",
    "query",
    "answer",
    "simulated_wrong_prediction",
    "postprocessor_prediction",
    "postprocessor_proof",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "selected_for_prompt_pack",
]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def reference_set(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            ids.add(rid)
        prompt = str(row.get("prompt") or row.get("generated_prompt") or "")
        if prompt:
            prompt_hashes.add(sha256_text(normalize_prompt(prompt)))
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows), "ids": ids, "prompt_hashes": prompt_hashes}


def candidate_wrong_predictions(examples: list[tuple[str, str]], query: str, answer: str) -> list[str]:
    normalized_answer = normalize_payload(answer)
    candidates: set[str] = set()
    if not normalized_answer:
        return []
    if normalized_answer.startswith("-"):
        candidates.add(normalized_answer[1:])
    else:
        candidates.add("-" + normalized_answer)
    if len(normalized_answer.lstrip("-")) > 1:
        candidates.add(reverse_normalized_keep_sign(normalized_answer))
    if normalized_answer.endswith("0"):
        candidates.add(normalized_answer.rstrip("0") or "0")

    parsed = parse_numeric_token(query)
    grouped_result = group_examples_by_operator(examples)
    if parsed and grouped_result is not None:
        grouped, _op_sequence = grouped_result
        query_op = parsed[1]
        group = grouped.get(query_op, [])
        if query_op in {")", "+"} and group:
            for item in numeric_candidates(group, query, {"add", "rev_add", "tens_add_ones_add"}):
                pred = normalize_payload(item.get("prediction", ""))
                if pred and pred != normalized_answer:
                    candidates.add(pred)
        if query_op == ":" and group:
            for item in numeric_candidates(group, query, {"abs_diff", "rev_abs_diff", "digit_absdiff_concat", "tens_absdiff_ones_absdiff_int"}):
                pred = normalize_payload(item.get("prediction", ""))
                if pred and pred != normalized_answer:
                    candidates.add(pred)

    return sorted(candidates)


def classify_target_rule(prompt: str, answer: str) -> tuple[str, str, str, str, str] | None:
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        return None
    if not parse_numeric_token(query):
        return None
    for wrong in candidate_wrong_predictions(examples, query, answer):
        replacement, rule, proof = choose_guarded_numeric_override(examples, query, wrong)
        rule_class = "v274_guarded_numeric_" + str(rule)
        if rule_class not in TARGET_RULES:
            continue
        if not replacement or not verify_answer(str(answer), str(replacement)):
            continue
        return rule_class, query, wrong, str(replacement), proof
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V457 PUBLIC TRAIN NUMERIC PROBE PACK START ===", flush=True)
    print("competition_train_csv =", args.competition_train_csv, "exists =", args.competition_train_csv.exists(), flush=True)
    print("reference_weak_csv =", args.reference_weak_csv, "exists =", args.reference_weak_csv.exists(), flush=True)
    print("reference_full_csv =", args.reference_full_csv, "exists =", args.reference_full_csv.exists(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("max_rows_per_rule =", args.max_rows_per_rule, flush=True)
    if not args.competition_train_csv.is_file():
        raise FileNotFoundError(args.competition_train_csv)
    if not args.reference_weak_csv.is_file():
        raise FileNotFoundError(args.reference_weak_csv)
    if not args.reference_full_csv.is_file():
        raise FileNotFoundError(args.reference_full_csv)

    weak_ref = reference_set(args.reference_weak_csv)
    full_ref = reference_set(args.reference_full_csv)
    reference_ids = set(weak_ref["ids"]) | set(full_ref["ids"])
    reference_prompt_hashes = set(weak_ref["prompt_hashes"]) | set(full_ref["prompt_hashes"])
    train_rows = read_csv(args.competition_train_csv)

    audit_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    selected_by_rule: Counter[str] = Counter()
    skipped = Counter()

    for row in sorted(train_rows, key=lambda item: str(item.get("id", ""))):
        rid = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        family = classify_puzzle(prompt)
        if family != "equation_transform":
            continue
        prompt_sha = sha256_text(prompt.replace("\r\n", "\n"))
        prompt_norm_sha = sha256_text(normalize_prompt(prompt))
        if rid in reference_ids:
            skipped["reference_id_overlap"] += 1
            continue
        if prompt_norm_sha in reference_prompt_hashes:
            skipped["reference_prompt_overlap"] += 1
            continue
        classified = classify_target_rule(prompt, answer)
        if classified is None:
            continue
        rule_class, query, wrong, replacement, proof = classified
        selected = selected_by_rule[rule_class] < args.max_rows_per_rule
        audit_rows.append(
            {
                "id": rid,
                "family": family,
                "target_rule_class": rule_class,
                "query": query,
                "answer": answer,
                "simulated_wrong_prediction": wrong,
                "postprocessor_prediction": replacement,
                "postprocessor_proof": proof,
                "prompt_sha256": prompt_sha,
                "prompt_normalized_sha256": prompt_norm_sha,
                "selected_for_prompt_pack": selected,
            }
        )
        if selected:
            prompt_rows.append(
                {
                    "id": rid,
                    "family": family,
                    "target_rule_class": rule_class,
                    "prompt_sha256": prompt_sha,
                    "prompt_normalized_sha256": prompt_norm_sha,
                    "prompt": prompt,
                }
            )
            selected_by_rule[rule_class] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_csv = args.output_dir / "v457_public_train_numeric_probe_pack_prompts.csv"
    prompts_jsonl = args.output_dir / "v457_public_train_numeric_probe_pack_prompts.jsonl"
    audit_csv = args.output_dir / "v457_public_train_numeric_probe_pack_audit.csv"
    manifest_json = args.output_dir / "v457_public_train_numeric_probe_pack_manifest.json"
    report_md = args.output_dir / "V457_PUBLIC_TRAIN_NUMERIC_PROBE_PACK.md"

    write_csv(prompts_csv, prompt_rows, PROMPT_COLUMNS)
    write_jsonl(prompts_jsonl, prompt_rows)
    write_csv(audit_csv, audit_rows, AUDIT_COLUMNS)

    audit_counts = Counter(str(row["target_rule_class"]) for row in audit_rows)
    selected_counts = Counter(str(row["target_rule_class"]) for row in prompt_rows)
    hf_raw_probe_allowed = bool(prompt_rows) and any(
        selected_counts.get(rule, 0) > 0 for rule in TARGET_RULES
    )
    manifest = {
        "schema_version": "kg1_v457_public_train_numeric_probe_pack_v1",
        "generated_at_utc": utc_now(),
        "source_policy": {
            "answers_exported_to_prompt_pack": False,
            "answers_in_audit_only": True,
            "weak_or_full_rows_exported": False,
            "purpose": "Collect adapter raw outputs on public-train numeric prompts before any training.",
        },
        "inputs": {
            "competition_train_csv": str(args.competition_train_csv),
            "competition_train_sha256": sha256_file(args.competition_train_csv),
            "reference_weak_csv": str(args.reference_weak_csv),
            "reference_weak_sha256": weak_ref["sha256"],
            "reference_full_csv": str(args.reference_full_csv),
            "reference_full_sha256": full_ref["sha256"],
        },
        "selection": {
            "audit_rows": len(audit_rows),
            "prompt_rows": len(prompt_rows),
            "audit_counts_by_rule": dict(sorted(audit_counts.items())),
            "selected_counts_by_rule": dict(sorted(selected_counts.items())),
            "skipped": dict(sorted(skipped.items())),
            "max_rows_per_rule": args.max_rows_per_rule,
            "hf_raw_probe_allowed": hf_raw_probe_allowed,
            "hf_gpu_train_allowed": False,
        },
        "outputs": {
            "prompts_csv": str(prompts_csv),
            "prompts_csv_sha256": sha256_file(prompts_csv),
            "prompts_jsonl": str(prompts_jsonl),
            "prompts_jsonl_sha256": sha256_file(prompts_jsonl),
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": sha256_file(audit_csv),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
        "decision": {
            "decision": "v457_prompt_pack_ready_for_raw_probe" if hf_raw_probe_allowed else "v457_no_public_train_targets_found",
            "hf_raw_probe_allowed": hf_raw_probe_allowed,
            "hf_gpu_train_allowed": False,
            "next_action": (
                "Run inference-only raw-output probe, then build legal hard negatives from actual adapter errors."
                if hf_raw_probe_allowed
                else "Return to CPU DSL search; do not open HF."
            ),
        },
    }
    write_json(manifest_json, manifest)

    lines = [
        "# V457 Public Train Numeric Probe Pack",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Audit rows | `{len(audit_rows)}` |",
        f"| Prompt rows exported | `{len(prompt_rows)}` |",
        f"| `hf_raw_probe_allowed` | `{str(hf_raw_probe_allowed).lower()}` |",
        f"| `hf_gpu_train_allowed` | `false` |",
        "",
        "## Selected Counts",
        "",
        "| Rule class | Rows |",
        "|---|---:|",
    ]
    for rule, count in sorted(selected_counts.items()):
        lines.append(f"| `{rule}` | `{count}` |")
    lines.extend(
        [
            "",
            "The prompt pack omits `answer` and exports only public-train prompts plus rule metadata.",
            "Training remains blocked until actual adapter raw outputs prove legal hard negatives.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("audit_counts_by_rule =", json.dumps(dict(sorted(audit_counts.items())), sort_keys=True), flush=True)
    print("selected_counts_by_rule =", json.dumps(dict(sorted(selected_counts.items())), sort_keys=True), flush=True)
    print("prompts_jsonl =", prompts_jsonl, flush=True)
    print("manifest_json =", manifest_json, flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("=== V457 PUBLIC TRAIN NUMERIC PROBE PACK END ===", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--reference-weak-csv", type=Path, default=DEFAULT_REFERENCE_WEAK_CSV)
    parser.add_argument("--reference-full-csv", type=Path, default=DEFAULT_REFERENCE_FULL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows-per-rule", type=int, default=96)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

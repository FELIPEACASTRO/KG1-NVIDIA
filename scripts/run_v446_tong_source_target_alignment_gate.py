#!/usr/bin/env python3
"""V446 CPU gate for public-source target alignment.

This gate is intentionally CPU-only. It inspects public Tong Hui Kang source
signals and local SFT trace candidates, then decides whether there is enough
clean, target-aligned material to justify a paid GPU job.

It does not train, run model inference, package an adapter, or submit.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for item in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import canonical_family, classify_puzzle  # noqa: E402
from run_v278_symbolic_pbe_dsl_audit_hf import EXPECTED_ROW_CONTRACT_SHA256, normalize_row, row_contract  # noqa: E402
from run_v333_tong_bit_reasoner_gate import DEFAULT_TONG_COMMIT, DEFAULT_TONG_REPO_URL, fetch_tong_source  # noqa: E402


DEFAULT_SFT_JSONL = Path(r"C:\Users\davis\Downloads\sft_reconstructed.jsonl")
DEFAULT_COMPETITION_TRAIN_CSV = Path(r"C:\Users\davis\Downloads\competition_train.csv")
DEFAULT_REFERENCE_WEAK_CSV = (
    REPO_ROOT
    / "artifacts/v290_rank19_micro_patch_reference/runtime_artifacts/v245_weak_eval_bridge/"
    / "v245-weak-bridge-hfonly-20260510T1950Z/v221_weak_315.csv"
)
DEFAULT_REFERENCE_FULL_CSV = REPO_ROOT / "artifacts/v293_gap_mining/inputs/v291_full_predictions.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/v446_tong_source_target_alignment_gate"

TRACE_STATUSES_ALLOWED = {"rule_found", "verified", "ok"}
TARGET_FAMILIES = {"bit_manipulation", "equation_transform"}
CSV_COLUMNS = [
    "row_no",
    "source_path",
    "id",
    "family",
    "status",
    "accepted",
    "block_reason",
    "prompt_sha256",
    "prompt_normalized_sha256",
    "assistant_sha256",
    "assistant_chars",
    "boxed_count",
    "reference_id_overlap",
    "reference_prompt_sha_overlap",
    "reference_normalized_prompt_overlap",
    "reference_13gram_overlap_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r\n", "\n")).strip()


def ngram_tokens(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    return re.findall(r"[a-z0-9_]+|[^\sA-Za-z0-9_]", normalized)


def token_ngrams(text: str, width: int = 13) -> set[str]:
    tokens = ngram_tokens(text)
    if len(tokens) < width:
        return set()
    return {" ".join(tokens[index : index + width]) for index in range(0, len(tokens) - width + 1)}


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


def boxed_count(text: str) -> int:
    return len(re.findall(r"\\boxed\{[^{}]*\}", text or ""))


def load_reference_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            source_counts[str(path)] = 0
            continue
        raw_rows = read_csv(path)
        normalized = [normalize_row(row) for row in raw_rows]
        source_counts[str(path)] = len(normalized)
        rows.extend(normalized)
    summary = {
        "source_counts": source_counts,
        "rows": len(rows),
        "unique_ids": len({str(row.get("id", "")) for row in rows}),
    }
    return rows, summary


def reference_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_ngrams: set[str] = set()
    high_frequency_ngrams: set[str] = set()
    ngram_counts: Counter[str] = Counter()
    for row in rows:
        grams = token_ngrams(str(row.get("prompt", "")))
        ngram_counts.update(grams)
    for gram, count in ngram_counts.items():
        if count > 3:
            high_frequency_ngrams.add(gram)
        else:
            prompt_ngrams.add(gram)
    return {
        "ids": {str(row.get("id", "")) for row in rows if str(row.get("id", ""))},
        "prompt_sha256": {sha256_text(str(row.get("prompt", ""))) for row in rows},
        "prompt_normalized_sha256": {sha256_text(normalize_text(row.get("prompt", ""))) for row in rows},
        "prompt_13grams": prompt_ngrams,
        "ignored_high_frequency_13grams": high_frequency_ngrams,
        "high_frequency_13gram_count": len(high_frequency_ngrams),
        "unique_13gram_count": len(prompt_ngrams),
    }


def extract_operations_from_ast(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions: list[str] = []
    tuple_names: Counter[str] = Counter()
    string_constants: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_constants[node.value] += 1
        if isinstance(node, ast.Tuple) and node.elts:
            first = node.elts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                tuple_names[first.value] += 1
    relevant_strings = [
        value
        for value, _count in string_constants.most_common()
        if re.search(r"(concat|reverse|add|sub|mul|div|mod|det|xor|and|or|not|shift|rot|bit|bitsum|stride|prefix|suffix)", value, re.I)
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "functions": sorted(functions),
        "tuple_candidate_names": sorted(tuple_names),
        "relevant_string_constants": relevant_strings[:200],
        "line_count": len(text.splitlines()),
    }


def source_inventory(repo_dir: Path, resolved_commit: str) -> dict[str, Any]:
    source_files = {
        "equation_numeric": repo_dir / "reasoners" / "equation_numeric.py",
        "bit_manipulation": repo_dir / "reasoners" / "bit_manipulation.py",
        "reasoning": repo_dir / "reasoning.py",
        "train_csv": repo_dir / "train.csv",
    }
    inventory = {
        "repo_url": DEFAULT_TONG_REPO_URL,
        "expected_commit": DEFAULT_TONG_COMMIT,
        "resolved_commit": resolved_commit,
        "commit_matches": resolved_commit == DEFAULT_TONG_COMMIT,
        "files": {},
    }
    for key, path in source_files.items():
        if path.suffix == ".py":
            inventory["files"][key] = extract_operations_from_ast(path)
        else:
            inventory["files"][key] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "bytes": path.stat().st_size if path.exists() else 0,
            }
    equation_terms = " ".join(
        inventory["files"].get("equation_numeric", {}).get("tuple_candidate_names", [])
        + inventory["files"].get("equation_numeric", {}).get("relevant_string_constants", [])
    ).lower()
    bit_terms = " ".join(
        inventory["files"].get("bit_manipulation", {}).get("functions", [])
        + inventory["files"].get("bit_manipulation", {}).get("relevant_string_constants", [])
    ).lower()
    inventory["equation_inventory_flags"] = {
        "concat": "concat" in equation_terms,
        "reverse": "reverse" in equation_terms or "rev" in equation_terms,
        "add_sub_mul": all(term in equation_terms for term in ("add", "sub", "mul")),
        "div_mod": "div" in equation_terms or "mod" in equation_terms,
        "digit_or_det": "digit" in equation_terms or "det" in equation_terms,
    }
    inventory["bit_inventory_flags"] = {
        "bitsum": "bitsum" in bit_terms,
        "stride": "stride" in bit_terms,
        "bit_pair": "pair" in bit_terms or "and" in bit_terms,
        "boolean_ops": all(term in bit_terms for term in ("and", "or", "xor")),
    }
    inventory["source_inventory_pass"] = bool(
        inventory["commit_matches"]
        and all(inventory["equation_inventory_flags"].values())
        and all(inventory["bit_inventory_flags"].values())
    )
    return inventory


def message_parts(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", ""))
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)
    return "\n".join(user_texts), "\n".join(assistant_texts)


def normalize_family(value: Any, prompt: str) -> str:
    raw = str(value or "").strip()
    if raw in {"transformation", "equation", "equation_transform", "symbol_transform"}:
        return "equation_transform"
    if raw in {"bitwise", "bit", "bit_manipulation"}:
        return "bit_manipulation"
    return canonical_family(raw or classify_puzzle(prompt))


def iter_sft_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for row_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                continue
            metadata = obj.get("_metadata") if isinstance(obj.get("_metadata"), dict) else {}
            prompt, assistant = message_parts(obj)
            family = normalize_family(metadata.get("category") or metadata.get("family"), prompt)
            rows.append(
                {
                    "row_no": row_no,
                    "source_path": str(path),
                    "id": str(metadata.get("problem_id") or metadata.get("id") or "").strip(),
                    "family": family,
                    "status": str(metadata.get("status") or "").strip(),
                    "prompt": prompt,
                    "assistant": assistant,
                }
            )
    return rows


def audit_candidates(
    candidates: list[dict[str, Any]],
    ref_index: dict[str, Any],
    *,
    max_target_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    family_status: Counter[tuple[str, str]] = Counter()
    accepted_by_family: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    for candidate in candidates:
        prompt = str(candidate.get("prompt", ""))
        assistant = str(candidate.get("assistant", ""))
        family = str(candidate.get("family", ""))
        status = str(candidate.get("status", ""))
        prompt_hash = sha256_text(prompt)
        normalized_hash = sha256_text(normalize_text(prompt))
        assistant_hash = sha256_text(assistant)
        grams = token_ngrams(prompt)
        overlap_grams = grams & ref_index["prompt_13grams"]
        id_overlap = bool(candidate.get("id") and candidate.get("id") in ref_index["ids"])
        prompt_overlap = prompt_hash in ref_index["prompt_sha256"]
        normalized_overlap = normalized_hash in ref_index["prompt_normalized_sha256"]
        box_count = boxed_count(assistant)

        reasons: list[str] = []
        if family not in TARGET_FAMILIES:
            reasons.append("non_target_family")
        if status and status not in TRACE_STATUSES_ALLOWED:
            reasons.append("status_not_allowed")
        if not prompt.strip():
            reasons.append("missing_prompt")
        if not assistant.strip():
            reasons.append("missing_assistant")
        if "<think>" not in assistant:
            reasons.append("missing_think_trace")
        if box_count < 1:
            reasons.append("missing_boxed_answer")
        if len(assistant) > max_target_chars:
            reasons.append("assistant_too_long")
        if id_overlap:
            reasons.append("reference_id_overlap")
        if prompt_overlap:
            reasons.append("reference_prompt_sha_overlap")
        if normalized_overlap:
            reasons.append("reference_normalized_prompt_overlap")
        if overlap_grams:
            reasons.append("reference_13gram_overlap")

        accepted = not reasons
        counters["rows"] += 1
        family_status[(family, status)] += 1
        if accepted:
            accepted_by_family[family] += 1
            counters["accepted"] += 1
        else:
            counters["blocked"] += 1
            for reason in reasons:
                blocked_reasons[reason] += 1
        audit_rows.append(
            {
                **candidate,
                "accepted": accepted,
                "block_reason": ";".join(reasons),
                "prompt_sha256": prompt_hash,
                "prompt_normalized_sha256": normalized_hash,
                "assistant_sha256": assistant_hash,
                "assistant_chars": len(assistant),
                "boxed_count": box_count,
                "reference_id_overlap": id_overlap,
                "reference_prompt_sha_overlap": prompt_overlap,
                "reference_normalized_prompt_overlap": normalized_overlap,
                "reference_13gram_overlap_count": len(overlap_grams),
            }
        )
    summary = {
        "rows": counters["rows"],
        "accepted": counters["accepted"],
        "blocked": counters["blocked"],
        "accepted_by_family": dict(sorted(accepted_by_family.items())),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "family_status_counts": {
            f"{family}::{status}": count for (family, status), count in sorted(family_status.items())
        },
    }
    return audit_rows, summary


def audit_competition_train(path: Path, ref_index: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = read_csv(path)
    family_counts: Counter[str] = Counter()
    overlaps: Counter[str] = Counter()
    for row in rows:
        prompt = str(row.get("prompt", ""))
        family_counts[canonical_family(row.get("family") or row.get("type") or classify_puzzle(prompt))] += 1
        row_id = str(row.get("id", "")).strip()
        if row_id in ref_index["ids"]:
            overlaps["id"] += 1
        if sha256_text(prompt) in ref_index["prompt_sha256"]:
            overlaps["prompt_sha256"] += 1
        if sha256_text(normalize_text(prompt)) in ref_index["prompt_normalized_sha256"]:
            overlaps["prompt_normalized_sha256"] += 1
        if token_ngrams(prompt) & ref_index["prompt_13grams"]:
            overlaps["target_13gram"] += 1
    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256_file(path),
        "family_counts": dict(sorted(family_counts.items())),
        "reference_overlap_counts": dict(sorted(overlaps.items())),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / utc_compact())
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== V446 TONG SOURCE TARGET ALIGNMENT GATE START ===", flush=True)
    print("output_dir =", output_dir, flush=True)
    print("sft_jsonl =", args.sft_jsonl, "exists =", args.sft_jsonl.exists(), flush=True)
    print("competition_train_csv =", args.competition_train_csv, "exists =", args.competition_train_csv.exists(), flush=True)
    print("reference_weak_csv =", args.reference_weak_csv, "exists =", args.reference_weak_csv.exists(), flush=True)
    print("reference_full_csv =", args.reference_full_csv, "exists =", args.reference_full_csv.exists(), flush=True)

    reference_rows, reference_summary = load_reference_rows([args.reference_weak_csv, args.reference_full_csv])
    reference_rows_weak_only, _weak_summary = load_reference_rows([args.reference_weak_csv])
    if reference_rows_weak_only:
        observed_contract = row_contract(reference_rows_weak_only)
    else:
        observed_contract = ""
    ref_index = reference_index(reference_rows)
    reference_summary.update(
        {
            "observed_weak_row_contract_sha256": observed_contract,
            "expected_weak_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
            "weak_row_contract_pass": observed_contract == EXPECTED_ROW_CONTRACT_SHA256,
            "unique_13gram_count": ref_index["unique_13gram_count"],
            "ignored_high_frequency_13gram_count": ref_index["high_frequency_13gram_count"],
        }
    )
    print("reference_summary =", json.dumps(reference_summary, sort_keys=True), flush=True)

    with tempfile.TemporaryDirectory(prefix="kg1_v446_tong_") as tmp:
        tmp_dir = Path(tmp)
        tong_repo_dir, resolved_commit = fetch_tong_source(tmp_dir, args.tong_repo_url, args.tong_commit)
        inventory = source_inventory(tong_repo_dir, resolved_commit)

    candidates = iter_sft_candidates(args.sft_jsonl)
    audit_rows, candidate_summary = audit_candidates(candidates, ref_index, max_target_chars=args.max_target_chars)
    train_summary = audit_competition_train(args.competition_train_csv, ref_index)

    accepted_by_family = candidate_summary["accepted_by_family"]
    equation_accepted = int(accepted_by_family.get("equation_transform", 0))
    bit_accepted = int(accepted_by_family.get("bit_manipulation", 0))
    hf_gpu_allowed = bool(
        reference_summary["weak_row_contract_pass"]
        and inventory["source_inventory_pass"]
        and args.sft_jsonl.exists()
        and equation_accepted >= args.min_equation_rows_for_gpu
        and bit_accepted >= args.min_bit_rows_for_gpu
    )
    decision = {
        "hf_gpu_allowed": hf_gpu_allowed,
        "finops_decision": "allow_next_gpu_smoke" if hf_gpu_allowed else "block_gpu_keep_cpu_only",
        "reason": (
            "clean target-aligned material meets minimums"
            if hf_gpu_allowed
            else (
                f"accepted_equation={equation_accepted} min={args.min_equation_rows_for_gpu}; "
                f"accepted_bit={bit_accepted} min={args.min_bit_rows_for_gpu}; "
                f"source_inventory_pass={inventory['source_inventory_pass']}; "
                f"weak_contract_pass={reference_summary['weak_row_contract_pass']}"
            )
        ),
        "next_action": (
            "Build a minimal trace dataset and run CPU tokenization/pair gates before a one-checkpoint GPU smoke."
            if hf_gpu_allowed
            else "Do not launch GPU. Mine public notebooks/source for a cleaner target-aligned dataset or implement CPU DSL gains first."
        ),
    }

    manifest = {
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "sft_jsonl": str(args.sft_jsonl),
            "competition_train_csv": str(args.competition_train_csv),
            "reference_weak_csv": str(args.reference_weak_csv),
            "reference_full_csv": str(args.reference_full_csv),
            "tong_repo_url": args.tong_repo_url,
            "tong_commit": args.tong_commit,
            "expected_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
        },
        "reference_summary": reference_summary,
        "source_inventory": inventory,
        "candidate_summary": candidate_summary,
        "competition_train_summary": train_summary,
        "decision": decision,
        "outputs": {
            "manifest_json": str(output_dir / f"{args.label}_manifest.json"),
            "candidate_audit_csv": str(output_dir / f"{args.label}_candidate_audit.csv"),
            "source_inventory_json": str(output_dir / f"{args.label}_source_inventory.json"),
        },
    }
    write_csv(output_dir / f"{args.label}_candidate_audit.csv", audit_rows, CSV_COLUMNS)
    write_json(output_dir / f"{args.label}_source_inventory.json", inventory)
    write_json(output_dir / f"{args.label}_manifest.json", manifest)
    print("candidate_summary =", json.dumps(candidate_summary, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, sort_keys=True), flush=True)
    print("manifest_json =", output_dir / f"{args.label}_manifest.json", flush=True)
    print("=== V446 TONG SOURCE TARGET ALIGNMENT GATE END ===", flush=True)
    return manifest


def run_self_test() -> None:
    print("=== V446 SELF TEST START ===", flush=True)
    assert normalize_text("a\r\n b\t c") == "a b c"
    assert token_ngrams("one two three four", width=2) == {"one two", "two three", "three four"}
    ref_rows = [{"id": "x1", "prompt": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}]
    ref = reference_index(ref_rows)
    candidates = [
        {
            "row_no": 1,
            "source_path": "mock.jsonl",
            "id": "x2",
            "family": "bit_manipulation",
            "status": "rule_found",
            "prompt": "fresh prompt with enough distinct tokens for a separate puzzle instance only",
            "assistant": "<think>ok</think> \\boxed{1010}",
        },
        {
            "row_no": 2,
            "source_path": "mock.jsonl",
            "id": "x1",
            "family": "bit_manipulation",
            "status": "rule_found",
            "prompt": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu",
            "assistant": "<think>ok</think> \\boxed{1010}",
        },
    ]
    audit, summary = audit_candidates(candidates, ref, max_target_chars=1000)
    assert audit[0]["accepted"] is True
    assert audit[1]["accepted"] is False
    assert summary["accepted"] == 1
    print("v446_self_test=ok", flush=True)
    print("=== V446 SELF TEST END ===", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--label", default="v446_tong_source_target_alignment_gate")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--competition-train-csv", type=Path, default=DEFAULT_COMPETITION_TRAIN_CSV)
    parser.add_argument("--reference-weak-csv", type=Path, default=DEFAULT_REFERENCE_WEAK_CSV)
    parser.add_argument("--reference-full-csv", type=Path, default=DEFAULT_REFERENCE_FULL_CSV)
    parser.add_argument("--tong-repo-url", default=DEFAULT_TONG_REPO_URL)
    parser.add_argument("--tong-commit", default=DEFAULT_TONG_COMMIT)
    parser.add_argument("--max-target-chars", type=int, default=12000)
    parser.add_argument("--min-equation-rows-for-gpu", type=int, default=400)
    parser.add_argument("--min-bit-rows-for-gpu", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

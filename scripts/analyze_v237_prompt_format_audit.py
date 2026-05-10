#!/usr/bin/env python3
"""Audit prompt formats in V232 solver workitems.

This script is CPU-only and diagnostic-only. It reads the V232 verified solver
workbench manifest, inspects equation and bit workitem prompt shapes, and emits
CSV summaries that show which parser families are actually needed before any
additional solver or rescue-eval work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"

AUDIT_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "solver_route",
    "priority_score",
    "prompt_sha256",
    "prompt_chars",
    "prompt_lines",
    "backtick_count",
    "arrow_token_count",
    "equals_count",
    "has_now_marker",
    "has_example_word",
    "first_query_marker",
    "query_candidate_excerpt",
    "query_candidate_chars",
    "backtick_pair_count",
    "arrow_pair_count",
    "io_pair_count",
    "total_candidate_pair_count",
    "single_equation_candidate_count",
    "numeric_expr_candidate_count",
    "symbolic_special_char_count",
    "v236_abstain_hint",
    "prompt_excerpt",
]
SUMMARY_COLUMNS = ["family", "solver_route", "first_query_marker", "v236_abstain_hint", "rows"]
SAMPLE_PREVIEW_KEYS = [
    "id",
    "family",
    "solver_route",
    "priority_score",
    "first_query_marker",
    "query_candidate_excerpt",
    "total_candidate_pair_count",
    "single_equation_candidate_count",
    "numeric_expr_candidate_count",
    "v236_abstain_hint",
    "prompt_excerpt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl row: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def compact_text(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_v232_paths(v232_manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = v232_manifest.get("outputs", {})
    required = {
        "equation_solver_workitems_jsonl",
        "bit_guardrail_workitems_jsonl",
        "acceptance_matrix_csv",
        "solver_contracts_json",
    }
    missing = sorted(name for name in required if not outputs.get(name))
    if missing:
        raise RuntimeError("V232 manifest missing outputs: " + json.dumps(missing))
    paths = {name: Path(str(outputs[name])) for name in required}
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return paths


def shared_row_contract_from_v232_manifest(v232_manifest: dict[str, Any]) -> str:
    inputs = v232_manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    return str(
        inputs.get("observed_shared_row_contract_sha256")
        or v232_manifest.get("observed_shared_row_contract_sha256")
        or inputs.get("expected_shared_row_contract_sha256")
        or v232_manifest.get("expected_shared_row_contract_sha256")
        or ""
    )


def query_marker(prompt: str) -> tuple[str, str]:
    patterns = [
        ("now_determine_result_for", r"Now,\s*determine\s+the\s+result\s+for:\s*(.+)$"),
        ("now_verb_colon", r"Now,\s*(?:solve|answer|write|compute)[^:]*:\s*(.+)$"),
        ("question_colon", r"(?:Question|Problem|Input)\s*:\s*(.+)$"),
    ]
    for name, pattern in patterns:
        match = re.search(pattern, prompt, flags=re.I | re.S)
        if match:
            return name, compact_text(match.group(1).splitlines()[0], 180)
    ticks = re.findall(r"`([^`]+)`", prompt)
    if ticks:
        return "last_backtick", compact_text(ticks[-1], 180)
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if lines:
        return "last_nonempty_line", compact_text(lines[-1], 180)
    return "missing", ""


def count_backtick_pairs(prompt: str) -> int:
    count = 0
    for chunk in re.findall(r"`([^`]+)`", prompt):
        if "->" in chunk or "=>" in chunk or " = " in chunk:
            count += 1
    return count


def count_arrow_pairs(prompt: str) -> int:
    return len(re.findall(r"[^\n`]{1,120}\s*(?:->|=>|maps?\s+to|becomes)\s*[^\n`]{1,120}", prompt, flags=re.I))


def count_io_pairs(prompt: str) -> int:
    return len(
        re.findall(
            r"(?:input|in)\s*[:=]\s*[^\n]{1,120}\s+(?:output|out)\s*[:=]\s*[^\n]{1,120}",
            prompt,
            flags=re.I,
        )
    )


def count_single_equations(prompt: str) -> int:
    candidates = re.findall(r"([A-Za-z0-9_().+\-*/^ ]{1,96}=[A-Za-z0-9_().+\-*/^ ]{1,96})", prompt)
    valid = set()
    for candidate in candidates:
        item = re.sub(r"\s+", " ", candidate).strip(" .,:;`")
        if item.count("=") != 1:
            continue
        variables = sorted(set(re.findall(r"[A-Za-z]", item)))
        if len(variables) == 1 and re.search(r"\d", item):
            valid.add(item)
    return len(valid)


def count_numeric_exprs(prompt: str) -> int:
    return len(re.findall(r"\b-?\d+\s*[+\-*/%]\s*-?\d+\b", prompt))


def abstain_hint(prompt: str) -> str:
    marker, query = query_marker(prompt)
    pair_count = count_backtick_pairs(prompt) + count_arrow_pairs(prompt) + count_io_pairs(prompt)
    if count_single_equations(prompt):
        return "single_algebraic_equation_candidate"
    if count_numeric_exprs(prompt) and pair_count:
        return "numeric_expr_with_examples"
    if count_numeric_exprs(prompt):
        return "numeric_expr_without_parseable_examples"
    if pair_count:
        return "example_pairs_nonuniform_or_symbolic"
    if marker in {"missing", "last_nonempty_line"} or not query:
        return "query_marker_missing_or_weak"
    return "prompt_format_requires_manual_parser"


def audit_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    marker, query = query_marker(prompt)
    backtick_pairs = count_backtick_pairs(prompt)
    arrow_pairs = count_arrow_pairs(prompt)
    io_pairs = count_io_pairs(prompt)
    return {
        "schema_version": "kg1_v237_prompt_format_audit_v1",
        "id": str(item.get("id", "")),
        "family": str(item.get("family", "")),
        "solver_route": str(item.get("solver_route", "")),
        "priority_score": int(item.get("priority_score", 0) or 0),
        "prompt_sha256": str(item.get("prompt_sha256") or prompt_sha256(prompt)),
        "prompt_chars": len(prompt),
        "prompt_lines": len(prompt.splitlines()),
        "backtick_count": prompt.count("`"),
        "arrow_token_count": len(re.findall(r"->|=>|maps?\s+to|becomes", prompt, flags=re.I)),
        "equals_count": prompt.count("="),
        "has_now_marker": bool(re.search(r"\bnow\b", prompt, flags=re.I)),
        "has_example_word": bool(re.search(r"\bexamples?\b", prompt, flags=re.I)),
        "first_query_marker": marker,
        "query_candidate_excerpt": query,
        "query_candidate_chars": len(query),
        "backtick_pair_count": backtick_pairs,
        "arrow_pair_count": arrow_pairs,
        "io_pair_count": io_pairs,
        "total_candidate_pair_count": backtick_pairs + arrow_pairs + io_pairs,
        "single_equation_candidate_count": count_single_equations(prompt),
        "numeric_expr_candidate_count": count_numeric_exprs(prompt),
        "symbolic_special_char_count": len(re.findall(r"[{}\\|`'\"!@#$%&<>\[\]]", prompt)),
        "v236_abstain_hint": abstain_hint(prompt),
        "prompt_excerpt": compact_text(prompt),
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(str(row.get(key, "")) for key in keys)] += 1
    return [
        {**{key: values[idx] for idx, key in enumerate(keys)}, "rows": count}
        for values, count in sorted(counts.items())
    ]


def top_prompt_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: (-int(row.get("priority_score", 0)), str(row.get("id", ""))))[:limit]
    return selected


def preview_prompt_samples(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    selected = top_prompt_samples(rows, limit)
    return [{key: row.get(key, "") for key in SAMPLE_PREVIEW_KEYS} for row in selected]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V237 PROMPT FORMAT AUDIT SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v232_analysis_manifest_json =", args.v232_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    v232_manifest = read_json(args.v232_analysis_manifest_json)
    observed_contract = shared_row_contract_from_v232_manifest(v232_manifest)
    if args.expected_shared_row_contract_sha256 and observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    paths = load_v232_paths(v232_manifest)
    equation_items = read_jsonl(paths["equation_solver_workitems_jsonl"])
    bit_items = read_jsonl(paths["bit_guardrail_workitems_jsonl"])
    if not equation_items:
        raise RuntimeError("V237 requires non-empty equation workitems")
    if not bit_items:
        raise RuntimeError("V237 requires non-empty bit workitems")

    audit_rows = [audit_item(item) for item in equation_items + bit_items]
    equation_rows = [row for row in audit_rows if row["family"] == "equation_transform"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "prompt_format_audit_csv": args.output_dir / f"{prefix}_prompt_format_audit.csv",
        "prompt_format_summary_csv": args.output_dir / f"{prefix}_prompt_format_summary.csv",
        "equation_prompt_sample_csv": args.output_dir / f"{prefix}_equation_prompt_samples.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary = summarize(audit_rows, ["family", "solver_route", "first_query_marker", "v236_abstain_hint"])
    equation_prompt_samples = top_prompt_samples(equation_rows, args.sample_limit)
    equation_prompt_sample_preview = preview_prompt_samples(equation_rows, min(args.preview_limit, args.sample_limit))
    write_csv(out_paths["prompt_format_audit_csv"], audit_rows, AUDIT_COLUMNS)
    write_csv(out_paths["prompt_format_summary_csv"], summary, SUMMARY_COLUMNS)
    write_csv(out_paths["equation_prompt_sample_csv"], equation_prompt_samples, AUDIT_COLUMNS)

    equation_hints = summarize(equation_rows, ["v236_abstain_hint"])
    zero_pair_rows = sum(1 for row in equation_rows if int(row["total_candidate_pair_count"]) == 0)
    decision = {
        "decision": "build_prompt_format_specific_parser_before_solver",
        "reason": f"equation_rows={len(equation_rows)}; zero_pair_rows={zero_pair_rows}; sample_limit={args.sample_limit}",
        "next_action": "Inspect prompt_format_audit and equation_prompt_samples, then implement V238 parser tests before any rescue eval.",
    }
    manifest = {
        "schema_version": "kg1_v237_prompt_format_audit_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v232_analysis_manifest_json": str(args.v232_analysis_manifest_json),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "observed_shared_row_contract_sha256": observed_contract,
            **{name: str(path) for name, path in sorted(paths.items())},
        },
        "input_artifact_hashes": {
            "v232_analysis_manifest_json": file_meta(args.v232_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(paths.items())},
        },
        "counts": {
            "equation_workitems": len(equation_items),
            "bit_workitems": len(bit_items),
            "audit_rows": len(audit_rows),
            "equation_zero_candidate_pair_rows": zero_pair_rows,
        },
        "equation_hint_summary": equation_hints,
        "prompt_format_summary": summary,
        "equation_prompt_sample_preview": equation_prompt_sample_preview,
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("equation_hint_summary =", json.dumps(equation_hints, indent=2, sort_keys=True), flush=True)
    print("prompt_format_summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(
        "equation_prompt_sample_preview =",
        json.dumps(equation_prompt_sample_preview, indent=2, sort_keys=True),
        flush=True,
    )
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V237 PROMPT FORMAT AUDIT SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-analysis-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v237_prompt_format_audit")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--sample-limit", type=int, default=30)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        equation_items = root / "equation.jsonl"
        bit_items = root / "bit.jsonl"
        acceptance = root / "acceptance.csv"
        contracts = root / "contracts.json"
        equation_rows = [
            {
                "id": "eq1",
                "family": "equation_transform",
                "solver_route": "sympy_symbolic_transform",
                "priority_score": 9,
                "expected_answer": "2",
                "prompt": "Examples: `a = b` `c = d`. Now, determine the result for: `e`",
            },
            {
                "id": "eq2",
                "family": "equation_transform",
                "solver_route": "numeric_formula_search",
                "priority_score": 8,
                "expected_answer": "7",
                "prompt": "Given 2 + 3 -> 5 and 4 + 1 -> 5. Question: 3 + 4",
            },
        ]
        bit_rows = [
            {
                "id": "bit1",
                "family": "bit_manipulation",
                "solver_route": "bitwise_named_operator_dsl",
                "priority_score": 1,
                "expected_answer": "01010101",
                "prompt": "Apply xor to 00000000 -> 11111111. Now solve 01010101.",
            }
        ]
        with equation_items.open("w", encoding="utf-8") as handle:
            for row in equation_rows:
                handle.write(json.dumps(row) + "\n")
        with bit_items.open("w", encoding="utf-8") as handle:
            for row in bit_rows:
                handle.write(json.dumps(row) + "\n")
        write_csv(acceptance, [{"area": "equation_transform", "criterion": "prompt_audit"}], ["area", "criterion"])
        write_json(contracts, {"schema_version": "kg1_v232_solver_contracts_v1"})
        v232_manifest = root / "v232_manifest.json"
        write_json(
            v232_manifest,
            {
                "expected_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "outputs": {
                    "equation_solver_workitems_jsonl": str(equation_items),
                    "bit_guardrail_workitems_jsonl": str(bit_items),
                    "acceptance_matrix_csv": str(acceptance),
                    "solver_contracts_json": str(contracts),
                },
            },
        )
        args = argparse.Namespace(
            v232_analysis_manifest_json=v232_manifest,
            output_dir=out_dir,
            label="v237_prompt_format_audit",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            sample_limit=2,
            preview_limit=2,
        )
        manifest = run_analysis(args)
        if manifest["counts"]["equation_workitems"] != 2:
            raise AssertionError("expected two equation workitems in self-test")
        if not Path(manifest["outputs"]["prompt_format_audit_csv"]).exists():
            raise AssertionError("prompt format audit output missing")
        if len(manifest["equation_prompt_sample_preview"]) != 2:
            raise AssertionError("expected two preview rows in self-test")
    print("v237_prompt_format_audit_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.v232_analysis_manifest_json is None:
        parser.error("--v232-analysis-manifest-json is required unless --self-test is used")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

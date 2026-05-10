#!/usr/bin/env python3
"""Mine V238 Alice parser abstains into auditable workpacks.

This script is CPU-only and diagnostic-only. It consumes the V238 Alice parser
probe manifest plus the V232 equation workitems referenced by that manifest.
It does not train, run model generation, run full scoring, package artifacts,
download payloads, or submit anything.
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

from analyze_v238_alice_parser_probes import parse_alice_prompt, parse_numeric_token


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
SYMBOLIC_COLUMNS = [
    "schema_version",
    "id",
    "prompt_sha256",
    "query",
    "expected_answer",
    "baseline_prediction",
    "proof",
    "reason_bucket",
    "example_count",
    "query_len",
    "expected_len",
    "baseline_len",
    "query_expected_relation",
    "query_baseline_relation",
    "example_lhs_lengths",
    "example_rhs_lengths",
    "example_len_delta_signature",
    "query_symbols",
    "expected_symbols",
    "baseline_symbols",
    "examples_preview",
    "recommended_next_action",
]
NUMERIC_COLUMNS = [
    "schema_version",
    "id",
    "prompt_sha256",
    "query",
    "query_op",
    "query_left",
    "query_right",
    "expected_answer",
    "baseline_prediction",
    "proof",
    "reason_bucket",
    "example_count",
    "example_ops",
    "same_operator_example_count",
    "examples_preview",
    "recommended_next_action",
]
SUMMARY_COLUMNS = ["reason_bucket", "prompt_kind", "rows"]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "exists": True,
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def symbol_chars(text: str) -> str:
    return "".join(sorted(set(ch for ch in str(text) if not ch.isalnum())))


def examples_preview(examples: list[tuple[str, str]], limit: int = 5) -> str:
    return " | ".join(f"{lhs}={rhs}" for lhs, rhs in examples[:limit])


def relation(query: str, answer: str) -> str:
    query = str(query)
    answer = str(answer)
    if query == answer:
        return "exact"
    if len(query) == len(answer):
        diffs = sum(1 for lhs, rhs in zip(query, answer) if lhs != rhs)
        return f"same_len_substitutions={diffs}"
    if len(query) == len(answer) + 1:
        matches = [
            idx
            for idx in range(len(query))
            if query[:idx] + query[idx + 1 :] == answer
        ]
        if len(matches) == 1:
            return f"single_query_char_deleted_at={matches[0]} char={query[matches[0]]!r}"
        if matches:
            return f"ambiguous_single_delete_matches={len(matches)}"
    if len(answer) == len(query) + 1:
        matches = [
            idx
            for idx in range(len(answer))
            if answer[:idx] + answer[idx + 1 :] == query
        ]
        if len(matches) == 1:
            return f"single_answer_char_inserted_at={matches[0]} char={answer[matches[0]]!r}"
        if matches:
            return f"ambiguous_single_insert_matches={len(matches)}"
    return f"length_delta={len(answer) - len(query)}"


def reason_bucket(prompt_kind: str, proof: str) -> str:
    proof = str(proof)
    if prompt_kind == "alice_symbolic_token_transform":
        if "diagnostic_only_candidate_disabled" in proof:
            return "symbolic_diagnostic_deletion_disabled"
        if "nonuniform_lengths" in proof:
            return "symbolic_nonuniform_lengths"
        if "ambiguous_or_missing_keep_positions=0" in proof:
            return "symbolic_no_keep_positions"
        if "ambiguous_or_missing_keep_positions" in proof:
            return "symbolic_ambiguous_keep_positions"
        if "candidate_rule_count=0" in proof:
            return "symbolic_no_prefix_suffix_rule"
        return "symbolic_other_abstain"
    if "no_examples_for_query_operator" in proof:
        return "numeric_no_examples_for_query_operator"
    match = re.search(r"candidate_rule_count=(\d+)\s+unique_prediction_count=(\d+)", proof)
    if match:
        rule_count = int(match.group(1))
        prediction_count = int(match.group(2))
        if rule_count == 0:
            return "numeric_no_candidate_rule"
        if prediction_count != 1:
            return "numeric_ambiguous_candidate_rules"
    return "numeric_other_abstain"


def recommended_action(bucket: str) -> str:
    if bucket == "symbolic_nonuniform_lengths":
        return "mine length-delta token rules; do not use positional deletion as deployable"
    if bucket == "symbolic_no_keep_positions":
        return "test substitution/permutation rules with exact examples before deployable use"
    if bucket == "symbolic_diagnostic_deletion_disabled":
        return "keep blocked; use only as negative regression fixture"
    if bucket == "numeric_no_examples_for_query_operator":
        return "inspect whether examples contain aliased operator semantics; otherwise abstain"
    if bucket == "numeric_no_candidate_rule":
        return "expand numeric DSL only with unit tests and zero incorrects"
    if bucket == "numeric_ambiguous_candidate_rules":
        return "add tie-breaker verifier or keep abstain"
    return "manual inspection required before any parser extension"


def load_v238_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = manifest.get("outputs", {})
    inputs = manifest.get("inputs", {})
    required_outputs = {"alice_parser_probe_results_csv", "manifest_json"}
    missing_outputs = sorted(name for name in required_outputs if not outputs.get(name))
    if missing_outputs:
        raise RuntimeError("V238 manifest missing outputs: " + json.dumps(missing_outputs))
    if not inputs.get("v232_analysis_manifest_json"):
        raise RuntimeError("V238 manifest missing inputs.v232_analysis_manifest_json")
    paths = {
        "v238_results_csv": Path(str(outputs["alice_parser_probe_results_csv"])),
        "v238_manifest_json": Path(str(outputs["manifest_json"])),
        "v232_manifest_json": Path(str(inputs["v232_analysis_manifest_json"])),
    }
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{name}: {path}")
    return paths


def load_v232_equation_workitems(v232_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(v232_manifest_path)
    outputs = manifest.get("outputs", {})
    path_text = outputs.get("equation_solver_workitems_jsonl")
    if not path_text:
        raise RuntimeError("V232 manifest missing outputs.equation_solver_workitems_jsonl")
    path = Path(str(path_text))
    return read_jsonl(path)


def build_symbolic_row(result: dict[str, str], item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        query = str(result.get("query", ""))
    expected = str(result.get("expected_answer", ""))
    baseline = str(result.get("baseline_prediction", ""))
    bucket = reason_bucket(str(result.get("prompt_kind", "")), str(result.get("proof", "")))
    lhs_lengths = [len(lhs) for lhs, _ in examples]
    rhs_lengths = [len(rhs) for _, rhs in examples]
    delta_signature = ",".join(str(len(rhs) - len(lhs)) for lhs, rhs in examples)
    return {
        "schema_version": "kg1_v239_symbolic_abstain_workitem_v1",
        "id": result.get("id", ""),
        "prompt_sha256": prompt_sha256(prompt),
        "query": query,
        "expected_answer": expected,
        "baseline_prediction": baseline,
        "proof": result.get("proof", ""),
        "reason_bucket": bucket,
        "example_count": len(examples),
        "query_len": len(query),
        "expected_len": len(expected),
        "baseline_len": len(baseline),
        "query_expected_relation": relation(query, expected),
        "query_baseline_relation": relation(query, baseline),
        "example_lhs_lengths": ",".join(map(str, lhs_lengths)),
        "example_rhs_lengths": ",".join(map(str, rhs_lengths)),
        "example_len_delta_signature": delta_signature,
        "query_symbols": symbol_chars(query),
        "expected_symbols": symbol_chars(expected),
        "baseline_symbols": symbol_chars(baseline),
        "examples_preview": examples_preview(examples),
        "recommended_next_action": recommended_action(bucket),
    }


def build_numeric_row(result: dict[str, str], item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    examples, query, parse_status = parse_alice_prompt(prompt)
    if parse_status != "ok":
        query = str(result.get("query", ""))
    parsed_query = parse_numeric_token(query)
    query_left = parsed_query[0] if parsed_query else ""
    query_op = parsed_query[1] if parsed_query else ""
    query_right = parsed_query[2] if parsed_query else ""
    example_ops: list[str] = []
    for lhs, _ in examples:
        parsed = parse_numeric_token(lhs)
        if parsed:
            example_ops.append(parsed[1])
    bucket = reason_bucket(str(result.get("prompt_kind", "")), str(result.get("proof", "")))
    return {
        "schema_version": "kg1_v239_numeric_abstain_workitem_v1",
        "id": result.get("id", ""),
        "prompt_sha256": prompt_sha256(prompt),
        "query": query,
        "query_op": query_op,
        "query_left": query_left,
        "query_right": query_right,
        "expected_answer": result.get("expected_answer", ""),
        "baseline_prediction": result.get("baseline_prediction", ""),
        "proof": result.get("proof", ""),
        "reason_bucket": bucket,
        "example_count": len(examples),
        "example_ops": "".join(sorted(set(example_ops))),
        "same_operator_example_count": sum(1 for op in example_ops if op == query_op),
        "examples_preview": examples_preview(examples),
        "recommended_next_action": recommended_action(bucket),
    }


def summarize_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter(
        (str(row.get("reason_bucket", "")), str(row.get("prompt_kind", ""))) for row in rows
    )
    return [
        {"reason_bucket": bucket, "prompt_kind": prompt_kind, "rows": rows_count}
        for (bucket, prompt_kind), rows_count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V239 ALICE ABSTAIN MINING SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v238_analysis_manifest_json =", args.v238_analysis_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)

    v238_manifest = read_json(args.v238_analysis_manifest_json)
    inputs = v238_manifest.get("inputs", {})
    observed_contract = str(inputs.get("observed_shared_row_contract_sha256") or inputs.get("expected_shared_row_contract_sha256") or "")
    if args.expected_shared_row_contract_sha256 and observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )
    paths = load_v238_paths(v238_manifest)
    v238_results = read_csv(paths["v238_results_csv"])
    if not v238_results:
        raise RuntimeError("V239 requires non-empty V238 result rows")
    equation_items = load_v232_equation_workitems(paths["v232_manifest_json"])
    item_by_id = {str(item.get("id", "")): item for item in equation_items}
    missing_ids = sorted(str(row.get("id", "")) for row in v238_results if str(row.get("id", "")) not in item_by_id)
    if missing_ids:
        raise RuntimeError("V238 results missing V232 workitems for ids: " + json.dumps(missing_ids[:20]))

    abstains = [row for row in v238_results if row.get("status") == "abstain"]
    verified = [row for row in v238_results if row.get("status") == "verified"]
    incorrect = [row for row in v238_results if row.get("status") == "incorrect"]
    symbolic_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    for row in abstains:
        item = item_by_id[str(row.get("id", ""))]
        prompt_kind = str(row.get("prompt_kind", ""))
        if prompt_kind == "alice_symbolic_token_transform":
            mined = build_symbolic_row(row, item)
            symbolic_rows.append(mined)
        elif prompt_kind == "alice_numeric_binary_operator":
            mined = build_numeric_row(row, item)
            numeric_rows.append(mined)
        else:
            continue
        all_bucket_rows.append({"reason_bucket": mined.get("reason_bucket", ""), "prompt_kind": prompt_kind})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "symbolic_abstain_workpack_csv": args.output_dir / f"{prefix}_symbolic_abstain_workpack.csv",
        "numeric_abstain_workpack_csv": args.output_dir / f"{prefix}_numeric_abstain_workpack.csv",
        "abstain_bucket_summary_csv": args.output_dir / f"{prefix}_abstain_bucket_summary.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary = summarize_buckets(all_bucket_rows)
    write_csv(out_paths["symbolic_abstain_workpack_csv"], symbolic_rows, SYMBOLIC_COLUMNS)
    write_csv(out_paths["numeric_abstain_workpack_csv"], numeric_rows, NUMERIC_COLUMNS)
    write_csv(out_paths["abstain_bucket_summary_csv"], summary, SUMMARY_COLUMNS)

    if incorrect:
        decision = {
            "decision": "stop_parser_development_until_incorrects_are_hardened",
            "reason": f"incorrect={len(incorrect)} in V238 input",
            "next_action": "Harden V238 parser before mining abstains.",
        }
    elif len(verified) >= int(args.target_gain):
        decision = {
            "decision": "v238_already_met_target_gain",
            "reason": f"verified={len(verified)} >= target_gain={args.target_gain}",
            "next_action": "Create separate gated measurement notebook only if weak gate policy allows it.",
        }
    else:
        decision = {
            "decision": "mine_abstains_before_any_rescue_measurement",
            "reason": f"verified={len(verified)}; incorrect=0; symbolic_abstains={len(symbolic_rows)}; numeric_abstains={len(numeric_rows)}",
            "next_action": "Use V239 workpacks to design unit-tested parser rules; keep all ambiguous routes abstain.",
        }

    manifest = {
        "schema_version": "kg1_v239_alice_abstain_mining_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "v238_analysis_manifest_json": str(args.v238_analysis_manifest_json),
            "expected_shared_row_contract_sha256": str(args.expected_shared_row_contract_sha256),
            "observed_shared_row_contract_sha256": observed_contract,
            **{name: str(path) for name, path in sorted(paths.items())},
        },
        "input_artifact_hashes": {
            "v238_analysis_manifest_json": file_meta(args.v238_analysis_manifest_json),
            **{name: file_meta(path) for name, path in sorted(paths.items())},
        },
        "counts": {
            "v238_rows": len(v238_results),
            "verified": len(verified),
            "incorrect": len(incorrect),
            "abstain": len(abstains),
            "symbolic_abstain": len(symbolic_rows),
            "numeric_abstain": len(numeric_rows),
            "target_gain": int(args.target_gain),
        },
        "abstain_bucket_summary": summary,
        "symbolic_preview": symbolic_rows[:12],
        "numeric_preview": numeric_rows[:12],
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("abstain_bucket_summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("symbolic_preview =", json.dumps(symbolic_rows[:12], indent=2, sort_keys=True), flush=True)
    print("numeric_preview =", json.dumps(numeric_rows[:12], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V239 ALICE ABSTAIN MINING SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v238-analysis-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v239_alice_abstain_mining")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-gain", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        v232_workitems = root / "equation_workitems.jsonl"
        prompts = {
            "num_abstain": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 72)27 = 99 26#48 = 22 Now, determine the result for: 11-50",
            "sym_abstain": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: abcd = ac xyzw = xzw Now, determine the result for: lmno",
            "num_verified": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 72)27 = 99 94)40 = 134 Now, determine the result for: 94)40",
        }
        with v232_workitems.open("w", encoding="utf-8") as handle:
            for key, prompt in prompts.items():
                handle.write(json.dumps({"id": key, "prompt": prompt}) + "\n")
        v232_manifest = root / "v232_manifest.json"
        write_json(v232_manifest, {"outputs": {"equation_solver_workitems_jsonl": str(v232_workitems)}})
        v238_results = root / "v238_results.csv"
        rows = [
            {
                "id": "num_abstain",
                "prompt_kind": "alice_numeric_binary_operator",
                "status": "abstain",
                "query": "11-50",
                "expected_answer": "-39",
                "baseline_prediction": "39",
                "proof": "no_examples_for_query_operator='-'",
            },
            {
                "id": "sym_abstain",
                "prompt_kind": "alice_symbolic_token_transform",
                "status": "abstain",
                "query": "lmno",
                "expected_answer": "lno",
                "baseline_prediction": "xxx",
                "proof": "candidate_count=0 unique_prediction_count=0; alice_symbolic_deletion_positions_probe:nonuniform_lengths",
            },
            {
                "id": "num_verified",
                "prompt_kind": "alice_numeric_binary_operator",
                "status": "verified",
                "query": "94)40",
                "expected_answer": "134",
                "baseline_prediction": "35",
                "proof": "rules=add",
            },
        ]
        write_csv(
            v238_results,
            rows,
            ["id", "prompt_kind", "status", "query", "expected_answer", "baseline_prediction", "proof"],
        )
        v238_manifest_self = root / "v238_manifest.json"
        write_json(
            v238_manifest_self,
            {
                "inputs": {
                    "v232_analysis_manifest_json": str(v232_manifest),
                    "observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                },
                "outputs": {
                    "alice_parser_probe_results_csv": str(v238_results),
                    "manifest_json": str(root / "v238_inner_manifest.json"),
                },
            },
        )
        write_json(root / "v238_inner_manifest.json", {"ok": True})
        args = argparse.Namespace(
            v238_analysis_manifest_json=v238_manifest_self,
            output_dir=out_dir,
            label="v239_alice_abstain_mining",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            target_gain=5,
        )
        manifest = run_analysis(args)
        counts = manifest["counts"]
        if counts["verified"] != 1 or counts["incorrect"] != 0:
            raise AssertionError("unexpected self-test verified/incorrect counts")
        if counts["symbolic_abstain"] != 1 or counts["numeric_abstain"] != 1:
            raise AssertionError("unexpected self-test abstain counts")
        if manifest["decision"]["decision"] != "mine_abstains_before_any_rescue_measurement":
            raise AssertionError("unexpected self-test decision")
    print("v239_alice_abstain_mining_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.v238_analysis_manifest_json is None:
        parser.error("--v238-analysis-manifest-json is required unless --self-test is used")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

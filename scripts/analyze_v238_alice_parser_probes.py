#!/usr/bin/env python3
"""Run conservative parser probes for Alice-format equation prompts.

This script consumes V232 solver workitems. It targets the prompt family shown
by V237: "In Alice's Wonderland..." prompts with inline examples like
``lhs = rhs`` followed by ``Now, determine the result for: query``.

It is CPU-only and diagnostic-only. It does not train, run model generation,
run full scoring, package artifacts, download payloads, or submit anything.
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
RESULT_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "solver_route",
    "prompt_sha256",
    "parse_status",
    "prompt_kind",
    "example_count",
    "query",
    "probe_name",
    "deployable",
    "status",
    "prediction",
    "expected_answer",
    "baseline_prediction",
    "proof",
]
SUMMARY_COLUMNS = ["prompt_kind", "probe_name", "status", "rows"]
PREVIEW_COLUMNS = [
    "id",
    "prompt_kind",
    "probe_name",
    "status",
    "query",
    "prediction",
    "expected_answer",
    "baseline_prediction",
    "proof",
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


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.strip("` $")
    text = re.sub(r"\s+", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text


def answers_equal(left: Any, right: Any) -> bool:
    lhs = normalize_answer(left)
    rhs = normalize_answer(right)
    if lhs == rhs:
        return True
    try:
        return Fraction(lhs) == Fraction(rhs)
    except Exception:
        return False


def strip_token(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1].strip()
    return text


def parse_alice_prompt(prompt: str) -> tuple[list[tuple[str, str]], str, str]:
    example_match = re.search(
        r"Below\s+are\s+a\s+few\s+examples:\s*(?P<body>.*?)(?:\bNow,\s*determine\s+the\s+result\s+for:\s*)(?P<query>.+)$",
        prompt,
        flags=re.I | re.S,
    )
    if not example_match:
        return [], "", "alice_marker_not_found"
    body = example_match.group("body")
    query = strip_token(example_match.group("query").splitlines()[0])
    examples: list[tuple[str, str]] = []
    for match in re.finditer(r"(?P<lhs>\S+)\s*=\s*(?P<rhs>\S+)", body):
        lhs = strip_token(match.group("lhs"))
        rhs = strip_token(match.group("rhs"))
        if lhs and rhs:
            examples.append((lhs, rhs))
    if not examples:
        return [], query, "alice_examples_not_parseable"
    return examples, query, "ok"


def parse_numeric_token(value: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(r"(-?\d+)([^\d\s])(-?\d+)", value)
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def classify_prompt(examples: list[tuple[str, str]], query: str) -> str:
    if examples and parse_numeric_token(query) and all(parse_numeric_token(lhs) for lhs, _ in examples):
        return "alice_numeric_binary_operator"
    if examples:
        return "alice_symbolic_token_transform"
    return "unknown"


def digitwise_add_mod10(a: int, b: int) -> str | None:
    sa, sb = str(abs(a)), str(abs(b))
    if len(sa) != len(sb):
        return None
    return "".join(str((int(x) + int(y)) % 10) for x, y in zip(sa, sb))


def digitwise_absdiff(a: int, b: int) -> str | None:
    sa, sb = str(abs(a)), str(abs(b))
    if len(sa) != len(sb):
        return None
    return "".join(str(abs(int(x) - int(y))) for x, y in zip(sa, sb))


def numeric_rules() -> dict[str, Callable[[int, int], str | None]]:
    return {
        "add": lambda a, b: str(a + b),
        "sub_ab": lambda a, b: str(a - b),
        "sub_ba": lambda a, b: str(b - a),
        "abs_diff": lambda a, b: str(abs(a - b)),
        "mul": lambda a, b: str(a * b),
        "concat_ab": lambda a, b: f"{abs(a)}{abs(b)}",
        "concat_ba": lambda a, b: f"{abs(b)}{abs(a)}",
        "digitwise_absdiff": digitwise_absdiff,
        "sum_digits": lambda a, b: str(sum(int(ch) for ch in str(abs(a)) + str(abs(b)))),
    }


def numeric_operator_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    query_expr = parse_numeric_token(query)
    if query_expr is None:
        return probe_abstain("alice_numeric_operator_rule_probe", "query_not_numeric_binary")
    query_op = query_expr[1]
    same_op: list[tuple[tuple[int, str, int], str]] = []
    for lhs, rhs in examples:
        expr = parse_numeric_token(lhs)
        if expr and expr[1] == query_op:
            same_op.append((expr, normalize_answer(rhs)))
    if not same_op:
        return probe_abstain("alice_numeric_operator_rule_probe", f"no_examples_for_query_operator={query_op!r}")
    candidates: list[tuple[str, str]] = []
    for name, func in numeric_rules().items():
        outputs: list[str] = []
        ok = True
        for expr, _ in same_op:
            value = func(expr[0], expr[2])
            if value is None:
                ok = False
                break
            outputs.append(normalize_answer(value))
        if ok and outputs == [rhs for _, rhs in same_op]:
            prediction = func(query_expr[0], query_expr[2])
            if prediction is not None:
                candidates.append((name, str(prediction)))
    unique_predictions = sorted(set(prediction for _, prediction in candidates))
    if len(unique_predictions) != 1:
        return probe_abstain(
            "alice_numeric_operator_rule_probe",
            f"candidate_rule_count={len(candidates)} unique_prediction_count={len(unique_predictions)}",
        )
    return {
        "probe_name": "alice_numeric_operator_rule_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": unique_predictions[0],
        "proof": "rules=" + ",".join(name for name, _ in candidates),
    }


def char_map_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    mapping: dict[str, str] = {}
    for lhs, rhs in examples:
        if len(lhs) != len(rhs):
            return probe_abstain("alice_symbolic_char_map_probe", "example_length_mismatch")
        for src, dst in zip(lhs, rhs):
            if src in mapping and mapping[src] != dst:
                return probe_abstain("alice_symbolic_char_map_probe", f"conflicting_mapping_for={src!r}")
            mapping[src] = dst
    missing = sorted(set(query) - set(mapping))
    if missing:
        return probe_abstain("alice_symbolic_char_map_probe", "query_contains_unmapped_symbols=" + repr("".join(missing)))
    return {
        "probe_name": "alice_symbolic_char_map_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": "".join(mapping[ch] for ch in query),
        "proof": f"consistent_char_map_size={len(mapping)}",
    }


def deletion_positions_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    if not examples:
        return probe_abstain("alice_symbolic_deletion_positions_probe", "missing_examples")
    lhs_len = len(examples[0][0])
    rhs_len = len(examples[0][1])
    if rhs_len > lhs_len or any(len(lhs) != lhs_len or len(rhs) != rhs_len for lhs, rhs in examples):
        return probe_abstain("alice_symbolic_deletion_positions_probe", "nonuniform_lengths")
    keep_sets: list[tuple[int, ...]] = []
    for lhs, rhs in examples:
        matches: list[tuple[int, ...]] = []
        for indexes in combinations(range(lhs_len), rhs_len):
            if "".join(lhs[idx] for idx in indexes) == rhs:
                matches.append(indexes)
        if len(matches) != 1:
            return probe_abstain("alice_symbolic_deletion_positions_probe", f"ambiguous_or_missing_keep_positions={len(matches)}")
        keep_sets.append(matches[0])
    if len(set(keep_sets)) != 1:
        return probe_abstain("alice_symbolic_deletion_positions_probe", "inconsistent_keep_positions")
    keep = keep_sets[0]
    if len(query) != lhs_len:
        return probe_abstain("alice_symbolic_deletion_positions_probe", "query_length_mismatch")
    return {
        "probe_name": "alice_symbolic_deletion_positions_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": "".join(query[idx] for idx in keep),
        "proof": "keep_positions=" + ",".join(map(str, keep)),
    }


def reverse_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    if all(rhs == lhs[::-1] for lhs, rhs in examples):
        return {
            "probe_name": "alice_symbolic_reverse_probe",
            "deployable": True,
            "status": "candidate",
            "prediction": query[::-1],
            "proof": "all_examples_match_reverse",
        }
    return probe_abstain("alice_symbolic_reverse_probe", "examples_do_not_match_reverse")


def prefix_suffix_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    rules: list[tuple[str, int]] = []
    lhs_lengths = {len(lhs) for lhs, _ in examples}
    rhs_lengths = {len(rhs) for _, rhs in examples}
    if len(lhs_lengths) != 1 or len(rhs_lengths) != 1:
        return probe_abstain("alice_symbolic_prefix_suffix_probe", "nonuniform_lengths")
    rhs_len = next(iter(rhs_lengths))
    if all(rhs == lhs[:rhs_len] for lhs, rhs in examples):
        rules.append(("prefix", rhs_len))
    if all(rhs == lhs[-rhs_len:] for lhs, rhs in examples):
        rules.append(("suffix", rhs_len))
    if len(rules) != 1:
        return probe_abstain("alice_symbolic_prefix_suffix_probe", f"candidate_rule_count={len(rules)}")
    name, length = rules[0]
    prediction = query[:length] if name == "prefix" else query[-length:]
    return {
        "probe_name": "alice_symbolic_prefix_suffix_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": prediction,
        "proof": f"{name}_length={length}",
    }


def combinations(values: range, size: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def walk(start: int, picked: list[int]) -> None:
        if len(picked) == size:
            result.append(tuple(picked))
            return
        for value in range(start, len(values)):
            picked.append(values[value])
            walk(value + 1, picked)
            picked.pop()

    walk(0, [])
    return result


def probe_abstain(name: str, proof: str) -> dict[str, Any]:
    return {"probe_name": name, "deployable": True, "status": "abstain", "prediction": "", "proof": proof}


def symbolic_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    probes = [
        deletion_positions_probe,
        reverse_probe,
        prefix_suffix_probe,
    ]
    candidates: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for probe in probes:
        result = probe(examples, query)
        if result.get("status") == "candidate":
            candidates.append(result)
        else:
            abstains.append(result)
    unique_predictions = sorted(set(str(item.get("prediction", "")) for item in candidates))
    if len(unique_predictions) != 1:
        proofs = "; ".join(f"{item['probe_name']}:{item['proof']}" for item in abstains[:4])
        return probe_abstain(
            "alice_symbolic_rule_probe",
            f"candidate_count={len(candidates)} unique_prediction_count={len(unique_predictions)}; {proofs}",
        )
    names = ",".join(str(item.get("probe_name", "")) for item in candidates)
    return {
        "probe_name": "alice_symbolic_rule_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": unique_predictions[0],
        "proof": "candidate_probes=" + names,
    }


def choose_probe(prompt_kind: str, examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    if prompt_kind == "alice_numeric_binary_operator":
        return numeric_operator_probe(examples, query)
    if prompt_kind == "alice_symbolic_token_transform":
        return symbolic_probe(examples, query)
    return probe_abstain("alice_parser_probe", "unsupported_prompt_kind")


def load_v232_paths(v232_manifest: dict[str, Any]) -> dict[str, Path]:
    outputs = v232_manifest.get("outputs", {})
    required = {"equation_solver_workitems_jsonl", "bit_guardrail_workitems_jsonl"}
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


def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    examples, query, parse_status = parse_alice_prompt(prompt)
    kind = classify_prompt(examples, query)
    result = choose_probe(kind, examples, query)
    expected = str(item.get("expected_answer", ""))
    prediction = str(result.get("prediction", ""))
    status = str(result.get("status", "abstain"))
    if status == "candidate":
        status = "verified" if answers_equal(prediction, expected) else "incorrect"
    return {
        "schema_version": "kg1_v238_alice_parser_probe_result_v1",
        "id": str(item.get("id", "")),
        "family": str(item.get("family", "")),
        "solver_route": str(item.get("solver_route", "")),
        "prompt_sha256": str(item.get("prompt_sha256") or prompt_sha256(prompt)),
        "parse_status": parse_status,
        "prompt_kind": kind,
        "example_count": len(examples),
        "query": query,
        "probe_name": str(result.get("probe_name", "")),
        "deployable": bool(result.get("deployable", False)),
        "status": status,
        "prediction": prediction,
        "expected_answer": expected,
        "baseline_prediction": str(item.get("baseline_prediction", "")),
        "proof": str(result.get("proof", "")),
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(str(row.get(key, "")) for key in keys)] += 1
    return [
        {**{key: values[idx] for idx, key in enumerate(keys)}, "rows": count}
        for values, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def preview_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in rows[:limit]:
        preview.append({column: row.get(column, "") for column in PREVIEW_COLUMNS})
    return preview


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V238 ALICE PARSER PROBES SCRIPT START ===", flush=True)
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
    if not equation_items:
        raise RuntimeError("V238 requires non-empty equation workitems")
    results = [evaluate_item(item) for item in equation_items]
    verified = [row for row in results if row["deployable"] and row["status"] == "verified"]
    incorrect = [row for row in results if row["deployable"] and row["status"] == "incorrect"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "alice_parser_probe_results_csv": args.output_dir / f"{prefix}_alice_parser_probe_results.csv",
        "alice_parser_probe_summary_csv": args.output_dir / f"{prefix}_alice_parser_probe_summary.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    summary = summarize(results, ["prompt_kind", "probe_name", "status"])
    abstain_reason_summary = summarize(
        [row for row in results if row.get("status") == "abstain"],
        ["prompt_kind", "probe_name", "proof"],
    )
    verified_preview = preview_rows(verified)
    incorrect_preview = preview_rows(incorrect)
    abstain_preview = preview_rows([row for row in results if row.get("status") == "abstain"])
    write_csv(out_paths["alice_parser_probe_results_csv"], results, RESULT_COLUMNS)
    write_csv(out_paths["alice_parser_probe_summary_csv"], summary, SUMMARY_COLUMNS)

    if len(verified) >= args.target_gain and not incorrect:
        decision = {
            "decision": "prepare_gated_alice_parser_rescue_measurement",
            "reason": f"verified={len(verified)} >= target_gain={args.target_gain}; incorrect=0",
            "next_action": "Create a separate measurement notebook that applies only verified Alice parser overrides.",
        }
    else:
        decision = {
            "decision": "continue_alice_parser_development",
            "reason": f"verified={len(verified)}; incorrect={len(incorrect)}; target_gain={args.target_gain}",
            "next_action": "Inspect incorrect and abstain rows; add only unit-tested rules before any rescue measurement.",
        }

    manifest = {
        "schema_version": "kg1_v238_alice_parser_probes_manifest_v1",
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
            "deployable_verified_overrides": len(verified),
            "deployable_incorrect_overrides": len(incorrect),
            "target_gain": int(args.target_gain),
        },
        "summary": summary,
        "abstain_reason_summary_top": abstain_reason_summary[:30],
        "verified_preview": verified_preview,
        "incorrect_preview": incorrect_preview,
        "abstain_preview": abstain_preview,
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("summary =", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("abstain_reason_summary_top =", json.dumps(abstain_reason_summary[:30], indent=2, sort_keys=True), flush=True)
    print("verified_preview =", json.dumps(verified_preview, indent=2, sort_keys=True), flush=True)
    print("incorrect_preview =", json.dumps(incorrect_preview, indent=2, sort_keys=True), flush=True)
    print("abstain_preview =", json.dumps(abstain_preview, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V238 ALICE PARSER PROBES SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-analysis-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v238_alice_parser_probes")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-gain", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        equation_items = root / "equation.jsonl"
        bit_items = root / "bit.jsonl"
        equation_rows = [
            {
                "id": "num_add",
                "family": "equation_transform",
                "solver_route": "sympy_symbolic_transform",
                "expected_answer": "134",
                "baseline_prediction": "0",
                "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: 72)27 = 99 26#48 = 22 42#45 = 3 24#14 = 10 Now, determine the result for: 94)40",
            },
            {
                "id": "sym_delete",
                "family": "equation_transform",
                "solver_route": "sympy_symbolic_transform",
                "expected_answer": "lno",
                "baseline_prediction": "xxx",
                "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: abcd = acd xyzw = xzw pqrs = prs Now, determine the result for: lmno",
            },
        ]
        bit_rows = [{"id": "bit1", "family": "bit_manipulation", "prompt": "noop"}]
        with equation_items.open("w", encoding="utf-8") as handle:
            for row in equation_rows:
                handle.write(json.dumps(row) + "\n")
        with bit_items.open("w", encoding="utf-8") as handle:
            for row in bit_rows:
                handle.write(json.dumps(row) + "\n")
        v232_manifest = root / "v232_manifest.json"
        write_json(
            v232_manifest,
            {
                "expected_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256,
                "outputs": {
                    "equation_solver_workitems_jsonl": str(equation_items),
                    "bit_guardrail_workitems_jsonl": str(bit_items),
                },
            },
        )
        args = argparse.Namespace(
            v232_analysis_manifest_json=v232_manifest,
            output_dir=out_dir,
            label="v238_alice_parser_probes",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            target_gain=2,
        )
        manifest = run_analysis(args)
        if manifest["counts"]["deployable_verified_overrides"] != 2:
            raise AssertionError("expected two verified overrides in self-test")
        if manifest["counts"]["deployable_incorrect_overrides"] != 0:
            raise AssertionError("expected zero incorrect overrides in self-test")
    print("v238_alice_parser_probes_self_test=ok", flush=True)
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

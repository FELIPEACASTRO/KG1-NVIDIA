#!/usr/bin/env python3
"""Run local conservative DSL probes over V232 solver workitems.

This script is CPU-only. It consumes the V232 verified solver workbench
manifest and emits auditable local probe artifacts for equation_transform and
bit_manipulation. It does not train, run model generation, run full scoring,
package artifacts, download external payloads, or submit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
EQUATION_AUDIT_COLUMNS = [
    "schema_version",
    "id",
    "subtype",
    "example_count",
    "query",
    "expected_answer",
    "baseline_prediction",
    "prompt_sha256",
    "notes",
]
PROBE_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "subtype",
    "probe_name",
    "deployable",
    "status",
    "prediction",
    "expected_answer",
    "baseline_prediction",
    "prompt_sha256",
    "proof",
]
ABSTAIN_REASON_COLUMNS = ["subtype", "probe_name", "status", "proof", "rows"]
BIT_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "probe_name",
    "status",
    "allowed_ops_seen",
    "expected_answer",
    "baseline_prediction",
    "prompt_sha256",
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


def format_number(value: Any) -> str:
    numerator = getattr(value, "p", None)
    denominator = getattr(value, "q", None)
    if numerator is not None and denominator is not None:
        if int(denominator) == 1:
            return str(int(numerator))
        return f"{int(numerator)}/{int(denominator)}"
    try:
        number = float(value)
        if math.isfinite(number) and abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
    except Exception:
        pass
    return str(value).strip()


def import_sympy() -> Any | None:
    try:
        import sympy as sp  # type: ignore

        return sp
    except Exception:
        return None


def extract_algebraic_equation_from_text(text: str) -> tuple[str | None, str]:
    # Conservative extraction: one short equality, one alphabetic variable,
    # at least one digit, and only algebra-safe characters.
    candidates = re.findall(
        r"([A-Za-z0-9_().+\-*/^ ]{1,96}=[A-Za-z0-9_().+\-*/^ ]{1,96})",
        text,
    )
    cleaned: list[str] = []
    for candidate in candidates:
        item = re.sub(r"\s+", " ", candidate).strip(" .,:;`")
        if not item or item.count("=") != 1:
            continue
        variables = sorted(set(re.findall(r"[A-Za-z]", item)))
        if len(variables) == 1 and re.search(r"\d", item):
            cleaned.append(item)
    unique = sorted(set(cleaned))
    if not unique:
        return None, "no_single_algebraic_equation_found"
    if len(unique) > 1:
        return None, "ambiguous_multiple_algebraic_equations"
    return unique[0], "ok"


def extract_single_algebraic_equation(prompt: str, query: str) -> tuple[str | None, str]:
    if query:
        equation, reason = extract_algebraic_equation_from_text(query)
        if equation:
            return equation, "query:" + reason
    equation, reason = extract_algebraic_equation_from_text(prompt)
    return equation, "prompt:" + reason


def algebraic_equation_probe(prompt: str, query: str) -> dict[str, Any]:
    equation, reason = extract_single_algebraic_equation(prompt, query)
    if not equation:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": reason,
        }
    sp = import_sympy()
    if sp is None:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "sympy_unavailable",
        }
    variables = sorted(set(re.findall(r"[A-Za-z]", equation)))
    if len(variables) != 1:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "variable_count_not_one",
        }
    var = sp.symbols(variables[0])
    local_dict = {variables[0]: var}
    lhs_text, rhs_text = equation.split("=", 1)
    try:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (implicit_multiplication_application,)
        lhs = parse_expr(lhs_text.replace("^", "**"), local_dict=local_dict, transformations=transformations)
        rhs = parse_expr(rhs_text.replace("^", "**"), local_dict=local_dict, transformations=transformations)
        solutions = sp.solve(sp.Eq(lhs, rhs), var)
    except Exception as exc:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "sympy_parse_or_solve_failed: " + repr(exc),
        }
    if len(solutions) != 1:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "solution_count_not_one",
        }
    solution = solutions[0]
    try:
        verified = sp.simplify(lhs.subs(var, solution) - rhs.subs(var, solution)) == 0
    except Exception:
        verified = False
    if not verified:
        return {
            "probe_name": "sympy_single_equation_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "solution_failed_symbolic_verification",
        }
    return {
        "probe_name": "sympy_single_equation_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": format_number(solution),
        "proof": f"solved {equation} for {variables[0]} and verified substitution",
    }


def strip_ticks(text: str) -> str:
    return str(text).strip().strip("`").strip()


def split_examples_and_query(prompt: str) -> tuple[list[tuple[str, str]], str, str]:
    marker_match = re.search(r"Now,\s*determine\s+the\s+result\s+for:\s*(.+)$", prompt, flags=re.I | re.S)
    if not marker_match:
        marker_match = re.search(r"Now,\s*(?:solve|answer|write|compute)[^:]*:\s*(.+)$", prompt, flags=re.I | re.S)
    if not marker_match:
        return [], "", "query_marker_not_found"
    query = strip_ticks(marker_match.group(1).splitlines()[0].strip())
    before = prompt[: marker_match.start()]
    examples: list[tuple[str, str]] = []
    for chunk in re.findall(r"`([^`]+)`", before):
        item = strip_ticks(chunk)
        if "->" in item:
            lhs, rhs = item.split("->", 1)
        elif " = " in item:
            lhs, rhs = item.split(" = ", 1)
        else:
            continue
        lhs = strip_ticks(lhs)
        rhs = strip_ticks(rhs)
        if lhs and rhs:
            examples.append((lhs, rhs))
    return examples, query, "ok"


def classify_equation(prompt: str, examples: list[tuple[str, str]], query: str) -> tuple[str, str]:
    example_query_text = " ".join([query, " ".join(x + " " + y for x, y in examples)])
    all_text = " ".join([prompt, example_query_text])
    algebraic_equation, algebraic_reason = extract_single_algebraic_equation(prompt, query)
    if algebraic_equation:
        return "algebraic_equation", "single_equation_parseable;" + algebraic_reason
    if re.search(r"\d+\s*[+\-*/%]\s*\d+", example_query_text):
        return "numeric_operator_transform", "numeric_binary_operator_signature"
    if examples and (
        re.search(r"[{}\\|`'\"!@#$%&<>\[\]]", example_query_text)
        or any(re.search(r"[A-Za-z]{2,}", lhs + rhs + query) for lhs, rhs in examples)
    ):
        return "symbolic_mixed_token_rewrite", "symbolic_or_special_examples_present"
    if re.search(r"\b[a-zA-Z]\b", all_text) and "=" in all_text:
        return "algebraic_equation_unparsed", "single_letter_variable_or_equation_signature_without_safe_parse"
    if re.search(r"[{}\\|`'\"!@#$%&<>\[\]]", example_query_text):
        return "symbolic_mixed_token_rewrite", "special_symbols_present"
    if examples:
        return "sequence_token_transform", "examples_present_without_numeric_signature"
    return "unknown", "no_supported_signature"


def symbolic_char_map_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    if not examples or not query:
        return {
            "probe_name": "symbolic_char_map_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "missing_examples_or_query",
        }
    mapping: dict[str, str] = {}
    for lhs, rhs in examples:
        if len(lhs) != len(rhs):
            return {
                "probe_name": "symbolic_char_map_probe",
                "deployable": True,
                "status": "abstain",
                "prediction": "",
                "proof": "example_length_mismatch",
            }
        for src, dst in zip(lhs, rhs):
            if src in mapping and mapping[src] != dst:
                return {
                    "probe_name": "symbolic_char_map_probe",
                    "deployable": True,
                    "status": "abstain",
                    "prediction": "",
                    "proof": f"conflicting_mapping_for_{src!r}",
                }
            mapping[src] = dst
    missing = sorted(set(query) - set(mapping))
    if missing:
        return {
            "probe_name": "symbolic_char_map_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "query_contains_unmapped_symbols=" + repr("".join(missing)),
        }
    return {
        "probe_name": "symbolic_char_map_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": "".join(mapping[ch] for ch in query),
        "proof": f"consistent_char_map_size={len(mapping)}",
    }


def reverse_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    if not examples or not query:
        return {
            "probe_name": "reverse_token_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "missing_examples_or_query",
        }
    if all(rhs == lhs[::-1] for lhs, rhs in examples):
        return {
            "probe_name": "reverse_token_probe",
            "deployable": True,
            "status": "candidate",
            "prediction": query[::-1],
            "proof": "all_examples_match_reverse",
        }
    return {
        "probe_name": "reverse_token_probe",
        "deployable": True,
        "status": "abstain",
        "prediction": "",
        "proof": "examples_do_not_match_reverse",
    }


def parse_numeric_expr(text: str) -> tuple[int, str, int] | None:
    match = re.fullmatch(r"\s*(-?\d+)\s*([+\-*/%])\s*(-?\d+)\s*", text)
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def eval_op(a: int, op: str, b: int) -> int | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and b != 0 and a % b == 0:
        return a // b
    if op == "%" and b != 0:
        return a % b
    return None


def digitwise_add_mod10(a: int, b: int) -> str | None:
    sa, sb = str(abs(a)), str(abs(b))
    if len(sa) != len(sb):
        return None
    return "".join(str((int(x) + int(y)) % 10) for x, y in zip(sa, sb))


def numeric_operator_probe(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    parsed_examples = [(parse_numeric_expr(lhs), normalize_answer(rhs)) for lhs, rhs in examples]
    parsed_query = parse_numeric_expr(query)
    if not parsed_examples or parsed_query is None or any(item[0] is None for item in parsed_examples):
        return {
            "probe_name": "numeric_operator_dsl_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": "numeric_examples_or_query_not_parseable",
        }

    def direct(expr: tuple[int, str, int]) -> str | None:
        value = eval_op(*expr)
        return None if value is None else str(value)

    def reverse_result(expr: tuple[int, str, int]) -> str | None:
        value = direct(expr)
        return None if value is None else value[::-1]

    def reverse_operands(expr: tuple[int, str, int]) -> str | None:
        a, op, b = expr
        value = eval_op(int(str(abs(a))[::-1]), op, int(str(abs(b))[::-1]))
        return None if value is None else str(value)

    def digit_add(expr: tuple[int, str, int]) -> str | None:
        a, op, b = expr
        if op != "+":
            return None
        return digitwise_add_mod10(a, b)

    rules = {
        "direct_arithmetic": direct,
        "reverse_result_arithmetic": reverse_result,
        "reverse_operands_arithmetic": reverse_operands,
        "digitwise_add_mod10": digit_add,
    }
    candidates: list[tuple[str, str]] = []
    for name, func in rules.items():
        outputs = [func(expr) for expr, _ in parsed_examples if expr is not None]
        if None in outputs:
            continue
        if [normalize_answer(value) for value in outputs] == [rhs for _, rhs in parsed_examples]:
            prediction = func(parsed_query)
            if prediction is not None:
                candidates.append((name, prediction))
    unique_predictions = sorted(set(pred for _, pred in candidates))
    if len(unique_predictions) != 1:
        return {
            "probe_name": "numeric_operator_dsl_probe",
            "deployable": True,
            "status": "abstain",
            "prediction": "",
            "proof": f"candidate_rule_count={len(candidates)} unique_prediction_count={len(unique_predictions)}",
        }
    return {
        "probe_name": "numeric_operator_dsl_probe",
        "deployable": True,
        "status": "candidate",
        "prediction": unique_predictions[0],
        "proof": "rules=" + ",".join(name for name, _ in candidates),
    }


def choose_equation_probe(subtype: str, examples: list[tuple[str, str]], query: str, prompt: str) -> dict[str, Any]:
    if subtype.startswith("algebraic_equation"):
        return algebraic_equation_probe(prompt, query)
    if subtype == "numeric_operator_transform":
        return numeric_operator_probe(examples, query)
    symbolic = symbolic_char_map_probe(examples, query)
    if symbolic.get("status") == "candidate":
        return symbolic
    reverse = reverse_probe(examples, query)
    if reverse.get("status") == "candidate":
        return reverse
    return symbolic


def evaluate_equation_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = str(item.get("prompt", ""))
    examples, query, parse_status = split_examples_and_query(prompt)
    subtype, note = classify_equation(prompt, examples, query)
    expected = str(item.get("expected_answer", ""))
    baseline = str(item.get("baseline_prediction", ""))
    ps = str(item.get("prompt_sha256") or prompt_sha256(prompt))
    audit = {
        "schema_version": "kg1_v236_equation_subtype_audit_v1",
        "id": str(item.get("id", "")),
        "subtype": subtype,
        "example_count": len(examples),
        "query": query,
        "expected_answer": expected,
        "baseline_prediction": baseline,
        "prompt_sha256": ps,
        "notes": parse_status + ";" + note,
    }
    result = choose_equation_probe(subtype, examples, query, prompt)
    status = str(result.get("status", "abstain"))
    prediction = str(result.get("prediction", ""))
    if status == "candidate":
        status = "verified" if answers_equal(prediction, expected) else "incorrect"
    probe = {
        "schema_version": "kg1_v236_equation_solver_probe_result_v1",
        "id": audit["id"],
        "family": "equation_transform",
        "subtype": subtype,
        "probe_name": str(result.get("probe_name", "")),
        "deployable": bool(result.get("deployable", False)),
        "status": status,
        "prediction": prediction,
        "expected_answer": expected,
        "baseline_prediction": baseline,
        "prompt_sha256": ps,
        "proof": str(result.get("proof", "")),
    }
    return audit, probe


def evaluate_bit_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    lowered = prompt.lower()
    allowed = [
        name
        for name in ["shift", "rotation", "xor", "and", "or", "not", "majority", "choice"]
        if name in lowered
    ]
    has_binary = bool(re.search(r"\b[01]{8}\b", prompt))
    status = "guardrail_signature_verified" if has_binary and allowed else "guardrail_signature_incomplete"
    return {
        "schema_version": "kg1_v236_bit_guardrail_probe_result_v1",
        "id": str(item.get("id", "")),
        "family": "bit_manipulation",
        "probe_name": "bitvector_prompt_signature_guardrail",
        "status": status,
        "allowed_ops_seen": ";".join(allowed),
        "expected_answer": str(item.get("expected_answer", "")),
        "baseline_prediction": str(item.get("baseline_prediction", "")),
        "prompt_sha256": str(item.get("prompt_sha256") or prompt_sha256(prompt)),
        "proof": "no override emitted; this probe only verifies DSL search scope and regression guardrail input",
    }


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


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(str(row.get(key, "")) for key in keys)] += 1
    return [
        {**{key: values[idx] for idx, key in enumerate(keys)}, "rows": count}
        for values, count in sorted(counts.items())
    ]


def summarize_abstain_reasons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize(
        [row for row in rows if str(row.get("status", "")) == "abstain"],
        ["subtype", "probe_name", "status", "proof"],
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V236 LOCAL SOLVER DSL PROBES SCRIPT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("v232-analysis-manifest-json =", args.v232_analysis_manifest_json, flush=True)
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
        raise RuntimeError("V236 requires non-empty equation workitems")
    if not bit_items:
        raise RuntimeError("V236 requires non-empty bit guardrail workitems")

    equation_audit: list[dict[str, Any]] = []
    equation_probe: list[dict[str, Any]] = []
    for item in equation_items:
        audit, probe = evaluate_equation_item(item)
        equation_audit.append(audit)
        equation_probe.append(probe)
    bit_probe = [evaluate_bit_item(item) for item in bit_items]

    verified_overrides = [row for row in equation_probe if row["deployable"] and row["status"] == "verified"]
    incorrect_overrides = [row for row in equation_probe if row["deployable"] and row["status"] == "incorrect"]
    bit_guardrail_ready = all(row["status"] == "guardrail_signature_verified" for row in bit_probe)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label
    out_paths = {
        "equation_subtype_audit_csv": args.output_dir / f"{prefix}_equation_subtype_audit.csv",
        "equation_solver_probe_results_csv": args.output_dir / f"{prefix}_equation_solver_probe_results.csv",
        "bit_guardrail_probe_results_csv": args.output_dir / f"{prefix}_bit_guardrail_probe_results.csv",
        "equation_probe_summary_csv": args.output_dir / f"{prefix}_equation_probe_summary.csv",
        "equation_abstain_reason_summary_csv": args.output_dir / f"{prefix}_equation_abstain_reason_summary.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_csv(out_paths["equation_subtype_audit_csv"], equation_audit, EQUATION_AUDIT_COLUMNS)
    write_csv(out_paths["equation_solver_probe_results_csv"], equation_probe, PROBE_COLUMNS)
    write_csv(out_paths["bit_guardrail_probe_results_csv"], bit_probe, BIT_COLUMNS)
    write_csv(
        out_paths["equation_probe_summary_csv"],
        summarize(equation_probe, ["subtype", "probe_name", "status"]),
        ["subtype", "probe_name", "status", "rows"],
    )
    write_csv(
        out_paths["equation_abstain_reason_summary_csv"],
        summarize_abstain_reasons(equation_probe),
        ABSTAIN_REASON_COLUMNS,
    )

    verified_count = len(verified_overrides)
    if verified_count >= args.equation_target_gain and not incorrect_overrides and bit_guardrail_ready:
        decision = {
            "decision": "prepare_gated_solver_rescue_eval",
            "reason": f"verified_equation_overrides={verified_count} >= target_gain={args.equation_target_gain}; bit_guardrail_ready=true",
            "next_action": "Create a separate rescue-eval notebook that applies only verified overrides and measures weak gate.",
        }
    else:
        decision = {
            "decision": "continue_local_solver_development",
            "reason": (
                f"verified_equation_overrides={verified_count}; "
                f"incorrect_overrides={len(incorrect_overrides)}; "
                f"target_gain={args.equation_target_gain}; bit_guardrail_ready={bit_guardrail_ready}"
            ),
            "next_action": "Inspect equation_subtype_audit and equation_abstain_reason_summary; extend only routes with exact parsers before any eval.",
        }

    manifest = {
        "schema_version": "kg1_v236_local_solver_dsl_probes_manifest_v1",
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
        "probe_counts": {
            "equation_workitems": len(equation_items),
            "bit_guardrail_workitems": len(bit_items),
            "deployable_verified_equation_overrides": verified_count,
            "deployable_incorrect_equation_overrides": len(incorrect_overrides),
            "bit_guardrail_signature_verified_rows": sum(1 for row in bit_probe if row["status"] == "guardrail_signature_verified"),
            "target_gain": int(args.equation_target_gain),
        },
        "equation_subtype_summary": summarize(equation_audit, ["subtype"]),
        "equation_probe_summary": summarize(equation_probe, ["subtype", "probe_name", "status"]),
        "equation_abstain_reason_summary": summarize_abstain_reasons(equation_probe),
        "bit_probe_summary": summarize(bit_probe, ["status"]),
        "outputs": {name: str(path) for name, path in out_paths.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in out_paths.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(out_paths["manifest_json"], manifest)
    print("probe_counts =", json.dumps(manifest["probe_counts"], sort_keys=True), flush=True)
    print("equation_subtype_summary =", json.dumps(manifest["equation_subtype_summary"], indent=2, sort_keys=True), flush=True)
    print("equation_probe_summary =", json.dumps(manifest["equation_probe_summary"], indent=2, sort_keys=True), flush=True)
    print("equation_abstain_reason_summary =", json.dumps(manifest["equation_abstain_reason_summary"], indent=2, sort_keys=True), flush=True)
    print("bit_probe_summary =", json.dumps(manifest["bit_probe_summary"], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in out_paths.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V236 LOCAL SOLVER DSL PROBES SCRIPT END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v232-analysis-manifest-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v236_local_solver_dsl_probes")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--equation-target-gain", type=int, default=5)
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
                "id": "eq_symbolic",
                "family": "equation_transform",
                "expected_answer": "dc",
                "baseline_prediction": "xx",
                "prompt": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: `ab = cd` `aa = cc` Now, determine the result for: `ba`",
            },
            {
                "id": "eq_numeric",
                "family": "equation_transform",
                "expected_answer": "7",
                "baseline_prediction": "6",
                "prompt": "Below are a few examples: `2 + 3 = 5` `4 + 1 = 5` Now, determine the result for: `3 + 4`",
            },
            {
                "id": "eq_algebraic",
                "family": "equation_transform",
                "expected_answer": "2",
                "baseline_prediction": "3",
                "prompt": "Solve the single equation and return only the value. Now, determine the result for: `2*x + 3 = 7`",
            },
        ]
        bit_rows = [
            {
                "id": "bit1",
                "family": "bit_manipulation",
                "expected_answer": "01010101",
                "baseline_prediction": "01010101",
                "prompt": "A secret bit manipulation rule transforms 8-bit binary numbers using shift, rotation, XOR, AND, OR, NOT, majority, and choice. 00000000 -> 11111111 Now solve 01010101.",
            }
        ]
        with equation_items.open("w", encoding="utf-8") as handle:
            for row in equation_rows:
                handle.write(json.dumps(row) + "\n")
        with bit_items.open("w", encoding="utf-8") as handle:
            for row in bit_rows:
                handle.write(json.dumps(row) + "\n")
        write_csv(acceptance, [{"area": "equation_transform", "criterion": "minimum_verified_gain"}], ["area", "criterion"])
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
            label="v236_local_solver_dsl_probes",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            equation_target_gain=3 if import_sympy() is not None else 2,
        )
        manifest = run_analysis(args)
        expected_verified = 3 if import_sympy() is not None else 2
        if manifest["probe_counts"]["deployable_verified_equation_overrides"] != expected_verified:
            raise AssertionError(f"expected {expected_verified} verified equation overrides in self-test")
        if manifest["probe_counts"]["bit_guardrail_signature_verified_rows"] != 1:
            raise AssertionError("expected bit guardrail signature verification")
    print("v236_local_solver_dsl_probes_self_test=ok", flush=True)
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

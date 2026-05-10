#!/usr/bin/env python3
"""Run V246 exhaustive abstain rule audit from HF runtime artifacts.

This is CPU-only and diagnostic-only. It consumes the V236 parserfix probe
results plus the V240 bridge equation workitems from the HF dataset repo. It
does not train, run model generation, run full scoring, package artifacts, or
submit to Kaggle.

Weak labels are used only as an audit brake. A rule class is considered
promotable only when every candidate emitted by that class is verified and
there are zero incorrect candidates for that class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from analyze_v238_alice_parser_probes import answers_equal, parse_alice_prompt, parse_numeric_token
from analyze_v241_abstain_rule_candidate_audit import infer_symbolic_transducer, numeric_rule_functions


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_HF_DATASET_REPO = "felipesp1983/kg1-nemotron-training"
DEFAULT_V236_PREFIX = "runtime_artifacts/v236_local_solver_dsl_probes/v236-hf-cpu-bridge-parserfix-20260510T202819Z"
DEFAULT_V240_PREFIX = "runtime_artifacts/v240_hf_bridge/local_drive_mcp_20260510T172421Z"
DEFAULT_OUTPUT_PREFIX = "runtime_artifacts/v246_exhaustive_abstain_audit"

AUDIT_COLUMNS = [
    "schema_version",
    "id",
    "family",
    "subtype",
    "rule_class",
    "status",
    "prediction",
    "expected_answer",
    "baseline_prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "promotable_after_class_gate",
    "query",
    "example_count",
    "same_operator_example_count",
    "proof",
]
SUMMARY_COLUMNS = [
    "rule_class",
    "rows",
    "candidate_rows",
    "verified_candidates",
    "incorrect_candidates",
    "abstain_rows",
    "promotable_after_class_gate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download_file(repo_id: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V246 HF artifact audit") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def upload_outputs(repo_id: str, output_dir: Path, path_in_repo: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload V246 outputs") from exc
    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload KG1 V246 exhaustive abstain audit {path_in_repo.strip('/')}",
    )
    return str(info)


def load_hf_inputs(args: argparse.Namespace, download_root: Path) -> dict[str, Path]:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    files = {
        "v236_results_csv": f"{args.v236_prefix}/v236_local_solver_dsl_probes_from_hf_bridge_equation_solver_probe_results.csv",
        "v236_manifest_json": f"{args.v236_prefix}/v236_local_solver_dsl_probes_from_hf_bridge_manifest.json",
        "v232_equation_workitems_jsonl": f"{args.v240_prefix}/v232_equation_workitems.jsonl",
        "v232_manifest_json": f"{args.v240_prefix}/v232_manifest.json",
    }
    downloaded: dict[str, Path] = {}
    for label, filename in files.items():
        path = download_file(args.hf_dataset_repo, filename, download_root, token)
        downloaded[label] = path
        print(f"downloaded_{label} =", path, flush=True)
    return downloaded


def observed_contract(v236_manifest: dict[str, Any], v232_manifest: dict[str, Any]) -> str:
    for manifest in (v236_manifest, v232_manifest):
        inputs = manifest.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        value = (
            inputs.get("observed_shared_row_contract_sha256")
            or manifest.get("observed_shared_row_contract_sha256")
            or inputs.get("expected_shared_row_contract_sha256")
            or manifest.get("expected_shared_row_contract_sha256")
        )
        if value:
            return str(value)
    return ""


def classify_status(predictions: list[str]) -> tuple[str, str, str]:
    unique = sorted(set(predictions))
    if not unique:
        return "abstain", "", "no_unique_prediction"
    if len(unique) != 1:
        return "abstain", "", f"ambiguous_prediction_count={len(unique)}"
    return "candidate", unique[0], "unique_prediction"


def positional_subsequence_positions(lhs: str, rhs: str) -> set[tuple[int, ...]]:
    found: set[tuple[int, ...]] = set()

    def walk(start: int, rhs_index: int, positions: list[int]) -> None:
        if rhs_index == len(rhs):
            found.add(tuple(positions))
            return
        for index in range(start, len(lhs)):
            if lhs[index] == rhs[rhs_index]:
                walk(index + 1, rhs_index + 1, [*positions, index])

    walk(0, 0, [])
    return found


def positional_deletion_candidate(examples: list[tuple[str, str]], query: str) -> tuple[str, str, str]:
    possible: set[tuple[int, ...]] | None = None
    for lhs, rhs in examples:
        positions = positional_subsequence_positions(lhs, rhs)
        possible = positions if possible is None else possible & positions
    if not possible:
        return "abstain", "", "no_common_keep_positions"
    predictions: list[str] = []
    for positions in possible:
        if positions and max(positions) >= len(query):
            continue
        predictions.append("".join(query[index] for index in positions))
    status, prediction, proof = classify_status(predictions)
    return status, prediction, f"{proof}; keep_position_count={len(possible)}"


def position_specific_char_map_candidate(examples: list[tuple[str, str]], query: str) -> tuple[str, str, str]:
    lengths = sorted(set(len(rhs) for _, rhs in examples))
    if len(lengths) != 1:
        return "abstain", "", "nonuniform_rhs_lengths"
    output_len = lengths[0]
    choices: list[list[str]] = []
    proof_parts: list[str] = []
    for out_index in range(output_len):
        possible_values: set[str] = set()
        for input_index in range(5):
            mapping: dict[str, str] = {}
            ok = True
            for lhs, rhs in examples:
                if len(lhs) != 5 or len(rhs) != output_len or len(query) != 5:
                    ok = False
                    break
                source_char = lhs[input_index]
                target_char = rhs[out_index]
                if source_char in mapping and mapping[source_char] != target_char:
                    ok = False
                    break
                mapping[source_char] = target_char
            if ok and query[input_index] in mapping:
                possible_values.add(mapping[query[input_index]])
        if not possible_values:
            return "abstain", "", f"no_position_map_for_output_index={out_index}"
        choices.append(sorted(possible_values))
        proof_parts.append(f"out{out_index}_choices={len(possible_values)}")
    predictions = ["".join(parts) for parts in itertools.product(*choices)]
    status, prediction, proof = classify_status(predictions)
    return status, prediction, proof + "; " + ",".join(proof_parts)


def same_operator_examples(examples: list[tuple[str, str]], query: str) -> list[tuple[str, str]]:
    if len(query) != 5:
        return []
    query_operator = query[2]
    return [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[2] == query_operator]


def symbolic_audits(row: dict[str, str], item: dict[str, Any]) -> list[dict[str, Any]]:
    examples, query, parse_status = parse_alice_prompt(str(item.get("prompt", "")))
    if parse_status != "ok":
        return [
            build_audit_row(row, "symbolic_parse_gate", "abstain", "", query, examples, [], f"parse_status={parse_status}")
        ]
    audits: list[dict[str, Any]] = []
    all_examples = examples
    same_examples = same_operator_examples(examples, query)
    candidate_sets = [
        ("symbolic_all_examples_char_transducer", all_examples),
        ("symbolic_same_operator_char_transducer_min2", same_examples if len(same_examples) >= 2 else []),
        ("symbolic_all_examples_positional_deletion", all_examples),
        ("symbolic_same_operator_positional_deletion_min2", same_examples if len(same_examples) >= 2 else []),
        ("symbolic_same_operator_position_char_map_min2", same_examples if len(same_examples) >= 2 else []),
    ]
    for rule_class, selected in candidate_sets:
        if not selected:
            audits.append(build_audit_row(row, rule_class, "abstain", "", query, examples, same_examples, "insufficient_examples"))
            continue
        if "char_transducer" in rule_class:
            result = infer_symbolic_transducer(selected, query, pair_cap=3000, global_cap=12000)
            status = str(result.get("status", "abstain"))
            prediction = str(result.get("prediction", ""))
            proof = str(result.get("proof", ""))
        elif "positional_deletion" in rule_class:
            status, prediction, proof = positional_deletion_candidate(selected, query)
        else:
            status, prediction, proof = position_specific_char_map_candidate(selected, query)
        audits.append(build_audit_row(row, rule_class, status, prediction, query, examples, same_examples, proof))
    return audits


def numeric_rules_extended() -> dict[str, Callable[[int, int], str]]:
    rules = dict(numeric_rule_functions())
    rules.update(
        {
            "normal_mod": lambda a, b: str(a % b) if b else "",
            "normal_div_floor": lambda a, b: str(a // b) if b else "",
            "normal_div_round": lambda a, b: str(round(a / b)) if b else "",
            "normal_div_float2": lambda a, b: f"{a / b:.2f}" if b else "",
            "a_squared_plus_b": lambda a, b: str(a * a + b),
            "b_squared_plus_a": lambda a, b: str(b * b + a),
            "a_times_b_plus_a": lambda a, b: str(a * b + a),
            "a_times_b_plus_b": lambda a, b: str(a * b + b),
            "a_times_b_minus_a": lambda a, b: str(a * b - a),
            "a_times_b_minus_b": lambda a, b: str(a * b - b),
            "hundreds_a_plus_b": lambda a, b: str(a * 100 + b),
            "hundreds_b_plus_a": lambda a, b: str(b * 100 + a),
        }
    )
    return rules


def numeric_candidate(examples: list[tuple[str, str]], query: str, min_examples: int) -> tuple[str, str, str, int, int]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return "abstain", "", "query_not_numeric_binary_operator", 0, 0
    left, operator, right = parsed_query
    parsed_examples: list[tuple[int, int, str]] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if parsed and parsed[1] == operator:
            parsed_examples.append((parsed[0], parsed[2], rhs))
    if len(parsed_examples) < min_examples:
        return "abstain", "", f"same_operator_examples={len(parsed_examples)} below_min={min_examples}", len(parsed_examples), 0
    candidates: list[tuple[str, str]] = []
    for name, func in numeric_rules_extended().items():
        outputs: list[str] = []
        ok = True
        for ex_left, ex_right, ex_answer in parsed_examples:
            try:
                value = func(ex_left, ex_right)
            except Exception:
                ok = False
                break
            if value == "" or value != str(ex_answer):
                ok = False
                break
            outputs.append(value)
        if ok:
            try:
                candidates.append((name, func(left, right)))
            except Exception:
                pass
    predictions = [prediction for _, prediction in candidates if prediction != ""]
    status, prediction, proof = classify_status(predictions)
    proof = f"{proof}; same_operator_examples={len(parsed_examples)}; candidate_rules={','.join(name for name, _ in candidates)}"
    return status, prediction, proof, len(parsed_examples), len(candidates)


def numeric_audits(row: dict[str, str], item: dict[str, Any]) -> list[dict[str, Any]]:
    examples, query, parse_status = parse_alice_prompt(str(item.get("prompt", "")))
    if parse_status != "ok":
        return [
            build_audit_row(row, "numeric_parse_gate", "abstain", "", query, examples, [], f"parse_status={parse_status}")
        ]
    audits: list[dict[str, Any]] = []
    for min_examples in (2, 3, 4):
        status, prediction, proof, same_count, _candidate_count = numeric_candidate(examples, query, min_examples)
        same = [("", "")] * same_count
        audits.append(
            build_audit_row(
                row,
                f"numeric_same_operator_extended_dsl_min{min_examples}",
                status,
                prediction,
                query,
                examples,
                same,
                proof,
            )
        )
    return audits


def build_audit_row(
    row: dict[str, str],
    rule_class: str,
    status: str,
    prediction: str,
    query: str,
    examples: list[tuple[str, str]],
    same_examples: list[tuple[str, str]],
    proof: str,
) -> dict[str, Any]:
    candidate = status == "candidate"
    verified = candidate and answers_equal(prediction, row.get("expected_answer", ""))
    incorrect = candidate and not verified
    return {
        "schema_version": "kg1_v246_exhaustive_abstain_audit_row_v1",
        "id": row.get("id", ""),
        "family": row.get("family", "equation_transform"),
        "subtype": row.get("subtype", ""),
        "rule_class": rule_class,
        "status": status,
        "prediction": prediction,
        "expected_answer": row.get("expected_answer", ""),
        "baseline_prediction": row.get("baseline_prediction", ""),
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "promotable_after_class_gate": False,
        "query": query,
        "example_count": len(examples),
        "same_operator_example_count": len(same_examples),
        "proof": proof,
    }


def apply_class_gate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row.get("rule_class", ""))].append(row)
    for rule_rows in by_class.values():
        candidates = [row for row in rule_rows if row.get("status") == "candidate"]
        incorrect = [row for row in candidates if truthy(row.get("incorrect_by_weak_label"))]
        promotable = bool(candidates) and not incorrect
        for row in rule_rows:
            row["promotable_after_class_gate"] = bool(promotable and row.get("status") == "candidate")
    return rows


def summarize_rule_classes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row.get("rule_class", ""))].append(row)
    summary: list[dict[str, Any]] = []
    for rule_class, rule_rows in sorted(by_class.items()):
        candidates = [row for row in rule_rows if row.get("status") == "candidate"]
        verified = [row for row in candidates if truthy(row.get("verified_by_weak_label"))]
        incorrect = [row for row in candidates if truthy(row.get("incorrect_by_weak_label"))]
        abstains = [row for row in rule_rows if row.get("status") != "candidate"]
        summary.append(
            {
                "rule_class": rule_class,
                "rows": len(rule_rows),
                "candidate_rows": len(candidates),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "abstain_rows": len(abstains),
                "promotable_after_class_gate": bool(candidates and not incorrect),
            }
        )
    return summary


def run_analysis(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    print("=== V246 EXHAUSTIVE ABSTAIN AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    print("expected_shared_row_contract_sha256 =", args.expected_shared_row_contract_sha256, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v236_manifest = read_json(paths["v236_manifest_json"])
    v232_manifest = read_json(paths["v232_manifest_json"])
    contract = observed_contract(v236_manifest, v232_manifest)
    print("observed_shared_row_contract_sha256 =", contract, flush=True)
    if args.expected_shared_row_contract_sha256 and contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "shared row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + contract
        )

    v236_rows = read_csv(paths["v236_results_csv"])
    workitems = read_jsonl(paths["v232_equation_workitems_jsonl"])
    item_by_id = {str(item.get("id", "")): item for item in workitems}
    missing_items = sorted(row.get("id", "") for row in v236_rows if row.get("id", "") not in item_by_id)
    if missing_items:
        raise RuntimeError("V246 V236 rows reference missing workitems: " + json.dumps(missing_items[:20]))
    abstains = [row for row in v236_rows if row.get("status") == "abstain"]
    print("v236_rows =", len(v236_rows), flush=True)
    print("v236_abstain_rows =", len(abstains), flush=True)

    audit_rows: list[dict[str, Any]] = []
    for row in abstains:
        item = item_by_id[str(row.get("id", ""))]
        if row.get("subtype") == "symbolic_mixed_token_rewrite":
            audit_rows.extend(symbolic_audits(row, item))
        elif row.get("subtype") == "numeric_operator_transform":
            audit_rows.extend(numeric_audits(row, item))
        else:
            audit_rows.append(
                build_audit_row(row, "unsupported_subtype_gate", "abstain", "", "", [], [], "unsupported_subtype")
            )
    audit_rows = apply_class_gate(audit_rows)
    summary_rows = summarize_rule_classes(audit_rows)
    promotable_rows = [row for row in audit_rows if truthy(row.get("promotable_after_class_gate"))]
    incorrect_rows = [row for row in audit_rows if truthy(row.get("incorrect_by_weak_label"))]
    verified_rows = [row for row in audit_rows if truthy(row.get("verified_by_weak_label"))]

    prefix = args.label
    outputs = {
        "audit_csv": args.output_dir / f"{prefix}_audit.csv",
        "rule_class_summary_csv": args.output_dir / f"{prefix}_rule_class_summary.csv",
        "manifest_json": args.output_dir / f"{prefix}_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    write_csv(outputs["rule_class_summary_csv"], summary_rows, SUMMARY_COLUMNS)

    if promotable_rows:
        decision = {
            "decision": "prepare_strict_solver_rescue_probe",
            "reason": f"promotable_rows={len(promotable_rows)}; incorrect_rows={len(incorrect_rows)}",
            "next_action": "Promote only class-gated rows into a new solver rescue probe and remeasure weak.",
        }
    else:
        decision = {
            "decision": "no_safe_local_rule_promotion_found",
            "reason": f"verified_candidates={len(verified_rows)}; incorrect_candidates={len(incorrect_rows)}; promotable_rows=0",
            "next_action": "Do not spend GPU on this local-rule path; obtain gated external traces or design a new training set from non-leaking sources.",
        }

    manifest = {
        "schema_version": "kg1_v246_exhaustive_abstain_audit_manifest_v1",
        "generated_at_utc": utc_now(),
        "label": args.label,
        "inputs": {
            "hf_dataset_repo": args.hf_dataset_repo,
            "v236_prefix": args.v236_prefix,
            "v240_prefix": args.v240_prefix,
            "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
            "observed_shared_row_contract_sha256": contract,
        },
        "input_artifact_hashes": {name: file_meta(path) for name, path in paths.items()},
        "counts": {
            "v236_rows": len(v236_rows),
            "v236_abstain_rows": len(abstains),
            "audit_rows": len(audit_rows),
            "verified_candidates": len(verified_rows),
            "incorrect_candidates": len(incorrect_rows),
            "promotable_rows_after_class_gate": len(promotable_rows),
        },
        "v236_status_summary": [
            {"status": status, "rows": count} for status, count in sorted(Counter(row.get("status", "") for row in v236_rows).items())
        ],
        "v236_subtype_summary": [
            {"subtype": subtype, "rows": count}
            for subtype, count in sorted(Counter(row.get("subtype", "") for row in v236_rows).items())
        ],
        "rule_class_summary": summary_rows,
        "verified_preview": verified_rows[:20],
        "incorrect_preview": incorrect_rows[:20],
        "promotable_preview": promotable_rows[:20],
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_artifact_hashes": {
            name: file_meta(path) for name, path in outputs.items() if name != "manifest_json"
        },
        "decision": decision,
        "blocked_actions": ["train", "model_generation", "full_scoring", "package", "kaggle_submit"],
    }
    write_json(outputs["manifest_json"], manifest)
    print("counts =", json.dumps(manifest["counts"], sort_keys=True), flush=True)
    print("rule_class_summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print("outputs =", json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True), flush=True)
    print("=== V246 EXHAUSTIVE ABSTAIN AUDIT END ===", flush=True)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V246 HF RUNNER START ===", flush=True)
    print("hf_dataset_repo =", args.hf_dataset_repo, flush=True)
    print("v236_prefix =", args.v236_prefix, flush=True)
    print("v240_prefix =", args.v240_prefix, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("upload_to_hf =", args.upload_to_hf, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        download_root = Path(tmp) / "hf_download"
        download_root.mkdir(parents=True, exist_ok=True)
        paths = load_hf_inputs(args, download_root)
        manifest = run_analysis(args, paths)
    upload_info = "upload_disabled"
    if args.upload_to_hf:
        if not args.output_path_in_repo:
            raise RuntimeError("--output-path-in-repo is required when --upload-to-hf is used")
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --upload-to-hf is used")
        upload_info = upload_outputs(args.hf_dataset_repo, args.output_dir, args.output_path_in_repo, token)
    manifest["hf_upload"] = {
        "enabled": bool(args.upload_to_hf),
        "repo_id": args.hf_dataset_repo,
        "path_in_repo": str(args.output_path_in_repo or ""),
        "upload_info": upload_info,
    }
    manifest_path = Path(manifest["outputs"]["manifest_json"])
    write_json(manifest_path, manifest)
    print("hf_upload =", json.dumps(manifest["hf_upload"], sort_keys=True), flush=True)
    print("=== V246 HF RUNNER END ===", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset-repo", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--v236-prefix", default=DEFAULT_V236_PREFIX)
    parser.add_argument("--v240-prefix", default=DEFAULT_V240_PREFIX)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="v246_exhaustive_abstain_audit")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--output-path-in-repo", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v236_results = root / "v236_results.csv"
        v236_manifest = root / "v236_manifest.json"
        v232_manifest = root / "v232_manifest.json"
        workitems = root / "workitems.jsonl"
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples: 03#04 = 7 02#08 = 10 Now, determine the result for: 05#06"
        )
        with v236_results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
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
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "schema_version": "self_test",
                    "id": "num1",
                    "family": "equation_transform",
                    "subtype": "numeric_operator_transform",
                    "probe_name": "numeric_operator_dsl_probe",
                    "deployable": "True",
                    "status": "abstain",
                    "prediction": "",
                    "expected_answer": "11",
                    "baseline_prediction": "0",
                    "prompt_sha256": "self",
                    "proof": "self_test",
                }
            )
        workitems.write_text(json.dumps({"id": "num1", "prompt": prompt}) + "\n", encoding="utf-8")
        write_json(v236_manifest, {"inputs": {"observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256}})
        write_json(v232_manifest, {"inputs": {"observed_shared_row_contract_sha256": EXPECTED_ROW_CONTRACT_SHA256}})
        args = argparse.Namespace(
            output_dir=root / "out",
            label="v246_exhaustive_abstain_audit",
            expected_shared_row_contract_sha256=EXPECTED_ROW_CONTRACT_SHA256,
            hf_dataset_repo=DEFAULT_HF_DATASET_REPO,
            v236_prefix=DEFAULT_V236_PREFIX,
            v240_prefix=DEFAULT_V240_PREFIX,
        )
        manifest = run_analysis(
            args,
            {
                "v236_results_csv": v236_results,
                "v236_manifest_json": v236_manifest,
                "v232_equation_workitems_jsonl": workitems,
                "v232_manifest_json": v232_manifest,
            },
        )
        if manifest["counts"]["promotable_rows_after_class_gate"] != 1:
            raise AssertionError("expected one promotable row for the numeric min2 gate in self-test")
        if manifest["counts"]["incorrect_candidates"] != 0:
            raise AssertionError("expected zero incorrect self-test candidates")
    print("v246_exhaustive_abstain_audit_self_test=ok", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

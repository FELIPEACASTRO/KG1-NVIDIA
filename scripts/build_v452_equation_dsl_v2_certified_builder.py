#!/usr/bin/env python3
"""V452 CPU-certified equation DSL v2 builder.

V452 audits the public-train hard negatives produced by V439. It is deliberately
CPU-only: it does not train, run inference, package, submit, or launch HF jobs.

The gate objective is narrow:

* generate equation preference pairs only from rules frozen before answer audit;
* reject any rule class that produces an incorrect candidate on the same audit;
* report whether the result is strong enough to justify a later paid GPU job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"
for item in (ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from competition_utils import verify_answer  # noqa: E402
from kg1_v274_numeric_postprocessor import (  # noqa: E402
    choose_guarded_numeric_override,
    normalize_payload,
    parse_alice_prompt,
    parse_numeric_token,
)
from run_v278_symbolic_pbe_dsl_audit_hf import (  # noqa: E402
    classify_subtype,
    symbolic_candidates,
)
from run_v324_equation_expanded_solver_gate import symbolic_variable_operator_candidates  # noqa: E402


DEFAULT_V439_MANIFEST = (
    ROOT
    / "artifacts/v439_final_answer_only_pairs/20260515T_v439_final_answer_only/"
    / "v439_final_answer_only_pairs_manifest.json"
)


AUDIT_COLUMNS = [
    "id",
    "source_id",
    "split",
    "family",
    "subcategory",
    "parse_status",
    "subtype",
    "query",
    "answer",
    "adapter_prediction",
    "candidate_source",
    "rule_class",
    "status",
    "prediction",
    "candidate_correct",
    "candidate_conflict",
    "class_safe",
    "row_promoted",
    "rejection_reason",
    "candidate_program_count",
    "unique_prediction_count",
    "loo_checked",
    "loo_passed",
    "proof",
]


@dataclass(frozen=True)
class Candidate:
    source: str
    rule_class: str
    prediction: str
    proof: str
    program_count: int
    unique_prediction_count: int
    loo_checked: int = 0
    loo_passed: int = 0


class SymbolicArgs:
    pair_mapping_cap = 2000
    global_mapping_cap = 50000
    max_char_subset_size = 4
    max_position_sources = 8


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def to_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def boxed(answer: str) -> str:
    return f"Final answer: \\boxed{{{answer}}}"


def reverse_text(value: str) -> str:
    text = str(value)
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


def digits2(value: int) -> tuple[int, int]:
    text = f"{abs(int(value)):02d}"[-2:]
    return int(text[0]), int(text[1])


def reverse_int(value: int) -> int:
    return int(f"{abs(int(value)):02d}"[::-1])


def numeric_v2_functions() -> dict[str, Callable[[int, int], str | None]]:
    return {
        "add": lambda a, b: str(a + b),
        "sub_ab": lambda a, b: str(a - b),
        "sub_ba": lambda a, b: str(b - a),
        "abs_diff": lambda a, b: str(abs(a - b)),
        "neg_abs_diff": lambda a, b: str(-abs(a - b)),
        "mul": lambda a, b: str(a * b),
        "mul_plus1": lambda a, b: str(a * b + 1),
        "mul_minus1": lambda a, b: str(a * b - 1),
        "add_plus1": lambda a, b: str(a + b + 1),
        "add_minus1": lambda a, b: str(a + b - 1),
        "sub_ab_plus1": lambda a, b: str(a - b + 1),
        "sub_ab_minus1": lambda a, b: str(a - b - 1),
        "sub_ba_plus1": lambda a, b: str(b - a + 1),
        "sub_ba_minus1": lambda a, b: str(b - a - 1),
        "concat_ab": lambda a, b: f"{abs(a)}{abs(b)}",
        "concat_ba": lambda a, b: f"{abs(b)}{abs(a)}",
        "max_mod_min": lambda a, b: str(max(abs(a), abs(b)) % max(1, min(abs(a), abs(b)))),
        "div_ab": lambda a, b: str(a // b) if b else None,
        "div_ba": lambda a, b: str(b // a) if a else None,
        "mod_ab": lambda a, b: str(a % b) if b else None,
        "mod_ba": lambda a, b: str(b % a) if a else None,
        "digit_absdiff_concat": lambda a, b: "".join(str(abs(x - y)) for x, y in zip(digits2(a), digits2(b))),
        "digit_add_mod10_concat": lambda a, b: "".join(str((x + y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ab_mod10_concat": lambda a, b: "".join(str((x - y) % 10) for x, y in zip(digits2(a), digits2(b))),
        "digit_sub_ba_mod10_concat": lambda a, b: "".join(str((y - x) % 10) for x, y in zip(digits2(a), digits2(b))),
        "cross_mul": lambda a, b: str(digits2(a)[0] * digits2(b)[0] + digits2(a)[1] * digits2(b)[1]),
        "cross_mul_rev": lambda a, b: str(digits2(a)[0] * digits2(b)[1] + digits2(a)[1] * digits2(b)[0]),
        "digit_mul_concat": lambda a, b: str(digits2(a)[0] * digits2(b)[0]) + str(digits2(a)[1] * digits2(b)[1]),
        "digit_mul_rev_concat": lambda a, b: str(digits2(a)[0] * digits2(b)[1]) + str(digits2(a)[1] * digits2(b)[0]),
        "digit_sum_diff": lambda a, b: str(sum(digits2(a)) - sum(digits2(b))),
        "digit_sum_sum": lambda a, b: str(sum(digits2(a)) + sum(digits2(b))),
        "digit_product_diff": lambda a, b: str(digits2(a)[0] * digits2(a)[1] - digits2(b)[0] * digits2(b)[1]),
        "digit_product_sum": lambda a, b: str(digits2(a)[0] * digits2(a)[1] + digits2(b)[0] * digits2(b)[1]),
        "determinant": lambda a, b: str(digits2(a)[0] * digits2(b)[1] - digits2(a)[1] * digits2(b)[0]),
        "abs_determinant": lambda a, b: str(abs(digits2(a)[0] * digits2(b)[1] - digits2(a)[1] * digits2(b)[0])),
        "tens_add_ones_add": lambda a, b: str((digits2(a)[0] + digits2(b)[0]) * 10 + digits2(a)[1] + digits2(b)[1]),
    }


NUMERIC_V2_FUNCTIONS = numeric_v2_functions()


def apply_numeric_label(label: tuple[str, bool, bool], left: str, right: str) -> str | None:
    name, reverse_operands, reverse_result = label
    a, b = int(left), int(right)
    if reverse_operands:
        a, b = reverse_int(a), reverse_int(b)
    value = NUMERIC_V2_FUNCTIONS[name](a, b)
    if value is None:
        return None
    text = str(value)
    return reverse_text(text) if reverse_result else text


def infer_numeric_v2_candidates(
    examples: list[tuple[str, str]],
    query: str,
    scope: str,
    min_rows: int,
) -> list[Candidate]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return []
    query_left, query_op, query_right = parsed_query
    scoped: list[tuple[str, str, str, str]] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if not parsed:
            return []
        left, op, right = parsed
        if scope == "global" or op == query_op:
            scoped.append((left, op, right, str(rhs)))
    if len(scoped) < min_rows:
        return []

    labels: list[tuple[tuple[str, bool, bool], str]] = []
    for label in itertools.product(sorted(NUMERIC_V2_FUNCTIONS), (False, True), (False, True)):
        ok = True
        for left, _op, right, expected in scoped:
            try:
                prediction = apply_numeric_label(label, left, right)
            except Exception:
                ok = False
                break
            if prediction != expected:
                ok = False
                break
        if not ok:
            continue
        prediction = apply_numeric_label(label, query_left, query_right)
        if prediction:
            labels.append((label, prediction))

    predictions = sorted({prediction for _label, prediction in labels})
    if len(predictions) != 1:
        return []

    stable_labels: list[tuple[tuple[str, bool, bool], str, int]] = []
    for label, prediction in labels:
        checked = 0
        passed = 0
        for index, (left, _op, right, expected) in enumerate(scoped):
            train = scoped[:index] + scoped[index + 1 :]
            if not train:
                continue
            if all(apply_numeric_label(label, item_left, item_right) == item_expected for item_left, _item_op, item_right, item_expected in train):
                checked += 1
                if apply_numeric_label(label, left, right) == expected:
                    passed += 1
        if checked >= min(2, len(scoped)) and checked == passed:
            stable_labels.append((label, prediction, checked))

    stable_predictions = sorted({prediction for _label, prediction, _checked in stable_labels})
    if len(stable_predictions) != 1:
        return []

    best_label, prediction, checked = sorted(stable_labels, key=lambda item: (len(item[0][0]), item[0]))[0]
    name, reverse_operands, reverse_result = best_label
    return [
        Candidate(
            source="numeric_v2_loo",
            rule_class=f"numeric_v2_{scope}_{name}_revop{int(reverse_operands)}_revres{int(reverse_result)}",
            prediction=prediction,
            proof=f"scope={scope}; rows={len(scoped)}; label={best_label}; loo_checked={checked}",
            program_count=len(labels),
            unique_prediction_count=1,
            loo_checked=checked,
            loo_passed=checked,
        )
    ]


def guarded_adapter_relation_candidate(
    examples: list[tuple[str, str]],
    query: str,
    adapter_prediction: str,
) -> list[Candidate]:
    replacement, rule, proof = choose_guarded_numeric_override(examples, query, adapter_prediction)
    if not replacement:
        return []
    return [
        Candidate(
            source="v274_guarded_adapter_relation",
            rule_class=f"v274_guarded_numeric_{rule}",
            prediction=str(replacement),
            proof=proof,
            program_count=1,
            unique_prediction_count=1,
        )
    ]


def existing_symbolic_candidates(examples: list[tuple[str, str]], query: str) -> list[Candidate]:
    out: list[Candidate] = []
    for result in symbolic_candidates(examples, query, SymbolicArgs):
        if result.get("status") == "candidate" and result.get("prediction"):
            out.append(
                Candidate(
                    source="v278_symbolic_pbe",
                    rule_class=str(result.get("rule_class", "")),
                    prediction=str(result.get("prediction", "")),
                    proof=str(result.get("proof", "")),
                    program_count=int(result.get("candidate_program_count", 0) or 0),
                    unique_prediction_count=int(result.get("unique_prediction_count", 0) or 0),
                )
            )
    for result in symbolic_variable_operator_candidates(examples, query):
        if result.get("status") == "candidate" and result.get("prediction"):
            out.append(
                Candidate(
                    source="v324_variable_operator_symbolic",
                    rule_class=str(result.get("rule_class", "")),
                    prediction=str(result.get("prediction", "")),
                    proof=str(result.get("proof", "")),
                    program_count=int(result.get("candidate_program_count", 0) or 0),
                    unique_prediction_count=int(result.get("unique_prediction_count", 0) or 0),
                )
            )
    return out


def row_candidates(row: dict[str, Any], examples: list[tuple[str, str]], query: str, subtype: str) -> list[Candidate]:
    metadata = row.get("metadata", {})
    adapter_prediction = str(metadata.get("adapter_prediction", ""))
    candidates = guarded_adapter_relation_candidate(examples, query, adapter_prediction)
    if subtype == "equation_numeric_operator":
        for scope in ("same", "global"):
            candidates.extend(infer_numeric_v2_candidates(examples, query, scope=scope, min_rows=2))
    else:
        candidates.extend(existing_symbolic_candidates(examples, query))
    return candidates


def build_pair(row: dict[str, Any], candidate: Candidate, split: str) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    rejected_answer = str(metadata.get("adapter_prediction", ""))
    return {
        "id": "v452_" + str(row.get("id", "")),
        "source": "v452_equation_dsl_v2_certified_builder",
        "family": "equation_transform",
        "subcategory": f"v452_{candidate.source}_{candidate.rule_class}",
        "prompt": row.get("prompt", ""),
        "chosen": boxed(candidate.prediction),
        "rejected": boxed(rejected_answer),
        "messages": [
            {"role": "system", "content": "Solve the KG1 puzzle. End with exactly one final answer in \\boxed{}."},
            {"role": "user", "content": row.get("prompt", "")},
            {"role": "assistant", "content": boxed(candidate.prediction)},
        ],
        "metadata": {
            "schema_version": "kg1_v452_equation_dsl_v2_pair_v1",
            "source_id": metadata.get("source_id", ""),
            "source_pair_id": row.get("id", ""),
            "split": split,
            "adapter_prediction": rejected_answer,
            "adapter_exact_wrong_certificate": bool(metadata.get("adapter_exact_wrong_certificate")),
            "adapter_repo": metadata.get("adapter_repo", ""),
            "adapter_subfolder": metadata.get("adapter_subfolder", ""),
            "prompt_sha256": metadata.get("prompt_sha256", ""),
            "prompt_normalized_sha256": metadata.get("prompt_normalized_sha256", ""),
            "weak_gate_rows_used_for_training": False,
            "full_gate_rows_used_for_training": False,
            "labels_joined_after_collection_from_public_train": True,
            "rule_frozen_before_answer": True,
            "rule_class_no_loss_on_v452_audit": True,
            "candidate_source": candidate.source,
            "rule_class": candidate.rule_class,
            "proof": candidate.proof,
            "program_count": candidate.program_count,
            "unique_prediction_count": candidate.unique_prediction_count,
            "loo_checked": candidate.loo_checked,
            "loo_passed": candidate.loo_passed,
            "answer_audit": metadata.get("answer", ""),
            "audit_correct_after_freeze": verify_answer(str(metadata.get("answer", "")), candidate.prediction),
        },
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("metadata", {}).get("source_id", row.get("id", "")))
        bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest(), 16) % 5
        (val if bucket == 0 else train).append(row)
    return train, val


def audit_source_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_pass_rows: list[dict[str, Any]] = []
    candidate_rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_stats: dict[str, Counter[str]] = defaultdict(Counter)

    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"v452_audit_progress = {index}/{len(rows)}", flush=True)
        metadata = row.get("metadata", {})
        family = str(row.get("family", metadata.get("family", "")))
        split = str(metadata.get("split", ""))
        answer = str(metadata.get("answer", ""))
        adapter_prediction = str(metadata.get("adapter_prediction", ""))
        examples, query, parse_status = parse_alice_prompt(str(row.get("prompt", "")))
        subtype = classify_subtype(examples, query) if parse_status == "ok" else ""
        base_audit = {
            "id": row.get("id", ""),
            "source_id": metadata.get("source_id", ""),
            "split": split,
            "family": family,
            "subcategory": row.get("subcategory", metadata.get("rule_class", "")),
            "parse_status": parse_status,
            "subtype": subtype,
            "query": query,
            "answer": answer,
            "adapter_prediction": adapter_prediction,
            "candidate_source": "",
            "rule_class": "",
            "status": "abstain",
            "prediction": "",
            "candidate_correct": False,
            "candidate_conflict": False,
            "class_safe": False,
            "row_promoted": False,
            "rejection_reason": "",
            "candidate_program_count": 0,
            "unique_prediction_count": 0,
            "loo_checked": 0,
            "loo_passed": 0,
            "proof": "",
        }
        if family != "equation_transform":
            base_audit["rejection_reason"] = "not_equation"
            first_pass_rows.append(base_audit)
            continue
        if parse_status != "ok":
            base_audit["rejection_reason"] = parse_status
            first_pass_rows.append(base_audit)
            continue

        candidates = row_candidates(row, examples, query, subtype)
        if not candidates:
            base_audit["rejection_reason"] = "no_candidate"
            first_pass_rows.append(base_audit)
            continue

        predictions = sorted({candidate.prediction for candidate in candidates if candidate.prediction})
        row_conflict = len(predictions) > 1
        for candidate in candidates:
            candidate_correct = verify_answer(answer, candidate.prediction)
            rule_key = f"{candidate.source}:{candidate.rule_class}"
            class_stats[rule_key]["candidates"] += 1
            class_stats[rule_key]["correct"] += int(candidate_correct)
            class_stats[rule_key]["incorrect"] += int(not candidate_correct)
            class_stats[rule_key]["changed"] += int(normalize_payload(candidate.prediction) != normalize_payload(adapter_prediction))
            audit = {
                **base_audit,
                "candidate_source": candidate.source,
                "rule_class": candidate.rule_class,
                "status": "candidate",
                "prediction": candidate.prediction,
                "candidate_correct": candidate_correct,
                "candidate_conflict": row_conflict,
                "candidate_program_count": candidate.program_count,
                "unique_prediction_count": candidate.unique_prediction_count,
                "loo_checked": candidate.loo_checked,
                "loo_passed": candidate.loo_passed,
                "proof": candidate.proof,
            }
            first_pass_rows.append(audit)
            candidate_rows_by_id[str(row.get("id", ""))].append(audit)

    safe_classes = {rule for rule, stats in class_stats.items() if stats["candidates"] and not stats["incorrect"] and stats["correct"]}
    promoted_audits: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    row_by_id = {str(row.get("id", "")): row for row in rows}

    for audit in first_pass_rows:
        if audit["status"] != "candidate":
            promoted_audits.append(audit)
            continue
        rule_key = f"{audit['candidate_source']}:{audit['rule_class']}"
        class_safe = rule_key in safe_classes
        same_row_candidates = candidate_rows_by_id[str(audit["id"])]
        distinct_predictions = {str(item["prediction"]) for item in same_row_candidates}
        row_conflict = len(distinct_predictions) > 1
        old_correct = verify_answer(str(audit["answer"]), str(audit["adapter_prediction"]))
        row_promoted = (
            class_safe
            and not row_conflict
            and bool(audit["candidate_correct"])
            and not old_correct
            and normalize_payload(str(audit["prediction"])) != normalize_payload(str(audit["adapter_prediction"]))
        )
        reason = ""
        if not class_safe:
            reason = "rule_class_has_incorrect_candidates"
        elif row_conflict:
            reason = "row_has_conflicting_candidate_predictions"
        elif old_correct:
            reason = "adapter_already_correct"
        elif not audit["candidate_correct"]:
            reason = "candidate_not_correct"
        elif normalize_payload(str(audit["prediction"])) == normalize_payload(str(audit["adapter_prediction"])):
            reason = "candidate_same_as_adapter_prediction"
        promoted = {**audit, "class_safe": class_safe, "row_promoted": row_promoted, "rejection_reason": reason}
        promoted_audits.append(promoted)
        if row_promoted:
            source_row = row_by_id[str(audit["id"])]
            candidate = Candidate(
                source=str(audit["candidate_source"]),
                rule_class=str(audit["rule_class"]),
                prediction=str(audit["prediction"]),
                proof=str(audit["proof"]),
                program_count=int(audit["candidate_program_count"]),
                unique_prediction_count=int(audit["unique_prediction_count"]),
                loo_checked=int(audit["loo_checked"]),
                loo_passed=int(audit["loo_passed"]),
            )
            pairs.append(build_pair(source_row, candidate, str(audit["split"])))

    return promoted_audits, pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v439_manifest = read_json(args.v439_manifest_json)
    source_rows = read_jsonl(to_path(v439_manifest["train_jsonl"])) + read_jsonl(to_path(v439_manifest["val_jsonl"]))
    print("v452_source_rows =", len(source_rows), flush=True)

    audit_rows, pairs = audit_source_rows(source_rows)
    train_pairs, val_pairs = split_rows(pairs)

    audit_csv = args.output_dir / f"{args.label}_audit.csv"
    preferences_train_jsonl = args.output_dir / f"{args.label}_preferences_train.jsonl"
    preferences_val_jsonl = args.output_dir / f"{args.label}_preferences_val.jsonl"
    manifest_json = args.output_dir / f"{args.label}_manifest.json"

    write_csv(audit_csv, audit_rows, AUDIT_COLUMNS)
    write_jsonl(preferences_train_jsonl, train_pairs)
    write_jsonl(preferences_val_jsonl, val_pairs)

    status_counts = Counter(str(row.get("status", "")) for row in audit_rows)
    reason_counts = Counter(str(row.get("rejection_reason", "")) for row in audit_rows)
    class_summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        if row["status"] == "candidate":
            grouped[(str(row["candidate_source"]), str(row["rule_class"]))].append(row)
    for (source, rule_class), items in sorted(grouped.items()):
        class_summary.append(
            {
                "candidate_source": source,
                "rule_class": rule_class,
                "candidate_rows": len(items),
                "correct_candidates": sum(1 for item in items if bool(item["candidate_correct"])),
                "incorrect_candidates": sum(1 for item in items if not bool(item["candidate_correct"])),
                "promoted_rows": sum(1 for item in items if bool(item["row_promoted"])),
                "class_safe": all(bool(item["class_safe"]) for item in items),
            }
        )

    independent_modes = len({pair["subcategory"] for pair in pairs})
    hf_gpu_allowed = (
        len(pairs) >= args.min_pairs
        and len(val_pairs) >= args.min_val_pairs
        and independent_modes >= args.min_independent_modes
    )
    decision = {
        "decision": "v452_cpu_pairs_ready_for_integration_gate" if hf_gpu_allowed else "v452_cpu_pairs_insufficient_no_gpu",
        "reason": f"pairs={len(pairs)}, val_pairs={len(val_pairs)}, modes={independent_modes}",
        "next_action": (
            "Run tokenization/pre-paid integration gate before HF."
            if hf_gpu_allowed
            else "Do not launch GPU; expand DSL or use submit-safe postprocessor route."
        ),
    }

    manifest = {
        "schema_version": "kg1_v452_equation_dsl_v2_certified_builder_v1",
        "label": args.label,
        "generated_at_utc": utc_now(),
        "inputs": {
            "v439_manifest_json": repo_rel(args.v439_manifest_json),
            "source_rows": len(source_rows),
        },
        "summary": {
            "audit_rows": len(audit_rows),
            "candidate_rows": status_counts.get("candidate", 0),
            "certified_pair_rows": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "independent_modes": independent_modes,
            "status_counts": dict(status_counts),
            "reason_counts": dict(reason_counts),
            "class_summary": class_summary,
        },
        "hf_gpu_allowed": hf_gpu_allowed,
        "decision": decision,
        "outputs": {
            "audit_csv": repo_rel(audit_csv),
            "preferences_train_jsonl": repo_rel(preferences_train_jsonl),
            "preferences_train_sha256": sha256_path(preferences_train_jsonl),
            "preferences_val_jsonl": repo_rel(preferences_val_jsonl),
            "preferences_val_sha256": sha256_path(preferences_val_jsonl),
            "manifest_json": repo_rel(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest


def run_self_test() -> None:
    examples = [
        ("73*57", "6772"),
        ("29*49", "9468"),
        ("56+16", "1656"),
    ]
    query = "22-84"
    candidates = guarded_adapter_relation_candidate(examples, query, "62")
    if not candidates or candidates[0].prediction != "-62":
        raise AssertionError(candidates)
    numeric = infer_numeric_v2_candidates(
        [("79-12", "67"), ("27-05", "22"), ("65-21", "44")],
        "66-11",
        scope="same",
        min_rows=2,
    )
    if not numeric or numeric[0].prediction != "55":
        raise AssertionError(numeric)
    print("v452_equation_dsl_v2_certified_builder_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v439-manifest-json", type=Path, default=DEFAULT_V439_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v452_equation_dsl_v2_certified_builder")
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--min-val-pairs", type=int, default=4)
    parser.add_argument("--min-independent-modes", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    print("=== V452 EQUATION DSL V2 CERTIFIED BUILDER START ===", flush=True)
    print("v439_manifest_json =", args.v439_manifest_json, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("label =", args.label, flush=True)
    manifest = run(args)
    print("summary =", json.dumps(manifest["summary"], sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], sort_keys=True), flush=True)
    print("hf_gpu_allowed =", manifest["hf_gpu_allowed"], flush=True)
    print("manifest_json =", manifest["outputs"]["manifest_json"], flush=True)
    print("=== V452 EQUATION DSL V2 CERTIFIED BUILDER END ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

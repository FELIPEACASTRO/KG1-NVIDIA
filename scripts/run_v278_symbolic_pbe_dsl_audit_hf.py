#!/usr/bin/env python3
"""Run a stronger CPU-only symbolic PBE DSL audit on current equation misses.

V278 extends the V272/V246 conservative audits with small FlashFill/CEGIS-style
string DSL families. It is diagnostic-only: it downloads the current-best weak
prediction CSV, validates the V221 shared row contract, tests candidate
programs derived only from prompt examples, and uses weak labels only as an
audit brake. It does not train, run model generation, package, or submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from competition_utils import canonical_family, classify_puzzle, verify_answer  # noqa: E402
from analyze_v238_alice_parser_probes import answers_equal, parse_alice_prompt, parse_numeric_token  # noqa: E402
from analyze_v241_abstain_rule_candidate_audit import infer_symbolic_transducer, numeric_rule_functions  # noqa: E402


EXPECTED_ROW_CONTRACT_SHA256 = "bf055e3b9ebce79d4bfc9e48bce5a305b1d83da882f14afddec80d6afaba5fff"
DEFAULT_BASELINE_REPO = "felipesp1983/kg1-nemotron-lora-v259-v249-eqfocus-v257ckpt4-smoke"
DEFAULT_BASELINE_FILENAME = (
    "evals/v260b-h200-v221contract-v259-eqfocus-eval-20260511T025751Z/eval/"
    "v259_checkpoint_4_v221_contract/v245_hf_weak_v259_checkpoint_4_v221_contract_predictions.csv"
)

AUDIT_COLUMNS = [
    "id",
    "subtype",
    "rule_class",
    "status",
    "prediction",
    "answer",
    "baseline_prediction",
    "verified_by_weak_label",
    "incorrect_by_weak_label",
    "promotable_after_class_gate",
    "query",
    "example_count",
    "candidate_program_count",
    "unique_prediction_count",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download_file(repo_id: str, filename: str, local_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for V278") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename.strip("/"),
            local_dir=str(local_dir),
            token=token,
        )
    )


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    family = canonical_family(row.get("family") or row.get("task_type") or row.get("type") or classify_puzzle(prompt))
    return {
        **row,
        "id": str(row.get("id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "prediction": prediction,
        "family": family,
        "prompt_sha256": sha256_text(prompt),
        "correct_bool": verify_answer(answer, prediction),
        "truncated_bool": truthy(row.get("truncated", row.get("truncated_bool", "False"))),
    }


def row_contract(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 315:
        raise RuntimeError(f"expected 315 rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate ids in baseline predictions")
    payload = "\n".join(
        f"{row['id']}\t{row['family']}\t{row['answer']}\t{row['prompt_sha256']}"
        for row in sorted(rows, key=lambda item: item["id"])
    )
    return sha256_text(payload)


def classify_subtype(examples: list[tuple[str, str]], query: str) -> str:
    if examples and parse_numeric_token(query) and all(parse_numeric_token(lhs) for lhs, _ in examples):
        return "equation_numeric_operator"
    return "equation_symbolic_punct"


def candidate_from_predictions(
    rule_class: str,
    predictions: list[str],
    proof: str,
    *,
    candidate_program_count: int,
) -> dict[str, Any]:
    unique = sorted(set(prediction for prediction in predictions if prediction != ""))
    if len(unique) == 1:
        return {
            "rule_class": rule_class,
            "status": "candidate",
            "prediction": unique[0],
            "proof": proof,
            "candidate_program_count": candidate_program_count,
            "unique_prediction_count": 1,
        }
    return {
        "rule_class": rule_class,
        "status": "abstain",
        "prediction": "",
        "proof": proof + f"; unique_prediction_count={len(unique)}",
        "candidate_program_count": candidate_program_count,
        "unique_prediction_count": len(unique),
    }


def subset_candidates(values: list[str], max_size: int) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for size in range(1, min(max_size, len(values)) + 1):
        subsets.extend(itertools.combinations(values, size))
    return subsets


def delete_char_set_candidate(
    examples: list[tuple[str, str]],
    query: str,
    max_subset_size: int,
) -> dict[str, Any]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    programs: list[tuple[str, ...]] = []
    predictions: list[str] = []
    for delete_chars in subset_candidates(alphabet, max_subset_size):
        delete_set = set(delete_chars)
        if all("".join(ch for ch in lhs if ch not in delete_set) == rhs for lhs, rhs in examples):
            programs.append(delete_chars)
            predictions.append("".join(ch for ch in query if ch not in delete_set))
    proof = "delete_char_sets=" + ";".join("".join(chars) for chars in programs[:20])
    return candidate_from_predictions(
        "symbolic_delete_selected_chars",
        predictions,
        proof,
        candidate_program_count=len(programs),
    )


def keep_char_set_candidate(
    examples: list[tuple[str, str]],
    query: str,
    max_subset_size: int,
) -> dict[str, Any]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    programs: list[tuple[str, ...]] = []
    predictions: list[str] = []
    for keep_chars in subset_candidates(alphabet, max_subset_size):
        keep_set = set(keep_chars)
        if all("".join(ch for ch in lhs if ch in keep_set) == rhs for lhs, rhs in examples):
            prediction = "".join(ch for ch in query if ch in keep_set)
            if prediction:
                programs.append(keep_chars)
                predictions.append(prediction)
    proof = "keep_char_sets=" + ";".join("".join(chars) for chars in programs[:20])
    return candidate_from_predictions(
        "symbolic_keep_selected_chars",
        predictions,
        proof,
        candidate_program_count=len(programs),
    )


def marker_candidate(
    rule_class: str,
    examples: list[tuple[str, str]],
    query: str,
    transform: Callable[[str, str], str | None],
) -> dict[str, Any]:
    alphabet = sorted(set("".join(lhs for lhs, _ in examples)))
    programs: list[str] = []
    predictions: list[str] = []
    for marker in alphabet:
        values = [transform(lhs, marker) for lhs, _ in examples]
        if any(value is None for value in values):
            continue
        if all(str(value) == rhs for value, (_, rhs) in zip(values, examples)):
            prediction = transform(query, marker)
            if prediction:
                programs.append(marker)
                predictions.append(str(prediction))
    proof = "markers=" + repr("".join(programs[:40]))
    return candidate_from_predictions(rule_class, predictions, proof, candidate_program_count=len(programs))


def before_first_marker(text: str, marker: str) -> str | None:
    return text.split(marker, 1)[0] if marker in text else None


def after_first_marker(text: str, marker: str) -> str | None:
    return text.split(marker, 1)[1] if marker in text else None


def before_last_marker(text: str, marker: str) -> str | None:
    return text.rsplit(marker, 1)[0] if marker in text else None


def after_last_marker(text: str, marker: str) -> str | None:
    return text.rsplit(marker, 1)[1] if marker in text else None


def remove_first_marker(text: str, marker: str) -> str | None:
    return text.replace(marker, "", 1) if marker in text else None


def source_functions(max_len: int) -> list[tuple[str, Callable[[str], str | None]]]:
    funcs: list[tuple[str, Callable[[str], str | None]]] = []
    for index in range(max_len):
        funcs.append((f"start{index}", lambda text, idx=index: text[idx] if idx < len(text) else None))
        funcs.append((f"end{index}", lambda text, idx=index: text[-idx - 1] if idx < len(text) else None))
    return funcs


def position_template_candidate(examples: list[tuple[str, str]], query: str, max_sources: int) -> dict[str, Any]:
    rhs_lengths = {len(rhs) for _, rhs in examples}
    if len(rhs_lengths) != 1:
        return {
            "rule_class": "symbolic_position_template",
            "status": "abstain",
            "prediction": "",
            "proof": "nonuniform_rhs_lengths",
            "candidate_program_count": 0,
            "unique_prediction_count": 0,
        }
    output_len = next(iter(rhs_lengths))
    if output_len == 0 or output_len > 6:
        return {
            "rule_class": "symbolic_position_template",
            "status": "abstain",
            "prediction": "",
            "proof": f"unsupported_output_len={output_len}",
            "candidate_program_count": 0,
            "unique_prediction_count": 0,
        }
    max_len = min(max_sources, max(len(lhs) for lhs, _ in examples + [(query, "")]))
    funcs = source_functions(max_len)
    choices: list[list[tuple[str, Callable[[str], str | None]]]] = []
    for out_index in range(output_len):
        out_choices: list[tuple[str, Callable[[str], str | None]]] = []
        literal_value = examples[0][1][out_index]
        if all(rhs[out_index] == literal_value for _, rhs in examples):
            out_choices.append((f"literal:{literal_value}", lambda _text, value=literal_value: value))
        for name, func in funcs:
            values = [func(lhs) for lhs, _ in examples]
            if all(value is not None for value in values) and all(str(value) == rhs[out_index] for value, (_, rhs) in zip(values, examples)):
                if func(query) is not None:
                    out_choices.append((name, func))
        if not out_choices:
            return {
                "rule_class": "symbolic_position_template",
                "status": "abstain",
                "prediction": "",
                "proof": f"no_source_for_output_index={out_index}",
                "candidate_program_count": 0,
                "unique_prediction_count": 0,
            }
        choices.append(out_choices)
    predictions: list[str] = []
    program_count = 1
    for choice in choices:
        program_count *= len(choice)
    if program_count > 5000:
        return {
            "rule_class": "symbolic_position_template",
            "status": "abstain",
            "prediction": "",
            "proof": f"program_count_above_cap={program_count}",
            "candidate_program_count": program_count,
            "unique_prediction_count": 0,
        }
    for program in itertools.product(*choices):
        names = [name for name, _ in program]
        if all(name.startswith("literal:") for name in names):
            continue
        prediction = "".join(str(func(query)) for _, func in program)
        predictions.append(prediction)
    proof = "choice_counts=" + ",".join(str(len(choice)) for choice in choices)
    return candidate_from_predictions(
        "symbolic_position_template",
        predictions,
        proof,
        candidate_program_count=program_count,
    )


def op_index_template_candidate(examples: list[tuple[str, str]], query: str) -> dict[str, Any]:
    programs: list[str] = []
    predictions: list[str] = []

    def transforms(text: str, op_index: int) -> dict[str, str]:
        if op_index <= 0 or op_index >= len(text) - 1:
            return {}
        left = text[:op_index]
        right = text[op_index + 1 :]
        return {
            "drop_op": left + right,
            "left": left,
            "right": right,
            "right_left": right + left,
            "reverse_drop_op": (left + right)[::-1],
            "reverse_left_right": left[::-1] + right,
            "left_reverse_right": left + right[::-1],
            "reverse_right_left": right[::-1] + left,
            "right_reverse_left": right + left[::-1],
        }

    max_len = max(len(lhs) for lhs, _ in examples + [(query, "")])
    for op_index in range(1, max_len - 1):
        query_transforms = transforms(query, op_index)
        if not query_transforms:
            continue
        for name in sorted(query_transforms):
            ok = True
            for lhs, rhs in examples:
                if transforms(lhs, op_index).get(name) != rhs:
                    ok = False
                    break
            if ok:
                programs.append(f"{name}@{op_index}")
                predictions.append(query_transforms[name])
    proof = "programs=" + ",".join(programs[:40])
    return candidate_from_predictions(
        "symbolic_operator_index_template",
        predictions,
        proof,
        candidate_program_count=len(programs),
    )


def transducer_candidate(examples: list[tuple[str, str]], query: str, pair_cap: int, global_cap: int) -> dict[str, Any]:
    result = infer_symbolic_transducer(examples, query, pair_cap=pair_cap, global_cap=global_cap)
    return {
        "rule_class": "symbolic_char_transducer",
        "status": result.get("status", "abstain"),
        "prediction": result.get("prediction", ""),
        "proof": result.get("proof", ""),
        "candidate_program_count": result.get("mapping_count", 0),
        "unique_prediction_count": result.get("unique_prediction_count", 0),
    }


def numeric_candidate(examples: list[tuple[str, str]], query: str, min_examples: int) -> dict[str, Any]:
    parsed_query = parse_numeric_token(query)
    if not parsed_query:
        return {
            "rule_class": "numeric_same_operator_extended",
            "status": "abstain",
            "prediction": "",
            "proof": "query_not_numeric_binary",
            "candidate_program_count": 0,
            "unique_prediction_count": 0,
        }
    same_operator: list[tuple[int, int, str]] = []
    for lhs, rhs in examples:
        parsed = parse_numeric_token(lhs)
        if parsed and parsed[1] == parsed_query[1]:
            same_operator.append((parsed[0], parsed[2], str(rhs)))
    if len(same_operator) < min_examples:
        return {
            "rule_class": "numeric_same_operator_extended",
            "status": "abstain",
            "prediction": "",
            "proof": f"same_operator_examples={len(same_operator)} below_min={min_examples}",
            "candidate_program_count": 0,
            "unique_prediction_count": 0,
        }
    candidates: list[tuple[str, str]] = []
    for name, func in numeric_rule_functions().items():
        ok = True
        for left, right, expected in same_operator:
            try:
                prediction = func(left, right)
            except Exception:
                ok = False
                break
            if prediction != expected:
                ok = False
                break
        if ok:
            try:
                candidates.append((name, func(parsed_query[0], parsed_query[2])))
            except Exception:
                pass
    predictions = [prediction for _, prediction in candidates]
    proof = "same_operator_examples=" + str(len(same_operator)) + "; rules=" + ",".join(name for name, _ in candidates)
    return candidate_from_predictions(
        "numeric_same_operator_extended",
        predictions,
        proof,
        candidate_program_count=len(candidates),
    )


def symbolic_candidates(examples: list[tuple[str, str]], query: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        transducer_candidate(examples, query, args.pair_mapping_cap, args.global_mapping_cap),
        delete_char_set_candidate(examples, query, args.max_char_subset_size),
        keep_char_set_candidate(examples, query, args.max_char_subset_size),
        marker_candidate("symbolic_prefix_until_marker", examples, query, before_first_marker),
        marker_candidate("symbolic_suffix_after_marker", examples, query, after_first_marker),
        marker_candidate("symbolic_prefix_before_last_marker", examples, query, before_last_marker),
        marker_candidate("symbolic_suffix_after_last_marker", examples, query, after_last_marker),
        marker_candidate("symbolic_remove_first_marker", examples, query, remove_first_marker),
        position_template_candidate(examples, query, args.max_position_sources),
        op_index_template_candidate(examples, query),
    ]


def build_audit_row(
    row: dict[str, Any],
    result: dict[str, Any],
    examples: list[tuple[str, str]],
    query: str,
) -> dict[str, Any]:
    status = str(result.get("status", "abstain"))
    prediction = str(result.get("prediction", ""))
    is_candidate = status == "candidate"
    verified = is_candidate and answers_equal(prediction, row["answer"])
    incorrect = is_candidate and not verified
    return {
        "id": row["id"],
        "subtype": classify_subtype(examples, query) if examples else "parse_failed",
        "rule_class": str(result.get("rule_class", "")),
        "status": status,
        "prediction": prediction,
        "answer": row["answer"],
        "baseline_prediction": row["prediction"],
        "verified_by_weak_label": verified,
        "incorrect_by_weak_label": incorrect,
        "promotable_after_class_gate": False,
        "query": query,
        "example_count": len(examples),
        "candidate_program_count": result.get("candidate_program_count", 0),
        "unique_prediction_count": result.get("unique_prediction_count", 0),
        "proof": str(result.get("proof", ""))[:700],
    }


def summarize_rule_classes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rule_class"])].append(row)
    summaries: list[dict[str, Any]] = []
    for rule_class, items in sorted(grouped.items()):
        candidates = [row for row in items if row["status"] == "candidate"]
        verified = [row for row in candidates if row["verified_by_weak_label"]]
        incorrect = [row for row in candidates if row["incorrect_by_weak_label"]]
        promotable = bool(candidates) and not incorrect
        for row in items:
            if row["status"] == "candidate":
                row["promotable_after_class_gate"] = promotable
        summaries.append(
            {
                "rule_class": rule_class,
                "rows": len(items),
                "candidate_rows": len(candidates),
                "verified_candidates": len(verified),
                "incorrect_candidates": len(incorrect),
                "abstain_rows": len(items) - len(candidates),
                "promotable_after_class_gate": promotable,
            }
        )
    summaries.sort(
        key=lambda item: (
            int(item["verified_candidates"]),
            -int(item["incorrect_candidates"]),
            -int(item["candidate_rows"]),
            str(item["rule_class"]),
        ),
        reverse=True,
    )
    return summaries


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    print("=== V278 SYMBOLIC PBE DSL AUDIT START ===", flush=True)
    print("generated_at_utc =", utc_now(), flush=True)
    print("baseline_repo =", args.baseline_repo, flush=True)
    print("baseline_filename =", args.baseline_filename, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("target_verified_gain =", args.target_verified_gain, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="kg1_v278_") as temp_name:
        prediction_path = download_file(args.baseline_repo, args.baseline_filename, Path(temp_name), token)
        rows = [normalize_row(row) for row in read_csv(prediction_path)]
        input_meta = {
            "repo_id": args.baseline_repo,
            "filename": args.baseline_filename,
            "downloaded_name": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "rows": len(rows),
        }

    observed_contract = row_contract(rows)
    print("observed_shared_row_contract_sha256 =", observed_contract, flush=True)
    if observed_contract != args.expected_shared_row_contract_sha256:
        raise RuntimeError(
            "row contract mismatch: expected "
            + args.expected_shared_row_contract_sha256
            + ", got "
            + observed_contract
        )

    equation_misses = [
        row
        for row in rows
        if row["family"] == "equation_transform" and not row["correct_bool"] and not row["truncated_bool"]
    ]
    audit_rows: list[dict[str, Any]] = []
    parse_status_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    for row in equation_misses:
        examples, query, parse_status = parse_alice_prompt(str(row["prompt"]))
        parse_status_counts[parse_status] += 1
        if parse_status != "ok":
            audit_rows.append(
                build_audit_row(
                    row,
                    {
                        "rule_class": "alice_parse_gate",
                        "status": "abstain",
                        "prediction": "",
                        "proof": parse_status,
                    },
                    examples,
                    query,
                )
            )
            continue
        subtype = classify_subtype(examples, query)
        subtype_counts[subtype] += 1
        if subtype == "equation_numeric_operator":
            results = [numeric_candidate(examples, query, args.min_same_operator_examples)]
        else:
            results = symbolic_candidates(examples, query, args)
        for result in results:
            audit_rows.append(build_audit_row(row, result, examples, query))

    summary_rows = summarize_rule_classes(audit_rows)
    verified_promotable = [
        row
        for row in audit_rows
        if row["status"] == "candidate" and row["verified_by_weak_label"] and row["promotable_after_class_gate"]
    ]
    incorrect_promotable = [
        row
        for row in audit_rows
        if row["status"] == "candidate" and row["incorrect_by_weak_label"] and row["promotable_after_class_gate"]
    ]
    all_verified = [row for row in audit_rows if row["status"] == "candidate" and row["verified_by_weak_label"]]
    all_incorrect = [row for row in audit_rows if row["status"] == "candidate" and row["incorrect_by_weak_label"]]

    outputs = {
        "audit_csv": args.output_dir / "v278_symbolic_pbe_dsl_audit.csv",
        "rule_summary_csv": args.output_dir / "v278_rule_class_summary.csv",
        "verified_candidates_csv": args.output_dir / "v278_verified_candidates.csv",
        "verified_promotable_csv": args.output_dir / "v278_verified_promotable_candidates.csv",
        "manifest_json": args.output_dir / "v278_symbolic_pbe_dsl_audit_manifest.json",
    }
    write_csv(outputs["audit_csv"], audit_rows, AUDIT_COLUMNS)
    write_csv(outputs["rule_summary_csv"], summary_rows, SUMMARY_COLUMNS)
    write_csv(outputs["verified_candidates_csv"], all_verified, AUDIT_COLUMNS)
    write_csv(outputs["verified_promotable_csv"], verified_promotable, AUDIT_COLUMNS)

    if len(verified_promotable) >= args.target_verified_gain and not incorrect_promotable:
        decision = "symbolic_pbe_candidates_ready_for_postprocessor_gate"
        next_action = "Create a guarded V279 postprocessor gate using only class-promotable zero-loss rules."
    elif verified_promotable:
        decision = "partial_symbolic_pbe_signal_below_target"
        next_action = "Review promotable rows; do not spend GPU unless combined with existing V275 keeps zero-loss and passes gate."
    else:
        decision = "no_promotable_symbolic_pbe_signal"
        next_action = "Do not spend GPU on this route; request gated andy279 traces or expand the DSL with external solver evidence."

    manifest = {
        "schema_version": "kg1_v278_symbolic_pbe_dsl_audit_v1",
        "generated_at_utc": utc_now(),
        "input": input_meta,
        "expected_shared_row_contract_sha256": args.expected_shared_row_contract_sha256,
        "observed_shared_row_contract_sha256": observed_contract,
        "equation_miss_rows": len(equation_misses),
        "parse_status_counts": dict(parse_status_counts),
        "subtype_counts": dict(subtype_counts),
        "rule_summary": summary_rows,
        "candidate_counts": {
            "all_verified_candidates": len(all_verified),
            "all_incorrect_candidates": len(all_incorrect),
            "verified_promotable_candidates": len(verified_promotable),
            "incorrect_promotable_candidates": len(incorrect_promotable),
        },
        "decision": {
            "decision": decision,
            "reason": (
                f"equation_misses={len(equation_misses)}; "
                f"all_verified={len(all_verified)}; "
                f"all_incorrect={len(all_incorrect)}; "
                f"verified_promotable={len(verified_promotable)}; "
                f"incorrect_promotable={len(incorrect_promotable)}"
            ),
            "next_action": next_action,
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)

    print("parse_status_counts =", json.dumps(dict(parse_status_counts), sort_keys=True), flush=True)
    print("subtype_counts =", json.dumps(dict(subtype_counts), sort_keys=True), flush=True)
    print("rule_summary =", json.dumps(summary_rows, indent=2, sort_keys=True), flush=True)
    print("candidate_counts =", json.dumps(manifest["candidate_counts"], indent=2, sort_keys=True), flush=True)
    print("decision =", json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)
    print("manifest_json =", outputs["manifest_json"], flush=True)
    print("=== V278 SYMBOLIC PBE DSL AUDIT END ===", flush=True)
    return manifest


def run_self_test() -> None:
    examples = [("abc", "ac"), ("bbca", "ca"), ("dbd", "dd")]
    delete_result = delete_char_set_candidate(examples, "bdc", 2)
    if delete_result["status"] != "candidate" or delete_result["prediction"] != "dc":
        raise AssertionError(delete_result)
    keep_examples = [("abc", "ac"), ("bac", "ac"), ("cba", "ca")]
    keep_result = keep_char_set_candidate(keep_examples, "bdca", 3)
    if keep_result["status"] != "candidate" or keep_result["prediction"] != "ca":
        raise AssertionError(keep_result)
    marker_examples = [("ab#cd", "ab"), ("xy#z", "xy")]
    marker_result = marker_candidate("test_prefix", marker_examples, "pq#rs", before_first_marker)
    if marker_result["status"] != "candidate" or marker_result["prediction"] != "pq":
        raise AssertionError(marker_result)
    position_examples = [("abcde", "bd"), ("vwxyz", "wy")]
    position_result = position_template_candidate(position_examples, "12345", 6)
    if position_result["status"] != "candidate" or position_result["prediction"] != "24":
        raise AssertionError(position_result)
    print("v278_self_test=ok", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-filename", default=DEFAULT_BASELINE_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v278_symbolic_pbe_dsl_audit"))
    parser.add_argument("--run-id", default=f"v278-hf-cpu-symbolic-pbe-dsl-audit-{utc_compact()}")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--expected-shared-row-contract-sha256", default=EXPECTED_ROW_CONTRACT_SHA256)
    parser.add_argument("--target-verified-gain", type=int, default=4)
    parser.add_argument("--pair-mapping-cap", type=int, default=3000)
    parser.add_argument("--global-mapping-cap", type=int, default=12000)
    parser.add_argument("--max-char-subset-size", type=int, default=4)
    parser.add_argument("--max-position-sources", type=int, default=7)
    parser.add_argument("--min-same-operator-examples", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

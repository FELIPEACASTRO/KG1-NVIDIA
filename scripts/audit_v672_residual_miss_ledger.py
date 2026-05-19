"""Build the V672 residual-miss ledger for submit-safe gain decisions.

The ledger is intentionally conservative.  A row is marked trainable only when
there is a direct accepted no-loss rule candidate, not merely an integrated
prediction that happens to match the weak label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition_utils import verify_answer  # noqa: E402


ACTIONABLE_MISS_CLASSES = {"equation_numeric_miss", "bit_residual_miss"}

DEFAULT_MISSES = (
    REPO_ROOT
    / "artifacts"
    / "v670_today_family_gain_plan"
    / "v670_v290_baseline_missmap"
    / "v541_weak_missmap_misses_only.csv"
)
DEFAULT_V350_INTEGRATED = (
    REPO_ROOT
    / "artifacts"
    / "v542_cpu_equation_solver_gate"
    / "v350_on_v516_strict"
    / "v350_integrated_predictions.csv"
)
DEFAULT_V350_DECISIONS = (
    REPO_ROOT
    / "artifacts"
    / "v542_cpu_equation_solver_gate"
    / "v350_on_v516_strict"
    / "v350_candidate_decisions.csv"
)
DEFAULT_V366_INTEGRATED = (
    REPO_ROOT
    / "artifacts"
    / "v366_bit_fullbyte_ternary_op_gate"
    / "20260514T_cpu_gate"
    / "v366_integrated_predictions.csv"
)
DEFAULT_V366_DECISIONS = (
    REPO_ROOT
    / "artifacts"
    / "v366_bit_fullbyte_ternary_op_gate"
    / "20260514T_cpu_gate"
    / "v366_candidate_decisions.csv"
)
DEFAULT_V336_TRACE = (
    REPO_ROOT
    / "artifacts"
    / "v542_cpu_equation_solver_gate"
    / "v336_integrated_on_v516_strict"
    / "v336a_integrated_no_loss_candidate_trace.csv"
)
DEFAULT_V324_ACCEPTED = (
    REPO_ROOT
    / "artifacts"
    / "v542_cpu_equation_solver_gate"
    / "v324_on_v516_strict"
    / "v324_equation_expanded_solver_accepted_candidates.csv"
)
DEFAULT_V333_DETAIL = (
    REPO_ROOT
    / "artifacts"
    / "v333_tong_bit_reasoner_gate"
    / "20260513T171304Z"
    / "v333_tong_bit_reasoner_gate_tong_bit_detail.csv"
)
DEFAULT_V334_DETAIL = (
    REPO_ROOT
    / "artifacts"
    / "v334_tong_equation_numeric_reasoner_gate"
    / "20260513T172300Z"
    / "v334_tong_equation_numeric_reasoner_gate_tong_equation_detail.csv"
)

EQUATION_QUERY_RE = re.compile(r"^\s*([0-9]+)\s*([^0-9\s=]+)\s*([0-9]+)\s*$")
PROMPT_EXAMPLE_RE = re.compile(r"^\s*(.+?)\s*=\s*(.+?)\s*$")
SOLVER_QUERY_RE = re.compile(r"\bquery=([^\s]+)")
SOLVER_EXAMPLES_RE = re.compile(r"\bexamples=(\d+)")


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def index_first(df: pd.DataFrame, key: str = "id") -> dict[str, dict[str, str]]:
    if df.empty or key not in df.columns:
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in df.to_dict("records"):
        row_id = clean(row.get(key))
        if row_id and row_id not in out:
            out[row_id] = {str(k): clean(v) for k, v in row.items()}
    return out


def index_accepted(df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    if df.empty or "id" not in df.columns or "accepted" not in df.columns:
        return out
    for row in df.to_dict("records"):
        if not boolish(row.get("accepted")):
            continue
        row_id = clean(row.get("id"))
        if row_id:
            out[row_id].append({str(k): clean(v) for k, v in row.items()})
    return out


def prompt_lines(prompt: str) -> list[str]:
    return [line.strip() for line in clean(prompt).splitlines() if line.strip()]


def parse_equation_expr(expr: str) -> tuple[str, str, str] | None:
    match = EQUATION_QUERY_RE.match(clean(expr))
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def query_from_solver_trace(trace: str) -> str:
    match = SOLVER_QUERY_RE.search(clean(trace))
    return match.group(1) if match else ""


def example_count_from_solver_trace(trace: str) -> int | None:
    match = SOLVER_EXAMPLES_RE.search(clean(trace))
    if not match:
        return None
    return int(match.group(1))


def equation_support(prompt: str, solver_trace: str) -> dict[str, Any]:
    query = query_from_solver_trace(solver_trace)
    parsed_query = parse_equation_expr(query)
    if not parsed_query:
        return {
            "query": query,
            "query_operator": "",
            "same_operator_examples": "",
            "total_examples": example_count_from_solver_trace(solver_trace) or "",
            "query_operator_support": "unparsed",
        }
    query_operator = parsed_query[1]
    same = 0
    total = 0
    for line in prompt_lines(prompt):
        if line.lower().startswith("now, determine"):
            break
        example_match = PROMPT_EXAMPLE_RE.match(line)
        if not example_match:
            continue
        lhs = example_match.group(1)
        parsed_lhs = parse_equation_expr(lhs)
        if not parsed_lhs:
            continue
        total += 1
        if parsed_lhs[1] == query_operator:
            same += 1
    support = "absent"
    if same >= 3:
        support = "three_plus"
    elif same == 2:
        support = "two"
    elif same == 1:
        support = "single"
    return {
        "query": query,
        "query_operator": query_operator,
        "same_operator_examples": same,
        "total_examples": total or (example_count_from_solver_trace(solver_trace) or ""),
        "query_operator_support": support,
    }


def ambiguity_risk(miss_class: str, support: str, candidate_direct: bool) -> str:
    if miss_class == "bit_residual_miss":
        return "low" if candidate_direct else "medium_rule_not_proven"
    if support in {"three_plus", "two"}:
        return "low" if candidate_direct else "medium_rule_not_proven"
    if support == "single":
        return "medium_single_same_operator_example"
    if support in {"absent", "unparsed"}:
        return "high_operator_not_constrained"
    return "unknown"


def boxed_answer(value: str) -> str:
    return "\\boxed{" + clean(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}") + "}"


def token_estimate(value: str) -> int:
    # Cheap conservative estimate for the boxed answer-only target.
    return max(1, len(boxed_answer(value)))


def direct_candidate_for_row(
    row_id: str,
    miss_class: str,
    accepted_sources: dict[str, dict[str, list[dict[str, str]]]],
) -> tuple[dict[str, str] | None, str]:
    """Pick a direct accepted candidate, preferring equation proof for equation rows."""

    order = (
        ("v336_no_loss_trace", accepted_sources["v336"]),
        ("v324_equation_accepted", accepted_sources["v324"]),
        ("v366_bit_ternary", accepted_sources["v366"]),
        ("v350_no_loss", accepted_sources["v350"]),
    )
    for source_name, source_index in order:
        candidates = source_index.get(row_id, [])
        if not candidates:
            continue
        for candidate in candidates:
            if miss_class == "equation_numeric_miss" and source_name not in {
                "v336_no_loss_trace",
                "v324_equation_accepted",
            }:
                continue
            if miss_class == "bit_residual_miss" and source_name not in {
                "v366_bit_ternary",
                "v350_no_loss",
            }:
                continue
            return candidate, source_name
    return None, ""


def candidate_prediction(candidate: dict[str, str] | None) -> str:
    if not candidate:
        return ""
    for key in ("new_prediction", "prediction"):
        value = clean(candidate.get(key))
        if value:
            return value
    return ""


def candidate_rule(candidate: dict[str, str] | None) -> str:
    if not candidate:
        return ""
    return clean(candidate.get("rule_class") or candidate.get("subtype"))


def candidate_proof(candidate: dict[str, str] | None) -> str:
    if not candidate:
        return ""
    return clean(candidate.get("proof") or candidate.get("reason"))


def build_ledger(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    misses_df = load_csv(args.misses_csv)
    if misses_df.empty:
        raise FileNotFoundError(f"No miss-map rows loaded from {args.misses_csv}")

    rows_df = misses_df[misses_df["miss_class"].isin(ACTIONABLE_MISS_CLASSES)].copy()

    v350_int = index_first(load_csv(args.v350_integrated))
    v366_int = index_first(load_csv(args.v366_integrated))
    v333 = index_first(load_csv(args.v333_detail))
    v334 = index_first(load_csv(args.v334_detail))

    accepted_sources = {
        "v350": index_accepted(load_csv(args.v350_decisions)),
        "v366": index_accepted(load_csv(args.v366_decisions)),
        "v336": index_accepted(load_csv(args.v336_trace)),
        "v324": index_first(load_csv(args.v324_accepted)),
    }
    # v324 accepted candidates do not have an "accepted" column; normalize them
    # into the same id -> list[dict] shape.
    accepted_sources["v324"] = {
        key: [value] for key, value in accepted_sources["v324"].items()
    }

    ledger: list[dict[str, Any]] = []
    for row in rows_df.to_dict("records"):
        row = {str(k): clean(v) for k, v in row.items()}
        row_id = row["id"]
        answer = row["answer"]
        baseline_prediction = row.get("extracted_answer", "")
        miss_class = row["miss_class"]

        direct_candidate, direct_source = direct_candidate_for_row(
            row_id, miss_class, accepted_sources
        )
        direct_prediction = candidate_prediction(direct_candidate)
        direct_correct = verify_answer(answer, direct_prediction) if direct_prediction else False

        v350_row = v350_int.get(row_id, {})
        v366_row = v366_int.get(row_id, {})
        v333_row = v333.get(row_id, {})
        v334_row = v334.get(row_id, {})

        prompt = clean(v350_row.get("prompt") or v366_row.get("prompt"))
        support = {
            "query": "",
            "query_operator": "",
            "same_operator_examples": "",
            "total_examples": "",
            "query_operator_support": "",
        }
        if miss_class == "equation_numeric_miss":
            support = equation_support(prompt, row.get("solver_trace", ""))

        candidate_source = direct_source
        candidate_directness = "direct_accepted_no_loss_rule" if direct_candidate else ""
        candidate = direct_candidate
        candidate_pred = direct_prediction
        candidate_rule_text = candidate_rule(candidate)
        candidate_proof_text = candidate_proof(candidate)
        leakage_risk = "low_direct_rule_candidate" if direct_candidate else "none"
        trainability_decision = "drop"
        trainability_reason = "no_direct_candidate"

        if direct_candidate and direct_correct:
            risk = ambiguity_risk(miss_class, str(support["query_operator_support"]), True)
            if risk.startswith("low"):
                trainability_decision = "trainable"
                trainability_reason = "direct_accepted_rule_zero_loss_candidate"
            elif (
                risk.startswith("medium")
                and clean(direct_candidate.get("candidate_count")) in {"", "1"}
                and clean(direct_candidate.get("candidate_program_count")) in {"", "1"}
                and clean(direct_candidate.get("conflict_count")) in {"", "0"}
                and direct_source in {"v336_no_loss_trace", "v324_equation_accepted"}
            ):
                trainability_decision = "trainable_guarded"
                trainability_reason = (
                    "direct_unique_no_loss_candidate_but_single_operator_support"
                )
            else:
                trainability_decision = "protected-only"
                trainability_reason = risk
        elif direct_candidate:
            trainability_decision = "drop"
            trainability_reason = "direct_candidate_not_verified"

        inherited_candidates: list[str] = []
        for source_name, source_row, pred_key, correct_key, rule_key in [
            ("v350", v350_row, "v350_prediction", "v350_correct", "v350_source_rule"),
            ("v366", v366_row, "v366_prediction", "v366_correct", "v366_source_rule"),
        ]:
            pred = clean(source_row.get(pred_key))
            correct = boolish(source_row.get(correct_key)) or (
                bool(pred) and verify_answer(answer, pred)
            )
            source_rule = clean(source_row.get(rule_key))
            if correct and pred and not direct_candidate:
                inherited_candidates.append(f"{source_name}:{source_rule or 'integrated_unknown'}={pred}")

        if inherited_candidates and not direct_candidate:
            candidate_source = "integrated_inherited"
            candidate_directness = "inherited_correct_needs_rule_proof"
            candidate_pred = inherited_candidates[0].split("=", 1)[1]
            candidate_rule_text = "; ".join(inherited_candidates)
            leakage_risk = "medium_inherited_prediction_without_rule_trace"
            trainability_decision = "needs_rule_proof"
            trainability_reason = "integrated_prediction_correct_but_no_direct_accepted_candidate"

        support_label = str(support["query_operator_support"])
        row_out: dict[str, Any] = {
            "id": row_id,
            "family": row.get("family"),
            "miss_class": miss_class,
            "prompt_sha256": row.get("prompt_sha256"),
            "answer": answer,
            "baseline_prediction": baseline_prediction,
            "baseline_correct": boolish(row.get("verify_answer_ok")),
            "baseline_token_count": row.get("token_count"),
            "baseline_truncated": boolish(row.get("truncated")),
            "operator_tag": row.get("operator_tag"),
            "required_rule": row.get("required_rule"),
            "solver_trace": row.get("solver_trace"),
            "query": support["query"],
            "query_operator": support["query_operator"],
            "same_operator_examples": support["same_operator_examples"],
            "total_examples": support["total_examples"],
            "query_operator_support": support_label,
            "candidate_prediction": candidate_pred,
            "candidate_prediction_boxed": boxed_answer(candidate_pred) if candidate_pred else "",
            "candidate_source": candidate_source,
            "candidate_rule": candidate_rule_text,
            "candidate_directness": candidate_directness,
            "candidate_proof": candidate_proof_text,
            "candidate_correct": verify_answer(answer, candidate_pred) if candidate_pred else False,
            "gain_vs_baseline": (
                (not boolish(row.get("verify_answer_ok")))
                and bool(candidate_pred)
                and verify_answer(answer, candidate_pred)
            ),
            "ambiguity_risk": ambiguity_risk(miss_class, support_label, bool(direct_candidate)),
            "leakage_risk": leakage_risk,
            "trainability_decision": trainability_decision,
            "trainability_reason": trainability_reason,
            "verify_answer_candidate": verify_answer(answer, candidate_pred) if candidate_pred else False,
            "target_boxed_token_estimate": token_estimate(candidate_pred or answer),
            "prompt_available": bool(prompt),
            "v350_prediction": clean(v350_row.get("v350_prediction")),
            "v350_correct": boolish(v350_row.get("v350_correct")),
            "v350_source_rule": clean(v350_row.get("v350_source_rule")),
            "v350_direct_accepted": bool(row_id in accepted_sources["v350"]),
            "v366_prediction": clean(v366_row.get("v366_prediction")),
            "v366_correct": boolish(v366_row.get("v366_correct")),
            "v366_source_rule": clean(v366_row.get("v366_source_rule")),
            "v366_direct_accepted": bool(row_id in accepted_sources["v366"]),
            "v333_tong_prediction": clean(v333_row.get("tong_prediction")),
            "v333_correct": boolish(v333_row.get("tong_correct")),
            "v333_status": clean(v333_row.get("tong_status")),
            "v334_tong_prediction": clean(v334_row.get("tong_prediction")),
            "v334_correct": boolish(v334_row.get("tong_correct")),
            "v334_status": clean(v334_row.get("tong_status")),
        }
        ledger.append(row_out)

    summary = summarize_ledger(ledger, args)
    return ledger, summary


def summarize_ledger(ledger: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_miss = Counter(row["miss_class"] for row in ledger)
    by_decision = Counter(row["trainability_decision"] for row in ledger)
    trainable_by_miss = Counter(
        row["miss_class"]
        for row in ledger
        if row["trainability_decision"] == "trainable" and row["candidate_correct"]
    )
    guarded_trainable_by_miss = Counter(
        row["miss_class"]
        for row in ledger
        if row["trainability_decision"] == "trainable_guarded" and row["candidate_correct"]
    )
    direct_usable_by_miss = trainable_by_miss + guarded_trainable_by_miss
    trainable_ids = [
        row["id"]
        for row in ledger
        if row["trainability_decision"] in {"trainable", "trainable_guarded"}
        and row["candidate_correct"]
    ]
    needs_rule_proof_ids = [
        row["id"] for row in ledger if row["trainability_decision"] == "needs_rule_proof"
    ]
    high_ambiguity_ids = [
        row["id"]
        for row in ledger
        if str(row["ambiguity_risk"]).startswith("high")
        and row["trainability_decision"] in {"trainable", "trainable_guarded", "protected-only"}
    ]
    equation_trainable = trainable_by_miss["equation_numeric_miss"]
    equation_guarded = guarded_trainable_by_miss["equation_numeric_miss"]
    bit_trainable = trainable_by_miss["bit_residual_miss"]
    bit_guarded = guarded_trainable_by_miss["bit_residual_miss"]
    if equation_trainable >= 4:
        gpu_gate = "allow_a100_large_equation_transfer_probe"
    elif equation_trainable + equation_guarded >= 4:
        gpu_gate = "allow_a100_large_equation_transfer_probe_guarded"
    elif bit_trainable >= 4:
        gpu_gate = "bit_secondary_ready_but_equation_primary_blocked"
    else:
        gpu_gate = "blocked_no_family_has_four_direct_trainable_gains"

    input_paths = {
        "misses_csv": args.misses_csv,
        "v350_integrated": args.v350_integrated,
        "v350_decisions": args.v350_decisions,
        "v366_integrated": args.v366_integrated,
        "v366_decisions": args.v366_decisions,
        "v336_trace": args.v336_trace,
        "v324_accepted": args.v324_accepted,
        "v333_detail": args.v333_detail,
        "v334_detail": args.v334_detail,
    }
    return {
        "schema_version": "kg1_v672_residual_miss_ledger_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(ledger),
        "miss_class_counts": dict(sorted(by_miss.items())),
        "trainability_decision_counts": dict(sorted(by_decision.items())),
        "direct_trainable_counts": dict(sorted(trainable_by_miss.items())),
        "guarded_trainable_counts": dict(sorted(guarded_trainable_by_miss.items())),
        "direct_usable_counts": dict(sorted(direct_usable_by_miss.items())),
        "trainable_ids": trainable_ids,
        "needs_rule_proof_ids": needs_rule_proof_ids,
        "high_ambiguity_non_drop_ids": high_ambiguity_ids,
        "equation_numeric_direct_trainable": equation_trainable,
        "equation_numeric_guarded_trainable": equation_guarded,
        "equation_numeric_direct_usable": equation_trainable + equation_guarded,
        "bit_residual_direct_trainable": bit_trainable,
        "bit_residual_guarded_trainable": bit_guarded,
        "bit_residual_direct_usable": bit_trainable + bit_guarded,
        "gpu_gate": gpu_gate,
        "gpu_recommendation": (
            "Use a100-large only for a cheap transfer probe, not H200, because "
            "the ledger now has direct no-loss candidates."
            if gpu_gate
            in {
                "allow_a100_large_equation_transfer_probe",
                "allow_a100_large_equation_transfer_probe_guarded",
            }
            else "Do not spend GPU on equation transfer until direct equation candidates reach 4."
            if equation_trainable + equation_guarded < 4
            else "Do not spend H200; bit is secondary and equation primary still controls promotion."
        ),
        "input_files": {name: rel(path) for name, path in input_paths.items()},
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
    }


def write_outputs(ledger: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v672_residual_miss_ledger.csv"
    jsonl_path = out_dir / "v672_residual_miss_ledger.jsonl"
    manifest_path = out_dir / "v672_residual_miss_ledger_manifest.json"
    report_path = out_dir / "KG1_V672_RESIDUAL_MISS_LEDGER.md"

    fieldnames = list(ledger[0].keys()) if ledger else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ledger)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = dict(summary)
    summary["outputs"] = {
        "ledger_csv": rel(csv_path),
        "ledger_jsonl": rel(jsonl_path),
        "manifest_json": rel(manifest_path),
        "markdown_report": rel(report_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)

    decision_counts = summary["trainability_decision_counts"]
    trainable_rows = [
        row
        for row in ledger
        if row["trainability_decision"] in {"trainable", "trainable_guarded"}
    ]
    needs_proof_rows = [row for row in ledger if row["trainability_decision"] == "needs_rule_proof"]
    blocked_rows = [
        row
        for row in ledger
        if row["trainability_decision"]
        not in {"trainable", "trainable_guarded", "needs_rule_proof"}
    ]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# KG1 V672 Residual Miss Ledger\n\n")
        handle.write(f"Generated UTC: `{summary['generated_at_utc']}`\n\n")
        handle.write("## Gate Result\n\n")
        handle.write(f"- Rows audited: `{summary['rows']}`\n")
        handle.write(f"- Miss classes: `{summary['miss_class_counts']}`\n")
        handle.write(f"- Trainability decisions: `{decision_counts}`\n")
        handle.write(f"- Direct trainable counts: `{summary['direct_trainable_counts']}`\n")
        handle.write(f"- Guarded trainable counts: `{summary['guarded_trainable_counts']}`\n")
        handle.write(f"- Direct usable counts: `{summary['direct_usable_counts']}`\n")
        handle.write(f"- GPU gate: `{summary['gpu_gate']}`\n")
        handle.write(f"- Recommendation: {summary['gpu_recommendation']}\n\n")
        handle.write("## Direct Trainable Rows\n\n")
        if not trainable_rows:
            handle.write("None.\n\n")
        else:
            handle.write(
                "| id | miss_class | answer | candidate | source | rule | support | proof |\n"
            )
            handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for row in trainable_rows:
                handle.write(
                    f"| `{row['id']}` | `{row['miss_class']}` | `{row['answer']}` | "
                    f"`{row['candidate_prediction']}` | `{row['candidate_source']}` | "
                    f"`{row['candidate_rule']}` | `{row['query_operator_support']}` | "
                    f"`{row['candidate_proof']}` |\n"
                )
            handle.write("\n")
        handle.write("## Correct But Needs Rule Proof\n\n")
        if not needs_proof_rows:
            handle.write("None.\n\n")
        else:
            handle.write("| id | miss_class | answer | candidate | evidence | reason |\n")
            handle.write("| --- | --- | --- | --- | --- | --- |\n")
            for row in needs_proof_rows:
                handle.write(
                    f"| `{row['id']}` | `{row['miss_class']}` | `{row['answer']}` | "
                    f"`{row['candidate_prediction']}` | `{row['candidate_rule']}` | "
                    f"`{row['trainability_reason']}` |\n"
                )
            handle.write("\n")
        handle.write("## Blocked Rows\n\n")
        if not blocked_rows:
            handle.write("None.\n\n")
        else:
            handle.write("| id | miss_class | answer | baseline | reason | ambiguity |\n")
            handle.write("| --- | --- | --- | --- | --- | --- |\n")
            for row in blocked_rows:
                handle.write(
                    f"| `{row['id']}` | `{row['miss_class']}` | `{row['answer']}` | "
                    f"`{row['baseline_prediction']}` | `{row['trainability_reason']}` | "
                    f"`{row['ambiguity_risk']}` |\n"
                )
            handle.write("\n")
        handle.write("## Output Files\n\n")
        for key, path in summary["outputs"].items():
            handle.write(f"- `{key}`: `{path}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--misses-csv", type=Path, default=DEFAULT_MISSES)
    parser.add_argument("--v350-integrated", type=Path, default=DEFAULT_V350_INTEGRATED)
    parser.add_argument("--v350-decisions", type=Path, default=DEFAULT_V350_DECISIONS)
    parser.add_argument("--v366-integrated", type=Path, default=DEFAULT_V366_INTEGRATED)
    parser.add_argument("--v366-decisions", type=Path, default=DEFAULT_V366_DECISIONS)
    parser.add_argument("--v336-trace", type=Path, default=DEFAULT_V336_TRACE)
    parser.add_argument("--v324-accepted", type=Path, default=DEFAULT_V324_ACCEPTED)
    parser.add_argument("--v333-detail", type=Path, default=DEFAULT_V333_DETAIL)
    parser.add_argument("--v334-detail", type=Path, default=DEFAULT_V334_DETAIL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v672_residual_miss_ledger" / utc_now_tag(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger, summary = build_ledger(args)
    write_outputs(ledger, summary, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

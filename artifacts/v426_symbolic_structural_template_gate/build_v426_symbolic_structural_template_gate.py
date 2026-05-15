#!/usr/bin/env python3
"""V426 symbolic structural template gate.

CPU-only probe for the remaining punctuation-heavy equation_transform misses.
It tests a new DSL class not covered by V421/V422/V423:

* per-output-position templates that can copy an input position or emit a
  constant symbol learned from examples;
* optional operator-specific fitting when at least two same-operator examples
  exist;
* leave-one-out stability and abstain on ambiguity.

Weak labels are used only after predictions are generated to audit gains and
losses. This gate does not authorize training, packaging, or submission.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.competition_utils import classify_puzzle, verify_answer  # noqa: E402


BASELINE_CSV = ROOT / "artifacts/v342_acc_first_diagnostic/v290_checkpoint6_baseline_predictions.csv"
V414_ACCEPTED_CSV = (
    ROOT
    / "artifacts/v414_cpu_teacher_meta_gate/20260515T_v414_cpu_teacher_meta_gate/"
    / "v414_accepted_union.csv"
)
OUT_DIR = ROOT / "artifacts/v426_symbolic_structural_template_gate/20260515T_v426_symbolic_structural_template"

EXPECTED_BASELINE = {
    "correct": 192,
    "equation_transform_correct": 56,
    "bit_manipulation_correct": 136,
    "truncated": 0,
}


@dataclass(frozen=True)
class Term:
    kind: str
    value: str

    def apply(self, token: str) -> str | None:
        if self.kind == "pos":
            idx = int(self.value)
            return token[idx] if 0 <= idx < len(token) else None
        return self.value

    def label(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class Program:
    scope: str
    rhs_len: int
    terms: tuple[Term, ...]
    support_rows: int
    loo_checked: int
    loo_passed: int
    prediction: str
    score: tuple[int, int, int, int, str]

    def label(self) -> str:
        return self.scope + "::" + ",".join(term.label() for term in self.terms)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def family_for(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or row.get("task_type") or classify_puzzle(str(row.get("prompt", ""))))


def parse_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples: list[tuple[str, str]] = []
    for raw in str(prompt or "").splitlines():
        if " = " not in raw:
            continue
        lhs, rhs = raw.split(" = ", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if len(lhs) == 5 and rhs:
            examples.append((lhs, rhs))
    match = re.search(r"Now, determine the result for:\s*([^\n]+)", str(prompt or ""))
    query = match.group(1).strip() if match else None
    return examples, query


def operator_of(token: str | None) -> str:
    text = str(token or "")
    return text[2] if len(text) == 5 else ""


def apply_terms(token: str, terms: tuple[Term, ...]) -> str | None:
    out: list[str] = []
    for term in terms:
        val = term.apply(token)
        if val is None:
            return None
        out.append(val)
    return "".join(out)


def candidate_terms_for_position(examples: list[tuple[str, str]], out_idx: int) -> list[Term]:
    terms: list[Term] = []
    for pos in range(5):
        if all(len(rhs) > out_idx and lhs[pos] == rhs[out_idx] for lhs, rhs in examples):
            terms.append(Term("pos", str(pos)))
    chars = sorted({rhs[out_idx] for _lhs, rhs in examples if len(rhs) > out_idx})
    for ch in chars:
        if all(len(rhs) > out_idx and rhs[out_idx] == ch for _lhs, rhs in examples):
            terms.append(Term("const", ch))
    return terms


def synthesize_templates(
    examples: list[tuple[str, str]],
    query: str,
    *,
    scope: str,
    max_programs_per_len: int = 5000,
) -> list[Program]:
    programs: list[Program] = []
    if len(query) != 5 or len(examples) < 2:
        return programs
    by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for lhs, rhs in examples:
        if 0 < len(rhs) <= 6 and len(lhs) == 5:
            by_len[len(rhs)].append((lhs, rhs))

    for rhs_len, length_examples in sorted(by_len.items()):
        if len(length_examples) < 2:
            continue
        per_pos_terms = [candidate_terms_for_position(length_examples, idx) for idx in range(rhs_len)]
        if any(not items for items in per_pos_terms):
            continue
        total_programs = 1
        for items in per_pos_terms:
            total_programs *= len(items)
        if total_programs > max_programs_per_len:
            continue
        for terms in itertools.product(*per_pos_terms):
            terms_tuple = tuple(terms)
            if not all(apply_terms(lhs, terms_tuple) == rhs for lhs, rhs in length_examples):
                continue
            prediction = apply_terms(query, terms_tuple)
            if prediction is None:
                continue
            loo_checked = 0
            loo_passed = 0
            for omit_idx, (held_lhs, held_rhs) in enumerate(length_examples):
                train = [item for idx, item in enumerate(length_examples) if idx != omit_idx]
                if len(train) < 2:
                    continue
                train_terms = [candidate_terms_for_position(train, idx) for idx in range(rhs_len)]
                if any(not items for items in train_terms):
                    continue
                # The same concrete template must remain valid under LOO.
                if all(terms_tuple[idx] in train_terms[idx] for idx in range(rhs_len)):
                    pred = apply_terms(held_lhs, terms_tuple)
                    if pred is not None:
                        loo_checked += 1
                        loo_passed += int(pred == held_rhs)
            pos_count = sum(1 for term in terms_tuple if term.kind == "pos")
            const_count = len(terms_tuple) - pos_count
            op_pos_count = sum(1 for term in terms_tuple if term.kind == "pos" and term.value == "2")
            score = (-loo_passed, const_count, op_pos_count, -len(length_examples), ",".join(t.label() for t in terms_tuple))
            programs.append(
                Program(
                    scope=scope,
                    rhs_len=rhs_len,
                    terms=terms_tuple,
                    support_rows=len(length_examples),
                    loo_checked=loo_checked,
                    loo_passed=loo_passed,
                    prediction=prediction,
                    score=score,
                )
            )
    return programs


def predict_symbolic(prompt: str) -> tuple[str | None, dict[str, Any]]:
    examples, query = parse_examples(prompt)
    if not query or len(query) != 5:
        return None, {"status": "abstain", "reason": "query_not_len5_or_missing"}
    if len(examples) < 2:
        return None, {"status": "abstain", "reason": "too_few_examples"}

    programs: list[Program] = []
    programs.extend(synthesize_templates(examples, query, scope="all_examples"))
    q_op = operator_of(query)
    same_op = [(lhs, rhs) for lhs, rhs in examples if operator_of(lhs) == q_op]
    if len(same_op) >= 2:
        programs.extend(synthesize_templates(same_op, query, scope="same_operator_examples"))

    if not programs:
        return None, {"status": "abstain", "reason": "no_structural_template"}
    programs.sort(key=lambda item: item.score)
    top_score = programs[0].score[:4]
    top = [item for item in programs if item.score[:4] == top_score]
    predictions = sorted({item.prediction for item in top})
    if len(predictions) != 1:
        return None, {
            "status": "abstain",
            "reason": "ambiguous_top_structural_template",
            "top_predictions": predictions[:20],
            "top_programs": [item.label() for item in top[:20]],
        }
    return predictions[0], {
        "status": "candidate",
        "reason": "unique_structural_template",
        "top_score": list(top_score),
        "top_programs": [item.label() for item in top[:10]],
        "support_rows": max(item.support_rows for item in top),
        "loo_checked": max(item.loo_checked for item in top),
        "loo_passed": max(item.loo_passed for item in top),
    }


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    total = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = family_for(row)
        correct = verify_answer(row["answer"], row[prediction_key])
        truncated = boolish(row.get("truncated", False))
        total["rows"] += 1
        total["correct"] += int(correct)
        total["truncated"] += int(truncated)
        families[family]["rows"] += 1
        families[family]["correct"] += int(correct)
        families[family]["truncated"] += int(truncated)
    return {"total": dict(total), "families": {key: dict(value) for key, value in sorted(families.items())}}


def assert_baseline(summary: dict[str, Any]) -> None:
    observed = {
        "correct": int(summary["total"].get("correct", -1)),
        "equation_transform_correct": int(summary["families"].get("equation_transform", {}).get("correct", -1)),
        "bit_manipulation_correct": int(summary["families"].get("bit_manipulation", {}).get("correct", -1)),
        "truncated": int(summary["total"].get("truncated", -1)),
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError(f"baseline drift: expected {EXPECTED_BASELINE}, got {observed}")


def main() -> int:
    rows = read_csv(BASELINE_CSV)
    baseline = score_rows(rows, "prediction")
    assert_baseline(baseline)
    already_known = {row["id"] for row in read_csv(V414_ACCEPTED_CSV)}

    audit: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    integrated_rows: list[dict[str, Any]] = []

    for row in rows:
        out = dict(row)
        out["v426_prediction"] = row["prediction"]
        out["v426_applied"] = False
        if family_for(row) != "equation_transform":
            integrated_rows.append(out)
            continue

        candidate, meta = predict_symbolic(row.get("prompt", ""))
        baseline_correct = verify_answer(row["answer"], row["prediction"])
        candidate_correct = bool(candidate) and verify_answer(row["answer"], candidate)
        audit_row = {
            "id": row["id"],
            "answer": row["answer"],
            "baseline_prediction": row["prediction"],
            "baseline_correct": baseline_correct,
            "candidate_prediction": candidate or "",
            "candidate_correct": candidate_correct,
            "already_known_v414": row["id"] in already_known,
            "status": meta.get("status", ""),
            "reason": meta.get("reason", ""),
            "proof": json.dumps(meta, sort_keys=True),
        }
        audit.append(audit_row)
        if candidate:
            if baseline_correct and not candidate_correct:
                conflicts.append(audit_row)
            elif (not baseline_correct) and candidate_correct and row["id"] not in already_known:
                accepted.append(audit_row)
                out["v426_prediction"] = candidate
                out["v426_applied"] = True
        integrated_rows.append(out)

    integrated = score_rows(integrated_rows, "v426_prediction")
    candidate_count = sum(1 for row in audit if row["status"] == "candidate")
    accepted_gain = len(accepted)
    conflict_count = len(conflicts)
    projected_total = int(integrated["total"]["correct"])
    projected_eq = int(integrated["families"].get("equation_transform", {}).get("correct", 0))
    projected_bit = int(integrated["families"].get("bit_manipulation", {}).get("correct", 0))
    decision = (
        "v426_safe_gain_found"
        if accepted_gain > 0 and conflict_count == 0 and projected_total > EXPECTED_BASELINE["correct"] and projected_eq > EXPECTED_BASELINE["equation_transform_correct"]
        else "v426_no_safe_new_gain"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_columns = [
        "id",
        "answer",
        "baseline_prediction",
        "baseline_correct",
        "candidate_prediction",
        "candidate_correct",
        "already_known_v414",
        "status",
        "reason",
        "proof",
    ]
    write_csv(OUT_DIR / "v426_symbolic_structural_template_audit.csv", audit, audit_columns)
    write_csv(OUT_DIR / "v426_symbolic_structural_template_accepted.csv", accepted, audit_columns)
    write_csv(OUT_DIR / "v426_symbolic_structural_template_conflicts.csv", conflicts, audit_columns)
    manifest = {
        "schema_version": "kg1_v426_symbolic_structural_template_gate_v1",
        "generated_at_utc": utc_now(),
        "baseline": EXPECTED_BASELINE,
        "candidate_count": candidate_count,
        "accepted_new_gain_count": accepted_gain,
        "conflict_count": conflict_count,
        "projected": {
            "correct": projected_total,
            "equation_transform_correct": projected_eq,
            "bit_manipulation_correct": projected_bit,
            "truncated": int(integrated["total"].get("truncated", 0)),
        },
        "decision": {
            "decision": decision,
            "hf_gpu_allowed": False,
            "reason": f"accepted_new_gain={accepted_gain}; conflicts={conflict_count}; projected={projected_total}/315 eq={projected_eq}/155 bit={projected_bit}/160",
            "next_action": (
                "Use accepted rows only as CPU teacher signal; do not open GPU until adapter/package transfer gate exists."
                if decision == "v426_safe_gain_found"
                else "Reject this DSL class for training; continue with another structural class."
            ),
        },
        "outputs": {
            "manifest_json": str((OUT_DIR / "v426_symbolic_structural_template_manifest.json").relative_to(ROOT)),
            "report_md": str((OUT_DIR / "V426_SYMBOLIC_STRUCTURAL_TEMPLATE_GATE.md").relative_to(ROOT)),
            "audit_csv": str((OUT_DIR / "v426_symbolic_structural_template_audit.csv").relative_to(ROOT)),
            "accepted_csv": str((OUT_DIR / "v426_symbolic_structural_template_accepted.csv").relative_to(ROOT)),
            "conflicts_csv": str((OUT_DIR / "v426_symbolic_structural_template_conflicts.csv").relative_to(ROOT)),
        },
    }
    write_json(OUT_DIR / "v426_symbolic_structural_template_manifest.json", manifest)

    report = [
        "# V426 Symbolic Structural Template Gate",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Baseline Versus Projection",
        "",
        "| State | Total | equation_transform | bit_manipulation | Truncated |",
        "|---|---:|---:|---:|---:|",
        f"| V291/V290 baseline | `{EXPECTED_BASELINE['correct']}/315` | `{EXPECTED_BASELINE['equation_transform_correct']}/155` | `{EXPECTED_BASELINE['bit_manipulation_correct']}/160` | `0` |",
        f"| V426 projection | `{projected_total}/315` | `{projected_eq}/155` | `{projected_bit}/160` | `{manifest['projected']['truncated']}` |",
        "",
        "## Gate Metrics",
        "",
        f"- Candidate rows: `{candidate_count}`.",
        f"- Accepted new gains outside V414: `{accepted_gain}`.",
        f"- Conflicts/losses against baseline-correct rows: `{conflict_count}`.",
        "",
        "## Decision",
        "",
        f"`{decision}`: {manifest['decision']['reason']}",
        "",
        "This is a CPU-only rule probe. It is not a submit artifact and does not authorize HF GPU by itself.",
    ]
    (OUT_DIR / "V426_SYMBOLIC_STRUCTURAL_TEMPLATE_GATE.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
